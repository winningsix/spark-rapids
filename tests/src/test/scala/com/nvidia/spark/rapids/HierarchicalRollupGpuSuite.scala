/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.spark.rapids

import scala.collection.mutable

import ai.rapids.cudf.ColumnVector
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.RmmSpark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, ExprId}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.execution.LeafExecNode
import org.apache.spark.sql.rapids.GpuUnaryMinus
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuDecimal128Sum}
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.rapids.metrics.source.MockTaskContext
import org.apache.spark.sql.types.{DecimalType, LongType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class HierarchicalRollupGpuSuite extends RmmSparkRetrySuiteBase {
  private case class Input(override val output: Seq[Attribute], rdd: RDD[ColumnarBatch])
      extends LeafExecNode {
    override def supportsColumnar: Boolean = true
    override protected def doExecuteColumnar(): RDD[ColumnarBatch] = rdd
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("Columnar input only")
  }

  private case class Row(a: String, b: String, value: Option[BigInt])
  private type Key = (Option[String], Option[String], Long)

  private def rows(count: Int, allNull: Boolean): Seq[Row] = (0 until count).map { i =>
    Row(if (i % 5 == 0) null else s"a${i % 4}",
      if (i % 7 == 0) null else s"b${i % 9}",
      if (allNull || i % 11 == 0) None else Some(
        (BigInt(10).pow(27) + i) * (if (i % 2 == 0) 1 else -1)))
  }

  private def batch(rows: Seq[Row]): ColumnarBatch = {
    withResource(ColumnVector.fromStrings(rows.map(_.a): _*)) { a =>
      withResource(ColumnVector.fromStrings(rows.map(_.b): _*)) { b =>
        withResource(ColumnVector.decimalFromBigInt(-2,
            rows.map(_.value.map(_.bigInteger).orNull): _*)) { value =>
          new ColumnarBatch(Array(GpuColumnVector.from(a.incRefCount(), StringType),
            GpuColumnVector.from(b.incRefCount(), StringType),
            GpuColumnVector.from(value.incRefCount(), DecimalType(28, 2))), rows.size)
        }
      }
    }
  }

  private def mergeValue(a: Option[BigInt], b: Option[BigInt]): Option[BigInt] =
    if (a.isEmpty && b.isEmpty) None else Some(a.getOrElse(BigInt(0)) + b.getOrElse(BigInt(0)))

  private def runCase(count: Int, batchRows: Int, allNull: Boolean, injection: String,
      multipleSums: Boolean = false, includeEmptyBatches: Boolean = false): Unit = {
    val data = rows(count, allNull)
    val expected = mutable.Map.empty[Key, Option[BigInt]]
    data.foreach { row =>
      Seq((Option(row.a), Option(row.b), 0L), (Option(row.a), None, 1L),
        (None, None, 3L)).foreach { key =>
        expected.update(key, mergeValue(expected.getOrElse(key, None), row.value))
      }
    }
    val spark = SparkSession.builder().master("local[1]").appName("hierarchical-rollup-test")
      .config("spark.ui.enabled", "false").config(RapidsConf.METRICS_LEVEL.key, "DEBUG")
      .getOrCreate()
    try {
      val a = AttributeReference("a", StringType)()
      val b = AttributeReference("b", StringType)()
      val v = AttributeReference("v", DecimalType(28, 2))()
      val ga = AttributeReference("ga", StringType)()
      val gb = AttributeReference("gb", StringType)()
      val gid = AttributeReference("gid", LongType, nullable = false)()
      val nonemptyInputs = data.grouped(batchRows).map(batch).toSeq
      val inputs = if (includeEmptyBatches) {
        Seq(batch(Seq.empty)) ++ nonemptyInputs ++ Seq(batch(Seq.empty))
      } else nonemptyInputs
      val rdd = spark.sparkContext.parallelize(inputs, numSlices = 1)
      val nil = GpuLiteral(null, StringType)
      val expand = GpuExpandExec(Seq(
        Seq(a, b, GpuLiteral(0L, LongType), v),
        Seq(a, nil, GpuLiteral(1L, LongType), v),
        Seq(nil, nil, GpuLiteral(3L, LongType), v)),
        Seq(ga, gb, gid, v), Input(Seq(a, b, v), rdd))()
      val sum = GpuDecimal128Sum(v, DecimalType(38, 2),
        failOnErrorOverride = false, forceWindowSumToNotBeReplaced = false)
      val sums = if (multipleSums) {
        Seq(sum, GpuDecimal128Sum(GpuUnaryMinus(v, failOnError = false), DecimalType(38, 2),
          failOnErrorOverride = false, forceWindowSumToNotBeReplaced = false))
      } else {
        Seq(sum)
      }
      val exprs = sums.zipWithIndex.map { case (agg, index) =>
        GpuAggregateExpression(agg, Partial, isDistinct = false, None, ExprId(100 + index))
      }
      val groups = Seq(ga, gb, gid)
      val original = GpuHashAggregateExec(None, groups, exprs, Seq.empty,
        groups ++ sums.flatMap(_.aggBufferAttributes), expand, 1024 * 1024, 1.0,
        forceSinglePassAgg = false, allowSinglePassAgg = false,
        allowNonFullyAggregatedOutput = true, skipAggPassReductionRatio = 0.9)
      val node = GpuHierarchicalRollupAggregateExec.tryReplace(original)
        .asInstanceOf[GpuHierarchicalRollupAggregateExec]
      val outputRdd = node.doExecuteColumnar()
      val context = new MockTaskContext(taskAttemptId = 1, partitionId = 0)
      TrampolineUtil.setTaskContext(context)
      try {
        RmmSpark.getAndResetNumRetryThrow(1)
        RmmSpark.getAndResetNumSplitRetryThrow(1)
        injection match {
          case "retry" => RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case "split" => RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)
          case _ =>
        }
        val actual = mutable.Map.empty[Key, Option[BigInt]]
        val iterator = outputRdd.compute(outputRdd.partitions.head, context)
        iterator.foreach { cb =>
          withResource(cb) { _ =>
            withResource(cb.column(0).asInstanceOf[GpuColumnVector].copyToHost()) { ha =>
              withResource(cb.column(1).asInstanceOf[GpuColumnVector].copyToHost()) { hb =>
                withResource(cb.column(2).asInstanceOf[GpuColumnVector].copyToHost()) { hg =>
                  val sumColumn = cb.column(3).asInstanceOf[GpuColumnVector].getBase
                  withResource(sumColumn.copyToHost()) { hv =>
                    withResource(cb.column(4).asInstanceOf[GpuColumnVector].copyToHost()) { empty =>
                      (0 until cb.numRows()).foreach { i =>
                        val key = (if (ha.isNullAt(i)) None else Some(ha.getUTF8String(i).toString),
                          if (hb.isNullAt(i)) None else Some(hb.getUTF8String(i).toString),
                          hg.getLong(i))
                        val value = if (empty.getBoolean(i)) {
                          None
                        } else {
                          Some(BigInt(hv.getBigDecimal(i).unscaledValue()))
                        }
                        actual.update(key, mergeValue(actual.getOrElse(key, None), value))
                      }
                      if (multipleSums) {
                        val negativeSum = cb.column(5).asInstanceOf[GpuColumnVector].getBase
                        val negativeEmpty = cb.column(6).asInstanceOf[GpuColumnVector]
                        withResource(negativeSum.copyToHost()) { hn =>
                          withResource(negativeEmpty.copyToHost()) { he =>
                            (0 until cb.numRows()).foreach { i =>
                              assert(he.getBoolean(i) == empty.getBoolean(i))
                              if (!empty.getBoolean(i)) {
                                assert(hn.getBigDecimal(i) == hv.getBigDecimal(i).negate())
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        assertResult(expected.toMap)(actual.toMap)
        if (injection == "split") assert(RmmSpark.getAndResetNumSplitRetryThrow(1) > 0)
        if (injection == "retry") assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
      } finally {
        context.markTaskComplete()
        TrampolineUtil.unsetTaskContext()
        ScalableTaskCompletion.reset()
      }
    } finally {
      spark.stop()
    }
  }

  test("hierarchical partial buffers merge across batches with null keys and signed decimals") {
    runCase(513, 97, allNull = false, "none")
  }
  test("all-null values preserve the empty SUM buffer flag") {
    runCase(200, 37, allNull = true, "none")
  }
  test("empty input produces no partial groups") {
    runCase(0, 37, allNull = false, "none")
  }
  test("empty physical batches do not create a constant-key aggregate row") {
    runCase(0, 37, allNull = false, "none", includeEmptyBatches = true)
  }
  test("empty physical batches preserve populated nullable-key groups") {
    runCase(513, 97, allNull = false, "none", includeEmptyBatches = true)
  }
  test("plain retry preserves hierarchical partial sums") {
    runCase(513, 97, allNull = false, "retry")
  }
  test("split retry preserves hierarchical partial sums") {
    runCase(513, 97, allNull = false, "split")
  }
  test("multiple decimal sums retain independent buffers across rollup levels") {
    runCase(513, 97, allNull = false, "none", multipleSums = true)
  }
  test("multiple decimal sums preserve null flags and split retry results") {
    runCase(513, 97, allNull = true, "split", multipleSums = true)
  }
}
