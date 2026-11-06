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

import java.io.{ByteArrayOutputStream, ObjectOutputStream}

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, ExprId}
import org.apache.spark.sql.catalyst.expressions.aggregate.{Final, Partial}
import org.apache.spark.sql.execution.{LeafExecNode, LocalTableScanExec}
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuDecimal128Sum}
import org.apache.spark.sql.types.{DecimalType, LongType, StringType}

class HierarchicalRollupPlanSuite extends AnyFunSuite {
  private case class Input(override val output: Seq[Attribute]) extends LeafExecNode {
    override protected def doExecute(): RDD[InternalRow] =
      throw new UnsupportedOperationException("Planning only")
  }

  private def example(precision: Int = 28): GpuHashAggregateExec = {
    val left = AttributeReference("a", StringType)()
    val right = AttributeReference("b", StringType)()
    val value = AttributeReference("v", DecimalType(precision, 2))()
    val outLeft = AttributeReference("ga", StringType)()
    val outRight = AttributeReference("gb", StringType)()
    val gid = AttributeReference("gid", LongType, nullable = false)()
    val nil = GpuLiteral(null, StringType)
    val expand = GpuExpandExec(Seq(
      Seq(left, right, GpuLiteral(0L, LongType), value),
      Seq(left, nil, GpuLiteral(1L, LongType), value),
      Seq(nil, nil, GpuLiteral(3L, LongType), value)),
      Seq(outLeft, outRight, gid, value), Input(Seq(left, right, value)))()
    val sum = GpuDecimal128Sum(value, DecimalType(38, 2),
      failOnErrorOverride = false, forceWindowSumToNotBeReplaced = false)
    val expression = GpuAggregateExpression(sum, Partial, isDistinct = false, None, ExprId(100))
    val groups = Seq(outLeft, outRight, gid)
    GpuHashAggregateExec(None, groups, Seq(expression), Seq.empty,
      groups ++ sum.aggBufferAttributes, expand, 16 * 1024 * 1024, 1.0,
      forceSinglePassAgg = false, allowSinglePassAgg = false,
      allowNonFullyAggregatedOutput = true, skipAggPassReductionRatio = 0.9)
  }

  private def supported(agg: GpuHashAggregateExec): Boolean =
    GpuHierarchicalRollupAggregateExec.tryReplace(agg)
      .isInstanceOf[GpuHierarchicalRollupAggregateExec]

  test("accept a nullable-key decimal rollup and preserve output attributes") {
    val agg = example()
    val result = GpuHierarchicalRollupAggregateExec.tryReplace(agg)
      .asInstanceOf[GpuHierarchicalRollupAggregateExec]
    assertResult(agg.output)(result.output)
    assertResult(2)(result.mergeProjections.size)
    assert(result.child eq agg.child.children.head)
    assert(result.references.subsetOf(result.child.outputSet))
  }

  test("remove only constant keys and retain a key for the coarsest empty-input level") {
    val agg = example()
    val expand = agg.child.asInstanceOf[GpuExpandExec]
    val groups = expand.projections.map(_.take(3))
    val active = groups.map(GpuHierarchicalRollupAggregateExec.activeGroupingIndexes)
    assertResult(Seq(Seq(0, 1), Seq(0), Seq(2)))(active)
    assert(active.forall(_.nonEmpty), "An empty input must never become a zero-key reduction")
    assert(groups.head.head.nullable, "Nullable data keys must still participate in grouping")
  }

  test("reject input precision beyond the within-batch overflow proof") {
    assert(supported(example(28)))
    assert(!supported(example(29)))
    assert(!supported(example(38)))
  }

  test("reject grouping sets that restore an already removed key") {
    val agg = example()
    val expand = agg.child.asInstanceOf[GpuExpandExec]
    val broken = expand.projections.updated(2,
      expand.projections(2).updated(1, expand.projections.head(1)))
    assert(!supported(agg.copy(child = expand.copy(projections = broken)(false, true))))
  }

  test("reject aggregate inputs changed by an Expand projection") {
    val agg = example()
    val expand = agg.child.asInstanceOf[GpuExpandExec]
    val broken = expand.projections.updated(1,
      expand.projections(1).updated(3, GpuLiteral(null, DecimalType(28, 2))))
    assert(!supported(agg.copy(child = expand.copy(projections = broken)(false, true))))
  }

  test("reject distinct filtered and final aggregates") {
    val agg = example()
    val sum = agg.aggregateExpressions.head
    assert(!supported(agg.copy(aggregateExpressions = Seq(sum.copy(isDistinct = true)))))
    assert(!supported(agg.copy(aggregateExpressions = Seq(sum.copy(mode = Final)))))
    assert(!supported(agg.copy(aggregateExpressions = Seq(
      sum.copy(filter = Some(GpuLiteral(true)))))))
  }

  test("respect forced sorting and preserve non-rollup aggregates") {
    val agg = example()
    assert(!supported(agg.copy(forceSinglePassAgg = true)))
    val noExpand = agg.copy(child = agg.child.children.head)
    assert(GpuHierarchicalRollupAggregateExec.tryReplace(noExpand) eq noExpand)
  }

  test("reject repeated grouping identifiers") {
    val agg = example()
    val expand = agg.child.asInstanceOf[GpuExpandExec]
    val broken = expand.projections.updated(2,
      expand.projections(2).updated(2, GpuLiteral(1L, LongType)))
    assert(!supported(agg.copy(child = expand.copy(projections = broken)(false, true))))
  }

  test("plan remains serializable after reference validation") {
    val agg = example()
    val expand = agg.child.asInstanceOf[GpuExpandExec]
    val input = LocalTableScanExec(expand.child.output, Seq.empty, None)
    val plan = GpuHierarchicalRollupAggregateExec.tryReplace(
      agg.copy(child = expand.copy(child = input)(false, true)))
      .asInstanceOf[GpuHierarchicalRollupAggregateExec]
    assert(plan.references.subsetOf(plan.child.outputSet))
    val bytes = new ByteArrayOutputStream()
    val stream = new ObjectOutputStream(bytes)
    try {
      stream.writeObject(plan)
    } finally {
      stream.close()
    }
    assert(bytes.size() > 0)
  }
}
