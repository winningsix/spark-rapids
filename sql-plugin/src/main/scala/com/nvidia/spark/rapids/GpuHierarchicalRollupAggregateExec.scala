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

import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuAggregateIterator.computeAggregateAndClose
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.shims.ShimUnaryExecNode

import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference, AttributeSet, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.catalyst.plans.physical.UnknownPartitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuDecimal128Sum}
import org.apache.spark.sql.types.{DecimalType, LongType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuHierarchicalRollupAggregateExec {
  /** Constants do not distinguish groups; retain one for the empty-input case. */
  private[rapids] def activeGroupingIndexes(projection: Seq[Expression]): Seq[Int] = {
    val varying = projection.indices.filter(i => !projection(i).isInstanceOf[GpuLiteral])
    if (varying.nonEmpty) varying else {
      // A zero-key reduction emits a row for empty input. A constant grouping key
      // retains the original grouped-aggregate behavior, which emits no rows.
      val identifier = projection.indexWhere {
        case GpuLiteral(value, _) => value != null
        case _ => false
      }
      require(identifier >= 0, "A validated rollup level must have a grouping identifier")
      Seq(identifier)
    }
  }

  /** A strict chain of grouping sets; each coarser set removes one more grouping key. */
  private[rapids] def mergeProjections(
      agg: GpuHashAggregateExec,
      expand: GpuExpandExec): Option[Seq[Seq[Expression]]] = {
    val groups = agg.groupingExpressions
    val groupAttrs = groups.map(_.toAttribute)
    val buffers = agg.aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)
    val groupIndexes = groupAttrs.map(a => expand.output.indexWhere(_.exprId == a.exprId))
    val levels = expand.projections
    val sumsSupported = agg.aggregateExpressions.nonEmpty &&
      agg.aggregateExpressions.forall { expr =>
        expr.mode == Partial && !expr.isDistinct && expr.filter.isEmpty &&
          (expr.aggregateFunction match {
            case sum: GpuDecimal128Sum => sum.child.dataType match {
              // A cuDF batch has at most Int.MaxValue rows. With p <= 28, every
              // within-batch sum fits decimal(p + 10), including signed values.
              // Keep the normal merge/final overflow checks across input batches.
              case d: DecimalType => d.precision <= 28 &&
                sum.dataType.asInstanceOf[DecimalType].precision >= d.precision + 10 &&
                sum.dataType.asInstanceOf[DecimalType].scale == d.scale &&
                sum.child.deterministic
              case _ => false
            }
            case _ => false
          })
      }
    val shapeSupported = groups.nonEmpty && groups.forall(_.isInstanceOf[AttributeReference]) &&
      groupIndexes.forall(_ >= 0) && groupIndexes.distinct.size == groups.size &&
      levels.size == groups.size && levels.size >= 3 && levels.size <= 64 &&
      levels.forall(_.size == expand.output.size) && levels.flatten.forall(_.deterministic) &&
      agg.requiredChildDistributionExpressions.isEmpty && !agg.forceSinglePassAgg &&
      agg.resultExpressions.forall(_.isInstanceOf[AttributeReference]) &&
      agg.resultExpressions.map(_.dataType) == (groupAttrs ++ buffers).map(_.dataType) &&
      agg.resultExpressions.take(groups.size).map(_.exprId) == groupAttrs.map(_.exprId)
    if (!sumsSupported || !shapeSupported) {
      None
    } else {
      val groupLevels = levels.map(p => groupIndexes.map(p))
      val gidIndexes = groups.indices.filter { index =>
        groupLevels.forall(_(index) match {
          case GpuLiteral(_: Long, LongType) => true
          case _ => false
        }) && groupLevels.map(_(index)).distinct.size == levels.size
      }
      val invariantIndexes = expand.output.indices.filter { index =>
        levels.forall(_(index).semanticEquals(levels.head(index)))
      }
      val invariantAttrs = AttributeSet(invariantIndexes.map(expand.output))
      val inputIsInvariant = agg.aggregateExpressions.forall { expr =>
        expr.aggregateFunction.references.subsetOf(invariantAttrs)
      }
      if (gidIndexes.size != 1 || !inputIsInvariant) {
        None
      } else {
        val gid = gidIndexes.head
        val keyIndexes = groups.indices.filter(_ != gid)
        val keysAreAttributes = keyIndexes.forall { index =>
          groupLevels.head(index).isInstanceOf[AttributeReference]
        }
        val nullSets = groupLevels.map { projection =>
          keyIndexes.filter { index => projection(index) match {
            case GpuLiteral(null, _) => true
            case _ => false
          }}.toSet
        }
        val isChain = nullSets.zipWithIndex.forall { case (keys, index) =>
          keys.size == index && (index == 0 || nullSets(index - 1).subsetOf(keys))
        }
        val onlyNullOrOriginal = groupLevels.forall { projection =>
          keyIndexes.forall { index => projection(index) match {
            case GpuLiteral(null, dataType) => dataType == groups(index).dataType
            case other => other.semanticEquals(groupLevels.head(index))
          }}
        }
        if (!keysAreAttributes || !isChain || !onlyNullOrOriginal) {
          None
        } else {
          Some(groupLevels.tail.zipWithIndex.map { case (projection, level) =>
            groups.indices.map { index =>
              if (index == gid || nullSets(level + 1).contains(index)) {
                projection(index)
              } else {
                groupAttrs(index)
              }
            } ++ buffers
          })
        }
      }
    }
  }

  def tryReplace(agg: GpuHashAggregateExec): GpuExec = agg.child match {
    case expand: GpuExpandExec =>
      mergeProjections(agg, expand).map { projections =>
        GpuHierarchicalRollupAggregateExec(agg.groupingExpressions,
          agg.aggregateExpressions, agg.aggregateAttributes, agg.resultExpressions,
          expand.projections.head, expand.output, projections, expand.child,
          agg.configuredTargetBatchSize, agg.allowNonFullyAggregatedOutput,
          agg.skipAggPassReductionRatio)
      }.getOrElse(agg)
    case _ => agg
  }
}

/** Build each coarser grouping set from the previous set's partial aggregate buffers. */
case class GpuHierarchicalRollupAggregateExec(
    groupingExpressions: Seq[NamedExpression],
    aggregateExpressions: Seq[GpuAggregateExpression],
    aggregateAttributes: Seq[Attribute],
    resultExpressions: Seq[NamedExpression],
    firstProjection: Seq[Expression],
    expandedAttributes: Seq[Attribute],
    mergeProjections: Seq[Seq[Expression]],
    child: SparkPlan,
    configuredTargetBatchSize: Long,
    allowNonFullyAggregatedOutput: Boolean,
    skipAggPassReductionRatio: Double)
  extends ShimUnaryExecNode with GpuExec {

  override def output: Seq[Attribute] = resultExpressions.map(_.toAttribute)
  override def outputPartitioning: UnknownPartitioning = UnknownPartitioning(0)
  override protected val outputRowsLevel: MetricsLevel = ESSENTIAL_LEVEL
  override protected val outputBatchesLevel: MetricsLevel = MODERATE_LEVEL
  override def gpuExpressions: Seq[Expression] =
    groupingExpressions ++ aggregateExpressions ++ resultExpressions ++ firstProjection ++
      mergeProjections.flatten
  override def references: AttributeSet = AttributeSet(firstProjection.flatMap(_.references))

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    NUM_TASKS_REPARTITIONED -> createMetric(MODERATE_LEVEL, DESCRIPTION_NUM_TASKS_REPARTITIONED),
    NUM_TASKS_SKIPPED_AGG -> createMetric(MODERATE_LEVEL, DESCRIPTION_NUM_TASKS_SKIPPED_AGG),
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    AGG_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_AGG_TIME),
    CONCAT_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_CONCAT_TIME),
    SORT_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_SORT_TIME),
    REPARTITION_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_REPARTITION_TIME),
    "NUM_AGGS" -> createMetric(DEBUG_LEVEL, "num agg operations"),
    "NUM_PRE_SPLITS" -> createMetric(DEBUG_LEVEL, "num pre splits"),
    "NUM_TASKS_SINGLE_PASS" -> createMetric(MODERATE_LEVEL, "number of single pass tasks"),
    "HEURISTIC_TIME" -> createNanoTimingMetric(DEBUG_LEVEL, "time in heuristic"))

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val metrics = GpuHashAggregateMetrics(gpuLongMetric(NUM_OUTPUT_ROWS),
      gpuLongMetric(NUM_OUTPUT_BATCHES), gpuLongMetric(NUM_TASKS_REPARTITIONED),
      gpuLongMetric(NUM_TASKS_SKIPPED_AGG), gpuLongMetric(OP_TIME_LEGACY),
      gpuLongMetric(AGG_TIME), gpuLongMetric(CONCAT_TIME), gpuLongMetric(SORT_TIME),
      gpuLongMetric(REPARTITION_TIME), gpuLongMetric("NUM_AGGS"),
      gpuLongMetric("NUM_PRE_SPLITS"), gpuLongMetric("NUM_TASKS_SINGLE_PASS"),
      gpuLongMetric("HEURISTIC_TIME"))
    val inputs = expandedAttributes
    val groups = groupingExpressions
    val aggregates = aggregateExpressions
    val attributes = aggregateAttributes
    val results = resultExpressions
    val sqlConf = conf
    val metricMap = allMetrics
    val batchSize = configuredTargetBatchSize
    val allowPartial = allowNonFullyAggregatedOutput
    val skipRatio = skipAggPassReductionRatio
    val mode = AggregateModeInfo(Seq(Partial))
    val groupAttributes = groups.map(_.toAttribute)
    val buffers = aggregates.flatMap(_.aggregateFunction.aggBufferAttributes)
    val bufferAttributes = groupAttributes ++ buffers
    val firstGroups = groupAttributes.map { attribute =>
      firstProjection(inputs.indexWhere(_.exprId == attribute.exprId))
    }
    val levelProjections = firstGroups +: mergeProjections.map(_.take(groups.size))
    val activeIndexes = levelProjections.map(
      GpuHierarchicalRollupAggregateExec.activeGroupingIndexes)
    val levelGroups = activeIndexes.map(_.map(groups))
    val omittedFirstIds = groups.indices.filterNot(activeIndexes.head.contains)
      .map(groupAttributes(_).exprId).toSet
    val firstInputIndexes = inputs.indices.filterNot(i => omittedFirstIds(inputs(i).exprId))
    val firstInputs = firstInputIndexes.map(inputs)
    val first = GpuBindReferences.bindGpuReferencesTiered(
      firstInputIndexes.map(firstProjection), child.output, sqlConf, metricMap)
    val merges = mergeProjections.zip(activeIndexes.tail).map { case (projection, indexes) =>
      GpuBindReferences.bindGpuReferencesTiered(indexes.map(projection) ++
        projection.drop(groups.size), bufferAttributes, sqlConf, metricMap)
    }
    val restores = levelProjections.zip(activeIndexes).map { case (projection, indexes) =>
      val output = groups.indices.map { index =>
        if (indexes.contains(index)) groupAttributes(index) else projection(index)
      } ++ buffers
      GpuBindReferences.bindGpuReferencesTiered(output,
        indexes.map(groupAttributes) ++ buffers, sqlConf, metricMap)
    }
    child.executeColumnar().mapPartitions { batches =>
      // SQLConf has transient settings. Read the executor's task-local configuration
      // instead of capturing and serializing the driver's instance in this closure.
      val taskConf = SQLConf.get
      val localExpandedRows = new LocalGpuMetric
      val update = new AggHelper(firstInputs, levelGroups.head, aggregates,
        forceMerge = false, conf = taskConf, metrics = metricMap)
      val merge = levelGroups.tail.map { grouping =>
        new AggHelper(grouping.map(_.toAttribute) ++ buffers, grouping, aggregates,
          forceMerge = true, conf = taskConf, metrics = metricMap)
      }
      val firstPass = new GpuHierarchicalRollupIterator(batches, first, merges,
        update, merge, restores, metrics, localExpandedRows)
      val merged = new GpuMergeAggregateIterator(new CloseableBufferedIterator(firstPass),
        inputs, groups, aggregates, attributes, results, mode, metrics, batchSize, taskConf,
        allowPartial, skipRatio, localExpandedRows, metricMap)
      val finalRefs = GpuAggFinalPassIterator.setupReferences(groups, aggregates,
        attributes, results, mode, metricMap)
      GpuAggFinalPassIterator.makeIter(merged, finalRefs, metrics)
    }
  }

  override protected def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("Row-based execution is not supported")
}

private[rapids] class GpuHierarchicalRollupIterator(
    input: Iterator[ColumnarBatch],
    firstProjection: GpuTieredProject,
    mergeProjections: Seq[GpuTieredProject],
    update: AggHelper,
    merge: Seq[AggHelper],
    restores: Seq[GpuTieredProject],
    metrics: GpuHashAggregateMetrics,
    expandedRows: LocalGpuMetric)
  extends Iterator[SpillableColumnarBatch] with AutoCloseable {

  private var previous: Option[SpillableColumnarBatch] = None
  private var nextLevel = 0
  Option(TaskContext.get()).foreach(tc => onTaskCompletion(tc) { close() })

  override def hasNext: Boolean = previous.isDefined || input.hasNext

  private def restore(
      aggregate: SpillableColumnarBatch,
      projection: GpuTieredProject): SpillableColumnarBatch = {
    val batch = projection.projectAndCloseWithRetrySingleBatch(aggregate)
    closeOnExcept(batch) { cb =>
      SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
    }
  }

  override def next(): SpillableColumnarBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("No rollup batch available")
    }
    val old = previous
    previous = None
    val result = withResource(old) {
      case Some(last) =>
        val projected = mergeProjections(nextLevel - 1).projectWithRetrySingleBatch(last)
        restore(computeAggregateAndClose(metrics, projected, merge(nextLevel - 1)),
          restores(nextLevel))
      case None =>
        val batch = input.next()
        // This private heuristic counts the rows the replaced Expand would have produced.
        // It is not a reported physical-input metric.
        expandedRows += batch.numRows().toLong * (mergeProjections.size + 1)
        val spillable = closeOnExcept(batch) { cb =>
          SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
        }
        val projected = firstProjection.projectAndCloseWithRetrySingleBatch(spillable)
        restore(computeAggregateAndClose(metrics, projected, update), restores.head)
    }
    nextLevel += 1
    if (nextLevel > mergeProjections.size) {
      nextLevel = 0
      result
    } else {
      previous = Some(result)
      result.incRefCount()
    }
  }

  override def close(): Unit = withResource(previous) { _ => previous = None }
}
