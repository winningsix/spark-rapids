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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, AttributeMap}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, LogicalPlan, Project, Statistics}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{LogicalQueryStage, QueryStageExec}
import org.apache.spark.sql.execution.aggregate.BaseAggregateExec

/** Restore column estimates discarded when AQE substitutes materialized stage statistics. */
class GpuAqeGroupingStatistics(spark: SparkSession) extends Rule[LogicalPlan] {
  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (new RapidsConf(spark.sessionState.conf).isSqlEnabled) {
      GpuAqeGroupingStatistics.restore(plan)
    } else {
      plan
    }
  }
}

object GpuAqeGroupingStatistics {
  private def hasGroupedAggregate(stage: LogicalQueryStage): Boolean = stage.physicalPlan match {
    case aggregate: BaseAggregateExec => aggregate.groupingExpressions.nonEmpty
    case aggregate: GpuHashAggregateExec => aggregate.groupingExpressions.nonEmpty
    case _ => false
  }

  private[rapids] def supplement(stage: LogicalQueryStage, runtime: Statistics): Statistics = {
    stage.logicalPlan match {
      case aggregate: Aggregate if runtime.isRuntime && runtime.rowCount.exists(_ > 0) &&
          hasGroupedAggregate(stage) =>
        val rows = runtime.rowCount.get
        val missing = aggregate.aggregateExpressions.collect {
          case key: Attribute => (key, key)
          case alias @ Alias(key: Attribute, _) => (alias.toAttribute, key)
        }.filter { case (output, key) =>
          aggregate.groupingExpressions.exists(_.semanticEquals(key)) &&
            !runtime.attributeStats.contains(output)
        }.distinct.flatMap { case (output, key) =>
          // A computed sibling grouping key can make AggregateEstimation discard all
          // column estimates. An unchanged key still has its child's domain, even
          // when the aggregate exposes that key through an alias.
          aggregate.stats.attributeStats.get(output).filter(_.hasCountStats)
            .orElse(aggregate.child.stats.attributeStats.get(key).filter(_.hasCountStats))
            .map { column =>
            output -> column.copy(
              distinctCount = column.distinctCount.map(_.min(rows)),
              nullCount = column.nullCount.map(_.min(rows)), histogram = None)
          }
        }
        if (missing.isEmpty) runtime else runtime.copy(
          attributeStats = AttributeMap(runtime.attributeStats.toSeq ++ missing))
      case project: Project if runtime.isRuntime && runtime.rowCount.exists(_ > 0) &&
          stage.physicalPlan.isInstanceOf[QueryStageExec] =>
        // Losing a broadcast dimension's NDV makes downstream joins look Cartesian.
        // Keep the measured stage size and row bounds; only retain missing estimates.
        val rows = runtime.rowCount.get
        val missing = project.output.distinct.flatMap { output =>
          if (runtime.attributeStats.contains(output)) {
            None
          } else {
            project.stats.attributeStats.get(output).filter(_.hasCountStats).map { column =>
              output -> column.copy(distinctCount = column.distinctCount.map(_.min(rows)),
                nullCount = column.nullCount.map(_.min(rows)), histogram = None)
            }
          }
        }
        if (missing.isEmpty) runtime else runtime.copy(
          attributeStats = AttributeMap(runtime.attributeStats.toSeq ++ missing))
      case _ => runtime
    }
  }

  def restore(plan: LogicalPlan): LogicalPlan = plan.transformUp {
    case stage: LogicalQueryStage if !stage.isInstanceOf[GpuGroupingLogicalQueryStage] &&
        supplement(stage, stage.stats) != stage.stats =>
      new GpuGroupingLogicalQueryStage(stage.logicalPlan, stage.physicalPlan)
  }
}

/**
 * Keep the lower stage's runtime row/byte bounds, including maxRows used by limit elimination.
 * Only column estimates are supplemented; estimated group counts must never become row bounds.
 */
class GpuGroupingLogicalQueryStage(logical: LogicalPlan, physical: SparkPlan)
    extends LogicalQueryStage(logical, physical) {
  override def computeStats(): Statistics =
    GpuAqeGroupingStatistics.supplement(this, super.computeStats())

  override def copy(logicalPlan: LogicalPlan, physicalPlan: SparkPlan): LogicalQueryStage =
    new GpuGroupingLogicalQueryStage(logicalPlan, physicalPlan)

  // TreeNode.fastEquals must observe the replacement, with symmetric equality to the base class.
  override def canEqual(other: Any): Boolean = other.isInstanceOf[GpuGroupingLogicalQueryStage]
  override def equals(other: Any): Boolean = canEqual(other) && super.equals(other)
  override def hashCode(): Int = super.hashCode()
}
