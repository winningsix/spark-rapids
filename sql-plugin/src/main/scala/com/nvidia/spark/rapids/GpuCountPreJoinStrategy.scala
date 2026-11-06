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

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete, Count, Sum}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.execution.adaptive.LogicalQueryStage

/** Count repeated fact keys before materializing dimension grouping attributes. */
private[rapids] object GpuCountPreJoinReorder extends PredicateHelper {
  private val planned = TreeNodeTag[Boolean]("rapids.countPreJoinPlanned")
  private val initiallySelected = TreeNodeTag[Boolean]("rapids.countPreJoinInitiallySelected")
  private val minimumReduction = 4

  private def unknownScalarSelectivity(plan: LogicalPlan): Boolean = plan.exists {
    case Filter(condition, _) => condition.exists(_.isInstanceOf[ScalarSubquery])
    case join: Join => join.condition.exists(_.exists(_.isInstanceOf[ScalarSubquery]))
    case _ => false
  }

  private def supported(expression: AggregateExpression): Boolean = {
    expression.mode == Complete && !expression.isDistinct && expression.filter.isEmpty &&
      (expression.aggregateFunction match {
        case Count(Seq(_: Attribute)) | Count(Seq(_: Literal)) => true
        case _ => false
      })
  }

  private def safeProjection(expression: Expression): Boolean = expression match {
    case _: Attribute | _: Literal => true
    case Substring(_: Attribute, Literal(_: Int, _), Literal(_: Int, _)) => true
    case _ => false
  }

  private def underlyingJoin(plan: LogicalPlan): Option[Join] = plan match {
    case join: Join if join.joinType == Inner && join.hint == JoinHint.NONE => Some(join)
    case Project(output, child) if output.forall(_.isInstanceOf[Attribute]) => underlyingJoin(child)
    case Project(output, join: Join) if join.joinType == Inner && join.hint == JoinHint.NONE =>
      val aliases = output.collect { case alias: Alias => alias }
      val supportedOutput = output.forall {
        case _: Attribute => true
        case alias: Alias => safeProjection(alias.child) &&
          (alias.references.subsetOf(join.left.outputSet) ||
            alias.references.subsetOf(join.right.outputSet))
        case _ => false
      }
      if (!supportedOutput) None else {
        val (left, right) = aliases.partition(_.references.subsetOf(join.left.outputSet))
        def add(plan: LogicalPlan, aliases: Seq[Alias]): LogicalPlan = {
          if (aliases.isEmpty) plan else Project(plan.output ++ aliases, plan)
        }
        Some(join.copy(left = add(join.left, left), right = add(join.right, right)))
      }
    case _ => None
  }

  private[rapids] def rewrite(aggregate: Aggregate): Option[LogicalPlan] = {
    val outputsSupported = aggregate.aggregateExpressions.forall {
      case _: Attribute => true
      case Alias(_: Attribute, _) | Alias(_: Literal, _) => true
      case Alias(expression: AggregateExpression, _) => supported(expression)
      case _ => false
    }
    // An ungrouped COUNT already produces one partial buffer per partition.
    // Grouping it by join keys adds an exchange without reducing the final work.
    if (aggregate.groupingExpressions.isEmpty ||
        aggregate.getTagValue(planned).contains(true) || aggregate.isStreaming ||
        // AQE stages hide original filters. Preserve an initial selection, but never turn
        // a rejected input into a new candidate using the stage's stale row estimates.
        (aggregate.exists(_.isInstanceOf[LogicalQueryStage]) &&
          !aggregate.getTagValue(initiallySelected).contains(true)) ||
        !outputsSupported || !aggregate.groupingExpressions.forall(_.isInstanceOf[Attribute]) ||
        aggregate.exists(_.expressions.exists(!_.deterministic))) {
      None
    } else {
      val expressions = aggregate.aggregateExpressions.flatMap(_.collect {
        case expression: AggregateExpression => expression
      })
      underlyingJoin(aggregate.child).flatMap { join =>
        if (expressions.isEmpty || !aggregate.references.subsetOf(join.outputSet)) None else {
          val arguments = AttributeSet(expressions.flatMap(_.references))
          val choices = Seq((join.left, join.right, true), (join.right, join.left, false))
          choices.iterator.flatMap { case (fact, dimension, factOnLeft) =>
            val conditions = join.condition.toSeq.flatMap(splitConjunctivePredicates)
            val keys = conditions.map {
              case EqualTo(a: Attribute, b: Attribute)
                  if fact.outputSet.contains(a) && dimension.outputSet.contains(b) => Some((a, b))
              case EqualTo(a: Attribute, b: Attribute)
                  if fact.outputSet.contains(b) && dimension.outputSet.contains(a) => Some((b, a))
              case _ => None
            }
            if (!arguments.subsetOf(fact.outputSet) || keys.isEmpty || keys.exists(_.isEmpty)) {
              None
            } else {
              val grouping = (keys.flatten.map(_._1) ++ aggregate.groupingExpressions.collect {
                case a: Attribute if fact.outputSet.contains(a) => a
              }).distinct
              val counts = grouping.map { key =>
                fact.stats.attributeStats.get(key).flatMap(_.distinctCount)
              }
              // Runtime scalar predicates can leave pre-filter row counts in CBO statistics.
              // Without their selectivity, an apparent reduction can instead add an expensive
              // matched-key scan and aggregation to an already small input.
              val worthwhile = !unknownScalarSelectivity(fact) &&
                counts.forall(_.isDefined) && fact.stats.rowCount.exists { rows =>
                counts.flatten.product * minimumReduction < rows
              }
              if (!worthwhile) None else {
                // Only count keys that can contribute to the original inner join. In ANSI
                // mode this prevents an unmatched fact group from introducing a new overflow.
                // Every remaining fact group contributes to at least one original output group.
                val dimensionKeys = Project(keys.flatten.map(_._2).distinct, dimension)
                val matched = Join(fact, dimensionKeys, LeftSemi, join.condition, JoinHint.NONE)
                val partialOutputs = expressions.zipWithIndex.map { case (expression, index) =>
                  Alias(expression.copy(resultId = NamedExpression.newExprId),
                    s"pre_join_count_$index")()
                }
                val partial = Aggregate(grouping, grouping ++ partialOutputs, matched)
                val replacements = expressions.map(_.resultId).zip(
                  partialOutputs.map(_.toAttribute)).toMap
                val outputs = aggregate.aggregateExpressions.map { expression =>
                  expression.transformUp {
                    case count: AggregateExpression =>
                      // COUNT buffers are longs. SUM uses the same session ANSI arithmetic;
                      // coalesce retains COUNT's non-null output contract.
                      val sum = count.copy(aggregateFunction = Sum(replacements(count.resultId)))
                      Coalesce(Seq(sum, Literal(0L)))
                  }.asInstanceOf[NamedExpression]
                }
                val joined = if (factOnLeft) join.copy(left = partial)
                  else join.copy(right = partial)
                val result = aggregate.copy(aggregateExpressions = outputs, child = joined)
                aggregate.setTagValue(initiallySelected, true)
                partial.setTagValue(planned, true)
                result.setTagValue(planned, true)
                val threshold = join.conf.autoBroadcastJoinThreshold
                Some(result.transformAllExpressions {
                  case pruning: DynamicPruningSubquery if pruning.onlyInBroadcast &&
                      pruning.buildQuery.stats.sizeInBytes <= threshold =>
                    pruning.copy(onlyInBroadcast = false)
                })
              }
            }
          }.take(1).toSeq.headOption
        }
      }
    }
  }
}

class GpuCountPreJoinStrategy extends SparkStrategy {
  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case aggregate: Aggregate => GpuCountPreJoinReorder.rewrite(aggregate)
      .map(rewritten => Seq(planLater(rewritten))).getOrElse(Nil)
    case _ => Nil
  }
}
