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
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, AggregateFunction, Complete, Max, Min}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.types._

/** Reduce repeated fact keys before joining dimension attributes into a MIN/MAX aggregate. */
private[rapids] object GpuMinMaxPreJoinReorder extends PredicateHelper {
  private val planned = TreeNodeTag[Boolean]("rapids.minMaxPreJoinPlanned")
  // Require a substantial estimated reduction to justify the additional aggregation.
  private val minimumReduction = 4

  private def exactOrderedType(dataType: DataType): Boolean = dataType match {
    case ByteType | ShortType | IntegerType | LongType | DateType | TimestampType => true
    case _: DecimalType => true
    case _ => false
  }

  private def supported(expression: AggregateExpression): Boolean = {
    expression.mode == Complete && !expression.isDistinct && expression.filter.isEmpty &&
      (expression.aggregateFunction match {
        case Min(a: Attribute) => exactOrderedType(a.dataType)
        case Max(a: Attribute) => exactOrderedType(a.dataType)
        case _ => false
      })
  }

  private def underlyingJoin(plan: LogicalPlan): Option[Join] = plan match {
    case Project(output, child) if output.forall(_.isInstanceOf[Attribute]) => underlyingJoin(child)
    case join: Join if join.joinType == Inner && join.hint == JoinHint.NONE => Some(join)
    case _ => None
  }

  private[rapids] def rewrite(aggregate: Aggregate): Option[LogicalPlan] = {
    val outputsSupported = aggregate.aggregateExpressions.forall {
      case _: Attribute => true
      case Alias(_: Attribute, _) | Alias(_: Literal, _) => true
      case Alias(expression: AggregateExpression, _) => supported(expression)
      case _ => false
    }
    if (aggregate.getTagValue(planned).contains(true) || aggregate.isStreaming ||
        !outputsSupported || !aggregate.groupingExpressions.forall(_.isInstanceOf[Attribute]) ||
        aggregate.exists(_.expressions.exists(!_.deterministic))) {
      None
    } else {
      val expressions = aggregate.aggregateExpressions.flatMap(_.collect {
        case expression: AggregateExpression => expression
      })
      underlyingJoin(aggregate.child).flatMap { join =>
        if (expressions.isEmpty || !aggregate.references.subsetOf(join.outputSet)) {
          None
        } else {
          val arguments = AttributeSet(expressions.flatMap(_.references))
          val choices = Seq((join.left, join.right, true), (join.right, join.left, false))
          choices.iterator.flatMap { case (fact, dimension, factOnLeft) =>
            val conditions = join.condition.toSeq.flatMap(splitConjunctivePredicates)
            val keys = conditions.map {
              case EqualTo(a: Attribute, b: Attribute)
                  if fact.outputSet.contains(a) && dimension.outputSet.contains(b) => Some(a)
              case EqualTo(a: Attribute, b: Attribute)
                  if fact.outputSet.contains(b) && dimension.outputSet.contains(a) => Some(b)
              case _ => None
            }
            if (!arguments.subsetOf(fact.outputSet) || keys.isEmpty || keys.exists(_.isEmpty)) {
              None
            } else {
              val grouping = (keys.flatten ++ aggregate.groupingExpressions.collect {
                case a: Attribute if fact.outputSet.contains(a) => a
              }).distinct
              val distinctCounts = grouping.map { key =>
                fact.stats.attributeStats.get(key).flatMap(_.distinctCount)
              }
              val worthwhile = distinctCounts.forall(_.isDefined) && fact.stats.rowCount.exists {
                rows => distinctCounts.flatten.product * minimumReduction < rows
              }
              if (!worthwhile) {
                None
              } else {
                val partialOutputs = expressions.zipWithIndex.map { case (expression, index) =>
                  Alias(expression.copy(resultId = NamedExpression.newExprId),
                    s"pre_join_minmax_$index")()
                }
                val partial = Aggregate(grouping, grouping ++ partialOutputs, fact)
                val replacements = expressions.map(_.resultId).zip(
                  partialOutputs.map(_.toAttribute)).toMap
                val outputs = aggregate.aggregateExpressions.map { expression =>
                  expression.transform {
                    case agg: AggregateExpression =>
                      val function = agg.aggregateFunction.withNewChildren(
                        Seq(replacements(agg.resultId))).asInstanceOf[AggregateFunction]
                      agg.copy(aggregateFunction = function)
                  }.asInstanceOf[NamedExpression]
                }
                val joined = if (factOnLeft) join.copy(left = partial)
                  else join.copy(right = partial)
                val result = aggregate.copy(aggregateExpressions = outputs, child = joined)
                // Keep both levels; no key uniqueness assumption is needed, including
                // duplicate dimension rows and different keys with identical group values.
                partial.setTagValue(planned, true)
                result.setTagValue(planned, true)
                // Smaller fact output can change which side is broadcast. Keep small
                // pruning subqueries executable when their old broadcast is not reused.
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

class GpuMinMaxPreJoinStrategy extends SparkStrategy {
  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case aggregate: Aggregate => GpuMinMaxPreJoinReorder.rewrite(aggregate)
      .map(rewritten => Seq(planLater(rewritten))).getOrElse(Nil)
    case _ => Nil
  }
}
