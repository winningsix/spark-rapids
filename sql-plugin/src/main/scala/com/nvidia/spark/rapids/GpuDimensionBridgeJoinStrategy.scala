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
import org.apache.spark.sql.catalyst.optimizer.{CollapseProject, ColumnPruning, ReorderJoin, StarSchemaDetection}
import org.apache.spark.sql.catalyst.plans.{Inner, InnerLike}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.execution.datasources.LogicalRelation

/** Avoid multiplying a large fact input through a many-to-many dimension bridge. */
private[rapids] object GpuDimensionBridgeJoinReorder extends PredicateHelper {
  private val planned = TreeNodeTag[Boolean]("rapids.dimensionBridgeJoinPlanned")

  private def flatten(plan: LogicalPlan)
      : (Seq[(LogicalPlan, InnerLike)], Seq[Expression]) = plan match {
    case Join(left, right, Inner, condition, hint) if hint == JoinHint.NONE =>
      val (leftInputs, leftConditions) = flatten(left)
      val (rightInputs, rightConditions) = flatten(right)
      (leftInputs ++ rightInputs,
        leftConditions ++ rightConditions ++ condition.toSeq.flatMap(splitConjunctivePredicates))
    case Project(output, child) if output.forall(_.isInstanceOf[Attribute]) => flatten(child)
    case other => (Seq((other, Inner)), Seq.empty)
  }

  private def source(plan: LogicalPlan): Option[LogicalRelation] = plan match {
    case Project(output, child) if output.forall(_.isInstanceOf[Attribute]) => source(child)
    case Filter(condition, child) if condition.deterministic => source(child)
    case relation: LogicalRelation if !relation.isStreaming => Some(relation)
    case _ => None
  }

  // Moving a predicate must not introduce evaluation of a throwing expression on extra rows.
  private def safeExpression(expression: Expression): Boolean = expression match {
    case _: Attribute | _: Literal => true
    case DateAdd(_: Attribute, _: Literal) => true
    case comparison: BinaryComparison => comparison.children.forall(safeExpression)
    case And(left, right) => safeExpression(left) && safeExpression(right)
    case Or(left, right) => safeExpression(left) && safeExpression(right)
    case Not(child) => safeExpression(child)
    case IsNull(_: Attribute) | IsNotNull(_: Attribute) => true
    case _ => false
  }

  private[rapids] def rewrite(join: Join): Option[LogicalPlan] = {
    if (join.joinType != Inner || join.getTagValue(planned).contains(true)) {
      None
    } else {
      val (inputs, conditions) = flatten(join)
      val threshold = join.conf.autoBroadcastJoinThreshold
      val sources = inputs.map { case (input, _) => source(input) }
      val supported = threshold > 0 && inputs.size >= 3 && sources.forall(_.isDefined) &&
        inputs.count(_._1.stats.sizeInBytes > threshold) >= 2 &&
        conditions.nonEmpty && conditions.forall(safeExpression)
      if (!supported) {
        None
      } else {
        def smallNonUniqueOwner(attribute: Attribute): Option[Int] = {
          val owners = inputs.indices.filter { index =>
            val input = inputs(index)._1
            val relation = sources(index).get
            input.outputSet.contains(attribute) && input.stats.sizeInBytes <= threshold &&
              relation.stats.rowCount.exists { rows =>
                relation.stats.attributeStats.get(attribute).flatMap(_.distinctCount)
                  .exists(_ < rows)
              }
          }
          if (owners.size == 1) Some(owners.head) else None
        }
        val hasBridge = conditions.exists {
          case EqualTo(left: Attribute, right: Attribute) =>
            (smallNonUniqueOwner(left), smallNonUniqueOwner(right)) match {
              case (Some(l), Some(r)) => l != r
              case _ => false
            }
          case _ => false
        }
        val star = if (hasBridge) {
          StarSchemaDetection.reorderStarJoins(inputs, conditions)
        } else {
          Seq.empty
        }
        if (star.isEmpty) {
          None
        } else {
          val rest = inputs.filterNot(star.contains(_))
          val reordered = ReorderJoin.createOrderedJoin(star ++ rest, conditions)
          // The old dimension bridge's broadcast may disappear after reordering. Retain
          // its small pruning subquery independently, so fact partition pruning survives.
          val withPruning = reordered.transformAllExpressions {
            case pruning: DynamicPruningSubquery if pruning.onlyInBroadcast &&
                pruning.buildQuery.stats.sizeInBytes <= threshold =>
              pruning.copy(onlyInBroadcast = false)
          }
          val result = CollapseProject(ColumnPruning(Project(join.output, withPruning)))
          // All input leaves are table accesses, so these joins are newly constructed.
          // Keep planner recursion from applying the same associative rewrite again.
          result.foreach {
            case node: Join => node.setTagValue(planned, true)
            case _ =>
          }
          Some(result)
        }
      }
    }
  }
}

/** Run after CBO so it cannot immediately restore the dimension-first join ordering. */
class GpuDimensionBridgeJoinStrategy extends SparkStrategy {
  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case join: Join => GpuDimensionBridgeJoinReorder.rewrite(join)
      .map(reordered => Seq(planLater(reordered))).getOrElse(Nil)
    case _ => Nil
  }
}
