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
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeMap, BinaryComparison, Expression, IsNotNull, IsNull, Literal, NamedExpression, Not, Or, UnscaledValue}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Average, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.plans.{Cross, Inner}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.LogicalRelation

/** Share one scan among independent, one-row aggregates over the same relation. */
class GpuAggregateFusionOverrides(spark: SparkSession) extends Rule[LogicalPlan] {
  private def branches(plan: LogicalPlan): Seq[LogicalPlan] = plan match {
    case Join(left, right, Inner | Cross, None, _) => branches(left) ++ branches(right)
    case _ => Seq(plan)
  }

  // Joining once can evaluate a branch predicate on rows from another branch. Keep the new
  // joined-input case restricted to total boolean expressions and non-throwing COUNT arguments.
  private def safePredicate(expression: Expression): Boolean = expression match {
    case _: Attribute | _: Literal => true
    case _: And | _: Or | _: Not | _: IsNull | _: IsNotNull | _: BinaryComparison =>
      expression.children.forall(safePredicate)
    case _ => false
  }

  private def source(plan: LogicalPlan): Option[(LogicalPlan, Seq[Expression])] = plan match {
    case Project(expressions, child) if expressions.forall(_.isInstanceOf[Attribute]) =>
      source(child)
    case Filter(condition, child) if condition.deterministic =>
      source(child).map { case (base, filters) => (base, filters :+ condition) }
    case relation: LogicalRelation if !relation.isStreaming => Some((relation, Seq.empty))
    case join @ Join(left, right, Inner | Cross, condition, hint)
        if hint == JoinHint.NONE && condition.forall(safePredicate) =>
      for {
        l <- source(left)
        r <- source(right)
      } yield (join.copy(left = l._1, right = r._1), l._2 ++ r._2)
    case _ => None
  }

  // A GPU aggregate can project arguments before applying its FILTER. Restrict arguments
  // to values that are safe on rows excluded by a branch's original WHERE clause.
  private def safeArgument(expression: Expression): Boolean = expression match {
    case _: Attribute | _: Literal => true
    case UnscaledValue(_: Attribute) => true
    case _ => false
  }

  private def fuse(join: Join): LogicalPlan = {
    val inputs = branches(join)
    val aggregates = inputs.collect {
      case a: Aggregate if a.groupingExpressions.isEmpty &&
          a.aggregateExpressions.forall(_.deterministic) => a
    }
    val sources = aggregates.flatMap(a => source(a.child))
    if (inputs.size < 2 || aggregates.size != inputs.size || sources.size != inputs.size ||
        !sources.tail.forall(_._1.sameResult(sources.head._1))) {
      join
    } else {
      val base = sources.head._1
      val expressions = aggregates.zip(sources).flatMap { case (aggregate, (relation, filters)) =>
        val attributes = AttributeMap(relation.output.zip(base.output))
        def remap(expression: Expression): Expression = expression.transformDown {
          case a: Attribute if attributes.contains(a) => attributes(a)
        }
        val predicate = filters.map(remap).reduceOption(And)
        aggregate.aggregateExpressions.map { expression =>
          remap(expression).transformDown {
            case ae: AggregateExpression =>
              val filter = (predicate.toSeq ++ ae.filter.toSeq).reduceOption(And)
              ae.copy(filter = filter)
          }.asInstanceOf[NamedExpression]
        }
      }
      val functions = expressions.flatMap(_.collect { case ae: AggregateExpression => ae })
      val distinctArguments = functions.filter(_.isDistinct)
        .map(_.aggregateFunction.children.map(_.canonicalized)).distinct
      val joined = base.exists(_.isInstanceOf[Join])
      val safeJoined = !joined || (sources.forall(_._2.forall(safePredicate)) &&
        functions.forall(ae => ae.aggregateFunction.isInstanceOf[Count] &&
          ae.filter.forall(safePredicate)))
      val safe = safeJoined && functions.nonEmpty && functions.forall { ae =>
        val supported = ae.aggregateFunction match {
          case _: Average | _: Sum | _: Count | _: Min | _: Max => true
          case _ => false
        }
        supported && ae.aggregateFunction.children.forall(safeArgument)
      } && expressions.forall(_.references.subsetOf(base.outputSet)) &&
        distinctArguments.size <= 1
      if (safe) {
        val predicates = sources.map { case (relation, filters) =>
          val attributes = AttributeMap(relation.output.zip(base.output))
          filters.reduceOption(And).getOrElse(Literal.TrueLiteral).transformDown {
            case a: Attribute if attributes.contains(a) => attributes(a)
          }
        }
        aggregates.head.copy(aggregateExpressions = expressions,
          child = Filter(predicates.reduce(Or), base))
      } else {
        join
      }
    }
  }

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!new RapidsConf(spark.sessionState.conf).isSqlEnabled) {
      plan
    } else {
      plan.transformDown {
        case join: Join if join.condition.isEmpty => fuse(join)
      }
    }
  }
}
