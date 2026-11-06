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

import scala.util.control.NonFatal

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Average, Complete, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.types.{Decimal, DecimalType}

/** Share a grouped CTE used by a filtered result and its unfiltered scalar AVG. */
class GpuScalarAggregateSharingOverrides(spark: SparkSession) extends Rule[LogicalPlan] {
  private val planned = TreeNodeTag[Boolean]("rapids.scalarAggregateSharingPlanned")
  private val countType = DecimalType(38, 0)
  // The global window is only used for a bounded, actually observed number of groups.
  private val maximumWindowGroups = 1000000L

  private def strip(plan: LogicalPlan): LogicalPlan = plan match {
    case a: SubqueryAlias => strip(a.child)
    case p => p
  }

  private def safeExpression(e: Expression): Boolean = e match {
    case _: Attribute | _: Literal => true
    case _: And | _: Or | _: Not | _: IsNull | _: IsNotNull | _: BinaryComparison |
        _: Upper => e.children.forall(safeExpression)
    case c: Cast if c.child.isInstanceOf[Literal] =>
      try { c.eval(); true } catch { case NonFatal(_) => false }
    case _ => false
  }

  private def safeInput(plan: LogicalPlan): Boolean = strip(plan) match {
    case p: Project => p.projectList.forall(_.isInstanceOf[Attribute]) && safeInput(p.child)
    case f: Filter => safeExpression(f.condition) && safeInput(f.child)
    case j: Join => j.joinType == Inner && j.hint == JoinHint.NONE &&
      j.condition.forall(safeExpression) && safeInput(j.left) && safeInput(j.right)
    case leaf if leaf.children.isEmpty => leaf.deterministic && !leaf.isStreaming &&
      leaf.expressions.forall(safeExpression)
    case _ => false
  }

  private def references(plan: LogicalPlan): Seq[CTERelationRef] = {
    val direct = plan match { case r: CTERelationRef => Seq(r); case _ => Nil }
    direct ++ plan.children.flatMap(references) ++ plan.expressions.flatMap(_.collect {
      case s: ScalarSubquery => references(s.plan)
    }.flatten)
  }

  private def literal(value: BigInt, scale: Int): Literal = Literal.create(
    Decimal(new java.math.BigDecimal(value.bigInteger, scale)), DecimalType(38, scale))

  private def sumExpression(e: NamedExpression): Option[(Alias, AggregateExpression, Attribute)] =
    e match {
      case alias @ Alias(a: AggregateExpression, _) => a.aggregateFunction match {
        case s: Sum if a.mode == Complete && !a.isDistinct && a.filter.isEmpty &&
            s.evalMode == EvalMode.ANSI => s.child match {
          case v: Attribute if v.dataType.isInstanceOf[DecimalType] &&
              v.dataType.asInstanceOf[DecimalType].precision <= 18 => Some((alias, a, v))
          case _ => None
        }
        case _ => None
      }
      case _ => None
    }

  private def build(cte: WithCTE, grouped: Aggregate, sum: Alias, value: Attribute,
      having: Filter, outer: Aggregate, selected: Filter, mainRef: CTERelationRef,
      scalar: ScalarSubquery, scalarAggregate: Aggregate, avg: AggregateExpression,
      scalarRef: CTERelationRef): LogicalPlan = {
    val inputType = value.dataType.asInstanceOf[DecimalType]
    val sumType = sum.dataType.asInstanceOf[DecimalType]
    val bound = BigInt(10).pow(sumType.precision) - 1
    val cap = bound / (BigInt(10).pow(inputType.precision) - 1)
    val n = Alias(Count(Seq(Literal(1))).toAggregateExpression(), "shared_group_count")()
    val wide = Alias(Sum(Cast(value, DecimalType(38, inputType.scale)))
      .toAggregateExpression(), "shared_group_wide_sum")()
    val keys = grouped.aggregateExpressions.filterNot(_.exprId == sum.exprId)
    val partial = Aggregate(grouped.groupingExpressions, keys ++ Seq(wide, n), grouped.child)
    val total = Sum(Cast(n.toAttribute, countType)).toAggregateExpression()
    val groups = Count(Seq(Literal(1))).toAggregateExpression()
    val minimum = Min(wide.toAttribute).toAggregateExpression()
    val maximum = Max(wide.toAttribute).toAggregateExpression()
    val inRange = Coalesce(Seq(And(
      GreaterThanOrEqual(minimum, literal(-bound, inputType.scale)),
      LessThanOrEqual(maximum, literal(bound, inputType.scale))), Literal(true)))
    val safe = Alias(And(And(
      LessThanOrEqual(Coalesce(Seq(total, literal(BigInt(0), 0))), literal(cap, 0)),
      LessThanOrEqual(groups, Literal(maximumWindowGroups))), inRange),
      "scalar_sharing_safe")()
    val rawGuard = ScalarSubquery(Aggregate(Nil, Seq(safe), partial))
    val Subquery(optimizedGuard, _) = spark.sessionState.optimizer.execute(
      Subquery.fromExpression(rawGuard))
    val guard = rawGuard.withNewPlan(optimizedGuard)
    val localRange = Coalesce(Seq(And(
      GreaterThanOrEqual(wide.toAttribute, literal(-bound, inputType.scale)),
      LessThanOrEqual(wide.toAttribute, literal(bound, inputType.scale))), Literal(true)))
    val filtered = Filter(And(guard, And(
      LessThanOrEqual(Cast(n.toAttribute, countType), literal(cap, 0)), localRange)), partial)
    val restored = Project(grouped.aggregateExpressions.map { e =>
      if (e.exprId == sum.exprId) {
        Alias(Cast(wide.toAttribute, sumType), e.name)(exprId = e.exprId)
      } else e.toAttribute
    }, filtered)
    val scalarMapping = AttributeMap(scalarRef.output.zip(grouped.output))
    val mainMapping = AttributeMap(mainRef.output.zip(grouped.output))
    val mappedAverage = avg.transform { case a: Attribute if scalarMapping.contains(a) =>
      scalarMapping(a)
    }
    val window = Alias(WindowExpression(mappedAverage, WindowSpecDefinition(Nil, Nil,
      SpecifiedWindowFrame(RowFrame, UnboundedPreceding, UnboundedFollowing))),
      "shared_global_average")()
    val windowed = Window(Seq(window), Nil, Nil, restored)
    val thresholdExpression = scalarAggregate.aggregateExpressions.head
      .asInstanceOf[Alias].child.transform { case a: AggregateExpression if a == avg =>
        window.toAttribute
      }
    val threshold = Alias(thresholdExpression, "shared_global_threshold")()
    val thresholdInput = Project(restored.output :+ threshold, windowed)
    val selectedInput = Filter(selected.condition.transform {
      case a: Attribute if mainMapping.contains(a) => mainMapping(a)
    }, thresholdInput)
    val perGroup = Alias(Max(threshold.toAttribute).toAggregateExpression(),
      "shared_threshold_per_group")()
    val result = outer.copy(
      groupingExpressions = outer.groupingExpressions.map(_.transform {
        case a: Attribute if mainMapping.contains(a) => mainMapping(a)
      }),
      aggregateExpressions = outer.aggregateExpressions.map { original =>
        val mapped = original.transform {
          case a: Attribute if mainMapping.contains(a) => mainMapping(a)
        }.asInstanceOf[NamedExpression]
        if (mapped.exprId == original.exprId) mapped else {
          Alias(mapped, original.name)(exprId = original.exprId)
        }
      } :+ perGroup,
      child = selectedInput)
    val resultFilter = Filter(having.condition.transform {
      case s: ScalarSubquery if s.exprId == scalar.exprId => perGroup.toAttribute
    }, result)
    val replacement = Project(outer.output, resultFilter)
    val originalDef = cte.cteDefs.head
    val fallbackDef = originalDef.copy(child = originalDef.child.transformDown {
      case a: Aggregate if a eq grouped => a.copy(child = Filter(Not(guard), a.child))
    })
    val fallback = cte.copy(plan = having, cteDefs = Seq(fallbackDef))
    fallback.setTagValue(planned, true)
    val union = Union(Seq(replacement, fallback))
    union.setTagValue(planned, true)
    // Retain the original outer ORDER BY/LIMIT above the union of mutually exclusive branches.
    cte.plan.transformUp { case f: Filter if f eq having => union }
  }

  private[rapids] def rewrite(cte: WithCTE): Option[LogicalPlan] = {
    if (cte.getTagValue(planned).contains(true) || cte.cteDefs.size != 1 ||
        !cte.resolved || cte.isStreaming || !cte.deterministic) {
      return None
    }
    val grouped = strip(cte.cteDefs.head.child) match {
      case a: Aggregate if a.groupingExpressions.nonEmpty &&
          a.groupingExpressions.forall(_.isInstanceOf[Attribute]) && safeInput(a.child) => a
      case _ => return None
    }
    val sums = grouped.aggregateExpressions.flatMap(sumExpression)
    if (sums.size != 1 || grouped.aggregateExpressions.exists(e =>
        e.exprId != sums.head._1.exprId && !e.isInstanceOf[Attribute])) {
      return None
    }
    val (sum, _, value) = sums.head
    val refs = references(cte.plan)
    if (refs.size != 2 || refs.exists(_.cteId != cte.cteDefs.head.id)) return None
    val matches = cte.plan.collect {
      case h: Filter if h.child.isInstanceOf[Aggregate] &&
          h.child.asInstanceOf[Aggregate].child.isInstanceOf[Filter] =>
        val a = h.child.asInstanceOf[Aggregate]
        (h, a, a.child.asInstanceOf[Filter])
    }.flatMap { case (having, outer, selected) =>
      val scalars = having.condition.collect { case s: ScalarSubquery => s }
      if (scalars.size != 1 || outer.groupingExpressions.isEmpty ||
          !safeExpression(selected.condition) ||
          !outer.groupingExpressions.forall(_.isInstanceOf[Attribute])) {
        None
      } else {
        val scalar = scalars.head
        (strip(selected.child), strip(scalar.plan)) match {
          case (mainRef: CTERelationRef, a: Aggregate) if a.groupingExpressions.isEmpty &&
              a.aggregateExpressions.size == 1 && a.aggregateExpressions.head.isInstanceOf[Alias]
              && scalar.outerAttrs.isEmpty => strip(a.child) match {
            case scalarRef: CTERelationRef =>
              val averages = a.aggregateExpressions.head.collect {
                case e: AggregateExpression if e.mode == Complete && !e.isDistinct &&
                    e.filter.isEmpty && e.aggregateFunction.isInstanceOf[Average] => e
              }
              val inputIndex = grouped.output.indexWhere(_.exprId == sum.exprId)
              val validAverage = averages.size == 1 && averages.head.aggregateFunction
                .children == Seq(scalarRef.output(inputIndex))
              val validThreshold = a.aggregateExpressions.head.asInstanceOf[Alias].child match {
                case m: Multiply => Seq((m.left, m.right), (m.right, m.left)).exists {
                  case (Literal(d: Decimal, _: DecimalType), e: AggregateExpression) =>
                    d.toJavaBigDecimal.abs().compareTo(java.math.BigDecimal.ONE) <= 0 &&
                      averages.contains(e)
                  case _ => false
                }
                case _ => false
              }
              if (validAverage && validThreshold) Some(build(cte, grouped, sum, value, having,
                outer, selected, mainRef, scalar, a, averages.head, scalarRef)) else None
            case _ => None
          }
          case _ => None
        }
      }
    }
    if (matches.size == 1) matches.headOption else None
  }

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!new RapidsConf(spark.sessionState.conf).isSqlEnabled) plan else {
      plan.transformDown { case cte: WithCTE => rewrite(cte).getOrElse(cte) }
    }
  }
}
