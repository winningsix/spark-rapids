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
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.plans.LeftSemi
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.types.{Decimal, DecimalType, IntegerType, LongType}

/** Prune fine rollup groups using exact coarse upper bounds, with a runtime fallback. */
class GpuRollupRankPruningOverrides(spark: SparkSession) extends Rule[LogicalPlan] {
  private val injected = TreeNodeTag[Boolean]("rapids.rollupRankPruningInjected")
  private val minimumInputRows = BigInt(1000000)

  private def mark[T <: LogicalPlan](plan: T): T = {
    plan.setTagValue(injected, true)
    plan
  }

  private def rewrite(filter: Filter): Option[LogicalPlan] = {
    if (filter.getTagValue(injected).contains(true)) {
      None
    } else {
      (filter.condition, filter.child) match {
        case (LessThanOrEqual(rankAttribute: Attribute, Literal(limit: Int, IntegerType)),
            window: Window) if limit > 0 &&
            limit <= spark.sessionState.conf.windowGroupLimitThreshold &&
            window.windowExpressions.size == 1 && window.partitionSpec.size == 1 &&
            window.orderSpec.size == 1 =>
          val ranked = window.windowExpressions.head match {
            case alias @ Alias(WindowExpression(_: Rank, _), _) =>
              alias.toAttribute.semanticEquals(rankAttribute)
            case _ => false
          }
          val orderedSum = window.orderSpec.head match {
            case SortOrder(attribute: Attribute, Descending, NullsLast, _) => Some(attribute)
            case _ => None
          }
          val child = window.child match {
            case groupLimit: WindowGroupLimit => groupLimit.child
            case other => other
          }
          (ranked, orderedSum, child) match {
            case (true, Some(sumAttribute), aggregate: Aggregate) => aggregate.child match {
              case expand: Expand =>
                analyze(filter, window, aggregate, expand, sumAttribute, limit)
              case _ => None
            }
            case _ => None
          }
        case _ => None
      }
    }
  }

  private def analyze(filter: Filter, window: Window, aggregate: Aggregate, expand: Expand,
      sumAttribute: Attribute, limit: Int): Option[LogicalPlan] = {
    val groups = aggregate.groupingExpressions
    val source = expand.child
    val countsKnownLarge = source.stats.rowCount.exists(_ >= minimumInputRows)
    val attributes = groups.forall(_.isInstanceOf[AttributeReference])
    val groupCount = groups.size
    if (!countsKnownLarge || !attributes || groupCount < 5 || groupCount > 17 ||
        source.isStreaming || source.exists(_.expressions.exists(!_.deterministic)) ||
        expand.projections.size != groupCount ||
        !expand.projections.forall(_.size == expand.output.size)) {
      None
    } else {
      val keys = groups.dropRight(1).map(_.asInstanceOf[AttributeReference])
      val gid = groups.last.asInstanceOf[AttributeReference]
      val sums = aggregate.aggregateExpressions.collect {
        case alias @ Alias(expression: AggregateExpression, _) => (alias, expression)
      }
      val expectedOutput = keys.map(_.exprId)
      val shape = sums.size == 1 && aggregate.aggregateExpressions.size == keys.size + 1 &&
        aggregate.aggregateExpressions.take(keys.size).map(_.exprId) == expectedOutput &&
        aggregate.aggregateExpressions.take(keys.size).forall(_.isInstanceOf[AttributeReference]) &&
        window.partitionSpec.head.semanticEquals(keys.head)
      if (!shape) {
        None
      } else {
        val (sumAlias, expression) = sums.head
        val sum = expression.aggregateFunction
        val sumSupported = sum.isInstanceOf[Sum] &&
          sum.asInstanceOf[Sum].evalMode == EvalMode.ANSI &&
          sum.dataType.isInstanceOf[DecimalType] &&
          sum.children.head.dataType.isInstanceOf[DecimalType] && sum.deterministic &&
          expression.mode == Complete && !expression.isDistinct && expression.filter.isEmpty &&
          sumAlias.toAttribute.semanticEquals(sumAttribute)
        val positions = (keys :+ gid).map(a => expand.output.indexWhere(_.exprId == a.exprId))
        if (!sumSupported || positions.exists(_ < 0) || positions.distinct.size != groupCount) {
          None
        } else {
          val first = expand.projections.head
          val chain = expand.projections.zipWithIndex.forall { case (projection, level) =>
            projection(positions.last) == Literal((1L << level) - 1L) &&
              keys.indices.forall { index =>
                val value = projection(positions(index))
                if (index < keys.size - level) {
                  first(positions(index)).isInstanceOf[AttributeReference] &&
                    value.semanticEquals(first(positions(index)))
                } else value == Literal.create(null, keys(index).dataType)
              }
          }
          val inputSum = sum.children.head
          val invariant = inputSum.references.forall { attribute =>
            val index = expand.output.indexWhere(_.exprId == attribute.exprId)
            index >= 0 && expand.projections.forall(_(index).semanticEquals(first(index))) &&
              first(index).references.subsetOf(source.outputSet)
          }
          if (!chain || !invariant) {
            None
          } else {
            val measure = inputSum.transform {
              case attribute: AttributeReference =>
                first(expand.output.indexWhere(_.exprId == attribute.exprId))
            }
            val inputKeys = positions.dropRight(1).map(first)
            Some(build(filter, window, aggregate, expand, keys, gid, positions,
              measure, inputKeys))
          }
        }
      }
    }
  }

  private def build(filter: Filter, window: Window, aggregate: Aggregate, expand: Expand,
      keys: Seq[AttributeReference], gid: AttributeReference, positions: Seq[Int],
      measure: Expression, inputKeys: Seq[Expression]): LogicalPlan = {
    val source = expand.child
    val prefix = (keys.size + 1) / 2
    val firstFineLevel = keys.size - prefix
    val sourceCategory = inputKeys.head
    val minimum = Alias(Min(measure).toAggregateExpression(), "rollup_minimum")()
    val count = Alias(Count(Seq(Literal(1))).toAggregateExpression(), "rollup_count")()
    val categoryGuard = Aggregate(Seq(sourceCategory), Seq(minimum, count), source)
    // At most keyCount groups per source row belong to its category, plus one grand total
    // in the NULL category. Bound the rank partition using actual counts, never CBO estimates.
    val maximumCategoryRows = (Int.MaxValue.toLong - 1L) / (keys.size + 1)
    val safe = Alias(Coalesce(Seq(And(
      GreaterThanOrEqual(Min(minimum.toAttribute).toAggregateExpression(),
        Literal.create(Decimal(0), measure.dataType)),
      LessThanOrEqual(Max(count.toAttribute).toAggregateExpression(),
        Literal(maximumCategoryRows))),
      Literal(false))), "rollup_nonnegative_bounded")()
    val guard = ScalarSubquery(Aggregate(Nil, Seq(safe), categoryGuard), Nil)
    val active = Filter(guard, source)
    val fallbackInput = Filter(Not(guard), source)

    // Drop constant NULL suffix keys so the existing hierarchical GPU rollup still applies.
    val removed = positions.slice(prefix, keys.size).toSet
    val kept = expand.output.indices.filterNot(removed.contains)
    val coarseExpand = expand.copy(
      projections = expand.projections.drop(firstFineLevel).map(p => kept.map(p)),
      output = kept.map(expand.output), child = active)
    val coarseOutput = aggregate.aggregateExpressions.zipWithIndex.map { case (output, index) =>
      if (index >= prefix && index < keys.size) {
        Alias(Literal.create(null, output.dataType), output.name)(exprId = output.exprId)
      } else output
    }
    val coarse = aggregate.copy(groupingExpressions = keys.take(prefix) :+ gid,
      aggregateExpressions = coarseOutput :+ gid, child = coarseExpand)
    val coarseChild = window.child match {
      case groupLimit: WindowGroupLimit => groupLimit.copy(child = coarse)
      case _ => coarse
    }
    val coarseTop = mark(filter.copy(child = window.copy(child = coarseChild)))
    val parentKeys = keys.take(prefix).zipWithIndex.map { case (key, index) =>
      Alias(key, s"rollup_parent_key_$index")()
    }
    // A fine sum cannot exceed its nonnegative parent sum. If the parent has rank > K
    // among coarse rows, K coarse rows already outrank every child. RANK keeps boundary
    // ties; NULL parents remain when fewer than K non-NULL coarse rows exist.
    // Select by grouping id so real NULL keys cannot match a different rollup level.
    val parentGid = expand.projections(firstFineLevel)(positions.last)
    val allowed = Project(parentKeys, Filter(EqualTo(gid, parentGid), coarseTop))
    val matchesParent = inputKeys.take(prefix).zip(parentKeys).map { case (key, parent) =>
      EqualNullSafe(key, parent.toAttribute): Expression
    }.reduce(And)
    val eligible = Join(active, allowed, LeftSemi, Some(matchesParent), JoinHint.NONE)
    val fine = aggregate.copy(child = expand.copy(
      projections = expand.projections.take(firstFineLevel), child = eligible))
    val fallback = aggregate.copy(child = expand.copy(child = fallbackInput))
    // Guard outputs too: an inactive empty grouping-set branch must emit no extra row.
    // Adding fine rows can only worsen a coarse row's rank, so coarse losers can be dropped.
    val keptCoarse = Project(aggregate.output, coarseTop)
    val union = Union(Seq(Filter(guard, keptCoarse), Filter(guard, fine),
      Filter(Not(guard), fallback)))
    val newChild = window.child match {
      case groupLimit: WindowGroupLimit => groupLimit.copy(child = union)
      case _ => union
    }
    mark(filter.copy(child = window.copy(child = newChild)))
  }

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val conf = spark.sessionState.conf
    if (!new RapidsConf(conf).isSqlEnabled || !conf.ansiEnabled) {
      plan
    } else {
      plan.transformDown { case filter: Filter => rewrite(filter).getOrElse(filter) }
    }
  }
}
