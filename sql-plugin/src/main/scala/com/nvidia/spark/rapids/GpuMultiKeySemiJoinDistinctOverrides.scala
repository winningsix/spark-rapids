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
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, InSubquery, NamedExpression}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, LogicalPlan, Project, SubqueryAlias}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.types.{ByteType, IntegerType, LongType, ShortType}

/** Reduce a large, duplicate-heavy multi-key semi-join build before choosing its transport. */
class GpuMultiKeySemiJoinDistinctOverrides(spark: SparkSession) extends Rule[LogicalPlan] {
  // Require a substantial reduction before paying for an additional aggregation stage.
  private val minimumBuildRows = BigInt(1000000)
  private val minimumDuplicateRatio = 8

  private def sourceAttribute(expression: NamedExpression): Option[Attribute] = expression match {
    case attribute: Attribute => Some(attribute)
    case Alias(attribute: Attribute, _) => Some(attribute)
    case _ => None
  }

  // An unrelated grouping expression can make Spark drop all aggregate statistics.
  // Recover only unchanged grouping keys, never statistics for computed values.
  private def keyStatistics(plan: LogicalPlan, key: Attribute): Option[BigInt] = {
    plan.stats.attributeStats.get(key).flatMap(_.distinctCount).orElse {
      plan match {
        case Project(expressions, child) =>
          expressions.find(_.exprId == key.exprId).flatMap(sourceAttribute)
            .flatMap(keyStatistics(child, _))
        case Filter(_, child) => keyStatistics(child, key)
        case alias: SubqueryAlias => keyStatistics(alias.child, key)
        case join: Join if join.joinType == Inner || join.joinType == LeftSemi =>
          join.children.find(_.outputSet.contains(key)).flatMap(keyStatistics(_, key))
        case aggregate: Aggregate =>
          aggregate.aggregateExpressions.find(_.exprId == key.exprId).flatMap(sourceAttribute)
            .filter(attribute => aggregate.groupingExpressions.exists(_.semanticEquals(attribute)))
            .flatMap(keyStatistics(aggregate.child, _))
        case _ => None
      }
    }
  }

  private def worthDeduplicating(right: LogicalPlan): Boolean = {
    if (right.isStreaming || right.output.size < 2 || right.output.size > 4) {
      false
    } else if (right.isInstanceOf[Aggregate] && {
        val aggregate = right.asInstanceOf[Aggregate]
        aggregate.groupingExpressions == aggregate.aggregateExpressions
      }) {
      // Also makes the rule idempotent when optimizer batches run to a fixed point.
      false
    } else {
      val keys = right.output
      val supported = keys.forall { key => key.dataType match {
        case ByteType | ShortType | IntegerType | LongType => true
        case _ => false
      }}
      if (!supported) false else {
        val domains = keys.map(key => keyStatistics(right, key))
        val stats = right.stats
        val distinct = if (domains.forall(_.isDefined)) {
          Some(domains.zip(keys).map { case (value, key) =>
            // NULL is another grouping value. Adding one is conservative even when
            // statistics have no NULL rows and prevents treating an all-NULL key as empty.
            value.get + (if (key.nullable) 1 else 0)
          }.product)
        } else None
        val rowsAndDistinct = for {
          rows <- stats.rowCount
          groups <- distinct
        } yield (rows, groups)
        rowsAndDistinct.exists { case (rows, groups) =>
          rows >= minimumBuildRows && groups > 0 &&
            rows >= groups * minimumDuplicateRatio &&
            stats.sizeInBytes > spark.sessionState.conf.autoBroadcastJoinThreshold.max(0L)
        }
      }
    }
  }

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val conf = spark.sessionState.conf
    if (!new RapidsConf(conf).isSqlEnabled || !conf.cboEnabled) {
      plan
    } else {
      val semiJoins = plan.transformUp {
        case join @ Join(_, right, LeftSemi, condition, _)
            if condition.forall(_.deterministic) && worthDeduplicating(right) =>
          // A semi join only tests existence. Keep the same attributes and NULL values,
          // while removing duplicate build rows; probe-side multiplicity is untouched.
          join.copy(right = Aggregate(right.output, right.output, right))
      }
      // Extension optimizer rules run before RewritePredicateSubquery. Handle the
      // uncorrelated IN form as well, so SQL subqueries receive the same optimization.
      semiJoins.transformAllExpressions {
        case in: InSubquery if in.query.outerAttrs.isEmpty && in.query.joinCond.isEmpty &&
            in.values.forall(_.deterministic) && worthDeduplicating(in.query.plan) =>
          val build = in.query.plan
          in.copy(query = in.query.withNewPlan(Aggregate(build.output, build.output, build)))
      }
    }
  }
}
