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
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, EqualTo, InSubquery, IsNotNull, LessThan, NamedExpression, Not, PredicateHelper}
import org.apache.spark.sql.catalyst.expressions.aggregate.{Max, Min}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftAnti, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.types.{ByteType, DataType, IntegerType, LongType, ShortType}

/** Replace an existence-only same-key/different-value self join with a min/max test. */
class GpuSelfJoinExistsOverrides(spark: SparkSession)
    extends Rule[LogicalPlan] with PredicateHelper {
  private def attribute(expression: NamedExpression): Option[Attribute] = expression match {
    case value: Attribute => Some(value)
    case Alias(value: Attribute, _) => Some(value)
    case _ => None
  }

  // CBO can move a third relation between the self-join inputs. Recover the
  // same key set only when that relation needs no value columns from the pair.
  private def replaceJoinedSelfJoin(project: Project): Option[LogicalPlan] = project.child match {
    case top: Join if top.joinType == Inner && top.condition.exists(_.deterministic) &&
        project.projectList.size == 1 && !top.exists(_.expressions.exists(!_.deterministic)) =>
      val variants = Seq((top.left, top.right), (top.right, top.left))
      variants.flatMap {
        case (nested: Join, third) if nested.joinType == Inner &&
            nested.condition.forall(_.deterministic) =>
          Seq(true, false).flatMap { pairOnLeft =>
            val pair = if (pairOnLeft) nested.left else nested.right
            val other = if (pairOnLeft) nested.right else nested.left
            val pairColumns = pair.outputSet ++ third.outputSet
            if (!top.condition.get.references.subsetOf(pairColumns)) {
              None
            } else {
              val key = splitConjunctivePredicates(top.condition.get).collectFirst {
                case EqualTo(a: Attribute, _: Attribute) if pair.outputSet.contains(a) => a
                case EqualTo(_: Attribute, b: Attribute) if pair.outputSet.contains(b) => b
              }
              key.flatMap { key =>
                val available = other.outputSet + key
                if (!project.references.subsetOf(available) ||
                    !nested.condition.forall(_.references.subsetOf(available))) {
                  None
                } else {
                  replaceSelfJoin(Project(Seq(key), top.copy(left = pair, right = third)))
                    .map { eligible =>
                      val child = if (pairOnLeft) nested.copy(left = eligible)
                        else nested.copy(right = eligible)
                      project.copy(child = child)
                    }
                }
              }
            }
          }
        case _ => Seq.empty
      }.headOption
    case _ => None
  }

  private def replaceSelfJoin(project: Project): Option[LogicalPlan] = project.child match {
    case Join(left, right, Inner, Some(condition), _)
        if project.projectList.size == 1 && !left.isStreaming && !right.isStreaming &&
          left.sameResult(right) && condition.deterministic &&
          !left.exists(_.expressions.exists(!_.deterministic)) =>
      def corresponding(a: Attribute, b: Attribute): Option[Int] = {
        val li = left.output.indexWhere(_.exprId == a.exprId)
        val ri = right.output.indexWhere(_.exprId == b.exprId)
        if (li >= 0 && li == ri) Some(li) else None
      }
      def index(a: Attribute, b: Attribute): Option[Int] =
        corresponding(a, b).orElse(corresponding(b, a))
      val predicates = splitConjunctivePredicates(condition)
      val keys = predicates.collect { case EqualTo(a: Attribute, b: Attribute) => index(a, b) }
      val values = predicates.collect {
        case Not(EqualTo(a: Attribute, b: Attribute)) => index(a, b)
      }
      for {
        output <- attribute(project.projectList.head)
        keyIndex <- if (predicates.size == 2 && keys.size == 1 && values.size == 1) {
          keys.head
        } else {
          None
        }
        valueIndex <- values.head
        source <- Seq(left, right).find(_.output(keyIndex).exprId == output.exprId)
        value = source.output(valueIndex)
        if Set[DataType](ByteType, ShortType, IntegerType, LongType).contains(value.dataType)
      } yield {
        val key = source.output(keyIndex)
        val low = Alias(Min(value).toAggregateExpression(), "exists_min")()
        val high = Alias(Max(value).toAggregateExpression(), "exists_max")()
        // Equality in the original join excludes NULL keys. Min/max ignore NULL
        // values, and the comparison excludes groups with fewer than two values.
        val aggregate = Aggregate(Seq(key), Seq(key, low, high), Filter(IsNotNull(key), source))
        project.copy(child = Filter(LessThan(low.toAttribute, high.toAttribute), aggregate))
      }
    case _ => None
  }

  // Only these operators preserve tuple-set equivalence when input multiplicity changes.
  // Stop at aggregates, windows and limits; their outputs can depend on duplicates.
  private def rewriteSet(plan: LogicalPlan): LogicalPlan = {
    if (!plan.expressions.forall(_.deterministic)) {
      plan
    } else {
      plan match {
        case project: Project => replaceSelfJoin(project).orElse(replaceJoinedSelfJoin(project))
          .getOrElse(project.copy(child = rewriteSet(project.child)))
        case filter: Filter => filter.copy(child = rewriteSet(filter.child))
        case join: Join if join.joinType == Inner =>
          join.copy(left = rewriteSet(join.left), right = rewriteSet(join.right))
        case _ => plan
      }
    }
  }

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!new RapidsConf(spark.sessionState.conf).isSqlEnabled) {
      plan
    } else {
      val joins = plan.transformUp {
        case join: Join if (join.joinType == LeftSemi || join.joinType == LeftAnti) &&
            join.condition.forall(_.deterministic) => join.copy(right = rewriteSet(join.right))
      }
      joins.transformAllExpressions {
        case in: InSubquery if in.query.outerAttrs.isEmpty && in.query.joinCond.isEmpty &&
            in.values.forall(_.deterministic) =>
          in.copy(query = in.query.withNewPlan(rewriteSet(in.query.plan)))
      }
    }
  }
}
