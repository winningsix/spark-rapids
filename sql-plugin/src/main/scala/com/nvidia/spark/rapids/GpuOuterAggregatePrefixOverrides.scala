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

import scala.collection.mutable
import scala.util.control.NonFatal

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, CollectList, Complete, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.optimizer.PushDownPredicates
import org.apache.spark.sql.catalyst.plans.{Inner, LeftAnti, LeftOuter}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.types._

/** Prune complete leading-key groups of a bounded outer-joined aggregate before ordered LIMIT. */
class GpuOuterAggregatePrefixOverrides private[rapids] (
    spark: SparkSession, maximumJoinedRows: Int, prefixSpan: Int) extends Rule[LogicalPlan] {
  def this(spark: SparkSession) = this(spark, Int.MaxValue - 1, 32768)

  require(maximumJoinedRows > 0 && maximumJoinedRows < Int.MaxValue && prefixSpan > 0)
  private val planned = TreeNodeTag[Boolean]("rapids.outerAggregatePrefixPlanned")
  private val maximumResultRows = 1000
  private val quantityCap = Long.MaxValue / (2L * maximumJoinedRows)
  private def dec(n: Long): BigDecimal = BigDecimal(n, java.math.MathContext.UNLIMITED)
  private def ten(n: Int): BigDecimal = dec(10).pow(n)
  private def cap(t: DataType): Option[BigDecimal] = t match {
    case d: DecimalType => Some((ten(d.precision) - 1) / ten(d.scale))
    case IntegerType => Some(dec(Int.MaxValue))
    case LongType => Some(dec(Long.MaxValue))
    case DoubleType => Some(BigDecimal(java.lang.Double.toString(Double.MaxValue)))
    case _ => None
  }
  private def quantum(t: DataType): BigDecimal = t match {
    case d: DecimalType => dec(1) / ten(d.scale)
    case _ => dec(1)
  }
  private def fits(t: DataType, n: BigDecimal): Option[BigDecimal] =
    cap(t).filter(n <= _).map(_ => n)
  private def rowBound(e: Expression): Option[BigDecimal] = e match {
    case l: Literal if l.value == null => Some(dec(0))
    case l: Literal => l.dataType match {
      case _: DecimalType => Some(BigDecimal(l.value.asInstanceOf[Decimal]
        .toJavaBigDecimal.toPlainString, java.math.MathContext.UNLIMITED).abs)
      case IntegerType => Some(dec(l.value.asInstanceOf[Int]).abs)
      case LongType => Some(dec(l.value.asInstanceOf[Long]).abs)
      case _ => None
    }
    case a: AttributeReference if a.dataType.isInstanceOf[DecimalType] => cap(a.dataType)
    case a: Add => for (x <- rowBound(a.left); y <- rowBound(a.right);
      b <- fits(a.dataType, x + y)) yield b
    case a: Subtract => for (x <- rowBound(a.left); y <- rowBound(a.right);
      b <- fits(a.dataType, x + y)) yield b
    case a: Multiply => for (x <- rowBound(a.left); y <- rowBound(a.right);
      b <- fits(a.dataType, x * y + quantum(a.dataType))) yield b
    case a: UnscaledValue => a.child.dataType match {
      case d: DecimalType => rowBound(a.child).flatMap(x => fits(LongType, x * ten(d.scale)))
      case _ => None
    }
    case _ => None
  }
  private case class Bound(low: BigDecimal, high: BigDecimal) {
    def magnitude: BigDecimal = low.abs.max(high.abs)
  }
  private def aggregateBound(e: Expression): Option[Bound] = e match {
    case a: Alias => aggregateBound(a.child)
    case a: AggregateExpression if a.mode == Complete && !a.isDistinct && a.filter.isEmpty =>
      a.aggregateFunction match {
        case s: Sum if s.evalMode == EvalMode.ANSI => s.child match {
          case a: Attribute if a.dataType == LongType =>
            Some(Bound(dec(0), dec(quantityCap) * dec(maximumJoinedRows)))
          case child => rowBound(child).flatMap(x => fits(s.dataType,
            x * dec(maximumJoinedRows))).map(b => Bound(-b, b))
        }
        case _ => None
      }
    case d: MakeDecimal => d.dataType match {
      case t: DecimalType => aggregateBound(d.child).flatMap(b =>
        fits(t, b.magnitude / ten(t.scale)).map(n => Bound(-n, n)))
      case _ => None
    }
    case _ => None
  }
  private def conjuncts(e: Expression): Seq[Expression] = e match {
    case And(a, b) => conjuncts(a) ++ conjuncts(b)
    case x => Seq(x)
  }
  private def safeInputExpression(e: Expression): Boolean = e match {
    case _: Attribute | _: Literal => true
    case _: And | _: Or | _: Not | _: IsNull | _: IsNotNull | _: BinaryComparison =>
      e.children.forall(safeInputExpression)
    case _ => false
  }
  private def safeInput(p: LogicalPlan): Boolean = p match {
    case a: Project => a.projectList.forall(_.isInstanceOf[Attribute]) && safeInput(a.child)
    case a: Filter => safeInputExpression(a.condition) && safeInput(a.child)
    case a: Join => (a.joinType == Inner || a.joinType == LeftOuter || a.joinType == LeftAnti) &&
      a.hint == JoinHint.NONE && a.condition.forall(safeInputExpression) &&
      safeInput(a.left) && safeInput(a.right)
    case leaf if leaf.children.isEmpty => leaf.deterministic && !leaf.isStreaming &&
      leaf.expressions.forall(safeInputExpression)
    case _ => false
  }
  private case class Shape(k: Int, sort: Sort, peers: Seq[(Aggregate, Attribute)])

  private def inspect(plan: LogicalPlan): Option[Shape] = {
    val shape = plan match {
      case GlobalLimit(Literal(k: Int, _), LocalLimit(Literal(k2: Int, _), child))
          if k == k2 && k > 0 && k <= maximumResultRows => child match {
        case s: Sort => Some(k -> s)
        case Project(attrs, s: Sort) if attrs.forall(_.isInstanceOf[Attribute]) => Some(k -> s)
        case _ => None
      }
      case _ => None
    }
    shape.flatMap { case (k, sort) =>
      if (!sort.global || sort.order.isEmpty || sort.order.head.direction != Ascending ||
          sort.order.head.nullOrdering != NullsFirst || !sort.order.forall(s =>
            s.child.isInstanceOf[Attribute] && sort.output.exists(
              _.exprId == s.child.asInstanceOf[Attribute].exprId))) {
        None
      } else {
        val peers = mutable.ArrayBuffer.empty[Aggregate]
        val spine = mutable.ArrayBuffer.empty[LogicalPlan]
        def collect(p: LogicalPlan): Boolean = p match {
          case a: Aggregate => peers += a; true
          case a: Project => spine += a; collect(a.child)
          case a: Filter => spine += a; collect(a.child)
          case a: Join if a.joinType == LeftOuter && a.hint == JoinHint.NONE =>
            spine += a; collect(a.left) && collect(a.right)
          case _ => false
        }
        if (!collect(sort.child) || peers.size < 2 || !peers.forall(a =>
            a.groupingExpressions.forall(_.isInstanceOf[Attribute]) && safeInput(a.child))) {
          None
        } else {
          val aliases = mutable.Map.empty[ExprId, ExprId]
          val bindings = mutable.Map.empty[ExprId, Expression]
          spine.foreach {
            case p: Project => p.projectList.foreach {
              case a @ Alias(child: Attribute, _) =>
                aliases(a.exprId) = child.exprId; bindings(a.exprId) = child
              case a: Alias => bindings(a.exprId) = a.child
              case _ =>
            }
            case _ =>
          }
          def origin(id: ExprId): ExprId = aliases.get(id).map(origin).getOrElse(id)
          val leading = sort.order.head.child.asInstanceOf[Attribute]
          val grouped = peers.map { a => a.aggregateExpressions.flatMap {
            case alias @ Alias(key: Attribute, _) if a.groupingExpressions.exists(
                _.semanticEquals(key)) => Some(alias.toAttribute -> key)
            case key: Attribute if a.groupingExpressions.exists(_.semanticEquals(key)) =>
              Some(key -> key)
            case _ => None
          }}
          // Equality to a nullable right key does not make that key a preserved ordering key.
          val first = grouped.head.filter { case (out, _) => origin(out.exprId) ==
            origin(leading.exprId) }
          val parent = mutable.Map.empty[ExprId, ExprId]
          def root(id: ExprId): ExprId = parent.get(origin(id)) match {
            case Some(p) if p != origin(id) => root(p)
            case _ => origin(id)
          }
          spine.foreach {
            case j: Join => j.condition.toSeq.flatMap(conjuncts).foreach {
              case EqualTo(a: Attribute, b: Attribute) => parent(root(a.exprId)) = root(b.exprId)
              case _ =>
            }
            case _ =>
          }
          val keys = grouped.map(_.filter { case (out, _) =>
            root(out.exprId) == root(leading.exprId) })
          val measures = peers.flatMap(_.aggregateExpressions.flatMap(e =>
            aggregateBound(e).map(e.exprId -> _))).toMap
          def zero(e: Expression): Boolean = e match {
            case l: Literal => l.value != null && rowBound(l).contains(dec(0))
            case _ => false
          }
          def nonnegativeQuantity(e: Expression): Boolean = e match {
            case Coalesce(Seq(a: Attribute, z)) if zero(z) && a.dataType == LongType =>
              measures.get(a.exprId).exists(_.low >= 0)
            case _ => false
          }
          def globalFilters(p: LogicalPlan): Seq[Filter] = p match {
            case f: Filter => Seq(f) ++ globalFilters(f.child)
            case p: Project => globalFilters(p.child)
            case _ => Nil
          }
          def nonthrowingSpine(p: LogicalPlan): Boolean = p match {
            case _: Aggregate => true // Every aggregate has its independent arithmetic proof.
            case p: Project => p.projectList.forall {
              case _: Attribute | Alias(_: Attribute, _) => true
              case _ => false
            } && nonthrowingSpine(p.child)
            case f: Filter => safeInputExpression(f.condition) && nonthrowingSpine(f.child)
            case j: Join => j.condition.forall(safeInputExpression) &&
              nonthrowingSpine(j.left) && nonthrowingSpine(j.right)
            case _ => false
          }
          // A filter proves positivity only for expressions evaluated above that filter.
          // Reject computed projections below it, including in a nullable join branch.
          val positiveSums = globalFilters(sort.child).filter(f =>
            nonthrowingSpine(f.child)).flatMap(f => conjuncts(f.condition)).flatMap {
            case Or(GreaterThan(a, za), GreaterThan(b, zb))
                if zero(za) && zero(zb) && nonnegativeQuantity(a) && nonnegativeQuantity(b) =>
              Some(Add(a, b))
            case _ => None
          }
          def bounds(e: Expression): Option[Bound] = e match {
            case a: Attribute => measures.get(a.exprId).orElse(bindings.get(a.exprId)
              .flatMap(bounds))
            case a: Alias => bounds(a.child)
            case l: Literal => rowBound(l).map { n =>
              if (l.value == null) Bound(dec(0), dec(0)) else {
                val negative = l.dataType match {
                  case LongType => l.value.asInstanceOf[Long] < 0
                  case IntegerType => l.value.asInstanceOf[Int] < 0
                  case _: DecimalType => l.value.asInstanceOf[Decimal].toJavaBigDecimal.signum < 0
                  case _ => false
                }
                val value = if (negative) -n else n
                Bound(value, value)
              }
            }
            case c: Coalesce =>
              val bs = c.children.map(bounds)
              if (bs.forall(_.nonEmpty)) Some(Bound(bs.flatten.map(_.low).min,
                bs.flatten.map(_.high).max)) else None
            case a: Add => for (x <- bounds(a.left); y <- bounds(a.right);
              _ <- fits(a.dataType, (x.low + y.low).abs.max((x.high + y.high).abs)))
              yield Bound(x.low + y.low, x.high + y.high)
            case c: Cast if c.dataType == DoubleType &&
                (c.child.dataType == LongType || c.child.dataType.isInstanceOf[DecimalType]) =>
              // A deliberately loose enclosure covers integer/decimal-to-double rounding.
              bounds(c.child).map { b =>
                val n = b.magnitude * dec(2) + dec(1)
                Bound(if (b.low >= 0) dec(0) else -n, n)
              }.filter(b => fits(DoubleType, b.magnitude).nonEmpty)
            case d: Divide if d.dataType == DoubleType && positive(d.right) =>
              // The proven denominator is a positive integer cast to double, hence >= 1.
              bounds(d.left).flatMap(b => fits(DoubleType, b.magnitude * dec(2) + dec(1))
                .map(n => Bound(-n, n)))
            case r: Round if r.dataType == DoubleType && (r.scale match {
              case Literal(n: Int, _) => n >= 0 && n <= 18
              case _ => false
            }) => bounds(r.child).flatMap(b => fits(DoubleType, b.magnitude + dec(1))
              .map(n => Bound(-n, n)))
            case _ => None
          }
          def positive(e: Expression): Boolean = e match {
            case c: Cast if c.dataType == DoubleType && c.child.dataType == LongType =>
              positive(c.child)
            case a: Add => positiveSums.exists(_.semanticEquals(a)) && bounds(a).nonEmpty
            case _ => false
          }
          def safeOuter(e: Expression): Boolean = e match {
            case _: Attribute | _: Literal => true
            case a: Alias => safeOuter(a.child)
            case _: And | _: Or | _: Not | _: IsNull | _: IsNotNull | _: BinaryComparison |
                _: Coalesce => e.children.forall(safeOuter)
            case _ => bounds(e).nonEmpty
          }
          val safeMeasures = peers.forall(_.aggregateExpressions.forall {
            case _: Attribute | Alias(_: Attribute, _) => true
            case e => aggregateBound(e).nonEmpty
          })
          val safeSpine = spine.forall {
            case p: Project => p.projectList.forall(safeOuter)
            case f: Filter => safeOuter(f.condition)
            case j: Join => j.condition.forall(safeOuter)
            case _ => false
          }
          if (first.size == 1 && keys.forall(_.size == 1) && safeMeasures && safeSpine &&
              positiveSums.nonEmpty && keys.flatten.forall(_._2.dataType == IntegerType)) {
            Some(Shape(k, sort, peers.zip(keys.map(_.head._2)).toSeq))
          } else None
        }
      }
    }
  }

  private def scalar(plan: LogicalPlan): ScalarSubquery = {
    val raw = ScalarSubquery(plan)
    val Subquery(optimized, _) = spark.sessionState.optimizer.execute(Subquery.fromExpression(raw))
    raw.withNewPlan(optimized)
  }
  private def limited(k: Int, plan: LogicalPlan): LogicalPlan =
    GlobalLimit(Literal(k), LocalLimit(Literal(k), plan))

  // Guards need an upper bound on rows and a superset of the guarded values, not
  // the exact anti-join result. Removing an anti join avoids doing its large shuffle
  // solely to prove the prefix safe. Ordinary outer joins must remain: their right
  // side can multiply rows. A NULL test proves anti semantics only when it tests a
  // right attribute that every matching row must make non-NULL via ordinary equality.
  private[rapids] def guardInputSuperset(
      plan: LogicalPlan, required: AttributeSet): LogicalPlan = {
    // The NULL rejection can initially sit above an inner dimension join. Normalize
    // those nonthrowing input predicates before looking for the anti-join pattern.
    var normalized = plan
    var changed = true
    var remaining = 10
    while (changed && remaining > 0) {
      val next = PushDownPredicates(normalized)
      changed = !next.fastEquals(normalized)
      normalized = next
      remaining -= 1
    }
    widenGuardInput(normalized, required)
  }

  private def widenGuardInput(plan: LogicalPlan, required: AttributeSet): LogicalPlan = {
    val output = plan.output.filter(required.contains)
    plan match {
      case Project(attributes, child) if attributes.forall(_.isInstanceOf[Attribute]) =>
        Project(output, widenGuardInput(child, required))
      case Filter(condition, Join(left, right, LeftOuter, Some(on), _))
          if required.subsetOf(left.outputSet) && conjuncts(condition).exists {
            case IsNull(key: Attribute) if right.outputSet.contains(key) =>
              conjuncts(on).exists {
                case EqualTo(a: Attribute, b: Attribute) =>
                  (key.semanticEquals(a) && left.outputSet.contains(b)) ||
                    (key.semanticEquals(b) && left.outputSet.contains(a))
                case _ => false
              }
            case _ => false
          } =>
        val remaining = conjuncts(condition).filter(_.references.subsetOf(left.outputSet))
        val source = remaining.reduceOption[Expression](And).map(Filter(_, left)).getOrElse(left)
        widenGuardInput(source, required)
      case Filter(condition, child) =>
        Project(output, Filter(condition,
          widenGuardInput(child, required ++ condition.references)))
      case Join(left, _, LeftAnti, _, _) => widenGuardInput(left, required)
      case j @ Join(left, right, Inner, condition, _) =>
        val inputs = required ++ condition.map(_.references).getOrElse(AttributeSet.empty)
        Project(output, j.copy(
          left = widenGuardInput(left, inputs.intersect(left.outputSet)),
          right = widenGuardInput(right, inputs.intersect(right.outputSet))))
      case _ => Project(output, plan)
    }
  }

  private def build(plan: LogicalPlan, shape: Shape): LogicalPlan = {
    val guards = shape.peers.map { case (aggregate, key) =>
      val quantities = aggregate.aggregateExpressions.flatMap(_.collect {
        case a: AggregateExpression => a.aggregateFunction match {
          case s: Sum => s.child match {
            case a: Attribute if a.dataType == LongType => Some(a)
            case _ => None
          }
          case _ => None
        }
      }).flatten.distinct
      val count = Count(Seq(Literal(1))).toAggregateExpression()
      val rowSafe = LessThanOrEqual(count, Literal(maximumJoinedRows.toLong))
      val checks = quantities.flatMap { a => Seq(
        Coalesce(Seq(GreaterThanOrEqual(Min(a).toAggregateExpression(), Literal(0L)),
          Literal(true))),
        Coalesce(Seq(LessThanOrEqual(Max(a).toAggregateExpression(), Literal(quantityCap)),
          Literal(true))))
      }
      val safe = checks.foldLeft[Expression](rowSafe)(And)
      val proof = Alias(CreateStruct(Seq(safe, Min(key).toAggregateExpression())),
        "outer_prefix_guard")()
      // If the global count is <= B, no partition reached its local cap B+1, so
      // count and extrema describe the complete input. Otherwise retain the original plan.
      // Spark has at most Int.MaxValue partitions, so even the capped total fits in Long.
      // Unlike a GlobalLimit, this permits partial count/extrema before the global exchange.
      // The superset count/extrema are stronger sufficient bounds. Its minimum can
      // make the prefix smaller, which only increases the chance of original fallback.
      val input = guardInputSuperset(aggregate.child, AttributeSet(quantities :+ key))
      scalar(Aggregate(Nil, Seq(proof),
        LocalLimit(Literal(maximumJoinedRows + 1), input)))
    }
    val safe = guards.map(g => GetStructField(g, 0)).reduceLeft[Expression](And)
    val minimum = Cast(GetStructField(guards.head, 1), LongType)
    val cutoff = Least(Seq(Add(minimum, Literal(prefixSpan.toLong - 1)),
      Literal(Int.MaxValue.toLong)))
    val wide = limited(shape.k, shape.sort)
    val fast = wide.transformDown {
      case a: Aggregate if shape.peers.exists(_._1 eq a) =>
        val key = shape.peers.find(_._1 eq a).get._2
        val prefix = Or(IsNull(key), LessThanOrEqual(Cast(key, LongType), cutoff))
        a.copy(child = Filter(And(safe, prefix), a.child))
    }
    fast.setTagValue(planned, true)
    val collected = Alias(CollectList(CreateStruct(fast.output)).toAggregateExpression(),
      "outer_prefix_rows")()
    val rows = scalar(Aggregate(Nil, Seq(collected), fast))
    val useFast = And(safe, GreaterThanOrEqual(Size(rows, false), Literal(shape.k)))
    val row = AttributeReference("outer_prefix_row",
      rows.dataType.asInstanceOf[ArrayType].elementType, nullable = false)()
    val expanded = Generate(Explode(rows), Nil, false, None, Seq(row),
      Filter(useFast, OneRowRelation()))
    val fastOutput = Project(wide.output.zipWithIndex.map { case (original, index) =>
      Alias(GetStructField(row, index, Some(original.name)), original.name)(
        exprId = original.exprId, qualifier = original.qualifier,
        explicitMetadata = Some(original.metadata))
    }, expanded)
    val fallback = wide.transformDown {
      case a: Aggregate if shape.peers.exists(_._1 eq a) =>
        a.copy(child = Filter(Not(useFast), a.child))
    }
    fallback.setTagValue(planned, true)
    val result = Project(plan.output, limited(shape.k,
      shape.sort.copy(child = Union(Seq(fastOutput, fallback)))))
    result.setTagValue(planned, true)
    result
  }
  private def marked(plan: LogicalPlan): Boolean = plan.getTagValue(planned).contains(true) ||
    plan.exists(_.expressions.exists(_.exists {
      case s: ScalarSubquery => s.plan.output.exists(_.name.startsWith("outer_prefix_"))
      case _ => false
    }))
  private[rapids] def rewrite(plan: LogicalPlan): Option[LogicalPlan] = {
    if (!plan.resolved || plan.isStreaming || !plan.deterministic || marked(plan)) None
    else try inspect(plan).map(build(plan, _)) catch { case NonFatal(_) => None }
  }
  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!new RapidsConf(spark.sessionState.conf).isSqlEnabled ||
        !spark.sessionState.conf.ansiEnabled) {
      plan
    } else {
      plan.transformUp { case p: GlobalLimit => rewrite(p).getOrElse(p) }
    }
  }
}
