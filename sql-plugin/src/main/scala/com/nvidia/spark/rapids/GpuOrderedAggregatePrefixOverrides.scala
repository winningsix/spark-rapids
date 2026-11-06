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
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, CollectList, Complete, Count, Sum}
import org.apache.spark.sql.catalyst.optimizer.{PropagateEmptyRelation, RemoveRedundantAliases}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.types._

/** Evaluate a complete leading-key prefix when bounded arithmetic and enough rows prove top K. */
class GpuOrderedAggregatePrefixOverrides private[rapids] (
    spark: SparkSession, private val maximumJoinedRows: Int) extends Rule[LogicalPlan] {
  def this(spark: SparkSession) = this(spark, Int.MaxValue - 1)

  require(maximumJoinedRows > 0 && maximumJoinedRows < Int.MaxValue)
  private val planned = TreeNodeTag[Boolean]("rapids.orderedAggregatePrefixPlanned")
  private val maximumResultRows = 1000
  private val prefixKeys = 4096
  private def decimal(n: Long): BigDecimal =
    BigDecimal(n, java.math.MathContext.UNLIMITED)
  private def ten(n: Int): BigDecimal = decimal(10).pow(n)
  private def cap(t: DataType): Option[BigDecimal] = t match {
    case d: DecimalType => Some((ten(d.precision) - 1) / ten(d.scale))
    case IntegerType => Some(decimal(Int.MaxValue))
    case LongType => Some(decimal(Long.MaxValue))
    case _ => None
  }
  private def quantum(t: DataType): BigDecimal = t match {
    case d: DecimalType => decimal(1) / ten(d.scale)
    case _ => decimal(1)
  }
  private def fits(t: DataType, n: BigDecimal): Option[BigDecimal] =
    cap(t).filter(n <= _).map(_ => n)
  private def rowBound(e: Expression): Option[BigDecimal] = e match {
    case l: Literal if l.value == null => Some(decimal(0))
    case l: Literal => l.dataType match {
      case _: DecimalType => Some(BigDecimal(
        l.value.asInstanceOf[Decimal].toJavaBigDecimal.toPlainString,
        java.math.MathContext.UNLIMITED).abs)
      case IntegerType => Some(decimal(l.value.asInstanceOf[Int]).abs)
      case LongType => Some(decimal(l.value.asInstanceOf[Long]).abs)
      case _ => None
    }
    case a: AttributeReference => a.dataType match {
      case _: DecimalType => cap(a.dataType)
      case _ => None // Signed integer minimum needs asymmetric bounds; reject for now.
    }
    case a: Add => for (x <- rowBound(a.left); y <- rowBound(a.right);
      bound <- fits(a.dataType, x + y)) yield bound
    case a: Subtract => for (x <- rowBound(a.left); y <- rowBound(a.right);
      bound <- fits(a.dataType, x + y)) yield bound
    case a: Multiply => for (x <- rowBound(a.left); y <- rowBound(a.right);
      bound <- fits(a.dataType, x * y + quantum(a.dataType))) yield bound
    case a: Divide => a.right match {
      case l: Literal if l.value != null => for (x <- rowBound(a.left); y <- rowBound(l)
        if y > 0; bound <- fits(a.dataType, x / y + quantum(a.dataType))) yield bound
      case _ => None
    }
    case a: UnaryMinus => rowBound(a.child).flatMap(fits(a.dataType, _))
    case a: Cast if a.child.dataType.isInstanceOf[DecimalType] &&
        a.dataType.isInstanceOf[DecimalType] =>
      rowBound(a.child).flatMap(x => fits(a.dataType, x + quantum(a.dataType)))
    case a: UnscaledValue => a.child.dataType match {
      case d: DecimalType => rowBound(a.child).flatMap(x => fits(LongType, x * ten(d.scale)))
      case _ => None
    }
    case _ => None
  }
  private def aggregateBound(e: Expression, n: BigDecimal): Option[BigDecimal] = e match {
    case a: Alias => aggregateBound(a.child, n)
    case ae: AggregateExpression if ae.mode == Complete && !ae.isDistinct && ae.filter.isEmpty =>
      ae.aggregateFunction match {
        case s: Sum if s.evalMode == EvalMode.ANSI =>
          rowBound(s.child).flatMap(x => fits(s.dataType, x * n))
        case _ => None
      }
    case d: MakeDecimal => d.dataType match {
      case t: DecimalType => aggregateBound(d.child, n)
        .flatMap(x => fits(t, x / ten(t.scale)))
      case _ => None
    }
    case _ => None
  }
  private def safeRatio(c: CaseWhen, bounds: Map[ExprId, BigDecimal]): Boolean = {
    c.branches.size == 1 && c.elseValue.forall {
      case l: Literal => l.value == null || rowBound(l).contains(decimal(0))
      case _ => false
    } && (c.branches.head match {
      case (GreaterThan(den: AttributeReference, zero: Literal), d: Divide)
          if zero.value != null && rowBound(zero).contains(decimal(0)) &&
             d.right.semanticEquals(den) =>
        d.left match {
          case num: AttributeReference =>
            bounds.contains(den.exprId) && bounds.get(num.exprId).exists { bound =>
              den.dataType.isInstanceOf[DecimalType] &&
                fits(d.dataType, bound / quantum(den.dataType) + quantum(d.dataType)).nonEmpty
            }
          case _ => false
        }
      case _ => false
    })
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
    case a: Join => a.joinType == Inner && a.hint == JoinHint.NONE &&
      a.condition.forall(safeInputExpression) && safeInput(a.left) && safeInput(a.right)
    case leaf if leaf.children.isEmpty => leaf.deterministic && !leaf.isStreaming &&
      leaf.expressions.forall(safeInputExpression)
    case _ => false
  }
  private def inspect(plan: LogicalPlan): Option[Seq[(Aggregate, Attribute)]] = plan match {
    case GlobalLimit(Literal(k: Int, _), LocalLimit(Literal(k2: Int, _), sort: Sort))
        if k > 0 && k <= maximumResultRows && k == k2 && sort.global && sort.order.nonEmpty &&
          sort.order.head.direction == Ascending && sort.order.head.nullOrdering == NullsFirst &&
          sort.order.forall(s => s.child.isInstanceOf[Attribute] &&
            plan.output.exists(_.exprId == s.child.asInstanceOf[Attribute].exprId)) =>
      val peers = mutable.ArrayBuffer.empty[Aggregate]
      val spine = mutable.ArrayBuffer.empty[LogicalPlan]
      def collect(p: LogicalPlan): Boolean = p match {
        case a: Aggregate => peers += a; true
        case a: Project if a.projectList.forall {
          case _: Attribute => true
          case Alias(_: Attribute, _) => true
          case _ => false
        } => spine += a; collect(a.child)
        case a: Filter => spine += a; collect(a.child)
        case a: Join if a.joinType == Inner && a.hint == JoinHint.NONE =>
          spine += a; collect(a.left) && collect(a.right)
        case _ => false
      }
      if (!collect(sort.child) || peers.size < 2 || !peers.forall(a =>
          a.groupingExpressions.forall(_.isInstanceOf[Attribute]) && safeInput(a.child))) {
        None
      } else {
        val parent = mutable.Map.empty[ExprId, ExprId]
        def root(id: ExprId): ExprId = parent.get(id) match {
          case Some(p) if p != id => root(p)
          case _ => id
        }
        def connect(a: Attribute, b: Attribute): Unit = parent(root(a.exprId)) = root(b.exprId)
        spine.foreach {
          case p: Project => p.projectList.foreach {
            case a @ Alias(child: Attribute, _) => connect(a.toAttribute, child)
            case _ =>
          }
          case j: Join => j.condition.toSeq.flatMap(conjuncts).foreach {
            case EqualTo(a: Attribute, b: Attribute) => connect(a, b)
            case _ =>
          }
          case f: Filter => conjuncts(f.condition).foreach {
            case EqualTo(a: Attribute, b: Attribute) => connect(a, b)
            case _ =>
          }
          case _ =>
        }
        val leading = sort.order.head.child.asInstanceOf[Attribute]
        val groupedKeys = peers.map { a =>
          val keys = a.aggregateExpressions.flatMap {
            case alias @ Alias(key: Attribute, _) if a.groupingExpressions.exists(
                _.semanticEquals(key)) && root(alias.exprId) == root(leading.exprId) => Some(key)
            case key: Attribute if a.groupingExpressions.exists(_.semanticEquals(key)) &&
                root(key.exprId) == root(leading.exprId) => Some(key)
            case _ => None
          }
          if (keys.size == 1) Some(a -> keys.head) else None
        }
        val bound = decimal(maximumJoinedRows)
        val bounds = peers.flatMap(_.aggregateExpressions.flatMap(e =>
          aggregateBound(e, bound).map(e.exprId -> _))).toMap
        def safeOuter(e: Expression): Boolean = e match {
          case _: Attribute | _: Literal => true
          case c: CaseWhen => safeRatio(c, bounds)
          case _: And | _: Or | _: Not | _: IsNull | _: IsNotNull | _: BinaryComparison =>
            e.children.forall(safeOuter)
          case _ => false
        }
        val safeMeasures = peers.forall(_.aggregateExpressions.forall {
          case _: Attribute => true
          case Alias(_: Attribute, _) => true
          case e => aggregateBound(e, bound).nonEmpty
        })
        val safePredicates = spine.forall {
          case f: Filter => safeOuter(f.condition)
          case j: Join => j.condition.forall(safeOuter)
          case _: Project => true
          case _ => false
        }
        // Every accepted leading key must participate in a conjunctive equality on the
        // inner-join spine. Such an equality rejects NULL, including through aliases.
        // This permits an ordinary equality join to the bounded, distinct prefix keys.
        val nonNullLeadingKey = spine.exists {
          case j: Join => j.condition.toSeq.flatMap(conjuncts).exists {
            case EqualTo(a: Attribute, b: Attribute) =>
              root(a.exprId) == root(leading.exprId) && root(b.exprId) == root(leading.exprId)
            case _ => false
          }
          case f: Filter => conjuncts(f.condition).exists {
            case EqualTo(a: Attribute, b: Attribute) =>
              root(a.exprId) == root(leading.exprId) && root(b.exprId) == root(leading.exprId)
            case _ => false
          }
          case _ => false
        }
        val leafSources = groupedKeys.flatten.forall { case (a, key) =>
          val leaves = a.child.collect {
            case leaf if leaf.children.isEmpty && leaf.output.exists(_.exprId == key.exprId) => leaf
          }
          leaves.size == 1 && (key.dataType == StringType || key.dataType == IntegerType ||
            key.dataType == LongType)
        }
        if (groupedKeys.forall(_.nonEmpty) && safeMeasures && safePredicates &&
            nonNullLeadingKey && leafSources) {
          Some(groupedKeys.flatten.toSeq)
        } else None
      }
    case _ => None
  }

  private def scalar(plan: LogicalPlan): ScalarSubquery = {
    val raw = ScalarSubquery(plan)
    val Subquery(optimized, _) = spark.sessionState.optimizer.execute(Subquery.fromExpression(raw))
    raw.withNewPlan(optimized)
  }

  private def limited(k: Int, plan: LogicalPlan): LogicalPlan =
    GlobalLimit(Literal(k), LocalLimit(Literal(k), plan))

  private def marked(plan: LogicalPlan): Boolean = plan.getTagValue(planned).contains(true) ||
    plan.exists(_.expressions.exists(_.exists {
      case s: ScalarSubquery => s.plan.output.exists(_.name.startsWith("ordered_prefix_"))
      case _ => false
    }))

  private def build(plan: LogicalPlan, peers: Seq[(Aggregate, Attribute)]): LogicalPlan = {
    val GlobalLimit(Literal(k: Int, _), LocalLimit(_, sort: Sort)) = plan: @unchecked
    val guards = peers.map { case (aggregate, _) =>
      // Count exact joined input rows, capped just above the bound. CBO estimates are irrelevant.
      val count = Count(Seq(Literal(1))).toAggregateExpression()
      val safe = Alias(LessThanOrEqual(count, Literal(maximumJoinedRows.toLong)),
        "ordered_prefix_input_safe")()
      scalar(Aggregate(Nil, Seq(safe), limited(maximumJoinedRows + 1, aggregate.child)))
    }
    val safe = guards.reduceLeft[Expression](And)
    val (first, key) = peers.head
    val leaf = first.child.collectFirst {
      case p if p.children.isEmpty && p.output.exists(_.exprId == key.exprId) => p
    }.get
    val distinct = Aggregate(Seq(key), Seq(key),
      Project(Seq(key), Filter(IsNotNull(key), leaf)))
    val sortedKeys = Sort(Seq(SortOrder(key, Ascending)), true, distinct)
    val prefix = new GpuDistinctKeyPrefixLimit(Literal(prefixKeys),
      LocalLimit(Literal(prefixKeys), sortedKeys))
    val prefixKey = Alias(key, "ordered_prefix_key")()
    val keySet = Project(Seq(prefixKey), prefix)
    val fast = plan.transformDown {
      case aggregate: Aggregate if peers.exists(_._1 eq aggregate) =>
        val groupKey = peers.find(_._1 eq aggregate).get._2
        val restricted = aggregate.child.transformUp {
          case source if source.children.isEmpty &&
              source.output.exists(_.exprId == groupKey.exprId) =>
            // Distinct prefix keys preserve source multiplicity, including duplicate labels.
            // Its explicit LIMIT exposes a bounded build side to the cost-based optimizer;
            // no broadcast hint or uniqueness assumption about the source is required.
            Project(source.output, Join(source, keySet, Inner,
              Some(EqualTo(groupKey, prefixKey.toAttribute)), JoinHint.NONE))
        }
        aggregate.copy(child = Filter(safe, restricted))
    }
    fast.setTagValue(planned, true)
    val collected = Alias(CollectList(CreateStruct(fast.output)).toAggregateExpression(),
      "ordered_prefix_rows")()
    val rows = scalar(Aggregate(Nil, Seq(collected), fast))
    val useFast = And(safe, GreaterThanOrEqual(Size(rows, false), Literal(k)))
    val row = AttributeReference("ordered_prefix_row",
      rows.dataType.asInstanceOf[ArrayType].elementType, nullable = false)()
    val expanded = Generate(Explode(rows), Nil, false, None, Seq(row),
      Filter(useFast, OneRowRelation()))
    val fastOutput = Project(plan.output.zipWithIndex.map { case (original, index) =>
      Alias(GetStructField(row, index, Some(original.name)), original.name)(
        exprId = original.exprId, qualifier = original.qualifier,
        explicitMetadata = Some(original.metadata))
    }, expanded)
    val fallback = plan.transformDown {
      case aggregate: Aggregate if peers.exists(_._1 eq aggregate) =>
        aggregate.copy(child = Filter(Not(useFast), aggregate.child))
    }
    fallback.setTagValue(planned, true)
    // The branches are exclusive. Re-sort reconstructed rows; collect_list order is unspecified.
    val union = Union(Seq(fastOutput, fallback))
    val result = limited(k, sort.copy(child = union))
    result.setTagValue(planned, true)
    result
  }

  private[rapids] def rewrite(plan: LogicalPlan): Option[LogicalPlan] = {
    if (!plan.resolved || plan.isStreaming || !plan.deterministic || marked(plan)) None
    else try {
      // Extended optimizer rules run before Spark propagates empty UNION arms. Use Spark's own
      // cleanup so dead grouped arms and their output aliases do not hide a shared grouping key.
      val hasEmpty = plan.exists {
        case local: LocalRelation => local.data.isEmpty
        case _ => false
      }
      val normalized = if (hasEmpty && plan.exists(_.isInstanceOf[Union])) {
        RemoveRedundantAliases(PropagateEmptyRelation(plan))
      } else {
        plan
      }
      inspect(normalized).map(peers => build(normalized, peers))
    } catch {
      // Unsupported arithmetic bounds (for example nonterminating exact decimal division) reject
      // this optional rewrite, rather than rounding down a safety bound.
      case NonFatal(_) => None
    }
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

/** Preserve only provable key uniqueness through a limited, non-NULL, single-key relation. */
class GpuDistinctKeyPrefixLimit(limit: Expression, input: LogicalPlan)
    extends GlobalLimit(limit, input) {
  override def stats: Statistics = {
    val limited = super.stats
    val key = output.headOption
    val unique = output.size == 1 && child.distinctKeys.exists(
      _.subsetOf(ExpressionSet(output)))
    if (unique && key.exists(k => child.constraints.contains(IsNotNull(k)))) {
      limited.rowCount.map { rows =>
        // LIMIT changes the key distribution. Do not copy a histogram or old NDV:
        // for a distinct non-NULL key, its NDV equals the estimated output row count.
        val column = ColumnStat(distinctCount = Some(rows), nullCount = Some(BigInt(0)),
          avgLen = Some(output.head.dataType.defaultSize.toLong))
        limited.copy(attributeStats = AttributeMap(Seq(output.head -> column)), isRuntime = false)
      }.getOrElse(limited)
    } else {
      limited
    }
  }

  override def copy(limitExpr: Expression, child: LogicalPlan): GlobalLimit =
    new GpuDistinctKeyPrefixLimit(limitExpr, child)

  override protected def withNewChildInternal(newChild: LogicalPlan): GlobalLimit =
    copy(limitExpr, newChild)

  override def canEqual(other: Any): Boolean = other.isInstanceOf[GpuDistinctKeyPrefixLimit]
  override def equals(other: Any): Boolean = canEqual(other) && super.equals(other)
  override def hashCode(): Int = super.hashCode()
}
