/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.optimizer

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.LogicalRelation

/**
 * Optimizer rule to merge subqueries that access the same base tables as the main query.
 * 
 * This optimization targets TPC-H style queries:
 * 
 * Q11 Pattern (HAVING with scalar subquery on same tables):
 * {{{
 * SELECT ps_partkey, SUM(value) as total
 * FROM partsupp
 * GROUP BY ps_partkey
 * HAVING SUM(value) > (SELECT SUM(value) * 0.0001 FROM partsupp)
 * }}}
 * 
 * Transforms to:
 * {{{
 * SELECT ps_partkey, total
 * FROM (
 *   SELECT ps_partkey, SUM(value) as total, 
 *          SUM(SUM(value)) OVER () * 0.0001 as threshold
 *   FROM partsupp
 *   GROUP BY ps_partkey
 * )
 * WHERE total > threshold
 * }}}
 * 
 * Configuration: spark.rapids.sql.optimizer.subqueryMerge.enabled
 */
class SubqueryMergeOptimizer extends Rule[LogicalPlan] with Logging {
  
  override def apply(plan: LogicalPlan): LogicalPlan = {
    plan.transformUp {
      case f @ Filter(condition, agg @ Aggregate(groupingExprs, aggExprs, child)) 
          if hasScalarSubqueryOnSameTable(condition, child) =>
        tryRewriteQ11Pattern(f, condition, agg, groupingExprs, aggExprs, child)
    }
  }
  
  /**
   * Check if condition contains a scalar subquery that scans the same table as child.
   */
  private def hasScalarSubqueryOnSameTable(condition: Expression, child: LogicalPlan): Boolean = {
    val mainTables = extractTables(child)
    var found = false
    
    condition.foreach {
      case sq: ScalarSubquery =>
        val subqueryTables = extractTables(sq.plan)
        if (mainTables.intersect(subqueryTables).nonEmpty) {
          // Check if subquery is a global aggregate (no grouping)
          sq.plan match {
            case Aggregate(Nil, _, _) => found = true
            case Project(_, Aggregate(Nil, _, _)) => found = true
            case _ =>
          }
        }
      case _ =>
    }
    
    found
  }
  
  /**
   * Attempt to rewrite Q11 pattern: HAVING with scalar subquery on same table.
   * 
   * Original:
   *   Filter(agg_result > ScalarSubquery(global_agg * factor))
   *     Aggregate(groupBy, agg_func)
   *       Scan(table)
   * 
   * Rewritten:
   *   Filter(agg_result > threshold)
   *     Project(*, SUM(agg_result) OVER () * factor AS threshold)
   *       Aggregate(groupBy, agg_func AS agg_result)
   *         Scan(table)
   */
  private def tryRewriteQ11Pattern(
      filter: Filter,
      condition: Expression,
      agg: Aggregate,
      groupingExprs: Seq[Expression],
      aggExprs: Seq[NamedExpression],
      child: LogicalPlan): LogicalPlan = {
    
    // Extract subquery info from condition
    val subqueryInfoOpt = extractSubqueryInfo(condition, child)
    
    if (subqueryInfoOpt.isEmpty) {
      logDebug("Could not extract subquery info for Q11 optimization")
      return filter
    }
    
    val (scalarSubquery, subqueryAggExpr, factorExpr) = subqueryInfoOpt.get
    
    // Find matching aggregate in main query
    val matchingMainAgg = findMatchingAggregate(aggExprs, subqueryAggExpr)
    
    if (matchingMainAgg.isEmpty) {
      logDebug("No matching aggregate found in main query")
      return filter
    }
    
    val (mainAggAlias, mainAggExpr) = matchingMainAgg.get
    
    logInfo("Applying Q11-style subquery merge optimization")
    logInfo(s"  Main aggregate: ${mainAggAlias.name}")
    logInfo(s"  Subquery aggregate: ${subqueryAggExpr}")
    logInfo(s"  Factor expression: ${factorExpr}")
    
    // Build the window expression: SUM(main_agg) OVER ()
    val windowExpr = buildWindowAggregate(mainAggAlias.toAttribute, subqueryAggExpr)
    
    // Apply factor if present
    val thresholdExpr = if (factorExpr != null) {
      Multiply(windowExpr, factorExpr)
    } else {
      windowExpr
    }
    
    // Create threshold alias
    val thresholdAlias = Alias(thresholdExpr, "_subquery_threshold")()
    
    // Build new plan:
    // 1. Original aggregate
    // 2. Window to compute global sum
    // 3. Replace subquery reference in filter condition
    
    // Create window node
    val windowSpec = WindowSpecDefinition(Nil, Nil, UnspecifiedFrame)
    val windowFunction = WindowExpression(
      AggregateExpression(
        Sum(mainAggAlias.toAttribute),
        Complete,
        isDistinct = false
      ),
      windowSpec
    )
    
    val windowAlias = Alias(windowFunction, "_global_sum")()
    
    // Build the final threshold with factor
    val finalThreshold = if (factorExpr != null) {
      Alias(Multiply(windowAlias.toAttribute, factorExpr), "_threshold")()
    } else {
      windowAlias
    }
    
    // Create Window node
    val windowNode = Window(
      Seq(windowAlias),
      Nil,  // No partition by
      Nil,  // No order by
      agg
    )
    
    // Create Project to add threshold
    val projectExprs = agg.output :+ finalThreshold
    val projectNode = Project(projectExprs, windowNode)
    
    // Replace scalar subquery in condition with threshold attribute
    val newCondition = replaceScalarSubquery(condition, scalarSubquery, finalThreshold.toAttribute)
    
    // Create new filter
    val newFilter = Filter(newCondition, projectNode)
    
    // Project to remove threshold column from output (keep original schema)
    val finalProject = Project(agg.output, newFilter)
    
    logInfo("Q11 optimization applied successfully")
    finalProject
  }
  
  /**
   * Extract subquery information from condition.
   * Returns (ScalarSubquery, AggregateExpression in subquery, factor expression if any)
   */
  private def extractSubqueryInfo(
      condition: Expression,
      mainChild: LogicalPlan): Option[(ScalarSubquery, AggregateExpression, Expression)] = {
    
    val mainTables = extractTables(mainChild)
    
    // Find scalar subquery in condition
    var result: Option[(ScalarSubquery, AggregateExpression, Expression)] = None
    
    condition.foreach {
      case sq: ScalarSubquery if result.isEmpty =>
        val subqueryTables = extractTables(sq.plan)
        
        // Check if same tables
        if (mainTables.intersect(subqueryTables).nonEmpty) {
          // Extract aggregate and factor from subquery
          val aggInfo = extractAggregateAndFactor(sq.plan)
          if (aggInfo.isDefined) {
            val (aggExpr, factor) = aggInfo.get
            result = Some((sq, aggExpr, factor))
          }
        }
      case _ =>
    }
    
    result
  }
  
  /**
   * Extract the aggregate expression and any multiplication factor from subquery plan.
   * 
   * Handles patterns like:
   *   - SELECT SUM(x) FROM table
   *   - SELECT SUM(x) * 0.0001 FROM table
   */
  private def extractAggregateAndFactor(plan: LogicalPlan): Option[(AggregateExpression, Expression)] = {
    plan match {
      case Aggregate(Nil, Seq(Alias(aggExpr: AggregateExpression, _)), _) =>
        Some((aggExpr, null))
        
      case Aggregate(Nil, Seq(Alias(Multiply(aggExpr: AggregateExpression, factor), _)), _) =>
        Some((aggExpr, factor))
        
      case Aggregate(Nil, Seq(Alias(Multiply(factor, aggExpr: AggregateExpression), _)), _) =>
        Some((aggExpr, factor))
        
      case Project(Seq(Alias(expr, _)), agg @ Aggregate(Nil, _, _)) =>
        expr match {
          case aggExpr: AggregateExpression =>
            Some((aggExpr, null))
          case Multiply(aggExpr: AggregateExpression, factor) =>
            Some((aggExpr, factor))
          case Multiply(factor, aggExpr: AggregateExpression) =>
            Some((aggExpr, factor))
          case _ =>
            extractAggregateAndFactor(agg)
        }
        
      case _ =>
        None
    }
  }
  
  /**
   * Find aggregate in main query that matches subquery aggregate.
   */
  private def findMatchingAggregate(
      aggExprs: Seq[NamedExpression],
      subqueryAgg: AggregateExpression): Option[(Alias, AggregateExpression)] = {
    
    aggExprs.foreach {
      case alias @ Alias(ae: AggregateExpression, _) =>
        if (aggregatesMatch(ae, subqueryAgg)) {
          return Some((alias, ae))
        }
      case _ =>
    }
    
    None
  }
  
  /**
   * Check if two aggregate expressions compute the same thing.
   */
  private def aggregatesMatch(a1: AggregateExpression, a2: AggregateExpression): Boolean = {
    (a1.aggregateFunction, a2.aggregateFunction) match {
      case (Sum(e1), Sum(e2)) => expressionsEquivalent(e1, e2)
      case (Average(e1), Average(e2)) => expressionsEquivalent(e1, e2)
      case (Count(e1), Count(e2)) => e1.zip(e2).forall { case (x, y) => expressionsEquivalent(x, y) }
      case (Min(e1), Min(e2)) => expressionsEquivalent(e1, e2)
      case (Max(e1), Max(e2)) => expressionsEquivalent(e1, e2)
      case _ => false
    }
  }
  
  /**
   * Check if two expressions are equivalent (compute the same value).
   */
  private def expressionsEquivalent(e1: Expression, e2: Expression): Boolean = {
    (e1, e2) match {
      case (a1: AttributeReference, a2: AttributeReference) =>
        a1.name == a2.name
      case (Multiply(l1, r1), Multiply(l2, r2)) =>
        (expressionsEquivalent(l1, l2) && expressionsEquivalent(r1, r2)) ||
        (expressionsEquivalent(l1, r2) && expressionsEquivalent(r1, l2))
      case (Add(l1, r1), Add(l2, r2)) =>
        (expressionsEquivalent(l1, l2) && expressionsEquivalent(r1, r2)) ||
        (expressionsEquivalent(l1, r2) && expressionsEquivalent(r1, l2))
      case (Literal(v1, t1), Literal(v2, t2)) =>
        v1 == v2 && t1 == t2
      case (Cast(c1, dt1, _, _), Cast(c2, dt2, _, _)) =>
        dt1 == dt2 && expressionsEquivalent(c1, c2)
      case _ =>
        e1.semanticEquals(e2)
    }
  }
  
  /**
   * Build a window aggregate that computes the global sum.
   */
  private def buildWindowAggregate(
      attr: Attribute,
      subqueryAgg: AggregateExpression): Expression = {
    
    // Create the same aggregate function but as a window function
    val windowAggFunc = subqueryAgg.aggregateFunction match {
      case Sum(_) => Sum(attr)
      case Average(_) => Average(attr)
      case Count(_) => Count(Seq(attr))
      case Min(_) => Min(attr)
      case Max(_) => Max(attr)
      case other => other
    }
    
    WindowExpression(
      AggregateExpression(windowAggFunc, Complete, isDistinct = false),
      WindowSpecDefinition(Nil, Nil, UnspecifiedFrame)
    )
  }
  
  /**
   * Replace scalar subquery in expression with attribute reference.
   */
  private def replaceScalarSubquery(
      expr: Expression,
      subquery: ScalarSubquery,
      replacement: Attribute): Expression = {
    
    expr.transform {
      case sq: ScalarSubquery if sq.fastEquals(subquery) =>
        replacement
    }
  }
  
  /**
   * Extract table names from a plan.
   */
  private def extractTables(plan: LogicalPlan): Set[String] = {
    val tables = new mutable.HashSet[String]()
    
    plan.foreach {
      case lr: LogicalRelation =>
        val name = lr.catalogTable.map(_.identifier.table).getOrElse {
          s"table_${lr.relation.hashCode().abs}"
        }
        tables += name
      case _ =>
    }
    
    tables.toSet
  }
}

object SubqueryMergeOptimizer {
  val ENABLED_KEY = "spark.rapids.sql.optimizer.subqueryMerge.enabled"
  val DEFAULT_ENABLED = false
  
  def apply(): SubqueryMergeOptimizer = new SubqueryMergeOptimizer()
}
