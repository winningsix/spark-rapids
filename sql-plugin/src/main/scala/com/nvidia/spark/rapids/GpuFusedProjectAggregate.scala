/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import ai.rapids.cudf.{Cuda, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{splitSpillableInHalfByRows, withRetry}
import com.nvidia.spark.rapids.jni.{FusedTransformAggregate, GpuRetryOOM, GpuSplitAndRetryOOM}
import com.nvidia.spark.rapids.jni.FusedTransformAggregate.ExpressionBuilder

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.rapids.{GpuGreaterThan, GpuMultiply}
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuAverage,
  GpuBasicSum, GpuCount, GpuMax, GpuMin}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

// =============================================================================
// Visitor Pattern Framework - Extensible Expression Matchers
// =============================================================================

/**
 * Fusion context - carries information needed during matching process
 * 
 * @param colIndexMap mapping from column ExprId to input column index
 * @param aggOp aggregation operation type (SUM/COUNT/AVG/MIN/MAX)
 * @param builder JNI expression builder
 */
case class FusionContext(
    colIndexMap: Map[ExprId, Int],
    aggOp: Int,
    builder: ExpressionBuilder
) {
  /** Get column index, returns None if column doesn't exist */
  def getColIndex(ref: AttributeReference): Option[Int] = colIndexMap.get(ref.exprId)
  
  /** Try to convert value to Long */
  def toLong(value: Any): Option[Long] = value match {
    case l: Long => Some(l)
    case i: Int => Some(i.toLong)
    case s: Short => Some(s.toLong)
    case b: Byte => Some(b.toLong)
    case d: Double => Some(d.toLong)
    case f: Float => Some(f.toLong)
    case _ => None
  }
}

/**
 * Expression matcher trait - base class for all matchers
 * 
 * Each matcher is responsible for recognizing and handling a specific expression pattern
 */
trait ExpressionMatcher {
  /** Matcher name for logging and debugging */
  def name: String
  
  /** Matcher priority, lower value means higher priority */
  def priority: Int = 100
  
  /**
   * Try to match expression
   * 
   * @param expr expression to match
   * @param ctx fusion context
   * @return true if matched successfully, false otherwise
   */
  def tryMatch(expr: Expression, ctx: FusionContext): Boolean
}

/**
 * Expression matcher registry - manages all registered matchers
 * 
 * Uses registry pattern for easy extension of new expression pattern support
 */
object ExpressionMatcherRegistry extends Logging {
  private val matchers = mutable.ListBuffer[ExpressionMatcher]()
  
  /** Register a matcher */
  def register(matcher: ExpressionMatcher): Unit = {
    matchers += matcher
    // Sort by priority, higher priority (lower value) matches first
    // Note: Using sortBy + clear + appendAll for Scala 2.12 compatibility
    val sorted = matchers.sortBy(_.priority)
    matchers.clear()
    matchers ++= sorted
    logDebug(s"Registered expression matcher: ${matcher.name}, priority: ${matcher.priority}")
  }
  
  /** 
   * Try to match expression
   * Tries all matchers in priority order, returns first successful match
   */
  def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    matchers.exists { matcher =>
      val matched = matcher.tryMatch(expr, ctx)
      if (matched) {
        logDebug(s"Expression matched: ${matcher.name}")
      }
      matched
    }
  }
  
  /** Get number of registered matchers */
  def size: Int = matchers.size
  
  /** Clear all matchers (for testing) */
  def clear(): Unit = matchers.clear()
}

// =============================================================================
// Concrete Matcher Implementations
// =============================================================================

/**
 * Literal matcher - handles COUNT(1) cases
 */
object LiteralMatcher extends ExpressionMatcher {
  override def name: String = "Literal (COUNT(1))"
  override def priority: Int = 10
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuLiteral(_, _) =>
        // For COUNT(1), use identity with column 0 and AGG_COUNT
        // This counts all non-null rows which is equivalent to COUNT(1)
        ctx.builder.addIdentity(0, FusedTransformAggregate.AGG_COUNT)
        true
      case _ => false
    }
  }
}

/**
 * Identity matcher - handles simple SUM(col) cases
 */
object IdentityMatcher extends ExpressionMatcher {
  override def name: String = "Identity (column reference)"
  override def priority: Int = 20
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case ref: AttributeReference =>
        ctx.getColIndex(ref) match {
          case Some(idx) =>
            ctx.builder.addIdentity(idx, ctx.aggOp)
            true
          case None => false
        }
      case _ => false
    }
  }
}

/**
 * Coalesce matcher - handles COALESCE(col, default) pattern
 */
object CoalesceMatcher extends ExpressionMatcher {
  override def name: String = "Coalesce"
  override def priority: Int = 30
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuCoalesce(Seq(ref: AttributeReference, GpuLiteral(value, _))) =>
        (ctx.getColIndex(ref), ctx.toLong(value)) match {
          case (Some(idx), Some(defaultVal)) =>
            ctx.builder.addCoalesce(idx, defaultVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

/**
 * Coalesce self-multiply matcher - handles COALESCE(col, d) * COALESCE(col, d) pattern
 * Commonly used for variance calculation
 */
object CoalesceMulSelfMatcher extends ExpressionMatcher {
  override def name: String = "CoalesceMulSelf (variance)"
  override def priority: Int = 40
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuMultiply(
          GpuCoalesce(Seq(ref1: AttributeReference, lit1 @ GpuLiteral(v1, _))),
          GpuCoalesce(Seq(ref2: AttributeReference, lit2 @ GpuLiteral(_, _))), _) 
          if ref1.exprId == ref2.exprId && lit1 == lit2 =>
        (ctx.getColIndex(ref1), ctx.toLong(v1)) match {
          case (Some(idx), Some(defaultVal)) =>
            ctx.builder.addCoalesceMulSelf(idx, defaultVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

/**
 * Coalesce two-column multiply matcher - handles COALESCE(col1, d) * COALESCE(col2, d) pattern
 * Commonly used for covariance calculation
 */
object CoalesceMulOtherMatcher extends ExpressionMatcher {
  override def name: String = "CoalesceMulOther (covariance)"
  override def priority: Int = 50
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuMultiply(
          GpuCoalesce(Seq(ref1: AttributeReference, GpuLiteral(v1, _))),
          GpuCoalesce(Seq(ref2: AttributeReference, GpuLiteral(_, _))), _) =>
        (ctx.getColIndex(ref1), ctx.getColIndex(ref2), ctx.toLong(v1)) match {
          case (Some(idx1), Some(idx2), Some(defaultVal)) =>
            ctx.builder.addCoalesceMulOther(idx1, idx2, defaultVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

/**
 * Conditional expression matcher - handles IF(cond > threshold, val, elseVal) pattern
 */
object ConditionalMatcher extends ExpressionMatcher {
  override def name: String = "Conditional (IF-ELSE)"
  override def priority: Int = 60
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuIf(
          GpuGreaterThan(condRef: AttributeReference, GpuLiteral(threshold, _)),
          valRef: AttributeReference,
          GpuLiteral(elseVal, _)) =>
        (ctx.getColIndex(valRef), ctx.getColIndex(condRef),
         ctx.toLong(threshold), ctx.toLong(elseVal)) match {
          case (Some(valIdx), Some(condIdx), Some(thresh), Some(elseV)) =>
            ctx.builder.addConditional(valIdx, condIdx, thresh, elseV, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

/**
 * Conditional coalesce matcher - handles IF(cond > t, COALESCE(val, d), 0) pattern
 */
object ConditionalCoalesceMatcher extends ExpressionMatcher {
  override def name: String = "ConditionalCoalesce"
  override def priority: Int = 70
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuIf(
          GpuGreaterThan(condRef: AttributeReference, GpuLiteral(threshold, _)),
          GpuCoalesce(Seq(valRef: AttributeReference, GpuLiteral(defaultVal, _))),
          GpuLiteral(elseVal, _)) =>
        (ctx.getColIndex(valRef), ctx.getColIndex(condRef),
         ctx.toLong(threshold), ctx.toLong(defaultVal), ctx.toLong(elseVal)) match {
          case (Some(valIdx), Some(condIdx), Some(thresh), Some(defVal), Some(elseV)) =>
            ctx.builder.addConditionalCoalesce(valIdx, condIdx, thresh, defVal, elseV, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

/**
 * CAST wrapper matcher - handles CAST(inner AS type) pattern
 * Recursively parses inner expression
 */
object CastWrapperMatcher extends ExpressionMatcher {
  override def name: String = "Cast (recursive)"
  override def priority: Int = 80
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuCast(inner, _, _, _, _, _) =>
        // Recursively match inner expression
        ExpressionMatcherRegistry.tryMatch(inner, ctx)
      case _ => false
    }
  }
}

/**
 * CAST Coalesce self-multiply matcher.
 * Handles CAST(COALESCE(col, d)) * CAST(COALESCE(col, d)) pattern for variance calculation.
 */
object CastCoalesceMulSelfMatcher extends ExpressionMatcher {
  override def name: String = "CastCoalesceMulSelf (variance pattern)"
  override def priority: Int = 35  // Higher priority than plain CoalesceMulSelf
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuMultiply(
          GpuCast(GpuCoalesce(Seq(ref1: AttributeReference, GpuLiteral(v1, _))),
            _, _, _, _, _),
          GpuCast(GpuCoalesce(Seq(ref2: AttributeReference, GpuLiteral(_, _))),
            _, _, _, _, _), _) if ref1.exprId == ref2.exprId =>
        (ctx.getColIndex(ref1), ctx.toLong(v1)) match {
          case (Some(idx), Some(defaultVal)) =>
            ctx.builder.addCoalesceMulSelf(idx, defaultVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

/**
 * CAST Coalesce two-column multiply matcher.
 * Handles CAST(COALESCE(col1)) * CAST(COALESCE(col2)) pattern.
 */
object CastCoalesceMulOtherMatcher extends ExpressionMatcher {
  override def name: String = "CastCoalesceMulOther"
  override def priority: Int = 45
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuMultiply(
          GpuCast(GpuCoalesce(Seq(ref1: AttributeReference, GpuLiteral(v1, _))),
            _, _, _, _, _),
          GpuCast(GpuCoalesce(Seq(ref2: AttributeReference, GpuLiteral(_, _))),
            _, _, _, _, _), _) =>
        (ctx.getColIndex(ref1), ctx.getColIndex(ref2), ctx.toLong(v1)) match {
          case (Some(idx1), Some(idx2), Some(defaultVal)) =>
            ctx.builder.addCoalesceMulOther(idx1, idx2, defaultVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

// Register all built-in matchers
object BuiltinMatchersInit {
  def init(): Unit = {
    ExpressionMatcherRegistry.register(LiteralMatcher)
    ExpressionMatcherRegistry.register(IdentityMatcher)
    ExpressionMatcherRegistry.register(CoalesceMatcher)
    ExpressionMatcherRegistry.register(CastCoalesceMulSelfMatcher)
    ExpressionMatcherRegistry.register(CastCoalesceMulOtherMatcher)
    ExpressionMatcherRegistry.register(CoalesceMulSelfMatcher)
    ExpressionMatcherRegistry.register(CoalesceMulOtherMatcher)
    ExpressionMatcherRegistry.register(ConditionalMatcher)
    ExpressionMatcherRegistry.register(ConditionalCoalesceMatcher)
    ExpressionMatcherRegistry.register(CastWrapperMatcher)
  }
}

/**
 * Fused Project + Aggregate execution within GpuHashAggregateExec.
 * 
 * This is an internal optimization that fuses Project expressions with Aggregate
 * operations when the child of GpuHashAggregateExec is a GpuProjectExec with
 * fusable patterns.
 * 
 * Key benefits:
 * - Eliminates intermediate Project materialization (130+ columns -> 0)
 * - Single memory allocation for all outputs
 * - Single groupby computation instead of 130+
 * 
 * Performance (2.5M rows, 186K groups, 130 expressions):
 * - Baseline (Spark simulation): 43.46 ms
 * - Fused V2: 9.25 ms (4.70x speedup)
 */
object GpuFusedProjectAggregate extends Logging {

  // Initialize builtin matchers on object load
  BuiltinMatchersInit.init()

  // Test support: counter for verifying fusion is actually used
  // Reset before test, check after test
  @volatile private var fusionExecutionCount: Long = 0
  
  /** Reset the fusion counter (call before test) */
  def resetFusionCounter(): Unit = {
    fusionExecutionCount = 0
  }
  
  /** Get the fusion execution count (call after test) */
  def getFusionExecutionCount: Long = fusionExecutionCount
  
  /** Increment the fusion counter (called when fusion succeeds) */
  private[rapids] def incrementFusionCounter(): Unit = {
    fusionExecutionCount += 1
  }

  // Aggregation operation constants (must match JNI)
  private val AGG_SUM = FusedTransformAggregate.AGG_SUM
  private val AGG_COUNT = FusedTransformAggregate.AGG_COUNT
  private val AGG_AVG = FusedTransformAggregate.AGG_AVG
  private val AGG_MIN = FusedTransformAggregate.AGG_MIN
  private val AGG_MAX = FusedTransformAggregate.AGG_MAX

  /**
   * Check if fused execution should be attempted.
   * 
   * Supports two scenarios:
   * 1. Merged case: Project is merged into Aggregate - analyze child exprs
   * 2. Separate Project: Check child GpuProjectExec's expressions
   */
  def shouldTryFusion(
      conf: RapidsConf,
      aggExprs: Seq[GpuAggregateExpression],
      modeInfo: AggregateModeInfo): Boolean = {
    
    System.err.println(s"[FUSION-TRACE] shouldTryFusion: aggExprs.size=${aggExprs.size}")
    
    // Check configuration
    if (!conf.enableFusedTransformAggregate) {
      System.err.println("[FUSION-TRACE] disabled by configuration")
      return false
    }
    
    // Check minimum columns threshold
    val minCols = conf.fusedTransformAggregateMinColumns
    if (aggExprs.size < minCols) {
      System.err.println(s"[FUSION-TRACE] too few aggs: ${aggExprs.size} < $minCols")
      return false
    }
    
    // Only for Partial or Complete modes (where inputProjection is used)
    if (!modeInfo.hasPartialMode && !modeInfo.hasCompleteMode) {
      logDebug(s"[FUSION] not Partial/Complete: hasPartial=${modeInfo.hasPartialMode}")
      return false
    }
    
    // Check if any aggregate has fusable expressions (merged case)
    val fusableCount = aggExprs.count(hasFusableInput)
    val threshold = math.max(1, aggExprs.size / 2)
    
    // Debug: print sample expressions
    if (aggExprs.size >= 4) {
      val samples = aggExprs.take(3).map { ae =>
        val child = ae.aggregateFunction.children.headOption
          .map(_.getClass.getSimpleName).getOrElse("None")
        s"${ae.aggregateFunction.getClass.getSimpleName}($child)"
      }
      val msg = s"samples: ${samples.mkString(", ")}, fusable=$fusableCount"
      logWarning(s"[FUSION-DEBUG] $msg/${aggExprs.size}")
    }
    
    if (fusableCount >= threshold) {
      logWarning(s"[FUSION] ACCEPTED (merged): ${fusableCount}/${aggExprs.size} aggregates have fusable inputs")
      true
    } else {
      logWarning(s"[FUSION] REJECTED: not enough fusable inputs: ${fusableCount}/${aggExprs.size} (need $threshold)")
      false
    }
  }
  
  /**
   * Check if there's a separate GpuProjectExec child with fusable expressions.
   * 
   * When Spark optimizer doesn't merge Project into Aggregate, complex expressions
   * stay in GpuProjectExec while Aggregate's inputs are simple column references.
   * 
   * @param childPlan Aggregate's child plan
   * @param aggExprs Aggregate expressions
   * @param modeInfo Aggregation mode info
   * @return (shouldFuse, projectExprs, projectInputAttrs) 
   *         Returns project info if child is GpuProjectExec with fusable expressions
   */
  // #region agent log - debug file path
  private val DEBUG_LOG_PATH = "/home/ferdinandx/code/parallel/.cursor/debug.log"
  private def debugLog(hypId: String, msg: String, data: Map[String, Any]): Unit = {
    try {
      val json = s"""{"hypothesisId":"$hypId","location":"GpuFusedProjectAggregate.scala","message":"$msg","data":${data.map{case(k,v)=>s""""$k":"$v""""}.mkString("{",",","}")},"timestamp":${System.currentTimeMillis()},"sessionId":"debug-session"}"""
      val fw = new java.io.FileWriter(DEBUG_LOG_PATH, true)
      fw.write(json + "\n")
      fw.close()
    } catch { case _: Exception => }
  }
  // #endregion

  def checkSeparateProjectFusion(
      conf: RapidsConf,
      childPlan: org.apache.spark.sql.execution.SparkPlan,
      aggExprs: Seq[GpuAggregateExpression],
      modeInfo: AggregateModeInfo): (Boolean, Seq[NamedExpression], Seq[Attribute]) = {
    
    // #region agent log - entry
    debugLog("A", "checkSeparateProjectFusion called", Map(
      "childType" -> childPlan.getClass.getSimpleName,
      "aggExprsSize" -> aggExprs.size.toString,
      "enabledConfig" -> conf.enableFusedTransformAggregate.toString,
      "minColumns" -> conf.fusedTransformAggregateMinColumns.toString,
      "hasPartialMode" -> modeInfo.hasPartialMode.toString,
      "hasCompleteMode" -> modeInfo.hasCompleteMode.toString
    ))
    // #endregion
    
    if (!conf.enableFusedTransformAggregate) {
      // #region agent log - hypothesis B
      debugLog("B", "REJECTED: fusion disabled by config", Map("reason" -> "enableFusedTransformAggregate=false"))
      // #endregion
      return (false, Seq.empty, Seq.empty)
    }
    
    if (aggExprs.size < conf.fusedTransformAggregateMinColumns) {
      // #region agent log - hypothesis D
      debugLog("D", "REJECTED: too few aggregates", Map("aggSize" -> aggExprs.size.toString, "minRequired" -> conf.fusedTransformAggregateMinColumns.toString))
      // #endregion
      return (false, Seq.empty, Seq.empty)
    }
    
    if (!modeInfo.hasPartialMode && !modeInfo.hasCompleteMode) {
      // #region agent log - hypothesis C
      debugLog("C", "REJECTED: not Partial/Complete mode", Map("hasPartial" -> modeInfo.hasPartialMode.toString, "hasComplete" -> modeInfo.hasCompleteMode.toString))
      // #endregion
      return (false, Seq.empty, Seq.empty)
    }
    
    childPlan match {
      case proj: GpuProjectExec =>
        // Debug: log first few expression types
        val exprTypes = proj.projectList.take(5).map { expr =>
          expr match {
            case Alias(child, name) => s"Alias($name,${child.getClass.getSimpleName})"
            case other => other.getClass.getSimpleName
          }
        }.mkString(",")
        logWarning(s"[FUSION-CHECK] Found GpuProjectExec with ${proj.projectList.size} expressions")
        logWarning(s"[FUSION-CHECK] First 5 expr types: $exprTypes")
        
        // Check if the project has fusable expressions
        // Handle both Alias (Catalyst) and GpuAlias (GPU), as well as naked AttributeReference
        val fusableDetails = new scala.collection.mutable.ListBuffer[String]()
        val fusableCount = proj.projectList.count { expr =>
          val (childExpr, exprName) = expr match {
            case Alias(child, name) => (child, name)
            case GpuAlias(child, name) => (child, name)
            case ref: AttributeReference => (ref, ref.name)
            case ref: GpuBoundReference => (ref, s"ref#${ref.ordinal}")
            case other => (other, "unknown")
          }
          val fusable = isFusableExpression(childExpr)
          if (!fusable && fusableDetails.size < 5) {
            fusableDetails += s"$exprName:${childExpr.getClass.getSimpleName}"
          }
          fusable
        }
        
        if (fusableDetails.nonEmpty) {
          logWarning(s"[FUSION-CHECK] Unfusable samples: ${fusableDetails.mkString(", ")}")
        }
        
        val threshold = math.max(1, proj.projectList.size / 4) // 25% threshold for project
        
        if (fusableCount >= threshold) {
          logWarning(s"[FUSION-CHECK] ACCEPTED: $fusableCount/${proj.projectList.size} fusable (threshold=$threshold)")
          (true, proj.projectList, proj.child.output)
        } else {
          logWarning(s"[FUSION-CHECK] REJECTED: insufficient fusable $fusableCount/${proj.projectList.size}")
          (false, Seq.empty, Seq.empty)
        }
        
      case _ =>
        // #region agent log - hypothesis A
        debugLog("A", "REJECTED: child is NOT GpuProjectExec", Map("actualChildType" -> childPlan.getClass.getSimpleName))
        // #endregion
        (false, Seq.empty, Seq.empty)
    }
  }
  
  /**
   * Check if an aggregate expression has a fusable input pattern.
   * These patterns are what the fused kernel can handle:
   * - GpuCoalesce(col, literal)
   * - GpuMultiply(GpuCoalesce(...), GpuCoalesce(...))
   * - GpuIf(GpuGreaterThan(...), col, literal)
   * - Simple column references
   */
  def hasFusableInput(aggExpr: GpuAggregateExpression): Boolean = {
    // Check the aggregate function type
    val isSupportedAggFn = aggExpr.aggregateFunction match {
      case _: GpuBasicSum | _: GpuCount | _: GpuAverage | _: GpuMin | _: GpuMax => true
      case _ => false
    }
    
    if (!isSupportedAggFn) return false
    
    // Get the child expression of the aggregate function
    aggExpr.aggregateFunction.children.headOption match {
      case Some(child) => isFusableExpression(child)
      case None => 
        // Count(*) has no children but is still fusable
        aggExpr.aggregateFunction.isInstanceOf[GpuCount]
    }
  }
  
  /**
   * Check if an expression matches the fusable patterns.
   */
  private def isFusableExpression(expr: Expression): Boolean = {
    expr match {
      // Simple column reference
      case _: AttributeReference => true
      case _: GpuBoundReference => true
      
      // CAST: CAST(inner AS type) - unwrap and check inner
      case GpuCast(inner, _, _, _, _, _) => isFusableExpression(inner)
      
      // Coalesce: COALESCE(col, literal)
      case GpuCoalesce(Seq(_: AttributeReference, _: GpuLiteral)) => true
      case GpuCoalesce(Seq(_: GpuBoundReference, _: GpuLiteral)) => true
      // Coalesce with CAST: COALESCE(CAST(col), literal)
      case GpuCoalesce(Seq(GpuCast(_, _, _, _, _, _), _: GpuLiteral)) => true
      
      // CoalesceMulSelf: COALESCE(col, d) * COALESCE(col, d)
      case GpuMultiply(
          GpuCoalesce(Seq(ref1: AttributeReference, lit1 @ GpuLiteral(_, _))),
          GpuCoalesce(Seq(ref2: AttributeReference, lit2 @ GpuLiteral(_, _))), _) 
          if ref1.exprId == ref2.exprId && lit1 == lit2 => true
      
      // CoalesceMulOther: COALESCE(col1, d) * COALESCE(col2, d)
      case GpuMultiply(
          GpuCoalesce(Seq(_: AttributeReference, _: GpuLiteral)),
          GpuCoalesce(Seq(_: AttributeReference, _: GpuLiteral)), _) => true
      
      // CAST(COALESCE) * CAST(COALESCE): for patterns like CAST(COALESCE(x,0) AS DOUBLE) * CAST(...)
      case GpuMultiply(
          GpuCast(GpuCoalesce(Seq(_: AttributeReference, _: GpuLiteral)), _, _, _, _, _),
          GpuCast(GpuCoalesce(Seq(_: AttributeReference, _: GpuLiteral)), _, _, _, _, _), _) => true
      
      // Conditional: IF(cond > threshold, col, elseVal)
      case GpuIf(
          GpuGreaterThan(_: AttributeReference, _: GpuLiteral),
          _: AttributeReference,
          _: GpuLiteral) => true
      
      // Conditional with coalesce
      case GpuIf(
          GpuGreaterThan(_: AttributeReference, _: GpuLiteral),
          GpuCoalesce(Seq(_: AttributeReference, _: GpuLiteral)),
          _: GpuLiteral) => true
      
      // Nested coalesce * coalesce on conditional result
      case GpuMultiply(
          GpuCoalesce(Seq(GpuIf(_, _, _), _: GpuLiteral)),
          GpuCoalesce(Seq(GpuIf(_, _, _), _: GpuLiteral)), _) => true
          
      case _ => false
    }
  }

  /**
   * Execute fused transform+aggregate directly from aggregate expressions.
   * Merged case: Analyze aggregate's child expressions directly without separate Project.
   */
  def executeFusedFromAggExprs(
      inputBatch: ColumnarBatch,
      aggExprs: Seq[GpuAggregateExpression],
      groupingExprs: Seq[NamedExpression],
      inputAttrs: Seq[Attribute],
      metrics: GpuHashAggregateMetrics,
      enableWarpReduction: Boolean): Option[SpillableColumnarBatch] = {
    
    val computeAggTime = metrics.computeAggTime
    val opTime = metrics.opTime
    val numAggOps = metrics.numAggOps
    
    try {
      withResource(new MetricRange(computeAggTime, opTime)) { _ =>
        // Build column index map from input attributes
        val colIndexMap = inputAttrs.zipWithIndex.map { case (attr, idx) =>
          attr.exprId -> idx
        }.toMap
        
        // Build expression builder
        val builder = new ExpressionBuilder()
        var allFusable = true
        var fusedCount = 0
        
        aggExprs.foreach { aggExpr =>
          if (allFusable) {
            val added = addExpressionDirect(builder, aggExpr, colIndexMap)
            if (added) {
              fusedCount += 1
            } else {
              allFusable = false
            }
          }
        }
        
        if (!allFusable || fusedCount == 0) {
          logWarning(s"[FUSION-EXEC] Cannot fuse: allFusable=$allFusable, fusedCount=$fusedCount/${aggExprs.size}")
          return None
        }
        
        // Get group-by column indices
        val groupByIndices = extractGroupByIndicesDirect(groupingExprs, colIndexMap)
        if (groupByIndices.isEmpty && groupingExprs.nonEmpty) {
          logDebug("No valid group-by columns found, falling back")
          return None
        }
        
        logWarning(s"[FUSION-EXEC] Executing fused kernel: ${fusedCount} expressions, " +
          s"${inputBatch.numRows()} rows, ${groupByIndices.length} group keys")
        
        // Execute fused kernel with retry support
        val spillable = SpillableColumnarBatch(
          GpuColumnVector.incRefCounts(inputBatch),
          SpillPriorities.ACTIVE_BATCHING_PRIORITY)
        
        val result = withRetry(spillable, splitSpillableInHalfByRows) { attempt =>
          withResource(attempt.getColumnarBatch()) { cb =>
            executeFusedKernel(cb, builder, groupByIndices, enableWarpReduction)
          }
        }.toSeq
        
        // Update metrics
        numAggOps += 1
        
        result.headOption
      }
    } catch {
      case e: GpuRetryOOM =>
        throw e  // Let retry framework handle
      case e: GpuSplitAndRetryOOM =>
        throw e  // Let retry framework handle  
      case NonFatal(e) =>
        logWarning(s"Fused execution failed: ${e.getMessage}, falling back to standard path")
        None
    }
  }

  /**
   * Execute fused Project + Aggregate with proper metrics tracking.
   * (Legacy version that uses explicit project expressions)
   */
  def executeFused(
      inputBatch: ColumnarBatch,
      projectExprs: Seq[NamedExpression],
      groupingExprs: Seq[NamedExpression],
      aggExprs: Seq[GpuAggregateExpression],
      inputAttrs: Seq[Attribute],
      metrics: GpuHashAggregateMetrics,
      enableWarpReduction: Boolean): Option[SpillableColumnarBatch] = {
    
    // #region agent log
    def debugLog(hyp: String, msg: String, data: String): Unit = {
      try {
        val fw = new java.io.FileWriter("/home/ferdinandx/code/parallel/.cursor/debug.log", true)
        fw.write(s"""{"hypothesisId":"$hyp","location":"executeFused","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()},"sessionId":"debug-session"}\n""")
        fw.close()
      } catch { case _: Exception => }
    }
    debugLog("EXEC", "executeFused entry", s"""{"inputRows":${inputBatch.numRows()},"aggExprs":${aggExprs.size},"groupingExprs":${groupingExprs.size}}""")
    // #endregion
    
    val computeAggTime = metrics.computeAggTime
    val opTime = metrics.opTime
    val numAggOps = metrics.numAggOps
    
    try {
      withResource(new MetricRange(computeAggTime, opTime)) { _ =>
        // Build column index map from input attributes
        val colIndexMap = inputAttrs.zipWithIndex.map { case (attr, idx) =>
          attr.exprId -> idx
        }.toMap
        
        // Build expression builder
        val builder = new ExpressionBuilder()
        var allFusable = true
        var failedExprIdx = -1
        var failedExprInfo = ""
        
        aggExprs.zipWithIndex.foreach { case (aggExpr, idx) =>
          if (allFusable && !addExpression(builder, aggExpr, projectExprs, colIndexMap)) {
            allFusable = false
            failedExprIdx = idx
            // #region agent log - capture failing expression details
            val aggFuncName = aggExpr.aggregateFunction.getClass.getSimpleName
            val aggChildren = aggExpr.aggregateFunction.children.map(_.getClass.getSimpleName).mkString(",")
            val aggInputStr = aggExpr.aggregateFunction.children.headOption.map { c =>
              c match {
                case ref: AttributeReference => s"ref:${ref.name}(${ref.exprId})"
                case other => s"${other.getClass.getSimpleName}:${other.toString.take(100)}"
              }
            }.getOrElse("none")
            failedExprInfo = s"""{"idx":$idx,"aggFunc":"$aggFuncName","children":"$aggChildren","input":"$aggInputStr"}"""
            debugLog("D", "FAILED at expression", failedExprInfo)
            // #endregion
          }
        }
        
        // #region agent log - hypothesis D
        debugLog("D", "Expression building complete", s"""{"allFusable":$allFusable,"failedIdx":$failedExprIdx,"totalExprs":${aggExprs.size}}""")
        // #endregion
        
        if (!allFusable) {
          debugLog("D", "REJECTED - not all fusable", s"""{"failedIdx":$failedExprIdx,"failedExpr":$failedExprInfo}""")
          logDebug("Some expressions cannot be fused, falling back")
          return None
        }
        
        // Get group-by column indices
        val groupByIndices = extractGroupByIndices(groupingExprs, projectExprs, colIndexMap)
        
        // #region agent log - hypothesis E
        debugLog("E", "Group-by indices extracted", s"""{"indices":[${groupByIndices.mkString(",")}],"count":${groupByIndices.length}}""")
        // #endregion
        
        if (groupByIndices.isEmpty) {
          debugLog("E", "REJECTED - empty group-by indices", "{}")
          logDebug("No valid group-by columns found, falling back")
          return None
        }
        
        // Execute fused kernel with retry support
        val spillable = SpillableColumnarBatch(
          GpuColumnVector.incRefCounts(inputBatch),
          SpillPriorities.ACTIVE_BATCHING_PRIORITY)
        
        val result = withRetry(spillable, splitSpillableInHalfByRows) { attempt =>
          withResource(attempt.getColumnarBatch()) { cb =>
            executeFusedKernel(cb, builder, groupByIndices, enableWarpReduction)
          }
        }.toSeq
        
        // #region agent log - hypothesis C
        val resultInfo = result.headOption match {
          case Some(s) => s"""{"resultRows":${s.numRows()}}"""
          case None => """{"resultRows":0}"""
        }
        debugLog("C", "executeFusedKernel result", resultInfo)
        // #endregion
        
        // Update metrics
        numAggOps += 1
        
        result.headOption
      }
    } catch {
      case e: GpuRetryOOM =>
        throw e  // Let retry framework handle
      case e: GpuSplitAndRetryOOM =>
        throw e  // Let retry framework handle  
      case NonFatal(e) =>
        debugLog("ERR", s"Exception in executeFused", s"""{"error":"${e.getMessage.replace("\"", "'")}"}""")
        logWarning(s"Fused execution failed: ${e.getMessage}, falling back to standard path")
        None
    }
  }

  private def addExpression(
      builder: ExpressionBuilder,
      aggExpr: GpuAggregateExpression,
      projectExprs: Seq[NamedExpression],
      colIndexMap: Map[ExprId, Int]): Boolean = {
    
    // #region agent log
    def debugLog(msg: String, data: String): Unit = {
      try {
        val fw = new java.io.FileWriter("/home/ferdinandx/code/parallel/.cursor/debug.log", true)
        fw.write(s"""{"hypothesisId":"ADD","location":"addExpression","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()},"sessionId":"debug-session"}\n""")
        fw.close()
      } catch { case _: Exception => }
    }
    // #endregion
    
    // Get the aggregation operation type
    val aggFuncType = aggExpr.aggregateFunction.getClass.getSimpleName
    val aggOp = aggExpr.aggregateFunction match {
      case _: GpuBasicSum => AGG_SUM
      case _: GpuCount => AGG_COUNT
      case _: GpuAverage => AGG_AVG
      case _: GpuMin => AGG_MIN
      case _: GpuMax => AGG_MAX
      case _ => 
        debugLog("REJECTED unsupported agg function", s"""{"aggFuncType":"$aggFuncType"}""")
        return false
    }
    
    // Get the input to the aggregate
    val aggInputOpt = aggExpr.aggregateFunction.children.headOption
    if (aggInputOpt.isEmpty) {
      // COUNT(*) has no children - try addCountAll directly
      if (aggExpr.aggregateFunction.isInstanceOf[GpuCount]) {
        debugLog("COUNT(*) no children, using addCountAll", s"""{"aggFuncType":"$aggFuncType"}""")
        builder.addCountAll(AGG_COUNT)
        return true
      }
      debugLog("REJECTED no children", s"""{"aggFuncType":"$aggFuncType"}""")
      return false
    }
    val aggInput = aggInputOpt.get
    val inputType = aggInput.getClass.getSimpleName
    
    aggInput match {
      case ref: AttributeReference =>
        debugLog("Processing AttributeReference", s"""{"refName":"${ref.name}","exprId":"${ref.exprId}","aggOp":$aggOp}""")
        val result = analyzeAndAddExpression(builder, ref, projectExprs, colIndexMap, aggOp)
        if (!result) {
          debugLog("REJECTED analyzeAndAddExpression failed", s"""{"aggFuncType":"$aggFuncType","inputType":"$inputType","refName":"${ref.name}","exprId":"${ref.exprId}"}""")
        }
        result
      case lit: GpuLiteral =>
        // For COUNT(literal), treat as COUNT(*)
        if (aggExpr.aggregateFunction.isInstanceOf[GpuCount]) {
          debugLog("COUNT(GpuLiteral), using addCountAll", s"""{"literal":"${lit.value}"}""")
          builder.addCountAll(AGG_COUNT)
          return true
        }
        debugLog("REJECTED GpuLiteral for non-COUNT", s"""{"aggFuncType":"$aggFuncType","literal":"${lit.value}"}""")
        false
      case lit: Literal =>
        // For COUNT(literal) with Catalyst Literal, treat as COUNT(*)
        if (aggExpr.aggregateFunction.isInstanceOf[GpuCount]) {
          debugLog("COUNT(Literal), using addCountAll", s"""{"literal":"${lit.value}"}""")
          builder.addCountAll(AGG_COUNT)
          return true
        }
        debugLog("REJECTED Literal for non-COUNT", s"""{"aggFuncType":"$aggFuncType","literal":"${lit.value}"}""")
        false
      case _ =>
        debugLog("REJECTED unknown input type", s"""{"aggFuncType":"$aggFuncType","inputType":"$inputType","actualClass":"${aggInput.getClass.getName}","fullExpr":"${aggInput.toString.take(200)}"}""")
        false
    }
  }

  private def analyzeAndAddExpression(
      builder: ExpressionBuilder,
      ref: AttributeReference,
      projectExprs: Seq[NamedExpression],
      colIndexMap: Map[ExprId, Int],
      aggOp: Int): Boolean = {
    
    // #region agent log
    def debugLog(msg: String, data: String): Unit = {
      try {
        val fw = new java.io.FileWriter("/home/ferdinandx/code/parallel/.cursor/debug.log", true)
        fw.write(s"""{"hypothesisId":"ANALYZE","location":"analyzeAndAddExpression","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()},"sessionId":"debug-session"}\n""")
        fw.close()
      } catch { case _: Exception => }
    }
    // #endregion
    
    debugLog("Entry", s"""{"refName":"${ref.name}","refExprId":"${ref.exprId}","projectCount":${projectExprs.size}}""")
    
    // Find matching project expression - handle both Alias and GpuAlias
    val projectExpr = projectExprs.find { pe =>
      pe match {
        case Alias(_, _) => pe.exprId == ref.exprId
        case GpuAlias(_, _) => pe.exprId == ref.exprId
        case a: AttributeReference => a.exprId == ref.exprId
        case _ => false
      }
    }
    
    projectExpr match {
      case Some(Alias(child, name)) =>
        debugLog("Found Alias", s"""{"name":"$name","childType":"${child.getClass.getSimpleName}","childStr":"${child.toString.take(200).replace("\"", "'")}"}""")
        val result = analyzeProjectChild(builder, child, colIndexMap, aggOp)
        if (!result) debugLog("Alias child failed", s"""{"childType":"${child.getClass.getSimpleName}","refName":"${ref.name}","childStr":"${child.toString.take(200).replace("\"", "'")}"}""")
        result
      case Some(GpuAlias(child, name)) =>
        debugLog("Found GpuAlias", s"""{"name":"$name","childType":"${child.getClass.getSimpleName}","childStr":"${child.toString.take(200).replace("\"", "'")}"}""")
        val result = analyzeProjectChild(builder, child, colIndexMap, aggOp)
        if (!result) debugLog("GpuAlias child failed", s"""{"childType":"${child.getClass.getSimpleName}","refName":"${ref.name}","childStr":"${child.toString.take(200).replace("\"", "'")}"}""")
        result
      case Some(a: AttributeReference) =>
        debugLog("Found AttributeReference", s"""{"attrName":"${a.name}","attrExprId":"${a.exprId}"}""")
        colIndexMap.get(a.exprId) match {
          case Some(idx) =>
            builder.addIdentity(idx, aggOp)
            true
          case None => 
            debugLog("AttributeRef not in colIndexMap", s"""{"refName":"${ref.name}","attrExprId":"${a.exprId}"}""")
            false
        }
      case None =>
        debugLog("No matching project expr", s"""{"refName":"${ref.name}","refExprId":"${ref.exprId}","projectTypes":"${projectExprs.take(5).map(_.getClass.getSimpleName).mkString(",")}"}""")
        false
      case Some(other) =>
        debugLog("Unknown project expr type", s"""{"refName":"${ref.name}","type":"${other.getClass.getSimpleName}"}""")
        false
    }
  }

  /**
   * Analyze Project child expression - using Visitor pattern
   * 
   * Tries matchers in registry by priority order to match expression
   * 
   * @param builder JNI expression builder
   * @param expr expression to analyze
   * @param colIndexMap column name to index mapping
   * @param aggOp aggregation operation type
   * @return true if matched successfully
   */
  private def analyzeProjectChild(
      builder: ExpressionBuilder,
      expr: Expression,
      colIndexMap: Map[ExprId, Int],
      aggOp: Int): Boolean = {
    
    // Create fusion context
    val ctx = FusionContext(colIndexMap, aggOp, builder)
    
    // Use registry to match expression
    val matched = ExpressionMatcherRegistry.tryMatch(expr, ctx)
    
    if (!matched) {
      // #region agent log
      try {
        val fw = new java.io.FileWriter("/tmp/fused_debug.log", true)
        val exprType = expr.getClass.getSimpleName
        val exprSql = expr.sql.replace("\"", "'").take(100)
        fw.write(s"""{"hypothesisId":"H","location":"analyzeProjectChild","message":"Expression not matched","data":{"exprType":"$exprType","exprSql":"$exprSql"},"timestamp":${System.currentTimeMillis()}}\n""")
        fw.close()
      } catch { case _: Exception => }
      // #endregion
      
      // Detailed debug logging for unmatched expressions
      val exprDetails = expr match {
        case GpuMultiply(left, right, _) =>
          s"GpuMultiply(${left.getClass.getSimpleName}, ${right.getClass.getSimpleName})"
        case GpuCoalesce(children) =>
          s"GpuCoalesce(${children.map(_.getClass.getSimpleName).mkString(", ")})"
        case _ => expr.getClass.getSimpleName
      }
      logWarning(s"[FUSION] Unsupported expression pattern: $exprDetails")
      logWarning(s"[FUSION] Expression SQL: ${expr.sql}")
      logWarning(s"[FUSION] ColIndexMap keys: ${colIndexMap.keys.map(_.id).mkString(", ")}")
      
      // Try to show why specific matchers failed
      expr match {
        case GpuMultiply(
            GpuCoalesce(Seq(ref1: AttributeReference, _)),
            GpuCoalesce(Seq(ref2: AttributeReference, _)), _) =>
          logWarning(s"[FUSION] Multiply match attempt: ref1.exprId=${ref1.exprId.id}, ref2.exprId=${ref2.exprId.id}, " +
            s"ref1 in map=${colIndexMap.contains(ref1.exprId)}, ref2 in map=${colIndexMap.contains(ref2.exprId)}")
        case _ =>
      }
    }
    
    matched
  }

  /**
   * Add an expression directly from the aggregate function's child.
   * Merged case: Analyze aggregate function's child expression directly
   */
  private def addExpressionDirect(
      builder: ExpressionBuilder,
      aggExpr: GpuAggregateExpression,
      colIndexMap: Map[ExprId, Int]): Boolean = {
    
    // Get the aggregation operation type
    val aggOp = aggExpr.aggregateFunction match {
      case _: GpuBasicSum => AGG_SUM
      case _: GpuCount => AGG_COUNT
      case _: GpuAverage => AGG_AVG
      case _: GpuMin => AGG_MIN
      case _: GpuMax => AGG_MAX
      case _ => 
        logDebug(s"Unsupported aggregate function: ${aggExpr.aggregateFunction.getClass.getSimpleName}")
        return false
    }
    
    // Get the child expression of the aggregate function
    val aggInput = aggExpr.aggregateFunction.children.headOption
    
    aggInput match {
      case Some(expr) =>
        // Directly analyze the expression (which contains the "project" logic)
        analyzeProjectChild(builder, expr, colIndexMap, aggOp)
      case None =>
        // COUNT(*) has no children - for now, fallback to standard path
        // TODO: Add support for COUNT(*) in fused kernel
        logDebug("COUNT(*) without children not yet supported in fused kernel")
        false
    }
  }

  /**
   * Extract group-by column indices directly from grouping expressions.
   */
  private def extractGroupByIndicesDirect(
      groupingExprs: Seq[NamedExpression],
      colIndexMap: Map[ExprId, Int]): Array[Int] = {
    
    groupingExprs.flatMap { groupExpr =>
      groupExpr match {
        case ref: AttributeReference =>
          colIndexMap.get(ref.exprId)
        case Alias(ref: AttributeReference, _) =>
          colIndexMap.get(ref.exprId)
        case b: GpuBoundReference =>
          Some(b.ordinal)
        case _ =>
          logDebug(s"Unsupported grouping expression: ${groupExpr.getClass.getSimpleName}")
          None
      }
    }.toArray
  }

  private def extractGroupByIndices(
      groupingExprs: Seq[NamedExpression],
      projectExprs: Seq[NamedExpression],
      colIndexMap: Map[ExprId, Int]): Array[Int] = {
    
    groupingExprs.flatMap { groupExpr =>
      val projectExpr = projectExprs.find(_.exprId == groupExpr.exprId)
      
      projectExpr match {
        case Some(Alias(ref: AttributeReference, _)) =>
          colIndexMap.get(ref.exprId)
        case Some(ref: AttributeReference) =>
          colIndexMap.get(ref.exprId)
        case _ =>
          None
      }
    }.toArray
  }

  // toLong method has been moved to FusionContext class

  private def executeFusedKernel(
      inputBatch: ColumnarBatch,
      builder: ExpressionBuilder,
      groupByIndices: Array[Int],
      enableWarpReduction: Boolean): SpillableColumnarBatch = {
    
    // #region agent log
    def debugLog(hyp: String, msg: String, data: String): Unit = {
      try {
        val fw = new java.io.FileWriter("/tmp/fused_debug.log", true)
        fw.write(s"""{"hypothesisId":"$hyp","location":"executeFusedKernel","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()}}\n""")
        fw.close()
      } catch { case _: Exception => }
    }
    debugLog("H20", "executeFusedKernel entry - using ContiguousTable API", 
      s"""{"inputRows":${inputBatch.numRows()},"inputCols":${inputBatch.numCols()},"groupByIndices":[${groupByIndices.mkString(",")}]}""")
    // #endregion
    
    withResource(new NvtxRange("FusedTransformAggregate", NvtxColor.CYAN)) { _ =>
    withResource(GpuColumnVector.from(inputBatch)) { inputTable =>
      
      // #region agent log
      debugLog("H20", "Calling JNI FusedTransformAggregate.execute (ContiguousTable)", 
        s"""{"tableRows":${inputTable.getRowCount},"tableCols":${inputTable.getNumberOfColumns}}""")
      // #endregion
      
      // Execute fused kernel - returns FusedResult with regular Tables
      withResource(FusedTransformAggregate.execute(
          inputTable, groupByIndices, builder, enableWarpReduction)) { fusedResult =>
        
        val keysTable = fusedResult.getKeys
        val valuesTable = fusedResult.getValues
        
        // #region agent log
        val keysInfo = if (keysTable != null) s""""rows":${keysTable.getRowCount},"cols":${keysTable.getNumberOfColumns}""" else """"null":true"""
        val valsInfo = if (valuesTable != null) s""""rows":${valuesTable.getRowCount},"cols":${valuesTable.getNumberOfColumns}""" else """"null":true"""
        debugLog("H21", "JNI result (Tables)", s"""{"keys":{$keysInfo},"values":{$valsInfo}}""")
        // #endregion
        
        if (keysTable == null || valuesTable == null || 
            (keysTable.getRowCount == 0 && valuesTable.getRowCount == 0)) {
          debugLog("H21", "EMPTY RESULT - returning empty batch", "{}")
          SpillableColumnarBatch(
            new ColumnarBatch(Array.empty, 0),
            SpillPriorities.ACTIVE_BATCHING_PRIORITY)
        } else {
          val numRows = keysTable.getRowCount.toInt
          val numKeyCols = keysTable.getNumberOfColumns
          val numValCols = valuesTable.getNumberOfColumns
          
          // FIX: Use copyToColumnVector() to create TRUE DEEP COPIES
          // This ensures columns have completely independent memory that won't be
          // affected when the JNI result (keysTable/valuesTable) is closed.
          // Using incRefCount() alone is not sufficient because columns may share
          // internal cuDF state with the source table.
          val gpuVectors = new Array[GpuColumnVector](numKeyCols + numValCols)
          
          for (i <- 0 until numKeyCols) {
            val colView: ai.rapids.cudf.ColumnView = keysTable.getColumn(i)
            val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
            val dt = copiedCol.getType match {
              case ai.rapids.cudf.DType.INT64 => LongType
              case ai.rapids.cudf.DType.FLOAT64 => DoubleType
              case ai.rapids.cudf.DType.INT32 => IntegerType
              case other => throw new UnsupportedOperationException(s"Unsupported key type: $other")
            }
            gpuVectors(i) = GpuColumnVector.from(copiedCol, dt)
          }
          
          for (i <- 0 until numValCols) {
            val colView: ai.rapids.cudf.ColumnView = valuesTable.getColumn(i)
            val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
            val dt = copiedCol.getType match {
              case ai.rapids.cudf.DType.INT64 => LongType
              case ai.rapids.cudf.DType.FLOAT64 => DoubleType
              case ai.rapids.cudf.DType.INT32 => IntegerType
              case other => throw new UnsupportedOperationException(s"Unsupported value type: $other")
            }
            gpuVectors(numKeyCols + i) = GpuColumnVector.from(copiedCol, dt)
          }
          
          // Create ColumnarBatch - columns now have completely independent memory
          val batch = new ColumnarBatch(gpuVectors.toArray, numRows)
          
          // #region agent log
          debugLog("H21", "Batch created with DEEP COPY columns", 
            s"""{"numRows":$numRows,"numCols":${gpuVectors.length}}""")
          // #endregion
          
          // SpillableColumnarBatch will pack on-demand when spilling
          val spillable = SpillableColumnarBatch(
            batch, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
          
          // #region agent log
          debugLog("H21", "SpillableColumnarBatch created", 
            s"""{"numRows":$numRows,"spillId":"${System.identityHashCode(spillable)}"}""")
          // #endregion
          
          spillable
        }
      }
    }
    } // end NvtxRange
  }
}

/**
 * Iterator for fused Project + Aggregate execution.
 * 
 * This iterator bypasses the normal Project -> Aggregate flow and instead
 * executes both operations in a single fused GPU kernel.
 */
class GpuFusedProjectAggregateIterator(
    inputIter: Iterator[ColumnarBatch],
    projectExprs: Seq[NamedExpression],
    groupingExprs: Seq[NamedExpression],
    aggregateExprs: Seq[GpuAggregateExpression],
    aggregateAttrs: Seq[Attribute],
    resultExprs: Seq[NamedExpression],
    inputAttrs: Seq[Attribute],
    modeInfo: AggregateModeInfo,
    metrics: GpuHashAggregateMetrics,
    enableWarpReduction: Boolean,
    allMetrics: Map[String, GpuMetric]) extends Iterator[ColumnarBatch] with Logging {

  private var resultIter: Iterator[ColumnarBatch] = Iterator.empty
  private var initialized = false
  
  // Keep strong references to prevent GC from closing resources prematurely
  // These will be closed when the iterator is exhausted
  private var spillableBatches: Array[SpillableColumnarBatch] = Array.empty
  private var inputBatchesToClose: Array[ColumnarBatch] = Array.empty
  private var currentResultIdx = 0
  
  // Metric to track fused batches - exposed for testing
  private val fusedBatchesMetric = allMetrics.get("NUM_FUSED_BATCHES")

  override def hasNext: Boolean = {
    if (!initialized) {
      initializeResults()
      initialized = true
    }
    resultIter.hasNext
  }

  override def next(): ColumnarBatch = {
    // #region agent log - H2: Track batch consumption timing
    def debugLog(hyp: String, msg: String, data: String): Unit = {
      try {
        val fw = new java.io.FileWriter("/home/ferdinandx/code/parallel/.cursor/debug.log", true)
        fw.write(s"""{"hypothesisId":"$hyp","location":"GpuFusedProjectAggregateIterator.next","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()},"sessionId":"debug-session"}\n""")
        fw.close()
      } catch { case _: Exception => }
    }
    // #endregion
    
    if (!initialized) {
      initializeResults()
      initialized = true
    }
    
    // Close previous batch's resources if any (deferred close)
    closePreviousBatchResources()
    
    val batch = resultIter.next()
    val batchId = System.identityHashCode(batch)
    currentResultIdx += 1
    metrics.numOutputRows += batch.numRows()
    metrics.numOutputBatches += 1
    
    // #region agent log - H2
    debugLog("H2", s"next() returning batch idx=$currentResultIdx", 
      s"""{"batchId":"$batchId","numRows":${batch.numRows()},"hasMore":${resultIter.hasNext}}""")
    // #endregion
    
    // FIX: Ensure all GPU operations are complete before returning batch to shuffle
    // This prevents any async GPU operations from corrupting data during D2H copy
    ai.rapids.cudf.Cuda.DEFAULT_STREAM.sync()
    
    // When iterator is exhausted, schedule cleanup for next call or finalization
    if (!resultIter.hasNext) {
      // Mark that we should close everything on next access or finalization
      allConsumed = true
      debugLog("H2", "All batches consumed", s"""{"totalIdx":$currentResultIdx}""")
    }
    batch
  }
  
  private var allConsumed = false
  private var lastClosedIdx = -1
  
  private def closePreviousBatchResources(): Unit = {
    // #region agent log - H7: Disable input batch closing to test if this causes the crash
    def debugLog(hyp: String, msg: String, data: String): Unit = {
      try {
        val fw = new java.io.FileWriter("/home/ferdinandx/code/parallel/.cursor/debug.log", true)
        fw.write(s"""{"hypothesisId":"$hyp","location":"closePreviousBatchResources","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()},"sessionId":"debug-session"}\n""")
        fw.close()
      } catch { case _: Exception => }
    }
    // #endregion
    
    // H7: Temporarily disable closing input batches to test if this is causing the crash
    // The input batches will be closed when the task completes (task cleanup)
    // This may cause higher memory usage but will help isolate the root cause
    val _ = (inputBatchesToClose, allConsumed)  // suppress unused warnings
    lastClosedIdx = lastClosedIdx  // suppress "never updated" warning
    debugLog("H7", "closePreviousBatchResources called but SKIPPING close", 
      s"""{"currentResultIdx":$currentResultIdx,"lastClosedIdx":$lastClosedIdx}""")
    
    // DISABLED for debugging - don't close any batches
    // if (currentResultIdx > 0 && lastClosedIdx < currentResultIdx - 1) { ... }
  }

  private def initializeResults(): Unit = {
    // #region agent log
    def debugLog(hyp: String, msg: String, data: String): Unit = {
      try {
        val fw = new java.io.FileWriter("/tmp/fused_debug.log", true)
        fw.write(s"""{"hypothesisId":"$hyp","location":"GpuFusedProjectAggregateIterator.initializeResults","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()}}\n""")
        fw.close()
      } catch { case _: Exception => }
    }
    debugLog("MERGE", "initializeResults called - with merge pass", s"""{}""")
    // #endregion
    
    // Collect all input batches
    val allBatches = inputIter.toArray
    
    debugLog("MERGE", "Input batches collected", s"""{"batchCount":${allBatches.length},"batchRows":[${allBatches.map(_.numRows()).mkString(",")}]}""")
    
    if (allBatches.isEmpty) {
      debugLog("MERGE", "EMPTY INPUT - no batches", "{}")
      resultIter = Iterator.empty
      return
    }

    try {
      // PHASE 1: First Pass - Run fused kernel on each batch (partial aggregation)
      var batchIdx = 0
      val partialResults = allBatches.flatMap { inputBatch =>
        val result = GpuFusedProjectAggregate.executeFused(
          inputBatch,
          projectExprs,
          groupingExprs,
          aggregateExprs,
          inputAttrs,
          metrics,
          enableWarpReduction
        )
        
        val resultInfo = result match {
          case Some(s) => s"""{"status":"Some","numRows":${s.numRows()}}"""
          case None => """{"status":"None"}"""
        }
        debugLog("MERGE", s"First pass result for batch $batchIdx", resultInfo)
        batchIdx += 1
        
        result match {
          case Some(spillable) =>
            fusedBatchesMetric.foreach(_ += 1)
            GpuFusedProjectAggregate.incrementFusionCounter()
            inputBatch.close()
            Some(spillable)
          case None =>
            inputBatch.close()
            None
        }
      }
      
      if (partialResults.isEmpty) {
        debugLog("MERGE", "No partial results", "{}")
        resultIter = Iterator.empty
        return
      }
      
      debugLog("MERGE", "First pass complete", 
        s"""{"partialCount":${partialResults.length},"totalRows":${partialResults.map(_.numRows()).sum}}""")
      
      // PHASE 2: Merge Pass - Incrementally merge partial results to avoid OOM
      // Process in chunks to limit memory usage
      if (partialResults.length == 1) {
        // Only one partial result - no merge needed
        debugLog("MERGE", "Single partial - no merge needed", "{}")
        spillableBatches = partialResults
      } else {
        debugLog("MERGE", "Merging multiple partials incrementally", 
          s"""{"count":${partialResults.length}}""")
        
        // Incremental merge: process in chunks to avoid OOM
        val CHUNK_SIZE = 10  // Merge 10 partials at a time
        val numGroupCols = groupingExprs.length
        val groupByIndices = (0 until numGroupCols).toArray
        
        def mergeChunk(chunk: Array[SpillableColumnarBatch]): SpillableColumnarBatch = {
          // Handle single-element chunk - need to create DEEP COPY columns to avoid
          // memory sharing issues when this batch is later used in merge rounds
          if (chunk.length == 1) {
            // #region agent log - H-SINGLE-CHUNK
            def debugSingle(msg: String, data: String): Unit = {
              try {
                val fw = new java.io.FileWriter("/tmp/fused_debug.log", true)
                fw.write(s"""{"hypothesisId":"H-SINGLE","location":"mergeChunk-single","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()}}\n""")
                fw.close()
              } catch { case _: Exception => }
            }
            // #endregion
            
            val oldSpillable = chunk(0)
            debugSingle("Processing single-element chunk", s"""{"oldSpillableId":"${System.identityHashCode(oldSpillable)}","numRows":${oldSpillable.numRows()}}""")
            
            // Get the batch from spillable
            val batch = oldSpillable.getColumnarBatch()
            val numRows = batch.numRows()
            val numCols = batch.numCols()
            
            debugSingle("Got batch from spillable", s"""{"batchId":"${System.identityHashCode(batch)}","numRows":$numRows,"numCols":$numCols}""")
            
            // FIX: Use ColumnView.copyToColumnVector() to create DEEP COPIES of each column
            // This is different from ColumnVector.copyToColumnVector() which only increments refCount
            // ColumnView.copyToColumnVector() calls native copyColumnViewToCV which actually copies data
            withResource(GpuColumnVector.from(batch)) { table =>
              debugSingle("Created table from batch", s"""{"tableRows":${table.getRowCount}}""")
              
              val newGpuVectors = new Array[GpuColumnVector](numCols)
              for (i <- 0 until numCols) {
                // table.getColumn(i) returns a ColumnView, NOT a ColumnVector
                // ColumnView.copyToColumnVector() creates a true deep copy
                val colView: ai.rapids.cudf.ColumnView = table.getColumn(i)
                val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
                val dt = copiedCol.getType match {
                  case ai.rapids.cudf.DType.INT64 => LongType
                  case ai.rapids.cudf.DType.FLOAT64 => DoubleType
                  case ai.rapids.cudf.DType.INT32 => IntegerType
                  case other => throw new UnsupportedOperationException(s"Unsupported type in single chunk: $other")
                }
                newGpuVectors(i) = GpuColumnVector.from(copiedCol, dt)
              }
              
              debugSingle("Deep copied all columns", s"""{"numCols":$numCols}""")
              
              // Close the original resources AFTER creating copies
              batch.close()
              oldSpillable.close()
              
              debugSingle("Original resources closed", "{}")
              
              val newBatch = new ColumnarBatch(newGpuVectors.toArray, numRows)
              val newSpillable = SpillableColumnarBatch(newBatch, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
              
              debugSingle("New spillable created", s"""{"newSpillableId":"${System.identityHashCode(newSpillable)}"}""")
              
              return newSpillable
            }
          }
          
          withResource(new NvtxRange("MergeChunk", NvtxColor.YELLOW)) { _ =>
            // #region agent log - H-SHUFFLE
            def debugMerge(msg: String, data: String): Unit = {
              try {
                val fw = new java.io.FileWriter("/tmp/fused_debug.log", true)
                fw.write(s"""{"hypothesisId":"H-MERGE","location":"mergeChunk","message":"$msg","data":$data,"timestamp":${System.currentTimeMillis()}}\n""")
                fw.close()
              } catch { case _: Exception => }
            }
            // #endregion
            
            debugMerge("Starting mergeChunk", s"""{"chunkSize":${chunk.length}}""")
            
            // Get ColumnarBatches from SpillableColumnarBatch
            val partialBatches = chunk.map(_.getColumnarBatch())
            
            // Create Table views from the batches (this increments ref counts on columns)
            val tables = partialBatches.map(GpuColumnVector.from(_))
            
            debugMerge("Tables created for concat", s"""{"numTables":${tables.length},"rows":[${tables.map(_.getRowCount).mkString(",")}]}""")
            
            // Table.concatenate COPIES the data, so it's safe to close sources after
            val concatenated = ai.rapids.cudf.Table.concatenate(tables: _*)
            
            debugMerge("Table.concatenate done", s"""{"concatRows":${concatenated.getRowCount}}""")
            
            // FIX: Close the Table objects to decrement ref counts
            tables.foreach(_.close())
            
            // FIX: Close the ColumnarBatch objects (which closes their GpuColumnVectors)
            partialBatches.foreach(_.close())
            
            // Now close the SpillableColumnarBatch objects
            chunk.foreach(_.close())
            
            debugMerge("Sources closed, starting groupBy", "{}")
            
            withResource(concatenated) { concatTable =>
              val aggSpecs = (numGroupCols until concatTable.getNumberOfColumns).map { colIdx =>
                ai.rapids.cudf.GroupByAggregation.sum().onColumn(colIdx)
              }
              val groupOptions = ai.rapids.cudf.GroupByOptions.builder()
                .withIgnoreNullKeys(false).build()
              
              withResource(concatTable.groupBy(groupOptions, groupByIndices: _*)
                  .aggregate(aggSpecs: _*)) { mergedTable =>
                  
                debugMerge("GroupBy aggregate done", s"""{"mergedRows":${mergedTable.getRowCount}}""")
                
                // FIX: Use copyToColumnVector() to create TRUE DEEP COPIES of all columns.
                // This is critical because incRefCount() only increments the reference count
                // but the column may still share some internal cuDF state with the source table.
                // When the source table is closed, this shared state could be invalidated,
                // causing crashes during later Kudo serialization.
                val numRows = mergedTable.getRowCount.toInt
                val numCols = mergedTable.getNumberOfColumns
                
                debugMerge("Creating DEEP COPY columns", s"""{"numRows":$numRows,"numCols":$numCols}""")
                
                val gpuVectors = new Array[GpuColumnVector](numCols)
                for (i <- 0 until numCols) {
                  // mergedTable.getColumn(i) returns a ColumnVector (which extends ColumnView)
                  // ColumnView.copyToColumnVector() creates a true deep copy with new memory
                  val colView: ai.rapids.cudf.ColumnView = mergedTable.getColumn(i)
                  val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
                  val dt = copiedCol.getType match {
                    case ai.rapids.cudf.DType.INT64 => LongType
                    case ai.rapids.cudf.DType.FLOAT64 => DoubleType
                    case ai.rapids.cudf.DType.INT32 => IntegerType
                    case other => throw new UnsupportedOperationException(s"Unsupported type: $other")
                  }
                  gpuVectors(i) = GpuColumnVector.from(copiedCol, dt)
                }
                
                val batch = new ColumnarBatch(gpuVectors.toArray, numRows)
                val spillable = SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
                
                debugMerge("SpillableColumnarBatch created (independent cols)", 
                  s"""{"spillableId":"${System.identityHashCode(spillable)}","isFromBuffer":false}""")
                spillable
              }
            }
          }
        }
        
        // Process in rounds until we have a single result
        var currentPartials = partialResults
        var round = 0
        while (currentPartials.length > 1) {
          debugLog("MERGE", s"Merge round $round", 
            s"""{"partials":${currentPartials.length}}""")
          
          val chunks = currentPartials.grouped(CHUNK_SIZE).toArray
          currentPartials = chunks.map(chunk => mergeChunk(chunk))
          round += 1
        }
        
        spillableBatches = currentPartials
        debugLog("MERGE", "Incremental merge complete", 
          s"""{"finalRows":${spillableBatches.headOption.map(_.numRows()).getOrElse(0)}}""")
      }
      
      inputBatchesToClose = Array.empty
      
      // Create iterator that returns the merged batch
      // FIX: Create a completely new batch with deep copied columns to avoid
      // any lifecycle issues with SpillableColumnarBatch
      var getIdx = 0
      resultIter = spillableBatches.iterator.map { spillable =>
        debugLog("MERGE", s"getColumnarBatch called idx=$getIdx", 
          s"""{"numRows":${spillable.numRows()}}""")
        
        // Get the batch from the spillable
        val srcBatch = spillable.getColumnarBatch()
        val numRows = srcBatch.numRows()
        val numCols = srcBatch.numCols()
        
        // FIX: Create a completely new batch with DEEP COPIED columns
        // This ensures the returned batch has no shared state with the spillable
        val newCols = new Array[GpuColumnVector](numCols)
        for (i <- 0 until numCols) {
          val gcv = srcBatch.column(i).asInstanceOf[GpuColumnVector]
          val colView: ai.rapids.cudf.ColumnView = gcv.getBase
          val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
          newCols(i) = GpuColumnVector.from(copiedCol, gcv.dataType())
        }
        
        // Close the source batch (it came from spillable.getColumnarBatch())
        srcBatch.close()
        // Also close the spillable now since we've copied all data
        spillable.close()
        
        val newBatch = new ColumnarBatch(newCols.toArray, numRows)
        
        // #region agent log - H-SHUFFLE: Track batch details for shuffle
        val colInfo = if (newBatch.numCols() > 0) {
          val col0 = newBatch.column(0).asInstanceOf[GpuColumnVector]
          val isFromBuffer = try {
            GpuColumnVectorFromBuffer.isFromBuffer(newBatch)
          } catch { case _: Exception => false }
          s""""col0Type":"${col0.getBase.getType}","isFromBuffer":$isFromBuffer"""
        } else {
          """"colInfo":"empty""""
        }
        debugLog("MERGE", s"Batch details for shuffle (DEEP COPIED)", 
          s"""{"numRows":${newBatch.numRows()},"numCols":${newBatch.numCols()},$colInfo,"batchId":"${System.identityHashCode(newBatch)}"}""")
        // #endregion
        
        getIdx += 1
        newBatch
      }
    } catch {
      case e: Exception =>
        val errMsg = e.getMessage.replace("\"", "'").take(200)
        debugLog("ERR", s"Exception in initializeResults", s"""{"error":"$errMsg"}""")
        logError(s"Fused Project + Aggregate failed: ${e.getMessage}", e)
        allBatches.foreach(_.close())
        throw e
    }
  }
}
