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

import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{splitSpillableInHalfByRows, withRetry}
import com.nvidia.spark.rapids.jni.{FusedTransformAggregate, GpuRetryOOM, GpuSplitAndRetryOOM}
import com.nvidia.spark.rapids.jni.FusedTransformAggregate.ExpressionBuilder

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.rapids.{GpuAdd, GpuGreaterThan, GpuMultiply, GpuSubtract}
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuAverage,
  GpuBasicSum, GpuCount, GpuMax, GpuMin, GpuSum}
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.Decimal
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
  
  /**
   * Try to convert value to Long.
   * 
   * Supports: Long, Int, Short, Byte, Double, Float, BigDecimal, Decimal
   * For Decimal types: Only converts if value can be represented exactly as Long
   * (i.e., has no fractional part and is within Long range)
   */
  def toLong(value: Any): Option[Long] = value match {
    case l: Long => Some(l)
    case i: Int => Some(i.toLong)
    case s: Short => Some(s.toLong)
    case b: Byte => Some(b.toLong)
    case d: Double => 
      // Only convert if it's an exact integer value
      if (d.isWhole && d >= Long.MinValue && d <= Long.MaxValue) Some(d.toLong)
      else None
    case f: Float =>
      // Only convert if it's an exact integer value
      if (f.isWhole && f >= Long.MinValue && f <= Long.MaxValue) Some(f.toLong)
      else None
    // Handle Spark's Decimal type
    case dec: Decimal =>
      try {
        // Check if decimal can be represented as long without losing precision
        val bd = dec.toJavaBigDecimal
        if (bd.scale() <= 0 || bd.stripTrailingZeros().scale() <= 0) {
          // Integer value or value with only trailing zeros
          val longVal = bd.setScale(0, java.math.RoundingMode.UNNECESSARY).longValueExact()
          Some(longVal)
        } else {
          // Has fractional part - cannot convert to long without losing precision
          None
        }
      } catch {
        case _: ArithmeticException => None  // Overflow or precision loss
      }
    // Handle Java BigDecimal directly
    case bd: java.math.BigDecimal =>
      try {
        if (bd.scale() <= 0 || bd.stripTrailingZeros().scale() <= 0) {
          val longVal = bd.setScale(0, java.math.RoundingMode.UNNECESSARY).longValueExact()
          Some(longVal)
        } else {
          None
        }
      } catch {
        case _: ArithmeticException => None
      }
    // Handle Scala BigDecimal
    case bd: scala.math.BigDecimal =>
      try {
        if (bd.scale <= 0 || bd.underlying().stripTrailingZeros().scale() <= 0) {
          val longVal = bd.underlying().setScale(0, java.math.RoundingMode.UNNECESSARY).longValueExact()
          Some(longVal)
        } else {
          None
        }
      } catch {
        case _: ArithmeticException => None
      }
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
        val _ = ctx.builder.addIdentity(0, FusedTransformAggregate.AGG_COUNT)
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
          case Some(colIdx @ _) =>
            ctx.builder.addIdentity(colIdx, ctx.aggOp)
            true
          case None => false
        }
      case _ => false
    }
  }
}

/**
 * Coalesce matcher - handles COALESCE(col, default) pattern
 * 
 * Supports default values of types: Long, Int, Short, Byte, Double, Float, Decimal
 * For Decimal defaults, only exact integer values are supported (e.g., 0, 1, -1)
 * Fractional Decimal values cannot be converted to Long for JNI and will cause fallback.
 */
object CoalesceMatcher extends ExpressionMatcher with Logging {
  override def name: String = "Coalesce"
  override def priority: Int = 30
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuCoalesce(Seq(ref: AttributeReference, GpuLiteral(value, dataType))) =>
        val colIdx = ctx.getColIndex(ref)
        val longVal = ctx.toLong(value)
        
        (colIdx, longVal) match {
          case (Some(cIdx @ _), Some(defVal @ _)) =>
            ctx.builder.addCoalesce(cIdx, defVal, ctx.aggOp)
            true
          case (None, _) =>
            logDebug(s"[FUSION] CoalesceMatcher: column ${ref.name} not found in colIndexMap")
            false
          case (_, None) =>
            // Detailed logging for debugging Decimal conversion issues
            val valueInfo = value match {
              case d: Decimal => s"Decimal(${d.toString}, precision=${d.precision}, scale=${d.scale})"
              case bd: java.math.BigDecimal => s"BigDecimal(${bd.toString}, scale=${bd.scale})"
              case other => s"${other.getClass.getSimpleName}($other)"
            }
            logDebug(s"[FUSION] CoalesceMatcher: cannot convert default value to Long: " +
              s"value=$valueInfo, dataType=$dataType. " +
              "Only integer-representable values are supported for fusion.")
            false
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
          case (Some(colIdx @ _), Some(defVal @ _)) =>
            ctx.builder.addCoalesceMulSelf(colIdx, defVal, ctx.aggOp)
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
          case (Some(idx1 @ _), Some(idx2 @ _), Some(defVal @ _)) =>
            ctx.builder.addCoalesceMulOther(idx1, idx2, defVal, ctx.aggOp)
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
      // GpuIf pattern
      case GpuIf(
          GpuGreaterThan(condRef: AttributeReference, GpuLiteral(threshold, _)),
          valRef: AttributeReference,
          GpuLiteral(elseVal, _)) =>
        (ctx.getColIndex(valRef), ctx.getColIndex(condRef),
         ctx.toLong(threshold), ctx.toLong(elseVal)) match {
          case (Some(vIdx @ _), Some(cIdx @ _), Some(thr @ _), Some(elV @ _)) =>
            ctx.builder.addConditional(vIdx, cIdx, thr, elV, ctx.aggOp)
            true
          case _ => false
        }
      // GpuCaseWhen single-branch pattern (equivalent to IF)
      case GpuCaseWhen(
          Seq((GpuGreaterThan(condRef: AttributeReference, GpuLiteral(threshold, _)), 
               valRef: AttributeReference)),
          Some(GpuLiteral(elseVal, _)), _) =>
        (ctx.getColIndex(valRef), ctx.getColIndex(condRef),
         ctx.toLong(threshold), ctx.toLong(elseVal)) match {
          case (Some(vIdx @ _), Some(cIdx @ _), Some(thr @ _), Some(elV @ _)) =>
            ctx.builder.addConditional(vIdx, cIdx, thr, elV, ctx.aggOp)
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
      // GpuIf pattern
      case GpuIf(
          GpuGreaterThan(condRef: AttributeReference, GpuLiteral(threshold, _)),
          GpuCoalesce(Seq(valRef: AttributeReference, GpuLiteral(defaultVal, _))),
          GpuLiteral(elseVal, _)) =>
        (ctx.getColIndex(valRef), ctx.getColIndex(condRef),
         ctx.toLong(threshold), ctx.toLong(defaultVal), ctx.toLong(elseVal)) match {
          case (Some(vIdx @ _), Some(cIdx @ _), Some(thr @ _), Some(dVal @ _), Some(elV @ _)) =>
            ctx.builder.addConditionalCoalesce(vIdx, cIdx, thr, dVal, elV, ctx.aggOp)
            true
          case _ => false
        }
      // GpuCaseWhen single-branch with coalesce (equivalent to IF with coalesce)
      case GpuCaseWhen(
          Seq((GpuGreaterThan(condRef: AttributeReference, GpuLiteral(threshold, _)),
               GpuCoalesce(Seq(valRef: AttributeReference, GpuLiteral(defaultVal, _))))),
          Some(GpuLiteral(elseVal, _)), _) =>
        (ctx.getColIndex(valRef), ctx.getColIndex(condRef),
         ctx.toLong(threshold), ctx.toLong(defaultVal), ctx.toLong(elseVal)) match {
          case (Some(vIdx @ _), Some(cIdx @ _), Some(thr @ _), Some(dVal @ _), Some(elV @ _)) =>
            ctx.builder.addConditionalCoalesce(vIdx, cIdx, thr, dVal, elV, ctx.aggOp)
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
          case (Some(colIdx @ _), Some(defVal @ _)) =>
            ctx.builder.addCoalesceMulSelf(colIdx, defVal, ctx.aggOp)
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
          case (Some(col1 @ _), Some(col2 @ _), Some(defVal @ _)) =>
            ctx.builder.addCoalesceMulOther(col1, col2, defVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
}

// =============================================================================
// TPC-H Pattern Matchers (Phase 1a)
// =============================================================================

/**
 * Simple multiply matcher - handles col1 * col2 pattern
 * TPC-H Q6: SUM(l_extendedprice * l_discount)
 */
object MulMatcher extends ExpressionMatcher {
  override def name: String = "Mul (a * b)"
  override def priority: Int = 55  // After CoalesceMul patterns
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      case GpuMultiply(left, right, _) =>
        // Both sides must be simple column references
        (resolveColumnIndex(left, ctx.colIndexMap), 
         resolveColumnIndex(right, ctx.colIndexMap)) match {
          case (Some(leftIdx), Some(rightIdx)) =>
            ctx.builder.addMul(leftIdx, rightIdx, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
  
  private def resolveColumnIndex(
      expr: Expression, 
      colIndexMap: Map[ExprId, Int]): Option[Int] = {
    expr match {
      case ref: AttributeReference => colIndexMap.get(ref.exprId)
      case ref: GpuBoundReference => Some(ref.ordinal)
      case GpuCast(inner, _, _, _, _, _) => resolveColumnIndex(inner, colIndexMap)
      case _ => None
    }
  }
}

/**
 * Multiply with subtraction from constant matcher
 * TPC-H Q1: SUM(l_extendedprice * (1 - l_discount))
 * 
 * Pattern: col1 * (const - col2)
 */
object MulSubConstMatcher extends ExpressionMatcher {
  override def name: String = "MulSubConst (a * (const - b))"
  override def priority: Int = 52  // Before simple Mul
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      // Pattern: col1 * (const - col2)
      case GpuMultiply(left, GpuSubtract(constExpr, right, _), _) =>
        (resolveColumnIndex(left, ctx.colIndexMap),
         resolveColumnIndex(right, ctx.colIndexMap),
         extractConstant(constExpr, ctx)) match {
          case (Some(leftIdx), Some(rightIdx), Some(constVal)) =>
            ctx.builder.addMulSubConst(leftIdx, rightIdx, constVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
  
  private def resolveColumnIndex(
      expr: Expression, 
      colIndexMap: Map[ExprId, Int]): Option[Int] = {
    expr match {
      case ref: AttributeReference => colIndexMap.get(ref.exprId)
      case ref: GpuBoundReference => Some(ref.ordinal)
      case GpuCast(inner, _, _, _, _, _) => resolveColumnIndex(inner, colIndexMap)
      case _ => None
    }
  }
  
  private def extractConstant(expr: Expression, ctx: FusionContext): Option[Long] = {
    expr match {
      case GpuLiteral(value, _) => ctx.toLong(value)
      case Literal(value, _) => ctx.toLong(value)
      case GpuCast(inner, _, _, _, _, _) => extractConstant(inner, ctx)
      case _ => None
    }
  }
}

/**
 * TPC-H Q1 sum_charge pattern matcher
 * TPC-H Q1: SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax))
 * 
 * Pattern: col1 * (const1 - col2) * (const2 + col3)
 * Also handles: col1 * (const1 - col2) * (col3 + const2)  (commutative Add)
 */
object MulSubConstMulAddConstMatcher extends ExpressionMatcher {
  override def name: String = "MulSubConstMulAddConst (a * (c1 - b) * (c2 + c))"
  override def priority: Int = 50  // Before CaseMul and MulSubConst
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      // Pattern: (col1 * (const1 - col2)) * (const2 + col3) or (col3 + const2)
      case GpuMultiply(
          GpuMultiply(left, GpuSubtract(const1Expr, right1, _), _),
          GpuAdd(addLeft, addRight, _), _) =>
        // Determine which operand of Add is the constant and which is the column
        val (const2Opt, right2Opt) = (isConstantExpr(addLeft), isConstantExpr(addRight)) match {
          case (true, false) => (extractConstant(addLeft, ctx), resolveColumnIndex(addRight, ctx.colIndexMap))
          case (false, true) => (extractConstant(addRight, ctx), resolveColumnIndex(addLeft, ctx.colIndexMap))
          case _ => (None, None)  // Both constants or both columns - not a valid pattern
        }
        
        (resolveColumnIndex(left, ctx.colIndexMap),
         resolveColumnIndex(right1, ctx.colIndexMap),
         right2Opt,
         extractConstant(const1Expr, ctx),
         const2Opt) match {
          case (Some(valIdx), Some(otherIdx), Some(thirdIdx), Some(const1), Some(const2)) =>
            ctx.builder.addMulSubConstMulAddConst(valIdx, otherIdx, thirdIdx, const1, const2, ctx.aggOp)
            true
          case _ => false
        }
        
      case _ => false
    }
  }
  
  private def resolveColumnIndex(
      expr: Expression, 
      colIndexMap: Map[ExprId, Int]): Option[Int] = {
    expr match {
      case ref: AttributeReference => colIndexMap.get(ref.exprId)
      case ref: GpuBoundReference => Some(ref.ordinal)
      case GpuCast(inner, _, _, _, _, _) => resolveColumnIndex(inner, colIndexMap)
      case _ => None
    }
  }
  
  private def extractConstant(expr: Expression, ctx: FusionContext): Option[Long] = {
    expr match {
      case GpuLiteral(value, _) => ctx.toLong(value)
      case Literal(value, _) => ctx.toLong(value)
      case GpuCast(inner, _, _, _, _, _) => extractConstant(inner, ctx)
      case _ => None
    }
  }
  
  private def isConstantExpr(expr: Expression): Boolean = expr match {
    case _: GpuLiteral => true
    case _: Literal => true
    case GpuCast(inner, _, _, _, _, _) => isConstantExpr(inner)
    case _ => false
  }
}

/**
 * Case-when multiply matcher
 * TPC-H Q14: SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1-l_discount) ELSE 0)
 * 
 * Pattern: CASE WHEN cond THEN col1 * col2 ELSE 0
 * 
 * Note: For LIKE patterns, we expect a pre-computed boolean column (1 for match, 0 for no match).
 * The condition becomes: condCol > 0
 */
object CaseMulMatcher extends ExpressionMatcher {
  override def name: String = "CaseMul (CASE WHEN c THEN a*b ELSE 0)"
  override def priority: Int = 51  // Before MulSubConst
  
  override def tryMatch(expr: Expression, ctx: FusionContext): Boolean = {
    expr match {
      // Pattern: CaseWhen with single branch and else
      case cw: GpuCaseWhen if cw.branches.size == 1 =>
        val (condExpr, thenExpr) = cw.branches.head
        val elseExpr = cw.elseValue
        
        // Try to match: CASE WHEN cond THEN a*b ELSE 0
        (resolveCondition(condExpr, ctx),
         resolveMulExpr(thenExpr, ctx.colIndexMap),
         extractElseValue(elseExpr, ctx)) match {
          case (Some((condIdx, threshold)), Some((valIdx, otherIdx)), Some(elseVal)) =>
            ctx.builder.addCaseMul(valIdx, otherIdx, condIdx, threshold, elseVal, ctx.aggOp)
            true
          case _ => false
        }
      case _ => false
    }
  }
  
  // Returns (condColIdx, threshold) - condition is: condCol > threshold
  private def resolveCondition(
      expr: Expression,
      ctx: FusionContext): Option[(Int, Long)] = {
    expr match {
      // Pattern: col > literal
      case GpuGreaterThan(left, GpuLiteral(value, _)) =>
        resolveColumnIndex(left, ctx.colIndexMap).flatMap { idx =>
          ctx.toLong(value).map(v => (idx, v))
        }
      // Boolean column (for pre-computed LIKE result): treat as col > 0
      case ref: AttributeReference if ref.dataType == BooleanType =>
        ctx.colIndexMap.get(ref.exprId).map(idx => (idx, 0L))
      case ref: GpuBoundReference if ref.dataType == BooleanType =>
        Some((ref.ordinal, 0L))
      case _ => None
    }
  }
  
  // Returns (valColIdx, otherColIdx) for multiply expression
  private def resolveMulExpr(
      expr: Expression,
      colIndexMap: Map[ExprId, Int]): Option[(Int, Int)] = {
    expr match {
      case GpuMultiply(left, right, _) =>
        (resolveColumnIndex(left, colIndexMap),
         resolveColumnIndex(right, colIndexMap)) match {
          case (Some(l), Some(r)) => Some((l, r))
          case _ => None
        }
      case _ => None
    }
  }
  
  private def resolveColumnIndex(
      expr: Expression,
      colIndexMap: Map[ExprId, Int]): Option[Int] = {
    expr match {
      case ref: AttributeReference => colIndexMap.get(ref.exprId)
      case ref: GpuBoundReference => Some(ref.ordinal)
      case GpuCast(inner, _, _, _, _, _) => resolveColumnIndex(inner, colIndexMap)
      case _ => None
    }
  }
  
  private def extractElseValue(elseOpt: Option[Expression], ctx: FusionContext): Option[Long] = {
    elseOpt match {
      case Some(GpuLiteral(value, _)) => ctx.toLong(value)
      case Some(Literal(value, _)) => ctx.toLong(value)
      case None => Some(0L)  // No else means NULL, treat as 0 for SUM
      case _ => None
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
    // TPC-H patterns (Phase 1a)
    ExpressionMatcherRegistry.register(MulSubConstMulAddConstMatcher)  // Most complex first
    ExpressionMatcherRegistry.register(CaseMulMatcher)
    ExpressionMatcherRegistry.register(MulSubConstMatcher)
    ExpressionMatcherRegistry.register(MulMatcher)
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
  // Note: AGG_AVG is not used because partial AVG is expanded to (SUM, COUNT)
  private val AGG_SUM = FusedTransformAggregate.AGG_SUM
  private val AGG_COUNT = FusedTransformAggregate.AGG_COUNT
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
    
    
    // Check configuration
    if (!conf.enableFusedTransformAggregate) {
      return false
    }
    
    // Check minimum columns threshold
    val minCols = conf.fusedTransformAggregateMinColumns
    if (aggExprs.size < minCols) {
      return false
    }
    
    // Only for Partial mode - Complete mode receives already-aggregated data
    if (!modeInfo.hasPartialMode) {
      logDebug(s"[FUSION] not Partial mode: hasPartial=${modeInfo.hasPartialMode}")
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
  def checkSeparateProjectFusion(
      conf: RapidsConf,
      childPlan: org.apache.spark.sql.execution.SparkPlan,
      aggExprs: Seq[GpuAggregateExpression],
      modeInfo: AggregateModeInfo): (Boolean, Seq[NamedExpression], Seq[Attribute]) = {
    
    logWarning(s"[FUSION-CHECK] checkSeparateProjectFusion called, child=${childPlan.getClass.getSimpleName}, " +
      s"aggExprs=${aggExprs.size}, modeInfo.hasPartialMode=${modeInfo.hasPartialMode}")
    
    if (!conf.enableFusedTransformAggregate) {
      logWarning("[FUSION-CHECK] Fusion disabled by config")
      return (false, Seq.empty, Seq.empty)
    }
    
    if (aggExprs.size < conf.fusedTransformAggregateMinColumns) {
      logWarning(s"[FUSION-CHECK] Not enough agg expressions: ${aggExprs.size} < ${conf.fusedTransformAggregateMinColumns}")
      return (false, Seq.empty, Seq.empty)
    }
    
    // Only for Partial mode - Complete mode receives already-aggregated data
    if (!modeInfo.hasPartialMode) {
      logWarning("[FUSION-CHECK] Not partial mode")
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
          // CRITICAL: Must also check that ALL aggregate inputs can be fused
          // Build map of project expression IDs to their fusability
          val projectFusabilityMap = proj.projectList.map { pe =>
            val (childExpr, exprId) = pe match {
              case Alias(child, _) => (child, pe.exprId)
              case GpuAlias(child, _) => (child, pe.exprId)
              case ref: AttributeReference => (ref, ref.exprId)
              case other => (other, pe.exprId)
            }
            exprId -> isFusableExpression(childExpr)
          }.toMap
          
          // Check that all aggregate inputs reference fusable project expressions OR are directly fusable inline expressions
          // 1. AttributeReference → check the project expression it references
          // 2. Inline expression → check if it matches a fusable pattern directly
          val unfusableAggInputs = aggExprs.flatMap { aggExpr =>
            aggExpr.aggregateFunction.children.headOption.flatMap {
              case ref: AttributeReference =>
                projectFusabilityMap.get(ref.exprId) match {
                  case Some(false) => Some(ref.name)
                  case None => None // Not in project, might be direct column reference
                  case Some(true) => None
                }
              case _: GpuLiteral | _: Literal =>
                // COUNT(1) has a literal child - this is fusable as IDENTITY transform
                logWarning(s"[FUSION-CHECK] Literal child - fusable (COUNT(1) pattern)")
                None
              case other if isFusableExpression(other) => 
                // Inline expression is directly fusable (e.g., SUM(a*b) with fusable a*b)
                logWarning(s"[FUSION-CHECK] Inline expression is fusable: ${other.getClass.getSimpleName}")
                None
              case other => 
                // Non-fusable inline expression
                Some(s"inline:${other.getClass.getSimpleName}")
            }
          }.distinct
          
          if (unfusableAggInputs.nonEmpty) {
            // Log detailed expression types for debugging
            val unfusableDetails = proj.projectList.filter { pe =>
              val exprName = pe match {
                case Alias(_, n) => n
                case GpuAlias(_, n) => n
                case _ => ""
              }
              unfusableAggInputs.contains(exprName)
            }.take(3).map { pe =>
              val (childExpr, exprName) = pe match {
                case Alias(child, name) => (child, name)
                case GpuAlias(child, name) => (child, name)
                case _ => (pe, "?")
              }
              s"$exprName:${childExpr.getClass.getSimpleName}"
            }
            logWarning(s"[FUSION-CHECK] REJECTED: aggregate inputs reference unfusable project expressions: " +
              s"${unfusableAggInputs.take(5).mkString(", ")}, types: ${unfusableDetails.mkString(", ")}")
            (false, Seq.empty, Seq.empty)
          } else {
            logWarning(s"[FUSION-CHECK] ACCEPTED: $fusableCount/${proj.projectList.size} project fusable, " +
              s"all ${aggExprs.size} aggregate inputs fusable")
            (true, proj.projectList, proj.child.output)
          }
        } else {
          logWarning(s"[FUSION-CHECK] REJECTED: insufficient fusable $fusableCount/${proj.projectList.size}")
          (false, Seq.empty, Seq.empty)
        }
        
      case other =>
        logWarning(s"[FUSION-CHECK] Child is not GpuProjectExec: ${other.getClass.getSimpleName}")
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
   * Check if a GpuLiteral value can be converted to Long.
   * This is essential for planning phase to correctly identify unsupported patterns.
   * 
   * Only values that can be exactly represented as Long are supported for fusion.
   * Fractional Decimal values cannot be converted and will cause fusion to be rejected.
   */
  private def isLiteralConvertibleToLong(lit: GpuLiteral): Boolean = {
    lit.value match {
      case null => true  // null is handled by coalesce
      case _: Long | _: Int | _: Short | _: Byte => true
      case d: Double => d.isWhole && d >= Long.MinValue && d <= Long.MaxValue
      case f: Float => f.isWhole && f >= Long.MinValue && f <= Long.MaxValue
      case dec: Decimal =>
        try {
          val bd = dec.toJavaBigDecimal
          bd.scale() <= 0 || bd.stripTrailingZeros().scale() <= 0
        } catch {
          case _: Exception => false
        }
      case bd: java.math.BigDecimal =>
        try {
          bd.scale() <= 0 || bd.stripTrailingZeros().scale() <= 0
        } catch {
          case _: Exception => false
        }
      case _ => false
    }
  }

  /**
   * Check if an expression matches the fusable patterns.
   * 
   * This method validates both the expression structure AND the literal values.
   * Literals must be convertible to Long for the JNI interface.
   */
  /**
   * Check if a data type is supported by the fused JNI kernel.
   * 
   * Supported types:
   * - Integer types: INT64, INT32, INT16, INT8
   * - Floating point: FLOAT64, FLOAT32
   * - Boolean: BOOL8
   * - Decimal: DECIMAL64 (precision <= 18) - stored as scaled int64
   * 
   * DECIMAL128 (precision > 18) is NOT supported because:
   * - JNI kernel uses atomicAdd which doesn't support 128-bit integers
   * - Would require custom atomic implementation or different aggregation strategy
   */
  private[rapids] def isSupportedDataType(dt: DataType): Boolean = dt match {
    case LongType | IntegerType | ShortType | ByteType => true
    case DoubleType | FloatType => true
    case BooleanType => true
    // DECIMAL64 (precision <= 18) can be handled as int64 with scaled values
    // DECIMAL128 (precision > 18) is NOT supported - atomicAdd doesn't support 128-bit
    case dt: DecimalType => dt.precision <= Decimal.MAX_LONG_DIGITS // 18
    case _ => false
  }
  
  private def isFusableExpression(expr: Expression): Boolean = {
    expr match {
      // Simple column reference - must also check data type
      case ref: AttributeReference => isSupportedDataType(ref.dataType)
      case ref: GpuBoundReference => isSupportedDataType(ref.dataType)
      
      // CAST: CAST(inner AS type) - unwrap and check inner
      case GpuCast(inner, _, _, _, _, _) => isFusableExpression(inner)
      
      // Coalesce: COALESCE(col, literal) - also check literal is convertible
      case GpuCoalesce(Seq(_: AttributeReference, lit: GpuLiteral)) => 
        isLiteralConvertibleToLong(lit)
      case GpuCoalesce(Seq(_: GpuBoundReference, lit: GpuLiteral)) => 
        isLiteralConvertibleToLong(lit)
      // Coalesce with CAST: COALESCE(CAST(col), literal)
      case GpuCoalesce(Seq(GpuCast(_, _, _, _, _, _), lit: GpuLiteral)) => 
        isLiteralConvertibleToLong(lit)
      
      // CoalesceMulSelf: COALESCE(col, d) * COALESCE(col, d)
      case GpuMultiply(
          GpuCoalesce(Seq(ref1: AttributeReference, lit1: GpuLiteral)),
          GpuCoalesce(Seq(ref2: AttributeReference, lit2: GpuLiteral)), _) 
          if ref1.exprId == ref2.exprId && lit1 == lit2 => 
        isLiteralConvertibleToLong(lit1)
      
      // CoalesceMulOther: COALESCE(col1, d) * COALESCE(col2, d)
      case GpuMultiply(
          GpuCoalesce(Seq(_: AttributeReference, lit1: GpuLiteral)),
          GpuCoalesce(Seq(_: AttributeReference, lit2: GpuLiteral)), _) => 
        isLiteralConvertibleToLong(lit1) && isLiteralConvertibleToLong(lit2)
      
      // CAST(COALESCE) * CAST(COALESCE): for patterns like CAST(COALESCE(x,0) AS DOUBLE) * CAST(...)
      case GpuMultiply(
          GpuCast(GpuCoalesce(Seq(_: AttributeReference, lit1: GpuLiteral)), _, _, _, _, _),
          GpuCast(GpuCoalesce(Seq(_: AttributeReference, lit2: GpuLiteral)), _, _, _, _, _), _) => 
        isLiteralConvertibleToLong(lit1) && isLiteralConvertibleToLong(lit2)
      
      // Conditional: IF(cond > threshold, col, elseVal)
      case GpuIf(
          GpuGreaterThan(_: AttributeReference, threshLit: GpuLiteral),
          _: AttributeReference,
          elseLit: GpuLiteral) => 
        isLiteralConvertibleToLong(threshLit) && isLiteralConvertibleToLong(elseLit)
      
      // Conditional with coalesce
      case GpuIf(
          GpuGreaterThan(_: AttributeReference, threshLit: GpuLiteral),
          GpuCoalesce(Seq(_: AttributeReference, defLit: GpuLiteral)),
          elseLit: GpuLiteral) => 
        isLiteralConvertibleToLong(threshLit) && 
        isLiteralConvertibleToLong(defLit) && 
        isLiteralConvertibleToLong(elseLit)
      
      // Nested coalesce * coalesce on conditional result
      case GpuMultiply(
          GpuCoalesce(Seq(GpuIf(_, _, _), lit1: GpuLiteral)),
          GpuCoalesce(Seq(GpuIf(_, _, _), lit2: GpuLiteral)), _) => 
        isLiteralConvertibleToLong(lit1) && isLiteralConvertibleToLong(lit2)
      
      // GpuCaseWhen with single branch - equivalent to IF(cond, then, else)
      // Pattern: CASE WHEN cond > threshold THEN col ELSE literal END
      case GpuCaseWhen(
          Seq((GpuGreaterThan(_: AttributeReference, threshLit: GpuLiteral), _: AttributeReference)),
          Some(elseLit: GpuLiteral), _) => 
        isLiteralConvertibleToLong(threshLit) && isLiteralConvertibleToLong(elseLit)
      
      // GpuCaseWhen with coalesce in then branch
      // Pattern: CASE WHEN cond > threshold THEN COALESCE(col, d) ELSE literal END
      case GpuCaseWhen(
          Seq((GpuGreaterThan(_: AttributeReference, threshLit: GpuLiteral), 
               GpuCoalesce(Seq(_: AttributeReference, defLit: GpuLiteral)))),
          Some(elseLit: GpuLiteral), _) => 
        isLiteralConvertibleToLong(threshLit) && 
        isLiteralConvertibleToLong(defLit) && 
        isLiteralConvertibleToLong(elseLit)
      
      // Nested coalesce * coalesce on CaseWhen result
      case GpuMultiply(
          GpuCoalesce(Seq(GpuCaseWhen(_, _, _), lit1: GpuLiteral)),
          GpuCoalesce(Seq(GpuCaseWhen(_, _, _), lit2: GpuLiteral)), _) => 
        isLiteralConvertibleToLong(lit1) && isLiteralConvertibleToLong(lit2)
      
      // === TPC-H patterns (Phase 1a) ===
      
      // Simple multiply: col * col (TPC-H Q6: l_extendedprice * l_discount)
      case GpuMultiply(left, right, _) 
          if isSimpleColumnRef(left) && isSimpleColumnRef(right) => true
      
      // Multiply with subtraction: col * (const - col) (TPC-H Q1: l_extendedprice * (1 - l_discount))
      case GpuMultiply(left, GpuSubtract(constExpr, right, _), _)
          if isSimpleColumnRef(left) && isSimpleColumnRef(right) && isConstantExpr(constExpr) => true
      
      // TPC-H Q1 sum_charge: col * (const - col) * (const + col)
      // Pattern 1: (l_extendedprice * (1 - l_discount)) * (1 + l_tax)
      case GpuMultiply(
          GpuMultiply(left, GpuSubtract(c1, right1, _), _),
          GpuAdd(c2, right2, _), _)
          if isSimpleColumnRef(left) && isSimpleColumnRef(right1) && 
             isSimpleColumnRef(right2) && isConstantExpr(c1) && isConstantExpr(c2) => true
      
      // Pattern 2: (l_extendedprice * (1 - l_discount)) * (l_tax + 1) - swapped Add
      case GpuMultiply(
          GpuMultiply(left, GpuSubtract(c1, right1, _), _),
          GpuAdd(right2, c2, _), _)
          if isSimpleColumnRef(left) && isSimpleColumnRef(right1) && 
             isSimpleColumnRef(right2) && isConstantExpr(c1) && isConstantExpr(c2) => true
          
      case _ => false
    }
  }
  
  /** Check if expression is a simple column reference (AttributeReference or GpuBoundReference) */
  private def isSimpleColumnRef(expr: Expression): Boolean = expr match {
    case _: AttributeReference => true
    case _: GpuBoundReference => true
    case GpuCast(inner, _, _, _, _, _) => isSimpleColumnRef(inner)
    case _ => false
  }
  
  /** Check if expression is a constant (literal) */
  private def isConstantExpr(expr: Expression): Boolean = expr match {
    case _: GpuLiteral => true
    case _: Literal => true
    case GpuCast(inner, _, _, _, _, _) => isConstantExpr(inner)
    case _ => false
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
          logWarning(s"[FUSION-EXEC] Cannot fuse: allFusable=$allFusable, " +
            s"fusedCount=$fusedCount/${aggExprs.size}")
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
    
    val computeAggTime = metrics.computeAggTime
    val opTime = metrics.opTime
    val numAggOps = metrics.numAggOps
    
    try {
      withResource(new MetricRange(computeAggTime, opTime)) { _ =>
        // Build column index map from input attributes
        val colIndexMap = inputAttrs.zipWithIndex.map { case (attr, idx) =>
          attr.exprId -> idx
        }.toMap
        
        // THREE CASES LOGIC:
        // Case 1: Project部分支持, Agg全部支持 → fuse (Project保留不支持的)
        // Case 2: Project部分支持, Agg部分支持 → 不fuse
        // Case 3: Project全部支持, Agg部分支持 → 不fuse
        // 
        // Key: Only fuse when ALL aggregate expressions can be fused
        
        val builder = new ExpressionBuilder()
        var allAggsFusable = true
        var failedAggIdx = -1
        
        // First pass: check if ALL aggregates can be fused
        aggExprs.zipWithIndex.foreach { case (aggExpr, idx) =>
          if (allAggsFusable && !addExpression(builder, aggExpr, projectExprs, colIndexMap)) {
            allAggsFusable = false
            failedAggIdx = idx
          }
        }
        
        // If not all aggregates are fusable, fall back to standard path (Case 2 & 3)
        if (!allAggsFusable) {
          logWarning(s"[FUSION] Agg not fully supported (failed at idx $failedAggIdx), " +
            s"falling back to standard path")
          return None
        }
        
        // All aggregates are fusable - proceed with fusion (Case 1 or full fusion)
        // Get group-by column indices
        val groupByIndices = extractGroupByIndices(groupingExprs, projectExprs, colIndexMap)
        
        // Note: groupByIndices can be empty for scalar aggregation (no GROUP BY)
        // The JNI layer now supports this case
        
        logWarning(s"[FUSION] All ${aggExprs.size} aggregates fusable, executing fused kernel")
        
        // Execute fused kernel
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

  private def addExpression(
      builder: ExpressionBuilder,
      aggExpr: GpuAggregateExpression,
      projectExprs: Seq[NamedExpression],
      colIndexMap: Map[ExprId, Int]): Boolean = {
    
    // For PARTIAL mode AVG, we need to output (SUM, COUNT) = 2 columns
    // This is because Spark's partial AVG produces 2 intermediate columns
    // that will be merged in the final aggregation phase.
    aggExpr.aggregateFunction match {
      case _: GpuAverage =>
        // Add two expressions: SUM and COUNT for the same input
        val aggInputOpt = aggExpr.aggregateFunction.children.headOption
        if (aggInputOpt.isEmpty) return false
        val aggInput = aggInputOpt.get
        
        aggInput match {
          case ref: AttributeReference =>
            // First add SUM expression
            val sumOk = analyzeAndAddExpression(
              builder, ref, projectExprs, colIndexMap, AGG_SUM)
            if (!sumOk) return false
            // Then add COUNT expression  
            val countOk = analyzeAndAddExpression(
              builder, ref, projectExprs, colIndexMap, AGG_COUNT)
            return countOk
          case inlineExpr =>
            // Inline expression in AVG (e.g., AVG(a*b))
            // First add SUM expression
            val sumOk = analyzeProjectChild(builder, inlineExpr, colIndexMap, AGG_SUM)
            if (!sumOk) return false
            // Then add COUNT expression
            val countOk = analyzeProjectChild(builder, inlineExpr, colIndexMap, AGG_COUNT)
            return countOk
        }
      case _ => // Fall through to normal handling
    }
    
    // Get the aggregation operation type
    // Use abstract classes GpuSum/GpuMin/GpuMax to match all subclasses
    // (e.g., GpuBasicSum, GpuDecimal128Sum, GpuBasicMin, GpuFloatMin, etc.)
    val aggOp = aggExpr.aggregateFunction match {
      case _: GpuSum => AGG_SUM
      case _: GpuCount => AGG_COUNT
      case _: GpuMin => AGG_MIN
      case _: GpuMax => AGG_MAX
      case _ => 
        return false
    }
    
    // Get the input to the aggregate
    val aggInputOpt = aggExpr.aggregateFunction.children.headOption
    if (aggInputOpt.isEmpty) {
      // COUNT(*) has no children - try addCountAll directly
      if (aggExpr.aggregateFunction.isInstanceOf[GpuCount]) {
        builder.addCountAll(AGG_COUNT)
        return true
      }
      return false
    }
    val aggInput = aggInputOpt.get
    
    aggInput match {
      case ref: AttributeReference =>
        analyzeAndAddExpression(builder, ref, projectExprs, colIndexMap, aggOp)
      case _: GpuLiteral =>
        // For COUNT(literal), treat as COUNT(*)
        if (aggExpr.aggregateFunction.isInstanceOf[GpuCount]) {
          builder.addCountAll(AGG_COUNT)
          return true
        }
        false
      case _: Literal =>
        // For COUNT(literal) with Catalyst Literal, treat as COUNT(*)
        if (aggExpr.aggregateFunction.isInstanceOf[GpuCount]) {
          builder.addCountAll(AGG_COUNT)
          return true
        }
        false
      case inlineExpr =>
        // Inline expression in aggregate (e.g., SUM(a*b) instead of SUM(alias))
        // Try to analyze it directly using expression matchers
        logWarning(s"[FUSION] Trying to analyze inline expression: ${inlineExpr.getClass.getSimpleName}")
        analyzeProjectChild(builder, inlineExpr, colIndexMap, aggOp)
    }
  }

  private def analyzeAndAddExpression(
      builder: ExpressionBuilder,
      ref: AttributeReference,
      projectExprs: Seq[NamedExpression],
      colIndexMap: Map[ExprId, Int],
      aggOp: Int): Boolean = {
    
    
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
      case Some(Alias(child, _)) =>
        analyzeProjectChild(builder, child, colIndexMap, aggOp)
      case Some(GpuAlias(child, _)) =>
        analyzeProjectChild(builder, child, colIndexMap, aggOp)
      case Some(a: AttributeReference) =>
        colIndexMap.get(a.exprId) match {
          case Some(colIdx @ _) =>
            builder.addIdentity(colIdx, aggOp)
            true
          case None => 
            false
        }
      case None =>
        false
      case Some(_) =>
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
          logWarning(s"[FUSION] Multiply match attempt: " +
            s"ref1.exprId=${ref1.exprId.id}, ref2.exprId=${ref2.exprId.id}, " +
            s"ref1 in map=${colIndexMap.contains(ref1.exprId)}, " +
            s"ref2 in map=${colIndexMap.contains(ref2.exprId)}")
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
    
    // For PARTIAL mode AVG, we need to output (SUM, COUNT) = 2 columns
    aggExpr.aggregateFunction match {
      case _: GpuAverage =>
        val aggInput = aggExpr.aggregateFunction.children.headOption
        aggInput match {
          case Some(expr) =>
            // First add SUM expression
            val sumOk = analyzeProjectChild(builder, expr, colIndexMap, AGG_SUM)
            if (!sumOk) return false
            // Then add COUNT expression
            val countOk = analyzeProjectChild(builder, expr, colIndexMap, AGG_COUNT)
            return countOk
          case None =>
            return false
        }
      case _ => // Fall through to normal handling
    }
    
    // Get the aggregation operation type
    // Use abstract classes GpuSum/GpuMin/GpuMax to match all subclasses
    val aggOp = aggExpr.aggregateFunction match {
      case _: GpuSum => AGG_SUM
      case _: GpuCount => AGG_COUNT
      case _: GpuMin => AGG_MIN
      case _: GpuMax => AGG_MAX
      case _ => 
        val funcName = aggExpr.aggregateFunction.getClass.getSimpleName
        logDebug(s"Unsupported aggregate function: $funcName")
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
    withResource(new NvtxRange("FusedTransformAggregate", NvtxColor.CYAN)) { _ =>
    withResource(GpuColumnVector.from(inputBatch)) { inputTable =>
      // Execute fused kernel - returns FusedResult with regular Tables
      withResource(FusedTransformAggregate.execute(
          inputTable, groupByIndices, builder, enableWarpReduction)) { fusedResult =>
        
        val keysTable = fusedResult.getKeys
        val valuesTable = fusedResult.getValues
        
        if (keysTable == null || valuesTable == null || 
            (keysTable.getRowCount == 0 && valuesTable.getRowCount == 0)) {
          SpillableColumnarBatch(
            new ColumnarBatch(Array.empty, 0),
            SpillPriorities.ACTIVE_BATCHING_PRIORITY)
        } else {
          val keysRows = keysTable.getRowCount
          val valuesRows = valuesTable.getRowCount
          val numKeyCols = keysTable.getNumberOfColumns
          val numValCols = valuesTable.getNumberOfColumns
          
          // DEBUG: Check row count consistency
          logWarning(s"[FUSION-DEBUG] keysRows=$keysRows, valuesRows=$valuesRows, " +
            s"keyCols=$numKeyCols, valCols=$numValCols")
          
          // For grouped aggregation, keys and values should have the same row count
          // For scalar aggregation, keys has 0 rows and values has 1 row
          val numRows = if (numKeyCols == 0) {
            // Scalar aggregation: use values row count
            valuesRows.toInt
          } else if (keysRows != valuesRows) {
            // ERROR: Row count mismatch - this should not happen
            throw new IllegalStateException(
              s"[FUSION BUG] Row count mismatch: keysRows=$keysRows, valuesRows=$valuesRows. " +
              "This indicates a bug in fused kernel output.")
          } else {
            keysRows.toInt
          }
          
          // FIX: Use copyToColumnVector() to create TRUE DEEP COPIES
          // This ensures columns have completely independent memory that won't be
          // affected when the JNI result (keysTable/valuesTable) is closed.
          // Using incRefCount() alone is not sufficient because columns may share
          // internal cuDF state with the source table.
          val gpuVectors = new Array[GpuColumnVector](numKeyCols + numValCols)
          
          // Helper to convert cuDF DType to Spark DataType
          def cudfTypeToSpark(cudfType: ai.rapids.cudf.DType): DataType = cudfType match {
            case ai.rapids.cudf.DType.INT64 => LongType
            case ai.rapids.cudf.DType.FLOAT64 => DoubleType
            case ai.rapids.cudf.DType.INT32 => IntegerType
            case ai.rapids.cudf.DType.FLOAT32 => FloatType
            case ai.rapids.cudf.DType.INT16 => ShortType
            case ai.rapids.cudf.DType.INT8 => ByteType
            case ai.rapids.cudf.DType.BOOL8 => BooleanType
            case ai.rapids.cudf.DType.STRING => StringType
            case dt if dt.isTimestampType => TimestampType
            case dt if dt.hasTimeResolution => 
              // Handle DATE types
              DateType
            case dt if dt.isDecimalType =>
              // DECIMAL64 or DECIMAL128 - get precision and scale
              // Scale in cuDF can be negative for large numbers
              val scale = -dt.getScale  // cuDF uses negative scale
              val precision = if (dt.getTypeId == ai.rapids.cudf.DType.DTypeEnum.DECIMAL128) {
                38  // Max precision for DECIMAL128
              } else {
                18  // Max precision for DECIMAL64
              }
              DecimalType(precision, math.max(0, scale))
            case other =>
              throw new UnsupportedOperationException(s"Unsupported cuDF type: $other")
          }
          
          for (i <- 0 until numKeyCols) {
            val colView: ai.rapids.cudf.ColumnView = keysTable.getColumn(i)
            val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
            val dt = cudfTypeToSpark(copiedCol.getType)
            gpuVectors(i) = GpuColumnVector.from(copiedCol, dt)
          }
          
          for (i <- 0 until numValCols) {
            val colView: ai.rapids.cudf.ColumnView = valuesTable.getColumn(i)
            val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
            val dt = cudfTypeToSpark(copiedCol.getType)
            gpuVectors(numKeyCols + i) = GpuColumnVector.from(copiedCol, dt)
          }
          
          // Create ColumnarBatch - columns now have completely independent memory
          val batch = new ColumnarBatch(gpuVectors.toArray, numRows)
          
          // SpillableColumnarBatch will pack on-demand when spilling
          SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
        }
      }
    }
    } // end NvtxRange
  }
}

/**
 * Merge operation type for partial aggregation results.
 * Each partial column needs to know how to merge with other partials.
 */
sealed trait MergeAggType {
  def toCudfAggregation: ai.rapids.cudf.GroupByAggregation
}
case object MergeSum extends MergeAggType {
  override def toCudfAggregation: ai.rapids.cudf.GroupByAggregation = 
    ai.rapids.cudf.GroupByAggregation.sum()
}
case object MergeMin extends MergeAggType {
  override def toCudfAggregation: ai.rapids.cudf.GroupByAggregation = 
    ai.rapids.cudf.GroupByAggregation.min()
}
case object MergeMax extends MergeAggType {
  override def toCudfAggregation: ai.rapids.cudf.GroupByAggregation = 
    ai.rapids.cudf.GroupByAggregation.max()
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
  
  // Track spillable batches for proper cleanup if iterator is not fully consumed
  private var spillableBatches: Array[SpillableColumnarBatch] = Array.empty
  
  // Metric to track fused batches - exposed for testing
  private val fusedBatchesMetric = allMetrics.get("NUM_FUSED_BATCHES")
  
  /**
   * Build merge operation types for each output aggregate column.
   * This tells the merge phase how to correctly combine partial results.
   * 
   * CRITICAL: Different aggregates need different merge operations:
   * - SUM partial results → merge with SUM
   * - COUNT partial results → merge with SUM (COUNT is SUM of counts)
   * - MIN partial results → merge with MIN
   * - MAX partial results → merge with MAX
   * - AVG is expanded to (SUM, COUNT), both merge with SUM
   */
  private val mergeAggTypes: Seq[MergeAggType] = {
    aggregateExprs.flatMap { aggExpr =>
      aggExpr.aggregateFunction match {
        case _: GpuAverage =>
          // AVG is expanded to 2 columns: SUM and COUNT
          // Both merge with SUM (COUNT partials are summed)
          Seq(MergeSum, MergeSum)
        case _: GpuBasicSum | _: GpuCount =>
          Seq(MergeSum)
        case _: GpuMin =>
          Seq(MergeMin)
        case _: GpuMax =>
          Seq(MergeMax)
        case _ =>
          // Fallback - should not happen if planning is correct
          Seq(MergeSum)
      }
    }
  }

  override def hasNext: Boolean = {
    if (!initialized) {
      initializeResults()
      initialized = true
    }
    resultIter.hasNext
  }

  override def next(): ColumnarBatch = {
    if (!initialized) {
      initializeResults()
      initialized = true
    }
    
    val batch = resultIter.next()
    metrics.numOutputRows += batch.numRows()
    metrics.numOutputBatches += 1
    
    batch
  }

  private def initializeResults(): Unit = {
    // FIX: Stream processing - process each batch immediately, don't buffer all inputs
    // This reduces memory from O(all_input_data) to O(partial_results)
    import scala.collection.mutable.ArrayBuffer
    val partialResults = ArrayBuffer[SpillableColumnarBatch]()
    
    try {
      // PHASE 1: Stream input batches - process each immediately, don't buffer
      // This is critical for large datasets where input >> partial results
      var batchIdx = 0
      var hasInput = false
      
      while (inputIter.hasNext) {
        hasInput = true
        val inputBatch = inputIter.next()
        
        try {
          val result = GpuFusedProjectAggregate.executeFused(
            inputBatch,
            projectExprs,
            groupingExprs,
            aggregateExprs,
            inputAttrs,
            metrics,
            enableWarpReduction
          )
          
          batchIdx += 1
          
          result match {
            case Some(spillable) =>
              fusedBatchesMetric.foreach(_ += 1)
              GpuFusedProjectAggregate.incrementFusionCounter()
              partialResults += spillable
            case None =>
              // CRITICAL: Fusion failed at runtime - this should not happen if planning was correct
              // Throw exception to fail the task instead of silently losing data
              throw new IllegalStateException(
                s"[FUSION BUG] executeFused returned None for batch $batchIdx. " +
                "This indicates a mismatch between planning and execution. " +
                "Planning accepted fusion but execution failed. " +
                "Please report this bug and disable fusion with: " +
                "spark.rapids.sql.fusedTransformAggregate.enabled=false")
          }
        } finally {
          // Close input batch immediately after processing - critical for memory
          inputBatch.close()
        }
      }
      
      if (!hasInput || partialResults.isEmpty) {
        resultIter = Iterator.empty
        return
      }
      
      // PHASE 2: Merge Pass - Incrementally merge partial results to avoid OOM
      // Process in chunks to limit memory usage
      if (partialResults.length == 1) {
        // Only one partial result - no merge needed
        spillableBatches = partialResults.toArray
      } else {
        
        // Incremental merge: process in chunks to avoid OOM
        // Increased chunk size to reduce merge rounds (was 10, now 32)
        val CHUNK_SIZE = 32
        val numGroupCols = groupingExprs.length
        val groupByIndices = (0 until numGroupCols).toArray
        
        def mergeChunk(chunk: Array[SpillableColumnarBatch]): SpillableColumnarBatch = {
          // Handle single-element chunk - just return as-is without deep copy
          // Deep copy will be done in final iteration when creating result batch
          if (chunk.length == 1) {
            return chunk(0)
          }
          
          withResource(new NvtxRange("MergeChunk", NvtxColor.YELLOW)) { _ =>
            // Get ColumnarBatches from SpillableColumnarBatch
            val partialBatches = chunk.map(_.getColumnarBatch())
            
            // Create Table views from the batches (this increments ref counts on columns)
            val tables = partialBatches.map(GpuColumnVector.from(_))
            
            // Table.concatenate COPIES the data, so it's safe to close sources after
            val concatenated = ai.rapids.cudf.Table.concatenate(tables: _*)
            
            // FIX: Close the Table objects to decrement ref counts
            tables.foreach(_.close())
            
            // FIX: Only close spillables - they own the batches and will close them
            // DO NOT close partialBatches separately - that causes double-close and leaks!
            chunk.foreach(_.close())
            
            
            withResource(concatenated) { concatTable =>
              // FIX: Use correct merge aggregation based on original aggregate type
              // Different aggregates need different merge operations:
              // - SUM/COUNT partials → merge with SUM
              // - MIN partials → merge with MIN  
              // - MAX partials → merge with MAX
              val aggSpecs = (numGroupCols until concatTable.getNumberOfColumns).zipWithIndex.map { 
                case (colIdx, aggIdx) =>
                  val mergeType = if (aggIdx < mergeAggTypes.length) {
                    mergeAggTypes(aggIdx)
                  } else {
                    logWarning(s"[FUSION-MERGE] No merge type for col $aggIdx, default SUM")
                    MergeSum
                  }
                  mergeType.toCudfAggregation.onColumn(colIdx)
              }
              val groupOptions = ai.rapids.cudf.GroupByOptions.builder()
                .withIgnoreNullKeys(false).build()
              
              withResource(concatTable.groupBy(groupOptions, groupByIndices: _*)
                  .aggregate(aggSpecs: _*)) { mergedTable =>
                  
                
                // FIX: Use copyToColumnVector() to create TRUE DEEP COPIES of all columns.
                // This is critical because incRefCount() only increments the reference count
                // but the column may still share some internal cuDF state with the source table.
                // When the source table is closed, this shared state could be invalidated,
                // causing crashes during later Kudo serialization.
                val numRows = mergedTable.getRowCount.toInt
                val numCols = mergedTable.getNumberOfColumns
                
                
                val gpuVectors = new Array[GpuColumnVector](numCols)
                for (i <- 0 until numCols) {
                  // mergedTable.getColumn(i) returns a ColumnVector (which extends ColumnView)
                  // ColumnView.copyToColumnVector() creates a true deep copy with new memory
                  val colView: ai.rapids.cudf.ColumnView = mergedTable.getColumn(i)
                  val copiedCol: ai.rapids.cudf.ColumnVector = colView.copyToColumnVector()
                  val cudfType = copiedCol.getType
                  val dt = cudfType match {
                    case ai.rapids.cudf.DType.INT64 => LongType
                    case ai.rapids.cudf.DType.FLOAT64 => DoubleType
                    case ai.rapids.cudf.DType.INT32 => IntegerType
                    case ai.rapids.cudf.DType.FLOAT32 => FloatType
                    case ai.rapids.cudf.DType.INT16 => ShortType
                    case ai.rapids.cudf.DType.INT8 => ByteType
                    case ai.rapids.cudf.DType.BOOL8 => BooleanType
                    case t if t.isDecimalType =>
                      // DECIMAL64 or DECIMAL128 - preserve precision and scale
                      val scale = -t.getScale  // cuDF uses negative scale
                      val isDecimal128 = 
                        t.getTypeId == ai.rapids.cudf.DType.DTypeEnum.DECIMAL128
                      val precision = if (isDecimal128) 38 else 18
                      DecimalType(precision, math.max(0, scale))
                    case other => 
                      throw new UnsupportedOperationException(s"Unsupported: $other")
                  }
                  gpuVectors(i) = GpuColumnVector.from(copiedCol, dt)
                }
                
                val batch = new ColumnarBatch(gpuVectors.toArray, numRows)
                SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
              }
            }
          }
        }
        
        // Process in rounds until we have a single result
        var currentPartials: Array[SpillableColumnarBatch] = partialResults.toArray
        var round = 0
        while (currentPartials.length > 1) {
          val chunks = currentPartials.grouped(CHUNK_SIZE).toArray
          currentPartials = chunks.map(chunk => mergeChunk(chunk))
          round += 1
        }
        
        spillableBatches = currentPartials
      }
      
      // Create iterator that returns the merged batch
      // FIX: Must create NEW GpuColumnVectors because spillable.close()
      // will close the original ones. incRefCount keeps cudf columns alive,
      // but GpuColumnVector gets closed by spillable.
      resultIter = spillableBatches.iterator.map { spillable =>
        val srcBatch = spillable.getColumnarBatch()
        val numRows = srcBatch.numRows()
        val numCols = srcBatch.numCols()
        
        // FIX: Create NEW GpuColumnVectors that wrap the cudf columns
        // This ensures proper lifecycle: new GpuColumnVector -> cudf column (with incRefCount)
        val newCols = new Array[GpuColumnVector](numCols)
        for (i <- 0 until numCols) {
          val srcGcv = srcBatch.column(i).asInstanceOf[GpuColumnVector]
          val cudfCol = srcGcv.getBase
          cudfCol.incRefCount() // Keep cudf column alive after spillable.close()
          // Create NEW GpuColumnVector wrapping the same cudf column
          newCols(i) = GpuColumnVector.from(cudfCol, srcGcv.dataType())
        }
        
        // Close spillable - this closes srcBatch and its GpuColumnVectors
        // But cudf columns survive due to incRefCount
        spillable.close()
        
        // Return new batch with new GpuColumnVectors
        // When downstream closes this batch, it decrements cudf ref counts to 0
        val colArray = newCols.asInstanceOf[Array[
          org.apache.spark.sql.vectorized.ColumnVector]]
        new ColumnarBatch(colArray, numRows)
      }
    } catch {
      case e: Exception =>
        logError(s"Fused Project + Aggregate failed: ${e.getMessage}", e)
        // Clean up any partial results created so far
        // Note: input batches are closed immediately in the streaming loop
        partialResults.foreach(_.close())
        throw e
    }
  }
}
