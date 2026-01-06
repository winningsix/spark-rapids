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

package com.nvidia.spark.rapids

import ai.rapids.cudf
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Utility for batching multiple AST-compatible expressions into a single kernel launch.
 * 
 * This optimization is particularly effective for workloads with many simple expressions,
 * such as the pre-project phase in aggregations with many variance/covariance calculations.
 * 
 * Example expressions that benefit:
 * - gpucoalesce(x, 0) * gpucoalesce(x, 0)  // for variance
 * - gpucoalesce(x, 0) * gpucoalesce(y, 0)  // for covariance
 * - if (condition > 0) x else 0            // conditional aggregation
 */
object AstBatchProject extends Logging {

  /**
   * Check if an expression can be converted to AST by actually trying to convert it.
   * This is a runtime check that catches any expressions that don't support AST.
   */
  def canConvertToAst(expr: Expression): Boolean = {
    expr match {
      case ge: GpuExpression =>
        try {
          // Try to convert to AST - this will throw if not supported
          // We use numFirstTableColumns=0 for single table scenarios
          ge.convertToAst(0)
          // Also check that all children can be converted
          ge.children.forall(canConvertToAst)
        } catch {
          case _: IllegalStateException => false
          case _: UnsupportedOperationException => false
          case _: Exception => false
        }
      case _ => false
    }
  }

  /**
   * Partition expressions into AST-compatible and non-AST groups.
   * 
   * @param expressions The expressions to partition
   * @return A tuple of (AST-compatible expressions with indices, non-AST expressions with indices)
   */
  def partitionByAstCompatibility(expressions: Seq[GpuExpression]): 
      (Seq[(GpuExpression, Int)], Seq[(GpuExpression, Int)]) = {
    val indexed = expressions.zipWithIndex
    indexed.partition { case (expr, _) => canConvertToAst(expr) }
  }

  /**
   * Batch project AST-compatible expressions using cudf's AST compute_column.
   * 
   * @param table The input table
   * @param astExprs AST-compatible expressions to evaluate
   * @param numFirstTableColumns Number of columns in the first table (for join scenarios)
   * @return Array of computed columns
   */
  def batchComputeAstColumns(
      table: cudf.Table,
      astExprs: Seq[GpuExpression],
      numFirstTableColumns: Int = 0): Array[cudf.ColumnVector] = {
    
    if (astExprs.isEmpty) {
      return Array.empty
    }

    logDebug(s"Batch computing ${astExprs.length} AST expressions")
    
    val results = new Array[cudf.ColumnVector](astExprs.length)
    
    closeOnExcept(results) { _ =>
      astExprs.zipWithIndex.foreach { case (expr, idx) =>
        try {
          val astExpr = expr.convertToAst(numFirstTableColumns)
          withResource(astExpr.compile()) { compiled =>
            // computeColumn is a method on CompiledExpression, not Table
            results(idx) = compiled.computeColumn(table)
          }
        } catch {
          case e: Exception =>
            logWarning(s"Failed to compute AST for expression $expr: ${e.getMessage}")
            throw e
        }
      }
    }
    
    results
  }

  /**
   * Execute a batched AST project on a columnar batch.
   * 
   * This method:
   * 1. Partitions expressions into AST-compatible and non-AST groups
   * 2. Evaluates AST expressions in batch using compute_column
   * 3. Evaluates non-AST expressions traditionally
   * 4. Combines results in correct order
   * 
   * @param batch Input columnar batch
   * @param expressions Expressions to evaluate
   * @return Output columnar batch with projected columns
   */
  def project(batch: ColumnarBatch, expressions: Seq[GpuExpression]): ColumnarBatch = {
    val (astExprs, nonAstExprs) = partitionByAstCompatibility(expressions)
    
    logDebug(s"AST project: ${astExprs.length} AST-compatible, ${nonAstExprs.length} non-AST")
    
    if (astExprs.isEmpty) {
      // All non-AST, use traditional evaluation
      return GpuProjectExec.project(batch, expressions)
    }
    
    if (nonAstExprs.isEmpty && astExprs.length >= 2) {
      // All AST-compatible with multiple expressions - use batch AST
      return projectAllAst(batch, astExprs.map(_._1))
    }
    
    // Mixed case - evaluate separately and combine
    projectMixed(batch, astExprs, nonAstExprs, expressions.length)
  }

  /**
   * Project when all expressions are AST-compatible.
   */
  private def projectAllAst(
      batch: ColumnarBatch, 
      astExprs: Seq[GpuExpression]): ColumnarBatch = {
    
    withResource(GpuColumnVector.from(batch)) { table =>
      // For single-table operations, numFirstTableColumns should be the number of columns
      // in the input table so that all column references are treated as LEFT table references
      val numCols = table.getNumberOfColumns
      val resultColumns = batchComputeAstColumns(table, astExprs, numCols)
      // GpuColumnVector.from takes ownership of the cudf.ColumnVector, so we need to
      // track which columns have been wrapped. On exception, only close unwrapped columns.
      val gpuCols = new Array[GpuColumnVector](resultColumns.length)
      var i = 0
      try {
        while (i < resultColumns.length) {
          // from() takes ownership - set to null before wrapping to avoid double-close
          val cudfCol = resultColumns(i)
          resultColumns(i) = null
          gpuCols(i) = GpuColumnVector.from(cudfCol, astExprs(i).dataType)
          i += 1
        }
        new ColumnarBatch(gpuCols.map(_.asInstanceOf[ColumnVector]), batch.numRows())
      } catch {
        case e: Exception =>
          // Close any GpuColumnVectors we already created
          gpuCols.filter(_ != null).foreach(_.close())
          // Close any cudf.ColumnVectors we haven't wrapped yet
          resultColumns.filter(_ != null).foreach(_.close())
          throw e
      }
    }
  }

  /**
   * Project when there's a mix of AST and non-AST expressions.
   */
  private def projectMixed(
      batch: ColumnarBatch,
      astExprs: Seq[(GpuExpression, Int)],
      nonAstExprs: Seq[(GpuExpression, Int)],
      totalExprs: Int): ColumnarBatch = {
    
    val resultColumns = new Array[GpuColumnVector](totalExprs)
    
    try {
      // Evaluate AST expressions in batch
      if (astExprs.nonEmpty) {
        withResource(GpuColumnVector.from(batch)) { table =>
          // For single-table operations, numFirstTableColumns should be the number of columns
          val numCols = table.getNumberOfColumns
          val astResults = batchComputeAstColumns(table, astExprs.map(_._1), numCols)
          // Wrap each column, clearing the array entry to avoid double-close on exception
          var i = 0
          try {
            while (i < astResults.length) {
              val (expr, origIdx) = astExprs(i)
              val cudfCol = astResults(i)
              astResults(i) = null // Clear before wrapping to avoid double-close
              resultColumns(origIdx) = GpuColumnVector.from(cudfCol, expr.dataType)
              i += 1
            }
          } catch {
            case e: Exception =>
              // Close any remaining unwrapped cudf columns
              astResults.filter(_ != null).foreach(_.close())
              throw e
          }
        }
      }
      
      // Evaluate non-AST expressions traditionally
      nonAstExprs.foreach { case (expr, origIdx) =>
        resultColumns(origIdx) = expr.columnarEval(batch)
      }
      
      new ColumnarBatch(resultColumns.map(_.asInstanceOf[ColumnVector]), batch.numRows())
    } catch {
      case e: Exception =>
        // Close any GpuColumnVectors we created
        resultColumns.filter(_ != null).foreach(_.close())
        throw e
    }
  }

  /**
   * Statistics helper for logging AST optimization effectiveness.
   */
  case class AstProjectStats(
      totalExpressions: Int,
      astCompatible: Int,
      nonAst: Int) {
    
    def astRatio: Double = if (totalExpressions > 0) {
      astCompatible.toDouble / totalExpressions
    } else 0.0
    
    override def toString: String = {
      f"AstProjectStats(total=$totalExpressions, ast=$astCompatible (${astRatio * 100}%.1f%%), nonAst=$nonAst)"
    }
  }

  /**
   * Analyze expressions for AST compatibility without executing.
   */
  def analyzeExpressions(expressions: Seq[Expression]): AstProjectStats = {
    val (astExprs, nonAstExprs) = expressions.partition(canConvertToAst)
    AstProjectStats(expressions.length, astExprs.length, nonAstExprs.length)
  }
}

/**
 * A tiered project that uses AST batch compilation where possible.
 * 
 * This extends the standard tiered project approach by:
 * 1. Identifying AST-compatible expressions in each tier
 * 2. Batching those expressions into single kernel launches
 * 3. Falling back to traditional evaluation for non-AST expressions
 */
case class GpuAstTieredProject(exprTiers: Seq[Seq[GpuExpression]]) extends Logging {

  // Analyze each tier for AST compatibility
  lazy val tierStats: Seq[AstBatchProject.AstProjectStats] = exprTiers.map { tier =>
    AstBatchProject.analyzeExpressions(tier)
  }

  /**
   * Log statistics about AST compatibility.
   */
  def logAstStats(): Unit = {
    tierStats.zipWithIndex.foreach { case (stats, tier) =>
      logInfo(s"Tier $tier: $stats")
    }
  }

  /**
   * Project using AST batch compilation where possible.
   */
  def project(batch: ColumnarBatch): ColumnarBatch = {
    var currentBatch = batch
    var isFirst = true
    
    for (tier <- exprTiers) {
      val projected = try {
        AstBatchProject.project(currentBatch, tier)
      } finally {
        if (!isFirst) {
          currentBatch.close()
        }
      }
      currentBatch = projected
      isFirst = false
    }
    
    currentBatch
  }
}
