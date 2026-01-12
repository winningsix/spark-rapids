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

package com.nvidia.spark.rapids.parquet

import ai.rapids.cudf.ast.AstExpression
import com.nvidia.spark.rapids.{GpuExpression, GpuOverrides, RapidsConf}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types.StructType

/**
 * Utility object to convert Spark Catalyst Expressions to cuDF AST expressions.
 * This leverages the existing spark-rapids GpuExpression infrastructure.
 * 
 * Used for filter pushdown in cuDF Hybrid Scan.
 */
object SparkExpressionToCudfAst extends Logging {

  /**
   * Preprocess filters to convert equality comparisons to range comparisons.
   * This is needed because GpuEqualTo converts to NOT(NOT_EQUAL(...)) which doesn't work
   * correctly with cuDF's statistics-based row group filtering for certain types.
   * 
   * Converts: col = literal  ->  col >= literal AND col <= literal
   */
  private def preprocessEqualityFilters(filters: Seq[Expression]): Seq[Expression] = {
    filters.flatMap { filter =>
      filter match {
        case EqualTo(attr: AttributeReference, lit: Literal) =>
          logInfo(s"[AST Preprocess] Converting equality to range: ${filter.sql}")
          Seq(GreaterThanOrEqual(attr, lit), LessThanOrEqual(attr, lit))
        case EqualTo(lit: Literal, attr: AttributeReference) =>
          logInfo(s"[AST Preprocess] Converting equality to range: ${filter.sql}")
          Seq(GreaterThanOrEqual(attr, lit), LessThanOrEqual(attr, lit))
        case _ => Seq(filter)
      }
    }
  }

  /**
   * Try to convert a sequence of Spark filter expressions to a single cuDF AST expression
   * by combining them with AND.
   * 
   * This method reuses spark-rapids' existing expression conversion mechanism:
   * 1. Wrap Spark Expression with GpuOverrides.wrapExpr()
   * 2. Check if it can be converted to AST with canThisBeAst
   * 3. Convert to GpuExpression with convertToGpu()
   * 4. Convert to cuDF AST with convertToAst()
   * 
   * @param filters The Spark filter expressions
   * @param readSchema The schema being read (used for column binding)
   * @param rapidsConf The Rapids configuration
   * @return Some(AstExpression) if all filters can be converted, None otherwise
   */
  def convertFilters(
      filters: Seq[Expression],
      readSchema: StructType,
      rapidsConf: RapidsConf): Option[AstExpression] = {
    
    if (filters.isEmpty) {
      logDebug("No filters to convert")
      return None
    }

    // Preprocess equality filters to range filters (works better with stats filtering)
    val preprocessedFilters = preprocessEqualityFilters(filters)

    try {
      // Debug: log filter expression types
      preprocessedFilters.foreach { filter =>
        val children = filter.children.map { c =>
          s"${c.getClass.getSimpleName}(${c.sql})"
        }.mkString(", ")
        logInfo(s"[AST Debug] Filter: ${filter.sql}, " +
          s"Class: ${filter.getClass.getSimpleName}, Children: $children")
      }
      logInfo(s"[AST Debug] readSchema fields: ${readSchema.fieldNames.mkString(", ")}")
      
      // First, bind the filter expressions to the schema
      val boundFilters = preprocessedFilters.map { filter =>
        val bound = bindExpression(filter, readSchema)
        logInfo(s"[AST Debug] After bind: ${bound.sql}, Class: ${bound.getClass.getSimpleName}")
        bound
      }

      // Combine all filters with AND
      val combinedFilter = boundFilters.reduce { (left, right) =>
        And(left, right)
      }

      // Wrap and convert using spark-rapids mechanism
      val exprMeta = GpuOverrides.wrapExpr(combinedFilter, rapidsConf, None)
      exprMeta.tagForGpu()

      // Check if this expression can be converted to AST
      if (!exprMeta.canThisBeAst) {
        logInfo(s"Filter cannot be converted to AST: ${exprMeta.explainAst(all = false)}")
        return None
      }

      // Convert to GPU expression
      val gpuExpr = exprMeta.convertToGpu()
      
      // Ensure it's a GpuExpression
      gpuExpr match {
        case gpu: GpuExpression =>
          // Convert to cuDF AST
          // numFirstTableColumns = readSchema.length to ensure all columns use LEFT table reference
          // (cuDF's stats_filter_helpers.cpp:34 requires LEFT table reference for statistics AST)
          val ast = gpu.convertToAst(readSchema.length)
          logInfo(s"Successfully converted ${preprocessedFilters.size} filters to cuDF AST")
          Some(ast)
        case _ =>
          logWarning(s"Converted expression is not a GpuExpression: ${gpuExpr.getClass}")
          None
      }
    } catch {
      case e: Exception =>
        logWarning(s"Failed to convert filters to cuDF AST: ${e.getMessage}")
        None
    }
  }

  /**
   * Bind an expression to a schema, resolving attribute references.
   * This creates bound references with correct ordinals.
   */
  private def bindExpression(expr: Expression, schema: StructType): Expression = {
    expr.transform {
      case attr: AttributeReference =>
        val index = schema.fieldIndex(attr.name)
        BoundReference(index, schema(index).dataType, schema(index).nullable)
    }
  }

  /**
   * Check if a sequence of filters can be converted to cuDF AST.
   * This actually attempts the conversion to ensure consistency with convertFilters.
   * 
   * @param filters The Spark filter expressions
   * @param readSchema The schema being read
   * @param rapidsConf The Rapids configuration
   * @return true if filters can be converted to AST
   */
  def canConvertFilters(
      filters: Seq[Expression],
      readSchema: StructType,
      rapidsConf: RapidsConf): Boolean = {
    
    if (filters.isEmpty) {
      return false
    }

    try {
      // Actually try the conversion to ensure consistency
      val result = convertFilters(filters, readSchema, rapidsConf)
      val canConvert = result.isDefined
      
      if (!canConvert) {
        // Log individual filter status for debugging
        logWarning(s"Filters cannot be converted to AST")
        filters.foreach { filter =>
          val bound = bindExpression(filter, readSchema)
          val meta = GpuOverrides.wrapExpr(bound, rapidsConf, None)
          meta.tagForGpu()
          val explain = meta.explainAst(all = false)
          logWarning(s"  Filter '${filter.sql}' canBeAst=${meta.canThisBeAst}: $explain")
        }
      }
      canConvert
    } catch {
      case e: Exception => 
        logWarning(s"Exception checking AST conversion: ${e.getMessage}")
        false
    }
  }

  /**
   * Result of partial filter conversion.
   * @param astFilters Filters that were successfully converted to AST
   * @param astExpression The combined AST expression (if any astFilters exist)
   * @param remainingFilters Filters that could not be converted to AST
   */
  case class PartialFilterResult(
      astFilters: Seq[Expression],
      astExpression: Option[AstExpression],
      remainingFilters: Seq[Expression]
  )

  /**
   * Convert filters to AST with partial pushdown support.
   * Filters that can be converted to AST are combined and returned.
   * Filters that cannot be converted are returned separately for GPU filtering.
   * 
   * @param filters The Spark filter expressions
   * @param readSchema The schema being read (used for column binding)
   * @param rapidsConf The Rapids configuration
   * @return PartialFilterResult containing AST-compatible and remaining filters
   */
  def convertFiltersPartial(
      filters: Seq[Expression],
      readSchema: StructType,
      rapidsConf: RapidsConf): PartialFilterResult = {
    
    if (filters.isEmpty) {
      return PartialFilterResult(Seq.empty, None, Seq.empty)
    }

    // Separate filters into AST-compatible and non-AST
    val (astCompatible, nonAstCompatible) = filters.partition { filter =>
      canConvertSingleFilter(filter, readSchema, rapidsConf)
    }

    logInfo(s"Partial filter pushdown: ${astCompatible.size} AST-compatible, " +
      s"${nonAstCompatible.size} remaining")

    // Convert AST-compatible filters
    val astExpr = if (astCompatible.nonEmpty) {
      convertFilters(astCompatible, readSchema, rapidsConf)
    } else {
      None
    }

    // Log details
    if (astCompatible.nonEmpty) {
      logInfo(s"AST-compatible filters: ${astCompatible.map(_.sql).mkString(", ")}")
    }
    if (nonAstCompatible.nonEmpty) {
      logInfo(s"Remaining filters (GPU): ${nonAstCompatible.map(_.sql).mkString(", ")}")
    }

    PartialFilterResult(astCompatible, astExpr, nonAstCompatible)
  }

  /**
   * Check if a single filter can be converted to AST.
   * This performs a full trial conversion including AST generation.
   */
  def canConvertSingleFilter(
      filter: Expression,
      readSchema: StructType,
      rapidsConf: RapidsConf): Boolean = {
    try {
      val bound = bindExpression(filter, readSchema)
      val meta = GpuOverrides.wrapExpr(bound, rapidsConf, None)
      meta.tagForGpu()
      
      if (!meta.canThisBeAst) {
        return false
      }
      
      // Try actual conversion to GPU expression
      val gpuExpr = meta.convertToGpu()
      gpuExpr match {
        case ge: GpuExpression =>
          // Also try AST conversion to catch literal type issues
          // Use readSchema.length to ensure LEFT table reference (required by cuDF stats filtering)
          ge.convertToAst(readSchema.length) // This will throw if AST conversion fails
          true
        case _ => false
      }
    } catch {
      case e: Exception => 
        logDebug(s"Filter ${filter.sql} cannot be converted to AST: ${e.getMessage}")
        false
    }
  }

  /**
   * Check if any filters can be converted to AST (for partial pushdown).
   * Returns true if at least one filter can be converted.
   */
  def canConvertAnyFilters(
      filters: Seq[Expression],
      readSchema: StructType,
      rapidsConf: RapidsConf): Boolean = {
    filters.exists(f => canConvertSingleFilter(f, readSchema, rapidsConf))
  }
}
