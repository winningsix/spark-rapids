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

import java.util.concurrent.{Callable, Executors, ExecutorService, Future, TimeUnit}

import scala.annotation.nowarn
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ColumnVector, HostMemoryBuffer, Table}
import ai.rapids.cudf.ast.CompiledExpression
import com.nvidia.spark.rapids.{GpuColumnVector, GpuMetric, GpuSemaphore, NoopMetric,
  RapidsConf, ThreadFactoryBuilder}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.ParquetHybridScan
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FSDataInputStream, Path}

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{And, AttributeReference,
  EqualTo, Expression, GreaterThan, GreaterThanOrEqual, In, LessThan, LessThanOrEqual, Literal}
import org.apache.spark.sql.connector.read.PartitionReader
import org.apache.spark.sql.execution.datasources.PartitionedFile
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Utility object for cuDF Hybrid Scan operations.
 */
object CudfHybridScanUtils extends Logging {

  /**
   * Determine if cuDF Hybrid Scan should be used for this query.
   * 
   * @param rapidsConf The Rapids configuration
   * @param dataFilters The data filters to evaluate
   * @param readSchema The schema being read
   * @return true if Hybrid Scan should be used
   */
  def shouldUseCudfHybridScan(
      rapidsConf: RapidsConf,
      dataFilters: Seq[Expression],
      readSchema: StructType): Boolean = {
    
    // Check if feature is enabled
    if (!rapidsConf.cudfHybridScanEnabled) {
      logDebug("cuDF Hybrid Scan disabled: feature not enabled in config")
      return false
    }

    // Hybrid Scan is beneficial when there are selective filters
    if (dataFilters.isEmpty) {
      logInfo("cuDF Hybrid Scan disabled: no data filters present")
      return false
    }

    // Check if we have filter columns (columns used in filters)
    val filterColumns = extractFilterColumns(dataFilters)
    if (filterColumns.isEmpty) {
      logDebug("cuDF Hybrid Scan disabled: no filter columns extracted")
      return false
    }

    // NOTE: Filter type support is determined by SparkExpressionToCudfAst conversion below.
    // Avoid hard-coding type blacklists here (e.g. DATE/TIMESTAMP), since native support evolves.

    // Check if we have payload columns (columns not used in filters)
    val payloadColumns = readSchema.fieldNames.filterNot(filterColumns.contains)
    if (payloadColumns.isEmpty) {
      logDebug("cuDF Hybrid Scan disabled: all columns are filter columns")
      return false
    }

    // Check if ANY filters can be converted to cuDF AST (partial pushdown support)
    // At least one filter must be AST-compatible for Hybrid Scan to be beneficial
    if (!SparkExpressionToCudfAst.canConvertAnyFilters(dataFilters, readSchema, rapidsConf)) {
      logInfo("cuDF Hybrid Scan disabled: no filters can be converted to cuDF AST")
      return false
    }

    // Get partial filter result for logging
    val partialResult = SparkExpressionToCudfAst.convertFiltersPartial(
      dataFilters, readSchema, rapidsConf)
    logInfo(s"cuDF Hybrid Scan enabled with ${filterColumns.size} filter columns, " +
      s"${payloadColumns.length} payload columns, " +
      s"${partialResult.astFilters.size} AST filters, " +
      s"${partialResult.remainingFilters.size} GPU filters")
    true
  }
  
  /**
   * Extract column names used in filter expressions.
   */
  def extractFilterColumns(filters: Seq[Expression]): Set[String] = {
    filters.flatMap { expr =>
      expr.references.map(_.name)
    }.toSet
  }

  /**
   * Separate columns into filter columns (used in predicates) and payload columns.
   * IMPORTANT: Both arrays must be ordered according to readSchema to ensure
   * the combined table matches the expected schema order.
   */
  def separateColumns(
      readSchema: StructType,
      dataFilters: Seq[Expression]): (Array[String], Array[String]) = {
    val filterColumnSet = extractFilterColumns(dataFilters)
    // Preserve readSchema order for filter columns
    val filterColumns = readSchema.fieldNames.filter(filterColumnSet.contains)
    // Preserve readSchema order for payload columns (non-filter columns)
    val payloadColumns = readSchema.fieldNames.filterNot(filterColumnSet.contains)
    (filterColumns, payloadColumns)
  }

  /**
   * Estimate filter selectivity based on row group statistics (min/max values).
   * This helps decide if Hybrid Scan is worthwhile for this query.
   * 
   * Selectivity estimation logic:
   * - For equality filters (col = value): estimate based on value range
   * - For range filters (col > value): estimate based on overlap with value range
   * - For compound filters (AND): multiply individual selectivities
   * 
   * @param filters The filter expressions
   * @param rowGroupStats Map of column name to (min, max) statistics
   * @param totalRows Total rows in all row groups
   * @return Estimated selectivity between 0.0 and 1.0
   */
  def estimateSelectivity(
      filters: Seq[Expression],
      rowGroupStats: Map[String, (Any, Any)],
      totalRows: Long): Double = {
    
    if (filters.isEmpty || totalRows == 0) {
      return 1.0 // No filtering, all rows selected
    }

    // Calculate selectivity for each filter and combine with AND semantics
    val filterSelectivities = filters.map(estimateSingleFilterSelectivity(_, rowGroupStats))
    
    // Combine selectivities - for AND, multiply them
    // Cap at minimum of 0.001 to avoid underestimation
    val combined = filterSelectivities.product
    math.max(0.001, math.min(1.0, combined))
  }

  /**
   * Estimate selectivity for a single filter expression.
   */
  private def estimateSingleFilterSelectivity(
      filter: Expression,
      stats: Map[String, (Any, Any)]): Double = {
    
    filter match {
      // Equality filter: col = literal
      case EqualTo(attr: AttributeReference, lit: Literal) =>
        estimateEqualitySelectivity(attr.name, lit.value, stats)
      case EqualTo(lit: Literal, attr: AttributeReference) =>
        estimateEqualitySelectivity(attr.name, lit.value, stats)
        
      // Range filters
      case GreaterThan(attr: AttributeReference, lit: Literal) =>
        estimateRangeSelectivity(attr.name, lit.value, stats, isGreater = true, inclusive = false)
      case GreaterThanOrEqual(attr: AttributeReference, lit: Literal) =>
        estimateRangeSelectivity(attr.name, lit.value, stats, isGreater = true, inclusive = true)
      case LessThan(attr: AttributeReference, lit: Literal) =>
        estimateRangeSelectivity(attr.name, lit.value, stats, isGreater = false, inclusive = false)
      case LessThanOrEqual(attr: AttributeReference, lit: Literal) =>
        estimateRangeSelectivity(attr.name, lit.value, stats, isGreater = false, inclusive = true)
        
      // IN filter: col IN (v1, v2, ...)
      case In(attr: AttributeReference, list) =>
        // Each value in list contributes equality selectivity
        val numValues = list.size
        math.min(1.0, numValues * estimateEqualitySelectivity(attr.name, null, stats))
        
      // AND filter: combine sub-filter selectivities
      case And(left, right) =>
        val leftSel = estimateSingleFilterSelectivity(left, stats)
        val rightSel = estimateSingleFilterSelectivity(right, stats)
        leftSel * rightSel
        
      // Unknown filter type - assume moderate selectivity
      case _ =>
        logDebug(s"Unknown filter type for selectivity estimation: ${filter.getClass.getName}")
        0.5
    }
  }

  /**
   * Estimate selectivity for equality filter based on column value range.
   * If value is within range, estimate as 1/distinct_values.
   * If value is outside range, selectivity is 0.
   */
  private def estimateEqualitySelectivity(
      colName: String,
      value: Any,
      stats: Map[String, (Any, Any)]): Double = {
    
    stats.get(colName) match {
      case Some((minVal, maxVal)) =>
        // Estimate distinct values from range
        val distinctEstimate = estimateDistinctValues(minVal, maxVal)
        if (distinctEstimate > 0) {
          1.0 / distinctEstimate
        } else {
          0.1 // Default for unknown range
        }
      case None =>
        0.1 // No stats available, assume 10% selectivity
    }
  }

  /**
   * Estimate selectivity for range filter based on overlap with column value range.
   */
  private def estimateRangeSelectivity(
      colName: String,
      value: Any,
      stats: Map[String, (Any, Any)],
      isGreater: Boolean,
      inclusive: Boolean): Double = {
    
    stats.get(colName) match {
      case Some((minVal, maxVal)) =>
        // Calculate overlap ratio
        val overlap = calculateRangeOverlap(minVal, maxVal, value, isGreater)
        math.max(0.01, math.min(1.0, overlap))
      case None =>
        0.5 // No stats, assume 50% selectivity for range
    }
  }

  /**
   * Estimate number of distinct values from min/max range.
   */
  private def estimateDistinctValues(min: Any, max: Any): Long = {
    (min, max) match {
      case (minL: java.lang.Long, maxL: java.lang.Long) =>
        math.abs(maxL - minL) + 1
      case (minI: java.lang.Integer, maxI: java.lang.Integer) =>
        math.abs(maxI.toLong - minI.toLong) + 1
      case (minD: java.lang.Double, maxD: java.lang.Double) =>
        // For doubles, estimate based on range magnitude
        math.max(1, (maxD - minD).toLong)
      case (_: String, _: String) =>
        // For strings, use a heuristic
        1000 // Assume moderate cardinality
      case _ =>
        100 // Default estimate
    }
  }

  /**
   * Calculate what fraction of the range [min, max] satisfies the filter.
   */
  private def calculateRangeOverlap(min: Any, max: Any, value: Any, isGreater: Boolean): Double = {
    try {
      val (minD, maxD, valD) = (min, max, value) match {
        case (minL: java.lang.Long, maxL: java.lang.Long, v: java.lang.Long) =>
          (minL.toDouble, maxL.toDouble, v.toDouble)
        case (minI: java.lang.Integer, maxI: java.lang.Integer, v: java.lang.Integer) =>
          (minI.toDouble, maxI.toDouble, v.toDouble)
        case (minD: java.lang.Double, maxD: java.lang.Double, v: java.lang.Double) =>
          (minD.doubleValue, maxD.doubleValue, v.doubleValue)
        case _ =>
          return 0.5 // Cannot compute, assume 50%
      }
      
      val range = maxD - minD
      if (range <= 0) return 0.5
      
      if (isGreater) {
        // col > value: fraction of range above value
        if (valD >= maxD) 0.0
        else if (valD <= minD) 1.0
        else (maxD - valD) / range
      } else {
        // col < value: fraction of range below value
        if (valD <= minD) 0.0
        else if (valD >= maxD) 1.0
        else (valD - minD) / range
      }
    } catch {
      case _: Exception => 0.5
    }
  }

  /**
   * Check if Hybrid Scan is likely beneficial based on selectivity estimation.
   * According to cuDF documentation, Hybrid Scan benefits when:
   * - Filter selectivity is low (< 10% of rows match)
   * - There are many payload columns relative to filter columns
   * 
   * @param estimatedSelectivity Estimated filter selectivity (0.0 to 1.0)
   * @param numFilterColumns Number of filter columns
   * @param numPayloadColumns Number of payload columns
   * @param threshold Selectivity threshold from config
   * @return true if Hybrid Scan is recommended
   */
  def isHybridScanBeneficial(
      estimatedSelectivity: Double,
      numFilterColumns: Int,
      numPayloadColumns: Int,
      threshold: Double): Boolean = {
    
    // Primary criterion: selectivity must be below threshold
    if (estimatedSelectivity > threshold) {
      logInfo(f"Hybrid Scan not beneficial: selectivity $estimatedSelectivity%.2f > " +
        f"threshold $threshold%.2f")
      return false
    }
    
    // Secondary criterion: payload columns should be significant
    // If most columns are filter columns, there's no benefit
    val payloadRatio = numPayloadColumns.toDouble / (numFilterColumns + numPayloadColumns)
    if (payloadRatio < 0.3) {
      logInfo(f"Hybrid Scan not beneficial: payload ratio $payloadRatio%.2f < 0.3")
      return false
    }
    
    logInfo(f"Hybrid Scan beneficial: selectivity=$estimatedSelectivity%.2f, " +
      f"payloadRatio=$payloadRatio%.2f")
    true
  }
}

/**
 * Result of page index detection in a Parquet file.
 */
case class PageIndexInfo(
    hasColumnIndex: Boolean,
    hasOffsetIndex: Boolean,
    pageIndexByteRange: (Long, Long)) {
  
  def hasFullPageIndex: Boolean = hasColumnIndex && hasOffsetIndex
  def hasAnyPageIndex: Boolean = hasColumnIndex || hasOffsetIndex
}

/**
 * Utility object for detecting page index presence in Parquet files.
 */
object PageIndexDetector extends Logging {

  /**
   * Detect if a Parquet file has page index information.
   * Page index consists of:
   * - Column Index: min/max statistics per page
   * - Offset Index: page locations and row counts
   * 
   * Both are required for effective page-level pruning in Hybrid Scan.
   * 
   * @param hybridScanReader The ParquetHybridScan reader with parsed footer
   * @return PageIndexInfo with detection results
   */
  def detectPageIndex(hybridScanReader: ParquetHybridScan): PageIndexInfo = {
    try {
      val pageIndexRange = hybridScanReader.getPageIndexByteRange()
      val offset = pageIndexRange(0)
      val length = pageIndexRange(1)
      
      // If byte range length > 0, page index is present
      val hasPageIndex = length > 0
      
      logDebug(s"Page index detection: offset=$offset, length=$length, " +
        s"hasPageIndex=$hasPageIndex")
      
      // cuDF only exposes combined page index range, so we assume both parts are present
      PageIndexInfo(
        hasColumnIndex = hasPageIndex,
        hasOffsetIndex = hasPageIndex,
        pageIndexByteRange = (offset, length))
    } catch {
      case e: Exception =>
        logDebug(s"Page index detection failed: ${e.getMessage}")
        PageIndexInfo(
          hasColumnIndex = false,
          hasOffsetIndex = false,
          pageIndexByteRange = (0L, 0L))
    }
  }

  /**
   * Check if page index pruning should be enabled for this file.
   * 
   * @param configEnabled Whether page index is enabled in config
   * @param pageIndexInfo Detection result from file
   * @return true if page index pruning should be used
   */
  def shouldUsePageIndex(configEnabled: Boolean, pageIndexInfo: PageIndexInfo): Boolean = {
    if (!configEnabled) {
      logDebug("Page index disabled by configuration")
      return false
    }
    
    if (!pageIndexInfo.hasFullPageIndex) {
      logInfo("Page index disabled: file does not have complete page index " +
        s"(columnIndex=${pageIndexInfo.hasColumnIndex}, " +
        s"offsetIndex=${pageIndexInfo.hasOffsetIndex})")
      return false
    }
    
    logInfo(s"Page index enabled: byte range " +
      s"${pageIndexInfo.pageIndexByteRange._1}-${pageIndexInfo.pageIndexByteRange._2}")
    true
  }
}

/**
 * Result of reading a single row group's column chunks.
 */
case class RowGroupReadResult(
    rowGroupIndex: Int,
    filterBuffers: Array[HostMemoryBuffer],
    payloadBuffers: Array[HostMemoryBuffer],
    numRows: Long) extends AutoCloseable {
  
  override def close(): Unit = {
    filterBuffers.foreach(_.close())
    payloadBuffers.foreach(_.close())
  }
}

/**
 * Task for reading column chunks of a single row group in parallel.
 */
class RowGroupIOTask(
    fs: FileSystem,
    filePath: Path,
    rowGroupIndex: Int,
    filterRanges: Array[Long],  // pairs of (offset, size)
    payloadRanges: Array[Long], // pairs of (offset, size)
    numRows: Long) extends Callable[RowGroupReadResult] with Logging {

  override def call(): RowGroupReadResult = {
    val filterBuffers = new ArrayBuffer[HostMemoryBuffer]()
    val payloadBuffers = new ArrayBuffer[HostMemoryBuffer]()
    
    var in: FSDataInputStream = null
    try {
      in = fs.open(filePath)
      
      // Read filter column chunks
      var i = 0
      while (i < filterRanges.length) {
        val offset = filterRanges(i)
        val size = filterRanges(i + 1).toInt
        if (size > 0) {
          val buffer = readChunk(in, offset, size)
          filterBuffers += buffer
        }
        i += 2
      }
      
      // Read payload column chunks
      i = 0
      while (i < payloadRanges.length) {
        val offset = payloadRanges(i)
        val size = payloadRanges(i + 1).toInt
        if (size > 0) {
          val buffer = readChunk(in, offset, size)
          payloadBuffers += buffer
        }
        i += 2
      }
      
      RowGroupReadResult(rowGroupIndex, filterBuffers.toArray, payloadBuffers.toArray, numRows)
    } catch {
      case e: Exception =>
        filterBuffers.foreach(_.close())
        payloadBuffers.foreach(_.close())
        throw e
    } finally {
      if (in != null) {
        in.close()
      }
    }
  }

  private def readChunk(in: FSDataInputStream, offset: Long, size: Int): HostMemoryBuffer = {
    in.seek(offset)
    val buffer = HostMemoryBuffer.allocate(size)
    try {
      val bytes = new Array[Byte](size)
      in.readFully(bytes)
      buffer.setBytes(0, bytes, 0, size)
      buffer
    } catch {
      case e: Exception =>
        buffer.close()
        throw e
    }
  }
}

/**
 * Thread pool for parallel row group IO operations.
 */
object HybridScanThreadPool extends Logging {
  @volatile
  private var threadPool: Option[ExecutorService] = None
  
  def getOrCreateThreadPool(numThreads: Int): ExecutorService = synchronized {
    if (threadPool.isEmpty) {
      val factory = new ThreadFactoryBuilder()
        .setDaemon(true)
        .setNameFormat("cudf-hybrid-scan-io-%d")
        .build()
      threadPool = Some(Executors.newFixedThreadPool(numThreads, factory))
      logInfo(s"Created cuDF Hybrid Scan IO thread pool with $numThreads threads")
    }
    threadPool.get
  }
  
  def shutdown(): Unit = synchronized {
    threadPool.foreach { pool =>
      pool.shutdown()
      try {
        if (!pool.awaitTermination(5, TimeUnit.SECONDS)) {
          pool.shutdownNow()
        }
      } catch {
        case _: InterruptedException =>
          pool.shutdownNow()
      }
    }
    threadPool = None
  }
}

/**
 * A PartitionReader that uses cuDF Hybrid Scan for reading Parquet files
 * with row group level parallel IO and selectivity-based optimization.
 * 
 * Hybrid Scan is optimized for highly selective filters. It reads the file in two passes:
 * 1. First pass: Read only filter columns and build a row mask
 * 2. Second pass: Read only payload columns using the row mask to skip unnecessary data
 * 
 * Key optimizations:
 * - Selectivity estimation: Only uses Hybrid Scan when filter is selective enough
 * - Page index detection: Automatically detects and uses page index for better pruning
 * - Parallel IO: Multiple row groups can be read concurrently
 * - Pipelined I/O: Overlaps filter and payload column reading (when enabled)
 */
@nowarn("msg=never used")
class CudfHybridScanPartitionReader(
    conf: Configuration,
    split: PartitionedFile,
    filePath: Path,
    readSchema: StructType,
    filterColumns: Array[String],
    payloadColumns: Array[String],
    usePageIndexConfig: Boolean,  // Renamed: config setting, may be overridden
    useBloomFilter: Boolean,
    parallelIOEnabled: Boolean,
    numIOThreads: Int,
    maxRowGroupsParallel: Int,
    selectivityThreshold: Double,  // Added: threshold for Hybrid Scan benefit
    pipeliningEnabled: Boolean,    // Added: enable intra-file I/O pipelining
    metrics: Map[String, GpuMetric],
    dataFilters: Seq[Expression] = Seq.empty) extends PartitionReader[ColumnarBatch] with Logging {

  private var batch: Option[ColumnarBatch] = None
  private var hasNextBatch = true
  private var hybridScanReader: ParquetHybridScan = _
  
  // Dynamic page index state - determined at runtime
  private var effectiveUsePageIndex: Boolean = usePageIndexConfig
  private var pageIndexInfo: Option[PageIndexInfo] = None
  
  // Selectivity estimation state
  private var estimatedSelectivity: Double = 1.0
  private var hybridScanBeneficial: Boolean = true  // Assume beneficial until checked
  
  // Pipelining state - for async I/O
  private val pipelineExecutor: ExecutorService = if (pipeliningEnabled) {
    Executors.newFixedThreadPool(2, new ThreadFactoryBuilder()
      .setDaemon(true)
      .setNameFormat("cudf-hybrid-scan-pipeline-%d")
      .build())
  } else {
    null
  }
  
  // Schema in cuDF column order: filterColumns + payloadColumns
  // This is important for AST column index binding - cuDF reads columns in this order
  @transient private lazy val cudfOrderSchema: StructType = {
    val allColumns = filterColumns ++ payloadColumns
    StructType(allColumns.map(name => readSchema(readSchema.fieldIndex(name))))
  }
  
  // Partial filter pushdown result - separates AST-compatible and remaining filters
  // Use cudfOrderSchema to ensure column indices match cuDF's actual column order
  @transient private lazy val partialFilterResult: SparkExpressionToCudfAst.PartialFilterResult = {
    val confMap = scala.collection.mutable.Map[String, String]()
    val rapidsConf = new RapidsConf(confMap.toMap)
    SparkExpressionToCudfAst.convertFiltersPartial(dataFilters, cudfOrderSchema, rapidsConf)
  }

  // Compiled AST filter for Hybrid Scan filter pushdown (partial pushdown support)
  // This is created lazily to avoid serialization issues
  @transient private lazy val compiledAstFilter: Option[CompiledExpression] = {
    if (dataFilters.isEmpty) {
      logDebug("No data filters to compile for AST")
      None
    } else {
      try {
        partialFilterResult.astExpression match {
          case Some(astExpr) =>
            val compiled = astExpr.compile()
            val astCnt = partialFilterResult.astFilters.size
            val gpuCnt = partialFilterResult.remainingFilters.size
            logInfo(s"Successfully compiled $astCnt filters to cuDF AST ($gpuCnt for GPU)")
            Some(compiled)
          case None =>
            logInfo("No filters could be converted to cuDF AST for partial pushdown")
            None
        }
      } catch {
        case e: Exception =>
          logWarning(s"Failed to compile filters to cuDF AST: ${e.getMessage}")
          None
      }
    }
  }
  
  // Filters that couldn't be pushed to AST - will be applied on GPU
  def getRemainingFilters: Seq[Expression] = partialFilterResult.remainingFilters
  
  // Check if we have AST filter for pushdown
  def hasAstFilter: Boolean = compiledAstFilter.isDefined
  
  // Get AST filter handle for native code, or 0 if no filter
  private def getAstFilterHandle: Long = {
    val handle = compiledAstFilter.map(_.getNativeHandle).getOrElse(0L)
    handle
  }

  // Metrics
  private val readTime = metrics.getOrElse(GpuMetric.READ_FS_TIME, NoopMetric)
  private val filterTime = metrics.getOrElse(GpuMetric.FILTER_TIME, NoopMetric)
  private val numOutputRows = metrics.getOrElse(GpuMetric.NUM_OUTPUT_ROWS, NoopMetric)
  private val numOutputBatches = metrics.getOrElse(GpuMetric.NUM_OUTPUT_BATCHES, NoopMetric)
  // Hybrid Scan specific metrics
  private val filterColTime = metrics.getOrElse(GpuMetric.HYBRID_FILTER_COL_TIME, NoopMetric)
  private val payloadColTime = metrics.getOrElse(GpuMetric.HYBRID_PAYLOAD_COL_TIME, NoopMetric)
  private val rowMaskTime = metrics.getOrElse(GpuMetric.HYBRID_ROW_MASK_TIME, NoopMetric)
  private val statsFilterTime = metrics.getOrElse(GpuMetric.HYBRID_STATS_FILTER_TIME, NoopMetric)

  // Parallel IO configuration (now passed as constructor parameters)

  override def next(): Boolean = {
    if (!hasNextBatch) {
      return false
    }

    batch.foreach(_.close())
    batch = None

    try {
      val result = if (parallelIOEnabled) {
        readWithParallelHybridScan()
      } else if (pipeliningEnabled) {
        readWithPipelinedHybridScan()
      } else {
        readWithHybridScan()
      }
      // Always set hasNextBatch = false after reading - Hybrid Scan reads the entire file at once
      hasNextBatch = false
      if (result != null) {
        logInfo(s"[HYBRID DEBUG] next() returning batch with ${result.numRows()} rows")
        if (result.numRows() > 0) {
        batch = Some(result)
        numOutputBatches += 1
        numOutputRows += result.numRows()
        true
      } else {
          logInfo(s"[HYBRID DEBUG] Closing empty batch")
          result.close()
          false
        }
      } else {
        logInfo(s"[HYBRID DEBUG] next() returning null result")
        false
      }
    } catch {
      case e: Exception =>
        logWarning(s"cuDF Hybrid Scan failed for ${filePath}, falling back to regular scan", e)
        hasNextBatch = false
        false
    }
  }

  override def get(): ColumnarBatch = {
    batch.getOrElse {
      throw new NoSuchElementException("No batch available. Call next() first.")
    }
  }

  override def close(): Unit = {
    batch.foreach(_.close())
    batch = None
    if (hybridScanReader != null) {
      hybridScanReader.close()
      hybridScanReader = null
    }
    // Close compiled AST filter to release native resources
    compiledAstFilter.foreach(_.close())
    // Shutdown pipeline executor if used
    if (pipelineExecutor != null) {
      pipelineExecutor.shutdownNow()
    }
  }

  /**
   * Read the Parquet file footer bytes.
   */
  private def readFooterBytes(fs: FileSystem, 
                               path: Path, 
                               fileLength: Long): HostMemoryBuffer = {
    val FOOTER_LENGTH_SIZE = 4
    val MAGIC_SIZE = 4
    val FOOTER_SUFFIX_SIZE = FOOTER_LENGTH_SIZE + MAGIC_SIZE

    val in = fs.open(path)
    try {
      in.seek(fileLength - FOOTER_SUFFIX_SIZE)
      val suffixBytes = new Array[Byte](FOOTER_SUFFIX_SIZE)
      in.readFully(suffixBytes)

      val footerLength = ((suffixBytes(0) & 0xFF) |
        ((suffixBytes(1) & 0xFF) << 8) |
        ((suffixBytes(2) & 0xFF) << 16) |
        ((suffixBytes(3) & 0xFF) << 24))

      val footerStart = fileLength - footerLength - FOOTER_SUFFIX_SIZE
      val totalFooterSize = footerLength + FOOTER_SUFFIX_SIZE
      
      in.seek(footerStart)
      val footerBuffer = HostMemoryBuffer.allocate(totalFooterSize)
      try {
        val footerBytes = new Array[Byte](totalFooterSize.toInt)
        in.readFully(footerBytes)
        footerBuffer.setBytes(0, footerBytes, 0, totalFooterSize)
        footerBuffer
      } catch {
        case e: Exception =>
          footerBuffer.close()
          throw e
      }
    } finally {
      in.close()
    }
  }

  /**
   * Read the Parquet file using parallel hybrid scan with row group level IO concurrency.
   */
  private def readWithParallelHybridScan(): ColumnarBatch = {
    logInfo(s"cuDF Parallel Hybrid Scan reading ${filePath} with " +
      s"filter columns: [${filterColumns.mkString(", ")}], " +
      s"payload columns: [${payloadColumns.mkString(", ")}], " +
      s"parallelism: $numIOThreads threads, max $maxRowGroupsParallel row groups parallel")

    val fs = filePath.getFileSystem(conf)
    val fileStatus = fs.getFileStatus(filePath)

    // Acquire GPU semaphore
    GpuSemaphore.acquireIfNecessary(TaskContext.get())

    // Read footer bytes
    val footerBuffer = readTime.ns {
      readFooterBytes(fs, filePath, fileStatus.getLen)
    }

    try {
      withResource(footerBuffer) { footer =>
        // Create hybrid scan reader
        // Get AST filter handle before creating ParquetHybridScan
        val astFilterHandle = getAstFilterHandle
        hybridScanReader = new ParquetHybridScan(
          footer, 0, footer.getLength,
          filterColumns, payloadColumns,
          15, // DType.TIMESTAMP_MICROSECONDS.typeId (correct: 15, was incorrectly 10)
          astFilterHandle // AST filter handle for filter pushdown
        )

        // Get all row groups
        var rowGroupIndices = hybridScanReader.getAllRowGroups()
        if (rowGroupIndices.isEmpty) {
          return null
        }

        // Filter row groups with statistics
        // Note: filterRowGroupsWithStats requires a filter expression.
        // Skip this step if no filter expression is available.
        try {
          rowGroupIndices = statsFilterTime.ns {
            hybridScanReader.filterRowGroupsWithStats(rowGroupIndices)
          }
          if (rowGroupIndices.isEmpty) {
            logDebug("All row groups filtered out by statistics")
            return null
          }
        } catch {
          case e: Exception if e.getMessage != null &&
              e.getMessage.contains("empty converted filter expression") =>
            logWarning("Parallel scan: empty filter expression, skipping stats filtering")
            // Continue with all row groups if filter expression is empty
          case e: Exception =>
            logWarning(s"Parallel Hybrid scan failed during stats filtering: ${e.getMessage}")
            throw e
        }

        logDebug(s"After stats filtering: ${rowGroupIndices.length} row groups remaining")

        // Dynamic page index detection
        pageIndexInfo = Some(PageIndexDetector.detectPageIndex(hybridScanReader))
        effectiveUsePageIndex = PageIndexDetector.shouldUsePageIndex(
          usePageIndexConfig, pageIndexInfo.get)
        
        // Setup page index if enabled and detected
        if (effectiveUsePageIndex) {
          val range = pageIndexInfo.get.pageIndexByteRange
          if (range._2 > 0) {
            val pageIndexBuffer = readTime.ns {
              readByteRange(fs, filePath, range._1, range._2.toInt)
            }
            withResource(pageIndexBuffer) { pib =>
              hybridScanReader.setupPageIndex(pib, 0, pib.getLength)
            }
          }
        }

        // Get total rows in remaining row groups
        val totalRows = hybridScanReader.getTotalRowsInRowGroups(rowGroupIndices)
        if (totalRows == 0) {
          return null
        }

        // Build row mask using page index if available and enabled
        // Note: buildRowMaskWithPageIndex requires a filter expression.
        // Skip page index filtering if no filter expression is available.
        val rowMask = if (effectiveUsePageIndex) {
          try {
            rowMaskTime.ns {
              hybridScanReader.buildRowMaskWithPageIndex(rowGroupIndices)
            }
          } catch {
            case e: Exception if e.getMessage != null &&
                e.getMessage.contains("empty converted filter expression") =>
              logWarning("Parallel scan: empty filter, skipping page index filtering")
              // Fallback to all rows if filter expression is empty
              ai.rapids.cudf.ColumnVector.fromBooleans(
                Array.fill(totalRows.toInt)(true): _*)
            case e: Exception =>
              logWarning(s"Parallel scan page index filtering failed: ${e.getMessage}")
              throw e
          }
        } else {
          ai.rapids.cudf.ColumnVector.fromBooleans(
            Array.fill(totalRows.toInt)(true): _*)
        }

        try {
          // Parallel read row groups
          val rowGroupResults = readTime.ns {
            readRowGroupsInParallel(fs, rowGroupIndices)
          }

          try {
            // Process all row group results and materialize columns
            val combinedTable = processRowGroupResults(rowGroupResults, rowGroupIndices, rowMask)
            
            // Convert to ColumnarBatch - must close table after creating batch
            if (combinedTable != null) {
              withResource(combinedTable) { table =>
                // Debug: log column types
                val numCols = table.getNumberOfColumns
                val cudfTypes = (0 until numCols).map(i => table.getColumn(i).getType.toString)
                val sparkTypes = readSchema.fields.map(_.dataType.simpleString)
                logInfo(s"[DEBUG] cuDF table column types: ${cudfTypes.mkString(", ")}")
                logInfo(s"[DEBUG] Spark schema types: ${sparkTypes.mkString(", ")}")
                GpuColumnVector.from(table, readSchema.fields.map(_.dataType).toArray)
              }
            } else {
              null
            }
          } finally {
            rowGroupResults.foreach(_.close())
          }
        } finally {
          rowMask.close()
        }
      }
    } catch {
      case e: Exception =>
        logWarning(s"Parallel Hybrid scan failed: ${e.getMessage}", e)
        // Clean up the reader on exception
        if (hybridScanReader != null) {
          hybridScanReader.close()
          hybridScanReader = null
        }
        null
    }
  }

  /**
   * Read multiple row groups in parallel using the thread pool.
   */
  private def readRowGroupsInParallel(
      fs: FileSystem,
      rowGroupIndices: Array[Int]): Array[RowGroupReadResult] = {
    
    val executor = HybridScanThreadPool.getOrCreateThreadPool(numIOThreads)
    val futures = new ArrayBuffer[Future[RowGroupReadResult]]()
    val results = new ArrayBuffer[RowGroupReadResult]()
    
    // Limit concurrent row groups if configured
    val effectiveLimit = if (maxRowGroupsParallel > 0) {
      math.min(maxRowGroupsParallel, rowGroupIndices.length)
    } else {
      rowGroupIndices.length
    }

    try {
      // Test if filter expression is available by trying to get ranges for the first row group
      // If it fails, we need to recreate ParquetHybridScan with all columns as payload
      var useFallbackReader = false
      var fallbackReader: ParquetHybridScan = null
      
      if (rowGroupIndices.nonEmpty) {
        try {
          // Try to get ranges for the first row group to test if filter expression is available
          hybridScanReader.getFilterColumnChunkRanges(Array(rowGroupIndices(0)))
          hybridScanReader.getPayloadColumnChunkRanges(Array(rowGroupIndices(0)))
        } catch {
          case e: Exception if e.getMessage != null &&
              e.getMessage.contains("empty converted filter expression") =>
            logWarning("Parallel scan: recreating reader with all columns as payload")
            // Recreate ParquetHybridScan with all columns as payload columns
            val footerBuffer = readFooterBytes(fs, filePath, fs.getFileStatus(filePath).getLen)
            try {
              val allColumns = (filterColumns ++ payloadColumns).distinct
              hybridScanReader.close()
              fallbackReader = new ParquetHybridScan(
                footerBuffer, 0, footerBuffer.getLength,
                Array.empty[String], // No filter columns
                allColumns, // All columns as payload
                15, // DType.TIMESTAMP_MICROSECONDS.typeId (correct: 15, was incorrectly 10)
                0   // No AST filter for fallback reader
              )
              useFallbackReader = true
              hybridScanReader = fallbackReader
              logInfo(s"Fallback reader created with ${allColumns.length} payload columns")
            } finally {
              footerBuffer.close()
            }
          case e: Exception =>
            logWarning(s"Parallel Hybrid scan failed during range test: ${e.getMessage}")
            throw e
        }
      }
      
      // Submit tasks in batches to control memory usage
      var submitted = 0
      var completed = 0
      
      while (completed < rowGroupIndices.length) {
        // Submit more tasks up to the limit
        while (submitted < rowGroupIndices.length && 
               (submitted - completed) < effectiveLimit) {
          val rgIndex = rowGroupIndices(submitted)
          
          // Get ranges for this row group
          val (filterRanges, payloadRanges, numRows) = if (useFallbackReader) {
            // When using fallback reader, only get payload ranges (all columns are payload)
            try {
              val pr = hybridScanReader.getPayloadColumnChunkRanges(Array(rgIndex))
              val nr = hybridScanReader.getRowsInRowGroup(rgIndex)
              (None, pr, nr)
            } catch {
              case e: Exception =>
                logWarning(s"Parallel scan: fallback failed for rg $rgIndex: " +
                  e.getMessage)
                val nr = hybridScanReader.getRowsInRowGroup(rgIndex)
                (None, Array.empty[Long], nr) // Empty ranges
            }
          } else {
            // Normal path: get both filter and payload ranges
            try {
              val fr = hybridScanReader.getFilterColumnChunkRanges(Array(rgIndex))
              val pr = hybridScanReader.getPayloadColumnChunkRanges(Array(rgIndex))
              val nr = hybridScanReader.getRowsInRowGroup(rgIndex)
              (Some(fr), pr, nr)
            } catch {
              case e: Exception if e.getMessage != null &&
                  e.getMessage.contains("empty converted filter expression") =>
                logWarning(s"Parallel scan: empty filter for rg $rgIndex, using fallback")
                // Fallback: try to get payload ranges only
                try {
                  val pr = hybridScanReader.getPayloadColumnChunkRanges(Array(rgIndex))
                  val nr = hybridScanReader.getRowsInRowGroup(rgIndex)
                  (None, pr, nr)
                } catch {
                  case e2: Exception =>
                    logWarning(s"Parallel scan: no payload ranges: ${e2.getMessage}")
                    val nr = hybridScanReader.getRowsInRowGroup(rgIndex)
                    (None, Array.empty[Long], nr) // Empty ranges
                }
              case e: Exception =>
                logWarning(s"Parallel scan getColumnChunkRanges failed: ${e.getMessage}")
                throw e
            }
          }
          
          val task = new RowGroupIOTask(fs, filePath, rgIndex, 
            filterRanges.getOrElse(Array.empty[Long]), payloadRanges, numRows)
          futures += executor.submit(task)
          submitted += 1
        }
        
        // Wait for at least one task to complete before submitting more
        if (futures.nonEmpty) {
          val completedFuture = futures.head
          results += completedFuture.get()
          futures.remove(0)
          completed += 1
        }
      }
      
      // Wait for remaining futures
      futures.foreach { f =>
        results += f.get()
      }
      
      // Sort results by row group index to maintain order
      results.sortBy(_.rowGroupIndex).toArray
    } catch {
      case e: Exception =>
        // Cancel pending futures
        futures.foreach(_.cancel(true))
        // Close completed results
        results.foreach(_.close())
        throw e
    }
  }

  /**
   * Process row group read results and materialize columns using GPU.
   */
  private def processRowGroupResults(
      results: Array[RowGroupReadResult],
      rowGroupIndices: Array[Int],
      rowMask: ColumnVector): Table = {
    
    if (results.isEmpty) {
      return null
    }

    // Combine all filter buffers (host memory)
    val allFilterBuffers = results.flatMap(_.filterBuffers)
    val allPayloadBuffers = results.flatMap(_.payloadBuffers)

    try {
      // Handle case where filter expression is empty (no filter columns)
      if (allFilterBuffers.isEmpty) {
        // Fallback: read all columns as payload columns only
        logDebug("No filter buffers available, reading all columns as payload columns")
        // Pass host buffer addresses - JNI will copy to device
        val (payloadAddrs, payloadSizes) = getHostBufferAddrsAndSizes(allPayloadBuffers)
          val payloadTable = payloadColTime.ns {
            hybridScanReader.materializePayloadColumns(
              rowGroupIndices, payloadAddrs, payloadSizes,
              rowMask.getNativeView, effectiveUsePageIndex)
          }
          payloadTable
      } else {
        // Normal path: read filter and payload columns separately
        // Pass host buffer addresses - JNI will copy to device (cuDF manages GPU memory)
        val (filterAddrs, filterSizes) = getHostBufferAddrsAndSizes(allFilterBuffers)
          val filterTable = filterColTime.ns {
            hybridScanReader.materializeFilterColumns(
              rowGroupIndices, filterAddrs, filterSizes,
              rowMask.getNativeView, effectiveUsePageIndex)
          }
          
          try {
          // Materialize payload columns - pass host addresses
          val (payloadAddrs, payloadSizes) = getHostBufferAddrsAndSizes(allPayloadBuffers)
            val payloadTable = payloadColTime.ns {
              hybridScanReader.materializePayloadColumns(
                rowGroupIndices, payloadAddrs, payloadSizes,
                rowMask.getNativeView, effectiveUsePageIndex)
            }

          // Combine tables - this will close filterTable and payloadTable
            combineFilterAndPayloadTables(filterTable, payloadTable)
    } catch {
      case e: Exception =>
            // Clean up filterTable if payload processing fails
            filterTable.close()
        throw e
    }
  }
        } catch {
          case e: Exception =>
        logWarning(s"Failed to process row group results: ${e.getMessage}", e)
        throw e
    }
  }

  /**
   * Read the Parquet file using sequential hybrid scan with optimization checks.
   * Includes selectivity estimation and dynamic page index detection.
   */
  private def readWithHybridScan(): ColumnarBatch = {
    logInfo(s"cuDF Hybrid Scan reading ${filePath} with " +
      s"filter columns: [${filterColumns.mkString(", ")}], " +
      s"payload columns: [${payloadColumns.mkString(", ")}]")

    val fs = filePath.getFileSystem(conf)
    val fileStatus = fs.getFileStatus(filePath)

    GpuSemaphore.acquireIfNecessary(TaskContext.get())

    val footerBuffer = readTime.ns {
      readFooterBytes(fs, filePath, fileStatus.getLen)
    }

    try {
      withResource(footerBuffer) { footer =>
        hybridScanReader = new ParquetHybridScan(
          footer, 0, footer.getLength,
          filterColumns, payloadColumns,
          15, // DType.TIMESTAMP_MICROSECONDS.typeId (correct: 15, was incorrectly 10)
          getAstFilterHandle // AST filter handle for filter pushdown
        )

        var rowGroupIndices = hybridScanReader.getAllRowGroups()
        if (rowGroupIndices.isEmpty) {
          return null
        }

        // Get total rows before filtering for selectivity estimation
        val totalRowsBeforeFilter = hybridScanReader.getTotalRowsInRowGroups(rowGroupIndices)
        
        // Filter row groups with statistics. Skip if no filter expression.
        var rowGroupsFilteredByStats = 0
        try {
          val filteredIndices = statsFilterTime.ns {
            hybridScanReader.filterRowGroupsWithStats(rowGroupIndices)
          }
          rowGroupsFilteredByStats = rowGroupIndices.length - filteredIndices.length
          rowGroupIndices = filteredIndices
          if (rowGroupIndices.isEmpty) {
            logDebug("All row groups filtered out by statistics")
            return null
          }
        } catch {
          case e: Exception if e.getMessage != null &&
              e.getMessage.contains("empty converted filter expression") =>
            logWarning("Hybrid scan: empty filter, skipping stats filtering")
            // Continue with all row groups if filter expression is empty
          case e: Exception =>
            logWarning(s"Hybrid scan failed during stats filtering: ${e.getMessage}")
            throw e
        }

        // Estimate selectivity based on row group pruning results
        val totalRowsAfterFilter = hybridScanReader.getTotalRowsInRowGroups(rowGroupIndices)
        estimatedSelectivity = if (totalRowsBeforeFilter > 0) {
          totalRowsAfterFilter.toDouble / totalRowsBeforeFilter
        } else {
          1.0
        }
        
        // Check if Hybrid Scan is beneficial based on selectivity
        hybridScanBeneficial = CudfHybridScanUtils.isHybridScanBeneficial(
          estimatedSelectivity,
          filterColumns.length,
          payloadColumns.length,
          selectivityThreshold)
        
        logInfo(f"Selectivity estimation: $estimatedSelectivity%.4f " +
          f"(${rowGroupsFilteredByStats}/${rowGroupIndices.length + rowGroupsFilteredByStats} " +
          s"row groups pruned), beneficial=$hybridScanBeneficial")
        
        // Note: We continue even if not beneficial since we've already started
        // In future, this could trigger fallback to regular scan

        logDebug(s"After stats filtering: ${rowGroupIndices.length} row groups remaining")

        // Dynamic page index detection
        pageIndexInfo = Some(PageIndexDetector.detectPageIndex(hybridScanReader))
        effectiveUsePageIndex = PageIndexDetector.shouldUsePageIndex(
          usePageIndexConfig, pageIndexInfo.get)
        
        if (effectiveUsePageIndex) {
          val range = pageIndexInfo.get.pageIndexByteRange
          if (range._2 > 0) {
            val pageIndexBuffer = readTime.ns {
              readByteRange(fs, filePath, range._1, range._2.toInt)
            }
            withResource(pageIndexBuffer) { pib =>
              hybridScanReader.setupPageIndex(pib, 0, pib.getLength)
            }
          }
        }

        val totalRows = totalRowsAfterFilter
        if (totalRows == 0) {
          return null
        }

        // Build row mask using page index if available and enabled.
        // Skip page index filtering if no filter expression.
        val rowMask = if (effectiveUsePageIndex) {
          try {
            rowMaskTime.ns {
              hybridScanReader.buildRowMaskWithPageIndex(rowGroupIndices)
            }
          } catch {
            case e: Exception if e.getMessage != null &&
                e.getMessage.contains("empty converted filter expression") =>
              logWarning("Hybrid scan: empty filter, skipping page index filtering")
              // Fallback to all rows if filter expression is empty
              ai.rapids.cudf.ColumnVector.fromBooleans(
                Array.fill(totalRows.toInt)(true): _*)
            case e: Exception =>
              logWarning(s"Hybrid scan failed during page index filtering: ${e.getMessage}")
              throw e
          }
        } else {
          ai.rapids.cudf.ColumnVector.fromBooleans(
            Array.fill(totalRows.toInt)(true): _*)
        }

        try {
          // Get filter column chunk ranges. Fall back to all-payload if empty.
          val filterRangesOpt = try {
            Some(hybridScanReader.getFilterColumnChunkRanges(rowGroupIndices))
          } catch {
            case e: Exception if e.getMessage != null &&
                e.getMessage.contains("empty converted filter expression") =>
              logWarning("Hybrid scan: empty filter, reading all columns as payload")
              // Fallback: read all columns as payload columns
              // Create a new ParquetHybridScan with all columns as payload columns
              val allColumns = (filterColumns ++ payloadColumns).distinct
              hybridScanReader.close()
              try {
                hybridScanReader = new ParquetHybridScan(
                  footer, 0, footer.getLength,
                  Array.empty[String], // No filter columns
                  allColumns, // All columns as payload
                  15, // DType.TIMESTAMP_MICROSECONDS.typeId (correct: 15, was incorrectly 10)
                  0 // No AST filter for fallback reader
                )
              } catch {
                case e: Exception =>
                  logWarning(s"Hybrid scan fallback failed to create new reader: ${e.getMessage}")
                  throw e
              }
              // Re-get row groups after recreating reader
              val fallbackRowGroupIndices = hybridScanReader.getAllRowGroups()
              if (fallbackRowGroupIndices.isEmpty) {
                return null
              }
              // Skip filter column reading, read all columns as payload
              // Try to get payload ranges - may fail if cuDF requires filter expression
              val payloadRanges = try {
                hybridScanReader.getPayloadColumnChunkRanges(fallbackRowGroupIndices)
              } catch {
                case e: Exception if e.getMessage != null &&
                    e.getMessage.contains("empty converted filter expression") =>
                  logWarning("Hybrid scan fallback: payload ranges failed, using regular read")
                  // Cannot use hybrid scan at all, fall back to regular reader
                  return null
                case e: Exception =>
                  logWarning(s"Hybrid scan fallback failed: ${e.getMessage}")
                  throw e
              }
              
              val payloadHostBuffers = readTime.ns {
                readColumnChunksToHost(fs, filePath, payloadRanges)
              }
              try {
                val (payloadAddrs, payloadSizes) = getHostBufferAddrsAndSizes(payloadHostBuffers)
              val payloadTable = payloadColTime.ns {
                hybridScanReader.materializePayloadColumns(
                  fallbackRowGroupIndices, payloadAddrs, payloadSizes,
                  rowMask.getNativeView, effectiveUsePageIndex)
              }
                // Must close table after creating batch
                return withResource(payloadTable) { table =>
                  GpuColumnVector.from(table, readSchema.fields.map(_.dataType).toArray)
                }
              } finally {
                payloadHostBuffers.foreach(_.close())
              }
            case e: Exception =>
              logWarning(s"Hybrid scan failed during getFilterColumnChunkRanges: ${e.getMessage}")
              throw e
          }
          
          // Normal path: read filter and payload columns separately
          val filterRanges = filterRangesOpt.get
          val filterHostBuffers = readTime.ns {
            readColumnChunksToHost(fs, filePath, filterRanges)
          }

          try {
            val (filterAddrs, filterSizes) = getHostBufferAddrsAndSizes(filterHostBuffers)
          val filterTable = filterColTime.ns {
            hybridScanReader.materializeFilterColumns(
              rowGroupIndices, filterAddrs, filterSizes,
              rowMask.getNativeView, effectiveUsePageIndex)
          }

          // Try to get payload ranges - may fail if cuDF requires filter expression
          val payloadRanges = try {
            hybridScanReader.getPayloadColumnChunkRanges(rowGroupIndices)
          } catch {
            case e: Exception if e.getMessage != null &&
                e.getMessage.contains("empty converted filter expression") =>
              logWarning("Hybrid scan: payload ranges failed after filter, returning null")
              filterTable.close()
              return null
            case e: Exception =>
              filterTable.close()
              throw e
          }
          
            try {
              val payloadHostBuffers = readTime.ns {
                readColumnChunksToHost(fs, filePath, payloadRanges)
          }

              try {
                val (payloadAddrs, payloadSizes) = getHostBufferAddrsAndSizes(payloadHostBuffers)
          val payloadTable = payloadColTime.ns {
            hybridScanReader.materializePayloadColumns(
              rowGroupIndices, payloadAddrs, payloadSizes,
              rowMask.getNativeView, effectiveUsePageIndex)
          }

                // combineFilterAndPayloadTables will close both tables
          logInfo(s"[HYBRID] filter: ${filterTable.getRowCount} rows, " +
            s"${filterTable.getNumberOfColumns} cols")
          logInfo(s"[HYBRID] payload: ${payloadTable.getRowCount} rows, " +
            s"${payloadTable.getNumberOfColumns} cols")
          val combinedTable = combineFilterAndPayloadTables(filterTable, payloadTable)
          
                // Must close table after creating batch
                withResource(combinedTable) { table =>
                  logInfo(s"[HYBRID] combined: ${table.getRowCount} rows, " +
                    s"${table.getNumberOfColumns} cols")
                  GpuColumnVector.from(table, readSchema.fields.map(_.dataType).toArray)
                }
              } finally {
                payloadHostBuffers.foreach(_.close())
              }
            } catch {
              case e: Exception =>
                // Clean up filterTable if payload processing fails
                filterTable.close()
                throw e
            }
          } finally {
            filterHostBuffers.foreach(_.close())
          }
        } finally {
          rowMask.close()
        }
      }
    } catch {
      case e: Exception =>
        logWarning(s"Hybrid scan failed: ${e.getMessage}", e)
        // Clean up the reader on exception
        if (hybridScanReader != null) {
          hybridScanReader.close()
          hybridScanReader = null
        }
        null
    }
  }

  /**
   * Read the Parquet file using pipelined hybrid scan.
   * This overlaps filter column I/O with payload column I/O to improve throughput.
   * 
   * Pipeline stages:
   * 1. Read footer and setup → 2. Filter I/O (async) → 3. Payload I/O (async, overlapped)
   *                         → 4. Filter GPU materialize → 5. Payload GPU materialize
   */
  private def readWithPipelinedHybridScan(): ColumnarBatch = {
    logInfo(s"cuDF Pipelined Hybrid Scan reading ${filePath} with " +
      s"filter columns: [${filterColumns.mkString(", ")}], " +
      s"payload columns: [${payloadColumns.mkString(", ")}]")

    val fs = filePath.getFileSystem(conf)
    val fileStatus = fs.getFileStatus(filePath)

    GpuSemaphore.acquireIfNecessary(TaskContext.get())

    val footerBuffer = readTime.ns {
      readFooterBytes(fs, filePath, fileStatus.getLen)
    }

    try {
      withResource(footerBuffer) { footer =>
        hybridScanReader = new ParquetHybridScan(
          footer, 0, footer.getLength,
          filterColumns, payloadColumns,
          15, // DType.TIMESTAMP_MICROSECONDS.typeId
          getAstFilterHandle
        )

        var rowGroupIndices = hybridScanReader.getAllRowGroups()
        if (rowGroupIndices.isEmpty) {
          return null
        }

        // Get total rows before filtering for selectivity estimation
        val totalRowsBeforeFilter = hybridScanReader.getTotalRowsInRowGroups(rowGroupIndices)
        
        // Filter row groups with statistics
        var rowGroupsFilteredByStats = 0
        try {
          val filteredIndices = statsFilterTime.ns {
            hybridScanReader.filterRowGroupsWithStats(rowGroupIndices)
          }
          rowGroupsFilteredByStats = rowGroupIndices.length - filteredIndices.length
          rowGroupIndices = filteredIndices
          if (rowGroupIndices.isEmpty) {
            logDebug("All row groups filtered out by statistics")
            return null
          }
        } catch {
          case e: Exception if e.getMessage != null &&
              e.getMessage.contains("empty converted filter expression") =>
            logWarning("Pipelined scan: empty filter, skipping stats filtering")
          case e: Exception =>
            throw e
        }

        // Estimate selectivity
        val totalRowsAfterFilter = hybridScanReader.getTotalRowsInRowGroups(rowGroupIndices)
        estimatedSelectivity = if (totalRowsBeforeFilter > 0) {
          totalRowsAfterFilter.toDouble / totalRowsBeforeFilter
        } else {
          1.0
        }
        
        logInfo(f"Pipelined scan: selectivity=$estimatedSelectivity%.4f, " +
          s"${rowGroupsFilteredByStats} row groups pruned")

        // Dynamic page index detection
        pageIndexInfo = Some(PageIndexDetector.detectPageIndex(hybridScanReader))
        effectiveUsePageIndex = PageIndexDetector.shouldUsePageIndex(
          usePageIndexConfig, pageIndexInfo.get)
        
        if (effectiveUsePageIndex) {
          val range = pageIndexInfo.get.pageIndexByteRange
          if (range._2 > 0) {
            val pageIndexBuffer = readTime.ns {
              readByteRange(fs, filePath, range._1, range._2.toInt)
            }
            withResource(pageIndexBuffer) { pib =>
              hybridScanReader.setupPageIndex(pib, 0, pib.getLength)
            }
          }
        }

        val totalRows = totalRowsAfterFilter
        if (totalRows == 0) {
          return null
        }

        // Build row mask
        val rowMask = if (effectiveUsePageIndex) {
          try {
            rowMaskTime.ns {
              hybridScanReader.buildRowMaskWithPageIndex(rowGroupIndices)
            }
          } catch {
            case e: Exception if e.getMessage != null &&
                e.getMessage.contains("empty converted filter expression") =>
              ai.rapids.cudf.ColumnVector.fromBooleans(
                Array.fill(totalRows.toInt)(true): _*)
            case e: Exception =>
              throw e
          }
        } else {
          ai.rapids.cudf.ColumnVector.fromBooleans(
            Array.fill(totalRows.toInt)(true): _*)
        }

        try {
          // Get byte ranges for filter and payload columns
          val filterRanges = try {
            hybridScanReader.getFilterColumnChunkRanges(rowGroupIndices)
          } catch {
            case e: Exception if e.getMessage != null &&
                e.getMessage.contains("empty converted filter expression") =>
              Array.empty[Long]
            case e: Exception =>
              throw e
          }
          
          val payloadRanges = hybridScanReader.getPayloadColumnChunkRanges(rowGroupIndices)

          // PIPELINING: Submit both I/O operations concurrently
          val filterIOFuture: Future[Array[HostMemoryBuffer]] = if (filterRanges.nonEmpty) {
            pipelineExecutor.submit(new Callable[Array[HostMemoryBuffer]] {
              override def call(): Array[HostMemoryBuffer] = {
                readTime.ns {
                  readColumnChunksToHost(fs, filePath, filterRanges)
                }
              }
            })
          } else {
            null
          }

          val payloadIOFuture: Future[Array[HostMemoryBuffer]] = 
            pipelineExecutor.submit(new Callable[Array[HostMemoryBuffer]] {
              override def call(): Array[HostMemoryBuffer] = {
                readTime.ns {
                  readColumnChunksToHost(fs, filePath, payloadRanges)
                }
              }
            })

          // Wait for filter I/O and materialize on GPU
          val filterTable = if (filterIOFuture != null) {
            val filterHostBuffers = filterIOFuture.get()
            try {
              val (filterAddrs, filterSizes) = getHostBufferAddrsAndSizes(filterHostBuffers)
              val table = filterColTime.ns {
                hybridScanReader.materializeFilterColumns(
                  rowGroupIndices, filterAddrs, filterSizes,
                  rowMask.getNativeView, effectiveUsePageIndex)
              }
              table
            } finally {
              filterHostBuffers.foreach(_.close())
            }
          } else {
            null
          }

          try {
            // Wait for payload I/O and materialize on GPU
            val payloadHostBuffers = payloadIOFuture.get()
            try {
              val (payloadAddrs, payloadSizes) = getHostBufferAddrsAndSizes(payloadHostBuffers)
              val payloadTable = payloadColTime.ns {
                hybridScanReader.materializePayloadColumns(
                  rowGroupIndices, payloadAddrs, payloadSizes,
                  rowMask.getNativeView, effectiveUsePageIndex)
              }

              // Combine tables
              val combinedTable = if (filterTable != null) {
                combineFilterAndPayloadTables(filterTable, payloadTable)
              } else {
                payloadTable
              }

              withResource(combinedTable) { table =>
                logInfo(s"[PIPELINED] Result: ${table.getRowCount} rows, " +
                  s"${table.getNumberOfColumns} cols")
                GpuColumnVector.from(table, readSchema.fields.map(_.dataType).toArray)
              }
            } finally {
              payloadHostBuffers.foreach(_.close())
            }
          } catch {
            case e: Exception =>
              if (filterTable != null) filterTable.close()
              throw e
          }
        } finally {
          rowMask.close()
        }
      }
    } catch {
      case e: Exception =>
        logWarning(s"Pipelined Hybrid scan failed: ${e.getMessage}", e)
        if (hybridScanReader != null) {
          hybridScanReader.close()
          hybridScanReader = null
        }
        null
    }
  }

  private def readByteRange(fs: FileSystem, 
                            path: Path, 
                            offset: Long, 
                            length: Int): HostMemoryBuffer = {
    val in = fs.open(path)
    try {
      in.seek(offset)
      val buffer = HostMemoryBuffer.allocate(length)
      try {
        val bytes = new Array[Byte](length)
        in.readFully(bytes)
        buffer.setBytes(0, bytes, 0, length)
        buffer
      } catch {
        case e: Exception =>
          buffer.close()
          throw e
      }
    } finally {
      in.close()
    }
  }

  /**
   * Read column chunks from file into HOST memory buffers.
   * The JNI layer will copy these to device memory managed by cuDF/RMM.
   * This approach avoids memory ownership conflicts between Java and cuDF.
   */
  private def readColumnChunksToHost(fs: FileSystem,
                                path: Path,
                                      ranges: Array[Long]): Array[HostMemoryBuffer] = {
    val buffers = new ArrayBuffer[HostMemoryBuffer]()
    val in = fs.open(path)
    try {
      var i = 0
      while (i < ranges.length) {
        val offset = ranges(i)
        val size = ranges(i + 1).toInt
        
        in.seek(offset)
        val hostBytes = new Array[Byte](size)
        in.readFully(hostBytes)
        
        val hostBuffer = HostMemoryBuffer.allocate(size)
        try {
          hostBuffer.setBytes(0, hostBytes, 0, size)
          buffers += hostBuffer
          } catch {
            case e: Exception =>
            hostBuffer.close()
              throw e
        }
        
        i += 2
      }
      buffers.toArray
    } catch {
      case e: Exception =>
        buffers.foreach(_.close())
        throw e
    } finally {
      in.close()
    }
  }

  private def getHostBufferAddrsAndSizes(
      buffers: Array[HostMemoryBuffer]): (Array[Long], Array[Long]) = {
    val addrs = buffers.map(_.getAddress)
    val sizes = buffers.map(_.getLength)
    (addrs, sizes)
  }

  /**
   * Combine filter and payload tables into a single table with columns ordered
   * according to readSchema.
   * 
   * filterTable columns are ordered by filterColumns array
   * payloadTable columns are ordered by payloadColumns array
   * The combined table must have columns in readSchema order
   */
  private def combineFilterAndPayloadTables(filterTable: Table, 
                                             payloadTable: Table): Table = {
    if (filterTable == null && payloadTable == null) {
      return null
    }
    if (filterTable == null) {
      return payloadTable
    }
    if (payloadTable == null) {
      return filterTable
    }

    // Build a map from column name to (source table, column index)
    val columnMap = scala.collection.mutable.Map[String, (Table, Int)]()
    
    // filterColumns are in schema order, so filterTable columns are also in schema order
    filterColumns.zipWithIndex.foreach { case (name, idx) =>
      columnMap(name) = (filterTable, idx)
    }
    
    // payloadColumns are in schema order, so payloadTable columns are also in schema order
    payloadColumns.zipWithIndex.foreach { case (name, idx) =>
      columnMap(name) = (payloadTable, idx)
    }

    val allColumns = new ArrayBuffer[ColumnVector]()

    try {
      // Extract columns in readSchema order
      readSchema.fieldNames.foreach { fieldName =>
        columnMap.get(fieldName) match {
          case Some((table, idx)) =>
            allColumns += table.getColumn(idx).incRefCount()
          case None =>
            throw new IllegalStateException(
              s"Column $fieldName not found in filter or payload tables")
        }
      }

      // Create the combined table - this takes ownership of the columns
      new Table(allColumns.toArray: _*)
    } catch {
      case e: Exception =>
        // Clean up columns we've already extracted on error
        allColumns.foreach(_.close())
        throw e
    } finally {
      // Always close the input tables
      filterTable.close()
      payloadTable.close()
    }
  }
}

/**
 * Factory for creating CudfHybridScanPartitionReader instances.
 */
object CudfHybridScanPartitionReaderFactory extends Logging {

  /**
   * Create a hybrid scan partition reader if conditions are met.
   * 
   * @return Some(reader) if hybrid scan should be used, None otherwise
   */
  def createReaderIfApplicable(
      conf: Configuration,
      split: PartitionedFile,
      filePath: Path,
      readSchema: StructType,
      dataFilters: Seq[Expression],
      rapidsConf: RapidsConf,
      metrics: Map[String, GpuMetric]): Option[CudfHybridScanPartitionReader] = {

    if (!CudfHybridScanUtils.shouldUseCudfHybridScan(rapidsConf, dataFilters, readSchema)) {
      return None
    }

    val (filterColumns, payloadColumns) = 
      CudfHybridScanUtils.separateColumns(readSchema, dataFilters)

    if (filterColumns.isEmpty || payloadColumns.isEmpty) {
      logDebug("cuDF Hybrid Scan not applicable: need both filter and payload columns")
      return None
    }

    Some(new CudfHybridScanPartitionReader(
      conf,
      split,
      filePath,
      readSchema,
      filterColumns,
      payloadColumns,
      rapidsConf.cudfHybridScanUsePageIndex,
      rapidsConf.cudfHybridScanUseBloomFilter,
      rapidsConf.cudfHybridScanParallelIOEnabled,
      rapidsConf.cudfHybridScanParallelIONumThreads,
      rapidsConf.cudfHybridScanMaxRowGroupsParallel,
      rapidsConf.cudfHybridScanSelectivityThreshold,  // Selectivity threshold for benefit check
      rapidsConf.cudfHybridScanPipeliningEnabled,     // Pipelining for I/O overlap
      metrics,
      dataFilters  // Pass dataFilters for filter expression support
    ))
  }
}
