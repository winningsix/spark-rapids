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
import org.apache.spark.sql.catalyst.expressions.Expression
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
 * with row group level parallel IO.
 * 
 * Hybrid Scan is optimized for highly selective filters. It reads the file in two passes:
 * 1. First pass: Read only filter columns and build a row mask
 * 2. Second pass: Read only payload columns using the row mask to skip unnecessary data
 * 
 * With parallel IO enabled, multiple row groups can be read concurrently using
 * a thread pool, significantly improving IO throughput for files with many row groups.
 */
@nowarn("msg=never used")
class CudfHybridScanPartitionReader(
    conf: Configuration,
    split: PartitionedFile,
    filePath: Path,
    readSchema: StructType,
    filterColumns: Array[String],
    payloadColumns: Array[String],
    usePageIndex: Boolean,
    useBloomFilter: Boolean,
    parallelIOEnabled: Boolean,
    numIOThreads: Int,
    maxRowGroupsParallel: Int,
    metrics: Map[String, GpuMetric],
    dataFilters: Seq[Expression] = Seq.empty) extends PartitionReader[ColumnarBatch] with Logging {

  private var batch: Option[ColumnarBatch] = None
  private var hasNextBatch = true
  private var hybridScanReader: ParquetHybridScan = _
  
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

        // Setup page index if enabled
        if (usePageIndex) {
          val pageIndexRange = hybridScanReader.getPageIndexByteRange()
          if (pageIndexRange(1) > 0) {
            val pageIndexBuffer = readTime.ns {
              readByteRange(fs, filePath, pageIndexRange(0), pageIndexRange(1).toInt)
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

        // Build row mask using page index if available
        // Note: buildRowMaskWithPageIndex requires a filter expression.
        // Skip page index filtering if no filter expression is available.
        val rowMask = if (usePageIndex) {
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
              rowMask.getNativeView, usePageIndex)
          }
          payloadTable
      } else {
        // Normal path: read filter and payload columns separately
        // Pass host buffer addresses - JNI will copy to device (cuDF manages GPU memory)
        val (filterAddrs, filterSizes) = getHostBufferAddrsAndSizes(allFilterBuffers)
          val filterTable = filterColTime.ns {
            hybridScanReader.materializeFilterColumns(
              rowGroupIndices, filterAddrs, filterSizes,
              rowMask.getNativeView, usePageIndex)
          }
          
          try {
          // Materialize payload columns - pass host addresses
          val (payloadAddrs, payloadSizes) = getHostBufferAddrsAndSizes(allPayloadBuffers)
            val payloadTable = payloadColTime.ns {
              hybridScanReader.materializePayloadColumns(
                rowGroupIndices, payloadAddrs, payloadSizes,
                rowMask.getNativeView, usePageIndex)
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
   * Read the Parquet file using sequential hybrid scan (original implementation).
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

        // Filter row groups with statistics. Skip if no filter expression.
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
            logWarning("Hybrid scan: empty filter, skipping stats filtering")
            // Continue with all row groups if filter expression is empty
          case e: Exception =>
            logWarning(s"Hybrid scan failed during stats filtering: ${e.getMessage}")
            throw e
        }

        logDebug(s"After stats filtering: ${rowGroupIndices.length} row groups remaining")

        if (usePageIndex) {
          val pageIndexRange = hybridScanReader.getPageIndexByteRange()
          if (pageIndexRange(1) > 0) {
            val pageIndexBuffer = readTime.ns {
              readByteRange(fs, filePath, pageIndexRange(0), pageIndexRange(1).toInt)
            }
            withResource(pageIndexBuffer) { pib =>
              hybridScanReader.setupPageIndex(pib, 0, pib.getLength)
            }
          }
        }

        val totalRows = hybridScanReader.getTotalRowsInRowGroups(rowGroupIndices)
        if (totalRows == 0) {
          return null
        }

        // Build row mask using page index if available.
        // Skip page index filtering if no filter expression.
        val rowMask = if (usePageIndex) {
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
                  rowMask.getNativeView, usePageIndex)
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
              rowMask.getNativeView, usePageIndex)
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
              rowMask.getNativeView, usePageIndex)
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
      metrics,
      dataFilters  // Pass dataFilters for filter expression support
    ))
  }
}
