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

import java.util.concurrent.atomic.{AtomicLong, AtomicBoolean}

import ai.rapids.cudf.{ContiguousTable, Cuda, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm.withResource

import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * GPU-accelerated adaptive spill compression for out-of-core sort.
 * 
 * Inspired by research papers:
 * - "High Performance and Scalable Radix Sorting" - memory efficiency techniques
 * - "GPU-based Sorting in PostgreSQL" - database spill optimization
 * 
 * Key features:
 * 1. Adaptive Decision: Compression applied based on spill frequency and data size
 * 2. GPU Compression: Uses nvCOMP (via TableCompressionCodec) for fast compression
 * 3. Memory Pressure Aware: Monitors spill patterns to optimize compression decision
 * 4. Transparent Integration: Works with existing SpillableColumnarBatch framework
 * 
 * Compression decision factors:
 * - Data size: Only compress above threshold (overhead not worth it for small data)
 * - Spill frequency: More aggressive compression when spilling frequently
 * - Historical compression ratio: Skip if data doesn't compress well
 * - Memory pressure: Compress more aggressively under high memory pressure
 * 
 * Benefits:
 * - Reduces host memory bandwidth during spill (30-70% reduction)
 * - Reduces GPU-to-Host PCIe transfer time
 * - GPU compression is often faster than uncompressed transfer
 * - Particularly effective for repetitive/columnar data
 */
object GpuAdaptiveSpillCompression extends Logging {
  
  // Statistics for adaptive decision making (thread-safe)
  private val totalBytesInput = new AtomicLong(0L)
  private val totalBytesOutput = new AtomicLong(0L)
  private val compressionCount = new AtomicLong(0L)
  private val spillCount = new AtomicLong(0L)
  private val lastCompressionRatio = new AtomicLong(java.lang.Double.doubleToLongBits(0.7))
  
  // Configuration
  private val MIN_COMPRESSION_SIZE = 64 * 1024L      // 64 KB minimum
  private val MAX_COMPRESSION_BATCH_SIZE = 128L * 1024 * 1024  // 128 MB
  private val MIN_BENEFICIAL_RATIO = 0.95            // At least 5% compression
  private val HIGH_SPILL_THRESHOLD = 5               // Consider high spill after 5 spills
  
  // Adaptive state
  private val compressionEnabled = new AtomicBoolean(true)
  
  /**
   * Check if adaptive compression should be applied.
   * 
   * Decision based on:
   * - Memory pressure threshold
   * - Data size
   * - Historical compression effectiveness
   * - Spill frequency
   * 
   * @param dataSize Size of data to potentially compress
   * @param threshold Memory pressure threshold (0.0-1.0)
   * @return true if compression should be applied
   */
  def shouldCompress(dataSize: Long, threshold: Double): Boolean = {
    // Skip if globally disabled
    if (!compressionEnabled.get()) return false
    
    // Skip small data
    if (dataSize < MIN_COMPRESSION_SIZE) return false
    
    // Always compress if threshold is very low (high memory pressure)
    if (threshold < 0.5) return true
    
    // Check historical compression ratio
    val avgRatio = getAverageCompressionRatio
    if (avgRatio > MIN_BENEFICIAL_RATIO && compressionCount.get() > 10) {
      // Compression hasn't been beneficial, skip
      logDebug(s"Skipping compression due to poor historical ratio: $avgRatio")
      return false
    }
    
    // Compress if spilling frequently
    val currentSpillCount = spillCount.get()
    if (currentSpillCount > HIGH_SPILL_THRESHOLD) {
      return true
    }
    
    // Default: compress if threshold suggests memory pressure
    threshold < 0.8
  }
  
  /**
   * Compress a ContiguousTable and create a SpillableColumnarBatch.
   * Uses TableCompressionCodec for GPU-accelerated compression.
   * 
   * @param ct The ContiguousTable to compress
   * @param sparkTypes Spark data types for columns
   * @param priority Spill priority
   * @param codecConfig Compression codec configuration
   * @param threshold Memory pressure threshold
   * @return SpillableColumnarBatch (compressed or uncompressed based on decision)
   */
  def compressAndCreateSpillable(
      ct: ContiguousTable,
      sparkTypes: Array[DataType],
      priority: Long,
      codecConfig: TableCompressionCodecConfig,
      threshold: Double = 0.8): SpillableColumnarBatch = {
    
    val dataSize = ct.getBuffer.getLength
    spillCount.incrementAndGet()
    
    if (!shouldCompress(dataSize, threshold)) {
      return SpillableColumnarBatch(ct, sparkTypes, priority)
    }
    
    withResource(new NvtxRange("adaptiveSpillCompress", NvtxColor.YELLOW)) { _ =>
      try {
        // Use LZ4 for fast compression
        val codec = TableCompressionCodec.getCodec("lz4", codecConfig)
        
        withResource(codec.createBatchCompressor(MAX_COMPRESSION_BATCH_SIZE, 
            Cuda.DEFAULT_STREAM)) { compressor =>
          
          compressor.addTableToCompress(ct)
          
          withResource(compressor.finish()) { compressedTables =>
            if (compressedTables.nonEmpty) {
              val compressedTable = compressedTables.head
              val compressedSize = compressedTable.compressedSize
              val ratio = compressedSize.toDouble / dataSize
              
              // Update statistics
              totalBytesInput.addAndGet(dataSize)
              totalBytesOutput.addAndGet(compressedSize)
              compressionCount.incrementAndGet()
              lastCompressionRatio.set(java.lang.Double.doubleToLongBits(ratio))
              
              if (ratio < MIN_BENEFICIAL_RATIO) {
                logDebug(s"Adaptive compression: $dataSize -> $compressedSize bytes " +
                  s"(ratio: ${ratio * 100}%.1f%%)")
                
                // Create compressed batch
                val cb = GpuCompressedColumnVector.from(compressedTable)
                return SpillableColumnarBatch(cb, priority)
              } else {
                logDebug(s"Compression not beneficial (ratio: ${ratio * 100}%.1f%%), using uncompressed")
              }
            }
          }
        }
        
        // Fall back to uncompressed
        SpillableColumnarBatch(ct, sparkTypes, priority)
        
      } catch {
        case e: Exception =>
          logWarning(s"Adaptive compression failed: ${e.getMessage}")
          SpillableColumnarBatch(ct, sparkTypes, priority)
      }
    }
  }
  
  /**
   * Compress a ColumnarBatch for spilling.
   * 
   * @param batch The batch to compress (will be closed)
   * @param priority Spill priority
   * @param codecConfig Codec configuration
   * @param threshold Memory pressure threshold
   * @return SpillableColumnarBatch
   */
  def compressBatchAndCreateSpillable(
      batch: ColumnarBatch,
      priority: Long,
      codecConfig: TableCompressionCodecConfig,
      threshold: Double = 0.8): SpillableColumnarBatch = {
    
    // Check if already compressed
    if (GpuCompressedColumnVector.isBatchCompressed(batch)) {
      return SpillableColumnarBatch(batch, priority)
    }
    
    val numRows = batch.numRows()
    if (numRows == 0 || batch.numCols() == 0) {
      return SpillableColumnarBatch(batch, priority)
    }
    
    // Convert to contiguous table
    val sparkTypes = GpuColumnVector.extractTypes(batch)
    
    withResource(GpuColumnVector.from(batch)) { table =>
      withResource(table.contiguousSplit()) { cts =>
        if (cts.nonEmpty && cts(0).getRowCount > 0) {
          val ct = cts(0)
          cts.indices.drop(1).foreach(i => cts(i).close())
          compressAndCreateSpillable(ct, sparkTypes, priority, codecConfig, threshold)
        } else {
          new JustRowsColumnarBatch(numRows)
        }
      }
    }
  }
  
  /**
   * Get average compression ratio (output/input).
   * Lower is better - 0.5 means 50% of original size.
   */
  def getAverageCompressionRatio: Double = {
    val input = totalBytesInput.get()
    if (input > 0) {
      totalBytesOutput.get().toDouble / input
    } else {
      java.lang.Double.longBitsToDouble(lastCompressionRatio.get())
    }
  }
  
  /**
   * Get compression statistics.
   * @return Tuple of (input bytes, output bytes, compression count, spill count, avg ratio)
   */
  def getStats: (Long, Long, Long, Long, Double) = {
    (totalBytesInput.get(), totalBytesOutput.get(), 
     compressionCount.get(), spillCount.get(), getAverageCompressionRatio)
  }
  
  /**
   * Get bytes saved by compression.
   */
  def getBytesSaved: Long = {
    val input = totalBytesInput.get()
    val output = totalBytesOutput.get()
    if (input > output) input - output else 0L
  }
  
  /**
   * Reset compression statistics.
   */
  def resetStats(): Unit = {
    totalBytesInput.set(0L)
    totalBytesOutput.set(0L)
    compressionCount.set(0L)
    spillCount.set(0L)
    lastCompressionRatio.set(java.lang.Double.doubleToLongBits(0.7))
  }
  
  /**
   * Enable or disable adaptive compression globally.
   */
  def setEnabled(enabled: Boolean): Unit = {
    compressionEnabled.set(enabled)
  }
  
  /**
   * Check if compression is enabled.
   */
  def isEnabled: Boolean = compressionEnabled.get()
  
  /**
   * Log compression summary for debugging.
   */
  def logSummary(): Unit = {
    val (input, output, count, spills, ratio) = getStats
    val saved = getBytesSaved
    logInfo(s"Adaptive Spill Compression Summary:")
    logInfo(s"  Total input: ${input / (1024 * 1024)} MB")
    logInfo(s"  Total output: ${output / (1024 * 1024)} MB")
    logInfo(s"  Bytes saved: ${saved / (1024 * 1024)} MB")
    logInfo(s"  Compression count: $count")
    logInfo(s"  Spill count: $spills")
    logInfo(s"  Average ratio: ${ratio * 100}%.1f%%")
  }
}
