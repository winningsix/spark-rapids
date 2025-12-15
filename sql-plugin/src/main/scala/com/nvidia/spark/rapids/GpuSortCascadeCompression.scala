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

import ai.rapids.cudf.{ContiguousTable, Cuda, NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.Arm.withResource

import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * GPU-accelerated cascade compression for out-of-core sort.
 *
 * Cascade compression compresses data before device-to-host transfer during spill,
 * reducing memory bandwidth requirements and transfer time. The compression happens
 * entirely on GPU using nvCOMP codecs (LZ4/ZSTD).
 *
 * Benefits:
 * - Reduces GPU-to-Host PCIe bandwidth usage (30-70% depending on data)
 * - Decreases spill time by reducing transfer size
 * - GPU compression is fast, often faster than raw transfer
 * - Particularly effective for repetitive/compressible data patterns
 *
 * Compression is applied when creating SpillableColumnarBatch during out-of-core
 * sort operations, making it transparent to the rest of the sort logic.
 */
object GpuSortCascadeCompression extends Logging {
  
  // Maximum batch size for compression operations
  private val MAX_COMPRESSION_BATCH_SIZE = 128L * 1024 * 1024 // 128MB
  
  // Minimum size to consider for compression (smaller data not worth compressing)
  private val MIN_COMPRESSION_SIZE = 64 * 1024 // 64KB
  
  // Statistics for monitoring compression effectiveness
  @volatile private var totalBytesInput: Long = 0L
  @volatile private var totalBytesOutput: Long = 0L
  @volatile private var compressionCount: Long = 0L
  
  /**
   * Compress a ContiguousTable and create a SpillableColumnarBatch.
   * The batch will automatically decompress when accessed.
   * Note: Takes ownership of ct and will close it.
   *
   * @param ct The ContiguousTable to compress
   * @param sparkTypes Spark data types for the columns
   * @param priority Spill priority for the resulting batch
   * @param codecName Compression codec name ("lz4", "zstd", or "copy")
   * @param codecConfig Codec configuration for compression settings
   * @return SpillableColumnarBatch with compressed data
   */
  def compressAndCreateSpillable(
      ct: ContiguousTable,
      sparkTypes: Array[DataType],
      priority: Long,
      codecName: String,
      codecConfig: TableCompressionCodecConfig): SpillableColumnarBatch = {
    
    withResource(new NvtxRange("sortCascadeCompress", NvtxColor.YELLOW)) { _ =>
      compressAndCreateSpillableInternal(ct, sparkTypes, priority, codecName, codecConfig)
    }
  }
  
  /**
   * Compress a ColumnarBatch and create a SpillableColumnarBatch.
   *
   * @param batch The batch to compress (will be closed)
   * @param priority Spill priority
   * @param codecName Compression codec name
   * @param codecConfig Codec configuration
   * @return SpillableColumnarBatch with compressed data
   */
  def compressBatchAndCreateSpillable(
      batch: ColumnarBatch,
      priority: Long,
      codecName: String,
      codecConfig: TableCompressionCodecConfig): SpillableColumnarBatch = {
    
    // Check if already compressed
    if (GpuCompressedColumnVector.isBatchCompressed(batch)) {
      return SpillableColumnarBatch(batch, priority)
    }
    
    val numRows = batch.numRows()
    if (numRows == 0 || batch.numCols() == 0) {
      return SpillableColumnarBatch(batch, priority)
    }
    
    // Convert to contiguous table for compression
    val sparkTypes = GpuColumnVector.extractTypes(batch)
    
    withResource(new NvtxRange("sortCascadeCompressBatch", NvtxColor.YELLOW)) { _ =>
      withResource(GpuColumnVector.from(batch)) { table =>
        withResource(table.contiguousSplit()) { cts =>
          if (cts.nonEmpty && cts(0).getRowCount > 0) {
            // Take ownership of first split, close rest
            val ct = cts(0)
            cts.indices.drop(1).foreach(i => cts(i).close())
            compressAndCreateSpillableInternal(ct, sparkTypes, priority, codecName, codecConfig)
          } else {
            new JustRowsColumnarBatch(numRows)
          }
        }
      }
    }
  }
  
  /**
   * Internal method to compress a ContiguousTable.
   * Note: Takes ownership of ct and will close it.
   */
  private def compressAndCreateSpillableInternal(
      ct: ContiguousTable,
      sparkTypes: Array[DataType],
      priority: Long,
      codecName: String,
      codecConfig: TableCompressionCodecConfig): SpillableColumnarBatch = {
    
    val bufferSize = ct.getBuffer.getLength
    
    // Skip compression for small tables
    if (bufferSize < MIN_COMPRESSION_SIZE) {
      logDebug(s"Skipping compression for small table: $bufferSize bytes")
      return SpillableColumnarBatch(ct, sparkTypes, priority)
    }
    
    try {
      val codec = TableCompressionCodec.getCodec(codecName.toLowerCase, codecConfig)
      
      withResource(codec.createBatchCompressor(MAX_COMPRESSION_BATCH_SIZE, 
          Cuda.DEFAULT_STREAM)) { compressor =>
        
        compressor.addTableToCompress(ct)
        
        withResource(compressor.finish()) { compressedTables =>
          if (compressedTables.nonEmpty) {
            val compressedTable = compressedTables.head
            val compressedSize = compressedTable.compressedSize
            
            // Update statistics
            totalBytesInput += bufferSize
            totalBytesOutput += compressedSize
            compressionCount += 1
            
            val ratio = compressedSize.toDouble / bufferSize
            logDebug(s"Cascade compression: $bufferSize -> $compressedSize bytes " +
              s"(ratio: ${ratio * 100}%.1f%%)")
            
            // Create compressed batch - this takes ownership of compressedTable
            val cb = GpuCompressedColumnVector.from(compressedTable)
            SpillableColumnarBatch(cb, priority)
          } else {
            // Compression produced no output, fall back to uncompressed
            logWarning("Compression produced no output, using uncompressed")
            SpillableColumnarBatch(ct, sparkTypes, priority)
          }
        }
      }
    } catch {
      case e: Exception =>
        logWarning(s"Cascade compression failed, falling back to uncompressed: ${e.getMessage}")
        SpillableColumnarBatch(ct, sparkTypes, priority)
    }
  }
  
  /**
   * Get compression statistics.
   * @return Tuple of (total input bytes, total output bytes, compression count)
   */
  def getStats: (Long, Long, Long) = (totalBytesInput, totalBytesOutput, compressionCount)
  
  /**
   * Get average compression ratio (output/input).
   * @return Compression ratio, or 1.0 if no compression has occurred
   */
  def getAverageCompressionRatio: Double = {
    if (totalBytesInput > 0) {
      totalBytesOutput.toDouble / totalBytesInput
    } else {
      1.0
    }
  }
  
  /**
   * Reset compression statistics.
   */
  def resetStats(): Unit = {
    totalBytesInput = 0L
    totalBytesOutput = 0L
    compressionCount = 0L
  }
}

