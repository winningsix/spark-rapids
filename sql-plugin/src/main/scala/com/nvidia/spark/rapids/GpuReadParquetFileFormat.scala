/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import scala.collection.JavaConverters._

import com.nvidia.spark.rapids.parquet.{GpuParquetMultiFilePartitionReaderFactory,
  GpuParquetPartitionReaderFactory, GpuParquetScan}
import org.apache.hadoop.conf.Configuration

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.PartitionReaderFactory
import org.apache.spark.sql.execution.FileSourceScanExec
import org.apache.spark.sql.execution.datasources.{PartitionedFile, PartitioningAwareFileIndex}
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.rapids.GpuFileSourceScanExec
import org.apache.spark.sql.rapids.shims.SparkSessionUtils
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.util.SerializableConfiguration

/**
 * A FileFormat that allows reading Parquet files with the GPU.
 */
class GpuReadParquetFileFormat extends ParquetFileFormat
    with GpuReadFileFormatWithMetrics with Logging {
  override def buildReaderWithPartitionValuesAndMetrics(
      sparkSession: SparkSession,
      dataSchema: StructType,
      partitionSchema: StructType,
      requiredSchema: StructType,
      filters: Seq[Filter],
      options: Map[String, String],
      hadoopConf: Configuration,
      metrics: Map[String, GpuMetric])
    : PartitionedFile => Iterator[InternalRow] = {
    val sqlConf = sparkSession.sessionState.conf
    val broadcastedHadoopConf =
      sparkSession.sparkContext.broadcast(new SerializableConfiguration(hadoopConf))
    val factory = GpuParquetPartitionReaderFactory(
      sqlConf,
      broadcastedHadoopConf,
      dataSchema,
      requiredSchema,
      partitionSchema,
      filters.toArray,
      new RapidsConf(sqlConf),
      metrics,
      options)
    PartitionReaderIterator.buildReader(factory)
  }

  override def isPerFileReadEnabled(conf: RapidsConf): Boolean = conf.isParquetPerFileReadEnabled

  override def createMultiFileReaderFactory(
      broadcastedConf: Broadcast[SerializableConfiguration],
      pushedFilters: Array[Filter],
      fileScan: GpuFileSourceScanExec): PartitionReaderFactory = {
    // Use GpuParquetScan to create the reader factory so that Hybrid Scan can be enabled
    val hadoopConf = fileScan.sparkSession.sessionState.newHadoopConfWithOptions(
      fileScan.relation.options)
    val fileIndex = fileScan.relation.location match {
      case p: PartitioningAwareFileIndex => p
      case _ => throw new IllegalStateException("FileIndex must be PartitioningAwareFileIndex")
    }
    // Convert CaseInsensitiveMap to CaseInsensitiveStringMap
    val optionsMap = fileScan.relation.options match {
      case m: CaseInsensitiveStringMap => m
      case m: Map[_, _] => 
        // Convert Map[String, String] to CaseInsensitiveStringMap
        val stringMap = m.asInstanceOf[Map[String, String]]
        new CaseInsensitiveStringMap(stringMap.asJava)
      case _ =>
        // Fallback: try to convert to Map first
        val map = fileScan.relation.options.asInstanceOf[Map[String, String]]
        new CaseInsensitiveStringMap(map.asJava)
    }
    val gpuParquetScan = GpuParquetScan(
      fileScan.sparkSession,
      hadoopConf,
      fileIndex,
      fileScan.relation.dataSchema,
      fileScan.requiredSchema,
      fileScan.readPartitionSchema,
      pushedFilters,
      optionsMap,
      fileScan.partitionFilters,
      fileScan.dataFilters,
      fileScan.rapidsConf,
      fileScan.queryUsesInputFile)
    
    // cuDF Hybrid Scan disabled - use standard factory
    val createdFactory = gpuParquetScan.createReaderFactory()
    // If it's GpuParquetMultiFilePartitionReaderFactory, ensure it has the right metrics
    val factory = createdFactory match {
      case _: GpuParquetMultiFilePartitionReaderFactory =>
        // Create a new factory with correct metrics
        val poolConfBuilder = ThreadPoolConfBuilder(fileScan.rapidsConf)
        GpuParquetMultiFilePartitionReaderFactory(
          fileScan.conf,
          broadcastedConf,
          fileScan.relation.dataSchema,
          fileScan.requiredSchema,
          fileScan.readPartitionSchema,
          pushedFilters,
          fileScan.rapidsConf,
          poolConfBuilder,
          fileScan.allMetrics, // Use metrics from fileScan
          fileScan.queryUsesInputFile)
      case _ => createdFactory
    }
    factory
  }
}

object GpuReadParquetFileFormat {
  def tagSupport(meta: SparkPlanMeta[FileSourceScanExec]): Unit = {
    val fsse = meta.wrapped
    val session = SparkSessionUtils.sessionFromPlan(fsse)
    GpuParquetScan.tagSupport(session, fsse.requiredSchema, meta)
  }
}
