/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

/*** spark-rapids-shim-json-lines
{"spark": "400"}
{"spark": "400db173"}
{"spark": "401"}
{"spark": "402"}
{"spark": "403"}
{"spark": "404"}
{"spark": "411"}
{"spark": "412"}
{"spark": "413"}
{"spark": "420"}
spark-rapids-shim-json-lines ***/

package com.nvidia.spark.rapids.shims.parquet

import com.nvidia.spark.rapids.Arm.withResource
import org.apache.parquet.ParquetReadOptions
import org.apache.parquet.conf.HadoopParquetConfiguration
import org.apache.parquet.format.converter.ParquetMetadataConverter.MetadataFilter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.metadata.ParquetMetadata
import org.apache.parquet.io.InputFile

/** Reads in-memory footers without loading Hadoop's default XML resources for each file. */
class ParquetFooterReader {
  // Keep the generic InputFile API's default configuration semantics. Scope this to the
  // file filter handler, not a global cache, and never share mutable read-options builders.
  private lazy val defaultConf = new HadoopParquetConfiguration()

  def readFooter(input: InputFile, filter: MetadataFilter): ParquetMetadata = {
    val options = ParquetReadOptions.builder(defaultConf).withMetadataFilter(filter).build()
    withResource(input.newStream()) { stream =>
      ParquetFileReader.readFooter(input, options, stream)
    }
  }
}
