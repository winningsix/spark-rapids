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

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, GreaterThan, Literal}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

class CudfHybridScanPartitionReaderSuite extends AnyFunSuite {
  
  test("CudfHybridScanUtils should separate filter and payload columns correctly") {
    val readSchema = StructType(Seq(
      StructField("col1", IntegerType, nullable = false),
      StructField("col2", IntegerType, nullable = false),
      StructField("col3", IntegerType, nullable = false)
    ))
    
    val dataFilters = Seq(
      GreaterThan(
        AttributeReference("col1", IntegerType, nullable = false)(),
        Literal(10)
      ),
      GreaterThan(
        AttributeReference("col2", IntegerType, nullable = false)(),
        Literal(20)
      )
    )
    
    val (filterColumns, payloadColumns) =
      CudfHybridScanUtils.separateColumns(readSchema, dataFilters)
    
    assert(filterColumns.length === 2, s"Expected 2 filter columns, got ${filterColumns.length}")
    assert(filterColumns.contains("col1"), "Filter columns should contain col1")
    assert(filterColumns.contains("col2"), "Filter columns should contain col2")
    assert(payloadColumns.length === 1, s"Expected 1 payload column, got ${payloadColumns.length}")
    assert(payloadColumns.contains("col3"), "Payload columns should contain col3")
  }
  
  test("CudfHybridScanUtils shouldUseCudfHybridScan should return false when no filters") {
    val readSchema = StructType(Seq(
      StructField("col1", IntegerType, nullable = false)
    ))
    
    val sqlConf = new org.apache.spark.sql.internal.SQLConf()
    sqlConf.setConfString("spark.rapids.sql.parquet.cudfHybridScan.enabled", "true")
    val rapidsConf = new com.nvidia.spark.rapids.RapidsConf(sqlConf)
    
    val result = CudfHybridScanUtils.shouldUseCudfHybridScan(
      rapidsConf,
      Seq.empty, // No filters
      readSchema
    )
    
    assert(result === false, "Should return false when no filters are present")
  }
}

