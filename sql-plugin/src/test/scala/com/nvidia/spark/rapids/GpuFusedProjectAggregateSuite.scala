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

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkConf
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{Complete, Partial}
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuCount}
import org.apache.spark.sql.types.{DecimalType, LongType}

/**
 * Unit tests for GpuFusedProjectAggregate fusion detection logic.
 * These tests verify the shouldTryFusion conditions without requiring GPU execution.
 */
class GpuFusedProjectAggregateSuite extends AnyFunSuite {

  private def createConf(enabled: Boolean, minColumns: Int): RapidsConf = {
    val sparkConf = new SparkConf()
    sparkConf.set(RapidsConf.ENABLE_FUSED_TRANSFORM_AGGREGATE.key, enabled.toString)
    sparkConf.set(RapidsConf.FUSED_TRANSFORM_AGGREGATE_MIN_COLUMNS.key, minColumns.toString)
    new RapidsConf(sparkConf)
  }

  private def createMockAggExprs(count: Int): Seq[GpuAggregateExpression] = {
    // Create simple mock aggregate expressions for testing
    (0 until count).map { i =>
      val attr = AttributeReference(s"col$i", LongType)()
      GpuAggregateExpression(
        GpuCount(Seq(attr)),
        mode = Partial,
        isDistinct = false,
        filter = None,
        resultId = ExprId(i)
      )
    }
  }

  test("shouldTryFusion returns false when fusion is disabled in config") {
    val conf = createConf(enabled = false, minColumns = 4)
    val aggExprs = createMockAggExprs(10)
    val modeInfo = AggregateModeInfo(Seq(Partial), hasPartialMode = true, 
      hasPartialMergeMode = false, hasFinalMode = false, hasCompleteMode = false)
    
    // Merged case: shouldTryFusion now takes only conf, aggExprs, modeInfo
    val result = GpuFusedProjectAggregate.shouldTryFusion(conf, aggExprs, modeInfo)
    
    assert(!result, "Fusion should be disabled when config is false")
  }

  test("shouldTryFusion returns false when aggregate count is below threshold") {
    val conf = createConf(enabled = true, minColumns = 10)
    val aggExprs = createMockAggExprs(5) // Less than minColumns
    val modeInfo = AggregateModeInfo(Seq(Partial), hasPartialMode = true,
      hasPartialMergeMode = false, hasFinalMode = false, hasCompleteMode = false)
    
    val result = GpuFusedProjectAggregate.shouldTryFusion(conf, aggExprs, modeInfo)
    
    assert(!result, "Fusion should be disabled when agg count < minColumns")
  }

  test("shouldTryFusion returns false for Final mode (not Partial/Complete)") {
    val conf = createConf(enabled = true, minColumns = 4)
    val aggExprs = createMockAggExprs(10)
    val modeInfo = AggregateModeInfo(Seq.empty, hasPartialMode = false,
      hasPartialMergeMode = false, hasFinalMode = true, hasCompleteMode = false)
    
    val result = GpuFusedProjectAggregate.shouldTryFusion(conf, aggExprs, modeInfo)
    
    assert(!result, "Fusion should be disabled for Final mode")
  }
  
  test("hasFusableInput returns true for simple column references in aggregate") {
    val attr = AttributeReference("col", LongType)()
    val countAgg = GpuAggregateExpression(
      GpuCount(Seq(attr)),
      mode = Partial,
      isDistinct = false,
      filter = None,
      resultId = ExprId(0)
    )
    
    // GpuCount with a simple attribute reference should be fusable
    // Note: The actual isFusableExpression check happens on the child expression
    assert(GpuFusedProjectAggregate.hasFusableInput(countAgg), 
      "COUNT(col) should have fusable input")
  }

  test("configuration defaults are correct") {
    val conf = new RapidsConf(new SparkConf())
    
    assert(conf.enableFusedTransformAggregate, 
      "Fused transform aggregate should be enabled by default")
    assert(conf.fusedTransformAggregateMinColumns == 4,
      "Default min columns should be 4")
    assert(!conf.fusedTransformAggregateWarpReduction,
      "Warp reduction should be disabled by default")
  }

  test("AggregateModeInfo correctly identifies Partial mode") {
    val partialMode = AggregateModeInfo(Seq(Partial), hasPartialMode = true,
      hasPartialMergeMode = false, hasFinalMode = false, hasCompleteMode = false)
    
    assert(partialMode.hasPartialMode)
    assert(!partialMode.hasFinalMode)
    assert(!partialMode.hasCompleteMode)
  }

  test("AggregateModeInfo correctly identifies Complete mode") {
    val completeMode = AggregateModeInfo(Seq(Complete), hasPartialMode = false,
      hasPartialMergeMode = false, hasFinalMode = false, hasCompleteMode = true)
    
    assert(!completeMode.hasPartialMode)
    assert(!completeMode.hasFinalMode)
    assert(completeMode.hasCompleteMode)
  }

  // ==================== DECIMAL Type Support Tests ====================

  test("DECIMAL64 (precision <= 18) should be supported by isSupportedDataType") {
    // DECIMAL64 can be handled as int64 with scaled values
    assert(GpuFusedProjectAggregate.isSupportedDataType(DecimalType(10, 2)),
      "DECIMAL(10,2) should be supported")
    assert(GpuFusedProjectAggregate.isSupportedDataType(DecimalType(18, 0)),
      "DECIMAL(18,0) should be supported")
    assert(GpuFusedProjectAggregate.isSupportedDataType(DecimalType(18, 6)),
      "DECIMAL(18,6) should be supported")
    assert(GpuFusedProjectAggregate.isSupportedDataType(DecimalType(1, 0)),
      "DECIMAL(1,0) should be supported")
  }

  test("DECIMAL128 (precision > 18) should NOT be supported by isSupportedDataType") {
    // DECIMAL128 requires 128-bit atomics which are not available
    assert(!GpuFusedProjectAggregate.isSupportedDataType(DecimalType(19, 0)),
      "DECIMAL(19,0) should NOT be supported")
    assert(!GpuFusedProjectAggregate.isSupportedDataType(DecimalType(38, 10)),
      "DECIMAL(38,10) should NOT be supported")
    assert(!GpuFusedProjectAggregate.isSupportedDataType(DecimalType(20, 2)),
      "DECIMAL(20,2) should NOT be supported")
  }

  test("shouldTryFusion accepts DECIMAL64 aggregate columns") {
    val conf = createConf(enabled = true, minColumns = 1)
    // Create aggregate with DECIMAL64 type
    val decAttr = AttributeReference("dec_col", DecimalType(18, 2))()
    val countAgg = GpuAggregateExpression(
      GpuCount(Seq(decAttr)),
      mode = Partial,
      isDistinct = false,
      filter = None,
      resultId = ExprId(0)
    )
    val modeInfo = AggregateModeInfo(Seq(Partial), hasPartialMode = true,
      hasPartialMergeMode = false, hasFinalMode = false, hasCompleteMode = false)

    val result = GpuFusedProjectAggregate.shouldTryFusion(conf, Seq(countAgg), modeInfo)
    assert(result, "Fusion should be enabled for DECIMAL64 columns")
  }
}

