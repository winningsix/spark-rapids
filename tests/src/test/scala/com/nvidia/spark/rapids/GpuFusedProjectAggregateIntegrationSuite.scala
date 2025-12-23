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

import org.apache.spark.SparkConf
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions._

/**
 * Integration tests for GpuFusedProjectAggregate.
 * These tests run actual GPU operations to verify the fusion works correctly.
 * 
 * Key verification mechanisms:
 * 1. Results match CPU execution (correctness)
 * 2. Fusion counter increments (fusion actually triggered)
 * 3. GpuHashAggregateExec present in plan
 */
class GpuFusedProjectAggregateIntegrationSuite extends SparkQueryCompareTestSuite {

  private def fusedAggConf: SparkConf = {
    new SparkConf()
      .set("spark.rapids.sql.fusedTransformAggregate.enabled", "true")
      .set("spark.rapids.sql.fusedTransformAggregate.minColumns", "2")
      .set("spark.rapids.sql.explain", "ALL")
  }

  // Helper to check if fusion was attempted in the plan
  private def assertFusionAttempted(gpuPlan: SparkPlan): Unit = {
    val planStr = gpuPlan.toString
    // Check for GpuHashAggregateExec which is where fusion happens
    val gpuAggs = gpuPlan.collect {
      case agg: GpuHashAggregateExec => agg
    }
    assert(gpuAggs.nonEmpty, s"Expected GpuHashAggregateExec in plan: $planStr")
    
    // Log for verification
    System.err.println(s"[TEST] Found ${gpuAggs.size} GpuHashAggregateExec nodes")
    gpuAggs.foreach { agg =>
      val numAggs = agg.aggregateExpressions.size
      System.err.println(s"[TEST] GpuHashAggregateExec: $numAggs aggs")
    }
  }
  
  /**
   * Helper to verify fusion was actually executed (not just planned).
   * This is the definitive way to confirm fusion happened at runtime.
   */
  private def assertFusionExecuted(beforeCount: Long, testName: String): Unit = {
    val afterCount = GpuFusedProjectAggregate.getFusionExecutionCount
    val fusionCount = afterCount - beforeCount
    System.err.println(s"[TEST] $testName: Fusion executed $fusionCount times " +
      s"(before=$beforeCount, after=$afterCount)")
    assert(fusionCount > 0, 
      s"Fusion was NOT executed for test '$testName'. " +
      s"Counter before=$beforeCount, after=$afterCount. " +
      "Check if fusion is enabled and expressions are supported.")
  }

  // =========================================================================
  // Fusion Verification Tests - CRITICAL: These verify fusion actually runs
  // =========================================================================

  test("VERIFY: Fusion counter increments when fusion is triggered") {
    val conf = fusedAggConf
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      // Reset counter before test
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      assert(beforeCount == 0, "Counter should be 0 after reset")
      
      // Create test data with enough columns to trigger fusion
      val df = (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toLong)
        val v3 = if (i % 7 == 0) None else Some(i.toLong)
        (gid.toLong, v1, v2, v3)
      }.toDF("gid", "v1", "v2", "v3")
      
      // Execute aggregation that should trigger fusion
      val result = df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
          sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
          count(lit(1)).as("cnt")
        )
        .collect()
      
      // Verify fusion was executed
      assertFusionExecuted(beforeCount, "Basic fusion verification")
      
      System.err.println(s"[TEST] Result rows: ${result.length}")
    }, conf)
  }

  // Test that explicitly verifies fusion is triggered (plan check + counter)
  testSparkResultsAreEqualWithCapture(
    "Verify fusion is triggered with COALESCE pattern",
    spark => {
      import spark.implicits._
      // Reset counter before creating data
      GpuFusedProjectAggregate.resetFusionCounter()
      (1 to 200).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toLong)
        val v3 = if (i % 7 == 0) None else Some(i.toLong)
        val v4 = if (i % 11 == 0) None else Some(i.toLong)
        (gid.toLong, v1, v2, v3, v4)
      }.toDF("gid", "v1", "v2", "v3", "v4")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        sum(coalesce(col("v4"), lit(0L))).as("sum_v4"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  } { (cpuPlan, gpuPlan) =>
    // Verify plan contains GpuHashAggregateExec
    assertFusionAttempted(gpuPlan)
    // Verify fusion counter incremented
    val fusionCount = GpuFusedProjectAggregate.getFusionExecutionCount
    System.err.println(s"[TEST] Fusion execution count after test: $fusionCount")
    assert(fusionCount > 0, s"Fusion should have been executed, but counter is $fusionCount")
  }


  // Simple test: SUM with COALESCE pattern
  testSparkResultsAreEqual(
    "Fused aggregate with COALESCE pattern",
    spark => {
      import spark.implicits._
      Seq(
        (1L, Some(10L), Some(20L)),
        (1L, None, Some(30L)),
        (2L, Some(40L), None),
        (2L, Some(50L), Some(60L)),
        (3L, None, None)
      ).toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        sum(coalesce(col("val2"), lit(0L))).as("sum_val2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  // Test with multiple aggregation types
  testSparkResultsAreEqual(
    "Fused aggregate with SUM, AVG, COUNT",
    spark => {
      import spark.implicits._
      (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (gid.toLong, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("val1"), lit(0L)) * coalesce(col("val1"), lit(0L))).as("sum_sq")
      )
      .orderBy("gid")
  }

  // Test with conditional expressions (IF pattern)
  testSparkResultsAreEqual(
    "Fused aggregate with IF pattern",
    spark => {
      import spark.implicits._
      (1 to 50).map { i =>
        val gid = i % 5
        val cond = i % 2
        val val1 = i.toLong
        (gid.toLong, cond, val1)
      }.toDF("gid", "cond", "val1")
    },
    conf = fusedAggConf
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(when(col("cond") > 0, col("val1")).otherwise(0L)).as("sum_cond"),
        sum(col("val1")).as("sum_all"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  // Test with larger data to verify memory management
  testSparkResultsAreEqual(
    "Fused aggregate with 10k rows",
    spark => {
      import spark.implicits._
      (1 to 10000).map { i =>
        val gid = i % 100  // 100 groups
        val v1 = if (i % 7 == 0) None else Some(i.toLong)
        val v2 = if (i % 11 == 0) None else Some(i.toDouble)
        val v3 = if (i % 13 == 0) None else Some((i * 2).toLong)
        (gid.toLong, v1, v2, v3)
      }.toDF("gid", "val1", "val2", "val3")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        sum(coalesce(col("val3"), lit(0L))).as("sum_val3"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("val1"), lit(0L)) * coalesce(col("val1"), lit(0L))).as("sum_sq1"),
        sum(coalesce(col("val2"), lit(0.0)) * coalesce(col("val2"), lit(0.0))).as("sum_sq2")
      )
      .orderBy("gid")
  }

  // =========================================================================
  // SINGLE-PARTITION TESTS: Verify fusion is correctly NOT triggered
  // When using coalesce(1), Spark uses Complete mode (no Partial phase),
  // so fusion should NOT be triggered - this is correct behavior!
  // =========================================================================
  
  test("SINGLE-PARTITION: Fusion correctly NOT triggered in Complete-only mode") {
    val conf = fusedAggConf
      .set("spark.sql.shuffle.partitions", "1")
    withGpuSparkSession(spark => {
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // Use range + coalesce(1) which results in Complete-only aggregation
      val df = spark.range(1, 1001).selectExpr(
        "id % 20 as gid",
        "CASE WHEN id % 3 = 0 THEN null ELSE id END as v1",
        "CASE WHEN id % 5 = 0 THEN null ELSE id END as v2",
        "CASE WHEN id % 7 = 0 THEN null ELSE id END as v3"
      ).coalesce(1)  // Single partition = Complete mode only
      
      // Aggregate - should use Complete mode, NOT Partial mode
      val result = df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
          sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
          count(lit(1)).as("cnt")
        )
        .collect()
      
      // Fusion should NOT trigger in Complete mode (this is correct!)
      val afterCount = GpuFusedProjectAggregate.getFusionExecutionCount
      System.err.println(s"[TEST] Single-partition Complete mode: " +
        s"fusion count before=$beforeCount, after=$afterCount (should be same)")
      assert(afterCount == beforeCount, 
        s"Fusion should NOT trigger in Complete-only mode! before=$beforeCount, after=$afterCount")
      
      // Verify results are reasonable (20 groups)
      assert(result.length == 20, s"Expected 20 groups, got ${result.length}")
      System.err.println(s"[TEST] Single-partition test passed: no fusion (correct), ${result.length} groups")
    }, conf)
  }

  test("MULTI-PARTITION: Fusion triggers in Partial mode with merge") {
    val conf = fusedAggConf
      .set("spark.sql.shuffle.partitions", "4")  // Multiple partitions = Partial + Final
    withGpuSparkSession(spark => {
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // Use repartition to force multiple partitions
      val df = spark.range(1, 1001).selectExpr(
        "id % 20 as gid",
        "CASE WHEN id % 3 = 0 THEN null ELSE id END as v1",
        "CASE WHEN id % 5 = 0 THEN null ELSE id END as v2",
        "CASE WHEN id % 7 = 0 THEN null ELSE id END as v3"
      ).repartition(4)  // Multiple partitions = Partial mode will be used
      
      // This should trigger Partial mode aggregation where fusion happens
      val result = df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
          sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
          count(lit(1)).as("cnt")
        )
        .collect()
      
      // Fusion SHOULD trigger in Partial mode
      assertFusionExecuted(beforeCount, "Multi-partition Partial mode fusion")
      
      assert(result.length == 20, s"Expected 20 groups, got ${result.length}")
      System.err.println(s"[TEST] Multi-partition test: fusion triggered in Partial mode, ${result.length} groups")
    }, conf)
  }

  // Compare fusion ON vs OFF using the standard test framework
  testSparkResultsAreEqual(
    "Fusion ON produces same results as OFF (compare mode)",
    spark => {
      import spark.implicits._
      (1 to 1000).map { i =>
        val gid = i % 20
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (gid.toLong, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  // Test with many columns to force fusion
  testSparkResultsAreEqual(
    "Fused aggregate with many columns (force fusion)",
    spark => {
      import spark.implicits._
      (1 to 500).map { i =>
        val gid = i % 10
        (gid.toLong,
          if (i % 2 == 0) None else Some(i.toLong),
          if (i % 3 == 0) None else Some((i * 2).toLong),
          if (i % 5 == 0) None else Some((i * 3).toLong),
          if (i % 7 == 0) None else Some(i.toDouble),
          if (i % 11 == 0) None else Some((i * 0.5).toDouble))
      }.toDF("gid", "v1", "v2", "v3", "v4", "v5")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        avg(coalesce(col("v4"), lit(0.0))).as("avg_v4"),
        avg(coalesce(col("v5"), lit(0.0))).as("avg_v5"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq_v1"),
        sum(coalesce(col("v2"), lit(0L)) * coalesce(col("v2"), lit(0L))).as("sum_sq_v2")
      )
      .orderBy("gid")
  }

  // Test with explicit shuffle to trigger Kudo serialization
  // This is the scenario that can cause crashes if memory management is wrong
  testSparkResultsAreEqual(
    "Fused aggregate with shuffle (Kudo serialization test)",
    spark => {
      import spark.implicits._
      (1 to 5000).map { i =>
        val gid = i % 50  // 50 groups
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toLong)
        val v3 = if (i % 7 == 0) None else Some(i.toLong)
        (gid.toLong, v1, v2, v3)
      }.toDF("gid", "v1", "v2", "v3")
        .repartition(4)  // Force initial shuffle
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        count(lit(1)).as("cnt")
      )
      .repartition(2)  // Force shuffle after aggregation - triggers Kudo
      .orderBy("gid")
  }

  // Test with multiple aggregations and repartition to stress memory management
  testSparkResultsAreEqual(
    "Fused aggregate with large data and shuffle",
    spark => {
      import spark.implicits._
      (1 to 10000).map { i =>
        val gid = i % 100  // 100 groups
        val v1 = if (i % 2 == 0) None else Some(i.toLong)
        val v2 = if (i % 3 == 0) None else Some(i.toDouble)
        val v3 = if (i % 5 == 0) None else Some((i * 2).toLong)
        val v4 = if (i % 7 == 0) None else Some((i * 0.5).toDouble)
        (gid.toLong, v1, v2, v3, v4)
      }.toDF("gid", "v1", "v2", "v3", "v4")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    // Multiple aggregates to force fusion
    val result = df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        avg(coalesce(col("v4"), lit(0.0))).as("avg_v4"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq")
      )
    // Force shuffle to trigger serialization
    result.repartition(3).orderBy("gid")
  }

  // =========================================================================
  // MIN/MAX Aggregation Tests
  // =========================================================================

  testSparkResultsAreEqual(
    "Fused aggregate with MIN and MAX",
    spark => {
      import spark.implicits._
      (1 to 100).map { i =>
        val gid = i % 5
        val v1 = i.toLong
        val v2 = (100 - i).toLong
        (gid.toLong, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        min(col("val1")).as("min_val1"),
        max(col("val1")).as("max_val1"),
        min(col("val2")).as("min_val2"),
        max(col("val2")).as("max_val2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  // =========================================================================
  // Multiple Data Types Tests
  // =========================================================================

  testSparkResultsAreEqual(
    "Fused aggregate with Int values",
    spark => {
      import spark.implicits._
      (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i)
        val v2 = if (i % 5 == 0) None else Some(i * 2)
        (gid, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0))).as("sum_val1"),
        sum(coalesce(col("val2"), lit(0))).as("sum_val2"),
        count(lit(1)).as("cnt"),
        avg(coalesce(col("val1").cast("double"), lit(0.0))).as("avg_val1")
      )
      .orderBy("gid")
  }

  testSparkResultsAreEqual(
    "Fused aggregate with Double values",
    spark => {
      import spark.implicits._
      (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i * 0.1)
        val v2 = if (i % 5 == 0) None else Some(i * 0.5)
        (gid.toLong, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true,
    maxFloatDiff = 1e-9  // Allow small floating-point precision differences
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0.0))).as("sum_val1"),
        sum(coalesce(col("val2"), lit(0.0))).as("sum_val2"),
        avg(coalesce(col("val1"), lit(0.0))).as("avg_val1"),
        min(col("val1")).as("min_val1"),
        max(col("val2")).as("max_val2")
      )
      .orderBy("gid")
  }

  // =========================================================================
  // Multiple Group-By Columns Tests
  // =========================================================================

  testSparkResultsAreEqual(
    "Fused aggregate with two group-by columns",
    spark => {
      import spark.implicits._
      (1 to 200).map { i =>
        val gid1 = i % 5
        val gid2 = i % 4
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 7 == 0) None else Some(i.toDouble)
        (gid1.toLong, gid2.toLong, v1, v2)
      }.toDF("gid1", "gid2", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid1", "gid2")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("val1"), lit(0L)) * coalesce(col("val1"), lit(0L))).as("sum_sq")
      )
      .orderBy("gid1", "gid2")
  }

  testSparkResultsAreEqual(
    "Fused aggregate with String group-by column",
    spark => {
      import spark.implicits._
      val categories = Seq("apple", "banana", "cherry", "date", "elderberry")
      (1 to 100).map { i =>
        val category = categories(i % 5)
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (category, v1, v2)
      }.toDF("category", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("category")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("category")
  }

  testSparkResultsAreEqual(
    "Fused aggregate with mixed type group-by columns",
    spark => {
      import spark.implicits._
      val categories = Seq("A", "B", "C")
      (1 to 150).map { i =>
        val intKey = i % 5
        val strKey = categories(i % 3)
        val v1 = if (i % 4 == 0) None else Some(i.toLong)
        val v2 = if (i % 6 == 0) None else Some(i.toDouble)
        (intKey, strKey, v1, v2)
      }.toDF("int_key", "str_key", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("int_key", "str_key")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        min(col("val1")).as("min_val1"),
        max(col("val2")).as("max_val2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("int_key", "str_key")
  }

  // =========================================================================
  // Edge Cases and Boundary Tests
  // =========================================================================

  testSparkResultsAreEqual(
    "Fused aggregate with single group",
    spark => {
      import spark.implicits._
      (1 to 50).map { i =>
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (1L, v1, v2)  // All same group
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        count(lit(1)).as("cnt"),
        min(col("val1")).as("min_val1"),
        max(col("val2")).as("max_val2")
      )
      .orderBy("gid")
  }

  testSparkResultsAreEqual(
    "Fused aggregate with all nulls in one column",
    spark => {
      import spark.implicits._
      (1 to 50).map { i =>
        val gid = i % 5
        val v1: Option[Long] = None  // All nulls
        val v2 = Some(i.toLong)
        (gid.toLong, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        sum(coalesce(col("val2"), lit(0L))).as("sum_val2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  testSparkResultsAreEqual(
    "Fused aggregate with many groups (1000 groups)",
    spark => {
      import spark.implicits._
      (1 to 5000).map { i =>
        val gid = i % 1000  // 1000 unique groups
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toLong)
        (gid.toLong, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        sum(coalesce(col("val2"), lit(0L))).as("sum_val2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  // =========================================================================
  // Complex Expression Tests
  // =========================================================================

  testSparkResultsAreEqual(
    "Fused aggregate with nested COALESCE",
    spark => {
      import spark.implicits._
      (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 2 == 0) None else Some(i.toLong)
        val v2 = if (i % 3 == 0) None else Some(i.toLong)
        val v3 = if (i % 5 == 0) None else Some(i.toLong)
        (gid.toLong, v1, v2, v3)
      }.toDF("gid", "v1", "v2", "v3")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), col("v2"), col("v3"), lit(0L))).as("sum_coalesce"),
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  testSparkResultsAreEqual(
    "Fused aggregate with arithmetic expressions",
    spark => {
      import spark.implicits._
      (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some((i * 2).toLong)
        (gid.toLong, v1, v2)
      }.toDF("gid", "v1", "v2")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L)) + coalesce(col("v2"), lit(0L))).as("sum_add"),
        sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v2"), lit(1L))).as("sum_mul"),
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  // =========================================================================
  // Performance Stress Test (larger data)
  // =========================================================================

  testSparkResultsAreEqual(
    "Fused aggregate stress test (50k rows)",
    spark => {
      import spark.implicits._
      (1 to 50000).map { i =>
        val gid = i % 500
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        val v3 = if (i % 7 == 0) None else Some((i * 2).toLong)
        val v4 = if (i % 11 == 0) None else Some((i * 0.5).toDouble)
        (gid.toLong, v1, v2, v3, v4)
      }.toDF("gid", "v1", "v2", "v3", "v4")
    },
    conf = fusedAggConf,
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        avg(coalesce(col("v4"), lit(0.0))).as("avg_v4"),
        min(col("v1")).as("min_v1"),
        max(col("v2")).as("max_v2"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq")
      )
      .orderBy("gid")
  }

  // =========================================================================
  // Fusion Disabled Comparison Test
  // =========================================================================

  private def fusedAggDisabledConf: SparkConf = {
    new SparkConf()
      .set("spark.rapids.sql.fusedTransformAggregate.enabled", "false")
  }

  testSparkResultsAreEqual(
    "Compare fusion enabled vs disabled (correctness check)",
    spark => {
      import spark.implicits._
      (1 to 500).map { i =>
        val gid = i % 25
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (gid.toLong, v1, v2)
      }.toDF("gid", "val1", "val2")
    },
    conf = fusedAggConf,  // This will compare GPU (with fusion) vs CPU
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("val1"), lit(0L))).as("sum_val1"),
        avg(coalesce(col("val2"), lit(0.0))).as("avg_val2"),
        count(lit(1)).as("cnt"),
        min(col("val1")).as("min_val1"),
        max(col("val2")).as("max_val2")
      )
      .orderBy("gid")
  }

  // =========================================================================
  // Negative Tests - Verify fusion does NOT happen when disabled
  // =========================================================================

  test("VERIFY: Fusion does NOT trigger when disabled") {
    val conf = fusedAggDisabledConf
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      // Reset counter before test
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // Create test data
      val df = (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toLong)
        (gid.toLong, v1, v2)
      }.toDF("gid", "v1", "v2")
      
      // Execute aggregation
      df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
          count(lit(1)).as("cnt")
        )
        .collect()
      
      // Verify fusion was NOT executed
      val afterCount = GpuFusedProjectAggregate.getFusionExecutionCount
      val fusionCount = afterCount - beforeCount
      System.err.println(s"[TEST] Fusion disabled test: counter before=$beforeCount, after=$afterCount")
      assert(fusionCount == 0, 
        s"Fusion should NOT have been executed when disabled, but counter incremented by $fusionCount")
    }, conf)
  }

  test("VERIFY: Fusion does NOT trigger with too few columns") {
    // Set minColumns to 10 so our 3-column test won't trigger fusion
    val conf = new SparkConf()
      .set("spark.rapids.sql.fusedTransformAggregate.enabled", "true")
      .set("spark.rapids.sql.fusedTransformAggregate.minColumns", "10")
    
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      // Reset counter before test
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // Create test data with only 3 aggregate columns (below minColumns=10)
      val df = (1 to 100).map { i =>
        val gid = i % 10
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toLong)
        (gid.toLong, v1, v2)
      }.toDF("gid", "v1", "v2")
      
      // Execute aggregation with only 3 aggregates (below threshold)
      df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          sum(coalesce(col("v2"), lit(0L))).as("sum_v2"),
          count(lit(1)).as("cnt")
        )
        .collect()
      
      // Verify fusion was NOT executed (too few columns)
      val afterCount = GpuFusedProjectAggregate.getFusionExecutionCount
      val fusionCount = afterCount - beforeCount
      System.err.println(s"[TEST] Too few columns test: counter before=$beforeCount, after=$afterCount")
      assert(fusionCount == 0, 
        s"Fusion should NOT have been executed with too few columns, but counter incremented by $fusionCount")
    }, conf)
  }

  // =========================================================================
  // Comprehensive Fusion Verification Test
  // =========================================================================

  test("VERIFY: Comprehensive fusion verification with multiple patterns") {
    val conf = fusedAggConf
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      // Test 1: COALESCE + SUM pattern
      GpuFusedProjectAggregate.resetFusionCounter()
      var beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      val df1 = (1 to 100).map { i =>
        (i % 5, Option(i.toLong), Option((i * 2).toLong), Option((i * 3).toLong))
      }.toDF("gid", "v1", "v2", "v3")
      
      df1.groupBy("gid").agg(
        sum(coalesce(col("v1"), lit(0L))),
        sum(coalesce(col("v2"), lit(0L))),
        sum(coalesce(col("v3"), lit(0L))),
        count(lit(1))
      ).collect()
      
      assertFusionExecuted(beforeCount, "COALESCE + SUM pattern")
      
      // Test 2: Mixed aggregates (SUM, AVG, MIN, MAX)
      beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      val df2 = (1 to 100).map { i =>
        (i % 5, i.toLong, i.toDouble)
      }.toDF("gid", "val1", "val2")
      
      df2.groupBy("gid").agg(
        sum(col("val1")),
        avg(col("val2")),
        min(col("val1")),
        max(col("val2")),
        count(lit(1))
      ).collect()
      
      assertFusionExecuted(beforeCount, "Mixed aggregates (SUM, AVG, MIN, MAX)")
      
      // Test 3: Multiple group-by columns
      beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      val df3 = (1 to 100).map { i =>
        (i % 5, i % 3, Option(i.toLong), Option((i * 2).toLong))
      }.toDF("gid1", "gid2", "v1", "v2")
      
      df3.groupBy("gid1", "gid2").agg(
        sum(coalesce(col("v1"), lit(0L))),
        sum(coalesce(col("v2"), lit(0L))),
        count(lit(1)),
        avg(coalesce(col("v1").cast("double"), lit(0.0)))
      ).collect()
      
      assertFusionExecuted(beforeCount, "Multiple group-by columns")
      
      System.err.println(s"[TEST] All fusion verification tests passed!")
      System.err.println(s"[TEST] Total fusion executions: ${GpuFusedProjectAggregate.getFusionExecutionCount}")
    }, conf)
  }

  // =========================================================================
  // Multiple Batches + Fused Aggregation + Merge Pass Tests
  // These tests verify the complete fused aggregation pipeline including:
  // 1. Multiple input batches (controlled by small batch size)
  // 2. Fused kernel execution
  // 3. Merge pass for partial results
  // =========================================================================

  /**
   * Test with multiple batches by using small batch size.
   * This forces multiple executions of the fused kernel and tests
   * the merge pass that combines partial results.
   */
  test("MULTI-BATCH: Multiple batches with fused aggregation and merge pass") {
    // Use small batch size to force multiple batches
    val conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "10000")  // 10KB batches
      .set("spark.sql.shuffle.partitions", "4")
    
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // Generate data with ~750 rows per group (matching customer scenario)
      // 100 groups * 750 rows = 75,000 rows
      val numGroups = 100
      val rowsPerGroup = 750
      val totalRows = numGroups * rowsPerGroup
      
      val df = (1 to totalRows).map { i =>
        val gid = i % numGroups  // 100 unique groups
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        val v3 = if (i % 7 == 0) None else Some((i * 2).toLong)
        val v4 = if (i % 11 == 0) None else Some((i * 0.5).toDouble)
        (gid.toLong, v1, v2, v3, v4)
      }.toDF("gid", "v1", "v2", "v3", "v4")
        .repartition(8)  // Multiple partitions to ensure multiple batches
      
      // Execute with multiple aggregates to ensure fusion
      val result = df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
          sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
          avg(coalesce(col("v4"), lit(0.0))).as("avg_v4"),
          count(lit(1)).as("cnt"),
          sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq_v1"),
          min(col("v1")).as("min_v1"),
          max(col("v2")).as("max_v2")
        )
        .collect()
      
      // Verify fusion was executed (should be multiple times due to multiple batches)
      val afterCount = GpuFusedProjectAggregate.getFusionExecutionCount
      val fusionCount = afterCount - beforeCount
      System.err.println(s"[TEST] Multi-batch test: fusion executed $fusionCount times")
      assert(fusionCount > 0, 
        s"Fusion should have been executed, but counter is $fusionCount")
      
      // Verify correct number of groups
      assert(result.length == numGroups, 
        s"Expected $numGroups groups, got ${result.length}")
      
      System.err.println(s"[TEST] Multi-batch test passed: $fusionCount fusions, ${result.length} groups")
    }, conf)
  }

  /**
   * Correctness test: Multiple batches with small batch size.
   * Compare GPU (fused) results with CPU.
   */
  testSparkResultsAreEqual(
    "MULTI-BATCH: Correctness with small batch size forcing multiple batches",
    spark => {
      import spark.implicits._
      // 50 groups * 200 rows = 10,000 rows total
      (1 to 10000).map { i =>
        val gid = i % 50
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        val v3 = if (i % 7 == 0) None else Some((i * 2).toLong)
        (gid.toLong, v1, v2, v3)
      }.toDF("gid", "v1", "v2", "v3")
        .repartition(4)
    },
    conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "5000"),  // Very small batches
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq_v1")
      )
      .orderBy("gid")
  }

  /**
   * Test with high cardinality groups to stress merge pass.
   * Many groups mean many partial results to merge.
   */
  test("MERGE-PASS: High cardinality groups requiring merge pass") {
    val conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "10000")
      .set("spark.sql.shuffle.partitions", "8")
    
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // 1000 groups with ~100 rows per group = 100,000 rows
      // High group count will produce many partial aggregation results
      val numGroups = 1000
      val rowsPerGroup = 100
      val totalRows = numGroups * rowsPerGroup
      
      val df = (1 to totalRows).map { i =>
        val gid = i % numGroups
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (gid.toLong, v1, v2)
      }.toDF("gid", "v1", "v2")
        .repartition(8)
      
      val result = df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
          count(lit(1)).as("cnt"),
          sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq")
        )
        .collect()
      
      assertFusionExecuted(beforeCount, "High cardinality merge pass")
      assert(result.length == numGroups, 
        s"Expected $numGroups groups, got ${result.length}")
      
      System.err.println(s"[TEST] High cardinality test passed: ${result.length} groups")
    }, conf)
  }

  /**
   * Correctness test: High cardinality groups.
   */
  testSparkResultsAreEqual(
    "MERGE-PASS: Correctness with high cardinality groups (2000 groups)",
    spark => {
      import spark.implicits._
      // 2000 groups * 25 rows = 50,000 rows
      (1 to 50000).map { i =>
        val gid = i % 2000
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        val v3 = if (i % 7 == 0) None else Some((i * 2).toLong)
        (gid.toLong, v1, v2, v3)
      }.toDF("gid", "v1", "v2", "v3")
        .repartition(4)
    },
    conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "20000"),
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  /**
   * End-to-end test with shuffle: Tests the full pipeline including
   * fused aggregation -> shuffle -> final merge.
   * This is the scenario that can trigger memory issues.
   */
  test("E2E-SHUFFLE: Full pipeline with fused agg and shuffle") {
    val conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "10000")
      .set("spark.sql.shuffle.partitions", "4")
    
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // 200 groups * 500 rows = 100,000 rows
      val numGroups = 200
      val rowsPerGroup = 500
      val totalRows = numGroups * rowsPerGroup
      
      val df = (1 to totalRows).map { i =>
        val gid = i % numGroups
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        val v3 = if (i % 7 == 0) None else Some((i * 2).toLong)
        (gid.toLong, v1, v2, v3)
      }.toDF("gid", "v1", "v2", "v3")
        .repartition(8)  // Force shuffle before aggregation
      
      // Aggregate then shuffle again (triggers Kudo serialization)
      val result = df.groupBy("gid")
        .agg(
          sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
          avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
          sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
          count(lit(1)).as("cnt"),
          sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq")
        )
        .repartition(2)  // Shuffle after aggregation
        .collect()
      
      assertFusionExecuted(beforeCount, "E2E shuffle test")
      assert(result.length == numGroups, 
        s"Expected $numGroups groups, got ${result.length}")
      
      System.err.println(s"[TEST] E2E shuffle test passed: ${result.length} groups")
    }, conf)
  }

  /**
   * Correctness test: Full pipeline with shuffle.
   */
  testSparkResultsAreEqual(
    "E2E-SHUFFLE: Correctness with fused agg -> shuffle -> collect",
    spark => {
      import spark.implicits._
      // 100 groups * 200 rows = 20,000 rows
      (1 to 20000).map { i =>
        val gid = i % 100
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        val v3 = if (i % 7 == 0) None else Some((i * 2).toLong)
        val v4 = if (i % 11 == 0) None else Some((i * 0.5).toDouble)
        (gid.toLong, v1, v2, v3, v4)
      }.toDF("gid", "v1", "v2", "v3", "v4")
        .repartition(8)
    },
    conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "10000"),
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
        sum(coalesce(col("v3"), lit(0L))).as("sum_v3"),
        avg(coalesce(col("v4"), lit(0.0))).as("avg_v4"),
        count(lit(1)).as("cnt"),
        min(col("v1")).as("min_v1"),
        max(col("v2")).as("max_v2")
      )
      .repartition(2)
      .orderBy("gid")
  }

  /**
   * Customer scenario simulation: 750 rows per group with many columns.
   * Matches a typical production workload pattern with variance/covariance calculations.
   */
  test("CUSTOMER-SCENARIO: 750 rows/group with variance/covariance pattern") {
    val conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "50000")
      .set("spark.sql.shuffle.partitions", "4")
    
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // 50 groups * 750 rows = 37,500 rows (similar to customer scale)
      val numGroups = 50
      val rowsPerGroup = 750
      val totalRows = numGroups * rowsPerGroup
      
      val df = (1 to totalRows).map { i =>
        val gid = i % numGroups
        val pre = if (i % 3 == 0) None else Some(i.toLong)
        val post = if (i % 5 == 0) None else Some((i * 2).toLong)
        val cond = i % 2  // Condition for conditional aggregates
        (gid.toLong, pre, post, cond)
      }.toDF("gid", "pre", "post", "cond")
        .repartition(4)
      
      // Simulate production pattern: sum, avg, variance squares, covariance
      val result = df.groupBy("gid")
        .agg(
          sum(coalesce(col("pre"), lit(0L))).as("sum_pre"),
          sum(coalesce(col("post"), lit(0L))).as("sum_post"),
          avg(coalesce(col("pre"), lit(0L)) * coalesce(col("pre"), lit(0L))).as("avg_varsq_pre"),
          avg(coalesce(col("post"), lit(0L)) * coalesce(col("post"), lit(0L))).as("avg_varsq_post"),
          sum(coalesce(col("pre"), lit(0L)) * coalesce(col("post"), lit(0L))).as("sum_cov"),
          // Conditional aggregates
          sum(when(col("cond") > 0, col("pre")).otherwise(0L)).as("sum_cond_pre"),
          sum(when(col("cond") > 0, col("post")).otherwise(0L)).as("sum_cond_post"),
          count(lit(1)).as("cnt")
        )
        .collect()
      
      assertFusionExecuted(beforeCount, "Customer scenario (750 rows/group)")
      assert(result.length == numGroups, 
        s"Expected $numGroups groups, got ${result.length}")
      
      // Verify reasonable values (each group should have ~750 rows)
      val firstRow = result.head
      val cnt = firstRow.getAs[Long]("cnt")
      System.err.println(s"[TEST] Customer scenario: group cnt=$cnt (expected ~$rowsPerGroup)")
      
      System.err.println(s"[TEST] Customer scenario test passed: ${result.length} groups")
    }, conf)
  }

  /**
   * Correctness test: Customer scenario pattern.
   */
  testSparkResultsAreEqual(
    "CUSTOMER-SCENARIO: Correctness with variance/covariance pattern",
    spark => {
      import spark.implicits._
      // 30 groups * 500 rows = 15,000 rows
      (1 to 15000).map { i =>
        val gid = i % 30
        val pre = if (i % 3 == 0) None else Some(i.toLong)
        val post = if (i % 5 == 0) None else Some((i * 2).toLong)
        val cond = i % 2
        (gid.toLong, pre, post, cond)
      }.toDF("gid", "pre", "post", "cond")
        .repartition(4)
    },
    conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "20000"),
    sort = true,
    maxFloatDiff = 1e-6
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("pre"), lit(0L))).as("sum_pre"),
        sum(coalesce(col("post"), lit(0L))).as("sum_post"),
        avg(coalesce(col("pre").cast("double"), lit(0.0)) * 
            coalesce(col("pre").cast("double"), lit(0.0))).as("avg_varsq_pre"),
        sum(coalesce(col("pre"), lit(0L)) * coalesce(col("post"), lit(0L))).as("sum_cov"),
        sum(when(col("cond") > 0, col("pre")).otherwise(0L)).as("sum_cond_pre"),
        count(lit(1)).as("cnt")
      )
      .orderBy("gid")
  }

  /**
   * Extreme test: Very small batches forcing many merge passes.
   */
  testSparkResultsAreEqual(
    "EXTREME: Very small batches (1KB) forcing many merge operations",
    spark => {
      import spark.implicits._
      // 20 groups * 500 rows = 10,000 rows
      (1 to 10000).map { i =>
        val gid = i % 20
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (gid.toLong, v1, v2)
      }.toDF("gid", "v1", "v2")
        .repartition(8)
    },
    conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "1000"),  // 1KB - very small
    sort = true
  ) { df =>
    df.groupBy("gid")
      .agg(
        sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
        avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
        count(lit(1)).as("cnt"),
        sum(coalesce(col("v1"), lit(0L)) * coalesce(col("v1"), lit(0L))).as("sum_sq")
      )
      .orderBy("gid")
  }

  /**
   * Test with write to parquet to verify output format correctness.
   */
  test("OUTPUT: Fused aggregate result can be written to parquet") {
    val conf = fusedAggConf
      .set("spark.rapids.sql.batchSizeBytes", "10000")
    
    withGpuSparkSession(spark => {
      import spark.implicits._
      
      GpuFusedProjectAggregate.resetFusionCounter()
      val beforeCount = GpuFusedProjectAggregate.getFusionExecutionCount
      
      // 100 groups * 200 rows = 20,000 rows
      val df = (1 to 20000).map { i =>
        val gid = i % 100
        val v1 = if (i % 3 == 0) None else Some(i.toLong)
        val v2 = if (i % 5 == 0) None else Some(i.toDouble)
        (gid.toLong, v1, v2)
      }.toDF("gid", "v1", "v2")
        .repartition(4)
      
      val outputPath = s"/tmp/fused_agg_test_${System.currentTimeMillis()}"
      
      try {
        // Write aggregation result to parquet
        df.groupBy("gid")
          .agg(
            sum(coalesce(col("v1"), lit(0L))).as("sum_v1"),
            avg(coalesce(col("v2"), lit(0.0))).as("avg_v2"),
            count(lit(1)).as("cnt")
          )
          .write
          .mode("overwrite")
          .parquet(outputPath)
        
        // Read back and verify
        val readBack = spark.read.parquet(outputPath).collect()
        
        assertFusionExecuted(beforeCount, "Write to parquet test")
        assert(readBack.length == 100, s"Expected 100 groups, got ${readBack.length}")
        
        System.err.println(s"[TEST] Parquet write test passed: ${readBack.length} groups written/read")
      } finally {
        // Cleanup
        try {
          val fs = org.apache.hadoop.fs.FileSystem.get(spark.sparkContext.hadoopConfiguration)
          fs.delete(new org.apache.hadoop.fs.Path(outputPath), true)
        } catch {
          case _: Exception => // Ignore cleanup errors
        }
      }
    }, conf)
  }
}

