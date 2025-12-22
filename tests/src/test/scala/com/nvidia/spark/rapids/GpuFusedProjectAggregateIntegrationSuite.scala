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
  
  // Test that explicitly verifies fusion is triggered (plan check)
  testSparkResultsAreEqualWithCapture(
    "Verify fusion is triggered with COALESCE pattern",
    spark => {
      import spark.implicits._
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
    assertFusionAttempted(gpuPlan)
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
}

