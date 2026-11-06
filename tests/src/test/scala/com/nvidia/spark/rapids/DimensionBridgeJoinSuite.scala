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
package com.nvidia.spark.rapids

import java.nio.file.Files
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.LeftOuter
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

class DimensionBridgeJoinSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _
  private val enabled = new AtomicBoolean(false)
  private val applied = new AtomicInteger()
  private val query = "SELECT a.id, b.wh, d1.week, a.v, b.qty FROM bridge_a a " +
    "JOIN bridge_dates d1 ON a.day = d1.day " +
    "JOIN bridge_items i ON a.item = i.item " +
    "JOIN bridge_dates d2 ON d1.week = d2.week " +
    "JOIN bridge_b b ON b.day = d2.day AND a.item = b.item " +
    "WHERE d1.year = 2000 AND b.qty < a.v"

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("dimension-bridge-tests").config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.hadoop.fs.defaultFS", "file:///")
      .config("spark.sql.warehouse.dir", Files.createTempDirectory("bridge").toUri.toString)
      .config("spark.sql.cbo.enabled", "true")
      .config("spark.sql.cbo.joinReorder.enabled", "true")
      .config("spark.sql.autoBroadcastJoinThreshold", "1024")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .withExtensions { extensions =>
        extensions.injectPlannerStrategy { _ => new SparkStrategy {
          private val delegate = new GpuDimensionBridgeJoinStrategy
          override def apply(plan: LogicalPlan): Seq[SparkPlan] = {
            if (enabled.get()) {
              val result = delegate(plan)
              if (result.nonEmpty) applied.incrementAndGet()
              result
            } else Nil
          }
        }}
      }.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.sql("CREATE TABLE bridge_a USING parquet PARTITIONED BY(day) AS " +
      "SELECT * FROM VALUES (1,1,10,30),(1,1,10,30),(2,2,10,12),(3,3,20,9)," +
      "(4,4,10,20),(5,NULL,10,12),(6,1,NULL,50),(7,1,10,NULL) t(id,day,item,v)")
    spark.sql("CREATE TABLE bridge_b USING parquet PARTITIONED BY(day) AS " +
      "SELECT * FROM VALUES (1,10,100,5),(2,10,101,7),(2,10,101,7),(3,20,102,4)," +
      "(4,10,103,4),(NULL,10,100,1),(1,NULL,100,1),(1,10,104,NULL) t(day,item,wh,qty)")
    spark.sql("CREATE TABLE bridge_dates USING parquet AS SELECT * FROM VALUES " +
      "(1,1,2000),(2,1,2000),(2,1,2000),(3,2,2000),(4,2,2001)," +
      "(NULL,1,2000),(5,3,2001),(6,3,2001) t(day,week,year)")
    spark.sql("CREATE TABLE bridge_items USING parquet AS " +
      "SELECT * FROM VALUES (10,'ten'),(20,'twenty') t(item,name)")
    for (table <- Seq("bridge_a", "bridge_b", "bridge_dates", "bridge_items")) {
      spark.sql(s"ANALYZE TABLE $table COMPUTE STATISTICS FOR ALL COLUMNS")
      val id = TableIdentifier(table)
      val stats = spark.sessionState.catalog.getTableMetadata(id).stats.get
      // Deliberately imperfect estimates select the heuristic on tiny actual data.
      // They must never justify dropping duplicate rows or treating NULL as a match.
      val rows = table match {
        case "bridge_a" => BigInt(1000000)
        case "bridge_b" => BigInt(100000)
        case _ => stats.rowCount.get
      }
      val columns = if (table == "bridge_dates") {
        stats.colStats.updated("day", stats.colStats("day")
          .copy(distinctCount = Some(rows), nullCount = Some(BigInt(0))))
      } else stats.colStats
      spark.sessionState.catalog.alterTableStats(id,
        Some(stats.copy(rowCount = Some(rows), sizeInBytes = rows * 32, colStats = columns)))
    }
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally super.afterAll()
  }

  private def eligible(): Join = spark.sql(query).queryExecution.optimizedPlan.collectFirst {
    case join: Join if GpuDimensionBridgeJoinReorder.rewrite(join).isDefined => join
  }.getOrElse(fail("Synthetic dimension bridge did not select the rewrite"))

  private def check(sql: String): Unit = {
    enabled.set(false)
    val expected = spark.sql(sql).collect().map(_.toString).sorted.toSeq
    applied.set(0)
    enabled.set(true)
    try {
      val actual = spark.sql(sql).collect().map(_.toString).sorted.toSeq
      assertResult(expected)(actual)
      assert(applied.get() > 0, "The executed candidate must use the planner strategy")
    } finally enabled.set(false)
  }

  test("preserve duplicate and NULL rows through actual CPU planner execution") {
    check(query)
  }

  test("preserve aggregate multiplicity after the reordered join") {
    check(s"SELECT week, count(*) AS n, sum(v) AS s FROM ($query) t GROUP BY week")
  }

  test("retain output attributes and stop recursive planner rewrites") {
    val join = eligible()
    val result = GpuDimensionBridgeJoinReorder.rewrite(join).get
    assert(result.resolved)
    assertResult(join.output)(result.output)
    assert(result.collect {
      case j: Join if GpuDimensionBridgeJoinReorder.rewrite(j).isDefined => j
    }.isEmpty)
  }

  test("keep small pruning subqueries when their old broadcast disappears") {
    val result = GpuDimensionBridgeJoinReorder.rewrite(eligible()).get
    val pruning = result.flatMap(_.expressions.flatMap(_.collect {
      case p: DynamicPruningSubquery => p
    }))
    assert(pruning.nonEmpty, "Partitioned facts must exercise dynamic pruning")
    assert(pruning.exists(p => p.buildQuery.stats.sizeInBytes <= 1024 && !p.onlyInBroadcast))
  }

  test("reject outer joins and explicit join hints") {
    val join = eligible()
    assert(GpuDimensionBridgeJoinReorder.rewrite(join.copy(joinType = LeftOuter)).isEmpty)
    val hint = JoinHint(Some(HintInfo(Some(BROADCAST))), None)
    assert(GpuDimensionBridgeJoinReorder.rewrite(join.copy(hint = hint)).isEmpty)
  }

  test("reject nondeterministic and potentially throwing join predicates") {
    val join = eligible()
    val random = GreaterThan(Rand(1L), Literal(0.5))
    assert(GpuDimensionBridgeJoinReorder.rewrite(join.copy(condition = Some(random))).isEmpty)
    val arithmetic = EqualTo(Add(Literal(Int.MaxValue), Literal(1)), Literal(0))
    val condition = And(join.condition.get, arithmetic)
    assert(GpuDimensionBridgeJoinReorder.rewrite(join.copy(condition = Some(condition))).isEmpty)
  }
}
