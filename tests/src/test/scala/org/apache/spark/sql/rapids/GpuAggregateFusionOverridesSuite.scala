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

package org.apache.spark.sql.rapids

import java.nio.file.Files

import com.nvidia.spark.rapids.GpuAggregateFusionOverrides
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

class GpuAggregateFusionOverridesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("aggregate-fusion-tests")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.hadoop.fs.defaultFS", "file:///")
      .config("spark.sql.warehouse.dir", Files.createTempDirectory("aggregate-fusion")
        .toUri.toString)
      .config("spark.sql.catalogImplementation", "in-memory")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .withExtensions { extensions =>
        extensions.injectOptimizerRule(session => new GpuAggregateFusionOverrides(session))
      }.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.sql("SELECT k, v, CAST(price AS DECIMAL(5,2)) price FROM VALUES " +
      "(1, 3L, 12.25), (1, 3L, 12.25), (2, 0L, -10.00), " +
      "(CAST(NULL AS INT), 2L, 4.50), (2, CAST(NULL AS BIGINT), 1.00), " +
      "(3, 4L, NULL) t(k, v, price)").write.saveAsTable("fusion_source")
    spark.sql("SELECT * FROM fusion_source WHERE false").write.saveAsTable("fusion_empty")
    spark.sql("SELECT * FROM fusion_source WHERE k = 1").write.saveAsTable("fusion_other")
    spark.sql("SELECT * FROM VALUES (1,8,35),(1,9,20),(2,9,45),(2,9,45)," +
      "(3,10,10),(NULL,9,40) t(dk,hour,minute)").write.saveAsTable("fusion_dim")
    spark.sql("SELECT * FROM VALUES (1,'x'),(2,'x'),(2,'x'),(3,NULL) t(sk,label)")
      .write.saveAsTable("fusion_store")
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally {
      super.afterAll()
    }
  }

  private def check(sql: String, expectedRewrite: Boolean, expectedScans: Int = 1): Unit = {
    def rows(df: TrampolineConnectShims.DataFrame) = df.collect().toSeq.groupBy(identity)
      .map { case (row, copies) => row -> copies.size }
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val baseline = spark.sql(sql)
    val expected = rows(baseline)
    def scans(df: TrampolineConnectShims.DataFrame): Int =
      df.queryExecution.optimizedPlan.collect { case r: LogicalRelation => r }.size
    spark.conf.set("spark.rapids.sql.enabled", "true")
    val candidate = spark.sql(sql)
    assert(candidate.schema == baseline.schema, sql)
    assert(rows(candidate) == expected, sql)
    assert(scans(candidate) == (if (expectedRewrite) expectedScans else scans(baseline)),
      candidate.queryExecution.optimizedPlan.toString)
    val plan = candidate.queryExecution.optimizedPlan
    assert(new GpuAggregateFusionOverrides(spark)(plan).fastEquals(plan))
  }

  private def branch(table: String, predicate: String): String =
    "SELECT avg(price) av, count(price) n, count(DISTINCT price) nd FROM " +
      table + " WHERE " + predicate

  test("one scan preserves overlapping filters, decimals, duplicate rows and NULLs") {
    for (predicates <- Seq(Seq("k <= 2", "k >= 2"),
        Seq("k = 1", "k = 2", "k IS NULL", "price IS NULL", "v > 1", "v IS NULL"))) {
      val from = predicates.zipWithIndex.map { case (predicate, i) =>
        "(" + branch("fusion_source", predicate) + ") b" + i
      }.mkString(" CROSS JOIN ")
      check("SELECT * FROM " + from, expectedRewrite = true)
    }
  }

  test("empty inputs and predicates matching no rows still produce one result row") {
    for (table <- Seq("fusion_source", "fusion_empty")) {
      check("SELECT * FROM (" + branch(table, "k > 10") + ") a CROSS JOIN (" +
        branch(table, "k < 0") + ") b", expectedRewrite = true)
    }
  }

  test("existing aggregate filters combine with each original WHERE clause") {
    check("SELECT * FROM (SELECT count(*) n, sum(v) FILTER (WHERE v > 1) s " +
      "FROM fusion_source WHERE k <= 2) a CROSS JOIN (SELECT count(*) n, " +
      "max(v) m, min(v) lo FROM fusion_source WHERE k >= 2) b", expectedRewrite = true)
  }

  test("unsafe argument evaluation is not moved onto previously excluded rows") {
    spark.conf.set("spark.sql.ansi.enabled", "true")
    check("SELECT * FROM (SELECT sum(100 / v) x FROM fusion_source WHERE v <> 0) a " +
      "CROSS JOIN (SELECT count(*) n FROM fusion_source WHERE k > 0) b",
      expectedRewrite = false)
  }

  test("different relations, non-global aggregates and multiple DISTINCT columns stay separate") {
    check("SELECT * FROM (" + branch("fusion_source", "k > 0") + ") a CROSS JOIN (" +
      branch("fusion_other", "k > 0") + ") b", expectedRewrite = false)
    check("SELECT * FROM (SELECT k, count(*) FROM fusion_source GROUP BY k) a " +
      "CROSS JOIN (SELECT count(*) FROM fusion_source) b", expectedRewrite = false)
    check("SELECT * FROM (SELECT count(DISTINCT price) FROM fusion_source) a " +
      "CROSS JOIN (SELECT count(DISTINCT k) FROM fusion_source) b", expectedRewrite = false)
  }

  test("nondeterministic predicates retain their separate evaluations") {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val plan = spark.sql("SELECT * FROM (SELECT count(*) FROM fusion_source " +
      "WHERE rand() > 0.5) a CROSS JOIN (SELECT count(*) FROM fusion_source " +
      "WHERE rand() > 0.5) b").queryExecution.optimizedPlan
    spark.conf.set("spark.rapids.sql.enabled", "true")
    assert(new GpuAggregateFusionOverrides(spark)(plan).fastEquals(plan))
  }
  private def joined(table: String, predicate: String, joinType: String = "JOIN"): String =
    "SELECT count(*) n, count(f.price) np FROM " + table + " f " + joinType +
      " fusion_dim d ON f.k=d.dk JOIN fusion_store s ON f.k=s.sk WHERE " + predicate

  test("joined COUNT branches share one scan per relation with duplicate and NULL keys") {
    for (aqe <- Seq(false, true); table <- Seq("fusion_source", "fusion_empty")) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      val predicates = Seq("hour=8 AND minute>=30", "hour=9 AND minute<30",
        "hour=9 AND minute>=30", "hour=10 AND minute<30", "hour IS NULL")
      val from = predicates.zipWithIndex.map { case (predicate, i) =>
        "(" + joined(table, "(" + predicate + ") AND label='x'") + ") b" + i
      }.mkString(" CROSS JOIN ")
      check("SELECT * FROM " + from, expectedRewrite = true, expectedScans = 3)
    }
  }

  test("joined fusion preserves empty branch counts and existing aggregate filters") {
    val left = joined("fusion_source", "hour=25").replace("count(*)", "count(*) FILTER " +
      "(WHERE minute>30)")
    val right = joined("fusion_source", "hour=9")
    check("SELECT * FROM (" + left + ") a CROSS JOIN (" + right + ") b",
      expectedRewrite = true, expectedScans = 3)
  }

  test("outer joins, different joined relations and non-COUNT joined aggregates stay separate") {
    val normal = joined("fusion_source", "hour=9")
    val controls = Seq(joined("fusion_source", "hour IS NULL", "LEFT JOIN"),
      joined("fusion_other", "hour=9"), normal.replace("count(*)", "sum(f.v)"))
    controls.foreach { other =>
      check("SELECT * FROM (" + normal + ") a CROSS JOIN (" + other + ") b",
        expectedRewrite = false)
    }
  }

  test("joined branches with potentially throwing filters are not broadened") {
    val left = joined("fusion_source", "f.v<>0 AND 100/f.v>1")
    val right = joined("fusion_source", "hour=9")
    check("SELECT * FROM (" + left + ") a CROSS JOIN (" + right + ") b",
      expectedRewrite = false)
    val aggregateFilter = joined("fusion_source", "f.v<>0").replace("count(*)",
      "count(*) FILTER (WHERE 100/f.v>1)")
    check("SELECT * FROM (" + aggregateFilter + ") a CROSS JOIN (" + right + ") b",
      expectedRewrite = false)
  }

}
