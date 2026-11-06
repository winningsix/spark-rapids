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

import com.nvidia.spark.rapids.GpuSelfJoinExistsOverrides
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Min}
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

class GpuSelfJoinExistsOverridesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _
  private val pairs = "SELECT a.k FROM exists_build a JOIN exists_build b " +
    "ON a.k = b.k AND a.v <> b.v"

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("self-exists-tests")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.hadoop.fs.defaultFS", "file:///")
      .config("spark.sql.warehouse.dir", Files.createTempDirectory("self-exists").toUri.toString)
      .config("spark.sql.catalogImplementation", "in-memory")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .withExtensions { extensions =>
        extensions.injectOptimizerRule(session => new GpuSelfJoinExistsOverrides(session))
      }.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.sql("SELECT * FROM VALUES (1), (1), (2), (3), (4), (5), (6), " +
      "(CAST(NULL AS INT)) t(k)").createOrReplaceTempView("exists_probe")
    spark.sql("SELECT * FROM VALUES (1, 1), (1, 1), (1, 2), (2, 1), " +
      "(2, CAST(NULL AS INT)), (3, NULL), (CAST(NULL AS INT), 1), (NULL, 2), " +
      "(4, -2147483648), (4, 2147483647), (5, 1), (5, 1) t(k, v)")
      .createOrReplaceTempView("exists_build")
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally {
      super.afterAll()
    }
  }

  private def check(sql: String, expectedRewrite: Option[Boolean]): Unit = {
    def rows(df: TrampolineConnectShims.DataFrame) = df.collect().toSeq.groupBy(identity)
      .map { case (row, copies) => row -> copies.size }
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val expected = rows(spark.sql(sql))
    spark.conf.set("spark.rapids.sql.enabled", "true")
    val candidate = spark.sql(sql)
    val plan = candidate.queryExecution.optimizedPlan
    expectedRewrite.foreach { expectedMin =>
      val hasMin = plan.exists(_.expressions.exists(_.exists {
        case expression: AggregateExpression => expression.aggregateFunction.isInstanceOf[Min]
        case _ => false
      }))
      assert(hasMin == expectedMin, s"$sql\n$plan")
    }
    assert(rows(candidate) == expected, sql)
    assert(new GpuSelfJoinExistsOverrides(spark)(plan).fastEquals(plan))
  }

  test("IN, NOT IN and projected membership preserve NULLs, duplicates and integer extremes") {
    for (sql <- Seq(s"SELECT k FROM exists_probe WHERE k IN ($pairs)",
        s"SELECT k FROM exists_probe WHERE k NOT IN ($pairs)",
        s"SELECT k, k IN ($pairs) found FROM exists_probe")) {
      check(sql, Some(true))
    }
  }

  test("an alias or either self-join side can supply the selected key") {
    for (key <- Seq("a.k AS key", "b.k AS key")) {
      val query = "SELECT " + key + " FROM exists_build a JOIN exists_build b " +
        "ON b.k = a.k AND b.v <> a.v"
      check(s"SELECT k FROM exists_probe WHERE k IN ($query)", Some(true))
    }
  }

  test("set equivalence propagates through another inner join") {
    val query = s"SELECT p.k FROM exists_probe p JOIN ($pairs) eligible ON p.k = eligible.k"
    check(s"SELECT k FROM exists_probe WHERE k IN ($query)", Some(true))
  }

  test("a reordered third relation needs only the self-join key") {
    for (from <- Seq(
        "exists_probe p JOIN exists_build a ON p.k = a.k JOIN exists_build b " +
          "ON a.k = b.k AND a.v <> b.v",
        "exists_build a JOIN exists_probe p ON p.k = a.k JOIN exists_build b " +
          "ON a.k = b.k AND a.v <> b.v",
        "exists_build b JOIN (exists_probe p JOIN exists_build a ON p.k = a.k) " +
          "ON a.k = b.k AND a.v <> b.v",
        "exists_build b JOIN (exists_build a JOIN exists_probe p ON p.k = a.k) " +
          "ON a.k = b.k AND a.v <> b.v")) {
      check(s"SELECT k FROM exists_probe WHERE k IN (SELECT p.k FROM $from)", Some(true))
    }
    val needsValue = "SELECT p.k FROM exists_probe p JOIN exists_build a " +
      "ON p.k = a.k AND p.k < a.v JOIN exists_build b ON a.k = b.k AND a.v <> b.v"
    check(s"SELECT k FROM exists_probe WHERE k IN ($needsValue)", Some(false))
  }

  test("empty and NULL-only self joins preserve membership truth values") {
    for (condition <- Seq("false", "k IS NULL", "v IS NULL")) {
      val source = s"(SELECT * FROM exists_build WHERE $condition)"
      val query = s"SELECT a.k FROM $source a JOIN $source b ON a.k = b.k AND a.v <> b.v"
      check(s"SELECT k, k IN ($query) found FROM exists_probe", None)
    }
  }

  test("aggregation, limits and ordinary joins retain multiplicity") {
    check(s"SELECT k FROM exists_probe WHERE k IN (SELECT k FROM ($pairs) t " +
      "GROUP BY k HAVING count(*) >= 3)", Some(false))
    check(s"SELECT k FROM exists_probe WHERE k IN ($pairs LIMIT 1)", Some(false))
    check(pairs, Some(false))
  }

  test("different sources, extra conditions and non-integral values are not rewritten") {
    val sources = Seq(
      "SELECT a.k FROM exists_build a JOIN (SELECT * FROM exists_build WHERE v > 0) b " +
        "ON a.k = b.k AND a.v <> b.v",
      pairs + " AND a.v > 0",
      "SELECT a.k FROM (SELECT k, CAST(v AS DOUBLE) v FROM exists_build) a JOIN " +
        "(SELECT k, CAST(v AS DOUBLE) v FROM exists_build) b ON a.k = b.k AND a.v <> b.v")
    for (query <- sources) check(s"SELECT k FROM exists_probe WHERE k IN ($query)", Some(false))
  }
}
