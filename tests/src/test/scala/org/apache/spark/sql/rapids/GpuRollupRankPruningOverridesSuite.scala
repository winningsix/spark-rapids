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

import com.nvidia.spark.rapids.GpuRollupRankPruningOverrides
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.catalog.CatalogStatistics
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

class GpuRollupRankPruningOverridesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _
  private val fixtures = Seq(
    ("positive", 4000, "cast(id%17 as decimal(8,2))", false),
    ("null_keys", 4000, "cast(id%17 as decimal(8,2))", true),
    ("zero_ties", 500, "cast(0 as decimal(8,2))", true),
    ("all_null", 500, "cast(null as decimal(8,2))", true),
    ("sparse_nonnull", 500, "cast(IF(id%149=0,10,NULL) as decimal(8,2))", true),
    ("skew_parent", 240,
      "cast(IF(id>=200,NULL,IF(id<60,1000,1)) as decimal(8,2))", false),
    ("negative", 4000, "cast(IF(id%13=0,-100,id%17) as decimal(8,2))", true),
    ("few_groups", 9, "cast(id%17 as decimal(8,2))", true),
    ("empty", 0, "cast(id%17 as decimal(8,2))", true))

  private def query(table: String, limit: Int = 3): String = s"""SELECT * FROM (
    SELECT *,rank() OVER (PARTITION BY a ORDER BY s DESC NULLS LAST) rk FROM (
      SELECT a,b,c,d,e,sum(v) s FROM $table GROUP BY ROLLUP(a,b,c,d,e)
    ) g) r WHERE rk <= $limit"""

  private def estimates(table: String, rows: Long = 10000000L): Unit =
    spark.sessionState.catalog.alterTableStats(TableIdentifier(table),
      Some(CatalogStatistics(BigInt(rows) * 32, Some(BigInt(rows)))))

  override def beforeAll(): Unit = {
    super.beforeAll()
    val root = Files.createTempDirectory("rollup-rank-tests")
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("rollup-rank-tests").config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.sql.warehouse.dir", root.resolve("warehouse").toUri.toString)
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.sql.cbo.enabled", "true")
      .config("spark.sql.ansi.enabled", "true")
      .config("spark.rapids.sql.enabled", "false")
      .withExtensions(e => e.injectOptimizerRule(s => new GpuRollupRankPruningOverrides(s)))
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    fixtures.foreach { case (name, count, value, nullableKeys) =>
      val expressions = if (name == "skew_parent") {
        // Dominant fine groups must win even when many smaller parent groups exist.
        // Category 2 has only NULL measures while other categories keep the guard active.
        Seq("IF(id<60,0,IF(id>=200,2,IF(id%17=0,NULL,1)))",
          "IF(id<60,0,IF(id%19=0,NULL,id%3))", "IF(id<60,0,IF(id%23=0,NULL,id%5))",
          "id%4", "id%3")
      } else {
        Seq("id%4", "cast(id/4 as bigint)%7", "cast(id/28 as bigint)%19",
          "cast(id/532 as bigint)%11", "cast(id/5852 as bigint)%5")
      }
      val keys = expressions.zipWithIndex.map { case (expression, index) =>
        val key = ('a' + index).toChar
        val input = if (nullableKeys) s"IF(id%${13 + index * 2}=0,NULL,$expression)" else expression
        s"cast($input as int) $key"
      }.mkString(",")
      val table = "rr_" + name
      val path = root.resolve(table).toString
      spark.sql(s"SELECT $keys,$value v FROM range(0,$count,1,2)").write.parquet(path)
      spark.sql(s"CREATE TABLE $table USING parquet LOCATION '$path'")
      estimates(table)
    }
    val maximum = "99999999999999999999999999999999999999"
    for (negative <- Seq(false, true); cancellation <- Seq(false, true)) {
      val prefix = if (negative) "-" else ""
      val opposite = if (negative) "" else "-"
      val value = if (cancellation) {
        s"IF(id<2,'$prefix$maximum','$opposite$maximum')"
      } else s"IF(id=0,'$prefix$maximum','${prefix}1')"
      val rows = if (cancellation) 3 else 2
      val table = s"rr_overflow_${negative}_$cancellation"
      val path = root.resolve(table).toString
      spark.sql(s"SELECT 0 a,0 b,0 c,0 d,0 e,cast($value as decimal(38,0)) v " +
        s"FROM range(0,$rows,1,1)").write.parquet(path)
      spark.sql(s"CREATE TABLE $table USING parquet LOCATION '$path'")
      estimates(table)
    }
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally super.afterAll()
  }

  private def answer(sql: String): Either[String, Seq[String]] = {
    try Right(spark.sql(sql).collect().map(_.toString).sorted.toSeq) catch {
      case error: Exception =>
        val causes = Iterator.iterate[Throwable](error)(_.getCause).takeWhile(_ != null).toSeq
        val condition = causes.collectFirst {
          case e: org.apache.spark.SparkThrowable if e.getCondition != null => e.getCondition
        }.getOrElse(throw error)
        assert(condition == "ARITHMETIC_OVERFLOW", condition)
        Left(condition)
    }
  }

  private def check(sql: String, expectedSelection: Boolean = true): Unit = {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val expected = answer(sql)
    spark.conf.set("spark.rapids.sql.enabled", "true")
    val frame = spark.sql(sql)
    val plan = frame.queryExecution.optimizedPlan
    assert(plan.resolved, plan.treeString)
    assert(plan.toString.contains("rollup_nonnegative_bounded") == expectedSelection,
      plan.treeString)
    assert(new GpuRollupRankPruningOverrides(spark)(plan).fastEquals(plan), plan.treeString)
    assertResult(expected)(answer(sql))
  }

  fixtures.foreach { case (name, _, _, _) =>
    test(s"preserve $name rollup ranks and ties with AQE on and off") {
      spark.conf.set("spark.sql.ansi.enabled", "true")
      val limits = if (name == "skew_parent") Seq(3, 8, 100) else Seq(3, 100)
      for (aqe <- Seq(false, true); limit <- limits) {
        spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
        check(query("rr_" + name, limit))
      }
    }
  }

  test("preserve signed overflow and mixed-sign cancellation errors") {
    spark.conf.set("spark.sql.ansi.enabled", "true")
    for (aqe <- Seq(false, true); negative <- Seq(false, true);
         cancellation <- Seq(false, true)) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      check(query(s"rr_overflow_${negative}_$cancellation"))
    }
  }

  test("keep original rank sort and final limit output") {
    spark.conf.set("spark.sql.ansi.enabled", "true")
    check(query("rr_null_keys", 100) + " ORDER BY a,b,c,d,e,s,rk LIMIT 100")
  }

  test("preserve eight mixed-type rollup keys and a computed decimal measure") {
    spark.conf.set("spark.sql.ansi.enabled", "true")
    val sql = """SELECT * FROM (
      SELECT *,rank() OVER (PARTITION BY category ORDER BY s DESC NULLS LAST) rk FROM (
        SELECT category,cl,brand,product,yr,q,m,store,
          sum(coalesce(v*cast(quantity as decimal(20,0)),0)) s FROM (
            SELECT cast(a as string) category,cast(b as string) cl,cast(c as string) brand,
              cast(d as string) product,cast(e as bigint) yr,
              cast(coalesce(e,0)%4 as bigint) q,cast(coalesce(e,0)%12 as bigint) m,
              cast(coalesce(b,0)%3 as string) store,coalesce(e,1) quantity,v
            FROM rr_null_keys
          ) input GROUP BY ROLLUP(category,cl,brand,product,yr,q,m,store)
        ) grouped) ranked WHERE rk<=100
      ORDER BY category,cl,brand,product,yr,q,m,store,s,rk LIMIT 100"""
    for (aqe <- Seq(false, true)) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      check(sql)
    }
  }

  test("skip unsupported order, aggregate, partition and grouping-set shapes") {
    spark.conf.set("spark.sql.ansi.enabled", "true")
    val sql = query("rr_positive")
    for (unsupported <- Seq(
        sql.replace("DESC NULLS LAST", "ASC NULLS FIRST"),
        sql.replace("DESC NULLS LAST", "DESC NULLS FIRST"),
        sql.replace("rank()", "dense_rank()"),
        sql.replace("PARTITION BY a", "PARTITION BY b"),
        sql.replace("sum(v)", "sum(DISTINCT v)"),
        sql.replace("sum(v)", "try_sum(v)"),
        sql.replace("sum(v)", "sum(cast(v as double))"),
        sql.replace("sum(v) s", "sum(v) s,count(*) n"),
        sql.replace("ROLLUP(a,b,c,d,e)", "GROUPING SETS ((a,b,c,d,e),(a))"),
        sql.replace("sum(v) s", "sum(v) s,grouping_id() gid"))) {
      check(unsupported, expectedSelection = false)
    }
  }

  test("skip legacy SUM semantics and small estimated input") {
    spark.conf.set("spark.sql.ansi.enabled", "false")
    try check(query("rr_positive"), expectedSelection = false)
    finally spark.conf.set("spark.sql.ansi.enabled", "true")
    estimates("rr_positive", 4000)
    try check(query("rr_positive"), expectedSelection = false)
    finally estimates("rr_positive")
  }

  test("skip disabled GPU SQL, nondeterministic input and unknown cardinality") {
    spark.conf.set("spark.sql.ansi.enabled", "true")
    def selected(sql: String): Boolean =
      spark.sql(sql).queryExecution.optimizedPlan.toString.contains("rollup_nonnegative_bounded")
    spark.conf.set("spark.rapids.sql.enabled", "false")
    assert(!selected(query("rr_positive")))
    spark.conf.set("spark.rapids.sql.enabled", "true")
    val nondeterministic = query("rr_positive").replace("FROM rr_positive GROUP",
      "FROM rr_positive WHERE rand(42)<0.5 GROUP")
    assert(!selected(nondeterministic))
    spark.sessionState.catalog.alterTableStats(TableIdentifier("rr_positive"),
      Some(CatalogStatistics(BigInt(320000000))))
    try assert(!selected(query("rr_positive"))) finally estimates("rr_positive")
  }
}
