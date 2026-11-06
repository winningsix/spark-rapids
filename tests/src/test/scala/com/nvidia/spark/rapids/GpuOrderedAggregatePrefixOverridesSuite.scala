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

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.SparkThrowable
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, IsNotNull, Literal}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, GlobalLimit, LocalRelation, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.types.IntegerType

/** Execute unchanged SQL in both arms, including cases where pruning must preserve ANSI errors. */
class GpuOrderedAggregatePrefixOverridesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: SparkSession = _
  private var joinedRowLimit = Int.MaxValue - 1
  private val maximumDecimal = "9" * 38

  override def beforeAll(): Unit = {
    super.beforeAll()
    val root = Files.createTempDirectory("ordered-prefix-suite")
    spark = SparkSession.builder().master("local[2]").appName("ordered-prefix-suite")
      .config("spark.ui.enabled", "false").config("spark.driver.host", "127.0.0.1")
      .config("spark.sql.warehouse.dir", root.resolve("warehouse").toString)
      .config("spark.sql.ansi.enabled", "true").config("spark.sql.shuffle.partitions", "2")
      .config("spark.rapids.sql.enabled", "false")
      .withExtensions(e => e.injectOptimizerRule(s => new Rule[LogicalPlan] {
        override def apply(plan: LogicalPlan): LogicalPlan =
          new GpuOrderedAggregatePrefixOverrides(s, joinedRowLimit).apply(plan)
      })).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    def register(name: String, sql: String): Unit = {
      val path = root.resolve(name).toString
      spark.sql(sql).coalesce(2).write.parquet(path)
      spark.read.parquet(path).createOrReplaceTempView(name)
    }
    register("prefix_dense", "SELECT cast(id as int) k, cast(id % 3 as int) g," +
      "cast(1 as decimal(7,2)) v FROM range(10000)")
    register("prefix_gapped", "SELECT cast(id as int) k,cast(id % 3 as int) g," +
      "cast(1 as decimal(7,2)) v FROM range(20000) WHERE id % 2 = 0")
    register("prefix_sparse", "SELECT k,g," +
      "cast(CASE WHEN k<5000 THEN -1 ELSE 1 END as decimal(7,2)) v FROM prefix_dense")
    register("prefix_ties", "SELECT cast(id % 2 as int) k,cast(id as int) g," +
      "cast(1 as decimal(7,2)) v FROM range(100)")
    register("prefix_nulls", "SELECT cast(k as int) k,cast(g as int) g," +
      "cast(v as decimal(7,2)) v FROM VALUES (NULL,1,2),(0,1,NULL)," +
      "(1,1,3),(1,1,-1),(2,1,4),(3,2,5) t(k,g,v)")
    register("prefix_all_null", "SELECT k,g,cast(NULL as decimal(7,2)) v FROM prefix_dense")
    register("prefix_empty", "SELECT * FROM prefix_dense WHERE false")
    register("prefix_dimension", "SELECT cast(id % 3 as int) g FROM range(6)")
    register("prefix_overflow", "SELECT cast(k as int) k,cast(g as int) g," +
      "cast(v as decimal(38,0)) v FROM VALUES " +
      s"(0,0,'1'),(9999,0,'$maximumDecimal'),(9999,0,'$maximumDecimal') t(k,g,v)")
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally super.afterAll()
  }

  private def sql(
      table: String, dimension: Boolean = false, emptyUnion: Boolean = false): String = {
    val source = if (dimension) {
      s"$table f JOIN prefix_dimension d ON f.g=d.g"
    } else {
      s"$table f"
    }
    val grouped = s"SELECT f.k,f.g,sum(v) s FROM $source GROUP BY f.k,f.g"
    val arm = if (emptyUnion) {
      s"$grouped UNION ALL SELECT k,g,sum(v) s FROM $table WHERE false GROUP BY k,g"
    } else {
      grouped
    }
    s"WITH a AS ($arm),b AS ($arm) SELECT a.k,a.g ag,b.g bg,a.s x,b.s y " +
      "FROM a JOIN b ON a.k=b.k WHERE a.s>0 AND b.s>0 " +
      "ORDER BY a.k,ag,bg,x,y LIMIT 5"
  }

  private def answer(frame: DataFrame): Either[String, Seq[Row]] = {
    try {
      Right(frame.collect().toSeq)
    } catch {
      case e: Exception =>
        val condition = Iterator.iterate[Throwable](e)(_.getCause).takeWhile(_ != null)
          .collectFirst { case t: SparkThrowable if t.getCondition != null &&
            (t.getCondition == "ARITHMETIC_OVERFLOW" ||
              t.getCondition.startsWith("NUMERIC_VALUE_OUT_OF_RANGE")) => t.getCondition }
        condition.map(Left(_)).getOrElse(throw e)
    }
  }

  private def compare(query: String, selected: Boolean, overflow: Boolean = false): Unit = {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val original = spark.sql(query)
    assert(!original.queryExecution.optimizedPlan.treeString.contains("ordered_prefix_rows"))
    val expected = answer(original)
    spark.conf.unset("spark.rapids.sql.enabled")
    val candidate = spark.sql(query)
    val plan = candidate.queryExecution.optimizedPlan.treeString
    assert(plan.contains("ordered_prefix_rows") == selected, plan)
    assert(candidate.schema == original.schema)
    assert(answer(candidate) == expected)
    assert(expected.isLeft == overflow)
  }

  for (aqe <- Seq(false, true)) {
    for (fixture <- Seq("dense", "sparse", "ties", "nulls", "all_null", "empty")) {
      test(s"complete groups and ordered output: $fixture, AQE=$aqe") {
        spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
        compare(sql(s"prefix_$fixture"), selected = true)
      }
    }
    test(s"key join: gapped peer domains preserve the leading joined rows, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      val query = sql("prefix_dense").replace("a AS (SELECT f.k,f.g,sum(v) s FROM prefix_dense",
        "a AS (SELECT f.k,f.g,sum(v) s FROM prefix_gapped")
      compare(query, selected = true)
    }
    test(s"key join: NULL-safe leading equality keeps the original plan, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(sql("prefix_nulls").replace("a.k=b.k", "a.k<=>b.k"), selected = false)
    }
    test(s"key join: leading output from the second peer preserves order, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(sql("prefix_dense").replace("SELECT a.k,", "SELECT b.k,")
        .replace("ORDER BY a.k,", "ORDER BY b.k,"), selected = true)
    }
    test(s"duplicate dimension matches retain aggregate multiplicity, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(sql("prefix_dense", dimension = true), selected = true)
    }
    test(s"dead UNION arms do not hide grouping keys, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(sql("prefix_dense", emptyUnion = true), selected = true)
    }
    test(s"failed exact row bound retains overflow outside the prefix, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      joinedRowLimit = 1
      try {
        compare(sql("prefix_overflow"), selected = true, overflow = true)
      } finally joinedRowLimit = Int.MaxValue - 1
    }
  }

  test("descending leading key rejects pruning") {
    compare(sql("prefix_dense").replace("ORDER BY a.k,", "ORDER BY a.k DESC,"),
      selected = false)
  }

  test("unbounded floating point SUM rejects pruning") {
    compare(sql("prefix_dense").replace("sum(v)", "sum(cast(v as double))"),
      selected = false)
  }

  test("unsafe decimal SUM rejects pruning without suppressing overflow") {
    compare(sql("prefix_overflow"), selected = false, overflow = true)
  }

  test("ANSI disabled leaves the original plan") {
    spark.conf.set("spark.sql.ansi.enabled", "false")
    try {
      compare(sql("prefix_dense"), selected = false)
    } finally spark.conf.set("spark.sql.ansi.enabled", "true")
  }

  test("key limit: uniqueness statistics survive copy and clone without changing maxRows") {
    val key = AttributeReference("key", IntegerType)()
    val input = Aggregate(Seq(key), Seq(key), Filter(IsNotNull(key), LocalRelation(key)))
    val limited = new GpuDistinctKeyPrefixLimit(Literal(4), input)
    val base = GlobalLimit(Literal(4), input)
    assert(!limited.equals(base) && !base.equals(limited))
    for (copy <- Seq(limited, limited.copy(Literal(4), input), limited.clone())) {
      assert(copy.isInstanceOf[GpuDistinctKeyPrefixLimit])
      assert(copy.maxRows.contains(4L))
      assert(copy.stats.attributeStats(key).distinctCount == copy.stats.rowCount)
      assert(copy.stats.attributeStats(key).nullCount.contains(BigInt(0)))
      assert(copy.stats.attributeStats(key).maxLen.isEmpty)
      assert(!copy.stats.isRuntime)
    }
    val alias = Alias(key, "renamed")()
    val renamed = limited.copy(Literal(4), Project(Seq(alias), input))
    assert(renamed.stats.attributeStats.contains(alias.toAttribute))
    assert(!renamed.stats.attributeStats.contains(key))
  }

  test("key limit: arbitrary or nullable inputs do not acquire uniqueness statistics") {
    val key = AttributeReference("key", IntegerType)()
    // A range-like physical relation must not be mistaken for unique simply from column NDV.
    val nullable = Aggregate(Seq(key), Seq(key), LocalRelation(key))
    val noProof = new GpuDistinctKeyPrefixLimit(Literal(4), nullable)
    assert(noProof.stats.attributeStats.isEmpty)
    val duplicates = spark.sql("SELECT g FROM prefix_dense").queryExecution.optimizedPlan
    val bounded = new GpuDistinctKeyPrefixLimit(Literal(4), duplicates)
    assert(bounded.stats.attributeStats.isEmpty)
  }
}
