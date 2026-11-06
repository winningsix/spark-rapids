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

import com.nvidia.spark.rapids.GpuMultiKeySemiJoinDistinctOverrides
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.{And, AttributeMap, EqualNullSafe, EqualTo, Expression, LessThan, Literal, Rand}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftAnti, LeftOuter, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, ColumnStat, Join, JoinHint, LogicalPlan, Statistics}
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

class GpuMultiKeySemiJoinDistinctOverridesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _
  private var rule: GpuMultiKeySemiJoinDistinctOverrides = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("multi-key-semi-tests")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.hadoop.fs.defaultFS", "file:///")
      .config("spark.sql.warehouse.dir", Files.createTempDirectory("multi-key-semi").toUri.toString)
      .config("spark.sql.catalogImplementation", "in-memory")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .config("spark.sql.cbo.enabled", "true")
      .config("spark.sql.autoBroadcastJoinThreshold", "16777216")
      .config("spark.rapids.sql.enabled", "true")
      .withExtensions { extensions =>
        extensions.injectOptimizerRule(session => new GpuMultiKeySemiJoinDistinctOverrides(session))
      }
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    rule = new GpuMultiKeySemiJoinDistinctOverrides(spark)
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally {
      super.afterAll()
    }
  }

  private def build(sql: String = "SELECT * FROM VALUES (1, 2, 3), (1, 2, 3), " +
      "(2, 3, 4), (NULL, 2, 3) t(k, v, w)",
      rows: BigInt = BigInt(2000000), distinct: Option[BigInt] = Some(BigInt(8)),
      bytes: BigInt = BigInt(536870912)): LogicalPlan = {
    val plan = spark.sql(sql).queryExecution.optimizedPlan
    // Synthetic estimates trigger the cost gate on small, explicitly checked test data.
    val estimates = Some(Statistics(bytes, Some(rows),
      AttributeMap(plan.output.map(_ -> ColumnStat(distinctCount = distinct)))))
    plan.getClass.getMethod("statsCache_$eq", classOf[Option[_]]).invoke(plan, estimates)
    plan
  }

  private def join(right: LogicalPlan): Join = {
    val left = spark.sql(
      "SELECT * FROM VALUES (1, 2, 3), (1, 2, 3), (2, 3, 4), (3, 2, 3), " +
      "(NULL, 2, 3), (NULL, 2, 3), (1, NULL, 3), (NULL, NULL, NULL) t(k, v, w)")
      .queryExecution.optimizedPlan
    Join(left, right, LeftSemi, Some(left.output.zip(right.output).map { case (l, r) =>
      EqualTo(l, r): Expression
    }.reduce(And)), JoinHint.NONE)
  }

  test("large duplicate-heavy build is deduplicated without changing output identities") {
    val original = join(build())
    val candidate = rule(original).asInstanceOf[Join]
    assert(candidate.right.isInstanceOf[Aggregate])
    assert(candidate.output == original.output)
    assert(candidate.right.output == original.right.output)
    assert(rule(candidate) == candidate)
  }

  test("unknown, small and high-cardinality builds retain their original plans") {
    for (right <- Seq(build(distinct = None), build(rows = 100),
        build(distinct = Some(BigInt(1000000))), build(bytes = 1024),
        build("SELECT * FROM VALUES (1), (1) t(k)"),
        build("SELECT * FROM VALUES (1, 2, 3, 4, 5) t(a, b, c, d, e)"),
        build("SELECT * FROM VALUES ('a', 'b'), ('a', 'b') t(k, v)"))) {
      val original = join(right)
      assert(rule(original) == original)
    }
  }

  test("other join types and nondeterministic predicates are not rewritten") {
    val original = join(build())
    for (kind <- Seq(Inner, LeftOuter, LeftAnti)) {
      val plan = original.copy(joinType = kind)
      assert(rule(plan) == plan)
    }
    val random = original.copy(condition = Some(EqualTo(Rand(0L), Literal(0.5))))
    assert(rule(random) == random)
  }

  test("CPU mode and disabled CBO do not receive the rewrite") {
    val original = join(build())
    for (key <- Seq("spark.rapids.sql.enabled", "spark.sql.cbo.enabled")) {
      spark.conf.set(key, "false")
      try {
        assert(rule(original) == original)
      } finally {
        spark.conf.set(key, "true")
      }
    }
  }

  test("Spark execution preserves NULL semantics, empty builds and duplicate probe rows") {
    val inputs = Seq(
      "SELECT * FROM VALUES (1, 2, 3), (1, 2, 3), (2, 3, 4), (NULL, 2, 3) t(k, v, w)",
      "SELECT CAST(NULL AS INT) k, CAST(NULL AS INT) v, CAST(NULL AS INT) w WHERE false",
      "SELECT * FROM VALUES (CAST(NULL AS INT), CAST(NULL AS INT), " +
        "CAST(NULL AS INT)), (NULL, NULL, NULL) t(k, v, w)",
      "SELECT * FROM VALUES (1, NULL, 3), (1, NULL, 3), (2, 3, 4) t(k, v, w)")
    for (sql <- inputs; nullSafe <- Seq(false, true); residual <- Seq(false, true)) {
      val base = join(build(sql))
      val equality: Expression = if (nullSafe) {
        base.left.output.zip(base.right.output).map { case (l, r) =>
          EqualNullSafe(l, r): Expression
        }.reduce(And)
      } else {
        base.condition.get
      }
      val condition = if (residual) {
        And(equality, LessThan(base.right.output.head, Literal(3)))
      } else {
        equality
      }
      val original = base.copy(condition = Some(condition))
      val candidate = rule(original)
      assert(candidate.asInstanceOf[Join].right.isInstanceOf[Aggregate])
      def rows(plan: LogicalPlan) = TrampolineConnectShims.createDataFrame(spark, plan)
        .collect().toSeq.groupBy(identity).map { case (row, copies) => row -> copies.size }
      assert(rows(candidate) == rows(original), s"sql=$sql nullSafe=$nullSafe residual=$residual")
    }
  }

  test("two and four integral keys are eligible, and already distinct builds are unchanged") {
    for (sql <- Seq("SELECT * FROM VALUES (1, 2), (1, 2) t(k, v)",
        "SELECT * FROM VALUES (1L, CAST(2 AS SMALLINT), CAST(3 AS TINYINT), 4) t(a, b, c, d)")) {
      val right = build(sql)
      assert(rule(join(right)).asInstanceOf[Join].right.isInstanceOf[Aggregate])
      val distinct = Aggregate(right.output, right.output, right)
      val original = join(distinct)
      assert(rule(original) == original)
    }
  }

  test("unknown row counts and all-null domains are handled conservatively") {
    val unknown = build()
    unknown.getClass.getMethod("statsCache_$eq", classOf[Option[_]]).invoke(unknown,
      Some(unknown.stats.copy(rowCount = None)))
    val original = join(unknown)
    assert(rule(original) == original)
    val nulls = build("SELECT * FROM VALUES (CAST(NULL AS INT), CAST(NULL AS INT), " +
      "CAST(NULL AS INT)), (NULL, NULL, NULL) t(k, v, w)", distinct = Some(BigInt(0)))
    assert(rule(join(nulls)).asInstanceOf[Join].right.isInstanceOf[Aggregate])
  }

  test("optimizer extension preserves tuple IN, NOT IN and projected three-valued results") {
    spark.sql("CREATE TABLE multi_semi_build USING parquet AS " +
      "SELECT IF(id % 16 = 0, NULL, CAST(id % 8 AS INT)) k, " +
      "IF(id % 7 = 0, NULL, CAST(id % 3 AS INT)) v, " +
      "CAST(id % 2 AS INT) w FROM range(2000000)")
    try {
      spark.sql("ANALYZE TABLE multi_semi_build COMPUTE STATISTICS FOR ALL COLUMNS")
      val probe = "VALUES (1, 1, 1), (1, 1, 1), (7, 2, 1), (9, 1, 1), " +
        "(NULL, 1, 1), (1, NULL, 1), (NULL, NULL, NULL) probe(k, v, w)"
      val in = "(k, v, w) IN (SELECT k, v, w FROM multi_semi_build)"
      for (sql <- Seq(s"SELECT * FROM $probe WHERE $in",
          s"SELECT * FROM $probe WHERE NOT ($in)", s"SELECT *, $in AS found FROM $probe")) {
        spark.conf.set("spark.rapids.sql.enabled", "false")
        val baseline = spark.sql(sql)
        val expected = baseline.collect().toSeq.sortBy(_.toString)
        assert(!baseline.queryExecution.optimizedPlan.exists(_.isInstanceOf[Aggregate]))
        spark.conf.set("spark.rapids.sql.enabled", "true")
        val candidate = spark.sql(sql)
        assert(candidate.queryExecution.optimizedPlan.exists(_.isInstanceOf[Aggregate]),
          candidate.queryExecution.optimizedPlan.toString)
        assert(candidate.collect().toSeq.sortBy(_.toString) == expected)
      }
    } finally {
      spark.conf.set("spark.rapids.sql.enabled", "true")
      spark.sql("DROP TABLE multi_semi_build")
    }
  }
}
