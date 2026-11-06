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
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, AttributeSet, Divide, ScalarSubquery}
import org.apache.spark.sql.catalyst.plans.{Inner, LeftOuter}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, JoinHint, LocalRelation, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.types._

/** Run the real optimizer hook with unchanged SQL and exact error/guard boundary checks. */
class GpuOuterAggregatePrefixOverridesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: SparkSession = _
  private var rowLimit = 32
  private var span = 4
  private val modes = Seq("dense", "sparse", "missing", "duplicates", "null_keys",
    "all_null", "empty", "overflow_tail", "zero_divisor_tail", "gapped_keys", "maximum_keys",
    "minimum_keys")

  override def beforeAll(): Unit = {
    super.beforeAll()
    val root = Files.createTempDirectory("outer-prefix-suite")
    spark = SparkSession.builder().master("local[2]").appName("outer-prefix-suite")
      .config("spark.ui.enabled", "false").config("spark.driver.host", "127.0.0.1")
      .config("spark.sql.warehouse.dir", root.resolve("warehouse").toString)
      .config("spark.sql.ansi.enabled", "true").config("spark.sql.shuffle.partitions", "2")
      .config("spark.rapids.sql.enabled", "false")
      .withExtensions(e => e.injectOptimizerRule(s => new Rule[LogicalPlan] {
        override def apply(plan: LogicalPlan): LogicalPlan =
          new GpuOuterAggregatePrefixOverrides(s, rowLimit, span).apply(plan)
      })).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val schema = StructType(Seq(StructField("channel", IntegerType, false),
      StructField("k", IntegerType, true), StructField("item", IntegerType, false),
      StructField("qty", LongType, true), StructField("price", DecimalType(7, 2), true)))
    modes.foreach { mode =>
      val path = root.resolve(mode).toString
      spark.createDataFrame(spark.sparkContext.parallelize(rows(mode), 2), schema)
        .write.parquet(path)
      spark.read.parquet(path).createOrReplaceTempView("outer_" + mode)
    }
    val returnsSchema = StructType(Seq(StructField("rk", IntegerType, true),
      StructField("ri", IntegerType, false), StructField("marker", IntegerType, true)))
    val returned = (for (k <- 0 until 3; item <- 0 until 2) yield Row(k, item, null)) ++
      Seq(Row(0, 0, null), Row(null, 0, null), Row(null, 0, null), Row(100, 0, null))
    val returnPath = root.resolve("returns").toString
    spark.createDataFrame(spark.sparkContext.parallelize(returned, 2), returnsSchema)
      .write.parquet(returnPath)
    spark.read.parquet(returnPath).createOrReplaceTempView("outer_returns")
    val dimensionPath = root.resolve("dimension").toString
    spark.createDataFrame(spark.sparkContext.parallelize(
      (0 until 3).flatMap(ch => Seq(Row(ch), Row(ch))), 2),
      StructType(Seq(StructField("dc", IntegerType, false)))).write.parquet(dimensionPath)
    spark.read.parquet(dimensionPath).createOrReplaceTempView("outer_dimension")
    spark.sql("SELECT f.* FROM outer_dense f LEFT JOIN outer_returns r " +
      "ON f.k=r.rk AND f.item=r.ri JOIN outer_dimension d ON f.channel=d.dc " +
      "WHERE r.rk IS NULL").createOrReplaceTempView("outer_unreturned_inner")
    for (mode <- Seq("dense", "overflow_tail")) {
      spark.sql(s"SELECT f.* FROM outer_$mode f LEFT JOIN outer_returns r " +
        "ON f.k=r.rk AND f.item=r.ri WHERE r.rk IS NULL")
        .createOrReplaceTempView("outer_unreturned_" + mode)
    }
  }
  override def afterAll(): Unit = {
    try { if (spark != null) spark.stop() } finally super.afterAll()
  }
  private def query(table: String): String = {
    val groups = (0 until 3).map { ch =>
      s"g$ch AS (SELECT k,item,sum(qty) q,sum(price) p FROM $table " +
        s"WHERE channel=$ch GROUP BY k,item)"
    }.mkString(",")
    s"WITH $groups SELECT a.k,a.item,a.q sq," +
      "coalesce(b.q,0)+coalesce(c.q,0) oq," +
      "round(a.q/(coalesce(b.q,0)+coalesce(c.q,0)),2) ratio," +
      "a.p sp,coalesce(b.p,0)+coalesce(c.p,0) op FROM g0 a " +
      "LEFT JOIN g1 b ON a.k=b.k AND a.item=b.item " +
      "LEFT JOIN g2 c ON a.k=c.k AND a.item=c.item " +
      "WHERE coalesce(b.q,0)>0 OR coalesce(c.q,0)>0 " +
      "ORDER BY a.k,a.item,sq DESC,oq,ratio,sp,op LIMIT 5"
  }
  private def answer(frame: DataFrame): Either[String, Seq[Row]] = {
    try Right(frame.collect().toSeq) catch {
      case e: Exception =>
        val condition = Iterator.iterate[Throwable](e)(_.getCause).takeWhile(_ != null)
          .collectFirst { case t: SparkThrowable if Set("ARITHMETIC_OVERFLOW", "DIVIDE_BY_ZERO")
            .contains(t.getCondition) => t.getCondition }
        condition.map(Left(_)).getOrElse(throw e)
    }
  }
  private def compare(sql: String, selected: Boolean, error: Boolean = false): DataFrame = {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val original = spark.sql(sql)
    assert(!original.queryExecution.optimizedPlan.treeString.contains("outer_prefix_rows"))
    val expected = answer(original)
    spark.conf.unset("spark.rapids.sql.enabled")
    val candidate = spark.sql(sql)
    val plan = candidate.queryExecution.optimizedPlan.treeString
    assert(plan.contains("outer_prefix_rows") == selected, plan)
    assert(candidate.schema == original.schema)
    assert(answer(candidate) == expected)
    assert(expected.isLeft == error)
    candidate
  }
  private def safetyGuards(frame: DataFrame): Seq[Boolean] = {
    val plans = frame.queryExecution.optimizedPlan.flatMap { node =>
      node.expressions.flatMap(_.collect {
        case s: ScalarSubquery if s.plan.output.exists(_.name == "outer_prefix_guard") => s.plan
      })
    }
    assert(plans.nonEmpty)
    plans.groupBy(_.canonicalized).values.map { equivalent =>
      val rows = spark.sessionState.executePlan(equivalent.head).executedPlan.executeCollect()
      assert(rows.length == 1)
      rows.head.getStruct(0, 2).getBoolean(0)
    }.toSeq
  }

  for (aqe <- Seq(false, true)) {
    test(s"normalize anti rejection above a duplicating inner dimension join, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      spark.conf.set("spark.rapids.sql.enabled", "false")
      try {
        val original = spark.sql("SELECT f.k,f.item,f.qty FROM outer_null_keys f " +
          "LEFT JOIN outer_returns r ON f.k=r.rk AND f.item=r.ri " +
          "WHERE f.channel=0 AND r.rk IS NULL").queryExecution.optimizedPlan
        val dim = LocalRelation(Seq(AttributeReference("dim", IntegerType, false)()),
          Seq(InternalRow(1), InternalRow(2)))
        var lifted = 0
        val input = original.transformUp {
          case Filter(condition, j: Join) if j.joinType == LeftOuter =>
            lifted += 1
            Filter(condition, Join(j, dim, Inner, None, JoinHint.NONE))
        }
        assert(lifted == 1 && input.resolved)
        val rule = new GpuOuterAggregatePrefixOverrides(spark, 64, 12)
        val widened = rule.guardInputSuperset(input, AttributeSet(input.output))
        assert(widened.resolved && widened.output == input.output)
        assert(!widened.treeString.contains("LeftOuter"), widened.treeString)
        assert(widened.treeString.contains("Join Inner"), widened.treeString)
        assert(spark.sessionState.executePlan(input).executedPlan.executeCollect().length == 38)
        assert(spark.sessionState.executePlan(widened).executedPlan.executeCollect().length == 50)
      } finally spark.conf.unset("spark.rapids.sql.enabled")
    }
    for (bound <- Seq(40, 48)) {
      test(s"superset guard retains inner-join multiplicity: bound=$bound, AQE=$aqe") {
        spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
        rowLimit = bound
        try {
          val frame = compare(query("outer_unreturned_inner"), selected = true)
          assert(safetyGuards(frame).forall(identity) == (bound == 48))
        } finally rowLimit = 32
      }
    }
    for (mode <- Seq("outer_anti", "explicit_anti", "nullable_marker", "null_safe")) {
      test(s"guard superset preserves anti and outer multiplicities: $mode, AQE=$aqe") {
        spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
        spark.conf.set("spark.rapids.sql.enabled", "false")
        try {
          val join = if (mode == "explicit_anti") "LEFT ANTI JOIN" else "LEFT JOIN"
          val equal = if (mode == "null_safe") "<=>" else "="
          val predicate = if (mode == "explicit_anti") "" else {
            if (mode == "nullable_marker") " AND r.marker IS NULL" else " AND r.rk IS NULL"
          }
          val input = spark.sql(s"SELECT f.k,f.item,f.qty FROM outer_null_keys f $join " +
            s"outer_returns r ON f.k $equal r.rk AND f.item=r.ri " +
            s"WHERE f.channel=0$predicate").queryExecution.optimizedPlan
          val rule = new GpuOuterAggregatePrefixOverrides(spark, 32, 4)
          val widened = rule.guardInputSuperset(input, AttributeSet(input.output))
          assert(widened.resolved && widened.output == input.output)
          def counts(plan: LogicalPlan) = spark.sessionState.executePlan(plan).executedPlan
            .executeCollect().map(_.copy()).groupBy(identity).view.mapValues(_.length).toMap
          val original = counts(input)
          val superset = counts(widened)
          assert(original.forall { case (row, count) => superset.getOrElse(row, 0) >= count })
          if (mode == "outer_anti" || mode == "explicit_anti") {
            assert(!widened.treeString.contains("Join"), widened.treeString)
            assert(superset.values.sum > original.values.sum)
          } else {
            assert(widened.treeString.contains("LeftOuter"), widened.treeString)
            assert(superset == original)
          }
        } finally spark.conf.unset("spark.rapids.sql.enabled")
      }
    }
    for (bound <- Seq(20, 24)) {
      test(s"anti input guard uses stronger pre-return row bound: $bound, AQE=$aqe") {
        spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
        rowLimit = bound
        try {
          val frame = compare(query("outer_unreturned_dense"), selected = true)
          assert(safetyGuards(frame).forall(identity) == (bound == 24))
        } finally rowLimit = 32
      }
    }
    test(s"unsafe returned quantity conservatively retains original fallback, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      val frame = compare(query("outer_unreturned_overflow_tail"), selected = true)
      assert(!safetyGuards(frame).forall(identity))
    }
    for (mode <- modes; budget <- Seq(4, 12)) {
      test(s"outer prefix ordered rows and tail errors: $mode, span=$budget, AQE=$aqe") {
        spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
        span = budget
        val error = mode == "overflow_tail" || mode == "zero_divisor_tail"
        val frame = compare(query("outer_" + mode), selected = true, error = error)
        if (mode == "dense" || error) {
          assert(safetyGuards(frame).forall(identity) == !error)
        }
      }
    }
    for (bound <- Seq(1, 23, 24)) {
      test(s"exact joined input row guard: bound=$bound, AQE=$aqe") {
        spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
        rowLimit = bound
        try {
          val frame = compare(query("outer_dense"), selected = true)
          assert(safetyGuards(frame).forall(identity) == (bound == 24))
        } finally rowLimit = 32
      }
    }
    test(s"nullable right-side ordering key cannot prune the preserved input, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(query("outer_missing").replace("ORDER BY a.k,", "ORDER BY b.k,a.k,"),
        selected = false)
    }
    test(s"NULL-safe equality is outside the prefix proof, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(query("outer_null_keys").replace("a.k=b.k", "a.k<=>b.k")
        .replace("a.k=c.k", "a.k<=>c.k"), selected = false)
    }
    test(s"descending ordering keeps the original query, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(query("outer_dense").replace("ORDER BY a.k,", "ORDER BY a.k DESC,"),
        selected = false)
    }
    test(s"unbounded floating SUM keeps the original query, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      compare(query("outer_dense").replace("sum(qty)", "sum(cast(qty as double))"),
        selected = false)
    }
    test(s"ANSI disabled keeps the original query, AQE=$aqe") {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      spark.conf.set("spark.sql.ansi.enabled", "false")
      try compare(query("outer_dense"), selected = false)
      finally spark.conf.set("spark.sql.ansi.enabled", "true")
    }
  }
  test("a positivity filter cannot justify division evaluated below itself") {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    try {
      val original = spark.sql(query("outer_dense")).queryExecution.optimizedPlan
      val rule = new GpuOuterAggregatePrefixOverrides(spark, 32, 4)
      assert(rule.rewrite(original).nonEmpty)
      var moved = 0
      val unsafe = original.transformDown {
        case p @ Project(expressions, f: Filter) if expressions.exists(_.exists {
            case _: Divide => true
            case _ => false
        }) =>
          moved += 1
          val extra = (f.condition.references -- p.outputSet).toSeq
          Project(p.output, Filter(f.condition, Project(expressions ++ extra, f.child)))
      }
      assert(moved == 1 && unsafe.resolved)
      assert(rule.rewrite(unsafe).isEmpty)
    } finally spark.conf.unset("spark.rapids.sql.enabled")
  }

  private def rows(mode: String): Seq[Row] = {
      if (mode == "empty") Seq.empty else {
        val base = for (k <- 0 until 12; item <- 0 until 2; ch <- 0 until 3
            if mode != "missing" || ch == 0 || ch == (k % 2 + 1)) yield {
          val quantity: java.lang.Long = if (mode == "all_null" && ch != 0) null
            else if (mode == "sparse" && k < 8 && ch != 0) 0L
            else if (ch == 0) 100L else 1L
          Row(ch, k, item, quantity, new java.math.BigDecimal("1.25"))
        }
        mode match {
          case "gapped_keys" => base.map(r => Row(r.getInt(0), r.getInt(1) * 1000,
            r.getInt(2), r.get(3), r.get(4)))
          case "minimum_keys" => base.map(r => Row(r.getInt(0),
            Int.MinValue + r.getInt(1), r.getInt(2), r.get(3), r.get(4)))
          case "maximum_keys" => base.map(r => Row(r.getInt(0),
            Int.MaxValue - 11 + r.getInt(1), r.getInt(2), r.get(3), r.get(4)))
          case "duplicates" => base ++ base.take(9)
          case "null_keys" => base ++ (0 until 3).map(ch =>
            Row(ch, null, 0, Long.box(1), new java.math.BigDecimal("1.25")))
          case "overflow_tail" => base ++ Seq(
            Row(0, 100, 0, Long.box(Long.MaxValue), new java.math.BigDecimal("1.25")),
            Row(0, 100, 0, Long.box(1), new java.math.BigDecimal("1.25")),
            Row(1, 100, 0, Long.box(1), new java.math.BigDecimal("1.25")))
          case "zero_divisor_tail" => base ++ Seq(
            Row(0, 100, 0, Long.box(1), new java.math.BigDecimal("1.25")),
            Row(1, 100, 0, Long.box(1), new java.math.BigDecimal("1.25")),
            Row(2, 100, 0, Long.box(-1), new java.math.BigDecimal("1.25")))
          case _ => base
        }
      }
    }

}
