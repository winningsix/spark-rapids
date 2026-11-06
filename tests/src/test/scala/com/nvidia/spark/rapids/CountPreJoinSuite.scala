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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Count, Sum}
import org.apache.spark.sql.catalyst.plans.LeftOuter
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, LogicalQueryStage}
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

class CountPreJoinSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _
  private val enabled = new AtomicBoolean(false)
  private val applied = new AtomicInteger()
  private val from = " FROM cp_fact f JOIN cp_dim d ON f.k=d.k AND f.g=d.g "
  private val query = "SELECT d.label, f.bucket, count(*) AS n, count(f.v) AS valid" +
    from + "GROUP BY d.label, f.bucket"

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("count-prejoin-tests").config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.hadoop.fs.defaultFS", "file:///")
      .config("spark.sql.warehouse.dir", Files.createTempDirectory("count-prejoin").toUri.toString)
      .config("spark.sql.cbo.enabled", "true")
      .config("spark.sql.cbo.joinReorder.enabled", "true")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .config("spark.sql.ansi.enabled", "true")
      .withExtensions { extensions =>
        extensions.injectPlannerStrategy { _ => new SparkStrategy {
          private val delegate = new GpuCountPreJoinStrategy
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
    spark.sql("CREATE TABLE cp_fact USING parquet PARTITIONED BY(g) AS SELECT " +
      "CASE WHEN id % 13 = 0 THEN NULL ELSE cast(id % 6 as int) END AS k," +
      "cast(id % 2 as int) AS g,cast(id % 3 as int) AS bucket," +
      "CASE WHEN id % 6 = 3 OR id % 17 = 0 THEN NULL " +
      "ELSE cast((id % 17 - 8) * 0.25 as decimal(18,2)) END AS v FROM range(2400)")
    spark.sql("CREATE TABLE cp_dim USING parquet AS SELECT * FROM VALUES " +
      "(0,0,'a'),(0,0,'a'),(0,0,'b'),(0,0,'aa'),(0,0,'ab')," +
      "(1,1,'a'),(2,0,NULL),(3,1,'c')," +
      "(4,0,'unused'),(NULL,0,'null-key') t(k,g,label)")
    for (table <- Seq("cp_fact", "cp_dim")) {
      spark.sql(s"ANALYZE TABLE $table COMPUTE STATISTICS FOR ALL COLUMNS")
    }
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally super.afterAll()
  }

  private def eligible(): Aggregate = spark.sql(query).queryExecution.optimizedPlan.collectFirst {
    case a: Aggregate if GpuCountPreJoinReorder.rewrite(a).isDefined => a
  }.getOrElse(fail("Expected the many-row fact input to select pre-join aggregation"))

  private def check(sql: String): Unit = {
    enabled.set(false)
    val expected = spark.sql(sql).collect().map(_.toString).sorted.toSeq
    applied.set(0)
    enabled.set(true)
    try {
      val actual = spark.sql(sql).collect().map(_.toString).sorted.toSeq
      assertResult(expected)(actual)
      assert(applied.get() > 0, "Candidate execution must use the strategy")
    } finally enabled.set(false)
  }

  test("preserve counts, duplicates, unmatched keys and NULL groups") {
    check(query)
    check("SELECT d.label, count(*) AS n" + from + "GROUP BY d.label")
    check("SELECT d.label, count(f.v) AS n" + from +
      "WHERE d.label='c' GROUP BY d.label")
  }

  test("support either fact side and dimension substring grouping") {
    check("SELECT d.label, count(f.v) AS n FROM cp_dim d JOIN cp_fact f " +
      "ON d.k=f.k AND d.g=f.g GROUP BY d.label")
    check("SELECT substring(d.label,1,1) AS label, f.bucket, count(*) AS n" +
      from + "GROUP BY substring(d.label,1,1),f.bucket")
  }

  test("preserve grouped counts with either ANSI setting") {
    for (ansi <- Seq(false, true)) {
      spark.conf.set("spark.sql.ansi.enabled", ansi.toString)
      check(query)
    }
  }

  test("leave ungrouped counts on the existing partial-reduction path") {
    for (ansi <- Seq(false, true); aqe <- Seq(false, true)) {
      spark.conf.set("spark.sql.ansi.enabled", ansi.toString)
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      for (suffix <- Seq("", "WHERE f.k < 0")) {
        val sql = "SELECT count(*) AS n, count(f.v) AS valid" + from + suffix
        enabled.set(false)
        val original = spark.sql(sql)
        assert(original.queryExecution.optimizedPlan.collect {
          case a: Aggregate if GpuCountPreJoinReorder.rewrite(a).isDefined => a
        }.isEmpty)
        val expected = original.collect().toSeq
        applied.set(0)
        enabled.set(true)
        try {
          assertResult(expected)(spark.sql(sql).collect().toSeq)
          assert(applied.get() == 0)
        } finally enabled.set(false)
      }
    }
    spark.conf.set("spark.sql.adaptive.enabled", "false")
  }

  test("skip reduction estimates across runtime scalar predicates") {
    val sql = query.replace("GROUP BY",
      "WHERE f.k >= (SELECT min(k) FROM cp_dim) GROUP BY")
    for (aqe <- Seq(false, true)) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      enabled.set(false)
      val original = spark.sql(sql)
      val plan = original.queryExecution.optimizedPlan
      assert(plan.exists(_.expressions.exists(_.exists(_.isInstanceOf[ScalarSubquery]))))
      assert(plan.collect {
        case a: Aggregate if GpuCountPreJoinReorder.rewrite(a).isDefined => a
      }.isEmpty)
      val expected = original.collect().map(_.toString).sorted.toSeq
      assert(expected.nonEmpty)
      applied.set(0)
      enabled.set(true)
      try {
        assertResult(expected)(spark.sql(sql).collect().map(_.toString).sorted.toSeq)
        assert(applied.get() == 0)
      } finally enabled.set(false)
    }
    spark.conf.set("spark.sql.adaptive.enabled", "false")
  }

  test("do not introduce pre-aggregation after AQE hides input predicates") {
    val original = spark.sql(query).queryExecution.optimizedPlan.collectFirst {
      case a: Aggregate => a
    }.get
    val join = original.child.collectFirst { case j: Join => j }.get
    for (child <- Seq(join.left, join.right)) {
      val physical = TrampolineConnectShims.createDataFrame(spark, child)
        .queryExecution.sparkPlan
      val stage = LogicalQueryStage(child, physical)
      val hidden = original.transformDown { case node if node.eq(child) => stage }
        .asInstanceOf[Aggregate]
      assert(hidden.resolved)
      assert(GpuCountPreJoinReorder.rewrite(hidden).isEmpty,
        "A stage leaf must not bypass the initial cost/selectivity decision")
    }
  }

  test("retain an initial pre-aggregation selection after AQE materializes a child") {
    val original = eligible()
    val join = original.child.collectFirst { case j: Join => j }.get
    for (child <- Seq(join.left, join.right)) {
      val physical = TrampolineConnectShims.createDataFrame(spark, child)
        .queryExecution.sparkPlan
      val stage = LogicalQueryStage(child, physical)
      val hidden = original.transformDown { case node if node.eq(child) => stage }
        .asInstanceOf[Aggregate]
      assert(hidden.resolved)
      assert(GpuCountPreJoinReorder.rewrite(hidden).isDefined,
        "AQE must retain an eligible initial decision when the logical node is copied")
    }
  }

  test("retain pre-aggregation in the final executed adaptive plan") {
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    enabled.set(false)
    val expected = spark.sql(query).collect().map(_.toString).sorted.toSeq
    enabled.set(true)
    try {
      val frame = spark.sql(query)
      assertResult(expected)(frame.collect().map(_.toString).sorted.toSeq)
      val adaptive = frame.queryExecution.executedPlan.asInstanceOf[AdaptiveSparkPlanExec]
      assert(adaptive.isFinalPlan)
      assert(adaptive.executedPlan.toString.contains("pre_join_count_"),
        "Initial selection must survive real AQE stage completion and replanning")
    } finally {
      enabled.set(false)
      spark.conf.set("spark.sql.adaptive.enabled", "false")
    }
  }

  test("COUNT and its long SUM merge retain the same overflow mode") {
    for (ansi <- Seq(false, true)) {
      val conf = new SQLConf
      conf.setConfString("spark.sql.ansi.enabled", ansi.toString)
      SQLConf.withExistingConf(conf) {
        val count = Count(Seq(Literal(1)))
        val sum = Sum(Literal(1L))
        val expressions = Seq(count, sum).map { function =>
          BindReferences.bindReference(function.mergeExpressions.head,
            function.aggBufferAttributes ++ function.inputAggBufferAttributes)
        }
        expressions.foreach { expression =>
          assertResult(3L)(expression.eval(InternalRow(1L, 2L)))
          if (ansi) {
            val error = intercept[ArithmeticException] {
              expression.eval(InternalRow(Long.MaxValue, 1L))
            }
            assert(error.getMessage.contains("ARITHMETIC_OVERFLOW"))
          } else {
            assertResult(Long.MinValue)(expression.eval(InternalRow(Long.MaxValue, 1L)))
          }
        }
      }
    }
  }

  test("retain output identity, matched-key guard and nonrecursive planning") {
    val original = eligible()
    val result = GpuCountPreJoinReorder.rewrite(original).get
    assert(result.resolved)
    assertResult(original.output)(result.output)
    assert(result.collect {
      case j: Join if j.joinType == org.apache.spark.sql.catalyst.plans.LeftSemi => j
    }.nonEmpty, "Discard unmatched keys before an ANSI count can overflow")
    assert(result.collect {
      case a: Aggregate if GpuCountPreJoinReorder.rewrite(a).isDefined => a
    }.isEmpty)
  }

  test("retain no groups for grouped empty inputs") {
    enabled.set(false)
    for (sql <- Seq(query);
        emptyTable <- Seq("cp_fact", "cp_dim")) {
      val original = spark.sql(sql).queryExecution.optimizedPlan.collectFirst {
        case a: Aggregate if GpuCountPreJoinReorder.rewrite(a).isDefined => a
      }.get
      val rewritten = GpuCountPreJoinReorder.rewrite(original).get
      def empty(plan: LogicalPlan): LogicalPlan = plan.transformDown {
        case relation: LogicalRelation
            if relation.catalogTable.exists(_.identifier.table == emptyTable) =>
          LocalRelation(relation.output)
      }
      def rows(plan: LogicalPlan): Seq[String] = TrampolineConnectShims
        .createDataFrame(spark, empty(plan)).collect().map(_.toString).sorted.toSeq
      assertResult(rows(original))(rows(rewritten))
    }
  }

  test("preserve partition pruning when the fact becomes smaller") {
    val sql = query.replace("GROUP BY", "WHERE d.label = 'b' GROUP BY")
    val original = spark.sql(sql).queryExecution.optimizedPlan.collectFirst {
      case a: Aggregate if GpuCountPreJoinReorder.rewrite(a).isDefined => a
    }.get
    val result = GpuCountPreJoinReorder.rewrite(original).get
    val pruning = result.flatMap(_.expressions.flatMap(_.collect {
      case p: DynamicPruningSubquery => p
    }))
    assert(pruning.nonEmpty)
    assert(pruning.forall(!_.onlyInBroadcast))
    check(sql)
  }

  test("reject SUM, DISTINCT and filtered aggregate expressions") {
    val original = eligible()
    val expression = original.aggregateExpressions.collectFirst {
      case Alias(a: AggregateExpression, _) => a
    }.get
    val unsupported = Seq(expression.copy(aggregateFunction = Sum(Literal(1L))),
      expression.copy(isDistinct = true), expression.copy(filter = Some(Literal(true))))
    for (aggregate <- unsupported) {
      val plan = original.copy(aggregateExpressions = Seq(Alias(aggregate, "unsupported")()))
      assert(GpuCountPreJoinReorder.rewrite(plan).isEmpty)
    }
  }

  test("reject throwing dimension projections") {
    val sql = "SELECT cast(d.label as bigint) AS label, count(*) AS n" +
      from + "GROUP BY cast(d.label as bigint)"
    val plan = spark.sql(sql).queryExecution.optimizedPlan
    assert(plan.collect {
      case a: Aggregate if GpuCountPreJoinReorder.rewrite(a).isDefined => a
    }.isEmpty)
  }

  test("reject outer joins, hints and residual join predicates") {
    val original = eligible()
    val join = original.child.collectFirst { case j: Join => j }.get
    val value = join.left.output.head
    for (changed <- Seq(join.copy(joinType = LeftOuter),
        join.copy(hint = JoinHint(Some(HintInfo(Some(BROADCAST))), None)),
        join.copy(condition = Some(And(join.condition.get, IsNotNull(value)))),
        join.copy(condition = Some(And(join.condition.get,
          GreaterThan(Rand(1L), Literal(0.5))))))) {
      assert(GpuCountPreJoinReorder.rewrite(original.copy(child = changed)).isEmpty)
    }
  }
}
