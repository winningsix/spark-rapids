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

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Min, Sum}
import org.apache.spark.sql.catalyst.plans.LeftOuter
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims
import org.apache.spark.sql.types.DoubleType

class MinMaxPreJoinSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _
  private val enabled = new AtomicBoolean(false)
  private val applied = new AtomicInteger()
  private val from = " FROM mm_fact f JOIN mm_dim d ON f.k=d.k AND f.g=d.g "
  private val query = "SELECT d.label, f.bucket, min(f.v) AS lo, max(f.v) AS hi" +
    from + "GROUP BY d.label, f.bucket"

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("minmax-prejoin-tests").config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.hadoop.fs.defaultFS", "file:///")
      .config("spark.sql.warehouse.dir", Files.createTempDirectory("minmax").toUri.toString)
      .config("spark.sql.cbo.enabled", "true")
      .config("spark.sql.cbo.joinReorder.enabled", "true")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .config("spark.sql.ansi.enabled", "true")
      .withExtensions { extensions =>
        extensions.injectPlannerStrategy { _ => new SparkStrategy {
          private val delegate = new GpuMinMaxPreJoinStrategy
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
    spark.sql("CREATE TABLE mm_fact USING parquet PARTITIONED BY(g) AS SELECT " +
      "CASE WHEN id % 13 = 0 THEN NULL ELSE cast(id % 4 as int) END AS k," +
      "cast(id % 2 as int) AS g,cast(id % 3 as int) AS bucket," +
      "CASE WHEN id % 4 = 3 OR id % 17 = 0 THEN NULL " +
      "ELSE cast((id % 17 - 8) * 0.25 as decimal(18,2)) END AS v FROM range(2400)")
    spark.sql("CREATE TABLE mm_dim USING parquet AS SELECT * FROM VALUES " +
      "(0,0,'a'),(0,0,'a'),(0,0,'b'),(1,1,'a'),(2,0,NULL),(3,1,'c')," +
      "(4,0,'unused'),(NULL,0,'null-key') t(k,g,label)")
    for (table <- Seq("mm_fact", "mm_dim")) {
      spark.sql(s"ANALYZE TABLE $table COMPUTE STATISTICS FOR ALL COLUMNS")
    }
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally super.afterAll()
  }

  private def eligible(): Aggregate = spark.sql(query).queryExecution.optimizedPlan.collectFirst {
    case a: Aggregate if GpuMinMaxPreJoinReorder.rewrite(a).isDefined => a
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

  test("preserve decimal extrema with duplicate dimension keys and NULL groups") {
    check(query)
  }

  test("keep final aggregation when distinct foreign keys share dimension values") {
    check("SELECT d.label, min(f.v) AS lo" + from + "GROUP BY d.label")
  }

  test("support a fact on either join side and mixed exact input types") {
    check("SELECT d.label, min(f.v) AS lo, max(f.bucket) AS hi " +
      "FROM mm_dim d JOIN mm_fact f ON d.k=f.k AND d.g=f.g GROUP BY d.label")
  }

  test("preserve global extrema without grouping columns") {
    check("SELECT min(f.v) AS lo, max(f.v) AS hi" + from)
  }

  test("retain output identity and prevent recursive aggregation insertion") {
    val original = eligible()
    val result = GpuMinMaxPreJoinReorder.rewrite(original).get
    assert(result.resolved)
    assertResult(original.output)(result.output)
    assert(result.collect {
      case a: Aggregate if GpuMinMaxPreJoinReorder.rewrite(a).isDefined => a
    }.isEmpty)
  }

  test("keep partition pruning when preaggregation changes broadcast eligibility") {
    val sql = query.replace("GROUP BY", "WHERE d.label = 'b' GROUP BY")
    val original = spark.sql(sql).queryExecution.optimizedPlan.collectFirst {
      case a: Aggregate if GpuMinMaxPreJoinReorder.rewrite(a).isDefined => a
    }.get
    val result = GpuMinMaxPreJoinReorder.rewrite(original).get
    val pruning = result.flatMap(_.expressions.flatMap(_.collect {
      case p: DynamicPruningSubquery => p
    }))
    assert(pruning.nonEmpty)
    assert(pruning.forall(!_.onlyInBroadcast))
    check(sql)
  }

  test("reject SUM, floating extrema, DISTINCT and aggregate filters") {
    val original = eligible()
    val expression = original.aggregateExpressions.collectFirst {
      case Alias(a: AggregateExpression, _) => a
    }.get
    val child = expression.aggregateFunction.children.head
    val unsupported = Seq(expression.copy(aggregateFunction = Sum(child)),
      expression.copy(aggregateFunction = Min(Cast(child, DoubleType))),
      expression.copy(isDistinct = true), expression.copy(filter = Some(IsNotNull(child))))
    for (aggregate <- unsupported) {
      val plan = original.copy(aggregateExpressions = Seq(Alias(aggregate, "unsupported")()))
      assert(GpuMinMaxPreJoinReorder.rewrite(plan).isEmpty)
    }
  }

  test("retain grouped and global empty-input semantics") {
    enabled.set(false)
    for (sql <- Seq(query, "SELECT min(f.v) AS lo, max(f.v) AS hi" + from);
        emptyTable <- Seq("mm_fact", "mm_dim")) {
      val original = spark.sql(sql).queryExecution.optimizedPlan.collectFirst {
        case a: Aggregate if GpuMinMaxPreJoinReorder.rewrite(a).isDefined => a
      }.get
      val rewritten = GpuMinMaxPreJoinReorder.rewrite(original).get
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

  test("reject outer joins, hints and residual join predicates") {
    val original = eligible()
    val join = original.child.collectFirst { case j: Join => j }.get
    val value = original.aggregateExpressions.flatMap(_.collect {
      case a: AggregateExpression => a.aggregateFunction.children.head
    }).head
    for (changed <- Seq(join.copy(joinType = LeftOuter),
        join.copy(hint = JoinHint(Some(HintInfo(Some(BROADCAST))), None)),
        join.copy(condition = Some(And(join.condition.get, IsNotNull(value)))),
        join.copy(condition = Some(And(join.condition.get,
          GreaterThan(Rand(1L), Literal(0.5))))))) {
      assert(GpuMinMaxPreJoinReorder.rewrite(original.copy(child = changed)).isEmpty)
    }
  }
}
