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

import java.nio.file.{Files, Paths}

import scala.collection.mutable
import scala.jdk.CollectionConverters._

import com.fasterxml.jackson.databind.ObjectMapper
import com.nvidia.spark.rapids.GpuScalarAggregateSharingOverrides
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.rapids.shims.TrampolineConnectShims

/** SQL algebra only. Never submit these modified queries as benchmark measurements. */
class GpuScalarAggregateSharingOverridesSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: TrampolineConnectShims.SparkSession = _
  private val out = Paths.get(System.getProperty("q24.probe.output"))
  private val records = mutable.ArrayBuffer.empty[java.util.Map[String, String]]
  private val mapper = new ObjectMapper
  private val maximum = "99999999999999999999999999999999999999"

  private def record(values: (String, String)*): Unit = {
    records += values.toMap.asJava
    println("SCALAR_SHARING_CASE " + mapper.writeValueAsString(records.last))
  }

  override def beforeAll(): Unit = {
    super.beforeAll()
    Files.createDirectories(out)
    val root = Files.createTempDirectory("scalar-sharing-probe")
    spark = TrampolineConnectShims.getBuilder().master("local[2]")
      .appName("scalar-aggregate-sharing-probe").config("spark.ui.enabled", "false")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.sql.warehouse.dir", root.resolve("warehouse").toUri.toString)
      .config("spark.sql.ansi.enabled", "true").config("spark.sql.shuffle.partitions", "2")
      .config("spark.rapids.sql.enabled", "false")
      .withExtensions(e => e.injectPostHocResolutionRule(s =>
        new GpuScalarAggregateSharingOverrides(s))).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    def register(name: String, sql: String): Unit = {
      val path = root.resolve(name).toString
      spark.sql(sql).coalesce(1).write.parquet(path)
      spark.sql(s"CREATE TABLE $name USING parquet LOCATION '$path'")
    }
    val normal = "SELECT name,color,shape,cast(v as decimal(18,2)) v FROM VALUES " +
      "('a','red',1,6000),('a','red',1,-10),('a','red',2,20)," +
      "('b','blue',1,100000),(NULL,'red',1,5000),('c',NULL,1,7000)," +
      "('d','red',1,NULL),('d','blue',2,NULL) t(name,color,shape,v)"
    register("sha_normal", normal)
    register("sha_decimal7", "SELECT name,color,shape,cast(v/10 as decimal(7,2)) v FROM sha_normal")
    register("sha_decimal14", "SELECT name,color,shape,cast(v as decimal(14,6)) v FROM sha_normal")
    register("sha_empty", normal + " WHERE false")
    register("sha_no_selected", normal + " WHERE color='blue'")
    register("sha_all_null", normal + " WHERE v IS NULL")
    register("sha_weighted", "SELECT CASE WHEN id=0 THEN 'a' ELSE 'b' END name," +
      "CASE WHEN id=0 THEN 'red' ELSE 'blue' END color,1 shape," +
      "cast(CASE WHEN id=0 THEN 20 ELSE 10 END as decimal(18,2)) v FROM range(101)")
    register("sha_precision", "SELECT name,color,shape,cast(v as decimal(18,2)) v FROM VALUES " +
      "('a','red',1,'9999999999999999.99'),('a','red',1,'-9999999999999998.99')," +
      "('b','blue',2,'2.01') t(name,color,shape,v)")
    for ((name, color, grouped) <- Seq(("sha_group_overflow", "red", true),
        ("sha_avg_overflow", "red", false), ("sha_empty_overflow", "blue", true))) {
      register(name, "SELECT name,color,shape,cast(v as decimal(38,0)) v FROM VALUES " +
        s"('a','$color',1,'$maximum')," +
        s"('${if (grouped) "a" else "b"}','${if (grouped) color else "blue"}'," +
        s"1,'$maximum') t(name,color,shape,v)")
    }
  }

  override def afterAll(): Unit = {
    try {
      Files.writeString(out.resolve("cases.json"), mapper.writerWithDefaultPrettyPrinter()
        .writeValueAsString(records.asJava))
      if (spark != null) spark.stop()
    } finally super.afterAll()
  }

  private def grouped(table: String): String =
    "WITH g AS (SELECT name,color,shape,sum(v) netpaid " +
      s"FROM $table GROUP BY name,color,shape) "

  private def original(table: String): String = grouped(table) +
    "SELECT name,sum(netpaid) paid FROM g WHERE color='red' GROUP BY name " +
    "HAVING sum(netpaid) > (SELECT 0.05*avg(netpaid) FROM g) ORDER BY name"

  private def window(table: String, earlyFilter: Boolean = false): String = grouped(table) +
    "SELECT name,sum(netpaid) paid FROM (SELECT *,0.05*avg(netpaid) OVER () threshold FROM g " +
    (if (earlyFilter) "WHERE color='red'" else "") + ") shared " +
    "WHERE color='red' GROUP BY name HAVING sum(netpaid)>max(threshold) ORDER BY name"

  private def answer(sql: String, stem: String): Either[String, Seq[String]] = {
    val frame = spark.sql(sql)
    try {
      val rows = frame.collect().map(_.toString).toSeq
      Files.writeString(out.resolve(stem + "-plan.txt"), frame.queryExecution.toString)
      Right(rows)
    } catch {
      case error: Exception =>
        val conditions = Iterator.iterate[Throwable](error)(_.getCause)
          .takeWhile(_ != null).collect { case t: org.apache.spark.SparkThrowable
              if t.getCondition != null && (t.getCondition == "ARITHMETIC_OVERFLOW" ||
                t.getCondition == "DIVIDE_BY_ZERO" ||
                t.getCondition.startsWith("NUMERIC_VALUE_OUT_OF_RANGE")) => t.getCondition
          }.toSeq.distinct
        assert(conditions.size == 1, error.toString)
        Left(conditions.head)
    }
  }

  private def check(sql: String, stem: String, selected: Boolean): Unit = {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val expectedSchema = spark.sql(sql).schema
    val expected = answer(sql, stem + "-original")
    spark.conf.set("spark.rapids.sql.enabled", "true")
    val frame = spark.sql(sql)
    assert(frame.schema == expectedSchema)
    val plan = frame.queryExecution.analyzed
    Files.writeString(out.resolve(stem + "-analyzed.txt"), plan.treeString)
    assert(plan.treeString.contains("scalar_sharing_safe") == selected, plan.treeString)
    assert(new GpuScalarAggregateSharingOverrides(spark).apply(plan).fastEquals(plan))
    val actual = answer(sql, stem + "-candidate")
    record("case" -> stem, "selected" -> selected.toString,
      "original" -> expected.toString, "candidate" -> actual.toString,
      "match" -> (expected == actual).toString)
    assertResult(expected)(actual)
  }

  test("unset SQL enablement uses the RAPIDS default and explicit false disables sharing") {
    val sql = original("sha_normal")
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val expected = answer(sql, "default-disabled-reference")
    assert(!spark.sql(sql).queryExecution.analyzed.toString.contains("scalar_sharing_safe"))
    try {
      spark.conf.unset("spark.rapids.sql.enabled")
      val frame = spark.sql(sql)
      assert(frame.queryExecution.analyzed.toString.contains("scalar_sharing_safe"))
      assertResult(expected)(answer(sql, "default-enabled-candidate"))
    } finally {
      spark.conf.set("spark.rapids.sql.enabled", "false")
    }
  }

  test("original SQL shares exact decimal grouped AVG without changing order or schema") {
    for (aqe <- Seq(false, true); table <- Seq("sha_normal", "sha_empty", "sha_no_selected",
        "sha_all_null", "sha_weighted", "sha_precision", "sha_decimal7", "sha_decimal14")) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      check(original(table), s"$table-$aqe", selected = true)
    }
  }

  test("unsupported decimal precision retains original overflow and empty-result behavior") {
    for (aqe <- Seq(false, true); table <- Seq("sha_group_overflow", "sha_avg_overflow",
        "sha_empty_overflow")) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      check(original(table), s"$table-$aqe", selected = false)
    }
  }

  test("different aggregate scope and unsafe expressions remain unchanged") {
    val base = original("sha_normal")
    val global = grouped("sha_empty") + "SELECT sum(netpaid) paid FROM g WHERE color='red' " +
      "HAVING sum(netpaid)>(SELECT 0.05*avg(netpaid) FROM g) OR sum(netpaid) IS NULL"
    val controls = Seq(
      "global_empty_group" -> global,
      "distinct_avg" -> base.replace("avg(netpaid)", "avg(distinct netpaid)"),
      "distinct_sum" -> base.replace("sum(v)", "sum(distinct v)"),
      "throwing_measure" -> base.replace("sum(v)", "sum(v/0)"),
      "large_multiplier" -> base.replace("0.05*avg", "2.0*avg"),
      "correlated" -> base.replace("avg(netpaid) FROM g",
        "avg(netpaid) FROM g x WHERE x.name=g.name"))
    for (aqe <- Seq(false, true); (name, sql) <- controls) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      check(sql, s"$name-$aqe", selected = false)
    }
  }

  test("non-unit and negative multipliers and original limit retain exact results") {
    for (aqe <- Seq(false, true); coefficient <- Seq("0.25", "-0.5", "0.0")) {
      spark.conf.set("spark.sql.adaptive.enabled", aqe.toString)
      check(original("sha_normal").replace("0.05*avg", coefficient + "*avg") + " LIMIT 2",
        s"coefficient-$coefficient-$aqe", selected = true)
    }
  }

  test("runtime group cap preserves the complete original fallback") {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    val path = Files.createTempDirectory("scalar-sharing-many-groups").resolve("data").toString
    spark.sql("SELECT 'a' name,'red' color,id shape,cast(1 as decimal(7,2)) v " +
      "FROM range(1000001)").write.parquet(path)
    spark.sql(s"CREATE TABLE sha_many_groups USING parquet LOCATION '$path'")
    check(original("sha_many_groups"), "actual-group-cap-fallback", selected = true)
    val frame = spark.sql(original("sha_many_groups"))
    frame.collect()
    val finalPlan = frame.queryExecution.executedPlan.toString.split("== Initial Plan ==")(0)
    assert(!finalPlan.contains("Window"), finalPlan)
    Files.writeString(out.resolve("group-cap-final-plan.txt"), frame.queryExecution.toString)
  }

  test("scope and weighting mutants produce observably different answers") {
    spark.conf.set("spark.rapids.sql.enabled", "false")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    val expected = answer(original("sha_weighted"), "mutant-reference")
    val early = answer(window("sha_weighted", earlyFilter = true), "mutant-early-filter")
    val weighted = answer(original("sha_weighted").replace("avg(netpaid) FROM g",
      "avg(v) FROM sha_weighted"), "mutant-raw-weighting")
    assert(expected != early && expected != weighted)
    record("case" -> "negative_controls", "both_detected" -> "true")
  }
  test("safe path reuses the physical partial SUM and COUNT in its scalar guard") {
    spark.conf.set("spark.rapids.sql.enabled", "true")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    val frame = spark.sql(original("sha_normal"))
    frame.collect()
    val visited = new java.util.IdentityHashMap[
      org.apache.spark.sql.execution.SparkPlan, java.lang.Boolean]()
    val reused = mutable.ArrayBuffer.empty[
      org.apache.spark.sql.execution.exchange.ReusedExchangeExec]
    def visit(plan: org.apache.spark.sql.execution.SparkPlan): Unit = {
      if (!visited.containsKey(plan)) {
        visited.put(plan, true)
        plan match {
          case r: org.apache.spark.sql.execution.exchange.ReusedExchangeExec
              if r.child.toString.contains("partial_sum") &&
                r.child.toString.contains("partial_count") => reused += r
          case _ =>
        }
        plan.children.foreach(visit)
        plan.expressions.foreach(_.foreach {
          case e: org.apache.spark.sql.catalyst.expressions.PlanExpression[_] => e.plan match {
            case physical: org.apache.spark.sql.execution.SparkPlan => visit(physical)
            case _ =>
          }
          case _ =>
        })
      }
    }
    visit(frame.queryExecution.executedPlan)
    assert(reused.nonEmpty, frame.queryExecution.toString)
    Files.writeString(out.resolve("production-rule-reuse-plan.txt"), frame.queryExecution.toString)
  }

}
