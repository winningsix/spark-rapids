/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids.execution

import org.json4s.JsonAST

import org.apache.spark.{SparkConf, SparkContext, SparkEnv, SparkUpgradeException, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.executor.InputMetrics
import org.apache.spark.internal.config.EXECUTOR_ID
import org.apache.spark.memory.TaskMemoryManager
import org.apache.spark.sql.{AnalysisException, SparkSession}
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.physical.BroadcastMode
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.shims.DataTypeUtilsShim
import org.apache.spark.sql.rapids.shims.SparkUpgradeExceptionShims
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.storage.BlockManagerId
import org.apache.spark.util.{ShutdownHookManager, Utils}

object TrampolineUtil {
  def doExecuteBroadcast[T](child: SparkPlan): Broadcast[T] = child.doExecuteBroadcast()

  def isSupportedRelation(mode: BroadcastMode): Boolean = 
    ShimTrampolineUtil.isSupportedRelation(mode)

  def unionLikeMerge(left: DataType, right: DataType): DataType =
    ShimTrampolineUtil.unionLikeMerge(left, right)

  def fromAttributes(attrs: Seq[Attribute]): StructType = DataTypeUtilsShim.fromAttributes(attrs)

  def toAttributes(structType: StructType): Seq[Attribute] =
    DataTypeUtilsShim.toAttributes(structType)

  def jsonValue(dataType: DataType): JsonAST.JValue = dataType.jsonValue

  /** Get a human-readable string, e.g.: "4.0 MiB", for a value in bytes. */
  def bytesToString(size: Long): String = Utils.bytesToString(size)

  /** Returns true if called from code running on the Spark driver. */
  def isDriver(env: SparkEnv): Boolean = {
    if (env != null) {
      env.executorId == SparkContext.DRIVER_IDENTIFIER
    } else {
      false
    }
  }

  def isDriver(sparkConf: SparkConf): Boolean = {
    sparkConf.get(EXECUTOR_ID).map(_ == SparkContext.DRIVER_IDENTIFIER)
      .getOrElse(isDriver(SparkEnv.get))
  }

  /**
   * Return true if the provided predicate function returns true for any
   * type node within the datatype tree.
   */
  def dataTypeExistsRecursively(dt: DataType, f: DataType => Boolean): Boolean = {
    dt.existsRecursively(f)
  }

  /**
   * Check if `a` and `b` are the same data type when ignoring nullability
   * (`StructField.nullable`, `ArrayType.containsNull`, and `MapType.valueContainsNull`).
   */
  def sameType(a: DataType, b: DataType): Boolean = a.sameType(b)

  def incInputRecordsRows(inputMetrics: InputMetrics, rows: Long): Unit =
    inputMetrics.incRecordsRead(rows)

  def makeSparkUpgradeException(
                                version: String,
                                message: String,
                                cause: Throwable): SparkUpgradeException = {
    SparkUpgradeExceptionShims.newSparkUpgradeException(version, message, cause)
  }

  /** Shuts down and cleans up any existing Spark session */
  def cleanupAnyExistingSession(): Unit = SparkSession.cleanupAnyExistingSession()

  def asNullable(dt: DataType): DataType = dt.asNullable

  /** Return a new InputMetrics instance */
  def newInputMetrics(): InputMetrics = new InputMetrics()

  /**
   * Increment the task's memory bytes spilled metric. If the current thread does not
   * correspond to a Spark task then this call does nothing.
   * @param amountSpilled amount of memory spilled in bytes
   */
  def incTaskMetricsMemoryBytesSpilled(amountSpilled: Long): Unit = {
    Option(TaskContext.get).foreach(_.taskMetrics().incMemoryBytesSpilled(amountSpilled))
  }

  /**
   * Increment the task's disk bytes spilled metric. If the current thread does not
   * correspond to a Spark task then this call does nothing.
   * @param amountSpilled amount of memory spilled in bytes
   */
  def incTaskMetricsDiskBytesSpilled(amountSpilled: Long): Unit = {
    Option(TaskContext.get).foreach(_.taskMetrics().incDiskBytesSpilled(amountSpilled))
  }

  /**
   * Returns a function that can be called to find Hadoop FileSystem bytes read. If
   * getFSBytesReadOnThreadCallback is called from thread r at time t, the returned callback will
   * return the bytes read on r since t.
   */
  def getFSBytesReadOnThreadCallback(): () => Long = {
    SparkHadoopUtil.get.getFSBytesReadOnThreadCallback()
  }

  /** Set the bytes read task input metric */
  def incBytesRead(inputMetrics: InputMetrics, bytesRead: Long): Unit = {
    inputMetrics.incBytesRead(bytesRead)
  }

  /** Get the simple name of a class with fixup for any Scala internal errors */
  def getSimpleName(cls: Class[_]): String = {
    Utils.getSimpleName(cls)
  }

  /** Create a `BlockManagerId` instance */
  def newBlockManagerId(
      execId: String,
      host: String,
      port: Int,
      topologyInfo: Option[String] = None): BlockManagerId =
    BlockManagerId(execId, host, port, topologyInfo)

  def getTaskMemoryManager(): TaskMemoryManager = {
    TaskContext.get.taskMemoryManager()
  }

  /** Throw a Spark analysis exception */
  def throwAnalysisException(msg: String) = throw new AnalysisException(msg)

  /** Set the task context for the current thread */
  def setTaskContext(tc: TaskContext): Unit = TaskContext.setTaskContext(tc)

  /** Remove the task context for the current thread */
  def unsetTaskContext(): Unit = TaskContext.unset()

  /** Add shutdown hook with priority */
  def addShutdownHook(priority: Int, runnable: Runnable): AnyRef = {
    ShutdownHookManager.addShutdownHook(priority)(() => runnable.run())
  }

  def classForName[C](
      className: String,
      initialize: Boolean = true,
      noSparkClassLoader: Boolean = false): Class[C] = {
    Utils.classForName(className, initialize, noSparkClassLoader)
  }

  def getSparkConf(spark: SparkSession): SQLConf = {
    spark.sqlContext.conf
  }
}
