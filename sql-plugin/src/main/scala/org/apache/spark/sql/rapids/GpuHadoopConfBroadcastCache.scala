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

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap

import scala.jdk.CollectionConverters._

import org.apache.hadoop.conf.Configuration

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.util.SerializableConfiguration

/**
 * Reuses the broadcast Hadoop configuration across file scans whose configurations are
 * identical. Without it every GpuFileSourceScanExec serializes, compresses and broadcasts the
 * full Hadoop Configuration (all default properties) on the driver, once per scan and per query
 * execution, which shows up as driver time between stages.
 *
 * The key is a digest of every (name, value) pair of the configuration, so any difference (for
 * example per-relation options) yields a separate broadcast. Cached broadcasts are strongly
 * referenced, so the ContextCleaner does not remove them while they can still be handed out.
 */
object GpuHadoopConfBroadcastCache {
  val ENABLED_KEY = "spark.rapids.sql.scan.hadoopConfBroadcastCache.enabled"
  private val MAX_ENTRIES = 256

  private case class Key(contextId: Int, digest: String)

  private val cache = new ConcurrentHashMap[Key, Broadcast[SerializableConfiguration]]()

  private def digest(conf: Configuration): String = {
    val entries = conf.iterator().asScala.map(e => e.getKey + "\u0000" + e.getValue).toArray
    java.util.Arrays.sort(entries.asInstanceOf[Array[Object]])
    val md = MessageDigest.getInstance("SHA-256")
    entries.foreach { s =>
      md.update(s.getBytes(StandardCharsets.UTF_8))
      md.update(1.toByte)
    }
    md.digest().map("%02x".format(_)).mkString
  }

  def broadcast(sc: SparkContext, conf: Configuration): Broadcast[SerializableConfiguration] = {
    if (!sc.conf.getBoolean(ENABLED_KEY, defaultValue = true)) {
      return sc.broadcast(new SerializableConfiguration(conf))
    }
    val key = Key(System.identityHashCode(sc), digest(conf))
    val cached = cache.get(key)
    if (cached != null && !sc.isStopped) {
      cached
    } else {
      if (cache.size() >= MAX_ENTRIES) {
        cache.clear()
      }
      // A concurrent miss may create a second broadcast for the same key; both are valid.
      val created = sc.broadcast(new SerializableConfiguration(conf))
      cache.put(key, created)
      created
    }
  }
}
