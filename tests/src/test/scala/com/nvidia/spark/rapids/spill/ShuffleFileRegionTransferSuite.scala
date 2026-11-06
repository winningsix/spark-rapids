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

package com.nvidia.spark.rapids.spill

import java.io.{ByteArrayOutputStream, File, IOException}
import java.nio.ByteBuffer
import java.nio.channels.WritableByteChannel
import java.nio.file.Files

import com.nvidia.spark.rapids.{MultiSegmentFileRegion, PartitionSegment, RapidsConf}
import com.nvidia.spark.rapids.Arm.withResource
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

class ShuffleFileRegionTransferSuite extends AnyFunSuite with BeforeAndAfterEach {
  override def beforeEach(): Unit = {
    super.beforeEach()
    SpillFramework.initialize(new RapidsConf(Map.empty[String, String]))
  }

  override def afterEach(): Unit = {
    SpillFramework.shutdown()
    super.afterEach()
  }

  private def handle(memory: Boolean): SpillablePartialFileHandle = {
    val file = File.createTempFile("rapids-region-transfer", ".data")
    if (memory) {
      SpillablePartialFileHandle.createMemoryWithSpill(
        initialCapacity = 1024, maxBufferSize = 1024 * 1024,
        memoryThreshold = 0.9, spillFile = file)
    } else {
      SpillablePartialFileHandle.createFileOnly(file)
    }
  }

  private class PartialChannel extends WritableByteChannel {
    val bytes = new ByteArrayOutputStream()
    var allowance = 0
    var calls = 0
    var direct = false
    override def isOpen: Boolean = true
    override def close(): Unit = {}
    override def write(buffer: ByteBuffer): Int = {
      calls += 1
      // Fail deterministically rather than hang if transferTo spins after a zero-byte write.
      assert(calls == 1, "transferTo retried a blocked or partial socket write")
      direct = buffer.isDirect
      val count = math.min(allowance, buffer.remaining())
      val data = new Array[Byte](count)
      buffer.get(data)
      bytes.write(data)
      count
    }
  }

  for (memory <- Seq(false, true)) {
    test(s"zero-byte and partial writes yield and resume exactly, memory=$memory") {
      withResource(handle(memory)) { source =>
        val data = Array[Byte](0, 1, 2, 3, 4, 5, 6, 7)
        source.write(data, 0, data.length)
        source.finishWrite()
        val region = new MultiSegmentFileRegion(Seq(
          PartitionSegment(source, 5, 3), PartitionSegment(source, 1, 3)))
        try {
          val target = new PartialChannel()
          for (limit <- Seq(0, 2, 0, 1, 2, 1)) {
            target.allowance = limit
            target.calls = 0
            val before = region.transferred()
            assert(region.transferTo(target, before) == limit)
            assert(region.transferred() == before + limit)
            if (memory) assert(target.direct)
          }
          assert(region.count() == 6)
          assert(region.transferTo(target, 6) == 0)
          assert(target.bytes.toByteArray.toSeq == Seq[Byte](5, 6, 7, 1, 2, 3))
          intercept[IllegalArgumentException] { region.transferTo(target, 1) }
          if (memory) assert(!source.isSpilled)
        } finally {
          region.release()
        }
      }
    }
  }

  test("resume a partially sent memory segment after spill") {
    withResource(handle(memory = true)) { source =>
      source.write(Array[Byte](10, 11, 12, 13), 0, 4)
      source.finishWrite()
      val region = new MultiSegmentFileRegion(Seq(PartitionSegment(source, 0, 4)))
      try {
        val target = new PartialChannel()
        target.allowance = 2
        assert(region.transferTo(target, 0) == 2)
        assert(target.direct)
        source.spillForCommit()
        assert(source.isSpilled)
        target.calls = 0
        assert(region.transferTo(target, 2) == 2)
        assert(target.bytes.toByteArray.toSeq == Seq[Byte](10, 11, 12, 13))
      } finally {
        region.release()
      }
    }
  }

  test("truncated file fails instead of repeatedly returning zero progress") {
    val file = Files.createTempFile("rapids-region-truncated", ".data")
    withResource(SpillablePartialFileHandle.createFileOnly(file.toFile)) { source =>
      source.write(Array[Byte](1, 2, 3), 0, 3)
      source.finishWrite()
      Files.write(file, Array.emptyByteArray)
      intercept[IOException] { source.transferTo(0, 3, new PartialChannel()) }
    }
  }
}
