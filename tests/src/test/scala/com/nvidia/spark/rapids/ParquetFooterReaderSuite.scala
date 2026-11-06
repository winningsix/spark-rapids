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

import java.io.{ByteArrayInputStream, IOException}
import java.util.concurrent.{Callable, Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._

import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.shims.parquet.ParquetFooterReader
import org.apache.parquet.format.converter.ParquetMetadataConverter
import org.apache.parquet.format.converter.ParquetMetadataConverter.MetadataFilter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.metadata.ParquetMetadata
import org.apache.parquet.io.{DelegatingSeekableInputStream, InputFile, SeekableInputStream}
import org.scalatest.funsuite.AnyFunSuite

class ParquetFooterReaderSuite extends AnyFunSuite {
  private lazy val bytes = withResource(getClass.getResourceAsStream("/file-splits.parquet")) {
    stream => org.apache.commons.io.IOUtils.toByteArray(stream)
  }

  private class MemoryInputFile(data: Array[Byte]) extends InputFile {
    val closes = new AtomicInteger()
    override def getLength: Long = data.length
    override def newStream(): SeekableInputStream = {
      class SeekableBytes extends ByteArrayInputStream(data) {
        def position: Long = pos.toLong
        def seekTo(offset: Long): Unit = {
          if (offset < 0 || offset > data.length) {
            throw new IOException(s"Invalid offset $offset")
          }
          pos = offset.toInt
        }
      }
      val input = new SeekableBytes
      new DelegatingSeekableInputStream(input) {
        override def getPos: Long = input.position
        override def seek(offset: Long): Unit = input.seekTo(offset)
        override def close(): Unit = {
          closes.incrementAndGet()
          super.close()
        }
      }
    }
  }

  @scala.annotation.nowarn("msg=method readFooter in class ParquetFileReader is deprecated")
  private def original(input: InputFile, filter: MetadataFilter): ParquetMetadata =
    ParquetFileReader.readFooter(input, filter)

  private def filters: Seq[MetadataFilter] = Seq(
    ParquetMetadataConverter.NO_FILTER,
    ParquetMetadataConverter.SKIP_ROW_GROUPS,
    ParquetMetadataConverter.range(0, bytes.length / 2),
    ParquetMetadataConverter.range(bytes.length / 2, bytes.length),
    ParquetMetadataConverter.range(bytes.length / 4, bytes.length * 3 / 4),
    ParquetMetadataConverter.range(bytes.length, bytes.length + 1))

  test("cached default configuration preserves footer metadata and range filtering") {
    val reader = new ParquetFooterReader
    val input = new MemoryInputFile(bytes)
    assert(original(input, ParquetMetadataConverter.NO_FILTER).getBlocks.size() > 1)
    filters.foreach { filter =>
      val expected = original(input, filter)
      val actual = reader.readFooter(input, filter)
      assert(ParquetMetadata.toJSON(actual) == ParquetMetadata.toJSON(expected))
    }
    assert(input.closes.get() == 1 + filters.size * 2)
  }

  test("concurrent footer reads keep metadata filters independent") {
    val reader = new ParquetFooterReader
    val input = new MemoryInputFile(bytes)
    val expected = filters.map(f => ParquetMetadata.toJSON(original(input, f)))
    val pool = Executors.newFixedThreadPool(8)
    try {
      val jobs = (0 until 240).map { i =>
        new Callable[String] {
          override def call(): String =
            ParquetMetadata.toJSON(reader.readFooter(input, filters(i % filters.size)))
        }
      }
      val results = pool.invokeAll(jobs.asJava, 30, TimeUnit.SECONDS).asScala
      results.zipWithIndex.foreach { case (result, i) =>
        assert(result.get() == expected(i % expected.size))
      }
      assert(input.closes.get() == filters.size + jobs.size)
    } finally {
      pool.shutdownNow()
    }
  }

  test("footer streams close when parsing fails") {
    val input = new MemoryInputFile(Array.fill[Byte](32)(0))
    intercept[RuntimeException] {
      new ParquetFooterReader().readFooter(input, ParquetMetadataConverter.NO_FILTER)
    }
    assert(input.closes.get() == 1)
  }
}
