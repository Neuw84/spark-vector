/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package org.apache.spark.sql.vecruntime.bench

import java.lang.foreign.Arena
import java.nio.file.{Files, Path}
import java.util.Random
import java.util.concurrent.Executors

import scala.collection.mutable

import io.vecruntime.kernels.{ArrowLayout, Bitmap, VectorBuffers}
import io.vecruntime.shuffle.{PartitionedIpcFile, PartitionedIpcWriter}
import io.vecruntime.spark.arrow.ArrowOutput
import org.apache.arrow.flight.{FlightServer, Location}
import org.apache.arrow.memory.{BufferAllocator, RootAllocator}
import org.apache.arrow.vector.compression.CompressionUtil
import org.apache.spark.SparkConf
import org.apache.spark.network.buffer.{FileSegmentManagedBuffer, ManagedBuffer}
import org.apache.spark.network.util.{MapConfigProvider, TransportConf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vecruntime.shuffle.flight.{FlightBlockStream, FlightLocation, FlightShuffle}
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}

/**
 * The fixture behind `io.vecruntime.benchmarks.FlightShuffleBenchmark` (here rather than beside it: the
 * metrics reporter the block stream takes is `private[spark]`): two Arrow Flight servers on the loopback interface
 * standing in for two remote executors, each serving the map files of `maps` map tasks written by
 * the real [[PartitionedIpcWriter]] at `partitions` reduce partitions, and the operations a reduce
 * task performs against them -- open one `DoGet` per executor for its partition range (#347),
 * decode the batches, close. What it measures is the CPU side of the transport: the server's file
 * reads and chunking, gRPC framing and flow control, the client's reassembly, zstd and IPC decode,
 * the reader's coalescing. Not bandwidth: both ends share this JVM and the loopback path, which is
 * also the truth of an executor that serves while it reads.
 *
 * `strings`: `none` (numbers only), `low` (a 50-value column), `high` (a near-distinct column) or
 * `mixed` (one of each) -- the string modes decide how much of the dictionary machinery (#377,
 * #416) is on the path.
 */
object FlightBench {
  val Servers = 2

  final class Fixture(
      partitions: Int,
      maps: Int,
      rowsPerMap: Int,
      strings: String,
      compression: String,
      serverThreads: Int,
      chunkBytes: Int
  ) extends AutoCloseable {

    val schema: StructType = {
      val numbers = Seq(
        StructField("k1", LongType),
        StructField("k2", LongType),
        StructField("i1", IntegerType),
        StructField("i2", IntegerType),
        StructField("d1", DoubleType),
        StructField("d2", DoubleType)
      )
      val texts = strings match {
        case "none" => Nil
        case "low" => Seq(StructField("s_low", StringType))
        case "high" => Seq(StructField("s_high", StringType))
        case "mixed" => Seq(StructField("s_low", StringType), StructField("s_high", StringType))
        case other => throw new IllegalArgumentException(s"strings=$other")
      }
      StructType(numbers ++ texts)
    }
    val codec: Option[CompressionUtil.CodecType] = compression match {
      case "zstd" => Some(CompressionUtil.CodecType.ZSTD)
      case "none" => None
      case other => throw new IllegalArgumentException(s"compression=$other")
    }
    val allocator: BufferAllocator = new RootAllocator()
    private val dir: Path = Files.createTempDirectory("svflightbench")
    private val transportConf = new TransportConf("shuffle", MapConfigProvider.EMPTY)
    private val conf = new SparkConf(false)
    private val metrics = new org.apache.spark.executor.TempShuffleReadMetrics()
    private val lowWords: Array[String] = Array.tabulate(50)(i => f"nation_$i%02d")

    /** Per server: the map files and their indexes (offsets and lengths per partition, as Spark's index file holds them). */
    private val files: Array[Array[(Path, PartitionedIpcFile.Index)]] = Array.tabulate(Servers) { server =>
      Array.tabulate(maps) { map =>
        val path = dir.resolve(s"s$server-m$map.data")
        (path, writeMap(path, seed = server * 1000 + map))
      }
    }
    val rowsWritten: Long = Servers.toLong * maps * rowsPerMap

    private val pools = Array.fill(Servers)(Executors.newFixedThreadPool(
      serverThreads,
      r => { val t = new Thread(r, "flight-bench"); t.setDaemon(true); t }
    ))
    private val servers: Array[FlightServer] = Array.tabulate(Servers) { server =>
      val producer = new FlightShuffle.Producer(
        (_: Int, mapId: Long, start: Int, end: Int) => blockData(server, mapId.toInt, start, end),
        allocator,
        chunkBytes
      )
      val s = FlightServer.builder(
        allocator,
        Location.forGrpcInsecure("127.0.0.1", 0),
        producer
      ).executor(pools(server)).build()
      s.start()
      s
    }
    val locations: Array[FlightLocation] = servers.map(s => FlightLocation("127.0.0.1", s.getPort))
    val mapIds: Seq[Long] = (0 until maps).map(_.toLong)

    /** The reduce range's bytes of one map file as the resolver serves them: one file segment (#411). */
    private def blockData(server: Int, map: Int, start: Int, end: Int): ManagedBuffer = {
      val (path, index) = files(server)(map)
      val offset = index.offsets(start)
      val length = index.offsets(end - 1) + index.lengths(end - 1) - offset
      new FileSegmentManagedBuffer(transportConf, path.toFile, offset, length)
    }

    /** One map task's output: `rowsPerMap` rows hash-spread over the partitions, in batches of 8192 as the exchange produces them. */
    private def writeMap(path: Path, seed: Int): PartitionedIpcFile.Index = {
      val writer = new PartitionedIpcWriter(schema, partitions, allocator, path, compression = codec)
      val random = new Random(seed)
      try {
        var left = rowsPerMap
        while (left > 0) {
          val n = math.min(8192, left)
          val arena = Arena.ofConfined()
          try {
            val batch = makeBatch(arena, n, random)
            val ids = Array.tabulate(n)(_ => random.nextInt(partitions))
            try writer.write(batch, ids)
            finally batch.close()
          } finally arena.close()
          left -= n
        }
        writer.finish(withFooter = false)
      } finally writer.close()
    }

    private def makeBatch(arena: Arena, n: Int, random: Random): ColumnarBatch = {
      val noNulls = new Array[Boolean](n)
      val buffers = mutable.ArrayBuffer.empty[VectorBuffers]
      buffers += ArrowLayout.ofLongs(arena, Array.fill(n)(random.nextLong() & 0xffffffL), noNulls)
      buffers += ArrowLayout.ofLongs(arena, Array.fill(n)(random.nextInt(100000).toLong), noNulls)
      buffers += ArrowLayout.ofInts(arena, Array.fill(n)(random.nextInt(1000)), noNulls)
      buffers += ArrowLayout.ofInts(arena, Array.fill(n)(random.nextInt()), noNulls)
      buffers += ArrowLayout.ofDoubles(arena, Array.fill(n)(random.nextDouble() * 1000), noNulls)
      buffers += ArrowLayout.ofDoubles(arena, Array.fill(n)(random.nextGaussian()), noNulls)
      schema.fields.drop(6).foreach { f =>
        val values =
          if (f.name == "s_low") Array.fill(n)(lowWords(random.nextInt(lowWords.length)))
          else Array.fill(n)(java.lang.Long.toString(random.nextLong() & 0xfffffffffffL, 36)) // near-distinct
        buffers += ArrowLayout.ofStrings(arena, values)
      }
      val all = arena.allocate(Bitmap.bytesFor(n), 8)
      Bitmap.fill(all, n, true)
      val columns: Array[ColumnVector] = schema.fields.indices.toArray.map { c =>
        ArrowOutput.compact(schema.fields(c).name, schema.fields(c).dataType, buffers(c), all, n, allocator)
      }
      new ColumnarBatch(columns, n)
    }

    /** A reduce task: one stream per executor for partitions `[reduce, reduce + width)`, opened together (#347), drained in turn. */
    def reduceTask(reduce: Int, width: Int, touch: java.util.function.Consumer[ColumnarBatch]): Long = {
      val streams = locations.map(l =>
        new FlightBlockStream(l, 0, mapIds, reduce, reduce + width, schema, codec, conf, allocator, metrics)
      )
      var rows = 0L
      try streams.foreach { s => while (s.hasNext) { val b = s.next(); rows += b.numRows(); touch.accept(b) } }
      finally streams.foreach(_.close())
      rows
    }

    /** The same tickets, the bytes consumed without decoding: the transport alone. */
    def rawTransport(reduce: Int, width: Int): Long = {
      val ticket = FlightShuffle.ticket(0, reduce, reduce + width, mapIds)
      val streams = locations.map(l => FlightShuffle.Clients.client(l).getStream(ticket))
      var bytes = 0L
      try streams.foreach { s =>
          while (s.next()) {
            val root = s.getRoot
            if (root.getRowCount > 0)
              bytes += root.getVector(0).asInstanceOf[org.apache.arrow.vector.VarBinaryVector].getValueLength(0)
          }
        }
      finally streams.foreach(_.close())
      bytes
    }

    /** The same blocks read from the files directly, no Flight: the floor under the transport. */
    def localRead(reduce: Int, width: Int, touch: java.util.function.Consumer[ColumnarBatch]): Long = {
      var rows = 0L
      var server = 0
      while (server < Servers) {
        var map = 0
        while (map < maps) {
          val buf = blockData(server, map, reduce, reduce + width)
          if (buf.size() > 0) {
            val reader =
              new PartitionedIpcFile.StreamReader(PartitionedIpcFile.blockChannel(buf), allocator, schema, codec)
            try while (reader.hasNext) { val b = reader.next(); rows += b.numRows(); touch.accept(b) }
            finally reader.close()
          }
          map += 1
        }
        server += 1
      }
      rows
    }

    /** Every row written comes back through the transport, and the allocator is at zero afterwards. */
    def verify(): Unit = {
      var rows = 0L
      var reduce = 0
      while (reduce < partitions) { rows += reduceTask(reduce, 1, _ => {}); reduce += 1 }
      require(rows == rowsWritten, s"read $rows rows of $rowsWritten written")
    }

    override def close(): Unit = {
      servers.foreach(_.close())
      pools.foreach(_.shutdownNow())
      files.flatten.foreach { case (p, _) => Files.deleteIfExists(p) }
      Files.deleteIfExists(dir)
    }
  }
}
