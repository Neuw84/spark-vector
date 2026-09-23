package org.apache.spark.sql.vector.shuffle.flight

import java.lang.foreign.Arena
import java.nio.ByteBuffer
import java.nio.file.Files

import scala.collection.mutable

import io.sparkvector.shuffle.{PartitionedIpcFile, PartitionedIpcWriter}
import io.sparkvector.kernels.ArrowLayout
import io.sparkvector.spark.arrow.ArrowOutput
import org.apache.arrow.flight.{FlightServer, Location}
import org.apache.arrow.memory.RootAllocator
import org.apache.spark.SparkConf
import org.apache.spark.network.buffer.NioManagedBuffer
import org.apache.spark.sql.types._
import org.apache.spark.sql.vector.shuffle.VectorShuffleBackend
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}
import org.scalatest.funsuite.AnyFunSuite

/**
 * #338: a block's record batches each carry their own (replacement) dictionary for the string
 * columns -- the slice's distinct values. Flight's record-batch framing sends a stream's
 * dictionaries once, so a block re-framed by Flight reached the client with batches 2..n indexed
 * against batch 1's dictionary: the wrong string, or an index past the dictionary. The block now
 * travels as bytes and the client reads it with the local path's decoder. This is the in-process
 * check: one server, one block of two batches with different dictionaries, read back row for row.
 */
class FlightBlockStreamSuite extends AnyFunSuite {

  private val schema = StructType(Seq(StructField("k", IntegerType), StructField("s", StringType)))

  /** A batch of `n` rows whose strings all start with `prefix`: two batches share no dictionary entry. */
  private def batch(
      arena: Arena,
      allocator: org.apache.arrow.memory.BufferAllocator,
      n: Int,
      prefix: String
  ): (ColumnarBatch, IndexedSeq[(Int, String)]) = {
    val ks = Array.tabulate(n)(i => i * 7)
    val ss = Array.tabulate(n)(i => s"$prefix-${i % 37}") // 37 distinct per batch, repeats within it
    val buffers = Array(ArrowLayout.ofInts(arena, ks, Array.fill(n)(false)), ArrowLayout.ofStrings(arena, ss))
    val all = arena.allocate(io.sparkvector.kernels.Bitmap.bytesFor(n), 8)
    io.sparkvector.kernels.Bitmap.fill(all, n, true)
    val columns: Array[ColumnVector] = schema.fields.indices.toArray.map { c =>
      ArrowOutput.compact(schema.fields(c).name, schema.fields(c).dataType, buffers(c), all, n, allocator)
    }
    (new ColumnarBatch(columns, n), (0 until n).map(i => (ks(i), ss(i))))
  }

  test("a remote block of two record batches with different string dictionaries reads back row for row") {
    val allocator = new RootAllocator()
    val dir = Files.createTempDirectory("svflight")
    val path = dir.resolve("block.ipc")
    val expected = mutable.ArrayBuffer.empty[(Int, String)]
    // batchRows = 500: the 1500 staged rows leave as three record batches, each with its own dictionary (#416).
    val writer = new PartitionedIpcWriter(schema, 1, allocator, path, 1L << 20, batchRows = 500)
    val arena = Arena.ofConfined()
    try {
      Seq("alpha", "beta", "gamma").foreach { prefix =>
        val (b, rows) = batch(arena, allocator, 500, prefix)
        try { writer.write(b, new Array[Int](500)); expected ++= rows }
        finally b.close()
      }
      writer.finish()
    } finally { arena.close(); writer.close() }
    val block = PartitionedIpcFile.blockBytes(path, 0, 1) // the dictionary section, then partition 0 (#416)

    val producer = new FlightShuffle.Producer((_, _, _) => new NioManagedBuffer(ByteBuffer.wrap(block)), allocator)
    val server = FlightServer.builder(allocator, Location.forGrpcInsecure("127.0.0.1", 0), producer).build()
    server.start()
    try {
      val stream = new FlightBlockStream(
        FlightLocation("127.0.0.1", server.getPort),
        0,
        0L,
        0,
        schema,
        new SparkConf(false),
        FlightShuffle.Clients.allocatorForReads,
        new org.apache.spark.executor.TempShuffleReadMetrics()
      )
      try {
        val got = mutable.ArrayBuffer.empty[(Int, String)]
        var batches = 0
        while (stream.hasNext) {
          val b = stream.next()
          batches += 1
          (0 until b.numRows()).foreach(r => got += ((b.column(0).getInt(r), b.column(1).getUTF8String(r).toString)))
        }
        assert(
          batches === 3,
          "three 500-row encoded record batches are each above PassEncodedRows and go through as ids (#416)"
        )
        assert(got === expected)
      } finally stream.close()
    } finally {
      server.close()
      allocator.close()
    }
  }
  test("#347: one DoGet carries several map outputs for a reducer, back to back, an empty one included") {
    val allocator = new RootAllocator()
    val dir = Files.createTempDirectory("svflight")
    val arena = Arena.ofConfined()
    // Three map outputs of one reduce partition, each its own file; map 1 wrote nothing.
    val blocks = mutable.Map.empty[Long, Array[Byte]]
    val expected = mutable.Map.empty[Long, IndexedSeq[(Int, String)]]
    try {
      Seq(0L, 2L, 5L).foreach { mapId =>
        val path = dir.resolve(s"map$mapId.ipc")
        val writer = new PartitionedIpcWriter(schema, 1, allocator, path, 1L << 20, batchBytes = 1L)
        try {
          val (b, rows) = batch(arena, allocator, 300 + mapId.toInt, s"m$mapId")
          try { writer.write(b, new Array[Int](b.numRows())); expected(mapId) = rows }
          finally b.close()
          writer.finish()
        } finally writer.close()
        blocks(mapId) = PartitionedIpcFile.blockBytes(path, 0, 1)
      }
      blocks(1L) = Array.emptyByteArray
      expected(1L) = IndexedSeq.empty
    } finally arena.close()

    val served = mutable.ArrayBuffer.empty[Long]
    val producer = new FlightShuffle.Producer(
      (_, mapId, _) => { served += mapId; new NioManagedBuffer(ByteBuffer.wrap(blocks(mapId))) },
      allocator
    )
    val server = FlightServer.builder(allocator, Location.forGrpcInsecure("127.0.0.1", 0), producer).build()
    server.start()
    try {
      val mapIds = Seq(0L, 1L, 2L, 5L)
      val stream = new FlightBlockStream(
        FlightLocation("127.0.0.1", server.getPort),
        0,
        mapIds,
        0,
        schema,
        new SparkConf(false),
        FlightShuffle.Clients.allocatorForReads,
        new org.apache.spark.executor.TempShuffleReadMetrics()
      )
      try {
        val got = mutable.ArrayBuffer.empty[(Int, String)]
        while (stream.hasNext) {
          val b = stream.next()
          (0 until b.numRows()).foreach(r => got += ((b.column(0).getInt(r), b.column(1).getUTF8String(r).toString)))
        }
        assert(got === mapIds.flatMap(expected), "the blocks' rows in map-id order")
        assert(served.toSeq === mapIds, "each block asked of the resolver once, in the ticket's order")
      } finally stream.close()
    } finally {
      server.close()
      allocator.close()
    }
  }

  test(
    "#411: one DoGet carries a reduce task's partition range per map output -- one lookup per map, the empty partition a zero-length span"
  ) {
    val allocator = new RootAllocator()
    val dir = Files.createTempDirectory("svflight")
    val arena = Arena.ofConfined()
    // Two map outputs over four partitions; the task reads partitions [1, 4). Partition 2 gets no row
    // from either map. Rows go to partitions 1 and 3 alternately (partition 0 is outside the range).
    val files = mutable.Map.empty[Long, (java.nio.file.Path, PartitionedIpcFile.Index)]
    val expected = mutable.Map.empty[Long, IndexedSeq[(Int, String)]]
    try {
      Seq(0L, 3L).foreach { mapId =>
        val path = dir.resolve(s"map$mapId.ipc")
        val writer = new PartitionedIpcWriter(schema, 4, allocator, path, 1L << 20, batchBytes = 1L)
        try {
          val (b, rows) = batch(arena, allocator, 200 + mapId.toInt, s"m$mapId")
          val ids = Array.tabulate(b.numRows())(i => if (i % 3 == 0) 0 else if (i % 3 == 1) 1 else 3)
          try {
            writer.write(b, ids)
            // The stream delivers partition 1's rows, then partition 3's, in input order within each.
            expected(mapId) =
              rows.indices.filter(i => ids(i) == 1).map(rows) ++ rows.indices.filter(i => ids(i) == 3).map(rows)
          } finally b.close()
          writer.finish()
        } finally writer.close()
        val index = {
          val ch = java.nio.channels.FileChannel.open(path);
          try PartitionedIpcFile.readIndex(ch)
          finally ch.close()
        }
        files(mapId) = (path, index)
      }
    } finally arena.close()

    val asked = mutable.ArrayBuffer.empty[(Long, Int, Int)]
    val producer = new FlightShuffle.Producer(
      (_: Int, mapId: Long, start: Int, end: Int) => {
        asked += ((mapId, start, end))
        val (path, _) = files(mapId)
        new NioManagedBuffer(ByteBuffer.wrap(PartitionedIpcFile.blockBytes(path, start, end)))
      },
      allocator
    )
    val server = FlightServer.builder(allocator, Location.forGrpcInsecure("127.0.0.1", 0), producer).build()
    server.start()
    try {
      val mapIds = Seq(0L, 3L)
      val stream = new FlightBlockStream(
        FlightLocation("127.0.0.1", server.getPort),
        0,
        mapIds,
        1,
        4,
        schema,
        Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.ZSTD),
        new SparkConf(false),
        FlightShuffle.Clients.allocatorForReads,
        new org.apache.spark.executor.TempShuffleReadMetrics()
      )
      try {
        val got = mutable.ArrayBuffer.empty[(Int, String)]
        while (stream.hasNext) {
          val b = stream.next()
          (0 until b.numRows()).foreach(r => got += ((b.column(0).getInt(r), b.column(1).getUTF8String(r).toString)))
        }
        assert(
          got === mapIds.flatMap(expected),
          "partitions 1 and 3 of each map, in map order; partition 0 excluded, 2 empty"
        )
        assert(asked.toSeq === Seq((0L, 1, 4), (3L, 1, 4)), "one range lookup per map output")
      } finally stream.close()
    } finally {
      server.close()
      allocator.close()
    }
  }

  test(
    "#364: a remote executor that refuses connections is a FetchFailedException for its address, not a plain failure"
  ) {
    // A port nothing listens on: the connect fails at open or on the first read, depending on the transport.
    val closed = new java.net.ServerSocket(0)
    val port = closed.getLocalPort
    closed.close()
    val address = org.apache.spark.storage.BlockManagerId("exec-9", "127.0.0.1", 7079)
    val blockId = org.apache.spark.storage.ShuffleBlockId(3, 11L, 5)
    def fail(e: Throwable): Nothing =
      throw new org.apache.spark.shuffle.FetchFailedException(
        address,
        blockId.shuffleId,
        blockId.mapId,
        4,
        blockId.reduceId,
        s"refused: $e",
        e
      )
    def open(): Iterator[ColumnarBatch] with AutoCloseable =
      try {
        val s = new FlightBlockStream(
          FlightLocation("127.0.0.1", port),
          3,
          11L,
          5,
          schema,
          new SparkConf(false),
          FlightShuffle.Clients.allocatorForReads,
          new org.apache.spark.executor.TempShuffleReadMetrics()
        )
        VectorShuffleBackend.fetchFailing(s, fail)
      } catch { case e: Exception if !VectorShuffleBackend.isMemory(e) => fail(e) }
    // The connect fails at open (the stream's constructor asks for the DoGet) or on the first read.
    val ex = intercept[org.apache.spark.shuffle.FetchFailedException] { open().hasNext }
    val reason = ex.toTaskFailedReason.asInstanceOf[org.apache.spark.FetchFailed]
    assert(reason.bmAddress === address)
    assert(reason.shuffleId === 3 && reason.mapId === 11L && reason.mapIndex === 4 && reason.reduceId === 5)
    // Memory errors are not fetch failures.
    val oom = VectorShuffleBackend.fetchFailing(
      new Iterator[ColumnarBatch] with AutoCloseable {
        override def hasNext: Boolean = throw new org.apache.arrow.memory.OutOfMemoryException("full")
        override def next(): ColumnarBatch = throw new NoSuchElementException
        override def close(): Unit = ()
      },
      fail
    )
    intercept[org.apache.arrow.memory.OutOfMemoryException] { oom.hasNext }
  }
}
