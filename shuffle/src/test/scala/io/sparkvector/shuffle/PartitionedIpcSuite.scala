package io.sparkvector.shuffle

import java.lang.foreign.Arena
import java.math.BigInteger
import java.nio.file.Files
import scala.collection.mutable
import scala.util.Random

import io.sparkvector.kernels.{ArrowLayout, PartitionKernels, SegmentVectorBuffers, VectorBuffers}
import io.sparkvector.kernels.PartitionKernels.KeyKind
import io.sparkvector.spark.arrow.{ArrowOutput, VectorDictionaryColumnVector}
import org.apache.arrow.memory.RootAllocator
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/**
 * #288 slice 1: batches of every lane type, split by Spark-identical hash ids into one IPC stream per
 * partition, read back as the operators' column vectors with the same rows, in order, per partition.
 */
class PartitionedIpcSuite extends AnyFunSuite with BeforeAndAfterAll {

  private val allocator = new RootAllocator()
  private val rnd = new Random(2880)

  override def afterAll(): Unit = allocator.close()

  private val schema = StructType(Seq(
    StructField("i", IntegerType),
    StructField("d", DateType),
    StructField("l", LongType),
    StructField("ts", TimestampType),
    StructField("x", DoubleType),
    StructField("b", BooleanType),
    StructField("s", StringType),
    StructField("sd", StringType), // dictionary encoded at the source
    StructField("dec", DecimalType(12, 2)),
    StructField("wide", DecimalType(30, 4))))

  /** One row as Spark-visible values (null = SQL null), for comparison. */
  private type Row = IndexedSeq[Any]

  private def nulls(n: Int) = Array.fill(n)(rnd.nextDouble() < 0.15)

  /** Builds a batch of `n` rows and the rows it should read back as; `dictStrings` picks the encoding of `sd`. */
  /** The dictionary of the encoded string column: five words, or thousands of names (#345, a slice far smaller than its dictionary). */
  private var bigDictionary = false

  private def batch(n: Int, arena: Arena, dictStrings: Boolean): (ColumnarBatch, IndexedSeq[Row]) = {
    val ni = nulls(n); val vi = Array.fill(n)(rnd.nextInt())
    val nd = nulls(n); val vd = Array.fill(n)(rnd.nextInt(20000))
    val nl = nulls(n); val vl = Array.fill(n)(rnd.nextLong())
    val nt = nulls(n); val vt = Array.fill(n)(rnd.nextLong() / 1000)
    val nx = nulls(n); val vx = Array.fill(n)(rnd.nextGaussian())
    val nb = nulls(n); val vb = Array.fill(n)(rnd.nextBoolean())
    val vs = Array.fill[String](n)(if (rnd.nextDouble() < 0.15) null else rnd.alphanumeric.take(rnd.nextInt(9)).mkString)
    val dict = if (bigDictionary) Array.tabulate(3000)(i => s"name-$i-${rnd.alphanumeric.take(6).mkString}") else Array("alpha", "beta", "gamma", "δέλτα", "😀")
    val nsd = nulls(n); val ids = Array.fill(n)(rnd.nextInt(dict.length))
    val ndec = nulls(n); val vdec = Array.fill(n)(rnd.nextLong() % 1000000000000L)
    val nw = nulls(n); val vw = Array.fill(n)(new BigInteger(90, rnd.self).subtract(BigInteger.ONE.shiftLeft(89)))

    val buffers: Array[VectorBuffers] = Array(
      ArrowLayout.ofInts(arena, vi, ni),
      ArrowLayout.ofInts(arena, vd, nd),
      ArrowLayout.ofLongs(arena, vl, nl),
      ArrowLayout.ofLongs(arena, vt, nt),
      ArrowLayout.ofDoubles(arena, vx, nx),
      ArrowLayout.ofBooleans(arena, vb, nb),
      ArrowLayout.ofStrings(arena, vs),
      if (dictStrings) {
        val idb = ArrowLayout.ofInts(arena, ids, nsd)
        SegmentVectorBuffers.dictionaryUtf8(n, idb.validity(), idb.data(), ArrowLayout.ofStrings(arena, dict))
      } else ArrowLayout.ofStrings(arena, ids.indices.map(i => if (nsd(i)) null else dict(ids(i))).toArray),
      ArrowLayout.ofLongs(arena, vdec, ndec),
      ArrowLayout.ofDecimal128(arena, vw, nw))
    val all = arena.allocate(io.sparkvector.kernels.Bitmap.bytesFor(n), 8)
    io.sparkvector.kernels.Bitmap.fill(all, n, true)
    val columns: Array[ColumnVector] = schema.fields.indices.toArray.map { c =>
      ArrowOutput.compact(schema.fields(c).name, schema.fields(c).dataType, buffers(c), all, n, allocator)
    }
    assert(columns(7).isInstanceOf[VectorDictionaryColumnVector] == dictStrings)
    val rows = (0 until n).map { r =>
      IndexedSeq[Any](
        if (ni(r)) null else vi(r), if (nd(r)) null else vd(r), if (nl(r)) null else vl(r), if (nt(r)) null else vt(r),
        if (nx(r)) null else vx(r), if (nb(r)) null else vb(r), vs(r), if (nsd(r)) null else dict(ids(r)),
        if (ndec(r)) null else Decimal(vdec(r), 12, 2), if (nw(r)) null else Decimal(new java.math.BigDecimal(vw(r), 4), 30, 4))
    }
    (new ColumnarBatch(columns, n), rows)
  }

  private def read(b: ColumnarBatch): IndexedSeq[Row] = (0 until b.numRows()).map { r =>
    schema.fields.indices.map { c =>
      val cv = b.column(c)
      if (cv.isNullAt(r)) null
      else schema.fields(c).dataType match {
        case IntegerType | DateType => cv.getInt(r)
        case LongType | TimestampType => cv.getLong(r)
        case DoubleType => cv.getDouble(r)
        case BooleanType => cv.getBoolean(r)
        case StringType => cv.getUTF8String(r).toString
        case d: DecimalType => cv.getDecimal(r, d.precision, d.scale)
      }
    }
  }

  private def roundTrip(numPartitions: Int, batches: Seq[(Int, Boolean)], flushBytes: Long, batchRows: Int = 8192, bufferBytes: Long = 64L << 20,
      writerAllocator: org.apache.arrow.memory.BufferAllocator = allocator,
      compression: Option[org.apache.arrow.vector.compression.CompressionUtil.CodecType] = Some(org.apache.arrow.vector.compression.CompressionUtil.CodecType.ZSTD)): Long = {
    val dir = Files.createTempDirectory("svipc")
    val path = dir.resolve("map.ipc")
    val expected = Array.fill(numPartitions)(mutable.ArrayBuffer.empty[Row])
    val writer = new PartitionedIpcWriter(schema, numPartitions, writerAllocator, path, flushBytes,
      compression, batchRows, 1L << 20, bufferBytes)
    try {
      batches.foreach { case (n, dictStrings) =>
        val arena = Arena.ofConfined()
        try {
          val (b, rows) = batch(n, arena, dictStrings)
          try {
            // Partition on (i, s): ids from the same kernel the exchange uses.
            val keys = Array(
              io.sparkvector.spark.adapter.ColumnVectorAdapters.adapt(b.column(0), n, arena),
              io.sparkvector.spark.adapter.ColumnVectorAdapters.adapt(b.column(6), n, arena))
            val hashes = new Array[Int](n); val ids = new Array[Int](n)
            PartitionKernels.hashPartitionIds(keys, Array(KeyKind.INT, KeyKind.UTF8), n, numPartitions, hashes, ids)
            writer.write(b, ids)
            rows.indices.foreach(r => expected(ids(r)) += rows(r))
          } finally b.close()
        } finally arena.close()
      }
      val index = writer.finish()
      assert(index.rows.toSeq === expected.map(_.size.toLong).toSeq)
      val fileBytes = Files.size(path)
      for (p <- 0 until numPartitions) {
        val reader = new PartitionedIpcFile.PartitionReader(path, p, allocator, schema)
        try {
          val got = mutable.ArrayBuffer.empty[Row]
          while (reader.hasNext) {
            val b = reader.next()
            try got ++= read(b) finally b.close()
          }
          assert(got === expected(p), s"partition $p")
        } finally reader.close()
      }
      fileBytes
    } finally {
      writer.close()
      Files.deleteIfExists(path)
      Files.deleteIfExists(dir)
    }
  }

  test("#377: a dictionary-encoded column is staged as ids and remapped per block -- the aggregate's shared dictionary across batches") {
    import org.apache.arrow.vector.{IntVector, VarCharVector}
    // One Arrow dictionary vector shared by every batch, as the grouped aggregate emits its keys: the
    // writer maps its entries once and every later batch costs its rows only. Two partitions so a
    // block never sees the whole dictionary, and the blocks' dictionaries must be the used entries.
    val dir = Files.createTempDirectory("svipc")
    val path = dir.resolve("map.ipc")
    val words = Array.tabulate(500)(i => s"product-$i")
    val dictionary = new VarCharVector("k.dictionary", allocator)
    dictionary.allocateNew(words.map(_.length).sum, words.length)
    words.zipWithIndex.foreach { case (w, i) => dictionary.setSafe(i, w.getBytes("UTF-8")) }
    dictionary.setValueCount(words.length)
    val small = StructType(Seq(StructField("k", StringType), StructField("v", IntegerType)))
    val writer = new PartitionedIpcWriter(small, 2, allocator, path, 1L << 20)
    val expected = Array.fill(2)(mutable.ArrayBuffer.empty[(String, Int)])
    try {
      for (batchNo <- 0 until 6) {
        val n = 2000
        val indices = new IntVector("k", allocator)
        indices.allocateNew(n)
        val values = new IntVector("v", allocator)
        values.allocateNew(n)
        val ids = new Array[Int](n)
        (0 until n).foreach { r =>
          val e = (r * 7 + batchNo * 13) % 60 + batchNo * 60 // 60 entries per batch, disjoint across batches
          if (r % 11 == 0) indices.setNull(r) else indices.set(r, e)
          values.set(r, r)
          ids(r) = r % 2
          expected(ids(r)) += ((if (r % 11 == 0) null else words(e), r))
        }
        indices.setValueCount(n); values.setValueCount(n)
        val batch = new ColumnarBatch(Array[ColumnVector](
          new VectorDictionaryColumnVector(indices, dictionary, () => ()), // borrowed dictionary: the aggregate's release hook shape
          new io.sparkvector.spark.arrow.VectorArrowColumnVector(values)), n)
        try writer.write(batch, ids) finally batch.close()
      }
      writer.finish()
      for (p <- 0 until 2) {
        val reader = new PartitionedIpcFile.PartitionReader(path, p, allocator, small)
        val got = mutable.ArrayBuffer.empty[(String, Int)]
        var blocks = 0
        try while (reader.hasNext) {
          val b = reader.next()
          try {
            blocks += 1
            val k = b.column(0)
            assert(k.isInstanceOf[VectorDictionaryColumnVector], "ids over a per-block dictionary")
            val d = k.asInstanceOf[VectorDictionaryColumnVector].dictionary()
            assert(d.getValueCount <= 60 * 6 && d.getValueCount > 0, s"a block's dictionary holds the entries it uses, got ${d.getValueCount}")
            (0 until b.numRows()).foreach(r => got += ((if (k.isNullAt(r)) null else k.getUTF8String(r).toString, b.column(1).getInt(r))))
          } finally b.close()
        } finally reader.close()
        assert(got === expected(p), s"partition $p")
      }
    } finally { writer.close(); dictionary.close(); Files.deleteIfExists(path); Files.deleteIfExists(dir) }
  }

  test("#377: in ids mode a block whose distinct values exceed the ratio goes plain from the staging dictionary, and the cap empties it") {
    // `sd` dictionary-encoded at the source with the 3000-name dictionary and 3000-row batches: every
    // block is nearly all distinct, so #356 sends it plain -- gathered from the staging dictionary, not
    // decoded from the input. A cap of a few kilobytes forces the flush-and-clear between batches; the
    // values must survive both.
    bigDictionary = true
    val dir = Files.createTempDirectory("svipc")
    val path = dir.resolve("map.ipc")
    val writer = new PartitionedIpcWriter(schema, 1, allocator, path, 1L << 20, dictionaryCapBytes = 4096)
    val expected = mutable.ArrayBuffer.empty[Row]
    try {
      for (_ <- 0 until 3) {
        val arena = Arena.ofConfined()
        try {
          val (b, rows) = batch(3000, arena, dictStrings = true)
          try writer.write(b, new Array[Int](3000)) finally b.close()
          expected ++= rows
        } finally arena.close()
      }
      writer.finish()
      val reader = new PartitionedIpcFile.PartitionReader(path, 0, allocator, schema)
      val got = mutable.ArrayBuffer.empty[Row]
      var plainBlocks = 0
      try while (reader.hasNext) {
        val b = reader.next()
        try {
          if (b.column(7).isInstanceOf[io.sparkvector.spark.arrow.VectorArrowColumnVector]) plainBlocks += 1
          got ++= read(b)
        } finally b.close()
      } finally reader.close()
      assert(plainBlocks > 0, "the nearly-distinct column went plain in at least one block")
      assert(got === expected)
    } finally { bigDictionary = false; writer.close(); Files.deleteIfExists(path); Files.deleteIfExists(dir) }
  }

  test("#345: a slice smaller than its dictionary carries only the entries it uses -- 200 partitions, a 3000-name dictionary") {
    bigDictionary = true
    try {
      // 4 x 8192 rows over 200 partitions: ~41 rows per slice against 3000 entries. With the whole
      // dictionary copied per slice the file was ~800 bytes per row; the used entries make it ~100.
      val bytes = roundTrip(numPartitions = 200, batches = Seq.fill(4)((8192, true)), flushBytes = 1L << 20)
      val rows = 4 * 8192
      assert(bytes < rows * 250L, s"$bytes bytes for $rows rows: ${bytes / rows} per row")
    } finally bigDictionary = false
  }

  test("every lane type round-trips per partition, in order, dictionary strings staying encoded") {
    roundTrip(numPartitions = 5, batches = Seq((300, true), (200, true), (0, true), (257, true)), flushBytes = 1L << 20)
  }

  test("a column that is dictionary encoded in one batch and plain in the next shares one stream") {
    roundTrip(numPartitions = 3, batches = Seq((120, true), (130, false), (140, true)), flushBytes = 1L << 20)
  }

  test("streams overflow to a temporary file and are concatenated at finish") {
    // Up to 200 partitions each stream goes straight to its file; above, the heap staging spills past flushBytes.
    roundTrip(numPartitions = 4, batches = Seq((500, true), (500, false), (500, true)), flushBytes = 1024)
    roundTrip(numPartitions = 256, batches = Seq((3000, true), (3000, false)), flushBytes = 1024)
  }

  test("the stream reader decodes several map outputs' streams concatenated, each with its own schema and dictionaries") {
    // What an aggregated partition from a shuffle service holds (future work): map output A's stream for
    // partition p, then map output B's. Dictionaries differ between the two.
    val dir = Files.createTempDirectory("svipc")
    val paths = Seq(dir.resolve("a.ipc"), dir.resolve("b.ipc"))
    val expected = mutable.ArrayBuffer.empty[Row]
    try {
      paths.zip(Seq((70, true), (90, false))).foreach { case (path, (n, dict)) =>
        val writer = new PartitionedIpcWriter(schema, 1, allocator, path, 1L << 20)
        val arena = Arena.ofConfined()
        try {
          val (b, rows) = batch(n, arena, dict)
          try { writer.write(b, new Array[Int](n)); expected ++= rows } finally b.close()
          writer.finish()
        } finally { arena.close(); writer.close() }
      }
      // Each map output as a transport delivers it (#416): its dictionary section, then partition 0's stream.
      val bytes = paths.map(p => PartitionedIpcFile.blockBytes(p, 0, 1))
      val channel = java.nio.channels.Channels.newChannel(new java.io.ByteArrayInputStream(bytes.reduce(_ ++ _)))
      val reader = new PartitionedIpcFile.StreamReader(channel, allocator, schema)
      try {
        val got = mutable.ArrayBuffer.empty[Row]
        while (reader.hasNext) got ++= read(reader.next())
        assert(got === expected)
      } finally reader.close()
    } finally {
      paths.foreach(Files.deleteIfExists); Files.deleteIfExists(dir)
    }
  }

  test("slices of several input batches, dictionary and plain strings mixed, become one record batch with one merged dictionary") {
    // 8192-row record batches over 3 partitions: every partition holds slices of all six inputs until finish.
    roundTrip(numPartitions = 3, batches = Seq((300, true), (200, false), (250, true), (100, false), (400, true), (50, false)), flushBytes = 1L << 20)
    // batchRows = 150 forces flushes mid-way: single-slice and multi-slice record batches alternate in one stream.
    roundTrip(numPartitions = 3, batches = Seq((300, true), (200, false), (250, true), (100, false), (400, true), (50, false)), flushBytes = 1L << 20, batchRows = 150)
    // A tiny task-wide buffer: the fullest partition is written out whenever the cap is passed.
    roundTrip(numPartitions = 5, batches = Seq((500, true), (500, false), (500, true)), flushBytes = 1L << 20, bufferBytes = 4096)
  }

  test("#416: above StagingPartitions the rows are staged and partitioned at the flush -- 400 partitions, several flushes, sliced batches") {
    // The staged path: every partition's rows arrive in one gather per flush; a small bufferBytes forces
    // several flushes (a record batch per partition per flush) and batchRows = 100 slices a partition's rows.
    roundTrip(numPartitions = 400, batches = Seq((3000, true), (2000, false), (2500, true), (1000, false)), flushBytes = 1L << 20)
    roundTrip(numPartitions = 400, batches = Seq((3000, true), (2000, false), (2500, true), (1000, false)), flushBytes = 1L << 20, bufferBytes = 64L << 10)
    roundTrip(numPartitions = 400, batches = Seq((3000, true), (2000, false), (2500, true)), flushBytes = 4096, batchRows = 100)
  }

  test("#340: the writer's real allocation stays within bufferBytes -- 200 partitions of string-heavy batches under a 24 MB limit") {
    // Before #340 the flush decision counted the slices' used bytes while the allocator held their
    // doubled capacity, the per-slice string dictionaries and every partition's last record batch in
    // its root: a 64 MB budget was 1.1 GB in an executor. With the cap on the allocator's own figure
    // and the roots emptied after each batch, a limited allocator is enough for many partitions.
    // This is also the regression test for ShuffleCompression.SafeZstdCodec: with arrow-java 18.3.0's zstd
    // codec the dense flushing here put a compressed buffer next to a pending slice and zstd's 8-byte
    // overrun zeroed the first value of that slice's BigInt column (partition 26, row 85).
    val limited = allocator.newChildAllocator("writer-340", 0L, 24L << 20)
    try {
      roundTrip(numPartitions = 200, batches = Seq.fill(40)((8192, false)), flushBytes = 1L << 20, bufferBytes = 8L << 20, writerAllocator = limited)
      assert(limited.getPeakMemoryAllocation <= (24L << 20), s"peak ${limited.getPeakMemoryAllocation}")
      assert(limited.getAllocatedMemory === 0L, "everything released at close")
    } finally limited.close()
  }

  test("a partition with no rows reads as empty and one partition takes everything") {
    roundTrip(numPartitions = 1, batches = Seq((50, true)), flushBytes = 1L << 20)
    roundTrip(numPartitions = 64, batches = Seq((3, false)), flushBytes = 1L << 20)
  }

  test("#416: small blocks whose strings arrived dictionary-encoded are decoded into the coalesced batch") {
    // batchRows = 400: the 1200 staged rows leave as three record batches, each above DictionaryMinRows
    // (so the five-word column is encoded per batch, with its own dictionary) and below CoalesceRows (so
    // the reader accumulates them) -- one 1200-row batch comes out, every column plain, the rows in order.
    val dir = Files.createTempDirectory("svipc")
    val path = dir.resolve("map.ipc")
    val writer = new PartitionedIpcWriter(schema, 1, allocator, path, 1L << 20, batchRows = 400)
    val arena = Arena.ofConfined()
    val expected = mutable.ArrayBuffer.empty[Row]
    try {
      (0 until 3).foreach { _ =>
        val (b, rows) = batch(400, arena, dictStrings = false)
        try writer.write(b, new Array[Int](400)) finally b.close()
        expected ++= rows
      }
      writer.finish()
      val reader = new PartitionedIpcFile.PartitionReader(path, 0, allocator, schema)
      try {
        val got = reader.next()
        assert(got.numRows() === 1200, "three blocks coalesced into one batch")
        assert(got.column(7).isInstanceOf[io.sparkvector.spark.arrow.VectorArrowColumnVector], "the encoded column comes out plain")
        assert(read(got) === expected)
        assert(!reader.hasNext)
      } finally reader.close()
    } finally { arena.close(); writer.close(); Files.deleteIfExists(path); Files.deleteIfExists(dir) }
  }

  test("#356: a record batch's string column is dictionary-encoded only when the dictionary pays; the reader takes either per batch") {
    import org.apache.arrow.vector.{IntVector, VarCharVector}
    // The encoder itself: all-distinct gives up at the sample (nothing allocated stays behind), repeats encode.
    def strings(values: Seq[String]): VarCharVector = {
      val v = new VarCharVector("t", allocator); v.allocateNew(values.size * 8, values.size)
      values.zipWithIndex.foreach { case (x, i) => if (x == null) v.setNull(i) else v.setSafe(i, x.getBytes("UTF-8")) }
      v.setValueCount(values.size); v
    }
    val before = allocator.getAllocatedMemory
    val distinct = strings((0 until 2000).map(i => s"email-$i@example.com"))
    try assert(PartitionedIpcWriter.encodeStrings(distinct, "t", allocator) == null, "2000 distinct of 2000 goes plain")
    finally distinct.close()
    assert(allocator.getAllocatedMemory === before, "a rejected encoding leaves nothing allocated")
    val repeats = strings((0 until 2000).map(i => if (i % 7 == 0) null else s"state-${i % 40}"))
    try {
      val (ids, dict) = PartitionedIpcWriter.encodeStrings(repeats, "t", allocator)
      try { assert(dict.getValueCount === 40); assert(ids.getValueCount === 2000); assert(ids.isNull(0) && ids.get(1) === 0 && ids.get(41) === 0) }
      finally { ids.close(); dict.close() }
      // ratio 1 always encodes, 0 never.
      val distinct600 = strings((0 until 600).map(i => s"u$i"))
      try { val (i1, d1) = PartitionedIpcWriter.encodeStrings(distinct600, "t", allocator, maxRatio = 1.0); i1.close(); d1.close() }
      finally distinct600.close()
      assert(PartitionedIpcWriter.encodeStrings(repeats, "t", allocator, maxRatio = 0.0) == null)
    } finally repeats.close()

    // Through the writer and the reader (#416: the ratio is judged per task once FreezeSampleRows are
    // seen): `s` (random alphanumerics, nearly all distinct) is frozen during the second batch. With
    // the default batch size nothing has been flushed by then, so its pending ids are decoded in place
    // and every batch comes back plain -- no dictionary section for it (item 5). With 1000-row batches
    // ids reach the file before the freeze: those batches stay encoded over the map file's dictionary
    // and the later ones are plain. `sd` (five words) stays dictionary-encoded throughout; values survive
    // either way.
    def roundTripFreeze(batchRows: Int): (Int, Int, Int) = {
      val dir = Files.createTempDirectory("svipc")
      val path = dir.resolve("map.ipc")
      val writer = new PartitionedIpcWriter(schema, 1, allocator, path, 1L << 20, batchRows = batchRows)
      val arena = Arena.ofConfined()
      try {
        val expected = mutable.ArrayBuffer.empty[Row]
        (0 until 3).foreach { _ =>
          val (b, rows) = batch(3000, arena, dictStrings = false)
          try writer.write(b, new Array[Int](3000)) finally b.close()
          expected ++= rows
        }
        writer.finish()
        val reader = new PartitionedIpcFile.PartitionReader(path, 0, allocator, schema)
        try {
          val got = mutable.ArrayBuffer.empty[Row]
          var plainS = 0; var encodedS = 0; var batches = 0
          while (reader.hasNext) {
            val b = reader.next()
            batches += 1
            if (b.column(6).isInstanceOf[io.sparkvector.spark.arrow.VectorArrowColumnVector]) plainS += 1 else encodedS += 1
            assert(b.column(7).isInstanceOf[VectorDictionaryColumnVector], "dictionary for the five-word column")
            got ++= read(b)
          }
          assert(got === expected)
          (batches, encodedS, plainS)
        } finally reader.close()
      } finally { arena.close(); writer.close(); Files.deleteIfExists(path); Files.deleteIfExists(dir) }
    }
    val (b1, e1, p1) = roundTripFreeze(8192)
    assert(e1 == 0 && p1 >= 1, s"frozen before the first flush: every batch plain ($b1 batches: $e1 encoded, $p1 plain)")
    val (b2, e2, p2) = roundTripFreeze(1000)
    assert(e2 >= 1 && p2 >= 1, s"ids flushed before the freeze stay encoded, the rest plain ($b2 batches: $e2 encoded, $p2 plain)")
  }
}
