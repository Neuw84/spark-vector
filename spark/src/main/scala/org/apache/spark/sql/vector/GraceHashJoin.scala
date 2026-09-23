package org.apache.spark.sql.vector

import java.lang.foreign.Arena

import io.sparkvector.kernels.{ColumnBuilder, VecType, VectorBuffers}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{ArrowOutput, VectorAllocators}
import io.sparkvector.spark.expr.EvalContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * The shuffled hash join of one task past its memory budget (#416): a grace hash join.
 *
 * The build side is held in memory while it fits `budgetBytes`, and then joined as today. Past the
 * budget every build row -- those held and those still to come -- is bucketed by a hash of its keys
 * into `buckets` local Arrow files, every streamed row is bucketed the same way, and the buckets are
 * joined one at a time: bucket `b`'s build rows become a [[BuildTable]], bucket `b`'s streamed rows
 * probe it through the unchanged [[VectorHashJoinIterator]], which also emits the bucket's unmatched
 * build rows for a build-side outer join. A key's rows meet in one bucket on both sides, so every
 * join type is correct per bucket; memory is one bucket's table, not the build side.
 *
 * The bucket hash is [[AggregateSpill]]'s -- Spark's hash of the keys under a seed the shuffle does
 * not use (the rows of one task all agree on the shuffle's hash modulo the partition count, so that
 * one would fill `buckets / gcd(partitions, buckets)` buckets). Null keys hash to one bucket on both
 * sides and never match there, as in memory. A key with more rows than the budget still builds as one
 * bucket, with a warning: the same memory the in-memory join took for it.
 *
 * What the task holds is bounded by the budget, not proportional to it (#416, the second round; TPC-DS
 * q95 at 1 TB with 13 tasks an executor): the builders are owning [[ColumnBuilder]]s, whose grown-out
 * buffers are released at once, and a batch is appended only when the capacity it grows the builders
 * to stays within the budget -- so the held rows occupy at most the budget, not four times it through
 * the doubling's leftovers; and the held rows are bucketed in [[SpillChunkRows]]-row slices, so the
 * bucketing's scratch (two `int` arrays per row on the heap, one mask per bucket in the arena, the
 * compacted Arrow buffers) is that of a slice, not of 67 M rows at once.
 */
object GraceHashJoin extends Logging {

  /**
   * Rows of the held build side bucketed per call: at 32 buckets a record batch of ~8192 rows per
   * bucket, the join's own batch size. A multiple of 64, so a slice starts on a bitmap byte.
   */
  val SpillChunkRows: Int = 262144

  def iterator(
      buildIter: Iterator[ColumnarBatch],
      streamIter: Iterator[ColumnarBatch],
      spec: JoinSpec,
      metrics: VectorMetrics,
      budgetBytes: Long,
      buckets: Int
  ): Iterator[ColumnarBatch] = {
    val keyTypes: Array[DataType] = spec.buildKeys.map(_.dataType)
    // A key type the bucket hash has no lane for (float, binary...): the build side stays in memory.
    val canSpill =
      budgetBytes > 0 && budgetBytes < Long.MaxValue && buckets >= 2 && AggregateSpill.supportsKeys(keyTypes.toSeq)

    val arena = Arena.ofShared() // the table's scratch; the columns live in the builders
    val builders = spec.buildTypes.map(dt => ColumnBuilder.owning(TypeMapping.vecTypeOf(dt), 4096))
    var total = 0
    // The batch that would have grown the builders past the budget: not appended, bucketed first below.
    var overflowBatch: ColumnarBatch = null
    while (buildIter.hasNext && overflowBatch == null) {
      val batch = buildIter.next()
      if (batch.numRows() > 0) {
        EvalContexts.withBatch(batch) { ctx =>
          if (canSpill && bytesAfter(builders, ctx) > budgetBytes) overflowBatch = batch
          else {
            var c = 0
            while (c < builders.length) { builders(c).append(ctx.input(c), ctx.selection, ctx.selectedCount); c += 1 }
            total += ctx.selectedCount
          }
        }
      }
    }
    if (overflowBatch == null) {
      // As before #416: the whole build side in one table.
      val table = new BuildTable(arena, builders.map(_.view()), total, spec, builders = builders).build()
      return new VectorHashJoinIterator(streamIter, table, spec, metrics)
    }

    logInfo(
      s"hash join: build side past $budgetBytes bytes after $total rows (${heldBytes(builders)} bytes held), bucketing both sides into $buckets buckets on disk"
    )
    val allocator = VectorAllocators.newChild("VectorShuffledHashJoinExec.grace")
    val buildAttrs =
      if (spec.buildIsLeft) spec.joinedAttrs.take(spec.buildTypes.length) else spec.joinedAttrs.drop(spec.streamedWidth)
    val streamedAttrs =
      if (spec.buildIsLeft) spec.joinedAttrs.drop(spec.buildTypes.length) else spec.joinedAttrs.take(spec.streamedWidth)
    val buildSpill = new AggregateSpill(buckets, buildAttrs, null, allocator, AggregateSpill.BucketSeed, keyTypes)
    val probeSpill = new AggregateSpill(buckets, streamedAttrs, null, allocator, AggregateSpill.BucketSeed, keyTypes)
    var spilledRows = 0L

    // The rows held so far, a slice at a time (the class note), then the batch that overflowed and the
    // rest of the build side, into the build buckets.
    val views = builders.map(_.view())
    var from = 0
    while (from < total) {
      val to = math.min(total, from + SpillChunkRows)
      val scratch = Arena.ofConfined()
      try {
        val slice = views.map(v => plain(v.slice(from, to), scratch))
        val ctx = new EvalContext(scratch, to - from, c => slice(c))
        buildSpill.writeBuffers(slice, spec.buildKeys.map(_.eval(ctx)), to - from, scratch)
      } finally scratch.close()
      from = to
    }
    spilledRows += total
    builders.foreach(_.close())
    arena.close()
    def spillBuild(batch: ColumnarBatch): Unit = if (batch.numRows() > 0) EvalContexts.withBatch(batch) { ctx =>
      val cols = Array.tabulate(spec.buildTypes.length)(c => plain(ctx.input(c), ctx.arena))
      buildSpill.writeBuffers(cols, spec.buildKeys.map(_.eval(ctx)), ctx.numRows, ctx.arena, ctx.selection)
      spilledRows += ctx.selectedCount
    }
    spillBuild(overflowBatch)
    while (buildIter.hasNext) spillBuild(buildIter.next())
    // Every streamed row into the probe buckets.
    while (streamIter.hasNext) {
      val batch = streamIter.next()
      if (batch.numRows() > 0) EvalContexts.withBatch(batch) { ctx =>
        val cols = Array.tabulate(spec.streamedWidth)(c => plain(ctx.input(c), ctx.arena))
        probeSpill.writeBuffers(cols, spec.streamedKeys.map(_.eval(ctx)), ctx.numRows, ctx.arena, ctx.selection)
        spilledRows += ctx.selectedCount
      }
    }
    logInfo(
      s"hash join: $spilledRows rows bucketed, ${buildSpill.spilledBatches + probeSpill.spilledBatches} batches written; joining $buckets buckets"
    )

    // Bucket by bucket: the bucket's build table, probed by the bucket's streamed rows.
    new Iterator[ColumnarBatch] with AutoCloseable {
      private var bucket = 0
      private var current: VectorHashJoinIterator = _
      private var currentProbe: Iterator[ColumnarBatch] with AutoCloseable = _
      private var closed = false
      Option(org.apache.spark.TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

      private def openNext(): Boolean = {
        while (bucket < buckets) {
          val b = bucket
          bucket += 1
          val buildRows = buildSpill.read(b)
          val table =
            try BuildTable.fromBatches(buildRows, spec)
            finally buildRows.close()
          currentProbe = probeSpill.read(b)
          // Closed by this iterator (closeCurrent, and the one task listener above), never by a listener
          // of its own: one per bucket kept every bucket's table on the heap until the task ended.
          current = new VectorHashJoinIterator(currentProbe, table, spec, metrics, closeOnTaskEnd = false)
          if (current.hasNext) return true
          closeCurrent()
        }
        false
      }

      private def closeCurrent(): Unit = {
        if (current != null) { current.close(); current = null }
        if (currentProbe != null) { currentProbe.close(); currentProbe = null }
      }

      override def hasNext: Boolean = {
        if (closed) return false
        while (current == null || !current.hasNext) {
          closeCurrent()
          if (!openNext()) { close(); return false }
        }
        true
      }

      override def next(): ColumnarBatch = {
        if (!hasNext) throw new NoSuchElementException("no more joined batches")
        current.next()
      }

      override def close(): Unit = if (!closed) {
        closed = true
        closeCurrent()
        buildSpill.close()
        probeSpill.close()
        allocator.close()
      }
    }
  }

  /** The bytes the builders hold (capacity: data, offsets, validity), for the log line. */
  private def heldBytes(builders: Array[ColumnBuilder]): Long = {
    var bytes = 0L
    var c = 0
    while (c < builders.length) { bytes += builders(c).allocatedBytes(); c += 1 }
    bytes
  }

  /** The bytes the builders would hold after appending the batch of `ctx` -- the budget is checked before, not after. */
  private def bytesAfter(builders: Array[ColumnBuilder], ctx: EvalContext): Long = {
    var bytes = 0L
    var c = 0
    while (c < builders.length) {
      bytes += builders(c).bytesAfterAppend(ctx.input(c), ctx.selection, ctx.selectedCount)
      c += 1
    }
    bytes
  }

  /** The buckets are written plain: a dictionary-encoded string column is decoded once here. */
  private def plain(b: VectorBuffers, arena: Arena): VectorBuffers =
    if (b.`type`() == VecType.UTF8 && b.isDictionaryEncoded()) ArrowOutput.decodeDictionary(b, arena) else b
}
