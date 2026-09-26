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
package org.apache.spark.sql.vector

import java.lang.foreign.Arena

import scala.collection.mutable.ArrayBuffer

import io.vecruntime.kernels.{ColumnBuilder, RunMerge, SortKernels, VecType, VectorBuffers}
import io.vecruntime.spark.arrow.{ArrowOutput, ArrowVectorBuffers, VectorAllocators}
import io.vecruntime.spark.expr.{ColumnRef, VectorExpr}
import org.apache.arrow.memory.BufferAllocator
import org.apache.spark.TaskContext
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

/**
 * Drains the partition into sorted runs of `runRows` rows, then emits the rows in order -- at most
 * `limit` of them -- merging the runs when there are several.
 *
 * Memory is bounded by `spillBytes` (#416): every run lives in an arena of its own, the bytes
 * appended are counted, and once the resident runs pass the budget a newly sealed run is written
 * to a local Arrow IPC file in sorted order and its arena closed. What stays resident is at most the
 * budget plus one run. A spilled run takes part in the merge as a refillable run of [[RunMerge]]:
 * one batch of it is loaded at a time, the merge stops at a batch boundary so the rows already
 * emitted can be gathered from the batch they refer to, and the next batch is loaded after. An
 * output batch is therefore up to `OutputBatchSize` rows, fewer where a spilled run's batch ended.
 */
private[vector] class VectorSortIterator(
    input: Iterator[ColumnarBatch],
    keyExprs: Array[VectorExpr],
    ascending: Array[Boolean],
    nullsFirst: Array[Boolean],
    outputAttrs: Array[(String, DataType)],
    metrics: VectorMetrics,
    limit: Int = Int.MaxValue,
    runRows: Int = 1 << 20,
    spillBytes: Long = Long.MaxValue
) extends Iterator[ColumnarBatch]
    with AutoCloseable {

  /** A run the merge reads: its key columns and output columns as they currently stand. */
  private sealed trait Run {
    def kept: Int
    def keys: Array[VectorBuffers]
    def columns: Array[VectorBuffers]
    def permutation: Array[Int]
    def close(): Unit
  }

  /**
   * A sealed run held in memory: its output columns, its key columns and the permutation that
   * orders it, in an arena of its own so that spilling it frees its memory. Under a limit only the
   * first `limit` rows of the order can ever be emitted, so that is all the merge walks (`kept`): a
   * top-N over a large partition costs the merge n rows per run, not the partition.
   */
  private final class MemoryRun(
      val columns: Array[VectorBuffers],
      val keys: Array[VectorBuffers],
      val rows: Int,
      val bytes: Long,
      val arena: Arena
  ) extends Run {
    val permutation: Array[Int] = SortKernels.sortIndices(keys, ascending, nullsFirst, rows)
    val kept: Int = math.min(rows, limit)
    def close(): Unit = arena.close()
  }

  /**
   * A run spilled to a local Arrow IPC file in sorted order: its `kept` rows in batches of
   * `OutputBatchSize`, the output columns first and the computed keys after them. Read back one
   * batch at a time: `columns` and `keys` are the batch currently loaded, already in order.
   */
  private final class SpilledRun(val file: java.io.File, val kept: Int) extends Run {
    private var channel: java.nio.channels.FileChannel = _
    private var reader: org.apache.arrow.vector.ipc.ArrowStreamReader = _
    var columns: Array[VectorBuffers] = _
    var keys: Array[VectorBuffers] = _
    var rows: Int = 0
    def permutation: Array[Int] = null

    /** Loads the next batch; false when the run is read through. */
    def load(): Boolean = {
      if (reader == null) {
        channel = java.nio.channels.FileChannel.open(file.toPath, java.nio.file.StandardOpenOption.READ)
        reader = new org.apache.arrow.vector.ipc.ArrowStreamReader(channel, allocator)
      }
      if (!reader.loadNextBatch()) { rows = 0; return false }
      val root = reader.getVectorSchemaRoot
      rows = root.getRowCount
      columns = Array.tabulate(numColumns)(c => ArrowVectorBuffers.forRead(root.getVector(c)))
      keys = new Array[VectorBuffers](keyExprs.length)
      var computedIdx = 0
      var k = 0
      while (k < keys.length) {
        if (keyColumn(k) >= 0) keys(k) = columns(keyColumn(k))
        else { keys(k) = ArrowVectorBuffers.forRead(root.getVector(numColumns + computedIdx)); computedIdx += 1 }
        k += 1
      }
      true
    }

    def close(): Unit = {
      if (reader != null) { reader.close(); reader = null }
      if (channel != null) { channel.close(); channel = null }
      file.delete()
    }
  }

  private val OutputBatchSize = 4096

  private val allocator: BufferAllocator = VectorAllocators.newChild("VectorSortExec")

  private val numColumns = outputAttrs.length

  /** Key c is output column `keyColumn(c)`, or -1 when it is a computed expression with its own chunks. */
  private val keyColumn: Array[Int] = keyExprs.map {
    case ColumnRef(ordinal, _) => ordinal
    case _ => -1
  }
  private val computedKeys: Array[Int] = keyColumn.indices.filter(keyColumn(_) < 0).toArray

  private var sorted = false
  private val runs = ArrayBuffer.empty[Run]

  /** Bytes of the runs held in memory (sealed and being built), the quantity `spillBytes` bounds. */
  private var residentBytes = 0L

  /** Runs written to disk and their bytes, for the operator's description and the tests. */
  var spilledRuns = 0
  var spilledBytes = 0L

  /** One run: its columns and permutation. Several: the merge, and every output column's run columns. */
  private var columns: Array[VectorBuffers] = _
  private var permutation: Array[Int] = _
  private var merge: RunMerge = _
  private var runColumns: Array[Array[VectorBuffers]] = _
  private var runOf: Array[Int] = _
  private var rowOf: Array[Int] = _
  private var total = 0
  private var emitted = 0
  private var current: ColumnarBatch = _
  private var closed = false

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => close()))

  /** Rough size of `count` rows of `in`: the fixed width, or a string column's mean bytes per row plus its offset. */
  private def estimateBytes(in: VectorBuffers, count: Int): Long = {
    val n = math.max(in.length(), 1)
    val perRow: Long = in.`type`() match {
      case VecType.UTF8 =>
        val off = in.offsets()
        val data = if (off == null) 0L
        else off.get(VectorBuffers.LE_INT, n.toLong << 2).toLong - off.get(VectorBuffers.LE_INT, 0L)
        4L + math.max(0L, data) / n
      case VecType.BOOL => 1L
      case t => math.max(1, t.byteWidth)
    }
    perRow * count
  }

  /** Writes `run`'s kept rows in sorted order to a local file and frees its memory. */
  private def spill(run: MemoryRun): SpilledRun = {
    val file = AggregateSpill.newFile()
    val types: Array[(String, DataType)] = outputAttrs ++ computedKeys.map(k => (s"key$k", keyExprs(k).dataType))
    Option(file.getParentFile).foreach(_.mkdirs())
    val channel = java.nio.channels.FileChannel.open(
      file.toPath,
      java.nio.file.StandardOpenOption.CREATE,
      java.nio.file.StandardOpenOption.WRITE,
      java.nio.file.StandardOpenOption.TRUNCATE_EXISTING
    )
    var writer: org.apache.arrow.vector.ipc.ArrowStreamWriter = null
    var root: org.apache.arrow.vector.VectorSchemaRoot = null
    try {
      var from = 0
      while (from < run.kept) {
        val to = math.min(run.kept, from + OutputBatchSize)
        val vectors = new Array[org.apache.arrow.vector.FieldVector](types.length)
        var c = 0
        while (c < types.length) {
          val (name, dt) = types(c)
          val source = if (c < numColumns) run.columns(c) else run.keys(computedKeys(c - numColumns))
          vectors(c) =
            AggregateSpill.vectorOf(ArrowOutput.gather(name, dt, source, run.permutation, from, to, allocator))
          c += 1
        }
        if (writer == null) {
          root = new org.apache.arrow.vector.VectorSchemaRoot(java.util.Arrays.asList(vectors: _*))
          writer = new org.apache.arrow.vector.ipc.ArrowStreamWriter(root, null, channel)
          writer.start()
        } else {
          // The root keeps its first vectors: later batches move theirs in.
          c = 0
          while (c < types.length) {
            vectors(c).makeTransferPair(root.getVector(c)).transfer(); vectors(c).close(); c += 1
          }
        }
        root.setRowCount(to - from)
        writer.writeBatch()
        from = to
      }
      if (writer != null) writer.end()
      AggregateSpill.reportSpill(channel.size(), run.bytes) // before the writer's close, which closes the channel
    } finally {
      if (writer != null) writer.close()
      if (root != null) root.close()
      channel.close()
    }
    spilledRuns += 1
    spilledBytes += run.bytes
    run.close()
    new SpilledRun(file, run.kept)
  }

  private def drainAndSort(): Unit = {
    if (sorted) return
    sorted = true
    var columnBuilders: Array[ColumnBuilder] = null
    var keyBuilders: Array[ColumnBuilder] = null
    var runArena: Arena = null
    var runTotal = 0
    var runBytes = 0L

    /** Seals the builders into a run: plain string columns (a dictionary decoded once), the keys, the permutation. */
    def seal(): Unit = {
      if (runTotal > 0) {
        val cols = columnBuilders.map(_.view()).map { v =>
          if (v.`type`() == VecType.UTF8 && v.isDictionaryEncoded) ArrowOutput.decodeDictionary(v, runArena) else v
        }
        val computed = keyBuilders.map(_.view())
        val keys = new Array[VectorBuffers](keyExprs.length)
        var computedIdx = 0
        var k = 0
        while (k < keys.length) {
          if (keyColumn(k) >= 0) keys(k) = cols(keyColumn(k))
          else { keys(k) = computed(computedIdx); computedIdx += 1 }
          k += 1
        }
        val run = new MemoryRun(cols, keys, runTotal, runBytes, runArena)
        total += runTotal
        // Past the budget the newest run goes to disk in sorted order and its memory is returned;
        // what stays resident is at most the budget plus one run.
        if (residentBytes > spillBytes) {
          runs += spill(run)
          residentBytes -= runBytes
        } else {
          runs += run
        }
        runArena = null
      } else if (runArena != null) {
        runArena.close(); runArena = null
      }
      columnBuilders = null
      keyBuilders = null
      runTotal = 0
      runBytes = 0L
    }

    while (input.hasNext) {
      val batch = input.next()
      if (batch.numRows() > 0) {
        metrics.timed {
          metrics.numInputBatches += 1
          EvalContexts.withBatch(batch) { ctx =>
            val count = ctx.selectedCount
            if (count > 0) {
              if (columnBuilders == null) {
                val expected = math.max(math.min(runRows, count), 1)
                runArena = Arena.ofShared()
                columnBuilders =
                  Array.tabulate(numColumns)(c => new ColumnBuilder(runArena, ctx.input(c).`type`(), expected))
                keyBuilders = Array.tabulate(computedKeys.length)(k =>
                  new ColumnBuilder(runArena, keyExprs(computedKeys(k)).vecType, expected)
                )
              }
              var c = 0
              while (c < numColumns) {
                val in = ctx.input(c)
                columnBuilders(c).append(in, ctx.selection, count)
                val b = estimateBytes(in, count)
                runBytes += b; residentBytes += b
                c += 1
              }
              var k = 0
              while (k < computedKeys.length) {
                val v = keyExprs(computedKeys(k)).eval(ctx)
                keyBuilders(k).append(v, ctx.selection, count)
                val b = estimateBytes(v, count)
                runBytes += b; residentBytes += b
                k += 1
              }
              runTotal += count
              if (runTotal >= runRows) seal()
            }
          }
        }
      }
    }
    metrics.timed {
      seal()
      // Every spilled run's first batch is loaded so the merge can seat it; an empty one leaves.
      val loaded = runs.filter {
        case r: SpilledRun => r.load()
        case _ => true
      }
      runs.clear(); runs ++= loaded
      if (runs.length == 1 && runs.head.isInstanceOf[MemoryRun]) {
        columns = runs.head.columns
        permutation = runs.head.permutation
      } else if (runs.nonEmpty) {
        merge = new RunMerge(
          runs.map(_.keys).toArray,
          runs.map(_.permutation).toArray,
          runs.map { case r: MemoryRun => r.kept; case r: SpilledRun => r.rows }.toArray,
          ascending,
          nullsFirst,
          runs.map(_.isInstanceOf[SpilledRun]).toArray
        )
        runColumns = Array.tabulate(numColumns)(c => runs.map(_.columns(c)).toArray)
        runOf = new Array[Int](OutputBatchSize)
        rowOf = new Array[Int](OutputBatchSize)
      }
      // A top-N emits only the head of the order (every run was still sorted whole).
      total = math.min(total, limit)
    }
  }

  override def hasNext: Boolean = {
    drainAndSort()
    emitted < total
  }

  override def next(): ColumnarBatch = {
    if (!hasNext) throw new NoSuchElementException("no more sorted rows")
    releaseCurrent()
    val from = emitted
    val out = new Array[ColumnVector](numColumns)
    var count = 0
    metrics.timed {
      if (merge == null) {
        val to = math.min(total, from + OutputBatchSize)
        count = to - from
        var c = 0
        while (c < numColumns) {
          val (name, dt) = outputAttrs(c)
          out(c) = ArrowOutput.gather(name, dt, columns(c), permutation, from, to, allocator)
          c += 1
        }
      } else {
        // A batch ends early when a spilled run's loaded piece is used up: the rows so far are
        // gathered from the pieces as they stand, then that run is refilled from its file.
        count = merge.next(runOf, rowOf, math.min(OutputBatchSize, total - from))
        var c = 0
        while (c < numColumns) {
          val (name, dt) = outputAttrs(c)
          out(c) = ArrowOutput.gatherRuns(name, dt, runColumns(c), runOf, rowOf, count, allocator)
          c += 1
        }
        val ex = merge.exhausted()
        if (ex >= 0) {
          val run = runs(ex).asInstanceOf[SpilledRun]
          if (run.load()) {
            merge.refill(ex, run.keys, null, run.rows)
            c = 0
            while (c < numColumns) { runColumns(c)(ex) = run.columns(c); c += 1 }
          } else merge.finish(ex)
        }
      }
    }
    emitted = from + count
    metrics.numOutputBatches += 1
    metrics.numOutputRows += count
    current = new ColumnarBatch(out, count)
    current
  }

  private def releaseCurrent(): Unit = {
    if (current != null) { current.close(); current = null }
  }

  override def close(): Unit = {
    if (!closed) {
      closed = true
      releaseCurrent()
      columns = null
      permutation = null
      merge = null
      runColumns = null
      runs.foreach(_.close())
      runs.clear()
      allocator.close()
    }
  }
}
