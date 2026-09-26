/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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

import scala.collection.mutable
import scala.jdk.CollectionConverters._

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.connector.write.{DeltaWrite, WriterCommitMessage}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.v2.V2CommandExec
import org.apache.spark.sql.catalyst.util.WriteDeltaProjections
import org.apache.spark.TaskContext

/**
 * The columnar Iceberg v3 deletion-vector delete command (#20, slice 4-live). It replaces Spark's
 * row-by-row `WriteDeltaExec` on a format-version-3 merge-on-read table: instead of routing each
 * deleted `InternalRow` through Iceberg's per-row `DeltaWriter.delete`, each task groups the delta
 * rows by their `_file` metadata column (they arrive clustered by the write's REBALANCE exchange),
 * builds one deletion-vector bitmap per data file through the optional `iceberg-bridge` module, and
 * emits one `DeltaTaskCommit`. The driver hands the messages to Iceberg's OWN `DeltaBatchWrite.commit`
 * (obtained from `write.toBatch()`), so the commit -- snapshot summary, previous-DV merge, metrics --
 * is entirely Iceberg's; on failure it calls `abort`.
 *
 * <p>This first landing handles the DELETE half only (delete-only tables, and the delete side of
 * UPDATE/MERGE); insert rows are out of scope here (slice 5). The planner strategy only produces this
 * node when every input row is a delete (see [[VectorWriteDeltaStrategy]]).
 *
 * @param table          the Iceberg table, as `AnyRef` (the core has no Iceberg compile dep)
 * @param write          the Iceberg `DeltaWrite`
 * @param projections    Spark's row/rowId/metadata projections for the delta rows
 * @param query          the child plan producing the delta rows
 */
case class VectorWriteDeltaExec(
    table: AnyRef,
    write: DeltaWrite,
    projections: WriteDeltaProjections,
    query: SparkPlan
) extends V2CommandExec
    with Logging {

  override def output: Seq[Attribute] = Nil
  override def children: Seq[SparkPlan] = Seq(query)
  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(query = newChildren.head)

  override protected def run(): Seq[InternalRow] = {
    val batchWrite = write.toBatch
    val rdd: RDD[InternalRow] = query.execute()
    val proj = projections
    val tbl = table
    val messages = new mutable.ArrayBuffer[WriterCommitMessage]()
    try {
      val collected: Array[WriterCommitMessage] =
        rdd.mapPartitions(iter => Iterator.single(VectorWriteDeltaExec.writePartition(tbl, proj, iter))).collect()
      messages ++= collected.filter(_ != null)
      batchWrite.commit(messages.toArray)
      logInfo(s"spark-vector: columnar v3 DV delete committed ${messages.size} task message(s)")
    } catch {
      case t: Throwable =>
        try batchWrite.abort(messages.toArray)
        catch { case a: Throwable => t.addSuppressed(a) }
        throw t
    }
    Nil
  }
}

object VectorWriteDeltaExec {

  /**
   * One task: read the delta rows, extract each row's `_spec_id` / `_partition` / `_file` / `_pos`
   * from the rowId projection, group consecutive equal `_file` into runs, and write one deletion
   * vector per data file through the bridge. Returns the task's commit message.
   */
  private def writePartition(
      table: AnyRef,
      projections: WriteDeltaProjections,
      rows: Iterator[InternalRow]
  ): WriterCommitMessage = {
    val tc = TaskContext.get()
    val partitionId = if (tc != null) tc.partitionId() else 0
    val taskId = if (tc != null) tc.taskAttemptId() else 0L

    val rowId = projections.rowIdProjection
    // The rowId projection carries {_spec_id, _partition, _file, _pos}; resolve ordinals by name so a
    // schema reordering across Iceberg versions cannot silently mis-read them.
    val rowIdSchema = rowId.schema
    def ordinal(name: String): Int = {
      val i = rowIdSchema.fieldIndex(name)
      i
    }

    val fileOrd = ordinal("_file")
    val posOrd = ordinal("_pos")
    val specOrdOpt = if (rowIdSchema.fieldNames.contains("_spec_id")) Some(ordinal("_spec_id")) else None
    val partOrdRowId = if (rowIdSchema.fieldNames.contains("_partition")) Some(ordinal("_partition")) else None

    val writer = IcebergDvBridge.createTaskWriter(table, partitionId, taskId)
    try {
      var currentFile: String = null
      var currentSpec: Int = 0
      var currentPartition: InternalRow = null
      val positions = new mutable.ArrayBuilder.ofLong
      var count = 0

      def flush(): Unit = {
        if (currentFile != null && count > 0) {
          writer.deleteFile(currentFile, positions.result(), count, currentSpec, currentPartition)
        }
        positions.clear()
        count = 0
      }

      while (rows.hasNext) {
        val row = rows.next()
        rowId.project(row)
        val file = rowId.getUTF8String(fileOrd).toString
        val pos = rowId.getLong(posOrd)
        val specId = specOrdOpt.map(rowId.getInt).getOrElse(0)
        val partition: InternalRow = partOrdRowId match {
          case Some(o) if !rowId.isNullAt(o) =>
            rowId.getStruct(o, rowIdSchema.fields(o).dataType.asInstanceOf[org.apache.spark.sql.types.StructType].size)
          case _ => null
        }
        if (file != currentFile || specId != currentSpec) {
          flush()
          currentFile = file
          currentSpec = specId
          currentPartition = if (partition != null) partition.copy() else null
        }
        positions += pos
        count += 1
      }
      flush()
      writer.commit()
    } finally {
      writer.close()
    }
  }
}
