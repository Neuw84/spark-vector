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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write.WriterCommitMessage

/**
 * Reflective hook to the optional `spark-vector-iceberg-bridge` module (#20), which is compiled
 * against Iceberg and holds the columnar deletion-vector writer's Iceberg-typed code
 * (`DvDeltaTaskWriter`, `IcebergDvCommitBridge`). The plugin core has no compile-time Iceberg
 * dependency, so -- exactly like [[VectorShuffle]] and the reader adapter -- every call here is
 * reflective and returns/takes only Spark and primitive types. When the bridge (or Iceberg) is not on
 * the classpath the hooks report "unavailable" and the planner leaves Spark's own `WriteDeltaExec` in
 * place.
 */
object IcebergDvBridge {
  private val CommitBridge = "org.apache.iceberg.spark.source.IcebergDvCommitBridge"
  private val TaskWriter = "io.vecruntime.iceberg.bridge.DvDeltaTaskWriter"

  private lazy val commitBridgeClass: Option[Class[_]] =
    try Some(Class.forName(CommitBridge, true, getClass.getClassLoader))
    catch { case _: ClassNotFoundException | _: NoClassDefFoundError => None }

  private lazy val taskWriterClass: Option[Class[_]] =
    try Some(Class.forName(TaskWriter, true, getClass.getClassLoader))
    catch { case _: ClassNotFoundException | _: NoClassDefFoundError => None }

  /** Whether the bridge module and Iceberg are on the classpath. */
  def isAvailable: Boolean = commitBridgeClass.isDefined && taskWriterClass.isDefined

  /** Whether an Iceberg table (passed as `AnyRef` to avoid a compile dep) writes deletion vectors (v3). */
  def isDvEligible(table: AnyRef): Boolean = commitBridgeClass.exists { c =>
    val m = c.getMethods.find(x => x.getName == "isDvEligible" && x.getParameterCount == 1)
    m.exists(_.invoke(null, table).asInstanceOf[Boolean])
  }

  /** Whether the table's current spec is unpartitioned (the operator's supported case this landing). */
  def isUnpartitioned(table: AnyRef): Boolean = commitBridgeClass.exists { c =>
    val m = c.getMethods.find(x => x.getName == "isUnpartitioned" && x.getParameterCount == 1)
    m.exists(_.invoke(null, table).asInstanceOf[Boolean])
  }

  /**
   * Whether the table already carries committed delete files. When it does, the columnar operator
   * declines (a repeated delete would index a second DV for a data file that already has one, which
   * Iceberg rejects); merging previous DVs is a later slice. Fails safe to `true` (decline).
   */
  def hasCommittedDeletes(table: AnyRef): Boolean = commitBridgeClass.forall { c =>
    val m = c.getMethods.find(x => x.getName == "hasCommittedDeletes" && x.getParameterCount == 1)
    m.forall(_.invoke(null, table).asInstanceOf[Boolean])
  }

  /** Creates a per-task DV writer handle (reflective wrapper over the bridge's DvDeltaTaskWriter). */
  def createTaskWriter(table: AnyRef, partitionId: Int, taskId: Long): IcebergDvTaskWriterHandle = {
    val c = taskWriterClass.getOrElse(
      throw new IllegalStateException("spark-vector: iceberg-bridge module not on the classpath")
    )
    val create = c.getMethods.find(m => m.getName == "create" && m.getParameterCount == 4).get
    // rewritableDeletes = null in this first landing (append-only deletes; previous DVs are merged by
    // Iceberg's BaseDVFileWriter.close only when a rewrite map is supplied -- slice for repeated merges).
    val instance = create.invoke(null, table, Integer.valueOf(partitionId), java.lang.Long.valueOf(taskId), null)
    new IcebergDvTaskWriterHandle(instance)
  }
}

/**
 * Reflective wrapper over one `io.vecruntime.iceberg.bridge.DvDeltaTaskWriter` instance. Created and
 * used entirely on the executor, so it holds the instance and looks its methods up once.
 */
final class IcebergDvTaskWriterHandle(instance: AnyRef) {
  private val cls = instance.getClass
  private val deleteFileM =
    cls.getMethods.find(m => m.getName == "deleteFile" && m.getParameterCount == 5).get
  private val commitM = cls.getMethod("commit")
  private val closeM = cls.getMethod("close")

  def deleteFile(path: String, positions: Array[Long], count: Int, specId: Int, partition: InternalRow): Unit =
    deleteFileM.invoke(
      instance,
      path,
      positions,
      Integer.valueOf(count),
      Integer.valueOf(specId),
      partition
    )

  def commit(): WriterCommitMessage = commitM.invoke(instance).asInstanceOf[WriterCommitMessage]
  def close(): Unit = closeM.invoke(instance)
}
