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
package org.apache.spark.sql.vector.shuffle

import org.apache.spark.{ShuffleDependency, SparkConf, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.shuffle.{
  ShuffleBlockResolver,
  ShuffleHandle,
  ShuffleManager,
  ShuffleReadMetricsReporter,
  ShuffleReader,
  ShuffleWriteMetricsReporter,
  ShuffleWriter
}
import org.apache.spark.sql.vecruntime.shuffle.{VectorShuffleManager => VecRuntimeShuffleManager}

/**
 * Deprecated compatibility alias for the columnar shuffle manager.
 *
 * The class was renamed in 0.0.2: its package moved from `org.apache.spark.sql.vector.shuffle` to
 * `org.apache.spark.sql.vecruntime.shuffle`. Use
 * `org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager` in
 * `spark.shuffle.manager` instead. This alias lets an existing
 * `spark.shuffle.manager=org.apache.spark.sql.vector.shuffle.VectorShuffleManager` keep working; it
 * is a thin delegate to the new manager (the new manager is `final`, so this cannot subclass it) and
 * will be removed in a later release.
 *
 * @deprecated renamed in 0.0.2; use
 *   `org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager`. To be removed in a later release.
 */
@deprecated(
  "renamed in 0.0.2 to org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager; " +
    "this alias will be removed in a later release",
  "0.0.2"
)
final class VectorShuffleManager(conf: SparkConf) extends ShuffleManager with Logging {

  logWarning(
    "spark.shuffle.manager=org.apache.spark.sql.vector.shuffle.VectorShuffleManager is deprecated " +
      "(renamed in 0.0.2). Use org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager; " +
      "this alias will be removed in a later release."
  )

  private val delegate = new VecRuntimeShuffleManager(conf)

  override def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]
  ): ShuffleHandle = delegate.registerShuffle(shuffleId, dependency)

  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter
  ): ShuffleWriter[K, V] = delegate.getWriter(handle, mapId, context, metrics)

  override def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter
  ): ShuffleReader[K, C] =
    delegate.getReader(handle, startMapIndex, endMapIndex, startPartition, endPartition, context, metrics)

  override def unregisterShuffle(shuffleId: Int): Boolean = delegate.unregisterShuffle(shuffleId)

  override def shuffleBlockResolver: ShuffleBlockResolver = delegate.shuffleBlockResolver

  override def stop(): Unit = delegate.stop()
}
