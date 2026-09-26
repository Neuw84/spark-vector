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
package org.apache.spark.sql.vector.ui

import java.util.concurrent.ConcurrentHashMap

import scala.jdk.CollectionConverters._

/** One query as the tab shows it. */
final case class ExecutionSummary(
    executionId: Long,
    description: String,
    submissionTime: Long,
    completionTime: Option[Long],
    failed: Boolean,
    plan: AcceleratedPlan,
    /** True once classified from the real operators rather than the serialised plan. */
    exact: Boolean,
    physicalPlanDescription: String
) {

  def status: String =
    if (failed) "FAILED" else if (completionTime.isDefined) "COMPLETED" else "RUNNING"

  def duration: Option[Long] = completionTime.map(_ - submissionTime)
}

/**
 * The tab's state: the last `retained` SQL executions with their classified plans.
 *
 * Written by [[VectorAccelerationListener]] on the listener bus thread, read by the UI pages on
 * Jetty threads, hence the concurrent map and the single `synchronized` block for eviction.
 */
class VectorAccelerationStore(val retained: Int) {

  private val executions = new ConcurrentHashMap[Long, ExecutionSummary]()

  def update(summary: ExecutionSummary): Unit = {
    executions.put(summary.executionId, summary)
    evict()
  }

  /** Keeps an existing entry's fields when only some of them are known. */
  def updateWith(executionId: Long)(f: ExecutionSummary => ExecutionSummary): Unit =
    Option(executions.get(executionId)).foreach(existing => update(f(existing)))

  def get(executionId: Long): Option[ExecutionSummary] = Option(executions.get(executionId))

  /** Newest first, which is the order the list page shows. */
  def list(): Seq[ExecutionSummary] =
    executions.values().asScala.toSeq.sortBy(-_.executionId)

  private def evict(): Unit = synchronized {
    val excess = executions.size() - retained
    if (excess > 0) {
      executions.keySet().asScala.toSeq.sorted.take(excess).foreach(executions.remove)
    }
  }
}
