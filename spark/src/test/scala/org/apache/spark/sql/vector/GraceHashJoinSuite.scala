package org.apache.spark.sql.vector

import scala.collection.mutable.ArrayBuffer

import io.sparkvector.spark.expr.ColumnRef
import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.types.{IntegerType, LongType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.{TaskCompletionListener, TaskFailureListener}
import org.scalatest.funsuite.AnyFunSuite

/**
 * The grace hash join (#416) driven directly, under a task context that only records what is
 * registered on it: the memory the bucketing path holds per task must not grow with the buckets.
 */
class GraceHashJoinSuite extends AnyFunSuite {

  /** Streamed `(k: long)` on the left, build `(bk: long, v: int)` on the right, joined on `k = bk`. */
  private val spec = JoinSpec(
    joinType = Inner,
    buildIsLeft = false,
    buildKeys = Array(ColumnRef(0, LongType)),
    streamedKeys = Array(ColumnRef(0, LongType)),
    condition = None,
    outputAttrs = Array("k" -> LongType, "bk" -> LongType, "v" -> IntegerType),
    joinedAttrs = Array("k" -> LongType, "bk" -> LongType, "v" -> IntegerType),
    buildTypes = Array(LongType, IntegerType),
    streamedWidth = 1
  )

  private def metrics =
    new VectorMetrics(new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("sum"), new SQLMetric("sum"))

  private def batches(
      schema: StructType,
      batchRows: Int,
      numBatches: Int
  )(row: Long => InternalRow): Iterator[ColumnarBatch] =
    Iterator.tabulate(numBatches) { b =>
      VectorRowStages.toBatch(schema, Array.tabulate(batchRows)(i => row(b.toLong * batchRows + i)))
    }

  private def rowsOf(it: Iterator[ColumnarBatch]): (Long, Long) = {
    var rows = 0L
    var keySum = 0L
    while (it.hasNext) {
      val batch = it.next()
      rows += batch.numRows()
      var i = 0
      while (i < batch.numRows()) { keySum += batch.column(0).getLong(i); i += 1 }
    }
    (rows, keySum)
  }

  private def join(budgetBytes: Long, buckets: Int): ((Long, Long), RecordingTaskContext) = {
    val ctx = new RecordingTaskContext
    TaskContext.setTaskContext(ctx)
    try {
      val build = batches(new StructType().add("bk", LongType).add("v", IntegerType), 4096, 8) { i =>
        InternalRow(i, (i % 7).toInt)
      }
      val streamed = batches(new StructType().add("k", LongType), 4096, 8) { i => InternalRow(i * 2) }
      val result = rowsOf(GraceHashJoin.iterator(build, streamed, spec, metrics, budgetBytes, buckets))
      ctx.complete()
      (result, ctx)
    } finally TaskContext.unset()
  }

  test("#416: a bucketed join registers one task-completion listener, not one per bucket") {
    // Keys 0 until 32768 on the build side, even keys 0 until 65536 streamed: 16384 matches.
    val expected = (16384L, (0L until 32768L by 2).sum)
    val (inMemory, ctxMem) = join(budgetBytes = Long.MaxValue, buckets = 32)
    assert(inMemory === expected)
    val (bucketed, ctxBuckets) = join(budgetBytes = 1, buckets = 32)
    assert(bucketed === expected)
    // Every registration keeps its closure -- the iterator, its build table and heap arrays -- reachable
    // until the task ends, so the bucketing path must register exactly what the in-memory path does.
    assert(ctxMem.listeners === 1)
    assert(ctxBuckets.listeners === 1, "one listener per bucket would hold every bucket's table until task end")
  }

  /** A task context that records the completion listeners and runs them on `complete()`; nothing else. */
  private final class RecordingTaskContext extends TaskContext {
    private val completion = new ArrayBuffer[TaskCompletionListener]
    def listeners: Int = completion.size
    def complete(): Unit = completion.reverseIterator.foreach(_.onTaskCompletion(this))

    override def addTaskCompletionListener(listener: TaskCompletionListener): TaskContext = {
      completion += listener; this
    }
    override def addTaskFailureListener(listener: TaskFailureListener): TaskContext = this
    override def isCompleted(): Boolean = false
    override def isFailed(): Boolean = false
    override def isInterrupted(): Boolean = false
    override def stageId(): Int = 0
    override def stageAttemptNumber(): Int = 0
    override def partitionId(): Int = 0
    override def numPartitions(): Int = 1
    override def attemptNumber(): Int = 0
    override def taskAttemptId(): Long = 0L
    override def getLocalProperty(key: String): String = null
    override def cpus(): Int = 1
    override def resources(): Map[String, org.apache.spark.resource.ResourceInformation] = Map.empty
    override def resourcesJMap(): java.util.Map[String, org.apache.spark.resource.ResourceInformation] =
      java.util.Collections.emptyMap()
    override def taskMetrics(): org.apache.spark.executor.TaskMetrics = null
    override def getMetricsSources(sourceName: String): Seq[org.apache.spark.metrics.source.Source] = Nil
    override private[spark] def killTaskIfInterrupted(): Unit = ()
    override private[spark] def getKillReason(): Option[String] = None
    override private[spark] def taskMemoryManager(): org.apache.spark.memory.TaskMemoryManager = null
    override private[spark] def registerAccumulator(a: org.apache.spark.util.AccumulatorV2[_, _]): Unit = ()
    override private[spark] def setFetchFailed(e: org.apache.spark.shuffle.FetchFailedException): Unit = ()
    override private[spark] def markInterrupted(reason: String): Unit = ()
    override private[spark] def markTaskFailed(error: Throwable): Unit = ()
    override private[spark] def markTaskCompleted(error: Option[Throwable]): Unit = ()
    override private[spark] def fetchFailed: Option[org.apache.spark.shuffle.FetchFailedException] = None
    override private[spark] def getLocalProperties: java.util.Properties = new java.util.Properties
    override private[spark] def interruptible(): Boolean = false
    override private[spark] def pendingInterrupt(threadToInterrupt: Option[Thread], reason: String): Unit = ()
    override private[spark] def createResourceUninterruptibly[T <: java.io.Closeable](resourceBuilder: => T): T =
      resourceBuilder
  }
}
