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

import io.sparkvector.spark.comet.CometBatchBridge
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.{Distribution, Partitioning, UnspecifiedDistribution}
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.exchange.{ShuffleExchangeExec, ShuffleExchangeLike, ShuffleOrigin}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Hands spark-vector batches to Comet: every column is exported through the Arrow C Data Interface
 * and imported by Comet's Arrow as a `CometVector`, over the same memory. Inserted below a Comet
 * native shuffle whose input is one of our operators.
 */
case class VectorToCometExec(child: SparkPlan) extends UnaryExecNode {

  override def output: Seq[Attribute] = child.output
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def requiredChildDistribution: Seq[Distribution] = Seq(UnspecifiedDistribution)
  override def supportsColumnar: Boolean = true
  override def nodeName: String = "VectorToComet"

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows")
  )

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val names = output.map(_.name).toArray
    val types = output.map(_.dataType).toArray
    val batches = longMetric("numOutputBatches")
    val rows = longMetric("numOutputRows")
    child.executeColumnar().mapPartitionsInternal { iter =>
      val bridge = CometBatchBridge.tryCreate()
      if (bridge == null) throw new IllegalStateException("Comet classes are not on the executor classpath")
      // Listeners run in reverse registration order, so this frees whatever Comet left unreleased
      // before the child iterators (registered earlier) close their allocators.
      Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => bridge.releaseOutstanding()))
      iter.filter(_.numRows() > 0).map { batch =>
        val converted = bridge.convert(batch, names, types)
        batches += 1
        rows += converted.numRows()
        converted
      }
    }
  }

  override protected def doExecute(): RDD[InternalRow] = ColumnarToRowExec(this).doExecute()

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/**
 * Reflective access to Comet's shuffle exchange, so the plugin compiles without Comet.
 *
 * Comet's planner only gives a Comet shuffle to children it recognises; at run time, though, its
 * native shuffle accepts any columnar child whose batches hold `CometVector`s (it exports them
 * through the Arrow C Stream to the native writer). [[VectorToCometExec]] provides exactly that, so
 * this factory builds a native `CometShuffleExchangeExec` over it.
 */
object CometShuffle {
  private val ExchangeClass = "org.apache.spark.sql.comet.execution.shuffle.CometShuffleExchangeExec"
  private val NativeType = "org.apache.spark.sql.comet.execution.shuffle.CometNativeShuffle$"
  private val ManagerClass = "org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager"

  private lazy val exchange: Option[Class[_]] =
    try Some(Class.forName(ExchangeClass, false, getClass.getClassLoader))
    catch { case _: ClassNotFoundException | _: LinkageError => None }

  def isAvailable: Boolean = exchange.isDefined && CometBatchBridge.isCometLoaded

  /** Comet shuffle on, and Spark configured with Comet's shuffle manager (a static setting). */
  def isEnabled(conf: SQLConf, shuffleManager: String): Boolean = {
    def flag(key: String, alt: String) =
      conf.getConfString(key, conf.getConfString(alt, "true")).trim.equalsIgnoreCase("true")
    isAvailable &&
    flag("spark.comet.shuffle.enabled", "spark.comet.exec.shuffle.enabled") &&
    shuffleManager == ManagerClass
  }

  def isCometExchange(plan: SparkPlan): Boolean = plan.getClass.getName == ExchangeClass

  /** Comet's own switch for native range partitioning (default on). */
  def rangePartitioningEnabled(conf: SQLConf): Boolean =
    conf.getConfString(
      "spark.comet.shuffle.native.partitioning.range.enabled",
      conf.getConfString("spark.comet.native.shuffle.partitioning.range.enabled", "true")
    ).trim.equalsIgnoreCase("true")

  /** True for a Comet exchange already using the native writer. */
  def isNative(plan: SparkPlan): Boolean =
    isCometExchange(plan) && plan.getClass.getMethod("shuffleType").invoke(plan).getClass.getName == NativeType

  /** Comet's native shuffle over `child`, which must produce `CometVector` batches. */
  def native(
      partitioning: Partitioning,
      child: SparkPlan,
      originalPlan: ShuffleExchangeLike,
      origin: ShuffleOrigin,
      advisoryPartitionSize: Option[Long]
  ): SparkPlan = {
    val cls = exchange.getOrElse(throw new IllegalStateException("Comet is not on the classpath"))
    val nativeType = Class.forName(NativeType, true, cls.getClassLoader).getField("MODULE$").get(null)
    val ctor = cls.getConstructors.find(_.getParameterCount == 6)
      .getOrElse(throw new IllegalStateException("unexpected CometShuffleExchangeExec constructor"))
    ctor.newInstance(
      partitioning,
      child,
      originalPlan,
      origin,
      nativeType,
      advisoryPartitionSize
    ).asInstanceOf[SparkPlan]
  }

  /** Rebuilds an existing Comet exchange (any type) as a native one over `newChild`. */
  def toNative(cometExchange: SparkPlan, newChild: SparkPlan): SparkPlan = {
    def field[T](name: String): T = cometExchange.getClass.getMethod(name).invoke(cometExchange).asInstanceOf[T]
    native(
      field[Partitioning]("outputPartitioning"),
      newChild,
      field[ShuffleExchangeLike]("originalPlan"),
      field[ShuffleOrigin]("shuffleOrigin"),
      field[Option[Long]]("advisoryPartitionSize")
    )
  }

  def native(spark: ShuffleExchangeExec, newChild: SparkPlan): SparkPlan =
    native(spark.outputPartitioning, newChild, spark, spark.shuffleOrigin, spark.advisoryPartitionSize)
}
