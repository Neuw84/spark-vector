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
package org.apache.spark.sql.vecruntime

import org.apache.spark.SparkConf
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec

/**
 * Reflective hook to the columnar shuffle of the `vecruntime-shuffle` module (#288), which is
 * optional on the classpath (it carries Arrow Flight and gRPC) and only works under its own
 * `spark.shuffle.manager`, so the planner checks both before it replaces an exchange.
 */
object VectorShuffle {
  val ExchangeClass = "org.apache.spark.sql.vecruntime.VectorShuffleExchangeExec"
  val ManagerClass = "org.apache.spark.sql.vecruntime.shuffle.VectorShuffleManager"

  private lazy val companion: Option[AnyRef] =
    try {
      val cls = Class.forName(ExchangeClass + "$", true, getClass.getClassLoader)
      Some(cls.getField("MODULE$").get(null))
    } catch { case _: ClassNotFoundException | _: NoClassDefFoundError => None }

  def isAvailable(conf: SparkConf): Boolean =
    companion.isDefined && conf.get("spark.shuffle.manager", "sort") == ManagerClass

  def supports(partitioning: Partitioning, output: Seq[Attribute]): Boolean = companion.exists { c =>
    c.getClass.getMethod(
      "supports",
      classOf[Partitioning],
      classOf[Seq[_]]
    ).invoke(c, partitioning, output).asInstanceOf[Boolean]
  }

  val RegistryClass = "org.apache.spark.sql.vecruntime.shuffle.flight.FlightRegistry"

  private lazy val registry: Option[AnyRef] =
    try {
      val cls = Class.forName(RegistryClass + "$", true, getClass.getClassLoader)
      Some(cls.getField("MODULE$").get(null))
    } catch { case _: ClassNotFoundException | _: NoClassDefFoundError => None }

  /** Driver plugin `receive`: the Flight registry answers, or null when the module is absent. */
  def driverReceive(message: AnyRef): AnyRef = registry.map { r =>
    r.getClass.getMethod("driverReceive", classOf[AnyRef]).invoke(r, message)
  }.orNull

  def executorInit(ctx: org.apache.spark.api.plugin.PluginContext): Unit = registry.foreach { r =>
    r.getClass.getMethod("executorInit", classOf[org.apache.spark.api.plugin.PluginContext]).invoke(r, ctx)
  }

  def executorShutdown(): Unit = registry.foreach { r =>
    r.getClass.getMethod("executorShutdown").invoke(r)
  }

  /** `VectorShuffleExchangeExec(partitioning, child, origin, advisoryPartitionSize)` over the same child. */
  def exchange(s: ShuffleExchangeExec, materializedKeys: Int = 0): SparkPlan = {
    val cls = Class.forName(ExchangeClass, true, getClass.getClassLoader)
    val ctor = cls.getConstructors.find(_.getParameterCount == 5).get
    ctor.newInstance(
      s.outputPartitioning,
      s.child,
      s.shuffleOrigin,
      s.advisoryPartitionSize,
      Integer.valueOf(materializedKeys)
    ).asInstanceOf[SparkPlan]
  }
}
