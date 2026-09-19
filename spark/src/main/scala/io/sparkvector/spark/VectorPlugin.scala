package io.sparkvector.spark

import java.util.Collections

import org.apache.spark.SparkContext
import org.apache.spark.api.plugin.{DriverPlugin, ExecutorPlugin, PluginContext, SparkPlugin}
import org.apache.spark.internal.Logging

/**
 * Spark plugin entry point (`spark.plugins=io.sparkvector.spark.VectorPlugin`). The driver side
 * appends [[VectorSparkSessionExtensions]] to `spark.sql.extensions` so users need a single config
 * key, and attaches the Vector Acceleration tab to the Spark UI; there is no executor-side
 * component beyond the Comet adapter registration.
 *
 * The JVM must be started with `--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED`
 * on both driver and executors; the plugin fails fast if the Vector API module is missing.
 */
class VectorPlugin extends SparkPlugin {

  override def driverPlugin(): DriverPlugin = new DriverPlugin with Logging {
    override def init(sc: SparkContext, ctx: PluginContext): java.util.Map[String, String] = {
      VectorPlugin.requireVectorApi()
      val key = "spark.sql.extensions"
      val ext = classOf[VectorSparkSessionExtensions].getName
      val existing = ctx.conf().getOption(key).map(_.split(",").map(_.trim).filter(_.nonEmpty)).getOrElse(Array.empty)
      if (!existing.contains(ext)) {
        ctx.conf().set(key, (existing :+ ext).mkString(","))
        logInfo(s"spark-vector: registered $ext in $key")
      }
      // The UI tab needs the SparkContext, which only exists here.
      val get: String => Option[String] = k => ctx.conf().getOption(k)
      org.apache.spark.sql.vector.ui.VectorUi.attach(
        sc,
        enabled = VectorConf.uiEnabled(get),
        retainedExecutions = VectorConf.uiRetainedExecutions(get))
      Collections.emptyMap()
    }

    /** The columnar shuffle's Flight location registry (#288), when its module is on the classpath. */
    override def receive(message: AnyRef): AnyRef =
      org.apache.spark.sql.vector.VectorShuffle.driverReceive(message)
  }

  override def executorPlugin(): ExecutorPlugin = new ExecutorPlugin {
    override def init(ctx: PluginContext, extraConf: java.util.Map[String, String]): Unit = {
      VectorPlugin.requireVectorApi()
      io.sparkvector.spark.comet.CometVectorAdapter.tryRegister()
      // Starts the Flight shuffle server on this executor when the shuffle module is present, its
      // manager configured and the backend is Flight (#288).
      org.apache.spark.sql.vector.VectorShuffle.executorInit(ctx)
    }

    override def shutdown(): Unit = org.apache.spark.sql.vector.VectorShuffle.executorShutdown()
  }
}

object VectorPlugin {
  def requireVectorApi(): Unit = {
    if (!ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent) {
      throw new IllegalStateException(
        "spark-vector requires the Java Vector API: start the JVM with " +
          "--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED " +
          "(spark.driver.extraJavaOptions / spark.executor.extraJavaOptions)")
    }
  }
}
