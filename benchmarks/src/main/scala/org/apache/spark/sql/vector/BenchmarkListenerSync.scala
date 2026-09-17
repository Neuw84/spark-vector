package org.apache.spark.sql.vector

import org.apache.spark.sql.SparkSession

/**
 * Waits until the listener bus has delivered every queued event: `SparkListenerStageCompleted`
 * arrives asynchronously, so a runner reading stage metrics right after `collect()` would miss
 * the last stages. `waitUntilEmpty` is `private[spark]`, hence this bridge in Spark's package.
 */
object BenchmarkListenerSync {
  def drain(spark: SparkSession): Unit = spark.sparkContext.listenerBus.waitUntilEmpty()
}
