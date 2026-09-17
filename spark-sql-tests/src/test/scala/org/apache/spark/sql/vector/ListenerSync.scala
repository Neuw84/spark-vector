package org.apache.spark.sql.vector

import org.apache.spark.sql.SparkSession

/** Waits until the asynchronous listener bus has delivered every posted event (`private[spark]` in Spark). */
object ListenerSync {
  def drain(spark: SparkSession): Unit = spark.sparkContext.listenerBus.waitUntilEmpty()
}
