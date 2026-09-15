package io.sparkvector.spark

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.vector.VectorColumnarRule

/**
 * Registers the planner rule. Enable with
 * `--conf spark.sql.extensions=io.sparkvector.spark.VectorSparkSessionExtensions`, or let
 * [[VectorPlugin]] do it through `--conf spark.plugins=io.sparkvector.spark.VectorPlugin`.
 */
class VectorSparkSessionExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectColumnar(session => VectorColumnarRule(session))
  }
}
