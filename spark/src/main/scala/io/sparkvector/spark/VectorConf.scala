package io.sparkvector.spark

import org.apache.spark.sql.internal.SQLConf

/** Configuration keys for the spark-vector plugin. All keys are read from the session's SQLConf. */
object VectorConf {

  val Enabled = "spark.vector.enabled"
  val FilterEnabled = "spark.vector.exec.filter.enabled"
  val ProjectEnabled = "spark.vector.exec.project.enabled"
  val AggregateEnabled = "spark.vector.exec.aggregate.enabled"
  val ExplainFallbackEnabled = "spark.vector.explainFallback.enabled"

  def isEnabled(conf: SQLConf): Boolean = bool(conf, Enabled, default = true)
  def filterEnabled(conf: SQLConf): Boolean = bool(conf, FilterEnabled, default = true)
  def projectEnabled(conf: SQLConf): Boolean = bool(conf, ProjectEnabled, default = true)
  def aggregateEnabled(conf: SQLConf): Boolean = bool(conf, AggregateEnabled, default = true)
  def explainFallback(conf: SQLConf): Boolean = bool(conf, ExplainFallbackEnabled, default = false)

  private def bool(conf: SQLConf, key: String, default: Boolean): Boolean =
    conf.getConfString(key, default.toString).trim.equalsIgnoreCase("true")
}
