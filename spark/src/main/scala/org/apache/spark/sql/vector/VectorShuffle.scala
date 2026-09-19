package org.apache.spark.sql.vector

import org.apache.spark.SparkConf
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec

/**
 * Reflective hook to the columnar shuffle of the `spark-vector-shuffle` module (#288), which is
 * optional on the classpath (it carries Arrow Flight and gRPC) and only works under its own
 * `spark.shuffle.manager`, so the planner checks both before it replaces an exchange.
 */
object VectorShuffle {
  val ExchangeClass = "org.apache.spark.sql.vector.VectorShuffleExchangeExec"
  val ManagerClass = "org.apache.spark.sql.vector.shuffle.VectorShuffleManager"

  private lazy val companion: Option[AnyRef] =
    try {
      val cls = Class.forName(ExchangeClass + "$", true, getClass.getClassLoader)
      Some(cls.getField("MODULE$").get(null))
    } catch { case _: ClassNotFoundException | _: NoClassDefFoundError => None }

  def isAvailable(conf: SparkConf): Boolean =
    companion.isDefined && conf.get("spark.shuffle.manager", "sort") == ManagerClass

  def supports(partitioning: Partitioning, output: Seq[Attribute]): Boolean = companion.exists { c =>
    c.getClass.getMethod("supports", classOf[Partitioning], classOf[Seq[_]]).invoke(c, partitioning, output).asInstanceOf[Boolean]
  }

  /** `VectorShuffleExchangeExec(partitioning, child, origin, advisoryPartitionSize)` over the same child. */
  def exchange(s: ShuffleExchangeExec): SparkPlan = {
    val cls = Class.forName(ExchangeClass, true, getClass.getClassLoader)
    val ctor = cls.getConstructors.find(_.getParameterCount == 4).get
    ctor.newInstance(s.outputPartitioning, s.child, s.shuffleOrigin, s.advisoryPartitionSize).asInstanceOf[SparkPlan]
  }
}
