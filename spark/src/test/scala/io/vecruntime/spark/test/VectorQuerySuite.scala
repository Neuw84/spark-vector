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
package io.sparkvector.spark.test

import io.sparkvector.spark.{VectorConf, VectorSparkSessionExtensions}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.vector.{PlanUtils, VectorFallback}

/**
 * Base class for plugin-on/plugin-off comparison suites. Every query is run twice on the same
 * session, toggling `spark.vector.enabled`, and the results are compared with a tolerance for
 * doubles (SIMD reductions reorder floating-point additions).
 */
abstract class VectorQuerySuite extends SparkVectorFunSuite {

  override protected def extraSparkConf: Map[String, String] = Map(
    "spark.sql.extensions" -> classOf[VectorSparkSessionExtensions].getName,
    "spark.sql.parquet.enableVectorizedReader" -> "true",
    "spark.sql.adaptive.enabled" -> "true"
  )

  protected def withConf[T](pairs: (String, String)*)(f: => T): T = {
    val previous = pairs.map { case (k, _) => k -> spark.conf.getOption(k) }
    pairs.foreach { case (k, v) => spark.conf.set(k, v) }
    try f
    finally previous.foreach {
        case (k, Some(v)) => spark.conf.set(k, v)
        case (k, None) => spark.conf.unset(k)
      }
  }

  protected def withPlugin[T](enabled: Boolean)(f: => T): T =
    withConf(VectorConf.Enabled -> enabled.toString)(f)

  /** The final physical plan, after adaptive execution has run. */
  protected def finalPlan(df: DataFrame): SparkPlan = df.queryExecution.executedPlan

  protected def nodesOf[T <: SparkPlan](df: DataFrame)(implicit ct: scala.reflect.ClassTag[T]): Seq[T] =
    PlanUtils.allNodes(finalPlan(df)).collect { case p: T => p }

  /**
   * Runs `sql` with and without the plugin, asserts identical results and that each of
   * `expectedOperators` appears in the final plan. Returns the plugin-enabled DataFrame.
   */
  protected def checkVectorized(
      sql: String,
      expectedOperators: Seq[Class[_ <: SparkPlan]],
      tolerance: Double = 1e-9
  ): DataFrame = {
    val expected = withPlugin(enabled = false)(spark.sql(sql).collect())
    val df = withPlugin(enabled = true) {
      val d = spark.sql(sql)
      d.collect() // materialise so AQE has finalised the plan
      d
    }
    val actual = df.collect()
    assertRowsEqual(expected, actual, tolerance, sql)
    val nodes = PlanUtils.allNodes(finalPlan(df))
    expectedOperators.foreach { cls =>
      assert(
        nodes.exists(n => cls.isInstance(n)),
        s"expected ${cls.getSimpleName} in plan for: $sql\n${finalPlan(df).treeString}\n" +
          s"fallbacks: ${VectorFallback.reasons(finalPlan(df)).map(_._2).mkString("; ")}"
      )
    }
    df
  }

  /** Runs `sql` with the plugin and asserts none of `operators` appear, with a matching reason. */
  protected def checkFallback(
      sql: String,
      operators: Seq[Class[_ <: SparkPlan]],
      reasonContains: String
  ): DataFrame = {
    val df = withPlugin(enabled = true) {
      val d = spark.sql(sql)
      d.collect()
      d
    }
    val nodes = PlanUtils.allNodes(finalPlan(df))
    operators.foreach { cls =>
      assert(!nodes.exists(n => cls.isInstance(n)), s"did not expect ${cls.getSimpleName} in plan for: $sql")
    }
    val reasons = VectorFallback.reasons(finalPlan(df)).map(_._2)
    assert(
      reasons.exists(_.contains(reasonContains)),
      s"expected a fallback reason containing '$reasonContains', got: ${reasons.mkString("; ")}\n${finalPlan(df).treeString}"
    )
    df
  }

  protected def assertRowsEqual(expected: Array[Row], actual: Array[Row], tolerance: Double, context: String): Unit = {
    assert(expected.length === actual.length, s"row count differs for: $context")
    // Sort on a key that rounds doubles so last-bit differences cannot reorder rows.
    def key(r: Seq[Any]): String = r.map {
      case d: Double => f"$d%.6f"
      case null => "<null>"
      case v => v.toString
    }.mkString("|")
    val e = expected.map(normalise).sortBy(key)
    val a = actual.map(normalise).sortBy(key)
    e.zip(a).zipWithIndex.foreach { case ((er, ar), i) =>
      assert(er.length === ar.length, s"column count differs at row $i for: $context")
      er.zip(ar).zipWithIndex.foreach { case ((ev, av), c) =>
        (ev, av) match {
          case (x: Double, y: Double) =>
            val ok = (x.isNaN && y.isNaN) || x == y || math.abs(x - y) <= tolerance * math.max(
              1.0,
              math.max(math.abs(x), math.abs(y))
            )
            assert(ok, s"double mismatch at row $i col $c: $x vs $y for: $context")
          case (x, y) => assert(x == y, s"mismatch at row $i col $c: $x vs $y for: $context")
        }
      }
    }
  }

  /** Rows as plain Seqs; doubles are rounded for sorting only. */
  private def normalise(r: Row): Seq[Any] = r.toSeq.map {
    case d: java.lang.Double => d.doubleValue()
    case f: java.lang.Float => f.doubleValue()
    case bd: java.math.BigDecimal => bd.doubleValue()
    case other => other
  }
}
