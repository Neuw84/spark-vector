package org.apache.spark.sql.vector

import org.apache.spark.sql.Column
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.classic.ExpressionUtils

/** Test-only bridge: a `Column` over a hand-built catalyst expression (the factory is `private[sql]`). */
object TestExprs {
  def column(e: Expression): Column = ExpressionUtils.column(e)
}
