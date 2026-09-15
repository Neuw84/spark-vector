package org.apache.spark.sql.vector

import org.apache.spark.QueryContext
import org.apache.spark.sql.errors.QueryExecutionErrors

/** Spark's error factories are `private[sql]`; this bridge lives inside that package tree. */
object VectorErrors {
  def divideByZero(context: QueryContext): ArithmeticException =
    QueryExecutionErrors.divideByZeroError(context)
}
