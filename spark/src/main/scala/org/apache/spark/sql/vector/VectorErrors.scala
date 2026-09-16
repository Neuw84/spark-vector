package org.apache.spark.sql.vector

import org.apache.spark.QueryContext
import org.apache.spark.sql.errors.QueryExecutionErrors
import org.apache.spark.sql.types.{DataType, Decimal}

/** Spark's error factories are `private[sql]`; this bridge lives inside that package tree. */
object VectorErrors {
  def divideByZero(context: QueryContext): ArithmeticException =
    QueryExecutionErrors.divideByZeroError(context)

  /** `%` / `pmod` by zero in ANSI mode: Spark's REMAINDER_BY_ZERO, distinct from DIVIDE_BY_ZERO. */
  def remainderByZero(context: QueryContext): ArithmeticException =
    QueryExecutionErrors.remainderByZeroError(context)

  /** A decimal result that does not fit the target precision (ANSI arithmetic and casts). */
  def decimalPrecisionOverflow(value: Decimal, precision: Int, scale: Int, context: QueryContext): ArithmeticException =
    QueryExecutionErrors.cannotChangeDecimalPrecisionError(value, precision, scale, context)

  /** ANSI integer overflow (a checked bigint sum). */
  def arithmeticOverflow(message: String, hint: String, context: QueryContext): ArithmeticException =
    QueryExecutionErrors.arithmeticOverflowError(message, hint, context)

  /** An ANSI cast whose value is outside the target type's range. */
  def castOverflow(value: Any, from: DataType, to: DataType): ArithmeticException =
    QueryExecutionErrors.castingCauseOverflowError(value, from, to)
}
