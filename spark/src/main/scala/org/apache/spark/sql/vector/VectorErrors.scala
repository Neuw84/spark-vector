/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
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
package org.apache.spark.sql.vector

import org.apache.spark.QueryContext
import org.apache.spark.sql.catalyst.util.StringUtils
import org.apache.spark.sql.errors.QueryExecutionErrors
import org.apache.spark.sql.types.{DataType, Decimal}
import org.apache.spark.unsafe.types.UTF8String

/** Spark's error factories are `private[sql]`; this bridge lives inside that package tree. */
object VectorErrors {

  /** Spark's `SparkArithmeticException` is `private[spark]`: the overflow a table-insert cast re-raises. */
  def isArithmeticOverflow(t: Throwable): Boolean = t.isInstanceOf[org.apache.spark.SparkArithmeticException]

  def castOverflowInTableInsert(from: DataType, to: DataType, columnName: String): ArithmeticException =
    QueryExecutionErrors.castingCauseOverflowErrorInTableInsert(from, to, columnName)
  def divideByZero(context: QueryContext): ArithmeticException =
    QueryExecutionErrors.divideByZeroError(context)

  /** `%` / `pmod` by zero in ANSI mode: Spark's REMAINDER_BY_ZERO, distinct from DIVIDE_BY_ZERO. */
  def remainderByZero(context: QueryContext): ArithmeticException =
    QueryExecutionErrors.remainderByZeroError(context)

  /** `elt` with an index outside `1..count` in ANSI mode: Spark's INVALID_ARRAY_INDEX. */
  def invalidArrayIndex(index: Int, count: Int, context: QueryContext): ArrayIndexOutOfBoundsException =
    QueryExecutionErrors.invalidArrayIndexError(index, count, context)

  /** `split_part` with part 0: Spark's INVALID_INDEX_OF_ZERO, raised regardless of ANSI mode. */
  def invalidIndexOfZero(context: QueryContext): RuntimeException =
    QueryExecutionErrors.invalidIndexOfZeroError(context)

  /** `make_date` with an invalid civil date in ANSI mode: Spark's DATETIME_FIELD_OUT_OF_BOUNDS family. */
  def dateTimeArgumentOutOfRange(cause: Exception): RuntimeException =
    QueryExecutionErrors.ansiDateTimeArgumentOutOfRange(cause)

  /** An ANSI `cast(string AS boolean)` over a spelling Spark does not accept. */
  def invalidBooleanInput(value: UTF8String, context: QueryContext): RuntimeException =
    QueryExecutionErrors.invalidInputSyntaxForBooleanError(value, context)

  /** Spark's accepted boolean spellings: `t`/`true`/`y`/`yes`/`1` and `f`/`false`/`n`/`no`/`0`, trimmed, any case. */
  def isTrueString(s: UTF8String): Boolean = StringUtils.isTrueString(s)
  def isFalseString(s: UTF8String): Boolean = StringUtils.isFalseString(s)

  /** A decimal result that does not fit the target precision (ANSI arithmetic and casts). */
  def decimalPrecisionOverflow(value: Decimal, precision: Int, scale: Int, context: QueryContext): ArithmeticException =
    QueryExecutionErrors.cannotChangeDecimalPrecisionError(value, precision, scale, context)

  /** `CheckOverflowInSum` on a sum buffer that overflowed earlier (a null sum of a non-empty group). */
  def overflowInSumOfDecimal(context: QueryContext): ArithmeticException =
    QueryExecutionErrors.overflowInSumOfDecimalError(context, "try_sum")

  /** ANSI integer overflow (a checked bigint sum). */
  def arithmeticOverflow(message: String, hint: String, context: QueryContext): ArithmeticException =
    QueryExecutionErrors.arithmeticOverflowError(message, hint, context)

  /** An ANSI cast whose value is outside the target type's range. */
  def castOverflow(value: Any, from: DataType, to: DataType): ArithmeticException =
    QueryExecutionErrors.castingCauseOverflowError(value, from, to)
}
