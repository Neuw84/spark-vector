package org.apache.spark.sql.vector

import org.apache.spark.sql.types.{Decimal, NullType}
import org.apache.spark.sql.vectorized.{ColumnarArray, ColumnarMap, ColumnVector}
import org.apache.spark.unsafe.types.UTF8String

/**
 * A joined column a join condition does not read: never gathered, never adapted, never touched.
 * A join evaluates its residual condition over a batch laid out as the full joined row so the
 * compiled expression's ordinals hold, but gathers only the columns the condition references
 * (#310 for the merge join, #332 for the hash joins); the others are this object. Any access is a
 * bug in the ordinal bookkeeping and throws.
 */
private[vector] object PlaceholderColumn extends ColumnVector(NullType) {
  private def no = throw new UnsupportedOperationException("a joined column the condition does not read")
  override def close(): Unit = ()
  override def hasNull: Boolean = no
  override def numNulls: Int = no
  override def isNullAt(rowId: Int): Boolean = no
  override def getBoolean(rowId: Int): Boolean = no
  override def getByte(rowId: Int): Byte = no
  override def getShort(rowId: Int): Short = no
  override def getInt(rowId: Int): Int = no
  override def getLong(rowId: Int): Long = no
  override def getFloat(rowId: Int): Float = no
  override def getDouble(rowId: Int): Double = no
  override def getArray(rowId: Int): ColumnarArray = no
  override def getMap(ordinal: Int): ColumnarMap = no
  override def getDecimal(rowId: Int, precision: Int, scale: Int): Decimal = no
  override def getUTF8String(rowId: Int): UTF8String = no
  override def getBinary(rowId: Int): Array[Byte] = no
  override def getChild(ordinal: Int): ColumnVector = no
}
