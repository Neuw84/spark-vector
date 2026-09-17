package io.sparkvector.spark.adapter;

import io.sparkvector.kernels.VecType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.TimestampType;

/** Spark logical type to kernel physical type. */
public final class TypeMapping {

  private TypeMapping() {}

  /** Physical type for a supported Spark type, or {@code null} if unsupported. */
  public static VecType vecTypeOf(DataType dt) {
    if (dt instanceof IntegerType || dt instanceof DateType) {
      return VecType.INT32;
    }
    if (dt instanceof LongType || dt instanceof TimestampType) {
      return VecType.INT64;
    }
    if (dt instanceof DoubleType) {
      return VecType.FLOAT64;
    }
    if (dt instanceof BooleanType) {
      return VecType.BOOL;
    }
    if (dt instanceof StringType st) {
      // Kernels compare, group and sort bytes: only UTF8_BINARY strings are lanes. A collated
      // string (STRING COLLATE UTF8_LCASE, ...) has no lane and the operator falls back -- Spark's
      // own RowToColumnarExec cannot convert one either.
      return st.isUTF8BinaryCollation() ? VecType.UTF8 : null;
    }
    if (dt instanceof DecimalType d && d.precision() <= MAX_DECIMAL_PRECISION) {
      // Unscaled value in long lanes, like Spark's own WritableColumnVector for Decimal(p <= 18).
      return VecType.INT64;
    }
    return null;
  }

  /** Decimals wider than this have no 64-bit representation and fall back. */
  public static final int MAX_DECIMAL_PRECISION = 18;

  public static boolean isDecimal(DataType dt) {
    return dt instanceof DecimalType;
  }

  public static boolean isSupported(DataType dt) {
    return vecTypeOf(dt) != null;
  }
}
