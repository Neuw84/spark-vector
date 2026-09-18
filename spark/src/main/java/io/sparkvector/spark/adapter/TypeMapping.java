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
    if (dt instanceof DecimalType d) {
      // Unscaled value in long lanes, like Spark's own WritableColumnVector for Decimal(p <= 18);
      // wider decimals are two-limb DECIMAL128 lanes (#28, option 1).
      return d.precision() <= MAX_DECIMAL_PRECISION ? VecType.INT64 : VecType.DECIMAL128;
    }
    return null;
  }

  /**
   * Whether values of this type have a physical lane: they can be adapted from a Spark vector,
   * compacted, gathered, appended and emitted. Wider than {@link #isSupported}, which also asks
   * that the kernels compute on the lane -- the DECIMAL128 lane can be carried (#257) but no
   * expression or key kernel reads it yet (#258, #259), so operators consult this only where they
   * merely move a column.
   */
  public static boolean hasLane(DataType dt) {
    return vecTypeOf(dt) != null;
  }

  /** Decimals wider than this have no 64-bit representation and fall back. */
  public static final int MAX_DECIMAL_PRECISION = 18;

  public static boolean isDecimal(DataType dt) {
    return dt instanceof DecimalType;
  }

  /** Whether the kernels compute on this type (a lane the expression and key kernels read). */
  public static boolean isSupported(DataType dt) {
    VecType t = vecTypeOf(dt);
    return t != null && t != VecType.DECIMAL128;
  }

  public static boolean isWideDecimal(DataType dt) {
    return dt instanceof DecimalType d && d.precision() > MAX_DECIMAL_PRECISION;
  }
}
