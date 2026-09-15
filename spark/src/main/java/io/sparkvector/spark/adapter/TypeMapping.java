package io.sparkvector.spark.adapter;

import io.sparkvector.kernels.VecType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
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
    if (dt instanceof StringType) {
      return VecType.UTF8;
    }
    return null;
  }

  public static boolean isSupported(DataType dt) {
    return vecTypeOf(dt) != null;
  }
}
