package io.sparkvector.kernels;

/**
 * Physical vector types understood by the kernels. Logical types are mapped onto these by the
 * Spark layer (e.g. DateType is INT32 days, TimestampType is INT64 micros).
 */
public enum VecType {
  /** Arrow boolean: values are a bitmap, one bit per element. */
  BOOL(0),
  INT32(4),
  INT64(8),
  FLOAT64(8),
  /** Arrow Utf8: int32 offsets (length + 1 entries) plus a byte buffer. */
  UTF8(-1);

  private final int byteWidth;

  VecType(int byteWidth) {
    this.byteWidth = byteWidth;
  }

  /** Bytes per element for fixed-width types; 0 for BOOL (bit-packed); -1 for variable width. */
  public int byteWidth() {
    return byteWidth;
  }

  public boolean isFixedWidth() {
    return byteWidth > 0;
  }
}
