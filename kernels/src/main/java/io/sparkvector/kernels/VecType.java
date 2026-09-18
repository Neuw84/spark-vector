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
  UTF8(-1),
  /**
   * Arrow Decimal128: 16 bytes per value, a two's complement 128-bit integer stored little-endian
   * as two {@code long} limbs (low at offset 0, high at offset 8; see {@link Decimal128}). Holds
   * the unscaled value of a decimal wider than 18 digits. The lane is deliberately scalar: no
   * {@code Species} covers it and every kernel walks it limb by limb (#28, option 1).
   */
  DECIMAL128(16);

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
