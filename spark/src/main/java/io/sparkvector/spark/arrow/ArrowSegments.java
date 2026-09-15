package io.sparkvector.spark.arrow;

import java.lang.foreign.MemorySegment;
import org.apache.arrow.memory.ArrowBuf;

/**
 * Zero-copy views of Arrow buffers as {@link MemorySegment}s.
 *
 * <p>{@code MemorySegment.ofAddress(...).reinterpret(...)} is a restricted FFM method; the JVM must
 * run with {@code --enable-native-access=ALL-UNNAMED} (otherwise JDK 25 prints a warning).
 *
 * <p>The returned segment aliases the buffer's current address. It becomes invalid if the buffer
 * is reallocated (vector grows) or released, so callers wrap buffers after the vector has been
 * sized and drop the segment before the vector is closed.
 */
public final class ArrowSegments {

  private ArrowSegments() {}

  /** View over {@code [0, buf.capacity())}. */
  public static MemorySegment of(ArrowBuf buf) {
    return of(buf, buf.capacity());
  }

  public static MemorySegment of(ArrowBuf buf, long byteSize) {
    if (byteSize == 0) {
      return MemorySegment.NULL.reinterpret(0);
    }
    return MemorySegment.ofAddress(buf.memoryAddress()).reinterpret(byteSize);
  }
}
