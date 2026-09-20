package io.sparkvector.shuffle

import com.github.luben.zstd.Zstd
import org.apache.arrow.compression.{CommonsCompressionFactory, ZstdCompressionCodec}
import org.apache.arrow.memory.{ArrowBuf, BufferAllocator}
import org.apache.arrow.vector.compression.{CompressionCodec, CompressionUtil}

/**
 * The compression codecs of the shuffle's IPC streams (#340).
 *
 * arrow-compression 18.3.0's `ZstdCompressionCodec.doCompress` offsets the destination by the 8-byte
 * uncompressed-length prefix but hands zstd the whole buffer's size as the destination capacity, so
 * zstd may write up to 8 bytes past the end of the compressed buffer -- into whatever the pool placed
 * next to it. Under a busy writer that was a neighbouring column's first bytes: a wrong value, once
 * per few hundred thousand rows, with every reference count intact (apache/arrow-java GH-1116, fixed
 * for 20.0.0, unreleased). This codec passes the capacity zstd actually has; LZ4 is Arrow's own.
 */
object ShuffleCompression {
  /** arrow-java's zstd codec with the destination capacity it should have passed. */
  final class SafeZstdCodec(level: Int) extends ZstdCompressionCodec(level) {
    override protected def doCompress(allocator: BufferAllocator, uncompressed: ArrowBuf): ArrowBuf = {
      val srcLength = uncompressed.writerIndex()
      val maxSize = Zstd.compressBound(srcLength)
      val compressed = allocator.buffer(CompressionUtil.SIZE_OF_UNCOMPRESSED_LENGTH + maxSize)
      val written = Zstd.compressUnsafe(
        compressed.memoryAddress() + CompressionUtil.SIZE_OF_UNCOMPRESSED_LENGTH, maxSize,
        uncompressed.memoryAddress(), srcLength, level)
      if (Zstd.isError(written)) {
        compressed.close()
        throw new RuntimeException("Error compressing: " + Zstd.getErrorName(written))
      }
      compressed.writerIndex(CompressionUtil.SIZE_OF_UNCOMPRESSED_LENGTH + written)
      compressed
    }
  }

  private val DefaultZstdLevel = 3

  /** The factory every stream writer and reader of the shuffle uses. */
  val Factory: CompressionCodec.Factory = new CompressionCodec.Factory {
    override def createCodec(codecType: CompressionUtil.CodecType): CompressionCodec = codecType match {
      case CompressionUtil.CodecType.ZSTD => new SafeZstdCodec(DefaultZstdLevel)
      case other => CommonsCompressionFactory.INSTANCE.createCodec(other)
    }
    override def createCodec(codecType: CompressionUtil.CodecType, compressionLevel: Int): CompressionCodec = codecType match {
      case CompressionUtil.CodecType.ZSTD => new SafeZstdCodec(compressionLevel)
      case other => CommonsCompressionFactory.INSTANCE.createCodec(other, compressionLevel)
    }
  }
}
