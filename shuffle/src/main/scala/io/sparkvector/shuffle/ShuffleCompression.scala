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

  /**
   * The partition stream's compression (#411): the writer compresses a partition's IPC messages as
   * one stream and the reader decompresses the block's bytes before its message reader -- one
   * compressor call and one frame per block instead of one per buffer of every record batch, and
   * the messages' repeated metadata compresses away. `None` when the configured codec is none.
   */
  def compressing(out: java.io.OutputStream, codec: Option[CompressionUtil.CodecType]): java.io.OutputStream = codec match {
    case Some(CompressionUtil.CodecType.ZSTD) => new com.github.luben.zstd.ZstdOutputStreamNoFinalizer(out, DefaultZstdLevel)
    case Some(CompressionUtil.CodecType.LZ4_FRAME) => new net.jpountz.lz4.LZ4BlockOutputStream(out)
    case Some(other) => throw new IllegalArgumentException(s"no stream compression for $other")
    case None => null
  }

  /** The reading side of [[compressing]]: several frames back to back (a range of partitions, several map outputs) read as one stream. */
  def decompressing(in: java.io.InputStream, codec: Option[CompressionUtil.CodecType]): java.io.InputStream = codec match {
    case Some(CompressionUtil.CodecType.ZSTD) => new com.github.luben.zstd.ZstdInputStreamNoFinalizer(in)
    case Some(CompressionUtil.CodecType.LZ4_FRAME) =>
      new net.jpountz.lz4.LZ4BlockInputStream(in, net.jpountz.lz4.LZ4Factory.fastestInstance().fastDecompressor(),
        net.jpountz.xxhash.XXHashFactory.fastestInstance().newStreamingHash32(0x9747b28c).asChecksum(), false)
    case Some(other) => throw new IllegalArgumentException(s"no stream compression for $other")
    case None => in
  }

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
