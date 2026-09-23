package io.sparkvector.spark.expr

import io.sparkvector.kernels.{StringSearchKernels, VectorBuffers}
import io.sparkvector.kernels.StringConcatKernels.Part
import org.apache.spark.QueryContext
import org.apache.spark.sql.types.{DataType, IntegerType, StringType}
import org.apache.spark.sql.vector.VectorErrors

/** The search family over `StringSearchKernels`; string inputs are lanes or literals (`StringConcats.parts`). */
private[expr] object StringSearches {
  def part(e: VectorExpr, ctx: EvalContext): (Part, VectorBuffers) = {
    val (ps, lanes) = StringConcats.parts(Seq(e), ctx)
    (ps(0), lanes.headOption.orNull)
  }
}

/** `locate(needle, hay, pos)` / `instr(hay, needle)` / `position`: Spark's null rules in the kernel. */
final case class LocateExpr(needle: VectorExpr, hay: VectorExpr, pos: VectorExpr) extends VectorExpr {
  override def dataType: DataType = IntegerType
  override def children: Seq[VectorExpr] = Seq(needle, hay, pos)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (np, _) = StringSearches.part(needle, ctx)
    val (hp, _) = StringSearches.part(hay, ctx)
    val (pc, p) = StringSlices.intArg(pos, ctx)
    StringSearchKernels.locate(np, hp, pc, p, ctx.numRows, ctx.arena)
  }
}

final case class ReplaceExpr(hay: VectorExpr, search: VectorExpr, replacement: VectorExpr) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(hay, search, replacement)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (hp, h) = StringSearches.part(hay, ctx)
    val (sp, s) = StringSearches.part(search, ctx)
    val (rp, r) = StringSearches.part(replacement, ctx)
    StringSearchKernels.replace(hp, sp, rp, ctx.numRows, StringSlices.validity(ctx, h, s, r), ctx.arena)
  }
}

/** `translate(hay, from, to)` with literal `from`/`to`, the dictionary built as Spark's `buildDict`. */
final case class TranslateExpr(hay: VectorExpr, from: Array[Int], to: Array[Array[Byte]]) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(hay)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (hp, h) = StringSearches.part(hay, ctx)
    StringSearchKernels.translate(hp, from, to, ctx.numRows, if (h == null) null else h.validity(), ctx.arena)
  }
}

object TranslateExpr {

  /** Spark's `StringTranslate.buildDict`: first mapping per matching code point wins; a missing replacement deletes. */
  def dictionary(matching: String, replace: String): (Array[Int], Array[Array[Byte]]) = {
    val m = matching.codePoints().toArray
    val r = replace.codePoints().toArray
    val seen = scala.collection.mutable.LinkedHashMap.empty[Int, Array[Byte]]
    m.zipWithIndex.foreach { case (cp, k) =>
      if (!seen.contains(cp))
        seen(cp) = if (k < r.length) new String(Character.toChars(r(k))).getBytes("UTF-8") else null
    }
    (seen.keys.toArray, seen.values.toArray)
  }
}

final case class SubstringIndexExpr(hay: VectorExpr, delim: VectorExpr, count: VectorExpr) extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(hay, delim, count)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (hp, h) = StringSearches.part(hay, ctx)
    val (dp, d) = StringSearches.part(delim, ctx)
    val (cc, c) = StringSlices.intArg(count, ctx)
    StringSearchKernels.substringIndex(hp, dp, cc, c, ctx.numRows, StringSlices.validity(ctx, h, d, cc), ctx.arena)
  }
}

/** `split_part(hay, delim, part)`: a zero part raises Spark's INVALID_INDEX_OF_ZERO on an active row. */
final case class SplitPartExpr(hay: VectorExpr, delim: VectorExpr, part: VectorExpr, queryContext: QueryContext)
    extends VectorExpr {
  override def dataType: DataType = StringType
  override def children: Seq[VectorExpr] = Seq(hay, delim, part)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (hp, h) = StringSearches.part(hay, ctx)
    val (dp, d) = StringSearches.part(delim, ctx)
    val (pc, p) = StringSlices.intArg(part, ctx)
    val validity = StringSlices.validity(ctx, h, d, pc)
    if (StringSearchKernels.firstZeroPart(pc, p, validity, ctx.active, ctx.numRows) >= 0)
      throw VectorErrors.invalidIndexOfZero(queryContext)
    StringSearchKernels.splitPart(hp, dp, pc, p, ctx.numRows, validity, ctx.arena)
  }
}

final case class FindInSetExpr(word: VectorExpr, set: VectorExpr) extends VectorExpr {
  override def dataType: DataType = IntegerType
  override def children: Seq[VectorExpr] = Seq(word, set)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val (wp, w) = StringSearches.part(word, ctx)
    val (sp, s) = StringSearches.part(set, ctx)
    StringSearchKernels.findInSet(wp, sp, ctx.numRows, StringSlices.validity(ctx, w, s), ctx.arena)
  }
}
