package io.sparkvector.spark.expr

import java.lang.foreign.Arena

import io.sparkvector.kernels._
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.types.{BooleanType, DataType}

/**
 * Per-batch evaluation context. Input columns are adapted lazily so that a predicate touching two
 * columns out of twenty never pays for adapting the other eighteen unless the batch survives.
 */
final class EvalContext(val arena: Arena, val numRows: Int, adaptColumn: Int => VectorBuffers) {
  private val adapted = new java.util.HashMap[Int, VectorBuffers]()

  def input(ordinal: Int): VectorBuffers = {
    var v = adapted.get(ordinal)
    if (v == null) {
      v = adaptColumn(ordinal)
      adapted.put(ordinal, v)
    }
    v
  }

  def bitmap(): java.lang.foreign.MemorySegment = ArrowLayout.allocateBitmap(arena, numRows)
}

/**
 * Compiled, serializable expression tree evaluated column-at-a-time with the kernels. Built by
 * [[ExpressionCompiler]] on the driver and shipped inside the physical operator.
 */
sealed trait VectorExpr extends Serializable {
  def dataType: DataType
  final def vecType: VecType = TypeMapping.vecTypeOf(dataType)
  def eval(ctx: EvalContext): VectorBuffers
  def children: Seq[VectorExpr]
}

/** Reference to an input column by ordinal in the child's output. */
final case class ColumnRef(ordinal: Int, dataType: DataType) extends VectorExpr {
  override def eval(ctx: EvalContext): VectorBuffers = ctx.input(ordinal)
  override def children: Seq[VectorExpr] = Nil
}

/**
 * A non-null literal in Spark's internal representation (Int for dates, Long for timestamps).
 * Literals are only valid as operands of a [[CompareExpr]] (and later arithmetic); they cannot be
 * materialised as a column on their own, which is why [[eval]] throws.
 */
final case class LiteralExpr(value: Any, dataType: DataType) extends VectorExpr {
  def number: Number = value.asInstanceOf[Number]
  override def eval(ctx: EvalContext): VectorBuffers =
    throw new UnsupportedOperationException("literal cannot be evaluated as a column")
  override def children: Seq[VectorExpr] = Nil
}

/** Comparison of two same-typed numeric operands, at most one of which is a literal. */
final case class CompareExpr(op: CompareOp, left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val bits = ctx.bitmap()
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        CompareKernels.compareScalar(a, lit.number, op, bits)
        // Result is null exactly where the column is null: share its validity.
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, a.validity(), bits)
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        CompareKernels.compareScalar(b, lit.number, op.flip(), bits)
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, b.validity(), bits)
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        CompareKernels.compare(a, b, op, bits)
        val validity =
          if (a.validity() == null && b.validity() == null) null
          else {
            val v = ctx.bitmap()
            BitmapKernels.combineValidity(a.validity(), b.validity(), v, n)
            v
          }
        SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
    }
  }
}

/** Spark's three-valued AND. */
final case class AndExpr(left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = left.eval(ctx)
    val b = right.eval(ctx)
    val bits = ctx.bitmap()
    if (a.validity() == null && b.validity() == null) {
      BitmapKernels.and(a.data(), b.data(), bits, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
    } else {
      val validity = ctx.bitmap()
      BitmapKernels.kleeneAnd(a.data(), a.validity(), b.data(), b.validity(), bits, validity, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
    }
  }
}

/** Spark's three-valued OR. */
final case class OrExpr(left: VectorExpr, right: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(left, right)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val a = left.eval(ctx)
    val b = right.eval(ctx)
    val bits = ctx.bitmap()
    if (a.validity() == null && b.validity() == null) {
      BitmapKernels.or(a.data(), b.data(), bits, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, null, bits)
    } else {
      val validity = ctx.bitmap()
      BitmapKernels.kleeneOr(a.data(), a.validity(), b.data(), b.validity(), bits, validity, n)
      SegmentVectorBuffers.fixedWidth(VecType.BOOL, n, validity, bits)
    }
  }
}

final case class NotExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    BitmapKernels.not(a.data(), bits, ctx.numRows)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, a.validity(), bits)
  }
}

final case class IsNullExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    BitmapKernels.isNull(a.validity(), bits, ctx.numRows)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, null, bits)
  }
}

final case class IsNotNullExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = BooleanType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val bits = ctx.bitmap()
    BitmapKernels.isNotNull(a.validity(), bits, ctx.numRows)
    SegmentVectorBuffers.fixedWidth(VecType.BOOL, ctx.numRows, null, bits)
  }
}

object VectorExpr {

  /** All column ordinals an expression reads. */
  def referencedColumns(e: VectorExpr): Set[Int] = e match {
    case ColumnRef(o, _) => Set(o)
    case other => other.children.flatMap(referencedColumns).toSet
  }

  /**
   * Selection bitmap for a boolean result: true and not null. Returns the number of selected rows.
   */
  def selection(pred: VectorBuffers, ctx: EvalContext): (java.lang.foreign.MemorySegment, Int) = {
    val sel = ctx.bitmap()
    BitmapKernels.selection(pred.data(), pred.validity(), sel, ctx.numRows)
    (sel, Bitmap.popcount(sel, ctx.numRows))
  }
}

/**
 * Binary arithmetic on same-typed operands (at most one literal). Division is only defined for
 * doubles; a zero divisor yields null in legacy mode and raises DIVIDE_BY_ZERO in ANSI mode,
 * exactly like Spark's `Divide`.
 */
final case class ArithExpr(
    op: ArithOp,
    left: VectorExpr,
    right: VectorExpr,
    dataType: DataType,
    ansiDivideByZero: Boolean,
    queryContext: org.apache.spark.QueryContext)
    extends VectorExpr {

  override def children: Seq[VectorExpr] = Seq(left, right)

  override def eval(ctx: EvalContext): VectorBuffers = {
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, vecType, n)
    var validity: java.lang.foreign.MemorySegment = null
    var divisorZero: java.lang.foreign.MemorySegment = null
    (left, right) match {
      case (l, lit: LiteralExpr) =>
        val a = l.eval(ctx)
        ArithKernels.arithScalar(op, a, lit.number, data)
        validity = a.validity()
        if (op == ArithOp.DIV && lit.number.doubleValue() == 0.0) {
          divisorZero = ctx.bitmap()
          Bitmap.fill(divisorZero, n, true)
        }
      case (lit: LiteralExpr, r) =>
        val b = r.eval(ctx)
        ArithKernels.scalarArith(op, lit.number, b, data)
        validity = b.validity()
        if (op == ArithOp.DIV) divisorZero = zeroMask(b, ctx)
      case (l, r) =>
        val a = l.eval(ctx)
        val b = r.eval(ctx)
        ArithKernels.arith(op, a, b, data)
        if (a.validity() != null || b.validity() != null) {
          validity = ctx.bitmap()
          BitmapKernels.combineValidity(a.validity(), b.validity(), validity, n)
        }
        if (op == ArithOp.DIV) divisorZero = zeroMask(b, ctx)
    }
    if (divisorZero != null) {
      // Lanes that are otherwise valid but divide by zero.
      val affected = ctx.bitmap()
      if (validity == null) BitmapKernels.copy(divisorZero, affected, n)
      else BitmapKernels.and(divisorZero, validity, affected, n)
      val count = Bitmap.popcount(affected, n)
      if (count > 0) {
        if (ansiDivideByZero) {
          throw org.apache.spark.sql.vector.VectorErrors.divideByZero(queryContext)
        }
        val newValidity = ctx.bitmap()
        if (validity == null) BitmapKernels.not(divisorZero, newValidity, n)
        else BitmapKernels.andNot(validity, divisorZero, newValidity, n)
        validity = newValidity
      }
    }
    SegmentVectorBuffers.fixedWidth(vecType, n, validity, data)
  }

  private def zeroMask(b: VectorBuffers, ctx: EvalContext): java.lang.foreign.MemorySegment = {
    val zero = ctx.bitmap()
    CompareKernels.compareScalar(b, java.lang.Double.valueOf(0.0), CompareOp.EQ, zero)
    zero
  }
}

/** Widening numeric cast; validity is shared with the child. */
final case class CastExpr(child: VectorExpr, dataType: DataType) extends VectorExpr {
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, vecType, ctx.numRows)
    CastKernels.cast(a, vecType, data)
    SegmentVectorBuffers.fixedWidth(vecType, ctx.numRows, a.validity(), data)
  }
}

final case class NegateExpr(child: VectorExpr) extends VectorExpr {
  override def dataType: DataType = child.dataType
  override def children: Seq[VectorExpr] = Seq(child)
  override def eval(ctx: EvalContext): VectorBuffers = {
    val a = child.eval(ctx)
    val data = ArrowLayout.allocateData(ctx.arena, vecType, ctx.numRows)
    ArithKernels.negate(a, data)
    SegmentVectorBuffers.fixedWidth(vecType, ctx.numRows, a.validity(), data)
  }
}
