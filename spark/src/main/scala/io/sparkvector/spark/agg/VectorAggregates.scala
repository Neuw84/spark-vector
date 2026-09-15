package io.sparkvector.spark.agg

import io.sparkvector.kernels.{AggKernels, CompareOp, VecType, VectorBuffers}
import io.sparkvector.spark.expr.{CastExpr, EvalContext, ExpressionCompiler, LiteralExpr, VectorExpr}
import org.apache.spark.sql.catalyst.expressions.{Attribute, EvalMode, Literal}
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.types.{DataType, DoubleType, LongType}

/**
 * Running state of one aggregate function within one task. `bufferValues` yields the partial
 * aggregation buffer in Spark's internal representation (boxed, `null` for SQL null), one entry
 * per `aggBufferAttribute` of the corresponding Catalyst function.
 */
trait AggState {
  def update(ctx: EvalContext): Unit
  def bufferValues: Array[Any]
}

/** Serializable description of a supported aggregate function; states are created per task. */
sealed trait VectorAggFunction extends Serializable {
  def bufferTypes: Seq[DataType]
  def newState(): AggState
}

/** SUM over doubles: buffer `sum` is null until the first non-null input. */
final case class SumDoubleAgg(input: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType)
  override def newState(): AggState = new AggState {
    private var sum = 0.0
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = input.eval(ctx)
      val c = AggKernels.countValid(v)
      if (c > 0) { sum += AggKernels.sumDouble(v); count += c }
    }
    override def bufferValues: Array[Any] = Array(if (count == 0) null else java.lang.Double.valueOf(sum))
  }
}

/** SUM over ints or longs into a long buffer (legacy wrap-around semantics for longs). */
final case class SumLongAgg(input: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType)
  override def newState(): AggState = new AggState {
    private var sum = 0L
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = input.eval(ctx)
      val c = AggKernels.countValid(v)
      if (c > 0) {
        sum += (if (v.`type`() == VecType.INT32) AggKernels.sumInt(v) else AggKernels.sumLong(v))
        count += c
      }
    }
    override def bufferValues: Array[Any] = Array(if (count == 0) null else java.lang.Long.valueOf(sum))
  }
}

/** COUNT(*) when `input` is None, otherwise COUNT of non-null values of the expression. */
final case class CountAgg(input: Option[VectorExpr]) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType)
  override def newState(): AggState = new AggState {
    private var count = 0L
    override def update(ctx: EvalContext): Unit = input match {
      case None => count += ctx.numRows
      case Some(e) => count += AggKernels.countValid(e.eval(ctx))
    }
    override def bufferValues: Array[Any] = Array(java.lang.Long.valueOf(count))
  }
}

/** MIN / MAX over Int (incl. Date), Long (incl. Timestamp) or Double, with Spark's NaN ordering. */
final case class MinMaxAgg(input: VectorExpr, isMin: Boolean, dataType: DataType) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(dataType)
  override def newState(): AggState = new AggState {
    private var any = false
    private var bestLong = 0L
    private var bestDouble = 0.0
    override def update(ctx: EvalContext): Unit = {
      val v = input.eval(ctx)
      if (AggKernels.countValid(v) > 0) {
        v.`type`() match {
          case VecType.FLOAT64 =>
            val m = if (isMin) AggKernels.minDouble(v) else AggKernels.maxDouble(v)
            if (!any) bestDouble = m
            else {
              val cmp = CompareOp.nanSafeCompare(m, bestDouble)
              if ((isMin && cmp < 0) || (!isMin && cmp > 0)) bestDouble = m
            }
          case VecType.INT32 =>
            val m: Long = if (isMin) AggKernels.minInt(v) else AggKernels.maxInt(v)
            if (!any) bestLong = m else bestLong = if (isMin) math.min(bestLong, m) else math.max(bestLong, m)
          case VecType.INT64 =>
            val m = if (isMin) AggKernels.minLong(v) else AggKernels.maxLong(v)
            if (!any) bestLong = m else bestLong = if (isMin) math.min(bestLong, m) else math.max(bestLong, m)
          case other => throw new IllegalStateException(s"min/max on $other")
        }
        any = true
      }
    }
    override def bufferValues: Array[Any] =
      if (!any) Array(null)
      else input.vecType match {
        case VecType.FLOAT64 => Array(java.lang.Double.valueOf(bestDouble))
        case VecType.INT32 => Array(java.lang.Integer.valueOf(bestLong.toInt))
        case _ => Array(java.lang.Long.valueOf(bestLong))
      }
  }
}

/** AVG over a double-typed input (integers are cast first): buffer is (sum, count). */
final case class AverageAgg(input: VectorExpr) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(DoubleType, LongType)
  override def newState(): AggState = new AggState {
    private var sum = 0.0
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val v = input.eval(ctx)
      val c = AggKernels.countValid(v)
      if (c > 0) { sum += AggKernels.sumDouble(v); count += c }
    }
    override def bufferValues: Array[Any] = Array(java.lang.Double.valueOf(sum), java.lang.Long.valueOf(count))
  }
}

object VectorAggregates {

  private val numeric: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.FLOAT64)

  /** Compiles a Partial-mode aggregate expression, or explains why it cannot be vectorized. */
  def compile(agg: AggregateExpression, input: Seq[Attribute]): Either[String, VectorAggFunction] = {
    if (agg.mode != Partial) Left(s"aggregate mode ${agg.mode} not supported (only Partial)")
    else if (agg.isDistinct) Left("distinct aggregates not supported")
    else if (agg.filter.isDefined) Left("aggregates with FILTER not supported")
    else compileFunction(agg.aggregateFunction, input)
  }

  private def compileFunction(f: AggregateFunction, input: Seq[Attribute]): Either[String, VectorAggFunction] = f match {
    case s: Sum =>
      numericChild(s.child, input).flatMap { child =>
        (s.dataType, child.vecType) match {
          case (DoubleType, VecType.FLOAT64) => Right(SumDoubleAgg(child))
          case (LongType, VecType.INT32) => Right(SumLongAgg(child))
          case (LongType, VecType.INT64) if s.evalContext.evalMode == EvalMode.ANSI =>
            Left("ANSI sum of bigint (overflow check) not supported")
          case (LongType, VecType.INT64) => Right(SumLongAgg(child))
          case (dt, _) => Left(s"sum over ${s.child.dataType.simpleString} producing ${dt.simpleString} not supported")
        }
      }

    case c: Count =>
      c.children match {
        case Seq(Literal(v, _)) if v != null => Right(CountAgg(None))
        case Seq(child) =>
          ExpressionCompiler.compile(child, input).flatMap {
            case _: LiteralExpr => Right(CountAgg(None))
            case e => Right(CountAgg(Some(e)))
          }
        case _ => Left("count with several arguments not supported")
      }

    case m: Min => numericChild(m.child, input).map(child => MinMaxAgg(child, isMin = true, m.dataType))
    case m: Max => numericChild(m.child, input).map(child => MinMaxAgg(child, isMin = false, m.dataType))

    case a: Average =>
      if (a.dataType != DoubleType) Left(s"avg producing ${a.dataType.simpleString} not supported")
      else numericChild(a.child, input).map { child =>
        AverageAgg(if (child.vecType == VecType.FLOAT64) child else CastExpr(child, DoubleType))
      }

    case other => Left(s"unsupported aggregate function ${other.getClass.getSimpleName}: ${other.sql}")
  }

  private def numericChild(child: org.apache.spark.sql.catalyst.expressions.Expression, input: Seq[Attribute]): Either[String, VectorExpr] =
    ExpressionCompiler.compile(child, input).flatMap {
      case _: LiteralExpr => Left("aggregate over a literal")
      case e if !numeric.contains(e.vecType) => Left(s"aggregate over ${child.dataType.simpleString} not supported")
      case e => Right(e)
    }

  /** Bridges a kernel-level buffers object to a Spark-side accessor, for tests. */
  def countValid(v: VectorBuffers): Long = AggKernels.countValid(v)
}
