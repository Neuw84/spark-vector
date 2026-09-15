package io.sparkvector.spark.expr

import io.sparkvector.kernels.{CompareOp, VecType}
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeReference, BoundReference, EqualTo, Expression, GreaterThan, GreaterThanOrEqual, IsNotNull, IsNull, LessThan, LessThanOrEqual, Literal, Not, Or}
import org.apache.spark.sql.types.{BooleanType, DataType, DateType, DoubleType, IntegerType, LongType, TimestampType}

/**
 * Translates Catalyst expressions into [[VectorExpr]] trees. Returns a human-readable reason on
 * failure so the planner rule can tag the Spark operator and fall back.
 */
object ExpressionCompiler {

  type Result = Either[String, VectorExpr]

  private val comparableTypes: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.FLOAT64)

  /** Types whose literals can be used as compare operands. */
  private def isLiteralType(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | DoubleType | DateType | TimestampType => true
    case _ => false
  }

  def compile(expr: Expression, input: Seq[Attribute]): Result = expr match {
    case a: AttributeReference =>
      val ordinal = input.indexWhere(_.exprId == a.exprId)
      if (ordinal < 0) Left(s"unbound attribute ${a.name}")
      else if (!TypeMapping.isSupported(a.dataType)) Left(s"unsupported type ${a.dataType.simpleString} for ${a.name}")
      else Right(ColumnRef(ordinal, a.dataType))

    case b: BoundReference =>
      if (!TypeMapping.isSupported(b.dataType)) Left(s"unsupported type ${b.dataType.simpleString}")
      else Right(ColumnRef(b.ordinal, b.dataType))

    case Literal(null, _) => Left("null literal")
    case Literal(v, dt) if isLiteralType(dt) => Right(LiteralExpr(v, dt))
    case Literal(_, dt) => Left(s"unsupported literal type ${dt.simpleString}")

    case Alias(child, _) => compile(child, input)

    case EqualTo(l, r) => comparison(CompareOp.EQ, l, r, input)
    case LessThan(l, r) => comparison(CompareOp.LT, l, r, input)
    case LessThanOrEqual(l, r) => comparison(CompareOp.LE, l, r, input)
    case GreaterThan(l, r) => comparison(CompareOp.GT, l, r, input)
    case GreaterThanOrEqual(l, r) => comparison(CompareOp.GE, l, r, input)
    case Not(EqualTo(l, r)) => comparison(CompareOp.NE, l, r, input)

    case And(l, r) => binaryBoolean(l, r, input)(AndExpr.apply)
    case Or(l, r) => binaryBoolean(l, r, input)(OrExpr.apply)
    case Not(child) => booleanChild(child, input).map(NotExpr.apply)

    case IsNull(child) => nullTestChild(child, input).map(IsNullExpr.apply)
    case IsNotNull(child) => nullTestChild(child, input).map(IsNotNullExpr.apply)

    case other => Left(s"unsupported expression ${other.getClass.getSimpleName}: ${other.sql}")
  }

  /** A filter condition must produce a non-literal boolean column. */
  def compilePredicate(expr: Expression, input: Seq[Attribute]): Result =
    compile(expr, input).flatMap {
      case _: LiteralExpr => Left("literal predicate")
      case e if e.dataType != BooleanType => Left(s"predicate is not boolean: ${e.dataType.simpleString}")
      case e => Right(e)
    }

  private def comparison(op: CompareOp, l: Expression, r: Expression, input: Seq[Attribute]): Result =
    for {
      le <- compile(l, input)
      re <- compile(r, input)
      _ <- check(le, re, l, r)
    } yield CompareExpr(op, le, re)

  private def check(le: VectorExpr, re: VectorExpr, l: Expression, r: Expression): Either[String, Unit] = {
    if (le.isInstanceOf[LiteralExpr] && re.isInstanceOf[LiteralExpr]) Left("comparison of two literals")
    else if (l.dataType != r.dataType) Left(s"comparison operands differ: ${l.dataType.simpleString} vs ${r.dataType.simpleString}")
    else if (!comparableTypes.contains(le.vecType)) Left(s"comparison not supported for ${l.dataType.simpleString}")
    else Right(())
  }

  private def binaryBoolean(l: Expression, r: Expression, input: Seq[Attribute])(
      make: (VectorExpr, VectorExpr) => VectorExpr): Result =
    for {
      le <- booleanChild(l, input)
      re <- booleanChild(r, input)
    } yield make(le, re)

  private def booleanChild(e: Expression, input: Seq[Attribute]): Result =
    compile(e, input).flatMap {
      case _: LiteralExpr => Left("boolean literal operand")
      case v if v.dataType != BooleanType => Left(s"expected boolean, got ${v.dataType.simpleString}")
      case v => Right(v)
    }

  private def nullTestChild(e: Expression, input: Seq[Attribute]): Result =
    compile(e, input).flatMap {
      case _: LiteralExpr => Left("null test on literal")
      case v => Right(v)
    }
}
