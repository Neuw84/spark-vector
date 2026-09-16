package io.sparkvector.spark.expr

import io.sparkvector.kernels.{ArithOp, CastKernels, CompareOp, VecType}
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.catalyst.expressions.{Add, Alias, And, Attribute, AttributeReference, BoundReference, Cast, Divide, EqualTo, EvalMode, Expression, GreaterThan, GreaterThanOrEqual, IsNotNull, IsNull, KnownFloatingPointNormalized, LessThan, LessThanOrEqual, Literal, Multiply, Not, Or, Subtract, UnaryMinus}
import org.apache.spark.sql.catalyst.optimizer.NormalizeNaNAndZero
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

    // Spark (and Comet's plan normalisation) wrap doubles used in comparisons and grouping keys in
    // NormalizeNaNAndZero so that all NaNs and both zeros compare equal. Our compare kernels already
    // implement that ordering and double grouping keys are rejected, so both are identities here.
    case KnownFloatingPointNormalized(child) => compile(child, input)
    case NormalizeNaNAndZero(child) => compile(child, input)

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

    case e: Add => arithmetic(ArithOp.ADD, e.left, e.right, e.evalMode, e, input)
    case e: Subtract => arithmetic(ArithOp.SUB, e.left, e.right, e.evalMode, e, input)
    case e: Multiply => arithmetic(ArithOp.MUL, e.left, e.right, e.evalMode, e, input)
    case e: Divide => arithmetic(ArithOp.DIV, e.left, e.right, e.evalMode, e, input)

    case UnaryMinus(child, failOnError) =>
      compile(child, input).flatMap {
        case _: LiteralExpr => Left("negation of a literal")
        case c if !arithmeticTypes.contains(c.vecType) => Left(s"negation not supported for ${c.dataType.simpleString}")
        case c if failOnError && c.vecType != VecType.FLOAT64 => Left("ANSI integer negation (overflow check) not supported")
        case c => Right(NegateExpr(c))
      }

    // Casts to the operand's own type (Spark's Average emits `sum.cast(double)` on a double sum).
    case c: Cast if c.child.dataType == c.dataType => compile(c.child, input)

    case c: Cast =>
      val child = c.child
      val dt = c.dataType
      compile(child, input).flatMap {
        case _: LiteralExpr => Left("cast of a literal")
        case c if !TypeMapping.isSupported(dt) => Left(s"unsupported cast target ${dt.simpleString}")
        case c if !CastKernels.isSupported(c.vecType, TypeMapping.vecTypeOf(dt)) =>
          Left(s"unsupported cast ${child.dataType.simpleString} -> ${dt.simpleString}")
        case c => Right(CastExpr(c, dt))
      }

    case other => Left(s"unsupported expression ${other.getClass.getSimpleName}: ${other.sql}")
  }

  private val arithmeticTypes: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.FLOAT64)

  /**
   * Spark 4 defaults to ANSI mode. Double arithmetic is identical in both modes except that
   * division by zero raises instead of yielding null, which the kernel wrapper handles. Integer
   * arithmetic in ANSI mode needs overflow checks the kernels do not implement, so it falls back.
   */
  private def arithmetic(
      op: ArithOp,
      l: Expression,
      r: Expression,
      mode: EvalMode.Value,
      e: Expression,
      input: Seq[Attribute]): Result =
    for {
      le <- compile(l, input)
      re <- compile(r, input)
      _ <- checkArithmetic(op, le, re, l, r, mode)
    } yield ArithExpr(op, le, re, e.dataType, mode == EvalMode.ANSI, e.origin.context)

  private def checkArithmetic(
      op: ArithOp,
      le: VectorExpr,
      re: VectorExpr,
      l: Expression,
      r: Expression,
      mode: EvalMode.Value): Either[String, Unit] = {
    if (le.isInstanceOf[LiteralExpr] && re.isInstanceOf[LiteralExpr]) Left("arithmetic on two literals")
    else if (l.dataType != r.dataType) Left(s"arithmetic operands differ: ${l.dataType.simpleString} vs ${r.dataType.simpleString}")
    else if (!arithmeticTypes.contains(le.vecType)) Left(s"arithmetic not supported for ${l.dataType.simpleString}")
    else if (op == ArithOp.DIV && le.vecType != VecType.FLOAT64) Left(s"division not supported for ${l.dataType.simpleString}")
    else if (mode == EvalMode.TRY) Left("try_* arithmetic not supported")
    else if (mode == EvalMode.ANSI && le.vecType != VecType.FLOAT64) Left("ANSI integer arithmetic (overflow checks) not supported")
    else Right(())
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
