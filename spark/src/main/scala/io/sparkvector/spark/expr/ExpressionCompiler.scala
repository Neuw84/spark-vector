package io.sparkvector.spark.expr

import io.sparkvector.kernels.{ArithOp, CastKernels, CompareOp, VecType}
import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.sql.catalyst.expressions.{Add, Alias, And, Attribute, AttributeReference, BoundReference, CaseWhen, Cast, Coalesce, Divide, EqualTo, EvalMode, Expression, GreaterThan, GreaterThanOrEqual, If, In, IsNotNull, IsNull, KnownFloatingPointNormalized, LessThan, LessThanOrEqual, Literal, MakeDecimal, Multiply, Not, Or, Subtract, UnaryMinus, UnscaledValue}
import org.apache.spark.sql.catalyst.optimizer.NormalizeNaNAndZero
import org.apache.spark.sql.types.{BooleanType, DataType, DateType, DecimalType, DoubleType, IntegerType, LongType, StringType, TimestampType}

/**
 * Translates Catalyst expressions into [[VectorExpr]] trees. Returns a human-readable reason on
 * failure so the planner rule can tag the Spark operator and fall back.
 */
object ExpressionCompiler {

  type Result = Either[String, VectorExpr]

  private val comparableTypes: Set[VecType] = Set(VecType.INT32, VecType.INT64, VecType.FLOAT64, VecType.UTF8)

  /** Types whose literals can be used as compare / IN operands. */
  private def isLiteralType(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | DoubleType | DateType | TimestampType | StringType => true
    case d: DecimalType => TypeMapping.isSupported(d)
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

    case In(value, list) => inList(value, list, input)

    case And(l, r) => binaryBoolean(l, r, input)(AndExpr.apply)
    case Or(l, r) => binaryBoolean(l, r, input)(OrExpr.apply)
    case Not(child) => booleanChild(child, input).map(NotExpr.apply)

    case IsNull(child) => nullTestChild(child, input).map(IsNullExpr.apply)
    case IsNotNull(child) => nullTestChild(child, input).map(IsNotNullExpr.apply)

    // Conditionals blend same-typed branches by per-branch masks (CaseWhenExpr). IF is the
    // one-branch case; COALESCE picks the first non-null operand, i.e. IS NOT NULL conditions over
    // the operands themselves (an operand is evaluated once for its test and once for its value).
    case CaseWhen(branches, elseValue) => conditional(branches, elseValue, expr, input)
    case If(predicate, trueValue, falseValue) => conditional(Seq((predicate, trueValue)), Some(falseValue), expr, input)
    case Coalesce(children) if children.length >= 2 =>
      conditional(children.init.map(c => (IsNotNull(c), c)), Some(children.last), expr, input)

    case e: Add => arithmetic(ArithOp.ADD, e.left, e.right, e.evalMode, e, input)
    case e: Subtract => arithmetic(ArithOp.SUB, e.left, e.right, e.evalMode, e, input)
    case e: Multiply => arithmetic(ArithOp.MUL, e.left, e.right, e.evalMode, e, input)
    case e: Divide => arithmetic(ArithOp.DIV, e.left, e.right, e.evalMode, e, input)

    case e @ UnaryMinus(child, failOnError) =>
      compile(child, input).flatMap {
        case _: LiteralExpr => Left("negation of a literal")
        case c if !arithmeticTypes.contains(c.vecType) => Left(s"negation not supported for ${c.dataType.simpleString}")
        // Doubles and decimals of at most 18 digits cannot overflow here; integers raise in ANSI mode.
        case c => Right(NegateExpr(c, ansi = failOnError && c.vecType != VecType.FLOAT64 && !TypeMapping.isDecimal(c.dataType), e.origin.context))
      }

    // The optimizer's DecimalAggregates rewrite: sum(decimal) becomes MakeDecimal(sum(UnscaledValue(x))).
    case UnscaledValue(child) =>
      compile(child, input).flatMap {
        case _: LiteralExpr => Left("unscaled value of a literal")
        case c if !TypeMapping.isDecimal(c.dataType) => Left(s"unscaled value of ${c.dataType.simpleString}")
        case c => Right(UnscaledValueExpr(c))
      }
    case m: MakeDecimal =>
      compile(m.child, input).flatMap {
        case _: LiteralExpr => Left("make_decimal of a literal")
        case c if c.vecType != VecType.INT64 => Left(s"make_decimal of ${c.dataType.simpleString}")
        case c if !TypeMapping.isSupported(m.dataType) => Left(s"make_decimal into ${m.dataType.simpleString} exceeds ${TypeMapping.MAX_DECIMAL_PRECISION} digits")
        case c => Right(MakeDecimalExpr(c, m.dataType.asInstanceOf[DecimalType], m.nullOnOverflow))
      }

    // Casts to the operand's own type (Spark's Average emits `sum.cast(double)` on a double sum).
    case c: Cast if c.child.dataType == c.dataType => compile(c.child, input)

    case c: Cast if TypeMapping.isDecimal(c.child.dataType) || TypeMapping.isDecimal(c.dataType) =>
      decimalCast(c, input)

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
   * arithmetic in ANSI mode is computed wrapping and checked afterwards with an overflow lane mask
   * (`OverflowKernels`); the error is raised only if an active row overflowed.
   */
  private def arithmetic(
      op: ArithOp,
      l: Expression,
      r: Expression,
      mode: EvalMode.Value,
      e: Expression,
      input: Seq[Attribute]): Result =
    if (TypeMapping.isDecimal(l.dataType) || TypeMapping.isDecimal(r.dataType)) decimalArithmetic(op, l, r, mode, e, input)
    else for {
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
    else Right(())
  }

  /**
   * Decimal arithmetic on unscaled long lanes. Spark computes the result type from the operand
   * types (`max(p1-s1, p2-s2) + max(s1, s2) + 1` digits for `+`/`-`, `p1+p2+1` for `*`), which
   * leaves room for every result: only division needs an overflow check. Results wider than 18
   * digits have no lane representation and fall back.
   */
  private def decimalArithmetic(
      op: ArithOp,
      l: Expression,
      r: Expression,
      mode: EvalMode.Value,
      e: Expression,
      input: Seq[Attribute]): Result = (l.dataType, r.dataType, e.dataType) match {
    case (lt: DecimalType, rt: DecimalType, dt: DecimalType) =>
      if (!TypeMapping.isSupported(dt)) Left(s"decimal result ${dt.simpleString} exceeds ${TypeMapping.MAX_DECIMAL_PRECISION} digits")
      else if (!TypeMapping.isSupported(lt) || !TypeMapping.isSupported(rt)) Left(s"decimal operand wider than ${TypeMapping.MAX_DECIMAL_PRECISION} digits")
      else if (mode == EvalMode.TRY) Left("try_* arithmetic not supported")
      else {
        val shapeOk = op match {
          case ArithOp.ADD | ArithOp.SUB => dt.scale == math.max(lt.scale, rt.scale)
          case ArithOp.MUL => dt.scale == lt.scale + rt.scale
          case ArithOp.DIV => rt.scale + dt.scale - lt.scale >= 0
        }
        if (!shapeOk) Left(s"unexpected decimal result scale for ${e.sql}")
        else for {
          le <- compile(l, input)
          re <- compile(r, input)
          _ <- if (le.isInstanceOf[LiteralExpr] && re.isInstanceOf[LiteralExpr]) Left("arithmetic on two literals") else Right(())
        } yield DecimalArithExpr(op, le, re, lt, rt, dt, mode == EvalMode.ANSI, e.origin.context)
      }
    case _ => Left(s"mixed decimal and non-decimal arithmetic: ${e.sql}")
  }

  private def decimalCast(c: Cast, input: Seq[Attribute]): Result = {
    val from = c.child.dataType
    val to = c.dataType
    val supportedPair = (from, to) match {
      case (_: DecimalType, _: DecimalType) => true
      case (IntegerType | LongType | DoubleType, _: DecimalType) => true
      case (_: DecimalType, DoubleType | LongType | IntegerType) => true
      case _ => false
    }
    if (!TypeMapping.isSupported(from) || !TypeMapping.isSupported(to)) Left(s"unsupported cast ${from.simpleString} -> ${to.simpleString}")
    else if (!supportedPair) Left(s"unsupported cast ${from.simpleString} -> ${to.simpleString}")
    else if (c.evalMode == EvalMode.TRY) Left("try_cast not supported")
    else compile(c.child, input).flatMap {
      case _: LiteralExpr => Left("cast of a literal")
      case child => Right(DecimalCastExpr(child, from, to, c.evalMode == EvalMode.ANSI, c.origin.context))
    }
  }

  private def conditional(
      branches: Seq[(Expression, Expression)],
      elseValue: Option[Expression],
      e: Expression,
      input: Seq[Attribute]): Result = {
    if (!TypeMapping.isSupported(e.dataType)) Left(s"unsupported result type ${e.dataType.simpleString} for ${e.sql}")
    else {
      def value(v: Expression): Either[String, Option[VectorExpr]] = v match {
        case Literal(null, _) => Right(None)
        case Literal(x, dt) if dt == e.dataType && CaseWhenExpr.isBranchLiteralType(dt) => Right(Some(LiteralExpr(x, dt)))
        case other if other.dataType != e.dataType => Left(s"branch type ${other.dataType.simpleString} differs from ${e.dataType.simpleString}")
        case other => compile(other, input).map(Some(_))
      }
      val compiled = branches.foldLeft[Either[String, Vector[(VectorExpr, Option[VectorExpr])]]](Right(Vector.empty)) {
        case (acc, (cond, v)) =>
          for {
            done <- acc
            c <- booleanChild(cond, input)
            bv <- value(v)
          } yield done :+ ((c, bv))
      }
      for {
        bs <- compiled
        rest <- elseValue match {
          case Some(x) => value(x).map(Some(_))
          case None => Right(None)
        }
      } yield CaseWhenExpr(bs, rest, e.dataType)
    }
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

  /**
   * `value IN (list)` where every element is a non-null literal of the value's type (Spark's own
   * type coercion has already folded any casts into the literals). A list with a `NULL`, a
   * non-literal element or a mismatched type falls back; so does `InSet`, the optimizer's rewrite
   * above `spark.sql.optimizer.inSetConversionThreshold` literals (#48).
   */
  private def inList(value: Expression, list: Seq[Expression], input: Seq[Attribute]): Result = {
    if (list.isEmpty) Left("empty IN list")
    else if (list.exists { case Literal(null, _) => true; case _ => false }) Left("NULL in IN list")
    else if (!list.forall(_.isInstanceOf[Literal])) Left("IN list is not all literals")
    else if (list.exists(_.dataType != value.dataType)) Left(s"IN operands differ: ${value.dataType.simpleString} vs ${list.map(_.dataType.simpleString).distinct.mkString("/")}")
    else compile(value, input).flatMap {
      case _: LiteralExpr => Left("IN over a literal")
      case c if !comparableTypes.contains(c.vecType) => Left(s"IN not supported for ${value.dataType.simpleString}")
      case c =>
        val lits = list.map(compile(_, input))
        lits.collectFirst { case Left(reason) => reason } match {
          case Some(reason) => Left(reason)
          case None => Right(InExpr(c, lits.collect { case Right(lit: LiteralExpr) => lit }))
        }
    }
  }

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
