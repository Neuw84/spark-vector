package io.sparkvector.spark.expr

import io.sparkvector.kernels.{ArithOp, BitKernels, CastKernels, CompareOp, DateKernels, MathKernels, PredicateKernels, RoundKernels, StringCaseKernels, StringLengthKernels, StringMatchKernels, VecType}
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.kernels.TranscendentalKernels
import org.apache.spark.sql.catalyst.expressions.{Abs, Acos, Acosh, Add, AddMonths, Alias, And, Ascii, Asin, Asinh, Atan, Atan2, Atanh, Attribute, AttributeReference, BitLength, BitwiseAnd, BitwiseCount, BitwiseGet, BitwiseNot, BitwiseOr, BitwiseXor, BloomFilterMightContain, BoundReference, BRound, CaseWhen, Cast, Cbrt, Ceil, Chr, Coalesce, Concat, ConcatWs, Contains, Crc32, Cos, Cosh, Cot, Csc, DateAdd, DateDiff, DateFromUnixDate, DateSub, DayOfMonth, DayOfWeek, DayOfYear, Divide, ElementAt, Elt, EndsWith, EqualNullSafe, EqualTo, EvalMode, Exp, Expm1, Expression, FindInSet, Floor, GreaterThan, Greatest, GreaterThanOrEqual, Hour, Hypot, If, In, InitCap, InSet, IntegralDivide, IsNaN, IsNotNull, IsNull, KnownFloatingPointNormalized, LastDay, Least, Length, LessThan, LessThanOrEqual, Literal, Log, Log10, Log1p, Log2, Logarithm, Lower, Md5, Murmur3Hash, MakeDate, MakeDecimal, MicrosToTimestamp, MillisToTimestamp, Minute, MonotonicallyIncreasingID, Month, MonthsBetween, Multiply, NaNvl, NextDay, Not, OctetLength, Or, Pmod, Pow, Quarter, Remainder, Rint, Round, RoundCeil, RoundFloor, Overlay, Sec, Second, SecondsToTimestamp, Sha1, Sha2, ShiftLeft, ShiftRight, ShiftRightUnsigned, Signum, Sin, Sinh, Sqrt, StartsWith, StringInstr, StringLocate, StringLPad, StringRepeat, StringReplace, StringRPad, StringSpace, StringSplitSQL, StringTranslate, StringTrim, StringTrimLeft, StringTrimRight, Substring, SubstringIndex, Subtract, Tan, Tanh, ToDegrees, ToRadians, TruncDate, UnaryMathExpression, UnaryMinus, UnaryPositive, UnixDate, UnixMicros, UnixMillis, UnixSeconds, UnscaledValue, Upper, WeekDay, WeekOfYear, XxHash64, Year}
import org.apache.spark.sql.catalyst.optimizer.NormalizeNaNAndZero
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.sql.types.{BinaryType, BooleanType, DataType, DateType, DecimalType, DoubleType, IntegerType, LongType, StringType, TimestampType}

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
    // A scalar subquery result in any reference-free shape (the ScalarSubquery, a struct field of a
    // merged one, a CASE over such fields): a literal by the time the operator runs, read then.
    case e if SubqueryLiteralExpr.isDeferred(e) =>
      if (SubqueryLiteralExpr.supports(e.dataType)) Right(SubqueryLiteralExpr(e))
      else Left(s"subquery result of type ${e.dataType.simpleString} not supported")

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
    case InSet(value, hset) => inSet(value, hset, input)
    case EqualNullSafe(l, r) => nullSafeEquality(l, r, input)
    case IsNaN(child) =>
      if (child.dataType != DoubleType) Left(s"isnan over ${child.dataType.simpleString}")
      else numericChild(child, input, "isnan").map(IsNaNExpr(_))

    case StartsWith(l, r) => stringMatch(StringMatchKernels.Kind.PREFIX, l, r, input)

    // Hashes: Spark's own Murmur3 steps per child (the xxhash64 chain's per-type rules); digests per row.
    case Murmur3Hash(children, seed) if children.nonEmpty && children.forall(c => XxHash64Expr.supports(c.dataType)) =>
      children.foldLeft[Either[String, Vector[VectorExpr]]](Right(Vector.empty)) { (acc, c) =>
        for (done <- acc; v <- compile(c, input)) yield done :+ v
      }.map(vs => Murmur3HashExpr(vs, children.map(_.dataType), seed))
    case Md5(BinaryFromString(str)) => stringSubject(str, input, "md5").map(DigestExpr(DigestExpr.Md5, _))
    case Sha1(BinaryFromString(str)) => stringSubject(str, input, "sha1").map(DigestExpr(DigestExpr.Sha1, _))
    case Sha2(BinaryFromString(str), Literal(bits: Int, IntegerType)) => stringSubject(str, input, "sha2").map(DigestExpr(DigestExpr.Sha2(bits), _))
    case Sha2(_, _) => Left("sha2 with a non-literal bit length not supported")
    case Crc32(BinaryFromString(str)) => stringSubject(str, input, "crc32").map(DigestExpr(DigestExpr.Crc32, _))

    // The search family: one byte-search primitive under every function; string children only.
    case StringInstr(str, sub) if str.dataType == StringType =>
      for (h <- stringArg(str, input, "instr"); nd <- stringArg(sub, input, "instr")) yield LocateExpr(nd, h, LiteralExpr(1, IntegerType))
    case StringLocate(sub, str, start) if str.dataType == StringType =>
      for (nd <- stringArg(sub, input, "locate"); h <- stringArg(str, input, "locate"); st <- intArg(start, input, "locate")) yield LocateExpr(nd, h, st)
    case StringReplace(str, search, replacement) if str.dataType == StringType =>
      for (h <- stringSubject(str, input, "replace"); se <- stringArg(search, input, "replace"); r <- stringArg(replacement, input, "replace")) yield ReplaceExpr(h, se, r)
    case StringTranslate(str, Literal(m: UTF8String, StringType), Literal(r: UTF8String, StringType)) if str.dataType == StringType =>
      stringSubject(str, input, "translate").map { h =>
        val (from, to) = TranslateExpr.dictionary(m.toString, r.toString)
        TranslateExpr(h, from, to)
      }
    case StringTranslate(_, _, _) => Left("translate with a non-literal from/to string not supported")
    case SubstringIndex(str, delim, count) if str.dataType == StringType =>
      for (h <- stringSubject(str, input, "substring_index"); d <- stringArg(delim, input, "substring_index"); c <- intArg(count, input, "substring_index")) yield SubstringIndexExpr(h, d, c)
    case e @ ElementAt(StringSplitSQL(str, delim), part, Some(Literal(dflt: UTF8String, StringType)), _) if str.dataType == StringType && dflt.numBytes == 0 =>
      for (h <- stringSubject(str, input, "split_part"); d <- stringArg(delim, input, "split_part"); k <- intArg(part, input, "split_part")) yield SplitPartExpr(h, d, k, e.origin.context)
    case FindInSet(word, set) if set.dataType == StringType =>
      for (w <- stringArg(word, input, "find_in_set"); st <- stringArg(set, input, "find_in_set")) yield FindInSetExpr(w, st)

    // Case mapping: the ASCII rows in the kernel, the rest through Spark's own CollationSupport.
    // A collated column follows ICU rules and is not byte-mapped -- the line Comet draws too.
    case Upper(child) => caseMap(StringCaseKernels.Kind.UPPER, child, input, "upper")
    case Lower(child) => caseMap(StringCaseKernels.Kind.LOWER, child, input, "lower")
    case InitCap(child) => caseMap(StringCaseKernels.Kind.INITCAP, child, input, "initcap")
    case StringTrim(src, trimStr) => trim(StringCaseKernels.Side.BOTH, src, trimStr, input, "trim")
    case StringTrimLeft(src, trimStr) => trim(StringCaseKernels.Side.LEFT, src, trimStr, input, "ltrim")
    case StringTrimRight(src, trimStr) => trim(StringCaseKernels.Side.RIGHT, src, trimStr, input, "rtrim")

    // Several string inputs: per-row lengths summed across them, one buffer, a per-input copy loop.
    // Array and binary forms are declined by type; an all-literal call folds in Spark.
    case Concat(children) if children.nonEmpty && children.forall(_.dataType == StringType) =>
      stringArgs(children, input, "concat").map(ConcatExpr)
    case ConcatWs(children) if children.nonEmpty && children.forall(_.dataType == StringType) =>
      stringArgs(children, input, "concat_ws").map(ps => ConcatWsExpr(ps.head, ps.tail))
    case e @ Elt(children, failOnError) if children.length > 1 && children.tail.forall(_.dataType == StringType) =>
      for (i <- intArg(children.head, input, "elt"); ps <- stringArgs(children.tail, input, "elt")) yield EltExpr(i, ps, failOnError, e.origin.context)

    // The measuring family: an INT32 per string, once per dictionary entry on dictionary input.
    case Length(child) if child.dataType == StringType => stringSubject(child, input, "length").map(StringMeasureExpr(StringLengthKernels.Measure.CHARS, _))
    case OctetLength(child) if child.dataType == StringType => stringSubject(child, input, "octet_length").map(StringMeasureExpr(StringLengthKernels.Measure.BYTES, _))
    case BitLength(child) if child.dataType == StringType => stringSubject(child, input, "bit_length").map(StringMeasureExpr(StringLengthKernels.Measure.BITS, _))
    case Ascii(child) => stringSubject(child, input, "ascii").map(StringMeasureExpr(StringLengthKernels.Measure.ASCII, _))
    case Chr(child) =>
      compile(child, input).flatMap {
        case _: LiteralExpr => Left("chr of a literal")
        case c if c.vecType == VecType.INT32 || c.vecType == VecType.INT64 => Right(ChrExpr(c))
        case _ => Left(s"chr over ${child.dataType.simpleString} not supported")
      }

    // The slicing family writes new UTF8 data: Spark's UTF8String semantics per lane (left/right are
    // Spark's own rewrites onto substring; binary inputs are not lanes and fall back).
    case Substring(str, pos, len) if str.dataType == StringType =>
      for (s <- stringSubject(str, input, "substring"); p <- intArg(pos, input, "substring"); l <- intArg(len, input, "substring")) yield SubstringExpr(s, p, l)
    case StringLPad(str, len, pad) =>
      for (s <- stringSubject(str, input, "lpad"); l <- intArg(len, input, "lpad", StringSlices.MaxLiteralCount); pd <- stringArg(pad, input, "lpad")) yield PadExpr(s, l, pd, left = true)
    case StringRPad(str, len, pad) =>
      for (s <- stringSubject(str, input, "rpad"); l <- intArg(len, input, "rpad", StringSlices.MaxLiteralCount); pd <- stringArg(pad, input, "rpad")) yield PadExpr(s, l, pd, left = false)
    case StringRepeat(str, times) =>
      for (s <- stringSubject(str, input, "repeat"); t <- intArg(times, input, "repeat", StringSlices.MaxLiteralCount)) yield RepeatExpr(s, t)
    case StringSpace(n) =>
      intArg(n, input, "space", StringSlices.MaxLiteralCount).flatMap {
        case _: LiteralExpr => Left("space of a literal") // folds in Spark; a constant column has no producer here
        case c => Right(SpaceExpr(c))
      }
    case Overlay(in, replace, pos, len) if in.dataType == StringType =>
      for {
        s <- stringSubject(in, input, "overlay"); r <- stringArg(replace, input, "overlay")
        p <- intArg(pos, input, "overlay"); l <- intArg(len, input, "overlay")
      } yield OverlayExpr(s, r, p, l)
    case EndsWith(l, r) => stringMatch(StringMatchKernels.Kind.SUFFIX, l, r, input)
    case Contains(l, r) => stringMatch(StringMatchKernels.Kind.CONTAINS, l, r, input)

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

    // timestamp -> date is zone dependent: compiled only under a UTC or fixed-offset session zone.
    case c @ Cast(child, DateType, _, _) if child.dataType == TimestampType =>
      DateExprs.fixedOffsetMicros(c.timeZoneId) match {
        case None => Left(s"cast timestamp -> date needs a fixed-offset session zone, not ${c.timeZoneId.getOrElse("none")}")
        case Some(offset) =>
          compile(child, input).flatMap {
            case _: LiteralExpr => Left("cast of a literal")
            case ce => Right(TimestampToDateExpr(ce, offset))
          }
      }

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

    case Year(child) => dateField(DateKernels.Field.YEAR, child, input)
    case Month(child) => dateField(DateKernels.Field.MONTH, child, input)
    case DayOfMonth(child) => dateField(DateKernels.Field.DAY, child, input)
    case DayOfYear(child) => dateField(DateKernels.Field.DAY_OF_YEAR, child, input)
    case Quarter(child) => dateField(DateKernels.Field.QUARTER, child, input)
    case DayOfWeek(child) => dateField(DateKernels.Field.DAY_OF_WEEK, child, input)
    case WeekDay(child) => dateField(DateKernels.Field.WEEKDAY, child, input)

    case TruncDate(date, format) =>
      format match {
        case Literal(f: UTF8String, StringType) =>
          DateExprs.truncUnit(f.toString) match {
            case None => Left(s"trunc unit '$f' not supported")
            case Some(unit) => dateChild(date, input).map(DateTruncExpr(unit, _))
          }
        case _ => Left("trunc unit is not a string literal")
      }

    // date +/- days and date - date are INT32 lane arithmetic; Spark does not overflow-check them.
    case e @ DateAdd(start, days) => dateArith(ArithOp.ADD, start, days, e.dataType, input)
    case e @ DateSub(start, days) => dateArith(ArithOp.SUB, start, days, e.dataType, input)
    case e @ DateDiff(end, start) if end.dataType == DateType && start.dataType == DateType =>
      dateArith(ArithOp.SUB, end, start, e.dataType, input)

    // Day-number relabels and epoch scaling: the lane is the same, only Spark's type changes.
    case UnixDate(child) => dateChild(child, input).map(RelabelExpr(_, IntegerType))
    case DateFromUnixDate(child) => intLane(child, input, "date_from_unix_date").map(RelabelExpr(_, DateType))
    case MicrosToTimestamp(child) => integralLane(child, input, "timestamp_micros").map(RelabelExpr(_, TimestampType))
    case UnixMicros(child) => timestampLane(child, input, "unix_micros").map(RelabelExpr(_, LongType))
    case SecondsToTimestamp(child) if child.dataType == IntegerType || child.dataType == LongType =>
      integralLane(child, input, "timestamp_seconds").map(EpochScaleExpr(_, 1000000L, toMicros = true, TimestampType))
    case SecondsToTimestamp(child) => Left(s"timestamp_seconds over ${child.dataType.simpleString} not supported")
    case MillisToTimestamp(child) => integralLane(child, input, "timestamp_millis").map(EpochScaleExpr(_, 1000L, toMicros = true, TimestampType))
    case UnixSeconds(child) => timestampLane(child, input, "unix_seconds").map(EpochScaleExpr(_, 1000000L, toMicros = false, LongType))
    case UnixMillis(child) => timestampLane(child, input, "unix_millis").map(EpochScaleExpr(_, 1000L, toMicros = false, LongType))

    // Month and week arithmetic on the civil-date conversion.
    case LastDay(child) => dateChild(child, input).map(DateScalarExpr(DateScalarExpr.LastDay, _, None, DateType))
    case WeekOfYear(child) => dateChild(child, input).map(DateScalarExpr(DateScalarExpr.WeekOfYear, _, None, IntegerType))
    case AddMonths(start, months) =>
      for (d <- dateChild(start, input); m <- intArg(months, input, "add_months")) yield DateScalarExpr(DateScalarExpr.AddMonths, d, Some(m), DateType)
    case NextDay(start, Literal(day: UTF8String, StringType), _) =>
      val code = DateKernels.dayOfWeekCode(day.toString)
      if (code < 0) Left(s"next_day with an unknown day name '$day'") else dateChild(start, input).map(DateScalarExpr(DateScalarExpr.NextDay(code), _, None, DateType))
    case NextDay(_, _, _) => Left("next_day with a non-literal day name not supported")
    case e @ MonthsBetween(a, b, Literal(roundOff: Boolean, BooleanType), tz) =>
      DateExprs.fixedOffsetMicros(tz) match {
        case None => Left(s"months_between needs a fixed-offset session zone, not ${tz.getOrElse("none")}")
        case Some(offset) =>
          for (ca <- instantLane(a, input, "months_between"); cb <- instantLane(b, input, "months_between"))
            yield MonthsBetweenExpr(ca._1, cb._1, ca._2, cb._2, roundOff, offset)
      }
    case e @ MakeDate(y, m, d, failOnError) =>
      for (cy <- intLane(y, input, "make_date"); cm <- intLane(m, input, "make_date"); cd <- intLane(d, input, "make_date"))
        yield MakeDateExpr(cy, cm, cd, failOnError, e.origin.context)

    // Per-partition prefix plus a running row number; state lives in the node, per task.
    case _: MonotonicallyIncreasingID => Right(MonotonicIdExpr())

    case e @ Abs(child, failOnError) =>
      numericChild(child, input, "abs").map(c => AbsExpr(c, ansi = failOnError && c.vecType != VecType.FLOAT64, e.origin.context))
    case UnaryPositive(child) => compile(child, input)
    case Signum(child) =>
      if (child.dataType != DoubleType) Left(s"signum over ${child.dataType.simpleString}")
      else numericChild(child, input, "signum").map(SignumExpr(_))
    case Sqrt(child) =>
      if (child.dataType != DoubleType) Left(s"sqrt over ${child.dataType.simpleString}")
      else numericChild(child, input, "sqrt").map(SqrtExpr(_))

    // The transcendental family: one scalar call per lane, exactly Spark's (Math or StrictMath per
    // function); the analyzer has cast the argument to double. pi() and e() fold to literals.
    case e: UnaryMathExpression if transcendental.contains(e.getClass) =>
      unaryMath(transcendental(e.getClass), e.child, e.prettyName, input)
    case x: XxHash64 =>
      x.children.find(c => !XxHash64Expr.supports(c.dataType)) match {
        case Some(c) => Left(s"xxhash64 over ${c.dataType.simpleString} not supported")
        case None =>
          x.children.foldRight[Either[String, List[VectorExpr]]](Right(Nil)) { (c, acc) =>
            for (rest <- acc; e <- compile(c, input)) yield e :: rest
          }.map(cs => XxHash64Expr(cs, x.children.map(_.dataType), x.seed))
      }
    case b: BloomFilterMightContain =>
      for {
        filter <- {
          val f = b.bloomFilterExpression
          if (SubqueryLiteralExpr.isDeferred(f)) Right(SubqueryLiteralExpr(f)) // binary: read as bytes, never a lane
          else Left("bloom filter is not a subquery result")
        }
        hash <- compile(b.valueExpression, input).flatMap {
          case h if h.vecType == VecType.INT64 && !h.isInstanceOf[LiteralExpr] => Right(h)
          case _ => Left("bloom filter probe value is not a long lane")
        }
      } yield BloomProbeExpr(filter, hash)

    case Pow(l, r) => binaryMath(TranscendentalKernels.Fn2.POW, l, r, "pow", input)
    case Atan2(l, r) => binaryMath(TranscendentalKernels.Fn2.ATAN2, l, r, "atan2", input)
    case Hypot(l, r) => binaryMath(TranscendentalKernels.Fn2.HYPOT, l, r, "hypot", input)
    case Logarithm(l, r) => binaryMath(TranscendentalKernels.Fn2.LOG_BASE, l, r, "log", input)

    case e: Remainder => divideLike(DivideLikeExpr.Rem, e.left, e.right, e.dataType, e.evalMode, e, input)
    case e: Pmod => divideLike(DivideLikeExpr.Pmod, e.left, e.right, e.dataType, e.evalMode, e, input)
    case e: IntegralDivide => divideLike(DivideLikeExpr.Div, e.left, e.right, e.dataType, e.evalMode, e, input)

    case e @ Greatest(children) => pick(MathKernels.Pick.GREATEST, children, e.dataType, input)
    case e @ Least(children) => pick(MathKernels.Pick.LEAST, children, e.dataType, input)

    case NaNvl(l, r) =>
      if (l.dataType != DoubleType || r.dataType != DoubleType) Left(s"nanvl over ${l.dataType.simpleString}")
      else
        for {
          le <- compile(l, input)
          re <- compile(r, input)
          _ <- if (le.isInstanceOf[LiteralExpr] && re.isInstanceOf[LiteralExpr]) Left("nanvl of two literals") else Right(())
        } yield NanvlExpr(le, re)

    case e @ Ceil(child) => ceilFloor(child, ceil = true, e.dataType, input)
    case e @ Floor(child) => ceilFloor(child, ceil = false, e.dataType, input)
    case Rint(child) =>
      if (child.dataType != DoubleType) Left(s"rint over ${child.dataType.simpleString}")
      else numericChild(child, input, "rint").map(RintExpr(_))
    case r: Round => rounding(RoundKernels.Mode.HALF_UP, "round", r.child, r.scale, r.dataType, r.ansiEnabled, r, input)
    case r: BRound => rounding(RoundKernels.Mode.HALF_EVEN, "bround", r.child, r.scale, r.dataType, r.ansiEnabled, r, input)
    case r: RoundCeil => rounding(RoundKernels.Mode.CEILING, "ceil", r.child, r.scale, r.dataType, ansi = false, r, input)
    case r: RoundFloor => rounding(RoundKernels.Mode.FLOOR, "floor", r.child, r.scale, r.dataType, ansi = false, r, input)

    case e @ BitwiseAnd(l, r) => bitwise(BitKernels.BitOp.AND, "&", l, r, e.dataType, input)
    case e @ BitwiseOr(l, r) => bitwise(BitKernels.BitOp.OR, "|", l, r, e.dataType, input)
    case e @ BitwiseXor(l, r) => bitwise(BitKernels.BitOp.XOR, "^", l, r, e.dataType, input)
    case e @ ShiftLeft(l, r) => bitwise(BitKernels.BitOp.SHL, "shiftleft", l, r, e.dataType, input)
    case e @ ShiftRight(l, r) => bitwise(BitKernels.BitOp.SHR, "shiftright", l, r, e.dataType, input)
    case e @ ShiftRightUnsigned(l, r) => bitwise(BitKernels.BitOp.USHR, "shiftrightunsigned", l, r, e.dataType, input)
    case BitwiseNot(child) => integralChild(child, input, "~").map(BitNotExpr(_))
    case BitwiseCount(child) => integralChild(child, input, "bit_count").map(BitCountExpr(_))
    case _: BitwiseGet => Left("bit_get returns tinyint, which has no lane")

    case h @ Hour(child, _) => timeField(DateKernels.TimeField.HOUR, child, h.timeZoneId, input)
    case m @ Minute(child, _) => timeField(DateKernels.TimeField.MINUTE, child, m.timeZoneId, input)
    case s @ Second(child, _) => timeField(DateKernels.TimeField.SECOND, child, s.timeZoneId, input)

    case other => Left(s"unsupported expression ${other.getClass.getSimpleName}: ${other.sql}")
  }

  /** A compiled non-literal operand on an INT32 / INT64 / FLOAT64 lane (decimals are #49). */
  private val transcendental: Map[Class[_], TranscendentalKernels.Fn] = {
    import TranscendentalKernels.Fn._
    Map(
      classOf[Exp] -> EXP, classOf[Expm1] -> EXPM1, classOf[Log] -> LOG, classOf[Log2] -> LOG2,
      classOf[Log10] -> LOG10, classOf[Log1p] -> LOG1P, classOf[Cbrt] -> CBRT,
      classOf[Sin] -> SIN, classOf[Cos] -> COS, classOf[Tan] -> TAN,
      classOf[Asin] -> ASIN, classOf[Acos] -> ACOS, classOf[Atan] -> ATAN,
      classOf[Sinh] -> SINH, classOf[Cosh] -> COSH, classOf[Tanh] -> TANH,
      classOf[Asinh] -> ASINH, classOf[Acosh] -> ACOSH, classOf[Atanh] -> ATANH,
      classOf[Cot] -> COT, classOf[Sec] -> SEC, classOf[Csc] -> CSC,
      classOf[ToDegrees] -> DEGREES, classOf[ToRadians] -> RADIANS)
  }

  private def doubleChild(e: Expression, input: Seq[Attribute], what: String): Result =
    if (e.dataType != DoubleType) Left(s"$what over ${e.dataType.simpleString}")
    else numericChild(e, input, what)

  private def unaryMath(fn: TranscendentalKernels.Fn, child: Expression, what: String, input: Seq[Attribute]): Result =
    doubleChild(child, input, what).map(UnaryMathExpr(fn, _))

  /** A binary math function: both sides double, a literal on at most one side (both would have folded). */
  private def binaryMath(fn: TranscendentalKernels.Fn2, l: Expression, r: Expression, what: String, input: Seq[Attribute]): Result =
    if (l.dataType != DoubleType || r.dataType != DoubleType) Left(s"$what over ${l.dataType.simpleString}, ${r.dataType.simpleString}")
    else for (a <- compile(l, input); b <- compile(r, input)) yield {
      (a, b) match {
        case (_: LiteralExpr, _: LiteralExpr) => return Left(s"$what of two literals")
        case _ =>
      }
      if (!a.isInstanceOf[LiteralExpr] && !arithmeticTypes.contains(a.vecType)) return Left(s"$what over ${l.dataType.simpleString} not supported")
      if (!b.isInstanceOf[LiteralExpr] && !arithmeticTypes.contains(b.vecType)) return Left(s"$what over ${r.dataType.simpleString} not supported")
      BinaryMathExpr(fn, a, b)
    }

  private def numericChild(e: Expression, input: Seq[Attribute], what: String): Result =
    compile(e, input).flatMap {
      case _: LiteralExpr => Left(s"$what of a literal")
      case c if TypeMapping.isDecimal(c.dataType) => Left(s"$what over ${c.dataType.simpleString} not supported")
      case c if !arithmeticTypes.contains(c.vecType) => Left(s"$what over ${c.dataType.simpleString} not supported")
      case c => Right(c)
    }

  /**
   * `%`, `pmod` and `div`: same Spark type on both sides (Spark's coercion has cast them), numeric
   * lanes only, `try_*` mode falls back. The kernel handles a zero divisor; ANSI raises it.
   */
  private def divideLike(
      kind: DivideLikeExpr.Kind,
      l: Expression,
      r: Expression,
      resultType: DataType,
      mode: EvalMode.Value,
      e: Expression,
      input: Seq[Attribute]): Result = {
    val what = kind match { case DivideLikeExpr.Rem => "%"; case DivideLikeExpr.Pmod => "pmod"; case DivideLikeExpr.Div => "div" }
    if (mode == EvalMode.TRY) Left("try_* arithmetic not supported")
    else if (l.dataType != r.dataType) Left(s"$what operands differ: ${l.dataType.simpleString} vs ${r.dataType.simpleString}")
    else if (TypeMapping.isDecimal(l.dataType)) Left(s"$what over ${l.dataType.simpleString} not supported")
    else if (kind == DivideLikeExpr.Div && TypeMapping.vecTypeOf(l.dataType) == VecType.FLOAT64) Left("div over double not supported")
    else
      for {
        le <- compile(l, input)
        re <- compile(r, input)
        _ <- if (le.isInstanceOf[LiteralExpr] && re.isInstanceOf[LiteralExpr]) Left("arithmetic on two literals") else Right(())
        _ <- if (!arithmeticTypes.contains(le.vecType)) Left(s"$what over ${l.dataType.simpleString} not supported") else Right(())
      } yield DivideLikeExpr(kind, le, re, resultType, mode == EvalMode.ANSI, e.origin.context)
  }

  /** `greatest` / `least`: every child the same Spark type on a numeric lane, at least one non-literal. */
  private def pick(pick: MathKernels.Pick, children: Seq[Expression], resultType: DataType, input: Seq[Attribute]): Result = {
    val what = if (pick == MathKernels.Pick.GREATEST) "greatest" else "least"
    if (children.size < 2) Left(s"$what with fewer than two arguments")
    else if (children.exists(_.dataType != resultType)) Left(s"$what operands differ: ${children.map(_.dataType.simpleString).distinct.mkString("/")}")
    else if (TypeMapping.isDecimal(resultType) || !TypeMapping.isSupported(resultType) || !arithmeticTypes.contains(TypeMapping.vecTypeOf(resultType))) Left(s"$what over ${resultType.simpleString} not supported")
    else {
      val compiled = children.map {
        case Literal(null, _) => Left("null literal")
        case c => compile(c, input)
      }
      compiled.collectFirst { case Left(reason) => reason } match {
        case Some(reason) => Left(reason)
        case None =>
          val exprs = compiled.collect { case Right(c) => c }
          if (exprs.forall(_.isInstanceOf[LiteralExpr])) Left(s"$what of literals only") else Right(PickExpr(pick, exprs, resultType))
      }
    }
  }

  /** `ceil` / `floor`: long is the identity, double goes to long, decimal to Spark's `bounded(p - s + 1, 0)`. */
  private def ceilFloor(child: Expression, ceil: Boolean, resultType: DataType, input: Seq[Attribute]): Result = {
    val what = if (ceil) "ceil" else "floor"
    child.dataType match {
      case LongType => compile(child, input)
      case DoubleType => numericChild(child, input, what).map(CeilFloorExpr(_, ceil, resultType))
      case d: DecimalType if TypeMapping.isSupported(d) && TypeMapping.isSupported(resultType) =>
        compile(child, input).flatMap {
          case _: LiteralExpr => Left(s"$what of a literal")
          case c => Right(CeilFloorExpr(c, ceil, resultType))
        }
      case d: DecimalType => Left(s"$what over ${d.simpleString} -> ${resultType.simpleString} not supported")
      case t => Left(s"$what over ${t.simpleString} not supported")
    }
  }

  /**
   * `round` / `bround` / two-argument `ceil` / `floor`: the scale must be an int literal (Spark
   * requires it foldable), the child a numeric or decimal lane, and the result -- which for decimals
   * Spark widens by a digit -- must still fit 18 digits.
   */
  private def rounding(
      mode: RoundKernels.Mode,
      what: String,
      child: Expression,
      scale: Expression,
      resultType: DataType,
      ansi: Boolean,
      e: Expression,
      input: Seq[Attribute]): Result =
    scale match {
      case Literal(k: Int, IntegerType) =>
        if (!RoundExpr.supports(child.dataType, resultType)) Left(s"$what over ${child.dataType.simpleString} -> ${resultType.simpleString} not supported")
        else
          compile(child, input).flatMap {
            case _: LiteralExpr => Left(s"$what of a literal")
            // An integer rounded to a non-negative scale is itself.
            case c if k >= 0 && !TypeMapping.isDecimal(c.dataType) && c.vecType != VecType.FLOAT64 => Right(c)
            case c => Right(RoundExpr(c, resultType, mode, k, ansi, e.origin.context))
          }
      case Literal(null, _) => Left(s"$what with a null scale")
      case other => Left(s"$what with a non-literal scale ${other.sql}")
    }

  /** A compiled non-literal INT32 / INT64 operand (bytes, shorts and booleans have no lane here). */
  private def integralChild(e: Expression, input: Seq[Attribute], what: String): Result =
    if (e.dataType != IntegerType && e.dataType != LongType) Left(s"$what over ${e.dataType.simpleString} not supported")
    else
      compile(e, input).flatMap {
        case _: LiteralExpr => Left(s"$what of a literal")
        case c => Right(c)
      }

  /**
   * `& | ^` (same integral type on both sides after Spark's coercion) and the shifts (an INT32 amount
   * on the right); a literal may sit on either side but not both.
   */
  private def bitwise(op: BitKernels.BitOp, what: String, l: Expression, r: Expression, resultType: DataType, input: Seq[Attribute]): Result = {
    val leftOk = l.dataType == IntegerType || l.dataType == LongType
    val rightOk = if (op.isShift) r.dataType == IntegerType else r.dataType == l.dataType
    if (!leftOk || !rightOk) Left(s"$what over ${l.dataType.simpleString}, ${r.dataType.simpleString} not supported")
    else
      for {
        le <- compile(l, input)
        re <- compile(r, input)
        _ <- if (le.isInstanceOf[LiteralExpr] && re.isInstanceOf[LiteralExpr]) Left(s"$what of two literals") else Right(())
      } yield BitBinaryExpr(op, le, re, resultType)
  }

  /** A compiled non-literal date operand. */
  private def dateChild(e: Expression, input: Seq[Attribute]): Result =
    if (e.dataType != DateType) Left(s"date function over ${e.dataType.simpleString}")
    else compile(e, input).flatMap {
      case _: LiteralExpr => Left("date function on a literal")
      case c => Right(c)
    }

  private def intLane(e: Expression, input: Seq[Attribute], what: String): Result =
    if (e.dataType != IntegerType) Left(s"$what argument ${e.dataType.simpleString} is not an int") else compile(e, input)

  private def integralLane(e: Expression, input: Seq[Attribute], what: String): Result =
    if (e.dataType != IntegerType && e.dataType != LongType) Left(s"$what over ${e.dataType.simpleString} not supported")
    else compile(e, input).flatMap { case _: LiteralExpr => Left(s"$what of a literal"); case c => Right(c) }

  private def timestampLane(e: Expression, input: Seq[Attribute], what: String): Result =
    if (e.dataType != TimestampType) Left(s"$what over ${e.dataType.simpleString} not supported")
    else compile(e, input).flatMap { case _: LiteralExpr => Left(s"$what of a literal"); case c => Right(c) }

  /** A timestamp lane, or a date lane behind Spark's date -> timestamp cast (the flag says which). */
  private def instantLane(e: Expression, input: Seq[Attribute], what: String): Either[String, (VectorExpr, Boolean)] = e match {
    case Cast(child, TimestampType, _, _) if child.dataType == DateType => compile(child, input).map(c => (c, true))
    case _ if e.dataType == TimestampType => compile(e, input).map(c => (c, false))
    case _ if e.dataType == DateType => compile(e, input).map(c => (c, true))
    case _ => Left(s"$what over ${e.dataType.simpleString} not supported")
  }

  private def dateField(field: DateKernels.Field, child: Expression, input: Seq[Attribute]): Result =
    dateChild(child, input).map(DateFieldExpr(field, _))

  /** `date_add` / `date_sub` take a date and an int; `datediff` two dates. Both are INT32 lanes. */
  private def dateArith(op: ArithOp, l: Expression, r: Expression, resultType: DataType, input: Seq[Attribute]): Result =
    if (l.dataType != DateType) Left(s"date arithmetic over ${l.dataType.simpleString}")
    else if (r.dataType != IntegerType && r.dataType != DateType) Left(s"date arithmetic with ${r.dataType.simpleString} days")
    else
      for {
        le <- compile(l, input)
        re <- compile(r, input)
        _ <- if (le.isInstanceOf[LiteralExpr] && re.isInstanceOf[LiteralExpr]) Left("arithmetic on two literals") else Right(())
      } yield ArithExpr(op, le, re, resultType, ansiDivideByZero = false, queryContext = null)

  private def timeField(field: DateKernels.TimeField, child: Expression, zoneId: Option[String], input: Seq[Attribute]): Result =
    if (child.dataType != TimestampType) Left(s"time field over ${child.dataType.simpleString}")
    else DateExprs.fixedOffsetMicros(zoneId) match {
      case None => Left(s"time field needs a fixed-offset session zone, not ${zoneId.getOrElse("none")}")
      case Some(offset) =>
        compile(child, input).flatMap {
          case _: LiteralExpr => Left("time field on a literal")
          case c => Right(TimeFieldExpr(field, c, offset))
        }
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
    if (l.dataType == BooleanType && r.dataType == BooleanType) booleanComparison(op, l, r, input)
    else
      for {
        le <- compile(l, input)
        re <- compile(r, input)
        _ <- check(le, re, l, r)
      } yield CompareExpr(op, le, re)

  /** Comparisons between BOOL columns, or a BOOL column and a boolean literal, on the packed words. */
  private def booleanComparison(op: CompareOp, l: Expression, r: Expression, input: Seq[Attribute]): Result =
    (l, r) match {
      case (Literal(null, _), _) | (_, Literal(null, _)) => Left("comparison against NULL")
      case (_: Literal, _: Literal) => Left("comparison of two literals")
      case (Literal(v: Boolean, _), c) => booleanChild(c, input).map(BoolCompareScalarExpr(op.flip(), _, v))
      case (c, Literal(v: Boolean, _)) => booleanChild(c, input).map(BoolCompareScalarExpr(op, _, v))
      case (a, b) => binaryBoolean(a, b, input)(BoolCompareExpr(op, _, _))
    }

  /** `a <=> b`: same type on both sides, never null; a `NULL` literal side is `IS NULL` of the other. */
  private def nullSafeEquality(l: Expression, r: Expression, input: Seq[Attribute]): Result =
    (l, r) match {
      case (Literal(null, _), Literal(null, _)) => Left("<=> of two literals")
      case (Literal(null, _), c) => nullTestChild(c, input).map(IsNullExpr.apply)
      case (c, Literal(null, _)) => nullTestChild(c, input).map(IsNullExpr.apply)
      case _ if l.dataType == BooleanType && r.dataType == BooleanType =>
        if (l.isInstanceOf[Literal] || r.isInstanceOf[Literal]) Left("<=> of a boolean literal")
        else binaryBoolean(l, r, input)(NullSafeEqExpr(_, _))
      case _ =>
        for {
          le <- compile(l, input)
          re <- compile(r, input)
          _ <- check(le, re, l, r)
        } yield NullSafeEqExpr(le, re)
    }

  /**
   * `value IN set` for the optimizer's `InSet`: numeric lanes take one binary search per row over
   * the sorted keys; strings go through the same per-literal path as `IN`. A set holding `NULL`
   * falls back (Spark's result is then null for non-members).
   */
  private def inSet(value: Expression, hset: Set[Any], input: Seq[Attribute]): Result = {
    if (hset.isEmpty) Left("empty IN set")
    else if (hset.contains(null)) Left("NULL in IN set")
    else compile(value, input).flatMap {
      case _: LiteralExpr => Left("IN over a literal")
      case c if c.vecType == VecType.UTF8 =>
        if (!hset.forall(_.isInstanceOf[UTF8String])) Left("IN set elements are not strings")
        else Right(InExpr(c, hset.toSeq.map(v => LiteralExpr(v, value.dataType))))
      case c if c.vecType == VecType.FLOAT64 =>
        if (!hset.forall(_.isInstanceOf[Double])) Left("IN set elements are not doubles")
        else Right(InSetExpr(c, PredicateKernels.doubleKeys(hset.toArray.map(_.asInstanceOf[Double]))))
      case c if c.vecType == VecType.INT32 || c.vecType == VecType.INT64 =>
        val keys = hset.toArray.map(LiteralExpr(_, value.dataType).number.longValue())
        java.util.Arrays.sort(keys)
        Right(InSetExpr(c, keys))
      case _ => Left(s"IN not supported for ${value.dataType.simpleString}")
    }
  }

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

  /**
   * `startswith` / `endswith` / `contains` of a string column against a non-null string literal.
   * A column pattern, a non-string operand or a `NULL` pattern falls back; `LIKE` with inner
   * wildcards never reaches here (the optimizer leaves it as `Like`, #3 follow-up).
   */
  /** The string a slicing function works on: a UTF8 lane (a literal subject folds in Spark). */
  private def stringSubject(e: Expression, input: Seq[Attribute], what: String): Result =
    compile(e, input).flatMap {
      case _: LiteralExpr => Left(s"$what of a literal")
      case c if c.vecType != VecType.UTF8 => Left(s"$what over ${e.dataType.simpleString} not supported")
      case c => Right(c)
    }

  /** A string argument: a UTF8 lane or a non-null string literal. */
  private def stringArg(e: Expression, input: Seq[Attribute], what: String): Result = e match {
    case Literal(null, _) => Left(s"$what with a null argument")
    case _ =>
      compile(e, input).flatMap {
        case lit: LiteralExpr => Right(lit)
        case c if c.vecType != VecType.UTF8 => Left(s"$what argument ${e.dataType.simpleString} not supported")
        case c => Right(c)
      }
  }

  private def caseMap(kind: StringCaseKernels.Kind, child: Expression, input: Seq[Attribute], what: String): Result =
    if (child.dataType != StringType) Left(s"$what over ${child.dataType.sql.toLowerCase} not supported (a collated string follows ICU rules)")
    else stringSubject(child, input, what).map(CaseMapExpr(kind, _, SQLConf.get.getConf(SQLConf.ICU_CASE_MAPPINGS_ENABLED)))

  /** A trim with no trim string, or a literal one taken as a set of code points; a column trim set falls back. */
  private def trim(side: StringCaseKernels.Side, src: Expression, trimStr: Option[Expression], input: Seq[Attribute], what: String): Result =
    if (src.dataType != StringType) Left(s"$what over ${src.dataType.sql.toLowerCase} not supported")
    else trimStr match {
      case None => stringSubject(src, input, what).map(TrimExpr(side, _, None))
      case Some(Literal(t: UTF8String, StringType)) => stringSubject(src, input, what).map(TrimExpr(side, _, Some(t.toString.codePoints().toArray)))
      case Some(Literal(null, _)) => Left(s"$what with a null trim string")
      case Some(_) => Left(s"$what with a non-literal trim string not supported")
    }

  /** Spark's `string -> binary` cast, a reinterpretation of the bytes: the UTF8 lane underneath serves as the binary. */
  private object BinaryFromString {
    def unapply(e: Expression): Option[Expression] = e match {
      case Cast(child, BinaryType, _, _) if child.dataType == StringType => Some(child)
      case _ => None
    }
  }

  /** Several string arguments, each a UTF8 lane or a non-null literal; at least one must be a lane. */
  private def stringArgs(es: Seq[Expression], input: Seq[Attribute], what: String): Either[String, Seq[VectorExpr]] = {
    val compiled = es.foldLeft[Either[String, Vector[VectorExpr]]](Right(Vector.empty)) { (acc, e) =>
      for (done <- acc; c <- stringArg(e, input, what)) yield done :+ c
    }
    compiled.flatMap { ps =>
      if (ps.forall(_.isInstanceOf[LiteralExpr])) Left(s"$what of literals only") else Right(ps)
    }
  }

  /** An int argument: an INT32 lane or an int literal, the literal bounded when it sizes the output. */
  private def intArg(e: Expression, input: Seq[Attribute], what: String, maxLiteral: Int = Int.MaxValue): Result = e match {
    case Literal(null, _) => Left(s"$what with a null argument")
    case Literal(v: Int, IntegerType) if v > maxLiteral => Left(s"$what count $v exceeds the batch output cap")
    case _ if e.dataType != IntegerType => Left(s"$what argument ${e.dataType.simpleString} is not an int")
    case _ => compile(e, input)
  }

  private def stringMatch(kind: StringMatchKernels.Kind, l: Expression, r: Expression, input: Seq[Attribute]): Result =
    r match {
      case Literal(null, _) => Left("null pattern")
      case Literal(_, StringType) =>
        for {
          c <- compile(l, input)
          _ <- c match {
            case _: LiteralExpr => Left("string match on a literal")
            case c if c.vecType != VecType.UTF8 => Left(s"string match not supported for ${l.dataType.simpleString}")
            case _ => Right(())
          }
          p <- compile(r, input)
        } yield StringMatchExpr(kind, c, p.asInstanceOf[LiteralExpr])
      case _: Literal => Left(s"string pattern of type ${r.dataType.simpleString}")
      case _ => Left("string pattern is not a literal")
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
