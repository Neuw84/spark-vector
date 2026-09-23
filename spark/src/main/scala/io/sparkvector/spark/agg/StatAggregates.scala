package io.sparkvector.spark.agg

import io.sparkvector.kernels.{GroupAssignment, VectorBuffers}
import io.sparkvector.spark.expr.{EvalContext, VectorExpr}
import org.apache.spark.sql.types.{DataType, DoubleType, LongType}

/**
 * Spark's one-pass moment statistics -- `CentralMomentAgg` (stddev / variance / skewness / kurtosis and
 * the `regr_*` replacement), `Covariance` and `PearsonCorrelation` -- as one accumulator over doubles.
 *
 * The interesting part is not the arithmetic, it is agreement with Spark: the update is Spark's own
 * streaming Welford step applied row by row in partition order, exactly the sequence Spark's operator
 * runs, so a Partial over the same rows produces the same doubles; the merging modes apply Spark's
 * `mergeExpressions` (the parallel form of Welford) to the incoming buffers, so a Final over the same
 * partials agrees to the last bits given the same merge order. Rows with a null in any argument are
 * skipped, as Spark's `If(isNull, old, new)` does. The buffers are emitted as Spark's DoubleType
 * columns in every mode and the Final's result expression (`sqrt(m2 / (n - 1.0))`, `ck / sqrt(xMk *
 * yMk)`, ...) is Spark's own, compiled over them by the operator's result projection.
 *
 * @param inputs the arguments cast to doubles in the update modes; the buffer columns in the merging modes
 * @param kind   which state and formulas
 * @param merge  whether `inputs` are buffers to merge rather than values to accumulate
 */
final case class MomentsAgg(inputs: Seq[VectorExpr], kind: MomentsAgg.Kind, merge: Boolean) extends VectorAggFunction {
  private val slots = kind.slots
  require(
    inputs.length == (if (merge) slots else kind.arity),
    s"${kind} takes ${kind.arity} arguments / $slots buffers, got ${inputs.length}"
  )

  override def bufferTypes: Seq[DataType] = Seq.fill(slots)(DoubleType)

  @inline private def read(v: VectorBuffers, i: Int): Double = v.data().getAtIndex(VectorBuffers.LE_DOUBLE, i)

  /** One row into the state at `base`: Spark's update over the values, or Spark's merge over a buffer row. */
  private def step(s: Array[Double], base: Int, vs: Array[VectorBuffers], i: Int): Unit = {
    var k = 0
    while (k < vs.length) { if (!Rows.valid(vs(k), i)) return; k += 1 }
    if (merge) {
      val r = new Array[Double](slots)
      k = 0
      while (k < slots) { r(k) = read(vs(k), i); k += 1 }
      kind.merge(s, base, r)
    } else if (vs.length == 1) kind.update(s, base, read(vs(0), i), 0.0)
    else kind.update(s, base, read(vs(0), i), read(vs(1), i))
  }

  override def newState(): AggState = new AggState {
    private val s = new Array[Double](slots)
    override def update(ctx: EvalContext): Unit = {
      val vs = inputs.map(_.eval(ctx)).toArray
      Rows.ungrouped(ctx) { i => step(s, 0, vs, i) }
    }
    override def bufferValues: Array[Any] = s.map(java.lang.Double.valueOf(_): Any)
  }

  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private var s = new Array[Double](16 * slots)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      val need = groups.numGroups() * slots
      if (need > s.length) s = java.util.Arrays.copyOf(s, math.max(need, s.length * 2))
      val vs = inputs.map(_.eval(ctx)).toArray
      Rows.grouped(ctx, groups) { (g, i) => step(s, g * slots, vs, i) }
    }
    override def bufferValue(g: Int, slot: Int): Any =
      java.lang.Double.valueOf(if ((g + 1) * slots <= s.length) s(g * slots + slot) else 0.0)
  }
}

/** `count(a, b, ...)` -- the rows where every argument is non-null (`regr_count`'s replacement). */
final case class CountAllAgg(inputs: Seq[VectorExpr]) extends VectorAggFunction {
  override def bufferTypes: Seq[DataType] = Seq(LongType)
  private def allValid(vs: Array[VectorBuffers], i: Int): Boolean = {
    var k = 0
    while (k < vs.length) { if (!Rows.valid(vs(k), i)) return false; k += 1 }
    true
  }
  override def newState(): AggState = new AggState {
    private var count = 0L
    override def update(ctx: EvalContext): Unit = {
      val vs = inputs.map(_.eval(ctx)).toArray
      Rows.ungrouped(ctx) { i => if (allValid(vs, i)) count += 1 }
    }
    override def bufferValues: Array[Any] = Array(java.lang.Long.valueOf(count))
  }
  override def newGroupedState(): GroupedAggState = new GroupedAggState {
    private var counts = new Array[Long](16)
    override def update(ctx: EvalContext, groups: GroupAssignment): Unit = {
      if (groups.numGroups() > counts.length)
        counts = java.util.Arrays.copyOf(counts, math.max(groups.numGroups(), counts.length * 2))
      val vs = inputs.map(_.eval(ctx)).toArray
      Rows.grouped(ctx, groups) { (g, i) => if (allValid(vs, i)) counts(g) += 1 }
    }
    override def bufferValue(g: Int, slot: Int): Any = java.lang.Long.valueOf(if (g < counts.length) counts(g) else 0L)
  }
}

object MomentsAgg {

  /** A state layout with Spark's update and merge formulas, written in Spark's evaluation order. */
  sealed trait Kind extends Serializable {
    def slots: Int
    def arity: Int
    def update(s: Array[Double], b: Int, x: Double, y: Double): Unit
    def merge(s: Array[Double], b: Int, r: Array[Double]): Unit
  }

  /** `CentralMomentAgg`: (n, avg, m2[, m3[, m4]]) -- `order` 2 for stddev/variance, 3 for skewness, 4 for kurtosis. */
  final case class Central(order: Int) extends Kind {
    require(order >= 2 && order <= 4)
    override def slots: Int = order + 1
    override def arity: Int = 1
    override def update(s: Array[Double], b: Int, x: Double, y: Double): Unit = {
      val n = s(b); val avg = s(b + 1); val m2 = s(b + 2)
      val newN = n + 1.0
      val delta = x - avg
      val deltaN = delta / newN
      val newAvg = avg + deltaN
      val newM2 = m2 + delta * (delta - deltaN)
      val delta2 = delta * delta
      val deltaN2 = deltaN * deltaN
      s(b) = newN; s(b + 1) = newAvg; s(b + 2) = newM2
      if (order >= 3) {
        val m3 = s(b + 3)
        val newM3 = m3 - 3.0 * deltaN * newM2 + delta * (delta2 - deltaN2)
        s(b + 3) = newM3
        if (order >= 4) {
          val m4 = s(b + 4)
          s(b + 4) = m4 - 4.0 * deltaN * newM3 - 6.0 * deltaN2 * newM2 + delta * (delta * delta2 - deltaN * deltaN2)
        }
      }
    }
    override def merge(s: Array[Double], b: Int, r: Array[Double]): Unit = {
      val n1 = s(b); val n2 = r(0)
      val newN = n1 + n2
      val delta = r(1) - s(b + 1)
      val deltaN = if (newN == 0.0) 0.0 else delta / newN
      val newAvg = s(b + 1) + deltaN * n2
      val m2l = s(b + 2); val m2r = r(2)
      val newM2 = m2l + m2r + delta * deltaN * n1 * n2
      s(b) = newN; s(b + 1) = newAvg; s(b + 2) = newM2
      if (order >= 3) {
        val m3l = s(b + 3); val m3r = r(3)
        s(b + 3) = m3l + m3r + deltaN * deltaN * delta * n1 * n2 * (n1 - n2) + 3.0 * deltaN * (n1 * m2r - n2 * m2l)
        if (order >= 4) {
          val m4l = s(b + 4); val m4r = r(4)
          s(b + 4) = m4l + m4r + deltaN * deltaN * deltaN * delta * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) +
            6.0 * deltaN * deltaN * (n1 * n1 * m2r + n2 * n2 * m2l) + 4.0 * deltaN * (n1 * m3r - n2 * m3l)
        }
      }
    }
  }

  /** `Covariance`: (n, xAvg, yAvg, ck). */
  case object Covariance extends Kind {
    override def slots: Int = 4
    override def arity: Int = 2
    override def update(s: Array[Double], b: Int, x: Double, y: Double): Unit = {
      val n = s(b); val xAvg = s(b + 1); val yAvg = s(b + 2); val ck = s(b + 3)
      val newN = n + 1.0
      val dx = x - xAvg
      val dy = y - yAvg
      val dyN = dy / newN
      val newXAvg = xAvg + dx / newN
      val newYAvg = yAvg + dyN
      s(b) = newN; s(b + 1) = newXAvg; s(b + 2) = newYAvg; s(b + 3) = ck + dx * (y - newYAvg)
    }
    override def merge(s: Array[Double], b: Int, r: Array[Double]): Unit = {
      val n1 = s(b); val n2 = r(0)
      val newN = n1 + n2
      val dx = r(1) - s(b + 1)
      val dxN = if (newN == 0.0) 0.0 else dx / newN
      val dy = r(2) - s(b + 2)
      val dyN = if (newN == 0.0) 0.0 else dy / newN
      s(b + 1) = s(b + 1) + dxN * n2
      s(b + 2) = s(b + 2) + dyN * n2
      s(b + 3) = s(b + 3) + r(3) + dx * dyN * n1 * n2
      s(b) = newN
    }
  }

  /**
   * `RegrSlope` / `RegrIntercept`: a `CovPopulation(x, y)` state followed by a `VariancePop(x)` state, both
   * updated only on pairs without a null. The arguments arrive as (x, y), the independent variable first.
   */
  case object Regression extends Kind {
    override def slots: Int = 7
    override def arity: Int = 2
    override def update(s: Array[Double], b: Int, x: Double, y: Double): Unit = {
      Covariance.update(s, b, x, y)
      Central(2).update(s, b + 4, x, 0.0)
    }
    override def merge(s: Array[Double], b: Int, r: Array[Double]): Unit = {
      Covariance.merge(s, b, r)
      Central(2).merge(s, b + 4, java.util.Arrays.copyOfRange(r, 4, 7))
    }
  }

  /** `PearsonCorrelation`: (n, xAvg, yAvg, ck, xMk, yMk). */
  case object Correlation extends Kind {
    override def slots: Int = 6
    override def arity: Int = 2
    override def update(s: Array[Double], b: Int, x: Double, y: Double): Unit = {
      val n = s(b); val xAvg = s(b + 1); val yAvg = s(b + 2); val ck = s(b + 3); val xMk = s(b + 4); val yMk = s(b + 5)
      val newN = n + 1.0
      val dx = x - xAvg
      val dxN = dx / newN
      val dy = y - yAvg
      val dyN = dy / newN
      val newXAvg = xAvg + dxN
      val newYAvg = yAvg + dyN
      s(b) = newN; s(b + 1) = newXAvg; s(b + 2) = newYAvg
      s(b + 3) = ck + dx * (y - newYAvg)
      s(b + 4) = xMk + dx * (x - newXAvg)
      s(b + 5) = yMk + dy * (y - newYAvg)
    }
    override def merge(s: Array[Double], b: Int, r: Array[Double]): Unit = {
      val n1 = s(b); val n2 = r(0)
      val newN = n1 + n2
      val dx = r(1) - s(b + 1)
      val dxN = if (newN == 0.0) 0.0 else dx / newN
      val dy = r(2) - s(b + 2)
      val dyN = if (newN == 0.0) 0.0 else dy / newN
      s(b + 1) = s(b + 1) + dxN * n2
      s(b + 2) = s(b + 2) + dyN * n2
      s(b + 3) = s(b + 3) + r(3) + dx * dyN * n1 * n2
      s(b + 4) = s(b + 4) + r(4) + dx * dxN * n1 * n2
      s(b + 5) = s(b + 5) + r(5) + dy * dyN * n1 * n2
      s(b) = newN
    }
  }
}
