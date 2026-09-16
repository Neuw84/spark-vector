package io.sparkvector.spark.expr

import io.sparkvector.kernels.{ArrowLayout, SegmentVectorBuffers, SequenceKernels, VecType, VectorBuffers}
import org.apache.spark.TaskContext
import org.apache.spark.sql.types.{DataType, LongType}

/**
 * `monotonically_increasing_id()`: Spark's contract is `partitionIndex.toLong << 33 | rowNumber`,
 * with the row number counting the rows that reach the expression within the partition. The
 * compiled expression tree is deserialised once per task, so the counter below is per task -- one
 * partition -- exactly like Spark's `Stateful` implementation; the partition prefix is read from the
 * task context on first use.
 *
 * With a forwarded selection only the selected rows are numbered, in order: those are the rows
 * Spark would have seen after the filter below; unselected lanes are masked by the consumer.
 * Rows outside an `active` mask (a CASE branch) are still numbered -- Spark evaluates the id only on
 * the branch's rows, a difference this node accepts rather than reproducing, since the id is a tag
 * and only its monotonicity and uniqueness within the partition are contractual.
 */
final case class MonotonicIdExpr() extends VectorExpr {
  override def dataType: DataType = LongType
  override def children: Seq[VectorExpr] = Nil

  // Not transient on purpose: the driver never evaluates, so -1 travels to every task and marks
  // "prefix not yet read" (a transient Long would come back as 0 and lose the partition prefix).
  private var next: Long = -1L

  override def eval(ctx: EvalContext): VectorBuffers = {
    if (next < 0L) {
      val partition = Option(TaskContext.get()).map(_.partitionId()).getOrElse(0)
      next = partition.toLong << 33
    }
    val n = ctx.numRows
    val data = ArrowLayout.allocateData(ctx.arena, VecType.INT64, n)
    next = SequenceKernels.iotaSelected(data, n, ctx.selection, next)
    SegmentVectorBuffers.fixedWidth(VecType.INT64, n, null, data)
  }
}
