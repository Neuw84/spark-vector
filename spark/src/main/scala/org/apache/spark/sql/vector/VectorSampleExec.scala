package org.apache.spark.sql.vector

import io.sparkvector.kernels.Bitmap
import io.sparkvector.spark.adapter.TypeMapping
import io.sparkvector.spark.arrow.{ArrowOutput, SelectedColumnarBatch}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{LeafExecNode, LocalTableScanExec, SampleExec, SparkPlan}
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}
import org.apache.spark.util.random.BernoulliCellSampler

/**
 * Columnar replacement for SampleExec without replacement (`TABLESAMPLE (p PERCENT)`, `df.sample`).
 * Sampling is a Bernoulli trial per row, so it is a selection: the rows are never copied here, the
 * bitmap is forwarded to a spark-vector parent or compacted for Spark exactly like the filter's.
 *
 * The rows match Spark's for a given seed because the random sequence is Spark's own: SampleExec's
 * row path and its generated code both create a `BernoulliCellSampler(lowerBound, upperBound)`,
 * seed it with `seed + partitionIndex`, and draw once per row in input order. This operator does the
 * same over the live rows of each batch (rows a forwarded selection already dropped draw nothing,
 * as they never reached Spark's sampler either). `GapSamplingIterator`, which skips rather than
 * draws, belongs to `RDD.sample` and is not on the SQL path.
 */
case class VectorSampleExec(
    lowerBound: Double,
    upperBound: Double,
    seed: Long,
    child: SparkPlan,
    emitSelection: Boolean = false)
    extends VectorExec {

  override def output: Seq[Attribute] = child.output
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val (lb, ub, s) = (lowerBound, upperBound, seed)
    val outputAttrs = output.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    val selectionOut = emitSelection
    child.executeColumnar().mapPartitionsWithIndexInternal { (index, iter) =>
      val sampler = new BernoulliCellSampler[AnyRef](lb, ub, false)
      sampler.setSeed(s + index)
      new VectorSampleIterator(iter, sampler, outputAttrs, selectionOut, m)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

private[vector] class VectorSampleIterator(
    input: Iterator[ColumnarBatch],
    sampler: BernoulliCellSampler[AnyRef],
    outputAttrs: Array[(String, DataType)],
    emitSelection: Boolean,
    metrics: VectorMetrics)
    extends VectorBatchIterator(input, "VectorSampleExec") {

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val n = ctx.numRows
      val selection = ctx.bitmap()
      val live = ctx.selection
      var count = 0
      var i = 0
      while (i < n) {
        if ((live == null || Bitmap.isSet(live, i)) && sampler.sample() != 0) {
          Bitmap.set(selection, i)
          count += 1
        }
        i += 1
      }
      if (count == 0) {
        null
      } else if (count == n && live == null) {
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        batch
      } else if (emitSelection && SelectionPolicy.keep(count, n)) {
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        SelectedColumnarBatch.of(SelectedColumnarBatch.columnsOf(batch), n, selection, count, false)
      } else {
        val columns = new Array[ColumnVector](outputAttrs.length)
        var c = 0
        while (c < columns.length) {
          val (name, dt) = outputAttrs(c)
          columns(c) = ArrowOutput.compact(name, dt, ctx.input(c), selection, count, allocator)
          c += 1
        }
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        new ColumnarBatch(columns, count)
      }
    }
  }
}

/**
 * Columnar replacement for LocalTableScanExec (`VALUES`, small `createDataFrame`, most test data).
 * There is nothing to accelerate -- the rows are already on the driver -- so this only removes the
 * `RowToColumnarExec` transition above such a relation and lets small-table tests run our operators.
 * Off by default (`spark.vector.exec.localTableScan.enabled`), as Comet's equivalent is. The rows are
 * distributed as Spark does it (`min(max(rows, 1), leafNodeDefaultParallelism)` partitions) and each
 * partition becomes one on-heap batch through the row writer the collect-limit stage already uses.
 */
case class VectorLocalTableScanExec(output: Seq[Attribute], rows: Seq[InternalRow])
    extends LeafExecNode with VectorPlan {

  @transient private lazy val numParallelism: Int =
    math.min(math.max(rows.length, 1), session.leafNodeDefaultParallelism)

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    if (rows.isEmpty) {
      sparkContext.emptyRDD[ColumnarBatch]
    } else {
      val schema = StructType(output.map(a => StructField(a.name, a.dataType, a.nullable, a.metadata)))
      val m = vectorMetrics
      sparkContext.parallelize(rows, numParallelism).mapPartitionsInternal { iter =>
        val part = iter.toArray
        if (part.isEmpty) Iterator.empty
        else {
          m.numOutputBatches += 1
          m.numOutputRows += part.length
          Iterator.single(VectorRowStages.toBatch(schema, part))
        }
      }
    }
  }

  override protected def stringArgs: Iterator[Any] = {
    if (rows.isEmpty) Iterator("<empty>", output) else Iterator(output)
  }
}

object VectorSamplePlanner {
  def plan(s: SampleExec): Either[String, VectorSampleExec] =
    if (s.withReplacement) Left("sampling with replacement (Poisson) not supported")
    else Right(VectorSampleExec(s.lowerBound, s.upperBound, s.seed, s.child))

  def planLocalTableScan(l: LocalTableScanExec): Either[String, VectorLocalTableScanExec] =
    l.output.find(a => !TypeMapping.hasLane(a.dataType)) match {
      case Some(a) => Left(s"unsupported type ${a.dataType.simpleString} for local table column ${a.name}")
      case None if l.stream.isDefined => Left("streaming local table scan")
      case None => Right(VectorLocalTableScanExec(l.output, l.rows))
    }
}
