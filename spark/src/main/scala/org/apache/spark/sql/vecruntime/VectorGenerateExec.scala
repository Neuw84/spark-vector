/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.vecruntime

import io.vecruntime.kernels.Bitmap
import io.vecruntime.spark.adapter.TypeMapping
import io.vecruntime.spark.arrow.{ArrowOutput, RemappedColumnVector}
import io.vecruntime.spark.expr.{EvalContext, ExpressionCompiler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Explode, Generator, PosExplode}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{GenerateExec, SparkPlan}
import org.apache.spark.sql.types.{
  ArrayType,
  BooleanType,
  DataType,
  DateType,
  DecimalType,
  DoubleType,
  IntegerType,
  LongType,
  MapType,
  StringType,
  TimestampType
}
import org.apache.spark.sql.vectorized.{ColumnarArray, ColumnarBatch, ColumnVector}

/**
 * Columnar replacement for GenerateExec with `explode` / `posexplode` (and their `_outer` forms) over
 * an array column. The array column reaches the operator as Spark's own vector (#19 passes columns
 * without a lane through), so exploding is a gather driven by the array lengths:
 *
 *  - a repeat index is built once per batch (row `i` repeated `len(i)` times; a null or empty array
 *    contributes one row under `outer`, none otherwise);
 *  - every forwarded column is gathered through it -- our lanes with `ArrowOutput.gather`, a column
 *    without a lane (another struct, array or map) as a [[RemappedColumnVector]] view;
 *  - the exploded column is copied once per array from the array vector's elements, the position
 *    column (`posexplode`) is an iota per array; both are null on the `outer` row of an empty or null
 *    array, as Spark's are.
 *
 * `inline`, `stack`, `json_tuple`, generators over maps and user-defined generators are refused with
 * a reason (`VectorGeneratePlanner`). The operator emits one batch per input batch, however many
 * rows the arrays expand to.
 */
case class VectorGenerateExec(
    generator: Generator,
    requiredChildOutput: Seq[Attribute],
    outer: Boolean,
    generatorOutput: Seq[Attribute],
    child: SparkPlan
) extends VectorExec {

  override def output: Seq[Attribute] = requiredChildOutput ++ generatorOutput
  override def outputPartitioning: Partitioning = child.outputPartitioning

  @transient private lazy val spec: GenerateSpec = VectorGeneratePlanner.spec(this) match {
    case Right(s) => s
    case Left(reason) => throw new IllegalStateException(s"cannot vectorize generate ${generator.sql}: $reason")
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val s = spec
    val required =
      requiredChildOutput.map(a => (a.name, a.dataType, child.output.indexWhere(_.exprId == a.exprId))).toArray
    val outs = generatorOutput.map(a => (a.name, a.dataType)).toArray
    val m = vectorMetrics
    child.executeColumnar().mapPartitionsInternal { iter => new VectorGenerateIterator(iter, s, required, outs, m) }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/** What the iterator needs: where the array column is, its element type, and the forms. */
final case class GenerateSpec(
    arrayOrdinal: Int,
    path: Seq[Int],
    elementType: DataType,
    position: Boolean,
    outer: Boolean
) extends Serializable

object VectorGeneratePlanner {
  def plan(g: GenerateExec): Either[String, SparkPlan] =
    spec(g.generator, g.child.output, g.generatorOutput, g.outer)
      .map(_ => VectorGenerateExec(g.generator, g.requiredChildOutput, g.outer, g.generatorOutput, g.child))

  private[vecruntime] def spec(v: VectorGenerateExec): Either[String, GenerateSpec] =
    spec(v.generator, v.child.output, v.generatorOutput, v.outer)

  private def spec(
      generator: Generator,
      input: Seq[Attribute],
      generatorOutput: Seq[Attribute],
      outer: Boolean
  ): Either[String, GenerateSpec] = {
    val (arrayExpr, position) = generator match {
      case Explode(c) => (c, false)
      case PosExplode(c) => (c, true)
      case other =>
        return Left(s"generator ${other.prettyName} not supported (explode and posexplode over an array are)")
    }
    val elementType = arrayExpr.dataType match {
      case ArrayType(et, _) => et
      case _: MapType => return Left(s"${generator.prettyName} over a map not supported (arrays are)")
      case other => return Left(s"${generator.prettyName} over ${other.simpleString} not supported")
    }
    if (!TypeMapping.isSupported(elementType))
      return Left(s"array element type ${elementType.simpleString} not supported in ${generator.prettyName}")
    if (generatorOutput.length != (if (position) 2 else 1))
      return Left(s"unexpected generator output ${generatorOutput.map(_.name).mkString(", ")}")
    ExpressionCompiler.nestedColumnPath(arrayExpr, input).left.map(r =>
      s"${generator.prettyName} over ${arrayExpr.sql}: $r (a column or a struct field of one is required)"
    )
      .map { case (ordinal, path) => GenerateSpec(ordinal, path, elementType, position, outer) }
  }
}

private[vecruntime] class VectorGenerateIterator(
    input: Iterator[ColumnarBatch],
    spec: GenerateSpec,
    required: Array[(String, DataType, Int)],
    outs: Array[(String, DataType)],
    metrics: VectorMetrics
) extends VectorBatchIterator(input, "VectorGenerateExec") {

  private val path = spec.path.toArray

  /** The array vector, and whether row `i` of it is null once its ancestors' nulls are counted. */
  private def arrayColumn(ctx: EvalContext): (ColumnVector, Int => Boolean) = {
    var cv = ctx.column(spec.arrayOrdinal)
    val ancestors = new scala.collection.mutable.ArrayBuffer[ColumnVector]
    var s = 0
    while (s < path.length) { if (cv.hasNull) ancestors += cv; cv = cv.getChild(path(s)); s += 1 }
    val leaf = cv
    val parents = ancestors.toArray
    (leaf, i => leaf.isNullAt(i) || parents.exists(_.isNullAt(i)))
  }

  private def element(arr: ColumnarArray, k: Int, dt: DataType): Any =
    if (arr.isNullAt(k)) null
    else dt match {
      case IntegerType | DateType => java.lang.Integer.valueOf(arr.getInt(k))
      case LongType | TimestampType => java.lang.Long.valueOf(arr.getLong(k))
      case DoubleType => java.lang.Double.valueOf(arr.getDouble(k))
      case BooleanType => java.lang.Boolean.valueOf(arr.getBoolean(k))
      case StringType => arr.getUTF8String(k)
      case d: DecimalType => java.lang.Long.valueOf(arr.getDecimal(k, d.precision, d.scale).toUnscaledLong)
      case other => throw new IllegalStateException(s"unexpected element type $other")
    }

  override protected def process(batch: ColumnarBatch): ColumnarBatch = metrics.timed {
    metrics.numInputBatches += 1
    withEvalContext(batch) { ctx =>
      val n = ctx.numRows
      val (arrays, isNull) = arrayColumn(ctx)
      val selection = if (ctx.hasSelection) ctx.selection else null
      // Pass 1: the output row count.
      var total = 0L
      var i = 0
      while (i < n) {
        if (selection == null || Bitmap.isSet(selection, i)) {
          val len = if (isNull(i)) 0 else arrays.getArray(i).numElements()
          total += (if (len == 0) (if (spec.outer) 1 else 0) else len)
        }
        i += 1
      }
      if (total == 0) null
      else {
        if (total > Int.MaxValue)
          throw new IllegalStateException(s"exploding one batch to $total rows exceeds the batch limit")
        val count = total.toInt
        // Pass 2: the repeat index and the element index (-1 on the outer row of an empty or null array).
        val rowIdx = new Array[Int](count)
        val elemIdx = new Array[Int](count)
        var o = 0
        i = 0
        while (i < n) {
          if (selection == null || Bitmap.isSet(selection, i)) {
            val len = if (isNull(i)) 0 else arrays.getArray(i).numElements()
            if (len == 0) { if (spec.outer) { rowIdx(o) = i; elemIdx(o) = -1; o += 1 } }
            else { var k = 0; while (k < len) { rowIdx(o) = i; elemIdx(o) = k; o += 1; k += 1 } }
          }
          i += 1
        }
        val columns = new Array[ColumnVector](required.length + outs.length)
        var c = 0
        while (c < required.length) {
          val (name, dt, ordinal) = required(c)
          columns(c) =
            if (TypeMapping.isSupported(dt))
              ArrowOutput.gather(name, dt, ctx.input(ordinal), rowIdx, 0, count, allocator)
            else RemappedColumnVector.of(ctx.column(ordinal), rowIdx)
          c += 1
        }
        // The elements, one array read per source row.
        var lastRow = -1
        var lastArr: ColumnarArray = null
        val (elemName, elemDt) = if (spec.position) outs(1) else outs(0)
        val elements = AggBufferColumns.values(
          elemName,
          elemDt,
          count,
          { r =>
            val k = elemIdx(r)
            if (k < 0) null
            else {
              val row = rowIdx(r)
              if (row != lastRow) { lastArr = arrays.getArray(row); lastRow = row }
              element(lastArr, k, spec.elementType)
            }
          },
          allocator
        )
        if (spec.position) {
          val (posName, posDt) = outs(0)
          columns(required.length) = AggBufferColumns.values(
            posName,
            posDt,
            count,
            { r =>
              val k = elemIdx(r); if (k < 0) null else java.lang.Integer.valueOf(k)
            },
            allocator
          )
          columns(required.length + 1) = elements
        } else columns(required.length) = elements
        metrics.numOutputBatches += 1
        metrics.numOutputRows += count
        new ColumnarBatch(columns, count)
      }
    }
  }
}
