package io.sparkvector.spark

import io.sparkvector.spark.test.VectorQuerySuite
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.{
  And,
  Attribute,
  AttributeReference,
  EqualTo,
  Expression,
  GreaterThan,
  IsNotNull,
  LessThan,
  Literal,
  Multiply,
  Pmod,
  Add
}
import org.apache.spark.sql.catalyst.plans.logical.MergeRows
import org.apache.spark.sql.catalyst.plans.logical.MergeRows.{
  Copy,
  Delete,
  Discard,
  Insert,
  Instruction,
  Keep,
  Split,
  Update
}
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlan}
import org.apache.spark.sql.execution.datasources.v2.MergeRowsExec
import org.apache.spark.sql.types._
import org.apache.spark.sql.vector.{PlanUtils, VectorFilterExec, VectorMergeRowsExec, VectorMergeRowsPlanner}

/**
 * The columnar MergeRows (#21) against Spark's `MergeRowsExec`, both built by hand over the same
 * columnar scan of a "joined" table: Spark only plans the operator inside a row-level `MERGE INTO`,
 * whose join carries Iceberg's struct `_partition` column that our joins do not pass through yet,
 * so the operator itself is exercised here with every construct it has to reproduce -- the two
 * presence predicates and the three groups, clause order with conditions, `Keep` in every context,
 * `Discard`, `Split` (an update as delete + insert), rows no clause takes, projections mixing
 * literals, nulls, columns, computed values and a struct passed through, the cardinality check --
 * and the results compared row by row.
 */
class VectorMergeRowsSuite extends VectorQuerySuite {

  private val Rows = 20000

  /** A joined table: target columns null where the target row is absent, source columns likewise, markers, a row id, a struct. */
  private def joined(name: String, rowIdExpr: String): SparkPlan = {
    val path = newTempPath(s"merge/$name")
    spark.range(0, Rows).selectExpr(
      "if(id % 10 < 7, true, null) AS t_marker",
      "if(id % 10 >= 3, true, null) AS s_marker",
      "if(id % 10 < 7, cast(id as int), null) AS t_i",
      "if(id % 10 < 7, id * 2, null) AS t_v",
      "if(id % 10 < 7, concat('f', id % 5), null) AS t_file",
      "if(id % 10 >= 3, cast(id as int), null) AS s_i",
      "if(id % 10 >= 3, id * 3 - 500, null) AS s_v",
      s"if(id % 10 < 7, $rowIdExpr, null) AS __row_id",
      "named_struct('a', cast(id % 4 as int), 'b', concat('p', id % 3)) AS part"
    )
      .write.mode("overwrite").parquet(path)
    val df = spark.read.parquet(path)
    df.collect()
    PlanUtils.allNodes(
      finalPlan(df)
    ).find(_.supportsColumnar).getOrElse(fail("no columnar scan in " + finalPlan(df).treeString))
  }

  private def attr(plan: SparkPlan, name: String): AttributeReference =
    plan.output.find(_.name == name).get.asInstanceOf[AttributeReference]

  /** The merge's output: an operation code, the row's columns, the row id, the struct. */
  private val output: Seq[Attribute] = Seq(
    AttributeReference("op", IntegerType, nullable = false)(),
    AttributeReference("i", IntegerType)(),
    AttributeReference("v", LongType)(),
    AttributeReference("file", StringType)(),
    AttributeReference("rid", LongType)(),
    AttributeReference("part", StructType(Seq(StructField("a", IntegerType), StructField("b", StringType))))()
  )

  private def program(child: SparkPlan)
      : (Expression, Expression, Seq[Instruction], Seq[Instruction], Seq[Instruction]) = {
    val tM = attr(child, "t_marker"); val sM = attr(child, "s_marker")
    val tI = attr(child, "t_i"); val tV = attr(child, "t_v"); val tF = attr(child, "t_file")
    val sI = attr(child, "s_i"); val sV = attr(child, "s_v")
    val rid = attr(child, "__row_id"); val part = attr(child, "part")
    val nullLong = Literal(null, LongType)
    val nullStr = Literal(null, StringType)
    def target(op: Int) = Seq(Literal(op), tI, tV, tF, rid, part)
    def fromSource(op: Int, v: Expression) = Seq(Literal(op), sI, v, nullStr, nullLong, part)
    val matched = Seq(
      Discard(LessThan(sV, Literal(0L))), // a matched delete of the rows with a negative source value
      Split(
        EqualTo(Pmod(tI, Literal(3)), Literal(0)),
        target(1),
        fromSource(3, Add(sV, tV))
      ), // an update: delete + insert
      Keep(
        Update,
        EqualTo(Pmod(tI, Literal(3)), Literal(1)),
        Seq(Literal(2), tI, Multiply(sV, Literal(10L)), tF, rid, part)
      )
    )
    // t_i % 3 = 2 matches no clause and is dropped, like Spark drops it.
    val notMatched = Seq(
      Keep(Insert, EqualTo(Pmod(sI, Literal(2)), Literal(0)), fromSource(3, sV))
    )
    val notMatchedBySource = Seq(
      Keep(Delete, EqualTo(Pmod(tI, Literal(5)), Literal(0)), target(1)),
      Keep(Delete, GreaterThan(tV, Literal(39000L)), target(1)),
      Keep(Copy, Literal.TrueLiteral, target(4))
    ) // the unconditional copy Spark adds for group-based merges
    (IsNotNull(sM), IsNotNull(tM), matched, notMatched, notMatchedBySource)
  }

  private def collect(plan: SparkPlan): Seq[Row] =
    ColumnarToRowExec(plan).executeCollect().map(_.copy()).toSeq.map { r =>
      Row.fromSeq(output.indices.map(i =>
        r.get(i, output(i).dataType) match {
          case u: org.apache.spark.unsafe.types.UTF8String => u.toString
          case s: org.apache.spark.sql.catalyst.InternalRow => (s.getInt(0), s.getUTF8String(1).toString)
          case v => v
        }
      ))
    }

  private def sorted(rows: Seq[Row]): Seq[Row] = rows.sortBy(_.toString)

  test("our MergeRows and Spark's agree on every clause construct over the same columnar child (#21)") {
    val child = joined("plain", "id")
    val (s, t, m, nm, nbs) = program(child)
    val theirs = MergeRowsExec(s, t, m, nm, nbs, checkCardinality = true, output, ColumnarToRowExec(child))
    val ours = VectorMergeRowsExec(s, t, m, nm, nbs, checkCardinality = true, output, child)
    assert(VectorMergeRowsPlanner.reason(ours).isEmpty, VectorMergeRowsPlanner.reason(ours).getOrElse(""))
    theirs.metrics // Spark initialises them lazily, inside the task, where a raw plan has no session
    val expected = sorted(theirs.executeCollect().map(_.copy()).toSeq.map { r =>
      Row.fromSeq(output.indices.map(i =>
        r.get(i, output(i).dataType) match {
          case u: org.apache.spark.unsafe.types.UTF8String => u.toString
          case st: org.apache.spark.sql.catalyst.InternalRow => (st.getInt(0), st.getUTF8String(1).toString)
          case v => v
        }
      ))
    })
    val actual = sorted(collect(ours))
    assert(actual.length === expected.length, s"row counts: ours ${actual.length}, Spark ${expected.length}")
    assert(actual === expected)
    // Every operation code appears: 1 delete (discard rows never do), 2 update, 3 insert, 4 copy.
    assert(actual.map(_.getInt(0)).distinct.sorted === Seq(1, 2, 3, 4))
  }

  test("a selected child batch (a filter forwarding its selection) is merged over the live rows only (#21)") {
    val child = joined("selected", "id")
    val (s, t, m, nm, nbs) = program(child)
    val keep = GreaterThan(
      Pmod(attr(child, "s_i"), Literal(7)),
      Literal(2)
    ) // drops some source rows before the merge; nulls are false
    val filtered = VectorFilterExec(
      org.apache.spark.sql.catalyst.expressions.Or(keep, IsNotNull(attr(child, "t_marker"))),
      child,
      emitSelection = true
    )
    val ours = VectorMergeRowsExec(s, t, m, nm, nbs, checkCardinality = false, output, filtered)
    val theirs = MergeRowsExec(
      s,
      t,
      m,
      nm,
      nbs,
      checkCardinality = false,
      output,
      ColumnarToRowExec(VectorFilterExec(
        org.apache.spark.sql.catalyst.expressions.Or(keep, IsNotNull(attr(child, "t_marker"))),
        child
      ))
    )
    theirs.metrics // Spark initialises them lazily, inside the task, where a raw plan has no session
    val expected = sorted(theirs.executeCollect().map(_.copy()).toSeq.map { r =>
      Row.fromSeq(output.indices.map(i =>
        r.get(i, output(i).dataType) match {
          case u: org.apache.spark.unsafe.types.UTF8String => u.toString
          case st: org.apache.spark.sql.catalyst.InternalRow => (st.getInt(0), st.getUTF8String(1).toString)
          case v => v
        }
      ))
    })
    assert(sorted(collect(ours)) === expected)
  }

  test("the cardinality check raises Spark's MERGE_CARDINALITY_VIOLATION on a repeated target row id (#21)") {
    val child = joined("dup", "id % 100") // matched rows share row ids
    val (s, t, m, nm, nbs) = program(child)
    val ours = VectorMergeRowsExec(s, t, m, nm, nbs, checkCardinality = true, output, child)
    val e = intercept[Exception](collect(ours))
    val cause = Iterator.iterate[Throwable](e)(_.getCause).takeWhile(_ != null).collectFirst {
      case st: org.apache.spark.SparkThrowable => st
    }
    assert(cause.exists(_.getCondition == "MERGE_CARDINALITY_VIOLATION"), e.toString)
    // Without the check the same rows merge.
    val relaxed = VectorMergeRowsExec(s, t, m, nm, nbs, checkCardinality = false, output, child)
    assert(collect(relaxed).nonEmpty)
  }

  test("the planner refuses what it cannot compile, with the reason") {
    val child = joined("refuse", "id")
    val (s, t, m, nm, nbs) = program(child)
    val tI = attr(child, "t_i")
    val bad = Seq(Keep(
      Update,
      EqualTo(org.apache.spark.sql.catalyst.expressions.SoundEx(attr(child, "t_file")), Literal("F000")),
      Seq(Literal(2), tI, attr(child, "t_v"), attr(child, "t_file"), attr(child, "__row_id"), attr(child, "part"))
    ))
    val noRowId = VectorMergeRowsExec(
      s,
      t,
      m,
      nm,
      nbs,
      checkCardinality = true,
      output,
      org.apache.spark.sql.vector.VectorProjectExec(child.output.filterNot(_.name == MergeRows.ROW_ID), child)
    )
    assert(VectorMergeRowsPlanner.reason(noRowId).exists(_.contains("without a __row_id column")))
    val unsupported = VectorMergeRowsExec(s, t, bad, nm, nbs, checkCardinality = false, output, child)
    assert(VectorMergeRowsPlanner.reason(
      unsupported
    ).exists(_.contains("merge clause condition: unsupported expression SoundEx")))
  }
}
