package org.apache.spark.sql.vector

import io.sparkvector.spark.adapter.TypeMapping
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.physical.{Partitioning, SinglePartition, UnknownPartitioning}
import org.apache.spark.sql.execution.{CoalesceExec, SparkPlan, UnionExec}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Columnar replacement for UnionExec: the children's batch RDDs, one after the other, as Spark's own
 * `UnionExec.doExecuteColumnar` does when every child is columnar. The difference is planning: this
 * operator is columnar whatever the children are, so a union with one row child (a `VALUES`
 * relation, a row shuffle) keeps the columnar chain and Spark's transitions convert that child
 * through `RowToColumnarExec`, where Spark's union would have dropped every child to rows. The rule
 * asks for at least one columnar child, otherwise there is nothing to keep. Output nullability is
 * merged across children exactly as Spark's operator does.
 */
case class VectorUnionExec(children: Seq[SparkPlan]) extends VectorPassThrough {

  override def output: Seq[Attribute] =
    children.map(_.output).transpose.map { attrs =>
      val first = attrs.head
      first.withNullability(attrs.exists(_.nullable))
    }

  override def outputPartitioning: Partitioning =
    UnknownPartitioning(children.map(_.outputPartitioning.numPartitions).sum)

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] =
    sparkContext.union(children.map(_.executeColumnar()))

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan = copy(children = newChildren)

  override def simpleString(maxFields: Int): String = s"VectorUnion(children=${children.length})"
}

/**
 * Columnar replacement for CoalesceExec: the child's batches coalesced into `numPartitions` without a
 * shuffle. Within an output partition the child's partition iterators are drained one after the
 * other; every child iterator releases its memory from its task-completion listener, so the last
 * batch of one child partition lives until the task ends rather than until the next batch is
 * requested -- the same lifecycle Spark's row `CoalesceExec` gives its children.
 */
case class VectorCoalesceExec(numPartitions: Int, child: SparkPlan) extends VectorExec with VectorPassThrough {

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning =
    if (numPartitions == 1) SinglePartition else UnknownPartitioning(numPartitions)

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val rdd = child.executeColumnar()
    if (numPartitions == 1 && rdd.getNumPartitions < 1) {
      // As Spark's CoalesceExec: an empty child still yields one (empty) partition.
      sparkContext.parallelize(Seq.empty[ColumnarBatch], 1)
    } else {
      rdd.coalesce(numPartitions, shuffle = false)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

/** Planning-time checks for the structural operators, shared by the rule. */
object VectorStructuralPlanner {

  /**
   * A union converts when at least one child is columnar and every child's output types are
   * supported lanes (a row child is converted by Spark's `RowToColumnarExec`, whose batches must
   * be adaptable).
   */
  def planUnion(u: UnionExec): Either[String, VectorUnionExec] = {
    val typeFailures = u.children.flatMap(c => c.output.find(a => !TypeMapping.isSupported(a.dataType)).map(a => s"unsupported column type ${a.dataType.simpleString} for ${a.name}"))
    if (u.children.size < 2) Left("union with fewer than two children")
    else if (!u.children.exists(_.supportsColumnar)) Left("no columnar child")
    else if (typeFailures.nonEmpty) Left(typeFailures.distinct.mkString("; "))
    else Right(VectorUnionExec(u.children))
  }

  def planCoalesce(c: CoalesceExec): Either[String, VectorCoalesceExec] =
    if (c.numPartitions < 1) Left(s"coalesce to ${c.numPartitions} partitions") else Right(VectorCoalesceExec(c.numPartitions, c.child))
}
