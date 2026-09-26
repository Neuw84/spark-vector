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

import io.vecruntime.spark.adapter.TypeMapping
import org.apache.spark.rdd.{RDD, SQLPartitioningAwareUnionRDD}
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
 *
 * Partitioning is Spark's contract too, and it is not optional: when every child is hash-partitioned
 * the same way (`spark.sql.unionOutputPartitioning`, on by default), Spark's `UnionExec` reports that
 * partitioning and `EnsureRequirements` -- which runs before this operator replaces it -- omits the
 * shuffle an aggregate or join above the union would otherwise need. The union must then keep the
 * i-th partitions of its children together (`SQLPartitioningAwareUnionRDD`), as Spark's does; a plain
 * concatenation would put one key in as many partitions as there are children, and a Final aggregate
 * above it would emit that key once per child (TPC-DS q33, q56, q60 -- #128).
 */
case class VectorUnionExec(children: Seq[SparkPlan]) extends VectorPassThrough {

  override def output: Seq[Attribute] =
    children.map(_.output).transpose.map { attrs =>
      val first = attrs.head
      first.withNullability(attrs.exists(_.nullable))
    }

  /** Exactly what Spark's operator would report for these children. */
  override def outputPartitioning: Partitioning = UnionExec(children).outputPartitioning

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    // Read once, before the children execute, so the partitioning reported and the one built agree.
    val partitioning = outputPartitioning
    val rdds = children.map(_.executeColumnar())
    partitioning match {
      case _: UnknownPartitioning => sparkContext.union(rdds)
      case known =>
        new SQLPartitioningAwareUnionRDD(sparkContext, rdds.filter(_.partitions.nonEmpty), known.numPartitions)
    }
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(children = newChildren)

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
  /**
   * The union forwards its children's batches untouched, so it is planned whatever their column types:
   * a type it does not know is a type it never reads. Refusing on types was worse than useless -- a
   * union left to Spark over columnar children runs Spark 4.1.3's columnar `UnionExec`, which
   * concatenates co-partitioned children (the #128 bug upstream), and TPC-DS q66 returned every row
   * twice once the wide decimal sum (#87) made the channel aggregates columnar with a `decimal(28,2)`
   * result the union refused.
   */
  def planUnion(u: UnionExec): Either[String, VectorUnionExec] = {
    if (u.children.size < 2) Left("union with fewer than two children")
    else if (!u.children.exists(_.supportsColumnar)) Left("no columnar child")
    else Right(VectorUnionExec(u.children))
  }

  def planCoalesce(c: CoalesceExec): Either[String, VectorCoalesceExec] =
    if (c.numPartitions < 1) Left(s"coalesce to ${c.numPartitions} partitions")
    else Right(VectorCoalesceExec(c.numPartitions, c.child))
}
