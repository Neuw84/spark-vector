package org.apache.spark.sql.vector

import java.util.concurrent.atomic.AtomicReference
import scala.concurrent.Future
import scala.jdk.CollectionConverters._

import org.apache.spark.{Dependency, FutureAction, MapOutputStatistics, Partition, Partitioner, RangePartitioner, ShuffleDependency, SparkEnv, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, BoundReference, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.exchange.{Exchange, ShuffleExchangeExec, ShuffleExchangeLike, ShuffleOrigin}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics, SQLShuffleReadMetricsReporter, SQLShuffleWriteMetricsReporter}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vector.shuffle.{VectorPartitioning, VectorShuffleDependency}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.MutablePair

/**
 * The columnar exchange (#288): a `ShuffleExchangeLike` over a `VectorPlan` child whose map side
 * writes Arrow IPC streams per reduce partition and whose reduce side reads them back as our column
 * vectors, so the operators above (`VectorHashAggregate` final, `VectorSortExec`, the merge join)
 * see columnar input without a `RowToColumnar`. AQE reads the map output statistics through the
 * real `ShuffleDependency` below and drives coalescing and skew handling through
 * `getShuffleRDD(partitionSpecs)`, exactly as with Spark's exchange.
 *
 * Range partitioning samples the child through Spark's `RangePartitioner` over a row projection of
 * the sort keys (the child runs once for the sample, as with Spark's and Comet's exchanges) and the
 * map task applies the partitioner row by row.
 */
case class VectorShuffleExchangeExec(
    override val outputPartitioning: Partitioning,
    child: SparkPlan,
    shuffleOrigin: ShuffleOrigin,
    advisoryPartitionSize: Option[Long] = None) extends Exchange with ShuffleExchangeLike with VectorPlan {

  override def nodeName: String = "VectorShuffleExchange"
  override def supportsColumnar: Boolean = true
  /**
   * Both outputs: AQE creates the stage from Spark's row exchange and applies the columnar rules with
   * `outputsColumnar = false`, so a shuffle that were columnar-only would be wrapped in a
   * `ColumnarToRowExec` and fail Spark's "cannot transform shuffle node" check; with the row output
   * declared the transition is left to the parent, which pulls columns or rows as it needs.
   */
  override def supportsRowBased: Boolean = true

  private lazy val writeMetrics = SQLShuffleWriteMetricsReporter.createShuffleWriteMetrics(sparkContext)
  private[sql] lazy val readMetrics = SQLShuffleReadMetricsReporter.createShuffleReadMetrics(sparkContext)
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "dataSize" -> SQLMetrics.createSizeMetric(sparkContext, "data size"),
    "numPartitions" -> SQLMetrics.createMetric(sparkContext, "partitions")) ++ readMetrics ++ writeMetrics

  @transient private lazy val inputRDD: RDD[ColumnarBatch] = child.executeColumnar()

  @transient override lazy val mapOutputStatisticsFuture: Future[MapOutputStatistics] =
    if (inputRDD.getNumPartitions == 0) Future.successful(null)
    else sparkContext.submitMapStage(shuffleDependency)

  override def numMappers: Int = shuffleDependency.rdd.getNumPartitions
  override def numPartitions: Int = shuffleDependency.partitioner.numPartitions
  override def shuffleId: Int = shuffleDependency.shuffleId

  override def getShuffleRDD(partitionSpecs: Array[ShufflePartitionSpec]): RDD[ColumnarBatch] =
    new ShuffledColumnarRDD(shuffleDependency, readMetrics, partitionSpecs)

  override def runtimeStatistics: Statistics = {
    val dataSize = metrics("dataSize").value
    val rowCount = metrics(SQLShuffleWriteMetricsReporter.SHUFFLE_RECORDS_WRITTEN).value
    Statistics(dataSize, Some(rowCount))
  }

  @transient lazy val shuffleDependency: VectorShuffleDependency = {
    val dep = VectorShuffleExchangeExec.prepareShuffleDependency(inputRDD, child.output, outputPartitioning, writeMetrics, metrics("dataSize"))
    metrics("numPartitions").set(dep.partitioner.numPartitions)
    val executionId = sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY)
    SQLMetrics.postDriverMetricUpdates(sparkContext, executionId, metrics("numPartitions") :: Nil)
    dep
  }

  private var cachedShuffleRDD: ShuffledColumnarRDD = _

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    if (cachedShuffleRDD == null) cachedShuffleRDD = new ShuffledColumnarRDD(shuffleDependency, readMetrics)
    cachedShuffleRDD
  }

  override protected def doExecute(): RDD[InternalRow] = ColumnarToRowExec(this).doExecute()

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = copy(child = newChild)
}

object VectorShuffleExchangeExec {

  /** The partitionings this exchange takes: hash over lane keys, round robin, single, range over lane keys. */
  def supports(partitioning: Partitioning, output: Seq[Attribute]): Boolean = {
    def laneKey(dt: org.apache.spark.sql.types.DataType) = io.sparkvector.spark.adapter.TypeMapping.hasLane(dt)
    output.forall(a => laneKey(a.dataType)) && (partitioning match {
      case h: HashPartitioning => h.expressions.forall(e => e.isInstanceOf[Attribute] && laneKey(e.dataType))
      case _: RoundRobinPartitioning => true
      case SinglePartition => true
      case r: RangePartitioning => r.ordering.forall(o => laneKey(o.dataType))
      case _ => false
    })
  }

  def prepareShuffleDependency(
      rdd: RDD[ColumnarBatch],
      output: Seq[Attribute],
      partitioning: Partitioning,
      writeMetrics: Map[String, SQLMetric],
      dataSize: SQLMetric): VectorShuffleDependency = {
    val schema = org.apache.spark.sql.catalyst.types.DataTypeUtils.fromAttributes(output)
    val spec: VectorPartitioning = partitioning match {
      case HashPartitioning(expressions, n) =>
        val ordinals = expressions.map { case a: Attribute => output.indexWhere(_.exprId == a.exprId) }.toArray
        require(ordinals.forall(_ >= 0), s"hash key not in the child's output: $expressions")
        VectorPartitioning.Hash(ordinals, ordinals.map(o => VectorPartitioning.keyKind(output(o).dataType)), n)
      case RoundRobinPartitioning(n) => VectorPartitioning.RoundRobin(n)
      case SinglePartition => VectorPartitioning.Single
      case RangePartitioning(sortingExpressions, numPartitions) =>
        // Spark's sampling (ShuffleExchangeExec.prepareShuffleDependency), over the batches' rows.
        val bound = sortingExpressions.map(_.child).map(e => org.apache.spark.sql.catalyst.expressions.BindReferences.bindReference(e, output))
        val rddForSampling = rdd.mapPartitionsInternal { batches =>
          val projection = UnsafeProjection.create(bound)
          val mutablePair = new MutablePair[InternalRow, Null]()
          batches.flatMap(b => b.rowIterator().asScala).map(row => mutablePair.update(projection(row).copy(), null))
        }
        val orderingAttributes = sortingExpressions.zipWithIndex.map { case (ord, i) =>
          ord.copy(child = BoundReference(i, ord.dataType, ord.nullable))
        }
        implicit val ordering: Ordering[InternalRow] = new LazilyGeneratedOrdering(orderingAttributes)
        val partitioner = new RangePartitioner(numPartitions, rddForSampling, ascending = true,
          samplePointsPerPartitionHint = SQLConf.get.rangeExchangeSampleSizePerPartition)
        VectorPartitioning.Range(partitioner, sortingExpressions.map(_.child), output)
      case other => throw new IllegalArgumentException(s"columnar shuffle over $other")
    }
    val partitioner = new Partitioner {
      override def numPartitions: Int = spec.numPartitions
      override def getPartition(key: Any): Int = 0 // the writer assigns rows, not Spark
    }
    val keyed: RDD[Product2[Int, ColumnarBatch]] = rdd.mapPartitionsInternal(_.map(b => (0, b)))
    new VectorShuffleDependency(keyed, partitioner, schema, spec, ShuffleExchangeExec.createShuffleWriteProcessor(writeMetrics), dataSize)
  }
}

/** One reduce task's input, as AQE describes it (the same partition kinds as `ShuffledRowRDD`). */
final class ShuffledColumnarRDDPartition(val index: Int, val spec: ShufflePartitionSpec) extends Partition

/**
 * The reduce side RDD: mirrors `ShuffledRowRDD` over our shuffle manager's reader, honouring the
 * coalesced, partial-reducer (skew) and partial/coalesced-mapper specs AQE produces.
 */
final class ShuffledColumnarRDD(
    dependency: ShuffleDependency[Int, ColumnarBatch, ColumnarBatch],
    metrics: Map[String, SQLMetric],
    partitionSpecs: Array[ShufflePartitionSpec]) extends RDD[ColumnarBatch](dependency.rdd.context, Nil) {

  def this(dependency: ShuffleDependency[Int, ColumnarBatch, ColumnarBatch], metrics: Map[String, SQLMetric]) =
    this(dependency, metrics, Array.tabulate(dependency.partitioner.numPartitions)(i => CoalescedPartitionSpec(i, i + 1)))

  override def getDependencies: Seq[Dependency[_]] = List(dependency)

  override val partitioner: Option[Partitioner] =
    if (partitionSpecs.forall(_.isInstanceOf[CoalescedPartitionSpec])) {
      val indices = partitionSpecs.map(_.asInstanceOf[CoalescedPartitionSpec].startReducerIndex)
      if (indices.toSet.size == partitionSpecs.length) Some(new CoalescedPartitioner(dependency.partitioner, indices)) else None
    } else None

  override def getPartitions: Array[Partition] =
    Array.tabulate[Partition](partitionSpecs.length)(i => new ShuffledColumnarRDDPartition(i, partitionSpecs(i)))

  override def getPreferredLocations(partition: Partition): Seq[String] = {
    val tracker = SparkEnv.get.mapOutputTracker.asInstanceOf[org.apache.spark.MapOutputTrackerMaster]
    partition.asInstanceOf[ShuffledColumnarRDDPartition].spec match {
      case CoalescedPartitionSpec(startReducerIndex, endReducerIndex, _) =>
        startReducerIndex.until(endReducerIndex).flatMap(r => tracker.getPreferredLocationsForShuffle(dependency, r))
      case PartialReducerPartitionSpec(_, startMapIndex, endMapIndex, _) =>
        tracker.getMapLocation(dependency, startMapIndex, endMapIndex)
      case PartialMapperPartitionSpec(mapIndex, _, _) =>
        tracker.getMapLocation(dependency, mapIndex, mapIndex + 1)
      case CoalescedMapperPartitionSpec(startMapIndex, endMapIndex, _) =>
        tracker.getMapLocation(dependency, startMapIndex, endMapIndex)
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val tempMetrics = context.taskMetrics().createTempShuffleReadMetrics()
    val sqlMetricsReporter = new SQLShuffleReadMetricsReporter(tempMetrics, metrics)
    val manager = SparkEnv.get.shuffleManager
    val reader = split.asInstanceOf[ShuffledColumnarRDDPartition].spec match {
      case CoalescedPartitionSpec(startReducerIndex, endReducerIndex, _) =>
        manager.getReader(dependency.shuffleHandle, startReducerIndex, endReducerIndex, context, sqlMetricsReporter)
      case PartialReducerPartitionSpec(reducerIndex, startMapIndex, endMapIndex, _) =>
        manager.getReader(dependency.shuffleHandle, startMapIndex, endMapIndex, reducerIndex, reducerIndex + 1, context, sqlMetricsReporter)
      case PartialMapperPartitionSpec(mapIndex, startReducerIndex, endReducerIndex) =>
        manager.getReader(dependency.shuffleHandle, mapIndex, mapIndex + 1, startReducerIndex, endReducerIndex, context, sqlMetricsReporter)
      case CoalescedMapperPartitionSpec(startMapIndex, endMapIndex, numReducers) =>
        manager.getReader(dependency.shuffleHandle, startMapIndex, endMapIndex, 0, numReducers, context, sqlMetricsReporter)
    }
    reader.read().asInstanceOf[Iterator[Product2[Int, ColumnarBatch]]].map(_._2)
  }

  override def clearDependencies(): Unit = super.clearDependencies()
}
