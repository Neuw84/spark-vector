/*
 * Copyright 2025-2026 Angel Conde and the spark-vector contributors
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
package org.apache.spark.sql.vector

import java.util.concurrent.atomic.AtomicReference
import scala.concurrent.Future
import scala.jdk.CollectionConverters._

import org.apache.spark.{
  Dependency,
  FutureAction,
  MapOutputStatistics,
  Partition,
  Partitioner,
  RangePartitioner,
  ShuffleDependency,
  SparkEnv,
  TaskContext
}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, BoundReference, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.exchange.{
  Exchange,
  REBALANCE_PARTITIONS_BY_COL,
  REBALANCE_PARTITIONS_BY_NONE,
  ShuffleExchangeExec,
  ShuffleExchangeLike,
  ShuffleOrigin
}
import org.apache.spark.sql.execution.metric.{
  SQLMetric,
  SQLMetrics,
  SQLShuffleReadMetricsReporter,
  SQLShuffleWriteMetricsReporter
}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vector.shuffle.{
  RebalanceAdvisory,
  RecordsByPartitionAccumulator,
  RowProportionalSizes,
  VectorPartitioning,
  VectorShuffleDependency
}
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
    /** The advisory size the plan asked for (Spark's exchange's); AQE reads [[advisoryPartitionSize]]. */
    requestedAdvisoryPartitionSize: Option[Long] = None,
    /**
     * Trailing columns of the child that exist only to hold computed hash keys (the columnar rule
     * materialises `rn + 1` and the like in a projection under the exchange): they partition the rows
     * and are not written, so the exchange's output is the child's without them.
     */
    materializedKeys: Int = 0
) extends Exchange with ShuffleExchangeLike with VectorPlan {

  override def output: Seq[Attribute] = child.output.dropRight(materializedKeys)

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
    "numPartitions" -> SQLMetrics.createMetric(sparkContext, "partitions")
  ) ++ readMetrics ++ writeMetrics

  @transient private lazy val inputRDD: RDD[ColumnarBatch] = child.executeColumnar()

  /**
   * Rebalance exchanges only (#20; `spark.vector.shuffle.rebalance.rowSizing`, default on): the map
   * tasks' per-partition record counts, from which AQE gets row-proportional partition sizes. Every
   * other origin -- the joins' and aggregates' exchanges -- keeps the real bytes.
   */
  @transient lazy val recordsByPartition: Option[RecordsByPartitionAccumulator] =
    if (VectorShuffleExchangeExec.rowSized(shuffleOrigin, conf)) {
      val acc = new RecordsByPartitionAccumulator
      sparkContext.register(acc)
      Some(acc)
    } else None

  @transient override lazy val mapOutputStatisticsFuture: Future[MapOutputStatistics] =
    if (inputRDD.getNumPartitions == 0) Future.successful(null)
    else recordsByPartition match {
      case Some(acc) => VectorShuffleExchangeExec.submitRowSizedMapStage(sparkContext, shuffleDependency, acc)
      case None => sparkContext.submitMapStage(shuffleDependency)
    }

  @transient private var scaledAdvisory: Option[Long] = None

  /**
   * The size AQE packs and splits this exchange's partitions to (#20). A rebalance's size that the
   * user did not set -- a data source's default, sized for Spark's shuffle -- is put on our scale
   * ([[RebalanceAdvisory.scale]]) once the map stage has written, which is when AQE's rules read it;
   * before that, and for every other exchange, it is the requested size.
   */
  override def advisoryPartitionSize: Option[Long] = requestedAdvisoryPartitionSize.map { size =>
    if (scaledAdvisory.isDefined) scaledAdvisory.get
    else if (!VectorShuffleExchangeExec.advisoryScaled(shuffleOrigin, size, conf)) size
    else {
      val rows = metrics(SQLShuffleWriteMetricsReporter.SHUFFLE_RECORDS_WRITTEN).value
      val bytes = metrics("dataSize").value
      if (rows <= 0) size
      else {
        val scaled = RebalanceAdvisory.scale(size, output.map(_.dataType), rows, bytes)
        logInfo(
          s"Rebalance exchange: advisory partition size $size -> $scaled for our shuffle " +
            f"(${bytes.toDouble / rows}%.1f bytes per row uncompressed over $rows rows)"
        )
        scaledAdvisory = Some(scaled)
        scaled
      }
    }
  }

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
    val dep = VectorShuffleExchangeExec.prepareShuffleDependency(
      inputRDD,
      child.output,
      output,
      VectorShuffleExchangeExec.keysAsColumns(outputPartitioning, child),
      writeMetrics,
      metrics("dataSize"),
      recordsByPartition
    )
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

  /** `false` turns the row-proportional sizing of rebalance exchanges off (#20). */
  val RebalanceRowSizingKey = "spark.vector.shuffle.rebalance.rowSizing"

  /** Only the origins whose partition sizes AQE uses to pack and split rebalanced output (#20). */
  def rowSized(origin: ShuffleOrigin, conf: SQLConf): Boolean = origin match {
    case REBALANCE_PARTITIONS_BY_COL | REBALANCE_PARTITIONS_BY_NONE =>
      conf.getConfString(RebalanceRowSizingKey, "true").toBoolean
    case _ => false
  }

  /** `false` keeps a rebalance's requested advisory size as is, on our shuffle's bytes (#20). */
  val RebalanceAdvisoryScalingKey = "spark.vector.shuffle.rebalance.advisoryScaling"

  /** Iceberg's session setting for the write's advisory size: a value the user chose. */
  val IcebergAdvisorySizeKey = "spark.sql.iceberg.advisory-partition-size"

  /**
   * Whether a rebalance's requested `size` is scaled to our shuffle (#20): the switch is on and the
   * size is not the user's own Iceberg session setting, which is taken as the size wanted in our
   * bytes. (A size set as an Iceberg write option or table property cannot be told apart from
   * Iceberg's default here and is scaled; turn the switch off to keep it exact.)
   */
  def advisoryScaled(origin: ShuffleOrigin, size: Long, conf: SQLConf): Boolean = origin match {
    case REBALANCE_PARTITIONS_BY_COL | REBALANCE_PARTITIONS_BY_NONE =>
      conf.getConfString(RebalanceAdvisoryScalingKey, "true").toBoolean &&
      !Option(conf.getConfString(IcebergAdvisorySizeKey, null)).exists(v =>
        scala.util.Try(org.apache.spark.network.util.JavaUtils.byteStringAsBytes(v.trim)).toOption.contains(size)
      )
    case _ => false
  }

  /**
   * `SparkContext.submitMapStage`, with the statistics' sizes made row-proportional
   * ([[RowProportionalSizes.reweight]]) once the stage has finished -- the DAG scheduler applies a
   * task's accumulator updates before it completes the map-stage job, so every map's counts are in by
   * then. Still a `SimpleFutureAction` over the job's waiter, so AQE can cancel the stage as before;
   * the map output tracker and the reducers' fetch sizes keep the real bytes.
   */
  private[vector] def submitRowSizedMapStage(
      sc: org.apache.spark.SparkContext,
      dependency: VectorShuffleDependency,
      records: RecordsByPartitionAccumulator
  ): FutureAction[MapOutputStatistics] = {
    sc.assertNotStopped()
    var result: MapOutputStatistics = null
    val waiter = sc.dagScheduler.submitMapStage(
      dependency,
      (r: MapOutputStatistics) => { result = r },
      sc.getCallSite(),
      sc.localProperties.get
    )
    lazy val sized: MapOutputStatistics =
      if (result == null) null
      else RowProportionalSizes.reweight(result.bytesByPartitionId, records.value, dependency.rdd.getNumPartitions)
        .fold(result)(new MapOutputStatistics(result.shuffleId, _))
    new org.apache.spark.SimpleFutureAction[MapOutputStatistics](waiter, sized)
  }

  /**
   * The partitionings this exchange takes: hash over lane keys (a struct of lanes hashes its leaves),
   * round robin, single, range over lane keys. Columns are lanes or structs of lanes (flattened).
   */
  def supports(partitioning: Partitioning, output: Seq[Attribute]): Boolean = {
    def laneKey(dt: org.apache.spark.sql.types.DataType) = io.sparkvector.spark.adapter.TypeMapping.hasLane(dt)
    def column(dt: org.apache.spark.sql.types.DataType) = io.sparkvector.shuffle.StructFlattening.supported(dt)
    output.forall(a => column(a.dataType)) && (partitioning match {
      case h: HashPartitioning => h.expressions.forall(e => e.isInstanceOf[Attribute] && column(e.dataType))
      case _: RoundRobinPartitioning => true
      case SinglePartition => true
      case r: RangePartitioning => r.ordering.forall(o => laneKey(o.dataType))
      case _ => false
    })
  }

  /**
   * The partitioning with every computed hash key replaced by the column that holds its value: the
   * columnar rule materialises such keys in a [[VectorProjectExec]] directly under the exchange while
   * the exchange keeps declaring the original expressions (so the requirements it satisfies do not
   * change). A key that is neither a column nor an alias of the projection is an error here, since
   * the rule only takes the exchange when every key has one.
   */
  def keysAsColumns(partitioning: Partitioning, child: SparkPlan): Partitioning = partitioning match {
    case h: HashPartitioning if h.expressions.exists(!_.isInstanceOf[Attribute]) =>
      val aliases = child match {
        case p: VectorProjectExec => p.projectList.collect { case a: org.apache.spark.sql.catalyst.expressions.Alias =>
            a
          }
        case _ => Nil
      }
      h.copy(expressions = h.expressions.map {
        case a: Attribute => a
        case e => aliases.find(_.child.semanticEquals(e)).map(_.toAttribute)
            .getOrElse(throw new IllegalStateException(s"hash key $e is not a column of the exchange's input"))
      })
    case other => other
  }

  def prepareShuffleDependency(
      rdd: RDD[ColumnarBatch],
      output: Seq[Attribute],
      written: Seq[Attribute],
      partitioning: Partitioning,
      writeMetrics: Map[String, SQLMetric],
      dataSize: SQLMetric,
      recordsByPartition: Option[RecordsByPartitionAccumulator] = None
  ): VectorShuffleDependency = {
    // `output` is the child's (keys are resolved against it); `written` is what the shuffle carries --
    // the same, less the trailing materialised key columns.
    val schema = org.apache.spark.sql.catalyst.types.DataTypeUtils.fromAttributes(written)
    // Structs cross as lanes (StructFlattening): the streams carry the flat schema, and a hash key on
    // a struct hashes its leaves. Ordinals past the written columns are materialised keys, which the
    // writer appends after the flat lanes.
    val layout = io.sparkvector.shuffle.StructFlattening.plan(schema)
    val flatSchema = layout.fold(schema)(_.flatSchema)
    def flatOrdinals(o: Int): Seq[Int] =
      if (o >= written.length) Seq(flatSchema.fields.length + (o - written.length))
      else layout.fold(Seq(o))(_.hashOrdinals(o))
    def flatType(f: Int): org.apache.spark.sql.types.DataType =
      if (f < flatSchema.fields.length) flatSchema.fields(f).dataType
      else output(written.length + (f - flatSchema.fields.length)).dataType
    val spec: VectorPartitioning = partitioning match {
      case HashPartitioning(expressions, n) =>
        val ordinals = expressions.map { case a: Attribute => output.indexWhere(_.exprId == a.exprId) }.toArray
        require(ordinals.forall(_ >= 0), s"hash key not in the child's output: $expressions")
        val flat = ordinals.flatMap(flatOrdinals)
        VectorPartitioning.Hash(flat, flat.map(f => VectorPartitioning.keyKind(flatType(f))), n)
      case RoundRobinPartitioning(n) => VectorPartitioning.RoundRobin(n)
      case SinglePartition => VectorPartitioning.Single
      case RangePartitioning(sortingExpressions, numPartitions) =>
        // Spark's sampling (ShuffleExchangeExec.prepareShuffleDependency), over the batches' rows.
        val bound = sortingExpressions.map(_.child).map(e =>
          org.apache.spark.sql.catalyst.expressions.BindReferences.bindReference(e, output)
        )
        val rddForSampling = rdd.mapPartitionsInternal { batches =>
          val projection = UnsafeProjection.create(bound)
          val mutablePair = new MutablePair[InternalRow, Null]()
          batches.flatMap(b => b.rowIterator().asScala).map(row => mutablePair.update(projection(row).copy(), null))
        }
        val orderingAttributes = sortingExpressions.zipWithIndex.map { case (ord, i) =>
          ord.copy(child = BoundReference(i, ord.dataType, ord.nullable))
        }
        implicit val ordering: Ordering[InternalRow] = new LazilyGeneratedOrdering(orderingAttributes)
        val partitioner = new RangePartitioner(
          numPartitions,
          rddForSampling,
          ascending = true,
          samplePointsPerPartitionHint = SQLConf.get.rangeExchangeSampleSizePerPartition
        )
        VectorPartitioning.Range(partitioner, sortingExpressions.map(_.child), output)
      case other => throw new IllegalArgumentException(s"columnar shuffle over $other")
    }
    val partitioner = new Partitioner {
      override def numPartitions: Int = spec.numPartitions
      override def getPartition(key: Any): Int = 0 // the writer assigns rows, not Spark
    }
    val keyed: RDD[Product2[Int, ColumnarBatch]] = rdd.mapPartitionsInternal(_.map(b => (0, b)))
    new VectorShuffleDependency(
      keyed,
      partitioner,
      flatSchema,
      spec,
      ShuffleExchangeExec.createShuffleWriteProcessor(writeMetrics),
      dataSize,
      layout,
      recordsByPartition
    )
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
    partitionSpecs: Array[ShufflePartitionSpec]
) extends RDD[ColumnarBatch](dependency.rdd.context, Nil) {

  def this(dependency: ShuffleDependency[Int, ColumnarBatch, ColumnarBatch], metrics: Map[String, SQLMetric]) =
    this(
      dependency,
      metrics,
      Array.tabulate(dependency.partitioner.numPartitions)(i => CoalescedPartitionSpec(i, i + 1))
    )

  override def getDependencies: Seq[Dependency[_]] = List(dependency)

  override val partitioner: Option[Partitioner] =
    if (partitionSpecs.forall(_.isInstanceOf[CoalescedPartitionSpec])) {
      val indices = partitionSpecs.map(_.asInstanceOf[CoalescedPartitionSpec].startReducerIndex)
      if (indices.toSet.size == partitionSpecs.length) Some(new CoalescedPartitioner(dependency.partitioner, indices))
      else None
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
        manager.getReader(
          dependency.shuffleHandle,
          startMapIndex,
          endMapIndex,
          reducerIndex,
          reducerIndex + 1,
          context,
          sqlMetricsReporter
        )
      case PartialMapperPartitionSpec(mapIndex, startReducerIndex, endReducerIndex) =>
        manager.getReader(
          dependency.shuffleHandle,
          mapIndex,
          mapIndex + 1,
          startReducerIndex,
          endReducerIndex,
          context,
          sqlMetricsReporter
        )
      case CoalescedMapperPartitionSpec(startMapIndex, endMapIndex, numReducers) =>
        manager.getReader(
          dependency.shuffleHandle,
          startMapIndex,
          endMapIndex,
          0,
          numReducers,
          context,
          sqlMetricsReporter
        )
    }
    reader.read().asInstanceOf[Iterator[Product2[Int, ColumnarBatch]]].map(_._2)
  }

  override def clearDependencies(): Unit = super.clearDependencies()
}
