package io.sparkvector.spark

import org.apache.spark.sql.internal.SQLConf

/** Configuration keys for the spark-vector plugin. All keys are read from the session's SQLConf. */
object VectorConf {

  val Enabled = "spark.vector.enabled"
  val FilterEnabled = "spark.vector.exec.filter.enabled"
  val ProjectEnabled = "spark.vector.exec.project.enabled"
  val AggregateEnabled = "spark.vector.exec.aggregate.enabled"
  val FinalAggregateEnabled = "spark.vector.exec.aggregate.final.enabled"
  val SelectionEnabled = "spark.vector.exec.selection.enabled"
  val CometShuffleEnabled = "spark.vector.comet.shuffle.enabled"
  val SortEnabled = "spark.vector.exec.sort.enabled"
  val TakeOrderedEnabled = "spark.vector.exec.takeOrdered.enabled"
  val LimitEnabled = "spark.vector.exec.limit.enabled"
  val UnionEnabled = "spark.vector.exec.union.enabled"
  val CoalesceEnabled = "spark.vector.exec.coalesce.enabled"
  val ExpandEnabled = "spark.vector.exec.expand.enabled"
  val SampleEnabled = "spark.vector.exec.sample.enabled"
  val WindowEnabled = "spark.vector.exec.window.enabled"
  val LocalTableScanEnabled = "spark.vector.exec.localTableScan.enabled"
  val BroadcastHashJoinEnabled = "spark.vector.exec.broadcastHashJoin.enabled"
  val BroadcastNestedLoopJoinEnabled = "spark.vector.exec.broadcastNestedLoopJoin.enabled"
  val ShuffledHashJoinEnabled = "spark.vector.exec.shuffledHashJoin.enabled"
  val SortMergeJoinEnabled = "spark.vector.exec.sortMergeJoin.enabled"
  val JoinMaxBuildSize = "spark.vector.join.maxBuildSize"
  val CometRangeShuffleEnabled = "spark.vector.comet.shuffle.range.enabled"
  val ExplainFallbackEnabled = "spark.vector.explainFallback.enabled"
  val UiEnabled = "spark.vector.ui.enabled"
  val UiRetainedExecutions = "spark.vector.ui.retainedExecutions"

  def isEnabled(conf: SQLConf): Boolean = bool(conf, Enabled, default = true)
  def filterEnabled(conf: SQLConf): Boolean = bool(conf, FilterEnabled, default = true)
  def projectEnabled(conf: SQLConf): Boolean = bool(conf, ProjectEnabled, default = true)
  def aggregateEnabled(conf: SQLConf): Boolean = bool(conf, AggregateEnabled, default = true)
  /** Convert Final-mode aggregates too (their input is a shuffle, converted to columnar by Spark). */
  def finalAggregateEnabled(conf: SQLConf): Boolean = bool(conf, FinalAggregateEnabled, default = true)
  /** Feed Comet's native shuffle from spark-vector operators when Comet's shuffle is configured. */
  def cometShuffleEnabled(conf: SQLConf): Boolean = bool(conf, CometShuffleEnabled, default = true)
  /** Pass selection bitmaps between spark-vector operators instead of compacting each batch. */
  def selectionEnabled(conf: SQLConf): Boolean = bool(conf, SelectionEnabled, default = true)
  /** Convert SortExec over a columnar child (in-memory, no spill). */
  def sortEnabled(conf: SQLConf): Boolean = bool(conf, SortEnabled, default = true)
  /** Convert TakeOrderedAndProjectExec (ORDER BY ... LIMIT) over a columnar child. */
  def takeOrderedEnabled(conf: SQLConf): Boolean = bool(conf, TakeOrderedEnabled, default = true)
  /** Convert LocalLimitExec / GlobalLimitExec / CollectLimitExec over a columnar child. */
  def limitEnabled(conf: SQLConf): Boolean = bool(conf, LimitEnabled, default = true)
  /** Convert UnionExec when at least one child is columnar (row children go through RowToColumnarExec). */
  def unionEnabled(conf: SQLConf): Boolean = bool(conf, UnionEnabled, default = true)
  /** Convert CoalesceExec over a columnar child. */
  def coalesceEnabled(conf: SQLConf): Boolean = bool(conf, CoalesceEnabled, default = true)
  /** Convert ExpandExec (grouping sets, the distinct rewrite) over a columnar child. */
  def expandEnabled(conf: SQLConf): Boolean = bool(conf, ExpandEnabled, default = true)
  /** Convert SampleExec without replacement over a columnar child (Spark's own Bernoulli sequence per partition). */
  def sampleEnabled(conf: SQLConf): Boolean = bool(conf, SampleEnabled, default = true)
  /** Convert WindowExec for the ranking functions (row_number, rank, dense_rank); the child may be a row sort. */
  def windowEnabled(conf: SQLConf): Boolean = bool(conf, WindowEnabled, default = true)
  /** Convert LocalTableScanExec (`VALUES`, small local relations) into one batch per partition; off by default. */
  def localTableScanEnabled(conf: SQLConf): Boolean = bool(conf, LocalTableScanEnabled, default = false)
  /** Convert BroadcastHashJoinExec over a columnar streamed side (the build side stays Spark's broadcast). */
  def broadcastHashJoinEnabled(conf: SQLConf): Boolean = bool(conf, BroadcastHashJoinEnabled, default = true)
  /** Convert BroadcastNestedLoopJoinExec (non-equi joins) when the streamed side is columnar. */
  def broadcastNestedLoopJoinEnabled(conf: SQLConf): Boolean = bool(conf, BroadcastNestedLoopJoinEnabled, default = true)
  /** Convert ShuffledHashJoinExec; over Spark's row shuffle both inputs go through RowToColumnarExec. */
  def shuffledHashJoinEnabled(conf: SQLConf): Boolean = bool(conf, ShuffledHashJoinEnabled, default = true)
  /**
   * Re-express SortMergeJoinExec as our shuffled hash join when the smaller side's statistics fit
   * `spark.vector.join.maxBuildSize` (#10). Opt-in: it trades Spark's streaming merge for a per-task
   * hash table, a different memory profile.
   */
  def sortMergeJoinEnabled(conf: SQLConf): Boolean = bool(conf, SortMergeJoinEnabled, default = false)
  /**
   * Largest build side (bytes, size strings like `512m` accepted) a hash-style join converts for (#86);
   * the joins hold the build side in memory per task. Default: a per-core share of the off-heap
   * budget (`spark.memory.offHeap.size / spark.executor.cores`) when off-heap is configured, else 1 GiB.
   */
  def joinMaxBuildSize(conf: SQLConf, sparkConf: org.apache.spark.SparkConf): Long = {
    val explicit = conf.getConfString(JoinMaxBuildSize, "")
    if (explicit.nonEmpty) org.apache.spark.network.util.JavaUtils.byteStringAsBytes(explicit)
    else {
      val offHeap = if (sparkConf.getBoolean("spark.memory.offHeap.enabled", false)) sparkConf.getSizeAsBytes("spark.memory.offHeap.size", "0") else 0L
      if (offHeap > 0) math.max(1L, offHeap / math.max(1, sparkConf.getInt("spark.executor.cores", 1))) else 1L << 30
    }
  }
  /** Also hand range-partitioned exchanges (global sorts) to Comet's native shuffle. */
  def cometRangeShuffleEnabled(conf: SQLConf): Boolean = bool(conf, CometRangeShuffleEnabled, default = true)
  def explainFallback(conf: SQLConf): Boolean = bool(conf, ExplainFallbackEnabled, default = false)

  private def bool(conf: SQLConf, key: String, default: Boolean): Boolean =
    conf.getConfString(key, default.toString).trim.equalsIgnoreCase("true")

  /**
   * The UI tab is attached from the driver plugin, before any session exists, so these two are
   * read from the [[org.apache.spark.SparkConf]] rather than a session's SQLConf.
   */
  def uiEnabled(get: String => Option[String]): Boolean =
    get(UiEnabled).forall(_.trim.equalsIgnoreCase("true"))

  def uiRetainedExecutions(get: String => Option[String]): Int =
    get(UiRetainedExecutions).flatMap(v => scala.util.Try(v.trim.toInt).toOption).filter(_ > 0).getOrElse(100)
}
