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
package org.apache.spark.sql.vector.shuffle

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection
import org.apache.spark.sql.catalyst.plans.logical.RebalancePartitions
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
import org.apache.spark.sql.execution.exchange.ENSURE_REQUIREMENTS
import org.apache.spark.sql.types._
import org.apache.spark.sql.vector.VectorShuffleExchangeExec
import org.apache.spark.unsafe.types.UTF8String
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

/** #20: a rebalance's advisory partition size, put on our shuffle's scale. */
class RebalanceAdvisorySuite extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _
  private var tempDir: java.nio.file.Path = _
  private val Requested = 64L << 20

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[4]")
      .appName("RebalanceAdvisorySuite")
      .config("spark.ui.enabled", "false")
      .config("spark.driver.host", "localhost")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.plugins", "io.sparkvector.spark.VectorPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.sql.vector.shuffle.VectorShuffleManager")
      .config("spark.vector.shuffle.enabled", "true")
      .getOrCreate()
    tempDir = java.nio.file.Files.createTempDirectory("spark-vector-advisory")
    // The write exchange's shape in small: narrow key columns plus a mostly-null payload.
    spark.sql(
      "select cast(id % 5 as int) g, id pos, cast(id % 3 as int) op, " +
        "case when id % 4 = 0 then cast(id as string) end payload, cast(id as double) d from range(0, 20000)"
    ).write.parquet(tempDir.resolve("rows").toString)
    spark.read.parquet(tempDir.resolve("rows").toString).createOrReplaceTempView("rows")
    // Iceberg's _file in small: a long path repeated over many rows, which our shuffle dictionary-encodes.
    spark.sql(
      "select cast(id % 5 as int) g, id pos, " +
        "case when id % 3 = 0 then null else concat('s3://warehouse/db/table/data/00042-1-file-', cast(id % 7 as string), '.parquet') end path " +
        "from range(0, 20000)"
    ).write.parquet(tempDir.resolve("paths").toString)
    spark.read.parquet(tempDir.resolve("paths").toString).createOrReplaceTempView("paths")
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
    SparkSession.clearActiveSession()
    SparkSession.clearDefaultSession()
  }

  private def nodes(plan: SparkPlan): Seq[SparkPlan] = plan +: (plan match {
    case q: QueryStageExec => nodes(q.plan)
    case p => p.children.flatMap(nodes)
  })

  /** A rebalance on `g` asking for [[Requested]] bytes, as a data source's write asks; our exchange after running it. */
  private def rebalanced(table: String = "rows"): VectorShuffleExchangeExec = {
    val session = spark.asInstanceOf[org.apache.spark.sql.classic.SparkSession]
    val child = session.table(table).queryExecution.analyzed
    val g = child.output.find(_.name == "g").get
    val df = org.apache.spark.sql.classic.Dataset.ofRows(
      session,
      RebalancePartitions(Seq(g), child, None, Some(Requested))
    )
    assert(df.collect().length === 20000)
    val plan = df.queryExecution.executedPlan match {
      case a: AdaptiveSparkPlanExec => a.executedPlan
      case p => p
    }
    nodes(
      plan
    ).collectFirst { case e: VectorShuffleExchangeExec => e }.getOrElse(fail(s"no exchange of ours in\n$plan"))
  }

  test("a data source's rebalance size is scaled to our bytes per row once the map stage has written") {
    val ex = rebalanced()
    assert(ex.requestedAdvisoryPartitionSize === Some(Requested))
    val scaled = ex.advisoryPartitionSize.get
    assert(scaled < Requested, s"not scaled: $scaled")
    assert(scaled >= (Requested * RebalanceAdvisory.MinFactor).toLong, s"below the floor: $scaled")
    val rows = ex.metrics("shuffleRecordsWritten").value
    val bytes = ex.metrics("dataSize").value
    assert(rows === 20000L)
    assert(scaled === RebalanceAdvisory.scale(
      Requested,
      ex.output.map(_.dataType),
      rows,
      bytes,
      ex.shuffleDependency.stringBytes.map(_.value)
    ))
    // Read twice, same answer.
    assert(ex.advisoryPartitionSize.get === scaled)
  }

  test("the user's own Iceberg advisory size is kept as is") {
    spark.conf.set(VectorShuffleExchangeExec.IcebergAdvisorySizeKey, "64m")
    try assert(rebalanced().advisoryPartitionSize === Some(Requested))
    finally spark.conf.unset(VectorShuffleExchangeExec.IcebergAdvisorySizeKey)
  }

  test("an Iceberg session size other than the requested one does not stop the scaling") {
    spark.conf.set(VectorShuffleExchangeExec.IcebergAdvisorySizeKey, "128m")
    try assert(rebalanced().advisoryPartitionSize.get < Requested)
    finally spark.conf.unset(VectorShuffleExchangeExec.IcebergAdvisorySizeKey)
  }

  test("with the switch off the requested size is kept") {
    spark.conf.set(VectorShuffleExchangeExec.RebalanceAdvisoryScalingKey, "false")
    try assert(rebalanced().advisoryPartitionSize === Some(Requested))
    finally spark.conf.unset(VectorShuffleExchangeExec.RebalanceAdvisoryScalingKey)
  }

  test("measured string bytes put the estimate within 5% of Spark's rows, dictionary-encoded paths included") {
    val ex = rebalanced("paths")
    val rows = ex.metrics("shuffleRecordsWritten").value
    val strings = ex.shuffleDependency.stringBytes.getOrElse(fail("no string bytes measured")).value
    assert(rows === 20000L)
    assert(strings > 0L)
    val schema = ex.output.map(_.dataType)
    val project = UnsafeProjection.create(ex.output, ex.output)
    val toCatalyst = org.apache.spark.sql.catalyst.CatalystTypeConverters
      .createToCatalystConverter(spark.table("paths").schema)
    val real = spark.table("paths").collect().map { r =>
      project(toCatalyst(r).asInstanceOf[InternalRow]).getSizeInBytes.toLong
    }.sum.toDouble / rows
    val estimate = RebalanceAdvisory.unsafeRowBytesWithStrings(schema, strings.toDouble / rows)
    assert(math.abs(estimate - real) / real < 0.05, s"estimate $estimate vs real $real")
    // The size is scaled by that estimate, not by the one from our bytes alone.
    val bytes = ex.metrics("dataSize").value
    assert(ex.advisoryPartitionSize.get === RebalanceAdvisory.scale(Requested, schema, rows, bytes, Some(strings)))
  }

  test("the UnsafeRow estimate is within 10% of Spark's real row bytes") {
    val schema = StructType(Seq(
      StructField("i", IntegerType),
      StructField("l", LongType),
      StructField("s", StringType),
      StructField("n", StringType),
      StructField("st", StructType(Seq(StructField("a", IntegerType), StructField("b", StringType)))),
      StructField("w", DecimalType(38, 10))
    ))
    val project = UnsafeProjection.create(schema)
    val rnd = new scala.util.Random(20)
    var real = 0L
    var ours = 0.0
    val n = 5000
    (0 until n).foreach { k =>
      val s = rnd.alphanumeric.take(5 + rnd.nextInt(30)).mkString
      val b = rnd.alphanumeric.take(rnd.nextInt(12)).mkString
      val nul = k % 2 == 0
      val row = InternalRow(
        k,
        k.toLong,
        UTF8String.fromString(s),
        if (nul) null else UTF8String.fromString(s),
        InternalRow(k, UTF8String.fromString(b)),
        Decimal(BigDecimal(k) / 7, 38, 10)
      )
      real += project(row).getSizeInBytes
      // Our Arrow bytes: a validity bit per leaf and the struct (8 x 1/8), the lanes (4 + 8 + 4 + 4
      // + 4 + 4 + 16, strings as 4-byte offsets), plus the string data.
      ours += 1.0 + 44 + s.length + (if (nul) 0 else s.length) + b.length
    }
    val estimate = RebalanceAdvisory.unsafeRowBytes(schema.fields.map(_.dataType).toSeq, ours / n)
    val actual = real.toDouble / n
    assert(math.abs(estimate - actual) / actual < 0.10, s"estimate $estimate vs real $actual (ours ${ours / n})")
  }

  test("scale never raises the size, floors it, and leaves it alone without rows") {
    val schema = Seq(IntegerType, LongType)
    assert(RebalanceAdvisory.scale(1000L, schema, 0L, 0L) === 1000L)
    // Our 12.25 B/row against UnsafeRow's 8 + 16 = 24: about half.
    val half = RebalanceAdvisory.scale(1000L, schema, 1000L, 12250L)
    assert(half > 400L && half < 600L, half.toString)
    // More bytes per row than an UnsafeRow (e.g. a bloated estimate): never above the request.
    assert(RebalanceAdvisory.scale(1000L, schema, 10L, 1000000L) === 1000L)
    // An absurdly small ratio is floored.
    assert(RebalanceAdvisory.scale(1600L, Seq.fill(64)(LongType), 1000L, 1L) === 100L)
  }

  test("mapSizeFactor: Spark's estimated on-disk bytes per row over ours, bounded, 1.0 without data") {
    val schema = Seq(IntegerType, LongType)
    // UnsafeRow 24 B/row; / 2.5 = 9.6 against our 5 on disk.
    val f = RebalanceAdvisory.mapSizeFactor(schema, 1000L, 12250L, 5000L, 2.5)
    assert(math.abs(f - 9.6 / 5) < 1e-9, f.toString)
    // Compression <= 0: the uncompressed ratio, 24 / 12.25.
    val g = RebalanceAdvisory.mapSizeFactor(schema, 1000L, 12250L, 5000L, 0.0)
    assert(math.abs(g - 24.0 / 12.25) < 1e-9, g.toString)
    assert(RebalanceAdvisory.mapSizeFactor(schema, 0L, 12250L, 5000L, 2.5) === 1.0)
    assert(RebalanceAdvisory.mapSizeFactor(schema, 1000L, 12250L, 0L, 2.5) === 1.0)
    assert(RebalanceAdvisory.mapSizeFactor(schema, 1000L, 12250L, 1L, 2.5) === RebalanceAdvisory.MaxMapSizeFactor)
    assert(
      RebalanceAdvisory.mapSizeFactor(schema, 1000L, 12250L, 1000000000L, 2.5) === 1.0 / RebalanceAdvisory.MaxMapSizeFactor
    )
    assert(RebalanceAdvisory.scaleSizes(Array(0L, 10L, 1L), 2.5).toSeq === Seq(0L, 25L, 3L))
  }

  /** An aggregate over `rows` with [[VectorShuffleExchangeExec.MapSizeScalingKey]] at `on`: its exchange and rows. */
  private def aggregated(on: Boolean): (VectorShuffleExchangeExec, Array[org.apache.spark.sql.Row]) = {
    spark.conf.set(VectorShuffleExchangeExec.MapSizeScalingKey, on.toString)
    try {
      val df = spark.sql("select g, count(*) c, sum(d) s, max(payload) p from rows group by g order by g")
      val out = df.collect()
      val plan = df.queryExecution.executedPlan match {
        case a: AdaptiveSparkPlanExec => a.executedPlan
        case p => p
      }
      val ex = nodes(plan).collectFirst {
        case e: VectorShuffleExchangeExec if e.shuffleOrigin == ENSURE_REQUIREMENTS => e
      }.getOrElse(fail(s"no exchange of ours in\n$plan"))
      (ex, out)
    } finally spark.conf.unset(VectorShuffleExchangeExec.MapSizeScalingKey)
  }

  private def statsOf(ex: VectorShuffleExchangeExec): org.apache.spark.MapOutputStatistics =
    scala.concurrent.Await.result(ex.mapOutputStatisticsFuture, scala.concurrent.duration.Duration(60, "s"))

  private def realSizes(ex: VectorShuffleExchangeExec): Array[Long] =
    org.apache.spark.SparkEnv.get.mapOutputTracker.asInstanceOf[org.apache.spark.MapOutputTrackerMaster]
      .getStatistics(ex.shuffleDependency).bytesByPartitionId

  test("with map size scaling on, AQE reads an aggregate exchange's sizes times the factor; the results are the same") {
    val (off, rowsOff) = aggregated(on = false)
    assert(statsOf(off).bytesByPartitionId.toSeq === realSizes(off).toSeq)
    assert(off.shuffleDependency.stringBytes.isEmpty)

    val (on, rowsOn) = aggregated(on = true)
    assert(rowsOn.toSeq === rowsOff.toSeq)
    val real = realSizes(on)
    val strings = on.shuffleDependency.stringBytes.getOrElse(fail("no string bytes measured")).value
    val factor = RebalanceAdvisory.mapSizeFactor(
      on.output.map(_.dataType),
      on.metrics("shuffleRecordsWritten").value,
      on.metrics("dataSize").value,
      real.sum,
      VectorShuffleExchangeExec.DefaultSparkCompression,
      Some(strings)
    )
    assert(factor != 1.0, s"factor $factor")
    assert(statsOf(on).bytesByPartitionId.toSeq === RebalanceAdvisory.scaleSizes(real, factor).toSeq)
  }

  test("map size scaling leaves a rebalance's row-proportional sizes alone") {
    spark.conf.set(VectorShuffleExchangeExec.MapSizeScalingKey, "true")
    try {
      val ex = rebalanced()
      assert(!VectorShuffleExchangeExec.mapSizesScaled(ex.shuffleOrigin, spark.sessionState.conf))
    } finally spark.conf.unset(VectorShuffleExchangeExec.MapSizeScalingKey)
  }
}
