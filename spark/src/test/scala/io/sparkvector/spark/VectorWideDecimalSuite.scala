package io.sparkvector.spark

import io.sparkvector.spark.adapter.SparkColumnVectorBuffers
import io.sparkvector.spark.test.VectorQuerySuite
import org.apache.spark.sql.vector.{VectorFilterExec, VectorProjectExec}

/**
 * Wide decimals (p > 18) as a DECIMAL128 lane (#257, the storage half of #28): a column read from
 * Parquet -- plain or dictionary encoded, Spark's big-endian byte strings -- becomes two little-endian
 * limbs, is compacted by our filter and forwarded by our projection as an Arrow Decimal128 vector,
 * and reads back through Spark's `getDecimal` exactly. No kernel computes on the lane yet (#258), so
 * an expression over a wide column still falls back with its type reason.
 */
class VectorWideDecimalSuite extends VectorQuerySuite {
  private val Filter = classOf[VectorFilterExec]
  private val Project = classOf[VectorProjectExec]

  private val wideExprs = Seq(
    "cast(id as int) as i",
    // decimal(38,10): the extremes of the type, values straddling the 64-bit limb boundary, nulls.
    "case when id % 101 = 1 then cast('9999999999999999999999999999.9999999999' as decimal(38,10)) " +
      "     when id % 101 = 2 then cast('-9999999999999999999999999999.9999999999' as decimal(38,10)) " +
      "     when id % 101 = 3 then cast('922337203.6854775807' as decimal(38,10)) " +
      "     when id % 101 = 4 then cast('-922337203.6854775808' as decimal(38,10)) " +
      "     when id % 101 = 5 then cast('922337203.6854775808' as decimal(38,10)) " +
      "     when id % 101 = 6 then cast(0 as decimal(38,10)) " +
      "     when id % 13 = 0 then null " +
      "     else cast(cast(id as decimal(38,10)) * cast('1234567890123.0000000001' as decimal(38,10)) as decimal(38,10)) end as w38",
    // decimal(27,2): TPC-DS's running-total shape, negative and small values, few distinct values (dictionary).
    "case when id % 17 = 0 then null else cast(cast((id % 40) - 20 as decimal(27,2)) * cast('1000000000000000.25' as decimal(27,2)) as decimal(27,2)) end as w27",
    // decimal(20,0): fits 9 bytes, the shortest wide form Parquet emits.
    "cast(cast(id as decimal(20,0)) * 10000000000 as decimal(20,0)) as w20")

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    val df = spark.range(0, 20000).selectExpr(wideExprs: _*)
    df.repartition(3).write.mode("overwrite").option("parquet.enable.dictionary", "true").parquet(newTempPath("wide/dict"))
    df.repartition(3).write.mode("overwrite").option("parquet.enable.dictionary", "false").parquet(newTempPath("wide/plain"))
    spark.read.parquet(newTempPath("wide/dict")).createOrReplaceTempView("tw_dict")
    spark.read.parquet(newTempPath("wide/plain")).createOrReplaceTempView("tw_plain")
  }

  private def lanesAdaptedBy[T](f: => T): Long = {
    val before = SparkColumnVectorBuffers.wideDecimalColumns()
    f
    SparkColumnVectorBuffers.wideDecimalColumns() - before
  }

  Seq("tw_dict", "tw_plain").foreach { t =>
    test(s"$t: a filter compacts wide decimal columns as DECIMAL128 lanes and the rows read back exactly") {
      val adapted = lanesAdaptedBy {
        checkVectorized(s"SELECT i, w38, w27, w20 FROM $t WHERE i % 3 = 0 AND i > 10", Seq(Filter))
      }
      // Three wide columns per batch, several batches: the lane path ran (a RemappedColumnVector never adapts).
      assert(adapted >= 3, s"expected the wide columns to be adapted into lanes, saw $adapted")
      // Every extreme survives the round trip.
      checkVectorized(s"SELECT w38 FROM $t WHERE i % 101 IN (1, 2, 3, 4, 5, 6) AND i < 2000", Seq(Filter))
      // A sparse selection (one survivor per word) and a dense one exercise both compaction paths.
      checkVectorized(s"SELECT w38, w27 FROM $t WHERE i % 64 = 5", Seq(Filter))
      checkVectorized(s"SELECT w38, w20 FROM $t WHERE i % 64 <> 5", Seq(Filter))
    }

    test(s"$t: a projection forwards wide decimal columns beside computed ones, compacting them under a selection") {
      val adapted = lanesAdaptedBy {
        checkVectorized(s"SELECT w38, i + 1 AS j, w27, w20 FROM $t WHERE i % 5 = 0", Seq(Filter, Project))
      }
      assert(adapted >= 3, s"expected the wide columns to be adapted into lanes, saw $adapted")
      // Without a selection the column is borrowed as is (no adaptation, no copy).
      checkVectorized(s"SELECT w38, i * 2 AS k FROM $t", Seq(Project))
    }
  }

  test("expressions over a wide decimal still fall back with the type reason until #258") {
    checkFallback("SELECT w38 + 1 AS x FROM tw_dict WHERE i % 101 > 6", Seq(Project), "exceeds 18 digits")
    checkFallback("SELECT i FROM tw_dict WHERE w27 > 0", Seq(Filter), "decimal(27,2)")
    checkFallback("SELECT cast(w20 AS double) AS d FROM tw_plain", Seq(Project), "decimal(20,0)")
    checkFallback("SELECT abs(w38) AS a FROM tw_plain WHERE i % 101 > 6", Seq(Project), "decimal(38,10)")
  }
}
