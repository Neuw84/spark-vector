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
package org.apache.spark.sql.vector.shuffle

import scala.jdk.CollectionConverters._

import org.apache.spark.util.AccumulatorV2

/**
 * Each map task's record count per reduce partition, keyed by the map's index (#20). The driver
 * keeps the last report of each map, so a retried or speculative attempt replaces its predecessor's
 * counts instead of adding to them.
 */
final class RecordsByPartitionAccumulator
    extends AccumulatorV2[(Int, Array[Long]), java.util.Map[Integer, Array[Long]]] {

  private val byMap = new java.util.HashMap[Integer, Array[Long]]()

  override def isZero: Boolean = byMap.synchronized(byMap.isEmpty)

  override def copy(): RecordsByPartitionAccumulator = {
    val c = new RecordsByPartitionAccumulator
    byMap.synchronized(c.byMap.putAll(byMap))
    c
  }

  override def reset(): Unit = byMap.synchronized(byMap.clear())

  override def add(v: (Int, Array[Long])): Unit = byMap.synchronized { byMap.put(v._1, v._2); () }

  override def merge(other: AccumulatorV2[(Int, Array[Long]), java.util.Map[Integer, Array[Long]]]): Unit =
    other match {
      case o: RecordsByPartitionAccumulator =>
        val theirs = o.byMap.synchronized(new java.util.HashMap[Integer, Array[Long]](o.byMap))
        byMap.synchronized(byMap.putAll(theirs))
      case _ => throw new UnsupportedOperationException(s"cannot merge ${other.getClass.getName}")
    }

  /** A snapshot of the counts reported so far. */
  override def value: java.util.Map[Integer, Array[Long]] =
    byMap.synchronized(new java.util.HashMap[Integer, Array[Long]](byMap))
}

object RowProportionalSizes {

  /**
   * The reduce partitions' sizes as AQE should weigh them for a rebalance (#20): each partition's
   * share of the exchange's bytes is its share of the rows, so the total -- and with it AQE's target
   * size -- is unchanged while the packing follows rows. Our columnar encoding makes bytes per row
   * vary several-fold between kinds of rows (dictionary-friendly rows shrink far more than wide
   * ones), and packing by those bytes put 9.9M rows into one write task where Spark's row bytes gave
   * 3.9M. An empty partition stays empty. `None` when a map's counts are missing or malformed, and
   * the caller keeps the real sizes.
   */
  def reweight(
      bytes: Array[Long],
      records: java.util.Map[Integer, Array[Long]],
      numMappers: Int
  ): Option[Array[Long]] = {
    val n = bytes.length
    val maps = records.asScala
    if (maps.size != numMappers || !(0 until numMappers).forall(m => maps.get(m).exists(_.length == n))) return None
    val rows = new Array[Long](n)
    maps.valuesIterator.foreach { counts =>
      var i = 0
      while (i < n) { rows(i) += counts(i); i += 1 }
    }
    val totalRows = rows.sum
    val totalBytes = bytes.sum
    if (totalRows <= 0 || totalBytes <= 0) return None
    val perRow = totalBytes.toDouble / totalRows
    Some(Array.tabulate(n)(i => if (rows(i) == 0) 0L else math.max(1L, math.round(rows(i) * perRow))))
  }
}
