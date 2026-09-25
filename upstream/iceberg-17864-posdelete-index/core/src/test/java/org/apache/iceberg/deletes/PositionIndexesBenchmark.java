/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.deletes;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.apache.iceberg.MetadataColumns;
import org.apache.iceberg.Schema;
import org.apache.iceberg.StructLike;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.util.CharSequenceMap;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Building one delete file's per-data-file indexes: Iceberg 1.11.0's loop (copied verbatim, since
 * the patched Deletes replaces it on this classpath) against the patched Deletes.toPositionIndexes.
 * 64 data files x 30,000 deletes, 144-character paths, a fresh String per row as the Parquet reader
 * produces them. {@code sorted} is the spec's (file_path, pos) order; {@code interleaved} is the
 * worst case, every row's path differing from the previous row's. Time per delete row.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(
    value = 3,
    jvmArgsAppend = {"-Xmx4g"})
@State(Scope.Benchmark)
@OperationsPerInvocation(PositionIndexesBenchmark.ROWS)
public class PositionIndexesBenchmark {

  static final int FILES = 64;
  static final int PER_FILE = 30_000;
  static final int ROWS = FILES * PER_FILE;

  @Param({"sorted", "interleaved"})
  String order;

  List<Record> rows;

  /** Iceberg 1.11.0's Deletes.toPositionIndexes loop. */
  static CharSequenceMap<PositionDeleteIndex> stock(
      CloseableIterable<? extends StructLike> deletes) {
    CharSequenceMap<PositionDeleteIndex> indexes = CharSequenceMap.create();
    for (StructLike delete : deletes) {
      CharSequence filePath = delete.get(0, CharSequence.class);
      long position = delete.get(1, Long.class);
      PositionDeleteIndex index =
          indexes.computeIfAbsent(filePath, key -> new BitmapPositionDeleteIndex());
      index.delete(position);
    }
    return indexes;
  }

  @Setup(Level.Trial)
  public void setup() {
    Schema schema = new Schema(MetadataColumns.DELETE_FILE_PATH, MetadataColumns.DELETE_FILE_POS);
    String prefix =
        "s3a://warehouse-bucket-name/warehouse/v2.db/mix_20_5/data/ss_sold_date_sk_bucket=12/";
    List<Record> sorted = new ArrayList<>(ROWS);
    for (int f = 0; f < FILES; f++) {
      String path =
          prefix
              + String.format("%05d-%d-4f1c2e9a-8d3b-4c7e-9f21-%012d-0-00001.parquet", f, f * 7, f);
      for (int p = 0; p < PER_FILE; p++) {
        GenericRecord r = GenericRecord.create(schema);
        r.set(0, new String(path.toCharArray()));
        r.set(1, (long) p * 3);
        sorted.add(r);
      }
    }
    if (order.equals("sorted")) {
      rows = sorted;
    } else {
      rows = new ArrayList<>(ROWS);
      for (int p = 0; p < PER_FILE; p++) {
        for (int f = 0; f < FILES; f++) {
          rows.add(sorted.get(f * PER_FILE + p));
        }
      }
    }
    CharSequenceMap<PositionDeleteIndex> a = stock(CloseableIterable.withNoopClose(rows));
    CharSequenceMap<PositionDeleteIndex> b =
        Deletes.toPositionIndexes(CloseableIterable.withNoopClose(rows));
    long ca = a.values().stream().mapToLong(PositionDeleteIndex::cardinality).sum();
    long cb = b.values().stream().mapToLong(PositionDeleteIndex::cardinality).sum();
    if (a.size() != FILES || b.size() != FILES || ca != ROWS || cb != ROWS) {
      throw new AssertionError(
          "indexes differ: " + a.size() + "/" + b.size() + " " + ca + "/" + cb);
    }
  }

  @Benchmark
  public void stock(Blackhole bh) {
    bh.consume(stock(CloseableIterable.withNoopClose(rows)));
  }

  @Benchmark
  public void patched(Blackhole bh) {
    bh.consume(Deletes.toPositionIndexes(CloseableIterable.withNoopClose(rows)));
  }
}
