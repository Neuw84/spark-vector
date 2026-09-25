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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import org.apache.iceberg.MetadataColumns;
import org.apache.iceberg.Schema;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.util.CharSequenceMap;
import org.junit.jupiter.api.Test;

/** Deletes.toPositionIndexes reusing the previous row's index while the data-file path repeats. */
public class TestDeletesPositionIndexes {

  private static final Schema SCHEMA =
      new Schema(MetadataColumns.DELETE_FILE_PATH, MetadataColumns.DELETE_FILE_POS);

  private static Record row(CharSequence path, long pos) {
    Record r = GenericRecord.create(SCHEMA);
    r.set(0, path);
    r.set(1, pos);
    return r;
  }

  /** The expected indexes, built independently of Deletes: path -> sorted positions. */
  private static Map<String, TreeSet<Long>> reference(List<Record> rows) {
    Map<String, TreeSet<Long>> expected = new HashMap<>();
    for (Record r : rows) {
      expected.computeIfAbsent(r.get(0).toString(), k -> new TreeSet<>()).add((Long) r.get(1));
    }
    return expected;
  }

  private static void assertIndexes(
      CharSequenceMap<PositionDeleteIndex> indexes, Map<String, TreeSet<Long>> expected) {
    assertThat(indexes).hasSize(expected.size());
    for (Map.Entry<String, TreeSet<Long>> e : expected.entrySet()) {
      PositionDeleteIndex index = indexes.get(e.getKey());
      assertThat(index).as(e.getKey()).isNotNull();
      assertThat(index.cardinality()).as(e.getKey()).isEqualTo(e.getValue().size());
      for (long pos : e.getValue()) {
        assertThat(index.isDeleted(pos)).as(e.getKey() + " @ " + pos).isTrue();
      }
      assertThat(index.isDeleted(e.getValue().last() + 1)).isFalse();
    }
  }

  private static String path(int f) {
    return "s3://bucket/warehouse/db/t/data/part="
        + (f % 3)
        + "/"
        + String.format("%05d-0-abc.parquet", f);
  }

  private static List<Record> sortedRows(int files, int perFile) {
    List<Record> rows = new ArrayList<>();
    for (int f = 0; f < files; f++) {
      String p = path(f);
      for (int i = 0; i < perFile; i++) {
        rows.add(row(new String(p.toCharArray()), (long) i * 2 + f));
      }
    }
    return rows;
  }

  @Test
  public void testSortedPaths() {
    List<Record> rows = sortedRows(20, 500);
    assertIndexes(
        Deletes.toPositionIndexes(CloseableIterable.withNoopClose(rows)), reference(rows));
  }

  @Test
  public void testEveryRowChangesPath() {
    List<Record> sorted = sortedRows(7, 300);
    List<Record> rows = new ArrayList<>();
    for (int i = 0; i < 300; i++) {
      for (int f = 0; f < 7; f++) {
        rows.add(sorted.get(f * 300 + i));
      }
    }
    assertIndexes(
        Deletes.toPositionIndexes(CloseableIterable.withNoopClose(rows)), reference(rows));
  }

  @Test
  public void testAPathThatReturnsKeepsOneIndex() {
    // A, A, B, A, C, C, A: A's rows land in the same index however they are interleaved.
    String a = path(1);
    String b = path(2);
    String c = path(3);
    List<Record> rows =
        List.of(row(a, 1), row(a, 2), row(b, 1), row(a, 3), row(c, 9), row(c, 10), row(a, 4));
    CharSequenceMap<PositionDeleteIndex> indexes =
        Deletes.toPositionIndexes(CloseableIterable.withNoopClose(rows));
    assertIndexes(indexes, reference(rows));
    assertThat(indexes.get(a).cardinality()).isEqualTo(4);
  }

  @Test
  public void testPathsOfEqualLengthDifferingInOneCharacter() {
    List<Record> rows = new ArrayList<>();
    Random rnd = new Random(20);
    for (int i = 0; i < 2000; i++) {
      int f =
          rnd.nextInt(4); // .../00000-, 00001-, 00002-, 00003-: same length, one character apart
      rows.add(row(path(f * 3), i));
    }
    assertIndexes(
        Deletes.toPositionIndexes(CloseableIterable.withNoopClose(rows)), reference(rows));
  }
}
