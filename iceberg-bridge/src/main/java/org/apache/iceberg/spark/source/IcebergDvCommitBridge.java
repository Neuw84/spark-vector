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
package org.apache.iceberg.spark.source;

import org.apache.iceberg.io.DeleteWriteResult;
import org.apache.spark.sql.connector.write.WriterCommitMessage;

/**
 * Same-package bridge for the columnar v3 deletion-vector writer (#20, option
 * B). It lives in {@code org.apache.iceberg.spark.source} so it can call the
 * package-private {@link SparkPositionDeltaWrite.DeltaTaskCommit} constructors
 * — the one piece of the write path that Iceberg does not expose publicly. The
 * plugin core has no compile-time Iceberg dependency and reaches this class
 * reflectively (its methods take and return only Iceberg / Spark public types,
 * so the reflective signatures are stable), exactly as it reaches Iceberg's
 * reader.
 *
 * <p>This class assembles ONLY the per-task commit message and hands off
 * construction of the deletion vectors to Iceberg's public {@link
 * org.apache.iceberg.deletes.BaseDVFileWriter} (see {@link
 * io.vecruntime.iceberg.bridge.ColumnarDvWriter}); the driver-side commit
 * stays Iceberg's own {@code RowDelta} through {@code
 * DeltaBatchWrite.commit(messages)}, so nothing about snapshot semantics,
 * metrics or previous-DV bookkeeping is reimplemented here. (The previous-DV
 * merge is Iceberg's too: {@code BaseDVFileWriter.close()} merges any prior
 * vector for a data file through the loader the caller supplies, which is built
 * from Iceberg's public {@code DeleteLoader}, not from the private {@code
 * PreviousDeleteLoader} nested here.)
 */
public final class IcebergDvCommitBridge {

    private IcebergDvCommitBridge() {}

    /**
     * Whether a target Iceberg table is the columnar DV writer's kind: format
     * version &ge; 3, where positional deletes are encoded as deletion vectors
     * (Puffin) rather than positional-delete Parquet files. This is the
     * public-API stand-in for Iceberg's package-private {@code
     * Context.useDVs()} (which is {@code deleteFileFormat == PUFFIN}, and
     * Iceberg selects PUFFIN exactly for v3): the plugin's planner strategy
     * checks it on the target table of a logical {@code WriteDelta}, and
     * declines to Spark's own {@code WriteDeltaExec} (with a printed reason)
     * for v2 or any non-Iceberg table. Reading the format version needs no
     * private access.
     */
    public static boolean isDvEligible(org.apache.iceberg.Table table) {
        if (table == null) {
            return false;
        }
        try {
            return org.apache.iceberg.TableUtil.formatVersion(table) >= 3;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Whether the table's current partition spec is unpartitioned. The columnar
     * DELETE operator only supports unpartitioned tables in this landing: for a
     * partitioned spec the per-file partition tuple must be threaded through to
     * Iceberg's commit, which is a later slice, so the strategy declines to
     * Spark's own writer for partitioned tables.
     */
    public static boolean isUnpartitioned(org.apache.iceberg.Table table) {
        if (table == null) {
            return false;
        }
        try {
            return !table.spec().isPartitioned();
        } catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Whether the table's current snapshot already carries delete files
     * (positional, equality, or deletion vectors). This landing writes an
     * append-only DV per touched data file and does NOT merge a
     * previously-committed DV for that file (Iceberg rejects two DVs indexing
     * the same data file), so the strategy declines when the table already has
     * deletes and lets Spark's own writer handle the merge; repeated-delete DV
     * merging is a later slice.
     */
    public static boolean hasCommittedDeletes(org.apache.iceberg.Table table) {
        if (table == null) {
            return true; // fail safe: if we cannot tell, decline
        }
        try {
            org.apache.iceberg.Snapshot snap = table.currentSnapshot();
            if (snap == null) {
                return false;
            }
            java.util.Map<String, String> summary = snap.summary();
            if (summary == null) {
                // A live snapshot with no summary at all is unexpected; be conservative and decline.
                return true;
            }
            // Iceberg stamps running totals on every snapshot's summary. A delete of any kind
            // (positional, equality, or deletion vector) leaves a positive count in one of these
            // keys. On an append-only snapshot (e.g. the INSERT that seeds a fresh v3 table) the
            // delete totals are either "0" or simply absent -- absent means "none ever committed",
            // NOT "unknown", so it must read as no deletes, or the strategy would wrongly decline
            // every clean table and the columnar DELETE would never run.
            return positiveCount(summary.get("total-delete-files"))
                    || positiveCount(summary.get("total-position-deletes"))
                    || positiveCount(summary.get("total-equality-deletes"));
        } catch (RuntimeException e) {
            return true;
        }
    }

    /** True when a snapshot-summary count string is present and parses to a value &gt; 0. */
    private static boolean positiveCount(String value) {
        if (value == null) {
            return false;
        }
        try {
            return Long.parseLong(value.trim()) > 0L;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /**
     * Resolves the partition tuple of a delete run as an Iceberg {@link
     * org.apache.iceberg.StructLike} from the row-level operation's {@code
     * specId} and the partition as the Spark {@code InternalRow} Iceberg's
     * {@code WriteDeltaProjections} produced. Returns {@code null} for an
     * unpartitioned spec. Uses the package-private {@code InternalRowWrapper},
     * which is why it lives in this same-package bridge.
     */
    public static org.apache.iceberg.StructLike wrapPartition(org.apache.iceberg.PartitionSpec spec, org.apache.spark.sql.catalyst.InternalRow partitionRow) {
        if (spec == null || !spec.isPartitioned() || partitionRow == null) {
            return null;
        }
        org.apache.spark.sql.types.StructType sparkType = (org.apache.spark.sql.types.StructType) org.apache.iceberg.spark.SparkSchemaUtil.convert(spec.partitionType());
        return new InternalRowWrapper(sparkType, spec.partitionType()).wrap(partitionRow);
    }

    /**
     * Wraps a {@link DeleteWriteResult} (the output of the columnar DV writer's
     * {@code close()}) as the {@link WriterCommitMessage} Iceberg's {@code
     * PositionDeltaBatchWrite.commit(...)} expects for a delete-only task. The
     * driver collects these and passes them straight to Iceberg's commit.
     */
    public static WriterCommitMessage deleteOnlyCommit(DeleteWriteResult deleteResult) {
        return new SparkPositionDeltaWrite.DeltaTaskCommit(deleteResult);
    }

    /**
     * Wraps a full {@link org.apache.iceberg.io.WriteResult} (data files from
     * inserts plus delete files / referenced data files from the DV writer,
     * combined by the caller) as the delta commit message, for a task that both
     * deleted and inserted rows (the full MERGE / UPDATE path).
     */
    public static WriterCommitMessage deleteAndDataCommit(org.apache.iceberg.io.WriteResult writeResult) {
        return new SparkPositionDeltaWrite.DeltaTaskCommit(writeResult);
    }
}
