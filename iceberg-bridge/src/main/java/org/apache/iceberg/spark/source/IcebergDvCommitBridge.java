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
 * io.sparkvector.iceberg.bridge.ColumnarDvWriter}); the driver-side commit
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
