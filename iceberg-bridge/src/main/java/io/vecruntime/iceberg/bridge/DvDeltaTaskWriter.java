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
package io.vecruntime.iceberg.bridge;

import java.io.IOException;
import java.util.Map;
import java.util.function.Function;

import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.StructLike;
import org.apache.iceberg.Table;
import org.apache.iceberg.data.BaseDeleteLoader;
import org.apache.iceberg.data.DeleteLoader;
import org.apache.iceberg.deletes.PositionDeleteIndex;
import org.apache.iceberg.io.DeleteWriteResult;
import org.apache.iceberg.io.FileIO;
import org.apache.iceberg.io.OutputFileFactory;
import org.apache.iceberg.spark.source.IcebergDvCommitBridge;
import org.apache.spark.sql.connector.write.WriterCommitMessage;

/**
 * One write task's columnar deletion-vector writer (#20, slice 4-live). It is
 * created on the executor from a broadcast {@link Table} and the task's
 * partition/task ids, fed one call per run of rows that share a data file (the
 * delta rows arrive already clustered by {@code _spec_id}/{@code _partition}/
 * {@code _file} from the write's REBALANCE exchange), and produces the per-task
 * {@link WriterCommitMessage} the driver hands to Iceberg's own {@code
 * DeltaBatchWrite.commit}.
 *
 * <p>Everything Iceberg-specific is Iceberg's: the bitmap and its Puffin
 * serialisation ({@link ColumnarDvWriter} → {@code BaseDVFileWriter}), the
 * merge of a data file's previously committed deletion vector ({@code
 * BaseDVFileWriter.close()} via the {@link DeleteLoader} loader below), and the
 * commit-message assembly ({@link IcebergDvCommitBridge}). The only thing that
 * is ours is filling each file's index from a run of positions in one call
 * instead of routing one {@code InternalRow} at a time.
 */
public final class DvDeltaTaskWriter implements AutoCloseable {

    private final ColumnarDvWriter dvWriter;
    private final Table table;

    private DvDeltaTaskWriter(ColumnarDvWriter dvWriter, Table table) {
        this.dvWriter = dvWriter;
        this.table = table;
    }

    /**
     * Builds a task writer. {@code rewritableDeletes} maps a data file path to
     * the set of previously-committed delete files whose positions must be
     * merged into the new blob (empty when nothing is being rewritten); the
     * merge is done by {@code BaseDVFileWriter.close()} through a loader built
     * from Iceberg's public {@link BaseDeleteLoader}. {@code
     * partitionId}/{@code taskId} name the output files uniquely per task.
     */
    public static DvDeltaTaskWriter create(Table table, int partitionId, long taskId,
            Map<String, java.util.List<DeleteFile>> rewritableDeletes) {
        OutputFileFactory outputFileFactory = OutputFileFactory.builderFor(table, partitionId, taskId)
                .format(org.apache.iceberg.FileFormat.PUFFIN)
                .operationId(java.util.UUID.randomUUID().toString())
                .build();
        FileIO io = table.io();
        DeleteLoader deleteLoader = new BaseDeleteLoader(deleteFile -> io.newInputFile(deleteFile.location()));
        Function<CharSequence, PositionDeleteIndex> previousDeletes = path -> {
            if (rewritableDeletes == null) {
                return null;
            }
            java.util.List<DeleteFile> files = rewritableDeletes.get(path.toString());
            if (files == null || files.isEmpty()) {
                return null;
            }
            return deleteLoader.loadPositionDeletes(files, path);
        };
        return new DvDeltaTaskWriter(new ColumnarDvWriter(outputFileFactory, previousDeletes), table);
    }

    /**
     * Records one data file's deletes (a run of positions) in a single bulk
     * call, resolving the partition spec and partition tuple from the row-level
     * operation's metadata: {@code specId} and the partition as the Spark
     * {@code InternalRow} Iceberg's {@code WriteDeltaProjections} produced (or
     * {@code null} for an unpartitioned table). Keeping this resolution in the
     * bridge means the plugin core hands over only Spark and primitive types.
     */
    public void deleteFile(String dataFilePath, long[] positions, int count,
                           int specId, org.apache.spark.sql.catalyst.InternalRow partitionRow) {
        PartitionSpec spec = table.specs().get(specId);
        if (spec == null) {
            throw new IllegalStateException("unknown partition spec id " + specId + " for " + table.name());
        }
        StructLike partition = IcebergDvCommitBridge.wrapPartition(spec, partitionRow);
        dvWriter.deleteFile(dataFilePath, positions, count, spec, partition);
    }

    /**
     * Closes the writer and returns the per-task commit message (a delete-only
     * {@code DeltaTaskCommit}) for the driver to hand to Iceberg's {@code DeltaBatchWrite.commit}.
     */
    public WriterCommitMessage commit() throws IOException {
        dvWriter.close();
        DeleteWriteResult result = dvWriter.result();
        return IcebergDvCommitBridge.deleteOnlyCommit(result);
    }

    @Override
    public void close() throws IOException {
        dvWriter.close();
    }
}
