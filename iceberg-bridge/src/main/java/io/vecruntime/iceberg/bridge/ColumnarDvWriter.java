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
import java.util.function.Function;

import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.StructLike;
import org.apache.iceberg.deletes.BaseDVFileWriter;
import org.apache.iceberg.deletes.PositionDeleteIndex;
import org.apache.iceberg.io.DeleteWriteResult;
import org.apache.iceberg.io.OutputFileFactory;

/**
 * The columnar deletion-vector build (#20, slice 3). One instance per write
 * task: for each run of rows sharing a data file path, {@link #deleteFile}
 * builds a {@link PositionDeleteIndex} from that run's positions and hands the
 * whole file's positions to Iceberg's public {@link
 * BaseDVFileWriter#delete(String, PositionDeleteIndex, PartitionSpec,
 * StructLike)} in one call, instead of Spark's row-at-a-time {@code
 * delete(row)}. {@link #result()} (after {@link #close()}) is Iceberg's {@link
 * DeleteWriteResult}: the writer has serialised each bitmap into one Puffin
 * blob and merged any previous DV for the file through the loader.
 *
 * <p>This is deliberately thin — the bitmap encoding, the Puffin serialisation,
 * the previous-DV merge and the {@code DeleteFile} metadata are all Iceberg's;
 * the only thing that is ours is filling the index from column runs rather than
 * one {@code InternalRow} at a time.
 */
public final class ColumnarDvWriter implements AutoCloseable {

    private final BaseDVFileWriter writer;

    public ColumnarDvWriter(OutputFileFactory outputFileFactory, Function<CharSequence, PositionDeleteIndex> previousDeleteLoader) {
        // BaseDVFileWriter's loader parameter is Function<String, PositionDeleteIndex>; adapt from the
        // CharSequence-keyed loader Iceberg's own writer uses (a data file path is a String at the call
        // site anyway).
        Function<String, PositionDeleteIndex> loader = previousDeleteLoader == null ? path -> null : previousDeleteLoader::apply;
        this.writer = new BaseDVFileWriter(outputFileFactory, loader);
    }

    /**
     * Builds a {@link PositionDeleteIndex} from {@code positions[0..count)}
     * (row positions within the data file at {@code dataFilePath}) and records
     * the whole file's deletes in one call. Positions need not be sorted or
     * unique — the roaring bitmap deduplicates. {@code count == 0} is a no-op
     * (an empty delete set for this file writes nothing).
     */
    public void deleteFile(String dataFilePath, long[] positions, int count,
                           PartitionSpec spec, StructLike partition) {
        if (count <= 0) {
            return;
        }
        PositionDeleteIndex index = buildIndex(positions, count);
        writer.delete(dataFilePath, index, spec, partition);
    }

    /**
     * Builds a bitmap of {@code positions[0..count)}. Exposed (package-private)
     * so the differential test can compare it against Iceberg's per-row-built
     * index for the same positions.
     */
    static PositionDeleteIndex buildIndex(long[] positions, int count) {
        return org.apache.iceberg.deletes.PositionDeleteIndexFactory.fromPositions(positions, count);
    }

    /**
     * Iceberg's per-task result after {@link #close()}: the DV delete files and
     * referenced data files.
     */
    public DeleteWriteResult result() {
        return writer.result();
    }

    @Override
    public void close() throws IOException {
        writer.close();
    }
}
