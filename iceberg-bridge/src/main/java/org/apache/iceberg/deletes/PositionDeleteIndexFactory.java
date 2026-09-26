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
package org.apache.iceberg.deletes;

/**
 * Same-package factory for a MUTABLE {@link PositionDeleteIndex} (#20). {@link
 * PositionDeleteIndex}'s public {@code empty()} returns an immutable singleton
 * that throws on {@code delete}, and the mutable {@link
 * BitmapPositionDeleteIndex} has only a package-private constructor — so this
 * tiny class lives in {@code org.apache.iceberg.deletes} to reach it, and
 * builds the whole file's index from a run of positions in one call. It
 * reimplements nothing: the bitmap, its serialisation and its merge are all
 * {@link BitmapPositionDeleteIndex}'s.
 */
public final class PositionDeleteIndexFactory {

    private PositionDeleteIndexFactory() {}

    /**
     * Builds a mutable {@link PositionDeleteIndex} holding {@code
     * positions[0..count)}. Positions need not be sorted or unique; the roaring
     * bitmap deduplicates. {@code count == 0} yields an empty mutable index
     * (not the immutable {@code empty()} singleton).
     */
    public static PositionDeleteIndex fromPositions(long[] positions, int count) {
        BitmapPositionDeleteIndex index = new BitmapPositionDeleteIndex();
        for (int i = 0; i < count; i++) {
            index.delete(positions[i]);
        }
        return index;
    }
}
