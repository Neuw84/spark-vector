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
package io.vecruntime.benchmarks;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.vecruntime.kernels.Bitmap;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1,
        jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED"})
@State(Scope.Thread)
public class BitmapBenchmark {

    @Param({"4096", "65536"})
    int numBits;

    Arena arena;
    MemorySegment bitmap;

    @Setup(Level.Trial)
    public void setup() {
        arena = Arena.ofShared();
        bitmap = Bitmap.allocate(arena, numBits);
        Random rnd = new Random(7);
        for (long i = 0; i < bitmap.byteSize(); i++) {
            bitmap.set(ValueLayout.JAVA_BYTE, i, (byte) rnd.nextInt(256));
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        arena.close();
    }

    @Benchmark
    public int popcountWords() {
        return Bitmap.popcount(bitmap, numBits);
    }

    @Benchmark
    public int popcountScalarBits() {
        int c = 0;
        for (int i = 0; i < numBits; i++) {
            if (Bitmap.isSet(bitmap, i)) {
                c++;
            }
        }
        return c;
    }
}
