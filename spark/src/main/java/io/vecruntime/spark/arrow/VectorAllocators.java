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
package io.sparkvector.spark.arrow;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

/**
 * One Arrow {@link RootAllocator} per JVM; operators create a child allocator
 * per task and close it (after closing every vector they produced) from a
 * task-completion listener.
 */
public final class VectorAllocators {

    private static final RootAllocator ROOT = new RootAllocator(Long.MAX_VALUE);

    private VectorAllocators() {}

    public static BufferAllocator root() {
        return ROOT;
    }

    public static BufferAllocator newChild(String name) {
        return ROOT.newChildAllocator(name, 0L, Long.MAX_VALUE);
    }
}
