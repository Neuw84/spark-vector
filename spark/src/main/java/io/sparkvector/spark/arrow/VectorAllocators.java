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
