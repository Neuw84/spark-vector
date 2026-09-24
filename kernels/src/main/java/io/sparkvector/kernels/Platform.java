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
package io.sparkvector.kernels;

import java.lang.management.ManagementFactory;

/**
 * The SIMD platform the kernels run on, probed once at class initialisation and
 * folded into {@code static final} booleans the JIT treats as constants (#283).
 * Kernels switch on these at the top of a loop, never per call, and never on
 * CPU flags read at runtime.
 *
 * <p>The probe reads HotSpot's own decision -- {@code UseAVX} (0-3), {@code
 * UseSVE} (0-2) and {@code MaxVectorSize} -- through the diagnostic MXBean, so
 * a JVM started with {@code -XX:UseAVX=2} on AVX-512 hardware is an {@code
 * avx2} platform, as the code the JIT emits will be. {@code
 * -Dsparkvector.platform=neon|sve|avx2|avx512} overrides the probe, for tests
 * that force a foreign path (emulated: the Vector API stays correct, only the
 * fast paths change) and for measuring one path against another on the same
 * machine.
 *
 * <ul>
 *   <li>{@link #MASK_REGISTERS}: masks live in predicate registers ({@code
 *       k0-k7} on AVX-512, the {@code p} registers on SVE), so {@code
 *       VectorMask.fromLong} is one move and masked lanewise operations take
 *       the mask directly. On NEON and AVX2 a mask is a vector, and {@code
 *       fromLong} has no fast path -- the kernels build masks with a
 *       broadcast-AND-compare.
 *   <li>{@link #NATIVE_COMPRESS}: {@code Vector.compress} is one instruction
 *       ({@code vpcompress} on AVX-512, {@code compact} on SVE); elsewhere a
 *       shuffle table and {@code rearrange} stand in.
 * </ul>
 */
public final class Platform {

    /** The platform names accepted by {@code -Dsparkvector.platform}. */
    public static final String NEON = "neon";

    public static final String SVE = "sve";
    public static final String AVX2 = "avx2";
    public static final String AVX512 = "avx512";

    /**
     * The platform in effect, one of the names above; {@code "unknown"} when no
     * probe applied.
     */
    public static final String NAME = probe();

    /**
     * Predicate registers: {@code VectorMask.fromLong} and masked operations
     * are native.
     */
    public static final boolean MASK_REGISTERS = NAME.equals(AVX512) || NAME.equals(SVE);

    /** {@code Vector.compress(mask)} is a single instruction. */
    public static final boolean NATIVE_COMPRESS = NAME.equals(AVX512) || NAME.equals(SVE);

    /** The JIT's maximum vector size in bytes, or 0 when unknown. */
    public static final int MAX_VECTOR_BYTES = vmOptionInt("MaxVectorSize");

    private Platform() {}

    private static String probe() {
        String override = System.getProperty("sparkvector.platform");
        if (override != null && !override.isEmpty()) {
            switch (override) {
                case NEON, SVE, AVX2, AVX512 -> {
                    return override;
                }
                default -> throw new IllegalArgumentException("sparkvector.platform must be neon, sve, avx2 or avx512, got " + override);
            }
        }
        String arch = System.getProperty("os.arch", "");
        if (arch.equals("aarch64")) {
            return vmOptionInt("UseSVE") > 0 ? SVE : NEON;
        }
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            int avx = vmOptionInt("UseAVX");
            if (avx >= 3) {
                return AVX512;
            }
            if (avx >= 2) {
                return AVX2;
            }
        }
        return "unknown";
    }

    /**
     * A HotSpot VM option as an int, 0 when the bean or the option is
     * unavailable (a non-HotSpot JVM).
     */
    static int vmOptionInt(String name) {
        try {
            com.sun.management.HotSpotDiagnosticMXBean bean = ManagementFactory.getPlatformMXBean(com.sun.management.HotSpotDiagnosticMXBean.class);
            if (bean == null) {
                return 0;
            }
            return Integer.parseInt(bean.getVMOption(name)
                    .getValue());
        } catch (RuntimeException | LinkageError e) {
            return 0;
        }
    }
}
