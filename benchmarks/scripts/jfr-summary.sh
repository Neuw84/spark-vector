#!/usr/bin/env bash
# Summarises a Java Flight Recorder file as text that can be attached to an issue: the hot-method
# table, the callers of the JDK-internal MemorySegment frames AGENTS.md section 4.7 tells you to look
# behind, the plugin's own frames by self time, the top allocation sites, the GC pauses, the waits
# and the native methods (Comet's JNI side). Reading order is the one in section 4.7.
#
#   benchmarks/scripts/jfr-summary.sh <recording.jfr> [--top N] [--width N] [--depth N]
#
#   --top    rows per table (default 15)
#   --width  table width for `jfr view` (default 160)
#   --depth  stack depth read for the caller analysis (default 12)
#
# Environment: JAVA_HOME (JDK 25; the `jfr` tool of the JDK that produced the recording works too).
# Only reads the recording; writes nothing.
set -euo pipefail
FILE="${1:?usage: jfr-summary.sh <recording.jfr> [--top N] [--width N] [--depth N]}"
shift
TOP=15
WIDTH=160
DEPTH=12
while [ $# -gt 0 ]; do
  case "$1" in
    --top) TOP="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -f "$FILE" ] || { echo "no such recording: $FILE" >&2; exit 2; }
JFR="${JAVA_HOME:?set JAVA_HOME to a JDK 25}/bin/jfr"

# `jfr view` prints the whole table; keep the header plus the first TOP non-blank rows.
view() {
  local name="$1"
  "$JFR" view --width "$WIDTH" "$name" "$FILE" 2>/dev/null | awk -v top="$TOP" '
    NR <= 3 { print; next }            # title, blank, header
    /^-+/ { print; next }               # separators
    /^[[:space:]]*$/ { if (rows <= top) print; next }
    { rows++; if (rows <= top) print }
  '
}

section() { printf '\n== %s\n' "$1"; }

echo "# JFR summary of $(basename "$FILE")"
"$JFR" summary "$FILE" | awk '/^ *Duration:|^ *Start:|^ *Version:|^ *Events:/ { print }'

section "1. Hot methods (top-of-stack samples; read after the runner's per-operator kernel times)"
view hot-methods

# The two analyses below walk the raw execution samples once with awk: the first frame of each
# stack is the sampled method, the rest are its callers, innermost first.
SAMPLES="$("$JFR" print --events jdk.ExecutionSample --stack-depth "$DEPTH" "$FILE" 2>/dev/null)"
TOTAL="$(printf '%s\n' "$SAMPLES" | grep -c '^jdk.ExecutionSample' || true)"
echo
echo "Execution samples read: $TOTAL (stack depth $DEPTH)"

section "2. Callers of JDK-internal MemorySegment frames (a hot one means an access the JIT did not hoist: megamorphic site, per-call session, tiny mismatch)"
printf '%s\n' "$SAMPLES" | awk -v total="$TOTAL" -v top="$TOP" '
  function flush() {
    if (n > 0 && top_is_internal) {
      caller = "(no non-JDK caller within depth)"
      for (i = 2; i <= n; i++) if (frames[i] !~ /^(jdk\.|java\.|sun\.)/) { caller = frames[i]; break }
      key = frames[1] " <- " caller
      count[key]++
    }
    n = 0; top_is_internal = 0; in_stack = 0
  }
  /^jdk.ExecutionSample/ { flush() }
  /stackTrace = \[/ { in_stack = 1; next }
  in_stack && /^\s*\]/ { in_stack = 0; next }
  in_stack {
    f = $0; sub(/^[ \t]+/, "", f); sub(/[ \t]+line:.*$/, "", f); sub(/\(.*$/, "", f)
    n++; frames[n] = f
    if (n == 1 && f ~ /MemorySessionImpl\.checkValidStateRaw|checkBounds|isAlignedForElement|ScopedMemoryAccess|AbstractMemorySegmentImpl\.checkAccess|Objects\.checkIndex|Buffer\.checkIndex/) top_is_internal = 1
  }
  END {
    flush()
    printed = 0
    PROCINFO["sorted_in"] = "@val_num_desc"
    for (k in count) { if (printed++ >= top) break; printf "%6d %5.1f%%  %s\n", count[k], 100.0 * count[k] / (total > 0 ? total : 1), k }
    if (printed == 0) print "  none"
  }'

section "3. Plugin frames by self time (first io.vecruntime frame on each stack: which kernel or expression owns the samples)"
printf '%s\n' "$SAMPLES" | awk -v total="$TOTAL" -v top="$TOP" '
  function flush() {
    if (n > 0) {
      own = ""
      for (i = 1; i <= n; i++) if (frames[i] ~ /sparkvector/) { own = frames[i]; break }
      if (own != "") count[own]++; else other++
    }
    n = 0; in_stack = 0
  }
  /^jdk.ExecutionSample/ { flush() }
  /stackTrace = \[/ { in_stack = 1; next }
  in_stack && /^\s*\]/ { in_stack = 0; next }
  in_stack { f = $0; sub(/^[ \t]+/, "", f); sub(/[ \t]+line:.*$/, "", f); sub(/\(.*$/, "", f); n++; frames[n] = f }
  END {
    flush()
    printed = 0
    PROCINFO["sorted_in"] = "@val_num_desc"
    for (k in count) { if (printed++ >= top) break; printf "%6d %5.1f%%  %s\n", count[k], 100.0 * count[k] / (total > 0 ? total : 1), k }
    printf "%6d %5.1f%%  (no plugin frame within depth: Spark, the reader, Comet, the JDK)\n", other, 100.0 * other / (total > 0 ? total : 1)
  }'

section "4. Allocation by site (when the stage metrics point at GC or allocation)"
view allocation-by-site

section "5. GC pauses"
view gc-pauses

section "6. Latencies by type (parking, sleeping, monitor waits: waiting rather than compute)"
view latencies-by-type

section "7. Native methods (Comet's time is native and invisible to JFR: what its JVM side shows is what we also pay)"
view native-methods
