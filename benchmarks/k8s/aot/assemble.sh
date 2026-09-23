#!/bin/bash
# Assembles the executor's AOT cache from a recording the executors made on the cluster (#416): the
# executors ran with -XX:AOTMode=record -XX:AOTConfiguration=<conf> (render-run.sh, AOT_RECORD=1), and
# this turns one such configuration into the cache under the same JVM options and class path, so any
# executor of this image accepts it. Those are fixed by the image, so one node's recording serves them
# all; the recording node's is used because the configuration refers to the class path as it saw it.
#
#   assemble.sh <recording .aotconf> <output .aot>
# Prints the linked-class count and fails when it is zero.
set -euo pipefail
CONF="${1:?recording .aotconf}"; OUT="${2:?output .aot}"
. "$(dirname "$0")/aot-env.sh"
rm -f "$OUT"
LOG="${OUT%.aot}.log"
"$JAVA" "${AOT_BASE_OPTS[@]}" -Xmx"${ASSEMBLE_HEAP:-4g}" \
  -XX:AOTMode=create -XX:AOTConfiguration="$CONF" -XX:AOTCache="$OUT" -Xlog:aot=info:file="$LOG" \
  -cp "$CP"
LINKED=$(grep -a "aot-linked" "$LOG" | grep -a "instance classes" | sed 's/.*aot-linked = *\([0-9]*\).*/\1/' | head -1)
echo "AOT cache: $(du -h "$OUT" | cut -f1), aot-linked instance classes = ${LINKED:-0}"
[ "${LINKED:-0}" -gt 0 ]
