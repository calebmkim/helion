#!/usr/bin/env bash
# Serial foreground driver: one fresh process per (corpus, kernel) — footgun #11.
# Skips any kernel whose output JSON already exists (resume-safe). NEVER backgrounds a
# GPU job. Run foreground under the turn.
#
# Usage: run_sweep.sh <RESULTS_DIR> [corpus_filter]
#   corpus_filter (optional): only run kernels whose corpus matches this substring.
set -u
RESULTS="${1:?results dir}"
FILTER="${2:-}"
WT=/home/dev/local/helion-redesign
PY=/home/dev/helion/.venv/bin/python
BENCH="$WT/_lab/perf_report/perf_report_bench.py"
WL=/tmp/perf_worklist.json

mkdir -p "$RESULTS"
mapfile -t PAIRS < <("$PY" -c "import json;[print(c+' '+k) for c,k in json.load(open('$WL'))]")

for pair in "${PAIRS[@]}"; do
  corpus="${pair%% *}"; kernel="${pair##* }"
  if [ -n "$FILTER" ] && [[ "$corpus" != *"$FILTER"* ]]; then continue; fi
  out="$RESULTS/${corpus}__${kernel}.json"
  if [ -f "$out" ]; then echo "[skip] $corpus/$kernel (exists)"; continue; fi
  echo "==================== $corpus/$kernel ===================="
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH="$WT" "$PY" "$BENCH" \
    --corpus "$corpus" --kernel "$kernel" --out-dir "$RESULTS" 2>&1 \
    | grep -E "^\[cell\]|^\[ERR |^=== DONE|Error|Traceback" | tail -40
done
echo "SWEEP CHUNK COMPLETE"
