#!/usr/bin/env bash
# Overnight supervisor: relaunch the resumable driver until the full matrix is complete.
# If the driver process dies (segfault / host hiccup / OOM-killer), the loop restarts it and
# --resume + kernel-level-skip continue from the last checkpointed cell. Bounded by MAX_ATTEMPTS.
set -u
OUTDIR="${1:?usage: overnight_supervisor.sh <out-dir>}"
PERF="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$PERF/.." && pwd)"
PY="${HELION_PY:-/home/dev/helion/.venv/bin/python}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
mkdir -p "$OUTDIR"
SUPLOG="$OUTDIR/supervisor.log"

echo "=== supervisor start $(date -u +%FT%TZ) out=$OUTDIR ===" | tee -a "$SUPLOG"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "--- attempt $attempt $(date -u +%FT%TZ) ---" | tee -a "$SUPLOG"
  # reap any orphaned compile workers from a prior killed attempt
  pkill -9 -f compile_worker 2>/dev/null; pkill -9 -f ptxas 2>/dev/null; sleep 2
  PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HELION_AUTOTUNE_EFFORT=none \
    "$PY" -u "$PERF/run_all.py" --out-dir "$OUTDIR" >> "$OUTDIR/driver.log" 2>&1
  rc=$?
  echo "attempt $attempt driver exit=$rc $(date -u +%FT%TZ)" | tee -a "$SUPLOG"
  # driver exits 0 when all pairs clean; exit 1 when some pair had a problem but it FINISHED the
  # sweep. Either way, if every kernel JSON is complete we're done. Check via the manifest.
  if [ "$rc" -eq 0 ]; then
    echo "=== supervisor: driver reports all clean, done ===" | tee -a "$SUPLOG"
    break
  fi
  # rc!=0: could be (a) finished-with-some-problem-pairs (real failures — restarting won't fix,
  # they'll re-run and re-fail), or (b) the driver itself was killed mid-sweep (restart helps).
  # Distinguish: if a run_manifest.json exists AND its pair count == expected, the sweep FINISHED
  # (problems are real) -> stop. Else the driver died mid-sweep -> restart.
  FINISHED=$("$PY" - "$OUTDIR/run_manifest.json" "$PERF/shapes.json" <<'PYEOF'
import json,sys,os
man_p, shapes_p = sys.argv[1], sys.argv[2]
if not os.path.exists(man_p): print("no"); sys.exit()
man=json.load(open(man_p)); sh=json.load(open(shapes_p))
n_pairs=sum(len(c["kernels"]) for c in sh["corpora"].values())
print("yes" if len(man.get("pairs",[]))>=n_pairs else "no")
PYEOF
)
  if [ "$FINISHED" = "yes" ]; then
    echo "=== supervisor: sweep FINISHED with some problem-pairs (real failures, restart won't help). Stopping. ===" | tee -a "$SUPLOG"
    break
  fi
  echo "supervisor: driver died mid-sweep, restarting (resume)..." | tee -a "$SUPLOG"
  sleep 5
done
echo "=== supervisor end $(date -u +%FT%TZ) ===" | tee -a "$SUPLOG"
