#!/bin/bash
# A billion tokens: the best configuration found, against its own baseline.
#
#   attend at embed+e+attn with real vectors, the arm that won at 400M.
#   Seed 0 only — one pair at this length is 8.7 hours, so the pair is the
#   unit, and a second seed is a separate night.
#
#   dim 768 (57.7M params), context 512, batch 24, bf16, lr 1e-3
#   85,000 steps = 1.044B tokens, just under the 1.05B prepared: no repeats.
#   ~4.3 h a run at the measured 67k tok/s.
#
#   batch stays 24: batch 32 at dim 768 peaks at 24.0 of the card's 24.5GB
#   and the concept arm wants about 1GB more than the baseline, so 32 would
#   OOM the concept run unattended.
# ONE RUN AT A TIME.
cd ~/OpenMythos
OUT=data/concept_bench/billion768
mkdir -p "$OUT/logs"
COMMON="--dim 768 --seq-len 512 --batch-size 24 --precision bf16 --lr 1e-3 \
  --steps 85000 --warmup 4000 --eval-every 10000 --eval-batches 20 --log-every 1000 \
  --device cuda --threads 8 --cache-dir data/concept_bench_1b"

# The night queue holds the card. Its driver is a bash loop, so between its
# runs there is a window with no python process at all: waiting on the python
# alone can slip into that window and put two jobs on one card. Wait for the
# driver too, and look twice with a gap, so a queue that was merely between
# runs has started its next one before the second look.
#
# Both patterns are deliberately narrow. A loose "run_night.sh" also matches
# the ssh wrapper that launched the night queue and every monitoring command
# that mentions it by name, and waiting on one of those would hang here for
# good; anchoring on the driver's own argv matches the one process that means
# the card is taken.
idle () {
  ! pgrep -f "^bash /home/pedro/run_night[.]sh" > /dev/null \
    && ! pgrep -f "python tests/concept_benchmark[.]py run" > /dev/null
}
wait_for_card () {
  while ! idle; do sleep 60; done
  sleep 120
  while ! idle; do sleep 60; done
}
echo "=== waiting for the card $(date +%F' '%H:%M:%S) ==="
wait_for_card
echo "=== card free $(date +%F' '%H:%M:%S) ==="

# The cache is written by a prepare running beside the night queue, and it may
# still be mid-write when the card comes free. Reading a half-written file
# would waste the night, so wait for the writer and then check the length:
# 85,000 steps of 24 x 512 need 1,044,480,000 tokens and must not wrap.
while pgrep -f "python tests/concept_benchmark[.]py prepare" > /dev/null; do sleep 60; done
.venv/bin/python - <<'PY' || exit 1
import sys, torch
need = 85_000 * 24 * 512
d = torch.load("data/concept_bench_1b/tokens.pt", map_location="cpu")
have, ev = d["train"].numel(), d["eval"].numel()
print(f"cache: train {have:,} tokens ({d['train'].dtype}), eval {ev:,}, need {need:,}")
if have < need + 512 or ev < 100_000:
    sys.exit(f"ABORT: cache too small, {have:,} train tokens for {need:,} needed")
PY

go () {
  name=$1; shift
  [ -f "$OUT/$name.json" ] && { echo "skip $name (already done)"; return; }
  # Last word before claiming the card: anything else resident on it means a
  # job this script cannot see, and starting beside it is the one thing that
  # must not happen.
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -gt 2000 ]; then
    echo "ABORT before $name: ${used} MiB already on the card, not starting"
    exit 1
  fi
  echo "=== start $name $(date +%F' '%H:%M:%S) ==="
  .venv/bin/python tests/concept_benchmark.py run "$@" $COMMON \
    --name "$name" --results-dir "$OUT" > "$OUT/logs/$name.log" 2>&1
  rc=$?
  echo "=== done $name rc=$rc $(date +%F' '%H:%M:%S) ==="
  [ $rc -ne 0 ] && echo "    (failed; see $OUT/logs/$name.log)"
  return 0
}

go "baseline-s0"                  --variant baseline --seed 0
go "real-attend-embed+e+attn-s0"  --variant real --combiner attend --sites embed,e,attn --seed 0
echo BILLION_FINISHED
