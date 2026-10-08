#!/bin/bash
# The three arms side by side on one card: at 2.1M params each run leaves most
# of a 4090 idle, so running them together costs little per run.
#   bash experiments/jev_reasoning/run_arms.sh [steps]
cd "$(dirname "$0")"
PY=${PY:-$HOME/OpenMythos/.venv/bin/python}
STEPS=${1:-20000}
mkdir -p results/logs
for arm in act fixed random; do
  nohup "$PY" train.py --arm "$arm" --steps "$STEPS" \
    > "results/logs/$arm-s0.log" 2>&1 &
done
wait
echo ARMS_FINISHED >> results/logs/done
