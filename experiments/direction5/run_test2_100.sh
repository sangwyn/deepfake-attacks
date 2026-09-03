#!/usr/bin/env bash
set -u
ROOT="/home/aiattacks/pengyang/deepfake-attacks"
mkdir -p "$ROOT/logs" "$ROOT/experiments/direction5"
exec >> "$ROOT/logs/direction5_test2_100.log" 2>&1
printf '[D5-test2-100] started %s\n' "$(date -Is)"
"$ROOT/.venv/bin/python" "$ROOT/evaluate.py" --config "$ROOT/experiments/direction5/scheduler_v3_test2_100.yaml"
"$ROOT/.venv/bin/python" "$ROOT/evaluate.py" --config "$ROOT/experiments/direction5/scheduler_v3_test2_equal_100.yaml"
"$ROOT/.venv/bin/python" "$ROOT/evaluate.py" --config "$ROOT/experiments/direction5/scheduler_v3_test2_spatial_100.yaml"
printf '[D5-test2-100] finished status=%s at %s\n' "$?" "$(date -Is)"
