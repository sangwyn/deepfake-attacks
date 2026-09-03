#!/usr/bin/env bash
set -u
ROOT="/home/aiattacks/pengyang/deepfake-attacks"
mkdir -p "$ROOT/logs" "$ROOT/experiments/direction5"
exec >> "$ROOT/logs/direction5_scheduler_46.log" 2>&1
printf '[D5] started %s\n' "$(date -Is)"
"$ROOT/.venv/bin/python" "$ROOT/evaluate.py" --config "$ROOT/experiments/direction5/scheduler_46.yaml"
