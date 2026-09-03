#!/usr/bin/env bash
set -u
ROOT="/home/aiattacks/pengyang/deepfake-attacks"
LOG="$ROOT/logs/direction3_shared_filter27_heldout.log"
mkdir -p "$ROOT/logs" "$ROOT/experiments/direction3"
exec >> "$LOG" 2>&1
printf '[D3-FILTER27] started %s\n' "$(date -Is)"
GPU=""
while [[ -z "$GPU" ]]; do
  GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=', ' read -r i m u; do if (( m < 12000 && u < 30 )); then printf '%s\n' "$i"; break; fi; done)"
  if [[ -z "$GPU" ]]; then printf '[D3-FILTER27] waiting for GPU\n'; sleep 60; fi
done
printf '[D3-FILTER27] selected GPU %s\n' "$GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" "$ROOT/workflows/direction3_shared_filter.py" --config "$ROOT/configs/direction3_shared_filter27_validation.yaml"
