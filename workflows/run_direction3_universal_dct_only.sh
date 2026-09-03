#!/usr/bin/env bash
set -u
ROOT="/home/aiattacks/pengyang/deepfake-attacks"
LOG="$ROOT/logs/direction3_universal_dct_only.log"
CONFIG="$ROOT/configs/direction3_universal_dct_only.yaml"
mkdir -p "$ROOT/logs" "$ROOT/experiments/direction3"
exec >> "$LOG" 2>&1
printf '[D3-UNIVERSAL-DCT] started %s\n' "$(date -Is)"
GPU=""
while [[ -z "$GPU" ]]; do
  GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=', ' read -r i m u; do if (( m < 12000 && u < 30 )); then printf '%s\n' "$i"; break; fi; done)"
  if [[ -z "$GPU" ]]; then printf '[D3-UNIVERSAL-DCT] waiting for GPU\n'; sleep 60; fi
done
printf '[D3-UNIVERSAL-DCT] selected GPU %s\n' "$GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" "$ROOT/workflows/direction3_universal.py" --config "$CONFIG"
