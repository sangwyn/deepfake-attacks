#!/usr/bin/env bash
set -u

ROOT="/home/aiattacks/pengyang/deepfake-attacks"
LOG="$ROOT/logs/direction3_lowres64.log"
CONFIG="$ROOT/configs/direction3_universal_lowres64.yaml"
mkdir -p "$ROOT/logs" "$ROOT/experiments/direction3"
exec >> "$LOG" 2>&1

printf '[D3-LOWRES64] started %s\n' "$(date -Is)"
choose_gpu() {
    while IFS=', ' read -r index used util; do
        if (( used < 4000 && util < 15 )); then printf '%s\n' "$index"; return 0; fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    return 1
}
while ! GPU="$(choose_gpu)"; do printf '[D3-LOWRES64] waiting for GPU\n'; sleep 60; done
printf '[D3-LOWRES64] selected GPU %s at %s\n' "$GPU" "$(date -Is)"
CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" "$ROOT/workflows/direction3_universal.py" --config "$CONFIG"
status=$?
printf '[D3-LOWRES64] finished status=%s at %s\n' "$status" "$(date -Is)"
exit "$status"
