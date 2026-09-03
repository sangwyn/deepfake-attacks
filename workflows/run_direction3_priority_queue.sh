#!/usr/bin/env bash
set -u
ROOT="/home/aiattacks/pengyang/deepfake-attacks"
mkdir -p "$ROOT/logs" "$ROOT/experiments/direction3"

run_when_available() {
  local label="$1"
  local config="$2"
  local log="$ROOT/logs/direction3_${label}.log"
  (
    exec >> "$log" 2>&1
    printf '[D3-QUEUE] %s waiting, started %s\n' "$label" "$(date -Is)"
    local gpu=""
    while [[ -z "$gpu" ]]; do
      gpu="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=', ' read -r i m u; do if (( m < 12000 && u < 30 )); then printf '%s\n' "$i"; break; fi; done)"
      if [[ -z "$gpu" ]]; then
        printf '[D3-QUEUE] %s no suitable GPU; retrying in 60s\n' "$label"
        sleep 60
      fi
    done
    printf '[D3-QUEUE] %s selected GPU %s\n' "$label" "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/.venv/bin/python" "$ROOT/workflows/direction3_ablation.py" --config "$config"
    local status=$?
    printf '[D3-QUEUE] %s finished status=%s at %s\n' "$label" "$status" "$(date -Is)"
    return "$status"
  )
}

run_when_available "residual_only_2_heldout" "$ROOT/configs/direction3_ablation_residual_only_2_heldout.yaml"
run_when_available "universal_plus_residual_2_heldout" "$ROOT/configs/direction3_ablation_universal_plus_residual_2_heldout.yaml"
