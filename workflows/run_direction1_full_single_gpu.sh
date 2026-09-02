#!/usr/bin/env bash
set -u

ROOT="/home/aiattacks/pengyang/deepfake-attacks"
LOG="$ROOT/logs/direction1_full_single_gpu.log"

mkdir -p "$ROOT/logs" "$ROOT/experiments/direction1"
exec >> "$LOG" 2>&1

printf '[FULL] started %s; opportunistic GPU selection\n' "$(date -Is)"

choose_gpu() {
    while IFS=', ' read -r index used util; do
        if (( used < 4000 && util < 15 )); then
            printf '%s\n' "$index"
            return 0
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits)
    return 1
}

wait_for_gpu() {
    while ! GPU="$(choose_gpu)"; do
        printf '[FULL] no GPU available, waiting\n'
        sleep 60
    done
    printf '[FULL] selected GPU %s\n' "$GPU"
}

run_one() {
    local name="$1"
    local config="$2"
    local result="$3"
    if [[ -f "$result" ]]; then
        printf '[FULL] skip %s, result already exists\n' "$name"
        return 0
    fi
    wait_for_gpu
    printf '[FULL] start %s %s\n' "$name" "$(date -Is)"
    CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" "$ROOT/evaluate.py" \
        --config "$ROOT/$config"
    printf '[FULL] done %s %s\n' "$name" "$(date -Is)"
}

run_one best configs/direction1/test2_full_best.yaml \
    "$ROOT/experiments/direction1/test2_full_results_best.json"
run_one dct65 configs/direction1/test2_full_dct65.yaml \
    "$ROOT/experiments/direction1/test2_full_results_dct65.json"

printf '[FULL] complete %s\n' "$(date -Is)"
