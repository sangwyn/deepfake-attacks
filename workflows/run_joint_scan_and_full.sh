#!/usr/bin/env bash
set -u

ROOT="/home/aiattacks/pengyang/deepfake-attacks"
LOG="$ROOT/logs/joint_scan_and_full.log"
SUMMARY="$ROOT/experiments/direction1/joint_scan_summary.json"

mkdir -p "$ROOT/logs" "$ROOT/experiments/direction1"
exec >> "$LOG" 2>&1

printf '[ORCH] started %s\n' "$(date -Is)"

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
        printf '[ORCH] no suitable GPU, retrying in 60s\n'
        sleep 60
    done
    printf '[ORCH] selected GPU %s\n' "$GPU"
}

run_config() {
    local name="$1"
    local config="$2"
    local result="$ROOT/experiments/direction1/joint_middle_results_${name}.json"
    local log="$ROOT/logs/joint_middle_${name}.log"

    if [[ -f "$result" ]]; then
        printf '[ORCH] skip %s, result exists\n' "$name"
        return 0
    fi
    wait_for_gpu
    printf '[ORCH] start scan %s on GPU %s at %s\n' "$name" "$GPU" "$(date -Is)"
    CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" "$ROOT/evaluate.py" \
        --config "$ROOT/configs/direction1/$config" > "$log" 2>&1
    local status=$?
    printf '[ORCH] finished scan %s status=%s at %s\n' "$name" "$status" "$(date -Is)"
    return "$status"
}

run_config vit10_dct90 joint_middle_vit10_dct90.yaml || exit $?
run_config vit20_dct80 joint_middle_vit20_dct80.yaml || exit $?
run_config vit30_dct70 joint_middle_vit30_dct70.yaml || exit $?
run_config vit50_dct50 joint_middle_vit50_dct50.yaml || exit $?

printf '[ORCH] all scans complete, selecting candidate\n'
ROOT="$ROOT" SUMMARY="$SUMMARY" "$ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
summary_path = Path(os.environ["SUMMARY"])
rows = []
for path in sorted((root / "experiments/direction1").glob("joint_middle_results_*.json")):
    if path.name == summary_path.name:
        continue
    try:
        report = json.loads(path.read_text())
        vit = report["per_classifier"]["vit_b_16"]
        dct = report["per_classifier"]["densenet121_dct"]
        row = {
            "result": str(path),
            "final_score": report["final_score"],
            "vit_success": vit["attack_success"],
            "dct_success": dct["attack_success"],
            "ssim": vit["mean_ssim"],
            "lpips": vit["mean_lpips"],
        }
        # Prefer candidates meeting both model and quality floors.
        row["eligible"] = (
            row["vit_success"] >= 0.80
            and row["dct_success"] >= 0.40
            and row["ssim"] >= 0.93
            and row["lpips"] <= 0.10
        )
        rows.append(row)
    except (KeyError, json.JSONDecodeError):
        continue

if not rows:
    raise SystemExit("No valid scan results found")
eligible = [row for row in rows if row["eligible"]]
pool = eligible or rows
selected = max(pool, key=lambda row: (row["dct_success"] + row["vit_success"], row["final_score"]))
summary_path.write_text(json.dumps({"results": rows, "selected": selected, "eligible_count": len(eligible)}, indent=2))
print(json.dumps({"selected": selected, "eligible_count": len(eligible)}, indent=2))
PY

SELECTED_RESULT="$($ROOT/.venv/bin/python -c 'import json; print(json.load(open("'"$SUMMARY"'"))["selected"]["result"])')"
SELECTED_NAME="$(basename "$SELECTED_RESULT" .json)"
SELECTED_NAME="${SELECTED_NAME#joint_middle_results_}"

ELIGIBLE="$($ROOT/.venv/bin/python -c 'import json; print(json.load(open("'"$SUMMARY"'"))["selected"]["eligible"])')"
if [[ "$ELIGIBLE" != "True" ]]; then
    printf '[ORCH] no candidate met full-run floors; stopping after scan\n'
    printf '[ORCH] complete %s\n' "$(date -Is)"
    exit 0
fi

FULL_CONFIG="$ROOT/configs/direction1/test2_full_joint_selected.yaml"
FULL_RESULT="$ROOT/experiments/direction1/test2_full_joint_selected_results.json"
SELECTED_RESULT="$SELECTED_RESULT" FULL_CONFIG="$FULL_CONFIG" FULL_RESULT="$FULL_RESULT" "$ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

result = Path(os.environ["SELECTED_RESULT"])
report = json.loads(result.read_text())
name = result.stem.removeprefix("joint_middle_results_")
vit, dct = name.split("_dct")
vit_weight = int(vit.removeprefix("vit")) / 100
dct_weight = int(dct) / 100
config = f'''original_root: /home/aiattacks/pengyang/Test/group1/test2(whole)/AADD_2026_Test
exclude_root: /home/aiattacks/pengyang/Test/group1/test1
attack: dual_pgd
save_attacked_dir: /home/aiattacks/pengyang/deepfake-attacks/experiments/direction1/test2_full_joint_selected_attacked
models_dir: /home/aiattacks/pengyang/deepfake-attacks/weights
save_json: {os.environ["FULL_RESULT"]}
classifiers: [vit_b_16, densenet121_dct]
dct_log_scale: true
dct_resize_mode: bicubic
weights: {{vit_b_16: 1.0, densenet121_dct: 1.0}}
aggregate: sum
device: cuda
alpha: 0.5
seed: 0
epsilon: 0.03137254901960784
step_size: 0.00392156862745098
iterations: 40
target: 0
vit_weight: {vit_weight}
dct_weight: {dct_weight}
normalize_gradients: true
'''
Path(os.environ["FULL_CONFIG"]).write_text(config)
print(f"selected {name}: vit_weight={vit_weight}, dct_weight={dct_weight}")
PY

wait_for_gpu
printf '[ORCH] start full run with %s on GPU %s at %s\n' "$SELECTED_NAME" "$GPU" "$(date -Is)"
CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/.venv/bin/python" "$ROOT/evaluate.py" \
    --config "$FULL_CONFIG" > "$ROOT/logs/direction1_full_joint_selected.log" 2>&1
printf '[ORCH] full run status=%s at %s\n' "$?" "$(date -Is)"
printf '[ORCH] complete %s\n' "$(date -Is)"
