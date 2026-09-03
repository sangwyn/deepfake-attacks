"""Create a compact cross-direction summary from five TEST_FAKE result files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--output", default="experiments/five_direction_test_fake_summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for direction, filename in (("direction1", "test_fake_full.json"),
                                ("direction2", "test_fake_full.json"),
                                ("direction4", "test_fake_full.json"),
                                ("direction5", "test_fake_full.json")):
        path = root / direction / filename
        if not path.exists():
            rows.append({"direction": direction, "status": "pending", "path": str(path)})
            continue
        data = json.loads(path.read_text())
        per = data.get("per_classifier", {})
        rows.append({"direction": direction, "status": "complete", "images": data.get("images_evaluated"),
                     "vit_real": per.get("vit_b_16", {}).get("attack_success"),
                     "dct_real": per.get("densenet121_dct", {}).get("attack_success"),
                     "mean_ssim": per.get("vit_b_16", {}).get("mean_ssim"),
                     "mean_lpips": per.get("vit_b_16", {}).get("mean_lpips"),
                     "final_score": data.get("final_score"), "path": str(path)})
    d3 = root / "direction3" / "test_fake_full.json"
    if d3.exists():
        data = json.loads(d3.read_text())
        # Direction 3 has a dedicated workflow, so reconstruct the canonical
        # two-detector similarity-weighted sum from its per-image rows.
        local_score = sum(
            (int(row["vit_real"]) + int(row["dct_real"]))
            * (0.5 * row["ssim"] + 0.5 * (1.0 - row["lpips"]))
            for row in data["rows"]
        )
        rows.append({"direction": "direction3", "status": "complete", "images": data.get("images"),
                     "vit_real": data.get("vit_real_rate"), "dct_real": data.get("dct_real_rate"),
                     "mean_ssim": data.get("mean_ssim"), "mean_lpips": data.get("mean_lpips"),
                     "final_score": local_score, "seconds": data.get("seconds"), "path": str(d3)})
    else:
        rows.append({"direction": "direction3", "status": "pending", "path": str(d3)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": "celebA/TEST/TEST_FAKE", "images": 100,
                                  "rows": sorted(rows, key=lambda row: row["direction"])}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
