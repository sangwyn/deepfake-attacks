"""Summarize clean-correct eligible sets from a prediction manifest."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text())
    detectors = data["detectors"]
    rows = []
    for source in detectors:
        for target in detectors:
            for source_label, direction in ((0, "real_to_fake"),
                                            (1, "fake_to_real")):
                eligible = [
                    item for item in data["records"]
                    if item["label"] == source_label
                    and item["predictions"][source]["prediction"] == source_label
                    and item["predictions"][target]["prediction"] == source_label
                ]
                rows.append({
                    "source": source,
                    "target": target,
                    "direction": direction,
                    "source_label": source_label,
                    "target_label": 1 - source_label,
                    "eligible_count": len(eligible),
                    "eligible_ids": [item["id"] for item in eligible],
                })
    result = {
        "schema_version": 1,
        "manifest": str(Path(args.manifest).resolve()),
        "detectors": detectors,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
