#!/usr/bin/env python3
"""Render the transfer matrix from completed runs.

Which direction a number belongs to is decided by two fields: `source_model` in
the summary is the detector the gradient came from, and each key under
`per_model` is a detector the result was measured on. Same name means white
box; different names mean transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "tracking" / "runs"


def completed_runs(runs_root: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for verification in sorted(runs_root.glob("*/*/attempt-*/verification.json")):
        attempt = verification.parent
        summary_path = attempt / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            verdict = json.loads(verification.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found.append(
            {
                "run": str(attempt.relative_to(runs_root)),
                "verdict": verdict.get("outcome"),
                "summary": summary,
            }
        )
    return found


def render(records: list[dict[str, Any]], only_passed: bool) -> None:
    if not records:
        print("Готовых прогонов нет.")
        return
    header = (
        f"{'эксперимент':<40}{'источник':<18}{'вердикт':<9}"
        f"{'цель':<18}{'роль':<11}{'ASR':>7}{'  n':>6}{'SSIM':>8}{'LPIPS':>8}"
    )
    print(header)
    print("-" * len(header))
    for record in sorted(records, key=lambda item: item["summary"].get("experiment_id", "")):
        summary = record["summary"]
        if only_passed and record["verdict"] != "passed":
            continue
        source = summary.get("source_model", "?")
        for target, metrics in sorted(summary.get("per_model", {}).items()):
            role = "белый ящик" if target == source else "перенос"
            print(
                f"{summary.get('experiment_id', '?'):<40}{source:<18}"
                f"{str(record['verdict']):<9}{target:<18}{role:<11}"
                f"{metrics.get('targeted_asr_on_source_eligible', 0):>7.3f}"
                f"{metrics.get('denominator', 0):>6}"
                f"{summary.get('mean_ssim') or 0:>8.4f}"
                f"{summary.get('mean_lpips') or 0:>8.4f}"
            )
        violations = summary.get("constraint_violations")
        if violations:
            print(f"  ВНИМАНИЕ: нарушений бюджета {violations}")


def matrix(records: list[dict[str, Any]]) -> None:
    """One transfer matrix per attack, sources down, targets across."""

    by_attack: dict[str, dict[tuple[str, str], float]] = {}
    for record in records:
        if record["verdict"] != "passed":
            continue
        summary = record["summary"]
        attack = str(summary.get("experiment_id", "?")).split("-")[0]
        source = summary.get("source_model", "?")
        for target, metrics in summary.get("per_model", {}).items():
            by_attack.setdefault(attack, {})[(source, target)] = metrics.get(
                "targeted_asr_on_source_eligible", 0.0
            )

    for attack, cells in sorted(by_attack.items()):
        targets = sorted({target for _, target in cells})
        sources = sorted({source for source, _ in cells})
        print(f"\n{attack}: строки — источник градиента, столбцы — где измеряли")
        print(f"  {'':<18}" + "".join(f"{t:>20}" for t in targets))
        for source in sources:
            cell_text = "".join(
                f"{cells[(source, t)]:>20.3f}" if (source, t) in cells else f"{'—':>20}"
                for t in targets
            )
            print(f"  {source:<18}{cell_text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--all", action="store_true", help="включая непройденные")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    records = completed_runs(args.runs_root)
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    render(records, only_passed=not args.all)
    matrix(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
