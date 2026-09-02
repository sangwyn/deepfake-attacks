#!/usr/bin/env python3
"""Stable preflight entry point referenced by agent instructions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from attacklab.config import PROJECT_ROOT
from attacklab.io import ContractError, atomic_write_json, project_relative_path
from attacklab.preflight import print_preflight, run_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the AADD server contract")
    parser.add_argument(
        "--config", default="configs/pipeline/server.yaml", help="project-relative YAML"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also validate exact packages, CUDA, and LPIPS calibration",
    )
    parser.add_argument("--output", help="optional project-relative JSON report")
    args = parser.parse_args(argv)
    try:
        config_path = project_relative_path(PROJECT_ROOT, args.config)
        report = run_preflight(config_path, deep=args.strict)
        if args.output:
            atomic_write_json(project_relative_path(PROJECT_ROOT, args.output), report)
        print_preflight(report)
        return 0 if report["status"] == "pass" else 2
    except (ContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
