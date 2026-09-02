"""Command-line entry points used by humans, OpenCode, and the GPU queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import verify_run
from .config import PROJECT_ROOT, load_server_config
from .io import ContractError, atomic_write_json, project_relative_path, utc_now
from .manifest import build_manifests
from .preflight import print_preflight, run_preflight
from .runner import run_experiment
from .status import load_and_validate_status


def _project_path(raw: str) -> Path:
    return project_relative_path(PROJECT_ROOT, raw)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="AADD reproducible experiment control layer")
    commands = result.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="validate paths, assets, and environment")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--deep", action="store_true")
    preflight.add_argument("--output")

    manifests = commands.add_parser("build-manifests", help="freeze CelebA inventory")
    manifests.add_argument("--config", required=True)
    manifests.add_argument("--output-dir", required=True)
    manifests.add_argument("--overwrite", action="store_true")

    run = commands.add_parser("run", help="execute one immutable experiment attempt")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir", required=True)

    verify = commands.add_parser("verify", help="validate a completed attempt")
    verify.add_argument("--run-dir", required=True)

    status = commands.add_parser("validate-status", help="validate agent status JSON")
    status.add_argument("--kind", choices=("attack", "review"), required=True)
    status.add_argument("--status-file", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            config_path = _project_path(args.config)
            report = run_preflight(config_path, deep=args.deep)
            if args.output:
                atomic_write_json(_project_path(args.output), report)
            print_preflight(report)
            return 0 if report["status"] == "pass" else 2

        if args.command == "build-manifests":
            server = load_server_config(_project_path(args.config))
            catalog = build_manifests(
                Path(server["dataset"]["celeb_a_root"]),
                server["dataset"]["classes"],
                _project_path(args.output_dir),
                overwrite=args.overwrite,
            )
            print(json.dumps(catalog, indent=2, sort_keys=True))
            return 0

        if args.command == "run":
            config_path = _project_path(args.config)
            run_dir = _project_path(args.run_dir)
            try:
                report = run_experiment(config_path, run_dir)
            except Exception as exc:
                run_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    run_dir / "failure.json",
                    {
                        "schema_version": 1,
                        "failed_at": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        if args.command == "verify":
            report = verify_run(_project_path(args.run_dir), write_report=True)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["outcome"] == "passed" else 2

        if args.command == "validate-status":
            value = load_and_validate_status(_project_path(args.status_file), args.kind)
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
    except (ContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
