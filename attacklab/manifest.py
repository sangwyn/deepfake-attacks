"""Deterministic, content-addressed dataset inventory generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .io import ContractError, atomic_write_json, atomic_write_text, sha256_file, utc_now


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _records_for_class(
    dataset_root: Path,
    class_name: str,
    relative_dir: str,
    label: int,
) -> Iterable[dict[str, Any]]:
    class_root = dataset_root / relative_dir
    if not class_root.is_dir():
        raise ContractError(f"Dataset class directory does not exist: {class_root}")
    paths = sorted(
        path
        for path in class_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ContractError(f"No images found in dataset class: {class_root}")
    split = class_name.split("_", 1)[0].lower()
    for path in paths:
        relative = path.relative_to(dataset_root).as_posix()
        digest = sha256_file(path)
        yield {
            "schema_version": 1,
            "sample_id": f"sha256:{digest}",
            "split": split,
            "class_name": class_name.lower(),
            "label": label,
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }


def build_manifests(
    dataset_root: Path,
    class_contract: dict[str, dict[str, Any]],
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not dataset_root.is_dir():
        raise ContractError(f"Dataset root does not exist: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {"TRAIN_REAL", "TRAIN_FAKE", "TEST_REAL", "TEST_FAKE"}
    if set(class_contract) != expected:
        raise ContractError("Class contract must define TRAIN/TEST x REAL/FAKE")

    catalog_entries: list[dict[str, Any]] = []
    seen_content: dict[str, str] = {}
    for class_name in sorted(class_contract):
        contract = class_contract[class_name]
        relative_dir = contract.get("path")
        label = contract.get("label")
        if not isinstance(relative_dir, str) or label not in {0, 1}:
            raise ContractError(f"Invalid class contract for {class_name}")
        output_path = output_dir / f"{class_name.lower()}.jsonl"
        if output_path.exists() and not overwrite:
            raise ContractError(f"Refusing to overwrite manifest: {output_path}")
        records = list(
            _records_for_class(dataset_root, class_name, relative_dir, int(label))
        )
        for record in records:
            digest = record["sha256"]
            previous = seen_content.get(digest)
            if previous is not None:
                raise ContractError(
                    f"Duplicate image content across dataset entries: {previous} and "
                    f"{record['relative_path']}"
                )
            seen_content[digest] = record["relative_path"]
        payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
        atomic_write_text(output_path, payload)
        catalog_entries.append(
            {
                "class_name": class_name,
                "label": label,
                "count": len(records),
                "manifest": output_path.name,
                "manifest_sha256": sha256_file(output_path),
            }
        )

    catalog = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset_root_recorded_for_audit": str(dataset_root.resolve()),
        "entries": catalog_entries,
        "total_count": sum(entry["count"] for entry in catalog_entries),
    }
    atomic_write_json(output_dir / "catalog.json", catalog)
    return catalog


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"Cannot read manifest {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ContractError(f"Invalid manifest record at {path}:{line_number}")
        required = {"sample_id", "relative_path", "sha256", "label", "class_name"}
        missing = required - record.keys()
        if missing:
            raise ContractError(
                f"Manifest record {line_number} is missing {', '.join(sorted(missing))}"
            )
        if record["label"] not in {0, 1}:
            raise ContractError(f"Invalid label at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ContractError(f"Manifest is empty: {path}")
    sample_ids = [record["sample_id"] for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ContractError(f"Manifest has duplicate sample ids: {path}")
    return records
