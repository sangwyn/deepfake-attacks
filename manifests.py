import hashlib
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
CLASS_DIRECTORIES = {"TEST_REAL": 0, "TEST_FAKE": 1}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_from_path(path: Path, dataset_root: Path) -> int:
    relative = path.relative_to(dataset_root)
    if not relative.parts or relative.parts[0] not in CLASS_DIRECTORIES:
        raise ValueError(
            f"Manifest path must be under TEST_REAL or TEST_FAKE: {relative}"
        )
    return CLASS_DIRECTORIES[relative.parts[0]]


def load_test_manifest(manifest_path: Path, dataset_root: Path) -> list[dict]:
    manifest_path = manifest_path.resolve()
    dataset_root = dataset_root.resolve()
    rows = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            relative_path = row.get("relative_path")
            expected_hash = row.get("sha256")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Manifest line {line_number}: missing sample_id")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"Manifest line {line_number}: missing relative_path")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"Manifest line {line_number}: invalid sha256")

            path = (dataset_root / relative_path).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest line {line_number}: path escapes dataset root"
                ) from exc
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise FileNotFoundError(f"Manifest image not found: {path}")

            label = _label_from_path(path, dataset_root)
            if "label" in row and row["label"] != label:
                raise ValueError(
                    f"Manifest line {line_number}: label disagrees with directory"
                )
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(f"Manifest line {line_number}: SHA-256 mismatch")
            rows.append({"sample_id": sample_id, "path": path, "label": label})

    if not rows:
        raise RuntimeError(f"No samples found in manifest: {manifest_path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest sample_id values must be unique")
    return rows
