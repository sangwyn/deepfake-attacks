import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from manifests import discover_test_samples, load_test_manifest


def _image(path, value):
    array = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_discovery_derives_labels_and_sorts(tmp_path):
    real = tmp_path / "TEST_REAL"
    fake = tmp_path / "TEST_FAKE"
    real.mkdir()
    fake.mkdir()
    _image(real / "b.png", 10)
    _image(fake / "a.png", 240)
    samples = discover_test_samples(tmp_path)
    assert [(sample["sample_id"], sample["label"]) for sample in samples] == [
        ("TEST_FAKE/a.png", 1),
        ("TEST_REAL/b.png", 0),
    ]


def test_manifest_verifies_hash_and_directory_label(tmp_path):
    (tmp_path / "TEST_REAL").mkdir()
    fake = tmp_path / "TEST_FAKE"
    fake.mkdir()
    path = fake / "sample.png"
    digest = _image(path, 230)
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "sample_id": "fixed-id",
        "relative_path": "TEST_FAKE/sample.png",
        "label": 1,
        "sha256": digest,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    samples = load_test_manifest(manifest, tmp_path)
    assert samples[0]["label"] == 1

    row["label"] = 0
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label disagrees"):
        load_test_manifest(manifest, tmp_path)
