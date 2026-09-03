import hashlib
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import evaluate


class ToyAdapter(nn.Module):
    def forward(self, image):
        score = image.mean(dim=(1, 2, 3)) - 0.5
        return torch.stack((-score, score), dim=1)


class ZeroMetric(nn.Module):
    def forward(self, original, attacked):
        return torch.zeros(1, 1, 1, 1, device=original.device)


def test_end_to_end_identity_pipeline(tmp_path, monkeypatch):
    fake_root = tmp_path / "data" / "TEST" / "TEST_FAKE"
    real_root = tmp_path / "data" / "TEST" / "TEST_REAL"
    fake_root.mkdir(parents=True)
    real_root.mkdir(parents=True)
    image_path = fake_root / "sample.png"
    Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8)).save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "relative_path": "TEST_FAKE/sample.png",
                "label": 1,
                "sha256": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "vit_b_16.pth").write_bytes(b"test checkpoint")
    output = tmp_path / "output"
    report_path = output / "result.json"

    monkeypatch.setattr(evaluate, "load_detector", lambda *args, **kwargs: ToyAdapter())
    monkeypatch.setattr(evaluate, "build_lpips_metric", lambda device: ZeroMetric())
    report = evaluate.evaluate(
        {
            "original_root": str(tmp_path / "data" / "TEST"),
            "manifest": str(manifest),
            "models_dir": str(weights),
            "save_attacked_dir": str(output / "attacked"),
            "save_json": str(report_path),
            "attack": "identity",
            "objective": "targeted_fake_to_real",
            "include_labels": [1],
            "source_classifiers": ["vit_b_16"],
            "target_classifiers": ["vit_b_16"],
            "progress": False,
            "device": "cpu",
        }
    )

    assert report["images_evaluated"] == 1
    assert report["max_linf"] == 0
    assert report["per_classifier"]["vit_b_16"]["clean_accuracy"] == 1
    assert report["per_classifier"]["vit_b_16"]["conditional_asr"] == 0
    assert (output / "attacked" / "TEST_FAKE" / "sample.png").is_file()
    assert json.loads(report_path.read_text())["images_evaluated"] == 1


def test_config_paths_are_relative_to_yaml(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "run.yaml"
    config.write_text("original_root: ../data/TEST\nmodels_dir: ../weights\n")
    loaded = evaluate.load_cfg(config)
    assert loaded["original_root"] == str((tmp_path / "data" / "TEST").resolve())
    assert loaded["models_dir"] == str((tmp_path / "weights").resolve())
