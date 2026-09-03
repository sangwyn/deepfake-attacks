import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from attacks.fgsm import attack as fgsm  # noqa: E402
from detectors import DetectorAdapter  # noqa: E402
from evaluate import get_metric_device, resolve_experiment_models  # noqa: E402
from manifests import load_test_manifest  # noqa: E402


class MeanBinaryModel(nn.Module):
    def forward(self, x):
        score = x.mean(dim=tuple(range(1, x.ndim)))
        return torch.stack([score, -score], dim=1)


class MeanNprModel(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3), keepdim=True).reshape(-1, 1)


class CampaignExtensionTests(unittest.TestCase):
    def test_lpips_uses_cpu_by_default_to_preserve_attack_gpu_memory(self):
        self.assertEqual(get_metric_device(), torch.device("cpu"))

    def test_reuse_cell_resolves_sources_before_loading_target_only(self):
        sources, targets, loaded = resolve_experiment_models({
            "source_classifiers": ["vit_b_16"],
            "target_classifiers": ["densenet121_dct"],
            "reuse_attacked_dir": "/tmp/generated-tree",
        })

        self.assertEqual(sources, ["vit_b_16"])
        self.assertEqual(targets, ["densenet121_dct"])
        self.assertEqual(loaded, ["densenet121_dct"])

    def test_manifest_verifies_directory_label_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "TEST_FAKE" / "sample.png"
            image.parent.mkdir()
            image.write_bytes(b"image bytes")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({
                    "sample_id": "fake/sample",
                    "relative_path": "TEST_FAKE/sample.png",
                    "label": 1,
                    "sha256": digest,
                }) + "\n",
                encoding="utf-8",
            )

            rows = load_test_manifest(manifest, root)

            self.assertEqual(rows[0]["sample_id"], "fake/sample")
            self.assertEqual(rows[0]["label"], 1)
            self.assertEqual(rows[0]["path"], image)

    def test_npr_adapter_produces_real_fake_logits_and_input_gradient(self):
        adapter = DetectorAdapter("npr", MeanNprModel())
        image = torch.rand(2, 3, 224, 224, requires_grad=True)

        logits = adapter(image)
        gradient = torch.autograd.grad(logits[:, 1].sum(), image)[0]

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.equal(logits[:, 0], torch.zeros(2)))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_fgsm_uses_selected_adapter_and_respects_budget(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        classifiers = {"source": {"adapter": MeanBinaryModel()}}

        adversarial = fgsm(
            image,
            classifiers,
            torch.device("cpu"),
            epsilon=4 / 255,
        )

        delta = np.abs(adversarial.astype(np.int16) - image.astype(np.int16))
        self.assertGreater(delta.max(), 0)
        self.assertLessEqual(delta.max(), 4)

    def test_fgsm_accepts_campaign_seed_and_is_deterministic(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        classifiers = {"source": {"adapter": MeanBinaryModel()}}

        first = fgsm(
            image,
            classifiers,
            torch.device("cpu"),
            epsilon=4 / 255,
            seed=7,
        )
        second = fgsm(
            image,
            classifiers,
            torch.device("cpu"),
            epsilon=4 / 255,
            seed=7,
        )

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
