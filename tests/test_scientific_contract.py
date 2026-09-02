import tempfile
import unittest
from pathlib import Path
from unittest import mock

from attacklab import attack_api, config
from attacklab.artifacts import verify_run
from attacklab.io import ContractError
from attacklab.manifest import build_manifests


def experiment_config() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "unit-test-ifgsm",
        "scope": "smoke",
        "seed": 0,
        "server_config": "configs/pipeline/server.yaml",
        "dataset": {
            "manifest": "manifests/celebA/test_fake.jsonl",
            "source_label": 1,
            "selection": "manifest-order",
            "sample_limit": 8,
            "require_clean_correct": True,
        },
        "attack": {
            "name": "ifgsm",
            "module": "attacks.ifgsm",
            "source_model": "vit_b_16",
            "target_class": 0,
            "parameters": {
                "epsilon": 8 / 255,
                "step_size": 2 / 255,
                "iterations": 10,
            },
        },
        "models": ["vit_b_16", "densenet121_dct"],
        "constraint": {
            "norm": "linf",
            "epsilon": 8 / 255,
            "pixel_range": [0.0, 1.0],
            "output_format": "png",
        },
        "metrics": {"targeted_asr": True, "ssim": True, "lpips": True},
    }


class ScientificContractTests(unittest.TestCase):
    def test_experiment_contract_accepts_complete_config(self):
        with mock.patch.object(config, "load_yaml", return_value=experiment_config()):
            loaded = config.load_experiment_config(Path("unused.yaml"))
        self.assertEqual(loaded["attack"]["source_model"], "vit_b_16")
        self.assertRegex(loaded["config_sha256"], r"^[0-9a-f]{64}$")

    def test_experiment_contract_rejects_budget_mismatch(self):
        value = experiment_config()
        value["attack"]["parameters"]["epsilon"] = 4 / 255
        with mock.patch.object(config, "load_yaml", return_value=value):
            with self.assertRaisesRegex(ContractError, "must equal constraint"):
                config.load_experiment_config(Path("unused.yaml"))

    def test_manifest_generation_is_deterministic(self):
        classes = {
            "TRAIN_REAL": {"path": "TRAIN/TRAIN_REAL", "label": 0},
            "TRAIN_FAKE": {"path": "TRAIN/TRAIN_FAKE", "label": 1},
            "TEST_REAL": {"path": "TEST/TEST_REAL", "label": 0},
            "TEST_FAKE": {"path": "TEST/TEST_FAKE", "label": 1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for index, contract in enumerate(classes.values()):
                directory = dataset / contract["path"]
                directory.mkdir(parents=True)
                (directory / f"image-{index}.png").write_bytes(
                    f"unique-{index}".encode()
                )
            first = root / "first"
            second = root / "second"
            build_manifests(dataset, classes, first)
            build_manifests(dataset, classes, second)
            for name in classes:
                filename = f"{name.lower()}.jsonl"
                self.assertEqual(
                    (first / filename).read_bytes(), (second / filename).read_bytes()
                )

    def test_manifest_rejects_duplicate_content(self):
        classes = {
            "TRAIN_REAL": {"path": "TRAIN/TRAIN_REAL", "label": 0},
            "TRAIN_FAKE": {"path": "TRAIN/TRAIN_FAKE", "label": 1},
            "TEST_REAL": {"path": "TEST/TEST_REAL", "label": 0},
            "TEST_FAKE": {"path": "TEST/TEST_FAKE", "label": 1},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for contract in classes.values():
                directory = dataset / contract["path"]
                directory.mkdir(parents=True)
                (directory / "same.png").write_bytes(b"duplicate")
            with self.assertRaisesRegex(ContractError, "Duplicate image content"):
                build_manifests(dataset, classes, root / "output")

    def test_attack_template_declares_versioned_contract(self):
        _, function = attack_api.load_attack_module("attacks.template", "vit_b_16")
        image = object()
        returned = attack_api.invoke_attack(
            function, image, {}, None, "vit_b_16", 0, {}
        )
        self.assertIs(returned, image)

    def test_verifier_fails_closed_on_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = verify_run(Path(temporary), write_report=False)
        self.assertEqual(report["outcome"], "failed")
        self.assertIn("missing required artifacts", report["errors"][0])




class SchemaTests(unittest.TestCase):
    """The committed configs and their JSON Schemas must not drift apart."""

    def _validate(self, config_name: str, schema_name: str) -> None:
        import json

        import jsonschema
        import yaml

        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schemas" / schema_name).read_text(encoding="utf-8"))
        document = yaml.safe_load((root / config_name).read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(document)

    def test_server_config_matches_its_schema(self):
        self._validate("configs/pipeline/server.yaml", "server.schema.json")

    def test_smoke_experiment_matches_its_schema(self):
        self._validate("configs/experiments/ifgsm-smoke.yaml", "experiment.schema.json")

    def test_contract_schemas_reject_unknown_fields(self):
        """A closed contract is what keeps a worker from inventing a field."""

        import json

        root = Path(__file__).resolve().parents[1]
        for name in (
            "attack-status.schema.json",
            "review-status.schema.json",
            "experiment.schema.json",
            "server.schema.json",
        ):
            schema = json.loads((root / "schemas" / name).read_text(encoding="utf-8"))
            self.assertFalse(
                schema.get("additionalProperties", True),
                f"{name} must reject unknown fields",
            )

    def test_run_summary_schema_accepts_the_summary_the_runner_writes(self):
        import json

        import jsonschema

        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "schemas" / "run-summary.schema.json").read_text(encoding="utf-8")
        )
        summary = {
            "schema_version": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "experiment_id": "ifgsm-vit-test-fake-smoke-seed0",
            "scope": "smoke",
            "seed": 0,
            "samples_selected": 8,
            "samples_eligible": 6,
            "samples_evaluated": 6,
            "source_model": "vit_b_16",
            "target_class": 0,
            "epsilon": 8 / 255,
            "constraint_violations": 0,
            "mean_ssim": 0.98,
            "mean_lpips": 0.01,
            "per_model": {
                "vit_b_16": {
                    "clean_accuracy_on_selected": 0.75,
                    "targeted_asr_on_source_eligible": 1.0,
                    "successes": 6,
                    "denominator": 6,
                }
            },
            "timing": {
                "schema_version": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:08+00:00",
                "elapsed_seconds": 8.0,
                "measurement": (
                    "wall-clock run_experiment preparation and evaluation through summary "
                    "assembly; excludes artifact serialization and final verification"
                ),
            },
        }
        jsonschema.Draft202012Validator(schema).validate(summary)

    def test_runner_timing_uses_monotonic_elapsed_time(self):
        from attacklab import runner

        with mock.patch.object(runner, "utc_now", return_value="2026-01-01T00:00:08+00:00"):
            with mock.patch.object(runner.time, "monotonic", return_value=108.25):
                timing = runner._run_timing("2026-01-01T00:00:00+00:00", 100.0)

        self.assertEqual(timing["started_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(timing["finished_at"], "2026-01-01T00:00:08+00:00")
        self.assertEqual(timing["elapsed_seconds"], 8.25)


class TestLayoutTests(unittest.TestCase):
    """A test defined after unittest.main() never runs under direct execution."""

    def test_no_test_class_follows_the_main_guard(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "tests").glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            marker = 'if __name__ == "__main__":'
            index = text.find(marker)
            if index == -1:
                continue
            trailing = text[index:]
            self.assertNotIn(
                "\nclass ",
                trailing,
                f"{path.name} defines a test class after the __main__ guard; "
                "python tests/<file>.py would exit before reaching it",
            )
            self.assertNotIn(
                "\ndef test_",
                trailing,
                f"{path.name} defines a test function after the __main__ guard",
            )


if __name__ == "__main__":
    unittest.main()
