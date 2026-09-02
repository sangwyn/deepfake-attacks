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
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            # Match the guard as a top-level statement, not the same text
            # quoted inside a string literal such as this test's own source.
            positions = [
                number
                for number, line in enumerate(lines)
                if line.startswith('if __name__ ==')
            ]
            if not positions:
                continue
            trailing = "".join(lines[positions[-1]:])
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


class SpecScheduleAgreementTests(unittest.TestCase):
    """A scheduled attack with no specification, or the reverse, misleads both
    the worker that must implement it and the reviewer that gates it."""

    @staticmethod
    def _root() -> Path:
        return Path(__file__).resolve().parents[1]

    def _scheduled_attacks(self) -> set[str]:
        import yaml

        manifest = yaml.safe_load(
            (self._root() / "CAMPAIGN.yaml").read_text(encoding="utf-8")
        )
        return {
            task["attack"]
            for task in manifest["profiles"]["development"]["tasks"]
            if task.get("kind", "attack") == "attack"
        }

    def _specified_attacks(self) -> set[str]:
        # specs/attacks is the source of truth for what an attack is.
        return {path.stem for path in (self._root() / "specs" / "attacks").glob("*.yaml")}

    def test_every_scheduled_attack_has_a_specification(self):
        missing = self._scheduled_attacks() - self._specified_attacks()
        self.assertEqual(
            missing,
            set(),
            "CAMPAIGN.yaml schedules attacks with no file in specs/attacks",
        )

    def test_every_specified_attack_is_scheduled(self):
        extra = self._specified_attacks() - self._scheduled_attacks()
        self.assertEqual(
            extra,
            set(),
            "specs/attacks describes attacks the campaign never runs; "
            "a reviewer could apply their gates to nothing",
        )


def _write_complete_run(root: Path, *, timing: dict | None = None,
                        linf: float | None = None) -> Path:
    """Build a run directory the verifier should accept, so individual checks
    can be removed one at a time and shown to fail."""

    import hashlib
    import json

    import yaml

    run_dir = root / "attempt-0001"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = experiment_config()
    epsilon = float(config["constraint"]["epsilon"])
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "resolved_server_config.yaml").write_text("placeholder: true\n", encoding="utf-8")
    (run_dir / "manifest.snapshot.jsonl").write_text(
        json.dumps({"sample_id": "sha256:aa", "label": 1}) + "\n", encoding="utf-8"
    )
    (run_dir / "selection.jsonl").write_text(
        json.dumps({"sample_id": "sha256:aa", "eligible": True}) + "\n", encoding="utf-8"
    )

    output = run_dir / "adversarial.png"
    output.write_bytes(b"not really a png, only hashed")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (run_dir / "per_sample_metrics.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sha256:aa",
                "linf": epsilon if linf is None else linf,
                "ssim": 0.99,
                "lpips": 0.01,
                "clean_predictions": {"vit_b_16": 1},
                "adversarial_predictions": {"vit_b_16": 0},
                "output": {"path": str(output), "sha256": digest, "format": "png"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    violations = 0 if (linf is None or linf <= epsilon + 1 / 255 + 1e-12) else 1
    (run_dir / "norm_audit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "epsilon": epsilon,
                "violations": violations,
                "samples": 1,
            }
        ),
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "samples_selected": 1,
        "samples_eligible": 1,
        "samples_evaluated": 1,
        "source_model": "vit_b_16",
        "target_class": 0,
        "epsilon": epsilon,
        "constraint_violations": violations,
        "per_model": {"vit_b_16": {"successes": 1, "denominator": 1}},
    }
    if timing is not None:
        summary["timing"] = timing
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def sha(name: str) -> str:
        return hashlib.sha256((run_dir / name).read_bytes()).hexdigest()

    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inputs": {
                    "config": {"sha256": sha("resolved_config.yaml")},
                    "manifest": {"sha256": sha("manifest.snapshot.jsonl")},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifacts.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    return run_dir


VALID_TIMING = {
    "schema_version": 1,
    "started_at": "2026-09-03T00:00:00+00:00",
    "finished_at": "2026-09-03T00:01:00+00:00",
    "elapsed_seconds": 60.0,
    "measurement": (
        "wall-clock run_experiment preparation and evaluation through summary "
        "assembly; excludes artifact serialization and final verification"
    ),
}


class VerifierBehaviourTests(unittest.TestCase):
    """Exercise the branch that actually decides whether a run counts."""

    def test_complete_run_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _write_complete_run(Path(temporary), timing=VALID_TIMING)
            report = verify_run(run_dir, write_report=False)
        self.assertEqual(report["outcome"], "passed", report["errors"])

    def test_run_without_timing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _write_complete_run(Path(temporary), timing=None)
            report = verify_run(run_dir, write_report=False)
        self.assertEqual(report["outcome"], "failed")
        self.assertTrue(
            any("timing" in error for error in report["errors"]), report["errors"]
        )

    def test_run_with_negative_elapsed_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            broken = dict(VALID_TIMING, elapsed_seconds=-1.0)
            run_dir = _write_complete_run(Path(temporary), timing=broken)
            report = verify_run(run_dir, write_report=False)
        self.assertEqual(report["outcome"], "failed")

    def test_budget_violation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _write_complete_run(
                Path(temporary), timing=VALID_TIMING, linf=0.5
            )
            report = verify_run(run_dir, write_report=False)
        self.assertEqual(report["outcome"], "failed")
        self.assertTrue(
            any("constraint violation" in e.lower() for e in report["errors"]),
            report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
