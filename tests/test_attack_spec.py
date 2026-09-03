"""Attack specifications must stay honest about what this checkout can run."""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from attacklab import attack_spec
from attacklab.io import ContractError

ROOT = Path(__file__).resolve().parents[1]


def _server() -> dict:
    return {
        "assets": {
            "models": {"vit_b_16": {"filename": "a.pth"}, "densenet121_dct": {"filename": "b.pth"}}
        }
    }


class SpecLoadingTests(unittest.TestCase):
    def test_every_committed_spec_validates(self):
        specs = attack_spec.load_all_specs()
        self.assertIn("fgsm", specs)
        for identifier, spec in specs.items():
            self.assertEqual(spec["idea_id"], identifier)
            self.assertTrue(spec["parameters"]["epsilon"])

    def test_idea_id_must_match_the_file_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "renamed.yaml"
            spec = attack_spec.load_spec(ROOT / "specs" / "attacks" / "fgsm.yaml")
            path.write_text(yaml.safe_dump(spec), encoding="utf-8")
            with self.assertRaises(ContractError):
                attack_spec.load_spec(path)

    def test_schema_rejects_an_unknown_field(self):
        spec = attack_spec.load_spec(ROOT / "specs" / "attacks" / "fgsm.yaml")
        spec["unexpected"] = True
        with self.assertRaises(ContractError):
            attack_spec.validate_against_schema(spec)


class StepSizeTests(unittest.TestCase):
    def _spec(self, **parameters) -> dict:
        return {"idea_id": "unit", "parameters": parameters}

    def test_ratio_over_iterations(self):
        spec = self._spec(step_size="epsilon / iterations", iterations=10)
        self.assertAlmostEqual(attack_spec.resolve_step_size(spec, 8 / 255), 8 / 2550)

    def test_ratio_over_a_literal_divisor(self):
        spec = self._spec(step_size="epsilon / 4", iterations=10)
        self.assertAlmostEqual(attack_spec.resolve_step_size(spec, 8 / 255), 2 / 255)

    def test_single_step_uses_the_whole_budget(self):
        spec = self._spec(iterations=1)
        self.assertAlmostEqual(attack_spec.resolve_step_size(spec, 8 / 255), 8 / 255)

    def test_iterative_attack_without_a_step_size_is_refused(self):
        spec = self._spec(iterations=10)
        with self.assertRaises(ContractError):
            attack_spec.resolve_step_size(spec, 8 / 255)

    def test_explicit_number_is_used_as_is(self):
        spec = self._spec(step_size=0.004, iterations=10)
        self.assertAlmostEqual(attack_spec.resolve_step_size(spec, 8 / 255), 0.004)


class ReadinessTests(unittest.TestCase):
    def _fgsm(self) -> dict:
        return copy.deepcopy(
            attack_spec.load_spec(ROOT / "specs" / "attacks" / "fgsm.yaml")
        )

    def test_unconfigured_target_model_blocks(self):
        spec = self._fgsm()
        spec["evaluation"]["target_models"] = ["vit_b_16", "npr"]
        blockers, _ = attack_spec.readiness(spec, _server())
        self.assertTrue(any("npr" in line for line in blockers), blockers)

    def test_configured_targets_do_not_block(self):
        spec = self._fgsm()
        spec["evaluation"]["target_models"] = ["vit_b_16", "densenet121_dct"]
        blockers, _ = attack_spec.readiness(spec, _server())
        self.assertEqual(blockers, [])

    def test_missing_manifest_blocks(self):
        spec = self._fgsm()
        spec["evaluation"]["manifest"] = "manifests/celebA/absent.jsonl"
        spec["evaluation"]["target_models"] = ["vit_b_16"]
        blockers, _ = attack_spec.readiness(spec, _server())
        self.assertTrue(any("manifest" in line for line in blockers), blockers)

    def test_targeted_attack_needs_a_different_target_class(self):
        spec = self._fgsm()
        spec["evaluation"]["target_models"] = ["vit_b_16"]
        spec["evaluation"]["target_class"] = spec["evaluation"]["source_label"]
        blockers, _ = attack_spec.readiness(spec, _server())
        self.assertTrue(any("target != source" in line for line in blockers), blockers)

    def test_zero_retention_gate_is_warned_about(self):
        spec = self._fgsm()
        spec["evaluation"]["target_models"] = ["vit_b_16"]
        spec["retention_gate"]["minimum_gain_percentage_points"] = 0
        _, warnings = attack_spec.readiness(spec, _server())
        self.assertTrue(any("does not discriminate" in w for w in warnings), warnings)


if __name__ == "__main__":
    unittest.main()
