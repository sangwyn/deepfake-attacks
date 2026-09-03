"""Focused CPU contract checks for ensemble MI-FGSM with EoT."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from attacklab.attack_api import invoke_attack, load_attack_module


class _MeanModel(nn.Module):
    def forward(self, x):
        value = x.flatten(1).mean(dim=1)
        return torch.stack((value, -value), dim=1)


class EnsembleMiEotTests(unittest.TestCase):
    def test_contract_output_budget_and_seeded_repeatability(self):
        module, function = load_attack_module("attacks.ensemble_mi_fgsm_eot", "densenet121_dct")
        self.assertEqual(module.ATTACK_CONTRACT["source_model"], "densenet121_dct")
        torch.manual_seed(0)
        model = _MeanModel()
        image = (np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3) % 251).astype(np.uint8)
        parameters = {"epsilon": 8 / 255, "step_size": 8 / 255 / 2, "iterations": 2,
                      "momentum": 1.0, "ensemble_weighting": "uniform", "eot_samples": 3,
                      "transformations": ["resize", "crop", "jpeg-like"]}
        classifiers = {"densenet121_dct": {"model": model}, "vit_b_16": {"model": model}}
        torch.manual_seed(3)
        first = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, parameters)
        torch.manual_seed(3)
        second = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, parameters)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, image.shape)
        self.assertLessEqual(int(np.abs(first.astype(int) - image.astype(int)).max()), 8)

    def test_identity_eot_is_valid(self):
        _, function = load_attack_module("attacks.ensemble_mi_fgsm_eot", "densenet121_dct")
        model = _MeanModel()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        output = invoke_attack(function, image, {"vit_b_16": {"model": model},
                              "densenet121_dct": {"model": model}}, torch.device("cpu"),
                              "densenet121_dct", 0, {"epsilon": 8 / 255, "iterations": 1,
                              "step_size": 8 / 255, "eot_samples": 1,
                              "transformations": ["identity"]})
        self.assertEqual(output.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
