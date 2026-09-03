"""Focused CPU contract checks for MI-DI-FGSM."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from attacklab.attack_api import invoke_attack, load_attack_module


class MiDiFgsmSmokeTests(unittest.TestCase):
    def test_output_gradient_budget_and_seeded_repeatability(self):
        module, function = load_attack_module("attacks.mi_di_fgsm", "densenet121_dct")
        self.assertEqual(module.ATTACK_CONTRACT["source_model"], "densenet121_dct")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(128 * 128, 2))
        image = (np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3) % 251).astype(np.uint8)
        parameters = {
            "epsilon": 8 / 255, "step_size": 8 / 255 / 10, "iterations": 10,
            "momentum": 1.0, "input_diversity_probability": 0.5,
            "resize_min_fraction": 0.9, "resize_max_fraction": 1.0, "padding": "random",
        }
        classifiers = {"densenet121_dct": {"model": model}}
        torch.manual_seed(3)
        first = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, parameters)
        torch.manual_seed(3)
        second = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, parameters)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, image.shape)
        self.assertLessEqual(int(np.abs(first.astype(int) - image.astype(int)).max()), 8)

    def test_zero_additions_recover_iterative_fgsm_shape(self):
        _, function = load_attack_module("attacks.mi_di_fgsm", "densenet121_dct")
        model = nn.Sequential(nn.Flatten(), nn.Linear(128 * 128, 2))
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        output = invoke_attack(function, image, {"densenet121_dct": {"model": model}}, torch.device("cpu"),
                               "densenet121_dct", 0, {"epsilon": 8 / 255, "step_size": 2 / 255,
                               "iterations": 2, "momentum": 0, "input_diversity_probability": 0,
                               "resize_min_fraction": 0.9, "resize_max_fraction": 1.0, "padding": "random"})
        self.assertEqual(output.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
