"""Focused CPU contract checks for MIG-COW."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from attacklab.attack_api import invoke_attack, load_attack_module
from attacks.mig_cow import _consensus_orthogonal


class MigCowTests(unittest.TestCase):
    def test_consensus_reconstruction_and_orthogonality(self):
        first = torch.tensor([[[[1.0, 2.0]]]])
        second = torch.tensor([[[[2.0, 1.0]]]])
        consensus, residuals = _consensus_orthogonal([first, second])
        self.assertTrue(torch.allclose(consensus + residuals[0], first))
        self.assertTrue(torch.allclose(consensus + residuals[1], second))
        self.assertAlmostEqual(float(torch.sum(residuals[0] * consensus)), 0.0, places=6)

    def test_seeded_shape_type_budget_and_repeatability(self):
        module, function = load_attack_module("attacks.mig_cow", "densenet121_dct")
        image = np.full((16, 16, 3), 127, dtype=np.uint8)
        model = nn.Sequential(nn.Flatten(), nn.Linear(128 * 128, 2))
        classifiers = {"densenet121_dct": {"model": model}}
        params = {"epsilon": 8 / 255, "step_size": 4 / 255, "iterations": 1,
                  "integrated_gradient_steps": 2}
        torch.manual_seed(4)
        first = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, params)
        torch.manual_seed(4)
        second = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, params)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, image.shape)
        self.assertLessEqual(int(np.abs(first.astype(int) - image.astype(int)).max()), 8)


if __name__ == "__main__":
    unittest.main()
