"""Focused CPU contract checks for SSA."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from attacklab.attack_api import invoke_attack, load_attack_module
from attacks.ssa_s2i_fgsm import _dct2, _idct2


class SsaTests(unittest.TestCase):
    def test_dct_round_trip_and_seeded_output(self):
        module, function = load_attack_module("attacks.ssa_s2i_fgsm", "densenet121_dct")
        image = (np.arange(16 * 18 * 3, dtype=np.uint32).reshape(16, 18, 3) % 251).astype(np.uint8)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
        self.assertTrue(torch.allclose(_idct2(_dct2(tensor)), tensor, atol=1e-5))
        model = nn.Sequential(nn.Flatten(), nn.Linear(128 * 128, 2))
        params = {"epsilon": 8 / 255, "step_size": 2 / 255, "iterations": 2,
                  "momentum": 1.0, "frequency_samples": 2,
                  "spectrum_noise_sigma": 0.01, "spectrum_amplitude_rho": 0.2,
                  "transform": "orthonormal-dct"}
        classifiers = {"densenet121_dct": {"model": model}}
        torch.manual_seed(9)
        first = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, params)
        torch.manual_seed(9)
        second = invoke_attack(function, image, classifiers, torch.device("cpu"), "densenet121_dct", 0, params)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, image.shape)
        self.assertLessEqual(int(np.abs(first.astype(int) - image.astype(int)).max()), 8)

    def test_identity_spectrum_is_valid(self):
        _, function = load_attack_module("attacks.ssa_s2i_fgsm", "densenet121_dct")
        model = nn.Sequential(nn.Flatten(), nn.Linear(128 * 128, 2))
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        output = invoke_attack(function, image, {"densenet121_dct": {"model": model}}, torch.device("cpu"),
                               "densenet121_dct", 0, {"iterations": 1, "frequency_samples": 1,
                               "spectrum_mask": "identity"})
        self.assertEqual(output.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
