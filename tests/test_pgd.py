"""Focused CPU contract checks for the PGD smoke task."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from attacklab.attack_api import invoke_attack, load_attack_module
from attacklab.preprocessing import from_uint8_image, preprocess_for


class PgdSmokeTests(unittest.TestCase):
    def test_dct_source_is_finite_budgeted_and_repeatable(self):
        module, function = load_attack_module("attacks.pgd", "densenet121_dct")
        self.assertEqual(module.ATTACK_CONTRACT["source_model"], "densenet121_dct")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 128 * 128, 2))
        image = (np.arange(32 * 32 * 3, dtype=np.uint32).reshape(32, 32, 3) % 251).astype(np.uint8)
        tensor = from_uint8_image(image, torch.device("cpu")).requires_grad_()
        loss = torch.nn.functional.cross_entropy(
            model(preprocess_for("densenet121_dct", tensor)), torch.tensor([0])
        )
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(loss, tensor)[0]).all()))
        parameters = {"epsilon": 8 / 255, "step_size": 2 / 255, "iterations": 10, "random_start": True}
        torch.manual_seed(1)
        first = invoke_attack(function, image, {"densenet121_dct": {"model": model}}, torch.device("cpu"), "densenet121_dct", 0, parameters)
        torch.manual_seed(1)
        second = invoke_attack(function, image, {"densenet121_dct": {"model": model}}, torch.device("cpu"), "densenet121_dct", 0, parameters)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, image.shape)
        self.assertLessEqual(int(np.abs(first.astype(int) - image.astype(int)).max()), 8)


if __name__ == "__main__":
    unittest.main()
