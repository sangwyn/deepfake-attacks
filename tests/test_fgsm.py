"""Focused CPU contract checks for the scheduled FGSM smoke task."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from attacklab.attack_api import invoke_attack, load_attack_module
from attacklab.preprocessing import from_uint8_image, preprocess_for


class FgsmSmokeTests(unittest.TestCase):
    @staticmethod
    def _image() -> np.ndarray:
        values = np.arange(32 * 32 * 3, dtype=np.uint32)
        return (values.reshape(32, 32, 3) % 251).astype(np.uint8)

    def test_dct_source_is_targeted_and_budgeted(self):
        module, function = load_attack_module("attacks.fgsm", "densenet121_dct")
        self.assertEqual(module.ATTACK_CONTRACT["source_model"], "densenet121_dct")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 128 * 128, 2))
        image = self._image()
        tensor = from_uint8_image(image, torch.device("cpu")).requires_grad_()
        loss = torch.nn.functional.cross_entropy(
            model(preprocess_for("densenet121_dct", tensor)), torch.tensor([0])
        )
        gradient = torch.autograd.grad(loss, tensor)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        output = invoke_attack(
            function, image, {"densenet121_dct": {"model": model}}, torch.device("cpu"),
            "densenet121_dct", 0, {"epsilon": 8 / 255}
        )
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        self.assertLessEqual(int(np.abs(output.astype(int) - image.astype(int)).max()), 8)

    def test_fixed_cpu_fixture_is_repeatable(self):
        _, function = load_attack_module("attacks.fgsm", "vit_b_16")
        torch.manual_seed(1)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        image = self._image()
        classifiers = {"vit_b_16": {"model": model}}
        parameters = {"epsilon": 8 / 255}
        first = invoke_attack(function, image, classifiers, torch.device("cpu"),
                              "vit_b_16", 0, parameters)
        second = invoke_attack(function, image, classifiers, torch.device("cpu"),
                               "vit_b_16", 0, parameters)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
