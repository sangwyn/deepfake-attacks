"""Focused CPU contract checks for the scheduled I-FGSM smoke task."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from attacklab.attack_api import invoke_attack, load_attack_module
from attacklab.preprocessing import from_uint8_image, preprocess_for


class IfgsmSmokeTests(unittest.TestCase):
    @staticmethod
    def _image() -> np.ndarray:
        values = np.arange(32 * 32 * 3, dtype=np.uint32)
        return (values.reshape(32, 32, 3) % 251).astype(np.uint8)

    def test_vit_source_has_finite_targeted_gradient_and_budgeted_output(self):
        module, function = load_attack_module("attacks.ifgsm", "vit_b_16")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        image = self._image()
        tensor = from_uint8_image(image, torch.device("cpu")).requires_grad_()
        loss = torch.nn.functional.cross_entropy(
            model(preprocess_for("vit_b_16", tensor)), torch.tensor([0])
        )
        gradient = torch.autograd.grad(loss, tensor)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        output = invoke_attack(
            function,
            image,
            {"vit_b_16": {"model": model}},
            torch.device("cpu"),
            "vit_b_16",
            0,
            {"epsilon": 8 / 255, "step_size": 2 / 255, "iterations": 10},
        )
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        self.assertLessEqual(int(np.abs(output.astype(int) - image.astype(int)).max()), 8)

    def test_fixed_cpu_fixture_is_repeatable(self):
        _, function = load_attack_module("attacks.ifgsm", "vit_b_16")
        torch.manual_seed(1)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        image = self._image()
        classifiers = {"vit_b_16": {"model": model}}
        parameters = {"epsilon": 8 / 255, "step_size": 2 / 255, "iterations": 10}
        first = invoke_attack(function, image, classifiers, torch.device("cpu"),
                              "vit_b_16", 0, parameters)
        second = invoke_attack(function, image, classifiers, torch.device("cpu"),
                               "vit_b_16", 0, parameters)
        self.assertTrue(np.array_equal(first, second))

    def test_dct_source_has_finite_gradient_and_budgeted_output(self):
        _, function = load_attack_module("attacks.ifgsm", "densenet121_dct")
        torch.manual_seed(2)
        model = nn.Sequential(nn.Flatten(), nn.Linear(128 * 128, 2))
        image = self._image()
        tensor = from_uint8_image(image, torch.device("cpu")).requires_grad_()
        loss = torch.nn.functional.cross_entropy(
            model(preprocess_for("densenet121_dct", tensor)), torch.tensor([0])
        )
        gradient = torch.autograd.grad(loss, tensor)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        output = invoke_attack(
            function,
            image,
            {"densenet121_dct": {"model": model}},
            torch.device("cpu"),
            "densenet121_dct",
            0,
            {"epsilon": 8 / 255, "step_size": 2 / 255, "iterations": 2},
        )
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        self.assertLessEqual(int(np.abs(output.astype(int) - image.astype(int)).max()), 8)


if __name__ == "__main__":
    unittest.main()
