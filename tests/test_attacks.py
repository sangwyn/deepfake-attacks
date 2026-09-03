import importlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from attacks.mig_cow import decompose_gradients


class ToyAdapter(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, image):
        score = self.scale * (image.mean(dim=(1, 2, 3), keepdim=False) - 0.5)
        return torch.stack((-score, score), dim=1)


@pytest.fixture
def image():
    return np.full((16, 16, 3), 192, dtype=np.uint8)


@pytest.fixture
def classifiers():
    return {
        "first": {"adapter": ToyAdapter()},
        "second": {"adapter": ToyAdapter(0.75)},
    }


@pytest.mark.parametrize(
    ("module_name", "parameters"),
    [
        ("fgsm", {"epsilon": 4 / 255}),
        ("pgd", {"epsilon": 4 / 255, "step_size": 2 / 255, "iterations": 2}),
        (
            "mi_di_fgsm",
            {
                "epsilon": 4 / 255,
                "step_size": 2 / 255,
                "iterations": 2,
                "diversity_probability": 0.0,
            },
        ),
        (
            "ssa_s2i_fgsm",
            {
                "epsilon": 4 / 255,
                "iterations": 1,
                "frequency_samples": 2,
                "transform_size": 16,
            },
        ),
        (
            "mig_cow",
            {
                "epsilon": 4 / 255,
                "iterations": 1,
                "integrated_gradient_steps": 2,
            },
        ),
        (
            "frequency_pgd",
            {
                "epsilon": 4 / 255,
                "step_size": 2 / 255,
                "iterations": 2,
                "band": "full",
            },
        ),
        (
            "isp_pgd",
            {
                "epsilon": 4 / 255,
                "step_size": 2 / 255,
                "iterations": 2,
            },
        ),
    ],
)
def test_attack_contract_and_budget(module_name, parameters, image, classifiers):
    function = importlib.import_module(f"attacks.{module_name}").attack
    attacked = function(
        image,
        classifiers,
        torch.device("cpu"),
        objective="targeted_fake_to_real",
        label=1,
        seed=7,
        **parameters,
    )
    assert attacked.dtype == np.uint8
    assert attacked.shape == image.shape
    assert np.abs(attacked.astype(int) - image.astype(int)).max() <= 4
    assert attacked.mean() < image.mean()


@pytest.mark.parametrize("module_name", ["pgd", "ssa_s2i_fgsm", "isp_pgd"])
def test_seeded_attacks_are_deterministic(module_name, image, classifiers):
    parameters = {"epsilon": 2 / 255, "iterations": 1, "seed": 19}
    if module_name == "ssa_s2i_fgsm":
        parameters.update(frequency_samples=2, transform_size=16)
    function = importlib.import_module(f"attacks.{module_name}").attack
    first = function(image, classifiers, torch.device("cpu"), **parameters)
    second = function(image, classifiers, torch.device("cpu"), **parameters)
    assert np.array_equal(first, second)


def test_consensus_residuals_are_orthogonal():
    gradients = [torch.tensor([[[[1.0, 0.0]]]]), torch.tensor([[[[0.0, 1.0]]]])]
    consensus, residuals = decompose_gradients(gradients, [1.0, 1.0])
    for residual in residuals:
        assert torch.dot(consensus.flatten(), residual.flatten()).abs() < 1e-6


@pytest.mark.parametrize(("value", "label", "direction"), [(48, 0, 1), (208, 1, -1)])
def test_untargeted_fgsm_moves_away_from_true_class(value, label, direction):
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    function = importlib.import_module("attacks.fgsm").attack
    attacked = function(
        image,
        {"source": {"adapter": ToyAdapter()}},
        torch.device("cpu"),
        epsilon=2 / 255,
        objective="untargeted",
        label=label,
    )
    assert np.sign(attacked.mean() - image.mean()) == direction
