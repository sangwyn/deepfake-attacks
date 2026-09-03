import numpy as np
import torch

from experiments.direction5.scheduler import PRIMITIVES, attack


class ToyModel(torch.nn.Module):
    def forward(self, x):
        score = x.mean((1, 2, 3))
        return torch.stack((score, -score), dim=1)


def test_scheduler_is_deterministic_and_budgeted():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    classifiers = {"vit_b_16": {"model": ToyModel().eval()},
                   "densenet121_dct": {"model": ToyModel().eval()}}
    kwargs = dict(iterations=3, step_size=1 / 255, epsilon=4 / 255,
                  schedule_interval=1, seed=9)
    first = attack(image, classifiers, torch.device("cpu"), **kwargs)
    metadata = attack.last_metadata
    second = attack(image, classifiers, torch.device("cpu"), **kwargs)
    assert np.array_equal(first, second)
    assert np.max(np.abs(first.astype(int) - image.astype(int))) <= 4
    assert metadata["primitives"] == list(PRIMITIVES)
    assert len(metadata["history"]) == 3
    assert abs(sum(metadata["final_weights"]) - 1) < 1e-6
    assert "accepted" in metadata["history"][0]


def test_fixed_and_equal_controls_are_available():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    classifiers = {"vit_b_16": {"model": ToyModel().eval()},
                   "densenet121_dct": {"model": ToyModel().eval()}}
    attack(image, classifiers, torch.device("cpu"), iterations=1,
           scheduler_mode="fixed", fixed_primitive="global_noise")
    assert attack.last_metadata["final_weights"][-1] == 1
    attack(image, classifiers, torch.device("cpu"), iterations=1,
           scheduler_mode="equal")
    assert all(abs(value - 0.25) < 1e-6 for value in attack.last_metadata["final_weights"])


def test_adaptive_scheduler_records_candidate_acceptance():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    classifiers = {"vit_b_16": {"model": ToyModel().eval()},
                   "densenet121_dct": {"model": ToyModel().eval()}}
    attack(image, classifiers, torch.device("cpu"), iterations=2,
           schedule_interval=1, acceptance_margin=0.0, stagnation_patience=1)
    assert all("accepted_primitive" in record for record in attack.last_metadata["history"])
