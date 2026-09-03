import torch
import torch.nn as nn

from detectors.aide.preprocessing import AIDEPreprocessor
from detectors.registry import DetectorAdapter, _load_npr_state_dict


class ScalarModel(nn.Module):
    def forward(self, image):
        return image.mean(dim=tuple(range(1, image.ndim))).unsqueeze(1)


class TwoClassModel(nn.Module):
    def forward(self, image):
        score = image.mean(dim=tuple(range(1, image.ndim)))
        return torch.stack((-score, score), dim=1)


class FixedAIDEModel(nn.Module):
    def forward(self, image):
        return image.new_tensor([[0.25, 0.75]]).repeat(image.shape[0], 1)


def test_npr_logits_are_real_fake_and_differentiable():
    adapter = DetectorAdapter("npr", ScalarModel())
    image = torch.rand(1, 3, 48, 48, requires_grad=True)
    logits = adapter(image)
    assert logits.shape == (1, 2)
    assert logits[0, 0] == 0
    logits[0, 1].backward()
    assert image.grad is not None and image.grad.abs().sum() > 0


def test_dct_adapter_is_differentiable():
    adapter = DetectorAdapter("densenet121_dct", TwoClassModel())
    image = torch.rand(1, 3, 48, 48, requires_grad=True)
    logits = adapter(image)
    assert logits.shape == (1, 2)
    logits.sum().backward()
    assert image.grad is not None


def test_aide_preprocessor_builds_five_gradient_views():
    image = torch.rand(1, 3, 48, 48, requires_grad=True)
    views = AIDEPreprocessor()(image)
    assert views.shape == (1, 5, 3, 256, 256)
    views.sum().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0


def test_aide_logits_are_swapped_to_shared_real_fake_order():
    adapter = DetectorAdapter("aide", FixedAIDEModel(), aide_preprocessor=nn.Identity())
    logits = adapter(torch.rand(1, 3, 16, 16))
    assert torch.equal(logits, torch.tensor([[0.75, 0.25]]))


def test_npr_checkpoint_requires_and_strips_module_prefix(tmp_path):
    path = tmp_path / "npr.pth"
    torch.save({"model": {"module.fc1.weight": torch.ones(1, 2)}}, path)
    state = _load_npr_state_dict(path)
    assert list(state) == ["fc1.weight"]
