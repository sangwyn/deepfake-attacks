import torch

from detectors.npr.adapter import NPRAdapter
from detectors.npr.model import NPRResNet


def test_npr_shape_and_gradient():
    model = NPRResNet().eval()
    image = torch.rand(1, 3, 224, 224, requires_grad=True)
    logits = model(image)
    logits.sum().backward()
    assert logits.shape == (1, 1)
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0


def test_npr_logits_follow_wrapper_mapping():
    model = NPRResNet().eval()
    adapter = NPRAdapter(model, torch.device("cpu"))
    image = torch.rand(1, 3, 224, 224)
    score = model(image)
    logits = adapter(image)
    assert torch.equal(logits[:, 0], score[:, 0])
    assert torch.equal(logits[:, 1], -score[:, 0])
