"""Small shared helpers for pixel-space attacks."""

import numpy as np
import torch
import torch.nn.functional as F


def image_to_tensor(image, device):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("attack input must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("attack input must have shape [H, W, 3]")
    return (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def tensor_to_image(image):
    return (
        image.detach()[0]
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def project_linf(candidate, original, epsilon):
    delta = (candidate - original).clamp(-epsilon, epsilon)
    return (original + delta).clamp(0.0, 1.0)


def make_generator(device, seed):
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return torch.Generator(device=device).manual_seed(seed)


def _adapter(pack, name):
    if not isinstance(pack, dict) or "adapter" not in pack:
        raise ValueError(f"source detector {name!r} has no differentiable adapter")
    return pack["adapter"]


def classification_loss(logits, objective, label):
    if objective == "targeted_fake_to_real":
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)
    if objective == "untargeted":
        if label not in {0, 1}:
            raise ValueError("untargeted attacks require label 0 or 1")
        target = torch.full(
            (logits.shape[0],), label, dtype=torch.long, device=logits.device
        )
        return -F.cross_entropy(logits, target)
    raise ValueError(f"unsupported objective: {objective}")


def ensemble_loss(image, classifiers, objective, label, source_weights=None):
    if not classifiers:
        raise ValueError("at least one source detector is required")
    source_weights = source_weights or {}
    losses = []
    weights = []
    for name, pack in classifiers.items():
        weight = float(source_weights.get(name, 1.0))
        if weight < 0:
            raise ValueError("source detector weights must be non-negative")
        if weight == 0:
            continue
        losses.append(
            classification_loss(_adapter(pack, name)(image), objective, label)
        )
        weights.append(weight)
    if not losses:
        raise ValueError("at least one source detector weight must be positive")
    denominator = sum(weights)
    return sum(weight * loss for weight, loss in zip(weights, losses)) / denominator


def checked_gradient(loss, image, attack_name, *, retain_graph=False):
    gradient = torch.autograd.grad(loss, image, retain_graph=retain_graph)[0]
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"{attack_name} produced a non-finite gradient")
    if gradient.abs().sum() == 0:
        raise RuntimeError(f"{attack_name} produced a zero gradient")
    return gradient


def validate_steps(epsilon, step_size=None, iterations=None):
    epsilon = float(epsilon)
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if step_size is not None and float(step_size) <= 0:
        raise ValueError("step_size must be positive")
    if iterations is not None and (not isinstance(iterations, int) or iterations < 1):
        raise ValueError("iterations must be a positive integer")
