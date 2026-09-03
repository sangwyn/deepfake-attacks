"""Momentum integrated gradients with consensus/orthogonal weighting."""

import torch

from ._utils import (
    _adapter,
    checked_gradient,
    classification_loss,
    image_to_tensor,
    project_linf,
    tensor_to_image,
    validate_steps,
)


def _source_entries(classifiers, source_weights):
    if not classifiers:
        raise ValueError("at least one source detector is required")
    source_weights = source_weights or {}
    entries = []
    for name, pack in classifiers.items():
        weight = float(source_weights.get(name, 1.0))
        if weight < 0:
            raise ValueError("source detector weights must be non-negative")
        if weight:
            entries.append((name, _adapter(pack, name), weight))
    if not entries:
        raise ValueError("at least one source detector weight must be positive")
    return entries


def _integrated_gradient(adapter, point, baseline, objective, label, steps, name):
    gradients = []
    for index in range(1, steps + 1):
        fraction = index / steps
        sample = (
            (baseline + fraction * (point - baseline)).detach().requires_grad_(True)
        )
        loss = classification_loss(adapter(sample), objective, label)
        gradients.append(checked_gradient(loss, sample, f"MIG-COW/{name}"))
    return torch.stack(gradients).mean(dim=0)


def decompose_gradients(gradients, weights):
    """Return a weighted consensus and residuals orthogonal to it."""
    denominator = sum(weights)
    consensus = sum(w * g for w, g in zip(weights, gradients)) / denominator
    basis = consensus.flatten()
    squared_norm = torch.dot(basis, basis).clamp_min(1e-12)
    residuals = []
    for gradient in gradients:
        difference = gradient - consensus
        projection = torch.dot(difference.flatten(), basis) / squared_norm
        residuals.append(difference - projection * consensus)
    return consensus, residuals


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    step_size=None,
    iterations=10,
    momentum=1.0,
    integrated_gradient_steps=8,
    consensus_weight=1.0,
    orthogonal_weight=1.0,
    objective="targeted_fake_to_real",
    label=None,
    source_weights=None,
    seed=0,
):
    """Combine common and model-specific directions without using a target model."""
    del seed
    if step_size is None:
        step_size = float(epsilon) / iterations
    validate_steps(epsilon, step_size, iterations)
    if momentum < 0:
        raise ValueError("momentum must be non-negative")
    if not isinstance(integrated_gradient_steps, int) or integrated_gradient_steps < 1:
        raise ValueError("integrated_gradient_steps must be a positive integer")
    if consensus_weight < 0 or orthogonal_weight < 0:
        raise ValueError("component weights must be non-negative")
    if consensus_weight + orthogonal_weight == 0:
        raise ValueError("at least one component weight must be positive")

    entries = _source_entries(classifiers, source_weights)
    names, adapters, weights = zip(*entries)
    original = image_to_tensor(image, device)
    baseline = torch.zeros_like(original)
    attacked = original.clone()
    velocity = torch.zeros_like(attacked)
    for _ in range(iterations):
        gradients = [
            _integrated_gradient(
                adapter,
                attacked,
                baseline,
                objective,
                label,
                integrated_gradient_steps,
                name,
            )
            for name, adapter in zip(names, adapters)
        ]
        consensus, residuals = decompose_gradients(gradients, weights)
        # Fixed rank weights avoid consulting any held-out detector.  With one
        # source its residual is exactly zero, recovering momentum IG.
        residual = sum(
            (index + 1) * weight * value
            for index, (weight, value) in enumerate(zip(weights, residuals))
        ) / sum((index + 1) * weight for index, weight in enumerate(weights))
        gradient = (
            float(consensus_weight) * consensus + float(orthogonal_weight) * residual
        )
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise RuntimeError("MIG-COW produced an invalid combined gradient")
        gradient = gradient / gradient.abs().mean(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(1e-12)
        velocity = float(momentum) * velocity + gradient
        attacked = project_linf(
            attacked - float(step_size) * velocity.sign(), original, epsilon
        ).detach()
    return tensor_to_image(attacked)
