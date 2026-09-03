"""Targeted Momentum Integrated Gradients with consensus/orthogonal weighting.

The source is explicit in the resolved experiment configuration.  The
leave-one-detector-out protocol therefore uses the selected detector for
generation and reserves the other configured detector for transfer scoring.
"""

import torch
import torch.nn.functional as F

from attacklab.preprocessing import from_uint8_image, preprocess_for, project_linf, to_uint8_image


ATTACK_CONTRACT = {
    "version": 1,
    "source_model": "densenet121_dct",
    "supported_source_models": ["vit_b_16", "densenet121_dct"],
    "description": "Targeted MIG-COW with integrated-gradient consensus weighting.",
}


def _integrated_gradients(model, source_model, point, baseline, target, steps):
    """Estimate the input gradient by a straight-line integrated gradient."""
    if steps < 1:
        raise ValueError("integrated-gradient steps must be positive")
    gradients = []
    for index in range(1, steps + 1):
        fraction = index / steps
        sample = (baseline + fraction * (point - baseline)).detach().requires_grad_()
        loss = F.cross_entropy(model(preprocess_for(source_model, sample)), target)
        gradient = torch.autograd.grad(loss, sample)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("MIG-COW encountered a non-finite integrated gradient")
        gradients.append(gradient)
    return torch.stack(gradients).mean(dim=0)


def _consensus_orthogonal(gradients):
    """Return consensus and residuals orthogonal to that consensus subspace."""
    stacked = torch.stack(gradients)
    consensus = stacked.mean(dim=0)
    basis = consensus.flatten()
    denominator = torch.dot(basis, basis).clamp_min(1e-12)
    residuals = []
    for gradient in gradients:
        residual = gradient - consensus
        coefficient = torch.dot(residual.flatten(), basis) / denominator
        residuals.append(residual - coefficient * consensus)
    return consensus, torch.stack(residuals)


def attack(image, classifiers, device, epsilon=8 / 255, step_size=None, iterations=10,
           momentum=1.0, integrated_gradient_steps=8, consensus_weight=1.0,
           orthogonal_weight=1.0, source_model="densenet121_dct", target_class=0):
    if source_model not in ATTACK_CONTRACT["supported_source_models"]:
        raise ValueError(f"mig_cow does not support source_model={source_model!r}")
    if source_model not in classifiers:
        raise ValueError("configured classifiers do not contain source_model")
    if target_class not in {0, 1} or epsilon <= 0 or iterations < 1:
        raise ValueError("invalid target_class, epsilon, or iterations")
    if step_size is None:
        step_size = epsilon / iterations
    if step_size <= 0 or momentum < 0 or integrated_gradient_steps < 1:
        raise ValueError("step_size, momentum, and integrated_gradient_steps must be positive")
    if consensus_weight < 0 or orthogonal_weight < 0:
        raise ValueError("consensus and orthogonal weights must be non-negative")

    model = classifiers[source_model]["model"]
    original = from_uint8_image(image, device)
    attacked = original.clone()
    accumulated = torch.zeros_like(original)
    target = torch.tensor([target_class], device=device)
    baseline = torch.zeros_like(original)
    for _ in range(iterations):
        gradient = _integrated_gradients(
            model, source_model, attacked, baseline, target, integrated_gradient_steps
        )
        consensus, residuals = _consensus_orthogonal([gradient])
        weighted = consensus_weight * consensus + orthogonal_weight * residuals[0]
        normalized = weighted / (weighted.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        accumulated = momentum * accumulated + normalized
        attacked = project_linf(attacked - step_size * accumulated.sign(), original, epsilon).detach()
    return to_uint8_image(attacked)
