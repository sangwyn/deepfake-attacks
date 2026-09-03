"""Targeted momentum diverse-input iterative FGSM in RGB pixel space."""

import torch
import torch.nn.functional as F

from attacklab.preprocessing import (
    from_uint8_image,
    preprocess_for,
    project_linf,
    to_uint8_image,
)


ATTACK_CONTRACT = {
    "version": 1,
    "source_model": "vit_b_16",
    "supported_source_models": ["vit_b_16", "densenet121_dct"],
    "description": "Targeted MI-DI-FGSM with differentiable random resize-padding.",
}


def _diverse_input(
    x: torch.Tensor,
    probability: float,
    resize_min_fraction: float,
    resize_max_fraction: float,
) -> torch.Tensor:
    """Apply the paper-style resize and random padding while preserving shape."""
    if torch.rand((), device=x.device) >= probability:
        return x
    height, width = x.shape[-2:]
    fraction = torch.empty((), device=x.device).uniform_(
        resize_min_fraction, resize_max_fraction
    )
    resized_height = max(1, int(round(height * float(fraction))))
    resized_width = max(1, int(round(width * float(fraction))))
    resized = F.interpolate(
        x, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    pad_height = height - resized_height
    pad_width = width - resized_width
    top = int(torch.randint(0, pad_height + 1, (), device=x.device))
    left = int(torch.randint(0, pad_width + 1, (), device=x.device))
    return F.pad(resized, (left, pad_width - left, top, pad_height - top))


def attack(
    image,
    classifiers,
    device,
    epsilon=8 / 255,
    step_size=None,
    iterations=10,
    momentum=1.0,
    input_diversity_probability=0.5,
    resize_min_fraction=0.9,
    resize_max_fraction=1.0,
    padding="random",
    source_model="vit_b_16",
    target_class=0,
):
    if source_model not in ATTACK_CONTRACT["supported_source_models"]:
        raise ValueError(f"mi_di_fgsm does not support source_model={source_model!r}")
    if target_class not in {0, 1}:
        raise ValueError("target_class must be 0 or 1")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    if step_size is None:
        step_size = epsilon / iterations
    if step_size <= 0 or momentum < 0:
        raise ValueError("step_size must be positive and momentum non-negative")
    if not 0 <= input_diversity_probability <= 1:
        raise ValueError("input_diversity_probability must be in [0, 1]")
    if not 0 < resize_min_fraction <= resize_max_fraction <= 1:
        raise ValueError("resize fractions must satisfy 0 < min <= max <= 1")
    if padding != "random":
        raise ValueError("padding must be 'random'")

    model = classifiers[source_model]["model"]
    original = from_uint8_image(image, device)
    attacked = original.clone()
    accumulated = torch.zeros_like(original)
    target = torch.tensor([target_class], device=device)

    for _ in range(iterations):
        attacked.requires_grad_()
        diverse = _diverse_input(
            attacked, input_diversity_probability, resize_min_fraction, resize_max_fraction
        )
        loss = F.cross_entropy(model(preprocess_for(source_model, diverse)), target)
        gradient = torch.autograd.grad(loss, attacked)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("MI-DI-FGSM encountered a non-finite gradient")
        normalized = gradient / (gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        accumulated = momentum * accumulated + normalized
        attacked = project_linf(attacked - step_size * accumulated.sign(), original, epsilon).detach()

    return to_uint8_image(attacked)
