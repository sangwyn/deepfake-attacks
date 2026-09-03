"""Targeted projected-gradient descent in RGB pixel space."""

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
    "source_model": "densenet121_dct",
    "supported_source_models": ["vit_b_16", "densenet121_dct"],
    "description": "Targeted PGD with a uniform random start in the RGB Linf ball.",
}


def attack(
    image,
    classifiers,
    device,
    epsilon=8 / 255,
    step_size=2 / 255,
    iterations=10,
    random_start=True,
    source_model="vit_b_16",
    target_class=0,
):
    if source_model not in ATTACK_CONTRACT["supported_source_models"]:
        raise ValueError(f"pgd does not support source_model={source_model!r}")
    if target_class not in {0, 1}:
        raise ValueError("target_class must be 0 or 1")
    if epsilon <= 0 or step_size <= 0 or iterations < 1:
        raise ValueError("epsilon, step_size, and iterations must be positive")

    model = classifiers[source_model]["model"]
    original = from_uint8_image(image, device)
    if random_start:
        attacked = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        attacked = project_linf(attacked, original, epsilon).detach()
    else:
        attacked = original.clone()
    target = torch.tensor([target_class], device=device)

    for _ in range(iterations):
        attacked.requires_grad_()
        loss = F.cross_entropy(
            model(preprocess_for(source_model, attacked)), target
        )
        gradient = torch.autograd.grad(loss, attacked)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("PGD encountered a non-finite gradient")
        attacked = project_linf(
            attacked - step_size * gradient.sign(), original, epsilon
        ).detach()

    return to_uint8_image(attacked)
