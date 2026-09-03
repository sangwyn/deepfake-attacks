"""Targeted or untargeted projected-gradient descent."""

import torch

from ._utils import (
    checked_gradient,
    ensemble_loss,
    image_to_tensor,
    make_generator,
    project_linf,
    tensor_to_image,
    validate_steps,
)


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    step_size=0.5 / 255,
    iterations=40,
    random_start=False,
    objective="targeted_fake_to_real",
    label=None,
    source_weights=None,
    seed=0,
):
    validate_steps(epsilon, step_size, iterations)
    original = image_to_tensor(image, device)
    attacked = original.clone()
    if random_start:
        generator = make_generator(device, seed)
        noise = torch.empty_like(attacked).uniform_(
            -float(epsilon), float(epsilon), generator=generator
        )
        attacked = project_linf(attacked + noise, original, epsilon).detach()
    for _ in range(iterations):
        attacked.requires_grad_(True)
        loss = ensemble_loss(attacked, classifiers, objective, label, source_weights)
        gradient = checked_gradient(loss, attacked, "PGD")
        attacked = project_linf(
            attacked - float(step_size) * gradient.sign(), original, epsilon
        ).detach()
    return tensor_to_image(attacked)
