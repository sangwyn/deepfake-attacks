"""Targeted one-step FGSM in RGB pixel space."""

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
    "supported_source_models": ["vit_b_16", "densenet121_dct"],
    "description": "Targeted one-step FGSM in RGB pixel space.",
}


def attack(image, classifiers, device, epsilon=8 / 255,
           source_model="vit_b_16", target_class=0):
    if source_model not in ATTACK_CONTRACT["supported_source_models"]:
        raise ValueError(
            f"fgsm does not support source_model={source_model!r}"
        )
    if target_class not in {0, 1}:
        raise ValueError("target_class must be 0 or 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    model = classifiers[source_model]["model"]
    original = from_uint8_image(image, device)
    original.requires_grad_()
    target = torch.tensor([target_class], device=device)
    loss = F.cross_entropy(model(preprocess_for(source_model, original)), target)
    gradient = torch.autograd.grad(loss, original)[0]

    # Descend the targeted loss, then re-enter the feasible set.
    attacked = project_linf(original - epsilon * gradient.sign(), original, epsilon)
    return to_uint8_image(attacked)
