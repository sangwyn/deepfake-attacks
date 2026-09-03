"""Targeted iterative FGSM in RGB pixel space.

The preprocessing and the L-infinity projection come from
``attacklab.preprocessing`` so that every attack differentiates through the
same surrogate of the evaluator. The operations and their order are unchanged
from the original inline implementation.
"""

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
    "description": "Targeted iterative FGSM in RGB pixel space.",
}


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=2 / 255, iterations=10,
           source_model="vit_b_16", target_class=0):
    if source_model not in ATTACK_CONTRACT["supported_source_models"]:
        raise ValueError(
            f"ifgsm does not support source_model={source_model!r}"
        )
    if target_class not in {0, 1}:
        raise ValueError("target_class must be 0 or 1")
    if epsilon <= 0 or step_size <= 0 or iterations < 1:
        raise ValueError("epsilon, step_size, and iterations must be positive")
    model = classifiers[source_model]['model']
    original = from_uint8_image(image, device)
    attacked = original.clone()
    target = torch.tensor([target_class], device=device)

    for _ in range(iterations):
        attacked.requires_grad_()
        model_input = preprocess_for(source_model, attacked)
        loss = F.cross_entropy(model(model_input), target)
        gradient = torch.autograd.grad(loss, attacked)[0]

        # Descend the targeted loss, then re-enter the feasible set.
        attacked = attacked - step_size * gradient.sign()
        attacked = project_linf(attacked, original, epsilon).detach()

    return to_uint8_image(attacked)
