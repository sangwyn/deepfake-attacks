"""
ifgsm.py — baseline targeted iterative FGSM (BIM), ensemble version.

This is the SANITY-CHECK baseline: a plain L-inf targeted BIM that drives every
provided detector toward the target class (0 = "Real"). No momentum, no input
diversity, no EOT. Use it to confirm the pipeline works white-box before moving
to midi_fgsm.py (MI/DI-FGSM + EOT) for transfer to unseen detectors.
"""

import torch
import torch.nn.functional as F

from attacks._common import to_tensor, to_numpy, build_preprocess


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=2 / 255, iterations=10, target=0, dct_log_scale=True):
    """Targeted ensemble BIM over every classifier in `classifiers`."""
    original = to_tensor(image, device)
    attacked = original.clone()
    tgt = torch.tensor([target], device=device)

    targets = [
        (pack["model"], build_preprocess(name, dct_log_scale))
        for name, pack in classifiers.items()
    ]

    for _ in range(iterations):
        attacked.requires_grad_(True)
        loss = sum(F.cross_entropy(model(prep(attacked)), tgt)
                   for model, prep in targets) / len(targets)
        gradient = torch.autograd.grad(loss, attacked)[0]

        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return to_numpy(attacked)
