"""
midi_fgsm.py — targeted ensemble MI-DI-FGSM with EOT.

Built for TRANSFER to unseen detectors (the challenge's hidden models + random
JPEG/resize post-processing), which plain BIM overfits away from. Combines:

  - Ensemble        : averages the targeted loss over every provided detector.
  - MI (momentum)   : accumulates an L1-normalised gradient across steps
                      (Dong et al., "Boosting Adversarial Attacks with Momentum").
  - DI (diversity)  : random resize+pad on each forward pass
                      (Xie et al., "Improving Transferability ... Input Diversity").
  - EOT             : averages the gradient over several stochastic transform
                      draws per step (Athalye et al.), optionally including
                      differentiable JPEG to harden against re-compression.

Config example (attack_params):
    attack: midi_fgsm
    attack_params:
      epsilon: 0.03137      # 8/255
      step_size: 0.00784    # 2/255
      iterations: 20
      decay: 1.0            # momentum
      di_prob: 0.7          # P(apply input diversity)
      di_pad_ratio: 0.1     # resize up to +10% then pad
      eot_samples: 4        # gradient draws per step
      jpeg_quality: [70, 85, 95]   # needs kornia; omit to disable JPEG EOT
"""

import torch
import torch.nn.functional as F

from attacks._common import (
    to_tensor, to_numpy, build_preprocess, input_diversity, jpeg_compress,
)


def _pick_quality(jpeg_quality):
    if jpeg_quality is None:
        return None
    if isinstance(jpeg_quality, (list, tuple)):
        return jpeg_quality[int(torch.randint(len(jpeg_quality), (1,)).item())]
    return jpeg_quality


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=2 / 255, iterations=20, target=0,
           decay=1.0, di_prob=0.7, di_pad_ratio=0.1,
           eot_samples=1, jpeg_quality=None, dct_log_scale=True):
    """Targeted ensemble MI-DI-FGSM with EOT over every classifier."""
    original = to_tensor(image, device)
    attacked = original.clone()
    tgt = torch.tensor([target], device=device)
    momentum = torch.zeros_like(original)

    targets = [
        (pack["model"], build_preprocess(name, dct_log_scale))
        for name, pack in classifiers.items()
    ]

    for _ in range(iterations):
        attacked.requires_grad_(True)

        # EOT: average loss over stochastic transform draws and detectors.
        total = 0.0
        n = 0
        for _ in range(eot_samples):
            q = _pick_quality(jpeg_quality)
            for model, prep in targets:
                xin = attacked
                if q is not None:
                    xin = jpeg_compress(xin, q)          # models re-compression
                xin = input_diversity(xin, di_prob, di_pad_ratio)  # DI
                total = total + F.cross_entropy(model(prep(xin)), tgt)
                n += 1
        loss = total / n
        gradient = torch.autograd.grad(loss, attacked)[0]

        # MI: accumulate L1-normalised gradient, step along its sign.
        gradient = gradient / gradient.abs().mean(dim=(1, 2, 3),
                                                   keepdim=True).clamp_min(1e-12)
        momentum = decay * momentum + gradient

        attacked = attacked - step_size * momentum.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return to_numpy(attacked)
