"""Targeted PGD optimized only against the RGB ViT detector."""

import torch
import torch.nn.functional as F

from attacks.dual_pgd import _vit_preprocess


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=0.5 / 255, iterations=40, target=0):
    if "vit_b_16" not in classifiers:
        raise ValueError("vit_pgd requires the 'vit_b_16' classifier")

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    original = original.to(device=device, dtype=torch.float32) / 255.0
    attacked = original.clone()
    target_tensor = torch.full((1,), target, dtype=torch.long, device=device)
    model = classifiers["vit_b_16"]["model"]

    for step in range(iterations):
        attacked.requires_grad_(True)
        logits = model(_vit_preprocess(attacked))
        loss = F.cross_entropy(logits, target_tensor)
        gradient = torch.autograd.grad(loss, attacked)[0]
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise RuntimeError(f"ViT gradient is invalid at iteration {step}")
        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return (attacked[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
