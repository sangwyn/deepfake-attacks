"""Targeted PGD optimized only against the DCT DenseNet detector."""

import torch
import torch.nn.functional as F

from attacks.dual_pgd import dct_preprocess


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=0.5 / 255, iterations=40, target=0,
           dct_log_scale=True, dct_resize_mode="bicubic"):
    if "densenet121_dct" not in classifiers:
        raise ValueError("dct_pgd requires the 'densenet121_dct' classifier")

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    original = original.to(device=device, dtype=torch.float32) / 255.0
    attacked = original.clone()
    target_tensor = torch.full((1,), target, dtype=torch.long, device=device)
    model = classifiers["densenet121_dct"]["model"]

    for step in range(iterations):
        attacked.requires_grad_(True)
        logits = model(dct_preprocess(
            attacked, log_scale=dct_log_scale, resize_mode=dct_resize_mode
        ))
        loss = F.cross_entropy(logits, target_tensor)
        gradient = torch.autograd.grad(loss, attacked)[0]
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise RuntimeError(f"DCT gradient is invalid at iteration {step}")
        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return (attacked[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
