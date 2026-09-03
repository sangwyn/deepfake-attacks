"""Joint ViT/DCT PGD regularized toward camera-like noise statistics."""

import numpy as np
import torch
import torch.nn.functional as F

from attacks.dual_pgd import _vit_preprocess, dct_preprocess


def _allocation(image, texture_mask, mask_floor):
    luminance = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2]
                 + 0.114 * image[:, 2:3])
    if not texture_mask:
        return torch.ones_like(luminance), luminance
    padded = F.pad(luminance, (1, 1, 1, 1), mode="reflect")
    local = F.avg_pool2d(padded, kernel_size=3, stride=1)
    detail = (luminance - local).abs()
    scale = torch.quantile(detail.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    allocation = mask_floor + (1.0 - mask_floor) * detail / scale.clamp_min(1e-6)
    return allocation.clamp(mask_floor, 1.0), luminance


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=0.5 / 255, iterations=40, target=0,
           vit_weight=0.5, dct_weight=0.5, normalize_gradients=True,
           dct_log_scale=True, dct_resize_mode="bicubic",
           isp_prior_weight=0.05, isp_prior_scale=1.0,
           isp_texture_mask=True, isp_mask_floor=0.25, seed=0):
    """Run targeted joint PGD with a bounded ISP-statistics prior."""
    required = {"vit_b_16", "densenet121_dct"}
    missing = required.difference(classifiers)
    if missing:
        raise ValueError(f"isp_joint_pgd requires classifiers: {sorted(required)}")
    if vit_weight < 0 or dct_weight < 0 or vit_weight + dct_weight <= 0:
        raise ValueError("vit_weight and dct_weight must be non-negative and not both zero")
    if isp_prior_weight < 0 or isp_prior_scale < 0:
        raise ValueError("ISP prior parameters must be non-negative")
    if not 0 < isp_mask_floor <= 1:
        raise ValueError("isp_mask_floor must be in (0, 1]")

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    original = original.to(device=device, dtype=torch.float32) / 255.0
    allocation, luminance = _allocation(original, isp_texture_mask, isp_mask_floor)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    noise = torch.randn(original.shape, generator=generator, device=device)
    camera_std = allocation * torch.sqrt(0.02 ** 2 * luminance + 0.005 ** 2)
    prior_delta = isp_prior_scale * noise * camera_std

    attacked = original.clone()
    target_tensor = torch.full((1,), target, dtype=torch.long, device=device)
    vit = classifiers["vit_b_16"]["model"]
    dct = classifiers["densenet121_dct"]["model"]

    for step in range(iterations):
        attacked.requires_grad_(True)
        vit_loss = F.cross_entropy(vit(_vit_preprocess(attacked)), target_tensor)
        dct_loss = F.cross_entropy(
            dct(dct_preprocess(attacked, log_scale=dct_log_scale,
                               resize_mode=dct_resize_mode)), target_tensor
        )
        classification_loss = vit_weight * vit_loss + dct_weight * dct_loss
        prior_loss = ((attacked - original - prior_delta) /
                      max(epsilon, 1e-12)).square().mean()
        gradient = torch.autograd.grad(
            classification_loss + isp_prior_weight * prior_loss, attacked
        )[0]
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise RuntimeError(f"invalid ISP-joint gradient at iteration {step}")
        if normalize_gradients:
            gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return (attacked[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
