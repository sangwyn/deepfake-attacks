"""Joint targeted PGD with an explicit RGB frequency-band constraint."""

import torch
import torch.nn.functional as F

from attacks.dual_pgd import _vit_preprocess, dct_preprocess, torch_dct2, torch_idct2


def _band_mask(size, band, low_cutoff, mid_cutoff, device, dtype):
    rows = torch.arange(size, device=device)
    cols = torch.arange(size, device=device)
    frequency = rows[:, None] + cols[None, :]
    if band == "full":
        allowed = torch.ones_like(frequency, dtype=torch.bool)
    elif band == "low":
        allowed = frequency <= low_cutoff
    elif band == "low_mid":
        allowed = frequency <= mid_cutoff
    elif band == "mid":
        allowed = (frequency > low_cutoff) & (frequency <= mid_cutoff)
    elif band == "high":
        allowed = frequency > mid_cutoff
    elif band == "adaptive_topk":
        raise ValueError("adaptive_topk requires a gradient-dependent mask")
    else:
        raise ValueError("band must be 'full', 'low', 'mid', 'low_mid', or 'high'")
    return allowed.to(dtype=dtype).view(1, 1, size, size)


def _filter_gradient(gradient, band, low_cutoff, mid_cutoff):
    height, width = gradient.shape[-2:]
    if height != width:
        raise ValueError("frequency_pgd requires square RGB inputs")
    if band == "full":
        return gradient
    work_size = min(height, 256)
    work = gradient
    if height != work_size:
        work = F.interpolate(work, size=(work_size, work_size),
                             mode="bilinear", align_corners=False,
                             antialias=True)
    mask = _band_mask(work_size, band, low_cutoff, mid_cutoff,
                      gradient.device, gradient.dtype)
    filtered = torch_idct2(torch_dct2(work) * mask)
    if height != work_size:
        filtered = F.interpolate(filtered, size=(height, width),
                                 mode="bilinear", align_corners=False,
                                 antialias=True)
    return filtered


def _adaptive_frequency_mask(gradient, keep_ratio, power=1.0,
                             previous=None, ema_decay=0.0, hard=False):
    """Keep frequency coefficients with the largest batch/channel energy."""
    height, width = gradient.shape[-2:]
    if height != width:
        raise ValueError("frequency_pgd requires square RGB inputs")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    work_size = min(height, 256)
    work = gradient
    if height != work_size:
        work = F.interpolate(work, size=(work_size, work_size),
                             mode="bilinear", align_corners=False,
                             antialias=True)
    spectrum = torch_dct2(work)
    energy = spectrum.abs().mean(dim=(0, 1), keepdim=True)
    if previous is not None:
        energy = ema_decay * previous + (1.0 - ema_decay) * energy
    if hard:
        count = max(1, int(keep_ratio * work_size * work_size))
        threshold = torch.topk(energy.flatten(), count).values[-1]
        mask = (energy >= threshold).to(dtype=gradient.dtype)
    else:
        normalized = energy / energy.amax().clamp_min(1e-12)
        mask = normalized.pow(power).clamp_min(1e-3)
    filtered = torch_idct2(spectrum * mask)
    if height != work_size:
        filtered = F.interpolate(filtered, size=(height, width),
                                 mode="bilinear", align_corners=False,
                                 antialias=True)
    return filtered, energy.detach()


def _apply_frequency_mask(gradient, energy, keep_ratio, power=1.0, hard=False):
    """Apply a mask derived from another gradient to the joint gradient."""
    height, width = gradient.shape[-2:]
    if height != width:
        raise ValueError("frequency_pgd requires square RGB inputs")
    work_size = min(height, 256)
    work = gradient
    if height != work_size:
        work = F.interpolate(work, size=(work_size, work_size),
                             mode="bilinear", align_corners=False,
                             antialias=True)
    if hard:
        count = max(1, int(keep_ratio * work_size * work_size))
        threshold = torch.topk(energy.flatten(), count).values[-1]
        mask = (energy >= threshold).to(dtype=gradient.dtype)
    else:
        normalized = energy / energy.amax().clamp_min(1e-12)
        mask = normalized.pow(power).clamp_min(1e-3)
    filtered = torch_idct2(torch_dct2(work) * mask)
    if height != work_size:
        filtered = F.interpolate(filtered, size=(height, width),
                                 mode="bilinear", align_corners=False,
                                 antialias=True)
    return filtered


def _filter_adaptive_gradient(gradient, keep_ratio):
    return _adaptive_frequency_mask(gradient, keep_ratio, hard=True)[0]


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=0.5 / 255, iterations=40, target=0,
           vit_weight=0.5, dct_weight=0.5, normalize_gradients=True,
           dct_log_scale=True, dct_resize_mode="bicubic", band="full",
           low_cutoff=16, mid_cutoff=48, keep_ratio=0.25,
           frequency_guidance="joint", soft_power=1.0, ema_decay=0.8):
    required = {"vit_b_16", "densenet121_dct"}
    missing = required.difference(classifiers)
    if missing:
        raise ValueError(f"frequency_pgd requires classifiers: {sorted(required)}")
    if vit_weight < 0 or dct_weight < 0 or vit_weight + dct_weight <= 0:
        raise ValueError("vit_weight and dct_weight must be non-negative and not both zero")

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    original = original.to(device=device, dtype=torch.float32) / 255.0
    attacked = original.clone()
    target_tensor = torch.full((1,), target, dtype=torch.long, device=device)
    vit = classifiers["vit_b_16"]["model"]
    dct = classifiers["densenet121_dct"]["model"]

    ema_energy = None
    valid_guidance = {"joint", "dct", "dct_guided_topk",
                      "dct_guided_soft", "dct_guided_soft_ema"}
    if frequency_guidance not in valid_guidance:
        raise ValueError(f"frequency_guidance must be one of {sorted(valid_guidance)}")

    for step in range(iterations):
        attacked.requires_grad_(True)
        vit_loss = F.cross_entropy(vit(_vit_preprocess(attacked)), target_tensor)
        dct_loss = F.cross_entropy(
            dct(dct_preprocess(attacked, log_scale=dct_log_scale,
                               resize_mode=dct_resize_mode)), target_tensor
        )
        vit_gradient = torch.autograd.grad(vit_loss, attacked, retain_graph=True)[0]
        dct_gradient = torch.autograd.grad(dct_loss, attacked)[0]
        if (not torch.isfinite(vit_gradient).all()
                or not torch.isfinite(dct_gradient).all()
                or dct_gradient.abs().sum() == 0):
            raise RuntimeError(f"invalid branch gradient at iteration {step}")
        if normalize_gradients:
            vit_gradient = vit_gradient / vit_gradient.abs().mean().clamp_min(1e-12)
            dct_gradient = dct_gradient / dct_gradient.abs().mean().clamp_min(1e-12)
        gradient = (vit_weight * vit_gradient + dct_weight * dct_gradient)
        if frequency_guidance != "joint" and band == "adaptive_topk":
            guidance = dct_gradient
            previous = ema_energy if frequency_guidance == "dct_guided_soft_ema" else None
            _, current_energy = _adaptive_frequency_mask(
                guidance, keep_ratio, power=soft_power, previous=previous,
                ema_decay=ema_decay, hard=False
            )
            if frequency_guidance == "dct_guided_soft_ema":
                ema_energy = current_energy
            if frequency_guidance == "dct_guided_topk":
                current_energy = _adaptive_frequency_mask(
                    guidance, keep_ratio, previous=previous,
                    ema_decay=ema_decay, hard=True
                )[1]
            gradient = _apply_frequency_mask(
                gradient, current_energy, keep_ratio, power=soft_power,
                hard=(frequency_guidance == "dct_guided_topk")
            )
        elif band == "adaptive_topk":
            gradient = _filter_adaptive_gradient(gradient, keep_ratio)
        else:
            gradient = _filter_gradient(gradient, band, low_cutoff, mid_cutoff)
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise RuntimeError(f"invalid masked gradient for band {band} at iteration {step}")
        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return (attacked[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
