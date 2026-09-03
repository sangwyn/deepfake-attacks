"""Targeted PGD against one selected detector, with post-hoc evaluation on all targets."""

import torch
import torch.nn.functional as F

from attacks.dual_pgd import _vit_preprocess, dct_preprocess


def _npr_preprocess(image):
    image = F.interpolate(image, size=(256, 256), mode="bilinear",
                          align_corners=False, antialias=True)
    image = image[:, :, 16:240, 16:240]
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=0.5 / 255, iterations=40, target=0,
           source="vit_b_16", dct_log_scale=True,
           dct_resize_mode="bicubic"):
    if source not in {"vit_b_16", "densenet121_dct", "npr"}:
        raise ValueError("source_pgd supports vit_b_16, densenet121_dct, or npr")
    if source not in classifiers:
        raise ValueError(f"source detector is not loaded: {source}")

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    original = original.to(device=device, dtype=torch.float32) / 255.0
    attacked = original.clone()
    target_tensor = torch.full((1,), target, dtype=torch.long, device=device)
    model = classifiers[source]["model"]

    for step in range(iterations):
        attacked.requires_grad_(True)
        if source == "vit_b_16":
            model_input = _vit_preprocess(attacked)
        elif source == "densenet121_dct":
            model_input = dct_preprocess(attacked, log_scale=dct_log_scale,
                                         resize_mode=dct_resize_mode)
        else:
            model_input = _npr_preprocess(attacked)
        loss = F.cross_entropy(model(model_input), target_tensor)
        gradient = torch.autograd.grad(loss, attacked)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"{source} gradient is invalid at iteration {step}")
        if gradient.abs().sum() == 0:
            break
        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return (attacked[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
