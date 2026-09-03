"""Joint targeted PGD for the RGB ViT and DCT DenseNet classifiers."""

import torch
import torch.nn.functional as F


_GRAY = (0.299, 0.587, 0.114)
_DCT_CACHE = {}


def _dct_matrix(size, device, dtype):
    key = (size, str(device), dtype)
    if key not in _DCT_CACHE:
        n = torch.arange(size, device=device, dtype=dtype)
        k = n[:, None]
        matrix = torch.cos(torch.pi / size * (n[None, :] + 0.5) * k)
        matrix[0] *= (1.0 / size) ** 0.5
        matrix[1:] *= (2.0 / size) ** 0.5
        _DCT_CACHE[key] = matrix
    return _DCT_CACHE[key]


def torch_dct2(x):
    """Apply an orthonormal 2-D DCT-II to BCHW input."""
    if x.ndim != 4:
        raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
    c = _dct_matrix(x.shape[-1], x.device, x.dtype)
    if x.shape[-2] != x.shape[-1]:
        c_h = _dct_matrix(x.shape[-2], x.device, x.dtype)
    else:
        c_h = c
    return c_h @ x @ c.transpose(-1, -2)


def torch_idct2(x):
    """Apply the inverse of the orthonormal 2-D DCT-II."""
    if x.ndim != 4:
        raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
    c_w = _dct_matrix(x.shape[-1], x.device, x.dtype)
    c_h = _dct_matrix(x.shape[-2], x.device, x.dtype)
    return c_h.transpose(-1, -2) @ x @ c_w


def dct_preprocess(image, log_scale=True, resize_mode="bilinear"):
    """Differentiable equivalent of evaluator.build_dct_transform.

    The evaluator's PIL grayscale image contains values in [0, 255], hence the
    explicit *255 below despite the attack tensor being represented in [0, 1].
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected RGB BCHW input, got shape {tuple(image.shape)}")
    gray = (image[:, 0:1] * _GRAY[0] + image[:, 1:2] * _GRAY[1]
            + image[:, 2:3] * _GRAY[2]) * 255.0
    if resize_mode not in {"bilinear", "bicubic"}:
        raise ValueError("resize_mode must be 'bilinear' or 'bicubic'")
    resized = F.interpolate(gray, size=(256, 256), mode=resize_mode,
                            align_corners=False, antialias=True)
    top = (256 - 128) // 2
    cropped = resized[:, :, top:top + 128, top:top + 128]
    result = torch_dct2(cropped)
    return torch.log(result.abs() + 1e-6) if log_scale else result


def _vit_preprocess(image):
    resized = F.interpolate(image, size=(256, 256), mode="bilinear",
                            align_corners=False, antialias=True)
    top = (256 - 224) // 2
    cropped = resized[:, :, top:top + 224, top:top + 224]
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (cropped - mean) / std


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=2 / 255, iterations=10, target=0,
           vit_weight=0.5, dct_weight=0.5, normalize_gradients=False,
           dct_log_scale=True, dct_resize_mode="bilinear",
           fusion_mode="weighted_sum"):
    """Run targeted PGD toward ``target`` in both image representations."""
    required = {"vit_b_16", "densenet121_dct"}
    missing = required.difference(classifiers)
    if missing:
        raise ValueError(f"dual_pgd requires classifiers: {sorted(required)}; missing {sorted(missing)}")

    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    original = original.to(device=device, dtype=torch.float32) / 255.0
    attacked = original.clone()
    if vit_weight < 0 or dct_weight < 0 or vit_weight + dct_weight <= 0:
        raise ValueError("vit_weight and dct_weight must be non-negative and not both zero")
    if fusion_mode not in {"weighted_sum", "weighted_sign"}:
        raise ValueError("fusion_mode must be 'weighted_sum' or 'weighted_sign'")
    target_tensor = torch.full((1,), target, dtype=torch.long, device=device)
    vit = classifiers["vit_b_16"]["model"]
    dct = classifiers["densenet121_dct"]["model"]

    for step in range(iterations):
        attacked.requires_grad_(True)
        vit_logits = vit(_vit_preprocess(attacked))
        dct_logits = dct(dct_preprocess(
            attacked, log_scale=dct_log_scale, resize_mode=dct_resize_mode
        ))
        vit_loss = F.cross_entropy(vit_logits, target_tensor)
        dct_loss = F.cross_entropy(dct_logits, target_tensor)
        dct_gradient = torch.autograd.grad(
            dct_loss, attacked, retain_graph=True
        )[0]
        # Compute the second branch last so autograd can release the graph.
        vit_gradient = torch.autograd.grad(vit_loss, attacked)[0]
        if not torch.isfinite(dct_gradient).all() or dct_gradient.abs().sum() == 0:
            raise RuntimeError(f"DCT gradient is invalid at iteration {step}")
        if not torch.isfinite(vit_gradient).all() or vit_gradient.abs().sum() == 0:
            raise RuntimeError(f"ViT gradient is invalid at iteration {step}")
        if normalize_gradients:
            vit_gradient = vit_gradient / vit_gradient.abs().mean().clamp_min(1e-12)
            dct_gradient = dct_gradient / dct_gradient.abs().mean().clamp_min(1e-12)
        if fusion_mode == "weighted_sign":
            gradient = (vit_weight * vit_gradient.sign()
                        + dct_weight * dct_gradient.sign())
        else:
            gradient = vit_weight * vit_gradient + dct_weight * dct_gradient
        if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise FloatingPointError(f"invalid joint gradient at iteration {step}")
        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    return (attacked[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
