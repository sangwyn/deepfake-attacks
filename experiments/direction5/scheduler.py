"""Direction 5 rule-based scheduler for four fixed attack primitives."""

import numpy as np
import torch
import torch.nn.functional as F

from attacks.dual_pgd import _vit_preprocess, dct_preprocess, torch_dct2, torch_idct2

PRIMITIVES = ("spatial", "di_fgsm", "low_frequency", "global_noise")


def _logits(x, classifiers):
    vit = classifiers["vit_b_16"]["model"](_vit_preprocess(x))
    dct = classifiers["densenet121_dct"]["model"](
        dct_preprocess(x, log_scale=True, resize_mode="bicubic"))
    return vit, dct


def _margins(x, classifiers):
    vit, dct = _logits(x, classifiers)
    return torch.stack((vit[:, 0] - vit[:, 1], dct[:, 0] - dct[:, 1]))


def _deficit_weights(margins, margin_temperature):
    """Emphasize branches still on the wrong side of the Real/Fake margin."""
    deficit = F.softplus(-margins / margin_temperature).mean(dim=1)
    return (deficit / deficit.sum().clamp_min(1e-12)).detach()


def _di_view(x, seed):
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    scale = float(torch.empty((), device=x.device).uniform_(0.9, 1.1,
                                                              generator=generator))
    size = max(4, int(round(x.shape[-1] * scale)))
    resized = F.interpolate(x, size=(size, size), mode="bilinear",
                            align_corners=False, antialias=True)
    if size >= x.shape[-1]:
        top = (size - x.shape[-1]) // 2
        return resized[:, :, top:top + x.shape[-2], top:top + x.shape[-1]]
    pad = x.shape[-1] - size
    return F.pad(resized, (pad // 2, pad - pad // 2, pad // 2, pad - pad // 2),
                 mode="reflect")[:, :, :x.shape[-2], :x.shape[-1]]


def _primitive_gradient(x, classifiers, primitive, target, seed, global_noise):
    if primitive == "global_noise":
        return global_noise
    target_tensor = torch.full((x.shape[0],), target, dtype=torch.long, device=x.device)
    view = _di_view(x, seed) if primitive == "di_fgsm" else x
    vit, dct = _logits(view, classifiers)
    loss = 0.5 * F.cross_entropy(vit, target_tensor) + 0.5 * F.cross_entropy(dct, target_tensor)
    gradient = torch.autograd.grad(loss, x, retain_graph=True)[0]
    if primitive == "low_frequency":
        size = gradient.shape[-1]
        rows = torch.arange(size, device=x.device)
        mask = (rows[:, None] + rows[None, :] <= size // 4).to(gradient.dtype)
        gradient = torch_idct2(torch_dct2(gradient) * mask.view(1, 1, size, size))
    return gradient


def _cosine_matrix(gradients):
    matrix = torch.eye(len(gradients), device=gradients[0].device)
    vectors = [gradient.flatten() for gradient in gradients]
    for i in range(len(vectors)):
        for j in range(i):
            value = F.cosine_similarity(vectors[i], vectors[j], dim=0)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def attack(image, classifiers, device, epsilon=8 / 255, step_size=0.5 / 255,
           iterations=40, target=0, schedule_interval=5, temperature=0.25,
           scheduler_mode="adaptive", fixed_primitive="spatial", seed=0,
           diversity_penalty=0.15, quality_penalty=0.05,
           margin_temperature=1.0, gain_ema_decay=0.7, momentum=0.75,
           min_primitive_weight=0.05, acceptance_margin=0.0,
           stagnation_patience=2, step_decay=0.5, **_):
    """Target Real with adaptive, equal, or fixed primitive scheduling."""
    required = {"vit_b_16", "densenet121_dct"}
    missing = required.difference(classifiers)
    if missing:
        raise ValueError(f"direction5 scheduler requires classifiers: {sorted(required)}")
    if scheduler_mode not in {"adaptive", "equal", "fixed"}:
        raise ValueError("scheduler_mode must be 'adaptive', 'equal', or 'fixed'")
    if fixed_primitive not in PRIMITIVES:
        raise ValueError(f"fixed_primitive must be one of {PRIMITIVES}")
    if iterations <= 0 or schedule_interval <= 0 or temperature <= 0:
        raise ValueError("iterations, schedule_interval, and temperature must be positive")
    if margin_temperature <= 0 or not 0 <= gain_ema_decay < 1 or not 0 <= momentum < 1:
        raise ValueError("invalid margin_temperature, gain_ema_decay, or momentum")
    if not 0 <= min_primitive_weight < 1 / len(PRIMITIVES):
        raise ValueError("min_primitive_weight must be below uniform allocation")
    if acceptance_margin < 0 or stagnation_patience <= 0 or not 0 < step_decay <= 1:
        raise ValueError("invalid acceptance_margin, stagnation_patience, or step_decay")

    torch.manual_seed(seed)
    np.random.seed(seed)
    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    attacked = original.clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    global_noise = torch.randn(original.shape, device=device, generator=generator)
    global_noise = F.avg_pool2d(global_noise, 5, 1, 2)
    global_noise = global_noise / global_noise.abs().mean().clamp_min(1e-12)
    weights = torch.full((len(PRIMITIVES),), 1 / len(PRIMITIVES), device=device)
    ema_scores = torch.zeros_like(weights)
    velocity = torch.zeros_like(attacked)
    history = []
    stagnation = 0
    effective_step_size = float(step_size)

    for step in range(iterations):
        attacked.requires_grad_(True)
        active_primitives = (PRIMITIVES if scheduler_mode != "fixed"
                             else (fixed_primitive,))
        gradients = [_primitive_gradient(attacked, classifiers, primitive, target,
                                         seed + step, global_noise)
                     for primitive in active_primitives]
        gradients = [gradient / gradient.abs().mean().clamp_min(1e-12)
                     for gradient in gradients]
        if step % schedule_interval == 0 or step == iterations - 1:
            current_margins = _margins(attacked.detach(), classifiers)
            branch_weights = _deficit_weights(current_margins, margin_temperature)
            current_quality = (attacked.detach() - original).square().mean()
            gains = []
            candidate_images = []
            for gradient in gradients:
                trial = torch.clamp(attacked.detach() - effective_step_size * gradient.sign(), 0, 1)
                candidate_images.append(trial)
                trial_margins = _margins(trial, classifiers)
                quality_loss = (trial - original).square().mean() - current_quality
                gains.append((branch_weights * (trial_margins - current_margins)).sum()
                             - quality_penalty * quality_loss)
            gains = torch.stack(gains)
            cosine = _cosine_matrix(gradients) if len(gradients) > 1 else torch.ones((1, 1), device=device)
            redundancy = ((cosine.sum(dim=1) - 1) / max(len(gradients) - 1, 1))
            normalized_gains = gains / gains.abs().mean().clamp_min(1e-6)
            scores = normalized_gains - diversity_penalty * redundancy
            ema_scores[:len(active_primitives)] = (
                gain_ema_decay * ema_scores[:len(active_primitives)]
                + (1 - gain_ema_decay) * scores.detach()
            )
            if scheduler_mode == "adaptive":
                proposed = torch.softmax(ema_scores[:len(active_primitives)] / temperature, dim=0)
                proposed = proposed * (1 - len(active_primitives) * min_primitive_weight)
                proposed = proposed + min_primitive_weight
                weights = torch.zeros_like(weights)
                for index, primitive in enumerate(active_primitives):
                    weights[PRIMITIVES.index(primitive)] = proposed[index]
            elif scheduler_mode == "equal":
                weights = torch.full_like(weights, 1 / len(PRIMITIVES))
            else:
                weights = torch.zeros_like(weights)
                weights[PRIMITIVES.index(fixed_primitive)] = 1
            candidate_scores = gains.detach()
            best_index = int(candidate_scores.argmax())
            best_score = candidate_scores[best_index]
            accepted = False
            if scheduler_mode == "adaptive":
                if best_score > acceptance_margin:
                    attacked = candidate_images[best_index].detach()
                    accepted = True
                    stagnation = 0
                else:
                    stagnation += 1
            if stagnation >= stagnation_patience:
                effective_step_size *= step_decay
                stagnation = 0
                weights = torch.full_like(weights, 1 / len(PRIMITIVES))
            history.append({
                "step": step,
                "real_fake_margins": current_margins.cpu().tolist(),
                "branch_deficit_weights": branch_weights.cpu().tolist(),
                "primitive_gains": gains.detach().cpu().tolist(),
                "primitive_names": list(active_primitives),
                "normalized_gains": normalized_gains.detach().cpu().tolist(),
                "gradient_cosine": cosine.detach().cpu().tolist(),
                "primitive_weights": weights.cpu().tolist(),
                "quality_proxy_mse": float(current_quality.detach().cpu()),
                "accepted": accepted,
                "accepted_primitive": (active_primitives[best_index]
                                       if accepted else None),
                "stagnation": stagnation,
                "effective_step_size": effective_step_size,
            })
        fused = sum(weights[PRIMITIVES.index(primitive)] * gradient
                    for primitive, gradient in zip(active_primitives, gradients))
        if not torch.isfinite(fused).all() or fused.abs().sum() == 0:
            raise RuntimeError(f"invalid scheduled gradient at iteration {step}")
        velocity = momentum * velocity + (1 - momentum) * fused
        if (not (step % schedule_interval == 0 or step == iterations - 1)
                or scheduler_mode != "adaptive"):
            attacked = attacked - effective_step_size * velocity.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    attack.last_metadata = {"primitives": list(PRIMITIVES), "scheduler_mode": scheduler_mode,
                           "schedule_interval": schedule_interval, "temperature": temperature,
                           "margin_temperature": margin_temperature,
                           "gain_ema_decay": gain_ema_decay, "momentum": momentum,
                           "acceptance_margin": acceptance_margin,
                           "stagnation_patience": stagnation_patience,
                           "seed": seed, "history": history,
                           "final_weights": weights.cpu().tolist()}
    return (attacked[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()


attack.last_metadata = {}
