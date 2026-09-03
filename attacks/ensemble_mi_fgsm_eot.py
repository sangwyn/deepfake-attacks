"""Targeted leave-one-detector-out ensemble MI-FGSM with differentiable EoT."""

import torch
import torch.nn.functional as F

from attacklab.preprocessing import from_uint8_image, preprocess_for, project_linf, to_uint8_image


ATTACK_CONTRACT = {
    "version": 1,
    "source_model": "densenet121_dct",
    "supported_source_models": ["vit_b_16", "densenet121_dct"],
    "description": "Targeted leave-one-detector-out ensemble MI-FGSM with EoT.",
}


def _transform(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply the differentiable generation-time approximation of an EoT transform."""
    height, width = x.shape[-2:]
    if name == "resize":
        reduced = F.interpolate(x, size=(max(1, round(height * 0.9)), max(1, round(width * 0.9))),
                                mode="bilinear", align_corners=False, antialias=True)
        return F.interpolate(reduced, size=(height, width), mode="bilinear", align_corners=False,
                             antialias=True)
    if name == "crop":
        margin_h, margin_w = max(1, height // 16), max(1, width // 16)
        cropped = x[..., margin_h:height - margin_h, margin_w:width - margin_w]
        return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False,
                             antialias=True)
    if name == "jpeg-like":
        # Smooth scalar quantization is differentiable and keeps the image shape intact.
        return x + (torch.round(x * 16.0) - x * 16.0).detach() / 16.0
    if name in ("identity", None):
        return x
    raise ValueError(f"Unsupported EoT transformation: {name!r}")


def attack(image, classifiers, device, epsilon=8 / 255, step_size=None, iterations=10,
           momentum=1.0, ensemble_weighting="uniform", eot_samples=5,
           transformations=("resize", "crop", "jpeg-like"), source_model="vit_b_16",
           target_class=0):
    if source_model not in ATTACK_CONTRACT["supported_source_models"]:
        raise ValueError(f"ensemble_mi_fgsm_eot does not support source_model={source_model!r}")
    if target_class not in {0, 1} or epsilon <= 0 or iterations < 1:
        raise ValueError("invalid target_class, epsilon, or iterations")
    if step_size is None:
        step_size = epsilon / iterations
    if step_size <= 0 or momentum < 0 or eot_samples < 1:
        raise ValueError("step_size, momentum, and eot_samples must be positive")
    if ensemble_weighting != "uniform":
        raise ValueError("only uniform ensemble weighting is supported")
    # SOURCE_MODEL is the detector whose gradient is used.  With the two
    # configured detectors this is the leave-one-detector-out white-box
    # variant: the other detector is reserved for transfer measurement.
    sources = [source_model]
    if source_model not in classifiers:
        raise ValueError("configured classifiers do not contain source_model")

    original = from_uint8_image(image, device)
    attacked = original.clone()
    accumulated = torch.zeros_like(original)
    target = torch.tensor([target_class], device=device)
    transform_names = tuple(transformations)
    for _ in range(iterations):
        attacked.requires_grad_()
        loss = torch.zeros((), device=device)
        for model_name in sources:
            model = classifiers[model_name]["model"]
            for sample in range(eot_samples):
                transformed = _transform(attacked, transform_names[sample % len(transform_names)])
                loss = loss + F.cross_entropy(model(preprocess_for(model_name, transformed)), target)
        loss = loss / (len(sources) * eot_samples)
        gradient = torch.autograd.grad(loss, attacked)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("ensemble MI-FGSM EoT encountered a non-finite gradient")
        normalized = gradient / (gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        accumulated = momentum * accumulated + normalized
        attacked = project_linf(attacked - step_size * accumulated.sign(), original, epsilon).detach()
    return to_uint8_image(attacked)
