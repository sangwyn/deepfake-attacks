"""Detector construction and differentiable preprocessing."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


SUPPORTED_DETECTORS = frozenset({"vit_b_16", "densenet121_dct", "npr", "aide"})
CLASS_COUNT = 2


def _dct_matrix(size):
    frequency = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    position = torch.arange(size, dtype=torch.float32).unsqueeze(0) + 0.5
    matrix = torch.cos(torch.pi * frequency * position / size)
    matrix[0] *= (1.0 / size) ** 0.5
    matrix[1:] *= (2.0 / size) ** 0.5
    return matrix


def _create_densenet121_dct():
    model = tv_models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, CLASS_COUNT),
    )
    return model


def _normalize_imagenet(x):
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


class DetectorAdapter(nn.Module):
    """Map RGB tensors in ``[0,1]`` to differentiable ``[Real, Fake]`` logits."""

    def __init__(self, name, model, *, log_dct=True, aide_preprocessor=None):
        super().__init__()
        if name not in SUPPORTED_DETECTORS:
            raise ValueError(f"Unsupported detector: {name}")
        self.name = name
        self.model = model
        self.log_dct = bool(log_dct)
        self.aide_preprocessor = aide_preprocessor
        if name == "densenet121_dct":
            self.register_buffer("dct_matrix", _dct_matrix(128), persistent=False)

    def _spatial_input(self, x):
        x = F.interpolate(
            x,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        if self.name in {"vit_b_16", "npr"}:
            x = x[:, :, 16:240, 16:240]
        return _normalize_imagenet(x)

    def _dct_input(self, x):
        x = F.interpolate(
            x,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x[:, :, 64:192, 64:192]
        weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        gray = (x * weights).sum(dim=1) * 255.0
        matrix = self.dct_matrix.to(dtype=x.dtype)
        coefficients = matrix @ gray @ matrix.t()
        if self.log_dct:
            coefficients = torch.log(coefficients.abs() + 1e-6)
        return coefficients.unsqueeze(1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Detector input must have shape [N, 3, H, W]")
        if self.name == "densenet121_dct":
            logits = self.model(self._dct_input(x))
        elif self.name == "aide":
            if self.aide_preprocessor is None:
                raise RuntimeError("AIDE preprocessor is not configured")
            raw_logits = self.model(self.aide_preprocessor(x))
            # The supplied AIDE wrapper maps raw class 1 to the shared Real
            # label and raw class 0 to Fake.
            logits = raw_logits[:, [1, 0]]
        else:
            logits = self.model(self._spatial_input(x))

        if self.name == "npr":
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeError(f"Unexpected NPR output: {tuple(logits.shape)}")
            # The official detector treats sigmoid(raw) > 0.5 as Fake.
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)
        if logits.ndim != 2 or logits.shape[1] != CLASS_COUNT:
            raise RuntimeError(
                f"{self.name} must return [N, 2] logits, got {tuple(logits.shape)}"
            )
        return logits


def _require_state_dict(value, description):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{description} must be a non-empty state_dict")
    return value


def _load_npr_state_dict(weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("NPR checkpoint must contain a 'model' state_dict")
    state = _require_state_dict(checkpoint["model"], "NPR checkpoint['model']")
    invalid = [key for key in state if not key.startswith("module.")]
    if invalid:
        raise ValueError(
            f"Every NPR checkpoint key must start with 'module.': {invalid[:3]}"
        )
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_detector(name, weight_path, device, *, log_dct=True):
    """Load one frozen detector and its canonical differentiable adapter."""
    if name not in SUPPORTED_DETECTORS:
        raise ValueError(
            f"Unsupported detector {name!r}; choose from {sorted(SUPPORTED_DETECTORS)}"
        )
    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {weight_path}")

    aide_preprocessor = None
    if name == "vit_b_16":
        model = tv_models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, CLASS_COUNT)
        state = _require_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True),
            "ViT checkpoint",
        )
    elif name == "densenet121_dct":
        model = _create_densenet121_dct()
        state = _require_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True),
            "DenseNet-DCT checkpoint",
        )
    elif name == "npr":
        from .npr import resnet50

        model = resnet50(num_classes=1)
        state = _load_npr_state_dict(weight_path)
    else:
        try:
            from .aide.preprocessing import AIDEPreprocessor
            from .aide.upstream.models.AIDE import AIDE_Model
        except ImportError as exc:
            raise ImportError(
                "AIDE requires open-clip-torch; install requirements.txt"
            ) from exc
        model = AIDE_Model(None, None)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("AIDE checkpoint must contain a 'model' state_dict")
        state = _require_state_dict(checkpoint["model"], "AIDE checkpoint['model']")
        aide_preprocessor = AIDEPreprocessor()

    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    adapter = DetectorAdapter(
        name,
        model,
        log_dct=log_dct,
        aide_preprocessor=aide_preprocessor,
    )
    return adapter.eval().to(device)
