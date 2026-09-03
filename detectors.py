import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision.models import vit_b_16
from torchvision.transforms import functional as TF


CLASS_COUNT = 2
SUPPORTED_DETECTORS = {
    "vit_b_16",
    "densenet121_dct",
    "npr",
    "aide",
}


def _create_densenet121_dct() -> nn.Module:
    model = tv_models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, CLASS_COUNT),
    )
    return model


def _architecture_parent(architecture_root: Path | None) -> Path:
    root = architecture_root or Path(__file__).resolve().parents[1] / "deepfakes_code"
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Detector architecture directory not found: {root}")
    return root.parent


def _prepare_reference_imports(architecture_root: Path | None) -> None:
    parent = _architecture_parent(architecture_root)
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


def _import_aide_architecture(architecture_root: Path | None):
    _prepare_reference_imports(architecture_root)
    from deepfakes_code.models.aide.data.image import AideImage
    from deepfakes_code.models.aide.models.AIDE import AIDE_Model

    return AideImage, AIDE_Model


def _import_npr_architecture(architecture_root: Path | None):
    _prepare_reference_imports(architecture_root)
    from deepfakes_code.models.npr.model import load_npr_state_dict
    from deepfakes_code.models.npr.networks.resnet import resnet50

    return load_npr_state_dict, resnet50


def _dct_matrix(size: int) -> torch.Tensor:
    frequency = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    position = torch.arange(size, dtype=torch.float32).unsqueeze(0) + 0.5
    matrix = torch.cos(torch.pi * frequency * position / size)
    matrix[0] *= (1.0 / size) ** 0.5
    matrix[1:] *= (2.0 / size) ** 0.5
    return matrix


class DetectorAdapter(nn.Module):
    """Map RGB tensors in [0, 1] to [Real, Fake] logits."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        log_dct: bool = True,
        aide_image_type=None,
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.log_dct = log_dct
        self.aide_image_type = aide_image_type
        if name == "densenet121_dct":
            self.register_buffer("dct_matrix", _dct_matrix(128))

    @staticmethod
    def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std

    def _spatial_input(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.resize(x, [256, 256], antialias=True)
        if self.name in {"vit_b_16", "npr"}:
            x = TF.center_crop(x, [224, 224])
        return self._imagenet_normalize(x)

    def _dct_input(self, x: torch.Tensor) -> torch.Tensor:
        if max(x.shape[-2:]) > 256:
            x = TF.resize(x, [256, 256], antialias=True)
        x = TF.center_crop(x, [128, 128])
        rgb_weights = x.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        gray = (x * rgb_weights).sum(dim=1, keepdim=True) * 255.0
        matrix = self.dct_matrix.to(dtype=x.dtype)
        dct = matrix @ gray.squeeze(1) @ matrix.t()
        if self.log_dct:
            dct = torch.log(torch.abs(dct) + 1e-6)
        return dct.unsqueeze(1)

    def _aide_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.aide_image_type is None:
            raise RuntimeError("AIDE preprocessing class is not configured")
        return torch.stack(
            [self.aide_image_type(sample).prepare() for sample in x], dim=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Detector input must have shape [N, 3, H, W]")

        if self.name == "densenet121_dct":
            logits = self.model(self._dct_input(x))
        elif self.name == "aide":
            raw_logits = self.model(self._aide_input(x))
            logits = raw_logits[:, [1, 0]]
        else:
            logits = self.model(self._spatial_input(x))

        if self.name == "npr":
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeError(f"Unexpected NPR output shape: {tuple(logits.shape)}")
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)

        if logits.ndim != 2 or logits.shape[1] != CLASS_COUNT:
            raise RuntimeError(
                f"{self.name} must produce [N, 2] logits, got {tuple(logits.shape)}"
            )
        return logits


def load_detector(
    name: str,
    weight_path: Path,
    device: torch.device,
    *,
    log_dct: bool = True,
    architecture_root: Path | None = None,
) -> DetectorAdapter:
    if name not in SUPPORTED_DETECTORS:
        raise ValueError(
            f"Unsupported detector '{name}'. Supported: {sorted(SUPPORTED_DETECTORS)}"
        )

    if name == "vit_b_16":
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, CLASS_COUNT)
        state_dict = torch.load(weight_path, map_location="cpu")
    elif name == "densenet121_dct":
        model = _create_densenet121_dct()
        state_dict = torch.load(weight_path, map_location="cpu")
    elif name == "npr":
        load_npr_state_dict, resnet50 = _import_npr_architecture(architecture_root)
        model = resnet50(num_classes=1)
        state_dict = load_npr_state_dict(weight_path, map_location="cpu")
    else:
        AideImage, AIDE_Model = _import_aide_architecture(architecture_root)
        model = AIDE_Model(None, None)
        checkpoint = torch.load(weight_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("AIDE checkpoint must contain a 'model' state_dict")
        state_dict = checkpoint["model"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    adapter = DetectorAdapter(
        name,
        model,
        log_dct=log_dct,
        aide_image_type=AideImage if name == "aide" else None,
    )
    adapter.eval().to(device)
    return adapter
