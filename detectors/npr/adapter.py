"""Differentiable NPR checkpoint loader and RGB input adapter."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from .model import NPRResNet


def build_npr(checkpoint_path, device):
    model = NPRResNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    cleaned = OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class NPRAdapter:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, image):
        image = F.interpolate(image, size=(256, 256), mode="bilinear",
                              align_corners=False, antialias=True)
        image = image[:, :, 16:240, 16:240]
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        score = self.model(image)
        # The labeled clean manifest verifies that positive raw scores
        # correspond to Fake for this checkpoint.
        return torch.cat((-score, score), dim=1)


__all__ = ["NPRAdapter", "build_npr"]
