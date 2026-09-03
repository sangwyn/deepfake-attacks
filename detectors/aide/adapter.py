"""AIDE checkpoint loader and differentiable input adapter."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

_UPSTREAM = Path(__file__).parent / "upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from models.AIDE import AIDE_Model  # noqa: E402
from data.dct import DCT_base_Rec_Module  # noqa: E402


def build_aide(checkpoint_path, device):
    """Construct the upstream architecture and load embedded model weights."""
    model = AIDE_Model(resnet_path=None, convnext_path=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class AIDEAdapter:
    """Map full-resolution RGB [0,1] tensors to official AIDE input."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.dct = DCT_base_Rec_Module().to(device)

    def __call__(self, image):
        # Match AideImage.prepare: select DCT patches at the original image
        # resolution, then resize each selected branch to 256.
        original = image
        features = self.dct(original[0])
        features = tuple(
            TF.normalize(
                F.interpolate(feature.unsqueeze(0), size=(256, 256),
                              mode="bilinear", align_corners=False,
                              antialias=True),
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )[0]
            for feature in features
        )
        rgb = TF.normalize(F.interpolate(original, size=(256, 256),
                                         mode="bilinear", align_corners=False),
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])[0]
        # The upstream DCT module has discrete patch selection, so this
        # adapter is for evaluation; it is not an AIDE attack gradient path.
        return torch.stack([features[0], features[1], features[2], features[3], rgb], dim=0).unsqueeze(0)


__all__ = ["AIDEAdapter", "build_aide"]
