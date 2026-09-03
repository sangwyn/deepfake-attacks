"""Data loading. Labels come from directory names (real=0, fake=1),
never from filenames — matching the team's contract."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# Canonical dataset locations. Oleg's aligned test set is celebA/TEST with
# TEST_FAKE / TEST_REAL subdirs; the earlier dev set uses fake/ real.
DATASETS = {
    "celebA": Path.home() / "dataset" / "celebA" / "TEST",
    "dev256": Path("/data2/aiattacks/dataset"),
}


def _label_from_dir(path: Path) -> int | None:
    """Label from the containing directory name (never the filename):
    a dir containing 'fake' -> 1, containing 'real' -> 0."""
    name = path.parent.name.lower()
    if "fake" in name:
        return 1
    if "real" in name:
        return 0
    return None


def build_manifest(root: str | Path) -> list[dict]:
    root = Path(root)
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            label = _label_from_dir(p)
            if label is None:
                continue
            items.append({"path": str(p), "label": label,
                          "cls": "fake" if label == 1 else "real"})
    return items


def load_image_tensor(path: str, device) -> tuple[torch.Tensor, np.ndarray]:
    """Return (x_full NCHW in [0,1], orig_uint8 HWC)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
    return t.unsqueeze(0), arr


def to_uint8_hwc(x_full_01: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) in [0,1] -> (H,W,3) uint8."""
    x = (x_full_01[0].detach().clamp(0, 1) * 255.0).round()
    return x.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def post_save_linf(orig_uint8: np.ndarray, adv_uint8: np.ndarray) -> float:
    """Max absolute pixel difference, in [0,1] units."""
    return float(np.abs(orig_uint8.astype(np.int16) - adv_uint8.astype(np.int16)).max() / 255.0)
