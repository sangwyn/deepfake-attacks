"""Quality metrics.

- official_quality(): matches the evaluator's SSIM (skimage, per-channel,
  data_range=255) and LPIPS (AlexNet, inputs in [-1,1]). Used for reporting
  and for post-save verification.
- DiffSSIM: a differentiable SSIM proxy (uniform 7x7 window) used only to give
  the Stage-B quality-recovery optimizer a gradient. It is a proxy; reported
  numbers always come from official_quality().
"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim


def np_to_lpips_tensor(arr_uint8_hwc: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(arr_uint8_hwc).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def official_quality(orig_uint8: np.ndarray, adv_uint8: np.ndarray,
                     lpips_fn, device) -> tuple[float, float]:
    """Return (ssim, lpips) exactly in the evaluator's convention."""
    ssim_val = sum(
        sk_ssim(orig_uint8[..., c], adv_uint8[..., c], data_range=255)
        for c in range(3)
    ) / 3.0
    with torch.no_grad():
        lpips_val = lpips_fn(
            np_to_lpips_tensor(orig_uint8, device),
            np_to_lpips_tensor(adv_uint8, device),
        ).item()
    return float(ssim_val), float(lpips_val)


def quality_score(ssim: float, lpips: float, alpha: float = 0.5) -> float:
    """Evaluator similarity weight: alpha*SSIM + (1-alpha)*(1-LPIPS)."""
    return alpha * ssim + (1.0 - alpha) * (1.0 - lpips)


class DiffSSIM(torch.nn.Module):
    """Differentiable SSIM proxy on (N,3,H,W) tensors in [0,1]."""

    def __init__(self, channels: int = 3, win: int = 7, data_range: float = 255.0):
        super().__init__()
        self.channels = channels
        self.win = win
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2
        kernel = torch.ones(channels, 1, win, win) / (win * win)
        self.register_buffer("kernel", kernel)

    def _mu(self, x):
        pad = self.win // 2
        return F.conv2d(x, self.kernel, padding=pad, groups=self.channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x * 255.0
        y = y * 255.0
        mu_x, mu_y = self._mu(x), self._mu(y)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sx = self._mu(x * x) - mu_x2
        sy = self._mu(y * y) - mu_y2
        sxy = self._mu(x * y) - mu_xy
        ssim_map = ((2 * mu_xy + self.c1) * (2 * sxy + self.c2)) / (
            (mu_x2 + mu_y2 + self.c1) * (sx + sy + self.c2)
        )
        return ssim_map.mean()
