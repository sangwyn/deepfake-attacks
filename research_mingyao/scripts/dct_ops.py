"""Differentiable DCT-II and grayscale ops matching the official evaluator.

Official (non-differentiable) path in evaluate.py:
    PIL.convert('L')  ->  resize 256 (LANCZOS, only if max(size)>256)
    -> center crop 128 -> scipy.fftpack.dct along axes 0 and 1, norm='ortho'
    -> log(|x| + 1e-6)

This module provides an end-to-end differentiable replica using fixed tensor
ops, so gradients can flow into the input image for white-box attacks.
"""
import math

import torch

# ITU-R 601 luma weights used by PIL convert('L'): L = 0.299R + 0.587G + 0.114B
GRAY_W = (0.299, 0.587, 0.114)


def dct_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix D of shape (n, n).

    2-D orthonormal DCT of x is  D @ x @ D.T .
    Matches scipy.fftpack.dct(..., norm='ortho') applied on both axes.
    """
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    j = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    d = torch.cos(torch.pi / n * (j + 0.5) * k)
    d = d * torch.sqrt(torch.tensor(2.0 / n, device=device, dtype=dtype))
    # k=0 row uses alpha_0 = sqrt(1/N) = sqrt(2/N) / sqrt(2)
    d[0, :] = d[0, :] / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return d


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: (N,3,H,W) in [0,1] -> (N,1,H,W) luma."""
    w = torch.tensor(GRAY_W, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def dct2_ortho(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Separable orthonormal 2-D DCT-II.

    x: (..., H, W); d: (H, H) DCT matrix (square, H==W for our crops).
    returns (..., H, W) DCT coefficients.
    """
    return torch.matmul(torch.matmul(d, x), d.t())


def _sinc(x: float) -> float:
    import math
    if abs(x) < 1e-12:
        return 1.0
    px = math.pi * x
    return math.sin(px) / px


def lanczos_matrix(n_in: int, n_out: int, support: float = 3.0,
                   device=None, dtype=torch.float64) -> torch.Tensor:
    """Differentiable replica of PIL Image.resize(..., LANCZOS) for ONE axis.

    Returns an (n_out, n_in) linear resampling matrix (Lanczos-3, antialiased),
    matching Pillow's Resampling.LANCZOS: for downscale the kernel support and
    filter scale stretch by n_in/n_out, kernel = sinc(x)*sinc(x/support) with a
    truncated support, then each output tap set is normalised to sum 1.
    """
    scale = n_in / n_out          # >1 for downsampling
    fscale = scale if scale > 1.0 else 1.0
    m = torch.zeros(n_out, n_in, dtype=dtype)
    half = support * fscale
    for i in range(n_out):
        center = (i + 0.5) * scale - 0.5
        lo = int(math.ceil(center - half))
        hi = int(math.floor(center + half))
        acc = 0.0
        for j in range(max(0, lo), min(n_in, hi + 1)):
            x = (j - center) / fscale
            if abs(x) < support:
                w = _sinc(x) * _sinc(x / support)
                m[i, j] = w
                acc += w
        if acc != 0.0:
            m[i] /= acc
    return m.to(device=device, dtype=torch.float32)


class _StableLogAbs(torch.autograd.Function):
    """log(|x| + eps_fwd) in the FORWARD pass (identical to the evaluator's
    numpy log), but with a magnitude-capped gradient in the BACKWARD pass.

    Naive d/dx log(|x|+e) = sign(x)/(|x|+e) explodes (~1/e ~ 1e6) on the many
    near-zero high-frequency DCT coefficients, which then dominate the sign
    gradient and make the attack ineffective. We cap the gradient magnitude via
    a soft denominator eps_grad while keeping the correct sign and the exact
    forward output.
    """

    @staticmethod
    def forward(ctx, x, eps_fwd, eps_grad):
        ctx.save_for_backward(x)
        ctx.eps_grad = float(eps_grad)
        return torch.log(x.abs() + eps_fwd)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        grad = grad_out * torch.sign(x) / (x.abs() + ctx.eps_grad)
        return grad, None, None


def stable_log_abs(x: torch.Tensor, eps_fwd: float = 1e-6,
                   eps_grad: float = 0.1) -> torch.Tensor:
    return _StableLogAbs.apply(x, eps_fwd, eps_grad)
