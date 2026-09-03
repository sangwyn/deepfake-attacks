"""Attack optimizers. All attacks operate on a full-res image tensor
(N,3,H,W) in [0,1] and return the adversarial full-res tensor in [0,1].

The optimization variable is delta at the canonical 256x256 grid; it is
bilinearly upsampled to full resolution before each detector forward pass
(so branches see the same full-res image the evaluator will score), and the
saved adversarial image preserves the original high-frequency detail plus a
smooth upsampled perturbation.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .masks import CANON, allowed_mask


def targeted_margin_loss(logits: torch.Tensor, target: int,
                         kappa: float = 0.0) -> torch.Tensor:
    """softplus(z_best_other - z_target + kappa); minimize for targeted attack."""
    wrong = logits.clone()
    wrong[:, target] = -1e9
    z_other = wrong.max(dim=1).values
    z_target = logits[:, target]
    return F.softplus(z_other - z_target + kappa).mean()


def identity(x_full, packs, device, **kw):
    return x_full.detach()


def ifgsm(x_full: torch.Tensor, packs: dict, device,
          target: int = 0, eps: float = 8 / 255, alpha: float = 2 / 255,
          steps: int = 10, use: tuple[str, ...] = ("densenet121_dct",),
          use_border_mask: bool = False, momentum: float = 0.0,
          verbose: bool = False):
    """Targeted I-FGSM (optionally MI-FGSM).

    The optimization variable is a FULL-RESOLUTION delta (same H,W as input);
    each detector branch differentiably resizes it to its canonical size, so
    gradients already encode only perturbation that survives the evaluator's
    resize. eps-Linf is enforced directly on the full-res delta, hence post-save
    norm is exact. target=0 -> fake->real ; target=1 -> real->fake.
    Targeted attack MINIMIZES the margin loss -> step opposite the sign.
    """
    n, _, h, w = x_full.shape
    delta = torch.zeros((n, 3, h, w), device=device)
    mask256 = allowed_mask(use_border_mask=use_border_mask, device=device)
    mask = F.interpolate(mask256, size=(h, w), mode="nearest")
    grad_acc = torch.zeros_like(delta)

    for it in range(steps):
        delta = delta.detach().requires_grad_(True)
        adv_full = (x_full + delta).clamp(0.0, 1.0)

        loss = delta.new_zeros(())
        for name in use:
            pack = packs[name]
            logits = pack["branch"](adv_full)
            loss = loss + targeted_margin_loss(logits, target)

        grad = torch.autograd.grad(loss, delta)[0]
        if momentum > 0:
            grad = grad / (grad.abs().mean() + 1e-12)
            grad_acc = momentum * grad_acc + grad
            step_dir = grad_acc
        else:
            step_dir = grad

        with torch.no_grad():
            # targeted: descend the loss
            delta = delta - alpha * step_dir.sign()
            delta = delta * mask
            delta = delta.clamp(-eps, eps)

        if verbose and (it % max(1, steps // 5) == 0 or it == steps - 1):
            print(f"    step {it:02d}  loss={loss.item():.4f}")

    with torch.no_grad():
        adv_full = (x_full + delta).clamp(0.0, 1.0)
    return adv_full.detach()
