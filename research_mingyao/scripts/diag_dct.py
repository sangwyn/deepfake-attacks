"""Diagnostic: is the DCT attack weak, or is perturbation lost to resize?

Runs targeted fake->real I-FGSM on the 8 fake images in three parameterizations:
  canon : optimize a 256x256 delta and feed the 256 image straight to the branch
          (full 8/255 budget reaches the model; isolates model-space strength).
  full  : optimize a full-res delta (evaluator-consistent; budget at 1024).
  full-smooth : full-res delta but step through a low-pass (momentum + sign on
          a blurred delta) to favour resize-surviving smooth perturbation.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from . import data as D
from . import attacks as A
from .detectors import load_detectors
from .data import build_manifest, load_image_tensor, to_uint8_hwc
from .attacks import targeted_margin_loss
from pathlib import Path

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/team_repo/weights")
DATA = Path("/data2/aiattacks/dataset")


def run(imgs, device, pack, mode, steps=20, eps=8/255, alpha=2/255, mom=0.0):
    model, branch = pack["model"], pack["branch"]
    fooled = 0
    for path in imgs:
        x_full, _ = load_image_tensor(path, device)
        with torch.no_grad():
            clean = branch(x_full).argmax(1).item()
        if mode == "canon":
            x256 = TF.resize(x_full, [256, 256], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            delta = torch.zeros_like(x256)
            acc = torch.zeros_like(delta)
            for it in range(steps):
                delta = delta.detach().requires_grad_(True)
                logits = branch((x256 + delta).clamp(0, 1))
                loss = targeted_margin_loss(logits, 0)
                g = torch.autograd.grad(loss, delta)[0]
                if mom > 0:
                    g = g / (g.abs().mean() + 1e-12); acc = mom * acc + g; g = acc
                with torch.no_grad():
                    delta = (delta - alpha * g.sign()).clamp(-eps, eps)
            with torch.no_grad():
                pred = branch((x256 + delta).clamp(0, 1)).argmax(1).item()
        else:
            n, _, h, w = x_full.shape
            delta = torch.zeros((n, 3, h, w), device=device)
            acc = torch.zeros_like(delta)
            blur = torch.tensor([0.25, 0.5, 0.25], device=device)
            for it in range(steps):
                delta = delta.detach().requires_grad_(True)
                logits = branch((x_full + delta).clamp(0, 1))
                loss = targeted_margin_loss(logits, 0)
                g = torch.autograd.grad(loss, delta)[0]
                if mom > 0:
                    g = g / (g.abs().mean() + 1e-12); acc = mom * acc + g; g = acc
                with torch.no_grad():
                    s = g.sign()
                    if mode == "full-smooth":
                        # separable 3x3 blur on the sign -> smoother, survives resize
                        k = blur.view(1, 1, 1, 3).expand(3, 1, 1, 3)
                        s = F.conv2d(s, k, padding=(0, 1), groups=3)
                        k2 = blur.view(1, 1, 3, 1).expand(3, 1, 3, 1)
                        s = F.conv2d(s, k2, padding=(1, 0), groups=3)
                    delta = (delta - alpha * s.sign() if mode == "full" else delta - alpha * s).clamp(-eps, eps)
            with torch.no_grad():
                pred = branch((x_full + delta).clamp(0, 1)).argmax(1).item()
        fooled += int(pred == 0)
    return fooled, len(imgs)


def main():
    device = torch.device("cuda")
    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=False)
    pack = packs["densenet121_dct"]
    fakes = [m["path"] for m in build_manifest(DATA) if m["cls"] == "fake"][:8]

    for steps in (50, 100):
        for mode in ("canon", "full"):
            f, n = run(fakes, device, pack, mode, steps=steps)
            print(f"steps={steps:3d}  mode={mode:12s}  ASR={f}/{n}")
    f, n = run(fakes, device, pack, "full", steps=100, mom=1.0)
    print(f"steps=100  mode=full + MI momentum      ASR={f}/{n}")

    # broader baseline: 32 fakes, full+MI, with quality metrics
    import lpips, numpy as np
    from .metrics import official_quality, quality_score
    from .data import to_uint8_hwc, post_save_linf
    lp = lpips.LPIPS(net="alex").to(device).eval()
    fakes32 = [m["path"] for m in build_manifest(DATA) if m["cls"] == "fake"][:32]
    ss, ll, linf_max, fl = [], [], 0.0, 0
    for pth in fakes32:
        x_full, orig = load_image_tensor(pth, device)
        adv = A.ifgsm(x_full, packs, device, target=0, steps=60,
                      use=("densenet121_dct",), momentum=1.0)
        adv_u = to_uint8_hwc(adv)
        s, l = official_quality(orig, adv_u, lp, device)
        pred = pack["branch"](adv).argmax(1).item()
        fl += int(pred == 0)
        ss.append(s); ll.append(l); linf_max = max(linf_max, post_save_linf(orig, adv_u))
    ss = np.mean(ss); ll = np.mean(ll)
    print(f"\n[DCT baseline full+MI 60steps on 32 fakes] "
          f"ASR={fl}/{len(fakes32)}={fl/len(fakes32):.3f}  "
          f"SSIM={ss:.4f} LPIPS={ll:.4f} Q={quality_score(ss,ll):.4f} "
          f"maxLinf={linf_max:.4f}")


if __name__ == "__main__":
    main()
