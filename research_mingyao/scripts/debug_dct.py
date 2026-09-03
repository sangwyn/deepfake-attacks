"""One-image deep debug: watch DCT logits/margin across I-FGSM steps."""
import torch
import torchvision.transforms.functional as TF
from pathlib import Path

from .detectors import load_detectors
from .data import build_manifest, load_image_tensor
from .attacks import targeted_margin_loss
from .dct_ops import stable_log_abs
from . import detectors as DET

WEIGHTS = Path("/data2/aiattacks/Mingyao-Duan/team_repo/weights")
DATA = Path("/data2/aiattacks/Mingyao-Duan/dataset".replace("Mingyao-Duan/", ""))
DATA = Path("/data2/aiattacks/dataset")


def margin(logits, target=0):
    z = logits[0]
    other = z[1 - target]
    return (other - z[target]).item()


def attack_watch(x256, branch, steps=50, alpha=2/255, eps=8/255, label=""):
    delta = torch.zeros_like(x256)
    print(f"  [{label}] start margin(z_fake-z_real)={margin(branch((x256).clamp(0,1))):.3f} "
          f"logits={branch(x256)[0].detach().cpu().numpy().round(2)}")
    for it in range(steps):
        delta = delta.detach().requires_grad_(True)
        logits = branch((x256 + delta).clamp(0, 1))
        loss = targeted_margin_loss(logits, 0)
        g = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = (delta - alpha * g.sign()).clamp(-eps, eps)
        if it % 10 == 0 or it == steps - 1:
            with torch.no_grad():
                lg = branch((x256 + delta).clamp(0, 1))[0].detach().cpu().numpy()
            print(f"    step {it:3d} margin={margin(branch((x256+delta).clamp(0,1))):8.3f} "
                  f"logits={lg.round(2)} |g|mean={g.abs().mean().item():.3e} "
                  f"|g|max={g.abs().max().item():.2f}")


def main():
    device = torch.device("cuda")
    packs = load_detectors(WEIGHTS, device, use_dct=True, use_vit=False)
    pack = packs["densenet121_dct"]
    branch = pack["branch"]
    fakes = [m["path"] for m in build_manifest(DATA) if m["cls"] == "fake"][:2]

    for i, p in enumerate(fakes):
        x_full, _ = load_image_tensor(p, device)
        x256 = TF.resize(x_full, [256, 256], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        print(f"Image {i}: {Path(p).name}")
        attack_watch(x256, branch, steps=50, label="eps_g=0.1 (current)")

    # try a much weaker gradient cap (closer to raw log) on image 0
    print("\n--- sweep eps_grad on image 0, 40 steps ---")
    x_full, _ = load_image_tensor(fakes[0], device)
    x256 = TF.resize(x_full, [256, 256], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    for eg in (0.1, 1e-2, 1e-3, 1e-4):
        # monkeypatch stable log eps via closure: rebuild feat by hooking
        orig_fwd = branch.forward
        def make(egv):
            def fwd(x):
                from .dct_ops import rgb_to_gray, dct_matrix
                g = rgb_to_gray(x)
                g = TF.resize(g, [256,256], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
                g = g[:, :, 64:192, 64:192][:,0]
                Dm = branch.dct_m
                coeff = torch.matmul(torch.matmul(Dm, g), Dm.t())
                feat = stable_log_abs(coeff, 1e-6, egv).unsqueeze(1)
                return branch.model(feat)
            return fwd
        branch.forward = make(eg)
        delta = torch.zeros_like(x256)
        for it in range(40):
            delta = delta.detach().requires_grad_(True)
            loss = targeted_margin_loss(branch((x256+delta).clamp(0,1)), 0)
            g = torch.autograd.grad(loss, delta)[0]
            with torch.no_grad():
                delta = (delta - (2/255)*g.sign()).clamp(-8/255, 8/255)
        m = margin(branch((x256+delta).clamp(0,1)))
        print(f"  eps_grad={eg:7.4f} -> final margin={m:8.3f} {'FLIPPED' if m<0 else ''}")
        branch.forward = orig_fwd


if __name__ == "__main__":
    main()
