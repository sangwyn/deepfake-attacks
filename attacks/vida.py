"""ViDA — Visibility-guided Dual-domain Adaptive attack (self-contained).

A white-box targeted attack for AADD-2026 against the two detectors
(``vit_b_16`` spatial and ``densenet121_dct`` frequency). It matches the
team template ``attack(image, classifiers, device)`` used by ``evaluate.py``.

Pipeline (research details in docs/PLAN.md):
  * Differentiable replicas of BOTH detector preprocessing paths, so white-box
    gradients flow end to end. The DCT log(|-|) uses a magnitude-capped custom
    backward (forward stays identical to the evaluator) to avoid gradient
    explosion on near-zero high-frequency coefficients.
  * Stage A (acquisition): MI-FGSM over the SUM of both detectors' targeted
    margin losses, with a border mask (pixels neither detector sees are never
    perturbed). Stops as soon as every available detector is fooled, then runs
    a short "tail" of extra joint steps for hard images.
  * Stage B (quality recovery): freeze the fooling margins and maximise
    perceptual quality (differentiable SSIM + LPIPS-Alex, the exact metric the
    evaluator uses), accepting a step only if it never loses an already-fooled
    detector. This is the main score driver for the LPIPS-weighted 2026 score.

Hyper-parameters are module-level constants (edit to taste). Returns a
same-size HWC uint8 RGB adversarial image.

Black-box transfer: set the module constant DI_PROB = 0.5 (e.g.
`import attacks.vida as vida; vida.DI_PROB = 0.5` before evaluation) to enable
Diverse-Inputs during acquisition. DI raises cross-model transfer at a small
cost to white-box fooling; leave DI_PROB = 0.0 for the best official score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

EPS = 8.0 / 255.0
STEP = 2.0 / 255.0
STEPS = 80
TAIL_STEPS = 60
REC_STEPS = 40
REC_STEP = 0.5 / 255.0
MU = 1.0
KAPPA = 0.5
LAMBDA_Q = 2.0
# Input-diversity (DI, Xie et al.) for black-box transfer. Off by default
# (it slightly lowers white-box fooling); set DI_PROB > 0 to randomly
# resize+pad the input during acquisition, which boosts cross-model transfer.
DI_PROB = 0.0
DI_SCALE_LO = 0.8
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAY = (0.299, 0.587, 0.114)


# --------------------------------------------------------------------------- #
# Differentiable DCT + stabilized log + Lanczos resize (PIL-faithful)
# --------------------------------------------------------------------------- #
def _dct_matrix(n, device, dtype=torch.float32):
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    j = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    d = torch.cos(torch.pi / n * (j + 0.5) * k)
    d = d * torch.sqrt(torch.tensor(2.0 / n, device=device, dtype=dtype))
    d[0, :] = d[0, :] / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return d


def _sinc(x):
    return torch.where(x.abs() < 1e-12, torch.ones_like(x),
                       torch.sin(torch.pi * x) / (torch.pi * x))


def _lanczos_matrix(n_in, n_out, device, support=3.0):
    """Differentiable replica of PIL Image.resize(LANCZOS) for one axis."""
    scale = n_in / n_out
    fscale = scale if scale > 1.0 else 1.0
    m = torch.zeros(n_out, n_in, device=device)
    half = support * fscale
    for i in range(n_out):
        center = (i + 0.5) * scale - 0.5
        lo = int(torch.ceil(torch.tensor(center - half)).item())
        hi = int(torch.floor(torch.tensor(center + half)).item())
        for j in range(max(0, lo), min(n_in, hi + 1)):
            x = (j - center) / fscale
            if abs(x) < support:
                m[i, j] = _sinc(torch.tensor(x)) * _sinc(torch.tensor(x / support))
        s = m[i].sum()
        if s != 0:
            m[i] /= s
    return m


class _StableLogAbs(torch.autograd.Function):
    """Forward = log(|x| + 1e-6) (identical to evaluator); backward capped."""

    @staticmethod
    def forward(ctx, x, eps_fwd, eps_grad):
        ctx.save_for_backward(x)
        ctx.eps_grad = float(eps_grad)
        return torch.log(x.abs() + eps_fwd)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.sign(x) / (x.abs() + ctx.eps_grad), None, None


def _stable_log_abs(x, eps_fwd=1e-6, eps_grad=0.01):
    return _StableLogAbs.apply(x, eps_fwd, eps_grad)


# --------------------------------------------------------------------------- #
# Detector branches (differentiable replicas of evaluate.py preprocessing)
# --------------------------------------------------------------------------- #
class ViTBranch(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = TF.resize(x, [256, 256], antialias=True)
        x = TF.center_crop(x, [224, 224])
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        return self.model(x)


class DCTBranch(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        w = torch.tensor(GRAY, device=device).view(1, 3, 1, 1)
        self.register_buffer("gray_w", w)
        self.register_buffer("dct_m", _dct_matrix(128, device))
        self._lz = {}

    def _resize_gray(self, g):
        """g: (N,1,H,W) in 0..255 -> (N,1,256,256) via PIL-faithful Lanczos."""
        n, _, h, w = g.shape
        key = (h, w)
        if key not in self._lz:
            self._lz[key] = (_lanczos_matrix(h, 256, g.device),
                             _lanczos_matrix(w, 256, g.device))
        mh, mw = self._lz[key]
        return (mh @ g[:, 0] @ mw.t()).unsqueeze(1)

    def forward(self, x):
        g = (x * self.gray_w).sum(dim=1, keepdim=True)   # grayscale in 0..1
        g = g * 255.0                                     # evaluator uses uint8 0..255
        g = self._resize_gray(g)                          # PIL LANCZOS -> 256
        g = g[:, :, 64:192, 64:192][:, 0]                 # center 128 grayscale
        coeff = torch.matmul(torch.matmul(self.dct_m, g), self.dct_m.t())
        feat = _stable_log_abs(coeff).unsqueeze(1)
        return self.model(feat)


class DiffSSIM(nn.Module):
    def __init__(self, channels=3, win=7):
        super().__init__()
        self.channels = channels
        self.c1 = (0.01 * 255.0) ** 2
        self.c2 = (0.03 * 255.0) ** 2
        k = torch.ones(channels, 1, win, win) / (win * win)
        self.register_buffer("kernel", k)
        self.pad = win // 2

    def _mu(self, x):
        return F.conv2d(x, self.kernel, padding=self.pad, groups=self.channels)

    def forward(self, x, y):
        x = x * 255.0
        y = y * 255.0
        mx, my = self._mu(x), self._mu(y)
        sx = self._mu(x * x) - mx * mx
        sy = self._mu(y * y) - my * my
        sxy = self._mu(x * y) - mx * my
        m = ((2 * mx * my + self.c1) * (2 * sxy + self.c2)) / (
            (mx * mx + my * my + self.c1) * (sx + sy + self.c2)
        )
        return m.mean()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _margin_loss(logits, target=0):
    wrong = logits.clone()
    wrong[:, target] = -1e9
    return F.softplus(wrong.max(dim=1).values - logits[:, target] + KAPPA).mean()


def _margin(branch, adv, target=0):
    with torch.no_grad():
        z = branch(adv)[0]
    return float(z[1 - target] - z[target])


def _di(x, prob=DI_PROB, lo=DI_SCALE_LO, hi=1.0):
    """Diverse-Inputs: with probability `prob`, randomly down-scale and
    zero-pad the adversarial image (differentiable). Improves transfer to
    unseen detectors; the official detector preprocessing is unchanged."""
    if prob <= 0.0 or torch.rand(1, device=x.device).item() > prob:
        return x
    n, c, h, w = x.shape
    s = lo + (hi - lo) * torch.rand(1, device=x.device).item()
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    xs = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    out = torch.zeros_like(x)
    top = int(torch.randint(0, h - nh + 1, (1,), device=x.device).item())
    left = int(torch.randint(0, w - nw + 1, (1,), device=x.device).item())
    out[:, :, top:top + nh, left:left + nw] = xs
    return out


# --------------------------------------------------------------------------- #
# Main entry point (team template)
# --------------------------------------------------------------------------- #
def attack(image, classifiers, device,
           epsilon=EPS, step=STEP, steps=STEPS,
           tail_steps=TAIL_STEPS, rec_steps=REC_STEPS, target=0):
    """Targeted attack. target=0 -> fake->real; target=1 -> real->fake."""
    device = torch.device(device)
    branches = {}
    if "vit_b_16" in classifiers:
        branches["vit_b_16"] = ViTBranch(classifiers["vit_b_16"]["model"]).to(device)
    if "densenet121_dct" in classifiers:
        branches["densenet121_dct"] = DCTBranch(
            classifiers["densenet121_dct"]["model"], device).to(device)
    if not branches:
        return image

    for b in branches.values():
        for p in b.model.parameters():
            p.requires_grad_(False)
        b.eval()

    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # border mask: forbid perturbation outside every detector's field of view.
    # ViT sees the central 224 of the 256 canonical grid; zero the outer ring.
    m256 = torch.ones(1, 1, 256, 256, device=device)
    m256[:, :, :16, :] = 0.0
    m256[:, :, -16:, :] = 0.0
    m256[:, :, :, :16] = 0.0
    m256[:, :, :, -16:] = 0.0
    allow = F.interpolate(m256, size=x.shape[-2:], mode="nearest")

    def adv_of(d):
        return (x + d).clamp(0.0, 1.0)

    def loss_on(d):
        adv = _di(adv_of(d)) if DI_PROB > 0 else adv_of(d)
        return sum(_margin_loss(b(adv), target) for b in branches.values())

    def fooled_count(d):
        return sum(1 for b in branches.values() if _margin(b, adv_of(d), target) < 0)

    n_br = len(branches)

    # ---- Stage A: MI-FGSM acquisition + early stop ----
    delta = torch.zeros_like(x)
    acc = torch.zeros_like(x)
    total = steps
    for it in range(steps):
        delta = delta.detach().requires_grad_(True)
        g = torch.autograd.grad(loss_on(delta), delta)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = MU * acc + g
        with torch.no_grad():
            delta = (delta - step * acc.sign()) * allow
            delta = delta.clamp(-epsilon, epsilon)
        if fooled_count(delta) == n_br:
            total = it + 1
            break

    # ---- Stage A.2: tail — keep pushing on images any detector still misses ----
    for _ in range(tail_steps):
        if fooled_count(delta) == n_br:
            break
        delta = delta.detach().requires_grad_(True)
        g = torch.autograd.grad(loss_on(delta), delta)[0]
        g = g / (g.abs().mean() + 1e-12)
        acc = MU * acc + g
        with torch.no_grad():
            delta = (delta.detach() - step * acc.sign()) * allow
            delta = delta.clamp(-epsilon, epsilon)
            total += 1

    # ---- Stage B: quality recovery (never lose an already-fooled detector) ----
    locks = [_margin(b, adv_of(delta), target) < 0 for b in branches.values()]
    if any(locks):
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
            has_lpips = True
        except Exception:
            has_lpips = False
        diff_ssim = DiffSSIM().to(device)
        base = (x * 2 - 1).detach()

        def holds(d):
            ok = True
            for b, locked in zip(branches.values(), locks):
                if locked:
                    ok = ok and (_margin(b, adv_of(d), target) < 0)
            return ok

        best_q, best_d = -1.0, delta.clone()
        for _ in range(rec_steps):
            delta = delta.detach().requires_grad_(True)
            adv = adv_of(delta)
            q_s = diff_ssim(x, adv)
            q_l = lpips_fn(base, adv * 2 - 1).mean() if has_lpips else torch.zeros((), device=device)
            q = 0.5 * q_s + 0.5 * (1.0 - q_l) if has_lpips else q_s
            margin_term = sum(_margin_loss(b(adv), target) for b in branches.values())
            loss = margin_term + LAMBDA_Q * (1.0 - q)
            g = torch.autograd.grad(loss, delta)[0]
            with torch.no_grad():
                cand = (delta - REC_STEP * g.sign()).clamp(-epsilon, epsilon)
                if holds(cand):
                    delta = cand
                    if float(q.item()) > best_q:
                        best_q, best_d = float(q.item()), cand.clone()
        delta = best_d

    with torch.no_grad():
        out = (x + delta).clamp(0.0, 1.0)[0].permute(1, 2, 0) * 255.0
    return out.round().to(torch.uint8).cpu().numpy()
