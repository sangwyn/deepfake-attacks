"""ViDA — Visibility-guided Dual-domain Adaptive attack (self-contained).

A white-box targeted attack for AADD-2026 against the two detectors
(``vit_b_16`` spatial and ``densenet121_dct`` frequency). It matches the
team template ``attack(image, classifiers, device)`` used by ``evaluate.py``.

Version: ViDA v3.4-final-no-stage0. EVERY image goes through the full attack
pipeline — there is no clean-skip and no clean-prediction shortcut.

Pipeline (research details in docs/PLAN.md):
  * Differentiable replicas of BOTH detector preprocessing paths, so white-box
    gradients flow end to end. The DCT log(|-|) uses a magnitude-capped custom
    backward (forward stays identical to the evaluator) to avoid gradient
    explosion on near-zero high-frequency coefficients.
  * Stage A (acquisition): MI-FGSM over the SUM of both detectors' targeted
    margin losses, with a border mask (pixels neither detector sees are never
    perturbed). Stops as soon as every available detector is fooled, then runs
    a short "tail" of extra joint steps for hard images.
  * Stage B (line search): binary search the smallest perturbation scale that
    still fools every detector, gated by the evaluator-EXACT uint8 prediction
    (PIL transforms + rounding included). Strips redundant magnitude.
  * Stage C (quality recovery): Adam on the continuous perturbation maximises
    perceptual quality (differentiable SSIM + LPIPS-Alex, the exact metric the
    evaluator uses) with a soft margin buffer annealed over phases; a step is
    accepted only if the evaluator-exact uint8 check still passes for every
    already-fooled detector. Phases reset momentum and re-run Stage B.
  * Stage D (uint8 verification): re-run the official transforms on the
    rounded image; any detector that slipped gets extra MI-FGSM steps. The best
    uint8 image (detectors fooled, then Q) seen anywhere is always kept.
  * Stage E (hard-case rescue): images that still end below the Q threshold
    (or not fully fooled) get additional passes with finer/different
    acquisition steps, random PGD starts and per-pass Diverse-Inputs; the best
    image across all passes is returned (multi-start / best-of-k).

LPIPS/SSIM models are cached globally (one load per process).

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
from PIL import Image

EPS = 8.0 / 255.0
STEP = 2.0 / 255.0
STEPS = 80
TAIL_STEPS = 60
REC_STEPS = 50
MU = 1.0
KAPPA = 0.5
# Quality-recovery (Stage B): Adam on the continuous perturbation, maximising
# 0.5*SSIM + 0.5*(1-LPIPS) with a soft margin buffer on fooled detectors.
REC_LR = 0.4 / 255.0
KAPPA_REC = 1.0
LAMBDA_REC = 3.0
# Stage B: binary line search on the perturbation scale (uint8-gated).
LINESEARCH_ITERS = 12
# Stage D: re-run the official transforms on the rounded image; any detector
# that slipped gets extra MI-FGSM steps.
VERIFY_ROUNDS = 3
REATTACK_STEPS = 20
# Stage E (rescue passes): images that end pass 1 with Q below RESCUE_Q (or with
# a detector still not fooled) get extra passes with different (step size,
# random-start) configurations. The uint8 accept gate sits right at the
# decision boundary, so the recovery trajectory is chaotic run-to-run — cheap
# solutions exist but are only hit on some runs; best-of-k with random PGD
# starts turns that variance into a reliable win. The best image across all
# passes (most detectors fooled, then highest Q) is kept.
RESCUE_Q = 0.92
RESCUE_CONFIGS = [
    # (acquisition step, acq steps, tail steps, random-start radius, DI prob)
    (1.0 / 255.0, 160, 80, 2.0 / 255.0, 0.0),
    (2.0 / 255.0, 120, 60, 2.0 / 255.0, 0.0),
    (1.0 / 255.0, 160, 80, 5.0 / 255.0, 0.5),   # DI + big random start
    (2.0 / 255.0, 120, 60, 5.0 / 255.0, 0.5),
    (1.0 / 255.0, 200, 100, 6.0 / 255.0, 0.3),
    (1.0 / 255.0, 160, 80, 0.0, 0.0),           # zero-start re-run (numeric lottery)
]
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
# Quality-model cache + evaluator-exact (ground-truth) helpers
# --------------------------------------------------------------------------- #
_QCACHE = {}


def _get_qmodels(device):
    """LPIPS-Alex + differentiable SSIM are loaded ONCE per process/device and
    reused across images (the evaluator scores hundreds of images; reloading
    the AlexNet weights per image is pure waste)."""
    key = str(device)
    if key not in _QCACHE:
        import lpips
        lp = lpips.LPIPS(net="alex").to(device).eval()
        for p in lp.parameters():
            p.requires_grad_(False)
        _QCACHE[key] = (lp, DiffSSIM().to(device))
    return _QCACHE[key]


def _uint8_hwc(x, d):
    """(x + d), clipped and rounded -> HWC uint8 numpy. This is exactly what
    the evaluator saves and scores, so all gates use it."""
    with torch.no_grad():
        u = ((x + d).clamp(0.0, 1.0)[0].permute(1, 2, 0) * 255.0)
    return u.round().to(torch.uint8).cpu().numpy()


def _gt_bad(u_img, classifiers, names, device, target):
    """Run the evaluator's EXACT transforms + models on a uint8 HWC image and
    return the names of detectors that do NOT predict `target`. Ground-truth
    gate: PIL resize/crop, scipy DCT and uint8 rounding are all included."""
    pil = Image.fromarray(u_img)
    bad = []
    for n in names:
        pack = classifiers.get(n)
        if pack is None or pack.get("transform") is None:
            continue
        with torch.no_grad():
            t = pack["transform"](pil).unsqueeze(0).to(device)
            pred = int(pack["model"](t).argmax(1).item())
        if pred != target:
            bad.append(n)
    return bad


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

    names = list(branches)
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # NOTE (v3.4-final-no-stage0): every image goes through the full attack
    # pipeline. There is deliberately NO clean-skip / no prediction shortcut:
    # we never inspect the clean prediction to decide whether to attack.

    # border mask: forbid perturbation outside every detector's field of view.
    # ViT sees the central 224 of the 256 canonical grid; zero the outer ring.
    m256 = torch.ones(1, 1, 256, 256, device=device)
    m256[:, :, :16, :] = 0.0
    m256[:, :, -16:, :] = 0.0
    m256[:, :, :, :16] = 0.0
    m256[:, :, :, -16:] = 0.0
    allow = F.interpolate(m256, size=x.shape[-2:], mode="nearest")

    lpips_fn, ssim_fn = _get_qmodels(device)
    base = (x * 2 - 1).detach()
    best_u, best_key = None, (-1, -1.0)

    def adv_of(d):
        return (x + d).clamp(0.0, 1.0)

    pass_cfg = {"di": DI_PROB}

    def loss_on(d):
        adv = _di(adv_of(d), prob=pass_cfg["di"]) if pass_cfg["di"] > 0 \
            else adv_of(d)
        return sum(_margin_loss(b(adv), target) for b in branches.values())

    def float_fooled(d):
        adv = adv_of(d)
        return all(_margin(b, adv, target) < 0 for b in branches.values())

    def q_of(d):
        with torch.no_grad():
            adv = adv_of(d)
            q_s = ssim_fn(x, adv)
            q_l = lpips_fn(base, adv * 2 - 1).mean()
        return float((0.5 * q_s + 0.5 * (1.0 - q_l)).item())

    def remember(d):
        """Keep the best uint8 image seen: first maximise detectors fooled,
        then perceptual quality Q."""
        nonlocal best_u, best_key
        u = _uint8_hwc(x, d)
        n_good = len(names) - len(_gt_bad(u, classifiers, names, device, target))
        key = (n_good, q_of(d))
        if key > best_key:
            best_key, best_u = key, u
        return u

    def run_pass(step=STEP, steps=STEPS, tail_steps=TAIL_STEPS,
                 rand_start=0.0, di=0.0):
        """One full acquisition -> line search -> Adam recovery -> verify
        cycle, updating the global best (best_u/best_key via remember).
        rand_start > 0 initialises the perturbation uniformly in a small L∞
        ball (PGD random start); di > 0 enables Diverse-Inputs during
        acquisition. Both diversify trajectories across rescue passes."""
        pass_cfg["di"] = di if di > 0 else DI_PROB
        # ---- Stage A: MI-FGSM acquisition + early stop ----
        # (both detectors' targeted margin losses, border mask, momentum)
        if rand_start > 0.0:
            delta = (torch.rand_like(x) * 2.0 - 1.0) * rand_start
            delta = (delta * allow).clamp(-epsilon, epsilon).detach()
        else:
            delta = torch.zeros_like(x)
        acc = torch.zeros_like(x)
        for it in range(steps):
            delta = delta.detach().requires_grad_(True)
            g = torch.autograd.grad(loss_on(delta), delta)[0]
            g = g / (g.abs().mean() + 1e-12)
            acc = MU * acc + g
            with torch.no_grad():
                delta = (delta - step * acc.sign()) * allow
                delta = delta.clamp(-epsilon, epsilon)
            if float_fooled(delta):
                break

        # ---- Stage A tail: extra steps for images any detector misses ----
        for _ in range(tail_steps):
            if float_fooled(delta):
                break
            delta = delta.detach().requires_grad_(True)
            g = torch.autograd.grad(loss_on(delta), delta)[0]
            g = g / (g.abs().mean() + 1e-12)
            acc = MU * acc + g
            with torch.no_grad():
                delta = (delta.detach() - step * acc.sign()) * allow
                delta = delta.clamp(-epsilon, epsilon)
        delta = delta.detach()

        # ---- Stage B: binary line search on perturbation scale ----
        # Invariant: scale hi passes the ground-truth gate, lo fails. Keep hi.
        if not _gt_bad(_uint8_hwc(x, delta), classifiers, names, device, target):
            remember(delta)
            lo, hi = 0.0, 1.0
            for _ in range(LINESEARCH_ITERS):
                mid = 0.5 * (lo + hi)
                if _gt_bad(_uint8_hwc(x, delta * mid),
                           classifiers, names, device, target):
                    lo = mid
                else:
                    hi = mid
            with torch.no_grad():
                delta = delta * hi
            remember(delta)

        # ---- Stage C: Adam quality recovery (evaluator-exact uint8 gate) --
        # Adaptive phases: fresh-momentum Adam + binary line search each phase.
        # kappa anneals 1.0 -> 0.1 so fooled margins hug the decision boundary
        # (the uint8 gate is the real safety net); Adam lr shrinks late for
        # fine polishing. Easy images stop early; hard ones spend up to ~2x.
        locked = [n for n in names
                  if n not in _gt_bad(_uint8_hwc(x, delta), classifiers, names,
                                      device, target)]
        if locked:
            b1, b2, eps_a = 0.9, 0.999, 1e-8
            psteps = max(10, rec_steps // 3)
            budget = rec_steps * 2
            done = 0
            stale = 0
            best_q = q_of(delta)

            def linescale(d):
                """Smallest perturbation scale passing the gt gate;
                returns (d*hi, hi). hi==1.0 means no slack."""
                if _gt_bad(_uint8_hwc(x, d), classifiers, names, device, target):
                    return d, 1.0
                lo, hi = 0.0, 1.0
                for _ in range(LINESEARCH_ITERS):
                    mid = 0.5 * (lo + hi)
                    if _gt_bad(_uint8_hwc(x, d * mid), classifiers, names,
                               device, target):
                        lo = mid
                    else:
                        hi = mid
                return (d * hi).detach(), hi

            while done < budget and stale < 2:
                lr_t = REC_LR * (0.6 if done >= rec_steps else 1.0)
                m_mom = torch.zeros_like(x)
                v_mom = torch.zeros_like(x)
                q_before = q_of(delta)
                for it in range(psteps):
                    kappa_t = KAPPA_REC + (0.1 - KAPPA_REC) * min(
                        1.0, (done + it) / max(1, budget - 1))
                    d_req = delta.detach().requires_grad_(True)
                    adv = adv_of(d_req)
                    q_s = ssim_fn(x, adv)
                    q_l = lpips_fn(base, adv * 2 - 1).mean()
                    q = 0.5 * q_s + 0.5 * (1.0 - q_l)
                    m_term = torch.zeros((), device=device)
                    for n, b in branches.items():
                        if n in locked:
                            z = b(adv)[0]
                            m_term = m_term + F.softplus(
                                z[1 - target] - z[target] + kappa_t)
                    loss = m_term + LAMBDA_REC * (1.0 - q)
                    g = torch.autograd.grad(loss, d_req)[0]
                    with torch.no_grad():
                        m_mom = b1 * m_mom + (1.0 - b1) * g
                        v_mom = b2 * v_mom + (1.0 - b2) * g * g
                        m_hat = m_mom / (1.0 - b1 ** (it + 1))
                        v_hat = v_mom / (1.0 - b2 ** (it + 1))
                        cand = (d_req - lr_t * m_hat / (v_hat.sqrt() + eps_a))
                        cand = (cand.clamp(-epsilon, epsilon) * allow).detach()
                        if not _gt_bad(_uint8_hwc(x, cand), classifiers, locked,
                                       device, target):
                            delta = cand
                            qv = float(q.item())
                            if qv > best_q:
                                best_q = qv
                                remember(cand)
                done += psteps
                delta, hi = linescale(delta)
                remember(delta)
                q_gain = q_of(delta) - q_before
                if hi > 0.999 and q_gain < 0.002:
                    stale += 1
                else:
                    stale = 0
            delta = delta.detach()

        # ---- Stage D: uint8 verification + re-attack on slippage ----
        for _ in range(VERIFY_ROUNDS):
            remember(delta)
            if not _gt_bad(_uint8_hwc(x, delta), classifiers, names,
                           device, target):
                break
            acc = torch.zeros_like(x)
            for _ in range(REATTACK_STEPS):
                d_req = delta.detach().requires_grad_(True)
                g = torch.autograd.grad(loss_on(d_req), d_req)[0]
                g = g / (g.abs().mean() + 1e-12)
                acc = MU * acc + g
                with torch.no_grad():
                    delta = ((d_req - step * acc.sign()) * allow
                             ).clamp(-epsilon, epsilon).detach()
                if not _gt_bad(_uint8_hwc(x, delta), classifiers, names,
                               device, target):
                    break
            remember(delta)

    # Main pass (Stages A->D), zero start.
    run_pass()
    # Stage E — hard-case rescue (multi-start / best-of-k): while the image
    # is still imperfect (any detector not fooled, or Q below RESCUE_Q),
    # re-run with different step sizes, random PGD starts and per-pass
    # Diverse-Inputs. Smaller L∞ steps trace a shorter path to the decision
    # boundary; random starts diversify the recovery trajectory (the
    # near-boundary uint8 gate is chaotic run-to-run). remember() keeps the
    # best image across all passes.
    for (r_step, r_steps, r_tail, r_rand, r_di) in RESCUE_CONFIGS:
        if best_key >= (len(names), RESCUE_Q):
            break
        run_pass(step=r_step, steps=r_steps, tail_steps=r_tail,
                 rand_start=r_rand, di=r_di)

    return best_u if best_u is not None else image
