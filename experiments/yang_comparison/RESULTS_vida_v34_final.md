# ViDA v3.4-final-no-stage0 — Final Results on the eval database

**Date:** 2026-09-03
**Attack:** ViDA — Visibility-guided Dual-domain Adaptive attack (`attacks/vida.py`)
**Version:** `v3.4-final-no-stage0` — every image goes through the full attack
pipeline. **No clean-skip, no clean-prediction shortcut.**
**Database:** `/home/aiattacks/dataset/celebA/TEST` (manifest:
`experiments/yang_comparison/manifest.json`) — 200 images: 100 fakes (PNG) +
100 reals (JPG), all 1024x1024, paired ids 001501-001600, **no duplicate files**.
**Evaluator:** unchanged official `evaluate.py`; detectors `vit_b_16` +
`densenet121_dct` (official checkpoints); target Real = class 0; direction
fake -> real; eps = 8/255 (L-inf); aggregate = `sum`; alpha = 0.5.
**Log:** `my_attack/results_vida_v34final_nostage0_dev200.log`
**JSON:** `experiments/yang_comparison/vida_v34_final_no_stage0.json`

> Historical development reference (kept, not overwritten): ViDA v3.3 *with*
> a clean-skip stage scored 397.20/400 on the same database. It is superseded
> by this compliant version; it is not used in any comparison.

## 1. Headline result

| Metric | Value |
|---|---|
| **Official score (sum)** | **397.97 / 400** (99.5% of maximum) |
| ViT ASR | **100%** (200/200) |
| DCT ASR | **100%** (200/200) |
| Mean SSIM | 0.9945 |
| Mean LPIPS (Alex) | 0.0046 |
| Wall-clock runtime | 1 h 30 min on one L40 (27.1 s/image avg) |

Per image-group breakdown (Q = 0.5*SSIM + 0.5*(1-LPIPS)):

| Group | n | Mean contrib | Median | Min | Mean SSIM | Mean LPIPS | Returned untouched |
|---|---|---|---|---|---|---|---|
| Fakes | 100 | 1.9798 | 1.9858 | 1.8955 | 0.9890 | 0.0092 | 0 |
| Reals | 100 | 1.9999 | 2.0000 | 1.9956 | 1.0000 | 0.0001 | **88/100** |

Without any explicit clean-skip, 88/100 real images are nevertheless returned
byte-identical (SSIM = 1.0000, LPIPS = 0.0000): MI-FGSM on already-Real inputs
has a near-zero targeted-margin gradient, Stage B's binary line search scales
the ~0 perturbation further down, and Stage C recovery drives it to zero. The
other 12 differ by sub-perceptual amounts (worst real contribution 1.9956).

Fake-score distribution: only 1 fake scores < 1.90; 6/100 score < 1.95;
10th percentile = 1.963, median = 1.986.

## 2. Pipeline (final version)

Differentiable replicas of both detector preprocessing paths (exact official
forward pass; capped-backward stable log for the DCT branch; differentiable
Lanczos). Every image passes through all stages:

- **Stage A - MI-FGSM acquisition.** Momentum (mu=1) sign attack on the sum of
  both detectors' targeted margin losses, border mask (outer ring unseen by
  both detectors never perturbed), early stop when both are fooled, tail
  steps for hard images.
- **Stage B - Binary line search.** Smallest perturbation scale t in [0,1]
  that still fools both detectors, gated by the official transforms run on
  the rounded uint8 image; removes sign-step overshoot.
- **Stage C - Adam quality recovery.** Maximises differentiable
  Q = 0.5*SSIM + 0.5*(1-LPIPS_Alex) with soft margin buffers annealed
  1.0 -> 0.1 (margins hug the decision boundary; the uint8 gate is the real
  safety net); adaptive phases with momentum reset and an inter-phase
  Stage B line search; each candidate accepted only if the official uint8
  check still fools every locked detector.
- **Stage D - Official uint8 verification.** The rounded image is re-run
  through the official transforms; any detector that slipped gets extra
  MI-FGSM steps. The best uint8 image by (detectors fooled, then Q) seen at
  any point is always retained.
- **Stage E - Hard-case rescue (multi-start / best-of-k).** Images ending
  Stages A-D below Q = 0.92 are re-attacked through independent passes with
  random PGD starts (radius up to 6/255), two acquisition step sizes and
  per-pass Diverse-Inputs; best across passes kept.

## 3. Hard-case diagnosis (001539, 001556)

`my_attack/diag_hard.py`; controls: 001553 (rescue-effective), 001549 (easy).
Logs: `my_attack/diag_hard.log`, `my_attack/diag_repeat*.log`.

| Measurement | 001539 | 001556 | 001553 | 001549 |
|---|---|---|---|---|
| Clean margin ViT / DCT (logit gap) | +2.24 / +2.17 | +2.19 / +2.15 | +2.19 / +2.17 | +2.21 / +2.18 |
| Min. flip scale after joint attack - ViT / DCT | **0.94 / 0.19** | **0.94 / 0.63** | 0.19 / 0.69 | 0.31 / 0.75 |
| ViT-only attack minimal Q | 0.767 | 0.799 | 0.978 | 0.953 |
| DCT-only attack minimal Q | **0.991** | **0.997** | 0.953 | 0.943 |
| Gradient sign agreement in shared centre region | 0.50 | 0.50 | 0.50 | 0.50 |
| ViT attack-gradient energy: luma / chroma | 0.03 / 0.89 | 0.05 / 0.81 | 0.06 / 0.80 | 0.05 / 0.83 |

- **ViT is the binding detector on hard images.** All images start with near-
  identical confidence (~90% Fake, margin ~ +2.2), but flipping ViT requires a
  spatially dense perturbation (56-73% of visible pixels at the eps cap in a
  ViT-only attack) whereas DCT flips with tiny centre-region perturbations
  (DCT-only Q ~ 0.99). This explains why naive joint optimisation wastes
  budget on DCT that ViT then cannot use.
- The two detectors' acquisition gradients are **near-orthogonal** in the
  shared region (sign agreement ~ 0.50, cosine ~ 0): not antagonistic, but the
  joint sum dilutes each direction. ViT's attack gradient is almost entirely
  chromatic (81-89%), consistent with its RGB view.

## 4. Hard-case optimization instability (presentation finding)

> **Same image + same method -> different optimization trajectories -> large
> Q variance -> high-quality valid solutions exist -> multi-start stabilises
> recovery.**

Independent single-trajectory runs on the same image (development builds
without multi-start rescue) gave, for 001539, Q in {0.819, 0.967, 0.977}
(LPIPS 0.019-0.184); for 001556, Q in {0.767, 0.767, 0.860}. The recovery
stage exhibits substantial run-to-run variability near the discrete decision
boundary: cheap valid solutions exist (LPIPS 0.01-0.05, perturbation 6-7/255,
logit margin hugging the boundary), but the discrete uint8 acceptance gate
combined with optimization/numerical variability can send Adam into different
recovery trajectories. (We describe this as boundary-gating + optimization
variability; we do not claim numeric noise is the proven sole cause.)

### 4.1 Best-of-k experiment (`my_attack/best_of_k.py`)

k independent full-pipeline runs (independent seeds -> independent acquisition
and Stage E trajectories) on the 4 representative images; selection rule
argmax(detectors fooled, Q). All k runs fooled **both detectors in every
case** (2/2, 32/32 runs across k=3 and k=5).

| Image | Case | k | best Q | mean Q | std Q | worst Q | best-of-k - mean |
|---|---|---|---|---|---|---|---|
| 001539 | hard | 3 | 0.988 | 0.973 | 0.012 | 0.959 | +0.015 |
| 001539 | hard | 5 | 0.971 | 0.953 | 0.016 | 0.925 | +0.018 |
| 001556 | hard | 3 | 0.991 | 0.960 | 0.022 | 0.943 | +0.031 |
| 001556 | hard | 5 | 0.988 | 0.962 | 0.019 | 0.941 | +0.026 |
| 001553 | rescue-effective | 3 | 0.998 | 0.996 | 0.003 | 0.992 | +0.002 |
| 001553 | rescue-effective | 5 | 0.998 | 0.997 | 0.001 | 0.996 | +0.001 |
| 001549 | easy | 3 | 0.998 | 0.998 | 0.001 | 0.997 | +0.001 |
| 001549 | easy | 5 | 0.998 | 0.998 | 0.0002 | 0.998 | +0.0004 |

Per-run data: `my_attack/best_of_k_results_k3.json`, `best_of_k_results_k5.json`.
Runtime: ~60-180 s per hard run, ~30-70 s per easy run on one L40.

**Reading of the result:**

- Variability is confined to hard cases (std ~ 0.015-0.022); easy/medium
  images are stable (std <= 0.003) and best-of-k gives them nothing.
- The internal Stage E multi-start rescue already raises the single-call floor
  from the historical 0.70-0.86 to >= 0.925 on the hardest images.
- Additional independent best-of-k restarts add roughly **+0.02-0.03 Q on the
  ~2 hardest images**, ~ +0.1-0.2 points of the 400-point total, at k x
  runtime per affected image.

## 5. Recommendation on best-of-k

1. **Keep Stage E (internal multi-start rescue) in the final ViDA** - it is
   the mechanism that converts the occasional cheap solution into the normal
   outcome: 397.97/400 with every fake >= 1.896 and no ASR loss.
2. **Do not make k independent full-pipeline restarts the default.** The gain
   over Stage E alone is marginal (~+0.02 Q on the hardest images), it costs
   k x runtime, and it applies to images already scoring above 0.95.
   Best-of-k across independent restarts is recommended only as an optional
   offline post-pass for any individual image still below Q = 0.92-0.95 after
   Stage E, if the submission runtime budget allows.
3. **Open direction (not implemented, no new modules added):** since the
   bottleneck on hard images is ViT (dense chromatic perturbation, orthogonal
   gradients), marginal future gains worth exploring are ViT-focused.

## 6. Artifacts

| Deliverable | Path |
|---|---|
| Database manifest | `experiments/yang_comparison/manifest.json` |
| Final config | `configs/v34_final_no_stage0.yaml` |
| Final results JSON | `experiments/yang_comparison/vida_v34_final_no_stage0.json` |
| Final eval log | `my_attack/results_vida_v34final_nostage0_dev200.log` |
| Best-of-k results | `my_attack/best_of_k_results_k3.json`, `best_of_k_results_k5.json` |
| Hard-case diagnosis | `my_attack/diag_hard.py`, `diag_hard.log` |
| Historical v3.3 log (reference) | `my_attack/results_vida_v3.3_dev200.log` |
