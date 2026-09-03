# ViDA — End-of-Day Summary (2026-09-02/03)

**Author:** Mingyao Duan (`mingyao-dev`)
**Attack:** ViDA — Visibility-guided Dual-domain Adaptive attack.
**Target challenge:** AADD-2026, white-box adversarial attacks against AI-generated-image
detectors. Primary code delivered as `attacks/vida.py` in the team repo.

## 1. Problem and scoring
- Two graded detectors: `vit_b_16` (spatial, 224×224 crop, ImageNet norm) and
  `densenet121_dct` (grayscale → 256 → centre-128 → orthonormal DCT-II → `log|·|`).
- Label semantics: Real=0, Fake=1; the official direction is **fake → real**.
- Official score (aggregate `sum`):
  `Σ_images Σ_detectors (0.5·SSIM + 0.5·(1 − LPIPS_Alex)) · 1(pred = Real)`,
  perturbation budget `ε = 8/255` (L∞, RGB).
- LPIPS counts for half of the quality weight, so a good attack must be both
  successful **and** perceptually close.

## 2. ViDA in one paragraph
ViDA derives each detector's **visibility geometry** from its preprocessing
(border vs. annulus vs. centre; grayscale/DCT vs. RGB) and then runs
(1) Stage A — an MI-FGSM acquisition over the sum of both detectors' targeted
margin losses, with a border mask (never perturb pixels neither detector sees),
early stop as soon as every detector is fooled, and a short extra-iteration
"tail" for hard images; and (2) Stage B — a **quality-recovery** phase that
maximises the differentiable score proxy `0.5·SSIM + 0.5·(1 − LPIPS_Alex)` while
a hard per-step rollback guarantees we never lose an already-fooled detector.
The two detectors' differentiable preprocessing paths are replicated exactly
(forward pass matches `evaluate.py`).

## 3. Key engineering findings (all verified experimentally)
1. **Differentiable DCT must match the evaluator exactly.** Two mismatches made
   in-branch "success" disagree with the official scipy/PIL path on adversarial
   images: (a) grayscale value scale — the evaluator DCT/log operate on **0–255**
   (a 0–1 branch shifts `log` by a constant and flips boundary predictions);
   (b) resize kernel — the evaluator uses **PIL Lanczos**; torch tensors have no
   Lanczos, so we implemented a differentiable Lanczos via fixed separable
   resampling matrices (matches PIL to <1/255; bicubic differed by up to 10).
2. **`log(|coef|+1e-6)` gradients explode (~1e6)** on near-zero high-frequency
   DCT coefficients and dominate the sign gradient. Fix: a custom autograd
   function with the exact official forward and a magnitude-capped backward
   (`eps_grad ≈ 0.01`).
3. **Momentum is essential for the frequency model**: plain sign I-FGSM
   saturates at ~1–2/8 even at 100 steps; MI-FGSM (µ=1, mean-abs normalised)
   reaches high ASR.
4. **Gates (development set, DCT side)**: border mask G1 = no harm; frequency
   shaping G3 = smooth/low-freq perturbation cannot fool DCT (high-freq moves
   it, LPIPS-cheap); adaptive budget G5 and quality recovery G4 = same ASR with
   markedly better quality.
5. **Chroma is DCT-blind but not perceptually cheap.** A chromatic perturbation
   (luma-zero) fully fools ViT while DCT never sees it — confirming the
   visibility-decoupling idea — but its LPIPS cost is similar to full RGB.
   Smooth global colour shifts are invisible to LPIPS yet cannot fool ViT: both
   detectors need **high-frequency** adversarial energy, which is intrinsically
   visible. Consequently the quality win comes from **minimal-sufficient budget
   + Stage-B recovery**, not from chroma/smoothing.
6. **Strict DCT-blind gradient routing for the joint attack underperforms**:
   the ViT chroma field saturates the shared RGB L∞ headroom the DCT grayscale
   perturbation needs. The effective joint recipe is the momentum-sum
   acquisition + border mask + early stop + tail + Stage-B recovery.

## 4. Results (celebA dev set, 100 fakes, `ε = 8/255`)
Score below is `mean_image [ Q · (I_ViT + I_DCT) ]`, `Q = 0.5·SSIM + 0.5·(1−LPIPS)` (max 2.0).

| Attack | ViT fooled | DCT fooled | SSIM | LPIPS | Q | Score |
|---|---|---|---|---|---|---|
| naive joint (fixed 80 it.) | 100/100 | 99/100 | 0.833 | 0.319 | 0.757 | 1.508 |
| ViDA (mask + early-stop + tail) | 100/100 | **100/100** | 0.866 | 0.274 | 0.796 | 1.592 |
| **ViDA + quality recovery** | **100/100** | **100/100** | **0.890** | **0.143** | **0.874** | **1.747** |

- **+15.9% score** over the naive joint attack; LPIPS cut by more than half;
  zero loss in fooling rate; mean acquisition stops at ~14 of 80 iterations.
- Verified end-to-end with the team **`evaluate.py`** (not just our harness):
  both detectors report `attack_success = 1.000`, mean SSIM ≈ 0.92, LPIPS ≈ 0.10
  on an 8-image smoke set.

## 5. Repository / deliverables
- `attacks/vida.py` — self-contained attack conforming to the team template
  `attack(image, classifiers, device)`; committed and pushed to `mingyao-dev`
  (commit `f257629`). Includes differentiable Lanczos, orthonormal DCT-II,
  stabilised log, MI acquisition, border mask, early stop, tail, quality
  recovery (differentiable SSIM + LPIPS-Alex, rollback).
- Research harness in `Mingyao-Duan/my_attack/`: gates G1–G5, the DCT/ViT
  branches, metrics, and the 4-detector transfer harness.

## 6. Black-box transfer (extra detectors)
Oleg added two **non-graded** detectors purely as black-box transfer targets —
`npr` (ResNet-50) and `aide` (ConvNeXt-XXL). ViDA gradients use only the two
graded detectors; these two are scored on the resulting adversarial images with
no gradient access. Fool→Real rate on the celebA dev fakes (`blackbox_transfer.py`,
clean-label orientation auto-calibrated):

| Detector | Role | clean accuracy | fooled (ViDA adv.) |
|---|---|---|---|
| vit_b_16 | white-box | 40/40 | **40/40 (100%)** |
| densenet121_dct | white-box | 40/40 | **40/40 (100%)** |
| npr (ResNet-50) | black-box | 37/40 | 13/40 (**32.5%**) |
| aide (ConvNeXt-XXL) | black-box | 35/40 | 0/40 (**0%**) |

### 6.1 Adding Diverse-Inputs (DI) raises transfer
A module-level `DI_PROB` switch in `attacks/vida.py` (default `0.0`) turns on
Diverse-Inputs during acquisition (random down-scale + zero-pad per step).
Fool→Real rate with MI vs DI (40 fakes; graded attack still white-box vs
vit+dct):

| Detector | Role | MI (DI off) | DI (DI_PROB=0.5) |
|---|---|---|---|
| vit_b_16 | white-box | 100% | 90% |
| densenet121_dct | white-box | 100% | 100% |
| **npr (ResNet-50)** | black-box | 57.5% | **82.5%** |
| aide (ConvNeXt-XXL) | black-box | 0% | 0% |

- DI lifts black-box transfer to NPR by **+25 points** (57.5% → 82.5%) at a
  small white-box ViT cost (100% → 90%). AIDE stays robust (it uses SRM/DCT
  preprocessing and a very large backbone).
- The 57.5% MI number (without quality recovery) vs the 32.5% in the table
  above (with recovery) shows quality recovery trades some transfer for
  perceptual quality — so there are two operating points:
  - **Official score** → `DI_PROB=0`, full quality recovery (best Q×success).
  - **Transfer showcase** → `DI_PROB=0.5`, no/light recovery (best black-box).
- Cross-architecture transfer is intrinsically limited (ViT↔DCT also transfer
  ~0); DI/momentum-diversity/surrogate ensembling are the known levers.

- Cross-architecture black-box transfer is low for the default config
  (NPR ~1/3, AIDE ~0), consistent with ViT↔DCT also transferring ~0 to each
  other: the attack is deliberately tuned white-box.
- **Integration note for the team:** the shared `DetectorAdapter` column swap
  (`raw_logits[:, [1, 0]]`) makes the AIDE checkpoint report an **inverted**
  label (clean fakes → "Real", clean reals → "Fake"). Raw AIDE logits are
  already `[Real, Fake]`; the swap double-flips it. NPR is oriented correctly.
  We calibrate orientation from clean labels; the adapter should be checked.

## 7. Dataset used
- All reported numbers use the designated development/test split
  `~/dataset/celebA/TEST` (**100 fakes + 100 reals**, 1024×1024), as indicated
  by Oleg. `~/dataset/celebA/TRAIN` (1500+1500) is for detector training and is
  not used for attack evaluation; the official score uses the organisers'
  separate `AADD_2026_Test` set (same `attacks/vida.py`, unchanged evaluator).
  Reports elsewhere with 1000+ images use the TRAIN split.

## 8. Open items / next steps
- Raise black-box transfer via DI / momentum-diversity / surrogate ensembling
  if transfer matters for the final comparison (not part of the official score).
- Official AADD-2026 test set (`AADD_2026_Test`, via the organisers/Dan).
- `real → fake` research direction: on the dev set the DCT model labels real
  photos as fake, so a clean-correct conditional set is not yet available; the
  official score only depends on fake → real.
- Multi-seed / random-start for robustness (MI-FGSM is deterministic).

## 9. Reproducibility notes
- Weights: official ViT/DCT via the team; `npr.pth`, `aide.pth` from
  `~/detector_weights`.
- GPU: shared L40s; pick the freest card (`nvidia-smi`) and use
  `CUDA_VISIBLE_DEVICES`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  `python -u` for unbuffered logs. AIDE needs `kornia`, `clip`, `open_clip`,
  `timm` (installed `--no-deps` where they conflict with the cu124 torch wheel).
