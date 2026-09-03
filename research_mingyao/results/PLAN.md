# Research Direction & Experiment Plan — AADD 2026

## Visibility-guided Dual-domain Decoupled Attack (ViDA v2)

**Author:** Mingyao Duan
**Branch:** `mingyao-dev`
**Date:** 2026-09-02

**Scope:** White-box adversarial attacks against the two AADD-2026 detectors (`vit_b_16`, `densenet121_dct`) in both directions:

* `fake → real` (official scoring direction)
* `real → fake` (research analysis)

---

# 1. Core Idea

The AADD-2026 score is:

$$
Score =
\sum_{images}
\sum_{detectors}
[
(0.5\cdot SSIM+
0.5\cdot(1-LPIPS))
\cdot
1(f\_fooled)
]
$$

Existing iterative attacks such as I-FGSM/PGD mainly treat perceptual quality as a hard constraint inside an ε-ball. They do not explicitly consider that different detectors observe different projections of the same image.

Our key observation is:

**The two AADD detectors have asymmetric visibility due to their preprocessing pipelines.**

The ViT detector observes:

* RGB information
* central 224×224 crop

The DCT detector observes:

* grayscale information only
* central 128×128 crop
* frequency-domain representation after DCT and logarithmic scaling

Therefore, the perturbation space can be decomposed into:

1. **ViT-exclusive directions**

   * visible to ViT
   * invisible or strongly attenuated by DCT

2. **DCT-oriented directions**

   * efficient for moving DCT-domain representations

3. **Shared battleground**

   * visible to both detectors
   * requires coordinated optimization

We propose:

**ViDA — Visibility-guided Dual-domain Decoupled Attack**

A visibility-aware attack framework that:

1. derives detector-specific perturbation subspaces from preprocessing geometry,
2. allocates perturbation budget according to detector visibility,
3. optimizes directly for the official score objective,
4. adaptively stops attack growth once fooling is achieved and reallocates remaining budget toward perceptual quality.

---

# 2. Visibility Model

## 2.1 Detector Views

At canonical 256×256 resolution:

| Region                               | ViT | DCT | Interpretation                  |
| ------------------------------------ | --- | --- | ------------------------------- |
| Outer border outside 224 crop        | No  | No  | Pure quality waste              |
| Annulus: 224 square minus 128 square | Yes | No  | ViT-exclusive spatial region    |
| Central 128×128 square               | Yes | Yes | Shared optimization region      |
| RGB chromatic subspace               | Yes | No  | Potential ViT-exclusive channel |
| DCT high-frequency bands             | Yes | Yes | Frequency-sensitive region      |

---

## 2.2 Visibility Assumptions

### H1 — Detector visibility asymmetry exists

Different preprocessing pipelines create different perturbation sensitivities.

### H2 — Detector-exclusive directions exist

There are perturbation directions satisfying:

$$
\nabla f_{ViT}\neq0
$$

while:

$$
\nabla f_{DCT}\approx0
$$

and vice versa.

### H3 — Adaptive perturbation allocation improves score efficiency

A fixed ε budget is not necessarily optimal because easy images do not require the same perturbation magnitude as difficult images.

---

# 3. ViDA Method

## 3.1 Differentiable Evaluator Replication

Optimization happens in canonical 256×256 RGB space.

Both detector pipelines are reproduced:

### ViT branch

```
RGB
→ crop 224×224
→ ImageNet normalization
→ vit_b_16
```

### DCT branch

```
RGB
→ grayscale
→ crop 128×128
→ separable orthonormal DCT-II
→ log(|x|+ε)
→ densenet121_dct
```

Gradient flows through the complete pipeline.

Engineering safeguards:

* use stabilized logarithm:

$$
log(|x|+\epsilon)
$$

with non-extreme ε.

* normalize or clip DCT branch gradients to prevent near-zero coefficient explosion.

---

# 3.2 Visibility-aware Perturbation Decomposition

The total perturbation:

$$
\delta=
\delta_{vit}
+
\delta_{dct}
+
\delta_{shared}
$$

is implemented using explicit components.

---

## Component A — Border Mask

Outer regions unseen by both detectors:

$$
\delta=0
$$

Reason:

They cannot improve fooling but always reduce SSIM/LPIPS.

---

## Component B — ViT-exclusive Branch

### δ_annulus

Luma perturbation in:

$$
224^2-128^2
$$

region.

Purpose:

* influence ViT
* avoid DCT crop

---

### δ_chroma

Chromatic perturbation satisfying:

$$
0.299\Delta R+
0.587\Delta G+
0.114\Delta B=0
$$

Purpose:

Potential DCT-blind auxiliary direction.

Important:

Chroma is NOT assumed free.

Although DCT removes chroma information, SSIM and LPIPS operate on RGB images. Therefore:

δ_chroma is accepted only if it improves score efficiency.

If it provides limited gain, it is retained only as a regularizer.

---

## Component C — DCT-oriented Branch

### δ_freq

Central 128×128 perturbation weighted toward DCT-sensitive frequency bands.

Purpose:

Move DCT detector efficiently with minimal perceptual cost.

The frequency weighting is validated experimentally rather than assumed.

---

## Component D — Shared Branch

### δ_shared

Central RGB/luma perturbation optimized jointly against both detectors.

Purpose:

Handle directions where both detectors respond.

---

# 3.3 Conditional Two-stage Optimization

Instead of one fixed weighted loss, ViDA uses conditional optimization.

---

## Stage A — Attack Acquisition

Goal:

maximize detector fooling.

Loss:

$$
L_{attack}
=
\sum_m
softplus(z_{wrong}-z_{target}+\kappa)
+
\alpha R(\delta)
$$

where:

* first term maximizes fooling margin
* second term controls perturbation structure

During this stage:

* attack effectiveness has priority
* quality term has low weight

---

## Stage B — Quality Recovery

When:

$$
f_{ViT}=1
$$

and:

$$
f_{DCT}=1
$$

the attack components are frozen.

Remaining iterations optimize:

$$
L_{repair}=-Q(x+\delta)
$$

where:

$$
Q=
0.5SSIM+
0.5(1-LPIPS)
$$

Goal:

recover perceptual quality while preserving fooling margins.

---

# 3.4 Block Coordinate Optimization

Default optimization:

1. update `{δ_chroma, δ_annulus}`
2. update `{δ_freq}`
3. update `{δ_shared}`

After every block:

* evaluate both detectors
* check fooling margin
* check official score proxy

If a block update destroys previously achieved fooling:

1. rollback update
2. reduce step size
3. retry or skip the block

This prevents oscillation between detector-specific objectives.

---

# 3.5 Adaptive Budget Allocation

Traditional attacks:

* fixed iterations
* fixed perturbation growth

ViDA:

* easy samples stop early
* difficult samples receive larger budget

Once both detectors are fooled:

remaining computation is used for quality recovery.

The goal is not maximum ASR alone.

The goal is:

$$
maximize\ Score
$$

---

# 4. Verification Gates

All gates are evaluated on:

* development set
* ε = 8/255
* paired comparisons
* bootstrap 95% confidence intervals

---

## G1 — Border Mask

Compare:

* masked attack
* unmasked attack

Expected:

* similar ASR
* better SSIM/LPIPS

Purpose:

Verify that invisible regions should never receive perturbation.

---

## G2 — ViT-exclusive Directions

Compare:

* chroma + annulus
* full RGB attack

Metric:

not only ASR.

Primary metric:

$$
MarginGain / QualityCost
$$

If chroma does not improve efficiency:

* remove it from main branch
* retain only as regularization

---

## G3 — DCT Frequency Shaping

Compare:

* frequency-weighted perturbation
* unrestricted spectrum perturbation

Measure:

$$
DCT\ MarginGain / PerceptualCost
$$

---

## G4 — Quality-aware Optimization

Compare:

* attack-only optimization
* attack + quality recovery

At identical fooling rate:

measure perceptual improvement.

---

## G5 — Adaptive Budget

Compare:

* fixed-step attack
* minimal-sufficient attack

Expected:

same fooling performance with improved average quality.

---

# 5. Main Experiments

## Baselines

1. Original I-FGSM
2. Fused ensemble I-FGSM
3. Border-masked fused I-FGSM
4. ViDA-full

---

## Visibility Ablations

Additional required experiments:

| Method               | Visibility model              |
| -------------------- | ----------------------------- |
| ViDA-full            | Full visibility decomposition |
| ViDA-no-mask         | Remove spatial visibility     |
| ViDA-no-chroma       | Remove chromatic branch       |
| ViDA-no-frequency    | Remove DCT shaping            |
| Random decomposition | Random allocation baseline    |

Purpose:

prove that gains come from visibility modeling, not arbitrary decomposition.

---

## Evaluation Metrics

Report:

* clean-correct conditional targeted ASR
* SSIM
* LPIPS-Alex
* official score
* per-image score distribution
* post-save L∞ verification
* runtime
* memory usage

---

## Experiment Protocol

Pipeline:

```
smoke test
    ↓
development validation
    ↓
gate evaluation
    ↓
full ViDA experiment
    ↓
multi-seed replication
```

Official evaluator:

* unchanged
* no post-evaluation tuning

---

# 6. Positioning

Existing attack families:

## Iterative attacks

Strength:

* reliable white-box optimization

Limitation:

* no detector visibility modeling
* fixed perturbation spending

---

## Transfer attacks

Strength:

* strong black-box transfer

Limitation:

* optimized for unknown targets
* does not exploit known preprocessing geometry

---

## Latent attacks

Strength:

* strong perceptual quality

Limitation:

* require generator assumptions

---

## Frequency attacks

Strength:

* exploit spectral weakness

Limitation:

* usually treat frequency as another optimization space rather than detector visibility allocation

---

## ViDA Contribution

The main contribution:

> An explicit detector visibility geometry model derived from preprocessing pipelines, used to allocate adversarial perturbation budget among detector-exclusive and shared subspaces.

ViDA is complementary to:

* MI-FGSM
* diversity methods
* integrated-gradient methods
* perceptual objectives

---

# 7. Dependencies and Risks

## Dependency

Need:

`vit_b_16.pth`

for 2026 detector validation.

Before availability:

can complete:

* DCT pipeline
* differentiable evaluator
* G1/G3/G4/G5 infrastructure

Only:

* G2
* joint experiments

depend on ViT weights.

---

## Engineering Risks

### DCT log instability

Mitigation:

* stabilized log
* gradient clipping

### Block optimization oscillation

Mitigation:

* rollback mechanism

### Chroma underperformance

Mitigation:

* automatic downgrade to regularizer

---

# 8. Implementation Plan

1. Create:

```
my_attack/
```

with:

* differentiable DCT-II
* detector wrappers
* metrics
* masks
* optimization framework

2. Validate:

* identity pipeline
* I-FGSM reproduction
* save/reload ε constraint

3. Run:

G1 → G5

4. Select validated components.

5. Run full ViDA benchmark.

---

# Final Research Claim

ViDA does not attempt to create another stronger PGD variant.

Its central idea is:

$$
Preprocessing
\rightarrow
Visibility Geometry
\rightarrow
Perturbation Allocation
\rightarrow
Score Optimization
$$

By explicitly modeling what each detector can and cannot see, ViDA aims to achieve higher AADD score efficiency under strict perceptual constraints.
