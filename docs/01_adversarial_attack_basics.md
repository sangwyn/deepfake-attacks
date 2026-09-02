# Adversarial Attack Basics — Study Notes

> Purpose: build a shared conceptual foundation so the team can write the agent prompt together.
> Target task: AADD 2026 "Attacking AI-Generated-Image Detectors".
> Date: 2026-09-01. Based on the papers we read (MR-CAS / MIG-COW / RoMa / challenge overview) and the real constraints in team_repo.

---

## 1. What we are doing (one sentence)

Given a **fake (AI-generated) image**, add an **almost invisible perturbation** so that a deepfake detector classifies it as **real**. The constraint is "keep the change small"; the goal is "fool the detector".

- Detector output: `real = 0`, `fake = 1`
- Attack success = the detector outputs `real (0)` on our adversarial image
- This is a **targeted attack**: the target class is explicitly `real`.

---

## 2. Core concepts (vocabulary needed when writing the prompt)

| Concept | Meaning | Value in this task |
|---|---|---|
| **Perturbation delta** | Noise added to the image, `x_adv = x + delta` | Must be invisibly small |
| **epsilon** | Upper bound of perturbation (L-inf norm), max change per pixel | **<= 8/255** |
| **alpha (step size)** | How far each iteration moves | Commonly 2/255 |
| **iterations** | Number of steps | ifgsm baseline uses 10 |
| **white-box** | We can see model architecture + weights + gradients | ResNet50/DenseNet121 (2025); 2026 TBD |
| **black-box** | Only input/output visible, no gradients | Held-out evaluation models |
| **transferability** | Whether an attack made on model A also fools an unseen model B | **The biggest difficulty** |
| **ensemble attack** | Attacking several models at once to improve transferability | Common technique |

---

## 3. The three most basic attack algorithms (must understand)

### FGSM (single step)
Take one large step along the sign of the loss gradient:
`x_adv = x + epsilon * sign(grad_x Loss)`
- Simplest, one-shot, moderate effectiveness.

### I-FGSM / BIM (multi-step iterative)
Break FGSM into many small steps, clamp back into the epsilon-ball each step:
`x_adv^(t+1) = clip( x_adv^(t) + alpha * sign(grad Loss), epsilon-ball )`
- Stronger than FGSM; this is our **baseline** (team_repo `attacks/ifgsm.py`).
- **Targeted version** (what we want): the target is real, so we **decrease** the loss for the real class:
  `x_adv = x_adv - alpha * sign(grad_x Loss(model(x_adv), real))`

### PGD (I-FGSM with random start)
I-FGSM + random initial perturbation; considered the strongest first-order white-box baseline.

### Common add-ons for transferability (key to black-box)
- **MI-FGSM**: add momentum, more stable gradients, higher transferability
- **DI-FGSM**: random resize/pad the input each step (input diversity), higher transferability
- **TI-FGSM**: translation-smoothing on the gradient, higher transferability
- These can be stacked.

---

## 4. Key difficulties of this task (the agent must be told about these)

1. **Black-box transfer is the crux**: in the papers, the top-3 teams all reach near-100% white-box ASR, but black-box is generally very low (around 7%). Whoever improves transferability wins.
2. **Constraints must not be violated**: epsilon <= 8/255, and the similarity score must stay high (see below). Attack too hard -> the image changes -> similarity drops -> the attack is wasted.
3. **Different detectors use different preprocessing**:
   - Spatial-domain model (ViT): resize -> crop -> normalize
   - **DCT model**: grayscale -> resize -> crop -> **2D-DCT -> log**. The attack must reach the frequency domain, which requires a **differentiable DCT**, otherwise gradients do not flow.

---

## 5. Important changes in AADD 2026 (different from 2025, must remember)

The papers we read are **AADD 2025**, but team_repo is **2026**, and the rules changed:

| | AADD 2025 | **AADD 2026 (now)** |
|---|---|---|
| Number of evaluation detectors | 4 | **2: vit_b_16 + densenet121_dct** |
| Similarity score | pure SSIM | **0.5*SSIM + 0.5*(1 - LPIPS)** |
| Detector weights | 2025 version | **retrained; 2025 weights cannot be used** (empirically confirmed) |

**Implications**:
- Perceptual similarity **LPIPS now counts for half the score** — we cannot only care about SSIM; the perturbation must also be perceptually close (lower LPIPS is better).
- Only 2 target models, one of which (DCT) works in the frequency domain — a **differentiable DCT is essentially required**.

---

## 6. Scoring formula (AADD 2026)

```
per image, per detector contribution = (0.5*SSIM + 0.5*(1 - LPIPS)) * [detector fooled to real]
total score = sum over all image x detector contributions (aggregate: sum)
```

- Higher SSIM is better (structural similarity); lower LPIPS is better (perceptual similarity)
- If the attack fails, that term = 0 (high similarity is useless without success)
- So it is a **dual objective: "must fool AND must look alike"**.

---

## 7. Four technical routes (see Direction-Comparison doc; one-liner version here)

- **A Pixel-level I-FGSM/PGD**: simple, strong white-box, weak black-box. Baseline.
- **B Latent-space (MR-CAS, 2025 rank 1)**: perturb in the diffusion latent space, stealthy, good transfer, but heavy engineering and compute.
- **C Integrated-gradient + consensus (MIG-COW, rank 2)**: extremely strong white-box, weak black-box, medium engineering.
- **D Transfer/surrogate (RoMa, rank 3)**: train surrogate models to guess the black-box; transfer still the weak point.

---

## 8. One-sentence summary (team consensus)

> We will write an `attack(image, classifiers, device)` function that, within epsilon <= 8/255, uses a (targeted, differentiable, possibly ensemble + differentiable-DCT + momentum/input-transform) iterative attack to turn a fake image into an adversarial image that fools vit_b_16 and densenet121_dct, while keeping SSIM high and LPIPS low.
