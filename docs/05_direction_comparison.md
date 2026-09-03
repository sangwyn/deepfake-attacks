# AADD / Adversarial-Attack Direction Comparison

> Use: help the team decide the technical direction for "attacking AI-generated-image detectors".
> This doc is **comparison only; it does not choose the direction for the team**. All numbers come from
> the local papers/code we measured; anything not found is marked "not specified".
> Date: 2026-08-31.

---

## 0. Decision background (one sentence)

Task: add an invisible perturbation to a fake image so a detector classifies it as **real**, constraints **SSIM >= 0.74, epsilon <= 8/255**, score **FS = sum(SSIM * attack-success indicator)** (real=0, fake=1).

**Current state**: data (1403 fake images), 4 detector weights, and the I-FGSM baseline code are all in place. The direction is not decided, so this doc is for selection, not for running attacks yet.

> NOTE (2026 update): team_repo is AADD 2026, which now evaluates only **2 detectors (vit_b_16 + densenet121_dct)** and scores with **0.5*SSIM + 0.5*(1 - LPIPS)**. The 2025 numbers below still inform the trade-offs, but the final rules follow the 2026 official spec.

---

## 1. Basic facts (from the AADD-2025 challenge overview and the solution papers)

- **Data**: from WILD, 1403 fake = 693 HQ (512^2 to 1024^2) + 710 LQ (256^2), GAN and Diffusion mixed, 16 subsets.
- **4 detectors (2025)**:
  - **White-box (public during dev)**: ResNet50, DenseNet121
  - **Black-box / transfer targets (held out for evaluation)**: ViT-B-16, DenseNet121-DCT
- **Official final top-3** (SSIMS = mean SSIM, ASS = mean attack success rate):

  | Team | Route | SSIMS | ASS | FS |
  |---|---|---|---|---|
  | MR-CAS | latent space | 0.742 | 0.672 | **2740 (1st)** |
  | Safe AI (MIG-COW) | integrated gradient / consensus | 0.915 | 0.528 | 2709 (2nd) |
  | RoMa | transfer / surrogate | 0.934 | 0.509 | 2679 (3rd) |

- **Key qualitative conclusions**: (1) latent-space attacks > pixel-level; (2) ensembling improves cross-model robustness; (3) **black-box transferability is a shared major bottleneck** for the top-3 (their FS gap is < 70, white-box ASR is near perfect for all, the difference is mainly in black-box).

---

## 2. Four technical routes

### A. Pixel-level gradient attack (I-FGSM / PGD / MI-FGSM) — the existing baseline route
- **Principle**: iterate small signed-gradient steps toward the target class (real) in pixel space, project back into the epsilon-ball and clamp each step; can add momentum (MI) / input diversity (DI) to improve transfer.
- **Representative**: the existing `aadd_attack.py`; mid-tier teams like GRADIANT, MICV, DeFakePol.
- **White/black-box**: strong white-box (MI-FGSM white-box ASR 91%), **weak black-box (4-6%)**.
- **Result**: per the papers, PGD FS=1655, MI-FGSM FS=2485.
- **Difficulty**: low. **Compute**: low (full set on a single GPU). **Reuse of existing code**: ~100%.

### B. Latent-space attack (MR-CAS, DDIM inversion + latent optimization) — rank 1
- **Principle**: VAE encode + DDIM inversion maps the fake image back into the SD latent space, do momentum gradient optimization on the latent at intermediate timesteps, then deterministically denoise + decode; the perturbation lands on the generative manifold, high stealth.
- **Representative**: MR-CAS (UCAS, arXiv 2506.23676).
- **White/black-box**: best transferability (the paper's main selling point, ASS=0.672, highest of top-3).
- **Result**: **FS=2740 (highest)**, but SSIMS=0.742 (lowest, near the constraint line).
- **Difficulty**: high (needs SD U-Net + VAE + VLM + DDIM inversion/denoise). **Compute**: high. **Reuse**: ~20-30%.

### C. Integrated gradient / consensus (MIG-COW, integrated gradient + consensus orthogonal decomposition) — rank 2
- **Principle**: compute momentum integrated gradients (IG) toward the target class per model, decompose the N-model gradients into a "consensus component" + "orthogonal component", weight-combine, then signed update.
- **Representative**: MIG-COW / Safe AI (UNIST, DOI 10.1145/3746027.3761986).
- **White/black-box**: extremely strong white-box (99.96%), **extremely weak black-box (7.16%)**. Counter-intuitive: blindly adding diverse but low-performance surrogates lowers both white and black-box ASR.
- **Result**: FS=2709, SSIMS=0.915, ASS=0.528. Hyperparameters epsilon=0.02, T=25, beta=0.7-0.8.
- **Difficulty**: medium. **Compute**: medium (IG costs about s times the gradient). **Reuse**: ~50-60%.

### D. Transfer / surrogate attack (RoMa, global noise + surrogate models) — rank 3
- **Principle**: inject global data-driven noise over the whole image; additionally train ViT-B16 + EfficientNet-B0 surrogates, optimize the noise with a combined (ASR+SSIM) loss. LQ uses Adam directly, HQ uses DI-FGSM.
- **Representative**: RoMa (Fraunhofer SIT / ATHENE, DOI 10.1145/3746027.3761984); also FPBA, arXiv 2410.01574.
- **White/black-box**: extremely strong white-box (CNN ~98%), **weak black-box** (ViT 3.2% / DCT 3.9%).
- **Result**: FS=2679, SSIMS=0.934 (highest), ASS=0.509.
- **Difficulty**: medium to high (needs self-trained surrogates + FFHQ data). **Compute**: medium to high. **Reuse**: ~40-50%.

---

## 3. Comparison table

| Dimension | A Pixel (baseline) | B Latent (1st) | C IntGrad/Consensus (2nd) | D Transfer/Surrogate (3rd) |
|---|---|---|---|---|
| Core mechanism | pixel signed gradient + epsilon proj | DDIM inversion + latent opt + decode | integrated gradient + consensus/orthogonal | global noise + surrogate + DI-FGSM/Adam |
| Representative team | existing baseline / GRADIANT | MR-CAS (UCAS) | Safe AI (UNIST) | RoMa (Fraunhofer) |
| Official FS | PGD1655 / MI-FGSM2485 (paper) | **2740 (1st)** | 2709 (2nd) | 2679 (3rd) |
| Official SSIMS | not specified (baseline) | 0.742 | 0.915 | 0.934 |
| Official ASS | not specified (baseline) | 0.672 (highest) | 0.528 | 0.509 |
| White-box | strong (91%) | strong | extreme (99.96%) | extreme (~98%) |
| Black-box / transfer | weak (4-6%) | strong (main point) | weak (7.16%) | weak (ViT3.2%/DCT3.9%) |
| Implementation difficulty | low | high | medium | medium to high |
| Compute / time | low | high (SD+VLM) | medium (IG ~s x) | medium to high (surrogate training) |
| Reuse of existing code | ~100% | ~20-30% | ~50-60% | ~40-50% |

---

## 4. Main risks of each route (objective)

- **A Pixel-level**: weak black-box transfer is the biggest shortfall; under epsilon=8/255 SSIM drops as attack strength grows (already ~0.79 at 8/255, near the constraint line).
- **B Latent**: highest compute/engineering cost, long cycle; **after decoding the latent perturbation back to pixels it does not necessarily satisfy epsilon<=8/255 / SSIM>=0.74** (MR-CAS SSIMS only 0.742, on the line), needs dedicated checking; high risk if done wrong (the similar diffusion approach by team MILab scored only FS 110, nearly a wipeout).
- **C IntGrad/Consensus**: near-perfect white-box but extremely weak black-box (7.16%); high FS is mainly from the two white-box models; surrogate choice must be careful (functional equivalence > architecture alignment).
- **D Transfer/Surrogate**: cross-model transfer is the fatal shortfall (surrogates fail to transfer even with matched architecture); needs self-trained surrogates (extra data + training + uncertainty).

**Cross-route commonality**: current FS is mainly contributed by the two white-box models; the two black-box models are the common bottleneck for all routes. **If this edition (NTIRE 2026) raises the black-box weight, B (best transferability) gains the most.**

---

## 5. Decision hints (for reference, not a conclusion)

- Want a **fast baseline + low cost** -> **A**, the existing `aadd_attack.py` works directly, add MI-FGSM/DI-FGSM first.
- Want the **highest FS with enough compute** -> **B**, but reserve an SD + VLM environment and tuning cycle, strictly verify pixel-domain constraints.
- **Low-cost white-box boost + some ensemble transfer** -> **C**, moderate effort, reuse half the existing code.
- **D**'s global noise + DI-FGSM can be a transfer-boost supplement to A; as a standalone main line the transfer risk is high.

A common robust path: **first use A to establish a white-box baseline (low cost, validates the full pipeline), then decide whether to invest in B/C/D for black-box transfer.** This remains the team's call.

---

## 6. To confirm / information gaps

- This edition is **NTIRE 2026**; the FS/SSIMS/ASS above are measured values from **AADD-2025** papers. If NTIRE 2026 adjusts scoring weights or the detector set, **follow the latest official rules** (not specified).
- Route A's official SSIMS/ASS have no independent data (the leaderboard only lists each team's method).
- The `real/` images are not included in the downloaded data; needed when computing the ROC AUC baseline in evaluation.

---

## Appendix: key reusable files
- `aadd_attack.py` — 4-model differentiable ensemble I-FGSM baseline (with differentiable DCT, official-consistent SSIM, I/O), largely reusable for A/C/D
- `ifgsm_attack.py` — generic I-FGSM reference
- `papers/` — MR-CAS / MIG-COW / RoMa / challenge overview
- `paper_cheatsheet.md` — quick index
