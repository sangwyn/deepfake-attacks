# AADD-2025 Paper Cheat Sheet

> For meeting prep. Three-sentence core summary per paper.

***

## Challenge paper (must read)

**AADD-2025: Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media**
Battiato et al., ACM MM 2025, pp.13714-13719
DOI: 10.1145/3746027.3761983

1. 16 subsets (8 HQ + 8 LQ, GAN + Diffusion), 4 detectors (ResNet50, DenseNet121, ViT-B-16, DenseNet121-DCT), score FS = sum SSIM(I, I_adv) x [detector fooled].
2. Key conclusion: **latent-space attacks outperform pixel-level methods** — MR-CAS optimizes in the latent space via DDIM inversion, achieving higher SSIM and better transferability.
3. real=0, fake=1; attack goal = make the detector classify a fake image as real(0); top-3 SSIM range 0.74-0.93.

***

## Solution 1 — MR-CAS (rank 1, FS=2740)

**A Unified Framework for Stealthy Adversarial Generation via Latent Optimization and Transferability Enhancement**
University of Chinese Academy of Sciences
arXiv: 2506.23676

1. **Core method**: use DDIM inversion to map the fake image back to latent space, do momentum gradient optimization on the latent at intermediate timesteps, then decode back to pixels.
2. **Why strong**: latent-space perturbations naturally lie on the generative manifold, giving very high visual imperceptibility (high SSIM), and remain compatible with classic transfer boosts (MI-FGSM, DIM, SIM, TIM, ensemble).
3. **Likely questions**: how are DDIM inversion steps/timesteps chosen? what is the latent optimization objective? how much SSIM improvement over pixel I-FGSM?

***

## Solution 2 — MIG-COW / Safe AI (rank 2, FS=2709)

**MIG-COW: Transferable Adversarial Attacks on Deepfake Detectors via Gradient Decomposition**
UNIST (Ulsan National Institute of Science and Technology)
DOI: 10.1145/3746027.3761986

1. **Core method**: Momentum Integrated Gradients + consensus orthogonal weighted decomposition — decompose multi-model gradients into a "shared vulnerability" (consensus) component and a "model-specific" (orthogonal) component, then weight-combine.
2. **Key metrics**: white-box ASR 99.96%, SSIM about 0.915, ASR about 0.528; but **black-box only 7.16%** — counter-intuitive finding: adding low-performance but diverse models to the ensemble actually lowers attack effectiveness.
3. **Likely questions**: how is consensus/orthogonal decomposed (SVD?)? how costly is integrated-gradient computation? why is it effective on non-convolutional structures like ViT?

***

## Solution 3 — RoMa (rank 3, FS=2679)

**Team RoMa @ AADD-2025**
Fraunhofer SIT / ATHENE Center
DOI: 10.1145/3746027.3761984

1. **Core method**: white-box framework using ViT-B-16 + EfficientNet-B0 as surrogates, generating global distributed data-driven noise, refined by Adam iteration.
2. **Comparison experiments**: global noise vs local adversarial patch vs post-processing transforms — global noise achieved the highest ASR and best SSIM on public detectors.
3. **Likely questions**: relation between surrogates and the official 4 classifiers? SSIM advantage of global noise vs per-pixel L-inf perturbation? did they add cross-model transfer boosts?

***

## Quick comparison

| Dimension | MR-CAS (1st) | MIG-COW (2nd) | RoMa (3rd) |
|---|---|---|---|
| FS | 2740 | 2709 | 2679 |
| Paradigm | latent optimization | pixel / gradient decomposition | pixel / global noise |
| Space | latent | pixel | pixel |
| Transfer means | MI-FGSM momentum | consensus/orthogonal weighting | surrogate models |
| Core novelty | DDIM inversion + latent opt | integrated gradient + SVD decomposition | global noise + Adam refinement |
| Lesson for baseline | pixel I-FGSM ceiling is low, try latent | low-cost ensemble transfer boost | simple and robust, good baseline reference |

***

## Additional literature

1. **arXiv 2410.01574** — *Adversarial Robustness of AIGI Detectors in the Real World*
   - 4 detection methods x 5 attack algorithms, shows performance can be greatly degraded without internal knowledge of the detector architecture
   - Lesson: feasibility of black-box transfer attacks

2. **arXiv 2407.20836** — *Vulnerabilities in AIGI Detection (FPBA)*
   - Frequency-domain perturbation + post-trained Bayesian surrogate (a single surrogate simulating diverse victim models)
   - Code: github.com/onotoa/fpba
   - Lesson: attack ideas for frequency-domain / DCT models

3. **arXiv 2505.03435** — *Robustness in AI-Generated Detection*
   - Defense perspective: adversarial training + diffusion-inversion reconstruction to harden detectors
   - Lesson: understand how detectors are hardened, reverse-infer attack weaknesses

4. **ARMOR++** (arXiv 2607.15246, 2026) — agentic-reasoning black-box transfer attack on the AADD-2025 benchmark
   - Directly targets this challenge, worth close tracking

***

## Our baseline (I-FGSM ensemble attack)

- **Adaptation points**: targeted attack (fake->real) / 4-model differentiable wrapper / differentiable DCT / ensemble gradient averaging / SSIM-aware tuning
- **Expected baseline**: epsilon=8/255, steps=20 -> SSIM ~0.8+, ASR TBD
- **Reference target**: rank 3 RoMa FS=2679; the gap decides whether to upgrade the method

***

## Questions to confirm with the team lead

1. Compute resources: how many GPUs? enough for latent-space optimization (needs Stable Diffusion U-Net)?
2. Direction trade-off: establish a baseline with pixel ensemble I-FGSM first, or go straight to the MR-CAS-style latent route?
3. Submission scope: full ~1400 images or a subset first? output format (PNG, any bit-depth/size limit)?
4. Black/white-box: is evaluation white-box (all weights available), or must we also consider transfer to unknown models?
5. Score interpretation: FS uses sum; does it weigh ASR or SSIM more (affects epsilon tuning strategy)?
