# My Experiment Plan — Mingyao Duan

> Use: for comparing each team member's experiment plan tomorrow. This is **my (Mingyao Duan)** plan.
> Date: 2026-09-01. Branch: `mingyao-dev`.

---

## 0. My scope

- I am not responsible for the dataset problem (that is Dan). I assume that when work starts tomorrow, the **AADD-2026 test data + the 2026 vit_b_16.pth** are already in place from Dan/Oleg.
- Before the data is ready, my plan can still do a pipeline-level dry run using the **local AADD-2025 1403 fake images + the densenet121_dct.pth shipped in team_repo** (already verified runnable).

---

## 1. Experiment goal

Under team_repo's `attacks/` interface, produce and compare **multiple attack versions**, and find the one with the highest **FS** under the AADD 2026 score (0.5*SSIM + 0.5*(1 - LPIPS), 2 detectors).

---

## 2. Experiment variables (what I compare)

| Dimension | Options |
|---|---|
| Base algorithm | I-FGSM (baseline) / MI-FGSM (momentum) / PGD |
| Transfer boost | none / momentum / input diversity (DI) / both |
| Ensemble | attack vit only / dct only / average both / weighted average |
| Perceptual constraint | epsilon only / add an LPIPS penalty in the loss |
| Hyperparameters | epsilon in {4,8}/255, alpha in {1,2}/255, iters in {10,20,40} |

---

## 3. Experiment steps (to run tomorrow)

1. **Confirm environment** (ready): venv + torch cu124 + lpips, GPU available (8x L40).
2. **Confirm data/weights in place**: after Dan delivers 2026 data and vit weights, put them into `team_repo/weights/` and the config's original_root.
3. **Reproduce baseline**: run evaluate once with the existing `attacks/ifgsm.py`, record baseline FS / ASR / SSIM / LPIPS.
4. **Experiment 1 — ensemble**: change ifgsm from "attack vit only" to "average vit+dct gradients", observe FS change (expected: dct is also fooled, FS up).
5. **Experiment 2 — transfer boost**: add MI-FGSM momentum + DI input diversity, observe FS change.
6. **Experiment 3 — perceptual optimization**: add an LPIPS term to the loss; see if we can lower LPIPS (raise similarity score) without dropping ASR.
7. **Small hyperparameter grid**: sweep epsilon/alpha/iters on the best version.
8. **Record comparison table**: one row per version, columns FS / ASR_vit / ASR_dct / mean_SSIM / mean_LPIPS.

---

## 4. Output of each experiment

- `attacks/mingyao_v1_ensemble.py`, `mingyao_v2_mi_di.py`, `mingyao_v3_lpips.py` (one attack file each, conforming to the team interface)
- One evaluate JSON result per version
- A summary comparison table (markdown)

---

## 5. Metrics (aligned with evaluate.py)

- **Primary**: final_score (FS, aggregate=sum)
- **Secondary**: per-classifier attack_success (ASR), mean_ssim, mean_lpips
- **Constraint check**: confirm L-inf <= 8/255 (guaranteed by clamp in the attack code)

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 2026 data/weights not ready in time | Do a pipeline dry run first with 2025 data + team dct weights, get the logic running |
| DCT model gradients do not flow | Use a differentiable DCT (matrix method, already verified equivalent to scipy ortho in aadd_attack.py) |
| Attack succeeds but evaluate marks it as failed | Numerical mismatch between the differentiable attack-time preprocessing and evaluate's PIL preprocessing; always cross-check with evaluate |
| LPIPS term slows down / is unstable | Skip it first, get the baseline running, then add incrementally |
| Poor black-box transfer (if evaluation includes held-out models) | Rely on MI/DI boost + ensemble; if needed refer to the MR-CAS latent-space idea (heavy engineering, as a fallback) |

---

## 7. Preparation already done tonight (not consuming tomorrow's time)

- [x] Understood team_repo interface and scoring (2 detectors + LPIPS)
- [x] Environment set up, GPU working, evaluate pipeline running locally (identity smoke test)
- [x] Created the `mingyao-dev` branch
- [x] Adversarial-attack basics notes + vibecode notes + prompt draft (the other three files in this folder)
- [x] Confirmed 2025 weights cannot be used for 2026 (empirically), reported the gap to the team
