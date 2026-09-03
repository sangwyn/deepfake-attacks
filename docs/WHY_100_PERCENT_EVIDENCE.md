# Why is the fooling rate 100%? — Evidence and explanation

**Question asked:** ViDA reports 100% fooling on both detectors. Why should this
be believed / how is it possible?

## 1. What "100%" actually refers to (scope)

- **Setting:** white-box. We have the detector weights and gradients for the
  two graded detectors (`vit_b_16`, `densenet121_dct`). This is exactly the
  AADD-2026 white-box protocol — not a black-box or hidden-model claim.
- **Data:** the designated dev split `/home/aiattacks/dataset/celebA/TEST`
  (100 fakes + 100 reals, 200 images). 100% means every one of these 200
  images is predicted **Real** by both detectors after attack.
- **Budget:** L∞ perturbation ≤ 8/255, the official budget.
- **Not claimed:** 100% on the hidden official test set, or 100% on unseen
  detectors. Black-box transfer experiments show exactly the expected drop
  (NPR ~57.5% without DI, 82.5% with DI; AIDE 0%). We report those low numbers
  openly — the white-box/black-box gap is a well-established phenomenon and
  matches the literature (Dong et al. 2018; Xie et al. 2019).

## 2. Why 100% white-box fooling is expected, not anomalous

1. **The detectors are standard undefended classifiers.** ViT-B/16 and
   DenseNet-121 are trained for detection accuracy, with no adversarial
   training, no certified defence, no input randomization. In the adversarial
   ML literature, iterative L∞ attacks (PGD/MI-FGSM) achieve essentially
   ~100% ASR on such models at ε = 8/255; 100% is the norm for strong
   white-box attacks, not a red flag.
2. **Independent corroboration from the team baseline.** Yang's Direction-1
   joint PGD (a plain 40-iteration attack) already reaches **ViT 97%**. Our
   ViT 100% is a marginal improvement from more iterations, momentum and
   adaptive per-image compute. The difficult detector was DCT, not ViT.
3. **Why DCT goes from Yang's 46–48% to our 100% — three specific, verified
   reasons (not "we tried harder"):**
   - **Momentum is necessary for the frequency model.** Our ablation: plain
     sign I-FGSM saturates at ~1–2/8 images on DCT even at 100 steps; MI-FGSM
     (µ=1, mean-abs gradient normalization) crosses reliably. DCT log-feature
     gradients are high-frequency/noisy, and momentum stabilizes the direction.
   - **Exact preprocessing match.** The DCT evaluator consumes 0–255
     grayscale with PIL LANCZOS resize. Mismatches (0–1 scale, or bicubic
     instead of LANCZOS) point gradients away from the real decision boundary.
     We replicate LANCZOS with differentiable resampling matrices (matches PIL
     to <1/255) and use the official scale; an early mismatch gave us
     in-branch "success" that disagreed with the official scorer — corrected.
   - **Stabilized log gradient.** `log(|DCT|+1e-6)` produces ~10^6-scale
     gradients on near-zero high-frequency coefficients; a magnitude-capped
     backward pass keeps sign updates usable, while the forward pass stays
     bit-identical to the evaluator.
4. **Guaranteed ASR by construction.** Stage D of the attack re-runs the
   official transforms on the rounded uint8 image and re-attacks any detector
   that slipped; the returned image is chosen as
   argmax(detectors fooled, Q). So a returned image is only ever kept when
   the official prediction check passes. Fooling is not "expected from
   gradients" — it is verified through the evaluator's own code path on the
   saved pixel values.

## 3. How to independently verify it

1. Same checkpoints: `vit_b_16.pth` sha256 =
   `5e9677d8...58d81`, identical to the hash printed in Yang's own report;
   `densenet121_dct.pth` = `5bbaf5c5...5e643`.
2. Same evaluator: `team_repo/evaluate.py` unchanged; attack code is only
   `team_repo/attacks/vida.py` (commit `66096c9`).
3. One-command reproduction:
   `cd team_repo && python evaluate.py --config configs/v34_final_no_stage0.yaml`
4. Independent rescoring of saved outputs (no attack code imported; uses
   only official model factory, official transforms, skimage SSIM, LPIPS-Alex):
   `python my_attack/rescore_saved.py --orig <PNG-staged data> \
       --adv my_attack/verify_outputs/vida_v34_images_png --weights team_repo/weights`
5. Per-image rows: `my_attack/verify_outputs/per_image_results.csv`
   (200 rows; ViT-fooled = 200/200, DCT-fooled = 200/200, total 397.97/400).

## 4. What would be suspicious (and is NOT the case here)

- Claiming 100% on hidden/black-box detectors — we do not (AIDE transfer = 0%).
- Scoring on modified evaluator code — unchanged `evaluate.py`.
- Using different checkpoints — hashes match.
- Round-trip artifacts (JPEG save) changing results — all verification outputs
  are lossless PNG; a JPEG-caveat check (2 boundary reals flipping after JPEG
  compression) is documented honestly in REPRODUCIBILITY.md.

## 5. Final independent confirmation (2026-09-04)

All 200 adversarial images were saved as lossless PNG and re-scored from disk
with `rescore_saved.py`, which imports only the official `evaluate.py`
factories (model construction, official transforms, skimage SSIM, LPIPS-Alex)
— the attack module is never imported:

```
images rescored : 200
ViT  Real rate  : 200/200
DCT  Real rate  : 200/200
mean SSIM       : 0.9941
mean LPIPS      : 0.0053
TOTAL SCORE     : 397.7590 / 400
```

This matches the official evaluator's JSON from the same run
(397.759003) exactly. Per-image rescoring log:
`my_attack/verify_outputs/rescore_independent.log`.

Note: the verification run scored 397.76 vs 397.97 in the earlier run — the
~0.2-point gap is run-to-run variance of the multi-start rescue on a few
borderline images (documented as the hard-case instability finding); both
runs give 200/200 fooling on both detectors.
