# Handoff — deepfake-attacks

Context for a new agent picking this up. Read this before touching code.

## What this project is

A **lightweight pipeline for running and evaluating adversarial attacks on
deepfake / AIGC image detectors**, for the ACM AADD-2026 challenge
(https://iplab.dmi.unict.it/mfs/acm-aadd-challenge-2026/). Goal: perturb "fake"
images so detectors classify them as "Real" (target class 0), while keeping them
visually close (SSIM / LPIPS).

## Scope & minimalism — READ THIS FIRST

This is a **hackathon project**. The bar is *bare minimum that works*, not a
framework. The author has explicitly and repeatedly asked to **avoid
overengineering**. Concretely:

- **No** registries, factories, plugin systems, Hydra, package builds, or config
  frameworks. Attacks are one-file-per-attack in `attacks/`, discovered by name
  via `import_module`.
- Keep Torch in the attacks; keep PIL/NumPy/file-encoding in `evaluate.py`.
- Prefer a few inline asserts / a tiny smoke script over a real test suite.
- Add a dependency only when it earns its place (see kornia note below).
- Do not "improve" the graded parts of the config (see below).

When in doubt, do the smaller thing and leave a note here.

## Repo layout

```
evaluate.py            # the pipeline (given by organizers, lightly modified)
configs/AADD_2026_config.yaml
attacks/
  _common.py           # shared: img<->tensor, differentiable preprocessing, DI, jpeg
  ifgsm.py             # BASELINE: targeted ensemble BIM (sanity check only)
  midi_fgsm.py         # targeted ensemble MI-DI-FGSM + EOT (for transfer)
  template.py          # attack interface example
weights/               # git-ignored; put <name>.pth here yourself
README.md
```

## How the pipeline works (30-second version)

`evaluate.py`: load config → build the detectors listed in `classifiers` (each
needs `weights/<name>.pth`) → for each image: run `attack(...)`, optionally
save+**reload** the attacked image, compute SSIM/LPIPS, run each detector →
score = `clf_weight * (alpha*SSIM + (1-alpha)*(1-LPIPS))` summed over detectors
where the prediction is class 0. Writes a JSON report.

Attack contract: `attack(image, classifiers, device, **attack_params)` where
`image` is `H×W×3 uint8 RGB`, returns the same. Extra kwargs come from the
config's `attack_params` block.

Run: `python evaluate.py --config configs/AADD_2026_config.yaml`

## Selectable vs. fixed

- **Selectable (your knobs):** `attack`, `attack_params`, `classifiers`
  (your surrogate set), `models_dir`, `original_root`, `save_attacked_dir`,
  `save_json`, `device`.
- **Fixed / graded — do NOT change:** `dct_log_scale`, `weights`,
  `aggregate: sum`, `alpha`. These define the challenge score.
- **Hidden detectors are not in this loop.** You only ever attack/score the
  detectors you have weights for. The challenge's held-out detectors + random
  JPEG/resize post-processing are unseen. Local score is a white-box proxy; the
  real game is *transfer*, which is why midi_fgsm exists.

## What was recently done

- De-hardcoded the attack (was `classifiers['vit_b_16']` only) → ensemble over
  all provided detectors.
- Added `attack_params` passthrough in `evaluate.py`.
- **Scoring fix:** when `save_attacked_dir` is set, images are re-loaded from
  disk before scoring, so the reported score matches the *submitted* file (JPEG
  re-encoding no longer silently diverges from the in-memory score).
- Added differentiable Torch preprocessing (spatial + a verified orthonormal
  2-D DCT) in `_common.py` so both detectors are attackable white-box.
- Added `midi_fgsm` (MI + DI + EOT, optional differentiable JPEG via kornia).
- Fixed README/docstring run command; documented attacks and params.

## Open issues (verify before trusting anything)

1. **Nothing is runtime-tested.** The dev box where this was last edited had no
   torch/scipy, no `vit_b_16.pth`, and no test images. Only syntax, imports, and
   DCT orthonormality were checked. **First task: a 2–3 image smoke run.**
2. **Config lists `vit_b_16` + `densenet121_dct`, but only
   `weights/densenet121_dct.pth` exists.** You need the organizer's ViT
   checkpoint, or trim `classifiers`. `weights/` is git-ignored by design.
3. **Preprocessing is a surrogate**, not bit-identical to `evaluate.py`:
   bilinear+antialias vs PIL LANCZOS, grayscale skips PIL rounding, DI pads with
   black. Affects gradient quality, not scored pixels. Worth: numeric check of
   `_common.dct_preprocess` vs the SciPy transform in `evaluate.build_dct_transform`.
4. **JPEG EOT ≠ organizer post-processing.** Theirs is randomized/hidden;
   `jpeg_quality` only approximates it and needs kornia (optional dep).
5. **Transfer is unverifiable locally** — released detectors are surrogates.
   Postmortems (see `../pdfs/`) show big white-box→black-box drops. Mitigate with
   more diverse surrogates + stronger EOT; can't be "fixed."
6. **Final uint8 rounding isn't re-projected** — a pixel can land ≤0.5/255
   outside the ε-ball. Negligible but real.
7. **No RNG seeding** — DI/EOT runs aren't reproducible.
8. **DCT resize mismatch:** eval resizes to 256 only if `max(size) > 256`; the
   attack surrogate always resizes. Only matters for images ≤256px.
9. **Cost** scales as `iterations × eot_samples × num_models × num_images`
   (e.g. 20×4×2×1600 = 256k model calls). Budget GPU time.
10. **LPIPS(alex)** downloads weights on first run (needs network once).

## Potential improvements (roughly high→low value; keep them minimal)

- **Smoke checks** (tiny script or inline asserts, not a suite): identity attack
  → SSIM≈1 / LPIPS≈0 / identical saved-reloaded pixels; attacked output in
  `[0,255]`, correct shape, respects ε after quantization; `dct_preprocess`
  matches SciPy within tolerance.
- **More/diverse surrogate detectors** in `classifiers` — the single biggest
  lever for transfer. Add a `build_preprocess` branch per new architecture.
- **Seed knob** in `attack_params` for reproducibility.
- **Per-image JSON** (path, preds, SSIM, LPIPS, L∞/L2) + **clean predictions**
  alongside attacked — cheap and very useful for debugging transfer.
- Tune midi_fgsm: `eot_samples`, `di_prob`, `di_pad_ratio`, `decay`, more
  `iterations`, `jpeg_quality` range.
- Ignore `experiments/` and generated attacked images in git.

## Reference code to follow (conventions, not to copy wholesale)

- pralab/RAID — closest AIGC-detector attack benchmark; good separation of raw
  pixels / model preprocessing / attack / evaluation.
- cihangxie/DI-2-FGSM — input diversity (transfer).
- PyTorch FGSM tutorial / Foolbox — naming and correctness references.

No public official code found for MIG-COW, Team RoMa, or "Stealthy Adversarial
Generation" as of 2026-09-01 — don't build on unofficial reproductions.
