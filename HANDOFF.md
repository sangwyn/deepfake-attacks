# Handoff — deepfake-attacks

Context for a new agent picking this up. Read this before touching code.

## What this project is

A **hackathon task: develop new adversarial attacks on deepfake / AIGC image
detectors.** The deliverable is the attacks, not a challenge ranking.

`evaluate.py` + `configs/AADD_2026_config.yaml` are taken from the ACM AADD-2026
challenge (https://iplab.dmi.unict.it/mfs/acm-aadd-challenge-2026/) and used as
the **fixed evaluation core** — treat them as the scoring harness, not something
to redesign. Your work happens in `attacks/`: add new attack modules and iterate,
scored by this pipeline.

Attack goal: perturb "fake" images so detectors predict "Real" (target class 0)
while staying visually close (SSIM / LPIPS). `ifgsm.py` (baseline) and
`midi_fgsm.py` (transfer-oriented) are starting points to build on, not the
endpoint.

## Scope — what's fixed vs open (READ before assuming constraints)

**Fixed (the only hard constraint):** the evaluation core — `evaluate.py` and the
graded config keys (`dct_log_scale`, `weights`, `aggregate: sum`, `alpha`). This
defines the score; don't change it.

**Open (do NOT treat as law):** everything about the attacks themselves. There
are **no strict limits on ideas.**
- **White-box** and **ε=8/255 L∞** are the *current baseline's* assumptions, not
  task rules. Proposals may be black-box, use other norms, looser/larger or
  unbounded budgets, query-based, generative — whatever. The baseline just
  happens to be white-box + L∞.
- The existing `attacks/` (BIM, MI-DI-FGSM) are **starting points, not a
  template you must conform to.** New attacks only need to satisfy the attack
  contract (numpy uint8 in → numpy uint8 out) so the pipeline can score them.

If you find yourself thinking "the constraint is X", check it against this
section — most apparent constraints are just baseline defaults.

## Working state (context, not a plan)

This is a **5-person hackathon team**; several members run their own agents on a
shared GPU server, each in a directory with its own copy of this repo plus
personal scripts/notes/artefacts. So **this directory is one of several parallel
copies, not the single source of truth** — don't assume it's authoritative and
don't clobber shared artefacts. **Experiment results** are what matters across
copies, but they may be logged in **different formats** and merged/synthesized
later by another agent — so don't impose or assume a single results schema; just
keep results legible enough to merge. No collaboration/merge plan is fixed yet.

## Experiment results

`evaluate.py` writes a JSON report when `save_json` is set: `final_score`,
`aggregate`, `alpha`, `images_evaluated`, `per_classifier` (attack_success,
clean_real_rate, mean_ssim, mean_lpips) and `per_image` (path, ssim, lpips, linf,
clean-vs-attacked prediction). This is a sensible default, **not a fixed schema** —
results across teammates may be logged in different formats and merged/synthesized
later by another agent, so don't treat these field names as a contract.

A failing image (corrupt file, attack error) is warned about and **skipped** via
a minimal `try/except … continue` around the per-image body, so a long run
doesn't die partway. (The organizer's original SSIM `try/except → 0.0` fallback
also remains; those two are the only error-handling kept.)

(Note: an earlier pass also added a heavy self-describing `meta` block, a
`failures` list, and `run_name`/`{timestamp}` filename placeholders. That was
judged too rigid/overengineered for a hackathon and **removed** — only the plain
skip above remains. Don't re-add the rest without a reason.)

## Engineering minimalism

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
tests/smoke.py         # `python tests/smoke.py` — DCT/identity/eps checks, no weights needed
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
  `save_json`, `device`, `seed`.
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
- Ran `tests/smoke.py` on CPU (torch 2.13): all checks pass. Fixed two bugs it
  surfaced — `sys.path` so it runs from any cwd, and a relative DCT tolerance.
  `tests/smoke.py` asserts the float32 DCT path (the one the attack uses) matches
  scipy to ~2e-6 relative. A one-off dev check (not in the committed test) also
  confirmed the float64 matrix matches scipy to ~3e-15 (machine precision).
- **Rolled back an over-engineered hardening pass** (added by another agent):
  removed the self-describing `meta` block, `failures` list, and
  `run_name`/`{timestamp}` filename placeholders from `evaluate.py` + config.
  Kept only a minimal `try/except … continue` that warns and skips a failing
  image. See the "Experiment results" section for details.

## First steps on the GPU server

1. `pip install -r requirements.txt` (add `kornia` only if you use midi_fgsm's
   `jpeg_quality`). Note: requirements pin `torch==2.3.0`; it was smoke-tested on
   2.13.0 — match the organizer's stack for the real run.
2. Put `weights/<name>.pth` for every name in the config's `classifiers`
   (`weights/` is git-ignored). You have `densenet121_dct.pth`; you still need
   the organizer's `vit_b_16.pth`, or trim `classifiers`.
3. `python tests/smoke.py` — no weights/data needed; confirms env + attack
   mechanics. Should print `SMOKE PASS`.
4. Set `original_root` to the test images and do a **2–3 image run first**
   (point it at a tiny subdir) before the full set — see the cost note below.
5. `python evaluate.py --config configs/AADD_2026_config.yaml`.

## Open issues (verify before trusting anything)

1. **Real detectors + data are still unexercised.** Only dummy-model smoke runs
   and CPU checks were done; no `vit_b_16.pth` or test images were available.
   The 2–3 image run in step 4 above is the first real end-to-end test.
2. **Config lists `vit_b_16` + `densenet121_dct`, but only
   `weights/densenet121_dct.pth` exists.** You need the organizer's ViT
   checkpoint, or trim `classifiers`. `weights/` is git-ignored by design.
3. **Preprocessing is a surrogate**, not bit-identical to `evaluate.py`:
   bilinear+antialias vs PIL LANCZOS, grayscale skips PIL rounding, DI pads with
   black. Affects gradient quality, not scored pixels. (The DCT step itself is
   verified correct — only the resize/grayscale differ.)
4. **JPEG EOT ≠ organizer post-processing.** Theirs is randomized/hidden;
   `jpeg_quality` only approximates it and needs kornia (optional dep).
5. **Transfer is unverifiable locally** — released detectors are surrogates.
   Postmortems (see `../pdfs/`) show big white-box→black-box drops. Mitigate with
   more diverse surrogates + stronger EOT; can't be "fixed."
6. **Final uint8 rounding isn't re-projected** — a pixel can land ≤0.5/255
   outside the ε-ball. Negligible but real.
7. **DCT resize mismatch:** eval resizes to 256 only if `max(size) > 256`; the
   attack surrogate always resizes. Only matters for images ≤256px.
8. **Cost** scales as `iterations × eot_samples × num_models × num_images`
   (e.g. 20×4×2×1600 = 256k model calls). Budget GPU time.
9. **LPIPS(alex)** downloads weights on first run (needs network once).

## Already done (don't redo)

- Seed knob (`seed` in config, seeds random/numpy/torch globally).
- `tests/smoke.py`: DCT-vs-scipy (float32 path, ~2e-6 rel), identity, and
  shape/range/ε checks — runs from any cwd, passes on CPU.
- Per-image JSON records + clean-vs-attacked predictions + per-classifier
  `clean_real_rate` in the report.
- `experiments/` git-ignored.

## Potential improvements (roughly high→low value; keep them minimal)

- **More/diverse surrogate detectors** in `classifiers` — the single biggest
  lever for transfer, but needs weights + a `_common.build_preprocess` branch
  per new architecture (data/infra, not just code).
- Tune midi_fgsm: `eot_samples`, `di_prob`, `di_pad_ratio`, `decay`, more
  `iterations`, `jpeg_quality` range.
- Add L2 perturbation / saved-vs-reloaded pixel-identity to `tests/smoke.py` if
  you start seeing quantization surprises.

## Reference code to follow (conventions, not to copy wholesale)

- pralab/RAID — closest AIGC-detector attack benchmark; good separation of raw
  pixels / model preprocessing / attack / evaluation.
- cihangxie/DI-2-FGSM — input diversity (transfer).
- PyTorch FGSM tutorial / Foolbox — naming and correctness references.

No public official code found for MIG-COW, Team RoMa, or "Stealthy Adversarial
Generation" as of 2026-09-01 — don't build on unofficial reproductions.
