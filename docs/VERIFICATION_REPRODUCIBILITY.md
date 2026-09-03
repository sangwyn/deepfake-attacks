# Verification / Reproducibility package — ViDA v3.4-final-no-stage0

Every number here is produced by the **unchanged official `evaluate.py`**.
No attack-side code is involved in scoring.

## 1. Provenance checksums

| Artifact | sha256 |
|---|---|
| `vit_b_16.pth` (checkpoint) | `5e9677d88a7af10791001796eb43d0d060fada3758369814d6d7832934758d81` |
| `densenet121_dct.pth` (checkpoint) | `5bbaf5c5c0e296d5e819a0b401198c73ad69c6bbc8f372579de5ee5c11d5e643` |

Note: the ViT hash **matches the checkpoint hash recorded in Yang's own
Five-Directions report** — identical weights are used.

- Source images: 200 files, md5 per file listed in
  `team_repo/experiments/yang_comparison/manifest.json`
- Attack code commit: `66096c9` on branch `mingyao-dev` (`team_repo/attacks/vida.py`)

## 2. One-command reproduction (uses official evaluator only)

```bash
cd team_repo
python evaluate.py --config configs/v34_final_no_stage0.yaml
# results JSON -> experiments/yang_comparison/vida_v34_final_no_stage0.json
```

## 3. Independently re-score the SAVED adversarial images (strongest check)

The verification run saves every output image under
`my_attack/verify_outputs/vida_v34_images/<TEST_FAKE|TEST_REAL>/`. You can
re-score these PNGs/JPGs with your own evaluator WITHOUT any attack code:

```bash
python my_attack/rescore_saved.py \
    --orig /home/aiattacks/dataset/celebA/TEST \
    --adv  my_attack/verify_outputs/vida_v34_images \
    --weights team_repo/weights
```

`rescore_saved.py` only imports the official `evaluate.py` factories
(model factory, official transforms, official SSIM/LPIPS); it never imports
the attack. If the saved images genuinely fool both detectors with high Q,
any compliant evaluator must reproduce the same score.

## 4. Reported result (200 images, aggregate=sum)

| Metric | Value |
|---|---|
| Official score | 397.97 / 400 |
| ViT / DCT fooled | 200/200, 200/200 |
| Mean SSIM / LPIPS | 0.9945 / 0.0046 |
| Per-image rows | `per_image_results.csv` (200 rows, from the official log) |

## 5. Visual sanity

`my_attack/samples/` contains before/after panels (original | adversarial |
perturbation amplified x10) for a hard fake (001539), another hard fake
(001556), an easy fake (001549) and a real image.

## 6. Note on save format (JPEG caveat found during verification)

When adversarial images are saved with a `.jpg` extension (PIL applies JPEG
compression), 2 of 200 boundary cases (reals that ViT naturally misclassifies
as Fake, e.g. TEST_REAL/001578, 001599) can flip back to Fake on the compressed
file, while the official in-memory prediction is Real. The official evaluator
scores the in-memory uint8 array before/independent of saving, so the official
397.97/400 is unaffected; but all artifacts in THIS verification package are
saved as lossless PNG (dataset staged via `.png` symlinks in `db_png/`), and
rescoring must be done on those files (`vida_v34_images_png/`). This is also a
useful robustness reminder: deliver/evaluate attack outputs as PNG.
