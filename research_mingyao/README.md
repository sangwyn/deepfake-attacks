# Mingyao's research workspace (ViDA development artifacts)

All code/results for the ViDA attack developed on branch `mingyao-dev`.
The graded deliverable is `attacks/vida.py` (run via the official
`evaluate.py`); this folder is the supporting research trail.

## Layout

- `scripts/` — research/diagnostic Python code:
  - `vida_attack.py` — research harness for ViDA stages (gates, routing, recovery)
  - `diag_hard.py`, `diag_repeat.py` — hard-case margin diagnosis and run-to-run
    variability study (Section 8 / Appendix of the final report)
  - `best_of_k.py` — best-of-k multi-start experiment (k=3/5)
  - `rescore_saved.py` (also in `../my_attack` origin; copy in verification/) —
    independent rescoring of saved adversarial PNGs; imports only the official
    evaluator factories, never the attack
  - `make_samples.py` — before/after visual panels
  - `gates_g1_g3.py`, `gates_g2.py`, `gates_g4_g5.py` — visibility gate experiments
  - `blackbox_transfer.py`, `di_transfer.py` — NPR/AIDE black-box transfer + DI study
  - `dct_ops.py`, `masks.py`, `metrics.py`, `detectors.py`, `attacks.py`, `data.py`
    — differentiable DCT/Lanczos, masks, SSIM/LPIPS, detector wrappers
- `logs/` — raw experiment logs (v3.0–v3.4 dev runs, black-box, DI, diagnosis)
- `results/` — result JSONs, `SUMMARY.md`, `PLAN.md`, REPORT markdown
- `samples/` — visual before/after panels (original | adversarial | x10 perturbation)
- `verification/` — independent verification trail:
  - `per_image_results.csv` — 200 per-image rows from the official log
  - `rescore_independent.log` — attack-free rescoring of saved PNGs
    (ViT 200/200, DCT 200/200, 397.7590/400)
  - `verify_run_png.log/.json`, `verify_saved.yaml` — lossless-PNG verification run

## Not included in git (regenerable / external)

- `vida_v34_images_png/` — the 200 saved adversarial PNGs (~214 MB). Regenerate with
  `python evaluate.py --config configs/verify_saved.yaml` (sets `save_attacked_dir`).
- Detector weights (see `weights/`), third-party reference code (`refcode/`).
