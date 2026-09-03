# Adversarial Attacks on AI Image Detectors

This branch adds a reproducible research pipeline and a restricted OpenCode
control plane around the original AADD evaluator. The original `evaluate.py`
and attack modules remain available; new orchestration lives beside them.

## Separation of concerns

| Purpose | Canonical location on the server | Git policy |
|---|---|---|
| Versioned project and agent instructions | `/home/aiattacks/oleg/aadd-attack-pipeline` | tracked |
| CelebA dataset | `/home/aiattacks/dataset/celebA` | external, read-only |
| Detector and LPIPS weights | `/home/aiattacks/oleg/aadd-attack-assets/weights` | external, hash-verified |
| Heavy run artifacts | `/home/aiattacks/oleg/aadd-attack-runs` | external |
| Compact run ledger | `tracking/` inside the project | tracked |
| GPU queue database and locks | `.gpuq/` inside the project | local, ignored |

Never put API keys in this repository. The project references the existing
OpenCode provider configuration but does not copy it.

## First server setup

- `weights/` contains deepfake classifiers for attacking

Campaign configs use `evaluate.py` with `manifest`, `source_classifiers`,
`target_classifiers`, `objective`, and `attack_params`. The immutable manifest is
JSONL with `sample_id`, dataset-relative `relative_path`, directory-derived
`label`, and `sha256` in every row. Supported detectors are `vit_b_16`,
`densenet121_dct`, `npr`, and `aide`.

LPIPS runs on CPU by default so detector gradients retain the GPU memory budget.
Set `metric_device: cuda` explicitly only when the GPU has sufficient headroom.
