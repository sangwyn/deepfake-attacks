# Adversarial Attacks on AI Image Detectors

## Motivation

## Structure

- `attacks/` contains `.py` files for all the attacks (one file per attack). See [attacks/template.py](attacks/template.py) for an example

- `evaluate.py` is used to run the evaluation pipeline from AADD 2026 challenge

- `configs/` contains config files for *evaluate.py*

- `weights/` contains deepfake classifiers for attacking

Campaign configs use `evaluate.py` with `manifest`, `source_classifiers`,
`target_classifiers`, `objective`, and `attack_params`. The immutable manifest is
JSONL with `sample_id`, dataset-relative `relative_path`, directory-derived
`label`, and `sha256` in every row. Supported detectors are `vit_b_16`,
`densenet121_dct`, `npr`, and `aide`.

LPIPS runs on CPU by default so detector gradients retain the GPU memory budget.
Set `metric_device: cuda` explicitly only when the GPU has sufficient headroom.
