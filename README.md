# Adversarial Attacks on AI Image Detectors

## Motivation

## Structure

- `attacks/` contains `.py` files for all the attacks (one file per attack). See [attacks/template.py](attacks/template.py) for an example

- `evaluate.py` is used to run the evaluation pipeline from AADD 2026 challenge

- `configs/` contains config files for *evaluate.py*

- `weights/` contains deepfake classifiers for attacking (git-ignored; place the
  `.pth` files here yourself)

## Running

```bash
python evaluate.py --config configs/AADD_2026_config.yaml
```

The config's `classifiers` list must have a matching `<name>.pth` in `weights/`
(e.g. `vit_b_16` needs `weights/vit_b_16.pth`). Remove a classifier from the
list to skip it.

Attacks receive extra keyword arguments from the optional `attack_params` block
in the config, e.g.:

```yaml
attack: ifgsm
attack_params:
  epsilon: 0.03137   # 8/255
  step_size: 0.00784 # 2/255
  iterations: 10
```

## Attacks

- `ifgsm` — baseline targeted ensemble BIM. Sanity check that the pipeline works
  white-box; no transfer tricks.
- `midi_fgsm` — targeted ensemble **MI-DI-FGSM + EOT**, built for transfer to
  unseen detectors and post-processing (JPEG/resize). See its docstring for the
  full `attack_params`. The `jpeg_quality` EOT option needs `kornia`
  (`pip install kornia`); omit it to run without.

Both attack *every* detector in the config's `classifiers` list (ensemble), so
add more surrogate `.pth` detectors there to improve transfer.