# Adversarial Attacks on AI Image Detectors

This repository contains a small AADD-2026-style evaluation pipeline for
researching targeted adversarial attacks against AI-generated-image detectors.
The code is intended for detector red-teaming and robustness analysis. It is
not a claim of effectiveness against hidden detectors, real camera pipelines,
or an official AADD-2026 leaderboard score.

## Motivation

## Structure

- `attacks/` contains `.py` files for all the attacks (one file per attack). See [attacks/template.py](attacks/template.py) for an example

- `evaluate.py` is used to run the evaluation pipeline from AADD 2026 challenge

- `configs/` contains config files for *evaluate.py*

- `weights/` contains deepfake classifiers for attacking

The repository includes the following attack modules:

- `identity`: no-op pipeline smoke test.
- `ifgsm`: historical targeted ViT-only I-FGSM baseline.
- `vit_pgd`: targeted PGD optimized only against the RGB ViT.
- `dct_pgd`: targeted PGD optimized only against the DCT DenseNet.
- `dual_pgd`: joint targeted PGD optimized against both ViT and DCT models.

`vit_pgd` and `dct_pgd` are single-model ablations. Their other-model result
is post-hoc transfer evaluation, not white-box attack success. `dual_pgd`
uses the same RGB image variable for both branches and combines their input
gradients.

## Run

Paths in a YAML file are resolved relative to that file. The recommended
environment is the project-local `.venv` running Python 3.12. Create it and
install the dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Provide the AADD test images and model checkpoints, then run from the
repository root:

```bash
python evaluate.py --config configs/AADD_2026_config.yaml
```

The default `ifgsm` attack requires both model checkpoints:

```text
weights/vit_b_16.pth
weights/densenet121_dct.pth
```

These weight files are intentionally ignored by Git and should be transferred
or downloaded separately. To verify the evaluator with only the DCT checkpoint,
use the included smoke configuration:

```bash
python evaluate.py --config configs/smoke.yaml
```

This uses the `identity` attack and the available DCT checkpoint. It reports
predictions, SSIM, LPIPS, and JSON output without modifying images.

The current local config expects the sample dataset at `../../Test` relative to
the config file, which resolves to `../Test` from the repository root. Override
`original_root` and `models_dir` when your local layout differs. The dataset
directory must contain at least one image with a supported extension (`png`,
`jpg`, `jpeg`, `bmp`, `tiff`, or `webp`).

## Direction 1

Direction 1 studies whether one targeted RGB-space perturbation can affect both
the spatial ViT-B/16 detector and the grayscale DCT DenseNet-121 detector.
The attack target is class `0`, interpreted by this local evaluator as `Real`.
The primary frozen configuration is:

```text
Torch bicubic DCT preprocessing
equal ViT/DCT branch weights
per-branch mean-absolute gradient normalization
epsilon = 8/255
step size = 0.5/255
iterations = 40
target = 0 (Real)
```

Run the frozen primary test2 configuration with:

```bash
python evaluate.py --config configs/direction1/test2_full_best.yaml
```

The primary configuration expects the local test2 layout at:

```text
/home/aiattacks/pengyang/Test/group1/test2(whole)/AADD_2026_Test
```

It excludes filenames found in `test1` through `exclude_root`. The resulting
full run contains 1,554 independent images. The DCT65 control can be run with:

```bash
python evaluate.py --config configs/direction1/test2_full_dct65.yaml
```

The control uses ViT/DCT weights `0.35/0.65`, step size `2/255`, and 10
iterations. It is intended to show that increasing the DCT branch weight alone
is not sufficient.

### Direction 1 results

On the 1,554 independent test2 images, the frozen primary configuration
achieved:

```text
ViT flip rate: 1499/1554 = 96.46%
DCT flip rate: 741/1554 = 47.68%
SSIM: 0.9437
LPIPS: 0.1384
local score: 2025.1901
```

The DCT65 control achieved:

```text
ViT flip rate: 1050/1554 = 67.57%
DCT flip rate: 418/1554 = 26.90%
SSIM: 0.9082
LPIPS: 0.1900
local score: 1264.7657
```

These are local evaluator prediction rates. A strict targeted clean-correct
ASR should use only images that were correctly classified as the source class
before attack:

```text
count(clean source prediction and attacked target prediction)
---------------------------------------------------------------
count(clean source prediction)
```

The detailed frozen reports are outside this repository in the shared project
folder:

```text
/home/aiattacks/pengyang/Documents/reports/
```

The relevant report files are named `AADD-2026-Direction1-Final-Frozen-Report`
with `.md`, `.tex`, `-CN.tex`, and `.docx` extensions.

## DCT preprocessing and tests

The DCT attack path is implemented in Torch so that gradients can flow from
the DCT DenseNet back to RGB pixels:

```text
RGB [0, 1]
 -> grayscale = 0.299 R + 0.587 G + 0.114 B
 -> differentiable bicubic resize to 256 x 256
 -> center crop to 128 x 128
 -> orthonormal 2-D DCT-II
 -> log(abs(DCT) + 1e-6)
 -> DenseNet-121-DCT
```

Run the DCT parity and gradient checks directly when `pytest` is unavailable:

```bash
python -m py_compile attacks/dual_pgd.py attacks/dct_pgd.py attacks/vit_pgd.py
python -c "import numpy as np, torch; from scipy.fftpack import dct; from attacks.dual_pgd import torch_dct2; x=np.random.default_rng(0).normal(size=(1,1,128,128)).astype('float32'); y=torch_dct2(torch.from_numpy(x)).numpy()[0,0]; z=dct(dct(x[0,0],axis=0,norm='ortho'),axis=1,norm='ortho'); assert np.max(np.abs(y-z)) < 1e-4; print('DCT parity: ok')"
```

The committed focused tests are:

```text
tests/test_dct_parity.py
```

The measured parity maximum absolute error is approximately `4.61e-05`, with
nonzero finite input gradients.

## Reproducibility and storage

Every experiment should use a separate YAML configuration, output directory,
JSON result, and log. Direction 1 configurations are under:

```text
configs/direction1/
```

Frozen full-run outputs, when present locally, are under:

```text
experiments/direction1/
logs/
```

Do not commit `.venv/`, model weights, input data, generated attack images,
large experiment outputs, credentials, or logs. The checkpoint files are
intentionally ignored or kept local. Record their SHA-256 hashes in reports.

The full Direction 1 experiment used one opportunistically selected NVIDIA L40
and did not terminate or migrate other users' processes.

## Current Validation

The following checks have been completed in the local `.venv`:

- All dependencies in `requirements.txt` install successfully.
- PyTorch `2.3.0+cu121` detects an NVIDIA L40 GPU.
- `vit_b_16.pth` loads into the ViT model and produces two-class output.
- `densenet121_dct.pth` loads into the DCT DenseNet model and produces
  two-class output.
- The model input shapes are `(1, 3, 224, 224)` for ViT and
  `(1, 1, 128, 128)` for DCT DenseNet.

## Branch Workflow

Keep experimental work on a separate branch and leave `main` as the stable
baseline:

```bash
git switch -c feature/joint-vit-dct
# make changes and run the evaluator
git status
git add <intended-files>
git commit -m "Add joint ViT and DCT attack"
git push -u origin feature/joint-vit-dct
```

Do not commit `.venv/`, model weights, test data, generated experiment output,
or credentials. Before opening a pull request, include the config, commit ID,
dataset split, model checkpoint names, attack budget, random seed, runtime,
ASR, SSIM, and LPIPS used for the result.

The historical baseline should be treated as `ViT-only attack with post-hoc
DCT evaluation`. The frozen Direction 1 result is a separate joint ViT+DCT
attack, supported by ViT-only and DCT-only ablations.
