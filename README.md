# Adversarial attacks on deepfake detectors

Reproducible targeted fake-to-real attacks against four image detectors:
ViT-B/16, DenseNet-121-DCT, NPR, and AIDE. The submission uses one evaluator,
strict checkpoint loading, directory-derived labels, and SHA-256-verified TEST
manifests. It is intended for detector robustness research.

## Environment

Use Linux and Python 3.11.11. A CUDA GPU is strongly recommended for the full
four-detector run; install the PyTorch 2.3.0 build appropriate for the host if
the default wheel is unsuitable.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies are PyTorch/torchvision, LPIPS, OpenCLIP (for AIDE),
NumPy/SciPy, Pillow, scikit-image, PyYAML, and tqdm. Exact versions are pinned
in `requirements.txt`; test dependencies are in `requirements-dev.txt`.
LPIPS may download its standard AlexNet weights on first use, so cache them
before moving the environment to an offline machine.

## Data and checkpoints

Place the evaluation snapshot under:

```text
data/TEST/TEST_FAKE/
data/TEST/TEST_REAL/
```

Place the four supplied checkpoints in `weights/` as documented in
`weights/README.md`. Dataset images, checkpoints, and generated outputs are
excluded from Git. The committed manifests cover 100 fake and 100 real images;
labels are accepted only from the two directory names.

## Reproduce

`configs/reproduce.yaml` declares the dataset/manifest paths, source and target
detectors, attack parameters, metric settings, and seed. It reproduces the
selected ISP-prior ensemble PGD candidate at an 8/255 L-infinity budget using
ViT and DenseNet-DCT as sources and all four detectors as targets.

```bash
python evaluate.py --config configs/reproduce.yaml
```

Expected output:

```text
outputs/attacked/TEST_FAKE/*.png  # lossless adversarial images
outputs/results.json              # config/checkpoint hashes, metrics, samples
```

The JSON reports clean accuracy, adversarial accuracy, clean-correct
conditional ASR, SSIM, LPIPS, L2/L-infinity distortion, and the configured
similarity-weighted score. Available attack modules are `identity`, `fgsm`,
`pgd`, `mi_di_fgsm`, `ssa_s2i_fgsm`, `mig_cow`, `frequency_pgd`, and `isp_pgd`.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python evaluate.py --help
```

Prior development and five-direction summaries are retained under `results/`.
They are diagnostic local-detector results, not hidden-detector or official
leaderboard claims. Third-party architecture attribution is in `NOTICE.md`.
