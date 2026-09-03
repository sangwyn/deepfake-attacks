# Adversarial Attacks on AI Image Detectors (AADD-2026)

Targeted adversarial attack research against AI-generated-image detectors, studying five distinct attack strategies under a unified evaluation protocol.

> **Disclaimer:** This code is for detector red-teaming and robustness analysis. It is not a claim of effectiveness against hidden detectors, real camera pipelines, or an official AADD-2026 leaderboard score.

## Motivation

AI-generated image detectors classify images as Real or Fake. This project asks: **given a Fake image, how small a perturbation is needed to make the detector output Real?** We study this question across two fundamentally different detector representations — RGB spatial (ViT-B/16) and grayscale DCT frequency (DenseNet-121-DCT) — to understand whether detector vulnerabilities are representation-specific or universal.

## Dataset

Unified evaluation uses `celebA/TEST/TEST_FAKE`:

| Split | Images | Format | Size | Role |
|-------|--------|--------|------|------|
| `TEST_FAKE` | 100 | PNG | 1024×1024 | **Five-direction attack set** |
| `TEST_REAL` | 100 | JPEG | 1024×1024 | Clean reference (not attacked) |
| `TRAIN_FAKE` | 1,500 | Mixed | 512–1792 | Not used in main experiments |
| `TRAIN_REAL` | 1,500 | JPEG | 1024×1024 | Not used in main experiments |

Labels come from directory membership (`TEST_FAKE → label 1 / Fake`), not detector predictions.

## Detectors

| Detector | Input | Preprocessing | Backbone |
|----------|-------|---------------|----------|
| ViT-B/16 | RGB 224×224 | resize → crop → ImageNet normalize | Vision Transformer |
| DenseNet-121-DCT | grayscale 128×128 DCT log-mag | grayscale → resize 256 → crop 128 → DCT-II → log(abs+1e-6) | DenseNet-121 |

The two detectors operate in **fundamentally different representation spaces**. A perturbation effective in RGB patch/token space may be invisible in DCT coefficient space, and vice versa.

## Unified Attack Protocol

| Parameter | Value |
|-----------|-------|
| Target class | 0 (Real) |
| $\epsilon$ | $8/255 \approx 0.0314$ |
| Step size | $0.5/255 \approx 0.00196$ |
| Iterations | 40 |
| Random seed | 0 |
| Quality gate | SSIM ≥ 0.94 **and** LPIPS ≤ 0.15 |

All reported rates are **raw post-attack Real rates**, not clean-corrected ASR or official AADD scores.

## Five-Direction Comparison

All five directions were evaluated on the complete 100-image `TEST_FAKE` set. Source: `experiments/five_direction_test_fake_summary.json`.

| Direction | ViT Real | DCT Real | SSIM | LPIPS | Score | Quality Gate |
|-----------|----------|----------|------|-------|-------|-------------|
| **D1**: Joint ViT+DCT PGD | **97%** | 46% | 0.9429 | 0.1592 | 127.93 | SSIM ✓, LPIPS ✗ |
| **D2**: Full-frequency PGD | **97%** | 48% | 0.9430 | 0.1590 | **129.56** | SSIM ✓, LPIPS ✗ |
| **D3**: Universal + residual | 16% | **82%** | 0.9375 | 0.1876 | 85.83* | Both ✗ |
| **D4**: ISP-prior joint PGD | 91% | 51% | **0.9464** | **0.1355** | 128.77 | **Both ✓** |
| **D5**: Adaptive scheduler | 44% | 41% | 0.7385 | 0.3669 | 59.42 | Both ✗ |

\* D3 uses a dedicated workflow with different compute budget; score is reconstructed.

### Key Findings

1. **No direction dominates all dimensions.** D1/D2 are strongest on ViT (97%), D3 is strongest on DCT (82%) but weak cross-representation, D4 has best quality trade-off (only direction passing both quality metrics), D5 performs worst (negative result).

2. **Detector vulnerabilities are representation-specific.** The 51-percentage-point gap between ViT (97%) and DCT (46%) for the same joint attack proves that RGB spatial and DCT frequency detectors have fundamentally different vulnerable directions.

3. **Adaptive scheduling hurts under current design.** D5's four-primitive scheduler (SSIM 0.7385) is drastically worse than fixed spatial PGD alone (SSIM 0.9433 on test1), because the four candidate gradients are too similar in RGB space.

---

## Direction 1: Joint ViT+DCT PGD

**Branch:** `feature/direction1-joint-pgd`

Studies whether one RGB-space perturbation can simultaneously fool both detectors. Computes gradients from both ViT and DCT branches and fuses them in pixel space.

### Implementation

- **Attack:** `attacks/dual_pgd.py` — joint targeted PGD
- **Config:** `configs/direction1/test_fake_full.yaml`
- **Key parameters:** `vit_weight=0.5, dct_weight=0.5, fusion_mode=weighted_sum`

### Weight Scan (100-image TEST_FAKE, intermediate)

| Vit/DCT Weight | ViT | DCT | SSIM | LPIPS | Score |
|---------------|-----|-----|------|-------|-------|
| 0.1 / 0.9 | 94% | **58%** | 0.9100 | 0.2010 | 130.21 |
| 0.2 / 0.8 | 95% | 52% | 0.9120 | 0.1961 | 126.04 |
| 0.3 / 0.7 | 97% | 53% | 0.9139 | 0.1920 | 129.37 |
| 0.5 / 0.5 | **97%** | 48% | **0.9168** | **0.1858** | 125.43 |

Source: `experiments/direction1/joint_scan_summary.json`

### Frozen Full-Run Results

```text
ViT Real:  97/100 = 97.00%
DCT Real:  46/100 = 46.00%
SSIM:      0.9429
LPIPS:     0.1592
Score:     127.9333
```

Source: `experiments/direction1/test_fake_full.json`

---

## Direction 2: Frequency-Band-Constrained PGD

**Branch:** `feature/direction2-frequency-bands`

Extends D1 by constraining the gradient update to specific DCT frequency bands. Tests whether集中扰动预算 on high frequencies improves DCT attack success.

### Implementation

- **Attack:** `attacks/frequency_pgd.py` — frequency-band-constrained joint PGD
- **Config:** `configs/direction2/test_fake_full.yaml`
- **Key parameters:** `mask_mode=full, vit_weight=0.5, dct_weight=0.5`

### Frequency Band Scan (46-image test1)

| Band | ViT | DCT | SSIM | LPIPS |
|------|-----|-----|------|-------|
| low | 91% | 39% | 0.946 | 0.149 |
| mid | 91% | 43% | 0.941 | 0.156 |
| high | 93% | **54%** | 0.931 | 0.170 |
| full | **96%** | 46% | **0.943** | **0.150** |

Source: `experiments/direction2/test1_results_{low,mid,high,full}.json`

### Frozen Full-Run Results

```text
ViT Real:  97/100 = 97.00%
DCT Real:  48/100 = 48.00%
SSIM:      0.9430
LPIPS:     0.1590
Score:     129.5576
```

Source: `experiments/direction2/test_fake_full.json`

### Cross-Detector Transfer (100-image eligible set)

| Source → Target | Target ASR | SSIM | LPIPS |
|----------------|-----------|------|-------|
| ViT → ViT | 75.00% | 0.9234 | 0.1889 |
| DCT → DCT | 29.00% | 0.9602 | 0.0638 |
| ViT → DCT | 1.00% | 0.9234 | 0.1889 |
| DCT → ViT | 0.00% | 0.9600 | 0.0639 |
| Joint → Joint | 73.00% | 0.9095 | 0.2059 |

Source: `experiments/direction2/eligible_*_fake_to_real.json`

---

## Direction 3: Universal + Residual

**Branch:** `feature/direction3-universal-residual`

Learns a reusable "universal" perturbation on a tuning set, then applies it to held-out images with per-image residual optimization.

### Implementation

- **Training:** `workflows/direction3_universal.py` — trains universal component
- **Evaluation:** `workflows/direction3_test_fake.py` — applies to TEST_FAKE
- **Ablation:** `workflows/direction3_ablation.py` — held-out ablation
- **Config:** `configs/direction3/test_fake_full.yaml`

### Training Configuration

```text
Training set:    AADD test2 tuning split (100 images, seed=0)
Validation:      100 images
Held-out:        1,354 images
Universal size:  256×256
Epsilon:         0.031 (~8/255)
Step size:       0.004
Batch size:      4
Epochs:          10
```

### Training Curve

| Epoch | Loss | Val ViT | Val DCT | Val SSIM | Val LPIPS |
|-------|------|---------|---------|----------|-----------|
| 1 | 2.07 | 2% | 3% | 0.964 | 0.111 |
| 5 | 1.45 | 8% | 4% | 0.947 | 0.154 |
| 10 | 1.40 | 10% | 3% | 0.944 | 0.161 |

Source: `experiments/direction3/universal_additive_corrected_seed0.json`

### Held-Out Ablation (100-image subset)

| Mode | ViT | DCT | Joint ASR | SSIM | LPIPS |
|------|-----|-----|-----------|------|-------|
| universal_only | 9% | 9% | 1.04% | 0.947 | 0.154 |
| residual_only (ε=2/255) | 15% | 72% | 10.4% | 0.990 | 0.026 |
| universal+residual (ε=2/255) | **28%** | **72%** | **25.0%** | 0.939 | 0.168 |

Source: `experiments/direction3/ablation_pilot_seed0.json`

### Frozen Full-Run Results (TEST_FAKE)

```text
ViT Real:  16/100 = 16.00%
DCT Real:  82/100 = 82.00%
SSIM:      0.9375
LPIPS:     0.1876
Score:     85.8264*
```

\* Score reconstructed from per-image rows; D3 uses a different workflow and compute budget.

Source: `experiments/direction3/test_fake_full.json`

---

## Direction 4: ISP-Prior Joint PGD

**Branch:** `feature/direction4-isp-prior`

Adds a simplified ISP noise prior to D1's joint PGD. The prior is derived from RGB residual statistics of real vs. fake images.

### Implementation

- **Attack:** `attacks/isp_joint_pgd.py` → `direction4/isp_joint_pgd.py`
- **Config:** `configs/direction4/test_fake_full.yaml`
- **Key parameter:** `w_isp=0.10`

### Dataset Statistics

| Class | Mean Abs Residual | P95 Abs Residual | Channel Covariance |
|-------|-------------------|------------------|--------------------|
| real | 0.0037 | 0.0196 | ~1.0×10⁻⁴ |
| fake | **0.0114** | **0.0588** | ~8.7×10⁻⁴ |

Fake images have **8.6× higher** RGB residual variance than real images. The ISP prior guides perturbation toward real-image statistics.

Source: `experiments/direction4/dataset_stats_standardized_1024.json`

### Frozen Full-Run Results

```text
ViT Real:  91/100 = 91.00%
DCT Real:  51/100 = 51.00%
SSIM:      0.9464
LPIPS:     0.1355
Score:     128.7722
```

**Only direction passing both quality metrics** (SSIM ≥ 0.94 and LPIPS ≤ 0.15).

Source: `experiments/direction4/test_fake_full.json`

---

## Direction 5: Adaptive Scheduler

**Branch:** `feature/direction5-scheduler`

Implements rule-based adaptive scheduling of four attack primitives, inspired by ARMOR++'s resource allocation idea but without VLM/LLM agents.

### Implementation

- **Scheduler:** `experiments/direction5/scheduler.py`
- **Config:** `configs/direction5/test_fake_full.yaml`
- **Entry point:** `attacks/direction5_scheduler.py`

### Four Primitives

| Primitive | Description |
|-----------|-------------|
| `spatial` | Standard PGD/I-FGSM in pixel space |
| `di_fgsm` | DI-FGSM with random resize/pad |
| `low_frequency` | DCT low-frequency constrained gradient |
| `global_noise` | Fixed-seed smoothed global noise direction |

### v1→v2→v3 Iteration Results

| Version | Variant | Images | ViT | DCT | SSIM | LPIPS | Score |
|---------|---------|--------|-----|-----|------|-------|-------|
| v1 | fixed spatial | 46 | **82.6%** | **56.5%** | **0.9433** | **0.1474** | **57.55** |
| v1 | equal mix | 46 | 6.5% | 43.5% | 0.7600 | 0.3147 | 16.82 |
| v1 | adaptive | 46 | 6.5% | 39.1% | 0.7645 | 0.3103 | 15.64 |
| v2 | adaptive | 16 | 18.8% | 43.8% | 0.7538 | 0.3269 | 7.33 |
| v2 | equal | 16 | 0.0% | 31.3% | 0.7381 | 0.3414 | 3.65 |
| v3 | adaptive | 16 | 50.0% | 31.3% | 0.7708 | 0.3123 | 9.59 |
| v3 | equal | 16 | 0.0% | 50.0% | 0.7381 | 0.3418 | 5.72 |

Source: `experiments/direction5/scheduler_{46,equal_46,spatial_46,v2_16,v2_equal_16,v3_16,v3_equal_16}.json`

### Post-Processing Robustness (v1, 46 images)

| Variant | Model | Clean | JPEG75 | Resize75 | Blur05 |
|---------|-------|-------|--------|----------|--------|
| fixed spatial | ViT | 82.6% | 78.3% | 82.6% | 82.6% |
| fixed spatial | DCT | 56.5% | 47.8% | 47.8% | 43.5% |
| adaptive | ViT | 6.5% | 6.5% | 6.5% | 6.5% |
| adaptive | DCT | 39.1% | 28.3% | 39.1% | 30.4% |

### Frozen Full-Run Results (100 images)

```text
ViT Real:  44/100 = 44.00%
DCT Real:  41/100 = 41.00%
SSIM:      0.7385
LPIPS:     0.3669
Score:     59.4220
```

**Negative result.** The scheduler degrades quality and attack success vs. fixed spatial PGD. Root cause: the four candidate gradients have cosine similarity > 0.9 in RGB space, making scheduling equivalent to noisy equal mixing.

Source: `experiments/direction5/test_fake_full.json`

---

## Structure

```
deepfake-attacks/
├── attacks/                  # Attack implementations
│   ├── dual_pgd.py          # D1: joint ViT+DCT PGD
│   ├── frequency_pgd.py     # D2: frequency-band-constrained PGD
│   ├── isp_joint_pgd.py     # D4: ISP-prior joint PGD (redirect)
│   ├── direction5_scheduler.py  # D5: scheduler entry point (redirect)
│   ├── source_pgd.py        # D2: single-source detector PGD
│   ├── vit_pgd.py           # ViT-only PGD (ablation)
│   ├── dct_pgd.py           # DCT-only PGD (ablation)
│   ├── ifgsm.py             # Historical ViT-only I-FGSM baseline
│   └── identity.py          # No-op smoke test
├── attacks/                  # Attack implementations
├── configs/                  # YAML configurations
│   ├── direction1/          # D1 configs (28 files)
│   ├── direction2/          # D2 configs (26 files)
│   ├── direction3/          # D3 configs (1 file)
│   ├── direction4/          # D4 configs (1 file)
│   └── direction5/          # D5 configs (1 file)
├── detectors/               # Detector adapters (NPR, AIDE)
├── direction4/              # D4 ISP implementation
├── experiments/             # Results, scripts, attacked images
│   ├── direction1/          # D1 results + 38 attacked image dirs
│   ├── direction2/          # D2 results + 35 attacked image dirs
│   ├── direction3/          # D3 results + universal checkpoints
│   ├── direction4/          # D4 results + dataset statistics
│   ├── direction5/          # D5 scheduler code + results
│   └── five_direction_*     # Cross-direction analysis
├── tests/                   # Unit tests
├── tools/                   # Utility scripts
├── workflows/               # D3/D5 workflow scripts
├── evaluate.py              # Main evaluation pipeline
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify environment
python evaluate.py --config configs/smoke.yaml

# Run unified five-direction comparison (requires all 5 configs + weights)
python evaluate.py --config configs/direction1/test_fake_full.yaml
python evaluate.py --config configs/direction2/test_fake_full.yaml
python evaluate.py --config configs/direction4/test_fake_full.yaml
python evaluate.py --config configs/direction5/test_fake_full.yaml
# D3 uses dedicated workflow:
python workflows/direction3_test_fake.py
```

## Reproducibility

| Element | Value |
|---------|-------|
| Python | 3.12 |
| PyTorch | 2.3.0+cu121 |
| GPU | NVIDIA L40 |
| Seed | 0 (`torch.manual_seed` + `np.random.seed`) |
| ViT checkpoint | `weights/vit_b_16.pth` (328 MB) |
| DCT checkpoint | `weights/densenet121_dct.pth` (28 MB) |
| AIDE checkpoint | `weights/aide.pth` (3.4 GB) |
| NPR checkpoint | `weights/npr.pth` (17 MB) |

Model weights are not committed to Git. Transfer them separately and record SHA-256 hashes.

## Branch Workflow

| Branch | Direction | Status |
|--------|-----------|--------|
| `main` | Stable baseline | ✅ |
| `feature/direction1-joint-pgd` | D1: Joint ViT+DCT PGD | ✅ Pushed |
| `feature/direction2-frequency-bands` | D2: Frequency-band PGD | ✅ Pushed |
| `feature/direction3-universal-residual` | D3: Universal + residual | ✅ Pushed |
| `feature/direction4-isp-prior` | D4: ISP-prior joint PGD | ✅ Pushed |
| `feature/direction5-scheduler` | D5: Adaptive scheduler | ✅ Pushed |

## License

Internal research use only.
