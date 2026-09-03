# Five-Direction TEST_FAKE Full Analysis

**Date:** September 3, 2026  
**Dataset:** `/home/aiattacks/dataset/celebA/TEST/TEST_FAKE`  
**Images:** 100  
**Target:** Fake -> Real, local class `0 = Real`  
**Status:** Five direction runs complete

## Scope and comparability

All five directions were evaluated on the complete 100-image `TEST_FAKE` set.
Directions 1, 2, 4, and 5 use the canonical evaluator with ViT-B/16 and
DenseNet-121-DCT. Direction 3 uses its dedicated universal-plus-residual
workflow and a different JSON schema and iteration budget. All runs use the
nominal `epsilon=8/255`, but Direction 3 uses a frozen universal component plus
a `2/255` residual and 10 residual iterations. The comparison is therefore a
full-data diagnostic, not a perfectly compute-matched benchmark.

The reported Real rates are raw post-attack target-hit rates. They are not
clean-corrected ASR, official AADD scores, or hidden-detector transfer rates.

## Aggregate results

| Direction | Images | ViT Real | DCT Real | SSIM | LPIPS | Local score |
|---|---:|---:|---:|---:|---:|---:|
| Direction 1: joint ViT+DCT PGD | 100 | 97.00% | 46.00% | 0.9429 | 0.1592 | 127.9333 |
| Direction 2: full-frequency PGD | 100 | 97.00% | 48.00% | 0.9430 | 0.1590 | 129.5576 |
| Direction 3: universal + residual | 100 | 16.00% | 82.00% | 0.9375 | 0.1876 | 85.8264* |
| Direction 4: ISP-prior joint PGD | 100 | 91.00% | 51.00% | 0.9464 | 0.1355 | 128.7722 |
| Direction 5: adaptive scheduler | 100 | 44.00% | 41.00% | 0.7385 | 0.3669 | 59.4220 |

## Interpretation

### Direction 1

Joint ViT+DCT PGD is tied for the best ViT rate and provides a strong general
baseline. Its LPIPS is slightly above the provisional `0.15` reference.

### Direction 2

Full-frequency PGD is tied for the best ViT rate, slightly improves DCT rate
over Direction 1, and has nearly identical quality. This supports its use as a
frequency-sensitivity baseline, not as proof of hidden-detector transfer.

### Direction 3

Universal-plus-residual is strongly detector-specific: 82% DCT Real but only
16% ViT Real. It gives useful evidence for a frequency-detector-specific prior,
but misses the provisional quality gate in mean metrics.

### Direction 4

The simplified ISP-prior joint attack has the best mean SSIM and LPIPS and
retains 91% ViT and 51% DCT Real rates. It is the strongest quality/attack
trade-off in this batch. This does not establish calibrated camera statistics
or validate a complete RAW/InvISP pipeline.

### Direction 5

The adaptive scheduler is inferior on the full set: 44%/41% ViT/DCT rates,
SSIM `0.7385`, and LPIPS `0.3669`. The 100-image result confirms the earlier
16/46-image diagnosis. The current four proxy primitives and source-only
controller should remain frozen as a negative result.

## Quality-gated view

Using the provisional screening gate SSIM `>=0.94` and LPIPS `<=0.15`:

- Direction 4 passes both mean quality metrics;
- Directions 1 and 2 pass SSIM but miss LPIPS narrowly;
- Direction 3 misses both in mean metrics;
- Direction 5 misses both substantially.

The gate is a project screening rule, not an official challenge rule.

## Limitations

- `TEST_FAKE` has 100 images and no generator-family labels in this analysis.
- Rates are raw post-attack Real rates, not clean-corrected ASR.
- Direction 3 uses a different workflow and compute budget.
- Only ViT-B/16 and DenseNet-DCT are white-box detectors in this comparison.
- No hidden detector, unseen generator, or official AADD score was measured.
- Interrupted Direction 5 expansion attempts are excluded; only complete outputs
  are summarized.

## Decision

Direction 4 is the best quality-constrained follow-up candidate. Directions 1
and 2 remain the strongest ViT and frequency baselines. Direction 3 remains a
detector-specific reusable-prior ablation. Direction 5 remains a negative
methodological result and should not receive further parameter tuning without
implementing genuinely distinct ARMOR++ primitives and matched compute.

## Reproducibility

This directory retains the machine-readable aggregate as `summary.json` and
`summary.csv`. The original per-direction configuration/output files remain
recoverable from the integrated branch history; they were removed from the
submission tree because they duplicated the canonical runner and included
partial or method-specific workflows. `configs/reproduce.yaml` declares the
selected Direction 4 follow-up with all four detectors evaluated as targets.
\* Direction 3 uses a dedicated workflow. Its local score is reconstructed
from per-image rows using the same two-detector similarity-weighted formula as
the canonical evaluator; it was not emitted directly by that workflow.
