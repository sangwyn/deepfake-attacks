# Detector Asset Status

## Available

The following checkpoint is copied from `/home/aiattacks/detector_weights`:

```text
npr.pth
SHA-256: 3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53
```

The checkpoint is a training checkpoint with top-level fields:

```text
model
optimizer
```

Its model state keys use the `module.` prefix. The verified NPR model
definition, preprocessing contract, class mapping, and differentiable raw-logit
adapter are now in `detectors/npr/`. The adapter removes the prefix, applies
resize-256/center-crop-224 ImageNet preprocessing, exposes logits ordered
`[Real, Fake]`, and has passed strict checkpoint loading, one-image inference,
and finite nonzero input-gradient checks.

## AIDE Checkpoint

The AIDE checkpoint has now been copied from `/home/aiattacks/detector_weights`:

```text
aide.pth
size: 3,592,077,976 bytes
SHA-256: ce3a9d66c124e4c24846a6e513d4c66a7e34a16c46bd46a8041f809c2a4a756e
```

The source and destination hashes were verified to be identical. The official
AIDE model, SRM filters, and local DCT sources are copied under
`detectors/aide/upstream/` from:

```text
https://github.com/shilinyan99/AIDE
```

The checkpoint contains 378 groups of `openclip_convnext_xxl.*` parameters,
including the ConvNeXt-XXLarge visual trunk. A separate ConvNeXt checkpoint is
therefore not required if the embedded trunk is loaded. The compatible
`open_clip` package/model constructor and an AIDE-specific preprocessing and
evaluator adapter are required. The local environment now has
`open-clip-torch==2.24.0` and `openai-clip==1.0.1`. The official AIDE
architecture was constructed without external pretrained weights and loaded
the checkpoint strictly with no missing or unexpected keys. A one-image GPU
forward smoke passed with input shape `[1, 5, 3, 256, 256]` and finite
two-class logits. The adapter is in `detectors/aide/adapter.py`.

Gradient-based AIDE attack integration is not complete: the upstream local DCT
patch selection contains discrete index selection, so a separate differentiable
AIDE preprocessing path is required before using AIDE as an attack source.

## Current AADD-2026 Pair

The existing runnable pair remains:

```text
vit_b_16.pth
densenet121_dct.pth
```

These files are the documented AADD-2026 checkpoints. Model weights and data
are intentionally kept outside Git commits.

## Required Follow-up

Before integrating NPR or AIDE, obtain and record:

- model architecture and source code;
- input size and preprocessing;
- output logits and class mapping;
- checkpoint loading procedure;
- source and version provenance;
- a clean prediction smoke test;
- a finite input-gradient test if the detector will be used as an attack source.
