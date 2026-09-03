# AIDE detector asset

This directory records the upstream source required to interpret
`weights/aide.pth`. The official repository is:

```text
https://github.com/shilinyan99/AIDE
```

Paper: `A Sanity Check for AI-generated Image Detection`, arXiv:2406.19435.

The checkpoint contains the trained AIDE head, SRM/ResNet branches, and
`openclip_convnext_xxl.*` parameters. The upstream model and SRM/DCT sources
are included under `upstream/` for inspection and later adapter work. The
remaining runtime dependency is the compatible `open_clip` package and its
model-construction code; a separate ConvNeXt checkpoint is not required if the
embedded `openclip_convnext_xxl.*` state is loaded. Do not run this checkpoint
through the two-detector AADD evaluator until the AIDE adapter and preprocessing
contract are tested.

The upstream AIDE license is included in `upstream/LICENSE`.
