# Detector checkpoints

Place the supplied checkpoints here with these exact names:

```text
vit_b_16.pth
densenet121_dct.pth
npr.pth
aide.pth
```

Checkpoints are deliberately excluded from Git. The evaluator loads every
state dictionary strictly and records its SHA-256 digest and byte size in the
result JSON.
