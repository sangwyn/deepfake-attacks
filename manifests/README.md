# Dataset manifests

Manifests are deterministic JSON Lines inventories. Each row records the
project-independent relative dataset path, explicit class label, byte length,
and SHA-256 digest. The image data itself is never committed.

Generate them on the server only after preflight succeeds:

```bash
.venv/bin/python -m attacklab.cli build-manifests \
  --config configs/pipeline/server.yaml \
  --output-dir manifests/celebA
git add manifests/celebA
git commit -m "data: freeze audited CelebA manifests"
```

Generation refuses to overwrite existing manifests by default. Use a new
directory for a changed dataset snapshot. `--overwrite` is intended only for
an uncommitted local correction that has been reviewed by a human.
