# Development run artifacts

Twelve 100-image runs cover FGSM, PGD, MI-DI-FGSM, ensemble MI/EOT, MIG-COW,
and SSA with either ViT or DenseNet-DCT as the white-box source. Every run has
the following compact audit artifacts:

- `summary.json` — aggregate source and transfer results;
- `verification.json` and `norm_audit.json` — protocol and budget checks;
- `per_sample_metrics.jsonl` and `selection.jsonl` — paired sample records;
- `provenance.json` and `resolved_config.yaml` — frozen run provenance.

The corresponding adversarial PNGs remain outside the repository. Paths in
these frozen files record the original server environment and are retained for
provenance, not as portable runtime configuration.
