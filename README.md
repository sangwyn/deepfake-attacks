# Adversarial Attacks on AI Image Detectors

This branch adds a reproducible research pipeline and a restricted OpenCode
control plane around the original AADD evaluator. The original `evaluate.py`
and attack modules remain available; new orchestration lives beside them.

## Separation of concerns

| Purpose | Canonical location on the server | Git policy |
|---|---|---|
| Versioned project and agent instructions | `/home/aiattacks/oleg/aadd-attack-pipeline` | tracked |
| CelebA dataset | `/home/aiattacks/dataset/celebA` | external, read-only |
| Detector and LPIPS weights | `/home/aiattacks/oleg/aadd-attack-assets/weights` | external, hash-verified |
| Heavy run artifacts | `/home/aiattacks/oleg/aadd-attack-runs` | external |
| Compact run ledger | `tracking/` inside the project | tracked |
| GPU queue database and locks | `.gpuq/` inside the project | local, ignored |

Never put API keys in this repository. The project references the existing
OpenCode provider configuration but does not copy it.

## First server setup

```bash
git clone --branch codex/aadd-agent-pipeline \
  https://github.com/sangwyn/deepfake-attacks.git \
  /home/aiattacks/oleg/aadd-attack-pipeline
cd /home/aiattacks/oleg/aadd-attack-pipeline
bash scripts/prepare_server_assets.sh
bash scripts/bootstrap_environment.sh
.venv/bin/python -m attacklab.cli preflight \
  --config configs/pipeline/server.yaml --deep
```

The bootstrap is intentionally explicit and must be run by a human. Agents
must not install packages, download weights, or modify the dataset.

## Reproducible flow

1. Run the deep preflight and resolve every failed check.
2. Build deterministic dataset manifests and commit them.
3. Create an immutable experiment YAML from the supplied template.
4. Submit it to the project GPU queue; never select or kill a GPU directly.
5. The worker writes artifacts to a unique attempt directory.
6. Verify the run. Only verified artifacts may produce a `passed` status.
7. Commit compact configs, manifests, hashes, summaries, statuses, and reviews
   under `tracking/`; do not commit weights, logs, or generated image trees.

Useful entry points:

```bash
.venv/bin/python -m attacklab.cli --help
.venv/bin/python -m ops.gpuq --help
python3 scripts/run_campaign.py --help
/home/aiattacks/.opencode/bin/opencode run --command campaign --dir "$PWD"
```

See `docs/SCIENTIFIC_PIPELINE.md`, `docs/AGENT_PIPELINE.md`, and
`ops/gpuq/README.md` before the first real run.

## Legacy evaluator

- `attacks/` contains one attack module per file.
- `evaluate.py` is the original AADD evaluator.
- `configs/AADD_2026_config.yaml` is retained as a legacy evaluator example.

The legacy evaluator is not itself sufficient evidence for a scientific run:
it does not freeze a sample manifest or write the complete provenance and
per-sample audit required by the new verifier.
