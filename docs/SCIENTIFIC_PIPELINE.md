# Scientific pipeline contract

## Purpose and current boundary

The pipeline evaluates targeted image-space attacks against two AI-image
detectors while keeping experiment selection, GPU scheduling, implementation,
execution, and review separate. An attack is swappable only if every other
contract stays fixed: dataset snapshot, labels, clean-correct denominator,
models, weights, preprocessing, perturbation budget, metrics, seeds, and
artifact layout.

The checked-in `evaluate.py` remains the legacy challenge evaluator. The
versioned `attacklab` wrapper adds the evidence needed for research runs. It
does not make the current IFGSM implementation novel; novelty must be assessed
only after reproducible baselines and transfer experiments exist.

## Canonical server layout

```text
/home/aiattacks/oleg/aadd-attack-pipeline/  Git checkout + OpenCode policy
/home/aiattacks/dataset/celebA/            read-only dataset
/home/aiattacks/oleg/aadd-attack-assets/   external weights/cache
/home/aiattacks/oleg/aadd-attack-runs/     generated images and heavy outputs
```

The repository records compact metadata under `tracking/`. Dataset bytes,
weights, generated image trees, SQLite queue state, and full logs are not Git
artifacts. They are referenced by absolute server path and SHA-256.

The following audited CelebA layout is the only default:

| Logical class | Relative directory | Label | Audited count |
|---|---|---:|---:|
| TRAIN_REAL | `TRAIN/TRAIN_REAL` | 0 | 1500 |
| TRAIN_FAKE | `TRAIN/TRAIN_FAKE` | 1 | 1500 |
| TEST_REAL | `TEST/TEST_REAL` | 0 | 100 |
| TEST_FAKE | `TEST/TEST_FAKE` | 1 | 100 |

Legacy dataset storage roots are invalid and must not be reintroduced.
The official AADD test set has no approved canonical location yet; keep
`official_aadd_root: null` until its ownership and immutable snapshot are
confirmed.

## Required external model assets

Two detector checkpoints are required for the current source/target matrix:

| Model | Filename | Required SHA-256 |
|---|---|---|
| RGB ViT | `vit_b_16.pth` | `5e9677d88a7af10791001796eb43d0d060fada3758369814d6d7832934758d81` |
| DCT DenseNet | `densenet121_dct.pth` | `5bbaf5c5c0e296d5e819a0b401198c73ad69c6bbc8f372579de5ee5c11d5e643` |

LPIPS 0.1.4 with the AlexNet backbone is also required. The audited shared
backbone is `/home/aiattacks/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
with SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
The package calibration `weights/v0.1/alex.pth` has SHA-256
`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`.
Preflight validates both; agents may not download a replacement automatically.

## Pinned execution environment

The reference contract is:

- Python 3.12.3;
- PyTorch 2.3.0+cu121;
- torchvision 0.18.0+cu121;
- CUDA wheel runtime 12.1;
- NVIDIA driver at least 525.60; server observation: 550.120;
- all Python packages exactly pinned in `requirements.lock`;
- OpenCode 1.18.26 at `/home/aiattacks/.opencode/bin/opencode`;
- OpenCode model `naapi/gpt-5.6-terra`.

The working environment observed in another user's directory is reference
evidence only. Do not run from or modify it. Create this project's own `.venv`.
The host driver may report CUDA 12.4 while the PyTorch wheel embeds CUDA 12.1;
that is expected and is recorded as two distinct facts.

## Before any experiment

1. Clone the exact branch into the canonical project path.
2. Create the isolated environment with `scripts/bootstrap_environment.sh`.
3. Place or link the two detector files into the configured external weights
   directory without changing their bytes.
4. Verify that the pinned shared AlexNet cache and packaged LPIPS calibration
   are readable and unchanged.
5. Run deep preflight. Do not downgrade a failed check to a warning.
6. Generate the four deterministic CelebA manifests.
7. Review counts, labels, paths, duplicate-content check, and hashes.
8. Commit manifests before creating an experiment config.
9. Confirm the repository commit and config hash used by the run.
10. Start exactly one GPU scheduler for this project. Agents submit jobs; they
    never directly claim GPUs or kill foreign processes.

Commands:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
bash scripts/bootstrap_environment.sh
.venv/bin/python -m attacklab.cli preflight \
  --config configs/pipeline/server.yaml --deep \
  --output tracking/preflight.json
.venv/bin/python -m attacklab.cli build-manifests \
  --config configs/pipeline/server.yaml \
  --output-dir manifests/celebA
```

## Dataset rules

- A manifest, never a directory glob, defines an experiment population.
- Each manifest row includes explicit label, class name, relative path, size,
  and content SHA-256.
- Manifest order is deterministic and is the only allowed smoke subset order.
- The source population for fake-to-real targeted attacks has label 1 and the
  target class is 0.
- Targeted ASR uses source-model clean-correct samples as its denominator by
  default. The pipeline separately reports selected count, eligible count, and
  evaluated count.
- Train and test sets must never be silently mixed.
- A changed byte means a new dataset snapshot and new manifests.
- REAL JPEG versus FAKE PNG can create a file-format shortcut. Report format
  distributions and, as a sensitivity study, compare a lossless decoded-pixel
  pipeline or balanced re-encoding. Never JPEG-save adversarial examples when
  claiming a strict L-infinity budget.

## Attack plug-in contract

An attack is one Python module under `attacks.*` with:

```python
ATTACK_CONTRACT = {
    "version": 1,
    "supported_source_models": ["vit_b_16"],
}

def attack(
    image,
    classifiers,
    device,
    source_model,
    target_class,
    **parameters,
):
    ...
```

Input is an RGB `numpy.uint8` array. Output must have the same shape and dtype.
The function must optimize the configured source model and target class, use
only declared parameters, and return no hidden state. The wrapper rejects an
undeclared source model and unknown function parameters.

To change an attack:

1. add one module under `attacks/`;
2. declare its contract and supported source models;
3. copy an experiment YAML and change only `experiment_id`, `attack`, and
   explicitly intended budget/seed fields;
4. run unit and smoke tests;
5. enqueue it through GPUQ;
6. compare it to frozen baselines on identical manifests and seeds.

The current IFGSM module supports ViT as a source only. A DCT-source experiment
requires an attack implementation whose gradient path includes the exact
grayscale, resize/crop, DCT, and log-scale preprocessing. Merely changing
`source_model` in YAML is rejected.

## Mandatory scientific matrix

Run both white-box and transfer directions where implemented:

| Source used for gradients | Evaluation on ViT | Evaluation on DCT DenseNet |
|---|---:|---:|
| ViT | white-box | transfer |
| DCT DenseNet | transfer | white-box |

For each cell, report:

- clean accuracy on selected samples;
- clean-correct eligible denominator;
- targeted ASR and successes/denominator;
- SSIM and LPIPS distributions, not only means;
- post-save L-infinity maximum and violation count;
- runtime and failure count;
- exact config, seed, code commit, weight hashes, and manifest hash.

Use a staged ladder:

1. `smoke`: 8 deterministic samples, seed 0, validity only;
2. `development`: frozen subset, seed 0, compare all baselines;
3. `replication`: frozen finalists, at least seeds 0, 1, and 2;
4. `budget`: frozen epsilon grid such as 4/255, 8/255, 12/255;
5. `full`: complete held-out fake manifest, frozen finalists only;
6. `official`: only after the official root and rules are frozen.

Do not tune on the full or official set. Select finalists from development
results using a predeclared decision rule, then freeze code and configuration.

## Artifact contract

Every attempt has a unique immutable metadata directory. Retrying creates a
new attempt number. A completed attempt contains:

| File | Meaning |
|---|---|
| `resolved_config.yaml` | exact experiment input |
| `resolved_server_config.yaml` | exact path/environment contract |
| `manifest.snapshot.jsonl` | exact selected population inventory |
| `preflight.json` | checks observed immediately before execution |
| `selection.jsonl` | clean predictions and eligibility for every selected row |
| `per_sample_metrics.jsonl` | predictions, distortion, metrics, output hashes |
| `norm_audit.json` | post-save budget audit |
| `summary.json` | aggregate denominators and results |
| `provenance.json` | Git, packages, CUDA, inputs, and weight hashes |
| `artifacts.json` | compact metadata index and heavy artifact location |
| `verification.json` | deterministic verifier result |

The verifier re-hashes output PNG files, checks row counts and unique sample
IDs, rejects non-finite metrics, and enforces the post-save L-infinity bound.
The one-byte tolerance exists only for converting a real-valued bound to uint8;
it must be reported, not silently expanded.

Only `verification.json.outcome == "passed"` permits an attack task to become
`passed`. Agent confidence, process exit code, or a plausible summary is not
scientific verification.

## Status lifecycle

`schemas/attack-status.schema.json` is authoritative:

```text
pending -> queued -> running -> passed
                          |-> failed
                          |-> blocked
                          |-> cancelled
```

- `queued` requires `job_id` and immutable `job_spec`.
- `passed` requires configs, results, evidence, and `verifier_report`.
- `blocked` means a named prerequisite is unavailable, not that an agent ran
  out of time.
- `failed` means execution or deterministic validation failed.
- existing terminal status files are immutable.

Review status has its own schema. A read-only review agent returns structured
JSON; the controller validates and writes it. Review agents receive no general
filesystem edit permission.

## Baselines and novelty gate

At minimum compare identity, FGSM, IFGSM/PGD, momentum iterative FGSM, and the
proposed method under the same budget, population, and source model. A novelty
claim is considered only if it includes:

- an explicit mechanism not reducible to an undeclared hyperparameter change;
- ablations isolating each new component;
- gains across seeds with uncertainty, not one best run;
- transfer results in both source directions where technically possible;
- perceptual/constraint trade-off curves;
- robustness to format/preprocessing controls;
- honest negative results and failure modes.

Potential directions such as frequency-aware EOT, cross-representation
gradient alignment, or detector-ensemble transfer are hypotheses, not novelty
by themselves. Literature search and source citations are a separate required
research phase before choosing the claim.

## Stop conditions

Do not launch when any of these holds:

- deep preflight fails;
- LPIPS or detector assets are missing or fail their pinned hashes;
- dataset manifest is uncommitted or its hashes changed;
- requested attack does not declare the source model;
- another scheduler owns the project lock;
- no GPU meets all idle samples and free-memory threshold;
- experiment config or job spec contains an absolute executable/command;
- output attempt already contains scientific artifacts;
- code/config/manifest identity cannot be recorded;
- review or status JSON fails its schema.
