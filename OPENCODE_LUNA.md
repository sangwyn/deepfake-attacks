# Attack and experiment specification

Invocation: `/attack <name> <smoke|development|full> [status-file] [task-id]`. Implement only that attack through the shared pipeline, add focused CPU tests, freeze its config/job spec, enqueue it through `gpuq`, write a provisional `queued` status when requested, and stop. The worker never launches or waits for the GPU experiment and never updates `PHASE_STATUS.md` from provisional evidence.

## Fixed protocol

- Research dataset: read-only `/home/aiattacks/dataset/celebA`. Reviewed manifests map `TRAIN/TRAIN_REAL` and `TEST/TEST_REAL` to class `0`, and `TRAIN/TRAIN_FAKE` and `TEST/TEST_FAKE` to class `1`. Never infer labels from filenames or build a replacement split during an attack task.
- Primary budget: `epsilon=8/255`, targeted attack. Evaluate primarily fake→real targeted evasion; untargeted attacks on both classes are a secondary experiment. Official AADD is fake→real only. Budget sweep is `2/255`, `4/255`, `8/255`.
- Evaluate ViT→ViT, DCT→DCT, ViT→DCT, and DCT→ViT. Generate once per source/direction/config/seed and reuse across targets.
- `development`: the frozen development manifest is `manifests/celebA/test_fake.jsonl`, sha256 `cee6c571063ef2d7550079d37f78100b1d5feb41d7cf31d39d3684752319d0ff`, 100 images of `TEST/TEST_FAKE`, label 1, seed 0. Every specification already points at it. Use the whole file; do not resample, subset, or build a replacement. If its hash differs from the value recorded in `manifests/celebA/catalog.json`, the dataset changed and the task is blocked.

  Recorded limitation: there is no separate held-out split at this stage, so development results are reported on the same 100 images every attack is compared on. Attack parameters are predeclared in the specification and never chosen from target predictions, which is what keeps this usable, but a held-out claim needs a disjoint manifest that does not yet exist. `full`: finalists only; run development budgets `2/255`, `4/255`, `8/255` at seed 0, development seeds 0–2 at `8/255`, full-manifest seeds 0–2 at `8/255`, and official fake→real evaluation of the saved trees.
- Fixed-step attacks never early-stop; record first-success iteration.
- Do not tune on held-out or official targets.
- Worker completion means pipeline selection/registration, focused tests, immutable configs recording all attack parameters/manifests/seeds/unique output paths, one immutable `attack-experiment` job spec, and successful idempotent queue submission. Technical experiment completion occurs later and requires the deterministic verifier report.

## Execution boundary

- OpenCode worker sessions are CPU/control-plane producers. They must not select a GPU, set CUDA environment variables, call the runner/verifier, or start the scheduler.
- A worker submits only `python3 -m ops.gpuq submit <job-spec.json>` and exits after recording `job_id`.
- The scheduler maps `task_kind=attack-experiment` to fixed runner/verifier argv and owns GPU allocation, leases, retry, timeout, logs, and process cleanup.
- The campaign controller reconciles `queued` and `running` jobs. Only a valid verifier report can produce attack outcome `passed`.
- OpenCode `1.18.26` and model `naapi/gpt-5.6-luna` are frozen; a mismatch is a blocker, not permission to fall back to another model.

## Attack tasks

`specs/attacks/<name>.yaml` is authoritative for every attack: its parameters,
budgets, evaluation protocol, required tests, and retention gate. Read the file
for the attack you were given and implement exactly that. Do not take
parameters from prose anywhere else, including this document.

Check the specification against this checkout before you write code:

```bash
python3 -m attacklab.cli validate-specs --config configs/pipeline/server.yaml
```

A `[BLOCK]` line means the task cannot run as specified; report it as `blocked`
rather than working around it. A `[WARN ]` line is context you must carry into
the handoff, not something to silence.

Two detectors are configured, `vit_b_16` and `densenet121_dct`. Any method the
literature describes as an ensemble therefore has at most two sources, and
leave-one-detector-out leaves exactly one. Report such a run as an
ensemble-of-two white-box variant, never as held-out transfer evidence.

Unified Latent Optimization and DAELTA are out of scope while the server has no
frozen diffusion model. Do not attempt either, and do not substitute an
approximation.

## Required checks

For every attack: targeted loss decreases on a CPU fixture; output shape/type is valid; gradients are finite; seeded unit fixtures repeat byte-for-byte; and existing focused tests still pass. The queued GPU run plus deterministic verifier must additionally prove every update and saved image respects the configured `Linf` budget within uint8 tolerance and that saved experiment outputs repeat as required by the protocol.

Additional checks:

- FGSM with one step at `alpha=epsilon` matches the first PGD step taken from a zero start.
- PGD start stays in the ball; PGD with a zero start and no random init matches I-FGSM.
- MI-DI-FGSM with `mu=0` and the identity input transform matches I-FGSM; the diversity transform preserves shape, device, and dtype.
- SSA DCT/IDCT round-trip is exact; the identity spectrum mask recovers the spatial-only branch; seeded transforms repeat.
- Ensemble EoT with a single transform and one source matches plain MI-FGSM.
- MIG-COW tests IG completeness approximately, consensus recovery, Gram/eigenvector stability, and orthogonality.

## Hypothesis decisions

- Retain PGD over FGSM for any white-box gain; it is a baseline, not a candidate.
- Retain MI-DI-FGSM for ≥5-point mean cross-model ASR gain over PGD with paired 95% CI excluding zero, and no SSIM/LPIPS worsening beyond 0.01.
- Retain SSA for ≥5 points on the worse transfer direction or ≥5% official-score gain with SSIM/LPIPS change ≤0.01. A white-box-only gain fails; the spectral branch must beat the spatial branch at matched compute and distortion.
- Retain ensemble EoT only for improved worst-transform ASR after unseen processing, reported separately from clean ASR.
- Retain COW for ≥5 points median held-out gain with no target losing more than 2 points. With two detectors this gate cannot be satisfied honestly; record the run as descriptive and defer the decision.
- Report negative ablations. Smoke success is engineering evidence, never a research decision.
