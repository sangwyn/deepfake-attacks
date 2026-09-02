# Attack and experiment specification

Invocation: `/attack <name> <smoke|development|full> [status-file] [task-id]`. Implement only that attack through the shared pipeline, add focused CPU tests, freeze its config/job spec, enqueue it through `gpuq`, write a provisional `queued` status when requested, and stop. The worker never launches or waits for the GPU experiment and never updates `PHASE_STATUS.md` from provisional evidence.

## Fixed protocol

- Research dataset: read-only `/home/aiattacks/dataset/celebA`. Reviewed manifests map `TRAIN/TRAIN_REAL` and `TEST/TEST_REAL` to class `0`, and `TRAIN/TRAIN_FAKE` and `TEST/TEST_FAKE` to class `1`. Never infer labels from filenames or build a replacement split during an attack task.
- Primary budget: `epsilon=8/255`, targeted attack. Evaluate primarily fake→real targeted evasion; untargeted attacks on both classes are a secondary experiment. Official AADD is fake→real only. Budget sweep is `2/255`, `4/255`, `8/255`.
- Evaluate ViT→ViT, DCT→DCT, ViT→DCT, and DCT→ViT. Generate once per source/direction/config/seed and reuse across targets.
- `development`: the frozen, hash-identified development manifest, seed 0. If that manifest is absent or cannot provide its declared clean-correct denominator without leakage, the task is blocked. `full`: finalists only; run development budgets `2/255`, `4/255`, `8/255` at seed 0, development seeds 0–2 at `8/255`, full-manifest seeds 0–2 at `8/255`, and official fake→real evaluation of the saved trees.
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

Two detectors are available: `vit_b_16` and `densenet121_dct`. Any method described in the literature as an ensemble method therefore has at most two sources, and leave-one-detector-out leaves exactly one. Report such a run as an ensemble-of-two white-box variant, never as held-out transfer evidence.

| Name | Prerequisite | Implementation |
|---|---|---|
| `ifgsm` | none | Pipeline regression only, not a research result. `epsilon=8/255`, `alpha=2/255`, 10 steps, no random start. Preserve the existing behavior exactly; do not modify this module. |
| `fgsm` | none | One targeted step, `epsilon=8/255`. Minimal classical baseline. |
| `pgd` | FGSM | `epsilon=8/255`, `alpha=2/255`, 10 steps, seeded uniform start in `[-epsilon,epsilon]`. Standard strong white-box baseline. |
| `mi-di-fgsm` | PGD | Momentum and input diversity in one method. `mu=1`; normalize each gradient by its mean absolute value before momentum. Apply differentiable resize `224..256` and random pad to 256 with `p=0.5` before detector preprocessing. Classical transferable baseline. |
| `ensemble-mi-eot` | MI-DI-FGSM | MI-FGSM over both detectors as the source ensemble, averaging gradients over `K=5` expectation-over-transformation samples of JPEG quality, resize kernel, and mild blur. Robustness-oriented baseline. With two detectors this is not a held-out result. |
| `ssa` | MI-DI-FGSM | Spectrum simulation, also called S2I-FGSM. Keep the variable in RGB. Average `N=20` gradients from `IDCT(DCT(x+xi)*M)`, `M~U(0.5,1.5)`, `sigma_xi=epsilon`; compare spatial, spectral, and 50/50 normalized fusion. See `modern/2407.20836v6.pdf`. |
| `mig-cow` | validated multi-source setup | From Algorithm 1 in `aadd-2025/3746027.3761986.pdf`: `epsilon=0.02`, 25 steps, `mu=1`, `beta=0.75`. Profile and freeze IG points on at most 32 images. Block the task if a genuine multi-source setup is unavailable rather than reporting a single-source run as ensemble evidence. |

Two methods from the reviewed plan are deliberately out of scope until a frozen diffusion model exists on the server: **Unified Latent Optimization** and **DAELTA**. Neither `diffusers` nor any diffusion checkpoint is installed, and `requirements.lock` is hash-verified by preflight, so adding one is a reviewed environment change, not an attack task. Do not attempt either, and do not substitute an approximation.

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
