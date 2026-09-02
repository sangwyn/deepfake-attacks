# Attack and experiment specification

Invocation: `/attack <name> <smoke|development|full> [status-file] [task-id]`. Implement only that attack through the shared pipeline, add focused CPU tests, freeze its config/job spec, enqueue it through `gpuq`, write a provisional `queued` status when requested, and stop. The worker never launches or waits for the GPU experiment and never updates `PHASE_STATUS.md` from provisional evidence.

## Fixed protocol

- Research dataset: read-only `/home/aiattacks/dataset/celebA`. Reviewed manifests map `TRAIN/TRAIN_REAL` and `TEST/TEST_REAL` to class `0`, and `TRAIN/TRAIN_FAKE` and `TEST/TEST_FAKE` to class `1`. Never infer labels from filenames or build a replacement split during an attack task.
- Primary budget: `epsilon=8/255`, targeted attack, both real→fake and fake→real on research data. Official AADD is fake→real only.
- Evaluate ViT→ViT, DCT→DCT, ViT→DCT, and DCT→ViT. Generate once per source/direction/config/seed and reuse across targets.
- `development`: the frozen, hash-identified development manifest, seed 0. If that manifest is absent or cannot provide its declared clean-correct denominator without leakage, the task is blocked. `full`: finalists only; run development budgets `4/255`, `8/255`, `16/255` at seed 0, development seeds 0–2 at `8/255`, full-manifest seeds 0–2 at `8/255`, and official fake→real evaluation of the saved trees.
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

| Name | Prerequisite | Implementation |
|---|---|---|
| `fgsm` | none | One targeted step, `epsilon=8/255`. |
| `ifgsm` | none | `epsilon=8/255`, `alpha=2/255`, 10 steps, no random start. Preserve the existing I-FGSM behavior. |
| `pgd` | I-FGSM | Same budget/steps, seeded uniform start in `[-epsilon,epsilon]`. |
| `mifgsm` | I-FGSM | `mu=1`; normalize each gradient by mean absolute value before momentum. |
| `di-mi-fgsm` | MI-FGSM | Apply differentiable resize `224..256` and random pad to 256 with `p=0.5` before detector preprocessing. |
| `ti-di-mi-fgsm` | DI-MI-FGSM | Convolve gradients with a unit-sum `15x15` Gaussian, `sigma=3`. |
| `frequency-eot` | best transfer baseline | Keep the variable in RGB. Average `K=5` gradients from `IDCT(DCT(x+xi)*M)`, `M~U(0.5,1.5)`, `sigma_xi=epsilon`; compare spatial, frequency, and 50/50 normalized fusion. See `modern/2407.20836v6.pdf`. |
| `mig-cow` | validated multi-source setup | From Algorithm 1 in `aadd-2025/3746027.3761986.pdf`: `epsilon=0.02`, 25 steps, `mu=1`, `beta=0.75`. Profile and freeze IG points on at most 32 images. |
| `dd-fcma` | retained DI/TI/frequency components | Combine separately normalized spatial and frequency gradients, momentum, targeted margin, and configurable `0.5*(1-SSIM)+0.5*LPIPS` regularization. Use COW only if MIG-COW improved held-out transfer. |
| `prototype` | explicit request and disjoint reference manifest | Add cosine distance to a frozen target-class pre-logit prototype; compare classification-only, feature-only, and combined losses without held-out tuning. |

## Required checks

For every attack: targeted loss decreases on a CPU fixture; output shape/type is valid; gradients are finite; seeded unit fixtures repeat byte-for-byte; and existing focused tests still pass. The queued GPU run plus deterministic verifier must additionally prove every update and saved image respects the configured `Linf` budget within uint8 tolerance and that saved experiment outputs repeat as required by the protocol.

Additional checks:

- PGD start stays in the ball; MI with `mu=0` matches I-FGSM.
- DI identity matches MI; TI kernel preserves shape/device/dtype and sums to one.
- Frequency DCT/IDCT round-trip and seeded transforms pass; disabling a branch recovers its baseline.
- MIG-COW tests IG completeness approximately, consensus recovery, Gram/eigenvector stability, and orthogonality.
- DD-FCMA recovers the strongest retained baseline when new terms are disabled and never uses a held-out target during generation/tuning.

## Hypothesis decisions

- Retain MI for ≥5 percentage-point cross-model ASR gain or ≥3% quality-weighted-score gain without SSIM/LPIPS worsening by more than 0.01.
- Retain DI for ≥5-point mean transfer gain with paired 95% CI excluding zero; retain TI for an additional ≥3 points or ≥3% score gain.
- Retain frequency EOT for ≥5 points on the worse transfer direction or ≥5% official-score gain with SSIM/LPIPS change ≤0.01; white-box-only gain fails.
- Retain COW for ≥5 points median held-out gain with no target losing >2 points. Retain prototypes for ≥3 points or ≥3% score gain without quality regression.
- Evaluate the frozen DD-FCMA finalist against the strongest baseline with seeds 0–2; report negative ablations.
