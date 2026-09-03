# Research experiment plan

This is a standalone campaign roadmap. Do not load it into every OpenCode session. Use `OPENCODE_LUNA.md` and `/attack <name> <scope>` for one attack at a time; return here only to choose the next run and apply decision gates.

> **Precedence.** The attack set was revised on 2026-09-03 to the reviewed
> baseline suite: `fgsm`, `pgd`, `mi-di-fgsm`, `ensemble-mi-eot`, `ssa`,
> `mig-cow`, with `ifgsm` retained only as a pipeline regression.
> `OPENCODE_LUNA.md` and `CAMPAIGN.yaml` are authoritative for which attacks
> exist, their parameters, and their order. The phases below still describe
> `mifgsm`, `di-mi-fgsm`, `ti-di-mi-fgsm`, `frequency-eot`, `prototype`, and
> `dd-fcma`, which are no longer scheduled; read them as methodology, not as a
> task list, and never apply a decision gate to an attack the campaign does not
> run. Unified Latent Optimization and DAELTA are out of scope while the server
> has no frozen diffusion model.

## 1. Goal and scope

Implement, integrate, configure, test, and evaluate selected targeted attacks against the spatial ViT and DCT DenseNet detectors. Shared pipeline components are contracts, not assumptions: each must pass preflight before use. An absent runner, detector adapter, manifest, checkpoint, validator, or storage interface blocks the dependent task and must not be replaced ad hoc inside an attack session.

The campaign covers:

1. FGSM, the existing I-FGSM regression, PGD, and MI-FGSM.
2. DI-MI-FGSM and TI-DI-MI-FGSM.
3. Frequency-EOT and its spatial/frequency/fused comparison.
4. MIG-COW when a valid multi-source evaluation is possible.
5. The target-prototype hypothesis when a disjoint reference manifest exists.
6. DD-FCMA and its required ablations.
7. Seed replication and full-data evaluation only for frozen finalists.

Out of scope: training detectors, changing dataset splits, redesigning the runner, adding unrelated models, tuning on an official or held-out target, and implementing latent/DDIM attacks unless separately requested.

## 2. Fixed experimental contract

- Labels are `0=Real`, `1=Fake`; the targeted research label is `1-y`.
- The research dataset root is read-only `/home/aiattacks/dataset/celebA`. Reviewed manifests map `TRAIN/TRAIN_REAL` and `TEST/TEST_REAL` to class 0, and `TRAIN/TRAIN_FAKE` and `TEST/TEST_FAKE` to class 1. Never derive labels from filenames and never create or change a split from inside an attack task.
- The primary budget is `epsilon=8/255` in RGB pixel space.
- Research evaluation uses both real→fake and fake→real. Official AADD evaluation is fake→real only and must use the unchanged official evaluator.
- Evaluate all four source→target cells: ViT→ViT, ViT→DCT, DCT→ViT, and DCT→DCT.
- Generate an adversarial tree once per dataset, source or source ensemble, direction, attack configuration, and seed. Reuse those exact saved images for every target evaluation.
- Use the frozen, hash-identified clean-correct development manifest, seed 0. Its source split, construction rule, disjointness, sample count, and checkpoint-training relationship must be recorded before the campaign. The current TEST tree contains only 100 examples per class, so a 200-per-class development manifest cannot be silently claimed to come from TEST. Use seeds 1 and 2 only for frozen finalists.
- Fixed-step attacks do not stop when they first succeed; record first-success iteration only as a diagnostic.
- Measure the perturbation after save/reload. It must remain inside the configured `Linf` budget, allowing only the expected uint8 rounding tolerance.
- Never overwrite configs, adversarial images, logs, or result JSON. A rerun gets a new run identifier.
- `OPENCODE_LUNA.md` is authoritative for attack constants and retention thresholds if this roadmap and the compact specification ever diverge.

For source detector `A`, target detector `B`, and direction `s→t`, define the eligible set before generating attacks:

```text
E(A,B,s→t) = {i : y_i=s, D_A(x_i)=s, D_B(x_i)=s}
conditional_ASR = mean[ D_B(x_adv_i)=t for i in E(A,B,s→t) ]
```

Always report `|E|`. Also report the all-sample target rate, but never substitute it for clean-correct conditional ASR.

## 3. Per-attack definition of done

An attack is not complete until all of the following are true:

1. **Implementation:** an attack module uses the existing API and shared differentiable preprocessing. It returns a same-size HWC RGB `uint8` image and does not mutate detector state.
2. **Pipeline integration:** the existing loader/registry can select it by its stable attack name. With the current dynamic loader this normally means `attacks/<name>.py`; update a registry only if the repository already has one.
3. **Config integration:** every tunable parameter comes from the existing attack/config path or is recorded explicitly in an immutable experiment config. Do not leave scientifically relevant values only as undocumented code defaults.
4. **Configs:** add smoke and development configs for every required source, direction, and seed. Add full configs only after finalist selection. Each config fixes the manifest, source, targets, seed, attack parameters, budget, output paths, and metric settings.
5. **Tests:** pass the common checks and all attack-specific checks in `OPENCODE_LUNA.md`.
6. **Smoke run:** complete tests plus at most 16 examples, inspect saved images, and verify the recorded post-save norms.
7. **Scoped run:** complete the requested development or full run without overwriting earlier outputs.
8. **Handoff:** the worker reports code/config/job-spec files, CPU checks, queue job ID, provisional status, failures, and Git status, then exits. The controller updates task state after queue reconciliation; `PHASE_STATUS.md` is updated only from verifier-approved evidence.

If the current runner lacks only the minimal plumbing needed to pass attack parameters, add that narrow plumbing and a focused test. Do not create a second runner or change official evaluation semantics.

## 4. Configuration and result discipline

Use the repository's existing schema and naming convention. If it does not yet prescribe unique config names, use this pattern:

```text
configs/experiments/<attack>/<scope>__src-<source>__dir-<direction>__seed-<seed>.yaml
```

Each research config must resolve, directly or through the existing base-config mechanism, these fields:

- attack name and complete attack parameters;
- input manifest and dataset root (`/home/aiattacks/dataset/celebA` for research runs);
- source detector or source ensemble;
- true source class and target class;
- evaluation target list;
- random seed and deterministic settings;
- output image tree, metrics JSON, and log path;
- fixed preprocessing and evaluation settings.

The official config must additionally preserve the challenge-controlled fields. Never edit `AADD-2026/AADD_2026_evaluation.py`. Keep official-score configs separate from research configs because the official metric assumes success means prediction as Real.

At launch time, copy or serialize the resolved config into the run directory. A result is valid only when its config, code revision or diff, manifest identity, command, start/end time, and output paths are recoverable.

## 5. Phase A — integration contract and regression fixture

**Objective:** verify the existing pipeline contract before multiplying runs.

**Work:**

- Inspect the attack API, differentiable ViT/DCT preprocessing, dynamic loader or registry, runner CLI, config parser, manifests, and result schema.
- Confirm which attack parameters are already configurable and add only minimal missing parameter plumbing.
- Reuse the existing deterministic fixture containing both classes and examples correctly classified by both detectors. If none exists, add only a minimal test fixture.
- Add shared checks for output shape/type, finite gradients, target-loss direction, deterministic byte output, post-save `Linf`, and source/target separation.
- Confirm a generated tree can be evaluated against a second target without regeneration.

**Launch:** run the existing I-FGSM on the fixture and on a smoke subset using its current constants.

**Gate:** do not start new attack modules until I-FGSM can be selected from config, saves budget-compliant images, and produces a reproducible result artifact. Treat the historical I-FGSM score as regression evidence only; do not silently replace it with a differently defined metric.

## 6. Phase B — classical baselines

Implement in dependency order: FGSM → I-FGSM regression → PGD → MI-FGSM.

| Attack | Frozen implementation | Extra acceptance check |
|---|---|---|
| FGSM | One targeted step at `8/255` | Update follows the targeted loss sign and reaches no more than the budget. |
| I-FGSM | `alpha=2/255`, 10 steps, no random start | Preserve current behavior; add regression coverage before refactoring shared code. |
| PGD | I-FGSM settings plus seeded uniform start in the epsilon ball | Initial and every later iterate remain in the ball. |
| MI-FGSM | `mu=1`, gradient divided by mean absolute value before momentum | `mu=0` recovers I-FGSM within tolerance. |

For each attack:

1. Integrate it and write smoke/development configs for both sources and directions.
2. Run focused tests, then all relevant existing tests.
3. Run smoke for all four source→target cells.
4. If smoke passes, run the 200-per-class development manifest with seed 0.
5. Produce one paired comparison table using identical eligible examples.

**Decision:** retain MI for the next phase if it improves cross-model ASR by at least 5 percentage points or the quality-weighted score by at least 3%, without worsening mean SSIM or LPIPS by more than 0.01. Keep the strongest simpler baseline even if MI fails.

## 7. Phase C — input diversity and translation invariance

Use the strongest retained iterative/momentum baseline.

### C1. DI-MI-FGSM

- With probability `0.5`, differentiably resize to a uniformly selected size from 224 through 256 and randomly pad to 256 before detector preprocessing.
- Seed every stochastic choice through the runner's seed path.
- An identity transform must recover MI-FGSM.

Run smoke, then seed-0 development on both directions and all target cells. Retain DI for a mean cross-model ASR gain of at least 5 points whose paired 95% bootstrap confidence interval excludes zero.

### C2. TI-DI-MI-FGSM

- Convolve the input gradient with a unit-sum 15×15 Gaussian kernel with `sigma=3`.
- Preserve tensor shape, device, and dtype.

Run only after freezing the DI comparison. Retain TI for at least 3 further cross-model ASR points or a 3% quality-weighted-score gain.

At the end of Phase C, freeze one strongest spatial transfer baseline and its config. Later hypotheses must compare against this exact run.

## 8. Phase D — frequency-EOT hypothesis

Keep the optimization variable in RGB space. For each step, average `K=5` gradients through seeded transforms

```text
IDCT(DCT(x + xi) * M),  M ~ Uniform(0.5, 1.5),  std(xi)=epsilon.
```

Implement three frozen variants using separately mean-absolute-normalized gradients:

1. spatial only;
2. frequency only;
3. 50/50 spatial-frequency fusion.

Required tests cover DCT/IDCT round-trip, differentiability, seed repeatability, transform bounds, and recovery of each baseline when the other branch is disabled.

**Runs:** smoke all variants; seed-0 development all variants; official fake→real evaluation only for the best development variant. Use the same source images and eligible indices as the frozen spatial baseline.

**Decision:** retain frequency-EOT if it gains at least 5 points on the worse cross-model direction or at least 5% official score, with mean SSIM and LPIPS changing by no more than 0.01. A white-box-only gain rejects the transfer hypothesis.

## 9. Phase E — MIG-COW reproduction

Implement Algorithm 1 from `aadd-2025/3746027.3761986.pdf` using `epsilon=0.02`, 25 steps, `mu=1`, and `beta=0.75`. Profile the number of integrated-gradient points on at most 32 images, record accuracy/runtime/memory, choose the smallest stable value, and freeze it before development runs.

Tests must cover approximate integrated-gradient completeness, consensus recovery, Gram/eigenvector stability, orthogonality, deterministic output, and budget compliance.

A genuine held-out COW claim requires at least three eligible detector sources so generation can exclude the evaluation detector. If only ViT and DCT exist, implement and smoke-test the method and label its two-model results as joint-source or reproduction results, not held-out ensemble transfer. Do not add detectors merely to satisfy this phase.

**Decision:** retain COW only for a median held-out ASR gain of at least 5 points with no held-out target losing more than 2 points. If that evaluation is impossible, do not enable COW in DD-FCMA.

## 10. Phase F — target-prototype hypothesis

Run this phase only when an existing, disjoint target-class reference manifest is available. Compute each target prototype once from frozen pre-logit features and persist its manifest identity; attack examples and official/held-out evaluation examples may not contribute to it.

Compare:

1. classification loss only;
2. cosine feature loss only;
3. their frozen combination.

Run smoke and seed-0 development against the strongest retained baseline. Retain the prototype term for at least 3 cross-model ASR points or a 3% quality-weighted-score gain without quality regression. Reject results that improve only the source model or rely on a non-disjoint reference set.

## 11. Phase G — DD-FCMA candidate and ablations

Build DD-FCMA only from components retained by earlier gates. The default objective is a targeted margin plus optional quality regularization:

```text
L_cls = softplus(z_source - z_target + kappa)
L_q   = 0.5 * (1 - SSIM_diff) + 0.5 * LPIPS
L     = L_cls + lambda_q * L_q
```

Combine separately mean-absolute-normalized spatial and frequency gradients, apply momentum, then take the projected targeted sign update. Include DI, TI, frequency-EOT, COW, and prototype terms only if their corresponding phases retained them. Never include a held-out target in generation or tuning.

Required recovery tests:

- disabling new terms exactly recovers the strongest retained baseline;
- spatial-only and frequency-only recover their frozen branch runs;
- `lambda_q=0` recovers the classification-only objective;
- seeded runs are byte-identical and all saved outputs remain in budget.

Run paired seed-0 development ablations for:

- strongest baseline;
- `+DI`, `+TI`, and `+frequency` in dependency order;
- spatial-only, frequency-only, and fused gradients;
- single-source versus eligible multi-source generation;
- mean consensus versus COW, only if COW passed;
- no quality term, differentiable SSIM, LPIPS, and combined quality term;
- prototype off/on, only if Phase F passed.

Freeze every choice before replication. Report negative ablations; do not silently remove them from the result table.

## 12. Phase H — replication, budget sensitivity, and final launch

Select no more than two finalists using seed-0 development results and the predeclared gates.

1. **Budget sensitivity:** run `epsilon=4/255`, `8/255`, and `16/255` for the strongest baseline and best candidate on the development manifest. Keep steps and all other choices fixed unless the protocol explicitly defines budget-dependent steps.
2. **Seed replication:** run seeds 0, 1, and 2 at the primary budget on the unchanged development manifest.
3. **Full research run:** only after explicit user approval, run the frozen finalists on the existing full manifest for both directions and all source→target cells.
4. **Official run:** evaluate the exact saved fake→real trees with the unchanged AADD evaluator and official config settings. Do not tune after seeing this result.

The final comparison is the frozen DD-FCMA finalist versus the strongest classical/transfer baseline. If DD-FCMA fails, the baseline remains the result; a negative result is complete evidence, not a reason to alter thresholds post hoc.

## 13. Metrics and statistical analysis

For every research run report:

- clean-correct conditional targeted ASR, eligible denominator, and all-sample target rate;
- ViT→ViT, DCT→DCT, ViT→DCT, and DCT→ViT separately;
- mean and worst-direction transfer, plus per-direction paired differences;
- paired bootstrap 95% confidence intervals using fixed resample indices;
- mean and median SSIM and LPIPS, including metrics on successful examples;
- post-save/reload `Linf`, `L2`, and MAE;
- target/source logits or margins and first-success iteration;
- wall time, peak memory when available, forward/backward counts, attack parameters, config path, and seed;
- official AADD score only for runs evaluated by the official procedure.

The research quality-weighted score may be used for declared gates, but label it clearly; do not call it the official score.

## 14. Launch order and stop rules

The resumable campaign controller implements this order and launches a fresh external OpenCode process using the `attack-worker` primary agent for each row. OpenCode's Task/subagent tool is disabled. A worker performs implementation and CPU checks, freezes the config/job spec, submits it to `gpuq`, records provisional state, and exits. The controller does not hold the model session open while a GPU is unavailable.

The durable technical lifecycle is:

```text
planned -> agent_running -> queued -> running -> validating -> passed
                                  \-> failed | blocked | cancelled
passed -> needs_review -> retain | reject | baseline
```

`gpuq` alone allocates a GPU and launches the fixed runner and verifier. The controller reconciles queue state and requires `verifier_report` before `passed`. The read-only `campaign-reviewer` considers only passed development evidence. Neither a worker's prose nor queue process exit alone advances a dependency.

The user-facing commands are:

```text
/campaign development
/campaign status
/campaign resume
/campaign resume --retry <failed-task-id>
/campaign full
```

`/campaign full` is separate explicit authorization and uses the finalist review from the most recent completed development campaign. The declarative graph is in `CAMPAIGN.yaml`; runtime state and per-session logs are under `.campaign/`.

The underlying per-attack order is:

| Order | Invocation | Continue when |
|---:|---|---|
| 1 | `/attack ifgsm smoke` | Regression fixture and integration contract pass. |
| 2 | `/attack fgsm development` | Smoke passes. |
| 3 | `/attack ifgsm development` | Regression remains compatible. |
| 4 | `/attack pgd development` | Smoke and random-start checks pass. |
| 5 | `/attack mifgsm development` | Baselines are paired on identical examples. |
| 6 | `/attack di-mi-fgsm development` | MI or strongest iterative prerequisite is frozen. |
| 7 | `/attack ti-di-mi-fgsm development` | DI comparison is frozen. |
| 8 | `/attack frequency-eot development` | Strongest spatial baseline is frozen. |
| 9 | `/attack mig-cow development` | Multi-source validity and IG profile are recorded. |
| 10 | `/attack prototype development` | Disjoint prototype manifest exists. |
| 11 | `/attack dd-fcma development` | Component decisions are frozen. |
| 12 | `/attack <finalist> full` | User explicitly approves the full run. |

Before each development run, its smoke run must pass deterministic validation. Pause the campaign on NaNs, budget violations, nondeterminism, data leakage, manifest drift, checkpoint drift, source/target preprocessing mismatch, invalid status schema, missing provenance, or unexplained regression. Analyze each phase before starting a more expensive phase. Never launch training, large downloads, full runs, or sweeps without explicit approval.

The canonical server paths and versions are frozen for this campaign:

```text
project:  /home/aiattacks/oleg/aadd-attack-pipeline
dataset:  /home/aiattacks/dataset/celebA
opencode: /home/aiattacks/.opencode/bin/opencode (1.18.26)
model:    naapi/gpt-5.6-terra
```

Mutable `.gpuq/` and live campaign databases are operational state, not hand-edited research records. Git stores the orchestration definitions, frozen configs/job specs/manifests, environment and checkpoint identities, finalized task/review statuses, result summaries, verifier reports, and content hashes needed to reconstruct what happened. Large adversarial image trees remain immutable in their recorded run locations pending an approved artifact-store policy.

## 15. Final deliverables

The completed campaign should contain:

- attack modules integrated through the existing selection path;
- focused unit/integration tests;
- immutable smoke, development, replication, budget, full, and official configs where authorized;
- reusable generated-image trees and machine-readable results;
- a table of every planned run marked completed, failed, blocked, or intentionally skipped with reason;
- paired metric tables and confidence intervals;
- compute/runtime accounting and reproducibility metadata;
- a final baseline-versus-DD-FCMA comparison, including negative ablations and limitations.

Update `PHASE_STATUS.md` only after evidence exists. Smoke success proves engineering viability, not a research hypothesis.
