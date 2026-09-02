# AADD attack research workspace

This repository contains two deliberately separated layers:

- the scientific data plane (detectors, attacks, configurations, manifests, evaluation, and validation); and
- the OpenCode control plane (`.opencode/`, campaign orchestration, and the GPU queue).

The canonical server checkout is `/home/aiattacks/oleg/aadd-attack-pipeline`. The research dataset is external and read-only at `/home/aiattacks/dataset/celebA`. Never copy or modify the dataset from an agent session.

The workspace is being hardened incrementally. Never assume that a runner, manifest, validator, checkpoint, or queue component exists merely because a planning document describes it. Run the documented preflight, inspect the actual interface, and report a missing prerequisite as `blocked` instead of silently inventing a parallel pipeline.

## OpenCode roles

- `coordinator` starts and reconciles independent OpenCode sessions and campaign tasks. It does not edit attack code or run GPU programs.
- `attack-worker` handles exactly one named attack in one isolated session and worktree. It loads `attack-execution`, writes the minimal implementation/config/tests, runs permitted CPU checks, and submits an immutable job to `gpuq`.
- `campaign-reviewer` reads verifier-approved evidence and applies frozen decision gates. It does not edit code, launch experiments, or call another agent.

All three are primary agents invoked in separate OpenCode processes. The built-in Task/subagent tool is disabled. Do not emulate concurrency inside a session; the campaign controller owns external session concurrency.

## Non-negotiable rules

1. Use OpenCode `1.18.26` with the exact model ID `naapi/gpt-5.6-luna`. Stop if either differs.
2. Start OpenCode at the canonical repository root. Do not run it from the dataset directory, any legacy checkout, or a parent directory.
3. Never place API keys, tokens, provider URLs containing credentials, SSH material, or complete environment dumps in this repository, prompts, status files, or logs. Provider credentials remain in the user's external OpenCode configuration.
4. Never edit the official evaluator or challenge-controlled semantics. Protected inputs include `evaluate.py`, `AADD-2026/AADD_2026_evaluation.py` when present, dataset files, manifests selected for a frozen run, detector preprocessing, detector checkpoints, metric definitions, and completed run artifacts.
5. Research labels are `0=Real` and `1=Fake`. Under the current dataset layout, `TRAIN/TRAIN_REAL` and `TEST/TEST_REAL` are class 0; `TRAIN/TRAIN_FAKE` and `TEST/TEST_FAKE` are class 1. Labels come from a reviewed manifest, never from filename text.
6. Record every scientifically relevant parameter in an immutable resolved config. Use targeted label `1-y`, the shared differentiable preprocessing and projector in `attacklab/preprocessing.py`, and unique output paths. Never re-derive resize, crop, normalization, the DCT, or the `Linf` projection inline in an attack module; an attack that differentiates through its own copy is not comparable with the ones it is measured against.
7. Do not overwrite data, checkpoints, configs used by a prior run, results, logs, or adversarial trees. A retry gets a new attempt/run identifier.
8. Do not install dependencies, download checkpoints, train models, commit, push, access the network, or touch another worktree from an automated agent session.
9. An agent must never select a GPU, set `CUDA_VISIBLE_DEVICES`, run `nvidia-smi` to race for a card, or invoke an experiment/validator directly. It submits a schema-validated job with `python3 -m ops.gpuq submit <job-spec.json>` and exits. Only the scheduler starts GPU work.
10. `smoke` permits focused tests and a queued experiment for at most 16 images. `development` uses the frozen development manifest. `full` requires explicit user authorization and frozen finalists; never infer authorization from an earlier scope.
11. Generated adversarial trees are evaluated unchanged across target detectors. Required evidence includes clean-correct conditional ASR and denominator, all-sample target rate, SSIM, LPIPS, save/reload perturbation norms, runtime, resolved config, seed, manifest/checkpoint hashes, and official score only where applicable.
12. `status.json` is a machine contract, not prose. A worker may create only a provisional `queued`, `failed`, or `blocked` status. `running`, `passed`, and scheduler failures are reconciled by the controller; `passed` requires the deterministic verifier report. A reviewer never treats smoke success as scientific retention evidence.
13. `PHASE_STATUS.md` is a human-readable ledger. Update it only from verifier-approved artifacts; queued/running work belongs in campaign/queue state, not in the verified-results table.

## Required attack-worker sequence

Before editing, load `attack-execution`, read the selected attack entry in `OPENCODE_LUNA.md`, inspect relevant code/config/tests, and run the permitted Git read-only checks. Then:

1. validate the attack name, scope, prerequisite state, clean worktree boundary, and required manifests/checkpoints;
2. implement only the selected attack through the shared API;
3. add focused tests and immutable configs/job spec;
4. run only permitted CPU/unit checks;
5. enqueue exactly one idempotent GPU job through `gpuq`;
6. write the provisional status contract supplied by `/attack` and stop without polling the GPU indefinitely.

The normal handoff lists changed files, integration/configs, tests, job ID and job-spec path, provisional status path, known blockers, and `git status --short`.
