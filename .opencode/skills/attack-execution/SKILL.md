---
name: attack-execution
description: Implement and CPU-test one named adversarial attack, freeze its experiment configuration, and enqueue it through the project GPU queue without selecting or running a GPU directly.
compatibility: OpenCode 1.18.26; project attack-worker only
metadata:
  owner: aadd-attack-pipeline
  model: naapi/gpt-5.6-luna
---

# Attack execution workflow

Use this skill only inside the project `attack-worker` primary agent. Require explicit `ATTACK`, `RUN_SCOPE`, and task identity. Never infer `development` or `full` from conversation history.

Before work, read `AGENTS.md`, the selected row and checks in `OPENCODE_LUNA.md`, `PHASE_STATUS.md`, relevant code/tests/configs, and `references/status-contract.md` next to this file.

## 1. Fail-closed preflight

- Confirm the working directory is the canonical Git worktree for `/home/aiattacks/oleg/aadd-attack-pipeline`, not the dataset or an old clone.
- Confirm OpenCode `1.18.26` and model `naapi/gpt-5.6-luna` were selected by the caller.
- Inspect Git status and preserve unrelated changes.
- Confirm the requested scope is one of `smoke`, `development`, or explicitly authorized `full`.
- Confirm prerequisites, manifest identity, detector checkpoint paths/hashes, shared attack API, projector, differentiable preprocessing, and validator/runner entrypoints exist.
- Treat missing or contradictory infrastructure as `blocked`; do not build a second runner or relax a scientific gate.

## 2. Bounded implementation

Implement only the named attack and the smallest existing registration/config plumbing it needs. Add focused tests. Do not edit another attack, the official evaluator, detector preprocessing/adapters, frozen manifests, checkpoint files, metric definitions, campaign graph, queue implementation, completed runs, or external data.

Every stochastic choice must use the runner-provided RNG/seed. Every scientifically relevant value belongs in the resolved config, not only a code default. Outputs must satisfy the shared RGB pixel-space projector and save/reload audit.

Take gradients through `attacklab/preprocessing.py`, never through a private copy of the preprocessing:

- `from_uint8_image` / `to_uint8_image` convert between the runner's HWC uint8 array and a `1x3xHxW` float tensor in `[0, 1]`;
- `preprocess_for(source_model, x)` dispatches to the differentiable surrogate of the detector's evaluation transform (`vit_b_16` spatial, `densenet121_dct` frequency);
- `project_linf(adversarial, original, epsilon)` re-enters the feasible set after every update.

`attacks/ifgsm.py` is the reference consumer. Read it before writing a new module.

## Staging frozen inputs

The runner starts by checking that the experiment config, server config, manifest, and environment lock are all present in Git, and aborts otherwise. Your config is new, so stage it — together with the attack module, its tests, and the job spec — before you submit:

```bash
git add attacks/<name>.py configs/experiments/<task>.yaml tests/<test file> tracking/jobs/<TASK_ID>/job-spec.json
```

Stage only what this task produced, and never commit: a commit is a human or controller action. An unstaged config does not fail early — it fails after the scheduler has already reserved a GPU.

## 3. Local checks are CPU-only

Run focused unit checks with `PYTHONPATH=. .venv/bin/python tests/<file>.py`. A bare `python3` cannot import numpy or torch, because the dependencies are installed in the project's `.venv`, and `pytest` is not part of the frozen environment. Run only checks allowed by the agent policy. Do not invoke a command that imports the project in a way that initializes CUDA unless that test is explicitly documented as CPU-only. Never set a GPU environment variable, probe GPU availability, run the experiment, run the verifier, install a package, or download an input.

If tests fail because a pinned dependency is missing, return `blocked`. Do not mutate the environment.

## 4. Freeze and enqueue

Create one immutable repository-local job spec for one experiment. It must contain only:

```json
{
  "schema_version": 1,
  "task_kind": "attack-experiment",
  "config_path": "configs/.../resolved.yaml",
  "run_dir": "runs/<unique-run-id>",
  "requested_memory_mb": 24000,
  "timeout_seconds": 7200,
  "priority": 0,
  "max_attempts": 1
}
```

Use values supported by the frozen config and scope. Never add a user-controlled command, shell fragment, environment mapping, GPU index/UUID, credential, or external output path. The queue maps `task_kind` to the fixed runner and validator argv.

Submit exactly once:

```bash
python3 -m ops.gpuq submit tracking/jobs/<task-id>/job-spec.json
```

Record the returned `job_id` and idempotency key. Duplicate submission of identical work must resolve through the queue's idempotency contract, not through a modified config.

## 5. Provisional status and stop

When the command supplies a status destination, create only the provisional schema in `references/status-contract.md`. A successfully submitted worker status is `queued` with `decision=pending`; it is never `passed`. With no submitted job, use `failed` or `blocked` and omit `job_id`.

The controller later reconciles queue state. The scheduler runs the fixed experiment and verifier. Only a valid verifier report permits the controller to produce `passed`; only the independent reviewer may produce a research decision.

Do not wait for a GPU, repeatedly poll the job, edit `PHASE_STATUS.md`, commit, push, or start another task. Finish with a concise handoff containing changed files, tests, config/job spec, job ID, provisional status, and blockers.
