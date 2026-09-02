# OpenCode agent and GPU execution pipeline

## Purpose

This document defines the operational contract for automated adversarial-attack research. The goal is to make the attack implementation replaceable while keeping datasets, detector semantics, execution, verification, and scientific decisions stable.

The design separates probabilistic model work from deterministic execution:

```text
campaign controller
        |
        | starts independent OpenCode processes
        v
attack-worker -> frozen config/job spec -> durable gpuq
                                             |
                                      eligible GPU lease
                                             |
                                  fixed runner -> verifier
                                             |
                                      controller reconcile
                                             |
                                  read-only campaign-reviewer
```

An OpenCode response is never sufficient evidence that an experiment passed.

## Canonical locations

```text
PROJECT_ROOT=/home/aiattacks/oleg/aadd-attack-pipeline
DATASET_ROOT=/home/aiattacks/dataset/celebA
OPENCODE_BIN=/home/aiattacks/.opencode/bin/opencode
OPENCODE_VERSION=1.18.26
OPENCODE_MODEL=naapi/gpt-5.6-luna
```

The repository contains the scientific code and agent control plane. The dataset is external and read-only. Detector checkpoints may be stored outside Git, but each run must resolve and hash both expected files:

- `vit_b_16.pth` for the spatial ViT detector;
- `densenet121_dct.pth` for the DCT DenseNet detector.

These are detector checkpoints. The attacks in this campaign do not require trained attack-model weights unless a later, separately approved method explicitly introduces them.

No path may be inferred from the caller's current directory. Preflight resolves paths from the canonical project configuration and rejects an old checkout or missing input.

## Version-controlled and runtime state

Git must record every small artifact required to explain and reproduce a run:

- OpenCode config, agent definitions, commands, and skills;
- campaign graph and frozen scientific protocol;
- environment lock/bootstrap contract;
- reviewed dataset manifests and their hashes;
- detector checkpoint identities and hashes (the large checkpoint bytes need not be in Git);
- attack source, tests, frozen resolved configs, and immutable job specs;
- finalized task/review statuses, result summaries, verifier reports, commands, and provenance/content hashes under `tracking/`.

The following are live mutable state and must not be committed while active:

- `.gpuq/` SQLite database, leases, scheduler heartbeats, and transient events;
- `.campaign/` live controller database/session logs;
- temporary files and process output;
- large adversarial image trees.

Large trees remain immutable at their recorded paths and are represented in Git by a complete content manifest/hash. Until a shared artifact store or Git LFS/DVC policy is approved, they must not be deleted after a reported run. A final status referencing an unhashable or missing tree is invalid.

## OpenCode configuration compatibility

`opencode.jsonc` targets stable OpenCode `1.18.26`, not OpenCode V2. It uses V1 `permission` rules for edit/bash/web/external access and the deprecated-but-supported V1 `tools` switches to turn off Task/subagents and other tools that V1 cannot express granularly in `permission`.

The root policy is fail-closed and built-in primary agents are disabled. Start with an explicit project primary agent. Every project agent also pins `naapi/gpt-5.6-luna` and disables `task`; concurrency is implemented by separate OS/OpenCode processes, not nested model calls.

OpenCode permissions are not an OS sandbox. In V1, `edit: allow` is not reliably restrictable to a safe subset of repository paths. Therefore the attack worker also requires:

- a dedicated Git worktree and task branch;
- read-only dataset/checkpoint mounts or filesystem permissions;
- no SSH/push/package-manager credentials in the worker environment;
- protected-file hash checks in the deterministic verifier;
- a narrow bash allowlist and no general Python command;
- integration by a separate controller/human gate.

## Project agents

### `coordinator`

Mode: primary, one durable campaign context.

Allowed responsibilities:

- invoke `scripts/run_campaign.py`;
- start fresh external workers through the controller;
- read campaign state and query `gpuq list/status/doctor`;
- reconcile queue/verifier state through deterministic code;
- request finalist review after development evidence is complete.

Forbidden:

- editing research code or state JSON;
- implementing an attack;
- running GPU code, a validator, or the scheduler daemon;
- spawning a built-in subagent;
- changing the model, scientific gates, or full authorization.

### `attack-worker`

Mode: primary, one new process/worktree per task.

Allowed responsibilities:

- load `attack-execution`;
- implement one attack through the shared interface;
- add its focused tests/config/job spec;
- run permitted CPU/unit checks;
- submit one idempotent `attack-experiment` job;
- emit provisional `queued`, `failed`, or `blocked` status.

Forbidden:

- selecting/probing/reserving a GPU;
- setting `CUDA_VISIBLE_DEVICES`;
- directly invoking the attack runner or verifier;
- waiting until a GPU becomes free;
- changing protected components or another attack;
- installing/downloading/training/committing/pushing;
- invoking another agent.

### `campaign-reviewer`

Mode: primary, read-only and independently invoked.

It reads only verifier-approved repository evidence, applies fixed gates, and returns a single review JSON object. It cannot edit even its destination status file. The controller validates the response and persists it atomically. It cannot run code or access the network.

## External session model

The campaign controller starts a new process for each attack task. The conceptual argv is:

```bash
/home/aiattacks/.opencode/bin/opencode run \
  --command attack \
  --dir /home/aiattacks/oleg/aadd-attack-pipeline \
  --model naapi/gpt-5.6-luna \
  --title <campaign-id>:<task-id> \
  <attack> <scope> <status-file> <task-id>
```

The command itself selects `attack-worker`. Do not add `--auto`: the project has no intentional `ask` path for unattended work, so a permission prompt signals a policy/config mismatch.

Parallelism and GPU capacity are separate limits. Start with:

```text
max external OpenCode workers: 1 (enforced: the controller is sequential)
max running GPU jobs:          1
max running jobs per campaign: 1
```

Two concurrent workers are the eventual target, not present behaviour: the controller starts one worker, waits for its queue job, and only then starts the next task. Raise the worker limit only after per-worker Git worktrees, deterministic concurrency, and scheduler recovery tests exist. Two workers must never edit the same worktree or overlapping attack/core files.

## GPU queue contract

The only submission interface available to an attack worker is:

```bash
python3 -m ops.gpuq submit tracking/jobs/<task-id>/job-spec.json
```

The canonical schema is:

```json
{
  "schema_version": 1,
  "task_kind": "attack-experiment",
  "config_path": "configs/experiments/mifgsm/smoke.yaml",
  "run_dir": "runs/mifgsm-smoke-<unique-id>",
  "requested_memory_mb": 24000,
  "timeout_seconds": 7200,
  "priority": 0,
  "max_attempts": 1
}
```

There is deliberately no arbitrary command or environment field. The scheduler maps `attack-experiment` to fixed argv:

```text
python -m attacklab.cli run --config <validated-config> --run-dir <attempt-dir>
python -m attacklab.cli verify --run-dir <attempt-dir>
```

The queue returns `job_id`, `idempotency_key`, and creation/reuse information. The same canonical spec must not create duplicate scientific work.

The scheduler starts in preview mode. Actual execution requires the operator's explicit `run-scheduler --execute`. Initial resource policy is one exclusive job per GPU, three consecutive idle samples, 4 GiB headroom, and one concurrently running GPU job. `gpuq` must never terminate a foreign process.

Useful interfaces:

```text
python3 -m ops.gpuq submit JOB_SPEC.json
python3 -m ops.gpuq list [--state STATE] [--json]
python3 -m ops.gpuq status JOB_ID [--json]
python3 -m ops.gpuq cancel JOB_ID
python3 -m ops.gpuq doctor [--json]
python3 -m ops.gpuq run-scheduler [--execute] [--once] [resource options]
```

This user-space queue can observe and avoid GPUs already used by other people, then take a newly eligible card. It cannot provide a race-free reservation against users who bypass the queue. Strict multi-user isolation requires all users to adopt one queue or an administrator to deploy Slurm/GRES/cgroups (or an equivalent root-managed broker).

## Attack status contract

Schema version remains `1`. Every status requires:

```json
{
  "schema_version": 1,
  "task_id": "mifgsm-smoke",
  "attack": "mifgsm",
  "scope": "smoke",
  "outcome": "queued",
  "decision": "pending",
  "summary": "CPU checks passed; immutable experiment was queued."
}
```

Allowed technical outcomes:

```text
queued | running | passed | failed | blocked | cancelled
```

Allowed scientific decisions:

```text
pending | baseline | retain | reject | not_applicable
```

`queued` requires `job_id` and `job_spec`. The worker also records `configs`; `results` and `verifier_report` are omitted until execution. Only the controller may reconcile later states.

`passed` requires non-empty `configs`, `results`, and `evidence`, and a valid `verifier_report`. A clean process exit without verifier approval is `failed`, not `passed`. A correct experiment that loses its scientific comparison is technically `passed` with decision `reject` after review.

## Review status contract

The reviewer returns exactly one object:

```json
{
  "schema_version": 1,
  "task": "select-finalists",
  "outcome": "passed",
  "summary": "Frozen gates applied to valid development evidence.",
  "finalists": ["mifgsm"],
  "evidence": ["tracking/runs/.../validation.json", "tracking/runs/.../summary.json"]
}
```

Required in all states: `schema_version`, `task`, `outcome`, and non-empty `summary`. `passed` additionally requires one or two unique known `finalists` and non-empty existing `evidence`. The controller, not the reviewer, writes the status file.

## Durable lifecycle and ownership

```text
planned
  -> agent_running       controller
  -> queued              attack-worker after gpuq submit
  -> running             controller reconciles scheduler
  -> validating          scheduler runs fixed verifier
  -> passed              controller accepts verifier report
  -> needs_review        controller after development set completes
  -> retain/reject       campaign-reviewer decision persisted by controller
```

Terminal alternatives are `failed`, `blocked`, and `cancelled`. A scheduler restart reconciles existing jobs by job ID/idempotency key; it must not resubmit them. A GPU collision returns our job to the queue without killing the other user's process. Application/config/verifier failures do not silently retry as a new scientific run.

## Reproducibility gates

Before a task can pass, deterministic code checks at minimum:

- status/result schema and expected sample count;
- Git revision and declared dirty-diff policy;
- resolved config, environment lock, manifest, and checkpoint hashes;
- protected-file hashes;
- source/target detector and preprocessing identities;
- output filenames, shapes, dtype/range, missing/duplicate files;
- post-save/reload `Linf` budget and requested quality metrics;
- seed/determinism metadata and complete per-sample results;
- runner/validator exit codes and queue attempt identity.

The reviewer never repairs a failed gate. The controller never treats model-generated text as verifier evidence.

## Deployment acceptance checklist

The first real smoke job is blocked until all are true:

1. The server checkout is exactly `/home/aiattacks/oleg/aadd-attack-pipeline` on the intended feature branch/commit.
2. `opencode --version` is exactly `1.18.26` and `opencode models` contains `naapi/gpt-5.6-luna`.
3. OpenCode discovers the three project agents, three commands, and `attack-execution` skill from this root.
4. A permission test proves `task`, network, package installation, direct runner/GPU commands, external writes, commit, and push are denied.
5. The clean project environment is reproducibly created from the reviewed lock and passes CUDA/detector load checks.
6. Both detector checkpoint paths and hashes are resolved; missing `vit_b_16.pth` is a blocker.
7. Dataset audit and disjoint frozen manifests exist; no task assumes 200 TEST samples per class when only 100 exist.
8. `gpuq doctor`, queue idempotency/concurrency/restart tests, and preview scheduling pass.
9. The controller accepts `queued`, reconciles it without keeping an OpenCode worker alive, and requires a valid verifier report for `passed`. The worker exits at submit time; the controller, not the agent, waits on the queue. Reconciliation is sequential today: one worker at a time, no per-worker worktrees.
10. One I-FGSM smoke task completes end-to-end before any automated development campaign begins.

If any item fails, record the exact missing prerequisite and stop at `blocked`; do not weaken the contract to obtain a green status.
