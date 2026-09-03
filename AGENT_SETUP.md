# OpenCode control-plane setup

This checkout is the canonical agent/scientific workspace. On the server it must live at:

```text
/home/aiattacks/oleg/aadd-attack-pipeline
```

The code and versioned control plane live in this Git repository. Mutable scheduler state remains under `.gpuq/`. The dataset remains separate and read-only:

```text
/home/aiattacks/dataset/celebA/
├── TRAIN/TRAIN_REAL
├── TRAIN/TRAIN_FAKE
├── TEST/TEST_REAL
└── TEST/TEST_FAKE
```

Do not launch from a parent directory, a legacy checkout, or the dataset directory. OpenCode discovers project commands, agents, the skill, and `AGENTS.md` from the repository root.

## 1. Required versions and inputs

- OpenCode: exactly `1.18.26` at `/home/aiattacks/.opencode/bin/opencode`.
- Model ID: exactly `naapi/gpt-5.6-terra`.
- Python/CUDA dependencies: install only through the reviewed lock/bootstrap procedure outside an automated agent session; `requirements.lock`, `pyproject.toml`, and `environment.lock.json` are the recorded contract.
- Detector checkpoints: `vit_b_16.pth` and `densenet121_dct.pth`. These are detector weights, not attack weights. Their absolute resolved paths and SHA-256 hashes must be recorded by preflight/run provenance. Agents may not download or modify them.
- Provider credentials: external user configuration only. This repository intentionally contains no API key or provider secret.

Before any agent run:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
/home/aiattacks/.opencode/bin/opencode --version
/home/aiattacks/.opencode/bin/opencode models
python3 scripts/preflight.py --strict
python3 -m ops.gpuq doctor --json
```

The first command must print `1.18.26`; the model listing must contain `naapi/gpt-5.6-terra`. Treat a missing preflight script, manifest, checkpoint, compatible environment, queue, or CUDA runtime as a blocker. Do not repair the environment from inside OpenCode.

## 2. Interactive use

Start with the explicitly named primary agent because the permissive built-in agents are disabled by `opencode.jsonc`:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
/home/aiattacks/.opencode/bin/opencode --agent coordinator --model naapi/gpt-5.6-terra .
```

Use the coordinator for campaign actions:

```text
/campaign status
/campaign development
/campaign resume
/campaign resume --retry <task-id>
/campaign full
```

`/campaign full` is the explicit full-run authorization boundary. Development, resume, or an earlier conversation never implies it.

For a manual single-task preparation, start a fresh process and invoke:

```text
/attack fgsm smoke
```

A manual invocation without a controller-supplied status path can enqueue work and show its provisional JSON, but cannot advance durable campaign state. Prefer the campaign driver for unattended work.

## 3. GPU scheduler

OpenCode agents never run a GPU experiment directly. Start one scheduler process outside OpenCode after `gpuq doctor` passes. Preview mode is the default; real execution requires the explicit `--execute` switch:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
python3 -m ops.gpuq run-scheduler \
  --execute \
  --max-running 1 \
  --poll-seconds 10 \
  --idle-samples 3 \
  --headroom-mb 4096
```

Begin with one concurrent GPU job. Increase the cap only after queue fault-injection tests and agreement with other server users. The scheduler waits for an eligible card, acquires its lease, and launches the fixed task-kind runner; an agent must not choose an index or set `CUDA_VISIBLE_DEVICES`.

Useful read-only queries:

```bash
python3 -m ops.gpuq list --json
python3 -m ops.gpuq status <job-id> --json
python3 -m ops.gpuq doctor --json
```

Queue submission accepts a reviewed repository-local job spec only:

```bash
python3 -m ops.gpuq submit tracking/jobs/<task-id>/job-spec.json
```

Do not put arbitrary commands, environment mappings, secrets, GPU IDs, or external output paths in a job spec.

## 4. Unattended campaign

Run the deterministic driver in `tmux` or a supervised service from the canonical root:

```bash
cd /home/aiattacks/oleg/aadd-attack-pipeline
python3 scripts/run_campaign.py development
```

The driver launches independent OpenCode processes; it does not use the built-in subagent mechanism. Every worker prepares one task, queues it, writes a provisional `queued` status, and exits. The controller then polls `gpuq`, requires the deterministic verifier report, rewrites the status itself, and only then advances dependencies or requests a read-only review.

The controller is currently sequential: it starts one worker, waits for that worker's queue job to reach a terminal state, and only then starts the next task. `control.max_parallel_agents` in `CAMPAIGN.yaml` is the declared target, not present behaviour; running two workers in one mutable checkout is still forbidden, and per-worker Git worktrees are not implemented.

Reconciliation is idempotent and also runs at the start of `resume` and `status`, so a controller stopped mid-poll recovers by rereading the queue. Interrupting the controller never cancels a queued GPU job; use `python3 -m ops.gpuq cancel <job-id>` for that. `--poll-timeout-seconds` bounds how long the controller waits, and gives up on waiting rather than on the job.

Do not pass `--auto`. The project uses explicit allow/deny rules; an unexpected permission request is a configuration error to investigate, not something to auto-approve. Do not override the model or agent in routine runs.

Runtime queue/campaign databases and live logs are mutable operational state and are not committed while processes are running. Git tracks every input needed to reproduce a run and the finalized ledger under `tracking/`: agent/command/skill definitions, frozen configs/job specs/manifests, environment contract, status/result summaries, verifier reports, and artifact hashes. Large adversarial image trees remain in their immutable run directory and must be content-hashed; do not delete them until an artifact-store policy is approved.

## 5. Safe stop and resume

Stopping an OpenCode worker does not cancel its queued GPU job. Inspect the job and request cancellation through the queue:

```bash
python3 -m ops.gpuq status <job-id> --json
python3 -m ops.gpuq cancel <job-id>
```

After restart, run `/campaign status` and then `/campaign resume`. Never resubmit by hand just because a job is waiting; identical specs rely on the queue idempotency key. Never edit SQLite state or a final status file manually.

## Security boundary

OpenCode permissions reduce accidental tool use but are not an operating-system sandbox. In OpenCode 1.18.26, the project uses V1-compatible `permission` rules plus the still-supported `tools` switches to disable Task/subagents and unwanted tools. A worker that can edit code and run tests can execute trusted repository code. For unattended production use, keep each worker in an isolated Git worktree, mount dataset/checkpoints read-only, restrict Unix credentials and writable paths, and let only the scheduler account access GPU devices.
