# GPUQ: conservative single-host GPU queue

GPUQ is a small, dependency-free control layer for shared NVIDIA hosts. It
persists jobs in SQLite, waits for a genuinely idle card, and supervises only
processes it started. It never accepts a command or environment mapping from a
job specification.

This is a cooperative user-space scheduler, not an isolation boundary. It can
avoid GPUs already used by other people, but it cannot reserve a device against
someone who bypasses GPUQ. Slurm with GRES/cgroups remains the production answer
when administrator support is available. GPUQ never signals a foreign process.

## Safety defaults

- `run-scheduler` is preview-only unless the operator passes `--execute`.
- At most one job runs by default (`--max-running 1`).
- A GPU needs three consecutive idle observations by default.
- A GPU is rejected if `nvidia-smi` reports any compute PID.
- Requested memory plus 4096 MiB headroom must be free.
- The GPU is queried again while a cooperative per-UUID `flock` is held,
  immediately before dispatch.
- `CUDA_VISIBLE_DEVICES` contains the selected full `GPU-...` UUID, so the child
  sees only that device as its local `cuda:0`.
- Children run with `shell=False`, a new process session, a timeout, and captured
  logs. Cancellation/timeout signals only that owned child process group.
- After a scheduler crash, recorded active PIDs become `orphaned`; a restarted
  scheduler does not signal or automatically duplicate them.

The singleton scheduler lock and per-GPU locks only coordinate GPUQ instances
that share the same state directory. `nvidia-smi` checks are still mandatory.

## Job schema v1

The machine-readable schema is
[`job-spec.schema.json`](job-spec.schema.json). The only accepted task kind is
`attack-experiment`:

```json
{
  "schema_version": 1,
  "task_kind": "attack-experiment",
  "config_path": "configs/AADD_2026_config.yaml",
  "run_dir": "tracking/runs/example-campaign/ifgsm-smoke",
  "requested_memory_mb": 16384,
  "timeout_seconds": 3600,
  "priority": 0,
  "max_attempts": 1
}
```

`config_path` and `run_dir` are normalized paths inside the canonical project.
`run_dir` must have the form `tracking/runs/<campaign>/<task>`. Absolute paths,
`..`, unknown fields, arbitrary commands, and caller-supplied environment
variables are rejected. Submission hashes the config and normalized spec; an
identical submission returns the existing job ID. Reusing a run directory for a
different spec is rejected.

GPUQ writes two small Git-visible files immediately:

```text
tracking/runs/<campaign>/<task>/job_spec.json
tracking/runs/<campaign>/<task>/gpuq_status.json
```

SQLite, advisory locks, and supervisor logs default to `.gpuq/` and should stay
out of Git. The scientific runner is responsible for resolved config,
provenance, per-sample results, summaries, and validator output under the
tracking run directory. Large generated images should be ignored or stored in
the configured artifact store.

## Fixed execution contract

The queue constructs these argument vectors internally with the scheduler's
Python interpreter and project root as `cwd`:

```text
python -m attacklab.cli run \
  --config <config_path> \
  --run-dir <run_dir>/attempt-0001

python -m attacklab.cli verify \
  --run-dir <run_dir>/attempt-0001
```

The validator must exit zero before the queue records `succeeded`. Runner or
validator errors are technical `failed` outcomes, not scientific rejection.

## CLI

Run from the canonical repository root:

```bash
python3 -m ops.gpuq submit ops/gpuq/examples/job-spec.json
python3 -m ops.gpuq list
python3 -m ops.gpuq status <job-id> --json
python3 -m ops.gpuq cancel <job-id>
python3 -m ops.gpuq doctor --json
python3 -m ops.gpuq run-scheduler
python3 -m ops.gpuq run-scheduler --execute
```

The first scheduler command performs one safe preview and exits. `--execute`
runs continuously; add `--once` for a single dispatch cycle that waits for its
started jobs. Useful policy flags are `--max-running`, `--poll-seconds`,
`--idle-samples`, `--headroom-mb`, and `--max-idle-utilization`.

Use `--project-root` and `--state-dir` only as operator options before the
subcommand:

```bash
python3 -m ops.gpuq \
  --project-root /home/aiattacks/oleg/aadd-attack-pipeline \
  --state-dir /home/aiattacks/.local/state/aadd-gpuq \
  doctor --json
```

## State machine

```text
queued ──► reserving ──► running ──► validating ──► succeeded
  │             │            │             │
  └─► cancelled ├─► failed   ├─► failed    ├─► failed
                ├─► retry_wait             ├─► retry_wait
                ├─► cancelled              ├─► cancelled
                └─► orphaned               └─► orphaned

retry_wait ──► reserving
```

`retry_wait` is used for a clean scheduler shutdown only when another configured
attempt remains. Code/config errors, non-zero exits, timeouts, and failed
validation do not retry automatically. Cancellation of an active job sets a
durable flag; the live scheduler then terminates only its own `Popen` child.

## Operating assumptions

Run one daemon for one state directory, under a dedicated Unix account if
possible. Put the state directory on a local reliable filesystem. Do not place
SQLite on NFS. GPUQ does not kill unexpected competing processes, reset GPUs,
change compute mode, install dependencies, or mutate datasets/checkpoints.
