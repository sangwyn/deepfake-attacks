---
description: "Prepare and enqueue one attack: /attack <name> <smoke|development|full> [status-file] [task-id]"
agent: attack-worker
model: naapi/gpt-5.6-luna
subtask: false
---

Treat `$1` as `ATTACK`, `$2` as `RUN_SCOPE`, optional `$3` as the controller-owned `STATUS_FILE`, optional `$4` as `TASK_ID`, and optional `$5` as `SOURCE_MODEL` — the detector the gradient is taken from. When `$5` is absent, use the source the specification names; if it names `each-valid-single-source`, this is a manual invocation and you must pick one explicitly and say which. The campaign controller always supplies both `$3` and `$4`. When `$4` is absent, this is a manual invocation: use `<ATTACK>-<RUN_SCOPE>` as the task ID.

Freeze `SOURCE_MODEL` into the experiment config as `attack.source_model`; the attack must declare it in `ATTACK_CONTRACT`. Write `TASK_ID` verbatim into the status document. The controller rejects a status whose `task_id` does not match the task it dispatched.

Reject an empty/unknown attack or any scope other than `smoke`, `development`, or `full`. A `full` request is valid only when this invocation itself is the explicitly authorized full task. Load `attack-execution`, follow `AGENTS.md` and the matching entry in `OPENCODE_LUNA.md`, then handle only this task.

Your work ends at queue submission:

1. Check prerequisites and the actual shared interfaces.
2. Implement only the minimal attack/config/test changes.
3. Run permitted CPU/unit checks; never run the experiment or verifier directly.
4. Freeze one JSON job spec at `tracking/jobs/<TASK_ID>/job-spec.json` with exactly the gpuq contract: `schema_version: 1`, `task_kind: "attack-experiment"`, `config_path`, unique `run_dir`, positive `requested_memory_mb`, positive `timeout_seconds`, `priority: 0`, and positive `max_attempts`. Do not add a shell command, environment override, GPU index, or secret.
5. Stage every scientific input you froze, before submitting: `git add attacks/<name>.py configs/experiments/<task>.yaml tests/<test file> tracking/jobs/<TASK_ID>/job-spec.json`. The runner refuses to start on an input that is not in Git, so an unstaged config fails the experiment after the GPU has already been allocated. Stage only the files this task produced. Never commit.
6. Submit it once with `python3 -m ops.gpuq submit <job-spec.json>`. Do not select a GPU or wait for completion.
7. If `STATUS_FILE` was supplied, create a provisional status JSON only after submission. Otherwise return the same JSON in the handoff and state that a manual invocation cannot advance campaign state.

The canonical attack status uses `schema_version: 1` and requires these fields:

- `task_id`, `attack`, and `scope`, exactly matching the invocation;
- `outcome`: one of `queued`, `running`, `passed`, `failed`, `blocked`, or `cancelled`;
- `decision`: one of `pending`, `baseline`, `retain`, `reject`, or `not_applicable`;
- non-empty `summary`.

For a worker-created `queued` status, also require non-empty `job_id` and repository-relative `job_spec`; set `decision` to `pending`, include non-empty `configs`, include CPU test artifacts in `evidence` when available, and omit `results` and `verifier_report`. If no job was submitted, write only `failed` or `blocked`, omit `job_id`, and explain the reason.

Only the deterministic controller may reconcile the status to `running`, `passed`, `failed`, or `cancelled`. It polls the queue after you exit, reads the verifier report, and overwrites `STATUS_FILE` with its own document, keeping yours beside it as `status.worker.json`. A final `passed` status additionally requires non-empty existing `configs`, `results`, and `evidence`, plus a valid `verifier_report`. Never claim `passed` yourself: the controller rejects a worker status carrying a controller-owned outcome, which fails the task. Never update `PHASE_STATUS.md` from a queued result.

Finish with changed files, tests, config/job-spec path, job ID, status path, blockers, and Git status, then stop.
