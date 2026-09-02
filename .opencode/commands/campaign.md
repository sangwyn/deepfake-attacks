---
description: "Start, resume, or inspect the durable campaign: /campaign <development|resume|status|full> [options]"
agent: coordinator
model: naapi/gpt-5.6-luna
subtask: false
---

Treat the complete user argument string as `$ARGUMENTS` and its first token `$1` as the action. Accept only `development`, `resume`, `status`, or `full`.

Run `python3 scripts/run_campaign.py` with that action and only the remaining options explicitly supplied by the user. For `full`, add `--confirm-full`; invoking `/campaign full` is explicit authorization for the already frozen finalists, not permission to redesign configs or add a sweep. Never add `--auto`, `--new`, `--retry`, `--finalists`, a model, or a variant unless explicitly supplied and accepted by the driver. Never change the exact project model `naapi/gpt-5.6-luna`.

The driver owns external isolated OpenCode sessions, task dependencies, queue reconciliation, verifier gates, retry policy, and durable state. Do not implement an attack in this session, invoke the Task tool, run GPU code, start the scheduler, edit status JSON, or fabricate a transition. A worker returning `queued` is normal: report the campaign state path and queue job ID, then let the controller reconcile it on `status`/`resume`.

For a read-only queue snapshot, only the documented `python3 -m ops.gpuq list --json` and `python3 -m ops.gpuq status <job-id> --json` commands are permitted. Report the exact controller outcome, queued/running work, verifier failures, and next authorized action.
