# Campaign status contract

All paths stored in status documents are repository-relative POSIX paths unless a controller-owned destination is explicitly passed on the command line. Status JSON must contain no comments, secrets, model prose, or unrecognized state names.

## Attack task

Required in every state:

```json
{
  "schema_version": 1,
  "task_id": "fgsm-smoke",
  "attack": "fgsm",
  "scope": "smoke",
  "outcome": "queued",
  "decision": "pending",
  "summary": "CPU checks passed; immutable experiment was queued."
}
```

Allowed `outcome` values are `queued`, `running`, `passed`, `failed`, `blocked`, and `cancelled`. Allowed `decision` values are `pending`, `baseline`, `retain`, `reject`, and `not_applicable`.

`queued` additionally requires:

```json
{
  "job_id": "<gpuq job id>",
  "job_spec": "tracking/jobs/fgsm-smoke/job-spec.json",
  "configs": ["configs/.../resolved.yaml"],
  "evidence": ["tracking/jobs/fgsm-smoke/cpu-tests.json"]
}
```

The worker may emit `queued`, `failed`, or `blocked` only. The deterministic controller owns `running`, terminal scheduler reconciliation, and cancellation.

`passed` requires all of the following in addition to common fields:

```json
{
  "job_id": "<gpuq job id>",
  "job_spec": "tracking/jobs/fgsm-smoke/job-spec.json",
  "configs": ["tracking/runs/<run-id>/resolved-config.yaml"],
  "results": ["tracking/runs/<run-id>/summary.json"],
  "evidence": ["tracking/runs/<run-id>/norm-audit.json"],
  "verifier_report": "tracking/runs/<run-id>/validation.json"
}
```

The verifier report must exist, match this job/config/manifest/checkpoint provenance, and declare a valid run. Technical `passed` does not imply scientific `retain`.

## Finalist review

Every review response requires `schema_version`, `task`, `outcome`, and a non-empty `summary`:

```json
{
  "schema_version": 1,
  "task": "select-finalists",
  "outcome": "passed",
  "summary": "Frozen gates applied to verifier-approved development runs.",
  "finalists": ["ssa"],
  "evidence": ["tracking/runs/.../validation.json", "tracking/runs/.../summary.json"]
}
```

A passed review requires one or two unique known finalists and non-empty existing evidence paths. A failed or blocked review omits fields that would otherwise be empty and states why evidence is insufficient. The reviewer returns this object as its only response; the controller validates and atomically persists it.
