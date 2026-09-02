# Verified attack status

This Git-tracked ledger records verifier-approved outcomes; it never authorizes larger runs. Live `planned`, `agent_running`, `queued`, `running`, and `validating` states are read from the campaign controller and `gpuq`, not copied here by an attack worker.

| Attack | Status | Evidence/results |
|---|---|---|
| FGSM | not started | — |
| I-FGSM regression | not started | — |
| PGD | not started | — |
| MI-FGSM | not started | — |
| DI-MI-FGSM | blocked on MI | — |
| TI-DI-MI-FGSM | blocked on DI | — |
| Frequency EOT | blocked on transfer baseline | — |
| MIG-COW | not started | — |
| DD-FCMA | blocked on component decisions | — |
| Prototype | optional | — |

Use `passed`, `failed`, or `blocked` only after controller reconciliation. Every `passed` row must link the frozen config, result summary, evidence, and deterministic verifier report; include the queue job ID in the evidence text. Smoke success is engineering evidence, not a research decision. Record `retain`, `reject`, or `baseline` only after the independent reviewer applies the frozen gates.

Automated campaign and queue state are stored separately under `.campaign/` and `.gpuq/`. They are mutable operational databases and must not be edited by hand. Finalized status/result summaries and their provenance hashes are committed through the integration workflow.

## Control-plane readiness

| Contract | Required value | State |
|---|---|---|
| Canonical checkout | `/home/aiattacks/oleg/aadd-attack-pipeline` | pending server deployment |
| Dataset | read-only `/home/aiattacks/dataset/celebA` | preflight required |
| OpenCode | `1.18.26` | preflight required |
| Model | `naapi/gpt-5.6-luna` | preflight required |
| Agents | `coordinator`, `attack-worker`, `campaign-reviewer` | configured in `.opencode/agents/` |
| GPU execution | `gpuq` only | scheduler validation required |
| Campaign controller | reconciles `queued` through the verifier | implemented; sequential, one worker |
| Shared attack API | differentiable preprocessing plus `Linf` projector | `attacklab/preprocessing.py` |
| Attack coverage | one module per campaign attack | only `ifgsm`; staged per-worker implementation |
| Detector checkpoints | `vit_b_16.pth`, `densenet121_dct.pth` plus SHA-256 | preflight required |
| Environment | reviewed lock/bootstrap plus CUDA smoke | preflight required |

Do not change a `pending ...` entry to ready without a machine-readable preflight artifact.

The controller owns the `queued` to `passed` transition. It polls `gpuq`, reads the deterministic verifier report from the scheduler's attempt directory, and rewrites the task status itself; a `passed` written by an agent is rejected. Reconciliation also runs at the start of `resume` and `status`, so an interrupted controller recovers without resubmitting GPU work. Interrupting the controller never cancels a queued job.

The runner sets `torch.use_deterministic_algorithms(True, warn_only=True)`. Strict mode is incompatible with the CUDA backward of the differentiable resize every attack needs. Seeds, `cudnn.deterministic`, and `CUBLAS_WORKSPACE_CONFIG` still apply, the mode is recorded under `runtime.determinism` in `provenance.json`, and byte-for-byte reproducibility remains an empirical check against the verifier's per-sample output hashes. Confirm it on the server with two identical smoke runs before trusting a replication result.
