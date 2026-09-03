# Mingyao's Notes & Research (docs/)

This folder collects the research notes, planning, and reusable tooling I (Mingyao Duan)
prepared for the AADD 2026 "attacking AI-generated-image detectors" task. All content is in English.

## Contents

| File | What it is |
|---|---|
| [01_adversarial_attack_basics.md](01_adversarial_attack_basics.md) | Shared conceptual foundation: FGSM/I-FGSM/PGD/MI/DI, task semantics, constraints, and the AADD 2026 changes (2 detectors, LPIPS scoring) |
| [02_vibecode_notes.md](02_vibecode_notes.md) | Vibecoding workflow and how to write good prompts for the coding agent |
| [03_agent_prompt_draft.md](03_agent_prompt_draft.md) | Draft prompt to give the agent tomorrow, written against the real team_repo interface |
| [04_my_experiment_plan.md](04_my_experiment_plan.md) | My experiment plan (variables, steps, metrics, risks) for tomorrow's comparison |
| [05_direction_comparison.md](05_direction_comparison.md) | Comparison of 4 technical routes (pixel / latent / integrated-gradient / transfer) with the AADD-2025 top-3 numbers |
| [06_paper_cheatsheet.md](06_paper_cheatsheet.md) | Three-sentence summaries of the challenge paper + top-3 solutions + extra literature |
| [tools/baseline_check.py](tools/baseline_check.py) | Direction-independent tool: measures how detectors classify clean fake images (no attack). Reuses evaluate.py's model/transform logic |

## Important context (for the team)

- **AADD 2026 differs from 2025**: only 2 evaluation detectors (vit_b_16 + densenet121_dct), and the score is `0.5*SSIM + 0.5*(1 - LPIPS)`, not pure SSIM. The 2025 numbers in docs 05/06 still inform the trade-offs but the official 2026 rules take precedence.
- **2025 detector weights cannot be reused for 2026** (empirically confirmed: same architecture, all tensors differ). The 2026 vit_b_16 weights and the 2026 test data are still needed.
- These are **notes and drafts** for tomorrow's team discussion, not final attack implementations. Actual attacks go into `attacks/` per Oleg's template.
