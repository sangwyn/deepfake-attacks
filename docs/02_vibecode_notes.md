# Vibecode Study Notes (Learn vibecode in one night)

> Purpose: quickly get hands-on with "vibecoding" in one night, to prepare for writing attack code with an AI agent tomorrow.
> Date: 2026-09-01.

---

## 1. What is vibecoding

**Vibecoding** = describe your intent to an AI coding agent in natural language, let it generate/modify/run the code, while you mainly **describe requirements, review results, and iterate on feedback** instead of hand-writing every line.

- The term was coined by Andrej Karpathy in early 2025; the core idea is "go with the flow and let the AI write most of the code".
- The point is not "don't read the code", but to **shift human effort from typing to "specifying requirements + judging correctness + giving direction"**.

---

## 2. Why our team uses it

Our task (writing adversarial attacks) is essentially: repeatedly trying different algorithms (I-FGSM / MI-FGSM / latent-space / ensemble ...) inside a **fixed interface** `attack(image, classifiers, device)`. This "fixed interface, fast internal iteration" scenario **fits vibecoding very well**:

- Tomorrow everyone uses an agent to write their own attack -> quickly produce multiple versions -> compare scores via `evaluate.py`.
- The better your prompt, the stronger the attack the agent produces -> so today we must **practice writing good prompts** (task 3).

---

## 3. The core vibecoding workflow (the loop to master in one night)

```
(1) describe goal -> (2) agent generates code -> (3) run it and see results -> (4) read errors / check score -> (5) give feedback and fix -> back to (2)
```

Three things to do every round:
1. **Give enough context**: interface signature, input/output format, constraints (epsilon, SSIM/LPIPS), available classifiers.
2. **Make it runnable**: require the agent to produce code that runs directly with `evaluate.py`, no TODOs left.
3. **Verify before trusting**: check the score / compare with the original image; don't just believe the agent saying "done".

---

## 4. Keys to writing a good prompt (make-or-break for vibecoding)

A good prompt = the agent can get it right **without guessing**. Elements:

| Element | Example (for our task) |
|---|---|
| **Role / goal** | "You are an adversarial-attack expert implementing an attack that fools deepfake detectors" |
| **Precise interface** | "Signature `def attack(image, classifiers, device)`; image is HWC uint8 [0,255] numpy RGB, return the same format" |
| **Hard constraints** | "L-inf perturbation <= 8/255; do not change image size" |
| **Available resources** | "classifiers['vit_b_16']['model'] is a differentiable PyTorch model; classifiers also has densenet121_dct" |
| **Algorithm** | "Use targeted I-FGSM, target class real=0, 10 steps, alpha=2/255" |
| **Caveats** | "The DCT model needs a differentiable DCT for gradients; the attack-time preprocessing must be differentiable" |
| **Acceptance** | "The code must be directly callable by evaluate.py and produce a non-zero score" |

**Anti-pattern**: just saying "write me an adversarial attack" -> the agent will guess the format, miss constraints, and may not run.

---

## 5. One-night practice checklist (what I did tonight)

- [x] Understand the vibecoding concept and workflow
- [x] Read and understand team_repo's attack interface `attack(image, classifiers, device)` and the evaluate flow
- [ ] (tomorrow) Use an agent to generate an I-FGSM attack from the prompt and run evaluate
- [ ] Practice the "read error -> feedback -> fix" loop
- [ ] Practice the "check score -> tune algorithm/hyperparameters -> rerun" loop

---

## 6. Practical suggestions for the team

1. **Fix the interface first** (Oleg done): everyone's attack goes into the same signature in `attacks/xxx.py`, so comparisons are fair.
2. **Put the real constraint numbers in the prompt** (epsilon=8/255, score includes LPIPS, only 2 models) — these are what the agent most easily misses.
3. **Have the agent produce "directly runnable" code every time**, verify with evaluate, don't rely on words.
4. **Each person works on their own branch** (I created `mingyao-dev`); compare scores tomorrow.
