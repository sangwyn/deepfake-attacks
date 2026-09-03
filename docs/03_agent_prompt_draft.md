# Draft Prompt for Agents (Draft of the prompt we give to agents tomorrow)

> Purpose: a shared prompt template the team uses tomorrow when writing attack code with an AI agent.
> Written against the real team_repo interface (attacks/template.py, attacks/ifgsm.py, evaluate.py),
> so the agent output is directly callable by evaluate.py.
> Date: 2026-09-01. This is a DRAFT to be finalized by the team tomorrow.

---

## A. The prompt (give this to the agent)

```
You are an expert in adversarial machine learning. Implement an adversarial attack
that fools AI-generated-image (deepfake) detectors into classifying a FAKE image as REAL.

## Exact interface (do NOT change the signature)
Create a single file `attacks/<name>.py` with:

    def attack(image, classifiers, device):
        # image  : HWC uint8 numpy array, RGB, values in [0, 255]
        # classifiers : dict; classifiers['vit_b_16']['model'] and
        #               classifiers['densenet121_dct']['model'] are differentiable
        #               torch nn.Module in eval mode on `device`.
        #               Each entry also has ['transform'] (PIL->tensor).
        # device : torch.device
        # RETURN : HWC uint8 numpy array, RGB, [0,255], SAME shape as input
        ...
        return adversarial_image

## Task semantics
- Labels: real = 0, fake = 1. Target class = real (0). This is a TARGETED attack.
- Success = detector outputs real(0) on the adversarial image.

## Hard constraints (must satisfy)
- L-inf perturbation budget: ||x_adv - x||_inf <= 8/255 (work in [0,1] float internally).
- Do NOT change image resolution.
- The final output must round to uint8 [0,255] with the same H,W,C as input.

## Scoring (optimize for this)
- Per detector: score += (0.5*SSIM + 0.5*(1 - LPIPS)) * 1[fooled to real]
- So the attack must (a) fool BOTH vit_b_16 and densenet121_dct, and
  (b) keep SSIM high and LPIPS low (imperceptible).

## Preprocessing you must replicate DIFFERENTIABLY inside the attack
- vit_b_16 : resize to 256, center-crop 224, normalize(mean/std ImageNet).
- densenet121_dct : grayscale -> resize 256 -> center-crop 128 -> 2D DCT (type-II,
  norm='ortho') -> log(|.|+1e-6). You MUST implement a DIFFERENTIABLE DCT so gradients
  flow back to pixels.

## Algorithm (baseline; may be upgraded)
- Targeted iterative FGSM (I-FGSM): x_adv = x_adv - alpha * sign(grad of CE(model(x_adv), real)).
- Ensemble the gradients of both detectors (average) for better joint fooling.
- Suggested: epsilon=8/255, alpha=2/255, iterations=10.
- Optional upgrades for transferability: momentum (MI-FGSM), input diversity (DI-FGSM).

## Requirements
- Code must run as-is when called by evaluate.py (no TODOs, no placeholders).
- Handle both HQ and LQ resolutions robustly.
- Keep everything on `device`; no CPU/GPU mismatch.
- Return uint8 numpy, not tensor.
```

---

## B. Why these points are in the prompt

The prompt covers the 6 points an agent most easily misses:
1. **Precise interface** (numpy HWC uint8, not a tensor) — from template.py
2. **Targeted semantics** (target = real = 0)
3. **Hard constraints** (epsilon <= 8/255, do not change resolution, output uint8)
4. **Score includes LPIPS** (not only SSIM) — this is a 2026 change the agent cannot know on its own
5. **Differentiable DCT** (otherwise gradients do not flow through the DCT model) — the easiest pitfall
6. **Runnable acceptance** (directly callable by evaluate.py)

---

## C. Points to discuss when finalizing with the team

- Should the prompt **specify a concrete algorithm** (restrict to I-FGSM), or **only give the goal and let the agent improvise**? The former is controllable and comparable; the latter may produce stronger solutions.
- Should we require the agent to **also optimize LPIPS** (e.g. a perceptual term in the loss), or only control it passively via the epsilon constraint.
- Ensemble weights: simple average of the two detector gradients, or difficulty-weighted.
- Whether to allow extra dependencies (a latent-space solution needs diffusers, etc.).

---

## D. Note: source of the interface facts
- Input/output format from `attacks/ifgsm.py` (`torch.from_numpy(image).permute(2,0,1)...` and the final `.round().to(uint8).cpu().numpy()`)
- classifiers structure from `evaluate.py` (`classifiers[name]['model'] / ['transform']`)
- Scoring formula from `evaluate.py`: `sim_weight = alpha*ssim + (1-alpha)*(1-lpips)`, alpha=0.5
- DCT preprocessing from `build_dct_transform` in `evaluate.py`
