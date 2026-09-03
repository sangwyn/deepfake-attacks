import torch
import torch.nn.functional as F


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    objective="targeted_fake_to_real",
    label=None,
    seed=0,
):
    if not classifiers:
        raise ValueError("FGSM requires at least one source detector")
    if not isinstance(seed, int):
        raise TypeError("FGSM seed must be an integer")

    # The campaign passes a seed to every attack. FGSM has no random operations.

    attacked = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    attacked = attacked.to(device=device, dtype=torch.float32).div_(255.0)
    attacked.requires_grad_(True)

    losses = []
    for pack in classifiers.values():
        logits = pack["adapter"](attacked)
        if objective == "targeted_fake_to_real":
            target = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits, target))
        elif objective == "untargeted":
            if label not in {0, 1}:
                raise ValueError("Untargeted FGSM requires label 0 or 1")
            true_label = torch.full(
                (logits.shape[0],), label, dtype=torch.long, device=device
            )
            losses.append(-F.cross_entropy(logits, true_label))
        else:
            raise ValueError(f"Unsupported FGSM objective: {objective}")

    loss = torch.stack(losses).mean()
    gradient = torch.autograd.grad(loss, attacked)[0]
    gradient.sign_()
    with torch.no_grad():
        attacked.add_(gradient, alpha=-float(epsilon)).clamp_(0.0, 1.0)
    return (
        attacked[0].detach().permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    )
