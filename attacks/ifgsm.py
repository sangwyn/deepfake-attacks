import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def attack(image, classifiers, device, epsilon=8 / 255,
           step_size=2 / 255, iterations=10):
    model = classifiers['vit_b_16']['model']
    original = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    original = original.float().to(device) / 255
    attacked = original.clone()
    target = torch.tensor([0], device=device)

    for _ in range(iterations):
        attacked.requires_grad_()
        model_input = TF.resize(attacked, [256, 256], antialias=True)
        model_input = TF.center_crop(model_input, [224, 224])
        model_input = TF.normalize(
            model_input,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        loss = F.cross_entropy(model(model_input), target)
        gradient = torch.autograd.grad(loss, attacked)[0]

        attacked = attacked - step_size * gradient.sign()
        perturbation = torch.clamp(attacked - original, -epsilon, epsilon)
        attacked = torch.clamp(original + perturbation, 0, 1).detach()

    attacked = attacked[0].permute(1, 2, 0) * 255
    return attacked.round().to(torch.uint8).cpu().numpy()
