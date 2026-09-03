"""One-step Fast Gradient Sign Method."""

from ._utils import (
    checked_gradient,
    ensemble_loss,
    image_to_tensor,
    project_linf,
    tensor_to_image,
    validate_steps,
)


def attack(
    image,
    classifiers,
    device,
    *,
    epsilon=8 / 255,
    objective="targeted_fake_to_real",
    label=None,
    source_weights=None,
    seed=0,
):
    del seed  # FGSM has no stochastic operations.
    validate_steps(epsilon)
    original = image_to_tensor(image, device).requires_grad_(True)
    loss = ensemble_loss(original, classifiers, objective, label, source_weights)
    gradient = checked_gradient(loss, original, "FGSM")
    attacked = project_linf(
        original - float(epsilon) * gradient.sign(), original, epsilon
    )
    return tensor_to_image(attacked)
