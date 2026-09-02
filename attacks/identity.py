"""No-op attack used to verify the evaluation pipeline."""


def attack(image, classifiers, device):
    return image.copy()
