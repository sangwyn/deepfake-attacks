"""No-op attack used for pipeline validation."""


def attack(image, classifiers, device, **parameters):
    del classifiers, device, parameters
    return image.copy()
