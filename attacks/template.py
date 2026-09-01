# Minimal attack template. One file per attack; expose an `attack` function.
#
#   image       : H×W×3 uint8 RGB numpy array (the original image)
#   classifiers : dict {name: {"model": nn.Module, "transform": ..., ...}}
#                 attack the models under classifiers[name]["model"].
#   device      : torch.device
#   **kwargs    : anything from the config's `attack_params` block
#
# Return an H×W×3 uint8 RGB numpy array (the attacked image).
def attack(image, classifiers, device, **kwargs):
    return image
