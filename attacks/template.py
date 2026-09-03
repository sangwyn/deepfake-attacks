ATTACK_CONTRACT = {
    "version": 1,
    "supported_source_models": ["vit_b_16", "densenet121_dct"],
    "description": "Identity baseline; copy this interface for a new attack.",
}


def attack(
    image,
    classifiers,
    device,
    source_model="vit_b_16",
    target_class=0,
    **parameters,
):
    del classifiers, device, source_model, target_class, parameters
    return image
