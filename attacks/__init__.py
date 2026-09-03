"""Pixel-space attacks loaded by module name from the YAML configuration."""

AVAILABLE_ATTACKS = (
    "identity",
    "fgsm",
    "pgd",
    "mi_di_fgsm",
    "ssa_s2i_fgsm",
    "mig_cow",
    "frequency_pgd",
    "isp_pgd",
)
