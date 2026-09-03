"""Versioned adapter for swappable attack modules."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

from .io import ContractError


CONTRACT_VERSION = 1


def load_attack_module(module_name: str, source_model: str) -> tuple[ModuleType, Callable[..., Any]]:
    if not module_name.startswith("attacks."):
        raise ContractError("Attack modules must live under attacks.*")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ContractError(f"Cannot import attack module {module_name}: {exc}") from exc
    function = getattr(module, "attack", None)
    if not callable(function):
        raise ContractError(f"Attack module {module_name} has no callable attack()")
    contract = getattr(module, "ATTACK_CONTRACT", None)
    if not isinstance(contract, dict) or contract.get("version") != CONTRACT_VERSION:
        raise ContractError(
            f"{module_name} must define ATTACK_CONTRACT with version={CONTRACT_VERSION}"
        )
    supported = contract.get("supported_source_models")
    if not isinstance(supported, list) or source_model not in supported:
        raise ContractError(
            f"{module_name} does not declare source model {source_model!r}"
        )
    return module, function


def invoke_attack(
    function: Callable[..., Any],
    image: Any,
    classifiers: dict[str, Any],
    device: Any,
    source_model: str,
    target_class: int,
    parameters: dict[str, Any],
) -> Any:
    kwargs = {
        "source_model": source_model,
        "target_class": target_class,
        **parameters,
    }
    try:
        inspect.signature(function).bind(image, classifiers, device, **kwargs)
    except TypeError as exc:
        raise ContractError(f"Attack parameters do not match attack() signature: {exc}") from exc
    return function(image, classifiers, device, **kwargs)
