"""Differentiable detector adapters with unified ``[Real, Fake]`` logits."""

from .registry import DetectorAdapter, SUPPORTED_DETECTORS, load_detector

__all__ = ["DetectorAdapter", "SUPPORTED_DETECTORS", "load_detector"]
