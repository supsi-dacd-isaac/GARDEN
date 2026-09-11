"""Parallel, registry-based building emulator training package.

This package intentionally coexists with :mod:`neural_building_emulator`.
The established state-space implementations are consumed through adapters so
that refactoring does not alter their equations or saved artifacts.
"""

from .config import ExperimentConfig, LSTMConfig, OptimizerConfig
from .registry import MODEL_REGISTRY, ModelSpec, get_model_spec

__all__ = [
    "ExperimentConfig",
    "LSTMConfig",
    "MODEL_REGISTRY",
    "ModelSpec",
    "OptimizerConfig",
    "get_model_spec",
]
