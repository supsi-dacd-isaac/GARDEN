"""Typed configuration for the refactored comparison pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neural_building_emulator.data import DEFAULT_DATASET_PATH


@dataclass(frozen=True)
class OptimizerConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    train_eval_max_windows: int = 512


@dataclass(frozen=True)
class LSTMConfig:
    hidden_dim: int = 64
    metadata_hidden_dim: int = 64
    metadata_depth: int = 2
    bptt_truncate_steps: int = 0
    output_weights: tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    dataset_path: Path = DEFAULT_DATASET_PATH
    output_dir: Path = Path("output/neural_building_emulator_refactor")
    max_profiles: int | None = 100
    test_fraction: float = 0.2
    seed: int = 13
    sequence_length: int = 960
    stride: int = 2024
    rotate_window_starts: bool = True
    heating_mode: str = "zone_thermal"
    heat_input_normalization: str = "per_floor_area"
    hp_power_area_normalization: str = "building_heated_area"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    legacy_overrides: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.max_profiles is not None and self.max_profiles < 1:
            raise ValueError("max_profiles must be positive or None")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1)")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.optimizer.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.optimizer.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.optimizer.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer.gradient_clip_norm < 0.0:
            raise ValueError("gradient_clip_norm must be non-negative")
        if self.optimizer.train_eval_max_windows < 0:
            raise ValueError("train_eval_max_windows must be non-negative")
        if self.lstm.hidden_dim < 1 or self.lstm.metadata_hidden_dim < 1:
            raise ValueError("LSTM hidden dimensions must be positive")
        if self.lstm.metadata_depth < 1:
            raise ValueError("metadata_depth must be positive")
        if self.lstm.bptt_truncate_steps < 0:
            raise ValueError("bptt_truncate_steps must be non-negative")
        if self.hp_power_area_normalization not in (
            "zone_floor_area",
            "building_heated_area",
        ):
            raise ValueError(
                "hp_power_area_normalization must be 'zone_floor_area' or "
                "'building_heated_area'"
            )
