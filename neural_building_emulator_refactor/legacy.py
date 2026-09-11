"""Compatibility adapter for the established state-space trainers."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from neural_building_emulator.train import TrainConfig, run_training

from .artifacts import write_legacy_manifest
from .config import ExperimentConfig
from .registry import ModelSpec


def _coerce_like_default(value: Any, default: Any) -> Any:
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(default, Path) and not isinstance(value, Path):
        return Path(value)
    return value


def build_legacy_config(config: ExperimentConfig, spec: ModelSpec) -> TrainConfig:
    """Translate the small comparison config into the legacy trainer config."""
    if spec.backend != "legacy" or spec.legacy_model_kind is None:
        raise ValueError(f"{spec.name!r} is not a legacy-backed model")

    defaults = TrainConfig()
    values = dataclasses.asdict(defaults)
    unknown = sorted(set(config.legacy_overrides).difference(values))
    if unknown:
        raise ValueError(f"Unknown legacy config fields: {', '.join(unknown)}")
    for name, value in config.legacy_overrides.items():
        values[name] = _coerce_like_default(value, getattr(defaults, name))

    values.update(
        dataset_path=Path(config.dataset_path),
        max_profiles=config.max_profiles,
        test_fraction=config.test_fraction,
        seed=config.seed,
        sequence_length=config.sequence_length,
        stride=config.stride,
        rotate_window_starts=config.rotate_window_starts,
        batch_size=config.optimizer.batch_size,
        epochs=config.optimizer.epochs,
        learning_rate=config.optimizer.learning_rate,
        gradient_clip_norm=config.optimizer.gradient_clip_norm,
        early_stopping_patience=config.optimizer.early_stopping_patience,
        early_stopping_min_delta=config.optimizer.early_stopping_min_delta,
        train_eval_max_windows=config.optimizer.train_eval_max_windows,
        output_dir=Path(config.output_dir),
        model_checkpoint_dir=Path(config.output_dir) / "artifacts",
        save_model=True,
        model_kind=spec.legacy_model_kind,
    )
    if spec.task == "q_to_t":
        values.update(
            heating_mode=config.heating_mode,
            heat_input_normalization=config.heat_input_normalization,
            target_alignment="next_step",
        )
    else:
        values.update(
            target_alignment="same_time",
            target_mode="absolute",
            heat_input_normalization="per_floor_area",
            hp_power_area_normalization=config.hp_power_area_normalization,
        )
    return TrainConfig(**values)


def train_legacy(config: ExperimentConfig, spec: ModelSpec) -> Path:
    legacy_config = build_legacy_config(config, spec)
    run_training(legacy_config)
    artifact_dir = (
        Path(legacy_config.model_checkpoint_dir)
        / str(legacy_config.model_kind)
        / "selected"
    )
    if not artifact_dir.exists():
        raise RuntimeError(f"Legacy trainer did not create selected artifact {artifact_dir}")
    return write_legacy_manifest(
        artifact_dir,
        spec=spec,
        experiment_config=config,
    )
