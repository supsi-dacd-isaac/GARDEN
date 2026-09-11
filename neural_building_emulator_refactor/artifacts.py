"""Unified artifact persistence for legacy state-space and refactored models."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import numpy as np

from neural_building_emulator.model_io import (
    SavedModelArtifact,
    load_training_artifact,
)
from neural_building_emulator.scaling import StandardScaler, WindowScalers

from .config import ExperimentConfig, LSTMConfig
from .models import AutoregressiveLSTM
from .registry import MODEL_REGISTRY, ModelSpec, get_model_spec

MANIFEST_FILENAME = "refactor_manifest.json"
MODEL_FILENAME = "model.eqx"
SCALERS_FILENAME = "scalers.npz"
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class LoadedArtifact:
    """Backend-neutral trained model bundle."""

    artifact_dir: Path
    spec: ModelSpec
    backend: Literal["legacy", "lstm"]
    model: Any
    scalers: WindowScalers
    metadata: dict[str, Any]
    legacy_artifact: SavedModelArtifact | None = None


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _save_scalers(path: Path, scalers: WindowScalers) -> None:
    np.savez(
        path,
        metadata_mean=scalers.metadata.mean,
        metadata_scale=scalers.metadata.scale,
        inputs_mean=scalers.inputs.mean,
        inputs_scale=scalers.inputs.scale,
        target_mean=scalers.target.mean,
        target_scale=scalers.target.scale,
    )


def _load_scalers(path: Path) -> WindowScalers:
    with np.load(path) as data:
        return WindowScalers(
            metadata=StandardScaler(
                mean=data["metadata_mean"].astype(np.float32),
                scale=data["metadata_scale"].astype(np.float32),
            ),
            inputs=StandardScaler(
                mean=data["inputs_mean"].astype(np.float32),
                scale=data["inputs_scale"].astype(np.float32),
            ),
            target=StandardScaler(
                mean=data["target_mean"].astype(np.float32),
                scale=data["target_scale"].astype(np.float32),
            ),
        )


def _write_manifest(artifact_dir: Path, metadata: dict[str, Any]) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(json_safe(metadata), indent=2, sort_keys=True))
    return artifact_dir


def write_legacy_manifest(
    artifact_dir: Path,
    *,
    spec: ModelSpec,
    experiment_config: ExperimentConfig,
) -> Path:
    """Mark an artifact produced by the original trainer as registry-loadable."""
    legacy = load_training_artifact(artifact_dir)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "backend": "legacy",
        "model_name": spec.name,
        "task": spec.task,
        "probabilistic": spec.probabilistic,
        "experiment_config": experiment_config,
        "legacy_metadata": legacy.metadata,
    }
    return _write_manifest(artifact_dir, metadata)


def save_lstm_artifact(
    artifact_dir: Path,
    *,
    model: AutoregressiveLSTM,
    scalers: WindowScalers,
    spec: ModelSpec,
    experiment_config: ExperimentConfig,
    input_columns: list[str] | tuple[str, ...],
    metadata_columns: list[str] | tuple[str, ...],
    target_columns: list[str] | tuple[str, ...],
    selected_ids: list[int] | tuple[int, ...],
    train_ids: list[int] | tuple[int, ...],
    test_ids: list[int] | tuple[int, ...],
    checkpoint_epoch: int,
    checkpoint_metric_value: float,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(artifact_dir / MODEL_FILENAME, model)
    _save_scalers(artifact_dir / SCALERS_FILENAME, scalers)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "backend": "lstm",
        "model_name": spec.name,
        "task": spec.task,
        "probabilistic": False,
        "model_file": MODEL_FILENAME,
        "scalers_file": SCALERS_FILENAME,
        "experiment_config": experiment_config,
        "lstm_config": experiment_config.lstm,
        "metadata_dim": int(model.initial_hidden.layers[0].weight.shape[1]),
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "input_columns": list(input_columns),
        "metadata_columns": list(metadata_columns),
        "hp_power_area_normalization": experiment_config.hp_power_area_normalization,
        "target_columns": list(target_columns),
        "selected_ids": list(selected_ids),
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_metric": "test_temperature_rmse",
        "checkpoint_metric_value": checkpoint_metric_value,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    return _write_manifest(artifact_dir, metadata)


def _legacy_spec(model_kind: str) -> ModelSpec:
    matches = [
        spec for spec in MODEL_REGISTRY.values() if spec.legacy_model_kind == model_kind
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot infer a unique refactor model for legacy model_kind={model_kind!r}; "
            "load an artifact containing refactor_manifest.json instead."
        )
    return matches[0]


def load_artifact(artifact_dir: Path) -> LoadedArtifact:
    """Load any artifact produced by the comparison package.

    Original state-space artifacts without a refactor manifest are also accepted
    when their legacy model kind maps unambiguously to the registry.
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        legacy = load_training_artifact(artifact_dir)
        spec = _legacy_spec(str(legacy.metadata["model_kind"]))
        return LoadedArtifact(
            artifact_dir=artifact_dir,
            spec=spec,
            backend="legacy",
            model=legacy.model,
            scalers=legacy.scalers,
            metadata=legacy.metadata,
            legacy_artifact=legacy,
        )

    metadata = json.loads(manifest_path.read_text())
    spec = get_model_spec(str(metadata["model_name"]))
    backend = str(metadata["backend"])
    if backend == "legacy":
        legacy = load_training_artifact(artifact_dir)
        return LoadedArtifact(
            artifact_dir=artifact_dir,
            spec=spec,
            backend="legacy",
            model=legacy.model,
            scalers=legacy.scalers,
            metadata=metadata,
            legacy_artifact=legacy,
        )
    if backend != "lstm":
        raise ValueError(f"Unknown artifact backend {backend!r}")

    lstm_values = metadata["lstm_config"]
    lstm_config = LSTMConfig(
        hidden_dim=int(lstm_values["hidden_dim"]),
        metadata_hidden_dim=int(lstm_values["metadata_hidden_dim"]),
        metadata_depth=int(lstm_values["metadata_depth"]),
        bptt_truncate_steps=int(lstm_values["bptt_truncate_steps"]),
        output_weights=tuple(float(value) for value in lstm_values["output_weights"]),
    )
    skeleton = AutoregressiveLSTM(
        metadata_dim=int(metadata["metadata_dim"]),
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata["output_dim"]),
        hidden_dim=lstm_config.hidden_dim,
        metadata_hidden_dim=lstm_config.metadata_hidden_dim,
        metadata_depth=lstm_config.metadata_depth,
        bptt_truncate_steps=lstm_config.bptt_truncate_steps,
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves(artifact_dir / metadata["model_file"], skeleton)
    scalers = _load_scalers(artifact_dir / metadata["scalers_file"])
    return LoadedArtifact(
        artifact_dir=artifact_dir,
        spec=spec,
        backend="lstm",
        model=model,
        scalers=scalers,
        metadata=metadata,
    )
