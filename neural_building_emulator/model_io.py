"""Save and load trained emulator artifacts for ex-post analysis."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np

from .columns import DISTURBANCE_COLUMNS
from .models import (
    ClosedLoopHPEmulator,
    ContractingClosedLoopHPEmulator,
    MetadataStateSpaceEmulator,
    ProbabilisticClosedLoopHPEmulator,
    ProbabilisticContractingClosedLoopHPEmulator,
    ProbabilisticStableStateSpaceEmulator,
)
from .scaling import StandardScaler, WindowScalers

MODEL_FILENAME = "model.eqx"
SCALERS_FILENAME = "scalers.npz"
METADATA_FILENAME = "metadata.json"
SETPOINT_METADATA_COLUMN = "shSetpoint"


@dataclass(frozen=True)
class SavedModelArtifact:
    model: (
        MetadataStateSpaceEmulator
        | ProbabilisticStableStateSpaceEmulator
        | ClosedLoopHPEmulator
        | ContractingClosedLoopHPEmulator
        | ProbabilisticClosedLoopHPEmulator
        | ProbabilisticContractingClosedLoopHPEmulator
    )
    scalers: WindowScalers
    metadata: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
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


def _artifact_filenames(model_kind: str) -> tuple[str, str, str]:
    return (
        f"model_{model_kind}.eqx",
        f"scalers_{model_kind}.npz",
        f"metadata_{model_kind}.json",
    )


def _find_metadata_file(artifact_dir: Path) -> Path:
    candidates = sorted(artifact_dir.glob("metadata_*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple metadata files found in {artifact_dir}: {names}")
    legacy_path = artifact_dir / METADATA_FILENAME
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(f"No metadata file found in {artifact_dir}")


def _remove_stale_artifact_files(artifact_dir: Path, model_kind: str) -> None:
    for path in (
        artifact_dir / MODEL_FILENAME,
        artifact_dir / SCALERS_FILENAME,
        artifact_dir / METADATA_FILENAME,
        artifact_dir / f"model_{model_kind}.eqx",
        artifact_dir / f"scalers_{model_kind}.npz",
        artifact_dir / f"metadata_{model_kind}.json",
    ):
        path.unlink(missing_ok=True)


def save_training_artifact(
    artifact_dir: Path,
    *,
    model: (
        MetadataStateSpaceEmulator
        | ProbabilisticStableStateSpaceEmulator
        | ClosedLoopHPEmulator
        | ContractingClosedLoopHPEmulator
        | ProbabilisticClosedLoopHPEmulator
        | ProbabilisticContractingClosedLoopHPEmulator
    ),
    scalers: WindowScalers,
    train_config: Any,
    splits: Any,
    metadata_dim: int,
    input_dim: int,
    checkpoint_epoch: int,
    checkpoint_metric: str,
    checkpoint_metric_value: float,
    train_metrics: dict[str, float] | None,
    test_metrics: dict[str, float] | None,
) -> Path:
    """Persist a model plus reconstruction metadata in ``artifact_dir``."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_kind = getattr(train_config, "model_kind")
    _remove_stale_artifact_files(artifact_dir, model_kind)
    model_filename, scalers_filename, metadata_filename = _artifact_filenames(model_kind)
    eqx.tree_serialise_leaves(artifact_dir / model_filename, model)
    _save_scalers(artifact_dir / scalers_filename, scalers)

    metadata = {
        "artifact_version": 1,
        "model_file": model_filename,
        "scalers_file": scalers_filename,
        "train_config": _json_safe(train_config),
        "model_kind": model_kind,
        "heating_mode": splits.heating_mode,
        "heat_input_normalization": splits.heat_input_normalization,
        "input_feature_mode": splits.input_feature_mode,
        "heating_regime_window_steps": int(splits.heating_regime_window_steps),
        "target_mode": getattr(train_config, "target_mode"),
        "target_alignment": getattr(train_config, "target_alignment", "same_time"),
        "input_columns": list(splits.input_columns),
        "metadata_columns": list(splits.metadata_columns),
        "metadata_dim": int(metadata_dim),
        "input_dim": int(input_dim),
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_metric_value": _json_safe(float(checkpoint_metric_value)),
        "train_metrics": _json_safe(train_metrics),
        "test_metrics": _json_safe(test_metrics),
        "selected_ids": list(splits.selected_ids),
        "train_ids": list(splits.train_ids),
        "test_ids": list(splits.test_ids),
    }
    (artifact_dir / metadata_filename).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return artifact_dir


def _input_encoder_feedback_scaler_kwargs(
    metadata: dict[str, Any],
    scalers: WindowScalers,
) -> dict[str, int | float]:
    config = metadata["train_config"]
    require_thermal_gaps = config.get("input_encoder_feedback", "none") == "thermal_gaps"
    input_columns = list(metadata.get("input_columns", []))
    metadata_columns = list(metadata.get("metadata_columns", []))

    try:
        outdoor_temperature_input_index = input_columns.index(DISTURBANCE_COLUMNS[0])
    except ValueError as exc:
        if require_thermal_gaps:
            raise ValueError(
                f"Artifact requires input column {DISTURBANCE_COLUMNS[0]!r} for thermal_gaps feedback"
            ) from exc
        outdoor_temperature_input_index = 0

    try:
        setpoint_metadata_index = metadata_columns.index(SETPOINT_METADATA_COLUMN)
    except ValueError as exc:
        if require_thermal_gaps:
            raise ValueError(
                f"Artifact requires metadata column {SETPOINT_METADATA_COLUMN!r} for thermal_gaps feedback"
            ) from exc
        setpoint_metadata_index = 0

    input_mean = np.asarray(scalers.inputs.mean, dtype=np.float32).reshape(-1)
    input_scale = np.asarray(scalers.inputs.scale, dtype=np.float32).reshape(-1)
    metadata_mean = np.asarray(scalers.metadata.mean, dtype=np.float32).reshape(-1)
    metadata_scale = np.asarray(scalers.metadata.scale, dtype=np.float32).reshape(-1)
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)

    return {
        "outdoor_temperature_input_index": int(outdoor_temperature_input_index),
        "outdoor_temperature_input_mean": float(input_mean[outdoor_temperature_input_index]),
        "outdoor_temperature_input_scale": float(input_scale[outdoor_temperature_input_index]),
        "setpoint_metadata_index": int(setpoint_metadata_index),
        "setpoint_metadata_mean": float(metadata_mean[setpoint_metadata_index]),
        "setpoint_metadata_scale": float(metadata_scale[setpoint_metadata_index]),
        "target_temperature_mean": float(target_mean[0]),
        "target_temperature_scale": float(target_scale[0]),
    }


def _switching_dynamics_scaler_kwargs(
    metadata: dict[str, Any],
    scalers: WindowScalers,
) -> dict[str, int | float]:
    config = metadata["train_config"]
    input_columns = list(metadata.get("input_columns", []))
    input_mean = np.asarray(scalers.inputs.mean, dtype=np.float32).reshape(-1)
    input_scale = np.asarray(scalers.inputs.scale, dtype=np.float32).reshape(-1)

    heat_input_index = 0
    heating_on_input_index = -1
    recently_on_input_index = -1
    for index, name in enumerate(input_columns):
        if name.endswith("_is_on"):
            heating_on_input_index = index
        elif "_recently_on_" in name:
            recently_on_input_index = index

    def mean_at(index: int) -> float:
        return float(input_mean[index]) if index >= 0 else 0.0

    def scale_at(index: int) -> float:
        return float(input_scale[index]) if index >= 0 else 1.0

    return {
        "heat_input_index": int(heat_input_index),
        "heat_input_mean": float(input_mean[heat_input_index]),
        "heat_input_scale": float(input_scale[heat_input_index]),
        "heating_on_input_index": int(heating_on_input_index),
        "heating_on_input_mean": mean_at(heating_on_input_index),
        "heating_on_input_scale": scale_at(heating_on_input_index),
        "recently_on_input_index": int(recently_on_input_index),
        "recently_on_input_mean": mean_at(recently_on_input_index),
        "recently_on_input_scale": scale_at(recently_on_input_index),
        "heat_on_threshold": float(config.get("heat_on_threshold", 1e-6)),
        "switching_alpha_heat_scale": float(config.get("switching_alpha_heat_scale", 1.0)),
        "switching_alpha_on_weight": float(config.get("switching_alpha_on_weight", 2.0)),
        "switching_alpha_recent_weight": float(config.get("switching_alpha_recent_weight", 1.0)),
    }


def _build_model_skeleton(
    metadata: dict[str, Any],
    scalers: WindowScalers,
) -> (
    MetadataStateSpaceEmulator
    | ProbabilisticStableStateSpaceEmulator
    | ClosedLoopHPEmulator
    | ContractingClosedLoopHPEmulator
    | ProbabilisticClosedLoopHPEmulator
    | ProbabilisticContractingClosedLoopHPEmulator
):
    config = metadata["train_config"]
    model_kind = metadata["model_kind"]
    key = jax.random.PRNGKey(0)
    feedback_scaler_kwargs = _input_encoder_feedback_scaler_kwargs(metadata, scalers)
    switching_scaler_kwargs = _switching_dynamics_scaler_kwargs(metadata, scalers)

    if model_kind == "deterministic":
        return MetadataStateSpaceEmulator(
            metadata_dim=int(metadata["metadata_dim"]),
            input_dim=int(metadata["input_dim"]),
            state_dim=int(config["state_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            input_encoder_dim=config["input_encoder_dim"],
            input_encoder_hidden_dim=config["input_encoder_hidden_dim"],
            input_encoder_depth=int(config["input_encoder_depth"]),
            input_encoder_feedback=config.get("input_encoder_feedback", "none"),
            zero_d=bool(config.get("zero_d", False)),
            output_timing=config.get("output_timing", "pre_update"),
            switching_dynamics=config.get("switching_dynamics", "none"),
            **switching_scaler_kwargs,
            **feedback_scaler_kwargs,
            schur_gamma=float(config["schur_gamma"]),
            pf_lambda_min=float(config.get("pf_lambda_min", 0.0)),
            schur_mode=config["schur_mode"],
            key=key,
        )

    if model_kind == "probabilistic":
        encoded_input_dim = config["input_encoder_dim"] or int(metadata["input_dim"])
        return ProbabilisticStableStateSpaceEmulator(
            metadata_dim=int(metadata["metadata_dim"]),
            input_dim=int(metadata["input_dim"]),
            state_dim=int(config["state_dim"]),
            encoded_input_dim=int(encoded_input_dim),
            latent_dim=int(config["prob_latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            input_encoder_hidden_dim=config["input_encoder_hidden_dim"],
            input_encoder_depth=int(config["input_encoder_depth"]),
            input_encoder_feedback=config.get("input_encoder_feedback", "none"),
            process_noise_mode=config["prob_process_noise"],
            process_noise_init=float(config["prob_process_noise_init"]),
            zero_d=bool(config.get("zero_d", False)),
            output_timing=config.get("output_timing", "pre_update"),
            switching_dynamics=config.get("switching_dynamics", "none"),
            **switching_scaler_kwargs,
            **feedback_scaler_kwargs,
            schur_gamma=float(config["schur_gamma"]),
            pf_lambda_min=float(config.get("pf_lambda_min", 0.0)),
            schur_mode=config["schur_mode"],
            key=key,
        )

    if model_kind == "closed_loop_hp":
        input_mean = tuple(float(value) for value in np.asarray(scalers.inputs.mean, dtype=np.float32).reshape(-1))
        input_scale = tuple(float(value) for value in np.asarray(scalers.inputs.scale, dtype=np.float32).reshape(-1))
        target_mean = tuple(float(value) for value in np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1))
        target_scale = tuple(float(value) for value in np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1))
        return ClosedLoopHPEmulator(
            metadata_dim=int(metadata["metadata_dim"]),
            input_dim=int(metadata["input_dim"]),
            state_dim=int(config["state_dim"]),
            controller_state_dim=int(config.get("hp_controller_state_dim", 2)),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            input_encoder_dim=config["input_encoder_dim"],
            input_encoder_hidden_dim=config["input_encoder_hidden_dim"],
            input_encoder_depth=int(config["input_encoder_depth"]),
            hp_dt_hours=float(config.get("hp_dt_hours", 0.25)),
            hp_cop_floor=float(config.get("hp_cop_floor", 1.0)),
            hp_cop_cap=float(config.get("hp_cop_cap", 0.0)),
            hp_pel_cap_w_m2=float(config.get("hp_pel_cap_w_m2", 0.0)),
            hp_qroom_cap_w_m2=float(config.get("hp_qroom_cap_w_m2", 0.0)),
            hp_energy_cap_wh_m2=float(config.get("hp_energy_cap_wh_m2", 0.0)),
            schur_gamma=float(config["schur_gamma"]),
            pf_lambda_min=float(config.get("pf_lambda_min", 0.0)),
            schur_mode=config["schur_mode"],
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            key=key,
        )

    if model_kind == "closed_loop_hp_contracting":
        input_mean = tuple(float(value) for value in np.asarray(scalers.inputs.mean, dtype=np.float32).reshape(-1))
        input_scale = tuple(float(value) for value in np.asarray(scalers.inputs.scale, dtype=np.float32).reshape(-1))
        target_mean = tuple(float(value) for value in np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1))
        target_scale = tuple(float(value) for value in np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1))
        return ContractingClosedLoopHPEmulator(
            metadata_dim=int(metadata["metadata_dim"]),
            input_dim=int(metadata["input_dim"]),
            state_dim=int(config["state_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            input_encoder_dim=config["input_encoder_dim"],
            input_encoder_hidden_dim=config["input_encoder_hidden_dim"],
            input_encoder_depth=int(config["input_encoder_depth"]),
            contraction_gamma=float(config.get("contracting_gamma", 0.99)),
            state_bound=float(config.get("contracting_state_bound", 5.0)),
            temperature_output_scale=float(config.get("contracting_temperature_scale", 8.0)),
            temperature_delta_max_c=float(config.get("contracting_temperature_delta_max_c", 0.0)),
            hp_dt_hours=float(config.get("hp_dt_hours", 0.25)),
            hp_cop_floor=float(config.get("hp_cop_floor", 1.0)),
            hp_cop_cap=float(config.get("hp_cop_cap", 0.0)),
            hp_pel_cap_w_m2=float(config.get("hp_pel_cap_w_m2", 0.0)),
            hp_qroom_cap_w_m2=float(config.get("hp_qroom_cap_w_m2", 0.0)),
            hp_energy_cap_wh_m2=float(config.get("hp_energy_cap_wh_m2", 0.0)),
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            key=key,
        )

    if model_kind == "closed_loop_hp_contracting_probabilistic":
        input_mean = tuple(float(value) for value in np.asarray(scalers.inputs.mean, dtype=np.float32).reshape(-1))
        input_scale = tuple(float(value) for value in np.asarray(scalers.inputs.scale, dtype=np.float32).reshape(-1))
        target_mean = tuple(float(value) for value in np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1))
        target_scale = tuple(float(value) for value in np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1))
        return ProbabilisticContractingClosedLoopHPEmulator(
            metadata_dim=int(metadata["metadata_dim"]),
            input_dim=int(metadata["input_dim"]),
            state_dim=int(config["state_dim"]),
            controller_state_dim=int(config.get("hp_controller_state_dim", 2)),
            latent_dim=int(config.get("prob_latent_dim", 4)),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            input_encoder_dim=config["input_encoder_dim"],
            input_encoder_hidden_dim=config["input_encoder_hidden_dim"],
            input_encoder_depth=int(config["input_encoder_depth"]),
            process_noise_mode=config.get("prob_process_noise", "constant"),
            process_noise_init=float(config.get("prob_process_noise_init", -6.0)),
            hp_emission_mode=config.get("prob_hp_emission_mode", "legacy_lognormal_mean"),
            contraction_gamma=float(config.get("contracting_gamma", 0.99)),
            state_bound=float(config.get("contracting_state_bound", 5.0)),
            temperature_output_scale=float(config.get("contracting_temperature_scale", 8.0)),
            temperature_delta_max_c=float(config.get("contracting_temperature_delta_max_c", 0.0)),
            hp_dt_hours=float(config.get("hp_dt_hours", 0.25)),
            hp_cop_floor=float(config.get("hp_cop_floor", 1.0)),
            hp_cop_cap=float(config.get("hp_cop_cap", 0.0)),
            hp_pel_cap_w_m2=float(config.get("hp_pel_cap_w_m2", 0.0)),
            hp_qroom_cap_w_m2=float(config.get("hp_qroom_cap_w_m2", 0.0)),
            hp_energy_cap_wh_m2=float(config.get("hp_energy_cap_wh_m2", 0.0)),
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            key=key,
        )

    if model_kind == "closed_loop_hp_probabilistic":
        input_mean = tuple(float(value) for value in np.asarray(scalers.inputs.mean, dtype=np.float32).reshape(-1))
        input_scale = tuple(float(value) for value in np.asarray(scalers.inputs.scale, dtype=np.float32).reshape(-1))
        target_mean = tuple(float(value) for value in np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1))
        target_scale = tuple(float(value) for value in np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1))
        return ProbabilisticClosedLoopHPEmulator(
            metadata_dim=int(metadata["metadata_dim"]),
            input_dim=int(metadata["input_dim"]),
            state_dim=int(config["state_dim"]),
            controller_state_dim=int(config.get("hp_controller_state_dim", 2)),
            latent_dim=int(config.get("prob_latent_dim", 4)),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            input_encoder_dim=config["input_encoder_dim"],
            input_encoder_hidden_dim=config["input_encoder_hidden_dim"],
            input_encoder_depth=int(config["input_encoder_depth"]),
            process_noise_mode=config.get("prob_process_noise", "constant"),
            process_noise_init=float(config.get("prob_process_noise_init", -6.0)),
            hp_emission_mode=config.get("prob_hp_emission_mode", "legacy_lognormal_mean"),
            hp_dt_hours=float(config.get("hp_dt_hours", 0.25)),
            hp_cop_floor=float(config.get("hp_cop_floor", 1.0)),
            hp_cop_cap=float(config.get("hp_cop_cap", 0.0)),
            hp_pel_cap_w_m2=float(config.get("hp_pel_cap_w_m2", 0.0)),
            hp_qroom_cap_w_m2=float(config.get("hp_qroom_cap_w_m2", 0.0)),
            hp_energy_cap_wh_m2=float(config.get("hp_energy_cap_wh_m2", 0.0)),
            schur_gamma=float(config["schur_gamma"]),
            pf_lambda_min=float(config.get("pf_lambda_min", 0.0)),
            schur_mode=config["schur_mode"],
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            key=key,
        )

    raise ValueError(f"Unknown model_kind in artifact: {model_kind!r}")


def load_training_artifact(
    artifact_dir: Path,
    *,
    train_config_overrides: dict[str, Any] | None = None,
) -> SavedModelArtifact:
    """Load a trained model artifact saved by ``save_training_artifact``."""
    artifact_dir = Path(artifact_dir)
    metadata = json.loads(_find_metadata_file(artifact_dir).read_text())
    if train_config_overrides:
        metadata = dict(metadata)
        train_config = dict(metadata.get("train_config", {}))
        train_config.update(train_config_overrides)
        metadata["train_config"] = train_config
    scalers = _load_scalers(artifact_dir / metadata["scalers_file"])
    skeleton = _build_model_skeleton(metadata, scalers)
    model = eqx.tree_deserialise_leaves(artifact_dir / metadata["model_file"], skeleton)
    return SavedModelArtifact(model=model, scalers=scalers, metadata=metadata)
