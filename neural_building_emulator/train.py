"""Train a metadata-conditioned stable state-space building emulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go

from .columns import DISTURBANCE_COLUMNS, HEATING_INPUT_COLUMNS, InputFeatureMode
from .data import (
    BuildingProfile,
    DEFAULT_DATASET_PATH,
    SplitConfig,
    WindowConfig,
    WindowTargetAlignment,
    WindowedArrays,
    load_result_splits,
    make_windows,
    to_profiles,
)
from .metrics import regression_metrics
from .model_io import save_training_artifact
from .models import MetadataStateSpaceEmulator, ProbabilisticStableStateSpaceEmulator, SwitchingDynamics, spectral_radius
from .models.state_space import OutputTiming
from .scaling import WindowScalers, fit_window_scalers, inverse_target, transform_windows

TargetMode = Literal["absolute", "delta", "residual"]
CheckpointMetric = Literal["auto", "train_rmse_c", "test_rmse_c"]
LossNormalization = Literal["none", "window_std"]
InputEncoderFeedbackMode = Literal["none", "predicted_temperature", "thermal_gaps"]
ModelKind = Literal["deterministic", "probabilistic"]
ProbProcessNoiseMode = Literal["none", "constant", "heteroscedastic"]
EmulatorModel = MetadataStateSpaceEmulator | ProbabilisticStableStateSpaceEmulator
SETPOINT_METADATA_COLUMN = "shSetpoint"


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    max_profiles: int | None = 10
    test_fraction: float = 0.2
    seed: int = 13
    heating_mode: str = "A"
    heat_input_normalization: str = "per_floor_area"
    input_feature_mode: InputFeatureMode = "base"
    heating_regime_window_steps: int = 96 * 7
    heat_on_threshold: float = 1e-6
    sequence_length: int = 96
    stride: int = 96
    target_alignment: WindowTargetAlignment = "same_time"
    state_dim: int = 6
    hidden_dim: int = 64
    depth: int = 3
    input_encoder_dim: int | None = None
    input_encoder_hidden_dim: int | None = None
    input_encoder_depth: int = 2
    input_encoder_feedback: InputEncoderFeedbackMode = "none"
    zero_d: bool = False
    output_timing: OutputTiming = "pre_update"
    switching_dynamics: SwitchingDynamics = "none"
    switching_alpha_heat_scale: float = 1.0
    switching_alpha_on_weight: float = 2.0
    switching_alpha_recent_weight: float = 1.0
    schur_gamma: float = 0.995
    pf_lambda_min: float = 0.0
    schur_mode: str = "near_identity"
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    max_train_batches: int | None = None
    output_dir: Path = Path("output/neural_building_emulator")
    model_checkpoint_dir: Path | None = None
    save_model: bool = False
    save_model_every_epochs: int = 0
    num_window_plots: int = 1
    num_full_profile_plots: int = 1
    monotonicity_weight: float = 0.0
    monotonicity_horizon: int | None = None
    monotonicity_features: tuple[str, ...] = ("heat", "outdoor_temperature", "solar")
    target_mode: TargetMode = "absolute"
    checkpoint_metric: CheckpointMetric = "auto"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    loss_normalization: LossNormalization = "none"
    loss_std_floor_c: float = 0.25
    model_kind: ModelKind = "deterministic"
    prob_particles: int = 8
    prob_eval_particles: int = 16
    prob_plot_particles: int = 100
    prob_latent_dim: int = 4
    prob_process_noise: ProbProcessNoiseMode = "constant"
    prob_process_noise_init: float = -6.0
    prob_softopt_weight: float = 0.1
    prob_softopt_temperature: float = 0.1
    prob_variogram_weight: float = 0.1
    prob_variogram_lags: tuple[int, ...] = (1, 4, 16, 96)
    prob_variogram_power: float = 0.5
    prob_physics_weight: float = 0.0
    prob_horizon_weight_power: float = 0.0


def minibatches(
    windows: WindowedArrays,
    *,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    n = windows.targets.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield (
            windows.metadata[batch_idx],
            windows.inputs[batch_idx],
            windows.initial_temperature[batch_idx],
            windows.targets[batch_idx],
        )


def predict_batch(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    target_mode: TargetMode = "absolute",
) -> jnp.ndarray:
    raw_prediction = jax.vmap(model)(metadata, inputs, initial_temperature)
    return reconstruct_batch_temperature(raw_prediction, initial_temperature, target_mode)


def reconstruct_temperature(
    raw_prediction: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    target_mode: TargetMode,
) -> jnp.ndarray:
    """Convert raw model outputs to normalized absolute temperatures."""
    if target_mode == "absolute":
        return raw_prediction
    if target_mode == "residual":
        initial = initial_temperature[jnp.newaxis, :]
        return initial + raw_prediction - raw_prediction[:1]
    if target_mode == "delta":
        initial = initial_temperature[jnp.newaxis, :]
        if raw_prediction.shape[0] == 1:
            return initial
        integrated = initial + jnp.cumsum(raw_prediction[:-1], axis=0)
        return jnp.concatenate([initial, integrated], axis=0)
    raise ValueError(f"Unknown target_mode {target_mode!r}")


def reconstruct_batch_temperature(
    raw_prediction: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    target_mode: TargetMode,
) -> jnp.ndarray:
    if target_mode == "absolute":
        return raw_prediction
    return jax.vmap(lambda pred, init: reconstruct_temperature(pred, init, target_mode))(
        raw_prediction,
        initial_temperature,
    )


def reconstruct_particle_temperatures(
    raw_prediction: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    target_mode: TargetMode,
) -> jnp.ndarray:
    """Convert raw particle outputs [batch, particles, time, output] to temperatures."""
    if target_mode == "absolute":
        return raw_prediction
    return jax.vmap(
        lambda particle_predictions, init: jax.vmap(
            lambda pred: reconstruct_temperature(pred, init, target_mode)
        )(particle_predictions)
    )(raw_prediction, initial_temperature)


def target_window_std(targets: jnp.ndarray, std_floor: jnp.ndarray) -> jnp.ndarray:
    """Return per-window, per-output target std with a stabilizing floor."""
    return jnp.maximum(jnp.std(targets, axis=1, keepdims=True), std_floor)


def normalize_loss_trajectories(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    loss_normalization: LossNormalization,
    std_floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Optionally scale trajectories by the true window std before scoring."""
    if loss_normalization == "none":
        return predictions, targets
    if loss_normalization != "window_std":
        raise ValueError(f"Unknown loss_normalization {loss_normalization!r}")
    scale = target_window_std(targets, std_floor)
    if predictions.ndim == targets.ndim + 1:
        return predictions / scale[:, jnp.newaxis, :, :], targets / scale
    return predictions / scale, targets / scale


def _logmeanexp(values: jnp.ndarray, axis: int) -> jnp.ndarray:
    max_value = jnp.max(values, axis=axis, keepdims=True)
    return jnp.squeeze(
        max_value + jnp.log(jnp.mean(jnp.exp(values - max_value), axis=axis, keepdims=True)),
        axis=axis,
    )


def _horizon_weights(length: int, power: float) -> jnp.ndarray:
    if power == 0.0:
        return jnp.ones((length,), dtype=jnp.float32)
    weights = jnp.linspace(1.0 / length, 1.0, length, dtype=jnp.float32) ** power
    return weights / jnp.mean(weights)


def trajectory_squared_errors(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    horizon_weight_power: float,
) -> jnp.ndarray:
    """Return J[b, k] trajectory losses for particle predictions."""
    sq_error = jnp.sum((predictions - targets[:, jnp.newaxis, :, :]) ** 2, axis=-1)
    weights = _horizon_weights(predictions.shape[2], horizon_weight_power)
    return jnp.mean(sq_error * weights[jnp.newaxis, jnp.newaxis, :], axis=-1)


def soft_optimistic_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    temperature: float,
    horizon_weight_power: float,
) -> jnp.ndarray:
    if temperature <= 0.0:
        raise ValueError("soft optimistic temperature must be positive")
    losses = trajectory_squared_errors(
        predictions,
        targets,
        horizon_weight_power=horizon_weight_power,
    )
    return jnp.mean(-temperature * _logmeanexp(-losses / temperature, axis=1))


def energy_score(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Trajectory-level energy score over particles."""
    centered = predictions - targets[:, jnp.newaxis, :, :]
    obs_distance = jnp.sqrt(jnp.sum(centered**2, axis=(2, 3)) + 1e-6)
    pairwise = predictions[:, :, jnp.newaxis, :, :] - predictions[:, jnp.newaxis, :, :, :]
    pairwise_distance = jnp.sqrt(jnp.sum(pairwise**2, axis=(3, 4)) + 1e-6)
    return jnp.mean(jnp.mean(obs_distance, axis=1) - 0.5 * jnp.mean(pairwise_distance, axis=(1, 2)))


def smooth_abs_power(values: jnp.ndarray, power: float) -> jnp.ndarray:
    """Smooth |x|**power to avoid singular gradients at x=0 for power < 1."""
    return (values**2 + 1e-6) ** (0.5 * power)


def variogram_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    lags: tuple[int, ...],
    power: float,
) -> jnp.ndarray:
    """Efficient lag-restricted trajectory variogram score."""
    scores = []
    horizon = predictions.shape[2]
    for lag in lags:
        if lag < 1 or lag >= horizon:
            continue
        target_delta = smooth_abs_power(targets[:, lag:, :] - targets[:, :-lag, :], power)
        prediction_delta = smooth_abs_power(
            predictions[:, :, lag:, :] - predictions[:, :, :-lag, :],
            power,
        )
        prediction_moment = jnp.mean(prediction_delta, axis=1)
        scores.append(jnp.mean((target_delta - prediction_moment) ** 2))
    if not scores:
        return jnp.asarray(0.0, dtype=predictions.dtype)
    return jnp.mean(jnp.stack(scores))


def predict_probabilistic_batch(
    model: ProbabilisticStableStateSpaceEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    target_mode: TargetMode,
    key: jax.Array,
    num_particles: int,
    sample_process_noise: bool,
) -> jnp.ndarray:
    keys = jax.random.split(key, metadata.shape[0])
    raw_prediction = jax.vmap(
        lambda row_metadata, row_inputs, row_initial_temperature, row_key: model.sample(
            row_metadata,
            row_inputs,
            row_initial_temperature,
            key=row_key,
            num_particles=num_particles,
            sample_process_noise=sample_process_noise,
        )
    )(metadata, inputs, initial_temperature, keys)
    return reconstruct_particle_temperatures(raw_prediction, initial_temperature, target_mode)


def predict_mean_batch(
    model: EmulatorModel,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    target_mode: TargetMode,
    model_kind: ModelKind,
    key: jax.Array | None = None,
    num_particles: int = 1,
    sample_process_noise: bool = False,
) -> jnp.ndarray:
    if model_kind == "deterministic":
        assert isinstance(model, MetadataStateSpaceEmulator)
        return predict_batch(model, metadata, inputs, initial_temperature, target_mode)
    assert isinstance(model, ProbabilisticStableStateSpaceEmulator)
    if key is None:
        key = jax.random.PRNGKey(0)
    particles = predict_probabilistic_batch(
        model,
        metadata,
        inputs,
        initial_temperature,
        target_mode=target_mode,
        key=key,
        num_particles=num_particles,
        sample_process_noise=sample_process_noise,
    )
    return jnp.mean(particles, axis=1)


def resolve_monotonicity_feature_indices(
    columns: Sequence[str],
    feature_names: Sequence[str],
) -> tuple[int, ...]:
    """Map physics feature aliases to input-channel indices."""
    aliases = {
        "heat": columns[0],
        "heating": columns[0],
        "q_heat": columns[0],
        "heating_power": columns[0],
        "outdoor": DISTURBANCE_COLUMNS[0],
        "outdoor_temperature": DISTURBANCE_COLUMNS[0],
        "t_out": DISTURBANCE_COLUMNS[0],
        "solar": DISTURBANCE_COLUMNS[1],
        "solar_radiation": DISTURBANCE_COLUMNS[1],
        "g_solar": DISTURBANCE_COLUMNS[1],
    }
    column_indices = {column: index for index, column in enumerate(columns)}

    indices: list[int] = []
    for feature_name in feature_names:
        key = feature_name.strip().lower().replace("-", "_")
        if key not in aliases:
            valid = ", ".join(sorted(aliases))
            raise ValueError(
                f"Unknown monotonicity feature {feature_name!r}. Expected one of: {valid}"
            )
        column = aliases[key]
        if column not in column_indices:
            raise ValueError(
                f"Monotonicity feature {feature_name!r} maps to missing input column {column!r}"
            )
        index = column_indices[column]
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def _impulse_response_monotonicity_penalty_one(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    feature_indices: tuple[int, ...],
    horizon: int,
    target_mode: TargetMode,
) -> jnp.ndarray:
    matrices = model.matrices(metadata)
    feature_idx = jnp.asarray(feature_indices)
    direct_response = matrices.d[:, feature_idx]

    if horizon == 1:
        responses = direct_response[jnp.newaxis, :, :]
    else:
        initial_state_response = matrices.b[:, feature_idx]

        def step(
            state_response: jnp.ndarray,
            _: None,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            output_response = matrices.c @ state_response
            next_state_response = matrices.a @ state_response
            return next_state_response, output_response

        _, future_responses = jax.lax.scan(
            step,
            initial_state_response,
            None,
            length=horizon - 1,
        )
        responses = jnp.concatenate(
            [direct_response[jnp.newaxis, :, :], future_responses],
            axis=0,
        )
    if target_mode == "delta":
        zero_response = jnp.zeros_like(responses[:1])
        responses = jnp.concatenate(
            [zero_response, jnp.cumsum(responses[:-1], axis=0)],
            axis=0,
        )
    return jnp.mean(jax.nn.relu(-responses) ** 2)


def impulse_response_monotonicity_penalty(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    feature_indices: tuple[int, ...],
    horizon: int,
    target_mode: TargetMode,
) -> jnp.ndarray:
    """Penalize negative d indoor-temperature / d input impulse responses."""
    penalties = jax.vmap(
        lambda row: _impulse_response_monotonicity_penalty_one(
            model,
            row,
            feature_indices,
            horizon,
            target_mode,
        )
    )(metadata)
    return jnp.mean(penalties)


@eqx.filter_value_and_grad
def loss_fn(
    model: MetadataStateSpaceEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    monotonicity_weight: float,
    monotonicity_feature_indices: tuple[int, ...],
    monotonicity_horizon: int,
    target_mode: TargetMode,
    loss_normalization: LossNormalization,
    loss_std_floor: jnp.ndarray,
) -> jnp.ndarray:
    prediction = predict_batch(model, metadata, inputs, initial_temperature, target_mode)
    prediction, loss_targets = normalize_loss_trajectories(
        prediction,
        targets,
        loss_normalization=loss_normalization,
        std_floor=loss_std_floor,
    )
    mse = jnp.mean((prediction - loss_targets) ** 2)
    if monotonicity_weight <= 0.0 or not monotonicity_feature_indices:
        return mse
    monotonicity = impulse_response_monotonicity_penalty(
        model,
        metadata,
        monotonicity_feature_indices,
        monotonicity_horizon,
        target_mode,
    )
    return mse + monotonicity_weight * monotonicity


@eqx.filter_jit
def train_step(
    model: MetadataStateSpaceEmulator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    monotonicity_weight: float,
    monotonicity_feature_indices: tuple[int, ...],
    monotonicity_horizon: int,
    target_mode: TargetMode,
    loss_normalization: LossNormalization,
    loss_std_floor: jnp.ndarray,
) -> tuple[MetadataStateSpaceEmulator, optax.OptState, jnp.ndarray]:
    loss, grads = loss_fn(
        model,
        metadata,
        inputs,
        initial_temperature,
        targets,
        monotonicity_weight,
        monotonicity_feature_indices,
        monotonicity_horizon,
        target_mode,
        loss_normalization,
        loss_std_floor,
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_value_and_grad
def probabilistic_loss_fn(
    model: ProbabilisticStableStateSpaceEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    key: jax.Array,
    num_particles: int,
    target_mode: TargetMode,
    softopt_weight: float,
    softopt_temperature: float,
    variogram_weight: float,
    variogram_lags: tuple[int, ...],
    variogram_power: float,
    physics_weight: float,
    horizon_weight_power: float,
    loss_normalization: LossNormalization,
    loss_std_floor: jnp.ndarray,
) -> jnp.ndarray:
    predictions = predict_probabilistic_batch(
        model,
        metadata,
        inputs,
        initial_temperature,
        target_mode=target_mode,
        key=key,
        num_particles=num_particles,
        sample_process_noise=True,
    )
    predictions, loss_targets = normalize_loss_trajectories(
        predictions,
        targets,
        loss_normalization=loss_normalization,
        std_floor=loss_std_floor,
    )
    loss = energy_score(predictions, loss_targets)
    if variogram_weight > 0.0:
        loss = loss + variogram_weight * variogram_score(
            predictions,
            loss_targets,
            lags=variogram_lags,
            power=variogram_power,
        )
    if softopt_weight > 0.0:
        loss = loss + softopt_weight * soft_optimistic_loss(
            predictions,
            loss_targets,
            temperature=softopt_temperature,
            horizon_weight_power=horizon_weight_power,
        )
    if physics_weight > 0.0:
        regularization = jnp.mean(jax.vmap(model.parameter_regularization)(metadata))
        loss = loss + physics_weight * regularization
    return loss


@eqx.filter_jit
def probabilistic_train_step(
    model: ProbabilisticStableStateSpaceEmulator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    key: jax.Array,
    num_particles: int,
    target_mode: TargetMode,
    softopt_weight: float,
    softopt_temperature: float,
    variogram_weight: float,
    variogram_lags: tuple[int, ...],
    variogram_power: float,
    physics_weight: float,
    horizon_weight_power: float,
    loss_normalization: LossNormalization,
    loss_std_floor: jnp.ndarray,
) -> tuple[ProbabilisticStableStateSpaceEmulator, optax.OptState, jnp.ndarray]:
    loss, grads = probabilistic_loss_fn(
        model,
        metadata,
        inputs,
        initial_temperature,
        targets,
        key,
        num_particles,
        target_mode,
        softopt_weight,
        softopt_temperature,
        variogram_weight,
        variogram_lags,
        variogram_power,
        physics_weight,
        horizon_weight_power,
        loss_normalization,
        loss_std_floor,
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def evaluate(
    model: EmulatorModel,
    windows: WindowedArrays,
    scalers: WindowScalers,
    *,
    batch_size: int,
    target_mode: TargetMode,
    model_kind: ModelKind = "deterministic",
    num_particles: int = 1,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(1234)
    for metadata, inputs, initial_temperature, target in minibatches(
        windows,
        batch_size=batch_size,
        rng=rng,
        shuffle=False,
    ):
        key, batch_key = jax.random.split(key)
        pred = predict_mean_batch(
            model,
            jnp.asarray(metadata),
            jnp.asarray(inputs),
            jnp.asarray(initial_temperature),
            target_mode=target_mode,
            model_kind=model_kind,
            key=batch_key,
            num_particles=num_particles,
            sample_process_noise=False,
        )
        predictions.append(np.asarray(pred))
        targets.append(target)

    pred_norm = np.concatenate(predictions, axis=0)
    target_norm = np.concatenate(targets, axis=0)
    normalized = regression_metrics(pred_norm, target_norm)
    physical = regression_metrics(
        inverse_target(pred_norm, scalers),
        inverse_target(target_norm, scalers),
    )
    return {
        "loss": normalized.rmse**2,
        "rmse_c": physical.rmse,
        "mae_c": physical.mae,
        "nmae": physical.nmae,
        "bias_c": physical.bias,
    }


def resolve_checkpoint_metric(
    metric: CheckpointMetric,
    *,
    has_test_windows: bool,
) -> CheckpointMetric:
    if metric == "auto":
        return "test_rmse_c" if has_test_windows else "train_rmse_c"
    if metric == "test_rmse_c" and not has_test_windows:
        raise ValueError("checkpoint_metric='test_rmse_c' requires a non-empty test split")
    return metric


def checkpoint_metric_value(
    metric: CheckpointMetric,
    train_eval: dict[str, float],
    test_eval: dict[str, float] | None,
) -> float:
    if metric == "train_rmse_c":
        return train_eval["rmse_c"]
    if metric == "test_rmse_c" and test_eval is not None:
        return test_eval["rmse_c"]
    raise ValueError(f"Cannot compute checkpoint metric {metric!r}")


def target_alignment_offset(target_alignment: WindowTargetAlignment) -> int:
    if target_alignment == "same_time":
        return 0
    if target_alignment == "next_step":
        return 1
    raise ValueError("target_alignment must be 'same_time' or 'next_step'")


def full_profile_arrays(
    profile: BuildingProfile,
    target_alignment: WindowTargetAlignment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return inputs, target, datetimes, and initial temperature for a continuous rollout."""
    offset = target_alignment_offset(target_alignment)
    if offset == 0:
        return profile.inputs, profile.target, profile.datetime, profile.target[0]
    if profile.target.shape[0] <= offset:
        raise ValueError(f"Profile {profile.profile_id} is too short for target_alignment={target_alignment!r}")
    return (
        profile.inputs[:-offset],
        profile.target[offset:],
        profile.datetime[offset:],
        profile.target[0],
    )


def predict_full_profile(
    model: EmulatorModel,
    profile: BuildingProfile,
    scalers: WindowScalers,
    target_mode: TargetMode,
    *,
    target_alignment: WindowTargetAlignment = "same_time",
    model_kind: ModelKind = "deterministic",
    key: jax.Array | None = None,
    num_particles: int = 1,
) -> np.ndarray:
    """Run one continuous rollout over a complete profile."""
    profile_inputs, _, _, initial_temperature_c = full_profile_arrays(profile, target_alignment)
    metadata = scalers.metadata.transform(profile.metadata)
    inputs = scalers.inputs.transform(profile_inputs)
    initial_temperature = scalers.target.transform(initial_temperature_c)
    prediction = predict_mean_batch(
        model,
        jnp.asarray(metadata)[jnp.newaxis, :],
        jnp.asarray(inputs)[jnp.newaxis, :, :],
        jnp.asarray(initial_temperature)[jnp.newaxis, :],
        target_mode=target_mode,
        model_kind=model_kind,
        key=jax.random.PRNGKey(0) if key is None else key,
        num_particles=num_particles,
        sample_process_noise=False,
    )
    return inverse_target(np.asarray(prediction[0]), scalers)


def sample_full_profile_scenarios(
    model: ProbabilisticStableStateSpaceEmulator,
    profile: BuildingProfile,
    scalers: WindowScalers,
    target_mode: TargetMode,
    *,
    target_alignment: WindowTargetAlignment = "same_time",
    key: jax.Array,
    num_particles: int,
) -> np.ndarray:
    """Sample full-profile probabilistic scenarios in physical temperature units."""
    profile_inputs, _, _, initial_temperature_c = full_profile_arrays(profile, target_alignment)
    metadata = scalers.metadata.transform(profile.metadata)
    inputs = scalers.inputs.transform(profile_inputs)
    initial_temperature = scalers.target.transform(initial_temperature_c)
    raw_predictions = model.sample(
        jnp.asarray(metadata),
        jnp.asarray(inputs),
        jnp.asarray(initial_temperature),
        key=key,
        num_particles=num_particles,
        sample_process_noise=True,
    )
    predictions = jax.vmap(
        lambda raw_prediction: reconstruct_temperature(
            raw_prediction,
            jnp.asarray(initial_temperature),
            target_mode,
        )
    )(raw_predictions)
    return inverse_target(np.asarray(predictions), scalers)


def evaluate_full_profiles(
    model: EmulatorModel,
    profiles: list[BuildingProfile],
    scalers: WindowScalers,
    target_mode: TargetMode,
    *,
    target_alignment: WindowTargetAlignment = "same_time",
    model_kind: ModelKind = "deterministic",
    num_particles: int = 1,
) -> dict[str, float]:
    """Evaluate one continuous rollout per full profile."""
    if not profiles:
        return {
            "profile_count": 0.0,
            "rmse_c": float("nan"),
            "mae_c": float("nan"),
            "nmae": float("nan"),
            "bias_c": float("nan"),
        }

    predictions = []
    targets = []
    key = jax.random.PRNGKey(5678)
    for profile in profiles:
        key, profile_key = jax.random.split(key)
        predictions.append(
            predict_full_profile(
                model,
                profile,
                scalers,
                target_mode,
                target_alignment=target_alignment,
                model_kind=model_kind,
                key=profile_key,
                num_particles=num_particles,
            )
        )
        _, profile_target, _, _ = full_profile_arrays(profile, target_alignment)
        targets.append(profile_target)

    metrics = regression_metrics(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
    )
    return {
        "profile_count": float(len(profiles)),
        "rmse_c": metrics.rmse,
        "mae_c": metrics.mae,
        "nmae": metrics.nmae,
        "bias_c": metrics.bias,
    }


def _profile_by_id(profiles: list[BuildingProfile]) -> dict[int, BuildingProfile]:
    return {profile.profile_id: profile for profile in profiles}


def _evenly_spaced_indices(count: int, limit: int) -> list[int]:
    if count <= 0 or limit <= 0:
        return []
    n = min(count, limit)
    return [int(index) for index in np.linspace(0, limit - 1, n, dtype=np.int64)]


def _prediction_metrics_text(prediction: np.ndarray, target: np.ndarray) -> str:
    metrics = regression_metrics(prediction, target)
    return (
        f"RMSE={metrics.rmse:.3f} degC, "
        f"MAE={metrics.mae:.3f} degC, "
        f"bias={metrics.bias:.3f} degC"
    )


def _write_temperature_comparison(
    *,
    path: Path,
    title: str,
    datetimes: np.ndarray,
    simulated: np.ndarray,
    emulated: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    interval_label: str = "95% scenario interval",
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=datetimes,
            y=simulated.ravel(),
            mode="lines",
            name="Simulated",
            line={"color": "#1f77b4"},
        )
    )
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=datetimes,
                y=upper.ravel(),
                mode="lines",
                line={"width": 0, "color": "rgba(214, 39, 40, 0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=datetimes,
                y=lower.ravel(),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(214, 39, 40, 0.18)",
                line={"width": 0, "color": "rgba(214, 39, 40, 0)"},
                name=interval_label,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scattergl(
            x=datetimes,
            y=emulated.ravel(),
            mode="lines",
            name="Emulated",
            line={"color": "#d62728"},
        )
    )
    fig.update_layout(
        title=f"{title}<br><sup>{_prediction_metrics_text(emulated, simulated)}</sup>",
        xaxis_title="Time",
        yaxis_title="Indoor temperature [degC]",
        template="plotly_white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    fig.write_html(path, include_plotlyjs=True)


def save_prediction_visualizations(
    model: EmulatorModel,
    test_windows: WindowedArrays,
    test_profiles: list[BuildingProfile],
    scalers: WindowScalers,
    output_dir: Path,
    target_mode: TargetMode,
    num_window_plots: int,
    num_full_profile_plots: int,
    *,
    target_alignment: WindowTargetAlignment = "same_time",
    model_kind: ModelKind = "deterministic",
    num_particles: int = 1,
    prob_plot_particles: int = 100,
) -> dict[str, Path]:
    """Save fixed-window and full-profile rollout plots."""
    if test_windows.targets.shape[0] == 0 or not test_profiles:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_by_id = _profile_by_id(test_profiles)
    paths: dict[str, Path] = {}
    key = jax.random.PRNGKey(91011)
    filename_suffix = "_prob" if model_kind == "probabilistic" else ""

    for plot_number, window_index in enumerate(
        _evenly_spaced_indices(num_window_plots, test_windows.targets.shape[0]),
        start=1,
    ):
        key, window_key = jax.random.split(key)
        window_profile_id = int(test_windows.profile_ids[window_index])
        window_start = int(test_windows.start_indices[window_index])
        window_profile = profiles_by_id[window_profile_id]
        window_prediction_norm = predict_mean_batch(
            model,
            jnp.asarray(test_windows.metadata[window_index])[jnp.newaxis, :],
            jnp.asarray(test_windows.inputs[window_index])[jnp.newaxis, :, :],
            jnp.asarray(test_windows.initial_temperature[window_index])[jnp.newaxis, :],
            target_mode=target_mode,
            model_kind=model_kind,
            key=window_key,
            num_particles=num_particles,
            sample_process_noise=False,
        )
        window_prediction = inverse_target(np.asarray(window_prediction_norm[0]), scalers)
        window_target = inverse_target(test_windows.targets[window_index], scalers)
        target_offset = target_alignment_offset(target_alignment)
        window_end = window_start + window_prediction.shape[0]
        window_path = (
            output_dir
            / (
                f"test_window_{plot_number:02d}_profile_{window_profile_id}"
                f"_start_{window_start}{filename_suffix}.html"
            )
        )
        _write_temperature_comparison(
            path=window_path,
            title=(
                f"{window_prediction.shape[0]}-step test rollout, "
                f"profile {window_profile_id}, start {window_start}, alignment {target_alignment}"
            ),
            datetimes=window_profile.datetime[window_start + target_offset : window_end + target_offset],
            simulated=window_target,
            emulated=window_prediction,
        )
        paths[f"test_window_{plot_number:02d}"] = window_path

    for plot_number, profile_index in enumerate(
        _evenly_spaced_indices(num_full_profile_plots, len(test_profiles)),
        start=1,
    ):
        key, profile_key = jax.random.split(key)
        full_profile = test_profiles[profile_index]
        lower = None
        upper = None
        interval_label = "95% scenario interval"
        if model_kind == "probabilistic":
            assert isinstance(model, ProbabilisticStableStateSpaceEmulator)
            scenarios = sample_full_profile_scenarios(
                model,
                full_profile,
                scalers,
                target_mode,
                target_alignment=target_alignment,
                key=profile_key,
                num_particles=prob_plot_particles,
            )
            full_prediction = np.mean(scenarios, axis=0)
            lower = np.quantile(scenarios, 0.025, axis=0)
            upper = np.quantile(scenarios, 0.975, axis=0)
            interval_label = f"95% scenario interval ({prob_plot_particles} samples)"
        else:
            full_prediction = predict_full_profile(
                model,
                full_profile,
                scalers,
                target_mode,
                target_alignment=target_alignment,
                model_kind=model_kind,
                key=profile_key,
                num_particles=num_particles,
            )
        _, full_target, full_datetimes, _ = full_profile_arrays(full_profile, target_alignment)
        full_path = (
            output_dir
            / f"test_full_profile_{plot_number:02d}_{full_profile.profile_id}{filename_suffix}.html"
        )
        _write_temperature_comparison(
            path=full_path,
            title=(
                f"Full-profile continuous rollout, profile {full_profile.profile_id}, "
                f"alignment {target_alignment}"
            ),
            datetimes=full_datetimes,
            simulated=full_target,
            emulated=full_prediction,
            lower=lower,
            upper=upper,
            interval_label=interval_label,
        )
        paths[f"full_profile_{plot_number:02d}"] = full_path

    return paths


def sample_spectral_radius(
    model: EmulatorModel,
    windows: WindowedArrays,
    n: int = 32,
) -> float:
    count = min(n, windows.metadata.shape[0])
    radii = []
    for metadata in windows.metadata[:count]:
        if getattr(model, "switching_dynamics", "none") != "none":
            off_matrices, on_matrices = model.regime_matrices(jnp.asarray(metadata))
            radii.append(float(spectral_radius(off_matrices.a)))
            radii.append(float(spectral_radius(on_matrices.a)))
        else:
            matrices = model.matrices(jnp.asarray(metadata))
            radii.append(float(spectral_radius(matrices.a)))
    return float(np.max(radii)) if radii else float("nan")


def input_encoder_feedback_scaler_kwargs(
    input_columns: Sequence[str],
    metadata_columns: Sequence[str],
    scalers: WindowScalers,
    *,
    require_thermal_gaps: bool,
) -> dict[str, int | float]:
    """Build static conversion constants for thermal feedback encoder features."""
    try:
        outdoor_temperature_input_index = list(input_columns).index(DISTURBANCE_COLUMNS[0])
    except ValueError as exc:
        if require_thermal_gaps:
            raise ValueError(
                f"--input-encoder-feedback thermal_gaps requires input column {DISTURBANCE_COLUMNS[0]!r}"
            ) from exc
        outdoor_temperature_input_index = 0

    try:
        setpoint_metadata_index = list(metadata_columns).index(SETPOINT_METADATA_COLUMN)
    except ValueError as exc:
        if require_thermal_gaps:
            raise ValueError(
                f"--input-encoder-feedback thermal_gaps requires metadata column {SETPOINT_METADATA_COLUMN!r}"
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


def switching_dynamics_scaler_kwargs(
    input_columns: Sequence[str],
    scalers: WindowScalers,
    *,
    heat_on_threshold: float,
    alpha_heat_scale: float,
    alpha_on_weight: float,
    alpha_recent_weight: float,
) -> dict[str, int | float]:
    """Build constants that recover physical heat/regime signals from scaled inputs."""
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
        "heat_on_threshold": float(heat_on_threshold),
        "switching_alpha_heat_scale": float(alpha_heat_scale),
        "switching_alpha_on_weight": float(alpha_on_weight),
        "switching_alpha_recent_weight": float(alpha_recent_weight),
    }


def run_training(config: TrainConfig) -> EmulatorModel:
    if config.model_kind not in ("deterministic", "probabilistic"):
        raise ValueError("model_kind must be 'deterministic' or 'probabilistic'")
    if config.monotonicity_weight < 0.0:
        raise ValueError("monotonicity_weight must be non-negative")
    if config.target_mode not in ("absolute", "delta", "residual"):
        raise ValueError("target_mode must be 'absolute', 'delta', or 'residual'")
    if config.heat_input_normalization not in ("raw", "per_floor_area"):
        raise ValueError("heat_input_normalization must be 'raw' or 'per_floor_area'")
    if config.input_feature_mode not in ("base", "heating_regime"):
        raise ValueError("input_feature_mode must be 'base' or 'heating_regime'")
    if config.heating_regime_window_steps < 1:
        raise ValueError("heating_regime_window_steps must be positive")
    if config.heat_on_threshold < 0.0:
        raise ValueError("heat_on_threshold must be non-negative")
    if config.target_alignment not in ("same_time", "next_step"):
        raise ValueError("target_alignment must be 'same_time' or 'next_step'")
    if config.output_timing not in ("pre_update", "post_update"):
        raise ValueError("output_timing must be 'pre_update' or 'post_update'")
    if config.switching_dynamics not in ("none", "heating", "heating_bias"):
        raise ValueError("switching_dynamics must be 'none', 'heating', or 'heating_bias'")
    if config.switching_alpha_heat_scale <= 0.0:
        raise ValueError("switching_alpha_heat_scale must be positive")
    if config.schur_mode not in ("dense", "near_identity", "pf"):
        raise ValueError("schur_mode must be 'dense', 'near_identity', or 'pf'")
    if config.pf_lambda_min < 0.0:
        raise ValueError("pf_lambda_min must be non-negative")
    if config.pf_lambda_min >= config.schur_gamma:
        raise ValueError("pf_lambda_min must be smaller than schur_gamma")
    if config.input_encoder_dim is not None and config.input_encoder_dim < 1:
        raise ValueError("input_encoder_dim must be positive or None")
    if config.input_encoder_hidden_dim is not None and config.input_encoder_hidden_dim < 1:
        raise ValueError("input_encoder_hidden_dim must be positive or None")
    if config.input_encoder_depth < 1:
        raise ValueError("input_encoder_depth must be at least 1")
    if config.input_encoder_feedback not in ("none", "predicted_temperature", "thermal_gaps"):
        raise ValueError(
            "input_encoder_feedback must be 'none', 'predicted_temperature', or 'thermal_gaps'"
        )
    if config.input_encoder_feedback != "none" and not config.zero_d:
        raise ValueError("--input-encoder-feedback requires --zero-d to avoid direct-feedthrough loops")
    if config.input_encoder_feedback != "none" and config.target_mode != "absolute":
        raise ValueError(
            "--input-encoder-feedback currently requires --target-mode absolute"
        )
    if config.model_kind == "deterministic" and config.input_encoder_dim is not None and config.monotonicity_weight > 0.0:
        raise ValueError(
            "monotonicity regularization currently assumes linear raw inputs; "
            "disable --monotonicity-weight or omit --input-encoder-dim."
        )
    if config.model_kind == "deterministic" and config.input_encoder_feedback != "none" and config.monotonicity_weight > 0.0:
        raise ValueError("monotonicity regularization is not implemented with --input-encoder-feedback")
    if config.model_kind == "probabilistic" and config.monotonicity_weight > 0.0:
        raise ValueError("monotonicity regularization is not implemented for --model-kind probabilistic")
    if config.prob_particles < 1:
        raise ValueError("prob_particles must be positive")
    if config.prob_eval_particles < 1:
        raise ValueError("prob_eval_particles must be positive")
    if config.prob_plot_particles < 1:
        raise ValueError("prob_plot_particles must be positive")
    if config.prob_latent_dim < 1:
        raise ValueError("prob_latent_dim must be positive")
    if config.prob_process_noise not in ("none", "constant", "heteroscedastic"):
        raise ValueError("prob_process_noise must be 'none', 'constant', or 'heteroscedastic'")
    if config.prob_softopt_weight < 0.0:
        raise ValueError("prob_softopt_weight must be non-negative")
    if config.prob_softopt_temperature <= 0.0:
        raise ValueError("prob_softopt_temperature must be positive")
    if config.prob_variogram_weight < 0.0:
        raise ValueError("prob_variogram_weight must be non-negative")
    if config.prob_variogram_power <= 0.0:
        raise ValueError("prob_variogram_power must be positive")
    if config.prob_physics_weight < 0.0:
        raise ValueError("prob_physics_weight must be non-negative")
    if config.prob_horizon_weight_power < 0.0:
        raise ValueError("prob_horizon_weight_power must be non-negative")
    if any(lag < 1 for lag in config.prob_variogram_lags):
        raise ValueError("prob_variogram_lags must be positive")
    if config.checkpoint_metric not in ("auto", "train_rmse_c", "test_rmse_c"):
        raise ValueError("checkpoint_metric must be 'auto', 'train_rmse_c', or 'test_rmse_c'")
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if config.loss_normalization not in ("none", "window_std"):
        raise ValueError("loss_normalization must be 'none' or 'window_std'")
    if config.loss_std_floor_c <= 0.0:
        raise ValueError("loss_std_floor_c must be positive")
    if config.save_model_every_epochs < 0:
        raise ValueError("save_model_every_epochs must be non-negative")
    if config.num_window_plots < 0:
        raise ValueError("num_window_plots must be non-negative")
    if config.num_full_profile_plots < 0:
        raise ValueError("num_full_profile_plots must be non-negative")

    split_config = SplitConfig(
        dataset_path=config.dataset_path,
        max_profiles=config.max_profiles,
        test_fraction=config.test_fraction,
        seed=config.seed,
        heat_input_normalization=config.heat_input_normalization,  # type: ignore[arg-type]
        input_feature_mode=config.input_feature_mode,
        heating_regime_window_steps=config.heating_regime_window_steps,
    )
    splits = load_result_splits(split_config, heating_mode=config.heating_mode)
    window_config = WindowConfig(
        sequence_length=config.sequence_length,
        stride=config.stride,
        target_alignment=config.target_alignment,
    )
    monotonicity_horizon = config.monotonicity_horizon or config.sequence_length
    if monotonicity_horizon < 1:
        raise ValueError("monotonicity_horizon must be positive")
    monotonicity_feature_indices = resolve_monotonicity_feature_indices(
        splits.input_columns,
        config.monotonicity_features,
    )
    train_profiles = to_profiles(
        splits.train,
        splits.heating_mode,
        splits.heat_input_normalization,
        splits.input_feature_mode,
        splits.heating_regime_window_steps,
        config.heat_on_threshold,
    )
    test_profiles = (
        to_profiles(
            splits.test,
            splits.heating_mode,
            splits.heat_input_normalization,
            splits.input_feature_mode,
            splits.heating_regime_window_steps,
            config.heat_on_threshold,
        )
        if splits.test_ids
        else []
    )
    train_windows = make_windows(train_profiles, window_config)
    test_windows = None
    if splits.test_ids:
        test_windows = make_windows(test_profiles, window_config)
    resolved_checkpoint_metric = resolve_checkpoint_metric(
        config.checkpoint_metric,
        has_test_windows=test_windows is not None,
    )

    scalers = fit_window_scalers(train_windows)
    feedback_scaler_kwargs = input_encoder_feedback_scaler_kwargs(
        splits.input_columns,
        splits.metadata_columns,
        scalers,
        require_thermal_gaps=config.input_encoder_feedback == "thermal_gaps",
    )
    switching_scaler_kwargs = switching_dynamics_scaler_kwargs(
        splits.input_columns,
        scalers,
        heat_on_threshold=config.heat_on_threshold,
        alpha_heat_scale=config.switching_alpha_heat_scale,
        alpha_on_weight=config.switching_alpha_on_weight,
        alpha_recent_weight=config.switching_alpha_recent_weight,
    )
    loss_std_floor = jnp.asarray(
        config.loss_std_floor_c / np.asarray(scalers.target.scale, dtype=np.float32),
        dtype=jnp.float32,
    )
    train_windows = transform_windows(train_windows, scalers)
    if test_windows is not None:
        test_windows = transform_windows(test_windows, scalers)

    key = jax.random.PRNGKey(config.seed)
    if config.model_kind == "deterministic":
        model: EmulatorModel = MetadataStateSpaceEmulator(
            metadata_dim=train_windows.metadata.shape[-1],
            input_dim=train_windows.inputs.shape[-1],
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            input_encoder_dim=config.input_encoder_dim,
            input_encoder_hidden_dim=config.input_encoder_hidden_dim,
            input_encoder_depth=config.input_encoder_depth,
            input_encoder_feedback=config.input_encoder_feedback,
            zero_d=config.zero_d,
            output_timing=config.output_timing,
            switching_dynamics=config.switching_dynamics,
            **switching_scaler_kwargs,
            **feedback_scaler_kwargs,
            schur_gamma=config.schur_gamma,
            pf_lambda_min=config.pf_lambda_min,
            schur_mode=config.schur_mode,  # type: ignore[arg-type]
            key=key,
        )
    else:
        encoded_input_dim = config.input_encoder_dim or train_windows.inputs.shape[-1]
        model = ProbabilisticStableStateSpaceEmulator(
            metadata_dim=train_windows.metadata.shape[-1],
            input_dim=train_windows.inputs.shape[-1],
            state_dim=config.state_dim,
            encoded_input_dim=encoded_input_dim,
            latent_dim=config.prob_latent_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            input_encoder_hidden_dim=config.input_encoder_hidden_dim,
            input_encoder_depth=config.input_encoder_depth,
            input_encoder_feedback=config.input_encoder_feedback,
            process_noise_mode=config.prob_process_noise,
            process_noise_init=config.prob_process_noise_init,
            zero_d=config.zero_d,
            output_timing=config.output_timing,
            switching_dynamics=config.switching_dynamics,
            **switching_scaler_kwargs,
            **feedback_scaler_kwargs,
            schur_gamma=config.schur_gamma,
            pf_lambda_min=config.pf_lambda_min,
            schur_mode=config.schur_mode,  # type: ignore[arg-type]
            key=key,
        )
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(config.seed)
    model_checkpoint_dir = config.model_checkpoint_dir or config.output_dir / "model_checkpoints"

    print(f"heating_mode={splits.heating_mode}")
    print(f"heat_input_normalization={splits.heat_input_normalization}")
    print(
        "input_feature_mode="
        f"{splits.input_feature_mode} "
        f"heating_regime_window_steps={splits.heating_regime_window_steps} "
        f"heat_on_threshold={config.heat_on_threshold}"
    )
    print(f"model_kind={config.model_kind}")
    print(f"target_mode={config.target_mode}")
    print(f"target_alignment={config.target_alignment}")
    print(f"zero_d={config.zero_d}")
    print(f"output_timing={config.output_timing}")
    print(f"input_encoder_feedback={config.input_encoder_feedback}")
    print(
        "switching_dynamics="
        f"{config.switching_dynamics} "
        f"alpha_heat_scale={config.switching_alpha_heat_scale} "
        f"alpha_on_weight={config.switching_alpha_on_weight} "
        f"alpha_recent_weight={config.switching_alpha_recent_weight}"
    )
    if config.switching_dynamics != "none":
        print(
            "switching_alpha_inputs="
            f"heat_index={switching_scaler_kwargs['heat_input_index']} "
            f"heating_on_index={switching_scaler_kwargs['heating_on_input_index']} "
            f"recently_on_index={switching_scaler_kwargs['recently_on_input_index']}"
        )
    if config.input_encoder_feedback == "thermal_gaps":
        print(
            "thermal_gap_features="
            "[y_pred, outdoor_temperature-y_pred, shSetpoint-y_pred] "
            f"outdoor_input_index={feedback_scaler_kwargs['outdoor_temperature_input_index']} "
            f"setpoint_metadata_index={feedback_scaler_kwargs['setpoint_metadata_index']}"
        )
    print(
        "loss_normalization="
        f"{config.loss_normalization} "
        f"std_floor_c={config.loss_std_floor_c}"
    )
    print(
        f"schur_mode={config.schur_mode} "
        f"schur_gamma={config.schur_gamma} "
        f"pf_lambda_min={config.pf_lambda_min}"
    )
    print(f"input_columns={list(splits.input_columns)}")
    if config.model_kind == "probabilistic" and config.schur_mode == "dense":
        print(
            "warning=probabilistic_dense_schur_can_initialize_almost_memoryless; "
            "near_identity is usually a stronger prior for 15-minute building thermal dynamics"
        )
    if config.model_kind == "probabilistic":
        print(
            "probabilistic_model=enabled "
            f"particles={config.prob_particles} "
            f"eval_particles={config.prob_eval_particles} "
            f"plot_particles={config.prob_plot_particles} "
            f"latent_dim={config.prob_latent_dim} "
            f"process_noise={config.prob_process_noise}"
        )
        print(
            "probabilistic_loss "
            "energy_score_weight=1.0 "
            f"variogram_weight={config.prob_variogram_weight} "
            f"variogram_lags={list(config.prob_variogram_lags)} "
            f"softopt_weight={config.prob_softopt_weight} "
            f"softopt_temperature={config.prob_softopt_temperature} "
            f"physics_weight={config.prob_physics_weight}"
        )
        print(
            "input_encoder=enabled "
            f"encoded_dim={config.input_encoder_dim or train_windows.inputs.shape[-1]} "
            f"hidden_dim={config.input_encoder_hidden_dim or config.hidden_dim} "
            f"depth={config.input_encoder_depth}"
        )
    elif config.input_encoder_dim is not None:
        print(
            "input_encoder=enabled "
            f"encoded_dim={config.input_encoder_dim} "
            f"hidden_dim={config.input_encoder_hidden_dim or config.hidden_dim} "
            f"depth={config.input_encoder_depth}"
        )
    else:
        print("input_encoder=disabled")
    print(f"train_profiles={len(splits.train_ids)} test_profiles={len(splits.test_ids)}")
    print(f"train_windows={train_windows.targets.shape[0]}")
    if test_windows is not None:
        print(f"test_windows={test_windows.targets.shape[0]}")
    print(f"checkpoint_metric={resolved_checkpoint_metric}")
    if config.early_stopping_patience is not None:
        print(
            "early_stopping=enabled "
            f"patience={config.early_stopping_patience} "
            f"min_delta={config.early_stopping_min_delta}"
        )
    if config.save_model or config.save_model_every_epochs > 0:
        print(f"model_checkpoint_dir={model_checkpoint_dir}")
    if config.save_model:
        print("save_selected_model=enabled")
    if config.save_model_every_epochs > 0:
        print(f"save_model_every_epochs={config.save_model_every_epochs}")
    if config.monotonicity_weight > 0.0:
        selected_columns = [splits.input_columns[index] for index in monotonicity_feature_indices]
        print(
            "monotonicity_regularization=enabled "
            f"weight={config.monotonicity_weight} "
            f"horizon={monotonicity_horizon} "
            f"input_columns={selected_columns}"
        )

    best_model = model
    best_epoch = 0
    best_metric = float("inf")
    best_train_eval: dict[str, float] | None = None
    best_test_eval: dict[str, float] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        losses = []
        for batch_idx, (metadata, inputs, initial_temperature, targets) in enumerate(
            minibatches(train_windows, batch_size=config.batch_size, rng=rng, shuffle=True),
            start=1,
        ):
            if config.model_kind == "deterministic":
                assert isinstance(model, MetadataStateSpaceEmulator)
                model, opt_state, loss = train_step(
                    model,
                    opt_state,
                    optimizer,
                    jnp.asarray(metadata),
                    jnp.asarray(inputs),
                    jnp.asarray(initial_temperature),
                    jnp.asarray(targets),
                    config.monotonicity_weight,
                    monotonicity_feature_indices,
                    monotonicity_horizon,
                    config.target_mode,
                    config.loss_normalization,
                    loss_std_floor,
                )
            else:
                assert isinstance(model, ProbabilisticStableStateSpaceEmulator)
                key, step_key = jax.random.split(key)
                model, opt_state, loss = probabilistic_train_step(
                    model,
                    opt_state,
                    optimizer,
                    jnp.asarray(metadata),
                    jnp.asarray(inputs),
                    jnp.asarray(initial_temperature),
                    jnp.asarray(targets),
                    step_key,
                    config.prob_particles,
                    config.target_mode,
                    config.prob_softopt_weight,
                    config.prob_softopt_temperature,
                    config.prob_variogram_weight,
                    config.prob_variogram_lags,
                    config.prob_variogram_power,
                    config.prob_physics_weight,
                    config.prob_horizon_weight_power,
                    config.loss_normalization,
                    loss_std_floor,
                )
            losses.append(float(loss))
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break

        train_eval = evaluate(
            model,
            train_windows,
            scalers,
            batch_size=config.batch_size,
            target_mode=config.target_mode,
            model_kind=config.model_kind,
            num_particles=config.prob_eval_particles,
        )
        message = (
            f"epoch={epoch:03d} train_loss={np.mean(losses):.6f} "
            f"train_rmse_c={train_eval['rmse_c']:.4f} "
            f"train_nmae={train_eval['nmae']:.4f} "
            f"rho_max={sample_spectral_radius(model, train_windows):.5f}"
        )
        test_eval: dict[str, float] | None = None
        if test_windows is not None:
            test_eval = evaluate(
                model,
                test_windows,
                scalers,
                batch_size=config.batch_size,
                target_mode=config.target_mode,
                model_kind=config.model_kind,
                num_particles=config.prob_eval_particles,
            )
            message += f" test_rmse_c={test_eval['rmse_c']:.4f} test_nmae={test_eval['nmae']:.4f}"
        metric_value = checkpoint_metric_value(
            resolved_checkpoint_metric,
            train_eval,
            test_eval if test_windows is not None else None,
        )
        if metric_value < best_metric - config.early_stopping_min_delta:
            best_model = model
            best_epoch = epoch
            best_metric = metric_value
            best_train_eval = train_eval
            best_test_eval = test_eval if test_windows is not None else None
            epochs_without_improvement = 0
            message += f" checkpoint=best_{resolved_checkpoint_metric}"
        else:
            epochs_without_improvement += 1
            message += f" checkpoint_wait={epochs_without_improvement}"
        print(message)

        if config.save_model_every_epochs > 0 and epoch % config.save_model_every_epochs == 0:
            artifact_path = save_training_artifact(
                model_checkpoint_dir / f"epoch_{epoch:03d}",
                model=model,
                scalers=scalers,
                train_config=config,
                splits=splits,
                metadata_dim=train_windows.metadata.shape[-1],
                input_dim=train_windows.inputs.shape[-1],
                checkpoint_epoch=epoch,
                checkpoint_metric=resolved_checkpoint_metric,
                checkpoint_metric_value=metric_value,
                train_metrics=train_eval,
                test_metrics=test_eval,
            )
            print(f"saved_epoch_model={artifact_path}")

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                "early_stopped "
                f"epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_{resolved_checkpoint_metric}={best_metric:.4f}"
            )
            break

    model = best_model
    print(
        "selected_checkpoint "
        f"epoch={best_epoch} "
        f"{resolved_checkpoint_metric}={best_metric:.4f}"
    )
    if best_train_eval is not None:
        print(
            "selected_train_metrics "
            f"rmse_c={best_train_eval['rmse_c']:.4f} "
            f"nmae={best_train_eval['nmae']:.4f}"
        )
    if best_test_eval is not None:
        print(
            "selected_test_metrics "
            f"rmse_c={best_test_eval['rmse_c']:.4f} "
            f"nmae={best_test_eval['nmae']:.4f}"
        )
    if config.save_model:
        artifact_path = save_training_artifact(
            model_checkpoint_dir / "selected",
            model=model,
            scalers=scalers,
            train_config=config,
            splits=splits,
            metadata_dim=train_windows.metadata.shape[-1],
            input_dim=train_windows.inputs.shape[-1],
            checkpoint_epoch=best_epoch,
            checkpoint_metric=resolved_checkpoint_metric,
            checkpoint_metric_value=best_metric,
            train_metrics=best_train_eval,
            test_metrics=best_test_eval,
        )
        print(f"saved_selected_model={artifact_path}")

    if test_windows is not None:
        full_profile_eval = evaluate_full_profiles(
            model,
            test_profiles,
            scalers,
            config.target_mode,
            target_alignment=config.target_alignment,
            model_kind=config.model_kind,
            num_particles=config.prob_eval_particles,
        )
        print(
            "full_profile_test "
            f"profiles={int(full_profile_eval['profile_count'])} "
            f"rmse_c={full_profile_eval['rmse_c']:.4f} "
            f"mae_c={full_profile_eval['mae_c']:.4f} "
            f"nmae={full_profile_eval['nmae']:.4f} "
            f"bias_c={full_profile_eval['bias_c']:.4f}"
        )
        visualization_paths = save_prediction_visualizations(
            model,
            test_windows,
            test_profiles,
            scalers,
            config.output_dir,
            config.target_mode,
            config.num_window_plots,
            config.num_full_profile_plots,
            target_alignment=config.target_alignment,
            model_kind=config.model_kind,
            num_particles=config.prob_eval_particles,
            prob_plot_particles=config.prob_plot_particles,
        )
        for name, path in visualization_paths.items():
            print(f"saved_{name}_plot={path}")

    return model


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-profiles", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--heating-mode", choices=(*HEATING_INPUT_COLUMNS.keys(), "A", "B"), default="A")
    parser.add_argument(
        "--model-kind",
        choices=("deterministic", "probabilistic"),
        default="deterministic",
        help="Select the original deterministic SS model or the probabilistic stable SS model.",
    )
    parser.add_argument(
        "--heat-input-normalization",
        choices=("raw", "per_floor_area"),
        default="per_floor_area",
        help="Use selected heat input as raw W or divide it by floor_area to W/m2 before scaling.",
    )
    parser.add_argument(
        "--input-feature-mode",
        choices=("base", "heating_regime"),
        default="base",
        help=(
            "Additional input features derived from available signals. "
            "'heating_regime' appends heat_is_on and a rolling recently-on flag."
        ),
    )
    parser.add_argument(
        "--heating-regime-window-steps",
        type=int,
        default=96 * 7,
        help="Rolling window, in timesteps, for the heating_regime recently-on input feature.",
    )
    parser.add_argument(
        "--heat-on-threshold",
        type=float,
        default=1e-6,
        help="Heat threshold, in the selected heat unit, used by heating_regime features.",
    )
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument(
        "--target-alignment",
        choices=("same_time", "next_step"),
        default="same_time",
        help=(
            "'same_time' keeps the historical u[t] -> T[t] pairing. "
            "'next_step' trains u[t] with initial T[t] to predict T[t+1]."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "delta", "residual"),
        default="absolute",
        help=(
            "Output reconstruction mode: absolute, delta integration, or residual anchoring "
            "to the true initial temperature."
        ),
    )
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--input-encoder-dim",
        type=int,
        default=None,
        help=(
            "Enable nonlinear z[t]=e(u[t]) forcing with this encoded input dimension. "
            "Disabled by default, which keeps raw linear inputs."
        ),
    )
    parser.add_argument(
        "--input-encoder-hidden-dim",
        type=int,
        default=None,
        help="Hidden width for the input encoder MLP. Defaults to --hidden-dim.",
    )
    parser.add_argument(
        "--input-encoder-depth",
        type=int,
        default=2,
        help="Number of linear layers in the input encoder MLP.",
    )
    parser.add_argument(
        "--input-encoder-feedback",
        "--input-feedback",
        choices=("none", "predicted_temperature", "thermal_gaps"),
        default="none",
        help=(
            "Augment the forcing encoder input at each rollout step. "
            "'predicted_temperature' uses [u[t], y_pred[t]]. "
            "'thermal_gaps' uses [u[t], y_pred[t], Tout[t]-y_pred[t], Tset-y_pred[t]]. "
            "Feedback modes require --zero-d and --target-mode absolute."
        ),
    )
    parser.add_argument(
        "--zero-d",
        "--zero-D",
        dest="zero_d",
        action="store_true",
        help="Force the output direct-feedthrough matrix D to zero: y[t] = C x[t] + d.",
    )
    parser.add_argument(
        "--output-timing",
        choices=("pre_update", "post_update"),
        default="pre_update",
        help=(
            "Whether y[t] is emitted before or after applying u[t] to the state. "
            "Use post_update with --target-alignment next_step to test end-of-interval outputs."
        ),
    )
    parser.add_argument(
        "--switching-dynamics",
        choices=("none", "heating", "heating_bias"),
        default="none",
        help=(
            "Experimental switched/LPV dynamics. 'heating_bias' shares A and B and "
            "switches only b. 'heating' is the older full A/B/b switched variant."
        ),
    )
    parser.add_argument(
        "--switching-alpha-heat-scale",
        type=float,
        default=1.0,
        help="Positive smoothing scale, in selected heat units, for sigmoid((Q_heat-threshold)/scale).",
    )
    parser.add_argument(
        "--switching-alpha-on-weight",
        type=float,
        default=2.0,
        help="Score weight for the optional heat_is_on input feature when present.",
    )
    parser.add_argument(
        "--switching-alpha-recent-weight",
        type=float,
        default=1.0,
        help="Score weight for the optional recently_on input feature when present.",
    )
    parser.add_argument("--schur-gamma", type=float, default=0.995)
    parser.add_argument(
        "--pf-lambda-min",
        type=float,
        default=0.0,
        help=(
            "Lower bound for M_ij in --schur-mode pf. Keep at 0.0 to allow very fast "
            "thermal modes; must be smaller than --schur-gamma."
        ),
    )
    parser.add_argument("--schur-mode", choices=("dense", "near_identity", "pf"), default="near_identity")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator"))
    parser.add_argument(
        "--model-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for saved model artifacts. Defaults to <output-dir>/model_checkpoints.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the selected best model artifact for later ex-post analysis.",
    )
    parser.add_argument(
        "--save-model-every-epochs",
        type=int,
        default=0,
        help="Also save a model artifact every N epochs. Use 0 to disable periodic snapshots.",
    )
    parser.add_argument(
        "--num-window-plots",
        type=int,
        default=1,
        help="Number of held-out fixed-window HTML comparisons to save after training.",
    )
    parser.add_argument(
        "--num-full-profile-plots",
        type=int,
        default=1,
        help="Number of held-out full-profile HTML comparisons to save after training.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("auto", "train_rmse_c", "test_rmse_c"),
        default="auto",
        help="Metric used to select the best in-memory checkpoint. Auto prefers test_rmse_c when available.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without checkpoint-metric improvement. Disabled by default.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum checkpoint-metric improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=("none", "window_std"),
        default="none",
        help=(
            "Normalize training trajectories inside the loss. 'window_std' divides each "
            "prediction window by std(y_true) with --loss-std-floor-c as a stabilizing floor."
        ),
    )
    parser.add_argument(
        "--loss-std-floor-c",
        type=float,
        default=0.25,
        help="Minimum true-window temperature std, in degC, used by --loss-normalization window_std.",
    )
    parser.add_argument(
        "--monotonicity-weight",
        type=float,
        default=0.0,
        help="Weight for non-negative heat/outdoor/solar impulse-response regularization.",
    )
    parser.add_argument(
        "--monotonicity-horizon",
        type=int,
        default=None,
        help="Impulse-response horizon for monotonicity regularization. Defaults to sequence length.",
    )
    parser.add_argument(
        "--monotonicity-features",
        nargs="+",
        default=["heat", "outdoor_temperature", "solar"],
        help="Input features to constrain. Aliases include heat, outdoor_temperature, and solar.",
    )
    parser.add_argument("--prob-particles", type=int, default=8, help="Training particles for probabilistic mode.")
    parser.add_argument(
        "--prob-eval-particles",
        type=int,
        default=16,
        help="Particles used for probabilistic validation.",
    )
    parser.add_argument(
        "--prob-plot-particles",
        type=int,
        default=100,
        help="Scenarios sampled for full-profile probabilistic plot mean and 95%% interval.",
    )
    parser.add_argument(
        "--prob-latent-dim",
        type=int,
        default=4,
        help="Dimension of persistent xi sampled once per trajectory particle.",
    )
    parser.add_argument(
        "--prob-process-noise",
        choices=("none", "constant", "heteroscedastic"),
        default="constant",
        help="Process-noise model for probabilistic mode. Measurement noise is not used.",
    )
    parser.add_argument(
        "--prob-process-noise-init",
        type=float,
        default=-6.0,
        help="Initial raw process-noise scale; softplus(raw) is used.",
    )
    parser.add_argument(
        "--prob-softopt-weight",
        type=float,
        default=0.1,
        help="Weight for trajectory-level soft Best-of-K loss.",
    )
    parser.add_argument(
        "--prob-softopt-temperature",
        type=float,
        default=0.1,
        help="Temperature tau for soft Best-of-K aggregation.",
    )
    parser.add_argument(
        "--prob-variogram-weight",
        type=float,
        default=0.1,
        help="Weight for the lag-restricted variogram score.",
    )
    parser.add_argument(
        "--prob-variogram-lags",
        nargs="+",
        type=int,
        default=[1, 4, 16, 96],
        help="Timestep lags used in the efficient variogram score.",
    )
    parser.add_argument(
        "--prob-variogram-power",
        type=float,
        default=0.5,
        help="Power p used in the variogram score.",
    )
    parser.add_argument(
        "--prob-physics-weight",
        type=float,
        default=0.0,
        help="Weight for simple parameter regularization in probabilistic mode.",
    )
    parser.add_argument(
        "--prob-horizon-weight-power",
        type=float,
        default=0.0,
        help="Power for increasing horizon weights in soft Best-of-K trajectory errors.",
    )
    args = parser.parse_args()
    return TrainConfig(
        dataset_path=args.dataset,
        max_profiles=args.max_profiles,
        test_fraction=args.test_fraction,
        seed=args.seed,
        heating_mode=args.heating_mode,
        model_kind=args.model_kind,
        heat_input_normalization=args.heat_input_normalization,
        input_feature_mode=args.input_feature_mode,
        heating_regime_window_steps=args.heating_regime_window_steps,
        heat_on_threshold=args.heat_on_threshold,
        sequence_length=args.sequence_length,
        stride=args.stride,
        target_alignment=args.target_alignment,
        target_mode=args.target_mode,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        input_encoder_dim=args.input_encoder_dim,
        input_encoder_hidden_dim=args.input_encoder_hidden_dim,
        input_encoder_depth=args.input_encoder_depth,
        input_encoder_feedback=args.input_encoder_feedback,
        zero_d=args.zero_d,
        output_timing=args.output_timing,
        switching_dynamics=args.switching_dynamics,
        switching_alpha_heat_scale=args.switching_alpha_heat_scale,
        switching_alpha_on_weight=args.switching_alpha_on_weight,
        switching_alpha_recent_weight=args.switching_alpha_recent_weight,
        schur_gamma=args.schur_gamma,
        pf_lambda_min=args.pf_lambda_min,
        schur_mode=args.schur_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_batches=args.max_train_batches,
        output_dir=args.output_dir,
        model_checkpoint_dir=args.model_checkpoint_dir,
        save_model=args.save_model,
        save_model_every_epochs=args.save_model_every_epochs,
        num_window_plots=args.num_window_plots,
        num_full_profile_plots=args.num_full_profile_plots,
        checkpoint_metric=args.checkpoint_metric,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        loss_normalization=args.loss_normalization,
        loss_std_floor_c=args.loss_std_floor_c,
        monotonicity_weight=args.monotonicity_weight,
        monotonicity_horizon=args.monotonicity_horizon,
        monotonicity_features=tuple(args.monotonicity_features),
        prob_particles=args.prob_particles,
        prob_eval_particles=args.prob_eval_particles,
        prob_plot_particles=args.prob_plot_particles,
        prob_latent_dim=args.prob_latent_dim,
        prob_process_noise=args.prob_process_noise,
        prob_process_noise_init=args.prob_process_noise_init,
        prob_softopt_weight=args.prob_softopt_weight,
        prob_softopt_temperature=args.prob_softopt_temperature,
        prob_variogram_weight=args.prob_variogram_weight,
        prob_variogram_lags=tuple(args.prob_variogram_lags),
        prob_variogram_power=args.prob_variogram_power,
        prob_physics_weight=args.prob_physics_weight,
        prob_horizon_weight_power=args.prob_horizon_weight_power,
    )


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
