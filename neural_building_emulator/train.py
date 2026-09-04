"""Train a metadata-conditioned stable state-space building emulator."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    CLOSED_LOOP_CALENDAR_COLUMNS,
    CLOSED_LOOP_TARGET_COLUMNS,
    DISTURBANCE_COLUMNS,
    HEATING_INPUT_COLUMNS,
    HPPowerAreaNormalization,
    InputFeatureMode,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from .data import (
    BuildingProfile,
    ClosedLoopProfile,
    ClosedLoopWindowedArrays,
    DEFAULT_DATASET_PATH,
    SplitConfig,
    WindowConfig,
    WindowTargetAlignment,
    WindowedArrays,
    load_closed_loop_result_splits,
    load_result_splits,
    make_closed_loop_windows,
    make_windows,
    to_closed_loop_profiles,
    to_profiles,
)
from .metrics import regression_metrics
from .model_io import load_training_artifact, save_training_artifact
from .models import (
    ClosedLoopHPEmulator,
    ContractingClosedLoopHPEmulator,
    HPElectricRolloutMode,
    HPElectricScenarioMode,
    HPActivationModel,
    HPTrainingMode,
    MetadataStateSpaceEmulator,
    ProbabilisticClosedLoopHPEmulator,
    ProbabilisticContractingClosedLoopHPEmulator,
    ProbHpEmissionMode,
    ProbabilisticStableStateSpaceEmulator,
    SwitchingDynamics,
    ThermalQResponseMode,
    ThermostatDemandMode,
    spectral_radius,
)
from .models.state_space import OutputTiming
from .models.truncated_bptt import DEFAULT_BPTT_TRUNCATE_STEPS
from .scaling import (
    WindowScalers,
    fit_window_scalers,
    inverse_target,
    transform_closed_loop_windows,
    transform_closed_loop_profiles,
    transform_profiles,
    transform_windows,
)

TargetMode = Literal["absolute", "delta", "residual"]
CheckpointMetric = Literal[
    "auto",
    "train_rmse_c",
    "test_rmse_c",
    "train_total_nrmse",
    "test_total_nrmse",
]
LossNormalization = Literal["none", "window_std"]
InputEncoderFeedbackMode = Literal["none", "predicted_temperature", "thermal_gaps"]
ProbProcessNoiseMode = Literal["none", "constant", "heteroscedastic"]
ContractingTemperatureUpdateMode = Literal[
    "auto",
    "absolute",
    "delta",
    "leaky_equilibrium",
    "bounded_equilibrium",
    "alpha_bounded_equilibrium",
]
ContractingTransitionConditioning = Literal[
    "state_feedback",
    "exogenous",
    "exogenous_additive_feedback",
]
ClosedLoopStabilityMethod = Literal["exact_svd", "power_iteration"]
FlexibilityKpiControls = Literal["none", "weather", "full"]
ModelKind = Literal[
    "deterministic",
    "probabilistic",
    "closed_loop_hp",
    "closed_loop_hp_contracting",
    "closed_loop_hp_probabilistic",
    "closed_loop_hp_contracting_probabilistic",
]
EmulatorModel = (
    MetadataStateSpaceEmulator
    | ProbabilisticStableStateSpaceEmulator
    | ClosedLoopHPEmulator
    | ContractingClosedLoopHPEmulator
    | ProbabilisticClosedLoopHPEmulator
    | ProbabilisticContractingClosedLoopHPEmulator
)
ProbabilisticClosedLoopModel = (
    ProbabilisticClosedLoopHPEmulator | ProbabilisticContractingClosedLoopHPEmulator
)
SETPOINT_METADATA_COLUMN = "shSetpoint"
PROB_CLOSED_LOOP_LOSS_COMPONENT_NAMES = (
    "energy",
    "variogram",
    "ires",
    "flex_kpi_crps",
    "softopt",
    "hp_bce",
    "hp_active_nll",
    "hp_inactive_leakage",
    "physics",
    "stability",
)
UPDATE_FINITE_FLAG_NAMES = (
    "loss",
    "components",
    "grads",
    "updates",
    "opt_state",
    "params",
    "forward",
)
UPDATE_STAT_NAMES = (
    "grad_norm",
    "update_norm",
    "param_norm",
    "max_abs_update",
    "max_abs_param",
)
PROB_CLOSED_LOOP_AUX_ENERGY = 2
PROB_CLOSED_LOOP_AUX_LOG_MU = 8
PROB_CLOSED_LOOP_AUX_LOG_SIGMA = 9
PROB_CLOSED_LOOP_AUX_X_STATE = 10
PROB_CLOSED_LOOP_AUX_W_STATE = 11
PROB_CLOSED_LOOP_AUX_TEMPERATURE_STATE = 12
PROB_CLOSED_LOOP_AUX_XI = 13
PROB_CLOSED_LOOP_AUX_HP_P_START = 14
PROB_CLOSED_LOOP_AUX_HP_P_STOP = 15
PROB_CLOSED_LOOP_AUX_HP_LATENT = 16
PROB_CLOSED_LOOP_AUX_HP_POWER_HISTORY = 17
PROB_CLOSED_LOOP_AUX_HP_MODE_HISTORY = 18
PROB_CLOSED_LOOP_AUX_HP_PREVIOUS_MODE = 19
PROB_CLOSED_LOOP_AUX_Q_RESPONSE_STATE = 20


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    max_profiles: int | None = 10
    test_fraction: float = 0.2
    fixed_test_profile_ids: tuple[int, ...] = ()
    excluded_profile_ids: tuple[int, ...] = ()
    seed: int = 13
    heating_mode: str = "A"
    heat_input_normalization: str = "per_floor_area"
    hp_power_area_normalization: HPPowerAreaNormalization = "building_heated_area"
    input_feature_mode: InputFeatureMode = "base"
    heating_regime_window_steps: int = 96 * 7
    heat_on_threshold: float = 1e-6
    sequence_length: int = 96
    stride: int = 96
    rotate_window_starts: bool = True
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
    gradient_clip_norm: float = 1.0
    skip_grad_norm_above: float = 1e4
    skip_nonfinite_updates: bool = True
    max_consecutive_nonfinite_updates: int = 8
    validate_candidate_updates: bool = False
    log_update_diagnostics: bool = False
    max_train_batches: int | None = None
    train_eval_max_windows: int = 512
    full_train_eval_every_epochs: int = 5
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
    closed_loop_trajectory_output_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    prob_variogram_weight: float = 0.1
    prob_variogram_lags: tuple[int, ...] = (1, 4, 16, 96)
    prob_variogram_power: float = 0.5
    prob_variogram_output_weights: tuple[float, ...] = ()
    prob_variogram_constant_setpoint_only: bool = False
    prob_variogram_setpoint_threshold_c: float = 0.05
    prob_ires_weight: float = 0.0
    prob_ires_horizons_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    prob_ires_setpoint_threshold_c: float = 0.05
    prob_flex_kpi_crps_weight: float = 0.0
    prob_flex_kpi_crps_horizons_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    prob_flex_kpi_crps_setpoint_threshold_c: float = 0.05
    prob_flex_kpi_crps_min_events: int = 20
    prob_flex_kpi_crps_ridge: float = 1e-3
    prob_flex_kpi_crps_controls: FlexibilityKpiControls = "full"
    prob_physics_weight: float = 0.0
    prob_horizon_weight_power: float = 0.0
    hp_controller_state_dim: int = 2
    hp_controller_calendar_features: bool = True
    hp_thermostat_demand_mode: ThermostatDemandMode = "unconstrained"
    hp_thermostat_slope_min: float = 0.1
    hp_thermostat_slope_max: float = 6.0
    hp_thermostat_threshold_min_c: float = -1.0
    hp_thermostat_threshold_max_c: float = 1.0
    hp_dt_hours: float = 0.25
    hp_mode_loss_weight: float = 0.1
    hp_cop_floor: float = 1.0
    hp_cop_cap: float = 8.0
    hp_pel_cap_w_m2: float = 0.0
    hp_qroom_cap_w_m2: float = 0.0
    hp_energy_cap_wh_m2: float = 0.0
    hp_energy_cap_hours: float = 24.0
    hp_cap_factor: float = 1.25
    closed_loop_stability_weight: float = 0.0
    closed_loop_stability_gamma: float = 0.995
    closed_loop_stability_samples: int = 4
    closed_loop_stability_aggregation: Literal["mean", "max"] = "max"
    closed_loop_stability_every_steps: int = 8
    closed_loop_stability_method: ClosedLoopStabilityMethod = "power_iteration"
    closed_loop_stability_power_iterations: int = 4
    init_from_deterministic_artifact: Path | None = None
    init_xi_weight_scale: float = 0.05
    init_hp_active_log_sigma: float = 0.25
    contracting_gamma: float = 0.99
    contracting_state_bound: float = 5.0
    contracting_temperature_scale: float = 8.0
    contracting_temperature_delta_max_c: float = 0.0
    contracting_temperature_update: ContractingTemperatureUpdateMode = "auto"
    contracting_transition_conditioning: ContractingTransitionConditioning = "state_feedback"
    contracting_additive_feedback_gain_bound: float = 1.0
    contracting_q_to_t_mode: ThermalQResponseMode = "unconstrained"
    contracting_q_to_t_time_constants_hours: tuple[float, ...] = (
        0.25,
        1.0,
        4.0,
        16.0,
        24.0,
        48.0,
    )
    contracting_q_to_t_gain_min_c_per_w_m2: float = 0.01
    contracting_q_to_t_gain_max_c_per_w_m2: float = 2.0
    prob_hp_scenario_mode: HPElectricScenarioMode = "bernoulli"
    prob_hp_emission_mode: ProbHpEmissionMode = "bounded"
    prob_hp_training_mode: HPTrainingMode = "expected"
    prob_hp_concrete_temperature: float = 0.5
    prob_hp_activation_model: HPActivationModel = "independent"
    prob_hp_persistent_latent_dim: int = 2
    prob_hp_controller_leak: float = 0.25
    prob_hp_controller_noise_scale: float = 0.0
    prob_hp_history_hours: float = 3.0
    hp_active_power_nll_weight: float = 1.0
    hp_inactive_leakage_weight: float = 0.1
    bptt_truncate_steps: int = DEFAULT_BPTT_TRUNCATE_STEPS


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


def rotating_window_start_offset(epoch: int, epochs: int, stride: int) -> int:
    """Spread epoch window grids over one stride while keeping epoch 1 canonical."""
    if epoch < 1 or epochs < 1:
        raise ValueError("epoch and epochs must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    if epochs == 1 or stride == 1:
        return 0
    fraction = float(epoch - 1) / float(epochs - 1)
    return int(round(fraction * float(stride - 1)))


def fixed_evaluation_subset(
    windows: WindowedArrays | ClosedLoopWindowedArrays,
    *,
    max_windows: int,
    seed: int,
) -> WindowedArrays | ClosedLoopWindowedArrays:
    """Return one reproducible window subset for comparable per-epoch metrics."""
    n_windows = windows.targets.shape[0]
    if max_windows <= 0 or n_windows <= max_windows:
        return windows
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_windows, size=max_windows, replace=False))
    return replace(
        windows,
        profile_ids=windows.profile_ids[indices],
        start_indices=windows.start_indices[indices],
        metadata=windows.metadata[indices],
        inputs=windows.inputs[indices],
        targets=windows.targets[indices],
        initial_temperature=windows.initial_temperature[indices],
    )


def build_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    if config.gradient_clip_norm < 0.0:
        raise ValueError("gradient_clip_norm must be non-negative; use 0 to disable clipping")
    if config.max_consecutive_nonfinite_updates < 1:
        raise ValueError("max_consecutive_nonfinite_updates must be positive")
    transforms: list[optax.GradientTransformation] = []
    if config.gradient_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.gradient_clip_norm))
    transforms.append(optax.adam(config.learning_rate))
    optimizer = optax.chain(*transforms)
    if config.skip_nonfinite_updates:
        optimizer = optax.apply_if_finite(
            optimizer,
            max_consecutive_errors=config.max_consecutive_nonfinite_updates,
        )
    return optimizer


def _array_tree_all_finite(tree: object) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _array_tree_global_norm(tree: object) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return jnp.asarray(0.0)
    total = jnp.sum(
        jnp.stack(
            [
                jnp.sum(leaf.astype(jnp.float32) ** 2)
                for leaf in leaves
            ]
        )
    )
    return jnp.sqrt(total + jnp.asarray(1e-12, dtype=jnp.float32))


def _array_tree_max_abs(tree: object) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(leaf.astype(jnp.float32)))
                for leaf in leaves
            ]
        )
    )


def _select_array_tree(predicate: jnp.ndarray, new_tree: object, old_tree: object) -> object:
    new_arrays, new_static = eqx.partition(new_tree, eqx.is_array)
    old_arrays, _ = eqx.partition(old_tree, eqx.is_array)
    selected_arrays = jax.tree_util.tree_map(
        lambda new, old: jnp.where(predicate, new, old),
        new_arrays,
        old_arrays,
    )
    return eqx.combine(selected_arrays, new_static)


def _update_stats(*, grads: object, updates: object, model: object) -> jnp.ndarray:
    return jnp.stack(
        [
            _array_tree_global_norm(grads),
            _array_tree_global_norm(updates),
            _array_tree_global_norm(model),
            _array_tree_max_abs(updates),
            _array_tree_max_abs(model),
        ]
    )


def _update_finite_flags(
    *,
    loss: jnp.ndarray,
    components: object | None,
    grads: object,
    updates: object,
    opt_state: object,
    model: object,
    forward_is_finite: jnp.ndarray,
    max_grad_norm: float = 0.0,
) -> jnp.ndarray:
    components_are_finite = (
        jnp.asarray(True)
        if components is None
        else _array_tree_all_finite(components)
    )
    grads_ok = _array_tree_all_finite(grads)
    if max_grad_norm > 0.0:
        grad_norm = _array_tree_global_norm(grads)
        grads_ok = jnp.logical_and(
            grads_ok,
            jnp.isfinite(grad_norm) & (grad_norm <= jnp.asarray(max_grad_norm, dtype=jnp.float32)),
        )
    return jnp.stack(
        [
            jnp.isfinite(loss),
            components_are_finite,
            grads_ok,
            _array_tree_all_finite(updates),
            _array_tree_all_finite(opt_state),
            _array_tree_all_finite(model),
            forward_is_finite,
        ]
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
    output_weights: tuple[float, ...] = (),
) -> jnp.ndarray:
    """Return J[b, k] trajectory losses for particle predictions."""
    weights = normalized_output_loss_weights(
        predictions.shape[-1],
        output_weights,
        dtype=predictions.dtype,
    )
    sq_error = jnp.sum(
        (predictions - targets[:, jnp.newaxis, :, :]) ** 2
        * weights[jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
        axis=-1,
    )
    weights = _horizon_weights(predictions.shape[2], horizon_weight_power)
    return jnp.mean(sq_error * weights[jnp.newaxis, jnp.newaxis, :], axis=-1)


def soft_optimistic_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    temperature: float,
    horizon_weight_power: float,
    output_weights: tuple[float, ...] = (),
) -> jnp.ndarray:
    if temperature <= 0.0:
        raise ValueError("soft optimistic temperature must be positive")
    losses = trajectory_squared_errors(
        predictions,
        targets,
        horizon_weight_power=horizon_weight_power,
        output_weights=output_weights,
    )
    return jnp.mean(-temperature * _logmeanexp(-losses / temperature, axis=1))


def normalized_output_loss_weights(
    output_dim: int,
    output_weights: tuple[float, ...],
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return nonnegative channel weights with mean one."""
    if not output_weights:
        return jnp.ones((output_dim,), dtype=dtype)
    if len(output_weights) != output_dim:
        raise ValueError(f"Expected {output_dim} output weights, got {len(output_weights)}")
    if any(weight < 0.0 for weight in output_weights):
        raise ValueError("Output loss weights must be non-negative")
    if sum(output_weights) <= 0.0:
        raise ValueError("Output loss weights must contain at least one positive value")
    weights = jnp.asarray(output_weights, dtype=dtype)
    return weights * (output_dim / jnp.sum(weights))


def energy_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    output_weights: tuple[float, ...] = (),
) -> jnp.ndarray:
    """Trajectory-level energy score over particles."""
    weights = normalized_output_loss_weights(
        predictions.shape[-1],
        output_weights,
        dtype=predictions.dtype,
    )
    root_weights = jnp.sqrt(weights)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    centered = (predictions - targets[:, jnp.newaxis, :, :]) * root_weights
    obs_distance = jnp.sqrt(jnp.sum(centered**2, axis=(2, 3)) + 1e-6)
    pairwise = (
        predictions[:, :, jnp.newaxis, :, :]
        - predictions[:, jnp.newaxis, :, :, :]
    ) * root_weights[:, :, jnp.newaxis, :, :]
    pairwise_distance = jnp.sqrt(jnp.sum(pairwise**2, axis=(3, 4)) + 1e-6)
    return jnp.mean(jnp.mean(obs_distance, axis=1) - 0.5 * jnp.mean(pairwise_distance, axis=(1, 2)))


def smooth_abs_power(values: jnp.ndarray, power: float) -> jnp.ndarray:
    """Smooth |x|**power to avoid singular gradients at x=0 for power < 1."""
    return (values**2 + 1e-6) ** (0.5 * power)


def constant_setpoint_pair_mask(
    inputs: jnp.ndarray,
    *,
    lag: int,
    setpoint_delta_threshold: float,
) -> jnp.ndarray:
    """Mask lagged pairs whose interval does not cross a setpoint jump."""
    setpoint = inputs[:, :, 0]
    threshold = jnp.asarray(setpoint_delta_threshold, dtype=setpoint.dtype)
    jumps = (jnp.abs(setpoint[:, 1:] - setpoint[:, :-1]) > threshold).astype(setpoint.dtype)
    jump_count = jnp.concatenate(
        [jnp.zeros_like(jumps[:, :1]), jnp.cumsum(jumps, axis=1)],
        axis=1,
    )
    interval_jump_count = jump_count[:, lag:] - jump_count[:, :-lag]
    return (interval_jump_count <= 0.0).astype(setpoint.dtype)


def variogram_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    *,
    lags: tuple[int, ...],
    power: float,
    output_weights: tuple[float, ...] = (),
    constant_setpoint_inputs: jnp.ndarray | None = None,
    setpoint_delta_threshold: float = 0.0,
) -> jnp.ndarray:
    """Efficient lag-restricted trajectory variogram score."""
    scores = []
    horizon = predictions.shape[2]
    if output_weights:
        weights = jnp.asarray(output_weights, dtype=predictions.dtype)
    else:
        weights = jnp.ones((predictions.shape[-1],), dtype=predictions.dtype)
    weights = weights.reshape((1, 1, -1))
    weight_sum = jnp.sum(weights)
    zero = jnp.asarray(0.0, dtype=predictions.dtype)
    for lag in lags:
        if lag < 1 or lag >= horizon:
            continue
        target_delta = smooth_abs_power(targets[:, lag:, :] - targets[:, :-lag, :], power)
        prediction_delta = smooth_abs_power(
            predictions[:, :, lag:, :] - predictions[:, :, :-lag, :],
            power,
        )
        prediction_moment = jnp.mean(prediction_delta, axis=1)
        squared_error = (target_delta - prediction_moment) ** 2
        weighted_error = squared_error * weights
        if constant_setpoint_inputs is None:
            denominator = (
                jnp.asarray(squared_error.shape[0] * squared_error.shape[1], dtype=predictions.dtype)
                * weight_sum
            )
            scores.append(jnp.sum(weighted_error) / jnp.maximum(denominator, 1e-6))
        else:
            mask = constant_setpoint_pair_mask(
                constant_setpoint_inputs,
                lag=lag,
                setpoint_delta_threshold=setpoint_delta_threshold,
            )
            denominator = jnp.sum(mask) * weight_sum
            scores.append(
                jnp.where(
                    denominator > 0.0,
                    jnp.sum(weighted_error * mask[:, :, jnp.newaxis]) / jnp.maximum(denominator, 1e-6),
                    zero,
                )
            )
    if not scores:
        return zero
    return jnp.mean(jnp.stack(scores))


def _window_means_at_starts(
    values: jnp.ndarray,
    *,
    starts: jnp.ndarray,
    window_steps: int,
) -> jnp.ndarray:
    cumsum = jnp.concatenate(
        [jnp.zeros_like(values[..., :1]), jnp.cumsum(values, axis=-1)],
        axis=-1,
    )
    end_values = jnp.take(cumsum, starts + window_steps, axis=-1)
    start_values = jnp.take(cumsum, starts, axis=-1)
    return (end_values - start_values) / jnp.asarray(window_steps, dtype=values.dtype)


def intervention_response_energy_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    inputs: jnp.ndarray,
    *,
    horizon_steps: tuple[int, ...],
    horizon_hours: tuple[float, ...],
    setpoint_delta_threshold: float,
    target_channel: int = 2,
) -> jnp.ndarray:
    """Energy score on windowed pre/post responses at setpoint interventions.

    The functional scored for each event k and horizon H is

        H * (mean(P_after_H) - mean(P_before_H)).

    This is the energy version of the flexibility KPI response and is computed
    on normalized target units so its scale remains well behaved during
    training.
    """
    scores = []
    horizon = predictions.shape[2]
    setpoint = inputs[:, :, 0]
    target_power = targets[:, :, target_channel]
    prediction_power = predictions[:, :, :, target_channel]
    threshold = jnp.asarray(setpoint_delta_threshold, dtype=predictions.dtype)
    zero = jnp.asarray(0.0, dtype=predictions.dtype)

    for steps, hours in zip(horizon_steps, horizon_hours):
        if steps < 1 or 2 * steps > horizon:
            continue
        starts = jnp.arange(steps, horizon - steps + 1)
        setpoint_delta = jnp.take(setpoint, starts, axis=-1) - jnp.take(
            setpoint,
            starts - 1,
            axis=-1,
        )
        event_mask = (jnp.abs(setpoint_delta) >= threshold).astype(predictions.dtype)

        target_pre = _window_means_at_starts(
            target_power,
            starts=starts - steps,
            window_steps=steps,
        )
        target_post = _window_means_at_starts(
            target_power,
            starts=starts,
            window_steps=steps,
        )
        prediction_pre = _window_means_at_starts(
            prediction_power,
            starts=starts - steps,
            window_steps=steps,
        )
        prediction_post = _window_means_at_starts(
            prediction_power,
            starts=starts,
            window_steps=steps,
        )

        hours_value = jnp.asarray(hours, dtype=predictions.dtype)
        target_response = hours_value * (target_post - target_pre)
        prediction_response = hours_value * (prediction_post - prediction_pre)
        centered = prediction_response - target_response[:, jnp.newaxis, :]
        obs_distance = jnp.sqrt(centered**2 + 1e-6)
        pairwise = prediction_response[:, :, jnp.newaxis, :] - prediction_response[:, jnp.newaxis, :, :]
        pairwise_distance = jnp.sqrt(pairwise**2 + 1e-6)
        per_event_score = jnp.mean(obs_distance, axis=1) - 0.5 * jnp.mean(
            pairwise_distance,
            axis=(1, 2),
        )
        event_count = jnp.sum(event_mask)
        scores.append(
            jnp.where(
                event_count > 0.0,
                jnp.sum(per_event_score * event_mask) / jnp.maximum(event_count, 1.0),
                zero,
            )
        )
    if not scores:
        return zero
    return jnp.mean(jnp.stack(scores))


def _masked_standardize_controls(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Standardize fixed regression controls over valid intervention rows."""
    weights = mask[..., jnp.newaxis]
    count = jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1.0)
    mean = jnp.sum(values * weights, axis=1, keepdims=True) / count
    centered = values - mean
    variance = jnp.sum(centered**2 * weights, axis=1, keepdims=True) / count
    scale = jnp.maximum(jnp.sqrt(variance), 1e-4)
    return centered / scale


def _previous_setpoint_event_delta(
    setpoint_delta: jnp.ndarray,
    event_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return the previous qualifying setpoint jump at every timestep."""

    def scan_profile(deltas: jnp.ndarray, events: jnp.ndarray) -> jnp.ndarray:
        def step(previous: jnp.ndarray, values: tuple[jnp.ndarray, jnp.ndarray]):
            delta, is_event = values
            output = previous
            previous = jnp.where(is_event, delta, previous)
            return previous, output

        _, previous_deltas = jax.lax.scan(
            step,
            jnp.asarray(0.0, dtype=deltas.dtype),
            (deltas, events),
        )
        return previous_deltas

    return jax.vmap(scan_profile)(setpoint_delta, event_mask)


def _time_window_means(
    values: jnp.ndarray,
    *,
    starts: jnp.ndarray,
    window_steps: int,
) -> jnp.ndarray:
    """Window means for arrays whose time dimension is penultimate."""
    return jnp.swapaxes(
        _window_means_at_starts(
            jnp.swapaxes(values, -2, -1),
            starts=starts,
            window_steps=window_steps,
        ),
        -2,
        -1,
    )


def flexibility_coefficient_crps(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    horizon_steps: tuple[int, ...],
    horizon_hours: tuple[float, ...],
    setpoint_delta_threshold: float,
    min_events: int,
    ridge: float,
    controls: FlexibilityKpiControls,
    target_channel: int = 2,
) -> jnp.ndarray:
    """CRPS on fitted upward/downward event-study flexibility gains.

    For every batch window and horizon, one fixed regression design is built
    from observed interventions, disturbances, and (for ``full`` controls)
    observed pre-event state. The same ridge projection is then applied to the
    target and every particle's predicted Pel response. This makes the fitted
    coefficients linear in the predicted power trace and keeps backpropagation
    substantially cheaper than the recurrent rollout itself.

    The scored variables are ``H * beta_plus`` and ``H * beta_minus`` in
    normalized training units. This differs from the physical Wh/(m2 K) KPI by
    one constant scaler ratio and therefore preserves its calibration target.
    """
    scores = []
    rollout_horizon = predictions.shape[2]
    setpoint = inputs[:, :, 0]
    target_power = targets[:, :, target_channel]
    prediction_power = predictions[:, :, :, target_channel]
    threshold = jnp.asarray(setpoint_delta_threshold, dtype=predictions.dtype)
    zero = jnp.asarray(0.0, dtype=predictions.dtype)

    full_setpoint_delta = jnp.concatenate(
        [jnp.zeros_like(setpoint[:, :1]), setpoint[:, 1:] - setpoint[:, :-1]],
        axis=1,
    )
    full_event_mask = jnp.abs(full_setpoint_delta) >= threshold
    previous_event_delta = _previous_setpoint_event_delta(
        full_setpoint_delta,
        full_event_mask,
    )
    initial = jnp.reshape(initial_temperature, (initial_temperature.shape[0], -1))[:, :1]
    current_target_temperature = jnp.concatenate([initial, targets[:, :-1, 0]], axis=1)

    for steps, hours in zip(horizon_steps, horizon_hours):
        if steps < 1 or 2 * steps > rollout_horizon:
            continue
        starts = jnp.arange(steps, rollout_horizon - steps + 1)
        setpoint_delta = jnp.take(setpoint, starts, axis=1) - jnp.take(
            setpoint,
            starts - 1,
            axis=1,
        )
        event_mask = (jnp.abs(setpoint_delta) >= threshold).astype(predictions.dtype)

        target_pre = _window_means_at_starts(
            target_power,
            starts=starts - steps,
            window_steps=steps,
        )
        target_post = _window_means_at_starts(
            target_power,
            starts=starts,
            window_steps=steps,
        )
        prediction_pre = _window_means_at_starts(
            prediction_power,
            starts=starts - steps,
            window_steps=steps,
        )
        prediction_post = _window_means_at_starts(
            prediction_power,
            starts=starts,
            window_steps=steps,
        )
        target_response = target_post - target_pre
        prediction_response = prediction_post - prediction_pre

        columns = [
            jnp.ones_like(setpoint_delta),
            jnp.maximum(setpoint_delta, 0.0),
            jnp.minimum(setpoint_delta, 0.0),
        ]
        if controls != "none":
            event_disturbances = jnp.take(inputs[:, :, 1:], starts, axis=1)
            disturbance_values = inputs[:, :, 1:4]
            pre_disturbance = _time_window_means(
                disturbance_values,
                starts=starts - steps,
                window_steps=steps,
            )
            post_disturbance = _time_window_means(
                disturbance_values,
                starts=starts,
                window_steps=steps,
            )
            control_parts = [event_disturbances, post_disturbance - pre_disturbance]
            if controls == "full":
                pre_state = jnp.stack(
                    [
                        target_pre,
                        jnp.take(current_target_temperature, starts, axis=1),
                        jnp.take(setpoint, starts - 1, axis=1),
                        jnp.take(previous_event_delta, starts, axis=1),
                    ],
                    axis=-1,
                )
                control_parts.insert(0, pre_state)
            fixed_controls = _masked_standardize_controls(
                jnp.concatenate(control_parts, axis=-1),
                event_mask,
            )
            columns.extend(jnp.moveaxis(fixed_controls, -1, 0))

        design = jnp.stack(columns, axis=-1)
        design = jax.lax.stop_gradient(design)
        weighted_design = design * event_mask[..., jnp.newaxis]
        xtx = jnp.einsum("bep,beq->bpq", design, weighted_design)
        design_dim = design.shape[-1]
        ridge_diagonal = jnp.ones((design_dim,), dtype=predictions.dtype).at[0].set(0.0)
        penalty = (
            jnp.asarray(ridge, dtype=predictions.dtype) * jnp.diag(ridge_diagonal)
            + jnp.asarray(1e-6, dtype=predictions.dtype) * jnp.eye(design_dim)
        )
        xtx = xtx + penalty[jnp.newaxis, :, :]
        projection = jnp.linalg.solve(xtx, jnp.swapaxes(weighted_design, -2, -1))
        target_beta = jnp.einsum("bpe,be->bp", projection, target_response)
        prediction_beta = jnp.einsum("bpe,bke->bkp", projection, prediction_response)

        hours_value = jnp.asarray(hours, dtype=predictions.dtype)
        target_kpi = hours_value * target_beta[:, 1:3]
        prediction_kpi = hours_value * prediction_beta[:, :, 1:3]
        observation_delta = prediction_kpi - target_kpi[:, jnp.newaxis, :]
        observation_distance = jnp.sqrt(observation_delta**2 + 1e-6)
        pairwise_delta = (
            prediction_kpi[:, :, jnp.newaxis, :]
            - prediction_kpi[:, jnp.newaxis, :, :]
        )
        pairwise_distance = jnp.sqrt(pairwise_delta**2 + 1e-6)
        per_direction_crps = jnp.mean(observation_distance, axis=1) - 0.5 * jnp.mean(
            pairwise_distance,
            axis=(1, 2),
        )
        per_profile_score = jnp.mean(per_direction_crps, axis=-1)
        event_count = jnp.sum(event_mask, axis=1)
        valid = (
            (event_count >= jnp.asarray(min_events, dtype=predictions.dtype))
            & (event_count > jnp.asarray(design_dim, dtype=predictions.dtype))
        ).astype(predictions.dtype)
        valid_count = jnp.sum(valid)
        scores.append(
            jnp.where(
                valid_count > 0.0,
                jnp.sum(per_profile_score * valid) / jnp.maximum(valid_count, 1.0),
                zero,
            )
        )

    if not scores:
        return zero
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
    variogram_output_weights: tuple[float, ...],
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
            output_weights=variogram_output_weights,
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
    variogram_output_weights: tuple[float, ...],
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
        variogram_output_weights,
        physics_weight,
        horizon_weight_power,
        loss_normalization,
        loss_std_floor,
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def predict_closed_loop_batch(
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(model)(metadata, inputs, initial_temperature)


def _physical_target_component(
    targets: jnp.ndarray,
    scalers_target_mean: jnp.ndarray,
    scalers_target_scale: jnp.ndarray,
    index: int,
) -> jnp.ndarray:
    return targets[..., index] * scalers_target_scale[index] + scalers_target_mean[index]


@eqx.filter_value_and_grad
def closed_loop_loss_fn(
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    target_mean: jnp.ndarray,
    target_scale: jnp.ndarray,
    trajectory_output_weights: tuple[float, ...],
    hp_mode_loss_weight: float,
    heat_on_threshold: float,
) -> jnp.ndarray:
    predictions, aux = jax.vmap(
        lambda row_metadata, row_inputs, row_initial_temperature: model.rollout_with_aux(
            row_metadata,
            row_inputs,
            row_initial_temperature,
        )
    )(metadata, inputs, initial_temperature)
    output_weights = normalized_output_loss_weights(
        predictions.shape[-1],
        trajectory_output_weights,
        dtype=predictions.dtype,
    )
    mse = jnp.mean(
        (predictions - targets) ** 2
        * output_weights[jnp.newaxis, jnp.newaxis, :]
    )
    if hp_mode_loss_weight <= 0.0:
        return mse
    pi = aux[0]
    pel_target = _physical_target_component(targets, target_mean, target_scale, 2)
    mode_target = (pel_target > heat_on_threshold).astype(predictions.dtype)
    pi = jnp.clip(pi, 1e-5, 1.0 - 1e-5)
    bce = -jnp.mean(mode_target * jnp.log(pi) + (1.0 - mode_target) * jnp.log(1.0 - pi))
    return mse + hp_mode_loss_weight * bce


@eqx.filter_jit
def closed_loop_train_step(
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    target_mean: jnp.ndarray,
    target_scale: jnp.ndarray,
    trajectory_output_weights: tuple[float, ...],
    hp_mode_loss_weight: float,
    heat_on_threshold: float,
    skip_nonfinite_updates: bool,
    validate_candidate_update: bool,
    skip_grad_norm_above: float = 0.0,
) -> tuple[
    ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    optax.OptState,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    loss, grads = closed_loop_loss_fn(
        model,
        metadata,
        inputs,
        initial_temperature,
        targets,
        target_mean,
        target_scale,
        trajectory_output_weights,
        hp_mode_loss_weight,
        heat_on_threshold,
    )
    updates, candidate_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    candidate_model = eqx.apply_updates(model, updates)
    candidate_forward_is_finite = jnp.asarray(True)
    if validate_candidate_update:
        candidate_predictions, candidate_aux = jax.vmap(
            lambda row_metadata, row_inputs, row_initial_temperature: candidate_model.rollout_with_aux(
                row_metadata,
                row_inputs,
                row_initial_temperature,
            )
        )(metadata, inputs, initial_temperature)
        candidate_forward_is_finite = _array_tree_all_finite((candidate_predictions, candidate_aux))
    update_flags = _update_finite_flags(
        loss=loss,
        components=None,
        grads=grads,
        updates=updates,
        opt_state=candidate_opt_state,
        model=candidate_model,
        forward_is_finite=candidate_forward_is_finite,
        max_grad_norm=skip_grad_norm_above,
    )
    update_stats = _update_stats(grads=grads, updates=updates, model=candidate_model)
    update_is_finite = jnp.all(update_flags)
    if skip_nonfinite_updates:
        model = _select_array_tree(update_is_finite, candidate_model, model)
        opt_state = _select_array_tree(update_is_finite, candidate_opt_state, opt_state)
    else:
        model = candidate_model
        opt_state = candidate_opt_state
    return model, opt_state, loss, update_flags, update_stats


def predict_probabilistic_closed_loop_batch(
    model: ProbabilisticClosedLoopModel,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    key: jax.Array,
    num_particles: int,
    sample_process_noise: bool,
    hp_scenario_mode: HPElectricRolloutMode,
    hp_concrete_temperature: jnp.ndarray | float = 0.5,
) -> jnp.ndarray:
    keys = jax.random.split(key, metadata.shape[0])
    return jax.vmap(
        lambda row_metadata, row_inputs, row_initial_temperature, row_key: model.sample(
            row_metadata,
            row_inputs,
            row_initial_temperature,
            key=row_key,
            num_particles=num_particles,
            sample_process_noise=sample_process_noise,
            hp_scenario_mode=hp_scenario_mode,
            hp_concrete_temperature=hp_concrete_temperature,
        )
    )(metadata, inputs, initial_temperature, keys)


def predict_probabilistic_closed_loop_batch_with_aux(
    model: ProbabilisticClosedLoopModel,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    key: jax.Array,
    num_particles: int,
    sample_process_noise: bool,
    hp_scenario_mode: HPElectricRolloutMode,
    hp_concrete_temperature: jnp.ndarray | float = 0.5,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    keys = jax.random.split(key, metadata.shape[0])
    return jax.vmap(
        lambda row_metadata, row_inputs, row_initial_temperature, row_key: model.sample_with_aux(
            row_metadata,
            row_inputs,
            row_initial_temperature,
            key=row_key,
            num_particles=num_particles,
            sample_process_noise=sample_process_noise,
            hp_scenario_mode=hp_scenario_mode,
            hp_concrete_temperature=hp_concrete_temperature,
        )
    )(metadata, inputs, initial_temperature, keys)


def local_jacobian_spectral_norm(
    step_fn,
    state: jnp.ndarray,
    direction: jnp.ndarray,
    *,
    method: ClosedLoopStabilityMethod,
    power_iterations: int,
) -> jnp.ndarray:
    """Estimate the spectral norm of a square local state Jacobian."""
    if method == "exact_svd":
        jacobian = jax.jacrev(step_fn)(state)
        return jnp.max(jnp.linalg.svd(jacobian, compute_uv=False))
    if method != "power_iteration":
        raise ValueError(f"Unknown closed-loop stability method {method!r}")

    eps = jnp.asarray(1e-12, dtype=state.dtype)
    vector = direction / jnp.maximum(jnp.linalg.norm(direction), eps)
    _, jvp_fn = jax.linearize(step_fn, state)
    transpose_fn = jax.linear_transpose(jvp_fn, state)
    for _ in range(power_iterations):
        jacobian_vector = jvp_fn(vector)
        transpose_vector = transpose_fn(jacobian_vector)[0]
        vector = transpose_vector / jnp.maximum(jnp.linalg.norm(transpose_vector), eps)
    return jnp.linalg.norm(jvp_fn(vector))


def closed_loop_stability_penalty(
    model: ProbabilisticClosedLoopModel,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    aux: tuple[jnp.ndarray, ...],
    key: jax.Array,
    *,
    gamma: float,
    num_samples: int,
    aggregation: Literal["mean", "max"],
    method: ClosedLoopStabilityMethod = "exact_svd",
    power_iterations: int = 4,
) -> jnp.ndarray:
    """Sample active closed-loop states and penalize local non-contraction."""
    if num_samples < 1:
        return jnp.asarray(0.0, dtype=inputs.dtype)
    batch_size, num_particles, horizon = aux[PROB_CLOSED_LOOP_AUX_ENERGY].shape
    (
        batch_key,
        particle_key,
        time_key,
        direction_key,
        process_key,
        controller_key,
    ) = jax.random.split(key, 6)
    batch_indices = jax.random.randint(batch_key, (num_samples,), 0, batch_size)
    particle_indices = jax.random.randint(particle_key, (num_samples,), 0, num_particles)
    time_indices = jax.random.randint(time_key, (num_samples,), 0, horizon)

    x_state = aux[PROB_CLOSED_LOOP_AUX_X_STATE]
    w_state = aux[PROB_CLOSED_LOOP_AUX_W_STATE]
    energy = aux[PROB_CLOSED_LOOP_AUX_ENERGY]
    temperature = aux[PROB_CLOSED_LOOP_AUX_TEMPERATURE_STATE]
    xi = aux[PROB_CLOSED_LOOP_AUX_XI]

    if isinstance(model, ProbabilisticContractingClosedLoopHPEmulator):
        hp_latent = aux[PROB_CLOSED_LOOP_AUX_HP_LATENT]
        hp_power_history = aux[PROB_CLOSED_LOOP_AUX_HP_POWER_HISTORY]
        hp_mode_history = aux[PROB_CLOSED_LOOP_AUX_HP_MODE_HISTORY]
        previous_mode = aux[PROB_CLOSED_LOOP_AUX_HP_PREVIOUS_MODE]
        q_response_state = aux[PROB_CLOSED_LOOP_AUX_Q_RESPONSE_STATE]
        state_size = model.stability_state_dim()
        directions = jax.random.normal(
            direction_key,
            (num_samples, state_size),
            dtype=inputs.dtype,
        )
        if model.process_noise_mode == "none":
            sampled_process_noise = jnp.zeros(
                (num_samples, model.state_dim), dtype=inputs.dtype
            )
        else:
            sampled_process_noise = jax.random.normal(
                process_key,
                (num_samples, model.state_dim),
                dtype=inputs.dtype,
            )
        sampled_controller_noise = jax.random.normal(
            controller_key,
            (num_samples, model.controller_state_dim),
            dtype=inputs.dtype,
        )

        def sampled_excess(
            batch_index: jnp.ndarray,
            particle_index: jnp.ndarray,
            time_index: jnp.ndarray,
            direction: jnp.ndarray,
            process_noise_t: jnp.ndarray,
            controller_noise_t: jnp.ndarray,
        ) -> jnp.ndarray:
            augmented_state = model.pack_stability_state(
                x_state[batch_index, particle_index, time_index],
                energy[batch_index, particle_index, time_index],
                temperature[batch_index, particle_index, time_index],
                w_state[batch_index, particle_index, time_index],
                previous_mode[batch_index, particle_index, time_index],
                hp_power_history[batch_index, particle_index, time_index],
                hp_mode_history[batch_index, particle_index, time_index],
                q_response_state[batch_index, particle_index, time_index],
            )
            step_fn = lambda state: model.one_step_augmented_state(
                metadata[batch_index],
                inputs[batch_index, time_index],
                xi[batch_index, particle_index],
                hp_latent[batch_index, particle_index],
                state,
                process_noise_t,
                controller_noise_t,
            )
            sigma_max = local_jacobian_spectral_norm(
                step_fn,
                augmented_state,
                direction,
                method=method,
                power_iterations=power_iterations,
            )
            return jax.nn.relu(
                sigma_max - jnp.asarray(gamma, dtype=sigma_max.dtype)
            ) ** 2

        excess = jax.vmap(sampled_excess)(
            batch_indices,
            particle_indices,
            time_indices,
            directions,
            sampled_process_noise,
            sampled_controller_noise,
        )
        if aggregation == "max":
            return jnp.max(excess)
        return jnp.mean(excess)

    directions = jax.random.normal(
        direction_key,
        (num_samples, model.state_dim + model.controller_state_dim + 2),
        dtype=inputs.dtype,
    )

    def sampled_excess(
        batch_index: jnp.ndarray,
        particle_index: jnp.ndarray,
        time_index: jnp.ndarray,
        direction: jnp.ndarray,
    ) -> jnp.ndarray:
        augmented_state = jnp.concatenate(
            [
                x_state[batch_index, particle_index, time_index],
                w_state[batch_index, particle_index, time_index],
                energy[batch_index, particle_index, time_index, jnp.newaxis],
                temperature[batch_index, particle_index, time_index, jnp.newaxis],
            ],
            axis=0,
        )
        sigma_max = local_jacobian_spectral_norm(
            lambda state: model.one_step_augmented_state(
                metadata[batch_index],
                inputs[batch_index, time_index],
                xi[batch_index, particle_index],
                state,
            ),
            augmented_state,
            direction,
            method=method,
            power_iterations=power_iterations,
        )
        return jax.nn.relu(sigma_max - jnp.asarray(gamma, dtype=sigma_max.dtype)) ** 2

    excess = jax.vmap(sampled_excess)(
        batch_indices,
        particle_indices,
        time_indices,
        directions,
    )
    if aggregation == "max":
        return jnp.max(excess)
    return jnp.mean(excess)


@eqx.filter_value_and_grad(has_aux=True)
def probabilistic_closed_loop_loss_fn(
    model: ProbabilisticClosedLoopModel,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    key: jax.Array,
    num_particles: int,
    hp_training_mode: HPTrainingMode,
    hp_concrete_temperature: jnp.ndarray,
    hp_activation_model: HPActivationModel,
    target_mean: jnp.ndarray,
    target_scale: jnp.ndarray,
    hp_mode_loss_weight: float,
    heat_on_threshold: float,
    hp_active_power_nll_weight: float,
    hp_inactive_leakage_weight: float,
    softopt_weight: float,
    softopt_temperature: float,
    trajectory_output_weights: tuple[float, ...],
    variogram_weight: float,
    variogram_lags: tuple[int, ...],
    variogram_power: float,
    variogram_output_weights: tuple[float, ...],
    variogram_constant_setpoint_only: bool,
    variogram_setpoint_delta_threshold: float,
    ires_weight: float,
    ires_horizon_steps: tuple[int, ...],
    ires_horizon_hours: tuple[float, ...],
    ires_setpoint_delta_threshold: float,
    flex_kpi_crps_weight: float,
    flex_kpi_crps_horizon_steps: tuple[int, ...],
    flex_kpi_crps_horizon_hours: tuple[float, ...],
    flex_kpi_crps_setpoint_delta_threshold: float,
    flex_kpi_crps_min_events: int,
    flex_kpi_crps_ridge: float,
    flex_kpi_crps_controls: FlexibilityKpiControls,
    physics_weight: float,
    horizon_weight_power: float,
    stability_weight: float,
    stability_gamma: float,
    stability_samples: int,
    stability_aggregation: Literal["mean", "max"],
    stability_method: ClosedLoopStabilityMethod,
    stability_power_iterations: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rollout_key, stability_key = jax.random.split(key)
    predictions, aux = predict_probabilistic_closed_loop_batch_with_aux(
        model,
        metadata,
        inputs,
        initial_temperature,
        key=rollout_key,
        num_particles=num_particles,
        sample_process_noise=True,
        hp_scenario_mode=hp_training_mode,
        hp_concrete_temperature=hp_concrete_temperature,
    )
    zero = jnp.asarray(0.0, dtype=predictions.dtype)
    energy_component = energy_score(
        predictions,
        targets,
        output_weights=trajectory_output_weights,
    )
    variogram_component = zero
    if variogram_weight > 0.0:
        variogram_component = variogram_weight * variogram_score(
            predictions,
            targets,
            lags=variogram_lags,
            power=variogram_power,
            output_weights=variogram_output_weights,
            constant_setpoint_inputs=inputs if variogram_constant_setpoint_only else None,
            setpoint_delta_threshold=variogram_setpoint_delta_threshold,
        )
    ires_component = zero
    if ires_weight > 0.0:
        ires_component = ires_weight * intervention_response_energy_score(
            predictions,
            targets,
            inputs,
            horizon_steps=ires_horizon_steps,
            horizon_hours=ires_horizon_hours,
            setpoint_delta_threshold=ires_setpoint_delta_threshold,
            target_channel=2,
        )
    flex_kpi_crps_component = zero
    if flex_kpi_crps_weight > 0.0:
        flex_kpi_crps_component = flex_kpi_crps_weight * flexibility_coefficient_crps(
            predictions,
            targets,
            inputs,
            initial_temperature,
            horizon_steps=flex_kpi_crps_horizon_steps,
            horizon_hours=flex_kpi_crps_horizon_hours,
            setpoint_delta_threshold=flex_kpi_crps_setpoint_delta_threshold,
            min_events=flex_kpi_crps_min_events,
            ridge=flex_kpi_crps_ridge,
            controls=flex_kpi_crps_controls,
            target_channel=2,
        )
    softopt_component = zero
    if softopt_weight > 0.0:
        softopt_component = softopt_weight * soft_optimistic_loss(
            predictions,
            targets,
            temperature=softopt_temperature,
            horizon_weight_power=horizon_weight_power,
            output_weights=trajectory_output_weights,
        )

    pi = aux[0]
    log_mu = aux[PROB_CLOSED_LOOP_AUX_LOG_MU]
    log_sigma = aux[PROB_CLOSED_LOOP_AUX_LOG_SIGMA]
    expected_active = model.expected_active_power(log_mu, log_sigma)
    expected_pel = pi * expected_active
    pel_target = _physical_target_component(targets, target_mean, target_scale, 2)
    mode_target = (pel_target > heat_on_threshold).astype(predictions.dtype)
    mode_target_particles = mode_target[:, jnp.newaxis, :]

    hp_bce_component = zero
    if hp_mode_loss_weight > 0.0:
        if hp_activation_model == "persistent_markov" and inputs.shape[1] > 1:
            p_start = jnp.mean(
                aux[PROB_CLOSED_LOOP_AUX_HP_P_START][:, :, 1:],
                axis=1,
                keepdims=True,
            )
            p_stop = jnp.mean(
                aux[PROB_CLOSED_LOOP_AUX_HP_P_STOP][:, :, 1:],
                axis=1,
                keepdims=True,
            )
            previous_mode = mode_target_particles[:, :, :-1]
            current_mode = mode_target_particles[:, :, 1:]
            availability = jax.vmap(
                jax.vmap(model._space_heating_availability)
            )(inputs)[:, jnp.newaxis, 1:]
            eps = jnp.asarray(1e-5, dtype=predictions.dtype)
            clipped_start = jnp.clip(p_start, eps, 1.0 - eps)
            clipped_stop = jnp.clip(p_stop, eps, 1.0 - eps)
            start_risk = (1.0 - previous_mode) * availability
            stop_risk = previous_mode * availability
            start_nll = -(
                current_mode * jnp.log(clipped_start)
                + (1.0 - current_mode) * jnp.log(1.0 - clipped_start)
            )
            stop_target = 1.0 - current_mode
            stop_nll = -(
                stop_target * jnp.log(clipped_stop)
                + (1.0 - stop_target) * jnp.log(1.0 - clipped_stop)
            )
            start_loss = jnp.sum(start_risk * start_nll) / jnp.maximum(
                jnp.sum(start_risk), 1.0
            )
            stop_loss = jnp.sum(stop_risk * stop_nll) / jnp.maximum(
                jnp.sum(stop_risk), 1.0
            )
            bce = 0.5 * (start_loss + stop_loss)
        else:
            clipped_pi = jnp.clip(pi, 1e-5, 1.0 - 1e-5)
            bce = -jnp.mean(
                mode_target_particles * jnp.log(clipped_pi)
                + (1.0 - mode_target_particles) * jnp.log(1.0 - clipped_pi)
            )
        hp_bce_component = hp_mode_loss_weight * bce

    hp_active_nll_component = zero
    if hp_active_power_nll_weight > 0.0:
        active_mask = jnp.broadcast_to(mode_target_particles, log_mu.shape)
        log_target = jnp.log1p(jnp.maximum(pel_target, 0.0))[:, jnp.newaxis, :]
        log_two_pi = jnp.asarray(np.log(2.0 * np.pi), dtype=predictions.dtype)
        nll = 0.5 * ((log_target - log_mu) / log_sigma) ** 2 + jnp.log(log_sigma) + 0.5 * log_two_pi
        active_count = jnp.maximum(jnp.sum(active_mask), 1.0)
        active_nll = jnp.sum(active_mask * nll) / active_count
        hp_active_nll_component = hp_active_power_nll_weight * active_nll

    hp_inactive_leakage_component = zero
    if hp_inactive_leakage_weight > 0.0:
        inactive_mask = jnp.broadcast_to(1.0 - mode_target_particles, expected_pel.shape)
        pel_scale = jnp.asarray(target_scale[2], dtype=predictions.dtype)
        inactive_count = jnp.maximum(jnp.sum(inactive_mask), 1.0)
        leakage = jnp.sum(inactive_mask * (expected_pel / pel_scale) ** 2) / inactive_count
        hp_inactive_leakage_component = hp_inactive_leakage_weight * leakage

    physics_component = zero
    if physics_weight > 0.0:
        regularization = jnp.mean(jax.vmap(model.parameter_regularization)(metadata))
        physics_component = physics_weight * regularization

    stability_component = zero
    if stability_weight > 0.0 and stability_samples > 0:
        stability_component = stability_weight * closed_loop_stability_penalty(
            model,
            metadata,
            inputs,
            aux,
            stability_key,
            gamma=stability_gamma,
            num_samples=stability_samples,
            aggregation=stability_aggregation,
            method=stability_method,
            power_iterations=stability_power_iterations,
        )

    components = jnp.stack(
        [
            energy_component,
            variogram_component,
            ires_component,
            flex_kpi_crps_component,
            softopt_component,
            hp_bce_component,
            hp_active_nll_component,
            hp_inactive_leakage_component,
            physics_component,
            stability_component,
        ]
    )
    return jnp.sum(components), components


@eqx.filter_jit
def probabilistic_closed_loop_train_step(
    model: ProbabilisticClosedLoopModel,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    key: jax.Array,
    num_particles: int,
    hp_training_mode: HPTrainingMode,
    hp_concrete_temperature: jnp.ndarray,
    hp_activation_model: HPActivationModel,
    target_mean: jnp.ndarray,
    target_scale: jnp.ndarray,
    hp_mode_loss_weight: float,
    heat_on_threshold: float,
    hp_active_power_nll_weight: float,
    hp_inactive_leakage_weight: float,
    softopt_weight: float,
    softopt_temperature: float,
    trajectory_output_weights: tuple[float, ...],
    variogram_weight: float,
    variogram_lags: tuple[int, ...],
    variogram_power: float,
    variogram_output_weights: tuple[float, ...],
    variogram_constant_setpoint_only: bool,
    variogram_setpoint_delta_threshold: float,
    ires_weight: float,
    ires_horizon_steps: tuple[int, ...],
    ires_horizon_hours: tuple[float, ...],
    ires_setpoint_delta_threshold: float,
    flex_kpi_crps_weight: float,
    flex_kpi_crps_horizon_steps: tuple[int, ...],
    flex_kpi_crps_horizon_hours: tuple[float, ...],
    flex_kpi_crps_setpoint_delta_threshold: float,
    flex_kpi_crps_min_events: int,
    flex_kpi_crps_ridge: float,
    flex_kpi_crps_controls: FlexibilityKpiControls,
    physics_weight: float,
    horizon_weight_power: float,
    stability_weight: float,
    stability_gamma: float,
    stability_samples: int,
    stability_aggregation: Literal["mean", "max"],
    stability_method: ClosedLoopStabilityMethod,
    stability_power_iterations: int,
    skip_nonfinite_updates: bool,
    validate_candidate_update: bool,
    skip_grad_norm_above: float = 0.0,
) -> tuple[
    ProbabilisticClosedLoopModel,
    optax.OptState,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    (loss, components), grads = probabilistic_closed_loop_loss_fn(
        model,
        metadata,
        inputs,
        initial_temperature,
        targets,
        key,
        num_particles,
        hp_training_mode,
        hp_concrete_temperature,
        hp_activation_model,
        target_mean,
        target_scale,
        hp_mode_loss_weight,
        heat_on_threshold,
        hp_active_power_nll_weight,
        hp_inactive_leakage_weight,
        softopt_weight,
        softopt_temperature,
        trajectory_output_weights,
        variogram_weight,
        variogram_lags,
        variogram_power,
        variogram_output_weights,
        variogram_constant_setpoint_only,
        variogram_setpoint_delta_threshold,
        ires_weight,
        ires_horizon_steps,
        ires_horizon_hours,
        ires_setpoint_delta_threshold,
        flex_kpi_crps_weight,
        flex_kpi_crps_horizon_steps,
        flex_kpi_crps_horizon_hours,
        flex_kpi_crps_setpoint_delta_threshold,
        flex_kpi_crps_min_events,
        flex_kpi_crps_ridge,
        flex_kpi_crps_controls,
        physics_weight,
        horizon_weight_power,
        stability_weight,
        stability_gamma,
        stability_samples,
        stability_aggregation,
        stability_method,
        stability_power_iterations,
    )
    updates, candidate_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    candidate_model = eqx.apply_updates(model, updates)
    candidate_forward_is_finite = jnp.asarray(True)
    if validate_candidate_update:
        candidate_predictions, candidate_aux = predict_probabilistic_closed_loop_batch_with_aux(
            candidate_model,
            metadata,
            inputs,
            initial_temperature,
            key=key,
            num_particles=num_particles,
            sample_process_noise=True,
            hp_scenario_mode=hp_training_mode,
            hp_concrete_temperature=hp_concrete_temperature,
        )
        candidate_forward_is_finite = _array_tree_all_finite((candidate_predictions, candidate_aux))
    update_flags = _update_finite_flags(
        loss=loss,
        components=components,
        grads=grads,
        updates=updates,
        opt_state=candidate_opt_state,
        model=candidate_model,
        forward_is_finite=candidate_forward_is_finite,
        max_grad_norm=skip_grad_norm_above,
    )
    update_stats = _update_stats(grads=grads, updates=updates, model=candidate_model)
    update_is_finite = jnp.all(update_flags)
    if skip_nonfinite_updates:
        model = _select_array_tree(update_is_finite, candidate_model, model)
        opt_state = _select_array_tree(update_is_finite, candidate_opt_state, opt_state)
    else:
        model = candidate_model
        opt_state = candidate_opt_state
    return model, opt_state, loss, components, update_flags, update_stats


def _component_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    index: int,
) -> dict[str, float]:
    metrics = regression_metrics(prediction[..., index], target[..., index])
    return {
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "nmae": metrics.nmae,
        "bias": metrics.bias,
    }


def closed_loop_metrics_dict(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    temp = _component_metrics(prediction, target, 0)
    q_room = _component_metrics(prediction, target, 1)
    p_el = _component_metrics(prediction, target, 2)
    return {
        "rmse_c": temp["rmse"],
        "mae_c": temp["mae"],
        "nmae": temp["nmae"],
        "bias_c": temp["bias"],
        "qroom_rmse_w_m2": q_room["rmse"],
        "qroom_mae_w_m2": q_room["mae"],
        "qroom_nmae": q_room["nmae"],
        "qroom_bias_w_m2": q_room["bias"],
        "pel_rmse_w_m2": p_el["rmse"],
        "pel_mae_w_m2": p_el["mae"],
        "pel_nmae": p_el["nmae"],
        "pel_bias_w_m2": p_el["bias"],
    }


@dataclass
class _RegressionAccumulator:
    count: int = 0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    error_sum: float = 0.0
    absolute_target_sum: float = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float64)
        true = np.asarray(target, dtype=np.float64)
        error = pred - true
        self.count += error.size
        self.squared_error_sum += float(np.sum(error**2, dtype=np.float64))
        self.absolute_error_sum += float(np.sum(np.abs(error), dtype=np.float64))
        self.error_sum += float(np.sum(error, dtype=np.float64))
        self.absolute_target_sum += float(np.sum(np.abs(true), dtype=np.float64))

    def metrics(self) -> dict[str, float]:
        if self.count == 0:
            return {name: float("nan") for name in ("rmse", "mae", "nmae", "bias")}
        mean_squared_error = self.squared_error_sum / self.count
        mae = self.absolute_error_sum / self.count
        denominator = self.absolute_target_sum / self.count
        return {
            "rmse": float(np.sqrt(mean_squared_error)),
            "mae": float(mae),
            "nmae": float(mae / denominator) if denominator > 1e-12 else float("nan"),
            "bias": float(self.error_sum / self.count),
        }


def _closed_loop_metrics_from_accumulators(
    accumulators: Sequence[_RegressionAccumulator],
) -> dict[str, float]:
    temp, q_room, p_el = (accumulator.metrics() for accumulator in accumulators)
    return {
        "rmse_c": temp["rmse"],
        "mae_c": temp["mae"],
        "nmae": temp["nmae"],
        "bias_c": temp["bias"],
        "qroom_rmse_w_m2": q_room["rmse"],
        "qroom_mae_w_m2": q_room["mae"],
        "qroom_nmae": q_room["nmae"],
        "qroom_bias_w_m2": q_room["bias"],
        "pel_rmse_w_m2": p_el["rmse"],
        "pel_mae_w_m2": p_el["mae"],
        "pel_nmae": p_el["nmae"],
        "pel_bias_w_m2": p_el["bias"],
    }


@dataclass
class _TemperatureDiagnosticsAccumulator:
    max_abs_error: float = -np.inf
    max_abs_error_profile_id: float = np.nan
    max_abs_error_start: float = np.nan
    max_abs_error_step: float = np.nan
    max_abs_error_pred: float = np.nan
    max_abs_error_target: float = np.nan
    min_pred: float = np.inf
    max_pred: float = -np.inf

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        profile_ids: np.ndarray,
        start_indices: np.ndarray,
    ) -> None:
        finite_prediction = prediction[np.isfinite(prediction)]
        if finite_prediction.size:
            self.min_pred = min(self.min_pred, float(np.min(finite_prediction)))
            self.max_pred = max(self.max_pred, float(np.max(finite_prediction)))
        absolute_error = np.abs(prediction - target)
        finite_error = np.where(np.isfinite(absolute_error), absolute_error, np.nan)
        if not np.isfinite(finite_error).any():
            return
        flat_index = int(np.nanargmax(finite_error))
        window_index, step_index = np.unravel_index(flat_index, finite_error.shape)
        error = float(finite_error[window_index, step_index])
        if error <= self.max_abs_error:
            return
        self.max_abs_error = error
        self.max_abs_error_profile_id = float(profile_ids[window_index])
        self.max_abs_error_start = float(start_indices[window_index])
        self.max_abs_error_step = float(step_index)
        self.max_abs_error_pred = float(prediction[window_index, step_index])
        self.max_abs_error_target = float(target[window_index, step_index])

    def as_dict(self) -> dict[str, float]:
        return {
            "max_abs_error_c": (
                float(self.max_abs_error) if np.isfinite(self.max_abs_error) else float("nan")
            ),
            "max_abs_error_profile_id": self.max_abs_error_profile_id,
            "max_abs_error_start": self.max_abs_error_start,
            "max_abs_error_step": self.max_abs_error_step,
            "max_abs_error_pred_c": self.max_abs_error_pred,
            "max_abs_error_target_c": self.max_abs_error_target,
            "min_pred_c": float(self.min_pred) if np.isfinite(self.min_pred) else float("nan"),
            "max_pred_c": float(self.max_pred) if np.isfinite(self.max_pred) else float("nan"),
        }


def add_closed_loop_temperature_diagnostics(
    metrics: dict[str, float],
    prediction: np.ndarray,
    target: np.ndarray,
    profile_ids: np.ndarray,
    start_indices: np.ndarray,
) -> dict[str, float]:
    """Add worst-window temperature diagnostics to a closed-loop metric dict."""
    temperature_prediction = prediction[..., 0]
    temperature_target = target[..., 0]
    for name in (
        "max_abs_error_c",
        "max_abs_error_profile_id",
        "max_abs_error_start",
        "max_abs_error_step",
        "max_abs_error_pred_c",
        "max_abs_error_target_c",
        "min_pred_c",
        "max_pred_c",
    ):
        metrics.setdefault(name, float("nan"))
    finite_temperature_prediction = temperature_prediction[np.isfinite(temperature_prediction)]
    if finite_temperature_prediction.size > 0:
        metrics["min_pred_c"] = float(np.min(finite_temperature_prediction))
        metrics["max_pred_c"] = float(np.max(finite_temperature_prediction))
    absolute_error = np.abs(temperature_prediction - temperature_target)
    if absolute_error.size == 0 or not np.isfinite(absolute_error).any():
        return metrics
    flat_index = int(np.nanargmax(absolute_error))
    window_index, step_index = np.unravel_index(flat_index, absolute_error.shape)
    metrics["max_abs_error_c"] = float(absolute_error[window_index, step_index])
    metrics["max_abs_error_profile_id"] = float(profile_ids[window_index])
    metrics["max_abs_error_start"] = float(start_indices[window_index])
    metrics["max_abs_error_step"] = float(step_index)
    metrics["max_abs_error_pred_c"] = float(temperature_prediction[window_index, step_index])
    metrics["max_abs_error_target_c"] = float(temperature_target[window_index, step_index])
    return metrics


def metric_int(metrics: dict[str, float], name: str) -> str:
    value = metrics.get(name, np.nan)
    if not np.isfinite(value):
        return "nan"
    return str(int(value))


def finite_mean(values: list[float]) -> tuple[float, int]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), int(array.size)
    return float(np.mean(finite)), int(array.size - finite.size)


def finite_column_means(values: list[np.ndarray]) -> tuple[np.ndarray | None, int]:
    if not values:
        return None, 0
    array = np.asarray(values, dtype=float)
    finite_rows = np.all(np.isfinite(array), axis=1)
    if not finite_rows.any():
        return np.full((array.shape[1],), np.nan), int(array.shape[0])
    return np.mean(array[finite_rows], axis=0), int(array.shape[0] - np.sum(finite_rows))


def diagnostic_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def csv_diagnostic_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def append_epoch_diagnostics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow({key: csv_diagnostic_value(value) for key, value in row.items()})


def evaluate_closed_loop(
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    windows: ClosedLoopWindowedArrays,
    scalers: WindowScalers,
    *,
    batch_size: int,
) -> dict[str, float]:
    normalized_accumulator = _RegressionAccumulator()
    physical_accumulators = [_RegressionAccumulator() for _ in CLOSED_LOOP_TARGET_COLUMNS]
    temperature_diagnostics = _TemperatureDiagnosticsAccumulator()
    rng = np.random.default_rng(0)
    for batch_start, (metadata, inputs, initial_temperature, target) in enumerate(
        minibatches(
            windows,
            batch_size=batch_size,
            rng=rng,
            shuffle=False,
        )
    ):
        pred = predict_closed_loop_batch(
            model,
            jnp.asarray(metadata),
            jnp.asarray(inputs),
            jnp.asarray(initial_temperature),
        )
        pred_norm = np.asarray(pred)
        target_norm = np.asarray(target)
        physical_pred = inverse_target(pred_norm, scalers)
        physical_target = inverse_target(target_norm, scalers)
        normalized_accumulator.update(pred_norm, target_norm)
        for index, accumulator in enumerate(physical_accumulators):
            accumulator.update(physical_pred[..., index], physical_target[..., index])
        index_start = batch_start * batch_size
        index_end = index_start + target.shape[0]
        temperature_diagnostics.update(
            physical_pred[..., 0],
            physical_target[..., 0],
            windows.profile_ids[index_start:index_end],
            windows.start_indices[index_start:index_end],
        )

    metrics = _closed_loop_metrics_from_accumulators(physical_accumulators)
    normalized_metrics = normalized_accumulator.metrics()
    metrics["loss"] = normalized_metrics["rmse"] ** 2
    metrics["total_nrmse"] = normalized_metrics["rmse"]
    metrics.update(temperature_diagnostics.as_dict())
    return metrics


def _update_particle_temperature_diagnostics(
    metrics: dict[str, float],
    particle_temperature: np.ndarray,
    target_temperature: np.ndarray,
    profile_ids: np.ndarray,
    start_indices: np.ndarray,
) -> None:
    finite_particle_temperature = particle_temperature[np.isfinite(particle_temperature)]
    if finite_particle_temperature.size:
        metrics["particle_min_pred_c"] = min(
            metrics["particle_min_pred_c"],
            float(np.min(finite_particle_temperature)),
        )
        metrics["particle_max_pred_c"] = max(
            metrics["particle_max_pred_c"],
            float(np.max(finite_particle_temperature)),
        )
    particle_abs_error = np.abs(
        particle_temperature - target_temperature[:, np.newaxis, :]
    )
    finite_abs_error = np.where(np.isfinite(particle_abs_error), particle_abs_error, np.nan)
    if not np.isfinite(finite_abs_error).any():
        return
    flat_index = int(np.nanargmax(finite_abs_error))
    window_index, particle_index, step_index = np.unravel_index(
        flat_index,
        finite_abs_error.shape,
    )
    error = float(finite_abs_error[window_index, particle_index, step_index])
    if error <= metrics["particle_max_abs_error_c"]:
        return
    metrics["particle_max_abs_error_c"] = error
    metrics["particle_worst_profile_id"] = float(profile_ids[window_index])
    metrics["particle_worst_start"] = float(start_indices[window_index])
    metrics["particle_worst_index"] = float(particle_index)
    metrics["particle_worst_step"] = float(step_index)
    metrics["particle_worst_pred_c"] = float(
        particle_temperature[window_index, particle_index, step_index]
    )
    metrics["particle_worst_target_c"] = float(target_temperature[window_index, step_index])


def evaluate_probabilistic_closed_loop(
    model: ProbabilisticClosedLoopModel,
    windows: ClosedLoopWindowedArrays,
    scalers: WindowScalers,
    *,
    batch_size: int,
    num_particles: int,
) -> dict[str, float]:
    normalized_accumulator = _RegressionAccumulator()
    normalized_median_accumulator = _RegressionAccumulator()
    physical_accumulators = [_RegressionAccumulator() for _ in CLOSED_LOOP_TARGET_COLUMNS]
    physical_median_accumulators = [_RegressionAccumulator() for _ in CLOSED_LOOP_TARGET_COLUMNS]
    mean_median_temperature_accumulator = _RegressionAccumulator()
    temperature_diagnostics = _TemperatureDiagnosticsAccumulator()
    key = jax.random.PRNGKey(2468)
    particle_diagnostics = {
        "particle_max_abs_error_c": -np.inf,
        "particle_worst_profile_id": np.nan,
        "particle_worst_start": np.nan,
        "particle_worst_index": np.nan,
        "particle_worst_step": np.nan,
        "particle_worst_pred_c": np.nan,
        "particle_worst_target_c": np.nan,
        "particle_min_pred_c": np.inf,
        "particle_max_pred_c": -np.inf,
    }
    n_windows = windows.targets.shape[0]
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        metadata = windows.metadata[batch_start:batch_end]
        inputs = windows.inputs[batch_start:batch_end]
        initial_temperature = windows.initial_temperature[batch_start:batch_end]
        target = windows.targets[batch_start:batch_end]
        key, batch_key = jax.random.split(key)
        particles = predict_probabilistic_closed_loop_batch(
            model,
            jnp.asarray(metadata),
            jnp.asarray(inputs),
            jnp.asarray(initial_temperature),
            key=batch_key,
            num_particles=num_particles,
            sample_process_noise=False,
            hp_scenario_mode="expected",
        )
        particles_np = np.asarray(particles)
        pred_norm = np.mean(particles_np, axis=1)
        median_pred_norm = np.median(particles_np, axis=1)
        target_norm = np.asarray(target)
        normalized_accumulator.update(pred_norm, target_norm)
        normalized_median_accumulator.update(median_pred_norm, target_norm)
        physical_pred = inverse_target(pred_norm, scalers)
        median_physical_pred = inverse_target(median_pred_norm, scalers)
        particle_physical = inverse_target(particles_np, scalers)
        target_physical = inverse_target(target_norm, scalers)
        for index, accumulator in enumerate(physical_accumulators):
            accumulator.update(physical_pred[..., index], target_physical[..., index])
        for index, accumulator in enumerate(physical_median_accumulators):
            accumulator.update(median_physical_pred[..., index], target_physical[..., index])
        mean_median_temperature_accumulator.update(
            physical_pred[..., 0],
            median_physical_pred[..., 0],
        )
        batch_profile_ids = windows.profile_ids[batch_start:batch_end]
        batch_start_indices = windows.start_indices[batch_start:batch_end]
        temperature_diagnostics.update(
            physical_pred[..., 0],
            target_physical[..., 0],
            batch_profile_ids,
            batch_start_indices,
        )
        particle_temperature = particle_physical[..., 0]
        _update_particle_temperature_diagnostics(
            particle_diagnostics,
            particle_temperature,
            target_physical[..., 0],
            batch_profile_ids,
            batch_start_indices,
        )

    metrics = _closed_loop_metrics_from_accumulators(physical_accumulators)
    median_metrics = _closed_loop_metrics_from_accumulators(physical_median_accumulators)
    metrics.update({f"median_{name}": value for name, value in median_metrics.items()})
    metrics["mean_median_rmse_c"] = mean_median_temperature_accumulator.metrics()["rmse"]
    metrics["loss"] = normalized_accumulator.metrics()["rmse"] ** 2
    metrics["total_nrmse"] = normalized_accumulator.metrics()["rmse"]
    metrics["median_loss"] = normalized_median_accumulator.metrics()["rmse"] ** 2
    particle_diagnostics["particle_max_abs_error_c"] = (
        float(particle_diagnostics["particle_max_abs_error_c"])
        if np.isfinite(particle_diagnostics["particle_max_abs_error_c"])
        else float("nan")
    )
    for name in ("particle_min_pred_c", "particle_max_pred_c"):
        if not np.isfinite(particle_diagnostics[name]):
            particle_diagnostics[name] = float("nan")
    metrics.update(particle_diagnostics)
    metrics.update(temperature_diagnostics.as_dict())
    return metrics


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
    normalized_accumulator = _RegressionAccumulator()
    physical_accumulator = _RegressionAccumulator()
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
        pred_norm = np.asarray(pred)
        target_norm = np.asarray(target)
        normalized_accumulator.update(pred_norm, target_norm)
        physical_accumulator.update(
            inverse_target(pred_norm, scalers),
            inverse_target(target_norm, scalers),
        )

    normalized = normalized_accumulator.metrics()
    physical = physical_accumulator.metrics()
    return {
        "loss": normalized["rmse"] ** 2,
        "total_nrmse": normalized["rmse"],
        "rmse_c": physical["rmse"],
        "mae_c": physical["mae"],
        "nmae": physical["nmae"],
        "bias_c": physical["bias"],
    }


def resolve_checkpoint_metric(
    metric: CheckpointMetric,
    *,
    has_test_windows: bool,
    prefer_total_nrmse: bool = False,
) -> CheckpointMetric:
    if metric == "auto":
        if prefer_total_nrmse:
            return "test_total_nrmse" if has_test_windows else "train_total_nrmse"
        return "test_rmse_c" if has_test_windows else "train_rmse_c"
    if metric.startswith("test_") and not has_test_windows:
        raise ValueError(f"checkpoint_metric={metric!r} requires a non-empty test split")
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
    if metric == "train_total_nrmse":
        return train_eval["total_nrmse"]
    if metric == "test_total_nrmse" and test_eval is not None:
        return test_eval["total_nrmse"]
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


def predict_closed_loop_full_profile(
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    profile: ClosedLoopProfile,
    scalers: WindowScalers,
) -> np.ndarray:
    """Run one closed-loop rollout over a complete profile."""
    metadata = scalers.metadata.transform(profile.metadata)
    inputs = scalers.inputs.transform(profile.inputs)
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = (
        (profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]
    ).astype(np.float32)
    prediction = predict_closed_loop_batch(
        model,
        jnp.asarray(metadata)[jnp.newaxis, :],
        jnp.asarray(inputs)[jnp.newaxis, :, :],
        jnp.asarray(initial_temperature)[jnp.newaxis, :],
    )
    return inverse_target(np.asarray(prediction[0]), scalers)


def predict_probabilistic_closed_loop_full_profile(
    model: ProbabilisticClosedLoopModel,
    profile: ClosedLoopProfile,
    scalers: WindowScalers,
    *,
    key: jax.Array,
    num_particles: int,
) -> np.ndarray:
    """Run one mean probabilistic closed-loop rollout over a complete profile."""
    metadata = scalers.metadata.transform(profile.metadata)
    inputs = scalers.inputs.transform(profile.inputs)
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = (
        (profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]
    ).astype(np.float32)
    particles = model.sample(
        jnp.asarray(metadata),
        jnp.asarray(inputs),
        jnp.asarray(initial_temperature),
        key=key,
        num_particles=num_particles,
        sample_process_noise=False,
        hp_scenario_mode="expected",
    )
    prediction = np.asarray(jnp.mean(particles, axis=0))
    return inverse_target(prediction, scalers)


def sample_probabilistic_closed_loop_full_profile_scenarios(
    model: ProbabilisticClosedLoopModel,
    profile: ClosedLoopProfile,
    scalers: WindowScalers,
    *,
    key: jax.Array,
    num_particles: int,
    hp_scenario_mode: HPElectricScenarioMode,
) -> np.ndarray:
    """Sample joint closed-loop scenarios in physical units.

    The returned channels are [Tin, Qroom, Pel_SH], preserving the per-scenario
    link between electric power, delivered heat, and temperature.
    """
    metadata = scalers.metadata.transform(profile.metadata)
    inputs = scalers.inputs.transform(profile.inputs)
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = (
        (profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]
    ).astype(np.float32)
    raw_predictions = model.sample(
        jnp.asarray(metadata),
        jnp.asarray(inputs),
        jnp.asarray(initial_temperature),
        key=key,
        num_particles=num_particles,
        sample_process_noise=True,
        hp_scenario_mode=hp_scenario_mode,
    )
    return inverse_target(np.asarray(raw_predictions), scalers)


def evaluate_closed_loop_full_profiles(
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    profiles: list[ClosedLoopProfile],
    scalers: WindowScalers,
) -> dict[str, float]:
    """Evaluate one continuous closed-loop rollout per full profile."""
    if not profiles:
        return {
            "profile_count": 0.0,
            "rmse_c": float("nan"),
            "mae_c": float("nan"),
            "nmae": float("nan"),
            "bias_c": float("nan"),
            "qroom_rmse_w_m2": float("nan"),
            "pel_rmse_w_m2": float("nan"),
        }
    predictions = [predict_closed_loop_full_profile(model, profile, scalers) for profile in profiles]
    targets = [profile.targets for profile in profiles]
    metrics = closed_loop_metrics_dict(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
    )
    metrics["profile_count"] = float(len(profiles))
    return metrics


def evaluate_probabilistic_closed_loop_full_profiles(
    model: ProbabilisticClosedLoopModel,
    profiles: list[ClosedLoopProfile],
    scalers: WindowScalers,
    *,
    num_particles: int,
) -> dict[str, float]:
    """Evaluate one continuous mean probabilistic closed-loop rollout per profile."""
    if not profiles:
        return {
            "profile_count": 0.0,
            "rmse_c": float("nan"),
            "mae_c": float("nan"),
            "nmae": float("nan"),
            "bias_c": float("nan"),
            "qroom_rmse_w_m2": float("nan"),
            "pel_rmse_w_m2": float("nan"),
        }
    key = jax.random.PRNGKey(8642)
    predictions = []
    for profile in profiles:
        key, profile_key = jax.random.split(key)
        predictions.append(
            predict_probabilistic_closed_loop_full_profile(
                model,
                profile,
                scalers,
                key=profile_key,
                num_particles=num_particles,
            )
        )
    targets = [profile.targets for profile in profiles]
    metrics = closed_loop_metrics_dict(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
    )
    metrics["profile_count"] = float(len(profiles))
    return metrics


def _profile_by_id(profiles: list[BuildingProfile]) -> dict[int, BuildingProfile]:
    return {profile.profile_id: profile for profile in profiles}


def _closed_loop_profile_by_id(profiles: list[ClosedLoopProfile]) -> dict[int, ClosedLoopProfile]:
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


def _closed_loop_metrics_text(prediction: np.ndarray, target: np.ndarray) -> str:
    metrics = closed_loop_metrics_dict(prediction, target)
    return (
        f"T RMSE={metrics['rmse_c']:.3f} degC, "
        f"Q RMSE={metrics['qroom_rmse_w_m2']:.3f} W/m2, "
        f"Pel RMSE={metrics['pel_rmse_w_m2']:.3f} W/m2"
    )


def _scenario_extreme_traces(
    scenarios: np.ndarray,
    simulated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-output closest and worst sampled traces versus simulation."""
    scenario_values = np.asarray(scenarios)
    simulated_values = np.asarray(simulated)
    if scenario_values.ndim == 2:
        scenario_values = scenario_values[..., np.newaxis]
    if simulated_values.ndim == 1:
        simulated_values = simulated_values[..., np.newaxis]
    if scenario_values.ndim != 3:
        raise ValueError("scenarios must have shape [scenario, time, output]")
    if simulated_values.ndim != 2:
        raise ValueError("simulated must have shape [time, output]")
    if scenario_values.shape[1:] != simulated_values.shape:
        raise ValueError(
            "scenario and simulated traces must agree on time/output dimensions; "
            f"got {scenario_values.shape[1:]} and {simulated_values.shape}"
        )

    errors = scenario_values - simulated_values[np.newaxis, :, :]
    rmse_by_scenario_and_output = np.sqrt(np.nanmean(errors**2, axis=1))
    best_indices = np.nanargmin(rmse_by_scenario_and_output, axis=0)
    worst_indices = np.nanargmax(rmse_by_scenario_and_output, axis=0)
    output_indices = np.arange(simulated_values.shape[1])
    closest = scenario_values[best_indices, :, output_indices].T
    worst = scenario_values[worst_indices, :, output_indices].T
    return closest, worst


def _write_temperature_comparison(
    *,
    path: Path,
    title: str,
    datetimes: np.ndarray,
    simulated: np.ndarray,
    emulated: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    closest: np.ndarray | None = None,
    worst: np.ndarray | None = None,
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
    if closest is not None:
        fig.add_trace(
            go.Scattergl(
                x=datetimes,
                y=np.asarray(closest).ravel(),
                mode="lines",
                name="Closest scenario",
                line={"color": "#2ca02c", "width": 1.5, "dash": "dash"},
            )
        )
    if worst is not None:
        fig.add_trace(
            go.Scattergl(
                x=datetimes,
                y=np.asarray(worst).ravel(),
                mode="lines",
                name="Worst scenario",
                line={"color": "#9467bd", "width": 1.5, "dash": "dot"},
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


def _write_closed_loop_comparison(
    *,
    path: Path,
    title: str,
    datetimes: np.ndarray,
    simulated: np.ndarray,
    emulated: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    closest: np.ndarray | None = None,
    worst: np.ndarray | None = None,
    interval_label: str = "95% scenario interval",
) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Indoor temperature",
            "Delivered room heat",
            "Space-heating HP electric power",
        ),
    )
    y_titles = ["Temperature [degC]", "Qroom [W/m2]", "Pel_SH [W/m2]"]
    for row, index in enumerate(range(3), start=1):
        fig.add_trace(
            go.Scattergl(
                x=datetimes,
                y=simulated[:, index],
                mode="lines",
                name="Simulated" if row == 1 else "Simulated",
                line={"color": "#1f77b4"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        if lower is not None and upper is not None:
            fig.add_trace(
                go.Scatter(
                    x=datetimes,
                    y=upper[:, index],
                    mode="lines",
                    line={"width": 0, "color": "rgba(214, 39, 40, 0)"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=datetimes,
                    y=lower[:, index],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(214, 39, 40, 0.18)",
                    line={"width": 0, "color": "rgba(214, 39, 40, 0)"},
                    name=interval_label,
                    showlegend=row == 1,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
        if closest is not None:
            fig.add_trace(
                go.Scattergl(
                    x=datetimes,
                    y=closest[:, index],
                    mode="lines",
                    name="Closest scenario",
                    line={"color": "#2ca02c", "width": 1.5, "dash": "dash"},
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
        if worst is not None:
            fig.add_trace(
                go.Scattergl(
                    x=datetimes,
                    y=worst[:, index],
                    mode="lines",
                    name="Worst scenario",
                    line={"color": "#9467bd", "width": 1.5, "dash": "dot"},
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
        fig.add_trace(
            go.Scattergl(
                x=datetimes,
                y=emulated[:, index],
                mode="lines",
                name="Emulated" if row == 1 else "Emulated",
                line={"color": "#d62728"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=y_titles[index], row=row, col=1)
    fig.update_layout(
        title=f"{title}<br><sup>{_closed_loop_metrics_text(emulated, simulated)}</sup>",
        template="plotly_white",
        hovermode="x unified",
        height=900,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.write_html(path, include_plotlyjs=True)


def save_closed_loop_prediction_visualizations(
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator,
    test_windows: ClosedLoopWindowedArrays,
    test_profiles: list[ClosedLoopProfile],
    scalers: WindowScalers,
    output_dir: Path,
    num_window_plots: int,
    num_full_profile_plots: int,
    *,
    filename_suffix: str = "_closed_loop_hp",
    title_label: str = "closed-loop HP",
) -> dict[str, Path]:
    """Save closed-loop fixed-window and full-profile rollout plots."""
    if test_windows.targets.shape[0] == 0 or not test_profiles:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_by_id = _closed_loop_profile_by_id(test_profiles)
    paths: dict[str, Path] = {}

    for plot_number, window_index in enumerate(
        _evenly_spaced_indices(num_window_plots, test_windows.targets.shape[0]),
        start=1,
    ):
        window_profile_id = int(test_windows.profile_ids[window_index])
        window_start = int(test_windows.start_indices[window_index])
        window_profile = profiles_by_id[window_profile_id]
        prediction_norm = predict_closed_loop_batch(
            model,
            jnp.asarray(test_windows.metadata[window_index])[jnp.newaxis, :],
            jnp.asarray(test_windows.inputs[window_index])[jnp.newaxis, :, :],
            jnp.asarray(test_windows.initial_temperature[window_index])[jnp.newaxis, :],
        )
        prediction = inverse_target(np.asarray(prediction_norm[0]), scalers)
        target = inverse_target(test_windows.targets[window_index], scalers)
        window_end = window_start + prediction.shape[0]
        window_path = (
            output_dir
            / (
                f"test_window_{plot_number:02d}_profile_{window_profile_id}"
                f"_start_{window_start}{filename_suffix}.html"
            )
        )
        _write_closed_loop_comparison(
            path=window_path,
            title=f"{prediction.shape[0]}-step {title_label} rollout, profile {window_profile_id}",
            datetimes=window_profile.datetime[window_start:window_end],
            simulated=target,
            emulated=prediction,
        )
        paths[f"test_window_{plot_number:02d}"] = window_path

    for plot_number, profile_index in enumerate(
        _evenly_spaced_indices(num_full_profile_plots, len(test_profiles)),
        start=1,
    ):
        full_profile = test_profiles[profile_index]
        prediction = predict_closed_loop_full_profile(model, full_profile, scalers)
        full_path = output_dir / (
            f"test_full_profile_{plot_number:02d}_{full_profile.profile_id}{filename_suffix}.html"
        )
        _write_closed_loop_comparison(
            path=full_path,
            title=f"Full-profile {title_label} rollout, profile {full_profile.profile_id}",
            datetimes=full_profile.datetime,
            simulated=full_profile.targets,
            emulated=prediction,
        )
        paths[f"full_profile_{plot_number:02d}"] = full_path

    return paths


def save_probabilistic_closed_loop_prediction_visualizations(
    model: ProbabilisticClosedLoopModel,
    test_windows: ClosedLoopWindowedArrays,
    test_profiles: list[ClosedLoopProfile],
    scalers: WindowScalers,
    output_dir: Path,
    num_window_plots: int,
    num_full_profile_plots: int,
    *,
    num_particles: int,
    hp_scenario_mode: HPElectricScenarioMode,
    filename_suffix: str = "_closed_loop_hp_prob",
    title_label: str = "probabilistic closed-loop HP",
) -> dict[str, Path]:
    """Save probabilistic closed-loop rollout plots with joint scenario intervals."""
    if test_windows.targets.shape[0] == 0 or not test_profiles:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_by_id = _closed_loop_profile_by_id(test_profiles)
    paths: dict[str, Path] = {}
    key = jax.random.PRNGKey(121314)

    for plot_number, window_index in enumerate(
        _evenly_spaced_indices(num_window_plots, test_windows.targets.shape[0]),
        start=1,
    ):
        key, window_key = jax.random.split(key)
        window_profile_id = int(test_windows.profile_ids[window_index])
        window_start = int(test_windows.start_indices[window_index])
        window_profile = profiles_by_id[window_profile_id]
        scenarios_norm = model.sample(
            jnp.asarray(test_windows.metadata[window_index]),
            jnp.asarray(test_windows.inputs[window_index]),
            jnp.asarray(test_windows.initial_temperature[window_index]),
            key=window_key,
            num_particles=num_particles,
            sample_process_noise=True,
            hp_scenario_mode=hp_scenario_mode,
        )
        scenarios = inverse_target(np.asarray(scenarios_norm), scalers)
        prediction = np.mean(scenarios, axis=0)
        lower = np.quantile(scenarios, 0.025, axis=0)
        upper = np.quantile(scenarios, 0.975, axis=0)
        target = inverse_target(test_windows.targets[window_index], scalers)
        closest, worst = _scenario_extreme_traces(scenarios, target)
        window_end = window_start + prediction.shape[0]
        window_path = (
            output_dir
            / (
                f"test_window_{plot_number:02d}_profile_{window_profile_id}"
                f"_start_{window_start}{filename_suffix}.html"
            )
        )
        _write_closed_loop_comparison(
            path=window_path,
            title=f"{prediction.shape[0]}-step {title_label} rollout, profile {window_profile_id}",
            datetimes=window_profile.datetime[window_start:window_end],
            simulated=target,
            emulated=prediction,
            lower=lower,
            upper=upper,
            closest=closest,
            worst=worst,
            interval_label=f"95% scenario interval ({num_particles} samples)",
        )
        paths[f"test_window_{plot_number:02d}"] = window_path

    for plot_number, profile_index in enumerate(
        _evenly_spaced_indices(num_full_profile_plots, len(test_profiles)),
        start=1,
    ):
        key, profile_key = jax.random.split(key)
        full_profile = test_profiles[profile_index]
        scenarios = sample_probabilistic_closed_loop_full_profile_scenarios(
            model,
            full_profile,
            scalers,
            key=profile_key,
            num_particles=num_particles,
            hp_scenario_mode=hp_scenario_mode,
        )
        prediction = np.mean(scenarios, axis=0)
        lower = np.quantile(scenarios, 0.025, axis=0)
        upper = np.quantile(scenarios, 0.975, axis=0)
        closest, worst = _scenario_extreme_traces(scenarios, full_profile.targets)
        full_path = output_dir / (
            f"test_full_profile_{plot_number:02d}_{full_profile.profile_id}{filename_suffix}.html"
        )
        _write_closed_loop_comparison(
            path=full_path,
            title=f"Full-profile {title_label} rollout, profile {full_profile.profile_id}",
            datetimes=full_profile.datetime,
            simulated=full_profile.targets,
            emulated=prediction,
            lower=lower,
            upper=upper,
            closest=closest,
            worst=worst,
            interval_label=f"95% scenario interval ({num_particles} samples)",
        )
        paths[f"full_profile_{plot_number:02d}"] = full_path

    return paths


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
        closest = None
        worst = None
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
        if model_kind == "probabilistic":
            closest, worst = _scenario_extreme_traces(scenarios, full_target)
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
            closest=closest,
            worst=worst,
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
    if isinstance(model, (ContractingClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator)):
        return float(model.contraction_gamma)
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


def _linear_with_copied_weights(
    source: eqx.nn.Linear,
    target: eqx.nn.Linear,
    *,
    allow_extra_input_columns: bool = False,
    extra_input_key: jax.Array | None = None,
    extra_input_scale: float = 0.0,
) -> eqx.nn.Linear:
    source_weight = jnp.asarray(source.weight)
    target_weight = jnp.asarray(target.weight)
    if source_weight.shape == target_weight.shape:
        new_weight = source_weight.astype(target_weight.dtype)
    elif (
        allow_extra_input_columns
        and source_weight.shape[0] == target_weight.shape[0]
        and source_weight.shape[1] < target_weight.shape[1]
    ):
        new_weight = jnp.zeros_like(target_weight).at[:, : source_weight.shape[1]].set(
            source_weight.astype(target_weight.dtype)
        )
        if extra_input_key is not None and extra_input_scale > 0.0:
            extra_width = target_weight.shape[1] - source_weight.shape[1]
            source_std = jnp.maximum(jnp.std(source_weight), jnp.asarray(1e-6, dtype=source_weight.dtype))
            noise = (
                jax.random.normal(extra_input_key, (target_weight.shape[0], extra_width))
                * jnp.asarray(extra_input_scale, dtype=target_weight.dtype)
                * source_std.astype(target_weight.dtype)
            )
            new_weight = new_weight.at[:, source_weight.shape[1] :].set(noise)
    else:
        raise ValueError(
            "Cannot copy linear layer with incompatible shapes: "
            f"source={source_weight.shape}, target={target_weight.shape}"
        )
    source_bias = jnp.asarray(source.bias)
    target_bias = jnp.asarray(target.bias)
    if source_bias.shape != target_bias.shape:
        raise ValueError(
            "Cannot copy linear bias with incompatible shapes: "
            f"source={source_bias.shape}, target={target_bias.shape}"
        )
    layer = eqx.tree_at(lambda item: item.weight, target, new_weight)
    return eqx.tree_at(lambda item: item.bias, layer, source_bias.astype(target_bias.dtype))


def _copy_mlp(
    source,
    target,
    *,
    allow_extra_input_columns: bool = False,
    extra_input_key: jax.Array | None = None,
    extra_input_scale: float = 0.0,
):
    if len(source.layers) != len(target.layers):
        raise ValueError(
            "Cannot copy MLP with different depths: "
            f"source={len(source.layers)}, target={len(target.layers)}"
        )
    layers = tuple(
        _linear_with_copied_weights(
            source_layer,
            target_layer,
            allow_extra_input_columns=allow_extra_input_columns and index == 0,
            extra_input_key=extra_input_key if index == 0 else None,
            extra_input_scale=extra_input_scale,
        )
        for index, (source_layer, target_layer) in enumerate(zip(source.layers, target.layers))
    )
    return eqx.tree_at(lambda item: item.layers, target, layers)


def _constant_output_mlp(target, output: jnp.ndarray):
    output = jnp.asarray(output)
    layers = []
    for index, layer in enumerate(target.layers):
        weight = jnp.zeros_like(layer.weight)
        if index == len(target.layers) - 1:
            if output.shape != layer.bias.shape:
                raise ValueError(
                    "Cannot set constant MLP output with incompatible shapes: "
                    f"output={output.shape}, bias={layer.bias.shape}"
                )
            bias = output.astype(layer.bias.dtype)
        else:
            bias = jnp.zeros_like(layer.bias)
        layer = eqx.tree_at(lambda item: item.weight, layer, weight)
        layer = eqx.tree_at(lambda item: item.bias, layer, bias)
        layers.append(layer)
    return eqx.tree_at(lambda item: item.layers, target, tuple(layers))


def _shift_final_bias(target, shift: float):
    final_index = len(target.layers) - 1
    final_layer = target.layers[final_index]
    shifted_layer = eqx.tree_at(
        lambda item: item.bias,
        final_layer,
        final_layer.bias + jnp.asarray(shift, dtype=final_layer.bias.dtype),
    )
    layers = list(target.layers)
    layers[final_index] = shifted_layer
    return eqx.tree_at(lambda item: item.layers, target, tuple(layers))


def _raw_for_hp_active_log_sigma(
    log_sigma: float,
    *,
    hp_emission_mode: ProbHpEmissionMode,
) -> jnp.ndarray:
    """Invert the HP active log-sigma parametrization for the selected emission mode."""
    width = 0.45 if hp_emission_mode == "bounded" else 0.70
    upper = 0.05 + width
    normalized = (log_sigma - 0.05) / width
    if not 0.0 < normalized < 1.0:
        raise ValueError(f"--init-hp-active-log-sigma must be in (0.05, {upper:.2f})")
    raw = np.log(normalized / (1.0 - normalized))
    return jnp.asarray([raw], dtype=jnp.float32)


def initialize_probabilistic_closed_loop_from_deterministic(
    probabilistic_model: ProbabilisticClosedLoopHPEmulator,
    deterministic_model: ClosedLoopHPEmulator,
    *,
    key: jax.Array,
    xi_weight_scale: float,
    hp_active_log_sigma: float,
) -> ProbabilisticClosedLoopHPEmulator:
    """Initialize probabilistic closed-loop model around a deterministic solution."""
    compatibility_fields = (
        "metadata_dim",
        "input_dim",
        "state_dim",
        "encoded_input_dim",
        "controller_state_dim",
    )
    for field in compatibility_fields:
        if getattr(probabilistic_model, field) != getattr(deterministic_model, field):
            raise ValueError(
                "Deterministic artifact is incompatible with the probabilistic model: "
                f"{field} differs ({getattr(deterministic_model, field)} != "
                f"{getattr(probabilistic_model, field)})"
            )
    model = probabilistic_model
    theta_key, x0_key, w0_key, e0_key = jax.random.split(key, 4)
    model = eqx.tree_at(
        lambda item: item.theta_net,
        model,
        _copy_mlp(
            deterministic_model.theta_net,
            model.theta_net,
            allow_extra_input_columns=True,
            extra_input_key=theta_key,
            extra_input_scale=xi_weight_scale,
        ),
    )
    model = eqx.tree_at(
        lambda item: item.x0_net,
        model,
        _copy_mlp(
            deterministic_model.x0_net,
            model.x0_net,
            allow_extra_input_columns=True,
            extra_input_key=x0_key,
            extra_input_scale=xi_weight_scale,
        ),
    )
    model = eqx.tree_at(
        lambda item: item.w0_net,
        model,
        _copy_mlp(
            deterministic_model.w0_net,
            model.w0_net,
            allow_extra_input_columns=True,
            extra_input_key=w0_key,
            extra_input_scale=xi_weight_scale,
        ),
    )
    model = eqx.tree_at(
        lambda item: item.e0_net,
        model,
        _copy_mlp(
            deterministic_model.e0_net,
            model.e0_net,
            allow_extra_input_columns=True,
            extra_input_key=e0_key,
            extra_input_scale=xi_weight_scale,
        ),
    )
    model = eqx.tree_at(
        lambda item: item.mode_net,
        model,
        _copy_mlp(deterministic_model.mode_net, model.mode_net),
    )
    pel_mu_net = _copy_mlp(deterministic_model.pel_net, model.pel_mu_net)
    pel_scale = float(model.target_scale[model.pel_target_index])
    pel_mean = float(model.target_mean[model.pel_target_index])
    max_active = max(pel_mean + 8.0 * pel_scale, 2.0 * pel_scale)
    if model.hp_pel_cap_w_m2 > 0.0:
        max_active = model.hp_pel_cap_w_m2
    reference_ratio = np.clip(
        np.log1p(max(pel_scale, 1e-6)) / max(np.log1p(max_active), 1e-6),
        1e-4,
        0.95,
    )
    bias_shift = float(np.log(reference_ratio / (1.0 - reference_ratio)))
    model = eqx.tree_at(
        lambda item: item.pel_mu_net,
        model,
        _shift_final_bias(pel_mu_net, bias_shift),
    )
    model = eqx.tree_at(
        lambda item: item.pel_sigma_net,
        model,
        _constant_output_mlp(
            model.pel_sigma_net,
            _raw_for_hp_active_log_sigma(
                hp_active_log_sigma,
                hp_emission_mode=model.hp_emission_mode,
            ),
        ),
    )
    model = eqx.tree_at(
        lambda item: item.qroom_net,
        model,
        _copy_mlp(deterministic_model.qroom_net, model.qroom_net),
    )
    model = eqx.tree_at(
        lambda item: item.w_net,
        model,
        _copy_mlp(deterministic_model.w_net, model.w_net),
    )
    if deterministic_model.thermal_encoder is None:
        if model.thermal_encoder is not None:
            raise ValueError("Deterministic artifact has no thermal encoder, but probabilistic model does")
    else:
        if model.thermal_encoder is None:
            raise ValueError("Deterministic artifact has a thermal encoder, but probabilistic model does not")
        model = eqx.tree_at(
            lambda item: item.thermal_encoder,
            model,
            _copy_mlp(deterministic_model.thermal_encoder, model.thermal_encoder),
        )

    cop_intercept = jnp.asarray(deterministic_model.cop_intercept)
    cop_slope = jnp.asarray(deterministic_model.cop_slope)
    raw_energy_loss_rate = jnp.asarray(deterministic_model.raw_energy_loss_rate)
    hp_raw = jnp.stack(
        [
            (cop_intercept - 3.0) / 0.1,
            (cop_slope - 0.02) / 0.005,
            (raw_energy_loss_rate + 4.0) / 0.1,
        ]
    ).astype(jnp.float32)
    model = eqx.tree_at(
        lambda item: item.hp_param_net,
        model,
        _constant_output_mlp(model.hp_param_net, hp_raw),
    )
    return model


def initialize_probabilistic_closed_loop_from_artifact(
    probabilistic_model: ProbabilisticClosedLoopHPEmulator,
    artifact_path: Path,
    *,
    key: jax.Array,
    xi_weight_scale: float,
    hp_active_log_sigma: float,
) -> ProbabilisticClosedLoopHPEmulator:
    artifact = load_training_artifact(artifact_path)
    if artifact.metadata.get("model_kind") != "closed_loop_hp":
        raise ValueError(
            "--init-from-deterministic-artifact must point to a closed_loop_hp artifact; "
            f"found {artifact.metadata.get('model_kind')!r}"
        )
    deterministic_model = artifact.model
    if not isinstance(deterministic_model, ClosedLoopHPEmulator):
        raise ValueError("Loaded artifact is not a ClosedLoopHPEmulator")
    return initialize_probabilistic_closed_loop_from_deterministic(
        probabilistic_model,
        deterministic_model,
        key=key,
        xi_weight_scale=xi_weight_scale,
        hp_active_log_sigma=hp_active_log_sigma,
    )


def observed_nonnegative_cap(
    values: np.ndarray,
    *,
    requested_cap: float,
    factor: float,
    floor: float,
) -> float:
    """Resolve a positive physical cap from a requested value or observed training data."""
    if requested_cap > 0.0:
        return float(requested_cap)
    finite_values = values[np.isfinite(values)]
    finite_values = finite_values[finite_values > 0.0]
    if finite_values.size == 0:
        return float(floor)
    return float(max(factor * float(np.max(finite_values)), floor))


def resolve_closed_loop_caps(
    train_windows: ClosedLoopWindowedArrays,
    config: TrainConfig,
) -> tuple[float, float, float, float]:
    """Resolve physical closed-loop caps in W/m2, Wh/m2, and COP units."""
    pel_cap_w_m2 = observed_nonnegative_cap(
        train_windows.targets[..., 2],
        requested_cap=config.hp_pel_cap_w_m2,
        factor=config.hp_cap_factor,
        floor=config.heat_on_threshold,
    )
    qroom_cap_w_m2 = observed_nonnegative_cap(
        train_windows.targets[..., 1],
        requested_cap=config.hp_qroom_cap_w_m2,
        factor=config.hp_cap_factor,
        floor=1.0,
    )
    cop_cap = float(config.hp_cop_cap)
    if config.hp_energy_cap_wh_m2 > 0.0:
        energy_cap_wh_m2 = float(config.hp_energy_cap_wh_m2)
    else:
        cop_for_energy = cop_cap if cop_cap > 0.0 else max(config.hp_cop_floor, 6.0)
        energy_cap_wh_m2 = max(
            config.hp_energy_cap_hours * qroom_cap_w_m2,
            config.hp_dt_hours * cop_for_energy * pel_cap_w_m2,
            1.0,
        )
    return pel_cap_w_m2, qroom_cap_w_m2, cop_cap, float(energy_cap_wh_m2)


def run_closed_loop_training(
    config: TrainConfig,
) -> ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator | ProbabilisticClosedLoopModel:
    is_probabilistic = config.model_kind in (
        "closed_loop_hp_probabilistic",
        "closed_loop_hp_contracting_probabilistic",
    )
    is_contracting = config.model_kind in (
        "closed_loop_hp_contracting",
        "closed_loop_hp_contracting_probabilistic",
    )
    is_contracting_probabilistic = config.model_kind == "closed_loop_hp_contracting_probabilistic"
    if config.target_mode != "absolute":
        raise ValueError(f"--model-kind {config.model_kind} requires --target-mode absolute")
    if config.target_alignment != "same_time":
        raise ValueError(
            f"--model-kind {config.model_kind} uses fixed u[t] -> (Tin[t+1], Qroom[t], Pel[t]) "
            "alignment; leave --target-alignment same_time."
        )
    if config.heat_input_normalization != "per_floor_area":
        raise ValueError(f"--model-kind {config.model_kind} requires --heat-input-normalization per_floor_area")
    if config.hp_power_area_normalization not in ("zone_floor_area", "building_heated_area"):
        raise ValueError(
            "hp_power_area_normalization must be 'zone_floor_area' or "
            "'building_heated_area'"
        )
    if config.hp_controller_state_dim < 1:
        raise ValueError("hp_controller_state_dim must be positive")
    if config.hp_thermostat_demand_mode not in ("unconstrained", "monotone"):
        raise ValueError("hp_thermostat_demand_mode must be 'unconstrained' or 'monotone'")
    if config.hp_thermostat_demand_mode == "monotone" and not is_contracting:
        raise ValueError(
            "--hp-thermostat-demand-mode monotone currently requires a contracting "
            "closed-loop HP model"
        )
    if (
        config.hp_thermostat_slope_min <= 0.0
        or config.hp_thermostat_slope_max <= config.hp_thermostat_slope_min
    ):
        raise ValueError("thermostat slope bounds must satisfy 0 < min < max")
    if config.hp_thermostat_threshold_max_c <= config.hp_thermostat_threshold_min_c:
        raise ValueError("thermostat threshold bounds must satisfy min < max")
    if config.hp_dt_hours <= 0.0:
        raise ValueError("hp_dt_hours must be positive")
    if config.bptt_truncate_steps < 0:
        raise ValueError("bptt_truncate_steps must be >= 0; use 0 for full BPTT")
    if config.skip_grad_norm_above < 0.0:
        raise ValueError("skip_grad_norm_above must be non-negative; use 0 to disable")
    if config.train_eval_max_windows < 0:
        raise ValueError("train_eval_max_windows must be non-negative; use 0 for all windows")
    if config.full_train_eval_every_epochs < 0:
        raise ValueError("full_train_eval_every_epochs must be non-negative; use 0 to disable")
    if config.hp_mode_loss_weight < 0.0:
        raise ValueError("hp_mode_loss_weight must be non-negative")
    if config.hp_cop_floor <= 0.0:
        raise ValueError("hp_cop_floor must be positive")
    if config.hp_cop_cap < 0.0:
        raise ValueError("hp_cop_cap must be non-negative; use 0 to disable the COP upper cap")
    if config.hp_cop_cap > 0.0 and config.hp_cop_cap <= config.hp_cop_floor:
        raise ValueError("hp_cop_cap must be larger than hp_cop_floor when enabled")
    if config.hp_pel_cap_w_m2 < 0.0:
        raise ValueError("hp_pel_cap_w_m2 must be non-negative; use 0 for an automatic cap")
    if config.hp_qroom_cap_w_m2 < 0.0:
        raise ValueError("hp_qroom_cap_w_m2 must be non-negative; use 0 for an automatic cap")
    if config.hp_energy_cap_wh_m2 < 0.0:
        raise ValueError("hp_energy_cap_wh_m2 must be non-negative; use 0 for an automatic cap")
    if config.hp_energy_cap_hours <= 0.0:
        raise ValueError("hp_energy_cap_hours must be positive")
    if config.hp_cap_factor <= 0.0:
        raise ValueError("hp_cap_factor must be positive")
    if len(config.closed_loop_trajectory_output_weights) != len(CLOSED_LOOP_TARGET_COLUMNS):
        raise ValueError(
            "--closed-loop-trajectory-output-weights must contain exactly three values "
            "for Tin Qroom Pel_SH"
        )
    if any(weight < 0.0 for weight in config.closed_loop_trajectory_output_weights):
        raise ValueError("closed_loop_trajectory_output_weights must be non-negative")
    if sum(config.closed_loop_trajectory_output_weights) <= 0.0:
        raise ValueError(
            "closed_loop_trajectory_output_weights must contain at least one positive value"
        )
    if config.monotonicity_weight > 0.0:
        raise ValueError(f"monotonicity regularization is not implemented for --model-kind {config.model_kind}")
    if config.loss_normalization != "none":
        raise ValueError(f"--model-kind {config.model_kind} already uses per-output target scaling; use --loss-normalization none")
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
    if is_contracting:
        if not 0.0 < config.contracting_gamma < 1.0:
            raise ValueError("contracting_gamma must be in (0, 1)")
        if config.contracting_state_bound <= 0.0:
            raise ValueError("contracting_state_bound must be positive")
        if config.contracting_temperature_scale <= 0.0:
            raise ValueError("contracting_temperature_scale must be positive")
        if config.contracting_temperature_delta_max_c < 0.0:
            raise ValueError("contracting_temperature_delta_max_c must be non-negative")
        if config.contracting_temperature_update not in (
            "auto",
            "absolute",
            "delta",
            "leaky_equilibrium",
            "bounded_equilibrium",
            "alpha_bounded_equilibrium",
        ):
            raise ValueError(
                "contracting_temperature_update must be 'auto', 'absolute', 'delta', "
                "'leaky_equilibrium', 'bounded_equilibrium', or "
                "'alpha_bounded_equilibrium'"
            )
        if config.contracting_transition_conditioning not in (
            "state_feedback",
            "exogenous",
            "exogenous_additive_feedback",
        ):
            raise ValueError(
                "contracting_transition_conditioning must be 'state_feedback' "
                "'exogenous', or 'exogenous_additive_feedback'"
            )
        if config.contracting_additive_feedback_gain_bound <= 0.0:
            raise ValueError("contracting_additive_feedback_gain_bound must be positive")
        if (
            config.contracting_temperature_update in (
                "leaky_equilibrium",
                "bounded_equilibrium",
                "alpha_bounded_equilibrium",
            )
            and config.contracting_temperature_delta_max_c <= 0.0
        ):
            raise ValueError(
                "equilibrium-based --contracting-temperature-update modes "
                "require --contracting-temperature-delta-max-c > 0"
            )
        if config.contracting_q_to_t_mode not in ("unconstrained", "positive_leaky"):
            raise ValueError(
                "contracting_q_to_t_mode must be 'unconstrained' or 'positive_leaky'"
            )
        if not config.contracting_q_to_t_time_constants_hours:
            raise ValueError(
                "contracting_q_to_t_time_constants_hours must contain at least one value"
            )
        if any(value <= 0.0 for value in config.contracting_q_to_t_time_constants_hours):
            raise ValueError(
                "contracting_q_to_t_time_constants_hours values must be positive"
            )
        if len(set(config.contracting_q_to_t_time_constants_hours)) != len(
            config.contracting_q_to_t_time_constants_hours
        ):
            raise ValueError(
                "contracting_q_to_t_time_constants_hours values must be unique"
            )
        if config.contracting_q_to_t_gain_min_c_per_w_m2 < 0.0:
            raise ValueError(
                "contracting_q_to_t_gain_min_c_per_w_m2 must be non-negative"
            )
        if (
            config.contracting_q_to_t_gain_max_c_per_w_m2
            <= config.contracting_q_to_t_gain_min_c_per_w_m2
        ):
            raise ValueError("contracting Q-to-T gain bounds must satisfy min < max")
        if (
            config.contracting_q_to_t_mode == "positive_leaky"
            and config.contracting_temperature_update not in (
                "leaky_equilibrium",
                "bounded_equilibrium",
                "alpha_bounded_equilibrium",
            )
        ):
            raise ValueError(
                "--contracting-q-to-t-mode positive_leaky requires "
                "an equilibrium-based --contracting-temperature-update"
            )
    if is_probabilistic:
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
        if config.prob_hp_scenario_mode not in ("expected", "bernoulli"):
            raise ValueError("prob_hp_scenario_mode must be 'expected' or 'bernoulli'")
        if config.prob_hp_emission_mode not in ("bounded", "legacy_lognormal_mean"):
            raise ValueError("prob_hp_emission_mode must be 'bounded' or 'legacy_lognormal_mean'")
        if config.prob_hp_training_mode not in ("expected", "straight_through"):
            raise ValueError("prob_hp_training_mode must be 'expected' or 'straight_through'")
        if config.prob_hp_concrete_temperature <= 0.0:
            raise ValueError("prob_hp_concrete_temperature must be positive")
        if config.prob_hp_activation_model not in (
            "independent",
            "persistent_markov",
            "power_history",
        ):
            raise ValueError(
                "prob_hp_activation_model must be 'independent', 'persistent_markov', "
                "or 'power_history'"
            )
        if config.prob_hp_persistent_latent_dim < 1:
            raise ValueError("prob_hp_persistent_latent_dim must be positive")
        if not 0.0 < config.prob_hp_controller_leak <= 1.0:
            raise ValueError("prob_hp_controller_leak must be in (0, 1]")
        if config.prob_hp_controller_noise_scale < 0.0:
            raise ValueError("prob_hp_controller_noise_scale must be non-negative")
        if config.prob_hp_history_hours <= 0.0:
            raise ValueError("prob_hp_history_hours must be positive")
        if (
            config.prob_hp_activation_model in ("persistent_markov", "power_history")
            and not is_contracting_probabilistic
        ):
            raise ValueError(
                f"--prob-hp-activation-model {config.prob_hp_activation_model} currently requires "
                "--model-kind closed_loop_hp_contracting_probabilistic"
            )
        if config.hp_active_power_nll_weight < 0.0:
            raise ValueError("hp_active_power_nll_weight must be non-negative")
        if config.hp_inactive_leakage_weight < 0.0:
            raise ValueError("hp_inactive_leakage_weight must be non-negative")
        if config.prob_softopt_weight < 0.0:
            raise ValueError("prob_softopt_weight must be non-negative")
        if config.prob_softopt_temperature <= 0.0:
            raise ValueError("prob_softopt_temperature must be positive")
        if config.prob_variogram_weight < 0.0:
            raise ValueError("prob_variogram_weight must be non-negative")
        if config.prob_variogram_power <= 0.0:
            raise ValueError("prob_variogram_power must be positive")
        if config.prob_variogram_output_weights:
            if len(config.prob_variogram_output_weights) != len(CLOSED_LOOP_TARGET_COLUMNS):
                raise ValueError(
                    "--prob-variogram-output-weights must contain exactly three values "
                    "for closed-loop models: Tin Qroom Pel_SH"
                )
            if any(weight < 0.0 for weight in config.prob_variogram_output_weights):
                raise ValueError("prob_variogram_output_weights must be non-negative")
            if sum(config.prob_variogram_output_weights) <= 0.0:
                raise ValueError("prob_variogram_output_weights must contain at least one positive value")
        if config.prob_variogram_setpoint_threshold_c < 0.0:
            raise ValueError("prob_variogram_setpoint_threshold_c must be non-negative")
        if config.prob_ires_weight < 0.0:
            raise ValueError("prob_ires_weight must be non-negative")
        if any(horizon <= 0.0 for horizon in config.prob_ires_horizons_hours):
            raise ValueError("prob_ires_horizons_hours must contain positive durations")
        if config.prob_ires_setpoint_threshold_c < 0.0:
            raise ValueError("prob_ires_setpoint_threshold_c must be non-negative")
        if config.prob_flex_kpi_crps_weight < 0.0:
            raise ValueError("prob_flex_kpi_crps_weight must be non-negative")
        if any(horizon <= 0.0 for horizon in config.prob_flex_kpi_crps_horizons_hours):
            raise ValueError("prob_flex_kpi_crps_horizons_hours must contain positive durations")
        if config.prob_flex_kpi_crps_setpoint_threshold_c < 0.0:
            raise ValueError("prob_flex_kpi_crps_setpoint_threshold_c must be non-negative")
        if config.prob_flex_kpi_crps_min_events < 1:
            raise ValueError("prob_flex_kpi_crps_min_events must be positive")
        if config.prob_flex_kpi_crps_ridge <= 0.0:
            raise ValueError("prob_flex_kpi_crps_ridge must be positive")
        if config.prob_flex_kpi_crps_controls not in ("none", "weather", "full"):
            raise ValueError("prob_flex_kpi_crps_controls must be 'none', 'weather', or 'full'")
        if config.prob_physics_weight < 0.0:
            raise ValueError("prob_physics_weight must be non-negative")
        if config.prob_horizon_weight_power < 0.0:
            raise ValueError("prob_horizon_weight_power must be non-negative")
        if any(lag < 1 for lag in config.prob_variogram_lags):
            raise ValueError("prob_variogram_lags must be positive")
        if config.closed_loop_stability_weight < 0.0:
            raise ValueError("closed_loop_stability_weight must be non-negative")
        if not (0.0 < config.closed_loop_stability_gamma < 1.0):
            raise ValueError("closed_loop_stability_gamma must be in (0, 1)")
        if config.closed_loop_stability_samples < 0:
            raise ValueError("closed_loop_stability_samples must be non-negative")
        if config.closed_loop_stability_aggregation not in ("mean", "max"):
            raise ValueError("closed_loop_stability_aggregation must be 'mean' or 'max'")
        if config.closed_loop_stability_every_steps < 1:
            raise ValueError("closed_loop_stability_every_steps must be positive")
        if config.closed_loop_stability_method not in ("exact_svd", "power_iteration"):
            raise ValueError(
                "closed_loop_stability_method must be 'exact_svd' or 'power_iteration'"
            )
        if config.closed_loop_stability_power_iterations < 1:
            raise ValueError("closed_loop_stability_power_iterations must be positive")
        if config.init_xi_weight_scale < 0.0:
            raise ValueError("init_xi_weight_scale must be non-negative")
        max_init_hp_active_log_sigma = 0.50 if config.prob_hp_emission_mode == "bounded" else 0.75
        if not (0.05 < config.init_hp_active_log_sigma < max_init_hp_active_log_sigma):
            raise ValueError(
                "init_hp_active_log_sigma must be in "
                f"(0.05, {max_init_hp_active_log_sigma:.2f}) for "
                f"prob_hp_emission_mode={config.prob_hp_emission_mode!r}"
            )
        if is_contracting_probabilistic and config.init_from_deterministic_artifact is not None:
            raise ValueError(
                "--init-from-deterministic-artifact is only implemented for "
                "--model-kind closed_loop_hp_probabilistic"
            )
    elif config.init_from_deterministic_artifact is not None:
        raise ValueError("--init-from-deterministic-artifact requires --model-kind closed_loop_hp_probabilistic")
    if config.checkpoint_metric not in (
        "auto",
        "train_rmse_c",
        "test_rmse_c",
        "train_total_nrmse",
        "test_total_nrmse",
    ):
        raise ValueError(
            "checkpoint_metric must be 'auto', 'train_rmse_c', 'test_rmse_c', "
            "'train_total_nrmse', or 'test_total_nrmse'"
        )
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
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
        fixed_test_profile_ids=config.fixed_test_profile_ids,
        excluded_profile_ids=config.excluded_profile_ids,
        heat_input_normalization="per_floor_area",
        hp_power_area_normalization=config.hp_power_area_normalization,
        input_feature_mode="base",
    )
    splits = load_closed_loop_result_splits(split_config)
    availability_input_index = list(splits.input_columns).index(
        SPACE_HEATING_AVAILABILITY_COLUMN
    )
    hp_controller_masked_input_indices = (
        ()
        if config.hp_controller_calendar_features
        else tuple(
            index
            for index, column in enumerate(splits.input_columns)
            if column in CLOSED_LOOP_CALENDAR_COLUMNS
        )
    )
    window_config = WindowConfig(
        sequence_length=config.sequence_length,
        stride=config.stride,
        target_alignment="same_time",
    )
    train_profiles = to_closed_loop_profiles(
        splits.train,
        hp_power_area_normalization=splits.hp_power_area_normalization,
        metadata_columns=splits.metadata_columns,
    )
    test_profiles = (
        to_closed_loop_profiles(
            splits.test,
            hp_power_area_normalization=splits.hp_power_area_normalization,
            metadata_columns=splits.metadata_columns,
        )
        if splits.test_ids
        else []
    )
    raw_train_windows = make_closed_loop_windows(train_profiles, window_config)
    raw_test_windows = (
        make_closed_loop_windows(test_profiles, window_config) if splits.test_ids else None
    )
    train_window_count = raw_train_windows.targets.shape[0]
    metadata_dim = raw_train_windows.metadata.shape[-1]
    input_dim = raw_train_windows.inputs.shape[-1]
    hp_pel_cap_w_m2, hp_qroom_cap_w_m2, hp_cop_cap, hp_energy_cap_wh_m2 = resolve_closed_loop_caps(
        raw_train_windows,
        config,
    )
    artifact_config = replace(
        config,
        hp_pel_cap_w_m2=hp_pel_cap_w_m2,
        hp_qroom_cap_w_m2=hp_qroom_cap_w_m2,
        hp_cop_cap=hp_cop_cap,
        hp_energy_cap_wh_m2=hp_energy_cap_wh_m2,
    )
    resolved_checkpoint_metric = resolve_checkpoint_metric(
        config.checkpoint_metric,
        has_test_windows=raw_test_windows is not None,
        prefer_total_nrmse=True,
    )

    scalers = fit_window_scalers(raw_train_windows)  # type: ignore[arg-type]
    input_mean = tuple(float(value) for value in np.asarray(scalers.inputs.mean, dtype=np.float32).reshape(-1))
    input_scale = tuple(float(value) for value in np.asarray(scalers.inputs.scale, dtype=np.float32).reshape(-1))
    target_mean_np = np.asarray(scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale_np = np.asarray(scalers.target.scale, dtype=np.float32).reshape(-1)
    target_mean = jnp.asarray(target_mean_np, dtype=jnp.float32)
    target_scale = jnp.asarray(target_scale_np, dtype=jnp.float32)
    prob_ires_horizon_steps = tuple(
        sorted(
            set(
                max(1, int(round(float(horizon) / config.hp_dt_hours)))
                for horizon in config.prob_ires_horizons_hours
            )
        )
    )
    prob_ires_horizons_hours = tuple(float(step * config.hp_dt_hours) for step in prob_ires_horizon_steps)
    if is_probabilistic and config.prob_ires_weight > 0.0:
        valid_ires_steps = [step for step in prob_ires_horizon_steps if 2 * step <= config.sequence_length]
        if not valid_ires_steps:
            raise ValueError(
                "prob_ires_horizons_hours has no valid horizons for the selected sequence_length; "
                "each horizon needs equally long pre and post windows."
            )
    prob_ires_setpoint_threshold_norm = (
        float(config.prob_ires_setpoint_threshold_c) / max(float(input_scale[0]), 1e-6)
    )
    prob_flex_kpi_crps_horizon_steps = tuple(
        sorted(
            set(
                max(1, int(round(float(horizon) / config.hp_dt_hours)))
                for horizon in config.prob_flex_kpi_crps_horizons_hours
            )
        )
    )
    prob_flex_kpi_crps_horizons_hours = tuple(
        float(step * config.hp_dt_hours) for step in prob_flex_kpi_crps_horizon_steps
    )
    if is_probabilistic and config.prob_flex_kpi_crps_weight > 0.0:
        valid_flex_kpi_steps = [
            step
            for step in prob_flex_kpi_crps_horizon_steps
            if 2 * step <= config.sequence_length
        ]
        if not valid_flex_kpi_steps:
            raise ValueError(
                "prob_flex_kpi_crps_horizons_hours has no valid horizons for the selected "
                "sequence_length; each horizon needs equally long pre and post windows."
            )
    prob_flex_kpi_crps_setpoint_threshold_norm = (
        float(config.prob_flex_kpi_crps_setpoint_threshold_c)
        / max(float(input_scale[0]), 1e-6)
    )
    flex_kpi_crps_design_dim = 3
    if config.prob_flex_kpi_crps_controls != "none":
        flex_kpi_crps_design_dim += (input_dim - 1) + min(3, input_dim - 1)
        if config.prob_flex_kpi_crps_controls == "full":
            flex_kpi_crps_design_dim += 4
    flex_kpi_crps_required_events = max(
        config.prob_flex_kpi_crps_min_events,
        flex_kpi_crps_design_dim + 1,
    )
    flex_kpi_crps_eligible_windows: tuple[int, ...] = ()
    if is_probabilistic and config.prob_flex_kpi_crps_weight > 0.0:
        raw_setpoint = np.asarray(raw_train_windows.inputs[:, :, 0], dtype=np.float32)
        eligible_counts = []
        for steps in prob_flex_kpi_crps_horizon_steps:
            if 2 * steps > config.sequence_length:
                eligible_counts.append(0)
                continue
            starts = np.arange(steps, config.sequence_length - steps + 1)
            setpoint_delta = raw_setpoint[:, starts] - raw_setpoint[:, starts - 1]
            event_count = np.sum(
                np.abs(setpoint_delta) >= config.prob_flex_kpi_crps_setpoint_threshold_c,
                axis=1,
            )
            eligible_counts.append(int(np.sum(event_count >= flex_kpi_crps_required_events)))
        flex_kpi_crps_eligible_windows = tuple(eligible_counts)
        if not any(flex_kpi_crps_eligible_windows):
            raise ValueError(
                "No training windows contain enough setpoint interventions for "
                "--prob-flex-kpi-crps-weight. Increase --sequence-length, reduce "
                "--prob-flex-kpi-crps-min-events, or use fewer regression controls. "
                f"required_events={flex_kpi_crps_required_events}"
            )
    prob_variogram_setpoint_threshold_norm = (
        float(config.prob_variogram_setpoint_threshold_c) / max(float(input_scale[0]), 1e-6)
    )
    raw_train_eval_windows = fixed_evaluation_subset(
        raw_train_windows,
        max_windows=config.train_eval_max_windows,
        seed=config.seed + 10_003,
    )
    assert isinstance(raw_train_eval_windows, ClosedLoopWindowedArrays)
    train_eval_windows = transform_closed_loop_windows(raw_train_eval_windows, scalers)
    test_windows = (
        transform_closed_loop_windows(raw_test_windows, scalers)
        if raw_test_windows is not None
        else None
    )
    scaled_train_profiles = transform_closed_loop_profiles(train_profiles, scalers)
    train_windows = (
        None
        if config.rotate_window_starts
        else make_closed_loop_windows(scaled_train_profiles, window_config)
    )
    del raw_train_windows, raw_train_eval_windows, raw_test_windows, train_profiles

    key = jax.random.PRNGKey(config.seed)
    model: ClosedLoopHPEmulator | ContractingClosedLoopHPEmulator | ProbabilisticClosedLoopModel
    if is_contracting_probabilistic:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            metadata_dim=metadata_dim,
            input_dim=input_dim,
            state_dim=config.state_dim,
            controller_state_dim=config.hp_controller_state_dim,
            latent_dim=config.prob_latent_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            input_encoder_dim=config.input_encoder_dim,
            input_encoder_hidden_dim=config.input_encoder_hidden_dim,
            input_encoder_depth=config.input_encoder_depth,
            process_noise_mode=config.prob_process_noise,
            process_noise_init=config.prob_process_noise_init,
            hp_emission_mode=config.prob_hp_emission_mode,
            hp_activation_model=config.prob_hp_activation_model,
            hp_persistent_latent_dim=config.prob_hp_persistent_latent_dim,
            hp_controller_leak=config.prob_hp_controller_leak,
            hp_controller_noise_scale=config.prob_hp_controller_noise_scale,
            hp_history_hours=config.prob_hp_history_hours,
            contraction_gamma=config.contracting_gamma,
            state_bound=config.contracting_state_bound,
            temperature_output_scale=config.contracting_temperature_scale,
            temperature_delta_max_c=config.contracting_temperature_delta_max_c,
            temperature_update_mode=config.contracting_temperature_update,
            transition_conditioning=config.contracting_transition_conditioning,
            additive_feedback_gain_bound=(
                config.contracting_additive_feedback_gain_bound
            ),
            hp_dt_hours=config.hp_dt_hours,
            hp_cop_floor=config.hp_cop_floor,
            hp_cop_cap=hp_cop_cap,
            hp_pel_cap_w_m2=hp_pel_cap_w_m2,
            hp_qroom_cap_w_m2=hp_qroom_cap_w_m2,
            hp_energy_cap_wh_m2=hp_energy_cap_wh_m2,
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=tuple(float(value) for value in target_mean_np),
            target_scale=tuple(float(value) for value in target_scale_np),
            availability_input_index=availability_input_index,
            bptt_truncate_steps=config.bptt_truncate_steps,
            hp_controller_masked_input_indices=hp_controller_masked_input_indices,
            thermostat_demand_mode=config.hp_thermostat_demand_mode,
            thermostat_slope_min=config.hp_thermostat_slope_min,
            thermostat_slope_max=config.hp_thermostat_slope_max,
            thermostat_threshold_min_c=config.hp_thermostat_threshold_min_c,
            thermostat_threshold_max_c=config.hp_thermostat_threshold_max_c,
            q_to_t_mode=config.contracting_q_to_t_mode,
            q_to_t_time_constants_hours=config.contracting_q_to_t_time_constants_hours,
            q_to_t_gain_min_c_per_w_m2=config.contracting_q_to_t_gain_min_c_per_w_m2,
            q_to_t_gain_max_c_per_w_m2=config.contracting_q_to_t_gain_max_c_per_w_m2,
            key=key,
        )
    elif is_probabilistic:
        model = ProbabilisticClosedLoopHPEmulator(
            metadata_dim=metadata_dim,
            input_dim=input_dim,
            state_dim=config.state_dim,
            controller_state_dim=config.hp_controller_state_dim,
            latent_dim=config.prob_latent_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            input_encoder_dim=config.input_encoder_dim,
            input_encoder_hidden_dim=config.input_encoder_hidden_dim,
            input_encoder_depth=config.input_encoder_depth,
            process_noise_mode=config.prob_process_noise,
            process_noise_init=config.prob_process_noise_init,
            hp_emission_mode=config.prob_hp_emission_mode,
            hp_dt_hours=config.hp_dt_hours,
            hp_cop_floor=config.hp_cop_floor,
            hp_cop_cap=hp_cop_cap,
            hp_pel_cap_w_m2=hp_pel_cap_w_m2,
            hp_qroom_cap_w_m2=hp_qroom_cap_w_m2,
            hp_energy_cap_wh_m2=hp_energy_cap_wh_m2,
            schur_gamma=config.schur_gamma,
            pf_lambda_min=config.pf_lambda_min,
            schur_mode=config.schur_mode,  # type: ignore[arg-type]
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=tuple(float(value) for value in target_mean_np),
            target_scale=tuple(float(value) for value in target_scale_np),
            availability_input_index=availability_input_index,
            bptt_truncate_steps=config.bptt_truncate_steps,
            key=key,
        )
        if config.init_from_deterministic_artifact is not None:
            key, init_key = jax.random.split(key)
            model = initialize_probabilistic_closed_loop_from_artifact(
                model,
                config.init_from_deterministic_artifact,
                key=init_key,
                xi_weight_scale=config.init_xi_weight_scale,
                hp_active_log_sigma=config.init_hp_active_log_sigma,
            )
    elif is_contracting:
        model = ContractingClosedLoopHPEmulator(
            metadata_dim=metadata_dim,
            input_dim=input_dim,
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            input_encoder_dim=config.input_encoder_dim,
            input_encoder_hidden_dim=config.input_encoder_hidden_dim,
            input_encoder_depth=config.input_encoder_depth,
            contraction_gamma=config.contracting_gamma,
            state_bound=config.contracting_state_bound,
            temperature_output_scale=config.contracting_temperature_scale,
            temperature_delta_max_c=config.contracting_temperature_delta_max_c,
            temperature_update_mode=config.contracting_temperature_update,
            transition_conditioning=config.contracting_transition_conditioning,
            additive_feedback_gain_bound=(
                config.contracting_additive_feedback_gain_bound
            ),
            hp_dt_hours=config.hp_dt_hours,
            hp_cop_floor=config.hp_cop_floor,
            hp_cop_cap=hp_cop_cap,
            hp_pel_cap_w_m2=hp_pel_cap_w_m2,
            hp_qroom_cap_w_m2=hp_qroom_cap_w_m2,
            hp_energy_cap_wh_m2=hp_energy_cap_wh_m2,
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=tuple(float(value) for value in target_mean_np),
            target_scale=tuple(float(value) for value in target_scale_np),
            availability_input_index=availability_input_index,
            bptt_truncate_steps=config.bptt_truncate_steps,
            hp_controller_masked_input_indices=hp_controller_masked_input_indices,
            thermostat_demand_mode=config.hp_thermostat_demand_mode,
            thermostat_slope_min=config.hp_thermostat_slope_min,
            thermostat_slope_max=config.hp_thermostat_slope_max,
            thermostat_threshold_min_c=config.hp_thermostat_threshold_min_c,
            thermostat_threshold_max_c=config.hp_thermostat_threshold_max_c,
            q_to_t_mode=config.contracting_q_to_t_mode,
            q_to_t_time_constants_hours=config.contracting_q_to_t_time_constants_hours,
            q_to_t_gain_min_c_per_w_m2=config.contracting_q_to_t_gain_min_c_per_w_m2,
            q_to_t_gain_max_c_per_w_m2=config.contracting_q_to_t_gain_max_c_per_w_m2,
            key=key,
        )
    else:
        model = ClosedLoopHPEmulator(
            metadata_dim=metadata_dim,
            input_dim=input_dim,
            state_dim=config.state_dim,
            controller_state_dim=config.hp_controller_state_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            input_encoder_dim=config.input_encoder_dim,
            input_encoder_hidden_dim=config.input_encoder_hidden_dim,
            input_encoder_depth=config.input_encoder_depth,
            hp_dt_hours=config.hp_dt_hours,
            hp_cop_floor=config.hp_cop_floor,
            hp_cop_cap=hp_cop_cap,
            hp_pel_cap_w_m2=hp_pel_cap_w_m2,
            hp_qroom_cap_w_m2=hp_qroom_cap_w_m2,
            hp_energy_cap_wh_m2=hp_energy_cap_wh_m2,
            schur_gamma=config.schur_gamma,
            pf_lambda_min=config.pf_lambda_min,
            schur_mode=config.schur_mode,  # type: ignore[arg-type]
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=tuple(float(value) for value in target_mean_np),
            target_scale=tuple(float(value) for value in target_scale_np),
            availability_input_index=availability_input_index,
            bptt_truncate_steps=config.bptt_truncate_steps,
            key=key,
        )
    optimizer = build_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(config.seed)
    model_checkpoint_dir = config.model_checkpoint_dir or config.output_dir / "model_checkpoints"
    model_artifact_dir = model_checkpoint_dir / config.model_kind
    diagnostics_path = (
        model_artifact_dir
        / f"epoch_diagnostics_{config.model_kind}_{diagnostic_run_id()}.csv"
    )

    print(f"model_kind={config.model_kind}")
    if config.init_from_deterministic_artifact is not None:
        print(
            "probabilistic_initialization=deterministic "
            f"artifact={config.init_from_deterministic_artifact} "
            f"xi_weight_scale={config.init_xi_weight_scale} "
            f"hp_active_log_sigma={config.init_hp_active_log_sigma}"
        )
    print("temperature_alignment=u[t] -> Tin[t+1]")
    print("power_alignment=u[t] -> Qroom[t], Pel_SH[t]")
    bptt_hours = config.bptt_truncate_steps * config.hp_dt_hours
    if config.bptt_truncate_steps > 0:
        print(
            "bptt_truncate="
            f"steps={config.bptt_truncate_steps} "
            f"hours={bptt_hours:.4g} "
            "forward_horizon=full "
            "use_0_for_full_bptt"
        )
    else:
        print("bptt_truncate=disabled full_bptt=true")
    print(f"target_columns={list(CLOSED_LOOP_TARGET_COLUMNS)}")
    print(f"input_columns={list(splits.input_columns)}")
    print(
        "hp_thermostat_demand "
        f"mode={config.hp_thermostat_demand_mode} "
        f"slope_bounds=[{config.hp_thermostat_slope_min:.6g},"
        f"{config.hp_thermostat_slope_max:.6g}]_per_C "
        f"threshold_bounds_c=[{config.hp_thermostat_threshold_min_c:.6g},"
        f"{config.hp_thermostat_threshold_max_c:.6g}] "
        f"setpoint_generic_encoder={'masked' if config.hp_thermostat_demand_mode == 'monotone' else 'visible'}"
    )
    masked_hp_columns = [
        splits.input_columns[index] for index in hp_controller_masked_input_indices
    ]
    print(
        "hp_controller_calendar_features="
        f"{'enabled' if config.hp_controller_calendar_features else 'disabled'} "
        f"masked_columns={masked_hp_columns} "
        "thermal_calendar_features=retained"
    )
    print(
        "space_heating_availability=hard_gate "
        f"input_index={availability_input_index} off=May15-Sep30 "
        "gated_outputs=[Qroom,Pel_SH] thermal_direct_input=false"
    )
    print(
        "closed_loop_hp_profile_filter=hp_ref_capacity_W>0,hp_size_binding=SH "
        f"kept_candidates={len(splits.candidate_ids)} "
        f"dropped_non_hp={len(splits.dropped_non_hp_ids)} "
        f"dropped_non_sh_hp={len(splits.dropped_non_sh_hp_ids)} "
        f"dropped_total={len(splits.dropped_ids)} "
        f"selected={len(splits.selected_ids)}"
    )
    print(f"train_profiles={len(splits.train_ids)} test_profiles={len(splits.test_ids)}")
    print(f"train_windows={train_window_count}")
    if test_windows is not None:
        print(f"test_windows={test_windows.targets.shape[0]}")
    if is_contracting:
        print(
            "closed_loop_hp_contracting=enabled "
            f"gamma={config.contracting_gamma} "
            f"state_bound={config.contracting_state_bound} "
            f"temperature_output_scale={config.contracting_temperature_scale} "
            f"temperature_delta_max_c={config.contracting_temperature_delta_max_c} "
            f"temperature_update={config.contracting_temperature_update} "
            f"transition_conditioning={config.contracting_transition_conditioning} "
            f"additive_feedback_gain_bound="
            f"{config.contracting_additive_feedback_gain_bound} "
            "thermal_matrix=time_varying "
            "pointwise_matrix_guarantee=||M_t||_2<=gamma"
        )
        print(
            "closed_loop_hp_contracting_causal_structure="
            "Tset_and_availability_to_HP_head; "
            + (
                (
                    "thermal_transition_uses=[u_without_Tset_or_availability]; "
                    if config.contracting_transition_conditioning == "exogenous"
                    else (
                        "thermal_matrix_uses=[u_without_Tset_or_availability]; "
                        "thermal_additive_feedback_uses="
                        "[u_without_Tset_or_availability,Tin,Tout_minus_Tin]; "
                        if config.contracting_transition_conditioning
                        == "exogenous_additive_feedback"
                        else "thermal_transition_uses="
                        "[u_without_Tset_or_availability,Tin,Tout_minus_Tin]; "
                    )
                )
                +
                "Qroom_to_temperature=separate_positive_leaky_path"
                if config.contracting_q_to_t_mode == "positive_leaky"
                else (
                    (
                        "thermal_transition_uses=[u_without_Tset_or_availability]; "
                        if config.contracting_transition_conditioning == "exogenous"
                        else (
                            "thermal_matrix_uses=[u_without_Tset_or_availability]; "
                            "thermal_additive_feedback_uses="
                            "[u_without_Tset_or_availability,Qroom,Tin,Tout_minus_Tin]; "
                            if config.contracting_transition_conditioning
                            == "exogenous_additive_feedback"
                            else "thermal_transition_uses="
                            "[u_without_Tset_or_availability,Qroom,Tin,Tout_minus_Tin]; "
                        )
                    )
                    + "Qroom_to_temperature=unconstrained"
                )
            )
        )
        print(
            "closed_loop_hp_contracting_q_to_t "
            f"mode={config.contracting_q_to_t_mode} "
            f"time_constants_hours={list(config.contracting_q_to_t_time_constants_hours)} "
            "total_gain_bounds_c_per_w_m2="
            f"[{config.contracting_q_to_t_gain_min_c_per_w_m2:.6g},"
            f"{config.contracting_q_to_t_gain_max_c_per_w_m2:.6g}]"
        )
        if config.contracting_temperature_update == "leaky_equilibrium":
            temperature_target_scale = max(abs(float(target_scale_np[0])), 1e-6)
            temperature_alpha_max = min(
                max(
                    config.contracting_temperature_delta_max_c
                    / (2.0 * config.contracting_temperature_scale * temperature_target_scale),
                    1e-6,
                ),
                1.0,
            )
            alpha_conditioning = (
                "metadata_and_xi"
                if is_contracting_probabilistic
                else "metadata"
            )
            print(
                "closed_loop_hp_contracting_temperature_update=leaky_equilibrium "
                f"alpha_max={temperature_alpha_max:.6g} "
                f"max_step_c={config.contracting_temperature_delta_max_c:.6g} "
                "alpha_is_rollout_constant=true "
                f"alpha_conditioning={alpha_conditioning}"
            )
        elif config.contracting_temperature_update == "bounded_equilibrium":
            print(
                "closed_loop_hp_contracting_temperature_update=bounded_equilibrium "
                f"max_step_c={config.contracting_temperature_delta_max_c:.6g} "
                "alpha=removed update=delta_max*tanh((T_eq-T)/delta_max)"
            )
        elif config.contracting_temperature_update == "alpha_bounded_equilibrium":
            alpha_conditioning = (
                "metadata_and_xi"
                if is_contracting_probabilistic
                else "metadata"
            )
            print(
                "closed_loop_hp_contracting_temperature_update=alpha_bounded_equilibrium "
                f"max_step_c={config.contracting_temperature_delta_max_c:.6g} "
                "alpha_range=(0,1) alpha_is_rollout_constant=true "
                f"alpha_conditioning={alpha_conditioning} "
                "update=delta_max*tanh(alpha*(T_eq-T)/delta_max)"
            )
    else:
        print(
            "closed_loop_hp "
            f"controller_state_dim={config.hp_controller_state_dim} "
            f"dt_hours={config.hp_dt_hours} "
            f"mode_loss_weight={config.hp_mode_loss_weight} "
            f"cop_floor={config.hp_cop_floor}"
        )
    print(
        "closed_loop_hp_power_normalization "
        f"qroom_area=zone_floor_area pel_area={config.hp_power_area_normalization}"
    )
    if is_contracting:
        print(
            "closed_loop_hp_contracting_caps "
            f"pel_cap_w_m2={hp_pel_cap_w_m2:.6g} "
            f"qroom_cap_w_m2={hp_qroom_cap_w_m2:.6g} "
            f"cop_cap={hp_cop_cap:.6g} "
            f"energy_cap_wh_m2={hp_energy_cap_wh_m2:.6g} "
            f"auto_cap_factor={config.hp_cap_factor:.6g}"
        )
    else:
        print(
            "closed_loop_hp_caps "
            f"pel_cap_w_m2={hp_pel_cap_w_m2:.6g} "
            f"qroom_cap_w_m2={hp_qroom_cap_w_m2:.6g} "
            f"cop_cap={hp_cop_cap:.6g} "
            f"energy_cap_wh_m2={hp_energy_cap_wh_m2:.6g} "
            f"auto_cap_factor={config.hp_cap_factor:.6g}"
        )
    print(
        "closed_loop_trajectory_output_weights="
        f"{list(config.closed_loop_trajectory_output_weights)} "
        "order=[Tin,Qroom,Pel_SH] normalized_to_mean_one=true"
    )
    if is_probabilistic:
        print(
            f"{config.model_kind}=enabled "
            f"particles={config.prob_particles} "
            f"eval_particles={config.prob_eval_particles} "
            f"plot_particles={config.prob_plot_particles} "
            f"latent_dim={config.prob_latent_dim} "
            f"process_noise={config.prob_process_noise} "
            f"hp_scenario_mode={config.prob_hp_scenario_mode} "
            f"hp_emission_mode={config.prob_hp_emission_mode} "
            f"hp_training_mode={config.prob_hp_training_mode} "
            f"hp_concrete_temperature={config.prob_hp_concrete_temperature} "
            f"hp_activation_model={config.prob_hp_activation_model} "
            f"hp_persistent_latent_dim={config.prob_hp_persistent_latent_dim} "
            f"hp_controller_leak={config.prob_hp_controller_leak} "
            f"hp_controller_noise_scale={config.prob_hp_controller_noise_scale} "
            f"hp_history_hours={config.prob_hp_history_hours}"
        )
        print(
            "probabilistic_closed_loop_loss "
            "energy_score_weight=1.0 "
            f"trajectory_output_weights={list(config.closed_loop_trajectory_output_weights)} "
            f"variogram_weight={config.prob_variogram_weight} "
            f"variogram_lags={list(config.prob_variogram_lags)} "
            f"variogram_power={config.prob_variogram_power} "
            f"variogram_output_weights="
            f"{list(config.prob_variogram_output_weights) if config.prob_variogram_output_weights else 'equal'} "
            f"variogram_constant_setpoint_only={config.prob_variogram_constant_setpoint_only} "
            f"variogram_setpoint_threshold_c={config.prob_variogram_setpoint_threshold_c} "
            f"ires_weight={config.prob_ires_weight} "
            f"ires_horizons_hours={list(prob_ires_horizons_hours)} "
            f"ires_horizon_steps={list(prob_ires_horizon_steps)} "
            f"ires_setpoint_threshold_c={config.prob_ires_setpoint_threshold_c} "
            f"flex_kpi_crps_weight={config.prob_flex_kpi_crps_weight} "
            f"flex_kpi_crps_horizons_hours={list(prob_flex_kpi_crps_horizons_hours)} "
            f"flex_kpi_crps_horizon_steps={list(prob_flex_kpi_crps_horizon_steps)} "
            f"flex_kpi_crps_setpoint_threshold_c="
            f"{config.prob_flex_kpi_crps_setpoint_threshold_c} "
            f"flex_kpi_crps_min_events={config.prob_flex_kpi_crps_min_events} "
            f"flex_kpi_crps_ridge={config.prob_flex_kpi_crps_ridge} "
            f"flex_kpi_crps_controls={config.prob_flex_kpi_crps_controls} "
            f"flex_kpi_crps_design_dim={flex_kpi_crps_design_dim} "
            f"flex_kpi_crps_required_events={flex_kpi_crps_required_events} "
            f"flex_kpi_crps_eligible_train_windows="
            f"{list(flex_kpi_crps_eligible_windows)} "
            f"softopt_weight={config.prob_softopt_weight} "
            f"softopt_temperature={config.prob_softopt_temperature} "
            f"physics_weight={config.prob_physics_weight} "
            f"hp_active_power_nll_weight={config.hp_active_power_nll_weight} "
            f"hp_inactive_leakage_weight={config.hp_inactive_leakage_weight}"
        )
        print(
            "closed_loop_stability="
            f"{'enabled' if config.closed_loop_stability_weight > 0.0 else 'disabled'} "
            f"weight={config.closed_loop_stability_weight} "
            f"gamma={config.closed_loop_stability_gamma} "
            f"samples={config.closed_loop_stability_samples} "
            f"aggregation={config.closed_loop_stability_aggregation} "
            f"every_steps={config.closed_loop_stability_every_steps} "
            f"method={config.closed_loop_stability_method} "
            f"power_iterations={config.closed_loop_stability_power_iterations} "
            "state=active_recurrent_state"
        )
    if config.input_encoder_feedback != "none":
        print(
            "warning=closed_loop_hp_ignores_input_encoder_feedback; "
            "temperature, outdoor gap, and setpoint gap are always included internally"
        )
    encoder_label = "hp_control_input_encoder" if is_contracting else "thermal_input_encoder"
    print(
        f"{encoder_label}="
        f"{'enabled' if config.input_encoder_dim is not None else 'disabled'} "
        f"encoded_dim={config.input_encoder_dim or input_dim} "
        f"hidden_dim={config.input_encoder_hidden_dim or config.hidden_dim} "
        f"depth={config.input_encoder_depth}"
    )
    if is_contracting:
        print(
            "stable_parametrization=time_varying_bounded_thermal_matrix "
            f"contraction_gamma={config.contracting_gamma}"
        )
    else:
        print(
            f"schur_mode={config.schur_mode} "
            f"schur_gamma={config.schur_gamma} "
            f"pf_lambda_min={config.pf_lambda_min}"
        )
    print(
        "optimizer="
        f"adam learning_rate={config.learning_rate} "
        f"gradient_clip_norm={config.gradient_clip_norm} "
        f"skip_nonfinite_updates={config.skip_nonfinite_updates} "
        f"max_consecutive_nonfinite_updates={config.max_consecutive_nonfinite_updates} "
        f"validate_candidate_updates={config.validate_candidate_updates} "
        f"skip_grad_norm_above={config.skip_grad_norm_above} "
        f"log_update_diagnostics={config.log_update_diagnostics}"
    )
    print(f"checkpoint_metric={resolved_checkpoint_metric}")
    print(
        "window_start_rotation="
        f"{'enabled' if config.rotate_window_starts else 'disabled'} "
        f"stride={config.stride} epochs={config.epochs} validation_offset=0"
    )
    print(
        "evaluation_schedule="
        f"train_fixed_subset:{train_eval_windows.targets.shape[0]}/{train_window_count} "
        f"full_train_every_epochs:{config.full_train_eval_every_epochs} "
        f"test_full_every_epoch:{test_windows is not None}"
    )
    if config.early_stopping_patience is not None:
        print(
            "early_stopping=enabled "
            f"patience={config.early_stopping_patience} "
            f"min_delta={config.early_stopping_min_delta}"
        )
    if config.save_model or config.save_model_every_epochs > 0:
        print(f"model_checkpoint_dir={model_checkpoint_dir}")
        print(f"model_artifact_dir={model_artifact_dir}")
    print(f"epoch_diagnostics_csv={diagnostics_path}")
    if config.save_model:
        print("save_selected_model=enabled")
    if config.save_model_every_epochs > 0:
        print(f"save_model_every_epochs={config.save_model_every_epochs}")

    best_model = model
    best_epoch = 0
    best_metric = float("inf")
    best_train_eval: dict[str, float] | None = None
    best_test_eval: dict[str, float] | None = None
    epochs_without_improvement = 0
    optimizer_step = 0
    training_started_at = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        if config.rotate_window_starts and epoch > 1:
            del epoch_train_windows
        train_window_offset = (
            rotating_window_start_offset(epoch, config.epochs, config.stride)
            if config.rotate_window_starts
            else 0
        )
        epoch_train_windows = (
            make_closed_loop_windows(
                scaled_train_profiles,
                window_config,
                start_offset=train_window_offset,
            )
            if config.rotate_window_starts
            else train_windows
        )
        assert epoch_train_windows is not None
        losses = []
        prob_loss_components = []
        update_stats_values = []
        skipped_update_batches = 0
        stability_penalty_batches = 0
        first_skipped_update_batch: int | None = None
        skipped_update_failure_counts = np.zeros((len(UPDATE_FINITE_FLAG_NAMES),), dtype=np.int64)
        for batch_idx, (metadata, inputs, initial_temperature, targets) in enumerate(
            minibatches(epoch_train_windows, batch_size=config.batch_size, rng=rng, shuffle=True),
            start=1,
        ):
            if is_probabilistic:
                assert isinstance(
                    model,
                    (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator),
                )
                key, step_key = jax.random.split(key)
                apply_stability_penalty = (
                    config.closed_loop_stability_weight > 0.0
                    and optimizer_step % config.closed_loop_stability_every_steps == 0
                )
                stability_weight = (
                    config.closed_loop_stability_weight
                    if apply_stability_penalty
                    else 0.0
                )
                if apply_stability_penalty:
                    stability_penalty_batches += 1
                model, opt_state, loss, components, update_flags, update_stats = (
                    probabilistic_closed_loop_train_step(
                    model,
                    opt_state,
                    optimizer,
                    jnp.asarray(metadata),
                    jnp.asarray(inputs),
                    jnp.asarray(initial_temperature),
                    jnp.asarray(targets),
                    step_key,
                    config.prob_particles,
                    config.prob_hp_training_mode,
                    jnp.asarray(config.prob_hp_concrete_temperature, dtype=jnp.float32),
                    config.prob_hp_activation_model,
                    target_mean,
                    target_scale,
                    config.hp_mode_loss_weight,
                    config.heat_on_threshold,
                    config.hp_active_power_nll_weight,
                    config.hp_inactive_leakage_weight,
                    config.prob_softopt_weight,
                    config.prob_softopt_temperature,
                    config.closed_loop_trajectory_output_weights,
                    config.prob_variogram_weight,
                    config.prob_variogram_lags,
                    config.prob_variogram_power,
                    config.prob_variogram_output_weights,
                    config.prob_variogram_constant_setpoint_only,
                    prob_variogram_setpoint_threshold_norm,
                    config.prob_ires_weight,
                    prob_ires_horizon_steps,
                    prob_ires_horizons_hours,
                    prob_ires_setpoint_threshold_norm,
                    config.prob_flex_kpi_crps_weight,
                    prob_flex_kpi_crps_horizon_steps,
                    prob_flex_kpi_crps_horizons_hours,
                    prob_flex_kpi_crps_setpoint_threshold_norm,
                    config.prob_flex_kpi_crps_min_events,
                    config.prob_flex_kpi_crps_ridge,
                    config.prob_flex_kpi_crps_controls,
                    config.prob_physics_weight,
                    config.prob_horizon_weight_power,
                    stability_weight,
                    config.closed_loop_stability_gamma,
                    config.closed_loop_stability_samples,
                    config.closed_loop_stability_aggregation,
                    config.closed_loop_stability_method,
                    config.closed_loop_stability_power_iterations,
                    config.skip_nonfinite_updates,
                    config.validate_candidate_updates,
                    config.skip_grad_norm_above,
                )
                )
                prob_loss_components.append(np.asarray(components))
            else:
                model, opt_state, loss, update_flags, update_stats = closed_loop_train_step(
                    model,
                    opt_state,
                    optimizer,
                    jnp.asarray(metadata),
                    jnp.asarray(inputs),
                    jnp.asarray(initial_temperature),
                    jnp.asarray(targets),
                    target_mean,
                    target_scale,
                    config.closed_loop_trajectory_output_weights,
                    config.hp_mode_loss_weight,
                    config.heat_on_threshold,
                    config.skip_nonfinite_updates,
                    config.validate_candidate_updates,
                    config.skip_grad_norm_above,
                )
            update_stats_values.append(np.asarray(update_stats, dtype=float))
            update_flags_np = np.asarray(update_flags, dtype=bool)
            if config.skip_nonfinite_updates and not bool(np.all(update_flags_np)):
                skipped_update_batches += 1
                if first_skipped_update_batch is None:
                    first_skipped_update_batch = batch_idx
                skipped_update_failure_counts += (~update_flags_np).astype(np.int64)
            losses.append(float(loss))
            optimizer_step += 1
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break

        if is_probabilistic:
            assert isinstance(
                model,
                (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator),
            )
            train_eval = evaluate_probabilistic_closed_loop(
                model,
                train_eval_windows,
                scalers,
                batch_size=config.batch_size,
                num_particles=config.prob_eval_particles,
            )
        else:
            train_eval = evaluate_closed_loop(
                model,
                train_eval_windows,
                scalers,
                batch_size=config.batch_size,
            )
        full_train_eval: dict[str, float] | None = None
        if (
            config.full_train_eval_every_epochs > 0
            and epoch % config.full_train_eval_every_epochs == 0
            and train_eval_windows.targets.shape[0] < epoch_train_windows.targets.shape[0]
        ):
            if is_probabilistic:
                assert isinstance(
                    model,
                    (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator),
                )
                full_train_eval = evaluate_probabilistic_closed_loop(
                    model,
                    epoch_train_windows,
                    scalers,
                    batch_size=config.batch_size,
                    num_particles=config.prob_eval_particles,
                )
            else:
                full_train_eval = evaluate_closed_loop(
                    model,
                    epoch_train_windows,
                    scalers,
                    batch_size=config.batch_size,
                )
        train_loss_mean, nonfinite_train_batches = finite_mean(losses)
        rho_max = sample_spectral_radius(model, epoch_train_windows)
        message = (
            f"epoch={epoch:03d} train_loss={train_loss_mean:.6f} "
            f"train_total_nrmse={train_eval['total_nrmse']:.4f} "
            f"train_rmse_c={train_eval['rmse_c']:.4f} "
            f"train_qroom_rmse_w_m2={train_eval['qroom_rmse_w_m2']:.4f} "
            f"train_pel_rmse_w_m2={train_eval['pel_rmse_w_m2']:.4f} "
            f"train_eval_windows={train_eval_windows.targets.shape[0]} "
            f"train_window_offset={train_window_offset} "
            f"rho_max={rho_max}"
        )
        if full_train_eval is not None:
            message += (
                f" full_train_total_nrmse={full_train_eval['total_nrmse']:.4f}"
                f" full_train_rmse_c={full_train_eval['rmse_c']:.4f}"
                f" full_train_qroom_rmse_w_m2={full_train_eval['qroom_rmse_w_m2']:.4f}"
                f" full_train_pel_rmse_w_m2={full_train_eval['pel_rmse_w_m2']:.4f}"
                f" full_train_eval_windows={epoch_train_windows.targets.shape[0]}"
            )
        if nonfinite_train_batches > 0:
            message += f" nonfinite_train_batches={nonfinite_train_batches}"
        if is_probabilistic and config.closed_loop_stability_weight > 0.0:
            message += f" stability_penalty_batches={stability_penalty_batches}"
        skipped_update_reason_counts = {
            name: int(count)
            for name, count in zip(UPDATE_FINITE_FLAG_NAMES, skipped_update_failure_counts)
        }
        if skipped_update_batches > 0:
            message += f" skipped_update_batches={skipped_update_batches}"
            if first_skipped_update_batch is not None:
                message += f" first_skipped_update_batch={first_skipped_update_batch}"
            reason_parts = [
                f"{name}:{count}"
                for name, count in skipped_update_reason_counts.items()
                if count > 0
            ]
            if reason_parts:
                message += f" skipped_update_reasons=[{','.join(reason_parts)}]"
        update_stat_maxima = {f"max_{name}": np.nan for name in UPDATE_STAT_NAMES}
        if update_stats_values and (config.log_update_diagnostics or skipped_update_batches > 0):
            stats_array = np.asarray(update_stats_values, dtype=float)
            finite_rows = np.all(np.isfinite(stats_array), axis=1)
            if finite_rows.any():
                max_stats = np.nanmax(stats_array[finite_rows], axis=0)
                update_stat_maxima = {
                    f"max_{name}": float(value)
                    for name, value in zip(UPDATE_STAT_NAMES, max_stats)
                }
                message += " " + " ".join(
                    f"max_{name}={value:.6g}"
                    for name, value in zip(UPDATE_STAT_NAMES, max_stats)
                )
        component_means: np.ndarray | None = None
        nonfinite_component_batches = 0
        if is_probabilistic and prob_loss_components:
            component_means, nonfinite_component_batches = finite_column_means(prob_loss_components)
            assert component_means is not None
            if nonfinite_component_batches > 0:
                message += f" nonfinite_component_batches={nonfinite_component_batches}"
            message += " " + " ".join(
                f"train_loss_{name}={value:.6f}"
                for name, value in zip(PROB_CLOSED_LOOP_LOSS_COMPONENT_NAMES, component_means)
            )
            message += (
                f" train_median_rmse_c={train_eval['median_rmse_c']:.4f}"
                f" train_mean_median_rmse_c={train_eval['mean_median_rmse_c']:.4f}"
                f" train_max_abs_error_c={train_eval['max_abs_error_c']:.4f}"
                f" train_worst_profile={metric_int(train_eval, 'max_abs_error_profile_id')}"
                f" train_worst_start={metric_int(train_eval, 'max_abs_error_start')}"
                f" train_worst_step={metric_int(train_eval, 'max_abs_error_step')}"
                f" train_worst_pred_c={train_eval['max_abs_error_pred_c']:.4f}"
                f" train_worst_target_c={train_eval['max_abs_error_target_c']:.4f}"
                f" train_pred_range_c=[{train_eval['min_pred_c']:.4f},{train_eval['max_pred_c']:.4f}]"
                f" train_particle_max_abs_error_c={train_eval['particle_max_abs_error_c']:.4f}"
                f" train_particle_worst_profile={metric_int(train_eval, 'particle_worst_profile_id')}"
                f" train_particle_worst_start={metric_int(train_eval, 'particle_worst_start')}"
                f" train_particle_worst_k={metric_int(train_eval, 'particle_worst_index')}"
                f" train_particle_worst_step={metric_int(train_eval, 'particle_worst_step')}"
                f" train_particle_worst_pred_c={train_eval['particle_worst_pred_c']:.4f}"
                f" train_particle_worst_target_c={train_eval['particle_worst_target_c']:.4f}"
                f" train_particle_pred_range_c=[{train_eval['particle_min_pred_c']:.4f},"
                f"{train_eval['particle_max_pred_c']:.4f}]"
            )
        test_eval: dict[str, float] | None = None
        if test_windows is not None:
            if is_probabilistic:
                assert isinstance(
                    model,
                    (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator),
                )
                test_eval = evaluate_probabilistic_closed_loop(
                    model,
                    test_windows,
                    scalers,
                    batch_size=config.batch_size,
                    num_particles=config.prob_eval_particles,
                )
            else:
                test_eval = evaluate_closed_loop(
                    model,
                    test_windows,
                    scalers,
                    batch_size=config.batch_size,
                )
            message += (
                f" test_total_nrmse={test_eval['total_nrmse']:.4f}"
                f" test_rmse_c={test_eval['rmse_c']:.4f}"
                f" test_qroom_rmse_w_m2={test_eval['qroom_rmse_w_m2']:.4f}"
                f" test_pel_rmse_w_m2={test_eval['pel_rmse_w_m2']:.4f}"
            )
            if is_probabilistic:
                message += (
                    f" test_median_rmse_c={test_eval['median_rmse_c']:.4f}"
                    f" test_mean_median_rmse_c={test_eval['mean_median_rmse_c']:.4f}"
                    f" test_max_abs_error_c={test_eval['max_abs_error_c']:.4f}"
                    f" test_worst_profile={metric_int(test_eval, 'max_abs_error_profile_id')}"
                    f" test_worst_start={metric_int(test_eval, 'max_abs_error_start')}"
                    f" test_worst_step={metric_int(test_eval, 'max_abs_error_step')}"
                    f" test_worst_pred_c={test_eval['max_abs_error_pred_c']:.4f}"
                    f" test_worst_target_c={test_eval['max_abs_error_target_c']:.4f}"
                    f" test_pred_range_c=[{test_eval['min_pred_c']:.4f},{test_eval['max_pred_c']:.4f}]"
                    f" test_particle_max_abs_error_c={test_eval['particle_max_abs_error_c']:.4f}"
                    f" test_particle_worst_profile={metric_int(test_eval, 'particle_worst_profile_id')}"
                    f" test_particle_worst_start={metric_int(test_eval, 'particle_worst_start')}"
                    f" test_particle_worst_k={metric_int(test_eval, 'particle_worst_index')}"
                    f" test_particle_worst_step={metric_int(test_eval, 'particle_worst_step')}"
                    f" test_particle_worst_pred_c={test_eval['particle_worst_pred_c']:.4f}"
                    f" test_particle_worst_target_c={test_eval['particle_worst_target_c']:.4f}"
                    f" test_particle_pred_range_c=[{test_eval['particle_min_pred_c']:.4f},"
                    f"{test_eval['particle_max_pred_c']:.4f}]"
                )
        metric_value = checkpoint_metric_value(
            resolved_checkpoint_metric,
            train_eval,
            test_eval if test_windows is not None else None,
        )
        checkpoint_is_best = metric_value < best_metric - config.early_stopping_min_delta
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
        diagnostics_row: dict[str, object] = {
            "epoch": epoch,
            "model_kind": config.model_kind,
            "train_loss": train_loss_mean,
            "nonfinite_train_batches": nonfinite_train_batches,
            "nonfinite_component_batches": nonfinite_component_batches,
            "rho_max": rho_max,
            "checkpoint_metric": resolved_checkpoint_metric,
            "checkpoint_metric_value": metric_value,
            "checkpoint_is_best": checkpoint_is_best,
            "checkpoint_wait": epochs_without_improvement,
            "best_epoch_so_far": best_epoch,
            "best_metric_so_far": best_metric,
            "skipped_update_batches": skipped_update_batches,
            "first_skipped_update_batch": first_skipped_update_batch,
            "learning_rate": config.learning_rate,
            "gradient_clip_norm": config.gradient_clip_norm,
            "skip_nonfinite_updates": config.skip_nonfinite_updates,
            "validate_candidate_updates": config.validate_candidate_updates,
            "log_update_diagnostics": config.log_update_diagnostics,
            "prob_particles": config.prob_particles if is_probabilistic else "",
            "prob_eval_particles": config.prob_eval_particles if is_probabilistic else "",
            "prob_process_noise": config.prob_process_noise if is_probabilistic else "",
            "prob_hp_emission_mode": config.prob_hp_emission_mode if is_probabilistic else "",
            "prob_hp_training_mode": config.prob_hp_training_mode if is_probabilistic else "",
            "prob_hp_activation_model": (
                config.prob_hp_activation_model if is_probabilistic else ""
            ),
            "prob_hp_history_hours": (
                config.prob_hp_history_hours if is_probabilistic else ""
            ),
            "prob_hp_concrete_temperature": (
                config.prob_hp_concrete_temperature if is_probabilistic else ""
            ),
            "train_eval_windows": train_eval_windows.targets.shape[0],
            "train_eval_total_windows": epoch_train_windows.targets.shape[0],
            "train_window_offset": train_window_offset,
            "full_train_eval_every_epochs": config.full_train_eval_every_epochs,
            "train_total_nrmse": train_eval["total_nrmse"],
            "train_rmse_c": train_eval["rmse_c"],
            "train_qroom_rmse_w_m2": train_eval["qroom_rmse_w_m2"],
            "train_pel_rmse_w_m2": train_eval["pel_rmse_w_m2"],
            "full_train_rmse_c": (
                full_train_eval["rmse_c"] if full_train_eval is not None else ""
            ),
            "full_train_total_nrmse": (
                full_train_eval["total_nrmse"] if full_train_eval is not None else ""
            ),
            "full_train_qroom_rmse_w_m2": (
                full_train_eval["qroom_rmse_w_m2"] if full_train_eval is not None else ""
            ),
            "full_train_pel_rmse_w_m2": (
                full_train_eval["pel_rmse_w_m2"] if full_train_eval is not None else ""
            ),
        }
        if test_eval is not None:
            diagnostics_row.update(
                {
                    "test_total_nrmse": test_eval["total_nrmse"],
                    "test_rmse_c": test_eval["rmse_c"],
                    "test_qroom_rmse_w_m2": test_eval["qroom_rmse_w_m2"],
                    "test_pel_rmse_w_m2": test_eval["pel_rmse_w_m2"],
                }
            )
        else:
            diagnostics_row.update(
                {
                    "test_total_nrmse": "",
                    "test_rmse_c": "",
                    "test_qroom_rmse_w_m2": "",
                    "test_pel_rmse_w_m2": "",
                }
            )
        diagnostics_row.update(update_stat_maxima)
        diagnostics_row.update(
            {
                f"skipped_update_reason_{name}": count
                for name, count in skipped_update_reason_counts.items()
            }
        )
        if component_means is not None:
            diagnostics_row.update(
                {
                    f"train_loss_{name}": float(value)
                    for name, value in zip(PROB_CLOSED_LOOP_LOSS_COMPONENT_NAMES, component_means)
                }
            )
        else:
            diagnostics_row.update(
                {
                    f"train_loss_{name}": ""
                    for name in PROB_CLOSED_LOOP_LOSS_COMPONENT_NAMES
                }
            )
        for prefix, metrics in (("train", train_eval), ("test", test_eval or {})):
            for key_name in (
                "median_rmse_c",
                "mean_median_rmse_c",
                "max_abs_error_c",
                "max_abs_error_profile_id",
                "max_abs_error_start",
                "max_abs_error_step",
                "max_abs_error_pred_c",
                "max_abs_error_target_c",
                "min_pred_c",
                "max_pred_c",
                "particle_max_abs_error_c",
                "particle_worst_profile_id",
                "particle_worst_start",
                "particle_worst_index",
                "particle_worst_step",
                "particle_worst_pred_c",
                "particle_worst_target_c",
                "particle_min_pred_c",
                "particle_max_pred_c",
            ):
                diagnostics_row[f"{prefix}_{key_name}"] = metrics.get(key_name, "")
        if config.save_model_every_epochs > 0 and epoch % config.save_model_every_epochs == 0:
            artifact_path = save_training_artifact(
                model_artifact_dir / f"epoch_{epoch:03d}",
                model=model,
                scalers=scalers,
                train_config=artifact_config,
                splits=splits,
                metadata_dim=metadata_dim,
                input_dim=input_dim,
                checkpoint_epoch=epoch,
                checkpoint_metric=resolved_checkpoint_metric,
                checkpoint_metric_value=metric_value,
                train_metrics=train_eval,
                test_metrics=test_eval,
            )
            print(f"saved_epoch_model={artifact_path}")

        epoch_elapsed_min = (time.perf_counter() - epoch_started_at) / 60.0
        total_elapsed_min = (time.perf_counter() - training_started_at) / 60.0
        message += (
            f" epoch_elapsed_min={epoch_elapsed_min:.2f}"
            f" total_elapsed_min={total_elapsed_min:.2f}"
        )
        diagnostics_row["epoch_elapsed_min"] = epoch_elapsed_min
        diagnostics_row["total_elapsed_min"] = total_elapsed_min
        append_epoch_diagnostics(diagnostics_path, diagnostics_row)
        print(message)

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
            f"total_nrmse={best_train_eval['total_nrmse']:.4f} "
            f"rmse_c={best_train_eval['rmse_c']:.4f} "
            f"qroom_rmse_w_m2={best_train_eval['qroom_rmse_w_m2']:.4f} "
            f"pel_rmse_w_m2={best_train_eval['pel_rmse_w_m2']:.4f}"
        )
    if best_test_eval is not None:
        print(
            "selected_test_metrics "
            f"total_nrmse={best_test_eval['total_nrmse']:.4f} "
            f"rmse_c={best_test_eval['rmse_c']:.4f} "
            f"qroom_rmse_w_m2={best_test_eval['qroom_rmse_w_m2']:.4f} "
            f"pel_rmse_w_m2={best_test_eval['pel_rmse_w_m2']:.4f}"
        )
    if config.save_model:
        artifact_path = save_training_artifact(
            model_artifact_dir / "selected",
            model=model,
            scalers=scalers,
            train_config=artifact_config,
            splits=splits,
            metadata_dim=metadata_dim,
            input_dim=input_dim,
            checkpoint_epoch=best_epoch,
            checkpoint_metric=resolved_checkpoint_metric,
            checkpoint_metric_value=best_metric,
            train_metrics=best_train_eval,
            test_metrics=best_test_eval,
        )
        print(f"saved_selected_model={artifact_path}")

    if test_windows is not None:
        if is_probabilistic:
            assert isinstance(
                model,
                (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator),
            )
            full_profile_eval = evaluate_probabilistic_closed_loop_full_profiles(
                model,
                test_profiles,
                scalers,
                num_particles=config.prob_eval_particles,
            )
        else:
            full_profile_eval = evaluate_closed_loop_full_profiles(model, test_profiles, scalers)
        print(
            "full_profile_test "
            f"profiles={int(full_profile_eval['profile_count'])} "
            f"rmse_c={full_profile_eval['rmse_c']:.4f} "
            f"mae_c={full_profile_eval['mae_c']:.4f} "
            f"qroom_rmse_w_m2={full_profile_eval['qroom_rmse_w_m2']:.4f} "
            f"pel_rmse_w_m2={full_profile_eval['pel_rmse_w_m2']:.4f}"
        )
        if is_probabilistic:
            assert isinstance(
                model,
                (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator),
            )
            visualization_paths = save_probabilistic_closed_loop_prediction_visualizations(
                model,
                test_windows,
                test_profiles,
                scalers,
                config.output_dir,
                config.num_window_plots,
                config.num_full_profile_plots,
                num_particles=config.prob_plot_particles,
                hp_scenario_mode=config.prob_hp_scenario_mode,
                filename_suffix=(
                    "_closed_loop_hp_contracting_prob"
                    if is_contracting_probabilistic
                    else "_closed_loop_hp_prob"
                ),
                title_label=(
                    "probabilistic contractive closed-loop HP"
                    if is_contracting_probabilistic
                    else "probabilistic closed-loop HP"
                ),
            )
        else:
            suffix = "_closed_loop_hp_contracting" if is_contracting else "_closed_loop_hp"
            title_label = "contracting closed-loop HP" if is_contracting else "closed-loop HP"
            visualization_paths = save_closed_loop_prediction_visualizations(
                model,
                test_windows,
                test_profiles,
                scalers,
                config.output_dir,
                config.num_window_plots,
                config.num_full_profile_plots,
                filename_suffix=suffix,
                title_label=title_label,
            )
        for name, path in visualization_paths.items():
            print(f"saved_{name}_plot={path}")

    return model


def run_training(config: TrainConfig) -> EmulatorModel:
    if config.model_kind in (
        "closed_loop_hp",
        "closed_loop_hp_contracting",
        "closed_loop_hp_probabilistic",
        "closed_loop_hp_contracting_probabilistic",
    ):
        return run_closed_loop_training(config)
    if config.model_kind not in ("deterministic", "probabilistic"):
        raise ValueError(
            "model_kind must be 'deterministic', 'probabilistic', "
            "'closed_loop_hp', 'closed_loop_hp_contracting', "
            "'closed_loop_hp_probabilistic', or 'closed_loop_hp_contracting_probabilistic'"
        )
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
    if config.prob_variogram_output_weights:
        if len(config.prob_variogram_output_weights) != 1:
            raise ValueError(
                "--prob-variogram-output-weights must contain exactly one value "
                "for the Q-to-T probabilistic model"
            )
        if any(weight < 0.0 for weight in config.prob_variogram_output_weights):
            raise ValueError("prob_variogram_output_weights must be non-negative")
        if sum(config.prob_variogram_output_weights) <= 0.0:
            raise ValueError("prob_variogram_output_weights must contain at least one positive value")
    if config.prob_variogram_constant_setpoint_only:
        raise ValueError("--prob-variogram-constant-setpoint-only is only available for closed-loop HP models")
    if config.prob_variogram_setpoint_threshold_c < 0.0:
        raise ValueError("prob_variogram_setpoint_threshold_c must be non-negative")
    if config.prob_physics_weight < 0.0:
        raise ValueError("prob_physics_weight must be non-negative")
    if config.prob_horizon_weight_power < 0.0:
        raise ValueError("prob_horizon_weight_power must be non-negative")
    if any(lag < 1 for lag in config.prob_variogram_lags):
        raise ValueError("prob_variogram_lags must be positive")
    if config.checkpoint_metric not in (
        "auto",
        "train_rmse_c",
        "test_rmse_c",
        "train_total_nrmse",
        "test_total_nrmse",
    ):
        raise ValueError(
            "checkpoint_metric must be 'auto', 'train_rmse_c', 'test_rmse_c', "
            "'train_total_nrmse', or 'test_total_nrmse'"
        )
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if config.loss_normalization not in ("none", "window_std"):
        raise ValueError("loss_normalization must be 'none' or 'window_std'")
    if config.loss_std_floor_c <= 0.0:
        raise ValueError("loss_std_floor_c must be positive")
    if config.train_eval_max_windows < 0:
        raise ValueError("train_eval_max_windows must be non-negative; use 0 for all windows")
    if config.full_train_eval_every_epochs < 0:
        raise ValueError("full_train_eval_every_epochs must be non-negative; use 0 to disable")
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
        fixed_test_profile_ids=config.fixed_test_profile_ids,
        excluded_profile_ids=config.excluded_profile_ids,
        heat_input_normalization=config.heat_input_normalization,  # type: ignore[arg-type]
        hp_power_area_normalization=config.hp_power_area_normalization,
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
        splits.hp_power_area_normalization,
    )
    test_profiles = (
        to_profiles(
            splits.test,
            splits.heating_mode,
            splits.heat_input_normalization,
            splits.input_feature_mode,
            splits.heating_regime_window_steps,
            config.heat_on_threshold,
            splits.hp_power_area_normalization,
        )
        if splits.test_ids
        else []
    )
    raw_train_windows = make_windows(train_profiles, window_config)
    raw_test_windows = None
    if splits.test_ids:
        raw_test_windows = make_windows(test_profiles, window_config)
    train_window_count = raw_train_windows.targets.shape[0]
    metadata_dim = raw_train_windows.metadata.shape[-1]
    input_dim = raw_train_windows.inputs.shape[-1]
    resolved_checkpoint_metric = resolve_checkpoint_metric(
        config.checkpoint_metric,
        has_test_windows=raw_test_windows is not None,
    )

    scalers = fit_window_scalers(raw_train_windows)
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
    raw_train_eval_windows = fixed_evaluation_subset(
        raw_train_windows,
        max_windows=config.train_eval_max_windows,
        seed=config.seed + 10_003,
    )
    assert isinstance(raw_train_eval_windows, WindowedArrays)
    train_eval_windows = transform_windows(raw_train_eval_windows, scalers)
    test_windows = (
        transform_windows(raw_test_windows, scalers)
        if raw_test_windows is not None
        else None
    )
    scaled_train_profiles = transform_profiles(train_profiles, scalers)
    train_windows = (
        None
        if config.rotate_window_starts
        else make_windows(scaled_train_profiles, window_config)
    )
    del raw_train_windows, raw_train_eval_windows, raw_test_windows, train_profiles

    key = jax.random.PRNGKey(config.seed)
    if config.model_kind == "deterministic":
        model: EmulatorModel = MetadataStateSpaceEmulator(
            metadata_dim=metadata_dim,
            input_dim=input_dim,
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
        encoded_input_dim = config.input_encoder_dim or input_dim
        model = ProbabilisticStableStateSpaceEmulator(
            metadata_dim=metadata_dim,
            input_dim=input_dim,
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
    optimizer = build_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(config.seed)
    model_checkpoint_dir = config.model_checkpoint_dir or config.output_dir / "model_checkpoints"
    model_artifact_dir = model_checkpoint_dir / config.model_kind

    print(f"heating_mode={splits.heating_mode}")
    print(f"heat_input_normalization={splits.heat_input_normalization}")
    if splits.heating_mode == "heating_electric":
        print(
            "hp_power_area_normalization="
            f"{splits.hp_power_area_normalization}"
        )
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
            f"variogram_power={config.prob_variogram_power} "
            f"variogram_output_weights="
            f"{list(config.prob_variogram_output_weights) if config.prob_variogram_output_weights else 'equal'} "
            f"softopt_weight={config.prob_softopt_weight} "
            f"softopt_temperature={config.prob_softopt_temperature} "
            f"physics_weight={config.prob_physics_weight}"
        )
        print(
            "input_encoder=enabled "
            f"encoded_dim={config.input_encoder_dim or input_dim} "
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
    print(f"train_windows={train_window_count}")
    if test_windows is not None:
        print(f"test_windows={test_windows.targets.shape[0]}")
    print(
        "optimizer="
        f"adam learning_rate={config.learning_rate} "
        f"gradient_clip_norm={config.gradient_clip_norm} "
        f"skip_nonfinite_updates={config.skip_nonfinite_updates} "
        f"max_consecutive_nonfinite_updates={config.max_consecutive_nonfinite_updates} "
        f"validate_candidate_updates={config.validate_candidate_updates} "
        f"log_update_diagnostics={config.log_update_diagnostics}"
    )
    print(f"checkpoint_metric={resolved_checkpoint_metric}")
    print(
        "window_start_rotation="
        f"{'enabled' if config.rotate_window_starts else 'disabled'} "
        f"stride={config.stride} epochs={config.epochs} validation_offset=0"
    )
    print(
        "evaluation_schedule="
        f"train_fixed_subset:{train_eval_windows.targets.shape[0]}/{train_window_count} "
        f"full_train_every_epochs:{config.full_train_eval_every_epochs} "
        f"test_full_every_epoch:{test_windows is not None}"
    )
    if config.early_stopping_patience is not None:
        print(
            "early_stopping=enabled "
            f"patience={config.early_stopping_patience} "
            f"min_delta={config.early_stopping_min_delta}"
        )
    if config.save_model or config.save_model_every_epochs > 0:
        print(f"model_checkpoint_dir={model_checkpoint_dir}")
        print(f"model_artifact_dir={model_artifact_dir}")
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
    training_started_at = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        if config.rotate_window_starts and epoch > 1:
            del epoch_train_windows
        train_window_offset = (
            rotating_window_start_offset(epoch, config.epochs, config.stride)
            if config.rotate_window_starts
            else 0
        )
        epoch_train_windows = (
            make_windows(
                scaled_train_profiles,
                window_config,
                start_offset=train_window_offset,
            )
            if config.rotate_window_starts
            else train_windows
        )
        assert epoch_train_windows is not None
        losses = []
        for batch_idx, (metadata, inputs, initial_temperature, targets) in enumerate(
            minibatches(epoch_train_windows, batch_size=config.batch_size, rng=rng, shuffle=True),
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
                    config.prob_variogram_output_weights,
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
            train_eval_windows,
            scalers,
            batch_size=config.batch_size,
            target_mode=config.target_mode,
            model_kind=config.model_kind,
            num_particles=config.prob_eval_particles,
        )
        full_train_eval: dict[str, float] | None = None
        if (
            config.full_train_eval_every_epochs > 0
            and epoch % config.full_train_eval_every_epochs == 0
            and train_eval_windows.targets.shape[0] < epoch_train_windows.targets.shape[0]
        ):
            full_train_eval = evaluate(
                model,
                epoch_train_windows,
                scalers,
                batch_size=config.batch_size,
                target_mode=config.target_mode,
                model_kind=config.model_kind,
                num_particles=config.prob_eval_particles,
            )
        train_loss_mean, nonfinite_train_batches = finite_mean(losses)
        message = (
            f"epoch={epoch:03d} train_loss={train_loss_mean:.6f} "
            f"train_rmse_c={train_eval['rmse_c']:.4f} "
            f"train_nmae={train_eval['nmae']:.4f} "
            f"train_eval_windows={train_eval_windows.targets.shape[0]} "
            f"train_window_offset={train_window_offset} "
            f"rho_max={sample_spectral_radius(model, epoch_train_windows):.5f}"
        )
        if full_train_eval is not None:
            message += (
                f" full_train_rmse_c={full_train_eval['rmse_c']:.4f}"
                f" full_train_nmae={full_train_eval['nmae']:.4f}"
                f" full_train_eval_windows={epoch_train_windows.targets.shape[0]}"
            )
        if nonfinite_train_batches > 0:
            message += f" nonfinite_train_batches={nonfinite_train_batches}"
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
        if config.save_model_every_epochs > 0 and epoch % config.save_model_every_epochs == 0:
            artifact_path = save_training_artifact(
                model_artifact_dir / f"epoch_{epoch:03d}",
                model=model,
                scalers=scalers,
                train_config=config,
                splits=splits,
                metadata_dim=metadata_dim,
                input_dim=input_dim,
                checkpoint_epoch=epoch,
                checkpoint_metric=resolved_checkpoint_metric,
                checkpoint_metric_value=metric_value,
                train_metrics=train_eval,
                test_metrics=test_eval,
            )
            print(f"saved_epoch_model={artifact_path}")

        epoch_elapsed_min = (time.perf_counter() - epoch_started_at) / 60.0
        total_elapsed_min = (time.perf_counter() - training_started_at) / 60.0
        message += (
            f" epoch_elapsed_min={epoch_elapsed_min:.2f}"
            f" total_elapsed_min={total_elapsed_min:.2f}"
        )
        print(message)

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
            model_artifact_dir / "selected",
            model=model,
            scalers=scalers,
            train_config=config,
            splits=splits,
            metadata_dim=metadata_dim,
            input_dim=input_dim,
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
        choices=(
            "deterministic",
            "probabilistic",
            "closed_loop_hp",
            "closed_loop_hp_contracting",
            "closed_loop_hp_probabilistic",
            "closed_loop_hp_contracting_probabilistic",
        ),
        default="deterministic",
        help=(
            "Select the original deterministic SS model, the probabilistic stable SS model, "
            "the closed-loop HP plus thermal model, its contractive data-driven variant, "
            "or either probabilistic closed-loop variant."
        ),
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
        "--rotate-window-starts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rotate the training window grid across one stride over the epochs while keeping "
            "validation starts fixed. Enabled by default; use --no-rotate-window-starts to "
            "reproduce the historical fixed grid."
        ),
    )
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
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Global gradient clipping norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--skip-grad-norm-above",
        type=float,
        default=1e4,
        help=(
            "Skip the optimizer update when the pre-clip gradient L2 norm exceeds this value. "
            "Clipping a 1e9 gradient to 1.0 still takes a full-size step in a poison direction. "
            "Use 0 to disable. Default 1e4 is above healthy closed-loop grads (~1e2) and below "
            "the T=64 cliff (~1e4-1e5) that wrecks the next epoch."
        ),
    )
    parser.add_argument(
        "--allow-nonfinite-updates",
        action="store_true",
        help="Disable the default guard that skips optimizer updates with non-finite gradients.",
    )
    parser.add_argument(
        "--max-consecutive-nonfinite-updates",
        type=int,
        default=8,
        help="Allowed consecutive skipped non-finite updates before optax raises.",
    )
    parser.add_argument(
        "--validate-candidate-updates",
        action="store_true",
        help=(
            "Run an extra candidate forward pass before accepting each closed-loop update. "
            "This is slower, but diagnoses finite-parameter updates that would make the next "
            "rollout nonfinite."
        ),
    )
    parser.add_argument(
        "--log-update-diagnostics",
        action="store_true",
        help=(
            "Print per-epoch maxima of gradient norms, update norms, parameter norms, "
            "and max absolute update/parameter values."
        ),
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument(
        "--train-eval-max-windows",
        type=int,
        default=512,
        help=(
            "Maximum number of windows in the fixed per-epoch training-evaluation subset. "
            "The subset is sampled once and reused across epochs. Use 0 to evaluate all train windows."
        ),
    )
    parser.add_argument(
        "--full-train-eval-every-epochs",
        type=int,
        default=5,
        help=(
            "Run an additional full-training-set evaluation every N epochs. "
            "The complete test set is still evaluated every epoch. Use 0 to disable."
        ),
    )
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
        choices=(
            "auto",
            "train_rmse_c",
            "test_rmse_c",
            "train_total_nrmse",
            "test_total_nrmse",
        ),
        default="auto",
        help=(
            "Metric used to select the best in-memory checkpoint. Auto uses equal-channel "
            "normalized RMSE for closed-loop HP models and temperature RMSE for Q-to-T."
        ),
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
        "--closed-loop-trajectory-output-weights",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("TIN", "QROOM", "PEL"),
        help=(
            "Per-output weights for deterministic closed-loop MSE and probabilistic "
            "energy/soft-Best-of-K trajectory scores. Weights are normalized to mean one; "
            "use 1 1 0 to remove pointwise Pel trajectory matching while retaining HP "
            "mode, active-power, IRES, and flexibility losses."
        ),
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
        "--prob-variogram-output-weights",
        nargs="+",
        type=float,
        default=[],
        help=(
            "Optional per-output variogram weights. For closed-loop models use "
            "three values in Tin Qroom Pel_SH order, e.g. 1 0 0."
        ),
    )
    parser.add_argument(
        "--prob-variogram-constant-setpoint-only",
        action="store_true",
        help=(
            "For closed-loop probabilistic models, compute variogram terms only "
            "on lagged pairs whose interval does not cross a setpoint jump."
        ),
    )
    parser.add_argument(
        "--prob-variogram-setpoint-threshold-c",
        type=float,
        default=0.05,
        help="Setpoint-change threshold in degC used by --prob-variogram-constant-setpoint-only.",
    )
    parser.add_argument(
        "--prob-ires-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the intervention response energy score on windowed "
            "pre/post Pel responses around setpoint jumps."
        ),
    )
    parser.add_argument(
        "--prob-ires-horizons-hours",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
        help="Activation durations in hours used by the intervention response energy score.",
    )
    parser.add_argument(
        "--prob-ires-setpoint-threshold-c",
        type=float,
        default=0.05,
        help="Minimum absolute thermostat setpoint jump, in degC, counted as an IRES intervention.",
    )
    parser.add_argument(
        "--prob-flex-kpi-crps-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for CRPS applied directly to fitted upward/downward flexibility "
            "gains H*beta at setpoint interventions. Disabled by default."
        ),
    )
    parser.add_argument(
        "--prob-flex-kpi-crps-horizons-hours",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
        help="Activation durations in hours used by the flexibility-coefficient CRPS.",
    )
    parser.add_argument(
        "--prob-flex-kpi-crps-setpoint-threshold-c",
        type=float,
        default=0.05,
        help="Minimum absolute thermostat jump counted by the flexibility-coefficient CRPS.",
    )
    parser.add_argument(
        "--prob-flex-kpi-crps-min-events",
        type=int,
        default=20,
        help="Minimum intervention count required to score one training-window KPI regression.",
    )
    parser.add_argument(
        "--prob-flex-kpi-crps-ridge",
        type=float,
        default=1e-3,
        help="Ridge coefficient stabilizing each training-window KPI regression.",
    )
    parser.add_argument(
        "--prob-flex-kpi-crps-controls",
        choices=("none", "weather", "full"),
        default="full",
        help=(
            "Regression controls used by the flexibility-coefficient CRPS. Full uses fixed "
            "observed pre-event power/temperature plus weather and calendar controls."
        ),
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
    parser.add_argument(
        "--closed-loop-stability-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for sampled closed-loop contraction penalty in "
            "probabilistic closed-loop models."
        ),
    )
    parser.add_argument(
        "--closed-loop-stability-gamma",
        type=float,
        default=0.995,
        help="Target upper bound for sampled ||dF/ds||_2 over active recurrent states.",
    )
    parser.add_argument(
        "--closed-loop-stability-samples",
        type=int,
        default=4,
        help="Number of random batch/particle/time Jacobians sampled when the penalty is evaluated.",
    )
    parser.add_argument(
        "--closed-loop-stability-aggregation",
        choices=("mean", "max"),
        default="max",
        help="Aggregate sampled Jacobian violations with mean or max.",
    )
    parser.add_argument(
        "--closed-loop-stability-every-steps",
        type=int,
        default=8,
        help="Evaluate the sampled Jacobian penalty once per this many optimizer steps.",
    )
    parser.add_argument(
        "--closed-loop-stability-method",
        choices=("exact_svd", "power_iteration"),
        default="power_iteration",
        help="Use an exact full Jacobian SVD or a cheaper differentiable power iteration.",
    )
    parser.add_argument(
        "--closed-loop-stability-power-iterations",
        type=int,
        default=4,
        help="Power iterations used by --closed-loop-stability-method power_iteration.",
    )
    parser.add_argument(
        "--init-from-deterministic-artifact",
        type=Path,
        default=None,
        help=(
            "Warm-start --model-kind closed_loop_hp_probabilistic from a saved "
            "closed_loop_hp artifact directory."
        ),
    )
    parser.add_argument(
        "--init-xi-weight-scale",
        type=float,
        default=0.05,
        help=(
            "Relative standard deviation for nonzero xi-column perturbations when "
            "warm-starting closed_loop_hp_probabilistic from a deterministic artifact. "
            "Use 0 to recover the old xi-blind warm start."
        ),
    )
    parser.add_argument(
        "--init-hp-active-log-sigma",
        type=float,
        default=0.25,
        help=(
            "Initial log-space sigma for active HP electric power when warm-starting "
            "closed_loop_hp_probabilistic from a deterministic artifact. Must be in "
            "(0.05, 0.50) for bounded HP emissions or (0.05, 0.75) for legacy_lognormal_mean."
        ),
    )
    parser.add_argument(
        "--contracting-gamma",
        type=float,
        default=0.99,
        help=(
            "Structural norm bound applied independently to every generated thermal matrix M_t "
            "in contracting closed-loop models. This does not by itself bound the complete HP "
            "feedback Jacobian because M_t and the forcing depend on the recurrent trajectory."
        ),
    )
    parser.add_argument(
        "--contracting-state-bound",
        type=float,
        default=5.0,
        help="Bound on each normalized recurrent state coordinate in closed_loop_hp_contracting.",
    )
    parser.add_argument(
        "--contracting-temperature-scale",
        type=float,
        default=8.0,
        help=(
            "Bound on the normalized temperature output magnitude for "
            "--model-kind closed_loop_hp_contracting."
        ),
    )
    parser.add_argument(
        "--contracting-temperature-delta-max-c",
        type=float,
        default=0.0,
        help=(
            "If positive, rate-limit the contracting closed-loop temperature readout to "
            "T[t+1] = T[t] +/- this many degC per model step. The default 0 keeps the "
            "legacy absolute bounded readout."
        ),
    )
    parser.add_argument(
        "--contracting-temperature-update",
        choices=(
            "auto",
            "absolute",
            "delta",
            "leaky_equilibrium",
            "bounded_equilibrium",
            "alpha_bounded_equilibrium",
        ),
        default="auto",
        help=(
            "Temperature readout used by contracting closed-loop models. 'auto' preserves "
            "the legacy behavior: absolute when --contracting-temperature-delta-max-c is 0, "
            "additive delta when it is positive. 'leaky_equilibrium' uses "
            "T[t+1]=(1-alpha)T[t]+alpha*T_eq[t] with rollout-constant learned alpha and "
            "alpha_max derived from --contracting-temperature-delta-max-c. "
            "'bounded_equilibrium' removes alpha and uses "
            "T[t+1]=T[t]+delta_max*tanh((T_eq[t]-T[t])/delta_max). "
            "'alpha_bounded_equilibrium' additionally learns rollout-constant alpha in "
            "(0,1) and places it inside tanh."
        ),
    )
    parser.add_argument(
        "--contracting-transition-conditioning",
        choices=(
            "state_feedback",
            "exogenous",
            "exogenous_additive_feedback",
        ),
        default="state_feedback",
        help=(
            "Inputs used by the neural generator of M_t and b_t. 'state_feedback' "
            "preserves the existing Tin/Tout-Tin (and unconstrained Qroom) feedback. "
            "'exogenous' restricts the generator to metadata, persistent xi, and "
            "exogenous time-varying inputs while preserving the same parameter shape. "
            "'exogenous_additive_feedback' generates bounded M_t, B_t, and b_t from "
            "exogenous inputs and injects encoded Tin/Tout-Tin feedback only through "
            "the additive B_t z_t path."
        ),
    )
    parser.add_argument(
        "--contracting-additive-feedback-gain-bound",
        type=float,
        default=1.0,
        help=(
            "Frobenius-norm bound for B_t in the "
            "exogenous_additive_feedback transition mode."
        ),
    )
    parser.add_argument(
        "--contracting-q-to-t-mode",
        choices=("unconstrained", "positive_leaky"),
        default="unconstrained",
        help=(
            "Qroom-to-temperature structure for contracting closed-loop models. "
            "'unconstrained' keeps Qroom in the neural transition forcing. "
            "'positive_leaky' removes Qroom from the generated matrix/bias inputs and "
            "injects it through separate positive leaky thermal modes; it requires "
            "an equilibrium-based --contracting-temperature-update."
        ),
    )
    parser.add_argument(
        "--contracting-q-to-t-time-constants-hours",
        type=float,
        nargs="+",
        default=(0.25, 1.0, 4.0, 16.0, 24.0, 48.0),
        help=(
            "Fixed positive time constants in hours for the structural positive "
            "Qroom-to-temperature modes."
        ),
    )
    parser.add_argument(
        "--contracting-q-to-t-gain-min-c-per-w-m2",
        type=float,
        default=0.01,
        help="Lower bound on the learned total steady Qroom-to-temperature gain.",
    )
    parser.add_argument(
        "--contracting-q-to-t-gain-max-c-per-w-m2",
        type=float,
        default=2.0,
        help="Upper bound on the learned total steady Qroom-to-temperature gain.",
    )
    parser.add_argument(
        "--prob-hp-scenario-mode",
        choices=("expected", "bernoulli"),
        default="bernoulli",
        help=(
            "HP electric scenario mode for --model-kind closed_loop_hp_probabilistic plots. "
            "'bernoulli' samples on/off and active power; 'expected' propagates expected power."
        ),
    )
    parser.add_argument(
        "--prob-hp-emission-mode",
        choices=("bounded", "legacy_lognormal_mean"),
        default="bounded",
        help=(
            "Active HP electric-power distribution for probabilistic closed-loop models. "
            "'bounded' caps log power and sigma for numerical safety. "
            "'legacy_lognormal_mean' restores the previous lognormal-mean path, which can "
            "produce larger tails and reproduce older NaN-prone runs."
        ),
    )
    parser.add_argument(
        "--prob-hp-training-mode",
        choices=("expected", "straight_through"),
        default="expected",
        help=(
            "HP path used by probabilistic closed-loop training. 'expected' preserves the "
            "deterministic expected-power path; 'straight_through' samples hard HP on/off "
            "events and active power, using a Binary Concrete surrogate gradient for mode logits."
        ),
    )
    parser.add_argument(
        "--prob-hp-concrete-temperature",
        type=float,
        default=0.5,
        help=(
            "Positive Binary Concrete temperature used when "
            "--prob-hp-training-mode straight_through. Lower values approach a hard threshold "
            "but produce sharper surrogate gradients."
        ),
    )
    parser.add_argument(
        "--prob-hp-activation-model",
        choices=("independent", "persistent_markov", "power_history"),
        default="independent",
        help=(
            "HP on/off dynamics for the contracting probabilistic closed-loop model. "
            "'independent' preserves per-step Bernoulli activation. 'persistent_markov' "
            "uses particle-persistent HP heterogeneity, a bounded recurrent controller state, "
            "and separate learned start/stop probabilities. 'power_history' conditions the "
            "emission on causal bounded EWMAs of generated HP power and duty cycle."
        ),
    )
    parser.add_argument(
        "--prob-hp-persistent-latent-dim",
        type=int,
        default=2,
        help="Dimension of the trajectory-persistent HP-controller latent in persistent_markov mode.",
    )
    parser.add_argument(
        "--prob-hp-controller-leak",
        type=float,
        default=0.25,
        help=(
            "Controller-state update fraction in persistent_markov mode. It must be in (0, 1]; "
            "smaller values retain controller memory for longer."
        ),
    )
    parser.add_argument(
        "--prob-hp-controller-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Fixed per-step Gaussian disturbance scale inside the bounded HP controller-state "
            "update. Zero keeps controller uncertainty entirely persistent."
        ),
    )
    parser.add_argument(
        "--prob-hp-history-hours",
        type=float,
        default=3.0,
        help=(
            "Positive EWMA time constant in hours for generated HP power and duty-cycle "
            "history when --prob-hp-activation-model power_history."
        ),
    )
    parser.add_argument(
        "--hp-controller-state-dim",
        type=int,
        default=2,
        help="Latent controller/buffer state dimension for --model-kind closed_loop_hp.",
    )
    parser.add_argument(
        "--hp-thermostat-demand-mode",
        choices=("unconstrained", "monotone"),
        default="unconstrained",
        help=(
            "Thermostat structure for contracting closed-loop HP models. 'monotone' masks "
            "Tset from the generic encoder and makes only requested room heat structurally "
            "nondecreasing in Tset-Tin. HP activation and active power remain unrestricted "
            "functions of the thermal gap, controller history, and buffer state."
        ),
    )
    parser.add_argument(
        "--hp-thermostat-slope-min",
        type=float,
        default=0.1,
        help="Positive lower bound for monotone thermostat sigmoid slopes, in 1/degC.",
    )
    parser.add_argument(
        "--hp-thermostat-slope-max",
        type=float,
        default=6.0,
        help="Upper bound for monotone thermostat sigmoid slopes, in 1/degC.",
    )
    parser.add_argument(
        "--hp-thermostat-threshold-min-c",
        type=float,
        default=-1.0,
        help="Lower bound for learned monotone thermostat switching thresholds, in degC.",
    )
    parser.add_argument(
        "--hp-thermostat-threshold-max-c",
        type=float,
        default=1.0,
        help="Upper bound for learned monotone thermostat switching thresholds, in degC.",
    )
    parser.add_argument(
        "--hp-controller-calendar-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow hour/day cyclic features into the HP control head. Use "
            "--no-hp-controller-calendar-features to mask them from HP control while "
            "retaining them in the thermal forcing path."
        ),
    )
    parser.add_argument(
        "--hp-dt-hours",
        type=float,
        default=0.25,
        help="Closed-loop HP energy balance timestep in hours.",
    )
    parser.add_argument(
        "--hp-power-area-normalization",
        choices=("building_heated_area", "zone_floor_area"),
        default="building_heated_area",
        help=(
            "Area used to convert whole-building HP electric power to W/m2. "
            "The corrected default uses floor_area*totalFloors; zone_floor_area "
            "reproduces legacy artifacts and should not be used for new training."
        ),
    )
    parser.add_argument(
        "--hp-mode-loss-weight",
        type=float,
        default=0.1,
        help="Binary cross-entropy weight for HP on/off in --model-kind closed_loop_hp.",
    )
    parser.add_argument(
        "--hp-cop-floor",
        type=float,
        default=1.0,
        help="Lower bound added to the learnable linear COP in --model-kind closed_loop_hp.",
    )
    parser.add_argument(
        "--hp-cop-cap",
        type=float,
        default=8.0,
        help="Upper bound for closed-loop HP COP; use 0 to disable.",
    )
    parser.add_argument(
        "--hp-pel-cap-w-m2",
        type=float,
        default=0.0,
        help="Upper bound for space-heating HP electric power in W/m2; use 0 for train-data auto cap.",
    )
    parser.add_argument(
        "--hp-qroom-cap-w-m2",
        type=float,
        default=0.0,
        help="Upper bound for delivered room heat in W/m2; use 0 for train-data auto cap.",
    )
    parser.add_argument(
        "--hp-energy-cap-wh-m2",
        type=float,
        default=0.0,
        help="Upper bound for latent stored heat in Wh/m2; use 0 for automatic cap.",
    )
    parser.add_argument(
        "--hp-energy-cap-hours",
        type=float,
        default=24.0,
        help="Automatic energy cap duration: Emax=max(this * Qcap, dt * COPcap * Pelcap).",
    )
    parser.add_argument(
        "--hp-cap-factor",
        type=float,
        default=1.25,
        help="Multiplier applied to observed train maxima when resolving automatic HP/Q caps.",
    )
    parser.add_argument(
        "--hp-active-power-nll-weight",
        type=float,
        default=1.0,
        help="Conditional log-power NLL weight for --model-kind closed_loop_hp_probabilistic.",
    )
    parser.add_argument(
        "--bptt-truncate-steps",
        type=int,
        default=DEFAULT_BPTT_TRUNCATE_STEPS,
        help=(
            "Detach the recurrent carry every K steps during closed-loop HP rollouts. "
            "Forward predictions and the 960-step loss are unchanged; only BPTT is truncated. "
            f"Default {DEFAULT_BPTT_TRUNCATE_STEPS} is 4 hours at --hp-dt-hours 0.25, "
            "long enough for IRES/variogram lags and inside the empirically stable T<=32 band. "
            "Use 0 for full BPTT."
        ),
    )
    parser.add_argument(
        "--hp-inactive-leakage-weight",
        type=float,
        default=0.1,
        help="Penalty weight for nonzero expected HP power on inactive timesteps.",
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
        hp_power_area_normalization=args.hp_power_area_normalization,
        input_feature_mode=args.input_feature_mode,
        heating_regime_window_steps=args.heating_regime_window_steps,
        heat_on_threshold=args.heat_on_threshold,
        sequence_length=args.sequence_length,
        stride=args.stride,
        rotate_window_starts=args.rotate_window_starts,
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
        gradient_clip_norm=args.gradient_clip_norm,
        skip_grad_norm_above=args.skip_grad_norm_above,
        skip_nonfinite_updates=not args.allow_nonfinite_updates,
        max_consecutive_nonfinite_updates=args.max_consecutive_nonfinite_updates,
        validate_candidate_updates=args.validate_candidate_updates,
        log_update_diagnostics=args.log_update_diagnostics,
        max_train_batches=args.max_train_batches,
        train_eval_max_windows=args.train_eval_max_windows,
        full_train_eval_every_epochs=args.full_train_eval_every_epochs,
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
        closed_loop_trajectory_output_weights=tuple(
            args.closed_loop_trajectory_output_weights
        ),
        prob_variogram_weight=args.prob_variogram_weight,
        prob_variogram_lags=tuple(args.prob_variogram_lags),
        prob_variogram_power=args.prob_variogram_power,
        prob_variogram_output_weights=tuple(args.prob_variogram_output_weights),
        prob_variogram_constant_setpoint_only=args.prob_variogram_constant_setpoint_only,
        prob_variogram_setpoint_threshold_c=args.prob_variogram_setpoint_threshold_c,
        prob_ires_weight=args.prob_ires_weight,
        prob_ires_horizons_hours=tuple(args.prob_ires_horizons_hours),
        prob_ires_setpoint_threshold_c=args.prob_ires_setpoint_threshold_c,
        prob_flex_kpi_crps_weight=args.prob_flex_kpi_crps_weight,
        prob_flex_kpi_crps_horizons_hours=tuple(args.prob_flex_kpi_crps_horizons_hours),
        prob_flex_kpi_crps_setpoint_threshold_c=args.prob_flex_kpi_crps_setpoint_threshold_c,
        prob_flex_kpi_crps_min_events=args.prob_flex_kpi_crps_min_events,
        prob_flex_kpi_crps_ridge=args.prob_flex_kpi_crps_ridge,
        prob_flex_kpi_crps_controls=args.prob_flex_kpi_crps_controls,
        prob_physics_weight=args.prob_physics_weight,
        prob_horizon_weight_power=args.prob_horizon_weight_power,
        prob_hp_scenario_mode=args.prob_hp_scenario_mode,
        prob_hp_emission_mode=args.prob_hp_emission_mode,
        prob_hp_training_mode=args.prob_hp_training_mode,
        prob_hp_concrete_temperature=args.prob_hp_concrete_temperature,
        prob_hp_activation_model=args.prob_hp_activation_model,
        prob_hp_persistent_latent_dim=args.prob_hp_persistent_latent_dim,
        prob_hp_controller_leak=args.prob_hp_controller_leak,
        prob_hp_controller_noise_scale=args.prob_hp_controller_noise_scale,
        prob_hp_history_hours=args.prob_hp_history_hours,
        hp_controller_state_dim=args.hp_controller_state_dim,
        hp_controller_calendar_features=args.hp_controller_calendar_features,
        hp_thermostat_demand_mode=args.hp_thermostat_demand_mode,
        hp_thermostat_slope_min=args.hp_thermostat_slope_min,
        hp_thermostat_slope_max=args.hp_thermostat_slope_max,
        hp_thermostat_threshold_min_c=args.hp_thermostat_threshold_min_c,
        hp_thermostat_threshold_max_c=args.hp_thermostat_threshold_max_c,
        hp_dt_hours=args.hp_dt_hours,
        hp_mode_loss_weight=args.hp_mode_loss_weight,
        hp_cop_floor=args.hp_cop_floor,
        hp_cop_cap=args.hp_cop_cap,
        hp_pel_cap_w_m2=args.hp_pel_cap_w_m2,
        hp_qroom_cap_w_m2=args.hp_qroom_cap_w_m2,
        hp_energy_cap_wh_m2=args.hp_energy_cap_wh_m2,
        hp_energy_cap_hours=args.hp_energy_cap_hours,
        hp_cap_factor=args.hp_cap_factor,
        closed_loop_stability_weight=args.closed_loop_stability_weight,
        closed_loop_stability_gamma=args.closed_loop_stability_gamma,
        closed_loop_stability_samples=args.closed_loop_stability_samples,
        closed_loop_stability_aggregation=args.closed_loop_stability_aggregation,
        closed_loop_stability_every_steps=args.closed_loop_stability_every_steps,
        closed_loop_stability_method=args.closed_loop_stability_method,
        closed_loop_stability_power_iterations=args.closed_loop_stability_power_iterations,
        init_from_deterministic_artifact=args.init_from_deterministic_artifact,
        init_xi_weight_scale=args.init_xi_weight_scale,
        init_hp_active_log_sigma=args.init_hp_active_log_sigma,
        contracting_gamma=args.contracting_gamma,
        contracting_state_bound=args.contracting_state_bound,
        contracting_temperature_scale=args.contracting_temperature_scale,
        contracting_temperature_delta_max_c=args.contracting_temperature_delta_max_c,
        contracting_temperature_update=args.contracting_temperature_update,
        contracting_transition_conditioning=(
            args.contracting_transition_conditioning
        ),
        contracting_additive_feedback_gain_bound=(
            args.contracting_additive_feedback_gain_bound
        ),
        contracting_q_to_t_mode=args.contracting_q_to_t_mode,
        contracting_q_to_t_time_constants_hours=tuple(
            args.contracting_q_to_t_time_constants_hours
        ),
        contracting_q_to_t_gain_min_c_per_w_m2=(
            args.contracting_q_to_t_gain_min_c_per_w_m2
        ),
        contracting_q_to_t_gain_max_c_per_w_m2=(
            args.contracting_q_to_t_gain_max_c_per_w_m2
        ),
        hp_active_power_nll_weight=args.hp_active_power_nll_weight,
        hp_inactive_leakage_weight=args.hp_inactive_leakage_weight,
        bptt_truncate_steps=args.bptt_truncate_steps,
    )


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
