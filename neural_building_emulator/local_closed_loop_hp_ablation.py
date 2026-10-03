"""Fit and ablate building-specific closed-loop HP emulators.

This diagnostic deliberately removes the global metadata-to-system mapping. A
separate model is fitted to each selected building with an empty metadata
vector and a fixed zero system latent. The first part of the trajectory is used
for fitting and the remaining part is held out as a chronological test set.

The default randomized design compares the structural choices most relevant to
the current HP-emission question. A JSON search space can add or replace factors
without changing this module.

Example:

    .venv/bin/python -m neural_building_emulator.local_closed_loop_hp_ablation \
        --dataset neural_building_emulator/tessin_results.parquet/all_hp \
        --num-profiles 2 --num-configurations 8 --epochs 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    CLOSED_LOOP_INPUT_COLUMNS,
    HP_SIZE_BINDING_COLUMN,
    PROFILE_ID_COLUMN,
    SPACE_HEATING_HP_SIZE_BINDING,
    closed_loop_required_columns,
)
from .data import (
    DEFAULT_DATASET_PATH,
    ClosedLoopProfile,
    SplitConfig,
    WindowConfig,
    _read_parquet,
    load_closed_loop_result_splits,
    make_closed_loop_windows,
    to_closed_loop_profiles,
)
from .models import ProbabilisticContractingClosedLoopHPEmulator
from .scaling import StandardScaler, WindowScalers, inverse_target, transform_closed_loop_profiles
from .train import (
    closed_loop_metrics_dict,
    closed_loop_stability_penalty,
    energy_score,
    local_jacobian_spectral_norm,
    soft_optimistic_loss,
    variogram_score,
)
from .flexibility_event_study_kpis import EventStudyConfig, estimate_trace_flexibility


HPActivationModel = Literal["independent", "persistent_markov", "power_history"]
HPTrainingMode = Literal["expected", "straight_through", "hard_expected_grad"]
HPHistoryFeedback = Literal["predicted", "teacher_forced", "scheduled"]
HPHistoryRepresentation = Literal["single", "multiscale"]
HPTemperatureFeedback = Literal["predicted", "teacher_forced", "scheduled"]
ProcessNoiseMode = Literal["none", "constant", "heteroscedastic"]
TemperatureUpdateMode = Literal[
    "delta",
    "leaky_equilibrium",
    "bounded_equilibrium",
    "alpha_bounded_equilibrium",
]
TrainingObjective = Literal["joint", "hp_only"]
ThermalMatrixMode = Literal["fixed", "time_varying"]

LOSS_COMPONENT_NAMES = (
    "energy",
    "pel_mse",
    "variogram",
    "softopt",
    "hp_bce",
    "hp_active_nll",
    "hp_inactive_leakage",
    "stability",
)

DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "hp_activation_model": ["independent", "persistent_markov", "power_history"],
    "hp_training_mode": ["expected", "straight_through"],
    "process_noise_mode": ["none", "heteroscedastic"],
    "temperature_update_mode": ["delta", "leaky_equilibrium"],
}

ALLOWED_FACTORS = {
    "hp_activation_model",
    "hp_training_mode",
    "hp_history_feedback",
    "hp_history_representation",
    "hp_temperature_feedback",
    "process_noise_mode",
    "temperature_update_mode",
    "training_objective",
    "pointwise_pel_weight",
    "state_dim",
    "hp_history_hours",
    "hp_controller_noise_scale",
    "softopt_weight",
    "variogram_weight",
    "prob_particles",
    "thermal_matrix_mode",
    "stability_weight",
}

INFLUENCE_METRICS = (
    "continuous_test_composite_nrmse",
    "continuous_test_rmse_c",
    "continuous_test_qroom_rmse_w_m2",
    "continuous_test_pel_rmse_w_m2",
    "reset_test_rmse_c",
    "test_hp_mode_brier",
    "test_active_pel_rmse_w_m2",
    "test_event_pel_response_rmse_w_m2",
    "test_event_qroom_response_rmse_w_m2",
    "test_flex_up_abs_error_wh_m2_k",
    "test_flex_down_abs_error_wh_m2_k",
    "max_grad_norm",
    "test_jacobian_mean",
    "test_jacobian_max",
    "test_jacobian_violation_fraction",
    "runtime_seconds",
)


@dataclass(frozen=True)
class LocalVariant:
    hp_activation_model: HPActivationModel = "independent"
    hp_training_mode: HPTrainingMode = "expected"
    hp_history_feedback: HPHistoryFeedback = "predicted"
    hp_history_representation: HPHistoryRepresentation = "single"
    hp_temperature_feedback: HPTemperatureFeedback = "predicted"
    process_noise_mode: ProcessNoiseMode = "none"
    temperature_update_mode: TemperatureUpdateMode = "delta"
    training_objective: TrainingObjective = "joint"
    pointwise_pel_weight: float = 1.0
    state_dim: int = 6
    hp_history_hours: float = 3.0
    hp_controller_noise_scale: float = 0.0
    softopt_weight: float = 0.1
    variogram_weight: float = 0.1
    prob_particles: int = 4
    thermal_matrix_mode: ThermalMatrixMode = "time_varying"
    stability_weight: float = 0.0


@dataclass(frozen=True)
class PreparedProfile:
    raw: ClosedLoopProfile
    local: ClosedLoopProfile
    train: ClosedLoopProfile
    test: ClosedLoopProfile
    split_index: int


@dataclass
class BestPlot:
    score: float
    profile_id: int
    config_id: str
    variant: LocalVariant
    datetime: np.ndarray
    split_index: int
    inputs: np.ndarray
    targets: np.ndarray
    continuous_prediction: np.ndarray
    reset_prediction: np.ndarray


def _finite_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return float("nan")
    left = left[mask]
    right = right[mask]
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(prediction) & np.isfinite(target)
    if not bool(mask.any()):
        return float("nan")
    return float(np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2)))


def _safe_scale(values: np.ndarray) -> float:
    scale = float(np.std(values))
    return scale if np.isfinite(scale) and scale > 1e-6 else 1.0


def _slice_profile(profile: ClosedLoopProfile, start: int, end: int) -> ClosedLoopProfile:
    return ClosedLoopProfile(
        profile_id=profile.profile_id,
        datetime=profile.datetime[start:end],
        metadata=profile.metadata,
        inputs=profile.inputs[start:end],
        targets=profile.targets[start:end],
        initial_temperature=profile.initial_temperature[start:end],
    )


def prepare_profiles(
    profiles: Sequence[ClosedLoopProfile],
    *,
    train_fraction: float,
) -> list[PreparedProfile]:
    prepared: list[PreparedProfile] = []
    for raw_profile in profiles:
        step_count = raw_profile.targets.shape[0]
        split_index = int(round(step_count * train_fraction))
        if split_index < 2 or step_count - split_index < 2:
            raise ValueError(
                f"Profile {raw_profile.profile_id} is too short for train_fraction={train_fraction}"
            )
        local = replace(raw_profile, metadata=np.empty((0,), dtype=np.float32))
        prepared.append(
            PreparedProfile(
                raw=raw_profile,
                local=local,
                train=_slice_profile(local, 0, split_index),
                test=_slice_profile(local, split_index, step_count),
                split_index=split_index,
            )
        )
    return prepared


def fit_common_scalers(profiles: Sequence[PreparedProfile]) -> WindowScalers:
    train_inputs = np.concatenate([profile.train.inputs for profile in profiles], axis=0)
    train_targets = np.concatenate([profile.train.targets for profile in profiles], axis=0)
    return WindowScalers(
        metadata=StandardScaler(
            mean=np.empty((0,), dtype=np.float32),
            scale=np.empty((0,), dtype=np.float32),
        ),
        inputs=StandardScaler.fit(train_inputs, axis=0),
        target=StandardScaler.fit(train_targets, axis=0),
    )


def load_profiles(args: argparse.Namespace) -> list[ClosedLoopProfile]:
    if args.profile_ids:
        requested_ids = tuple(sorted(set(int(value) for value in args.profile_ids)))
        filters = [(PROFILE_ID_COLUMN, "in", list(requested_ids))]
        columns = list(
            dict.fromkeys(
                [*closed_loop_required_columns(), "hp_ref_capacity_W", HP_SIZE_BINDING_COLUMN]
            )
        )
        try:
            frame = _read_parquet(args.dataset, columns=columns, filters=filters)
        except (ValueError, NotImplementedError):
            frame = _read_parquet(args.dataset, columns=columns)
            frame = frame[frame[PROFILE_ID_COLUMN].isin(requested_ids)]
        frame = frame[frame[PROFILE_ID_COLUMN].isin(requested_ids)].copy()
        found_ids = set(int(value) for value in frame[PROFILE_ID_COLUMN].unique())
        missing = sorted(set(requested_ids).difference(found_ids))
        if missing:
            raise ValueError(f"Profiles not found in {args.dataset}: {missing}")
        hp_rows = frame.groupby(PROFILE_ID_COLUMN).first()
        invalid = [
            int(profile_id)
            for profile_id, row in hp_rows.iterrows()
            if not (
                float(row["hp_ref_capacity_W"]) > 0.0
                and str(row[HP_SIZE_BINDING_COLUMN]).strip().upper()
                == SPACE_HEATING_HP_SIZE_BINDING
            )
        ]
        if invalid:
            raise ValueError(f"Requested profiles are not valid space-heating HP profiles: {invalid}")
        frame = frame.sort_values([PROFILE_ID_COLUMN, "datetime"]).reset_index(drop=True)
    else:
        profile_split_fraction = 0.5 if args.num_profiles > 1 else 0.0
        splits = load_closed_loop_result_splits(
            SplitConfig(
                dataset_path=args.dataset,
                max_profiles=args.num_profiles,
                test_fraction=profile_split_fraction,
                seed=args.seed,
                profile_selection=args.profile_selection,
            )
        )
        frame = pd.concat([splits.train, splits.test], ignore_index=True)
    profiles = to_closed_loop_profiles(frame)
    if not profiles:
        raise ValueError("No closed-loop profiles were loaded")
    return profiles


def load_search_space(path: Path | None) -> dict[str, list[Any]]:
    if path is None:
        return {name: list(values) for name, values in DEFAULT_SEARCH_SPACE.items()}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Search-space JSON must be a non-empty object of factor -> levels")
    unknown = sorted(set(raw).difference(ALLOWED_FACTORS))
    if unknown:
        raise ValueError(f"Unsupported search-space factors: {unknown}")
    space: dict[str, list[Any]] = {}
    for name, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Search-space factor {name!r} must contain a non-empty list")
        space[name] = values
    return space


def _coerce_variant(values: dict[str, Any], base: LocalVariant) -> LocalVariant:
    merged = asdict(base)
    merged.update(values)
    variant = LocalVariant(**merged)
    if variant.hp_activation_model not in ("independent", "persistent_markov", "power_history"):
        raise ValueError(f"Invalid hp_activation_model={variant.hp_activation_model!r}")
    if variant.hp_training_mode not in (
        "expected",
        "straight_through",
        "hard_expected_grad",
    ):
        raise ValueError(f"Invalid hp_training_mode={variant.hp_training_mode!r}")
    if variant.hp_history_feedback not in ("predicted", "teacher_forced", "scheduled"):
        raise ValueError(f"Invalid hp_history_feedback={variant.hp_history_feedback!r}")
    if variant.hp_history_representation not in ("single", "multiscale"):
        raise ValueError(
            f"Invalid hp_history_representation={variant.hp_history_representation!r}"
        )
    if variant.hp_temperature_feedback not in (
        "predicted",
        "teacher_forced",
        "scheduled",
    ):
        raise ValueError(
            f"Invalid hp_temperature_feedback={variant.hp_temperature_feedback!r}"
        )
    if (
        variant.hp_history_feedback != "predicted"
        and variant.hp_activation_model != "power_history"
    ):
        raise ValueError(
            "hp_history_feedback requires hp_activation_model='power_history'"
        )
    if (
        variant.hp_history_representation != "single"
        and variant.hp_activation_model != "power_history"
    ):
        raise ValueError(
            "hp_history_representation='multiscale' requires "
            "hp_activation_model='power_history'"
        )
    if variant.process_noise_mode not in ("none", "constant", "heteroscedastic"):
        raise ValueError(f"Invalid process_noise_mode={variant.process_noise_mode!r}")
    if variant.temperature_update_mode not in (
        "delta",
        "leaky_equilibrium",
        "bounded_equilibrium",
        "alpha_bounded_equilibrium",
    ):
        raise ValueError(f"Invalid temperature_update_mode={variant.temperature_update_mode!r}")
    if variant.training_objective not in ("joint", "hp_only"):
        raise ValueError(f"Invalid training_objective={variant.training_objective!r}")
    if variant.thermal_matrix_mode not in ("fixed", "time_varying"):
        raise ValueError(f"Invalid thermal_matrix_mode={variant.thermal_matrix_mode!r}")
    if variant.pointwise_pel_weight < 0.0:
        raise ValueError("pointwise_pel_weight must be non-negative")
    if variant.state_dim < 1 or variant.prob_particles < 2:
        raise ValueError("state_dim must be positive and prob_particles must be at least 2")
    if variant.hp_history_hours <= 0.0:
        raise ValueError("hp_history_hours must be positive")
    if variant.hp_controller_noise_scale < 0.0:
        raise ValueError("hp_controller_noise_scale must be non-negative")
    if variant.softopt_weight < 0.0 or variant.variogram_weight < 0.0:
        raise ValueError("softopt_weight and variogram_weight must be non-negative")
    if variant.stability_weight < 0.0:
        raise ValueError("stability_weight must be non-negative")
    return variant


def balanced_random_design(
    space: dict[str, list[Any]],
    *,
    num_configurations: int,
    seed: int,
    base: LocalVariant,
) -> list[LocalVariant]:
    if num_configurations < 1:
        raise ValueError("num_configurations must be positive")
    names = tuple(space)
    combinations = list(itertools.product(*(space[name] for name in names)))
    if num_configurations > len(combinations):
        raise ValueError(
            f"Requested {num_configurations} unique configurations, but the search "
            f"space contains only {len(combinations)}; broaden the search space or "
            "use fewer configurations"
        )

    rng = np.random.default_rng(seed)
    baseline = tuple(space[name][0] for name in names)
    selected = [baseline]
    candidates = [combination for combination in combinations if combination != baseline]
    rng.shuffle(candidates)

    # Greedily balance both marginal level counts and pairwise combinations. The
    # latter prevents a small random design from confounding two structural choices.
    while len(selected) < num_configurations:
        next_size = len(selected) + 1

        def imbalance(candidate: tuple[Any, ...]) -> float:
            proposed = selected + [candidate]
            score = 0.0
            for factor_index, name in enumerate(names):
                target = next_size / len(space[name])
                counts = [
                    sum(row[factor_index] == level for row in proposed)
                    for level in space[name]
                ]
                score += sum((count - target) ** 2 for count in counts)
            for left in range(len(names)):
                for right in range(left + 1, len(names)):
                    target = next_size / (len(space[names[left]]) * len(space[names[right]]))
                    for left_level in space[names[left]]:
                        for right_level in space[names[right]]:
                            count = sum(
                                row[left] == left_level and row[right] == right_level
                                for row in proposed
                            )
                            score += 0.25 * (count - target) ** 2
            return score

        scores = np.asarray([imbalance(candidate) for candidate in candidates])
        best = np.flatnonzero(np.isclose(scores, np.min(scores)))
        chosen_index = int(rng.choice(best))
        selected.append(candidates.pop(chosen_index))

    rows = [dict(zip(names, combination)) for combination in selected]
    return [_coerce_variant(row, base) for row in rows]


def full_factorial_design(
    space: dict[str, list[Any]],
    *,
    base: LocalVariant,
) -> list[LocalVariant]:
    names = tuple(space)
    return [
        _coerce_variant(dict(zip(names, combination)), base)
        for combination in itertools.product(*(space[name] for name in names))
    ]


def resolve_caps(
    profiles: Sequence[PreparedProfile],
    *,
    cap_factor: float,
    hp_dt_hours: float,
    cop_cap: float,
) -> tuple[float, float, float]:
    targets = np.concatenate([profile.train.targets for profile in profiles], axis=0)
    pel_cap = max(cap_factor * float(np.max(targets[:, 2])), 1.0)
    qroom_cap = max(cap_factor * float(np.max(targets[:, 1])), 1.0)
    energy_cap = max(24.0 * qroom_cap, hp_dt_hours * cop_cap * pel_cap, 1.0)
    return pel_cap, qroom_cap, energy_cap


def build_local_model(
    variant: LocalVariant,
    args: argparse.Namespace,
    scalers: WindowScalers,
    caps: tuple[float, float, float],
    *,
    key: jax.Array,
) -> ProbabilisticContractingClosedLoopHPEmulator:
    pel_cap, qroom_cap, energy_cap = caps
    return ProbabilisticContractingClosedLoopHPEmulator(
        metadata_dim=0,
        input_dim=int(scalers.inputs.mean.shape[0]),
        state_dim=variant.state_dim,
        controller_state_dim=args.hp_controller_state_dim,
        latent_dim=1,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        input_encoder_dim=args.input_encoder_dim,
        input_encoder_hidden_dim=args.input_encoder_hidden_dim,
        input_encoder_depth=args.input_encoder_depth,
        process_noise_mode=variant.process_noise_mode,
        process_noise_init=args.process_noise_init,
        hp_emission_mode="bounded",
        hp_activation_model=variant.hp_activation_model,
        hp_persistent_latent_dim=args.hp_persistent_latent_dim,
        hp_controller_leak=args.hp_controller_leak,
        hp_controller_noise_scale=variant.hp_controller_noise_scale,
        hp_history_hours=variant.hp_history_hours,
        hp_history_timescales_hours=(
            tuple(args.hp_history_timescales_hours)
            if variant.hp_history_representation == "multiscale"
            else ()
        ),
        contraction_gamma=args.contracting_gamma,
        state_bound=args.contracting_state_bound,
        temperature_output_scale=args.contracting_temperature_scale,
        temperature_delta_max_c=args.contracting_temperature_delta_max_c,
        temperature_update_mode=variant.temperature_update_mode,
        hp_dt_hours=args.hp_dt_hours,
        hp_cop_floor=1.0,
        hp_cop_cap=args.hp_cop_cap,
        hp_pel_cap_w_m2=pel_cap,
        hp_qroom_cap_w_m2=qroom_cap,
        hp_energy_cap_wh_m2=energy_cap,
        input_mean=tuple(float(value) for value in scalers.inputs.mean),
        input_scale=tuple(float(value) for value in scalers.inputs.scale),
        target_mean=tuple(float(value) for value in scalers.target.mean),
        target_scale=tuple(float(value) for value in scalers.target.scale),
        availability_input_index=list(CLOSED_LOOP_INPUT_COLUMNS).index("space_heating_available"),
        bptt_truncate_steps=args.bptt_truncate_steps,
        transition_matrix_mode=variant.thermal_matrix_mode,
        key=key,
    )


def local_sample_with_aux(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    key: jax.Array,
    num_particles: int,
    hp_scenario_mode: Literal[
        "expected",
        "straight_through",
        "hard_expected_grad",
        "bernoulli",
    ],
    sample_process_noise: bool,
    concrete_temperature: float,
    hp_feedback_pel: jnp.ndarray | None = None,
    hp_feedback_mode: jnp.ndarray | None = None,
    hp_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
    hp_feedback_temperature: jnp.ndarray | None = None,
    hp_temperature_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    """Sample emissions while holding building/system latent variables fixed at zero."""
    process_key, mode_key, power_key, controller_key = jax.random.split(key, 4)
    horizon = inputs.shape[0]
    xi = jnp.zeros((num_particles, model.latent_dim), dtype=inputs.dtype)
    hp_latent_dim = (
        model.hp_persistent_latent_dim
        if model.hp_activation_model == "persistent_markov"
        else 0
    )
    hp_latent = jnp.zeros((num_particles, hp_latent_dim), dtype=inputs.dtype)
    if sample_process_noise and model.process_noise_mode != "none":
        process_noise = jax.random.normal(
            process_key,
            (num_particles, horizon, model.state_dim),
        )
    else:
        process_noise = jnp.zeros((num_particles, horizon, model.state_dim), dtype=inputs.dtype)
    hp_mode_uniform = jax.random.uniform(mode_key, (num_particles, horizon))
    hp_power_noise = jax.random.normal(power_key, (num_particles, horizon))
    if model.hp_controller_noise_scale > 0.0:
        controller_noise = jax.random.normal(
            controller_key,
            (num_particles, horizon, model.controller_state_dim),
        )
    else:
        controller_noise = jnp.zeros(
            (num_particles, horizon, model.controller_state_dim),
            dtype=inputs.dtype,
        )
    metadata = jnp.empty((0,), dtype=inputs.dtype)
    predictions, aux = jax.vmap(
        lambda particle_xi, particle_process, particle_mode, particle_power, particle_hp_latent, particle_controller: model.rollout_particle(
            metadata,
            inputs,
            initial_temperature,
            particle_xi,
            particle_process,
            particle_mode,
            particle_power,
            particle_hp_latent,
            particle_controller,
            hp_scenario_mode=hp_scenario_mode,
            hp_concrete_temperature=jnp.asarray(concrete_temperature, dtype=inputs.dtype),
            hp_feedback_pel=hp_feedback_pel,
            hp_feedback_mode=hp_feedback_mode,
            hp_teacher_forcing_ratio=hp_teacher_forcing_ratio,
            hp_feedback_temperature=hp_feedback_temperature,
            hp_temperature_teacher_forcing_ratio=hp_temperature_teacher_forcing_ratio,
        )
    )(xi, process_noise, hp_mode_uniform, hp_power_noise, hp_latent, controller_noise)
    # Match the production auxiliary layout so shared stability diagnostics use
    # the exact local xi=0 and HP latent states rather than resampling them.
    return predictions, (
        *aux[:13],
        xi,
        *aux[13:15],
        hp_latent,
        *aux[15:],
    )


def local_sample_batch(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    key: jax.Array,
    num_particles: int,
    hp_scenario_mode: Literal[
        "expected",
        "straight_through",
        "hard_expected_grad",
        "bernoulli",
    ],
    sample_process_noise: bool,
    concrete_temperature: float,
    hp_feedback_pel: jnp.ndarray | None = None,
    hp_feedback_mode: jnp.ndarray | None = None,
    hp_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
    hp_feedback_temperature: jnp.ndarray | None = None,
    hp_temperature_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    keys = jax.random.split(key, inputs.shape[0])
    if hp_feedback_pel is not None and hp_feedback_temperature is not None:
        if hp_feedback_mode is None:
            raise ValueError("hp_feedback_mode is required with hp_feedback_pel")
        return jax.vmap(
            lambda row_inputs, row_initial_temperature, row_key, row_pel, row_mode, row_temperature: local_sample_with_aux(
                model,
                row_inputs,
                row_initial_temperature,
                key=row_key,
                num_particles=num_particles,
                hp_scenario_mode=hp_scenario_mode,
                sample_process_noise=sample_process_noise,
                concrete_temperature=concrete_temperature,
                hp_feedback_pel=row_pel,
                hp_feedback_mode=row_mode,
                hp_teacher_forcing_ratio=hp_teacher_forcing_ratio,
                hp_feedback_temperature=row_temperature,
                hp_temperature_teacher_forcing_ratio=hp_temperature_teacher_forcing_ratio,
            )
        )(
            inputs,
            initial_temperature,
            keys,
            hp_feedback_pel,
            hp_feedback_mode,
            hp_feedback_temperature,
        )
    if hp_feedback_pel is not None:
        if hp_feedback_mode is None:
            raise ValueError("hp_feedback_mode is required with hp_feedback_pel")
        return jax.vmap(
            lambda row_inputs, row_initial_temperature, row_key, row_pel, row_mode: local_sample_with_aux(
                model,
                row_inputs,
                row_initial_temperature,
                key=row_key,
                num_particles=num_particles,
                hp_scenario_mode=hp_scenario_mode,
                sample_process_noise=sample_process_noise,
                concrete_temperature=concrete_temperature,
                hp_feedback_pel=row_pel,
                hp_feedback_mode=row_mode,
                hp_teacher_forcing_ratio=hp_teacher_forcing_ratio,
            )
        )(inputs, initial_temperature, keys, hp_feedback_pel, hp_feedback_mode)
    if hp_feedback_temperature is not None:
        return jax.vmap(
            lambda row_inputs, row_initial_temperature, row_key, row_temperature: local_sample_with_aux(
                model,
                row_inputs,
                row_initial_temperature,
                key=row_key,
                num_particles=num_particles,
                hp_scenario_mode=hp_scenario_mode,
                sample_process_noise=sample_process_noise,
                concrete_temperature=concrete_temperature,
                hp_feedback_temperature=row_temperature,
                hp_temperature_teacher_forcing_ratio=hp_temperature_teacher_forcing_ratio,
            )
        )(inputs, initial_temperature, keys, hp_feedback_temperature)
    return jax.vmap(
        lambda row_inputs, row_initial_temperature, row_key: local_sample_with_aux(
            model,
            row_inputs,
            row_initial_temperature,
            key=row_key,
            num_particles=num_particles,
            hp_scenario_mode=hp_scenario_mode,
            sample_process_noise=sample_process_noise,
            concrete_temperature=concrete_temperature,
        )
    )(inputs, initial_temperature, keys)


def hp_history_teacher_forcing_ratio(
    variant: LocalVariant,
    args: argparse.Namespace,
    epoch: int,
) -> float:
    """Return the epoch-level observed-history mixing ratio."""
    if variant.hp_history_feedback == "predicted":
        return 0.0
    if variant.hp_history_feedback == "teacher_forced":
        return 1.0
    anneal_epochs = (
        args.hp_history_teacher_forcing_anneal_epochs
        if args.hp_history_teacher_forcing_anneal_epochs > 0
        else args.epochs
    )
    progress = min(max(epoch / max(anneal_epochs - 1, 1), 0.0), 1.0)
    start = args.hp_history_teacher_forcing_start
    end = args.hp_history_teacher_forcing_end
    return float(start + progress * (end - start))


def hp_temperature_teacher_forcing_ratio(
    variant: LocalVariant,
    args: argparse.Namespace,
    epoch: int,
) -> float:
    """Return the observed-current-temperature fraction seen by the HP head."""
    if variant.hp_temperature_feedback == "predicted":
        return 0.0
    if variant.hp_temperature_feedback == "teacher_forced":
        return 1.0
    anneal_epochs = (
        args.hp_temperature_teacher_forcing_anneal_epochs
        if args.hp_temperature_teacher_forcing_anneal_epochs > 0
        else args.epochs
    )
    progress = min(max(epoch / max(anneal_epochs - 1, 1), 0.0), 1.0)
    start = args.hp_temperature_teacher_forcing_start
    end = args.hp_temperature_teacher_forcing_end
    return float(start + progress * (end - start))


def observed_current_temperature_sequence(
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """Return the observed temperature preceding each interval target."""
    if initial_temperature.ndim != 2 or initial_temperature.shape[1] != 1:
        raise ValueError("initial_temperature must have shape [batch, 1]")
    if targets.ndim != 3 or targets.shape[2] < 1:
        raise ValueError("targets must have shape [batch, horizon, outputs]")
    return jnp.concatenate(
        [initial_temperature, targets[:, :-1, 0]],
        axis=1,
    )


def _tree_all_finite(tree: Any) -> jnp.ndarray:
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf)]
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def make_train_step(
    variant: LocalVariant,
    args: argparse.Namespace,
    optimizer: optax.GradientTransformation,
):
    history_warmup_steps = (
        max(0, int(round(args.hp_history_loss_warmup_hours / args.hp_dt_hours)))
        if variant.hp_activation_model == "power_history"
        else 0
    )

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_and_grad(
        model: ProbabilisticContractingClosedLoopHPEmulator,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        targets: jnp.ndarray,
        key: jax.Array,
        teacher_forcing_ratio: jnp.ndarray,
        temperature_teacher_forcing_ratio: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        target_mean = jnp.asarray(model.target_mean, dtype=targets.dtype)
        target_scale = jnp.asarray(model.target_scale, dtype=targets.dtype)
        pel_target = targets[..., 2] * target_scale[2] + target_mean[2]
        mode_target = (pel_target > args.heat_on_threshold_w_m2).astype(targets.dtype)
        feedback_pel = (
            None if variant.hp_history_feedback == "predicted" else pel_target
        )
        feedback_mode = (
            None if variant.hp_history_feedback == "predicted" else mode_target
        )
        observed_temperature = observed_current_temperature_sequence(
            initial_temperature,
            targets,
        )
        feedback_temperature = (
            None
            if variant.hp_temperature_feedback == "predicted"
            else observed_temperature
        )
        rollout_key, stability_key = jax.random.split(key)
        predictions, aux = local_sample_batch(
            model,
            inputs,
            initial_temperature,
            key=rollout_key,
            num_particles=variant.prob_particles,
            hp_scenario_mode=variant.hp_training_mode,
            sample_process_noise=variant.process_noise_mode != "none",
            concrete_temperature=args.hp_concrete_temperature,
            hp_feedback_pel=feedback_pel,
            hp_feedback_mode=feedback_mode,
            hp_teacher_forcing_ratio=teacher_forcing_ratio,
            hp_feedback_temperature=feedback_temperature,
            hp_temperature_teacher_forcing_ratio=temperature_teacher_forcing_ratio,
        )
        loss_predictions = predictions[:, :, history_warmup_steps:, :]
        loss_targets = targets[:, history_warmup_steps:, :]
        loss_inputs = inputs[:, history_warmup_steps:, :]
        zero = jnp.asarray(0.0, dtype=predictions.dtype)
        energy_component = zero
        pel_mse_component = zero
        if variant.training_objective == "joint":
            trajectory_output_weights = (1.0, 1.0, variant.pointwise_pel_weight)
            energy_component = energy_score(
                loss_predictions,
                loss_targets,
                output_weights=trajectory_output_weights,
            )
        else:
            pel_mse_component = jnp.mean(
                (loss_predictions[..., 2] - loss_targets[:, jnp.newaxis, :, 2]) ** 2
            )
        variogram_component = zero
        if variant.training_objective == "joint" and variant.variogram_weight > 0.0:
            variogram_component = variant.variogram_weight * variogram_score(
                loss_predictions,
                loss_targets,
                lags=args.variogram_lags,
                power=0.5,
                output_weights=(1.0, 0.0, 0.0),
                constant_setpoint_inputs=loss_inputs,
                setpoint_delta_threshold=(
                    args.setpoint_change_threshold_c / max(float(model.input_scale[0]), 1e-6)
                ),
            )
        softopt_component = zero
        if variant.training_objective == "joint" and variant.softopt_weight > 0.0:
            softopt_component = variant.softopt_weight * soft_optimistic_loss(
                loss_predictions,
                loss_targets,
                temperature=0.1,
                horizon_weight_power=0.0,
                output_weights=(1.0, 1.0, variant.pointwise_pel_weight),
            )

        loss_pel_target = pel_target[:, history_warmup_steps:]
        loss_mode_target = mode_target[:, history_warmup_steps:, jnp.newaxis]
        loss_mode_target = jnp.swapaxes(loss_mode_target, 1, 2)
        pi = jnp.clip(aux[0][..., history_warmup_steps:], 1e-5, 1.0 - 1e-5)
        hp_bce_component = args.hp_mode_loss_weight * -jnp.mean(
            loss_mode_target * jnp.log(pi)
            + (1.0 - loss_mode_target) * jnp.log1p(-pi)
        )

        log_mu = aux[8][..., history_warmup_steps:]
        log_sigma = jnp.maximum(
            aux[9][..., history_warmup_steps:],
            jnp.asarray(1e-5, dtype=targets.dtype),
        )
        log_target = jnp.log1p(jnp.maximum(loss_pel_target, 0.0))[:, jnp.newaxis, :]
        active_mask = jnp.broadcast_to(loss_mode_target, log_mu.shape)
        log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype=targets.dtype)
        active_nll = (
            0.5 * ((log_target - log_mu) / log_sigma) ** 2
            + jnp.log(log_sigma)
            + 0.5 * log_two_pi
        )
        hp_active_nll_component = args.hp_active_nll_weight * (
            jnp.sum(active_mask * active_nll) / jnp.maximum(jnp.sum(active_mask), 1.0)
        )
        expected_pel = pi * model.expected_active_power(log_mu, log_sigma)
        inactive_mask = 1.0 - active_mask
        inactive_leakage_component = args.hp_inactive_leakage_weight * (
            jnp.sum(inactive_mask * (expected_pel / target_scale[2]) ** 2)
            / jnp.maximum(jnp.sum(inactive_mask), 1.0)
        )
        stability_component = zero
        if variant.stability_weight > 0.0:
            stability_component = (
                variant.stability_weight
                * closed_loop_stability_penalty(
                    model,
                    jnp.empty((inputs.shape[0], 0), dtype=inputs.dtype),
                    inputs,
                    aux,
                    stability_key,
                    gamma=args.closed_loop_stability_gamma,
                    num_samples=args.closed_loop_stability_samples,
                    aggregation=args.closed_loop_stability_aggregation,
                    method=args.closed_loop_stability_method,
                    power_iterations=args.closed_loop_stability_power_iterations,
                )
            )
        components = jnp.stack(
            [
                energy_component,
                pel_mse_component,
                variogram_component,
                softopt_component,
                hp_bce_component,
                hp_active_nll_component,
                inactive_leakage_component,
                stability_component,
            ]
        )
        return jnp.sum(components), components

    @eqx.filter_jit
    def train_step(
        model: ProbabilisticContractingClosedLoopHPEmulator,
        opt_state: optax.OptState,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        targets: jnp.ndarray,
        key: jax.Array,
        teacher_forcing_ratio: jnp.ndarray,
        temperature_teacher_forcing_ratio: jnp.ndarray,
    ) -> tuple[
        ProbabilisticContractingClosedLoopHPEmulator,
        optax.OptState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        (loss, components), grads = loss_and_grad(
            model,
            inputs,
            initial_temperature,
            targets,
            key,
            teacher_forcing_ratio,
            temperature_teacher_forcing_ratio,
        )
        grad_norm = optax.global_norm(grads)
        updates, candidate_opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        candidate_model = eqx.apply_updates(model, updates)
        finite = (
            jnp.isfinite(loss)
            & _tree_all_finite(components)
            & _tree_all_finite(grads)
            & _tree_all_finite(updates)
            & _tree_all_finite(candidate_model)
        )
        model = jax.tree_util.tree_map(
            lambda candidate, current: jnp.where(finite, candidate, current)
            if eqx.is_array(candidate)
            else candidate,
            candidate_model,
            model,
        )
        opt_state = jax.tree_util.tree_map(
            lambda candidate, current: jnp.where(finite, candidate, current)
            if eqx.is_array(candidate)
            else candidate,
            candidate_opt_state,
            opt_state,
        )
        return model, opt_state, loss, components, grad_norm, finite

    return train_step


def padded_batches(
    window_count: int,
    *,
    batch_size: int,
    rng: np.random.Generator,
    max_batches: int,
) -> list[np.ndarray]:
    order = rng.permutation(window_count)
    batch_count = math.ceil(window_count / batch_size)
    if max_batches > 0:
        batch_count = min(batch_count, max_batches)
    batches: list[np.ndarray] = []
    for batch_index in range(batch_count):
        start = batch_index * batch_size
        indices = order[start : start + batch_size]
        if indices.size < batch_size:
            indices = np.concatenate([indices, order[: batch_size - indices.size]])
        batches.append(indices)
    return batches


def train_local_model(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    scaled_train_profile: ClosedLoopProfile,
    variant: LocalVariant,
    args: argparse.Namespace,
    *,
    seed: int,
    train_step,
    optimizer: optax.GradientTransformation,
) -> tuple[ProbabilisticContractingClosedLoopHPEmulator, dict[str, float]]:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    component_sum = np.zeros((len(LOSS_COMPONENT_NAMES),), dtype=np.float64)
    loss_sum = 0.0
    update_count = 0
    skipped_updates = 0
    max_grad_norm = 0.0
    first_teacher_forcing_ratio = hp_history_teacher_forcing_ratio(variant, args, 0)
    last_teacher_forcing_ratio = first_teacher_forcing_ratio
    first_temperature_teacher_forcing_ratio = hp_temperature_teacher_forcing_ratio(
        variant,
        args,
        0,
    )
    last_temperature_teacher_forcing_ratio = first_temperature_teacher_forcing_ratio
    started = time.perf_counter()
    for epoch in range(args.epochs):
        last_teacher_forcing_ratio = hp_history_teacher_forcing_ratio(variant, args, epoch)
        last_temperature_teacher_forcing_ratio = hp_temperature_teacher_forcing_ratio(
            variant,
            args,
            epoch,
        )
        start_offset = 0
        if args.rotate_window_starts and args.stride > 1:
            start_offset = (epoch * max(1, args.stride // max(args.epochs, 1))) % args.stride
        windows = make_closed_loop_windows(
            [scaled_train_profile],
            WindowConfig(
                sequence_length=args.sequence_length,
                stride=args.stride,
                target_alignment="same_time",
            ),
            start_offset=start_offset,
        )
        batches = padded_batches(
            windows.targets.shape[0],
            batch_size=args.batch_size,
            rng=rng,
            max_batches=args.max_train_batches,
        )
        for indices in batches:
            key, step_key = jax.random.split(key)
            model, opt_state, loss, components, grad_norm, finite = train_step(
                model,
                opt_state,
                jnp.asarray(windows.inputs[indices]),
                jnp.asarray(windows.initial_temperature[indices]),
                jnp.asarray(windows.targets[indices]),
                step_key,
                jnp.asarray(last_teacher_forcing_ratio, dtype=jnp.float32),
                jnp.asarray(
                    last_temperature_teacher_forcing_ratio,
                    dtype=jnp.float32,
                ),
            )
            loss_value = float(np.asarray(loss))
            grad_value = float(np.asarray(grad_norm))
            max_grad_norm = max(max_grad_norm, grad_value if np.isfinite(grad_value) else np.inf)
            if not bool(np.asarray(finite)):
                skipped_updates += 1
                continue
            loss_sum += loss_value
            component_sum += np.asarray(components, dtype=np.float64)
            update_count += 1
    elapsed = time.perf_counter() - started
    denominator = max(update_count, 1)
    stats = {
        "final_train_loss": loss_sum / denominator,
        "max_grad_norm": max_grad_norm,
        "update_count": float(update_count),
        "skipped_updates": float(skipped_updates),
        "runtime_seconds": elapsed,
        "teacher_forcing_ratio_first": first_teacher_forcing_ratio,
        "teacher_forcing_ratio_last": last_teacher_forcing_ratio,
        "temperature_teacher_forcing_ratio_first": (
            first_temperature_teacher_forcing_ratio
        ),
        "temperature_teacher_forcing_ratio_last": (
            last_temperature_teacher_forcing_ratio
        ),
        "history_loss_warmup_steps": float(
            max(0, int(round(args.hp_history_loss_warmup_hours / args.hp_dt_hours)))
            if variant.hp_activation_model == "power_history"
            else 0
        ),
    }
    stats.update(
        {
            f"train_loss_{name}": float(component_sum[index] / denominator)
            for index, name in enumerate(LOSS_COMPONENT_NAMES)
        }
    )
    return model, stats


def rollout_physical(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    profile: ClosedLoopProfile,
    scalers: WindowScalers,
    *,
    key: jax.Array,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    inputs = scalers.inputs.transform(profile.inputs)
    target_mean = np.asarray(scalers.target.mean, dtype=np.float32)
    target_scale = np.asarray(scalers.target.scale, dtype=np.float32)
    initial_temperature = (
        (profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]
    ).astype(np.float32)
    predictions, aux = local_sample_with_aux(
        model,
        jnp.asarray(inputs),
        jnp.asarray(initial_temperature),
        key=key,
        num_particles=1,
        hp_scenario_mode="expected",
        sample_process_noise=False,
        concrete_temperature=0.5,
    )
    prediction = inverse_target(np.asarray(predictions[0]), scalers)
    return prediction, tuple(np.asarray(value[0]) for value in aux)


def local_jacobian_metrics(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    profile: PreparedProfile,
    scalers: WindowScalers,
    aux: tuple[np.ndarray, ...],
    args: argparse.Namespace,
    *,
    key: jax.Array,
) -> dict[str, float]:
    """Estimate closed-loop Jacobian norms along held-out rollout states."""
    available = profile.local.targets.shape[0] - profile.split_index
    sample_count = min(args.jacobian_diagnostic_samples, available)
    if sample_count < 1:
        return {
            "test_jacobian_mean": float("nan"),
            "test_jacobian_max": float("nan"),
            "test_jacobian_violation_fraction": float("nan"),
        }

    index_key, direction_key, process_key, controller_key = jax.random.split(key, 4)
    indices = jax.random.choice(
        index_key,
        jnp.arange(profile.split_index, profile.local.targets.shape[0]),
        shape=(sample_count,),
        replace=False,
    )
    directions = jax.random.normal(
        direction_key,
        (sample_count, model.stability_state_dim()),
    )
    process_noise = (
        jax.random.normal(process_key, (sample_count, model.state_dim))
        if model.process_noise_mode != "none"
        else jnp.zeros((sample_count, model.state_dim))
    )
    controller_noise = (
        jax.random.normal(controller_key, (sample_count, model.controller_state_dim))
        if model.hp_controller_noise_scale > 0.0
        else jnp.zeros((sample_count, model.controller_state_dim))
    )
    scaled_inputs = jnp.asarray(scalers.inputs.transform(profile.local.inputs))
    metadata = jnp.empty((0,), dtype=scaled_inputs.dtype)
    xi = jnp.asarray(aux[13])
    hp_latent = jnp.asarray(aux[16])
    x_state = jnp.asarray(aux[10])
    controller_state = jnp.asarray(aux[11])
    energy = jnp.asarray(aux[2])
    temperature = jnp.asarray(aux[12])
    hp_power_history = jnp.asarray(aux[17])
    hp_mode_history = jnp.asarray(aux[18])
    previous_mode = jnp.asarray(aux[19])

    def norm_at(
        time_index: jnp.ndarray,
        direction: jnp.ndarray,
        process_noise_t: jnp.ndarray,
        controller_noise_t: jnp.ndarray,
    ) -> jnp.ndarray:
        state = model.pack_stability_state(
            x_state[time_index],
            energy[time_index],
            temperature[time_index],
            controller_state[time_index],
            previous_mode[time_index],
            hp_power_history[time_index],
            hp_mode_history[time_index],
        )
        step_fn = lambda current: model.one_step_augmented_state(
            metadata,
            scaled_inputs[time_index],
            xi,
            hp_latent,
            current,
            process_noise_t,
            controller_noise_t,
        )
        return local_jacobian_spectral_norm(
            step_fn,
            state,
            direction,
            method=args.closed_loop_stability_method,
            power_iterations=args.closed_loop_stability_power_iterations,
        )

    norms = np.asarray(
        jax.vmap(norm_at)(indices, directions, process_noise, controller_noise)
    )
    return {
        "test_jacobian_mean": float(np.mean(norms)),
        "test_jacobian_max": float(np.max(norms)),
        "test_jacobian_violation_fraction": float(
            np.mean(norms > args.closed_loop_stability_gamma)
        ),
    }


def event_response_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    setpoint: np.ndarray,
    *,
    start: int,
    end: int,
    horizon_steps: int,
    threshold_c: float,
) -> dict[str, float]:
    setpoint_delta = np.diff(setpoint, prepend=setpoint[0])
    candidates = np.flatnonzero(np.abs(setpoint_delta) >= threshold_c)
    events = candidates[
        (candidates - horizon_steps >= start)
        & (candidates + horizon_steps <= end)
    ]
    result: dict[str, float] = {"test_event_count": float(events.size)}
    for channel, name in ((1, "qroom"), (2, "pel")):
        true_response = np.asarray(
            [
                np.mean(targets[index : index + horizon_steps, channel])
                - np.mean(targets[index - horizon_steps : index, channel])
                for index in events
            ],
            dtype=np.float64,
        )
        predicted_response = np.asarray(
            [
                np.mean(predictions[index : index + horizon_steps, channel])
                - np.mean(predictions[index - horizon_steps : index, channel])
                for index in events
            ],
            dtype=np.float64,
        )
        result[f"test_event_{name}_response_rmse_w_m2"] = _rmse(
            predicted_response,
            true_response,
        )
        result[f"test_event_{name}_response_bias_w_m2"] = (
            float(np.mean(predicted_response - true_response))
            if events.size
            else float("nan")
        )
        result[f"test_event_{name}_response_corr"] = _finite_corr(
            predicted_response,
            true_response,
        )
    return result


def flexibility_kpi_metrics(
    profile: ClosedLoopProfile,
    targets: np.ndarray,
    predictions: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    horizon_steps = max(1, int(round(args.event_horizon_hours / args.hp_dt_hours)))
    config = EventStudyConfig(
        horizons_hours=(float(horizon_steps * args.hp_dt_hours),),
        horizon_steps=(horizon_steps,),
        dt_hours=args.hp_dt_hours,
        min_setpoint_change=args.setpoint_change_threshold_c,
        min_events=args.flex_kpi_min_events,
        controls="full",
    )
    true_kpi = estimate_trace_flexibility(profile, targets, config)[0]
    predicted_kpi = estimate_trace_flexibility(profile, predictions, config)[0]
    result = {
        "test_flex_event_count": float(min(true_kpi.event_count, predicted_kpi.event_count)),
        "test_flex_up_true_wh_m2_k": float(true_kpi.up_flex_wh_m2_k),
        "test_flex_up_pred_wh_m2_k": float(predicted_kpi.up_flex_wh_m2_k),
        "test_flex_down_true_wh_m2_k": float(true_kpi.down_flex_wh_m2_k),
        "test_flex_down_pred_wh_m2_k": float(predicted_kpi.down_flex_wh_m2_k),
    }
    if true_kpi.valid and predicted_kpi.valid:
        result.update(
            {
                "test_flex_up_error_wh_m2_k": float(
                    predicted_kpi.up_flex_wh_m2_k - true_kpi.up_flex_wh_m2_k
                ),
                "test_flex_down_error_wh_m2_k": float(
                    predicted_kpi.down_flex_wh_m2_k - true_kpi.down_flex_wh_m2_k
                ),
                "test_flex_up_abs_error_wh_m2_k": float(
                    abs(predicted_kpi.up_flex_wh_m2_k - true_kpi.up_flex_wh_m2_k)
                ),
                "test_flex_down_abs_error_wh_m2_k": float(
                    abs(predicted_kpi.down_flex_wh_m2_k - true_kpi.down_flex_wh_m2_k)
                ),
                "test_flex_up_sign_match": float(
                    np.sign(predicted_kpi.up_flex_wh_m2_k)
                    == np.sign(true_kpi.up_flex_wh_m2_k)
                ),
                "test_flex_down_sign_match": float(
                    np.sign(predicted_kpi.down_flex_wh_m2_k)
                    == np.sign(true_kpi.down_flex_wh_m2_k)
                ),
            }
        )
    else:
        result.update(
            {
                "test_flex_up_error_wh_m2_k": float("nan"),
                "test_flex_down_error_wh_m2_k": float("nan"),
                "test_flex_up_abs_error_wh_m2_k": float("nan"),
                "test_flex_down_abs_error_wh_m2_k": float("nan"),
                "test_flex_up_sign_match": float("nan"),
                "test_flex_down_sign_match": float("nan"),
            }
        )
    return result


def relationship_metrics(
    profile: PreparedProfile,
    continuous_prediction: np.ndarray,
    continuous_aux: tuple[np.ndarray, ...],
    reset_prediction: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    split = profile.split_index
    targets = profile.local.targets
    test_target = targets[split:]
    test_prediction = continuous_prediction[split:]
    reset_metrics = closed_loop_metrics_dict(reset_prediction, test_target)
    continuous_metrics = closed_loop_metrics_dict(test_prediction, test_target)
    metrics = {
        f"continuous_test_{name}": value for name, value in continuous_metrics.items()
    }
    metrics.update({f"reset_test_{name}": value for name, value in reset_metrics.items()})
    train_metrics = closed_loop_metrics_dict(
        continuous_prediction[:split],
        targets[:split],
    )
    metrics.update({f"continuous_train_{name}": value for name, value in train_metrics.items()})

    target_scales = [_safe_scale(test_target[:, index]) for index in range(3)]
    metrics["continuous_test_composite_nrmse"] = float(
        np.mean(
            [
                continuous_metrics["rmse_c"] / target_scales[0],
                continuous_metrics["qroom_rmse_w_m2"] / target_scales[1],
                continuous_metrics["pel_rmse_w_m2"] / target_scales[2],
            ]
        )
    )

    true_mode = (test_target[:, 2] > args.heat_on_threshold_w_m2).astype(np.float64)
    predicted_pi = np.clip(continuous_aux[0][split:], 1e-6, 1.0 - 1e-6)
    metrics["test_hp_mode_brier"] = float(np.mean((predicted_pi - true_mode) ** 2))
    metrics["test_hp_mode_log_loss"] = float(
        -np.mean(true_mode * np.log(predicted_pi) + (1.0 - true_mode) * np.log1p(-predicted_pi))
    )
    metrics["test_hp_mode_rate_true"] = float(np.mean(true_mode))
    metrics["test_hp_mode_rate_pred"] = float(np.mean(predicted_pi))
    active = true_mode > 0.5
    metrics["test_active_pel_rmse_w_m2"] = (
        _rmse(test_prediction[active, 2], test_target[active, 2])
        if bool(active.any())
        else float("nan")
    )

    true_current_temperature = profile.local.initial_temperature[:, 0]
    predicted_current_temperature = np.concatenate(
        [
            profile.local.initial_temperature[:1, 0],
            continuous_prediction[:-1, 0],
        ]
    )
    setpoint = profile.local.inputs[:, 0]
    true_gap = setpoint - true_current_temperature
    predicted_gap = setpoint - predicted_current_temperature
    test_slice = slice(split, None)
    metrics["test_gap_pel_corr_true"] = _finite_corr(true_gap[test_slice], test_target[:, 2])
    metrics["test_gap_pel_corr_pred"] = _finite_corr(
        predicted_gap[test_slice],
        test_prediction[:, 2],
    )
    metrics["test_delta_pel_corr_true_pred"] = _finite_corr(
        np.diff(test_target[:, 2]),
        np.diff(test_prediction[:, 2]),
    )
    metrics["test_delta_qroom_corr_true_pred"] = _finite_corr(
        np.diff(test_target[:, 1]),
        np.diff(test_prediction[:, 1]),
    )
    metrics["test_qroom_pel_corr_true"] = _finite_corr(test_target[:, 1], test_target[:, 2])
    metrics["test_qroom_pel_corr_pred"] = _finite_corr(
        test_prediction[:, 1],
        test_prediction[:, 2],
    )
    metrics.update(
        event_response_metrics(
            targets,
            continuous_prediction,
            setpoint,
            start=split,
            end=targets.shape[0],
            horizon_steps=max(1, int(round(args.event_horizon_hours / args.hp_dt_hours))),
            threshold_c=args.setpoint_change_threshold_c,
        )
    )
    metrics.update(
        flexibility_kpi_metrics(
            profile.test,
            test_target,
            test_prediction,
            args,
        )
    )
    return metrics


def factor_effects(
    runs: pd.DataFrame,
    search_space: dict[str, list[Any]],
) -> pd.DataFrame:
    numeric_metrics = [metric for metric in INFLUENCE_METRICS if metric in runs]
    config_means = runs.groupby("config_id", as_index=False)[numeric_metrics].mean(numeric_only=True)
    design = runs.drop_duplicates("config_id")[["config_id", *search_space.keys()]]
    config_means = config_means.merge(design, on="config_id", how="left")
    rows: list[dict[str, Any]] = []
    for factor, levels in search_space.items():
        for metric in numeric_metrics:
            values = pd.to_numeric(config_means[metric], errors="coerce")
            for level in levels:
                selected = config_means[factor].astype(str) == str(level)
                at_level = values[selected].dropna()
                elsewhere = values[~selected].dropna()
                rows.append(
                    {
                        "factor": factor,
                        "level": level,
                        "metric": metric,
                        "configuration_count": int(at_level.size),
                        "mean": float(at_level.mean()) if not at_level.empty else float("nan"),
                        "std": float(at_level.std(ddof=1)) if at_level.size > 1 else float("nan"),
                        "mean_effect_vs_other_levels": (
                            float(at_level.mean() - elsewhere.mean())
                            if not at_level.empty and not elsewhere.empty
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def save_profile_plot(plot: BestPlot, output_dir: Path) -> Path:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Indoor temperature", "Delivered room heat", "HP electric power"),
    )
    names = ((0, "Temperature", "degC"), (1, "Qroom", "W/m2"), (2, "Pel SH", "W/m2"))
    for row, (channel, _name, unit) in enumerate(names, start=1):
        figure.add_trace(
            go.Scatter(x=plot.datetime, y=plot.targets[:, channel], name="Simulated", line={"color": "#1f77b4"}),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=plot.datetime,
                y=plot.continuous_prediction[:, channel],
                name="Local continuous",
                line={"color": "#d62728"},
            ),
            row=row,
            col=1,
        )
        reset_full = np.full((plot.targets.shape[0],), np.nan, dtype=np.float32)
        reset_full[plot.split_index :] = plot.reset_prediction[:, channel]
        figure.add_trace(
            go.Scatter(
                x=plot.datetime,
                y=reset_full,
                name="Local July reset",
                line={"color": "#2ca02c", "dash": "dot"},
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=unit, row=row, col=1)
    figure.add_trace(
        go.Scatter(
            x=plot.datetime,
            y=plot.inputs[:, 0],
            name="Tset",
            line={"color": "#9467bd", "dash": "dash"},
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    split_time = plot.datetime[plot.split_index]
    for row in range(1, 4):
        figure.add_vline(x=split_time, line_dash="dash", line_color="#555", row=row, col=1)
    figure.update_layout(
        title=(
            f"Best local fit, profile {plot.profile_id}, {plot.config_id}<br>"
            f"<sup>train/test split={pd.Timestamp(split_time)}, variant={asdict(plot.variant)}</sup>"
        ),
        height=1050,
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"local_profile_{plot.profile_id}_{plot.config_id}.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def save_summary_plot(runs: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> Path:
    metrics = (
        "continuous_test_composite_nrmse",
        "continuous_test_rmse_c",
        "test_hp_mode_brier",
        "test_event_pel_response_rmse_w_m2",
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=metrics,
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )
    for index, metric in enumerate(metrics):
        row = index // 2 + 1
        col = index % 2 + 1
        subset = effects[effects["metric"] == metric].copy()
        subset["label"] = subset["factor"].astype(str) + "=" + subset["level"].astype(str)
        figure.add_trace(
            go.Bar(
                x=subset["label"],
                y=subset["mean_effect_vs_other_levels"],
                customdata=np.column_stack([subset["mean"], subset["configuration_count"]]),
                hovertemplate=(
                    "%{x}<br>effect vs other levels=%{y:.4g}<br>"
                    "level mean=%{customdata[0]:.4g}<br>n configs=%{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        figure.add_hline(y=0.0, line_color="#555", line_dash="dot", row=row, col=col)
        figure.update_xaxes(tickangle=35, row=row, col=col)
    figure.update_layout(
        title=(
            "Building-local randomized closed-loop HP ablation"
            f"<br><sup>{runs['profile_id'].nunique()} buildings, "
            f"{runs['config_id'].nunique()} configurations; negative error effect is better</sup>"
        ),
        height=1000,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "local_ablation_factor_effects.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def run_ablation(args: argparse.Namespace) -> None:
    profiles = prepare_profiles(load_profiles(args), train_fraction=args.train_fraction)
    scalers = fit_common_scalers(profiles)
    scaled_train_by_id = {
        profile.raw.profile_id: transform_closed_loop_profiles([profile.train], scalers)[0]
        for profile in profiles
    }
    caps = resolve_caps(
        profiles,
        cap_factor=args.hp_cap_factor,
        hp_dt_hours=args.hp_dt_hours,
        cop_cap=args.hp_cop_cap,
    )
    search_space = load_search_space(args.search_space)
    base = LocalVariant(
        state_dim=args.state_dim,
        hp_history_hours=args.hp_history_hours,
        hp_controller_noise_scale=args.hp_controller_noise_scale,
        softopt_weight=args.softopt_weight,
        variogram_weight=args.variogram_weight,
        prob_particles=args.prob_particles,
    )
    variants = (
        full_factorial_design(search_space, base=base)
        if args.full_factorial
        else balanced_random_design(
            search_space,
            num_configurations=args.num_configurations,
            seed=args.seed,
            base=base,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design_rows = [
        {"config_id": f"config_{index:03d}", **asdict(variant)}
        for index, variant in enumerate(variants, start=1)
    ]
    pd.DataFrame(design_rows).to_csv(args.output_dir / "local_ablation_design.csv", index=False)
    (args.output_dir / "local_ablation_manifest.json").write_text(
        json.dumps(
            {
                "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
                "profile_ids": [profile.raw.profile_id for profile in profiles],
                "search_space": search_space,
                "metadata_conditioning": False,
                "system_latent": "fixed_zero",
                "scaler_fit_period": "training_half_only",
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"local_ablation profiles={[profile.raw.profile_id for profile in profiles]} "
        f"configurations={len(variants)} train_fraction={args.train_fraction} "
        "metadata_dim=0 system_latent=fixed_zero"
    )
    print(f"search_space={search_space}")
    print(f"caps pel={caps[0]:.3f}W/m2 qroom={caps[1]:.3f}W/m2 energy={caps[2]:.3f}Wh/m2")

    rows: list[dict[str, Any]] = []
    best_plots: dict[int, BestPlot] = {}
    for config_index, variant in enumerate(variants, start=1):
        config_id = f"config_{config_index:03d}"
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.gradient_clip_norm),
            optax.adam(args.learning_rate),
        )
        train_step = make_train_step(variant, args, optimizer)
        print(f"starting {config_id} {asdict(variant)}")
        for profile_index, profile in enumerate(profiles):
            fit_seed = args.seed + 10_000 * profile_index
            model = build_local_model(
                variant,
                args,
                scalers,
                caps,
                key=jax.random.PRNGKey(fit_seed),
            )
            model, train_stats = train_local_model(
                model,
                scaled_train_by_id[profile.raw.profile_id],
                variant,
                args,
                seed=fit_seed,
                train_step=train_step,
                optimizer=optimizer,
            )
            continuous_prediction, continuous_aux = rollout_physical(
                model,
                profile.local,
                scalers,
                key=jax.random.PRNGKey(fit_seed + 1),
            )
            reset_prediction, _ = rollout_physical(
                model,
                profile.test,
                scalers,
                key=jax.random.PRNGKey(fit_seed + 2),
            )
            metrics = relationship_metrics(
                profile,
                continuous_prediction,
                continuous_aux,
                reset_prediction,
                args,
            )
            metrics.update(
                local_jacobian_metrics(
                    model,
                    profile,
                    scalers,
                    continuous_aux,
                    args,
                    key=jax.random.PRNGKey(fit_seed + 3),
                )
            )
            row = {
                "config_id": config_id,
                "profile_id": profile.raw.profile_id,
                **asdict(variant),
                **train_stats,
                **metrics,
            }
            rows.append(row)
            score = float(metrics["continuous_test_composite_nrmse"])
            if np.isfinite(score) and (
                profile.raw.profile_id not in best_plots
                or score < best_plots[profile.raw.profile_id].score
            ):
                best_plots[profile.raw.profile_id] = BestPlot(
                    score=score,
                    profile_id=profile.raw.profile_id,
                    config_id=config_id,
                    variant=variant,
                    datetime=profile.local.datetime,
                    split_index=profile.split_index,
                    inputs=profile.local.inputs,
                    targets=profile.local.targets,
                    continuous_prediction=continuous_prediction,
                    reset_prediction=reset_prediction,
                )
            print(
                f"finished {config_id} profile={profile.raw.profile_id} "
                f"test_rmse_c={metrics['continuous_test_rmse_c']:.4f} "
                f"q_rmse={metrics['continuous_test_qroom_rmse_w_m2']:.4f} "
                f"pel_rmse={metrics['continuous_test_pel_rmse_w_m2']:.4f} "
                f"composite={score:.4f} max_grad={train_stats['max_grad_norm']:.4g} "
                f"jac_max={metrics['test_jacobian_max']:.4f}"
            )

    runs = pd.DataFrame(rows)
    runs_path = args.output_dir / "local_ablation_runs.csv"
    runs.to_csv(runs_path, index=False)
    effects = factor_effects(runs, search_space)
    effects_path = args.output_dir / "local_ablation_factor_effects.csv"
    effects.to_csv(effects_path, index=False)
    summary_path = save_summary_plot(runs, effects, args.output_dir)
    for plot in best_plots.values():
        path = save_profile_plot(plot, args.output_dir / "best_profile_plots")
        print(f"saved_best_profile_plot={path}")
    print(f"saved_local_ablation_runs={runs_path}")
    print(f"saved_local_ablation_factor_effects={effects_path}")
    print(f"saved_local_ablation_summary={summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit building-specific closed-loop HP models on the first half-year and run a "
            "balanced randomized structural ablation."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator/local_hp_ablation"))
    parser.add_argument("--profile-ids", type=int, nargs="*", default=())
    parser.add_argument("--num-profiles", type=int, default=2)
    parser.add_argument("--profile-selection", choices=("random", "first"), default="random")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--sequence-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--rotate-window-starts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Zero uses all batches.")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--bptt-truncate-steps", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--input-encoder-dim", type=int, default=8)
    parser.add_argument("--input-encoder-hidden-dim", type=int, default=32)
    parser.add_argument("--input-encoder-depth", type=int, default=2)
    parser.add_argument("--prob-particles", type=int, default=4)
    parser.add_argument("--process-noise-init", type=float, default=-4.0)
    parser.add_argument("--hp-controller-state-dim", type=int, default=2)
    parser.add_argument("--hp-persistent-latent-dim", type=int, default=2)
    parser.add_argument("--hp-controller-leak", type=float, default=0.25)
    parser.add_argument("--hp-controller-noise-scale", type=float, default=0.0)
    parser.add_argument("--hp-history-hours", type=float, default=3.0)
    parser.add_argument(
        "--hp-history-timescales-hours",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 3.0, 6.0),
        help=(
            "EWMA timescales used by multiscale history in addition to the immediate lag."
        ),
    )
    parser.add_argument(
        "--hp-history-teacher-forcing-start",
        type=float,
        default=1.0,
        help="Initial observed-history fraction for scheduled feedback.",
    )
    parser.add_argument(
        "--hp-history-teacher-forcing-end",
        type=float,
        default=0.0,
        help="Final observed-history fraction for scheduled feedback.",
    )
    parser.add_argument(
        "--hp-history-teacher-forcing-anneal-epochs",
        type=int,
        default=0,
        help="Annealing duration; zero uses the complete training run.",
    )
    parser.add_argument(
        "--hp-history-loss-warmup-hours",
        type=float,
        default=0.0,
        help="Exclude this initial duration from each window loss while history initializes.",
    )
    parser.add_argument(
        "--hp-temperature-teacher-forcing-start",
        type=float,
        default=1.0,
        help="Initial observed-Tin fraction used by scheduled HP-head feedback.",
    )
    parser.add_argument(
        "--hp-temperature-teacher-forcing-end",
        type=float,
        default=0.0,
        help="Final observed-Tin fraction used by scheduled HP-head feedback.",
    )
    parser.add_argument(
        "--hp-temperature-teacher-forcing-anneal-epochs",
        type=int,
        default=0,
        help="Observed-Tin annealing duration; zero uses the complete training run.",
    )
    parser.add_argument("--hp-concrete-temperature", type=float, default=0.5)
    parser.add_argument("--hp-dt-hours", type=float, default=0.25)
    parser.add_argument("--hp-cop-cap", type=float, default=8.0)
    parser.add_argument("--hp-cap-factor", type=float, default=1.25)
    parser.add_argument("--hp-mode-loss-weight", type=float, default=0.1)
    parser.add_argument("--hp-active-nll-weight", type=float, default=1.0)
    parser.add_argument("--hp-inactive-leakage-weight", type=float, default=0.0)
    parser.add_argument("--heat-on-threshold-w-m2", type=float, default=1e-6)
    parser.add_argument("--contracting-gamma", type=float, default=0.99)
    parser.add_argument("--contracting-state-bound", type=float, default=5.0)
    parser.add_argument("--contracting-temperature-scale", type=float, default=8.0)
    parser.add_argument("--contracting-temperature-delta-max-c", type=float, default=2.0)
    parser.add_argument("--closed-loop-stability-gamma", type=float, default=0.995)
    parser.add_argument("--closed-loop-stability-samples", type=int, default=2)
    parser.add_argument(
        "--closed-loop-stability-aggregation",
        choices=("mean", "max"),
        default="max",
    )
    parser.add_argument(
        "--closed-loop-stability-method",
        choices=("exact_svd", "power_iteration"),
        default="power_iteration",
    )
    parser.add_argument("--closed-loop-stability-power-iterations", type=int, default=3)
    parser.add_argument("--jacobian-diagnostic-samples", type=int, default=32)
    parser.add_argument("--softopt-weight", type=float, default=0.1)
    parser.add_argument("--variogram-weight", type=float, default=0.1)
    parser.add_argument("--variogram-lags", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--setpoint-change-threshold-c", type=float, default=0.05)
    parser.add_argument("--event-horizon-hours", type=float, default=3.0)
    parser.add_argument("--flex-kpi-min-events", type=int, default=20)
    parser.add_argument("--search-space", type=Path, default=None)
    parser.add_argument("--num-configurations", type=int, default=8)
    parser.add_argument("--full-factorial", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    if args.num_profiles < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("--num-profiles, --epochs and --batch-size must be positive")
    if args.sequence_length < 2 or args.stride < 1:
        parser.error("--sequence-length must be at least 2 and --stride must be positive")
    if args.contracting_temperature_delta_max_c <= 0.0:
        parser.error("--contracting-temperature-delta-max-c must be positive")
    if not 0.0 < args.closed_loop_stability_gamma < 1.0:
        parser.error("--closed-loop-stability-gamma must be in (0, 1)")
    if args.closed_loop_stability_samples < 1:
        parser.error("--closed-loop-stability-samples must be positive")
    if args.closed_loop_stability_power_iterations < 1:
        parser.error("--closed-loop-stability-power-iterations must be positive")
    if args.jacobian_diagnostic_samples < 1:
        parser.error("--jacobian-diagnostic-samples must be positive")
    if args.event_horizon_hours <= 0.0:
        parser.error("--event-horizon-hours must be positive")
    if args.flex_kpi_min_events < 1:
        parser.error("--flex-kpi-min-events must be positive")
    if not 0.0 <= args.hp_history_teacher_forcing_start <= 1.0:
        parser.error("--hp-history-teacher-forcing-start must be in [0, 1]")
    if not 0.0 <= args.hp_history_teacher_forcing_end <= 1.0:
        parser.error("--hp-history-teacher-forcing-end must be in [0, 1]")
    if args.hp_history_teacher_forcing_anneal_epochs < 0:
        parser.error("--hp-history-teacher-forcing-anneal-epochs must be non-negative")
    if not 0.0 <= args.hp_temperature_teacher_forcing_start <= 1.0:
        parser.error("--hp-temperature-teacher-forcing-start must be in [0, 1]")
    if not 0.0 <= args.hp_temperature_teacher_forcing_end <= 1.0:
        parser.error("--hp-temperature-teacher-forcing-end must be in [0, 1]")
    if args.hp_temperature_teacher_forcing_anneal_epochs < 0:
        parser.error("--hp-temperature-teacher-forcing-anneal-epochs must be non-negative")
    if args.hp_history_loss_warmup_hours < 0.0:
        parser.error("--hp-history-loss-warmup-hours must be non-negative")
    if any(value <= 0.0 for value in args.hp_history_timescales_hours):
        parser.error("--hp-history-timescales-hours values must be positive")
    if len(set(args.hp_history_timescales_hours)) != len(args.hp_history_timescales_hours):
        parser.error("--hp-history-timescales-hours values must be unique")
    warmup_steps = int(round(args.hp_history_loss_warmup_hours / args.hp_dt_hours))
    if warmup_steps >= args.sequence_length:
        parser.error("--hp-history-loss-warmup-hours must be shorter than --sequence-length")
    return args


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
