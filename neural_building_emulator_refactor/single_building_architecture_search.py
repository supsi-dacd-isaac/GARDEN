"""Fast building-local search over closed-loop HP emulator architectures.

This module intentionally does not use building metadata or a global
metadata-to-parameter network.  It fits each candidate directly to one EGID,
using the setpoint-identification and force-off trajectories when both are
available.  The purpose is model-class selection before another expensive
global probabilistic experiment.

The default full factorial compares 27 combinations:

* HP emission: continuous, hurdle, recurrent hysteretic;
* hydronics: direct, leaky energy buffer, positive causal filter;
* thermal plant: bounded MLP, passive two-node RC, recurrent GRU.

All candidates are trained end-to-end with autonomous rollouts. Held-out
blocks are spread across the year, and no metadata or sampled system latent is
present. Multiple EGIDs can be fitted independently to distinguish a model
class that works generally from one that only works on a selected building.
Independent restarts are also propagated for an uninterrupted full force-off
year to expose long-horizon sensitivity hidden by block-wise evaluation.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    CLOSED_LOOP_TARGET_COLUMNS,
    PROFILE_ID_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    closed_loop_input_columns,
    closed_loop_required_columns,
)
from neural_building_emulator.data import (
    ClosedLoopProfile,
    read_dataset_frame,
    to_closed_loop_profiles,
)
from neural_building_emulator.dataset_adapter import read_entity_profile_ids


HPKind = Literal[
    "continuous",
    "hurdle",
    "hysteretic",
    "asymmetric_hurdle",
    "recovery_hurdle",
]
HydronicKind = Literal[
    "direct",
    "buffer",
    "causal_filter",
    "leaky_emitter",
    "positive_convolution",
]
ThermalKind = Literal[
    "bounded_mlp",
    "bounded_mlp_feedback",
    "bounded_mlp_dual_alpha",
    "bounded_mlp_physical_alpha",
    "passive_memory_positive_q",
    "state_aware_mlp",
    "setpoint_error_ssm",
    "stable_rc",
    "stable_rc_narx",
    "stable_rc_narx_positive_q",
    "passive_rc",
    "gru",
]

HP_KINDS: tuple[HPKind, ...] = ("continuous", "hurdle", "hysteretic")
HP_CHOICES: tuple[HPKind, ...] = (
    *HP_KINDS,
    "asymmetric_hurdle",
    "recovery_hurdle",
)
HYDRONIC_KINDS: tuple[HydronicKind, ...] = ("direct", "buffer", "causal_filter")
HYDRONIC_CHOICES: tuple[HydronicKind, ...] = (
    "direct",
    "buffer",
    "causal_filter",
    "leaky_emitter",
    "positive_convolution",
)
THERMAL_KINDS: tuple[ThermalKind, ...] = ("bounded_mlp", "passive_rc", "gru")
THERMAL_CHOICES: tuple[ThermalKind, ...] = (
    "bounded_mlp",
    "bounded_mlp_feedback",
    "bounded_mlp_dual_alpha",
    "bounded_mlp_physical_alpha",
    "passive_memory_positive_q",
    "state_aware_mlp",
    "setpoint_error_ssm",
    "stable_rc",
    "stable_rc_narx",
    "stable_rc_narx_positive_q",
    "passive_rc",
    "gru",
)


@dataclass(frozen=True)
class Architecture:
    hp: HPKind
    hydronic: HydronicKind
    thermal: ThermalKind

    @property
    def name(self) -> str:
        return f"hp={self.hp}__hyd={self.hydronic}__thermal={self.thermal}"


@dataclass(frozen=True)
class Scale:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Scale":
        mean = np.asarray(np.mean(values, axis=0), dtype=np.float32)
        std = np.asarray(np.std(values, axis=0), dtype=np.float32)
        std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.0).astype(np.float32)
        return cls(mean=mean, std=std)


@dataclass(frozen=True)
class PolicyProfile:
    policy: str
    profile: ClosedLoopProfile


@dataclass(frozen=True)
class Segment:
    policy: str
    start: int
    end: int
    held_out: bool


@dataclass
class FittedCandidate:
    profile_id: int
    architecture: Architecture
    restart: int
    params: dict[str, Any]
    held_out_traces: list[dict[str, Any]]
    selection_score: float
    annual_force_off_prediction: np.ndarray


class RecurrentState(NamedTuple):
    temperature: jax.Array
    hp_mode: jax.Array
    previous_pel: jax.Array
    previous_q: jax.Array
    energy: jax.Array
    hydronic_modes: jax.Array
    mass_temperature: jax.Array
    hidden: jax.Array
    previous_setpoint: jax.Array
    force_debt: jax.Array
    rc_mass_temperatures: jax.Array
    functional_memory: jax.Array
    thermal_q_modes: jax.Array


def _init_mlp(
    key: jax.Array,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    keys = jax.random.split(key, depth + 1)
    dims = (input_dim, *(hidden_dim for _ in range(depth)), output_dim)
    layers: list[tuple[jax.Array, jax.Array]] = []
    for layer_key, fan_in, fan_out in zip(keys, dims[:-1], dims[1:]):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        weight = jax.random.uniform(
            layer_key,
            (fan_out, fan_in),
            minval=-limit,
            maxval=limit,
        )
        bias = jnp.zeros((fan_out,), dtype=jnp.float32)
        layers.append((weight, bias))
    return tuple(layers)


def _mlp(
    layers: tuple[tuple[jax.Array, jax.Array], ...],
    values: jax.Array,
) -> jax.Array:
    hidden = values
    for weight, bias in layers[:-1]:
        hidden = jax.nn.silu(weight @ hidden + bias)
    weight, bias = layers[-1]
    return weight @ hidden + bias


def _init_gru(key: jax.Array, input_dim: int, hidden_dim: int) -> dict[str, jax.Array]:
    input_key, hidden_key = jax.random.split(key)
    input_weight = jax.random.normal(input_key, (3 * hidden_dim, input_dim)) / math.sqrt(input_dim)
    hidden_weight = jax.random.normal(hidden_key, (3 * hidden_dim, hidden_dim)) / math.sqrt(hidden_dim)
    return {
        "input_weight": input_weight,
        "hidden_weight": hidden_weight,
        "bias": jnp.zeros((3 * hidden_dim,), dtype=jnp.float32),
    }


def _gru_step(params: dict[str, jax.Array], values: jax.Array, hidden: jax.Array) -> jax.Array:
    input_terms = params["input_weight"] @ values + params["bias"]
    hidden_terms = params["hidden_weight"] @ hidden
    input_reset, input_update, input_new = jnp.split(input_terms, 3)
    hidden_reset, hidden_update, hidden_new = jnp.split(hidden_terms, 3)
    reset = jax.nn.sigmoid(input_reset + hidden_reset)
    update = jax.nn.sigmoid(input_update + hidden_update)
    candidate = jnp.tanh(input_new + reset * hidden_new)
    return (1.0 - update) * candidate + update * hidden


def _candidate_parameters(
    architecture: Architecture,
    *,
    input_dim: int,
    hidden_dim: int,
    depth: int,
    key: jax.Array,
    hydronic_capacity_min_hours: float = 0.25,
    hydronic_capacity_max_hours: float = 24.0,
    hydronic_capacity_init_hours: float = 2.0,
) -> dict[str, Any]:
    # Preserve the original key stream exactly for legacy candidates. New
    # optional blocks use folded-in keys so adding them cannot silently change
    # an existing architecture's initialization.
    keys = iter(jax.random.split(key, 12))
    # Feedback features are: normalized exogenous inputs, current Tin, thermal
    # gap, previous Pel/Q, previous mode, setpoint shock, and force-off debt.
    feature_dim = input_dim + 8
    # The thermal plant sees only physical disturbances and Qroom. Tset,
    # force-off policy, availability, DHW request, and Pel are deliberately
    # excluded so their effect on Tin must pass through delivered room heat.
    thermal_feature_dim = (
        8
        if architecture.thermal in ("bounded_mlp_feedback", "state_aware_mlp")
        else 6
    )
    functional_feature_dim = 7
    functional_memory_count = 4
    narx_input_dim = functional_feature_dim * (functional_memory_count + 1)
    hp_net = _init_mlp(next(keys), feature_dim, hidden_dim, 2, depth)
    hydronic_net = _init_mlp(next(keys), feature_dim + 2, hidden_dim, 1, depth)
    thermal_net = _init_mlp(next(keys), thermal_feature_dim, hidden_dim, 1, depth)
    narx_net = _init_mlp(
        jax.random.fold_in(key, 91_733),
        narx_input_dim,
        hidden_dim,
        1,
        depth,
    )
    narx_last_weight, narx_last_bias = narx_net[-1]
    narx_net = (
        *narx_net[:-1],
        (0.01 * narx_last_weight, narx_last_bias),
    )
    thermal_q_gain_net = _init_mlp(
        jax.random.fold_in(key, 91_734),
        narx_input_dim,
        hidden_dim,
        1,
        depth,
    )
    q_gain_last_weight, _ = thermal_q_gain_net[-1]
    thermal_q_gain_net = (
        *thermal_q_gain_net[:-1],
        (
            jnp.zeros_like(q_gain_last_weight),
            jnp.asarray([-2.0], dtype=jnp.float32),
        ),
    )
    capacity_fraction = (
        hydronic_capacity_init_hours - hydronic_capacity_min_hours
    ) / (hydronic_capacity_max_hours - hydronic_capacity_min_hours)
    capacity_fraction = min(max(capacity_fraction, 1e-5), 1.0 - 1e-5)
    capacity_raw = math.log(capacity_fraction / (1.0 - capacity_fraction))
    params: dict[str, Any] = {
        "hp_net": hp_net,
        "hydronic_net": hydronic_net,
        "thermal_net": thermal_net,
        "hp_recurrence": jnp.asarray(0.0, dtype=jnp.float32),
        "hp_gap_slope_raw": jnp.asarray(0.0, dtype=jnp.float32),
        "hp_debt_slope_raw": jnp.asarray(0.0, dtype=jnp.float32),
        "hp_active_floor_raw": jnp.asarray(-2.0, dtype=jnp.float32),
        "hp_deadband_raw": jnp.asarray(-1.5, dtype=jnp.float32),
        "cop_raw": jnp.asarray(0.0, dtype=jnp.float32),
        "buffer_loss_raw": jnp.asarray(-3.0, dtype=jnp.float32),
        "emitter_tau_raw": jnp.asarray(-1.0, dtype=jnp.float32),
        "energy_capacity_raw": jnp.asarray(capacity_raw, dtype=jnp.float32),
        "hydronic_weight_raw": jnp.zeros((3,), dtype=jnp.float32),
        "hydronic_efficiency_raw": jnp.asarray(1.4, dtype=jnp.float32),
        "rc_raw": jnp.asarray([-2.5, -2.5, -3.5, -4.0, -4.0, -4.0, -4.0]),
        "setpoint_error_raw": jnp.asarray(
            [-3.5, -3.0, -2.5, -2.8, -4.0, -3.0, -3.0, -3.0],
            dtype=jnp.float32,
        ),
        # Air coupling budget, five coupling allocations, and four positive
        # forcing gains. The air-to-state coefficients are normalized below
        # a shared budget in the rollout, making the passive recurrence stable.
        "stable_rc_raw": jnp.asarray(
            [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0, -3.0, -3.0],
            dtype=jnp.float32,
        ),
        "narx_net": narx_net,
        "thermal_q_gain_net": thermal_q_gain_net,
        "narx_residual_scale_raw": jnp.asarray(-2.0, dtype=jnp.float32),
        "thermal_q_gain_raw": jnp.asarray(-2.0, dtype=jnp.float32),
        "thermal_q_weight_raw": jnp.zeros((4,), dtype=jnp.float32),
        "thermal_out": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "thermal_out_bias": jnp.asarray(0.0, dtype=jnp.float32),
        "thermal_alpha_raw": jnp.asarray(-2.0, dtype=jnp.float32),
        "thermal_alpha_free_raw": jnp.asarray(-2.0, dtype=jnp.float32),
        "thermal_alpha_heating_raw": jnp.asarray(-2.0, dtype=jnp.float32),
        # Positive conductance/capacity rates [1/h]. The base initialization
        # reproduces alpha ~= 0.03 at 15-minute resolution; ventilation and
        # stack-effect corrections start close to zero.
        "thermal_rate_base_raw": jnp.asarray(-1.1, dtype=jnp.float32),
        "thermal_rate_vent_raw": jnp.asarray(-5.0, dtype=jnp.float32),
        "thermal_rate_stack_raw": jnp.asarray(-5.0, dtype=jnp.float32),
    }
    if architecture.thermal == "gru":
        params["gru"] = _init_gru(next(keys), thermal_feature_dim, hidden_dim)
        params["thermal_out"] = jax.random.normal(next(keys), (hidden_dim,)) / math.sqrt(hidden_dim)
    return params


def _input_indices() -> dict[str, int]:
    names = closed_loop_input_columns(include_hp_force_off=True)
    return {name: index for index, name in enumerate(names)}


INPUT_INDICES = _input_indices()
SETPOINT_INDEX = 0
TOUT_INDEX = INPUT_INDICES["Environment, Site Outdoor Air Drybulb Temperature"]
SOLAR_INDEX = INPUT_INDICES["Environment, Site Global Horizontal Solar Radiation Rate per Area"]
VENT_INDEX = INPUT_INDICES["FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate"]
INTERNAL_INDEX = INPUT_INDICES["input_non_people_internal_gain_net_w_per_m2"]
OCCUPANCY_INDEX = INPUT_INDICES["input_occupants_present_per_m2"]
FORCE_OFF_INDEX = INPUT_INDICES["hp_force_off_requested"]
AVAILABILITY_INDEX = INPUT_INDICES[SPACE_HEATING_AVAILABILITY_COLUMN]
HOUR_SIN_INDEX = INPUT_INDICES["hour_sin"]
HOUR_COS_INDEX = INPUT_INDICES["hour_cos"]
DAY_OF_YEAR_SIN_INDEX = INPUT_INDICES["day_of_year_sin"]
DAY_OF_YEAR_COS_INDEX = INPUT_INDICES["day_of_year_cos"]


def _bounded_mlp_alpha(
    params: dict[str, Any],
    architecture: Architecture,
    row: jax.Array,
    temperature: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    dt_hours: float,
) -> jax.Array:
    """Return the bounded relaxation coefficient for a legacy MLP variant."""

    if architecture.thermal == "bounded_mlp_dual_alpha":
        alpha_free = 0.25 * jax.nn.sigmoid(
            params["thermal_alpha_free_raw"]
        )
        alpha_heating = 0.25 * jax.nn.sigmoid(
            params["thermal_alpha_heating_raw"]
        )
        availability = jnp.clip(row[AVAILABILITY_INDEX], 0.0, 1.0)
        return (1.0 - availability) * alpha_free + availability * alpha_heating

    if architecture.thermal == "bounded_mlp_physical_alpha":
        ventilation_reference = jnp.maximum(
            input_mean[VENT_INDEX] + 2.0 * input_std[VENT_INDEX],
            1e-4,
        )
        ventilation_activity = jnp.clip(
            jax.nn.relu(row[VENT_INDEX]) / ventilation_reference,
            0.0,
            2.0,
        )
        stack_activity = jnp.sqrt(
            jnp.clip(
                jnp.abs(temperature - row[TOUT_INDEX]) / 10.0,
                0.0,
                4.0,
            )
            + 1e-4
        )
        # rate = H/C. Positive bounded components preserve 0 < alpha < 1
        # while allowing ventilation and stack effect to shorten tau.
        rate_per_hour = (
            0.5 * jax.nn.sigmoid(params["thermal_rate_base_raw"])
            + 0.5
            * jax.nn.sigmoid(params["thermal_rate_vent_raw"])
            * ventilation_activity
            + 0.25
            * jax.nn.sigmoid(params["thermal_rate_stack_raw"])
            * stack_activity
        )
        return 1.0 - jnp.exp(-dt_hours * rate_per_hour)

    return 0.25 * jax.nn.sigmoid(params["thermal_alpha_raw"])


def _bounded_mlp_temperature_step(
    params: dict[str, Any],
    architecture: Architecture,
    row: jax.Array,
    temperature: jax.Array,
    qroom: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    q_cap: float,
    temperature_delta_cap: float,
    dt_hours: float,
) -> jax.Array:
    """Apply one legacy bounded-equilibrium thermal transition."""

    normalized = (row - input_mean) / input_std
    thermal_features = jnp.asarray(
        [
            normalized[TOUT_INDEX],
            normalized[SOLAR_INDEX],
            normalized[VENT_INDEX],
            normalized[INTERNAL_INDEX],
            normalized[OCCUPANCY_INDEX],
            qroom / q_cap,
        ]
    )
    if architecture.thermal == "bounded_mlp_feedback":
        feedback_temperature = jax.lax.stop_gradient(temperature)
        temperature_scale = jnp.maximum(input_std[TOUT_INDEX], 1e-6)
        thermal_features = jnp.concatenate(
            (
                thermal_features,
                jnp.asarray(
                    [
                        (feedback_temperature - input_mean[TOUT_INDEX])
                        / temperature_scale,
                        (row[TOUT_INDEX] - feedback_temperature)
                        / temperature_scale,
                    ]
                ),
            )
        )
    equilibrium = row[TOUT_INDEX] + 30.0 * jnp.tanh(
        _mlp(params["thermal_net"], thermal_features)[0]
    )
    alpha = _bounded_mlp_alpha(
        params,
        architecture,
        row,
        temperature,
        input_mean=input_mean,
        input_std=input_std,
        dt_hours=dt_hours,
    )
    raw_delta = alpha * (equilibrium - temperature)
    delta_t = temperature_delta_cap * jnp.tanh(
        raw_delta / temperature_delta_cap
    )
    return jnp.clip(temperature + delta_t, 5.0, 40.0)


def _positive_thermal_q_forcing(
    qroom: jax.Array,
    previous_modes: jax.Array,
    *,
    gain_raw: jax.Array,
    weight_raw: jax.Array,
    dt_hours: float,
) -> tuple[jax.Array, jax.Array]:
    """Return a sign-preserving fast/slow Qroom contribution to delta Tin."""

    time_constants_hours = jnp.asarray([1.0, 4.0, 16.0])
    retention = jnp.exp(-dt_hours / time_constants_hours)
    modes = retention * previous_modes + (1.0 - retention) * qroom
    components = jnp.concatenate((jnp.ravel(qroom), modes))
    weights = jax.nn.softmax(weight_raw)
    gain_c_per_w_m2 = 0.02 * jax.nn.sigmoid(gain_raw)
    forcing = gain_c_per_w_m2 * jnp.dot(weights, components)
    return forcing, modes


def _passive_memory_terms(
    params: dict[str, Any],
    row: jax.Array,
    temperature: jax.Array,
    functional_memory: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
    dt_hours: float,
    temperature_delta_cap: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return bounded passive drift, positive Q gain, and updated memory."""

    normalized = (row - input_mean) / input_std
    functional_features = jnp.asarray(
        [
            (temperature - target_t_mean) / target_t_std,
            (row[TOUT_INDEX] - temperature) / target_t_std,
            normalized[SOLAR_INDEX],
            normalized[VENT_INDEX],
            normalized[INTERNAL_INDEX],
            normalized[OCCUPANCY_INDEX],
            0.0,
        ]
    )
    time_constants = jnp.asarray([0.5, 2.0, 8.0, 24.0])
    retention = jnp.exp(-dt_hours / time_constants)[:, None]
    next_memory = (
        retention * functional_memory
        + (1.0 - retention) * functional_features[None, :]
    )
    context = jnp.concatenate((functional_features, next_memory.reshape(-1)))
    passive_scale = 0.5 * jax.nn.sigmoid(
        params["narx_residual_scale_raw"]
    )
    passive_delta = passive_scale * jnp.tanh(
        _mlp(params["narx_net"], context)[0]
    )
    q_gain = 0.02 * jax.nn.sigmoid(
        _mlp(params["thermal_q_gain_net"], context)[0]
    )
    return passive_delta, q_gain, next_memory


def _passive_memory_positive_q_step(
    params: dict[str, Any],
    row: jax.Array,
    temperature: jax.Array,
    qroom: jax.Array,
    functional_memory: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
    dt_hours: float,
    temperature_delta_cap: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Bounded passive state feedback plus an explicitly positive Q branch."""

    passive_delta, q_gain, next_memory = _passive_memory_terms(
        params,
        row,
        temperature,
        functional_memory,
        input_mean=input_mean,
        input_std=input_std,
        target_t_mean=target_t_mean,
        target_t_std=target_t_std,
        dt_hours=dt_hours,
        temperature_delta_cap=temperature_delta_cap,
    )
    raw_delta = passive_delta + q_gain * qroom
    delta_t = temperature_delta_cap * jnp.tanh(
        raw_delta / temperature_delta_cap
    )
    next_temperature = jnp.clip(temperature + delta_t, 5.0, 40.0)
    return next_temperature, next_memory, q_gain


def _observed_q_thermal_rollout(
    params: dict[str, Any],
    architecture: Architecture,
    inputs: jax.Array,
    observed_qroom: jax.Array,
    initial_temperature: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
    q_cap: float,
    dt_hours: float,
    temperature_delta_cap: float,
) -> jax.Array:
    """Autonomously propagate the thermal block using observed delivered heat."""

    supported = (
        "bounded_mlp",
        "bounded_mlp_feedback",
        "bounded_mlp_dual_alpha",
        "bounded_mlp_physical_alpha",
        "passive_memory_positive_q",
    )
    if architecture.thermal not in supported:
        raise ValueError(
            "The observed-Q auxiliary rollout supports bounded_mlp variants "
            "and passive_memory_positive_q"
        )
    initial_t = jnp.ravel(initial_temperature)[0]
    first_normalized = (inputs[0] - input_mean) / input_std
    initial_features = jnp.asarray(
        [
            (initial_t - target_t_mean) / target_t_std,
            (inputs[0, TOUT_INDEX] - initial_t) / target_t_std,
            first_normalized[SOLAR_INDEX],
            first_normalized[VENT_INDEX],
            first_normalized[INTERNAL_INDEX],
            first_normalized[OCCUPANCY_INDEX],
            0.0,
        ]
    )
    initial_memory = jnp.tile(initial_features[None, :], (4, 1))

    def step(carry, values):
        temperature, memory = carry
        row, qroom = values
        if architecture.thermal == "passive_memory_positive_q":
            next_temperature, next_memory, _ = (
                _passive_memory_positive_q_step(
                    params,
                    row,
                    temperature,
                    qroom,
                    memory,
                    input_mean=input_mean,
                    input_std=input_std,
                    target_t_mean=target_t_mean,
                    target_t_std=target_t_std,
                    dt_hours=dt_hours,
                    temperature_delta_cap=temperature_delta_cap,
                )
            )
        else:
            next_temperature = _bounded_mlp_temperature_step(
                params,
                architecture,
                row,
                temperature,
                qroom,
                input_mean=input_mean,
                input_std=input_std,
                q_cap=q_cap,
                temperature_delta_cap=temperature_delta_cap,
                dt_hours=dt_hours,
            )
            next_memory = memory
        return (next_temperature, next_memory), next_temperature

    _, temperatures = jax.lax.scan(
        step,
        (initial_t, initial_memory),
        (inputs, observed_qroom),
    )
    return temperatures


def _rollout(
    params: dict[str, Any],
    architecture: Architecture,
    inputs: jax.Array,
    initial_temperature: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_mean: jax.Array,
    target_std: jax.Array,
    pel_cap: float,
    q_cap: float,
    energy_cap: float,
    hidden_dim: int,
    dt_hours: float,
    temperature_delta_cap: float,
    hp_recovery_emergency_threshold_c: float = 1.0,
    hydronic_capacity_min_hours: float = 0.25,
    hydronic_capacity_max_hours: float = 24.0,
    teacher_feedback: jax.Array | None = None,
    teacher_force_temperature: bool = False,
    teacher_force_power_history: bool = False,
    controller_calendar_mode: str = "all",
) -> tuple[jax.Array, jax.Array]:
    initial_t = jnp.ravel(initial_temperature)[0]
    target_t_mean = target_mean[0]
    target_t_std = target_std[0]
    first_normalized = (inputs[0] - input_mean) / input_std
    initial_functional_features = jnp.asarray(
        [
            (initial_t - target_t_mean) / target_t_std,
            (inputs[0, TOUT_INDEX] - initial_t) / target_t_std,
            first_normalized[SOLAR_INDEX],
            first_normalized[VENT_INDEX],
            first_normalized[INTERNAL_INDEX],
            first_normalized[OCCUPANCY_INDEX],
            0.0,
        ]
    )
    state = RecurrentState(
        temperature=initial_t,
        hp_mode=jnp.asarray(0.0),
        previous_pel=jnp.asarray(0.0),
        previous_q=jnp.asarray(0.0),
        energy=jnp.asarray(0.0),
        hydronic_modes=jnp.zeros((3,)),
        mass_temperature=initial_t,
        hidden=jnp.zeros((hidden_dim,)),
        previous_setpoint=inputs[0, SETPOINT_INDEX],
        force_debt=jnp.asarray(0.0),
        rc_mass_temperatures=jnp.full((3,), initial_t),
        functional_memory=jnp.tile(initial_functional_features[None, :], (4, 1)),
        thermal_q_modes=jnp.zeros((3,)),
    )

    if teacher_feedback is None:
        teacher_feedback = jnp.zeros((inputs.shape[0], 4), dtype=inputs.dtype)

    def step(current: RecurrentState, values):
        row, observed_feedback = values
        controller_temperature = (
            observed_feedback[0]
            if teacher_force_temperature
            else current.temperature
        )
        controller_previous_pel = (
            observed_feedback[1]
            if teacher_force_power_history
            else current.previous_pel
        )
        controller_previous_q = (
            observed_feedback[2]
            if teacher_force_power_history
            else current.previous_q
        )
        controller_previous_mode = (
            observed_feedback[3]
            if teacher_force_power_history
            else current.hp_mode
        )
        normalized = (row - input_mean) / input_std
        controller_normalized = normalized
        if controller_calendar_mode in ("hour_only", "none"):
            controller_normalized = controller_normalized.at[
                DAY_OF_YEAR_SIN_INDEX
            ].set(0.0)
            controller_normalized = controller_normalized.at[
                DAY_OF_YEAR_COS_INDEX
            ].set(0.0)
        if controller_calendar_mode == "none":
            controller_normalized = controller_normalized.at[
                HOUR_SIN_INDEX
            ].set(0.0)
            controller_normalized = controller_normalized.at[
                HOUR_COS_INDEX
            ].set(0.0)
        setpoint = row[SETPOINT_INDEX]
        gap = setpoint - controller_temperature
        force_off = jnp.clip(row[FORCE_OFF_INDEX], 0.0, 1.0)
        available = jnp.clip(row[AVAILABILITY_INDEX], 0.0, 1.0)
        shock = setpoint - current.previous_setpoint
        if architecture.hp == "recovery_hurdle":
            deadband = 0.5 * jax.nn.sigmoid(params["hp_deadband_raw"])
            heating_deficit_c = jax.nn.relu(gap - deadband)
            debt_retention = jnp.exp(-dt_hours / 6.0)
            # Degree-hours of heating that an external force-off intervention
            # blocked while space heating was otherwise available. Seasonal
            # shutdown therefore cannot wind up the controller state.
            debt = jnp.clip(
                debt_retention * current.force_debt
                + dt_hours * available * force_off * heating_deficit_c,
                0.0,
                24.0 * target_t_std,
            )
        else:
            # Preserve the legacy controller state and its fitted behavior.
            debt = 0.98 * current.force_debt + force_off * jax.nn.relu(gap)
        feedback = jnp.asarray(
            [
                (controller_temperature - target_t_mean) / target_t_std,
                gap / target_t_std,
                controller_previous_pel / pel_cap,
                controller_previous_q / q_cap,
                controller_previous_mode,
                shock / target_t_std,
                debt / target_t_std,
                force_off,
            ]
        )
        features = jnp.concatenate((controller_normalized, feedback))

        if architecture.hp in ("asymmetric_hurdle", "recovery_hurdle"):
            # Weather, prior mode/power, setpoint shocks, and force-off debt
            # may move the baseline demand. Current Tin, the signed gap, and
            # the absolute setpoint are removed from that baseline so their
            # effect cannot cancel the explicit one-sided thermostat term.
            hp_context = features.at[SETPOINT_INDEX].set(0.0)
            feedback_start = normalized.shape[0]
            hp_context = hp_context.at[feedback_start].set(0.0)
            hp_context = hp_context.at[feedback_start + 1].set(0.0)
            hp_raw = _mlp(params["hp_net"], hp_context)
        else:
            hp_raw = _mlp(params["hp_net"], features)
        base_active_fraction = jax.nn.sigmoid(hp_raw[1])
        active_power = pel_cap * base_active_fraction
        if architecture.hp == "continuous":
            mode = jnp.asarray(1.0)
        elif architecture.hp == "hurdle":
            mode = jax.nn.sigmoid(hp_raw[0])
        elif architecture.hp == "asymmetric_hurdle":
            deadband = 0.5 * jax.nn.sigmoid(params["hp_deadband_raw"])
            heating_deficit = jax.nn.relu(gap - deadband) / target_t_std
            gap_slope = 0.5 + jax.nn.softplus(params["hp_gap_slope_raw"])
            mode = jax.nn.sigmoid(hp_raw[0] + gap_slope * heating_deficit)
        elif architecture.hp == "recovery_hurdle":
            deadband = 0.5 * jax.nn.sigmoid(params["hp_deadband_raw"])
            heating_deficit = jax.nn.relu(
                gap - deadband - hp_recovery_emergency_threshold_c
            ) / target_t_std
            normalized_debt = debt / (6.0 * target_t_std)

            mode_gap_slope = 0.5 + jax.nn.softplus(
                params["hp_gap_slope_raw"]
            )
            mode_debt_slope = 0.1 + jax.nn.softplus(
                params["hp_debt_slope_raw"]
            )
            mode_recovery = (
                mode_gap_slope * heating_deficit
                + mode_debt_slope * normalized_debt
            )
            base_mode = jax.nn.sigmoid(hp_raw[0])
            # This saturating union cannot be cancelled by an arbitrarily
            # negative learned baseline: large deficit/debt implies mode -> 1.
            mode = 1.0 - (1.0 - base_mode) * jnp.exp(-mode_recovery)

            # Thermostat error changes duty cycle, not compressor modulation.
            # A learned strictly positive floor prevents an "active" mode
            # from secretly carrying zero power without forcing rated power.
            active_floor_fraction = 0.01 + 0.49 * jax.nn.sigmoid(
                params["hp_active_floor_raw"]
            )
            active_fraction = active_floor_fraction + (
                1.0 - active_floor_fraction
            ) * base_active_fraction
            active_power = pel_cap * active_fraction
        else:
            recurrence = 4.0 * jnp.tanh(params["hp_recurrence"])
            requested = jax.nn.sigmoid(
                hp_raw[0] + recurrence * (controller_previous_mode - 0.5)
            )
            # A leaky mode probability gives differentiable hysteresis and
            # non-geometric run lengths without stochastic training tricks.
            mode = 0.85 * controller_previous_mode + 0.15 * requested
        gate = available * (1.0 - force_off)
        pel = gate * mode * active_power

        cop = 1.0 + 7.0 * jax.nn.sigmoid(params["cop_raw"])
        if architecture.hydronic == "leaky_emitter":
            capacity_hours = hydronic_capacity_min_hours + (
                hydronic_capacity_max_hours - hydronic_capacity_min_hours
            ) * jax.nn.sigmoid(params["energy_capacity_raw"])
            effective_energy_cap = q_cap * capacity_hours
        else:
            effective_energy_cap = energy_cap
        hydronic_features = jnp.concatenate(
            (
                features,
                jnp.asarray(
                    [pel / pel_cap, current.energy / effective_energy_cap]
                ),
            )
        )
        q_request = q_cap * jax.nn.sigmoid(_mlp(params["hydronic_net"], hydronic_features)[0])
        if architecture.hydronic == "direct":
            # Instantaneous delivery: unlike the buffer/filter candidates this
            # has no hydronic memory, but it still cannot create heat without
            # contemporaneous HP production.
            qroom = jnp.minimum(q_request, cop * pel)
            energy = current.energy
            hydronic_modes = current.hydronic_modes
        elif architecture.hydronic == "buffer":
            loss = 0.25 * jax.nn.sigmoid(params["buffer_loss_raw"])
            available_heat = current.energy / dt_hours + cop * pel
            qroom = jnp.minimum(q_request, jax.nn.relu(available_heat))
            energy = jnp.clip(
                (1.0 - loss) * current.energy + dt_hours * (cop * pel - qroom),
                0.0,
                effective_energy_cap,
            )
            hydronic_modes = current.hydronic_modes
        elif architecture.hydronic == "leaky_emitter":
            # A finite emitter conductance prevents a charged buffer from
            # supplying Qrequest unchanged until it is almost empty. With the
            # HP off, both stored energy and room heat decay smoothly while
            # retaining exact non-negative energy accounting.
            charged_energy = jnp.clip(
                current.energy + dt_hours * cop * pel,
                0.0,
                effective_energy_cap,
            )
            emitter_tau_h = 0.25 + 7.75 * jax.nn.sigmoid(params["emitter_tau_raw"])
            release_fraction = 1.0 - jnp.exp(-dt_hours / emitter_tau_h)
            emitter_power = release_fraction * charged_energy / dt_hours
            qroom = jnp.minimum(q_request, jax.nn.relu(emitter_power))
            standing_loss_rate = 0.05 * jax.nn.sigmoid(params["buffer_loss_raw"])
            energy = jnp.clip(
                (charged_energy - dt_hours * qroom)
                * jnp.exp(-dt_hours * standing_loss_rate),
                0.0,
                effective_energy_cap,
            )
            hydronic_modes = current.hydronic_modes
        elif architecture.hydronic == "positive_convolution":
            # A normalized positive impulse response maps generated thermal
            # power to room delivery. It retains heat after Pel disappears,
            # cannot reverse the sign of the response, and has DC gain <= 1.
            time_constants = jnp.asarray([0.5, 2.0, 8.0])
            retention = jnp.exp(-dt_hours / time_constants)
            hydronic_modes = (
                retention * current.hydronic_modes
                + (1.0 - retention) * cop * pel
            )
            filtered_heat = jnp.dot(
                jax.nn.softmax(params["hydronic_weight_raw"]),
                hydronic_modes,
            )
            delivery_efficiency = jax.nn.sigmoid(
                params["hydronic_efficiency_raw"]
            )
            qroom = jnp.clip(
                delivery_efficiency * filtered_heat,
                0.0,
                q_cap,
            )
            energy = current.energy
        else:
            time_constants = jnp.asarray([0.5, 2.0, 8.0])
            retention = jnp.exp(-dt_hours / time_constants)
            hydronic_modes = retention * current.hydronic_modes + (1.0 - retention) * cop * pel
            filtered_heat = jnp.dot(jax.nn.softmax(params["hydronic_weight_raw"]), hydronic_modes)
            qroom = jnp.minimum(q_request, jax.nn.relu(filtered_heat))
            energy = current.energy

        thermal_features = jnp.asarray(
            [
                normalized[TOUT_INDEX],
                normalized[SOLAR_INDEX],
                normalized[VENT_INDEX],
                normalized[INTERNAL_INDEX],
                normalized[INPUT_INDICES["input_occupants_present_per_m2"]],
                qroom / q_cap,
            ]
        )
        rc_mass_temperatures = current.rc_mass_temperatures
        functional_memory = current.functional_memory
        thermal_q_modes = current.thermal_q_modes
        if architecture.thermal in (
            "stable_rc",
            "stable_rc_narx",
            "stable_rc_narx_positive_q",
        ):
            raw = params["stable_rc_raw"]
            coupling_budget = 0.12 * jax.nn.sigmoid(raw[0])
            coupling_fraction = jax.nn.softmax(raw[1:6])
            ventilation_reference = jnp.maximum(
                input_mean[VENT_INDEX] + 2.0 * input_std[VENT_INDEX],
                1e-4,
            )
            ventilation_activity = jnp.clip(
                jax.nn.relu(row[VENT_INDEX]) / ventilation_reference,
                0.0,
                1.0,
            )
            outside_rate = coupling_budget * (
                coupling_fraction[0]
                + coupling_fraction[4] * ventilation_activity
            )
            mass_rates = coupling_budget * coupling_fraction[1:4]
            q_gain = 0.02 * jax.nn.sigmoid(raw[6])
            solar_gain = 0.0005 * jax.nn.sigmoid(raw[7])
            internal_gain = 0.02 * jax.nn.sigmoid(raw[8])
            occupancy_gain = 1.0 * jax.nn.sigmoid(raw[9])
            if architecture.thermal == "stable_rc_narx_positive_q":
                q_forcing, thermal_q_modes = _positive_thermal_q_forcing(
                    qroom,
                    current.thermal_q_modes,
                    gain_raw=params["thermal_q_gain_raw"],
                    weight_raw=params["thermal_q_weight_raw"],
                    dt_hours=dt_hours,
                )
            else:
                q_forcing = q_gain * qroom
            physical_delta = (
                outside_rate * (row[TOUT_INDEX] - current.temperature)
                + jnp.dot(
                    mass_rates,
                    rc_mass_temperatures - current.temperature,
                )
                + q_forcing
                + solar_gain * jax.nn.relu(row[SOLAR_INDEX])
                + internal_gain * jax.nn.relu(row[INTERNAL_INDEX])
                + occupancy_gain * jax.nn.relu(row[OCCUPANCY_INDEX])
            )

            if architecture.thermal in (
                "stable_rc_narx",
                "stable_rc_narx_positive_q",
            ):
                # In the positive-Q variant, neither the instantaneous
                # residual nor its memories receive Qroom. Delivered heat can
                # affect Tin only through the explicitly nonnegative branch
                # above and through the resulting recursive temperature state.
                residual_q_feature = (
                    jnp.asarray(0.0)
                    if architecture.thermal == "stable_rc_narx_positive_q"
                    else qroom / q_cap
                )
                functional_features = jnp.asarray(
                    [
                        (current.temperature - target_t_mean) / target_t_std,
                        (row[TOUT_INDEX] - current.temperature) / target_t_std,
                        normalized[SOLAR_INDEX],
                        normalized[VENT_INDEX],
                        normalized[INTERNAL_INDEX],
                        normalized[OCCUPANCY_INDEX],
                        residual_q_feature,
                    ]
                )
                functional_time_constants = jnp.asarray([0.5, 2.0, 8.0, 24.0])
                functional_retention = jnp.exp(
                    -dt_hours / functional_time_constants
                )[:, None]
                functional_memory = (
                    functional_retention * current.functional_memory
                    + (1.0 - functional_retention)
                    * functional_features[None, :]
                )
                narx_features = jnp.concatenate(
                    (functional_features, functional_memory.reshape(-1))
                )
                residual_scale = 0.5 * jax.nn.sigmoid(
                    params["narx_residual_scale_raw"]
                )
                physical_delta = physical_delta + residual_scale * jnp.tanh(
                    _mlp(params["narx_net"], narx_features)[0]
                )

            delta_t = temperature_delta_cap * jnp.tanh(
                physical_delta / temperature_delta_cap
            )
            temperature = jnp.clip(current.temperature + delta_t, 5.0, 40.0)
            mass_time_constants = jnp.asarray([2.0, 12.0, 48.0])
            mass_retention = jnp.exp(-dt_hours / mass_time_constants)
            rc_mass_temperatures = jnp.clip(
                mass_retention * rc_mass_temperatures
                + (1.0 - mass_retention) * current.temperature,
                5.0,
                40.0,
            )
            mass_temperature = current.mass_temperature
            hidden = current.hidden
        elif architecture.thermal == "passive_memory_positive_q":
            temperature, functional_memory, _ = (
                _passive_memory_positive_q_step(
                    params,
                    row,
                    current.temperature,
                    qroom,
                    current.functional_memory,
                    input_mean=input_mean,
                    input_std=input_std,
                    target_t_mean=target_t_mean,
                    target_t_std=target_t_std,
                    dt_hours=dt_hours,
                    temperature_delta_cap=temperature_delta_cap,
                )
            )
            mass_temperature = current.mass_temperature
            hidden = current.hidden
        elif architecture.thermal == "state_aware_mlp":
            state_aware_features = jnp.concatenate(
                (
                    thermal_features,
                    jnp.asarray(
                        [
                            (current.temperature - target_t_mean) / target_t_std,
                            (row[TOUT_INDEX] - current.temperature) / target_t_std,
                        ]
                    ),
                )
            )
            # Predict a displacement around the current state. A zero neural
            # output therefore means exactly constant Tin, while Tin and the
            # envelope gradient remain available for state-dependent cooling.
            equilibrium = current.temperature + 30.0 * jnp.tanh(
                _mlp(params["thermal_net"], state_aware_features)[0]
            )
            alpha = 0.25 * jax.nn.sigmoid(params["thermal_alpha_raw"])
            raw_delta = alpha * (equilibrium - current.temperature)
            delta_t = temperature_delta_cap * jnp.tanh(
                raw_delta / temperature_delta_cap
            )
            temperature = jnp.clip(current.temperature + delta_t, 5.0, 40.0)
            mass_temperature = current.mass_temperature
            hidden = current.hidden
        elif architecture.thermal == "setpoint_error_ssm":
            raw = params["setpoint_error_raw"]
            outdoor_rate = 0.02 * jax.nn.sigmoid(raw[0])
            mass_rate = 0.02 * jax.nn.sigmoid(raw[1])
            mass_update_rate = 0.01 * jax.nn.sigmoid(raw[2])
            q_gain = 0.01 * jax.nn.sigmoid(raw[3])
            solar_gain = 0.001 * jax.nn.sigmoid(raw[4])
            internal_gain = 0.02 * jax.nn.sigmoid(raw[5])
            occupancy_gain = 2.0 * jax.nn.sigmoid(raw[6])
            ventilation_gain = 1.0 * jax.nn.sigmoid(raw[7])

            # Change coordinates at every step. A setpoint jump changes these
            # errors but cancels when Tin is reconstructed, so Tset cannot
            # create an instantaneous physical temperature jump.
            air_error = current.temperature - setpoint
            mass_error = current.mass_temperature - setpoint
            outdoor_error = row[TOUT_INDEX] - setpoint
            envelope_gradient = outdoor_error - air_error
            air_error_next = (
                air_error
                + outdoor_rate * envelope_gradient
                + mass_rate * (mass_error - air_error)
                + q_gain * qroom
                + solar_gain * row[SOLAR_INDEX]
                + internal_gain * row[INTERNAL_INDEX]
                + occupancy_gain
                * row[INPUT_INDICES["input_occupants_present_per_m2"]]
                + ventilation_gain
                * jax.nn.relu(row[VENT_INDEX])
                * envelope_gradient
            )
            proposed_temperature = setpoint + air_error_next
            raw_delta = proposed_temperature - current.temperature
            delta_t = temperature_delta_cap * jnp.tanh(
                raw_delta / temperature_delta_cap
            )
            temperature = jnp.clip(current.temperature + delta_t, 5.0, 40.0)
            mass_error_next = mass_error + mass_update_rate * (
                air_error - mass_error
            )
            mass_temperature = jnp.clip(
                setpoint + mass_error_next,
                5.0,
                40.0,
            )
            hidden = current.hidden
        elif architecture.thermal in (
            "bounded_mlp",
            "bounded_mlp_feedback",
            "bounded_mlp_dual_alpha",
            "bounded_mlp_physical_alpha",
        ):
            temperature = _bounded_mlp_temperature_step(
                params,
                architecture,
                row,
                current.temperature,
                qroom,
                input_mean=input_mean,
                input_std=input_std,
                q_cap=q_cap,
                temperature_delta_cap=temperature_delta_cap,
                dt_hours=dt_hours,
            )
            mass_temperature = current.mass_temperature
            hidden = current.hidden
        elif architecture.thermal == "passive_rc":
            raw = params["rc_raw"]
            a_out = 0.18 * jax.nn.sigmoid(raw[0])
            a_mass = 0.18 * jax.nn.sigmoid(raw[1])
            a_mass_update = 0.08 * jax.nn.sigmoid(raw[2])
            q_gain = 0.08 * jax.nn.sigmoid(raw[3])
            solar_gain = 0.002 * jax.nn.sigmoid(raw[4])
            internal_gain = 0.05 * jax.nn.sigmoid(raw[5])
            vent_gain = 0.5 * jax.nn.sigmoid(raw[6])
            tout = row[TOUT_INDEX]
            passive_delta = (
                a_out * (tout - current.temperature)
                + a_mass * (current.mass_temperature - current.temperature)
                + q_gain * qroom
                + solar_gain * row[SOLAR_INDEX]
                + internal_gain * row[INTERNAL_INDEX]
                - vent_gain * jax.nn.relu(row[VENT_INDEX]) * (current.temperature - tout)
            )
            delta_t = temperature_delta_cap * jnp.tanh(passive_delta / temperature_delta_cap)
            temperature = jnp.clip(current.temperature + delta_t, 5.0, 40.0)
            mass_temperature = current.mass_temperature + a_mass_update * (
                current.temperature - current.mass_temperature
            )
            hidden = current.hidden
        else:
            hidden = _gru_step(params["gru"], thermal_features, current.hidden)
            equilibrium = row[TOUT_INDEX] + 30.0 * jnp.tanh(
                jnp.dot(params["thermal_out"], hidden) + params["thermal_out_bias"]
            )
            alpha = 0.25 * jax.nn.sigmoid(params["thermal_alpha_raw"])
            raw_delta = alpha * (equilibrium - current.temperature)
            delta_t = temperature_delta_cap * jnp.tanh(raw_delta / temperature_delta_cap)
            temperature = jnp.clip(current.temperature + delta_t, 5.0, 40.0)
            mass_temperature = current.mass_temperature

        next_state = RecurrentState(
            temperature=temperature,
            hp_mode=mode,
            previous_pel=pel,
            previous_q=qroom,
            energy=energy,
            hydronic_modes=hydronic_modes,
            mass_temperature=mass_temperature,
            hidden=hidden,
            previous_setpoint=setpoint,
            force_debt=debt,
            rc_mass_temperatures=rc_mass_temperatures,
            functional_memory=functional_memory,
            thermal_q_modes=thermal_q_modes,
        )
        output = jnp.asarray([temperature, qroom, pel])
        return next_state, (output, mode)

    _, (outputs, modes) = jax.lax.scan(
        step, state, (inputs, teacher_feedback)
    )
    return outputs, modes


def _load_policy_profile(
    dataset: Path,
    profile_id: int,
    *,
    policy: str,
) -> PolicyProfile:
    columns = closed_loop_required_columns(
        (),
        include_internal_gains=True,
        include_dhw_request=True,
        include_hp_force_off=True,
    )
    frame = read_dataset_frame(dataset, columns=columns, profile_ids=(profile_id,))
    profiles = to_closed_loop_profiles(
        frame,
        include_space_heating_availability=True,
        include_internal_gains=True,
        include_dhw_request=True,
        include_hp_force_off=True,
        hp_power_area_normalization="building_heated_area",
        metadata_columns=(),
    )
    if len(profiles) != 1:
        raise ValueError(f"Expected one trajectory for EGID {profile_id} in {dataset}; found {len(profiles)}")
    return PolicyProfile(policy=policy, profile=profiles[0])


def _choose_profile_id(datasets: Sequence[Path], requested: int | None, seed: int) -> int:
    common: set[int] | None = None
    for dataset in datasets:
        ids = set(int(value) for value in read_entity_profile_ids(dataset))
        common = ids if common is None else common.intersection(ids)
    available = sorted(common or ())
    if not available:
        raise ValueError("The policy datasets do not share any EGID")
    if requested is not None:
        if requested not in available:
            raise ValueError(f"EGID {requested} is not present in every requested policy dataset")
        return requested
    return int(np.random.default_rng(seed).choice(available))


def _choose_profile_ids(
    datasets: Sequence[Path],
    requested: Sequence[int],
    *,
    focus_profile_id: int | None,
    num_profiles: int,
    seed: int,
) -> tuple[list[int], int]:
    common: set[int] | None = None
    for dataset in datasets:
        ids = set(int(value) for value in read_entity_profile_ids(dataset))
        common = ids if common is None else common.intersection(ids)
    available = sorted(common or ())
    if not available:
        raise ValueError("The policy datasets do not share any EGID")
    requested_ids = list(dict.fromkeys(int(value) for value in requested))
    if focus_profile_id is not None and focus_profile_id not in requested_ids:
        requested_ids.insert(0, focus_profile_id)
    missing = sorted(set(requested_ids).difference(available))
    if missing:
        raise ValueError(f"EGIDs {missing} are not present in every requested policy dataset")
    rng = np.random.default_rng(seed)
    if not requested_ids:
        focus_profile_id = int(rng.choice(available))
        requested_ids = [focus_profile_id]
    focus = int(focus_profile_id if focus_profile_id is not None else requested_ids[0])
    if num_profiles > len(requested_ids):
        pool = np.asarray(sorted(set(available).difference(requested_ids)), dtype=np.int64)
        count = min(num_profiles - len(requested_ids), len(pool))
        if count:
            requested_ids.extend(int(value) for value in rng.choice(pool, size=count, replace=False))
    return requested_ids, focus


def _blocked_segments(
    policy_profile: PolicyProfile,
    *,
    train_days: int,
    test_days: int,
    steps_per_day: int,
) -> list[Segment]:
    train_steps = train_days * steps_per_day
    test_steps = test_days * steps_per_day
    total = policy_profile.profile.targets.shape[0]
    segments: list[Segment] = []
    cursor = 0
    while cursor < total:
        train_end = min(cursor + train_steps, total)
        if train_end - cursor >= 2:
            segments.append(Segment(policy_profile.policy, cursor, train_end, False))
        test_end = min(train_end + test_steps, total)
        if test_end - train_end >= 2:
            segments.append(Segment(policy_profile.policy, train_end, test_end, True))
        cursor += train_steps + test_steps
    return segments


def _training_windows(
    policies: Sequence[PolicyProfile],
    segments: Sequence[Segment],
    *,
    sequence_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_policy = {entry.policy: entry.profile for entry in policies}
    rows: list[tuple[str, int]] = []
    for segment in segments:
        if segment.held_out or segment.end - segment.start < sequence_length:
            continue
        starts = list(range(segment.start, segment.end - sequence_length + 1, stride))
        final_start = segment.end - sequence_length
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        rows.extend((segment.policy, start) for start in starts)
    if not rows:
        raise ValueError("No training windows; shorten --sequence-length or enlarge training blocks")
    inputs = np.stack([by_policy[p].inputs[s : s + sequence_length] for p, s in rows])
    targets = np.stack([by_policy[p].targets[s : s + sequence_length] for p, s in rows])
    initials = np.stack([by_policy[p].initial_temperature[s] for p, s in rows])
    policies_out = np.asarray([p for p, _ in rows])
    return inputs.astype(np.float32), targets.astype(np.float32), initials.astype(np.float32), policies_out


def _fit_scales(
    policies: Sequence[PolicyProfile],
    segments: Sequence[Segment],
) -> tuple[Scale, Scale]:
    by_policy = {entry.policy: entry.profile for entry in policies}
    input_rows = [by_policy[s.policy].inputs[s.start : s.end] for s in segments if not s.held_out]
    target_rows = [by_policy[s.policy].targets[s.start : s.end] for s in segments if not s.held_out]
    return Scale.fit(np.concatenate(input_rows)), Scale.fit(np.concatenate(target_rows))


def _caps(policies: Sequence[PolicyProfile], segments: Sequence[Segment]) -> tuple[float, float, float]:
    by_policy = {entry.policy: entry.profile for entry in policies}
    targets = np.concatenate(
        [by_policy[s.policy].targets[s.start : s.end] for s in segments if not s.held_out]
    )
    pel = max(float(np.quantile(targets[:, 2], 0.999)) * 1.25, 1.0)
    qroom = max(float(np.quantile(targets[:, 1], 0.999)) * 1.25, 1.0)
    return pel, qroom, max(24.0 * qroom, 8.0 * pel, 1.0)


def _event_aligned_temperature_response_loss(
    predictions: jax.Array,
    targets: jax.Array,
    inputs: jax.Array,
    *,
    temperature_scale: jax.Array,
    horizon_steps: int,
    setpoint_threshold_c: float,
) -> jax.Array:
    """MSE of pre/post mean Tin responses at exogenous interventions."""

    sequence_steps = predictions.shape[1]
    if horizon_steps < 1 or sequence_steps < 2 * horizon_steps + 1:
        return jnp.asarray(0.0, dtype=predictions.dtype)

    def responses(values: jax.Array) -> jax.Array:
        temperature = values[..., 0]
        cumulative = jnp.pad(
            jnp.cumsum(temperature, axis=1),
            ((0, 0), (1, 0)),
        )
        centers = cumulative[:, horizon_steps : sequence_steps - horizon_steps + 1]
        pre = (
            centers - cumulative[:, : sequence_steps - 2 * horizon_steps + 1]
        ) / horizon_steps
        post = (
            cumulative[:, 2 * horizon_steps : sequence_steps + 1] - centers
        ) / horizon_steps
        return post - pre

    setpoint_delta = inputs[:, 1:, SETPOINT_INDEX] - inputs[:, :-1, SETPOINT_INDEX]
    force_delta = inputs[:, 1:, FORCE_OFF_INDEX] - inputs[:, :-1, FORCE_OFF_INDEX]
    event_slice = slice(horizon_steps - 1, sequence_steps - horizon_steps)
    event_mask = (
        jnp.abs(setpoint_delta[:, event_slice]) >= setpoint_threshold_c
    ) | (force_delta[:, event_slice] > 0.5)
    squared_error = jnp.square(
        (responses(predictions) - responses(targets))
        / jnp.maximum(temperature_scale, 1e-6)
    )
    event_weight = event_mask.astype(squared_error.dtype)
    return jnp.sum(event_weight * squared_error) / jnp.maximum(
        jnp.sum(event_weight), 1.0
    )


def _loss_function(
    params: dict[str, Any],
    architecture: Architecture,
    inputs: jax.Array,
    targets: jax.Array,
    initials: jax.Array,
    *,
    rollout_kwargs: dict[str, Any],
    mode_loss_weight: float,
    controller_training_feedback: str = "closed_loop",
    event_tin_loss_weight: float = 0.0,
    observed_q_thermal_loss_weight: float = 0.0,
    qroom_window_mean_loss_weight: float = 0.0,
    event_horizon_steps: int = 12,
    event_setpoint_threshold_c: float = 0.05,
) -> tuple[
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    if controller_training_feedback == "closed_loop":
        predictions, modes = jax.vmap(
            lambda row_inputs, row_initial: _rollout(
                params,
                architecture,
                row_inputs,
                row_initial,
                **rollout_kwargs,
            )
        )(inputs, initials)
    else:
        initial_t = jnp.reshape(initials[:, 0], (-1, 1))
        zeros = jnp.zeros_like(initial_t)
        previous_t = jnp.concatenate((initial_t, targets[:, :-1, 0]), axis=1)
        previous_pel = jnp.concatenate((zeros, targets[:, :-1, 2]), axis=1)
        previous_qroom = jnp.concatenate((zeros, targets[:, :-1, 1]), axis=1)
        previous_mode = (previous_pel > 1e-5).astype(targets.dtype)
        feedback = jnp.stack(
            (previous_t, previous_pel, previous_qroom, previous_mode), axis=-1
        )
        force_history = controller_training_feedback == "temperature_and_history"
        predictions, modes = jax.vmap(
            lambda row_inputs, row_initial, row_feedback: _rollout(
                params,
                architecture,
                row_inputs,
                row_initial,
                teacher_feedback=row_feedback,
                teacher_force_temperature=True,
                teacher_force_power_history=force_history,
                **rollout_kwargs,
            )
        )(inputs, initials, feedback)
    target_std = rollout_kwargs["target_std"]
    normalized_error = (predictions - targets) / target_std
    if controller_training_feedback == "closed_loop":
        mse = jnp.mean(normalized_error**2)
    else:
        mse = jnp.mean(normalized_error[..., 1:] ** 2)
    if architecture.hp == "continuous":
        mode_loss = jnp.asarray(0.0)
    else:
        true_mode = (targets[..., 2] > 1e-5).astype(predictions.dtype)
        clipped = jnp.clip(modes, 1e-5, 1.0 - 1e-5)
        mode_loss = -jnp.mean(true_mode * jnp.log(clipped) + (1.0 - true_mode) * jnp.log1p(-clipped))
    if controller_training_feedback == "closed_loop":
        event_tin_loss = _event_aligned_temperature_response_loss(
            predictions,
            targets,
            inputs,
            temperature_scale=target_std[0],
            horizon_steps=event_horizon_steps,
            setpoint_threshold_c=event_setpoint_threshold_c,
        )
    else:
        event_tin_loss = jnp.asarray(0.0, dtype=predictions.dtype)
    if observed_q_thermal_loss_weight > 0.0:
        observed_q_temperature = jax.vmap(
            lambda row_inputs, row_targets, row_initial: (
                _observed_q_thermal_rollout(
                    params,
                    architecture,
                    row_inputs,
                    row_targets[:, 1],
                    row_initial,
                    input_mean=rollout_kwargs["input_mean"],
                    input_std=rollout_kwargs["input_std"],
                    target_t_mean=rollout_kwargs["target_mean"][0],
                    target_t_std=target_std[0],
                    q_cap=rollout_kwargs["q_cap"],
                    dt_hours=rollout_kwargs["dt_hours"],
                    temperature_delta_cap=rollout_kwargs[
                        "temperature_delta_cap"
                    ],
                )
            )
        )(inputs, targets, initials)
        observed_q_thermal_loss = jnp.mean(
            jnp.square(
                (observed_q_temperature - targets[..., 0]) / target_std[0]
            )
        )
    else:
        observed_q_thermal_loss = jnp.asarray(0.0, dtype=predictions.dtype)
    qroom_window_mean_loss = jnp.mean(
        jnp.square(
            jnp.mean(predictions[..., 1] - targets[..., 1], axis=-1)
            / jnp.maximum(target_std[1], 1e-6)
        )
    )
    loss = (
        mse
        + mode_loss_weight * mode_loss
        + event_tin_loss_weight * event_tin_loss
        + observed_q_thermal_loss_weight * observed_q_thermal_loss
        + qroom_window_mean_loss_weight * qroom_window_mean_loss
    )
    return loss, (
        mse,
        mode_loss,
        event_tin_loss,
        observed_q_thermal_loss,
        qroom_window_mean_loss,
    )


def _train_candidate(
    architecture: Architecture,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    train_initials: np.ndarray,
    *,
    args: argparse.Namespace,
    input_scale: Scale,
    target_scale: Scale,
    caps: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    params = _candidate_parameters(
        architecture,
        input_dim=train_inputs.shape[-1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        key=jax.random.PRNGKey(seed),
        hydronic_capacity_min_hours=getattr(
            args, "hydronic_capacity_min_hours", 0.25
        ),
        hydronic_capacity_max_hours=getattr(
            args, "hydronic_capacity_max_hours", 24.0
        ),
        hydronic_capacity_init_hours=getattr(
            args, "hydronic_capacity_init_hours", 2.0
        ),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adam(args.learning_rate),
    )
    state = optimizer.init(params)
    rollout_kwargs = {
        "input_mean": jnp.asarray(input_scale.mean),
        "input_std": jnp.asarray(input_scale.std),
        "target_mean": jnp.asarray(target_scale.mean),
        "target_std": jnp.asarray(target_scale.std),
        "pel_cap": caps[0],
        "q_cap": caps[1],
        "energy_cap": caps[2],
        "hidden_dim": args.hidden_dim,
        "dt_hours": args.dt_hours,
        "temperature_delta_cap": args.temperature_delta_cap,
        "controller_calendar_mode": getattr(
            args, "controller_calendar_mode", "all"
        ),
        "hp_recovery_emergency_threshold_c": getattr(
            args, "hp_recovery_emergency_threshold_c", 1.0
        ),
        "hydronic_capacity_min_hours": getattr(
            args, "hydronic_capacity_min_hours", 0.25
        ),
        "hydronic_capacity_max_hours": getattr(
            args, "hydronic_capacity_max_hours", 24.0
        ),
    }

    @eqx.filter_jit
    def train_step(model_params, optimizer_state, batch_inputs, batch_targets, batch_initials):
        (loss, components), gradients = jax.value_and_grad(_loss_function, has_aux=True)(
            model_params,
            architecture,
            batch_inputs,
            batch_targets,
            batch_initials,
            rollout_kwargs=rollout_kwargs,
            mode_loss_weight=args.mode_loss_weight,
            controller_training_feedback=getattr(
                args, "controller_training_feedback", "closed_loop"
            ),
            event_tin_loss_weight=getattr(
                args, "event_tin_loss_weight", 0.0
            ),
            observed_q_thermal_loss_weight=getattr(
                args, "observed_q_thermal_loss_weight", 0.0
            ),
            qroom_window_mean_loss_weight=getattr(
                args, "qroom_window_mean_loss_weight", 0.0
            ),
            event_horizon_steps=max(
                1,
                int(
                    round(
                        getattr(
                            args,
                            "event_tin_loss_horizon_hours",
                            getattr(args, "event_horizon_hours", 3.0),
                        )
                        / args.dt_hours
                    )
                ),
            ),
            event_setpoint_threshold_c=getattr(
                args, "event_tin_loss_setpoint_threshold_c", 0.05
            ),
        )
        grad_norm = optax.global_norm(gradients)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        return model_params, optimizer_state, loss, components, grad_norm

    rng = np.random.default_rng(seed)
    max_grad = 0.0
    last_loss = float("nan")
    last_event_tin_loss = float("nan")
    last_observed_q_thermal_loss = float("nan")
    last_qroom_window_mean_loss = float("nan")
    started = time.perf_counter()
    for epoch in range(args.epochs):
        order = rng.permutation(train_inputs.shape[0])
        losses: list[float] = []
        event_tin_losses: list[float] = []
        observed_q_thermal_losses: list[float] = []
        qroom_window_mean_losses: list[float] = []
        for batch_start in range(0, len(order), args.batch_size):
            batch = order[batch_start : batch_start + args.batch_size]
            params, state, loss, components, grad_norm = train_step(
                params,
                state,
                jnp.asarray(train_inputs[batch]),
                jnp.asarray(train_targets[batch]),
                jnp.asarray(train_initials[batch]),
            )
            loss_value = float(loss)
            grad_value = float(grad_norm)
            if not np.isfinite(loss_value) or not np.isfinite(grad_value):
                raise FloatingPointError(f"Non-finite optimization state for {architecture.name}")
            losses.append(loss_value)
            event_tin_losses.append(float(components[2]))
            observed_q_thermal_losses.append(float(components[3]))
            qroom_window_mean_losses.append(float(components[4]))
            max_grad = max(max_grad, grad_value)
        last_loss = float(np.mean(losses))
        last_event_tin_loss = float(np.mean(event_tin_losses))
        last_observed_q_thermal_loss = float(
            np.mean(observed_q_thermal_losses)
        )
        last_qroom_window_mean_loss = float(
            np.mean(qroom_window_mean_losses)
        )
        print(
            f"candidate={architecture.name} epoch={epoch + 1:02d}/{args.epochs} "
            f"loss={last_loss:.5f} "
            f"event_tin_loss={last_event_tin_loss:.5f} "
            f"observed_q_thermal_loss={last_observed_q_thermal_loss:.5f} "
            f"qroom_window_mean_loss={last_qroom_window_mean_loss:.5f} "
            f"grad_max={max_grad:.3g}",
            flush=True,
        )
    if architecture.hydronic == "leaky_emitter":
        capacity_fraction = 1.0 / (
            1.0 + math.exp(-float(params["energy_capacity_raw"]))
        )
        capacity_hours = getattr(args, "hydronic_capacity_min_hours", 0.25) + (
            getattr(args, "hydronic_capacity_max_hours", 24.0)
            - getattr(args, "hydronic_capacity_min_hours", 0.25)
        ) * capacity_fraction
    elif architecture.hydronic == "positive_convolution":
        capacity_hours = float("nan")
    else:
        capacity_hours = caps[2] / caps[1]
    hydronic_weights = np.asarray(
        jax.nn.softmax(params["hydronic_weight_raw"])
    )
    stable_rc_coupling_budget = 0.12 / (
        1.0 + math.exp(-float(params["stable_rc_raw"][0]))
    )
    narx_residual_scale = 0.5 / (
        1.0 + math.exp(-float(params["narx_residual_scale_raw"]))
    )
    thermal_q_gain = 0.02 / (
        1.0 + math.exp(-float(params["thermal_q_gain_raw"]))
    )
    thermal_q_weights = np.asarray(
        jax.nn.softmax(params["thermal_q_weight_raw"])
    )
    hp_active_floor_fraction = 0.01 + 0.49 / (
        1.0 + math.exp(-float(params["hp_active_floor_raw"]))
    )
    thermal_alpha = 0.25 / (
        1.0 + math.exp(-float(params["thermal_alpha_raw"]))
    )
    thermal_alpha_free = 0.25 / (
        1.0 + math.exp(-float(params["thermal_alpha_free_raw"]))
    )
    thermal_alpha_heating = 0.25 / (
        1.0 + math.exp(-float(params["thermal_alpha_heating_raw"]))
    )
    thermal_rate_base = 0.5 / (
        1.0 + math.exp(-float(params["thermal_rate_base_raw"]))
    )
    thermal_rate_vent = 0.5 / (
        1.0 + math.exp(-float(params["thermal_rate_vent_raw"]))
    )
    thermal_rate_stack = 0.25 / (
        1.0 + math.exp(-float(params["thermal_rate_stack_raw"]))
    )
    return params, {
        "train_loss": last_loss,
        "train_event_tin_response_loss": last_event_tin_loss,
        "train_observed_q_thermal_loss": last_observed_q_thermal_loss,
        "train_qroom_window_mean_loss": last_qroom_window_mean_loss,
        "controller_training_feedback": getattr(
            args, "controller_training_feedback", "closed_loop"
        ),
        "max_grad_norm": max_grad,
        "training_seconds": time.perf_counter() - started,
        "parameter_count": float(
            sum(int(value.size) for value in jax.tree_util.tree_leaves(params))
        ),
        "hydronic_capacity_hours": float(capacity_hours),
        "hydronic_capacity_wh_m2": float(capacity_hours * caps[1]),
        "hydronic_delivery_efficiency": float(
            jax.nn.sigmoid(params["hydronic_efficiency_raw"])
        ),
        "hydronic_weight_fast": float(hydronic_weights[0]),
        "hydronic_weight_medium": float(hydronic_weights[1]),
        "hydronic_weight_slow": float(hydronic_weights[2]),
        "stable_rc_coupling_budget": float(stable_rc_coupling_budget),
        "narx_residual_scale_c": float(narx_residual_scale),
        "thermal_q_gain_c_per_w_m2": float(thermal_q_gain),
        "thermal_q_weight_direct": float(thermal_q_weights[0]),
        "thermal_q_weight_1h": float(thermal_q_weights[1]),
        "thermal_q_weight_4h": float(thermal_q_weights[2]),
        "thermal_q_weight_16h": float(thermal_q_weights[3]),
        "thermal_alpha": float(thermal_alpha),
        "thermal_alpha_free": float(thermal_alpha_free),
        "thermal_alpha_heating": float(thermal_alpha_heating),
        "thermal_rate_base_per_hour": float(thermal_rate_base),
        "thermal_rate_vent_per_hour": float(thermal_rate_vent),
        "thermal_rate_stack_per_hour": float(thermal_rate_stack),
        "hp_active_floor_fraction": float(hp_active_floor_fraction),
        "hp_mode_gap_slope": float(
            0.5 + jax.nn.softplus(params["hp_gap_slope_raw"])
        ),
        "hp_mode_debt_slope": float(
            0.1 + jax.nn.softplus(params["hp_debt_slope_raw"])
        ),
    }


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def _event_response_errors(
    inputs: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    horizon_steps: int,
) -> dict[str, float]:
    setpoint = inputs[:, SETPOINT_INDEX]
    changes = np.diff(setpoint, prepend=setpoint[0])
    events = np.flatnonzero(np.abs(changes) >= 0.05)
    events = events[(events >= horizon_steps) & (events + horizon_steps <= len(setpoint))]
    force = inputs[:, FORCE_OFF_INDEX]
    force_events = np.flatnonzero(np.diff(force, prepend=force[0]) > 0.5)
    force_events = force_events[
        (force_events >= horizon_steps) & (force_events + horizon_steps <= len(force))
    ]

    def response(values: np.ndarray, indices: np.ndarray, channel: int) -> np.ndarray:
        return np.asarray(
            [
                np.mean(values[index : index + horizon_steps, channel])
                - np.mean(values[index - horizon_steps : index, channel])
                for index in indices
            ]
        )

    result: dict[str, float] = {
        "setpoint_event_count": float(events.size),
        "force_off_event_count": float(force_events.size),
    }
    for label, indices in (("setpoint", events), ("force_off", force_events)):
        for channel, signal in ((0, "temperature"), (1, "qroom"), (2, "pel")):
            if indices.size:
                true_response = response(target, indices, channel)
                predicted_response = response(prediction, indices, channel)
                result[f"{label}_{signal}_response_rmse"] = _rmse(predicted_response, true_response)
                result[f"{label}_{signal}_response_bias"] = float(np.mean(predicted_response - true_response))
            else:
                result[f"{label}_{signal}_response_rmse"] = float("nan")
                result[f"{label}_{signal}_response_bias"] = float("nan")
    return result


def _evaluate_candidate(
    params: dict[str, Any],
    architecture: Architecture,
    policies: Sequence[PolicyProfile],
    segments: Sequence[Segment],
    *,
    args: argparse.Namespace,
    input_scale: Scale,
    target_scale: Scale,
    caps: tuple[float, float, float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    by_policy = {entry.policy: entry.profile for entry in policies}
    kwargs = {
        "input_mean": jnp.asarray(input_scale.mean),
        "input_std": jnp.asarray(input_scale.std),
        "target_mean": jnp.asarray(target_scale.mean),
        "target_std": jnp.asarray(target_scale.std),
        "pel_cap": caps[0],
        "q_cap": caps[1],
        "energy_cap": caps[2],
        "hidden_dim": args.hidden_dim,
        "dt_hours": args.dt_hours,
        "temperature_delta_cap": args.temperature_delta_cap,
        "controller_calendar_mode": getattr(
            args, "controller_calendar_mode", "all"
        ),
        "hp_recovery_emergency_threshold_c": getattr(
            args, "hp_recovery_emergency_threshold_c", 1.0
        ),
        "hydronic_capacity_min_hours": getattr(
            args, "hydronic_capacity_min_hours", 0.25
        ),
        "hydronic_capacity_max_hours": getattr(
            args, "hydronic_capacity_max_hours", 24.0
        ),
    }
    rollout = eqx.filter_jit(lambda x, t: _rollout(params, architecture, x, t, **kwargs)[0])
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    predictions_by_policy: dict[str, list[np.ndarray]] = {
        entry.policy: [] for entry in policies
    }
    targets_by_policy: dict[str, list[np.ndarray]] = {
        entry.policy: [] for entry in policies
    }
    trace_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, float]] = []
    for segment_index, segment in enumerate(segments):
        if not segment.held_out:
            continue
        profile = by_policy[segment.policy]
        warmup_steps = max(0, int(round(args.warmup_hours / args.dt_hours)))
        rollout_start = max(0, segment.start - warmup_steps)
        inputs_with_context = profile.inputs[rollout_start : segment.end]
        target = profile.targets[segment.start : segment.end]
        initial = profile.initial_temperature[rollout_start]
        prediction_with_context = np.asarray(
            rollout(jnp.asarray(inputs_with_context), jnp.asarray(initial))
        )
        score_offset = segment.start - rollout_start
        prediction = prediction_with_context[score_offset:]
        inputs = profile.inputs[segment.start : segment.end]
        predictions.append(prediction)
        targets.append(target)
        predictions_by_policy[segment.policy].append(prediction)
        targets_by_policy[segment.policy].append(target)
        event_rows.append(
            _event_response_errors(
                inputs,
                target,
                prediction,
                horizon_steps=max(1, int(round(args.event_horizon_hours / args.dt_hours))),
            )
        )
        trace_rows.append(
            {
                "policy": segment.policy,
                "segment": segment_index,
                "datetime": profile.datetime[segment.start : segment.end],
                "inputs": inputs,
                "target": target,
                "prediction": prediction,
            }
        )
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    errors = prediction - target
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    bias = np.mean(errors, axis=0)
    nrmse = rmse / target_scale.std
    result: dict[str, float] = {
        "temperature_rmse_c": float(rmse[0]),
        "qroom_rmse_w_m2": float(rmse[1]),
        "pel_rmse_w_m2": float(rmse[2]),
        "temperature_mae_c": float(mae[0]),
        "qroom_mae_w_m2": float(mae[1]),
        "pel_mae_w_m2": float(mae[2]),
        "temperature_bias_c": float(bias[0]),
        "qroom_bias_w_m2": float(bias[1]),
        "pel_bias_w_m2": float(bias[2]),
        "temperature_nrmse": float(nrmse[0]),
        "qroom_nrmse": float(nrmse[1]),
        "pel_nrmse": float(nrmse[2]),
        "composite_nrmse": float(np.sqrt(np.mean(nrmse**2))),
        "prediction_temperature_min_c": float(np.min(prediction[:, 0])),
        "prediction_temperature_max_c": float(np.max(prediction[:, 0])),
        "implausible_temperature_fraction": float(
            np.mean((prediction[:, 0] < 5.0) | (prediction[:, 0] > 40.0))
        ),
        "target_qroom_mean_w_m2": float(np.mean(target[:, 1])),
        "prediction_qroom_mean_w_m2": float(np.mean(prediction[:, 1])),
        "target_pel_mean_w_m2": float(np.mean(target[:, 2])),
        "prediction_pel_mean_w_m2": float(np.mean(prediction[:, 2])),
        "target_qroom_energy_kwh_m2": float(
            args.dt_hours * np.sum(target[:, 1]) / 1000.0
        ),
        "prediction_qroom_energy_kwh_m2": float(
            args.dt_hours * np.sum(prediction[:, 1]) / 1000.0
        ),
        "target_pel_energy_kwh_m2": float(
            args.dt_hours * np.sum(target[:, 2]) / 1000.0
        ),
        "prediction_pel_energy_kwh_m2": float(
            args.dt_hours * np.sum(prediction[:, 2]) / 1000.0
        ),
        "qroom_energy_ratio": float(
            np.sum(prediction[:, 1]) / max(np.sum(target[:, 1]), 1e-8)
        ),
        "pel_energy_ratio": float(
            np.sum(prediction[:, 2]) / max(np.sum(target[:, 2]), 1e-8)
        ),
    }
    for policy in predictions_by_policy:
        policy_prediction = np.concatenate(predictions_by_policy[policy])
        policy_target = np.concatenate(targets_by_policy[policy])
        policy_rmse = np.sqrt(np.mean((policy_prediction - policy_target) ** 2, axis=0))
        result.update(
            {
                f"{policy}_temperature_rmse_c": float(policy_rmse[0]),
                f"{policy}_qroom_rmse_w_m2": float(policy_rmse[1]),
                f"{policy}_pel_rmse_w_m2": float(policy_rmse[2]),
            }
        )
    for key in event_rows[0]:
        values = np.asarray([row[key] for row in event_rows], dtype=np.float64)
        result[key] = float(np.nanmean(values)) if bool(np.isfinite(values).any()) else float("nan")
    # Penalize intervention errors in normalized units, but keep point accuracy
    # dominant. This is an architecture-screening score, not a training loss.
    event_terms = [
        result["setpoint_temperature_response_rmse"] / target_scale.std[0],
        result["setpoint_qroom_response_rmse"] / target_scale.std[1],
        result["setpoint_pel_response_rmse"] / target_scale.std[2],
        result["force_off_temperature_response_rmse"] / target_scale.std[0],
        result["force_off_qroom_response_rmse"] / target_scale.std[1],
        result["force_off_pel_response_rmse"] / target_scale.std[2],
    ]
    finite_events = [value for value in event_terms if np.isfinite(value)]
    event_score = float(np.mean(finite_events)) if finite_events else 0.0
    result["event_response_nrmse"] = event_score
    result["selection_score"] = result["composite_nrmse"] + args.event_score_weight * event_score
    return result, trace_rows


def _annual_force_off_rollout(
    params: dict[str, Any],
    architecture: Architecture,
    profile: ClosedLoopProfile,
    *,
    args: argparse.Namespace,
    input_scale: Scale,
    target_scale: Scale,
    caps: tuple[float, float, float],
) -> np.ndarray:
    """Run one autonomous January-to-December trajectory without state resets."""
    kwargs = {
        "input_mean": jnp.asarray(input_scale.mean),
        "input_std": jnp.asarray(input_scale.std),
        "target_mean": jnp.asarray(target_scale.mean),
        "target_std": jnp.asarray(target_scale.std),
        "pel_cap": caps[0],
        "q_cap": caps[1],
        "energy_cap": caps[2],
        "hidden_dim": args.hidden_dim,
        "dt_hours": args.dt_hours,
        "temperature_delta_cap": args.temperature_delta_cap,
        "controller_calendar_mode": getattr(
            args, "controller_calendar_mode", "all"
        ),
        "hp_recovery_emergency_threshold_c": getattr(
            args, "hp_recovery_emergency_threshold_c", 1.0
        ),
        "hydronic_capacity_min_hours": getattr(
            args, "hydronic_capacity_min_hours", 0.25
        ),
        "hydronic_capacity_max_hours": getattr(
            args, "hydronic_capacity_max_hours", 24.0
        ),
    }
    rollout = eqx.filter_jit(
        lambda x, t: _rollout(params, architecture, x, t, **kwargs)[0]
    )
    return np.asarray(
        rollout(
            jnp.asarray(profile.inputs),
            jnp.asarray(profile.initial_temperature[0]),
        ),
        dtype=np.float32,
    )


def _annual_run_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_scale: Scale,
) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    nrmse = rmse / np.asarray(target_scale.std, dtype=np.float64)
    return {
        "annual_force_off_temperature_rmse_c": float(rmse[0]),
        "annual_force_off_qroom_rmse_w_m2": float(rmse[1]),
        "annual_force_off_pel_rmse_w_m2": float(rmse[2]),
        "annual_force_off_composite_nrmse": float(np.sqrt(np.mean(nrmse**2))),
        "annual_force_off_temperature_min_c": float(np.min(prediction[:, 0])),
        "annual_force_off_temperature_max_c": float(np.max(prediction[:, 0])),
        "annual_force_off_implausible_temperature_fraction": float(
            np.mean((prediction[:, 0] < 5.0) | (prediction[:, 0] > 40.0))
        ),
    }


def _winter_mask(datetime: np.ndarray) -> np.ndarray:
    months = pd.DatetimeIndex(pd.to_datetime(datetime)).month
    return np.asarray(np.isin(months, (1, 2, 3, 11, 12)), dtype=bool)


def _annual_ensemble_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    datetime: np.ndarray,
    target_scale: Scale,
) -> dict[str, float]:
    """Summarize sensitivity to independently fitted parameter solutions."""
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean = np.mean(predictions, axis=0)
    lower = np.quantile(predictions, 0.025, axis=0)
    upper = np.quantile(predictions, 0.975, axis=0)
    width = upper - lower
    rmse = np.sqrt(np.mean((mean - target) ** 2, axis=0))
    nrmse = rmse / np.asarray(target_scale.std, dtype=np.float64)
    restart_rmse = np.sqrt(np.mean((predictions - target[None, ...]) ** 2, axis=1))
    restart_composite = np.sqrt(
        np.mean((restart_rmse / np.asarray(target_scale.std)[None, :]) ** 2, axis=1)
    )
    winter = _winter_mask(datetime)
    metrics: dict[str, float] = {
        "annual_force_off_ensemble_temperature_rmse_c": float(rmse[0]),
        "annual_force_off_ensemble_qroom_rmse_w_m2": float(rmse[1]),
        "annual_force_off_ensemble_pel_rmse_w_m2": float(rmse[2]),
        "annual_force_off_ensemble_composite_nrmse": float(np.sqrt(np.mean(nrmse**2))),
        "annual_force_off_restart_composite_nrmse_mean": float(np.mean(restart_composite)),
        "annual_force_off_restart_composite_nrmse_std": float(np.std(restart_composite)),
        "annual_force_off_stability_score": float(
            np.mean(restart_composite) + np.std(restart_composite)
        ),
    }
    channel_names = ("temperature", "qroom", "pel")
    for channel, name in enumerate(channel_names):
        covered = (target[:, channel] >= lower[:, channel]) & (
            target[:, channel] <= upper[:, channel]
        )
        metrics[f"annual_force_off_{name}_restart_coverage95"] = float(np.mean(covered))
        metrics[f"annual_force_off_{name}_restart_band_mean_width"] = float(
            np.mean(width[:, channel])
        )
        metrics[f"annual_force_off_{name}_restart_band_p95_width"] = float(
            np.quantile(width[:, channel], 0.95)
        )
    if np.any(winter):
        temperature_width = width[winter, 0]
        metrics.update(
            {
                "annual_force_off_winter_temperature_rmse_c": _rmse(
                    mean[winter, 0], target[winter, 0]
                ),
                "annual_force_off_winter_temperature_restart_band_mean_width_c": float(
                    np.mean(temperature_width)
                ),
                "annual_force_off_winter_temperature_restart_band_p95_width_c": float(
                    np.quantile(temperature_width, 0.95)
                ),
                "annual_force_off_winter_temperature_lower_min_c": float(
                    np.min(lower[winter, 0])
                ),
                "annual_force_off_winter_temperature_upper_max_c": float(
                    np.max(upper[winter, 0])
                ),
            }
        )
    return metrics


def _component_effects(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component in ("hp", "hydronic", "thermal"):
        for level, group in results.groupby(component):
            rows.append(
                {
                    "component": component,
                    "level": level,
                    "mean_selection_score": float(group["selection_score"].mean()),
                    "std_selection_score": float(group["selection_score"].std(ddof=1)),
                    "candidate_count": int(group.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _write_dashboard(
    results: pd.DataFrame,
    component_effects: pd.DataFrame,
    output_dir: Path,
) -> Path:
    rank_column = "joint_rank_score" if "joint_rank_score" in results else "selection_score"
    ranked = results.sort_values(rank_column).reset_index(drop=True)
    top = ranked.head(min(15, len(ranked)))
    figure = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Held-out architecture score",
            "Signal RMSE",
            "Intervention-response error",
            "Accuracy versus runtime",
            "Average component effect",
            "Tin RMSE by policy",
            "Full-year force-off robustness",
            "Full-year winter restart spread",
        ),
        vertical_spacing=0.08,
    )
    labels = top["architecture"].str.replace("__", "<br>", regex=False)
    figure.add_trace(
        go.Bar(
            x=labels,
            y=top["selection_score"],
            error_y={"type": "data", "array": top["selection_score_std"]},
            name="Selection score",
        ),
        row=1,
        col=1,
    )
    colors = {"Tin": "#1f77b4", "Qroom": "#2ca02c", "Pel": "#d62728"}
    for column, name in (
        ("temperature_nrmse", "Tin"),
        ("qroom_nrmse", "Qroom"),
        ("pel_nrmse", "Pel"),
    ):
        figure.add_trace(go.Bar(x=labels, y=top[column], name=name, marker_color=colors[name]), row=1, col=2)
    for column, name in (
        ("setpoint_temperature_response_rmse", "Tset -> Tin"),
        ("setpoint_pel_response_rmse", "Tset -> Pel"),
        ("force_off_qroom_response_rmse", "Force-off -> Qroom"),
    ):
        figure.add_trace(go.Bar(x=labels, y=top[column], name=name), row=2, col=1)
    figure.add_trace(
        go.Scatter(
            x=results["training_seconds"] / 60.0,
            y=results["selection_score"],
            mode="markers+text",
            text=results["rank"],
            customdata=results[["architecture", "parameter_count"]],
            hovertemplate=(
                "%{customdata[0]}<br>minutes=%{x:.2f}<br>score=%{y:.4f}"
                "<br>parameters=%{customdata[1]:.0f}<extra></extra>"
            ),
            name="Candidates",
        ),
        row=2,
        col=2,
    )
    effect_labels = component_effects["component"] + "=" + component_effects["level"]
    figure.add_trace(
        go.Bar(
            x=effect_labels,
            y=component_effects["mean_selection_score"],
            error_y={"type": "data", "array": component_effects["std_selection_score"]},
            customdata=component_effects[["candidate_count"]],
            hovertemplate=(
                "%{x}<br>mean score=%{y:.4f}<br>n=%{customdata[0]}<extra></extra>"
            ),
            name="Component mean",
        ),
        row=3,
        col=1,
    )
    for column, name, color in (
        ("setpoint_temperature_rmse_c", "Setpoint policy", "#1f77b4"),
        ("force_off_temperature_rmse_c", "Force-off policy", "#d62728"),
    ):
        figure.add_trace(
            go.Bar(x=labels, y=top[column], name=name, marker_color=color),
            row=3,
            col=2,
        )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=top["annual_force_off_stability_score"],
            name="Annual robustness",
            marker_color="#6a1b9a",
        ),
        row=4,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=top["annual_force_off_winter_temperature_restart_band_mean_width_c"],
            name="Winter mean width",
            marker_color="#ef6c00",
        ),
        row=4,
        col=2,
    )
    figure.update_xaxes(tickangle=35, row=1, col=1)
    figure.update_xaxes(tickangle=35, row=1, col=2)
    figure.update_xaxes(tickangle=35, row=2, col=1)
    figure.update_xaxes(title_text="training minutes", row=2, col=2)
    figure.update_xaxes(tickangle=35, row=3, col=1)
    figure.update_xaxes(tickangle=35, row=3, col=2)
    figure.update_xaxes(tickangle=35, row=4, col=1)
    figure.update_xaxes(tickangle=35, row=4, col=2)
    figure.update_yaxes(title_text="lower is better", row=1, col=1)
    figure.update_yaxes(title_text="NRMSE", row=1, col=2)
    figure.update_yaxes(title_text="physical units", row=2, col=1)
    figure.update_yaxes(title_text="lower is better", row=4, col=1)
    figure.update_yaxes(title_text="degC", row=4, col=2)
    figure.update_layout(
        title="Building-local closed-loop architecture search",
        height=1800,
        barmode="group",
        hoverlabel={"namelength": -1},
    )
    path = output_dir / "architecture_search_dashboard.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def _write_best_traces(
    architecture: Architecture,
    traces: Sequence[dict[str, Any]],
    output_dir: Path,
) -> Path:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Indoor temperature", "Delivered room heat", "HP electric power"),
        vertical_spacing=0.04,
    )
    signal_names = ((0, "Tin", "C"), (1, "Qroom", "W/m2"), (2, "Pel", "W/m2"))
    policy_colors = {"setpoint": "#1f77b4", "force_off": "#d62728"}
    for trace in traces:
        policy = str(trace["policy"])
        for row, (channel, signal, unit) in enumerate(signal_names, start=1):
            figure.add_trace(
                go.Scatter(
                    x=trace["datetime"],
                    y=trace["target"][:, channel],
                    name=f"{policy} simulated {signal}",
                    legendgroup=f"{policy}_{signal}",
                    line={"color": policy_colors.get(policy, "#444"), "width": 1.2},
                    showlegend=trace["segment"] == min(item["segment"] for item in traces),
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=trace["datetime"],
                    y=trace["prediction"][:, channel],
                    name=f"{policy} emulated {signal}",
                    legendgroup=f"{policy}_{signal}",
                    line={"color": policy_colors.get(policy, "#444"), "dash": "dot", "width": 1.2},
                    showlegend=trace["segment"] == min(item["segment"] for item in traces),
                ),
                row=row,
                col=1,
            )
            figure.update_yaxes(title_text=unit, row=row, col=1)
    figure.update_layout(
        title=f"Best held-out autonomous rollouts: {architecture.name}",
        height=1000,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
    )
    path = output_dir / "best_candidate_traces.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def _write_focus_annual_force_off_traces(
    candidates: Sequence[FittedCandidate],
    architecture_names: Sequence[str],
    profile: ClosedLoopProfile,
    output_dir: Path,
) -> Path:
    selected = [
        (
            name,
            [
                value.annual_force_off_prediction
                for value in candidates
                if value.architecture.name == name
            ],
        )
        for name in architecture_names
    ]
    selected = [(name, values) for name, values in selected if values]
    figure = make_subplots(
        rows=3,
        cols=len(selected),
        shared_xaxes="columns",
        subplot_titles=tuple(name.replace("__", "<br>") for name, _ in selected),
        vertical_spacing=0.05,
        horizontal_spacing=0.04,
    )
    channels = ((0, "Tin", "degC"), (1, "Qroom", "W/m2"), (2, "Pel", "W/m2"))
    for column, (_, values) in enumerate(selected, start=1):
        predictions = np.stack(values)
        mean = np.mean(predictions, axis=0)
        lower = np.quantile(predictions, 0.025, axis=0)
        upper = np.quantile(predictions, 0.975, axis=0)
        for row, (channel, signal, unit) in enumerate(channels, start=1):
            show_legend = column == 1 and row == 1
            figure.add_trace(
                go.Scatter(
                    x=profile.datetime,
                    y=lower[:, channel],
                    mode="lines",
                    line={"color": "rgba(198,40,40,0)", "width": 0},
                    name="2.5% restart quantile",
                    legendgroup="restart_band",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=profile.datetime,
                    y=upper[:, channel],
                    mode="lines",
                    line={"color": "rgba(198,40,40,0)", "width": 0},
                    fill="tonexty",
                    fillcolor="rgba(198,40,40,0.18)",
                    name="95% restart envelope",
                    legendgroup="restart_band",
                    showlegend=show_legend,
                    hovertemplate=f"{signal} restart envelope<extra></extra>",
                ),
                row=row,
                col=column,
            )
            for restart, prediction in enumerate(predictions):
                figure.add_trace(
                    go.Scattergl(
                        x=profile.datetime,
                        y=prediction[:, channel],
                        mode="lines",
                        line={"color": "rgba(198,40,40,0.22)", "width": 0.6},
                        name="Individual restart",
                        legendgroup="individual_restart",
                        showlegend=show_legend and restart == 0,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=column,
                )
            figure.add_trace(
                go.Scattergl(
                    x=profile.datetime,
                    y=profile.targets[:, channel],
                    mode="lines",
                    line={"color": "#1565c0", "width": 1.2},
                    name="EnergyPlus",
                    legendgroup="target",
                    showlegend=show_legend,
                ),
                row=row,
                col=column,
            )
            figure.add_trace(
                go.Scattergl(
                    x=profile.datetime,
                    y=mean[:, channel],
                    mode="lines",
                    line={"color": "#c62828", "width": 1.2},
                    name="Restart mean",
                    legendgroup="mean",
                    showlegend=show_legend,
                ),
                row=row,
                col=column,
            )
            if column == 1:
                figure.update_yaxes(title_text=unit, row=row, col=column)
    figure.update_layout(
        title=f"Autonomous full-year force-off robustness, EGID {profile.profile_id}",
        height=1050,
        width=max(1300, 500 * len(selected)),
        hovermode="x unified",
        hoverlabel={"namelength": -1},
    )
    path = output_dir / f"focus_{profile.profile_id}_annual_force_off_architectures.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def _decode_plotly_array(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict) or "bdata" not in value:
        raise ValueError("Unsupported Plotly array encoding")
    dtypes = {
        "f8": "<f8",
        "f4": "<f4",
        "i4": "<i4",
        "u4": "<u4",
        "i2": "<i2",
        "u2": "<u2",
        "i1": "i1",
        "u1": "u1",
    }
    return np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtypes[value["dtype"]])


def _reference_failure_metrics(path: Path) -> dict[str, float]:
    """Extract the temperature failure metrics from an existing Plotly trace."""
    source = path.read_text()
    plot_start = source.rindex("Plotly.newPlot(")
    data_start = source.index("[", plot_start)
    traces, _ = json.JSONDecoder().raw_decode(source[data_start:])
    names = [trace.get("name") for trace in traces]
    simulated_index = names.index("Simulated Indoor temperature")
    upper_index = names.index("95% Indoor temperature")
    mean_index = names.index("Emulated Indoor temperature")
    lower_index = upper_index - 1
    target = _decode_plotly_array(traces[simulated_index]["y"]).astype(np.float64)
    lower = _decode_plotly_array(traces[lower_index]["y"]).astype(np.float64)
    upper = _decode_plotly_array(traces[upper_index]["y"]).astype(np.float64)
    mean = _decode_plotly_array(traces[mean_index]["y"]).astype(np.float64)
    datetime = pd.to_datetime(traces[simulated_index]["x"])
    winter = np.asarray(datetime.month.isin((1, 2, 3, 11, 12)), dtype=bool)
    width = upper - lower
    return {
        "temperature_mean_rmse_c": _rmse(mean, target),
        "temperature_coverage95": float(np.mean((target >= lower) & (target <= upper))),
        "temperature_band_mean_width_c": float(np.mean(width)),
        "temperature_band_p95_width_c": float(np.quantile(width, 0.95)),
        "winter_temperature_band_mean_width_c": float(np.mean(width[winter])),
        "winter_temperature_band_p95_width_c": float(np.quantile(width[winter], 0.95)),
        "winter_temperature_lower_min_c": float(np.min(lower[winter])),
        "winter_temperature_upper_max_c": float(np.max(upper[winter])),
    }


def _architectures(args: argparse.Namespace) -> list[Architecture]:
    hp = tuple(args.hp_architectures)
    hydronic = tuple(args.hydronic_architectures)
    thermal = tuple(args.thermal_architectures)
    rows = [Architecture(*values) for values in itertools.product(hp, hydronic, thermal)]
    if args.max_candidates and args.max_candidates < len(rows):
        rng = np.random.default_rng(args.seed)
        selected = sorted(rng.choice(len(rows), size=args.max_candidates, replace=False))
        rows = [rows[index] for index in selected]
    return rows


def _run_profile_search(
    profile_id: int,
    architectures: Sequence[Architecture],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[FittedCandidate], dict[str, Any]]:
    policies = [
        _load_policy_profile(args.setpoint_dataset, profile_id, policy="setpoint"),
        _load_policy_profile(args.force_off_dataset, profile_id, policy="force_off"),
    ]
    steps_per_day = max(1, int(round(24.0 / args.dt_hours)))
    segments = [
        segment
        for policy in policies
        for segment in _blocked_segments(
            policy,
            train_days=args.train_block_days,
            test_days=args.test_block_days,
            steps_per_day=steps_per_day,
        )
    ]
    input_scale, target_scale = _fit_scales(policies, segments)
    caps = _caps(policies, segments)
    train_inputs, train_targets, train_initials, _ = _training_windows(
        policies,
        segments,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    print(
        f"profile={profile_id} candidates={len(architectures)} train_windows={len(train_inputs)} "
        f"held_out_segments={sum(segment.held_out for segment in segments)} "
        "metadata_conditioning=disabled",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    fitted: list[FittedCandidate] = []
    force_off_profile = next(value.profile for value in policies if value.policy == "force_off")
    for index, architecture in enumerate(architectures, start=1):
        for restart in range(args.restarts):
            print(
                f"starting_candidate={index}/{len(architectures)} restart={restart + 1}/{args.restarts} "
                f"{architecture.name}",
                flush=True,
            )
            # The same restart seeds are shared across architectures. This
            # reduces initialization noise in component-level comparisons.
            fit_seed = args.seed + restart * 100_003
            params, train_metrics = _train_candidate(
                architecture,
                train_inputs,
                train_targets,
                train_initials,
                args=args,
                input_scale=input_scale,
                target_scale=target_scale,
                caps=caps,
                seed=fit_seed,
            )
            test_metrics, traces = _evaluate_candidate(
                params,
                architecture,
                policies,
                segments,
                args=args,
                input_scale=input_scale,
                target_scale=target_scale,
                caps=caps,
            )
            annual_prediction = _annual_force_off_rollout(
                params,
                architecture,
                force_off_profile,
                args=args,
                input_scale=input_scale,
                target_scale=target_scale,
                caps=caps,
            )
            annual_metrics = _annual_run_metrics(
                annual_prediction,
                force_off_profile.targets,
                target_scale,
            )
            row = {
                "profile_id": profile_id,
                "architecture": architecture.name,
                "restart": restart,
                **asdict(architecture),
                **train_metrics,
                **test_metrics,
                **annual_metrics,
            }
            rows.append(row)
            fitted.append(
                FittedCandidate(
                    profile_id=profile_id,
                    architecture=architecture,
                    restart=restart,
                    params=params,
                    held_out_traces=traces,
                    selection_score=float(test_metrics["selection_score"]),
                    annual_force_off_prediction=annual_prediction,
                )
            )
            print(
                f"finished_candidate={architecture.name} restart={restart + 1} "
                f"score={test_metrics['selection_score']:.4f} "
                f"Tin_RMSE={test_metrics['temperature_rmse_c']:.3f} "
                f"Q_RMSE={test_metrics['qroom_rmse_w_m2']:.3f} "
                f"Pel_RMSE={test_metrics['pel_rmse_w_m2']:.3f}",
                flush=True,
            )
    context = {
        "policies": policies,
        "segments": segments,
        "input_scale": input_scale,
        "target_scale": target_scale,
        "caps": caps,
        "train_windows": int(train_inputs.shape[0]),
        "held_out_segments": int(sum(segment.held_out for segment in segments)),
        "force_off_profile": force_off_profile,
    }
    return rows, fitted, context


def run(args: argparse.Namespace) -> None:
    datasets = [args.setpoint_dataset, args.force_off_dataset]
    requested = list(args.profile_ids or ())
    if args.profile_id is not None:
        requested.insert(0, args.profile_id)
    profile_ids, focus_profile_id = _choose_profile_ids(
        datasets,
        requested,
        focus_profile_id=args.focus_profile_id,
        num_profiles=args.num_profiles,
        seed=args.seed,
    )
    architectures = _architectures(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile_ids": profile_ids,
        "focus_profile_id": focus_profile_id,
        "metadata_conditioning": False,
        "system_parameter_noise": False,
        "annual_restart_ensemble": True,
        "policy_datasets": {
            "setpoint": str(args.setpoint_dataset),
            "force_off": str(args.force_off_dataset),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "architectures": [asdict(value) for value in architectures],
        "input_columns": closed_loop_input_columns(include_hp_force_off=True),
        "target_columns": CLOSED_LOOP_TARGET_COLUMNS,
    }
    (args.output_dir / "architecture_search_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    rows: list[dict[str, Any]] = []
    fitted: list[FittedCandidate] = []
    contexts: dict[int, dict[str, Any]] = {}
    for profile_index, profile_id in enumerate(profile_ids, start=1):
        print(
            f"starting_profile={profile_index}/{len(profile_ids)} profile={profile_id} "
            f"focus={profile_id == focus_profile_id}",
            flush=True,
        )
        profile_rows, profile_fitted, context = _run_profile_search(
            profile_id,
            architectures,
            args,
        )
        rows.extend(profile_rows)
        fitted.extend(profile_fitted)
        contexts[profile_id] = context

    runs = pd.DataFrame(rows)
    runs_path = args.output_dir / "architecture_search_runs.csv"
    runs.to_csv(runs_path, index=False)
    identity_columns = ["profile_id", "architecture", "hp", "hydronic", "thermal"]
    numeric_columns = [
        column
        for column in runs.select_dtypes(include=[np.number]).columns
        if column not in {"restart", "profile_id"}
    ]
    profile_results = runs.groupby(identity_columns, as_index=False)[numeric_columns].mean()
    score_std = (
        runs.groupby(identity_columns)["selection_score"]
        .std(ddof=1)
        .fillna(0.0)
        .rename("selection_score_std")
        .reset_index()
    )
    profile_results = profile_results.merge(score_std, on=identity_columns, how="left")
    ensemble_rows: list[dict[str, Any]] = []
    for (profile_id, architecture_name), values in itertools.groupby(
        sorted(fitted, key=lambda value: (value.profile_id, value.architecture.name)),
        key=lambda value: (value.profile_id, value.architecture.name),
    ):
        grouped = list(values)
        context = contexts[profile_id]
        force_off_profile = context["force_off_profile"]
        ensemble_rows.append(
            {
                "profile_id": profile_id,
                "architecture": architecture_name,
                **_annual_ensemble_metrics(
                    np.stack([value.annual_force_off_prediction for value in grouped]),
                    force_off_profile.targets,
                    force_off_profile.datetime,
                    context["target_scale"],
                ),
            }
        )
    profile_results = profile_results.merge(
        pd.DataFrame(ensemble_rows),
        on=["profile_id", "architecture"],
        how="left",
    )
    profile_results["restarts"] = args.restarts
    profile_results["held_out_profile_rank_pct"] = profile_results.groupby("profile_id")[
        "selection_score"
    ].rank(pct=True, method="average")
    profile_results["annual_profile_rank_pct"] = profile_results.groupby("profile_id")[
        "annual_force_off_stability_score"
    ].rank(pct=True, method="average")
    profile_results_path = args.output_dir / "architecture_search_profile_results.csv"
    profile_results.to_csv(profile_results_path, index=False)

    architecture_identity = ["architecture", "hp", "hydronic", "thermal"]
    aggregate_columns = [
        column
        for column in profile_results.select_dtypes(include=[np.number]).columns
        if column not in {"profile_id", "restarts"}
    ]
    results = profile_results.groupby(architecture_identity, as_index=False)[aggregate_columns].mean()
    results["evaluated_profiles"] = len(profile_ids)
    focus_columns = [
        "architecture",
        "selection_score",
        "annual_force_off_stability_score",
        "annual_force_off_winter_temperature_rmse_c",
        "annual_force_off_winter_temperature_restart_band_mean_width_c",
    ]
    focus = profile_results.loc[
        profile_results["profile_id"] == focus_profile_id,
        focus_columns,
    ].rename(
        columns={
            column: f"focus_{column}"
            for column in focus_columns
            if column != "architecture"
        }
    )
    results = results.merge(focus, on="architecture", how="left")
    results["general_held_out_rank_pct"] = results["selection_score"].rank(pct=True)
    results["general_annual_rank_pct"] = results["annual_force_off_stability_score"].rank(pct=True)
    results["focus_held_out_rank_pct"] = results["focus_selection_score"].rank(pct=True)
    results["focus_annual_rank_pct"] = results[
        "focus_annual_force_off_stability_score"
    ].rank(pct=True)
    results["joint_rank_score"] = results[
        [
            "general_held_out_rank_pct",
            "general_annual_rank_pct",
            "focus_held_out_rank_pct",
            "focus_annual_rank_pct",
        ]
    ].mean(axis=1)
    results = results.sort_values("joint_rank_score").reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results_path = args.output_dir / "architecture_search_results.csv"
    results.to_csv(results_path, index=False)
    best_name = str(results.iloc[0]["architecture"])
    best_focus_score = float(results.iloc[0]["focus_selection_score"])
    matching_fits = [
        value
        for value in fitted
        if value.profile_id == focus_profile_id and value.architecture.name == best_name
    ]
    representative = min(
        matching_fits,
        key=lambda value: abs(value.selection_score - best_focus_score),
    )
    eqx.tree_serialise_leaves(args.output_dir / "best_candidate.eqx", representative.params)
    (args.output_dir / "best_candidate.json").write_text(
        json.dumps(
            {
                "profile_id": focus_profile_id,
                "architecture": asdict(representative.architecture),
                "representative_restart": representative.restart,
                "metrics": {
                    key: float(value)
                    for key, value in results.iloc[0].items()
                    if isinstance(value, (float, np.floating, int, np.integer))
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    effects = _component_effects(results)
    effects_path = args.output_dir / "architecture_component_effects.csv"
    effects.to_csv(effects_path, index=False)
    dashboard = _write_dashboard(results, effects, args.output_dir)
    traces_path = _write_best_traces(
        representative.architecture,
        representative.held_out_traces,
        args.output_dir,
    )
    focus_fitted = [value for value in fitted if value.profile_id == focus_profile_id]
    annual_architectures = results.head(args.annual_plot_candidates)["architecture"].tolist()
    annual_path = _write_focus_annual_force_off_traces(
        focus_fitted,
        annual_architectures,
        contexts[focus_profile_id]["force_off_profile"],
        args.output_dir,
    )
    if args.reference_html is not None:
        reference_metrics = _reference_failure_metrics(args.reference_html)
        reference_path = args.output_dir / "reference_failure_metrics.json"
        reference_path.write_text(json.dumps(reference_metrics, indent=2, sort_keys=True))
        print(f"reference_failure_metrics={json.dumps(reference_metrics, sort_keys=True)}")
        print(f"saved_reference_metrics={reference_path}")
    print(f"best_architecture={best_name} score={results.iloc[0]['selection_score']:.5f}")
    print(f"saved_runs={runs_path}")
    print(f"saved_profile_results={profile_results_path}")
    print(f"saved_results={results_path}")
    print(f"saved_component_effects={effects_path}")
    print(f"saved_dashboard={dashboard}")
    print(f"saved_best_traces={traces_path}")
    print(f"saved_focus_annual_traces={annual_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setpoint-dataset",
        type=Path,
        default=Path("data/control=setpoint_identification_dhw_fixed"),
    )
    parser.add_argument(
        "--force-off-dataset",
        type=Path,
        default=Path("data/control=force_off"),
    )
    parser.add_argument("--profile-id", type=int, default=None)
    parser.add_argument(
        "--profile-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit EGIDs to compare. --profile-id remains a single-EGID alias.",
    )
    parser.add_argument(
        "--focus-profile-id",
        type=int,
        default=None,
        help="EGID receiving dedicated full-year plots and focus-specific ranks.",
    )
    parser.add_argument(
        "--num-profiles",
        type=int,
        default=1,
        help="Total number of shared EGIDs; random background EGIDs supplement explicit IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/neural_building_emulator_refactor/single_building_architecture_search"),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Independent initializations per architecture; rankings use their mean.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=384, help="Training rollout length in 15-minute steps.")
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--train-block-days", type=int, default=28)
    parser.add_argument("--test-block-days", type=int, default=14)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--temperature-delta-cap", type=float, default=2.0)
    parser.add_argument(
        "--hp-recovery-emergency-threshold-c",
        type=float,
        default=1.0,
        help=(
            "Thermostat deficit above which recovery_hurdle structurally "
            "increases activation; force-off debt remains active below it."
        ),
    )
    parser.add_argument(
        "--hydronic-capacity-min-hours",
        type=float,
        default=0.25,
        help="Lower bound for the learned leaky-emitter heat capacity, expressed at Q_cap.",
    )
    parser.add_argument(
        "--hydronic-capacity-max-hours",
        type=float,
        default=24.0,
        help="Upper bound for the learned leaky-emitter heat capacity, expressed at Q_cap.",
    )
    parser.add_argument(
        "--hydronic-capacity-init-hours",
        type=float,
        default=2.0,
        help="Initial learned leaky-emitter heat capacity, expressed at Q_cap.",
    )
    parser.add_argument("--mode-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--controller-training-feedback",
        choices=("closed_loop", "temperature", "temperature_and_history"),
        default="closed_loop",
        help=(
            "Diagnostic controller training mode. Non-closed modes optimize "
            "only Qroom/Pel while supplying observed previous feedback."
        ),
    )
    parser.add_argument(
        "--controller-calendar-mode",
        choices=("all", "hour_only", "none"),
        default="all",
        help=(
            "Calendar features exposed to HP/hydronic networks. Thermal "
            "disturbance features are unchanged."
        ),
    )
    parser.add_argument("--event-horizon-hours", type=float, default=3.0)
    parser.add_argument("--event-tin-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--observed-q-thermal-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of an autonomous Tin rollout driven by observed Qroom; "
            "supported by bounded_mlp variants and passive_memory_positive_q."
        ),
    )
    parser.add_argument(
        "--qroom-window-mean-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight applied to the squared normalized bias of mean Qroom "
            "over each training rollout window."
        ),
    )
    parser.add_argument(
        "--event-tin-loss-horizon-hours", type=float, default=3.0
    )
    parser.add_argument(
        "--event-tin-loss-setpoint-threshold-c", type=float, default=0.05
    )
    parser.add_argument(
        "--warmup-hours",
        type=float,
        default=24.0,
        help="Autonomous context preceding each held-out block, used only to initialize recurrent state.",
    )
    parser.add_argument("--event-score-weight", type=float, default=0.25)
    parser.add_argument(
        "--hp-architectures",
        nargs="+",
        choices=HP_CHOICES,
        default=HP_KINDS,
    )
    parser.add_argument(
        "--hydronic-architectures",
        nargs="+",
        choices=HYDRONIC_CHOICES,
        default=HYDRONIC_KINDS,
    )
    parser.add_argument(
        "--thermal-architectures",
        nargs="+",
        choices=THERMAL_CHOICES,
        default=THERMAL_KINDS,
    )
    parser.add_argument("--max-candidates", type=int, default=0, help="Zero evaluates the complete factorial.")
    parser.add_argument(
        "--annual-plot-candidates",
        type=int,
        default=4,
        help="Number of joint-ranked architectures shown in the focus full-year plot.",
    )
    parser.add_argument(
        "--reference-html",
        type=Path,
        default=None,
        help="Existing probabilistic profile HTML whose annual failure metrics are extracted.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.restarts < 1 or args.batch_size < 1 or args.sequence_length < 2:
        parser.error("--epochs, --restarts, --batch-size and --sequence-length must be positive")
    if args.stride < 1 or args.train_block_days < 1 or args.test_block_days < 1:
        parser.error("stride and block lengths must be positive")
    if args.dt_hours <= 0.0 or args.temperature_delta_cap <= 0.0:
        parser.error("--dt-hours and --temperature-delta-cap must be positive")
    if args.hp_recovery_emergency_threshold_c < 0.0:
        parser.error("--hp-recovery-emergency-threshold-c must be non-negative")
    if args.observed_q_thermal_loss_weight < 0.0:
        parser.error("--observed-q-thermal-loss-weight must be non-negative")
    if args.qroom_window_mean_loss_weight < 0.0:
        parser.error("--qroom-window-mean-loss-weight must be non-negative")
    if (
        args.controller_training_feedback != "closed_loop"
        and args.observed_q_thermal_loss_weight > 0.0
    ):
        parser.error(
            "Controller teacher-forcing diagnostics cannot be combined with "
            "--observed-q-thermal-loss-weight"
        )
    observed_q_supported_thermal = {
        "bounded_mlp",
        "bounded_mlp_feedback",
        "bounded_mlp_dual_alpha",
        "bounded_mlp_physical_alpha",
        "passive_memory_positive_q",
    }
    if (
        args.observed_q_thermal_loss_weight > 0.0
        and any(
            thermal not in observed_q_supported_thermal
            for thermal in args.thermal_architectures
        )
    ):
        parser.error(
            "--observed-q-thermal-loss-weight requires bounded_mlp variants "
            "or passive_memory_positive_q"
        )
    if not (
        0.0 < args.hydronic_capacity_min_hours
        < args.hydronic_capacity_init_hours
        < args.hydronic_capacity_max_hours
    ):
        parser.error(
            "hydronic capacity bounds must satisfy 0 < min < init < max"
        )
    if args.warmup_hours < 0.0:
        parser.error("--warmup-hours must be non-negative")
    if args.max_candidates < 0:
        parser.error("--max-candidates must be non-negative")
    if args.num_profiles < 1 or args.annual_plot_candidates < 1:
        parser.error("--num-profiles and --annual-plot-candidates must be positive")
    if args.profile_id is not None and args.profile_ids is not None:
        parser.error("Use either --profile-id or --profile-ids, not both")
    if args.reference_html is not None and not args.reference_html.exists():
        parser.error(f"--reference-html does not exist: {args.reference_html}")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
