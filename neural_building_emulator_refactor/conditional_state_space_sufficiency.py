"""Sufficiency and unseen-building tests for conditional thermal state spaces.

Each variant receives an observed thermal driver (Qroom, Pel_SH, or
COP(Tout)*Pel_SH), so HP/controller prediction errors cannot hide whether the
thermal state-space class can reproduce autonomous Tin trajectories. With
``--test-fraction 0`` this is an in-sample approximation-capacity test;
positive fractions hold out complete buildings and test metadata generalization.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    CLOSED_LOOP_METADATA_COLUMNS,
    HP_SIZE_BINDING_COLUMN,
    PROFILE_ID_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    closed_loop_input_columns,
    closed_loop_required_columns,
)
from neural_building_emulator.data import (
    ClosedLoopProfile,
    read_dataset_frame,
    read_profile_ids,
    select_profile_ids,
    split_profile_ids,
    to_closed_loop_profiles,
)


Variant = Literal[
    "local_fixed",
    "metadata_fixed",
    "metadata_fixed_positive_leaky",
    "metadata_fixed_pel",
    "metadata_fixed_pel_cop_positive_leaky",
    "exogenous_dynamic",
]
VARIANTS: tuple[Variant, ...] = (
    "local_fixed",
    "metadata_fixed",
    "metadata_fixed_positive_leaky",
    "metadata_fixed_pel",
    "metadata_fixed_pel_cop_positive_leaky",
    "exogenous_dynamic",
)

POSITIVE_LEAKY_VARIANTS = (
    "metadata_fixed_positive_leaky",
    "metadata_fixed_pel_cop_positive_leaky",
)
PEL_VARIANTS = (
    "metadata_fixed_pel",
    "metadata_fixed_pel_cop_positive_leaky",
)

INPUT_NAMES = closed_loop_input_columns(include_hp_force_off=True)
INPUT_INDICES = {name: index for index, name in enumerate(INPUT_NAMES)}
TOUT_INDEX = INPUT_INDICES[
    "Environment, Site Outdoor Air Drybulb Temperature"
]
SOLAR_INDEX = INPUT_INDICES[
    "Environment, Site Global Horizontal Solar Radiation Rate per Area"
]
VENT_INDEX = INPUT_INDICES[
    "FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate"
]
INTERNAL_INDEX = INPUT_INDICES["input_non_people_internal_gain_net_w_per_m2"]
OCCUPANCY_INDEX = INPUT_INDICES["input_occupants_present_per_m2"]
AVAILABILITY_INDEX = INPUT_INDICES[SPACE_HEATING_AVAILABILITY_COLUMN]
HOUR_SIN_INDEX = INPUT_INDICES["hour_sin"]
HOUR_COS_INDEX = INPUT_INDICES["hour_cos"]
DAY_SIN_INDEX = INPUT_INDICES["day_of_year_sin"]
DAY_COS_INDEX = INPUT_INDICES["day_of_year_cos"]

EXOGENOUS_INDICES = (
    TOUT_INDEX,
    SOLAR_INDEX,
    VENT_INDEX,
    INTERNAL_INDEX,
    OCCUPANCY_INDEX,
    AVAILABILITY_INDEX,
    HOUR_SIN_INDEX,
    HOUR_COS_INDEX,
    DAY_SIN_INDEX,
    DAY_COS_INDEX,
)
HP_REF_COP_METADATA_INDEX = tuple(CLOSED_LOOP_METADATA_COLUMNS).index("hp_ref_cop")


@dataclass(frozen=True)
class Trajectory:
    policy: str
    building_index: int
    profile: ClosedLoopProfile


@dataclass(frozen=True)
class Scales:
    metadata_mean: np.ndarray
    metadata_std: np.ndarray
    exogenous_mean: np.ndarray
    exogenous_std: np.ndarray
    temperature_mean: float
    temperature_std: float
    qroom_scale: float
    pel_scale: float


def _safe_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    return mean, std


def _load_profiles(
    dataset: Path,
    profile_ids: Sequence[int],
) -> dict[int, ClosedLoopProfile]:
    columns = closed_loop_required_columns(
        tuple(CLOSED_LOOP_METADATA_COLUMNS),
        include_internal_gains=True,
        include_dhw_request=True,
        include_hp_force_off=True,
    )
    frame = read_dataset_frame(dataset, columns=columns, profile_ids=profile_ids)
    profiles = to_closed_loop_profiles(
        frame,
        include_space_heating_availability=True,
        include_internal_gains=True,
        include_dhw_request=True,
        include_hp_force_off=True,
        hp_power_area_normalization="building_heated_area",
        metadata_columns=tuple(CLOSED_LOOP_METADATA_COLUMNS),
    )
    result = {int(profile.profile_id): profile for profile in profiles}
    missing = [profile_id for profile_id in profile_ids if profile_id not in result]
    if missing:
        raise ValueError(f"Missing profiles {missing} in {dataset}")
    return result


def _selected_profile_split(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    common_ids = tuple(
        sorted(
            set(read_profile_ids(args.setpoint_dataset)).intersection(
                read_profile_ids(args.force_off_dataset)
            )
        )
    )
    if args.profile_filter == "sh_hp":
        eligible: set[int] | None = None
        for dataset in (args.setpoint_dataset, args.force_off_dataset):
            frame = read_dataset_frame(
                dataset,
                columns=(
                    PROFILE_ID_COLUMN,
                    "hp_ref_capacity_W",
                    HP_SIZE_BINDING_COLUMN,
                ),
            )
            first = frame.groupby(PROFILE_ID_COLUMN, sort=True).first()
            dataset_eligible = {
                int(profile_id)
                for profile_id, row in first.iterrows()
                if pd.notna(row["hp_ref_capacity_W"])
                and float(row["hp_ref_capacity_W"]) > 0.0
                and str(row[HP_SIZE_BINDING_COLUMN]).strip().upper() == "SH"
            }
            eligible = (
                dataset_eligible
                if eligible is None
                else eligible.intersection(dataset_eligible)
            )
        common_ids = tuple(
            profile_id for profile_id in common_ids if profile_id in (eligible or set())
        )
    if args.profile_ids is not None:
        requested = tuple(sorted(int(value) for value in args.profile_ids))
        missing = tuple(sorted(set(requested).difference(common_ids)))
        if missing:
            raise ValueError(f"Profiles are not shared by both datasets: {missing}")
        selected_ids = requested
    else:
        selected_ids = select_profile_ids(
            common_ids,
            max_profiles=args.max_profiles,
            seed=args.seed,
            strategy=args.profile_selection,
        )
    if args.test_fraction > 0.0:
        train_ids, test_ids = split_profile_ids(
            selected_ids,
            test_fraction=args.test_fraction,
            seed=args.seed + 1,
        )
    else:
        train_ids, test_ids = selected_ids, ()
    return selected_ids, train_ids, test_ids


def _make_trajectories(
    profile_ids: Sequence[int],
    *,
    setpoint: dict[int, ClosedLoopProfile],
    force_off: dict[int, ClosedLoopProfile],
    building_indices: dict[int, int],
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for profile_id in profile_ids:
        building_index = building_indices[int(profile_id)]
        trajectories.append(
            Trajectory("setpoint", building_index, setpoint[profile_id])
        )
        trajectories.append(
            Trajectory("force_off", building_index, force_off[profile_id])
        )
    return trajectories


def _load_trajectory_split(
    args: argparse.Namespace,
) -> tuple[
    list[Trajectory],
    list[Trajectory],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    selected_ids, train_ids, test_ids = _selected_profile_split(args)
    setpoint = _load_profiles(args.setpoint_dataset, selected_ids)
    force_off = _load_profiles(args.force_off_dataset, selected_ids)
    building_indices = {
        profile_id: index for index, profile_id in enumerate(selected_ids)
    }
    train = _make_trajectories(
        train_ids,
        setpoint=setpoint,
        force_off=force_off,
        building_indices=building_indices,
    )
    evaluation_ids = test_ids if test_ids else selected_ids
    evaluation = _make_trajectories(
        evaluation_ids,
        setpoint=setpoint,
        force_off=force_off,
        building_indices=building_indices,
    )
    return train, evaluation, selected_ids, train_ids, test_ids


def _fit_scales(trajectories: Sequence[Trajectory]) -> Scales:
    metadata = np.stack(
        [
            next(
                trajectory.profile.metadata
                for trajectory in trajectories
                if trajectory.building_index == building_index
            )
            for building_index in sorted(
                {trajectory.building_index for trajectory in trajectories}
            )
        ]
    )
    exogenous = np.concatenate(
        [trajectory.profile.inputs[:, EXOGENOUS_INDICES] for trajectory in trajectories]
    )
    temperature = np.concatenate(
        [trajectory.profile.targets[:, 0] for trajectory in trajectories]
    )
    qroom = np.concatenate(
        [trajectory.profile.targets[:, 1] for trajectory in trajectories]
    )
    pel = np.concatenate(
        [trajectory.profile.targets[:, 2] for trajectory in trajectories]
    )
    metadata_mean, metadata_std = _safe_mean_std(metadata)
    exogenous_mean, exogenous_std = _safe_mean_std(exogenous)
    return Scales(
        metadata_mean=metadata_mean,
        metadata_std=metadata_std,
        exogenous_mean=exogenous_mean,
        exogenous_std=exogenous_std,
        temperature_mean=float(np.mean(temperature)),
        temperature_std=max(float(np.std(temperature)), 0.25),
        qroom_scale=max(float(np.sqrt(np.mean(np.square(qroom)))), 1e-3),
        pel_scale=max(float(np.sqrt(np.mean(np.square(pel)))), 1e-3),
    )


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length < window:
        return []
    starts = list(range(0, length - window + 1, stride))
    final_start = length - window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _make_windows(
    trajectories: Sequence[Trajectory],
    *,
    sequence_length: int,
    stride: int,
) -> dict[str, np.ndarray]:
    building_indices: list[int] = []
    metadata: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    qroom: list[np.ndarray] = []
    pel: list[np.ndarray] = []
    target: list[np.ndarray] = []
    initial: list[float] = []
    for trajectory in trajectories:
        profile = trajectory.profile
        for start in _window_starts(len(profile.targets), sequence_length, stride):
            end = start + sequence_length
            building_indices.append(trajectory.building_index)
            metadata.append(profile.metadata)
            inputs.append(profile.inputs[start:end])
            qroom.append(profile.targets[start:end, 1])
            pel.append(profile.targets[start:end, 2])
            target.append(profile.targets[start:end, 0])
            initial.append(float(profile.initial_temperature[start, 0]))
    if not inputs:
        raise ValueError("No fitting windows were generated")
    return {
        "building_index": np.asarray(building_indices, dtype=np.int32),
        "metadata": np.stack(metadata).astype(np.float32),
        "inputs": np.stack(inputs).astype(np.float32),
        "qroom": np.stack(qroom).astype(np.float32),
        "pel": np.stack(pel).astype(np.float32),
        "target": np.stack(target).astype(np.float32),
        "initial": np.asarray(initial, dtype=np.float32),
    }


def _init_mlp(
    key: jax.Array,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
    *,
    final_scale: float = 1.0,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    keys = jax.random.split(key, depth + 1)
    dims = (input_dim, *(hidden_dim for _ in range(depth)), output_dim)
    layers = []
    for index, (layer_key, fan_in, fan_out) in enumerate(
        zip(keys, dims[:-1], dims[1:])
    ):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        scale = final_scale if index == len(dims) - 2 else 1.0
        weight = scale * jax.random.uniform(
            layer_key,
            (fan_out, fan_in),
            minval=-limit,
            maxval=limit,
        )
        layers.append((weight, jnp.zeros((fan_out,), dtype=jnp.float32)))
    return tuple(layers)


def _mlp(
    layers: tuple[tuple[jax.Array, jax.Array], ...], values: jax.Array
) -> jax.Array:
    hidden = values
    for weight, bias in layers[:-1]:
        hidden = jax.nn.silu(weight @ hidden + bias)
    weight, bias = layers[-1]
    return weight @ hidden + bias


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def _base_state_space_raw(
    state_dim: int,
    encoded_dim: int,
    *,
    gamma: float,
    dt_hours: float,
) -> np.ndarray:
    time_constants = np.geomspace(1.0, 24.0 * 21.0, state_dim)
    rho = np.minimum(np.exp(-dt_hours / time_constants), gamma - 1e-4)
    rho_fraction = (rho - 0.05) / (gamma - 0.05)
    rho_raw = _logit(rho_fraction).astype(np.float32)
    b_matrix = np.zeros((state_dim, encoded_dim), dtype=np.float32)
    bias = np.zeros((state_dim,), dtype=np.float32)
    output = np.zeros((state_dim,), dtype=np.float32)
    output[0] = 1.0
    return np.concatenate(
        [rho_raw, b_matrix.reshape(-1), bias, output, np.zeros((1,), np.float32)]
    )


def _parameter_sizes(state_dim: int, encoded_dim: int) -> tuple[int, int]:
    static_size = state_dim + state_dim * encoded_dim + state_dim + state_dim + 1
    dynamic_size = state_dim + state_dim * encoded_dim + state_dim
    return static_size, dynamic_size


def _initial_parameters(
    variant: Variant,
    *,
    key: jax.Array,
    num_buildings: int,
    metadata_dim: int,
    exogenous_dim: int,
    state_dim: int,
    encoded_dim: int,
    hidden_dim: int,
    depth: int,
    gamma: float,
    dt_hours: float,
    q_response_modes: int = 6,
) -> dict[str, Any]:
    encoder_key, static_key, dynamic_key, local_key = jax.random.split(key, 4)
    static_size, dynamic_size = _parameter_sizes(state_dim, encoded_dim)
    params: dict[str, Any] = {
        "encoder": _init_mlp(
            encoder_key,
            exogenous_dim + 3,
            hidden_dim,
            encoded_dim,
            depth,
        ),
        "base_state_space": jnp.asarray(
            _base_state_space_raw(
                state_dim,
                encoded_dim,
                gamma=gamma,
                dt_hours=dt_hours,
            )
        ),
    }
    if variant == "local_fixed":
        params["local_delta"] = 1e-3 * jax.random.normal(
            local_key, (num_buildings, static_size)
        )
    else:
        params["metadata_net"] = _init_mlp(
            static_key,
            metadata_dim,
            hidden_dim,
            static_size,
            depth,
            final_scale=1e-3,
        )
    if variant == "exogenous_dynamic":
        params["dynamic_net"] = _init_mlp(
            dynamic_key,
            metadata_dim + exogenous_dim,
            hidden_dim,
            dynamic_size,
            depth,
            final_scale=1e-3,
        )
    if variant in POSITIVE_LEAKY_VARIANTS:
        params["q_response_net"] = _init_mlp(
            jax.random.fold_in(dynamic_key, 1),
            metadata_dim,
            hidden_dim,
            q_response_modes + 1,
            depth,
            final_scale=1e-3,
        )
        params["q_response_init_net"] = _init_mlp(
            jax.random.fold_in(dynamic_key, 2),
            metadata_dim + 1,
            hidden_dim,
            q_response_modes,
            depth,
            final_scale=1e-3,
        )
    return params


def _split_static_raw(
    raw: jax.Array,
    *,
    state_dim: int,
    encoded_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    cursor = 0
    rho_raw = raw[cursor : cursor + state_dim]
    cursor += state_dim
    matrix_size = state_dim * encoded_dim
    b_matrix_raw = raw[cursor : cursor + matrix_size].reshape(
        state_dim, encoded_dim
    )
    cursor += matrix_size
    bias_raw = raw[cursor : cursor + state_dim]
    cursor += state_dim
    output_raw = raw[cursor : cursor + state_dim]
    cursor += state_dim
    output_bias_raw = raw[cursor]
    return rho_raw, b_matrix_raw, bias_raw, output_raw, output_bias_raw


def _state_space_parameters(
    params: dict[str, Any],
    variant: Variant,
    building_index: jax.Array,
    metadata: jax.Array,
    exogenous: jax.Array,
    *,
    state_dim: int,
    encoded_dim: int,
    gamma: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if variant == "local_fixed":
        raw = params["base_state_space"] + params["local_delta"][building_index]
    else:
        raw = params["base_state_space"] + _mlp(params["metadata_net"], metadata)
    rho_raw, b_matrix_raw, bias_raw, output_raw, output_bias_raw = _split_static_raw(
        raw, state_dim=state_dim, encoded_dim=encoded_dim
    )
    if variant == "exogenous_dynamic":
        dynamic = _mlp(
            params["dynamic_net"], jnp.concatenate((metadata, exogenous))
        )
        cursor = 0
        rho_raw = rho_raw + 0.75 * jnp.tanh(dynamic[cursor : cursor + state_dim])
        cursor += state_dim
        matrix_size = state_dim * encoded_dim
        b_matrix_raw = b_matrix_raw + 0.25 * jnp.tanh(
            dynamic[cursor : cursor + matrix_size]
        ).reshape(state_dim, encoded_dim)
        cursor += matrix_size
        bias_raw = bias_raw + 0.10 * jnp.tanh(
            dynamic[cursor : cursor + state_dim]
        )

    rho = 0.05 + (gamma - 0.05) * jax.nn.sigmoid(rho_raw)
    b_matrix = 0.30 * jnp.tanh(b_matrix_raw)
    bias = 0.15 * jnp.tanh(bias_raw)
    output = jnp.tanh(output_raw)
    output_bias = 2.0 * jnp.tanh(output_bias_raw)
    return rho, b_matrix, bias, output, output_bias


def _normalized_context(
    row: jax.Array,
    temperature_normalized: jax.Array,
    thermal_driver_normalized: jax.Array,
    scales: Scales,
) -> tuple[jax.Array, jax.Array]:
    exogenous = (
        row[jnp.asarray(EXOGENOUS_INDICES)]
        - jnp.asarray(scales.exogenous_mean)
    ) / jnp.asarray(scales.exogenous_std)
    outdoor_normalized = (
        row[TOUT_INDEX] - scales.temperature_mean
    ) / scales.temperature_std
    gap_normalized = outdoor_normalized - temperature_normalized
    encoder_input = jnp.concatenate(
        (
            exogenous,
            jnp.asarray(
                [
                    temperature_normalized,
                    gap_normalized,
                    thermal_driver_normalized,
                ]
            ),
        )
    )
    return exogenous, encoder_input


def _rollout(
    params: dict[str, Any],
    variant: Variant,
    building_index: jax.Array,
    metadata_raw: jax.Array,
    inputs: jax.Array,
    qroom: jax.Array,
    pel: jax.Array,
    initial_temperature: jax.Array,
    scales: Scales,
    *,
    state_dim: int,
    encoded_dim: int,
    gamma: float,
    bptt_truncate_steps: int,
    dt_hours: float,
    q_to_t_time_constants_hours: tuple[float, ...],
    q_to_t_gain_min: float,
    q_to_t_gain_max: float,
    pel_cop_slope_per_c: float,
    pel_cop_reference_outdoor_c: float,
    pel_cop_floor: float,
    pel_cop_cap: float,
) -> jax.Array:
    metadata = (
        metadata_raw - jnp.asarray(scales.metadata_mean)
    ) / jnp.asarray(scales.metadata_std)
    initial_y = (
        initial_temperature - scales.temperature_mean
    ) / scales.temperature_std
    initial_exogenous, _ = _normalized_context(
        inputs[0], initial_y, jnp.asarray(0.0, dtype=initial_y.dtype), scales
    )
    _, _, _, output, output_bias = _state_space_parameters(
        params,
        variant,
        building_index,
        metadata,
        initial_exogenous,
        state_dim=state_dim,
        encoded_dim=encoded_dim,
        gamma=gamma,
    )
    state = output * ((initial_y - output_bias) / (jnp.dot(output, output) + 1e-5))
    q_response = jnp.zeros(
        (len(q_to_t_time_constants_hours),), dtype=initial_y.dtype
    )
    if variant in POSITIVE_LEAKY_VARIANTS:
        q_raw = _mlp(params["q_response_net"], metadata)
        q_gain = q_to_t_gain_min + (
            q_to_t_gain_max - q_to_t_gain_min
        ) * jax.nn.sigmoid(q_raw[0] - 3.0)
        q_weights = jax.nn.softmax(q_raw[1:])
        q_initial_raw = _mlp(
            params["q_response_init_net"],
            jnp.concatenate((metadata, jnp.atleast_1d(initial_y))),
        )
        response_reference_scale = scales.qroom_scale
        if variant == "metadata_fixed_pel_cop_positive_leaky":
            reference_cop = jnp.clip(
                metadata_raw[HP_REF_COP_METADATA_INDEX],
                pel_cop_floor,
                pel_cop_cap,
            )
            response_reference_scale = scales.pel_scale * reference_cop
        q_response = (
            q_weights
            * q_gain
            * response_reference_scale
            * jax.nn.sigmoid(q_initial_raw - 4.0)
        )
        q_retention = jnp.exp(
            -dt_hours
            / jnp.asarray(q_to_t_time_constants_hours, dtype=initial_y.dtype)
        )
    else:
        q_gain = jnp.asarray(0.0, dtype=initial_y.dtype)
        q_weights = jnp.zeros_like(q_response)
        q_retention = jnp.zeros_like(q_response)

    def step(carry: tuple[jax.Array, jax.Array, jax.Array], values):
        latent, temperature_normalized, q_response_state = carry
        step_index, row, heat, electric_power = values
        should_detach = (step_index > 0) & (
            jnp.mod(step_index, bptt_truncate_steps) == 0
        )
        latent = jax.lax.cond(
            should_detach, jax.lax.stop_gradient, lambda value: value, latent
        )
        temperature_normalized = jax.lax.cond(
            should_detach,
            jax.lax.stop_gradient,
            lambda value: value,
            temperature_normalized,
        )
        q_response_state = jax.lax.cond(
            should_detach,
            jax.lax.stop_gradient,
            lambda value: value,
            q_response_state,
        )
        if variant in POSITIVE_LEAKY_VARIANTS:
            encoder_driver = jnp.asarray(0.0, dtype=heat.dtype)
        elif variant == "metadata_fixed_pel":
            encoder_driver = electric_power / scales.pel_scale
        else:
            encoder_driver = heat / scales.qroom_scale
        exogenous, encoder_input = _normalized_context(
            row, temperature_normalized, encoder_driver, scales
        )
        encoded = jnp.tanh(_mlp(params["encoder"], encoder_input))
        rho, b_matrix, bias, output, output_bias = _state_space_parameters(
            params,
            variant,
            building_index,
            metadata,
            exogenous,
            state_dim=state_dim,
            encoded_dim=encoded_dim,
            gamma=gamma,
        )
        next_latent = rho * latent + b_matrix @ encoded + bias
        raw_temperature = jnp.dot(output, next_latent) + output_bias
        free_temperature_normalized = 5.0 * jnp.tanh(raw_temperature / 5.0)
        response_driver = jnp.maximum(heat, 0.0)
        if variant == "metadata_fixed_pel_cop_positive_leaky":
            cop_t = jnp.clip(
                metadata_raw[HP_REF_COP_METADATA_INDEX]
                + pel_cop_slope_per_c
                * (row[TOUT_INDEX] - pel_cop_reference_outdoor_c),
                pel_cop_floor,
                pel_cop_cap,
            )
            response_driver = cop_t * jnp.maximum(electric_power, 0.0)
        q_response_next = (
            q_retention * q_response_state
            + (1.0 - q_retention)
            * q_weights
            * q_gain
            * response_driver
        )
        combined_temperature_normalized = free_temperature_normalized + (
            jnp.sum(q_response_next) / scales.temperature_std
        )
        next_temperature_normalized = 5.0 * jnp.tanh(
            combined_temperature_normalized / 5.0
        )
        temperature = (
            scales.temperature_mean
            + scales.temperature_std * next_temperature_normalized
        )
        return (
            next_latent,
            next_temperature_normalized,
            q_response_next,
        ), temperature

    step_indices = jnp.arange(inputs.shape[0], dtype=jnp.int32)
    _, prediction = jax.lax.scan(
        step,
        (state, initial_y, q_response),
        (step_indices, inputs, qroom, pel),
    )
    return prediction


def _parameter_count(params: dict[str, Any]) -> int:
    return int(
        sum(
            value.size
            for value in jax.tree_util.tree_leaves(params)
            if hasattr(value, "size")
        )
    )


def _fit_variant(
    variant: Variant,
    windows: dict[str, np.ndarray],
    scales: Scales,
    *,
    args: argparse.Namespace,
    num_buildings: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    params = _initial_parameters(
        variant,
        key=jax.random.PRNGKey(args.seed),
        num_buildings=num_buildings,
        metadata_dim=windows["metadata"].shape[-1],
        exogenous_dim=len(EXOGENOUS_INDICES),
        state_dim=args.state_dim,
        encoded_dim=args.encoded_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        gamma=args.gamma,
        dt_hours=args.dt_hours,
        q_response_modes=len(args.q_to_t_time_constants_hours),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adam(args.learning_rate),
    )
    optimizer_state = optimizer.init(params)

    @jax.jit
    def loss_and_grad(model_params, batch):
        prediction = jax.vmap(
            lambda building, metadata, inputs, qroom, pel, initial: _rollout(
                model_params,
                variant,
                building,
                metadata,
                inputs,
                qroom,
                pel,
                initial,
                scales,
                state_dim=args.state_dim,
                encoded_dim=args.encoded_dim,
                gamma=args.gamma,
                bptt_truncate_steps=args.bptt_truncate_steps,
                dt_hours=args.dt_hours,
                q_to_t_time_constants_hours=tuple(
                    args.q_to_t_time_constants_hours
                ),
                q_to_t_gain_min=args.q_to_t_gain_min,
                q_to_t_gain_max=args.q_to_t_gain_max,
                pel_cop_slope_per_c=args.pel_cop_slope_per_c,
                pel_cop_reference_outdoor_c=args.pel_cop_reference_outdoor_c,
                pel_cop_floor=args.pel_cop_floor,
                pel_cop_cap=args.pel_cop_cap,
            )
        )(
            batch["building_index"],
            batch["metadata"],
            batch["inputs"],
            batch["qroom"],
            batch["pel"],
            batch["initial"],
        )
        residual = (prediction - batch["target"]) / scales.temperature_std
        return jnp.mean(jnp.square(residual))

    value_and_grad = jax.jit(jax.value_and_grad(loss_and_grad))
    rng = np.random.default_rng(args.seed)
    count = len(windows["target"])
    best_loss = float("inf")
    best_params = params
    wait = 0
    max_grad = 0.0
    started = time.perf_counter()
    epoch = -1
    for epoch in range(args.epochs):
        permutation = rng.permutation(count)
        epoch_losses = []
        for start in range(0, count, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            if len(indices) < args.batch_size:
                indices = np.concatenate(
                    (indices, permutation[: args.batch_size - len(indices)])
                )
            batch = {
                name: jnp.asarray(values[indices]) for name, values in windows.items()
            }
            loss, gradients = value_and_grad(params, batch)
            grad_norm = float(optax.global_norm(gradients))
            if not np.isfinite(float(loss)) or not np.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite update for {variant} at epoch {epoch + 1}"
                )
            updates, optimizer_state = optimizer.update(
                gradients, optimizer_state, params
            )
            params = optax.apply_updates(params, updates)
            epoch_losses.append(float(loss))
            max_grad = max(max_grad, grad_norm)
        epoch_loss = float(np.mean(epoch_losses))
        if epoch_loss < best_loss - args.early_stopping_min_delta:
            best_loss = epoch_loss
            best_params = params
            wait = 0
        else:
            wait += 1
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"variant={variant} epoch={epoch + 1:04d} "
                f"loss={epoch_loss:.6f} best={best_loss:.6f} "
                f"max_grad={max_grad:.3g}",
                flush=True,
            )
        if wait >= args.early_stopping_patience:
            break
    return best_params, {
        "train_window_normalized_mse": best_loss,
        "epochs_completed": float(epoch + 1),
        "max_grad_norm": max_grad,
        "training_seconds": time.perf_counter() - started,
        "parameter_count": float(_parameter_count(best_params)),
    }


def _make_annual_evaluator(
    params: dict[str, Any],
    variant: Variant,
    scales: Scales,
    args: argparse.Namespace,
) -> Any:
    # No gradient truncation is required for evaluation, but using the same
    # forward operation keeps this exactly aligned with training.
    return jax.jit(
        lambda building_index, metadata, inputs, qroom, pel, initial: _rollout(
            params,
            variant,
            building_index,
            metadata,
            inputs,
            qroom,
            pel,
            initial,
            scales,
            state_dim=args.state_dim,
            encoded_dim=args.encoded_dim,
            gamma=args.gamma,
            bptt_truncate_steps=args.bptt_truncate_steps,
            dt_hours=args.dt_hours,
            q_to_t_time_constants_hours=tuple(
                args.q_to_t_time_constants_hours
            ),
            q_to_t_gain_min=args.q_to_t_gain_min,
            q_to_t_gain_max=args.q_to_t_gain_max,
            pel_cop_slope_per_c=args.pel_cop_slope_per_c,
            pel_cop_reference_outdoor_c=args.pel_cop_reference_outdoor_c,
            pel_cop_floor=args.pel_cop_floor,
            pel_cop_cap=args.pel_cop_cap,
        )
    )


def _annual_prediction(evaluate: Any, trajectory: Trajectory) -> np.ndarray:
    profile = trajectory.profile
    return np.asarray(
        evaluate(
            jnp.asarray(trajectory.building_index, dtype=jnp.int32),
            jnp.asarray(profile.metadata),
            jnp.asarray(profile.inputs),
            jnp.asarray(profile.targets[:, 1]),
            jnp.asarray(profile.targets[:, 2]),
            jnp.asarray(profile.initial_temperature[0, 0]),
        )
    )


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    datetime: np.ndarray,
) -> dict[str, float]:
    residual = prediction - target
    month = pd.DatetimeIndex(datetime).month.to_numpy()
    winter = np.isin(month, (1, 2, 3, 10, 11, 12))
    return {
        "annual_temperature_rmse_c": float(np.sqrt(np.mean(np.square(residual)))),
        "annual_temperature_mae_c": float(np.mean(np.abs(residual))),
        "annual_temperature_bias_c": float(np.mean(residual)),
        "annual_final_error_c": float(residual[-1]),
        "winter_temperature_rmse_c": float(
            np.sqrt(np.mean(np.square(residual[winter])))
        ),
        "prediction_min_c": float(np.min(prediction)),
        "prediction_max_c": float(np.max(prediction)),
    }


def _write_dashboard(
    frame: pd.DataFrame,
    traces: dict[tuple[str, int, str], np.ndarray],
    trajectories: Sequence[Trajectory],
    variants: Sequence[Variant],
    in_sample: bool,
    output_dir: Path,
) -> tuple[Path, Path]:
    colors = {
        "local_fixed": "#167d78",
        "metadata_fixed": "#d97706",
        "metadata_fixed_positive_leaky": "#355ea8",
        "metadata_fixed_pel": "#b44e64",
        "metadata_fixed_pel_cop_positive_leaky": "#4d7f3a",
        "exogenous_dynamic": "#775da6",
    }
    summary = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Annual autonomous Tin RMSE", "Annual Tin bias"),
    )
    for variant in variants:
        selected = frame[frame["variant"].eq(variant)]
        labels = selected.apply(
            lambda row: f"{int(row.profile_id)} / {row.policy}", axis=1
        )
        summary.add_trace(
            go.Bar(
                x=labels,
                y=selected["annual_temperature_rmse_c"],
                name=variant,
                marker_color=colors[variant],
            ),
            row=1,
            col=1,
        )
        summary.add_trace(
            go.Bar(
                x=labels,
                y=selected["annual_temperature_bias_c"],
                name=variant,
                legendgroup=variant,
                showlegend=False,
                marker_color=colors[variant],
            ),
            row=1,
            col=2,
        )
    summary.update_layout(
        title=(
            "Thermal state-space in-sample sufficiency"
            if in_sample
            else "Thermal state-space generalization to unseen buildings"
        ),
        barmode="group",
        height=650,
        template="plotly_white",
    )
    summary_path = output_dir / "state_space_sufficiency_dashboard.html"
    summary.write_html(summary_path, include_plotlyjs="cdn")

    annual = make_subplots(
        rows=len(trajectories),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.025,
        subplot_titles=tuple(
            f"{trajectory.policy} / {trajectory.profile.profile_id}"
            for trajectory in trajectories
        ),
    )
    for row_index, trajectory in enumerate(trajectories, start=1):
        profile = trajectory.profile
        plot_slice = slice(None, None, 4)
        annual.add_trace(
            go.Scatter(
                x=profile.datetime[plot_slice],
                y=profile.targets[plot_slice, 0],
                name="EnergyPlus",
                legendgroup="EnergyPlus",
                showlegend=row_index == 1,
                line={"color": "#202124", "width": 1.8},
            ),
            row=row_index,
            col=1,
        )
        for variant in variants:
            prediction = traces[
                (variant, trajectory.profile.profile_id, trajectory.policy)
            ]
            annual.add_trace(
                go.Scatter(
                    x=profile.datetime[plot_slice],
                    y=prediction[plot_slice],
                    name=variant,
                    legendgroup=variant,
                    showlegend=row_index == 1,
                    line={"color": colors[variant], "width": 1.0},
                ),
                row=row_index,
                col=1,
            )
        annual.update_yaxes(title_text="Tin [C]", row=row_index, col=1)
    annual.update_layout(
        title=(
            "Full-year autonomous Tin with observed Qroom "
            + ("(fit = evaluation year)" if in_sample else "(unseen buildings)")
        ),
        height=330 * len(trajectories),
        hovermode="x unified",
        template="plotly_white",
    )
    annual_path = output_dir / "state_space_annual_traces.html"
    annual.write_html(annual_path, include_plotlyjs="cdn")
    return summary_path, annual_path


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        train_trajectories,
        evaluation_trajectories,
        selected_ids,
        train_ids,
        test_ids,
    ) = _load_trajectory_split(args)
    if test_ids and "local_fixed" in args.variants:
        raise ValueError(
            "local_fixed has no parameters for unseen buildings; omit it when "
            "--test-fraction is positive"
        )
    scales = _fit_scales(train_trajectories)
    windows = _make_windows(
        train_trajectories,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    print(
        f"in_sample_sufficiency={not bool(test_ids)} "
        f"selected_profiles={len(selected_ids)} train_profiles={len(train_ids)} "
        f"test_profiles={len(test_ids)} train_trajectories={len(train_trajectories)} "
        f"evaluation_trajectories={len(evaluation_trajectories)} "
        f"windows={len(windows['target'])}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    traces: dict[tuple[str, int, str], np.ndarray] = {}
    training_summary: dict[str, dict[str, float]] = {}
    for variant in args.variants:
        params, training = _fit_variant(
            variant,
            windows,
            scales,
            args=args,
            num_buildings=len(selected_ids),
        )
        training_summary[variant] = training
        evaluate = _make_annual_evaluator(params, variant, scales, args)
        for trajectory in evaluation_trajectories:
            prediction = _annual_prediction(
                evaluate, trajectory
            )
            profile = trajectory.profile
            row = {
                "variant": variant,
                "profile_id": profile.profile_id,
                "policy": trajectory.policy,
                "evaluation_split": "test" if test_ids else "train",
                **training,
                **_metrics(prediction, profile.targets[:, 0], profile.datetime),
            }
            records.append(row)
            traces[(variant, profile.profile_id, trajectory.policy)] = prediction
            print(
                f"evaluated variant={variant} policy={trajectory.policy} "
                f"profile={profile.profile_id} "
                f"annual_rmse_c={row['annual_temperature_rmse_c']:.4f}",
                flush=True,
            )

    frame = pd.DataFrame(records)
    metrics_path = args.output_dir / "state_space_sufficiency_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    summary_path, annual_path = _write_dashboard(
        frame,
        traces,
        evaluation_trajectories,
        args.variants,
        not bool(test_ids),
        args.output_dir,
    )
    manifest = {
        "purpose": (
            "in_sample_thermal_state_space_sufficiency"
            if not test_ids
            else "unseen_building_thermal_state_space_generalization"
        ),
        "warning": (
            "Training and evaluation use the same complete annual trajectories; "
            "results measure representation/optimization, not generalization."
            if not test_ids
            else "An observed thermal driver is supplied during evaluation; this "
            "measures only thermal-driver-to-Tin generalization, not closed-loop "
            "performance."
        ),
        "variants": list(args.variants),
        "selected_profile_ids": list(selected_ids),
        "train_profile_ids": list(train_ids),
        "test_profile_ids": list(test_ids),
        "policies": ["setpoint", "force_off"],
        "thermal_driver_by_variant": {
            variant: (
                "observed_pel_sh_times_cop"
                if variant == "metadata_fixed_pel_cop_positive_leaky"
                else "observed_pel_sh"
                if variant == "metadata_fixed_pel"
                else "observed_qroom"
            )
            for variant in args.variants
        },
        "training": training_summary,
        "scales": {
            "temperature_mean": scales.temperature_mean,
            "temperature_std": scales.temperature_std,
            "qroom_scale": scales.qroom_scale,
            "pel_scale": scales.pel_scale,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    manifest_path = args.output_dir / "state_space_sufficiency_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"saved_metrics={metrics_path}", flush=True)
    print(f"saved_dashboard={summary_path}", flush=True)
    print(f"saved_annual_traces={annual_path}", flush=True)
    return summary_path


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
    parser.add_argument(
        "--profile-ids",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--profile-selection", choices=("random", "first"), default="random"
    )
    parser.add_argument(
        "--profile-filter",
        choices=("sh_hp", "common"),
        default="sh_hp",
        help="Restrict to SH-bound heat pumps or use all profiles shared by both datasets.",
    )
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--encoded-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--bptt-truncate-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument(
        "--q-to-t-time-constants-hours",
        type=float,
        nargs="+",
        default=(0.25, 1.0, 4.0, 16.0, 24.0, 48.0),
    )
    parser.add_argument("--q-to-t-gain-min", type=float, default=0.01)
    parser.add_argument("--q-to-t-gain-max", type=float, default=2.0)
    parser.add_argument("--pel-cop-slope-per-c", type=float, default=0.05)
    parser.add_argument(
        "--pel-cop-reference-outdoor-c", type=float, default=7.0
    )
    parser.add_argument("--pel-cop-floor", type=float, default=1.0)
    parser.add_argument("--pel-cop-cap", type=float, default=8.0)
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/conditional_state_space_sufficiency"
        ),
    )
    args = parser.parse_args()
    if args.profile_ids is not None and len(set(args.profile_ids)) != len(
        args.profile_ids
    ):
        parser.error("--profile-ids must be unique")
    if args.max_profiles is not None and args.max_profiles < 1:
        parser.error("--max-profiles must be positive")
    if not 0.0 <= args.test_fraction < 1.0:
        parser.error("--test-fraction must lie in [0, 1)")
    if not 0.05 < args.gamma < 1.0:
        parser.error("--gamma must lie between 0.05 and 1")
    if any(value <= 0.0 for value in args.q_to_t_time_constants_hours):
        parser.error("--q-to-t-time-constants-hours values must be positive")
    if args.q_to_t_gain_max <= args.q_to_t_gain_min:
        parser.error("Q-to-T gain bounds must satisfy max > min")
    if args.pel_cop_slope_per_c < 0.0:
        parser.error("--pel-cop-slope-per-c must be non-negative")
    if args.pel_cop_floor <= 0.0 or args.pel_cop_cap <= args.pel_cop_floor:
        parser.error("Pel COP bounds must satisfy 0 < floor < cap")
    if min(
        args.state_dim,
        args.encoded_dim,
        args.hidden_dim,
        args.depth,
        args.sequence_length,
        args.stride,
        args.bptt_truncate_steps,
        args.batch_size,
        args.epochs,
    ) < 1:
        parser.error("model and training dimensions must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
