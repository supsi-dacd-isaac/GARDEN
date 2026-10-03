"""Diagnose whether a fitted legacy thermal block can explain Tin with oracle Q.

The diagnostic refits one building-specific legacy model on the first half of
the year, freezes its parameters, and bypasses the learned HP/hydronic blocks
over the held-out half-year. It then compares:

* the ordinary closed-loop prediction;
* the thermal block driven by observed EnergyPlus Qroom;
* a non-causal oracle selecting Qroom at every step to best reproduce Tin.

The oracle is evaluated with both the fitted physical Q cap and an expanded
cap. Teacher-forced inversion isolates one-step expressivity, while autonomous
inversion tests whether recursive Tin can remain on the simulated trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .deterministic_two_building_temporal_ablation import _temporal_segments
from .single_building_architecture_search import (
    INTERNAL_INDEX,
    OCCUPANCY_INDEX,
    SOLAR_INDEX,
    TOUT_INDEX,
    VENT_INDEX,
    Architecture,
    Scale,
    THERMAL_CHOICES,
    _bounded_mlp_alpha,
    _caps,
    _event_response_errors,
    _fit_scales,
    _load_policy_profile,
    _mlp,
    _passive_memory_positive_q_step,
    _passive_memory_terms,
    _rollout,
    _train_candidate,
    _training_windows,
)


LEGACY = Architecture("hurdle", "buffer", "bounded_mlp")


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
    """Apply exactly the legacy bounded-MLP thermal transition."""

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


def _initial_thermal_memory(
    inputs: jax.Array,
    initial_temperature: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
) -> jax.Array:
    normalized = (inputs[0] - input_mean) / input_std
    temperature = jnp.ravel(initial_temperature)[0]
    features = jnp.asarray(
        [
            (temperature - target_t_mean) / target_t_std,
            (inputs[0, TOUT_INDEX] - temperature) / target_t_std,
            normalized[SOLAR_INDEX],
            normalized[VENT_INDEX],
            normalized[INTERNAL_INDEX],
            normalized[OCCUPANCY_INDEX],
            0.0,
        ]
    )
    return jnp.tile(features[None, :], (4, 1))


def _thermal_temperature_step(
    params: dict[str, Any],
    architecture: Architecture,
    row: jax.Array,
    temperature: jax.Array,
    qroom: jax.Array,
    memory: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
    q_cap: float,
    temperature_delta_cap: float,
    dt_hours: float,
) -> tuple[jax.Array, jax.Array]:
    if architecture.thermal == "passive_memory_positive_q":
        next_temperature, next_memory, _ = _passive_memory_positive_q_step(
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
        return next_temperature, next_memory
    return (
        _bounded_mlp_temperature_step(
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
        ),
        memory,
    )


def _temperature_with_given_q(
    params: dict[str, Any],
    architecture: Architecture,
    inputs: jax.Array,
    qroom: jax.Array,
    initial_temperature: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
    q_cap: float,
    temperature_delta_cap: float,
    dt_hours: float,
) -> jax.Array:
    initial_memory = _initial_thermal_memory(
        inputs,
        initial_temperature,
        input_mean=input_mean,
        input_std=input_std,
        target_t_mean=target_t_mean,
        target_t_std=target_t_std,
    )

    def step(
        carry: tuple[jax.Array, jax.Array],
        values: tuple[jax.Array, jax.Array],
    ):
        temperature, memory = carry
        row, q_value = values
        next_temperature, next_memory = _thermal_temperature_step(
            params,
            architecture,
            row,
            temperature,
            q_value,
            memory,
            input_mean=input_mean,
            input_std=input_std,
            target_t_mean=target_t_mean,
            target_t_std=target_t_std,
            q_cap=q_cap,
            temperature_delta_cap=temperature_delta_cap,
            dt_hours=dt_hours,
        )
        return (next_temperature, next_memory), next_temperature

    _, temperatures = jax.lax.scan(
        step,
        (jnp.ravel(initial_temperature)[0], initial_memory),
        (inputs, qroom),
    )
    return temperatures


def _teacher_forced_temperature_with_given_q(
    params: dict[str, Any],
    architecture: Architecture,
    inputs: jax.Array,
    qroom: jax.Array,
    target_temperature: jax.Array,
    initial_temperature: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
    q_cap: float,
    temperature_delta_cap: float,
    dt_hours: float,
) -> jax.Array:
    previous_true = jnp.concatenate(
        (jnp.ravel(initial_temperature)[:1], target_temperature[:-1])
    )
    initial_memory = _initial_thermal_memory(
        inputs,
        initial_temperature,
        input_mean=input_mean,
        input_std=input_std,
        target_t_mean=target_t_mean,
        target_t_std=target_t_std,
    )

    def step(memory: jax.Array, values):
        row, q_value, temperature = values
        next_temperature, next_memory = _thermal_temperature_step(
            params,
            architecture,
            row,
            temperature,
            q_value,
            memory,
            input_mean=input_mean,
            input_std=input_std,
            target_t_mean=target_t_mean,
            target_t_std=target_t_std,
            q_cap=q_cap,
            temperature_delta_cap=temperature_delta_cap,
            dt_hours=dt_hours,
        )
        return next_memory, next_temperature

    _, predictions = jax.lax.scan(
        step,
        initial_memory,
        (inputs, qroom, previous_true),
    )
    return predictions


def _oracle_q_rollout(
    params: dict[str, Any],
    architecture: Architecture,
    inputs: jax.Array,
    target_temperature: jax.Array,
    initial_temperature: jax.Array,
    *,
    input_mean: jax.Array,
    input_std: jax.Array,
    target_t_mean: jax.Array,
    target_t_std: jax.Array,
    q_cap: float,
    oracle_q_max: float,
    temperature_delta_cap: float,
    dt_hours: float,
    grid_points: int,
    teacher_forced: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    q_grid = jnp.linspace(0.0, oracle_q_max, grid_points)
    initial_memory = _initial_thermal_memory(
        inputs,
        initial_temperature,
        input_mean=input_mean,
        input_std=input_std,
        target_t_mean=target_t_mean,
        target_t_std=target_t_std,
    )

    def choose_q(
        current_temperature: jax.Array,
        memory: jax.Array,
        row: jax.Array,
        target: jax.Array,
    ):
        if architecture.thermal == "passive_memory_positive_q":
            passive_delta, q_gain, next_memory = _passive_memory_terms(
                params,
                row,
                current_temperature,
                memory,
                input_mean=input_mean,
                input_std=input_std,
                target_t_mean=target_t_mean,
                target_t_std=target_t_std,
                dt_hours=dt_hours,
                temperature_delta_cap=temperature_delta_cap,
            )
            raw_delta = passive_delta + q_gain * q_grid
            candidates = jnp.clip(
                current_temperature
                + temperature_delta_cap
                * jnp.tanh(raw_delta / temperature_delta_cap),
                5.0,
                40.0,
            )
        else:
            candidates = jax.vmap(
                lambda q: _bounded_mlp_temperature_step(
                    params,
                    architecture,
                    row,
                    current_temperature,
                    q,
                    input_mean=input_mean,
                    input_std=input_std,
                    q_cap=q_cap,
                    temperature_delta_cap=temperature_delta_cap,
                    dt_hours=dt_hours,
                )
            )(q_grid)
            next_memory = memory
        selected = jnp.argmin(jnp.square(candidates - target))
        return (
            candidates[selected],
            q_grid[selected],
            jnp.min(candidates),
            jnp.max(candidates),
            next_memory,
        )

    if teacher_forced:
        previous_true = jnp.concatenate(
            (jnp.ravel(initial_temperature)[:1], target_temperature[:-1])
        )
        def teacher_step(memory: jax.Array, values):
            temperature, row, target = values
            selected_temperature, selected_q, low, high, next_memory = (
                choose_q(temperature, memory, row, target)
            )
            return next_memory, (selected_temperature, selected_q, low, high)

        _, outputs = jax.lax.scan(
            teacher_step,
            initial_memory,
            (previous_true, inputs, target_temperature),
        )
        return outputs

    def step(carry, values: tuple[jax.Array, jax.Array]):
        temperature, memory = carry
        row, target = values
        selected_temperature, selected_q, low, high, next_memory = choose_q(
            temperature, memory, row, target
        )
        return (selected_temperature, next_memory), (
            selected_temperature,
            selected_q,
            low,
            high,
        )

    _, outputs = jax.lax.scan(
        step,
        (jnp.ravel(initial_temperature)[0], initial_memory),
        (inputs, target_temperature),
    )
    return outputs


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def _controller_teacher_feedback(
    target: np.ndarray, initial_temperature: np.ndarray
) -> np.ndarray:
    previous_temperature = np.concatenate(
        (np.ravel(initial_temperature)[:1], target[:-1, 0])
    )
    previous_pel = np.concatenate(
        (np.zeros((1,), dtype=target.dtype), target[:-1, 2])
    )
    previous_qroom = np.concatenate(
        (np.zeros((1,), dtype=target.dtype), target[:-1, 1])
    )
    previous_mode = (previous_pel > 1e-5).astype(target.dtype)
    return np.column_stack(
        (previous_temperature, previous_pel, previous_qroom, previous_mode)
    ).astype(np.float32)


def _inverse_metrics(
    temperature: np.ndarray,
    qroom: np.ndarray,
    target_temperature: np.ndarray,
    target_qroom: np.ndarray,
    reachable_low: np.ndarray,
    reachable_high: np.ndarray,
    *,
    q_max: float,
    dt_hours: float,
) -> dict[str, float]:
    tolerance = 1e-4
    below_reachable = target_temperature < reachable_low - tolerance
    above_reachable = target_temperature > reachable_high + tolerance
    unreachable = below_reachable | above_reachable
    return {
        "temperature_rmse_c": _rmse(temperature, target_temperature),
        "temperature_mae_c": float(
            np.mean(np.abs(temperature - target_temperature))
        ),
        "temperature_bias_c": float(
            np.mean(temperature - target_temperature)
        ),
        "qroom_rmse_w_m2": _rmse(qroom, target_qroom),
        "qroom_energy_ratio": float(
            np.sum(qroom) / max(np.sum(target_qroom), 1e-8)
        ),
        "qroom_energy_kwh_m2": float(dt_hours * np.sum(qroom) / 1000.0),
        "qroom_zero_fraction": float(np.mean(qroom <= tolerance)),
        "qroom_upper_bound_fraction": float(
            np.mean(qroom >= q_max - max(1e-3, 1e-3 * q_max))
        ),
        "target_outside_one_step_reachable_fraction": float(
            np.mean(unreachable)
        ),
        "target_below_reachable_min_fraction": float(
            np.mean(below_reachable)
        ),
        "target_above_reachable_max_fraction": float(
            np.mean(above_reachable)
        ),
    }


def _plot(
    output_path: Path,
    thermal_kind: str,
    datetime: np.ndarray,
    target_temperature: np.ndarray,
    target_qroom: np.ndarray,
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Held-out Tin", "Oracle and EnergyPlus Qroom"),
    )
    figure.add_trace(
        go.Scatter(
            x=datetime,
            y=target_temperature,
            name="EnergyPlus",
            line={"color": "#202124", "width": 2.2},
        ),
        row=1,
        col=1,
    )
    colors = {
        "Legacy closed loop": "#68757d",
        "Observed Q": "#2a9d8f",
        "Oracle Q (capped)": "#e76f51",
        "Oracle Q (expanded)": "#7b61a8",
        "Teacher-forced inverse": "#d49b22",
    }
    for name, (temperature, _) in traces.items():
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=temperature,
                name=name,
                legendgroup=name,
                line={"color": colors[name], "width": 1.3},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=datetime,
            y=target_qroom,
            name="EnergyPlus Qroom",
            legendgroup="EnergyPlus Qroom",
            line={"color": "#202124", "width": 2.0},
        ),
        row=2,
        col=1,
    )
    for name, (_, qroom) in traces.items():
        if name == "Legacy closed loop":
            continue
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=qroom,
                name=f"{name} Qroom",
                legendgroup=name,
                showlegend=False,
                line={"color": colors[name], "width": 1.1},
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="Tin [C]", row=1, col=1)
    figure.update_yaxes(title_text="Qroom [W/m2]", row=2, col=1)
    figure.update_layout(
        title=f"Thermal-block Q inversion: {thermal_kind}",
        height=900,
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture = Architecture("hurdle", "buffer", args.thermal_kind)
    policy_profile = _load_policy_profile(
        args.dataset,
        args.profile_id,
        policy=args.policy,
    )
    profile = policy_profile.profile
    segments = _temporal_segments(policy_profile, args.train_fraction)
    train_segment, test_segment = segments
    input_scale, target_scale = _fit_scales((policy_profile,), segments)
    caps = _caps((policy_profile,), segments)
    train_inputs, train_targets, train_initials, _ = _training_windows(
        (policy_profile,),
        segments,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    params, training_metrics = _train_candidate(
        architecture,
        train_inputs,
        train_targets,
        train_initials,
        args=args,
        input_scale=input_scale,
        target_scale=target_scale,
        caps=caps,
        seed=args.seed,
    )

    warmup_steps = max(0, int(round(args.warmup_hours / args.dt_hours)))
    rollout_start = max(0, test_segment.start - warmup_steps)
    inputs = profile.inputs[rollout_start : test_segment.end]
    target = profile.targets[rollout_start : test_segment.end]
    initial = profile.initial_temperature[rollout_start]
    score_offset = test_segment.start - rollout_start

    input_mean = jnp.asarray(input_scale.mean)
    input_std = jnp.asarray(input_scale.std)
    target_t_mean = jnp.asarray(target_scale.mean[0])
    target_t_std = jnp.asarray(target_scale.std[0])
    rollout_kwargs = {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": jnp.asarray(target_scale.mean),
        "target_std": jnp.asarray(target_scale.std),
        "pel_cap": caps[0],
        "q_cap": caps[1],
        "energy_cap": caps[2],
        "hidden_dim": args.hidden_dim,
        "dt_hours": args.dt_hours,
        "temperature_delta_cap": args.temperature_delta_cap,
        "controller_calendar_mode": args.controller_calendar_mode,
        "hp_recovery_emergency_threshold_c": args.hp_recovery_emergency_threshold_c,
        "hydronic_capacity_min_hours": args.hydronic_capacity_min_hours,
        "hydronic_capacity_max_hours": args.hydronic_capacity_max_hours,
    }
    closed_loop = np.asarray(
        eqx.filter_jit(
            lambda x, t: _rollout(
                params, architecture, x, t, **rollout_kwargs
            )[0]
        )(jnp.asarray(inputs), jnp.asarray(initial))
    )
    teacher_feedback = _controller_teacher_feedback(target, initial)
    previous_true_temperature = teacher_feedback[:, 0]
    controller_temperature_forced = np.asarray(
        eqx.filter_jit(
            lambda x, t, feedback: _rollout(
                params,
                architecture,
                x,
                t,
                teacher_feedback=feedback,
                teacher_force_temperature=True,
                **rollout_kwargs,
            )[0]
        )(
            jnp.asarray(inputs),
            jnp.asarray(initial),
            jnp.asarray(teacher_feedback),
        )
    )
    controller_all_feedback_forced = np.asarray(
        eqx.filter_jit(
            lambda x, t, feedback: _rollout(
                params,
                architecture,
                x,
                t,
                teacher_feedback=feedback,
                teacher_force_temperature=True,
                teacher_force_power_history=True,
                **rollout_kwargs,
            )[0]
        )(
            jnp.asarray(inputs),
            jnp.asarray(initial),
            jnp.asarray(teacher_feedback),
        )
    )
    controller_training_fit: dict[str, float] | None = None
    if args.controller_training_feedback != "closed_loop":
        train_inputs = profile.inputs[train_segment.start : train_segment.end]
        train_target = profile.targets[train_segment.start : train_segment.end]
        train_initial = profile.initial_temperature[train_segment.start]
        train_feedback = _controller_teacher_feedback(
            train_target, train_initial
        )
        force_history = (
            args.controller_training_feedback == "temperature_and_history"
        )
        train_prediction = np.asarray(
            eqx.filter_jit(
                lambda x, t, feedback: _rollout(
                    params,
                    architecture,
                    x,
                    t,
                    teacher_feedback=feedback,
                    teacher_force_temperature=True,
                    teacher_force_power_history=force_history,
                    **rollout_kwargs,
                )[0]
            )(
                jnp.asarray(train_inputs),
                jnp.asarray(train_initial),
                jnp.asarray(train_feedback),
            )
        )
        controller_training_fit = {
            "qroom_rmse_w_m2": _rmse(
                train_prediction[:, 1], train_target[:, 1]
            ),
            "pel_rmse_w_m2": _rmse(
                train_prediction[:, 2], train_target[:, 2]
            ),
            "qroom_energy_ratio": float(
                np.sum(train_prediction[:, 1])
                / max(np.sum(train_target[:, 1]), 1e-8)
            ),
            "pel_energy_ratio": float(
                np.sum(train_prediction[:, 2])
                / max(np.sum(train_target[:, 2]), 1e-8)
            ),
        }
    observed_q_temperature = np.asarray(
        eqx.filter_jit(
            lambda x, q, t: _temperature_with_given_q(
                params,
                architecture,
                x,
                q,
                t,
                input_mean=input_mean,
                input_std=input_std,
                target_t_mean=target_t_mean,
                target_t_std=target_t_std,
                q_cap=caps[1],
                temperature_delta_cap=args.temperature_delta_cap,
                dt_hours=args.dt_hours,
            )
        )(
            jnp.asarray(inputs),
            jnp.asarray(target[:, 1]),
            jnp.asarray(initial),
        )
    )
    observed_q_teacher_forced_temperature = np.asarray(
        eqx.filter_jit(
            lambda x, q, y, t: _teacher_forced_temperature_with_given_q(
                params,
                architecture,
                x,
                q,
                y,
                t,
                input_mean=input_mean,
                input_std=input_std,
                target_t_mean=target_t_mean,
                target_t_std=target_t_std,
                q_cap=caps[1],
                temperature_delta_cap=args.temperature_delta_cap,
                dt_hours=args.dt_hours,
            )
        )(
            jnp.asarray(inputs),
            jnp.asarray(target[:, 1]),
            jnp.asarray(target[:, 0]),
            jnp.asarray(initial),
        )
    )

    oracle: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, multiplier, teacher_forced in (
        ("capped", 1.0, False),
        ("expanded", args.expanded_q_multiplier, False),
        ("teacher_forced", 1.0, True),
    ):
        oracle_max = multiplier * caps[1]
        result = eqx.filter_jit(
            lambda x, y, t: _oracle_q_rollout(
                params,
                architecture,
                x,
                y,
                t,
                input_mean=input_mean,
                input_std=input_std,
                target_t_mean=target_t_mean,
                target_t_std=target_t_std,
                q_cap=caps[1],
                oracle_q_max=oracle_max,
                temperature_delta_cap=args.temperature_delta_cap,
                dt_hours=args.dt_hours,
                grid_points=args.oracle_grid_points,
                teacher_forced=teacher_forced,
            )
        )(
            jnp.asarray(inputs),
            jnp.asarray(target[:, 0]),
            jnp.asarray(initial),
        )
        oracle[name] = tuple(np.asarray(value) for value in result)

    score = slice(score_offset, None)
    target_temperature = target[score, 0]
    target_qroom = target[score, 1]
    target_pel = target[score, 2]
    closed_loop_event_metrics = _event_response_errors(
        inputs[score],
        target[score],
        closed_loop[score],
        horizon_steps=max(
            1, int(round(args.event_tin_loss_horizon_hours / args.dt_hours))
        ),
    )
    alpha_statistics: tuple[float | None, float | None, float | None]
    q_gain_statistics: tuple[float | None, float | None, float | None]
    passive_statistics: tuple[float | None, float | None]
    if architecture.thermal == "passive_memory_positive_q":
        initial_memory = _initial_thermal_memory(
            jnp.asarray(inputs),
            jnp.asarray(initial),
            input_mean=input_mean,
            input_std=input_std,
            target_t_mean=target_t_mean,
            target_t_std=target_t_std,
        )

        def diagnostic_step(memory: jax.Array, values):
            row, temperature = values
            passive_delta, q_gain, next_memory = _passive_memory_terms(
                params,
                row,
                temperature,
                memory,
                input_mean=input_mean,
                input_std=input_std,
                target_t_mean=target_t_mean,
                target_t_std=target_t_std,
                dt_hours=args.dt_hours,
                temperature_delta_cap=args.temperature_delta_cap,
            )
            return next_memory, (passive_delta, q_gain)

        _, (passive_trace, q_gain_trace) = jax.lax.scan(
            diagnostic_step,
            initial_memory,
            (jnp.asarray(inputs), jnp.asarray(previous_true_temperature)),
        )
        passive_trace = np.asarray(passive_trace)[score]
        q_gain_trace = np.asarray(q_gain_trace)[score]
        alpha_statistics = (None, None, None)
        q_gain_statistics = (
            float(np.mean(q_gain_trace)),
            float(np.min(q_gain_trace)),
            float(np.max(q_gain_trace)),
        )
        passive_statistics = (
            float(np.mean(passive_trace)),
            float(np.std(passive_trace)),
        )
    else:
        alpha_trace = np.asarray(
            jax.vmap(
                lambda row, temperature: _bounded_mlp_alpha(
                    params,
                    architecture,
                    row,
                    temperature,
                    input_mean=input_mean,
                    input_std=input_std,
                    dt_hours=args.dt_hours,
                )
            )(
                jnp.asarray(inputs),
                jnp.asarray(previous_true_temperature),
            )
        )[score]
        alpha_statistics = (
            float(np.mean(alpha_trace)),
            float(np.min(alpha_trace)),
            float(np.max(alpha_trace)),
        )
        q_gain_statistics = (None, None, None)
        passive_statistics = (None, None)
    metrics: dict[str, Any] = {
        "profile_id": args.profile_id,
        "policy": args.policy,
        "thermal_kind": args.thermal_kind,
        "train_start_index": train_segment.start,
        "train_end_index": train_segment.end,
        "test_start_index": test_segment.start,
        "test_end_index": test_segment.end,
        "q_cap_w_m2": caps[1],
        "expanded_q_cap_w_m2": args.expanded_q_multiplier * caps[1],
        "observed_q_thermal_loss_weight": args.observed_q_thermal_loss_weight,
        "qroom_window_mean_loss_weight": args.qroom_window_mean_loss_weight,
        "controller_training_feedback": args.controller_training_feedback,
        "controller_calendar_mode": args.controller_calendar_mode,
        "controller_training_fit": controller_training_fit,
        "thermal_alpha_mean": alpha_statistics[0],
        "thermal_alpha_min": alpha_statistics[1],
        "thermal_alpha_max": alpha_statistics[2],
        "thermal_q_gain_mean_c_per_w_m2": q_gain_statistics[0],
        "thermal_q_gain_min_c_per_w_m2": q_gain_statistics[1],
        "thermal_q_gain_max_c_per_w_m2": q_gain_statistics[2],
        "thermal_passive_delta_mean_c": passive_statistics[0],
        "thermal_passive_delta_std_c": passive_statistics[1],
        "training": training_metrics,
        "legacy_closed_loop": {
            "temperature_rmse_c": _rmse(
                closed_loop[score, 0], target_temperature
            ),
            "qroom_rmse_w_m2": _rmse(closed_loop[score, 1], target_qroom),
            "pel_rmse_w_m2": _rmse(closed_loop[score, 2], target_pel),
            "qroom_energy_ratio": float(
                np.sum(closed_loop[score, 1])
                / max(np.sum(target_qroom), 1e-8)
            ),
            "temperature_bias_c": float(
                np.mean(closed_loop[score, 0] - target_temperature)
            ),
            **closed_loop_event_metrics,
        },
        "controller_teacher_forced_temperature": {
            "qroom_rmse_w_m2": _rmse(
                controller_temperature_forced[score, 1], target_qroom
            ),
            "pel_rmse_w_m2": _rmse(
                controller_temperature_forced[score, 2], target_pel
            ),
            "qroom_energy_ratio": float(
                np.sum(controller_temperature_forced[score, 1])
                / max(np.sum(target_qroom), 1e-8)
            ),
            "pel_energy_ratio": float(
                np.sum(controller_temperature_forced[score, 2])
                / max(np.sum(target_pel), 1e-8)
            ),
        },
        "controller_teacher_forced_all_history": {
            "qroom_rmse_w_m2": _rmse(
                controller_all_feedback_forced[score, 1], target_qroom
            ),
            "pel_rmse_w_m2": _rmse(
                controller_all_feedback_forced[score, 2], target_pel
            ),
            "qroom_energy_ratio": float(
                np.sum(controller_all_feedback_forced[score, 1])
                / max(np.sum(target_qroom), 1e-8)
            ),
            "pel_energy_ratio": float(
                np.sum(controller_all_feedback_forced[score, 2])
                / max(np.sum(target_pel), 1e-8)
            ),
        },
        "observed_q": {
            "temperature_rmse_c": _rmse(
                observed_q_temperature[score], target_temperature
            ),
            "qroom_rmse_w_m2": 0.0,
            "qroom_energy_ratio": 1.0,
            "temperature_bias_c": float(
                np.mean(observed_q_temperature[score] - target_temperature)
            ),
        },
        "observed_q_teacher_forced": {
            "temperature_rmse_c": _rmse(
                observed_q_teacher_forced_temperature[score],
                target_temperature,
            ),
            "qroom_rmse_w_m2": 0.0,
            "qroom_energy_ratio": 1.0,
            "temperature_bias_c": float(
                np.mean(
                    observed_q_teacher_forced_temperature[score]
                    - target_temperature
                )
            ),
        },
    }
    metric_names = {
        "capped": "oracle_q_capped",
        "expanded": "oracle_q_expanded",
        "teacher_forced": "teacher_forced_inverse",
    }
    q_maxima = {
        "capped": caps[1],
        "expanded": args.expanded_q_multiplier * caps[1],
        "teacher_forced": caps[1],
    }
    for name, (temperature, qroom, low, high) in oracle.items():
        metrics[metric_names[name]] = _inverse_metrics(
            temperature[score],
            qroom[score],
            target_temperature,
            target_qroom,
            low[score],
            high[score],
            q_max=q_maxima[name],
            dt_hours=args.dt_hours,
        )

    output_json = args.output_dir / "oracle_q_inversion_metrics.json"
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    traces = {
        "Legacy closed loop": (
            closed_loop[score, 0],
            closed_loop[score, 1],
        ),
        "Observed Q": (observed_q_temperature[score], target_qroom),
        "Oracle Q (capped)": (oracle["capped"][0][score], oracle["capped"][1][score]),
        "Oracle Q (expanded)": (
            oracle["expanded"][0][score],
            oracle["expanded"][1][score],
        ),
        "Teacher-forced inverse": (
            oracle["teacher_forced"][0][score],
            oracle["teacher_forced"][1][score],
        ),
    }
    _plot(
        args.output_dir / "oracle_q_inversion.html",
        args.thermal_kind,
        profile.datetime[test_segment.start : test_segment.end],
        target_temperature,
        target_qroom,
        traces,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"saved_oracle_q_inversion={args.output_dir}", flush=True)
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/control=force_off"),
    )
    parser.add_argument("--profile-id", type=int, default=11132764)
    parser.add_argument(
        "--thermal-kind",
        choices=(
            "bounded_mlp",
            "bounded_mlp_feedback",
            "bounded_mlp_dual_alpha",
            "bounded_mlp_physical_alpha",
            "passive_memory_positive_q",
        ),
        default="bounded_mlp",
    )
    parser.add_argument("--policy", choices=("setpoint", "force_off"), default="force_off")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--temperature-delta-cap", type=float, default=2.0)
    parser.add_argument("--mode-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--controller-training-feedback",
        choices=("closed_loop", "temperature", "temperature_and_history"),
        default="closed_loop",
    )
    parser.add_argument(
        "--controller-calendar-mode",
        choices=("all", "hour_only", "none"),
        default="all",
    )
    parser.add_argument("--event-tin-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--observed-q-thermal-loss-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--qroom-window-mean-loss-weight", type=float, default=0.0
    )
    parser.add_argument("--event-tin-loss-horizon-hours", type=float, default=3.0)
    parser.add_argument("--event-tin-loss-setpoint-threshold-c", type=float, default=0.05)
    parser.add_argument("--hp-recovery-emergency-threshold-c", type=float, default=1.0)
    parser.add_argument("--hydronic-capacity-min-hours", type=float, default=0.25)
    parser.add_argument("--hydronic-capacity-max-hours", type=float, default=24.0)
    parser.add_argument("--hydronic-capacity-init-hours", type=float, default=2.0)
    parser.add_argument("--warmup-hours", type=float, default=168.0)
    parser.add_argument("--oracle-grid-points", type=int, default=513)
    parser.add_argument("--expanded-q-multiplier", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "deterministic_oracle_q_inversion_11132764"
        ),
    )
    args = parser.parse_args()
    if args.oracle_grid_points < 2:
        parser.error("--oracle-grid-points must be at least 2")
    if args.expanded_q_multiplier < 1.0:
        parser.error("--expanded-q-multiplier must be at least 1")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
