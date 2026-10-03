"""Compare stable RC orders using observed Qroom and autonomous Tin rollouts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .deterministic_two_building_temporal_ablation import _temporal_segments
from .single_building_architecture_search import (
    AVAILABILITY_INDEX,
    FORCE_OFF_INDEX,
    INTERNAL_INDEX,
    OCCUPANCY_INDEX,
    SOLAR_INDEX,
    TOUT_INDEX,
    VENT_INDEX,
    _load_policy_profile,
)


def _logit(fraction: float) -> float:
    fraction = min(max(fraction, 1e-5), 1.0 - 1e-5)
    return math.log(fraction / (1.0 - fraction))


def _initial_parameters(order: int) -> dict[str, jax.Array]:
    mass_count = order - 1
    if mass_count == 0:
        initial_mass_rates = jnp.empty((0,), dtype=jnp.float32)
    elif mass_count == 1:
        initial_mass_rates = jnp.asarray([0.02], dtype=jnp.float32)
    else:
        # Break the otherwise permanent symmetry between a fast and a slow
        # thermal mass (roughly 4 h and 30 h initial time constants).
        initial_mass_rates = jnp.asarray([0.06, 0.008], dtype=jnp.float32)
    mass_rate_raw = jnp.asarray(
        [_logit(float(rate) / 0.15) for rate in initial_mass_rates],
        dtype=jnp.float32,
    )
    return {
        "base_coupling_raw": jnp.asarray(_logit(0.03 / 0.25), dtype=jnp.float32),
        "vent_coupling_raw": jnp.asarray(_logit(0.01 / 0.15), dtype=jnp.float32),
        "coupling_logits": jnp.linspace(-0.1, 0.1, order, dtype=jnp.float32),
        "mass_rate_raw": mass_rate_raw,
        "forcing_gain_raw": jnp.full(
            (4,), _logit(0.01 / 0.20), dtype=jnp.float32
        ),
        "qroom_routing_logits": jnp.zeros((order,), dtype=jnp.float32),
        "qroom_gain_increment_raw": jnp.asarray(
            _logit(0.01 / 0.20), dtype=jnp.float32
        ),
        "qroom_gain_threshold_raw": jnp.asarray(
            _logit(1.0 / 3.0), dtype=jnp.float32
        ),
    }


def _parameter_values(params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "base_coupling": 0.25 * jax.nn.sigmoid(params["base_coupling_raw"]),
        "vent_coupling": 0.15 * jax.nn.sigmoid(params["vent_coupling_raw"]),
        "coupling_weights": jax.nn.softmax(params["coupling_logits"]),
        "mass_rates": 0.15 * jax.nn.sigmoid(params["mass_rate_raw"]),
        "forcing_gains": 0.20 * jax.nn.sigmoid(params["forcing_gain_raw"]),
        "qroom_routing_weights": jax.nn.softmax(
            params["qroom_routing_logits"]
        ),
        "qroom_gain_increment": 0.20
        * jax.nn.sigmoid(params["qroom_gain_increment_raw"]),
        "qroom_gain_threshold": 3.0
        * jax.nn.sigmoid(params["qroom_gain_threshold_raw"]),
    }


def _rollout(
    params: dict[str, jax.Array],
    inputs: jax.Array,
    qroom: jax.Array,
    initial_temperature: jax.Array,
    scales: jax.Array,
    *,
    delta_cap_c: float,
    qroom_routing: str,
    qroom_gain_mode: str,
) -> jax.Array:
    values = _parameter_values(params)
    mass_count = params["mass_rate_raw"].shape[0]
    initial_air = jnp.ravel(initial_temperature)[0]
    initial_mass = jnp.full((mass_count,), initial_air)

    def step(state: tuple[jax.Array, jax.Array], row_values):
        air, mass = state
        row, heat = row_values
        next_air, next_mass = _rc_step(
            values,
            air,
            mass,
            row,
            heat,
            scales,
            delta_cap_c=delta_cap_c,
            qroom_routing=qroom_routing,
            qroom_gain_mode=qroom_gain_mode,
        )
        return (next_air, next_mass), next_air

    _, temperatures = jax.lax.scan(
        step, (initial_air, initial_mass), (inputs, qroom)
    )
    return temperatures


def _rc_step(
    values: dict[str, jax.Array],
    air: jax.Array,
    mass: jax.Array,
    row: jax.Array,
    heat: jax.Array,
    scales: jax.Array,
    *,
    delta_cap_c: float,
    qroom_routing: str,
    qroom_gain_mode: str,
) -> tuple[jax.Array, jax.Array]:
    vent_activity = jnp.maximum(row[VENT_INDEX] / scales[4], 0.0)
    total_coupling = values["base_coupling"] + values["vent_coupling"] * (
        vent_activity / (1.0 + vent_activity)
    )
    neighbours = jnp.concatenate((jnp.ravel(row[TOUT_INDEX]), mass))
    passive_delta = total_coupling * jnp.sum(
        values["coupling_weights"] * (neighbours - air)
    )
    normalized_heat = heat / scales[0]
    qroom_gain = values["forcing_gains"][0]
    if qroom_gain_mode == "monotone":
        qroom_gain = qroom_gain + values["qroom_gain_increment"] * jax.nn.sigmoid(
            4.0 * (normalized_heat - values["qroom_gain_threshold"])
        )
    if qroom_routing == "positive_split":
        air_heat = (
            qroom_gain
            * values["qroom_routing_weights"][0]
            * normalized_heat
        )
        mass_heat = (
            qroom_gain
            * values["qroom_routing_weights"][1:]
            * normalized_heat
        )
    else:
        air_heat = qroom_gain * normalized_heat
        mass_heat = jnp.zeros_like(mass)
    exogenous_forcing = jnp.asarray(
        [
            row[SOLAR_INDEX] / scales[1],
            row[INTERNAL_INDEX] / scales[2],
            row[OCCUPANCY_INDEX] / scales[3],
        ]
    )
    raw_delta = (
        passive_delta
        + air_heat
        + jnp.dot(values["forcing_gains"][1:], exogenous_forcing)
    )
    delta = delta_cap_c * jnp.tanh(raw_delta / delta_cap_c)
    next_air = jnp.clip(air + delta, 5.0, 40.0)
    next_mass = jnp.clip(
        mass + values["mass_rates"] * (air - mass) + mass_heat,
        5.0,
        40.0,
    )
    return next_air, next_mass


def _teacher_forced_rollout(
    params: dict[str, jax.Array],
    inputs: jax.Array,
    qroom: jax.Array,
    current_temperature: jax.Array,
    scales: jax.Array,
    *,
    delta_cap_c: float,
    qroom_routing: str,
    qroom_gain_mode: str,
) -> jax.Array:
    values = _parameter_values(params)
    mass_count = params["mass_rate_raw"].shape[0]
    initial_mass = jnp.full((mass_count,), current_temperature[0])

    def step(mass: jax.Array, row_values):
        row, heat, air = row_values
        next_air, next_mass = _rc_step(
            values,
            air,
            mass,
            row,
            heat,
            scales,
            delta_cap_c=delta_cap_c,
            qroom_routing=qroom_routing,
            qroom_gain_mode=qroom_gain_mode,
        )
        return next_mass, next_air

    _, prediction = jax.lax.scan(
        step, initial_mass, (inputs, qroom, current_temperature)
    )
    return prediction


def _binned_residual(
    values: np.ndarray,
    residual: np.ndarray,
    *,
    bins: int = 12,
) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(residual)
    values = values[finite]
    residual = residual[finite]
    unique_values = np.unique(values)
    if len(unique_values) <= 4:
        output = {"x": [], "mean": [], "sem": [], "count": []}
        for value in unique_values:
            selected = values == value
            count = int(np.sum(selected))
            output["x"].append(float(value))
            output["mean"].append(float(np.mean(residual[selected])))
            output["sem"].append(
                float(np.std(residual[selected]) / math.sqrt(max(count, 1)))
            )
            output["count"].append(count)
        return output
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return {"x": [], "mean": [], "sem": [], "count": []}
    index = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    output = {"x": [], "mean": [], "sem": [], "count": []}
    for bin_index in range(len(edges) - 1):
        selected = index == bin_index
        if not np.any(selected):
            continue
        count = int(np.sum(selected))
        output["x"].append(float(np.mean(values[selected])))
        output["mean"].append(float(np.mean(residual[selected])))
        output["sem"].append(
            float(np.std(residual[selected]) / math.sqrt(max(count, 1)))
        )
        output["count"].append(count)
    return output


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    starts = np.maximum(index + 1 - window, 0)
    counts = index + 1 - starts
    return (cumulative[index + 1] - cumulative[starts]) / counts


def _residual_feature_values(
    inputs: np.ndarray,
    current_temperature: np.ndarray,
    qroom: np.ndarray,
    datetime: np.ndarray,
) -> dict[str, np.ndarray]:
    dates = np.asarray(datetime).astype("datetime64[D]")
    day_of_year = (dates - dates.astype("datetime64[Y]")).astype(np.int64) + 1
    return {
        "Tout [C]": np.asarray(inputs[:, TOUT_INDEX]),
        "Tin - Tout [C]": np.asarray(current_temperature - inputs[:, TOUT_INDEX]),
        "Ventilation": np.asarray(inputs[:, VENT_INDEX]),
        "Solar [W/m2]": np.asarray(inputs[:, SOLAR_INDEX]),
        "Qroom [W/m2]": np.asarray(qroom),
        "Heating availability": np.asarray(inputs[:, AVAILABILITY_INDEX]),
        "Day of year": np.asarray(day_of_year),
    }


def _windows(
    inputs: np.ndarray,
    targets: np.ndarray,
    initials: np.ndarray,
    *,
    length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(inputs) < length:
        starts = np.asarray([0], dtype=np.int32)
        length = len(inputs)
    else:
        starts = np.arange(0, len(inputs) - length + 1, stride, dtype=np.int32)
        final = len(inputs) - length
        if starts[-1] != final:
            starts = np.append(starts, final)
    return (
        np.stack([inputs[start : start + length] for start in starts]),
        np.stack([targets[start : start + length] for start in starts]),
        np.stack([initials[start] for start in starts]),
        starts,
    )


def _event_temperature_rmse(
    inputs: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    policy: str,
    horizon_steps: int = 12,
) -> tuple[float, int]:
    if policy == "force_off":
        signal = inputs[:, FORCE_OFF_INDEX]
        starts = np.flatnonzero((signal > 0.5) & ~np.r_[False, signal[:-1] > 0.5])
    else:
        signal = inputs[:, 0]
        starts = np.flatnonzero(np.abs(np.diff(signal, prepend=signal[0])) >= 0.05)
    starts = starts[(starts > 0) & (starts + horizon_steps <= len(target))]
    if not len(starts):
        return float("nan"), 0
    target_response = np.asarray(
        [target[start : start + horizon_steps].mean() - target[start - 1] for start in starts]
    )
    prediction_response = np.asarray(
        [
            prediction[start : start + horizon_steps].mean() - prediction[start - 1]
            for start in starts
        ]
    )
    return (
        float(np.sqrt(np.mean(np.square(prediction_response - target_response)))),
        int(len(starts)),
    )


def _fit_order(
    order: int,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    train_initials: np.ndarray,
    scales: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, jax.Array], dict[str, float]]:
    params = _initial_parameters(order)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adam(args.learning_rate),
    )
    state = optimizer.init(params)
    temperature_scale = max(float(np.std(train_targets[..., 0])), 0.25)

    @jax.jit
    def loss_and_grad(model_params):
        predictions = jax.vmap(
            lambda x, y, t: _rollout(
                model_params,
                x,
                y[:, 1],
                t,
                jnp.asarray(scales),
                delta_cap_c=args.temperature_delta_cap,
                qroom_routing=args.qroom_routing,
                qroom_gain_mode=args.qroom_gain_mode,
            )
        )(
            jnp.asarray(train_inputs),
            jnp.asarray(train_targets),
            jnp.asarray(train_initials),
        )
        return jnp.mean(
            jnp.square((predictions - jnp.asarray(train_targets[..., 0])) / temperature_scale)
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss_and_grad))
    best_loss = float("inf")
    best_params = params
    wait = 0
    max_grad = 0.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        loss, gradients = value_and_grad(params)
        grad_norm = float(optax.global_norm(gradients))
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
        loss_value = float(loss)
        max_grad = max(max_grad, grad_norm)
        if loss_value < best_loss - args.early_stopping_min_delta:
            best_loss = loss_value
            best_params = params
            wait = 0
        else:
            wait += 1
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"order={order} epoch={epoch + 1:03d} loss={loss_value:.6f} "
                f"grad={grad_norm:.3g} best={best_loss:.6f}",
                flush=True,
            )
        if wait >= args.early_stopping_patience:
            break
    return best_params, {
        "train_loss": best_loss,
        "epochs_completed": float(epoch + 1),
        "max_grad_norm": max_grad,
        "training_seconds": time.perf_counter() - started,
    }


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_profile = _load_policy_profile(args.dataset, args.profile_id, policy=args.policy)
    profile = policy_profile.profile
    train, test = _temporal_segments(policy_profile, args.train_fraction)
    train_inputs_raw = np.asarray(profile.inputs[train.start : train.end])
    train_targets_raw = np.asarray(profile.targets[train.start : train.end])
    train_initials_raw = np.asarray(profile.initial_temperature[train.start : train.end])
    train_current_raw = np.asarray(train_initials_raw[:, 0])
    train_datetime = np.asarray(profile.datetime[train.start : train.end])
    train_inputs, train_targets, train_initials, _ = _windows(
        train_inputs_raw,
        train_targets_raw,
        train_initials_raw,
        length=args.sequence_length,
        stride=args.stride,
    )
    scales = np.asarray(
        [
            max(float(np.sqrt(np.mean(np.square(train_targets_raw[:, 1])))), 1e-3),
            max(float(np.sqrt(np.mean(np.square(train_inputs_raw[:, SOLAR_INDEX])))), 1e-3),
            max(float(np.sqrt(np.mean(np.square(train_inputs_raw[:, INTERNAL_INDEX])))), 1e-3),
            max(float(np.sqrt(np.mean(np.square(train_inputs_raw[:, OCCUPANCY_INDEX])))), 1e-3),
            max(float(np.sqrt(np.mean(np.square(train_inputs_raw[:, VENT_INDEX])))), 1e-5),
        ],
        dtype=np.float32,
    )
    test_inputs = np.asarray(profile.inputs[test.start : test.end])
    test_target = np.asarray(profile.targets[test.start : test.end, 0])
    test_qroom = np.asarray(profile.targets[test.start : test.end, 1])
    test_current = np.asarray(
        profile.initial_temperature[test.start : test.end, 0]
    )
    test_datetime = np.asarray(profile.datetime[test.start : test.end])
    initial_temperature = np.asarray(profile.initial_temperature[test.start])

    metrics: dict[str, Any] = {
        "profile_id": args.profile_id,
        "policy": args.policy,
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "scales": scales.tolist(),
    }
    traces: dict[int, np.ndarray] = {}
    residual_traces: dict[int, dict[str, Any]] = {}
    for order in args.orders:
        params, training = _fit_order(
            order,
            train_inputs,
            train_targets,
            train_initials,
            scales,
            args=args,
        )
        prediction = np.asarray(
            _rollout(
                params,
                jnp.asarray(test_inputs),
                jnp.asarray(test_qroom),
                jnp.asarray(initial_temperature),
                jnp.asarray(scales),
                delta_cap_c=args.temperature_delta_cap,
                qroom_routing=args.qroom_routing,
                qroom_gain_mode=args.qroom_gain_mode,
            )
        )
        one_step = np.asarray(
            jax.vmap(
                lambda row, heat, temperature: _rollout(
                    params,
                    row[None, :],
                    heat[None],
                    temperature,
                    jnp.asarray(scales),
                    delta_cap_c=args.temperature_delta_cap,
                    qroom_routing=args.qroom_routing,
                    qroom_gain_mode=args.qroom_gain_mode,
                )[0]
            )(
                jnp.asarray(test_inputs),
                jnp.asarray(test_qroom),
                jnp.asarray(profile.initial_temperature[test.start : test.end]),
            )
        )
        event_rmse, event_count = _event_temperature_rmse(
            test_inputs, test_target, prediction, policy=args.policy
        )
        train_teacher_prediction = np.asarray(
            _teacher_forced_rollout(
                params,
                jnp.asarray(train_inputs_raw),
                jnp.asarray(train_targets_raw[:, 1]),
                jnp.asarray(train_current_raw),
                jnp.asarray(scales),
                delta_cap_c=args.temperature_delta_cap,
                qroom_routing=args.qroom_routing,
                qroom_gain_mode=args.qroom_gain_mode,
            )
        )
        test_teacher_prediction = np.asarray(
            _teacher_forced_rollout(
                params,
                jnp.asarray(test_inputs),
                jnp.asarray(test_qroom),
                jnp.asarray(test_current),
                jnp.asarray(scales),
                delta_cap_c=args.temperature_delta_cap,
                qroom_routing=args.qroom_routing,
                qroom_gain_mode=args.qroom_gain_mode,
            )
        )
        train_residual = train_targets_raw[:, 0] - train_teacher_prediction
        test_residual = test_target - test_teacher_prediction
        train_feature_values = _residual_feature_values(
            train_inputs_raw,
            train_current_raw,
            train_targets_raw[:, 1],
            train_datetime,
        )
        test_feature_values = _residual_feature_values(
            test_inputs,
            test_current,
            test_qroom,
            test_datetime,
        )
        residual_bins = {
            feature: {
                "train": _binned_residual(values, train_residual),
                "test": _binned_residual(
                    test_feature_values[feature], test_residual
                ),
            }
            for feature, values in train_feature_values.items()
        }
        values = _parameter_values(params)
        metrics[f"rc_order_{order}"] = {
            **training,
            "one_step_temperature_rmse_c": _rmse(one_step, test_target),
            "autonomous_temperature_rmse_c": _rmse(prediction, test_target),
            "autonomous_temperature_bias_c": float(np.mean(prediction - test_target)),
            "event_temperature_response_rmse_c": event_rmse,
            "event_count": event_count,
            "base_coupling_per_step": float(values["base_coupling"]),
            "vent_coupling_per_step": float(values["vent_coupling"]),
            "coupling_weights": np.asarray(values["coupling_weights"]).tolist(),
            "mass_rates_per_step": np.asarray(values["mass_rates"]).tolist(),
            "forcing_gains_c_per_scaled_input": np.asarray(values["forcing_gains"]).tolist(),
            "qroom_routing": args.qroom_routing,
            "qroom_routing_weights": np.asarray(
                values["qroom_routing_weights"]
            ).tolist(),
            "qroom_gain_mode": args.qroom_gain_mode,
            "qroom_gain_increment_c_per_scaled_input": float(
                values["qroom_gain_increment"]
            ),
            "qroom_gain_threshold_scaled": float(
                values["qroom_gain_threshold"]
            ),
            "qroom_gain_threshold_w_m2": float(
                values["qroom_gain_threshold"] * scales[0]
            ),
            "qroom_gain_base_c_per_w_m2": float(
                values["forcing_gains"][0] / scales[0]
            ),
            "qroom_gain_asymptotic_c_per_w_m2": float(
                (
                    values["forcing_gains"][0]
                    + (
                        values["qroom_gain_increment"]
                        if args.qroom_gain_mode == "monotone"
                        else 0.0
                    )
                )
                / scales[0]
            ),
            "teacher_forced_residual": {
                "train_mean_c": float(np.mean(train_residual)),
                "train_rmse_c": _rmse(
                    train_teacher_prediction, train_targets_raw[:, 0]
                ),
                "test_mean_c": float(np.mean(test_residual)),
                "test_rmse_c": _rmse(test_teacher_prediction, test_target),
                "max_abs_test_bin_mean_c": {
                    feature: (
                        float(np.max(np.abs(parts["test"]["mean"])))
                        if parts["test"]["mean"]
                        else float("nan")
                    )
                    for feature, parts in residual_bins.items()
                },
                "binned_mean_c": residual_bins,
            },
        }
        traces[order] = prediction
        residual_traces[order] = {
            "bins": residual_bins,
            "train_residual": train_residual,
            "test_residual": test_residual,
        }

    datetime = test_datetime
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Autonomous Tin with observed Qroom", "Tin error"),
    )
    figure.add_trace(
        go.Scatter(x=datetime, y=test_target, name="EnergyPlus", line={"color": "#202124", "width": 2}),
        row=1,
        col=1,
    )
    colors = {1: "#68757d", 2: "#187b78", 3: "#d17837"}
    for order, prediction in traces.items():
        name = f"RC order {order}"
        figure.add_trace(
            go.Scatter(x=datetime, y=prediction, name=name, line={"color": colors.get(order, "#7b2cbf"), "width": 1.2}),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=prediction - test_target,
                name=f"{name} error",
                legendgroup=name,
                showlegend=False,
                line={"color": colors.get(order, "#7b2cbf"), "width": 1.0},
            ),
            row=2,
            col=1,
        )
    figure.add_hline(y=0.0, line={"color": "#666", "dash": "dot"}, row=2, col=1)
    figure.update_yaxes(title_text="Tin [C]", row=1, col=1)
    figure.update_yaxes(title_text="Error [C]", row=2, col=1)
    figure.update_layout(
        title=f"Stable RC order sufficiency: {args.policy} / {args.profile_id}",
        height=850,
        hovermode="x unified",
        template="plotly_white",
    )
    html_path = args.output_dir / "rc_order_sufficiency.html"
    json_path = args.output_dir / "rc_order_sufficiency_metrics.json"
    figure.write_html(html_path, include_plotlyjs="cdn")

    residual_figure = make_subplots(
        rows=4,
        cols=2,
        vertical_spacing=0.09,
        horizontal_spacing=0.10,
        subplot_titles=(
            "Residual vs Tout",
            "Residual vs Tin - Tout",
            "Residual vs ventilation",
            "Residual vs solar",
            "Residual vs Qroom",
            "Residual vs heating availability",
            "Residual vs day of year",
            "Seven-day rolling residual",
        ),
    )
    feature_panels = (
        ("Tout [C]", 1, 1),
        ("Tin - Tout [C]", 1, 2),
        ("Ventilation", 2, 1),
        ("Solar [W/m2]", 2, 2),
        ("Qroom [W/m2]", 3, 1),
        ("Heating availability", 3, 2),
        ("Day of year", 4, 1),
    )
    for order, diagnostics in residual_traces.items():
        color = colors.get(order, "#7b2cbf")
        for feature, row, col in feature_panels:
            for split_name, dash in (("train", "dot"), ("test", "solid")):
                binned = diagnostics["bins"][feature][split_name]
                residual_figure.add_trace(
                    go.Scatter(
                        x=binned["x"],
                        y=binned["mean"],
                        error_y={"type": "data", "array": binned["sem"], "visible": True},
                        mode="lines+markers",
                        name=f"RC{order} {split_name}",
                        legendgroup=f"RC{order}-{split_name}",
                        showlegend=(feature == "Tout [C]"),
                        line={"color": color, "dash": dash},
                    ),
                    row=row,
                    col=col,
                )
        for split_name, split_datetime, dash in (
            ("train", train_datetime, "dot"),
            ("test", test_datetime, "solid"),
        ):
            residual_figure.add_trace(
                go.Scatter(
                    x=split_datetime,
                    y=_rolling_mean(
                        diagnostics[f"{split_name}_residual"], 7 * 96
                    ),
                    mode="lines",
                    name=f"RC{order} {split_name} rolling",
                    legendgroup=f"RC{order}-{split_name}",
                    showlegend=False,
                    line={"color": color, "dash": dash},
                ),
                row=4,
                col=2,
            )
    for row in range(1, 5):
        for col in range(1, 3):
            residual_figure.add_hline(
                y=0.0,
                line={"color": "#777", "dash": "dot", "width": 1},
                row=row,
                col=col,
            )
    residual_figure.update_yaxes(title_text="True - predicted [C]")
    residual_figure.update_layout(
        title=f"Teacher-forced RC residuals: {args.policy} / {args.profile_id}",
        height=1350,
        hovermode="closest",
        template="plotly_white",
    )
    residual_html_path = args.output_dir / "rc_residual_diagnostics.html"
    residual_figure.write_html(residual_html_path, include_plotlyjs="cdn")
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"saved_rc_order_sufficiency={html_path}", flush=True)
    print(f"saved_rc_residual_diagnostics={residual_html_path}", flush=True)
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile-id", type=int, required=True)
    parser.add_argument("--policy", choices=("setpoint", "force_off"), required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--stride", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--temperature-delta-cap", type=float, default=2.0)
    parser.add_argument(
        "--qroom-routing",
        choices=("air_only", "positive_split"),
        default="air_only",
    )
    parser.add_argument(
        "--qroom-gain-mode",
        choices=("constant", "monotone"),
        default="constant",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(order not in (1, 2, 3) for order in args.orders):
        parser.error("--orders accepts only 1, 2 and 3")
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must lie strictly between zero and one")
    if min(args.sequence_length, args.stride, args.epochs) < 1:
        parser.error("sequence length, stride and epochs must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
