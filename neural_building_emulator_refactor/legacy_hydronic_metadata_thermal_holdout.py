"""Test the legacy HP/hydronic block with metadata-fixed thermal dynamics.

The metadata-fixed thermal model is first fitted with observed Qroom and then
frozen. In ``observed`` mode, the legacy controller/buffer receives observed
Pel_SH and isolates the Pel_SH-to-Qroom link. In ``predicted`` mode, its own HP
emission drives the buffer, Qroom, and the frozen thermal model autonomously.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    HP_FORCE_OFF_REQUESTED_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from neural_building_emulator.models.contracting_closed_loop_hp import (
    ContractingClosedLoopHPEmulator,
)
from neural_building_emulator.models.emulator import MLP

from .conditional_state_space_sufficiency import (
    INPUT_NAMES,
    EXOGENOUS_INDICES,
    Scales,
    Trajectory,
    _fit_scales,
    _fit_variant,
    _load_profiles,
    _make_annual_evaluator,
    _make_trajectories,
    _make_windows,
    _metrics,
    _mlp,
    _normalized_context,
    _state_space_parameters,
)
from .legacy_thermal_holdout_comparison import (
    LegacyScales,
    _array_norm,
    _detach,
    _initial_model,
    _legacy_scales,
)


ORACLE_VARIANT = "metadata_fixed_observed_qroom"
OBSERVED_PEL_VARIANT = "legacy_hydronic_observed_pel_metadata_fixed"
PREDICTED_PEL_VARIANT = "legacy_hp_hydronic_metadata_fixed"


class MetadataCapacityHydronic(eqx.Module):
    base: ContractingClosedLoopHPEmulator
    capacity_net: MLP
    minimum_hours: float = eqx.field(static=True)
    maximum_hours: float = eqx.field(static=True)

    def capacity_hours(self, metadata: jax.Array) -> jax.Array:
        probability = jax.nn.sigmoid(self.capacity_net(metadata)[0])
        return self.minimum_hours + (
            self.maximum_hours - self.minimum_hours
        ) * probability

    def capacity_wh_m2(self, metadata: jax.Array) -> jax.Array:
        return self.capacity_hours(metadata) * jnp.asarray(
            self.base.hp_qroom_cap_w_m2, dtype=metadata.dtype
        )


def _metadata_capacity_model(
    base: ContractingClosedLoopHPEmulator,
    *,
    metadata_dim: int,
    hidden_dim: int,
    depth: int,
    minimum_hours: float,
    maximum_hours: float,
    initial_hours: float,
    seed: int,
) -> MetadataCapacityHydronic:
    network = MLP(
        metadata_dim,
        1,
        hidden_dim=hidden_dim,
        depth=depth,
        key=jax.random.PRNGKey(seed),
    )
    probability = (initial_hours - minimum_hours) / (
        maximum_hours - minimum_hours
    )
    initial_logit = math.log(probability / (1.0 - probability))
    network = eqx.tree_at(
        lambda value: value.layers[-1].weight,
        network,
        jnp.zeros_like(network.layers[-1].weight),
    )
    network = eqx.tree_at(
        lambda value: value.layers[-1].bias,
        network,
        jnp.asarray([initial_logit], dtype=network.layers[-1].bias.dtype),
    )
    return MetadataCapacityHydronic(
        base=base,
        capacity_net=network,
        minimum_hours=minimum_hours,
        maximum_hours=maximum_hours,
    )


def _contiguous_events(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _event_aligned_records(
    trajectory: Trajectory,
    temperature: np.ndarray,
    qroom: np.ndarray,
    pel: np.ndarray,
    *,
    dt_hours: float,
    context_hours: float,
) -> list[dict[str, Any]]:
    profile = trajectory.profile
    force_off_index = INPUT_NAMES.index(HP_FORCE_OFF_REQUESTED_COLUMN)
    availability_index = INPUT_NAMES.index(SPACE_HEATING_AVAILABILITY_COLUMN)
    effective_force_off = (
        (profile.inputs[:, force_off_index] > 0.5)
        & (profile.inputs[:, availability_index] > 0.5)
    )
    context_steps = max(1, int(round(context_hours / dt_hours)))
    observed = {
        "tin": np.asarray(profile.targets[:, 0]),
        "qroom": np.asarray(profile.targets[:, 1]),
        "pel": np.asarray(profile.targets[:, 2]),
    }
    predicted = {
        "tin": np.asarray(temperature),
        "qroom": np.asarray(qroom),
        "pel": np.asarray(pel),
    }
    records: list[dict[str, Any]] = []
    for event_index, (start, stop) in enumerate(
        _contiguous_events(effective_force_off), start=1
    ):
        pre_start = max(0, start - context_steps)
        post_stop = min(len(effective_force_off), stop + context_steps)
        if start - pre_start < context_steps or post_stop - stop < context_steps:
            continue
        row: dict[str, Any] = {
            "profile_id": profile.profile_id,
            "policy": trajectory.policy,
            "event_index": event_index,
            "event_start": profile.datetime[start],
            "event_stop": profile.datetime[stop - 1],
            "duration_hours": (stop - start) * dt_hours,
        }
        slices = {
            "pre": slice(pre_start, start),
            "event": slice(start, stop),
            "post": slice(stop, post_stop),
        }
        for signal in ("tin", "qroom", "pel"):
            for source, values in (
                ("observed", observed[signal]),
                ("predicted", predicted[signal]),
            ):
                means = {
                    period: float(np.mean(values[period_slice]))
                    for period, period_slice in slices.items()
                }
                row[f"{source}_{signal}_pre_mean"] = means["pre"]
                row[f"{source}_{signal}_event_mean"] = means["event"]
                row[f"{source}_{signal}_post_mean"] = means["post"]
                row[f"{source}_{signal}_event_delta"] = (
                    means["event"] - means["pre"]
                )
                row[f"{source}_{signal}_rebound"] = (
                    means["post"] - means["event"]
                )
        records.append(row)
    return records


def _masked_rmse(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    if not np.any(mask):
        return float("nan")
    residual = np.asarray(prediction)[mask] - np.asarray(target)[mask]
    return float(np.sqrt(np.mean(np.square(residual))))


def _force_off_regime_metrics(
    trajectory: Trajectory,
    temperature: np.ndarray,
    qroom: np.ndarray,
    pel: np.ndarray,
    *,
    dt_hours: float,
    recovery_hours: float,
) -> dict[str, float]:
    profile = trajectory.profile
    force_off_index = INPUT_NAMES.index(HP_FORCE_OFF_REQUESTED_COLUMN)
    availability_index = INPUT_NAMES.index(SPACE_HEATING_AVAILABILITY_COLUMN)
    availability = profile.inputs[:, availability_index] > 0.5
    event_mask = (profile.inputs[:, force_off_index] > 0.5) & availability
    recovery_mask = np.zeros_like(event_mask)
    recovery_steps = max(1, int(round(recovery_hours / dt_hours)))
    for _, stop in _contiguous_events(event_mask):
        recovery_mask[stop : min(len(event_mask), stop + recovery_steps)] = True
    recovery_mask &= availability & ~event_mask
    ordinary_mask = availability & ~event_mask & ~recovery_mask
    result: dict[str, float] = {
        "force_off_fraction": float(np.mean(event_mask)),
        "recovery_fraction": float(np.mean(recovery_mask)),
    }
    for signal, prediction, target in (
        ("tin", temperature, profile.targets[:, 0]),
        ("qroom", qroom, profile.targets[:, 1]),
        ("pel", pel, profile.targets[:, 2]),
    ):
        result[f"{signal}_force_off_rmse"] = _masked_rmse(
            prediction, target, event_mask
        )
        result[f"{signal}_recovery_rmse"] = _masked_rmse(
            prediction, target, recovery_mask
        )
        result[f"{signal}_ordinary_heating_rmse"] = _masked_rmse(
            prediction, target, ordinary_mask
        )
    return result


def _combined_rollout(
    hydronic_model,
    thermal_params: dict[str, Any],
    metadata_raw: jax.Array,
    inputs_raw: jax.Array,
    pel_observed: jax.Array,
    initial_temperature_c: jax.Array,
    thermal_scales: Scales,
    hydronic_scales: LegacyScales,
    *,
    state_dim: int,
    encoded_dim: int,
    gamma: float,
    bptt_truncate_steps: int,
    pel_source: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    base_model = (
        hydronic_model.base
        if isinstance(hydronic_model, MetadataCapacityHydronic)
        else hydronic_model
    )
    thermal_metadata = (
        metadata_raw - jnp.asarray(thermal_scales.metadata_mean)
    ) / jnp.asarray(thermal_scales.metadata_std)
    hydronic_metadata = (
        metadata_raw - jnp.asarray(hydronic_scales.shared.metadata_mean)
    ) / jnp.asarray(hydronic_scales.shared.metadata_std)
    hydronic_inputs = (
        inputs_raw - jnp.asarray(hydronic_scales.input_mean)
    ) / jnp.asarray(hydronic_scales.input_std)
    initial_temperature = (
        initial_temperature_c - thermal_scales.temperature_mean
    ) / thermal_scales.temperature_std
    initial_vector = jnp.atleast_1d(initial_temperature)

    initial_exogenous, _ = _normalized_context(
        inputs_raw[0],
        initial_temperature,
        jnp.asarray(0.0, dtype=initial_temperature.dtype),
        thermal_scales,
    )
    _, _, _, output, output_bias = _state_space_parameters(
        thermal_params,
        "metadata_fixed",
        jnp.asarray(0, dtype=jnp.int32),
        thermal_metadata,
        initial_exogenous,
        state_dim=state_dim,
        encoded_dim=encoded_dim,
        gamma=gamma,
    )
    thermal_state = output * (
        (initial_temperature - output_bias)
        / (jnp.dot(output, output) + 1e-5)
    )
    energy = base_model.initial_energy(hydronic_metadata, initial_vector)
    dynamic_energy_cap = None
    if isinstance(hydronic_model, MetadataCapacityHydronic):
        dynamic_energy_cap = hydronic_model.capacity_wh_m2(hydronic_metadata)
        energy = jnp.minimum(energy, dynamic_energy_cap)

    def step(carry, values):
        state_t, temperature_t, energy_t = carry
        step_index, input_raw_t, input_hydronic_t, pel_t = values
        state_t = _detach(state_t, step_index, bptt_truncate_steps)
        temperature_t = _detach(
            temperature_t, step_index, bptt_truncate_steps
        )
        energy_t = _detach(energy_t, step_index, bptt_truncate_steps)

        feedback_temperature = (
            base_model.temperature_feedback_gradient_scale * temperature_t
            + (1.0 - base_model.temperature_feedback_gradient_scale)
            * jax.lax.stop_gradient(temperature_t)
        )
        encoded_input = base_model.encode_input(input_hydronic_t)
        (
            _,
            decoded_qroom,
            predicted_pel,
            decoded_energy_next,
            _,
            requested_qroom,
        ) = base_model.decode_hp(
            hydronic_metadata,
            encoded_input,
            input_hydronic_t,
            energy_t,
            feedback_temperature,
        )
        if pel_source == "observed":
            cop_t = base_model._cop(input_hydronic_t)
            qroom_t, _, energy_next = base_model._buffer_draw(
                energy_t,
                cop_t * jnp.maximum(pel_t, 0.0),
                requested_qroom,
                jnp.asarray(base_model.hp_dt_hours, dtype=pel_t.dtype),
                jax.nn.softplus(base_model.raw_energy_loss_rate),
            )
        else:
            qroom_t = decoded_qroom
            energy_next = decoded_energy_next
        if dynamic_energy_cap is not None:
            energy_next = jnp.minimum(energy_next, dynamic_energy_cap)

        exogenous, encoder_input = _normalized_context(
            input_raw_t,
            temperature_t,
            qroom_t / thermal_scales.qroom_scale,
            thermal_scales,
        )
        encoded = jnp.tanh(_mlp(thermal_params["encoder"], encoder_input))
        rho, b_matrix, bias, output_t, output_bias_t = _state_space_parameters(
            thermal_params,
            "metadata_fixed",
            jnp.asarray(0, dtype=jnp.int32),
            thermal_metadata,
            exogenous,
            state_dim=state_dim,
            encoded_dim=encoded_dim,
            gamma=gamma,
        )
        state_next = rho * state_t + b_matrix @ encoded + bias
        raw_temperature = jnp.dot(output_t, state_next) + output_bias_t
        free_temperature = 5.0 * jnp.tanh(raw_temperature / 5.0)
        temperature_next = 5.0 * jnp.tanh(free_temperature / 5.0)
        temperature_c = (
            thermal_scales.temperature_mean
            + thermal_scales.temperature_std * temperature_next
        )
        return (state_next, temperature_next, energy_next), (
            temperature_c,
            qroom_t,
            predicted_pel,
            energy_next,
        )

    steps = jnp.arange(inputs_raw.shape[0], dtype=jnp.int32)
    _, outputs = jax.lax.scan(
        step,
        (thermal_state, initial_temperature, energy),
        (steps, inputs_raw, hydronic_inputs, pel_observed),
    )
    return outputs


def _fit_hydronic(
    model,
    thermal_params: dict[str, Any],
    windows: dict[str, np.ndarray],
    thermal_scales: Scales,
    hydronic_scales: LegacyScales,
    args: argparse.Namespace,
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adam(args.learning_rate),
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(candidate, metadata, inputs, pel, initial, qroom, target):
        temperature_prediction, qroom_prediction, pel_prediction, _ = jax.vmap(
            lambda metadata_t, inputs_t, pel_t, initial_t: _combined_rollout(
                candidate,
                thermal_params,
                metadata_t,
                inputs_t,
                pel_t,
                initial_t,
                thermal_scales,
                hydronic_scales,
                state_dim=args.state_dim,
                encoded_dim=args.encoded_dim,
                gamma=args.gamma,
                bptt_truncate_steps=args.bptt_truncate_steps,
                pel_source=args.pel_source,
            )
        )(
            metadata,
            inputs,
            pel,
            initial,
        )
        temperature_residual = (
            temperature_prediction - target
        ) / thermal_scales.temperature_std
        qroom_residual = (
            qroom_prediction - qroom
        ) / hydronic_scales.qroom_std
        temperature_loss = jnp.mean(jnp.square(temperature_residual))
        qroom_loss = jnp.mean(jnp.square(qroom_residual))
        pel_residual = (pel_prediction - pel) / hydronic_scales.pel_std
        pel_loss = jnp.mean(jnp.square(pel_residual))
        total = (
            args.temperature_loss_weight * temperature_loss
            + args.qroom_loss_weight * qroom_loss
            + args.pel_loss_weight * pel_loss
        )
        return total, (temperature_loss, qroom_loss, pel_loss)

    @eqx.filter_jit
    def train_step(candidate, state, batch):
        (loss, components), gradients = loss_fn(
            candidate,
            batch["metadata"],
            batch["inputs"],
            batch["pel"],
            batch["initial"],
            batch["qroom"],
            batch["target"],
        )
        grad_norm = _array_norm(gradients)
        updates, state = optimizer.update(gradients, state, candidate)
        candidate = eqx.apply_updates(candidate, updates)
        return candidate, state, loss, components, grad_norm

    rng = np.random.default_rng(args.seed)
    count = len(windows["target"])
    best_loss = math.inf
    best_model = model
    wait = 0
    max_grad = 0.0
    best_temperature_loss = math.inf
    best_qroom_loss = math.inf
    best_pel_loss = math.inf
    started = time.perf_counter()
    epoch = -1
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        losses: list[float] = []
        temperature_losses: list[float] = []
        qroom_losses: list[float] = []
        pel_losses: list[float] = []
        for start in range(0, count, args.batch_size):
            indices = order[start : start + args.batch_size]
            if len(indices) < args.batch_size:
                indices = np.concatenate(
                    (indices, order[: args.batch_size - len(indices)])
                )
            batch = {
                key: jnp.asarray(value[indices])
                for key, value in windows.items()
            }
            model, optimizer_state, loss, components, grad_norm = train_step(
                model, optimizer_state, batch
            )
            values = (
                float(loss),
                float(components[0]),
                float(components[1]),
                float(components[2]),
            )
            grad_value = float(grad_norm)
            if not all(np.isfinite(value) for value in (*values, grad_value)):
                raise FloatingPointError(
                    f"Non-finite hydronic update at epoch {epoch + 1}"
                )
            losses.append(values[0])
            temperature_losses.append(values[1])
            qroom_losses.append(values[2])
            pel_losses.append(values[3])
            max_grad = max(max_grad, grad_value)
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss - args.early_stopping_min_delta:
            best_loss = epoch_loss
            best_temperature_loss = float(np.mean(temperature_losses))
            best_qroom_loss = float(np.mean(qroom_losses))
            best_pel_loss = float(np.mean(pel_losses))
            best_model = model
            wait = 0
        else:
            wait += 1
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"hydronic epoch={epoch + 1:04d} loss={epoch_loss:.6f} "
                f"temperature={np.mean(temperature_losses):.6f} "
                f"qroom={np.mean(qroom_losses):.6f} best={best_loss:.6f} "
                f"pel={np.mean(pel_losses):.6f} "
                f"max_grad={max_grad:.3g}",
                flush=True,
            )
        if wait >= args.early_stopping_patience:
            break
    return best_model, {
        "train_total_normalized_mse": best_loss,
        "train_temperature_normalized_mse": best_temperature_loss,
        "train_qroom_normalized_mse": best_qroom_loss,
        "train_pel_normalized_mse": best_pel_loss,
        "epochs_completed": float(epoch + 1),
        "max_grad_norm": max_grad,
        "training_seconds": time.perf_counter() - started,
    }


def _qroom_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    target_energy = float(np.sum(target))
    return {
        "qroom_rmse_w_m2": float(np.sqrt(np.mean(np.square(residual)))),
        "qroom_mae_w_m2": float(np.mean(np.abs(residual))),
        "qroom_bias_w_m2": float(np.mean(residual)),
        "qroom_energy_ratio": (
            float(np.sum(prediction)) / target_energy
            if abs(target_energy) > 1e-9
            else float("nan")
        ),
    }


def _pel_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    target_energy = float(np.sum(target))
    return {
        "pel_rmse_w_m2": float(np.sqrt(np.mean(np.square(residual)))),
        "pel_mae_w_m2": float(np.mean(np.abs(residual))),
        "pel_bias_w_m2": float(np.mean(residual)),
        "pel_energy_ratio": (
            float(np.sum(prediction)) / target_energy
            if abs(target_energy) > 1e-9
            else float("nan")
        ),
    }


def _write_dashboard(frame: pd.DataFrame, output: Path, pipeline_variant: str) -> None:
    figure = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Annual Tin RMSE",
            "Winter Tin RMSE",
            "Qroom RMSE",
            "Pel_SH RMSE",
        ),
    )
    colors = {ORACLE_VARIANT: "#d97706", pipeline_variant: "#355ea8"}
    for variant in (ORACLE_VARIANT, pipeline_variant):
        selected = frame[frame.variant.eq(variant)].sort_values(
            ["profile_id", "policy"]
        )
        labels = selected.apply(
            lambda row: f"{int(row.profile_id)} / {row.policy}", axis=1
        )
        for column, metric in enumerate(
            (
                "annual_temperature_rmse_c",
                "winter_temperature_rmse_c",
                "qroom_rmse_w_m2",
                "pel_rmse_w_m2",
            ),
            start=1,
        ):
            if metric not in selected:
                continue
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=selected[metric],
                    name=variant,
                    legendgroup=variant,
                    showlegend=column == 1,
                    marker_color=colors[variant],
                    hovertemplate=(
                        f"model={variant}<br>profile=%{{x}}<br>"
                        "value=%{y:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    figure.update_layout(
        title=(
            f"{pipeline_variant}: legacy HP/hydronic -> metadata-fixed thermal"
        ),
        barmode="group",
        height=720,
        template="plotly_white",
    )
    figure.write_html(output, include_plotlyjs="cdn")


def _write_focus_traces(
    trajectories: list[Trajectory],
    traces: dict[
        tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    worst_profile_ids: list[int],
    output: Path,
) -> None:
    setpoint_index = INPUT_NAMES.index(SETPOINT_TIMESERIES_COLUMN)
    selected = [
        trajectory
        for trajectory in trajectories
        if trajectory.profile.profile_id in worst_profile_ids
    ]
    figure = make_subplots(
        rows=3 * len(selected),
        cols=1,
        subplot_titles=tuple(
            title
            for trajectory in selected
            for title in (
                f"{trajectory.profile.profile_id} / {trajectory.policy}: Tin",
                f"{trajectory.profile.profile_id} / {trajectory.policy}: Qroom",
                f"{trajectory.profile.profile_id} / {trajectory.policy}: Pel_SH",
            )
        ),
        vertical_spacing=0.012,
    )
    for index, trajectory in enumerate(selected):
        profile = trajectory.profile
        temperature, qroom, pel = traces[(profile.profile_id, trajectory.policy)]
        subset = slice(None, None, 4)
        temperature_row = 3 * index + 1
        qroom_row = temperature_row + 1
        pel_row = qroom_row + 1
        for row, observed, predicted, label in (
            (temperature_row, profile.targets[:, 0], temperature, "Tin"),
            (qroom_row, profile.targets[:, 1], qroom, "Qroom"),
            (pel_row, profile.targets[:, 2], pel, "Pel_SH"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=profile.datetime[subset],
                    y=observed[subset],
                    name="EnergyPlus",
                    legendgroup="EnergyPlus",
                    showlegend=index == 0 and label == "Tin",
                    line={"color": "#202124", "width": 1.6},
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=profile.datetime[subset],
                    y=predicted[subset],
                    name="Legacy hydronic + metadata-fixed",
                    legendgroup="prediction",
                    showlegend=index == 0 and label == "Tin",
                    line={"color": "#355ea8", "width": 1.0},
                ),
                row=row,
                col=1,
            )
            if label == "Tin":
                figure.add_trace(
                    go.Scatter(
                        x=profile.datetime[subset],
                        y=profile.inputs[subset, setpoint_index],
                        name="Tset",
                        legendgroup="Tset",
                        showlegend=index == 0,
                        line={"color": "#d97706", "width": 1.0, "dash": "dash"},
                    ),
                    row=row,
                    col=1,
                )
            figure.update_yaxes(
                title_text="C" if label == "Tin" else "W/m2", row=row, col=1
            )
    figure.update_layout(
        title="Worst holdout profiles for the isolated hydronic pipeline",
        height=max(900, 660 * len(selected)),
        hovermode="x unified",
        template="plotly_white",
    )
    figure.write_html(output, include_plotlyjs="cdn")


def _write_event_dashboard(frame: pd.DataFrame, output: Path) -> None:
    panels = (
        ("tin", "event_delta", "Tin change during force-off", "C"),
        ("pel", "event_delta", "Pel change during force-off", "W/m2"),
        ("qroom", "event_delta", "Qroom change during force-off", "W/m2"),
        ("tin", "rebound", "Tin rebound after release", "C"),
        ("pel", "rebound", "Pel rebound after release", "W/m2"),
        ("qroom", "rebound", "Qroom rebound after release", "W/m2"),
    )
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=tuple(panel[2] for panel in panels),
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )
    colors = frame["profile_id"].astype(str)
    for index, (signal, response, _, unit) in enumerate(panels):
        row = index // 3 + 1
        col = index % 3 + 1
        observed_column = f"observed_{signal}_{response}"
        predicted_column = f"predicted_{signal}_{response}"
        observed = frame[observed_column].to_numpy()
        predicted = frame[predicted_column].to_numpy()
        lower = float(np.nanmin(np.concatenate((observed, predicted))))
        upper = float(np.nanmax(np.concatenate((observed, predicted))))
        figure.add_trace(
            go.Scatter(
                x=observed,
                y=predicted,
                mode="markers",
                marker={
                    "color": pd.factorize(colors)[0],
                    "colorscale": "Viridis",
                    "size": 6,
                    "opacity": 0.68,
                },
                customdata=np.column_stack(
                    (
                        frame["profile_id"],
                        frame["event_start"].astype(str),
                        frame["duration_hours"],
                    )
                ),
                hovertemplate=(
                    "profile=%{customdata[0]}<br>start=%{customdata[1]}<br>"
                    "duration=%{customdata[2]:.2f} h<br>"
                    "observed=%{x:.3f}<br>predicted=%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line={"color": "#202124", "dash": "dash", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        figure.update_xaxes(title_text=f"EnergyPlus [{unit}]", row=row, col=col)
        figure.update_yaxes(title_text=f"Prediction [{unit}]", row=row, col=col)
    figure.update_layout(
        title=(
            "Event-aligned force-off validation: event means relative to the "
            "preceding 3 h"
        ),
        height=900,
        template="plotly_white",
    )
    figure.write_html(output, include_plotlyjs="cdn")


def _event_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selections = {
        "all_events": np.ones(len(frame), dtype=bool),
        "active_pre_pel_gt_0.5": frame["observed_pel_pre_mean"] > 0.5,
        "tin_drop_gt_0.1_c": frame["observed_tin_event_delta"] < -0.1,
        "tin_drop_gt_0.2_c": frame["observed_tin_event_delta"] < -0.2,
        "tin_drop_gt_0.5_c": frame["observed_tin_event_delta"] < -0.5,
        "tin_drop_gt_1.0_c": frame["observed_tin_event_delta"] < -1.0,
    }
    rows: list[dict[str, Any]] = []
    for selection, mask in selections.items():
        selected = frame[np.asarray(mask)]
        if selected.empty:
            continue
        for response in ("event_delta", "rebound"):
            for signal in ("tin", "qroom", "pel"):
                observed = selected[f"observed_{signal}_{response}"].to_numpy(
                    dtype=float
                )
                predicted = selected[f"predicted_{signal}_{response}"].to_numpy(
                    dtype=float
                )
                residual = predicted - observed
                correlation = float("nan")
                if (
                    len(selected) > 1
                    and np.std(observed) > 1e-12
                    and np.std(predicted) > 1e-12
                ):
                    correlation = float(np.corrcoef(observed, predicted)[0, 1])
                rows.append(
                    {
                        "selection": selection,
                        "event_count": len(selected),
                        "signal": signal,
                        "response": response,
                        "observed_mean": float(np.mean(observed)),
                        "predicted_mean": float(np.mean(predicted)),
                        "bias": float(np.mean(residual)),
                        "mae": float(np.mean(np.abs(residual))),
                        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                        "correlation": correlation,
                        "sign_agreement": float(
                            np.mean(np.sign(observed) == np.sign(predicted))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_variant = (
        OBSERVED_PEL_VARIANT
        if args.pel_source == "observed"
        else PREDICTED_PEL_VARIANT
    )
    reference_manifest = json.loads(
        (args.reference_dir / "state_space_sufficiency_manifest.json").read_text()
    )
    selected_ids = tuple(int(value) for value in reference_manifest["selected_profile_ids"])
    train_ids = tuple(int(value) for value in reference_manifest["train_profile_ids"])
    test_ids = tuple(int(value) for value in reference_manifest["test_profile_ids"])
    if not test_ids:
        raise ValueError("Reference run must contain a holdout split")
    setpoint_dataset = Path(reference_manifest["arguments"]["setpoint_dataset"])
    force_off_dataset = Path(reference_manifest["arguments"]["force_off_dataset"])
    setpoint = _load_profiles(setpoint_dataset, selected_ids)
    force_off = _load_profiles(force_off_dataset, selected_ids)
    building_indices = {
        profile_id: index for index, profile_id in enumerate(selected_ids)
    }
    train_trajectories = _make_trajectories(
        train_ids,
        setpoint=setpoint,
        force_off=force_off,
        building_indices=building_indices,
    )
    test_trajectories = _make_trajectories(
        test_ids,
        setpoint=setpoint,
        force_off=force_off,
        building_indices=building_indices,
    )
    thermal_scales = _fit_scales(train_trajectories)
    hydronic_scales = _legacy_scales(train_trajectories)
    windows = _make_windows(
        train_trajectories,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    print(
        f"train_profiles={len(train_ids)} test_profiles={len(test_ids)} "
        f"train_trajectories={len(train_trajectories)} windows={len(windows['target'])}",
        flush=True,
    )

    if args.thermal_params is None:
        thermal_params, thermal_training = _fit_variant(
            "metadata_fixed",
            windows,
            thermal_scales,
            args=args,
            num_buildings=len(selected_ids),
        )
    else:
        with args.thermal_params.open("rb") as stream:
            thermal_params = pickle.load(stream)
        thermal_training = {
            "loaded_thermal_params": str(args.thermal_params),
        }
        print(f"loaded_thermal_params={args.thermal_params}", flush=True)
    hydronic_args = argparse.Namespace(**vars(args))
    hydronic_args.hidden_dim = args.hydronic_hidden_dim
    hydronic_args.depth = args.hydronic_depth
    if args.metadata_conditioned_energy_capacity:
        hydronic_args.hp_energy_cap_hours = args.metadata_capacity_max_hours
    hydronic_model = _initial_model(
        hydronic_scales,
        metadata_dim=windows["metadata"].shape[-1],
        args=hydronic_args,
    )
    if args.metadata_conditioned_energy_capacity:
        hydronic_model = _metadata_capacity_model(
            hydronic_model,
            metadata_dim=windows["metadata"].shape[-1],
            hidden_dim=args.metadata_capacity_hidden_dim,
            depth=args.metadata_capacity_depth,
            minimum_hours=args.metadata_capacity_min_hours,
            maximum_hours=args.metadata_capacity_max_hours,
            initial_hours=args.metadata_capacity_initial_hours,
            seed=args.seed + 701,
        )
    if args.hydronic_model is None:
        hydronic_model, hydronic_training = _fit_hydronic(
            hydronic_model,
            thermal_params,
            windows,
            thermal_scales,
            hydronic_scales,
            args,
        )
    else:
        hydronic_model = eqx.tree_deserialise_leaves(
            args.hydronic_model, hydronic_model
        )
        hydronic_training = {
            "loaded_hydronic_model": str(args.hydronic_model),
        }
        print(f"loaded_hydronic_model={args.hydronic_model}", flush=True)

    oracle_evaluate = _make_annual_evaluator(
        thermal_params, "metadata_fixed", thermal_scales, args
    )
    combined_evaluate = eqx.filter_jit(
        lambda metadata, inputs, pel, initial: _combined_rollout(
            hydronic_model,
            thermal_params,
            metadata,
            inputs,
            pel,
            initial,
            thermal_scales,
            hydronic_scales,
            state_dim=args.state_dim,
            encoded_dim=args.encoded_dim,
            gamma=args.gamma,
            bptt_truncate_steps=args.bptt_truncate_steps,
            pel_source=args.pel_source,
        )
    )
    records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    traces: dict[
        tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for trajectory in test_trajectories:
        profile = trajectory.profile
        oracle_temperature = np.asarray(
            oracle_evaluate(
                jnp.asarray(trajectory.building_index, dtype=jnp.int32),
                jnp.asarray(profile.metadata),
                jnp.asarray(profile.inputs),
                jnp.asarray(profile.targets[:, 1]),
                jnp.asarray(profile.targets[:, 2]),
                jnp.asarray(profile.initial_temperature[0, 0]),
            )
        )
        temperature, qroom, pel, _ = combined_evaluate(
            jnp.asarray(profile.metadata),
            jnp.asarray(profile.inputs),
            jnp.asarray(profile.targets[:, 2]),
            jnp.asarray(profile.initial_temperature[0, 0]),
        )
        temperature = np.asarray(temperature)
        qroom = np.asarray(qroom)
        pel = np.asarray(pel)
        common = {
            "profile_id": profile.profile_id,
            "policy": trajectory.policy,
            "evaluation_split": "test",
        }
        records.append(
            {
                "variant": ORACLE_VARIANT,
                **common,
                **thermal_training,
                **_metrics(
                    oracle_temperature, profile.targets[:, 0], profile.datetime
                ),
                "qroom_rmse_w_m2": 0.0,
                "qroom_mae_w_m2": 0.0,
                "qroom_bias_w_m2": 0.0,
                "qroom_energy_ratio": 1.0,
                "pel_rmse_w_m2": 0.0,
                "pel_mae_w_m2": 0.0,
                "pel_bias_w_m2": 0.0,
                "pel_energy_ratio": 1.0,
            }
        )
        pipeline_row = {
            "variant": pipeline_variant,
            **common,
            **hydronic_training,
            **_metrics(temperature, profile.targets[:, 0], profile.datetime),
            **_qroom_metrics(qroom, profile.targets[:, 1]),
            **_pel_metrics(pel, profile.targets[:, 2]),
        }
        if isinstance(hydronic_model, MetadataCapacityHydronic):
            normalized_metadata = (
                np.asarray(profile.metadata) - hydronic_scales.shared.metadata_mean
            ) / hydronic_scales.shared.metadata_std
            pipeline_row["energy_capacity_hours"] = float(
                hydronic_model.capacity_hours(
                    jnp.asarray(normalized_metadata)
                )
            )
        if trajectory.policy == "force_off":
            pipeline_row.update(
                _force_off_regime_metrics(
                    trajectory,
                    temperature,
                    qroom,
                    pel,
                    dt_hours=args.dt_hours,
                    recovery_hours=args.event_context_hours,
                )
            )
            event_records.extend(
                _event_aligned_records(
                    trajectory,
                    temperature,
                    qroom,
                    pel,
                    dt_hours=args.dt_hours,
                    context_hours=args.event_context_hours,
                )
            )
        records.append(pipeline_row)
        traces[(profile.profile_id, trajectory.policy)] = (
            temperature,
            qroom,
            pel,
        )
        print(
            f"evaluated profile={profile.profile_id} policy={trajectory.policy} "
            f"tin_rmse_c={pipeline_row['annual_temperature_rmse_c']:.4f} "
            f"qroom_rmse_w_m2={pipeline_row['qroom_rmse_w_m2']:.4f} "
            f"pel_rmse_w_m2={pipeline_row['pel_rmse_w_m2']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(records)
    metrics_path = args.output_dir / "legacy_hydronic_metadata_thermal_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    dashboard_path = args.output_dir / "legacy_hydronic_metadata_thermal_dashboard.html"
    _write_dashboard(frame, dashboard_path, pipeline_variant)
    pipeline = frame[frame.variant.eq(pipeline_variant)]
    worst_ids = (
        pipeline.groupby("profile_id").annual_temperature_rmse_c.mean()
        .nlargest(args.num_focus_profiles)
        .index.astype(int)
        .tolist()
    )
    traces_path = args.output_dir / "legacy_hydronic_metadata_thermal_traces.html"
    _write_focus_traces(test_trajectories, traces, worst_ids, traces_path)
    event_frame = pd.DataFrame(event_records)
    event_metrics_path = args.output_dir / "force_off_event_metrics.csv"
    event_summary_path = args.output_dir / "force_off_event_summary.csv"
    event_dashboard_path = args.output_dir / "force_off_event_dashboard.html"
    event_frame.to_csv(event_metrics_path, index=False)
    if not event_frame.empty:
        _event_summary(event_frame).to_csv(event_summary_path, index=False)
        _write_event_dashboard(event_frame, event_dashboard_path)

    with (args.output_dir / "metadata_fixed_thermal_params.pkl").open("wb") as stream:
        pickle.dump(jax.device_get(thermal_params), stream)
    model_path = args.output_dir / f"{pipeline_variant}.eqx"
    eqx.tree_serialise_leaves(model_path, hydronic_model)
    manifest = {
        "purpose": (
            "evaluate_legacy_hp_hydronic_with_frozen_metadata_fixed_thermal"
        ),
        "pipeline_variant": pipeline_variant,
        "selected_profile_ids": list(selected_ids),
        "train_profile_ids": list(train_ids),
        "test_profile_ids": list(test_ids),
        "thermal_training": thermal_training,
        "hydronic_training": hydronic_training,
        "worst_profile_ids": worst_ids,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "legacy_hydronic_metadata_thermal_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"saved_metrics={metrics_path}", flush=True)
    print(f"saved_dashboard={dashboard_path}", flush=True)
    print(f"saved_traces={traces_path}", flush=True)
    print(f"saved_event_metrics={event_metrics_path}", flush=True)
    if event_summary_path.exists():
        print(f"saved_event_summary={event_summary_path}", flush=True)
    if event_dashboard_path.exists():
        print(f"saved_event_dashboard={event_dashboard_path}", flush=True)
    return dashboard_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "conditional_state_space_metadata_holdout_sh_hp_100"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "legacy_hydronic_metadata_thermal_holdout_sh_hp_100"
        ),
    )
    parser.add_argument("--thermal-params", type=Path, default=None)
    parser.add_argument("--hydronic-model", type=Path, default=None)
    parser.add_argument(
        "--pel-source",
        choices=("observed", "predicted"),
        default="observed",
        help="Drive the buffer with EnergyPlus Pel or the legacy HP prediction.",
    )
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--encoded-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hydronic-hidden-dim", type=int, default=64)
    parser.add_argument("--hydronic-depth", type=int, default=3)
    parser.add_argument("--input-encoder-dim", type=int, default=8)
    parser.add_argument("--input-encoder-hidden-dim", type=int, default=64)
    parser.add_argument("--input-encoder-depth", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--stride", type=int, default=2024)
    parser.add_argument("--bptt-truncate-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--state-bound", type=float, default=5.0)
    parser.add_argument("--temperature-scale", type=float, default=4.0)
    parser.add_argument("--temperature-delta-max-c", type=float, default=2.0)
    parser.add_argument("--temperature-alpha-cap", type=float, default=0.25)
    parser.add_argument(
        "--temperature-feedback-gradient-scale", type=float, default=0.05
    )
    parser.add_argument("--additive-feedback-gain-bound", type=float, default=1.0)
    parser.add_argument("--hp-energy-cap-hours", type=float, default=24.0)
    parser.add_argument(
        "--metadata-conditioned-energy-capacity",
        action="store_true",
        help="Learn a bounded effective buffer capacity from static metadata.",
    )
    parser.add_argument("--metadata-capacity-min-hours", type=float, default=2.0)
    parser.add_argument("--metadata-capacity-max-hours", type=float, default=16.0)
    parser.add_argument("--metadata-capacity-initial-hours", type=float, default=8.0)
    parser.add_argument("--metadata-capacity-hidden-dim", type=int, default=16)
    parser.add_argument("--metadata-capacity-depth", type=int, default=2)
    parser.add_argument("--temperature-loss-weight", type=float, default=1.0)
    parser.add_argument("--qroom-loss-weight", type=float, default=1.0)
    parser.add_argument("--pel-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--q-to-t-time-constants-hours",
        type=float,
        nargs="+",
        default=(0.25, 1.0, 4.0, 16.0, 24.0, 48.0),
    )
    parser.add_argument("--q-to-t-gain-min", type=float, default=0.01)
    parser.add_argument("--q-to-t-gain-max", type=float, default=2.0)
    parser.add_argument("--pel-cop-slope-per-c", type=float, default=0.05)
    parser.add_argument("--pel-cop-reference-outdoor-c", type=float, default=7.0)
    parser.add_argument("--pel-cop-floor", type=float, default=1.0)
    parser.add_argument("--pel-cop-cap", type=float, default=8.0)
    parser.add_argument("--early-stopping-patience", type=int, default=35)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-focus-profiles", type=int, default=3)
    parser.add_argument("--event-context-hours", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.thermal_params is not None and not args.thermal_params.is_file():
        parser.error(f"Thermal parameter file does not exist: {args.thermal_params}")
    if args.hydronic_model is not None and not args.hydronic_model.is_file():
        parser.error(f"Hydronic model file does not exist: {args.hydronic_model}")
    if args.hp_energy_cap_hours <= 0.0:
        parser.error("--hp-energy-cap-hours must be positive")
    if args.metadata_capacity_min_hours <= 0.0:
        parser.error("--metadata-capacity-min-hours must be positive")
    if args.metadata_capacity_max_hours <= args.metadata_capacity_min_hours:
        parser.error(
            "--metadata-capacity-max-hours must exceed the minimum"
        )
    if not (
        args.metadata_capacity_min_hours
        < args.metadata_capacity_initial_hours
        < args.metadata_capacity_max_hours
    ):
        parser.error(
            "--metadata-capacity-initial-hours must lie strictly inside bounds"
        )
    if args.metadata_capacity_hidden_dim <= 0 or args.metadata_capacity_depth <= 0:
        parser.error("metadata-capacity network dimensions must be positive")
    if args.event_context_hours <= 0.0:
        parser.error("--event-context-hours must be positive")
    if (
        args.temperature_loss_weight < 0.0
        or args.qroom_loss_weight <= 0.0
        or args.pel_loss_weight < 0.0
    ):
        parser.error(
            "loss weights must satisfy temperature >= 0, Qroom > 0, Pel >= 0"
        )
    if args.pel_source == "predicted" and args.pel_loss_weight <= 0.0:
        parser.error("--pel-source predicted requires --pel-loss-weight > 0")
    return args


if __name__ == "__main__":
    run(parse_args())
