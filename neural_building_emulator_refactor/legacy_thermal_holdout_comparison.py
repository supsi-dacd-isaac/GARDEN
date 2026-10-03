"""Compare the legacy thermal block with a metadata-conditioned state space.

The cohort and split are loaded from a completed
``conditional_state_space_sufficiency`` run. Both models receive observed
Qroom, so this diagnostic compares only autonomous Qroom-to-Tin dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    HP_FORCE_OFF_REQUESTED_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from neural_building_emulator.models.contracting_closed_loop_hp import (
    ContractingClosedLoopHPEmulator,
)

from .conditional_state_space_sufficiency import (
    INPUT_NAMES,
    Scales,
    Trajectory,
    _fit_scales,
    _load_profiles,
    _make_trajectories,
    _make_windows,
    _metrics,
)


LEGACY_VARIANT = "legacy_dynamic_positive_leaky"
STATE_SPACE_VARIANT = "metadata_fixed"
HYBRID_VARIANT = "metadata_fixed_positive_leaky"
PEL_VARIANT = "metadata_fixed_pel"
PEL_COP_POSITIVE_VARIANT = "metadata_fixed_pel_cop_positive_leaky"


@dataclass(frozen=True)
class LegacyScales:
    shared: Scales
    input_mean: np.ndarray
    input_std: np.ndarray
    qroom_mean: float
    qroom_std: float
    pel_mean: float
    pel_std: float


def _online_mean_std(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total: np.ndarray | None = None
    total_square: np.ndarray | None = None
    for values in arrays:
        values64 = np.asarray(values, dtype=np.float64)
        if values64.ndim == 1:
            values64 = values64[:, None]
        count += values64.shape[0]
        current_sum = np.sum(values64, axis=0)
        current_square = np.sum(np.square(values64), axis=0)
        total = current_sum if total is None else total + current_sum
        total_square = (
            current_square
            if total_square is None
            else total_square + current_square
        )
    if count == 0 or total is None or total_square is None:
        raise ValueError("Cannot fit scales from an empty trajectory list")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _legacy_scales(trajectories: Sequence[Trajectory]) -> LegacyScales:
    shared = _fit_scales(trajectories)
    input_mean, input_std = _online_mean_std(
        [trajectory.profile.inputs for trajectory in trajectories]
    )
    qroom_mean, qroom_std = _online_mean_std(
        [trajectory.profile.targets[:, 1] for trajectory in trajectories]
    )
    pel_mean, pel_std = _online_mean_std(
        [trajectory.profile.targets[:, 2] for trajectory in trajectories]
    )
    return LegacyScales(
        shared=shared,
        input_mean=input_mean,
        input_std=input_std,
        qroom_mean=float(qroom_mean[0]),
        qroom_std=float(qroom_std[0]),
        pel_mean=float(pel_mean[0]),
        pel_std=float(pel_std[0]),
    )


def _normalized_windows(
    windows: dict[str, np.ndarray], scales: LegacyScales
) -> dict[str, np.ndarray]:
    result = dict(windows)
    result["metadata"] = (
        (windows["metadata"] - scales.shared.metadata_mean)
        / scales.shared.metadata_std
    ).astype(np.float32)
    result["inputs"] = (
        (windows["inputs"] - scales.input_mean) / scales.input_std
    ).astype(np.float32)
    return result


def _initial_model(
    scales: LegacyScales,
    *,
    metadata_dim: int,
    args: argparse.Namespace,
) -> ContractingClosedLoopHPEmulator:
    availability_index = INPUT_NAMES.index(SPACE_HEATING_AVAILABILITY_COLUMN)
    force_off_index = INPUT_NAMES.index(HP_FORCE_OFF_REQUESTED_COLUMN)
    thermal_mask = (
        INPUT_NAMES.index(DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN),
        force_off_index,
    )
    qroom_cap = max(1.0, scales.qroom_mean + 8.0 * scales.qroom_std)
    pel_cap = max(1.0, scales.pel_mean + 8.0 * scales.pel_std)
    energy_cap_hours = getattr(args, "hp_energy_cap_hours", None)
    energy_cap = (
        1.0
        if energy_cap_hours is None
        else float(energy_cap_hours) * qroom_cap
    )
    return ContractingClosedLoopHPEmulator(
        metadata_dim=metadata_dim,
        input_dim=len(INPUT_NAMES),
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        input_encoder_dim=args.input_encoder_dim,
        input_encoder_hidden_dim=args.input_encoder_hidden_dim,
        input_encoder_depth=args.input_encoder_depth,
        contraction_gamma=args.gamma,
        state_bound=args.state_bound,
        temperature_output_scale=args.temperature_scale,
        temperature_delta_max_c=args.temperature_delta_max_c,
        temperature_alpha_cap=args.temperature_alpha_cap,
        temperature_fixed_alpha=0.0,
        temperature_update_mode="alpha_bounded_equilibrium",
        temperature_feedback_gradient_scale=args.temperature_feedback_gradient_scale,
        transition_conditioning="exogenous_additive_feedback",
        additive_feedback_gain_bound=args.additive_feedback_gain_bound,
        hp_dt_hours=args.dt_hours,
        hp_cop_floor=1.0,
        hp_cop_cap=8.0,
        hp_pel_cap_w_m2=pel_cap,
        hp_qroom_cap_w_m2=qroom_cap,
        hp_energy_cap_wh_m2=energy_cap,
        input_mean=tuple(float(value) for value in scales.input_mean),
        input_scale=tuple(float(value) for value in scales.input_std),
        target_mean=(
            scales.shared.temperature_mean,
            scales.qroom_mean,
            scales.pel_mean,
        ),
        target_scale=(
            scales.shared.temperature_std,
            scales.qroom_std,
            scales.pel_std,
        ),
        availability_input_index=availability_index,
        hp_force_off_input_index=force_off_index,
        bptt_truncate_steps=args.bptt_truncate_steps,
        thermal_dynamics_masked_input_indices=thermal_mask,
        thermostat_demand_mode="monotone",
        q_to_t_mode="positive_leaky",
        q_to_t_time_constants_hours=tuple(args.q_to_t_time_constants_hours),
        q_to_t_gain_min_c_per_w_m2=args.q_to_t_gain_min,
        q_to_t_gain_max_c_per_w_m2=args.q_to_t_gain_max,
        key=jax.random.PRNGKey(args.seed),
    )


def _detach(value: jax.Array, step: jax.Array, interval: int) -> jax.Array:
    should_detach = (step > 0) & (jnp.mod(step, interval) == 0)
    return jax.lax.cond(
        should_detach, jax.lax.stop_gradient, lambda current: current, value
    )


def _legacy_rollout(
    model: ContractingClosedLoopHPEmulator,
    metadata: jax.Array,
    inputs: jax.Array,
    qroom: jax.Array,
    initial_temperature_c: jax.Array,
) -> jax.Array:
    temperature_mean = jnp.asarray(model.target_mean[0], dtype=inputs.dtype)
    temperature_scale = jnp.asarray(model.target_scale[0], dtype=inputs.dtype)
    qroom_mean = jnp.asarray(model.target_mean[1], dtype=inputs.dtype)
    qroom_scale = jnp.asarray(model.target_scale[1], dtype=inputs.dtype)
    initial_temperature = (
        initial_temperature_c - temperature_mean
    ) / temperature_scale
    initial_vector = jnp.ravel(initial_temperature)[:1]
    state = model.initial_state(metadata, initial_vector)
    q_response = model.initial_q_response_state(metadata, initial_vector)
    alpha = model.temperature_alpha(metadata)

    def step(carry, values):
        state_t, temperature_t, q_response_t = carry
        step_index, input_t, qroom_t = values
        state_t = _detach(state_t, step_index, model.bptt_truncate_steps)
        temperature_t = _detach(
            temperature_t, step_index, model.bptt_truncate_steps
        )
        q_response_t = _detach(
            q_response_t, step_index, model.bptt_truncate_steps
        )
        feedback_temperature = (
            model.temperature_feedback_gradient_scale * temperature_t
            + (1.0 - model.temperature_feedback_gradient_scale)
            * jax.lax.stop_gradient(temperature_t)
        )
        qroom_scaled = (qroom_t - qroom_mean) / qroom_scale
        forcing = model.thermal_forcing(
            input_t, qroom_scaled, feedback_temperature
        )
        state_next = model.one_step_state(metadata, forcing, state_t)
        q_response_next = model.next_q_response_state(
            metadata, q_response_t, qroom_t
        )
        temperature_next = model.decode_temperature(
            state_next,
            temperature_t,
            alpha,
            q_response_next,
        )
        temperature_c = temperature_mean + temperature_scale * temperature_next
        return (
            state_next,
            temperature_next,
            q_response_next,
        ), temperature_c

    steps = jnp.arange(inputs.shape[0], dtype=jnp.int32)
    _, prediction = jax.lax.scan(
        step,
        (state, initial_temperature, q_response),
        (steps, inputs, qroom),
    )
    return prediction


def _array_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(value)) for value in leaves) + 1e-12
    )


def _fit(
    model: ContractingClosedLoopHPEmulator,
    windows: dict[str, np.ndarray],
    scales: LegacyScales,
    args: argparse.Namespace,
) -> tuple[ContractingClosedLoopHPEmulator, dict[str, float]]:
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adam(args.learning_rate),
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(candidate, metadata, inputs, qroom, initial, target):
        prediction = jax.vmap(_legacy_rollout, in_axes=(None, 0, 0, 0, 0))(
            candidate, metadata, inputs, qroom, initial
        )
        residual = (prediction - target) / scales.shared.temperature_std
        return jnp.mean(jnp.square(residual))

    @eqx.filter_jit
    def train_step(candidate, state, batch):
        loss, gradients = loss_fn(
            candidate,
            batch["metadata"],
            batch["inputs"],
            batch["qroom"],
            batch["initial"],
            batch["target"],
        )
        grad_norm = _array_norm(gradients)
        updates, state = optimizer.update(gradients, state, candidate)
        candidate = eqx.apply_updates(candidate, updates)
        return candidate, state, loss, grad_norm

    rng = np.random.default_rng(args.seed)
    count = len(windows["target"])
    best_loss = math.inf
    best_model = model
    wait = 0
    max_grad = 0.0
    started = time.perf_counter()
    epoch = -1
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        losses: list[float] = []
        for start in range(0, count, args.batch_size):
            indices = order[start : start + args.batch_size]
            if len(indices) < args.batch_size:
                indices = np.concatenate(
                    (indices, order[: args.batch_size - len(indices)])
                )
            batch = {
                key: jnp.asarray(value[indices]) for key, value in windows.items()
            }
            model, optimizer_state, loss, grad_norm = train_step(
                model, optimizer_state, batch
            )
            loss_value = float(loss)
            grad_value = float(grad_norm)
            if not np.isfinite(loss_value) or not np.isfinite(grad_value):
                raise FloatingPointError(
                    f"Non-finite legacy update at epoch {epoch + 1}"
                )
            losses.append(loss_value)
            max_grad = max(max_grad, grad_value)
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss - args.early_stopping_min_delta:
            best_loss = epoch_loss
            best_model = model
            wait = 0
        else:
            wait += 1
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"legacy epoch={epoch + 1:04d} loss={epoch_loss:.6f} "
                f"best={best_loss:.6f} max_grad={max_grad:.3g}",
                flush=True,
            )
        if wait >= args.early_stopping_patience:
            break
    return best_model, {
        "train_window_normalized_mse": best_loss,
        "epochs_completed": float(epoch + 1),
        "max_grad_norm": max_grad,
        "training_seconds": time.perf_counter() - started,
    }


def _annual_prediction(
    evaluate: Any, trajectory: Trajectory, scales: LegacyScales
) -> np.ndarray:
    profile = trajectory.profile
    metadata = (
        profile.metadata - scales.shared.metadata_mean
    ) / scales.shared.metadata_std
    inputs = (profile.inputs - scales.input_mean) / scales.input_std
    return np.asarray(
        evaluate(
            jnp.asarray(metadata),
            jnp.asarray(inputs),
            jnp.asarray(profile.targets[:, 1]),
            jnp.asarray(profile.initial_temperature[0, 0]),
        )
    )


def _comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in frame.groupby("variant"):
        for policy, selected in group.groupby("policy"):
            rows.append(
                {
                    "variant": variant,
                    "policy": policy,
                    "count": len(selected),
                    "annual_rmse_mean_c": selected.annual_temperature_rmse_c.mean(),
                    "annual_rmse_median_c": selected.annual_temperature_rmse_c.median(),
                    "annual_rmse_p95_c": selected.annual_temperature_rmse_c.quantile(0.95),
                    "winter_rmse_mean_c": selected.winter_temperature_rmse_c.mean(),
                    "absolute_bias_mean_c": selected.annual_temperature_bias_c.abs().mean(),
                }
            )
    return pd.DataFrame(rows)


def _write_comparison_dashboard(frame: pd.DataFrame, output: Path) -> None:
    colors = {
        STATE_SPACE_VARIANT: "#d97706",
        LEGACY_VARIANT: "#355ea8",
        HYBRID_VARIANT: "#16877c",
        PEL_VARIANT: "#b44e64",
        PEL_COP_POSITIVE_VARIANT: "#4d7f3a",
    }
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Annual Tin RMSE", "Winter Tin RMSE"),
    )
    preferred_order = (
        STATE_SPACE_VARIANT,
        PEL_VARIANT,
        PEL_COP_POSITIVE_VARIANT,
        HYBRID_VARIANT,
        LEGACY_VARIANT,
    )
    variants = tuple(
        variant for variant in preferred_order if variant in set(frame.variant)
    )
    variants += tuple(
        variant for variant in frame.variant.unique() if variant not in variants
    )
    for variant in variants:
        selected = frame[frame.variant.eq(variant)].sort_values(
            ["profile_id", "policy"]
        )
        labels = selected.apply(
            lambda row: f"{int(row.profile_id)} / {row.policy}", axis=1
        )
        for col, metric in enumerate(
            ("annual_temperature_rmse_c", "winter_temperature_rmse_c"),
            start=1,
        ):
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=selected[metric],
                    name=variant,
                    legendgroup=variant,
                    showlegend=col == 1,
                    marker_color=colors.get(variant, "#7a7a7a"),
                    hovertemplate=(
                        f"model={variant}<br>profile=%{{x}}<br>"
                        "RMSE=%{y:.4f} C<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    figure.update_layout(
        title="Observed-Q thermal generalization on identical holdout buildings",
        barmode="group",
        height=700,
        template="plotly_white",
    )
    figure.write_html(output, include_plotlyjs="cdn")


def _write_legacy_traces(
    trajectories: Sequence[Trajectory],
    predictions: dict[tuple[int, str], np.ndarray],
    output: Path,
) -> None:
    figure = make_subplots(
        rows=len(trajectories),
        cols=1,
        subplot_titles=tuple(
            f"{trajectory.profile.profile_id} / {trajectory.policy}"
            for trajectory in trajectories
        ),
        vertical_spacing=0.025,
    )
    for row, trajectory in enumerate(trajectories, start=1):
        profile = trajectory.profile
        subset = slice(None, None, 4)
        figure.add_trace(
            go.Scatter(
                x=profile.datetime[subset],
                y=profile.targets[subset, 0],
                name="EnergyPlus",
                legendgroup="EnergyPlus",
                showlegend=row == 1,
                line={"color": "#202124", "width": 1.7},
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=profile.datetime[subset],
                y=predictions[(profile.profile_id, trajectory.policy)][subset],
                name="Legacy dynamic",
                legendgroup="Legacy dynamic",
                showlegend=row == 1,
                line={"color": "#355ea8", "width": 1.0},
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text="Tin [C]", row=row, col=1)
    figure.update_layout(
        title="Legacy thermal block with observed Qroom on unseen buildings",
        height=330 * len(trajectories),
        hovermode="x unified",
        template="plotly_white",
    )
    figure.write_html(output, include_plotlyjs="cdn")


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reference_dir / "state_space_sufficiency_manifest.json"
    metrics_path = args.reference_dir / "state_space_sufficiency_metrics.csv"
    manifest = json.loads(manifest_path.read_text())
    train_ids = tuple(int(value) for value in manifest["train_profile_ids"])
    test_ids = tuple(int(value) for value in manifest["test_profile_ids"])
    selected_ids = tuple(int(value) for value in manifest["selected_profile_ids"])
    if not test_ids:
        raise ValueError("The reference run does not contain a holdout split")

    setpoint_dataset = Path(manifest["arguments"]["setpoint_dataset"])
    force_off_dataset = Path(manifest["arguments"]["force_off_dataset"])
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
    scales = _legacy_scales(train_trajectories)
    raw_windows = _make_windows(
        train_trajectories,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    windows = _normalized_windows(raw_windows, scales)
    model = _initial_model(
        scales,
        metadata_dim=windows["metadata"].shape[-1],
        args=args,
    )
    print(
        f"legacy_observed_q=true train_profiles={len(train_ids)} "
        f"test_profiles={len(test_ids)} windows={len(windows['target'])}",
        flush=True,
    )
    model, training = _fit(model, windows, scales, args)
    model_path = args.output_dir / "legacy_thermal_observed_q.eqx"
    eqx.tree_serialise_leaves(model_path, model)

    evaluate = eqx.filter_jit(
        lambda metadata, inputs, qroom, initial: _legacy_rollout(
            model, metadata, inputs, qroom, initial
        )
    )
    rows = []
    predictions: dict[tuple[int, str], np.ndarray] = {}
    for trajectory in test_trajectories:
        prediction = _annual_prediction(evaluate, trajectory, scales)
        profile = trajectory.profile
        row = {
            "variant": LEGACY_VARIANT,
            "profile_id": profile.profile_id,
            "policy": trajectory.policy,
            "evaluation_split": "test",
            **training,
            **_metrics(prediction, profile.targets[:, 0], profile.datetime),
        }
        rows.append(row)
        predictions[(profile.profile_id, trajectory.policy)] = prediction
        print(
            f"legacy_evaluated policy={trajectory.policy} "
            f"profile={profile.profile_id} "
            f"annual_rmse_c={row['annual_temperature_rmse_c']:.4f}",
            flush=True,
        )

    legacy_frame = pd.DataFrame(rows)
    legacy_frame.to_csv(args.output_dir / "legacy_thermal_metrics.csv", index=False)
    state_frame = pd.read_csv(metrics_path)
    comparison = pd.concat([state_frame, legacy_frame], ignore_index=True)
    comparison.to_csv(args.output_dir / "thermal_model_comparison_metrics.csv", index=False)
    summary = _comparison_summary(comparison)
    summary.to_csv(args.output_dir / "thermal_model_comparison_summary.csv", index=False)
    dashboard = args.output_dir / "thermal_model_comparison.html"
    _write_comparison_dashboard(comparison, dashboard)
    _write_legacy_traces(
        test_trajectories,
        predictions,
        args.output_dir / "legacy_thermal_annual_traces.html",
    )
    result_manifest = {
        "reference_dir": str(args.reference_dir),
        "observed_qroom": True,
        "train_profile_ids": list(train_ids),
        "test_profile_ids": list(test_ids),
        "training": training,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "legacy_thermal_comparison_manifest.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True)
    )
    print(summary.round(5).to_string(index=False), flush=True)
    print(f"saved_comparison={dashboard}", flush=True)
    return dashboard


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
            "legacy_vs_metadata_state_space_holdout_sh_hp_100"
        ),
    )
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--input-encoder-dim", type=int, default=8)
    parser.add_argument("--input-encoder-hidden-dim", type=int, default=64)
    parser.add_argument("--input-encoder-depth", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--state-bound", type=float, default=5.0)
    parser.add_argument("--temperature-scale", type=float, default=4.0)
    parser.add_argument("--temperature-delta-max-c", type=float, default=2.0)
    parser.add_argument("--temperature-alpha-cap", type=float, default=0.25)
    parser.add_argument(
        "--temperature-feedback-gradient-scale", type=float, default=0.05
    )
    parser.add_argument("--additive-feedback-gain-bound", type=float, default=1.0)
    parser.add_argument(
        "--q-to-t-time-constants-hours",
        type=float,
        nargs="+",
        default=(0.25, 1.0, 4.0, 16.0, 24.0, 48.0),
    )
    parser.add_argument("--q-to-t-gain-min", type=float, default=0.01)
    parser.add_argument("--q-to-t-gain-max", type=float, default=2.0)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--stride", type=int, default=2024)
    parser.add_argument("--bptt-truncate-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=35)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.q_to_t_gain_max <= args.q_to_t_gain_min:
        parser.error("Q-to-T gain bounds must satisfy max > min")
    return args


if __name__ == "__main__":
    run(parse_args())
