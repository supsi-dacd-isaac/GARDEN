"""Diagnose learned thermostat feedback in probabilistic closed-loop HP models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import PROFILE_ID_COLUMN, closed_loop_required_columns
from .data import ClosedLoopProfile, read_dataset_frame, to_closed_loop_profiles
from .model_io import SavedModelArtifact, load_training_artifact
from .models import ProbabilisticContractingClosedLoopHPEmulator
from .scaling import transform_closed_loop_profiles


@dataclass(frozen=True)
class NamedArtifact:
    label: str
    artifact: SavedModelArtifact


def _parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifacts must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("artifacts must be LABEL=PATH")
    return label.strip(), Path(raw_path).expanduser()


def _read_profile(dataset_path: Path, profile_id: int) -> ClosedLoopProfile:
    frame = read_dataset_frame(
        dataset_path,
        columns=closed_loop_required_columns(),
        profile_ids=[profile_id],
    )
    frame = frame[frame[PROFILE_ID_COLUMN] == int(profile_id)].copy()
    profiles = to_closed_loop_profiles(frame, include_space_heating_availability=True)
    if len(profiles) != 1:
        raise ValueError(f"Expected one profile for {profile_id}, found {len(profiles)}")
    return profiles[0]


def _validate_artifact(named: NamedArtifact, profile_id: int) -> ProbabilisticContractingClosedLoopHPEmulator:
    model = named.artifact.model
    if not isinstance(model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError(
            f"{named.label}: expected ProbabilisticContractingClosedLoopHPEmulator, "
            f"found {type(model).__name__}"
        )
    input_columns = tuple(named.artifact.metadata.get("input_columns", ()))
    if len(input_columns) != model.input_dim:
        raise ValueError(f"{named.label}: artifact input columns do not match model input_dim")
    test_ids = {int(value) for value in named.artifact.metadata.get("test_ids", ())}
    if test_ids and profile_id not in test_ids:
        print(f"warning={named.label}:profile_{profile_id}_is_not_in_saved_test_split")
    return model


def _scaled_profile(profile: ClosedLoopProfile, artifact: SavedModelArtifact) -> ClosedLoopProfile:
    return transform_closed_loop_profiles([profile], artifact.scalers)[0]


def _physical_temperature_to_scaled(value_c: float, artifact: SavedModelArtifact) -> float:
    mean = float(np.asarray(artifact.scalers.target.mean).reshape(-1)[0])
    scale = float(np.asarray(artifact.scalers.target.scale).reshape(-1)[0])
    return (float(value_c) - mean) / scale


def _setpoint_to_scaled(value_c: float, artifact: SavedModelArtifact) -> float:
    mean = float(np.asarray(artifact.scalers.inputs.mean).reshape(-1)[0])
    scale = float(np.asarray(artifact.scalers.inputs.scale).reshape(-1)[0])
    return (float(value_c) - mean) / scale


def controller_gap_response(
    named: NamedArtifact,
    raw_profile: ClosedLoopProfile,
    *,
    anchor_index: int,
    gaps_c: np.ndarray,
    num_particles: int,
    seed: int,
) -> dict[str, np.ndarray]:
    model = _validate_artifact(named, raw_profile.profile_id)
    profile = _scaled_profile(raw_profile, named.artifact)
    metadata = jnp.asarray(profile.metadata)
    input_anchor = jnp.asarray(profile.inputs[anchor_index])
    current_temperature_c = float(raw_profile.initial_temperature[anchor_index, 0])
    temperature_scaled = jnp.asarray(
        _physical_temperature_to_scaled(current_temperature_c, named.artifact),
        dtype=input_anchor.dtype,
    )

    key = jax.random.PRNGKey(seed)
    xi = jax.random.normal(key, (num_particles, model.latent_dim))
    initial_temperature = temperature_scaled[jnp.newaxis]
    energies = jax.vmap(lambda particle_xi: model.initial_energy(metadata, initial_temperature, particle_xi))(xi)
    if model.hp_activation_model == "persistent_markov":
        hp_key = jax.random.fold_in(key, 17)
        hp_latent = jax.random.normal(
            hp_key,
            (num_particles, model.hp_persistent_latent_dim),
        )
    else:
        hp_latent = jnp.zeros((num_particles, 0), dtype=input_anchor.dtype)
    controller_state, previous_mode = jax.vmap(
        lambda particle_hp_latent: model.initial_controller_state(
            metadata,
            initial_temperature,
            particle_hp_latent,
        )
    )(hp_latent)
    hp_power_history, hp_mode_history = model.initial_hp_history(input_anchor.dtype)
    hp_power_history = jnp.broadcast_to(
        hp_power_history,
        (num_particles,) + hp_power_history.shape,
    )
    hp_mode_history = jnp.broadcast_to(
        hp_mode_history,
        (num_particles,) + hp_mode_history.shape,
    )

    def one_gap(gap_c: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        setpoint_scaled = (
            jnp.asarray(current_temperature_c, dtype=input_anchor.dtype) + gap_c
            - jnp.asarray(model.input_mean[model.setpoint_input_index], dtype=input_anchor.dtype)
        ) / jnp.asarray(model.input_scale[model.setpoint_input_index], dtype=input_anchor.dtype)
        modified_input = input_anchor.at[model.setpoint_input_index].set(setpoint_scaled)
        encoded_input = model.encode_input(modified_input)

        def one_particle(
            particle_xi: jnp.ndarray,
            energy: jnp.ndarray,
            particle_controller_state: jnp.ndarray,
            particle_previous_mode: jnp.ndarray,
            particle_hp_latent: jnp.ndarray,
            particle_power_history: jnp.ndarray,
            particle_mode_history: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            decoded = model.decode_hp(
                metadata,
                particle_xi,
                encoded_input,
                modified_input,
                energy,
                temperature_scaled,
                jnp.asarray(0.5, dtype=modified_input.dtype),
                jnp.asarray(0.0, dtype=modified_input.dtype),
                particle_controller_state,
                particle_previous_mode,
                particle_power_history,
                particle_mode_history,
                particle_hp_latent,
                jnp.zeros((model.controller_state_dim,), dtype=modified_input.dtype),
                hp_scenario_mode="expected",
                hp_concrete_temperature=jnp.asarray(0.5, dtype=modified_input.dtype),
            )
            pi_t, _, qroom_t, _, _, requested_qroom_t, _, _, pel_t = decoded[:9]
            return pi_t, pel_t, qroom_t, requested_qroom_t

        return jax.vmap(one_particle)(
            xi,
            energies,
            controller_state,
            previous_mode,
            hp_latent,
            hp_power_history,
            hp_mode_history,
        )

    pi, pel, qroom, qroom_requested = jax.vmap(one_gap)(jnp.asarray(gaps_c, dtype=input_anchor.dtype))
    result: dict[str, np.ndarray] = {"gap_c": np.asarray(gaps_c, dtype=np.float32)}
    for name, values in (
        ("pi", pi),
        ("pel_w_m2", pel),
        ("qroom_w_m2", qroom),
        ("qroom_requested_w_m2", qroom_requested),
    ):
        array = np.asarray(values)
        result[f"{name}_mean"] = np.mean(array, axis=1)
        result[f"{name}_p025"] = np.quantile(array, 0.025, axis=1)
        result[f"{name}_p975"] = np.quantile(array, 0.975, axis=1)
    return result


def _first_sustained_recovery_hours(
    gaps: np.ndarray,
    *,
    threshold_c: float,
    sustained_steps: int,
    dt_hours: float,
) -> np.ndarray:
    recovered = np.abs(gaps) <= threshold_c
    result = np.full((gaps.shape[0],), np.nan, dtype=np.float32)
    for particle in range(gaps.shape[0]):
        for start in range(0, gaps.shape[1] - sustained_steps + 1):
            if bool(np.all(recovered[particle, start : start + sustained_steps])):
                result[particle] = np.float32((start + 1) * dt_hours)
                break
    return result


def _energy_balance_metrics(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    aux: tuple[jax.Array, ...],
    qroom_w_m2: np.ndarray,
    pel_w_m2: np.ndarray,
) -> dict[str, float]:
    """Audit the exact charge-leak-draw buffer balance in physical units."""
    dt_h = float(model.hp_dt_hours)
    cop = np.asarray(aux[1], dtype=np.float64)
    energy_t = np.asarray(aux[2], dtype=np.float64)
    available_energy = dt_h * np.asarray(aux[3], dtype=np.float64)
    qroom_energy = dt_h * np.asarray(qroom_w_m2, dtype=np.float64)
    generated_energy = dt_h * cop * np.asarray(pel_w_m2, dtype=np.float64)

    charged_uncapped = energy_t + generated_energy
    charged_energy = np.maximum(charged_uncapped, 0.0)
    if model.hp_energy_cap_wh_m2 > 0.0:
        charged_energy = np.minimum(charged_energy, float(model.hp_energy_cap_wh_m2))
    cap_spill = charged_uncapped - charged_energy
    buffer_loss = charged_energy - available_energy

    energy_next = np.maximum(available_energy - qroom_energy, 0.0)
    if model.hp_energy_cap_wh_m2 > 0.0:
        energy_next = np.minimum(energy_next, float(model.hp_energy_cap_wh_m2))

    # The auxiliary trace contains the pre-step energy. Its next sample is the
    # post-step energy for every step except the final one.
    transition_residual = energy_t[:, 1:] - energy_next[:, :-1]
    final_energy = energy_next[:, -1]
    horizon_residual = (
        energy_t[:, 0]
        + np.sum(generated_energy, axis=1)
        - np.sum(cap_spill, axis=1)
        - np.sum(buffer_loss, axis=1)
        - np.sum(qroom_energy, axis=1)
        - final_energy
    )
    qroom_constraint_violation = np.maximum(qroom_energy - available_energy, 0.0)
    scale = np.maximum(
        energy_t[:, 0] + np.sum(generated_energy, axis=1),
        1.0,
    )
    return {
        "mean_initial_buffer_energy_wh_m2": float(np.mean(energy_t[:, 0])),
        "mean_final_buffer_energy_after_last_step_wh_m2": float(np.mean(final_energy)),
        "mean_generated_heat_energy_wh_m2": float(np.mean(np.sum(generated_energy, axis=1))),
        "mean_buffer_loss_energy_wh_m2": float(np.mean(np.sum(buffer_loss, axis=1))),
        "mean_buffer_cap_spill_energy_wh_m2": float(np.mean(np.sum(cap_spill, axis=1))),
        "mean_energy_balance_residual_wh_m2": float(np.mean(horizon_residual)),
        "max_abs_energy_balance_residual_wh_m2": float(np.max(np.abs(horizon_residual))),
        "max_relative_energy_balance_residual": float(np.max(np.abs(horizon_residual) / scale)),
        "max_abs_step_balance_residual_wh_m2": float(np.max(np.abs(transition_residual))),
        "max_qroom_energy_constraint_violation_wh_m2": float(
            np.max(qroom_constraint_violation)
        ),
        "buffer_cap_active_fraction": float(np.mean(cap_spill > 1e-6)),
    }


def recovery_rollouts(
    named: NamedArtifact,
    raw_profile: ClosedLoopProfile,
    *,
    anchor_index: int,
    deficits_c: Iterable[float],
    horizon_steps: int,
    num_particles: int,
    seed: int,
) -> tuple[dict[tuple[float, str], dict[str, np.ndarray]], list[dict[str, float | str]]]:
    model = _validate_artifact(named, raw_profile.profile_id)
    profile = _scaled_profile(raw_profile, named.artifact)
    end = min(anchor_index + horizon_steps, profile.inputs.shape[0])
    if end - anchor_index < 2:
        raise ValueError("Recovery horizon extends past the end of the profile")
    raw_setpoint_c = float(raw_profile.inputs[anchor_index, model.setpoint_input_index])
    constant_setpoint_scaled = _setpoint_to_scaled(raw_setpoint_c, named.artifact)
    inputs = np.asarray(profile.inputs[anchor_index:end]).copy()
    inputs[:, model.setpoint_input_index] = np.float32(constant_setpoint_scaled)
    metadata = jnp.asarray(profile.metadata)
    inputs_jax = jnp.asarray(inputs)
    target_mean = np.asarray(named.artifact.scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(named.artifact.scalers.target.scale, dtype=np.float32).reshape(-1)
    traces: dict[tuple[float, str], dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | str]] = []

    for deficit_index, deficit_c in enumerate(deficits_c):
        initial_temperature_c = raw_setpoint_c - float(deficit_c)
        initial_temperature = jnp.asarray(
            [_physical_temperature_to_scaled(initial_temperature_c, named.artifact)],
            dtype=inputs_jax.dtype,
        )
        for mode_index, mode in enumerate(("expected", "bernoulli")):
            rollout_key = jax.random.PRNGKey(seed + 1009 * deficit_index + 101 * mode_index)
            predictions, aux = model.sample_with_aux(
                metadata,
                inputs_jax,
                initial_temperature,
                key=rollout_key,
                num_particles=num_particles,
                sample_process_noise=False,
                hp_scenario_mode=mode,
            )
            physical = np.asarray(predictions) * target_scale + target_mean
            temperature = physical[:, :, 0]
            qroom = physical[:, :, 1]
            pel = physical[:, :, 2]
            gaps = raw_setpoint_c - temperature
            pi = np.asarray(aux[0])
            cop = np.asarray(aux[1])
            energy = np.asarray(aux[2])
            requested_qroom = np.asarray(aux[4])
            hp_on = np.asarray(aux[5])
            energy_balance = _energy_balance_metrics(model, aux, qroom, pel)
            recovery_hours = _first_sustained_recovery_hours(
                gaps,
                threshold_c=0.5,
                sustained_steps=max(1, int(round(1.0 / model.hp_dt_hours))),
                dt_hours=model.hp_dt_hours,
            )
            traces[(float(deficit_c), mode)] = {
                "temperature_mean": np.mean(temperature, axis=0),
                "temperature_p025": np.quantile(temperature, 0.025, axis=0),
                "temperature_p975": np.quantile(temperature, 0.975, axis=0),
                "pel_mean": np.mean(pel, axis=0),
                "qroom_mean": np.mean(qroom, axis=0),
                "pi_mean": np.mean(pi, axis=0),
                "hp_on_mean": np.mean(hp_on, axis=0),
            }
            rows.append(
                {
                    "artifact": named.label,
                    "deficit_c": float(deficit_c),
                    "mode": mode,
                    "setpoint_c": raw_setpoint_c,
                    "temperature_rmse_to_setpoint_c": float(np.sqrt(np.mean(gaps**2))),
                    "mean_final_gap_c": float(np.mean(gaps[:, -1])),
                    "p95_abs_final_gap_c": float(np.quantile(np.abs(gaps[:, -1]), 0.95)),
                    "recovered_within_horizon_fraction": float(np.mean(np.isfinite(recovery_hours))),
                    "median_recovery_hours": float(np.nanmedian(recovery_hours))
                    if np.any(np.isfinite(recovery_hours))
                    else float("nan"),
                    "mean_pel_energy_wh_m2": float(model.hp_dt_hours * np.mean(np.sum(pel, axis=1))),
                    "mean_qroom_energy_wh_m2": float(model.hp_dt_hours * np.mean(np.sum(qroom, axis=1))),
                    "mean_requested_qroom_w_m2": float(np.mean(requested_qroom)),
                    "mean_final_buffer_energy_wh_m2": float(np.mean(energy[:, -1])),
                    "mean_cop": float(np.mean(cop)),
                    "mean_activation_probability": float(np.mean(pi)),
                    "mean_realized_duty_cycle": float(np.mean(hp_on)),
                    **energy_balance,
                }
            )
    return traces, rows


def save_gap_plot(
    responses: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    *,
    profile_id: int,
    anchor_time: object,
) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("HP activation probability", "Expected HP electric power", "Delivered room heat"),
        vertical_spacing=0.08,
    )
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    quantities = (("pi", "Probability", 1), ("pel_w_m2", "Pel [W/m2]", 2), ("qroom_w_m2", "Qroom [W/m2]", 3))
    for artifact_index, (label, response) in enumerate(responses.items()):
        color = colors[artifact_index % len(colors)]
        for quantity, y_title, row in quantities:
            x = response["gap_c"]
            lower = response[f"{quantity}_p025"]
            upper = response[f"{quantity}_p975"]
            mean = response[f"{quantity}_mean"]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower,
                    mode="lines",
                    line={"color": color, "width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=upper,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.13),
                    line={"color": color, "width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=row == 1,
                    line={"color": color, "width": 2},
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text=y_title, row=row, col=1)
    fig.add_vline(x=0.0, line_dash="dot", line_color="#666")
    fig.update_xaxes(title_text="Tset - Tin [degC]", row=3, col=1)
    fig.update_layout(
        title=(
            f"Controller gap response, profile {profile_id}, anchor {anchor_time}<br>"
            "<sup>Weather, metadata, current Tin, and per-particle buffer energy held fixed; no process noise</sup>"
        ),
        height=900,
        hovermode="x unified",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")


def save_recovery_plot(
    all_traces: dict[str, dict[tuple[float, str], dict[str, np.ndarray]]],
    output_path: Path,
    *,
    profile_id: int,
    anchor_time: object,
    setpoint_c: float,
    deficits_c: tuple[float, ...],
    dt_hours: float,
) -> None:
    fig = make_subplots(
        rows=len(deficits_c),
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(f"Initial deficit {deficit:g} K" for deficit in deficits_c),
        vertical_spacing=0.05,
    )
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for row, deficit_c in enumerate(deficits_c, 1):
        for artifact_index, (label, traces) in enumerate(all_traces.items()):
            color = colors[artifact_index % len(colors)]
            for mode, dash in (("expected", "solid"), ("bernoulli", "dash")):
                trace = traces[(float(deficit_c), mode)]
                hours = dt_hours * (np.arange(trace["temperature_mean"].shape[0]) + 1)
                if mode == "bernoulli":
                    fig.add_trace(
                        go.Scatter(
                            x=hours,
                            y=trace["temperature_p025"],
                            mode="lines",
                            line={"color": color, "width": 0},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=hours,
                            y=trace["temperature_p975"],
                            mode="lines",
                            fill="tonexty",
                            fillcolor=_rgba(color, 0.09),
                            line={"color": color, "width": 0},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=1,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=hours,
                        y=trace["temperature_mean"],
                        mode="lines",
                        name=f"{label}: {mode}",
                        legendgroup=f"{label}:{mode}",
                        showlegend=row == 1,
                        line={"color": color, "dash": dash, "width": 2},
                    ),
                    row=row,
                    col=1,
                )
        fig.add_hline(y=setpoint_c, line_color="#222", line_dash="dot", row=row, col=1)
        fig.update_yaxes(title_text="Tin [degC]", row=row, col=1)
    fig.update_xaxes(title_text="Hours after imposed deficit", row=len(deficits_c), col=1)
    fig.update_layout(
        title=(
            f"Closed-loop setpoint recovery, profile {profile_id}, anchor {anchor_time}<br>"
            "<sup>Constant setpoint and real future weather; parameter uncertainty retained; process noise disabled</sup>"
        ),
        height=max(850, 270 * len(deficits_c)),
        hovermode="x unified",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def run(args: argparse.Namespace) -> None:
    named_artifacts = [
        NamedArtifact(label=label, artifact=load_training_artifact(path))
        for label, path in args.artifact
    ]
    if len({item.label for item in named_artifacts}) != len(named_artifacts):
        raise ValueError("Artifact labels must be unique")
    raw_profile = _read_profile(args.dataset, args.profile_id)
    anchor_index = int(round(args.anchor_day * 24.0 / args.dt_hours))
    if not 0 <= anchor_index < raw_profile.inputs.shape[0]:
        raise ValueError("anchor-day lies outside the selected profile")
    availability_index = 4
    if raw_profile.inputs.shape[1] > availability_index and raw_profile.inputs[anchor_index, availability_index] < 0.5:
        raise ValueError("Selected anchor lies outside the space-heating availability period")
    horizon_steps = int(round(args.recovery_hours / args.dt_hours))
    gaps_c = np.linspace(args.gap_min_c, args.gap_max_c, args.gap_points, dtype=np.float32)
    deficits_c = tuple(float(value) for value in args.deficits_c)

    responses: dict[str, dict[str, np.ndarray]] = {}
    all_recovery: dict[str, dict[tuple[float, str], dict[str, np.ndarray]]] = {}
    recovery_rows: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    for artifact_index, named in enumerate(named_artifacts):
        response = controller_gap_response(
            named,
            raw_profile,
            anchor_index=anchor_index,
            gaps_c=gaps_c,
            num_particles=args.num_particles,
            seed=args.seed + 100_003 * artifact_index,
        )
        responses[named.label] = response
        for index, gap_c in enumerate(gaps_c):
            response_rows.append(
                {
                    "artifact": named.label,
                    "gap_c": float(gap_c),
                    "pi_mean": float(response["pi_mean"][index]),
                    "pi_p025": float(response["pi_p025"][index]),
                    "pi_p975": float(response["pi_p975"][index]),
                    "pel_mean_w_m2": float(response["pel_w_m2_mean"][index]),
                    "qroom_mean_w_m2": float(response["qroom_w_m2_mean"][index]),
                    "qroom_requested_mean_w_m2": float(
                        response["qroom_requested_w_m2_mean"][index]
                    ),
                }
            )
        traces, rows = recovery_rollouts(
            named,
            raw_profile,
            anchor_index=anchor_index,
            deficits_c=deficits_c,
            horizon_steps=horizon_steps,
            num_particles=args.num_particles,
            seed=args.seed + 200_003 * artifact_index,
        )
        all_recovery[named.label] = traces
        recovery_rows.extend(rows)

    output_dir = args.output_dir
    anchor_time = raw_profile.datetime[anchor_index]
    gap_plot = output_dir / f"profile_{args.profile_id}_controller_gap_response.html"
    recovery_plot = output_dir / f"profile_{args.profile_id}_setpoint_recovery.html"
    save_gap_plot(
        responses,
        gap_plot,
        profile_id=args.profile_id,
        anchor_time=anchor_time,
    )
    save_recovery_plot(
        all_recovery,
        recovery_plot,
        profile_id=args.profile_id,
        anchor_time=anchor_time,
        setpoint_c=float(raw_profile.inputs[anchor_index, 0]),
        deficits_c=deficits_c,
        dt_hours=args.dt_hours,
    )
    response_csv = output_dir / f"profile_{args.profile_id}_controller_gap_response.csv"
    recovery_csv = output_dir / f"profile_{args.profile_id}_setpoint_recovery.csv"
    _write_rows(response_csv, response_rows)
    _write_rows(recovery_csv, recovery_rows)
    print(f"profile={args.profile_id} anchor_index={anchor_index} anchor_time={anchor_time}")
    print(f"saved_gap_plot={gap_plot}")
    print(f"saved_recovery_plot={recovery_plot}")
    print(f"saved_gap_csv={response_csv}")
    print(f"saved_recovery_csv={recovery_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=_parse_artifact,
        action="append",
        required=True,
        help="Model artifact as LABEL=PATH; repeat to compare checkpoints.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile-id", type=int, default=701524)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-day", type=float, default=14.0)
    parser.add_argument("--recovery-hours", type=float, default=48.0)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--deficits-c", type=float, nargs="+", default=(0.0, 1.0, 2.0, 4.0))
    parser.add_argument("--gap-min-c", type=float, default=-2.0)
    parser.add_argument("--gap-max-c", type=float, default=6.0)
    parser.add_argument("--gap-points", type=int, default=33)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.recovery_hours <= 0.0 or args.dt_hours <= 0.0:
        parser.error("recovery-hours and dt-hours must be positive")
    if args.gap_points < 2 or args.num_particles < 1:
        parser.error("gap-points must be at least 2 and num-particles must be positive")
    if any(value < 0.0 for value in args.deficits_c):
        parser.error("deficits-c must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
