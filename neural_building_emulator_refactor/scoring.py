"""Common physical and flexibility scoring for saved comparison artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Sequence

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator.flexibility_event_study_kpis import (
    EventStudyConfig,
    FlexibilityKpi,
    estimate_trace_flexibility,
)

from .artifacts import load_artifact
from .prediction import predict_profile
from .profiles import ProfileSource, load_profiles

KpiMode = Literal["mean", "scenario_average"]


def _output_name(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    if "/" in suffix or "\\" in suffix:
        raise ValueError("output_suffix must be a filename suffix, not a path")
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix}"


def _metric_names(channel_count: int) -> list[str]:
    return ["temperature_c"] if channel_count == 1 else ["temperature_c", "qroom_w_m2", "pel_w_m2"]


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    names = _metric_names(error.shape[-1])
    result: dict[str, float] = {}
    for index, name in enumerate(names):
        values = error[..., index]
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(values**2)))
        result[f"{name}_mae"] = float(np.mean(np.abs(values)))
        result[f"{name}_bias"] = float(np.mean(values))
    return result


def _average_kpis(kpis: list[list[FlexibilityKpi]]) -> list[FlexibilityKpi]:
    if not kpis:
        return []
    result: list[FlexibilityKpi] = []
    for horizon_index in range(len(kpis[0])):
        valid = [values[horizon_index] for values in kpis if values[horizon_index].valid]
        reference = kpis[0][horizon_index]
        if not valid:
            result.append(reference)
            continue
        beta_plus = float(np.mean([value.beta_plus_w_m2_k for value in valid]))
        beta_minus = float(np.mean([value.beta_minus_w_m2_k for value in valid]))
        result.append(
            FlexibilityKpi(
                profile_id=reference.profile_id,
                horizon_hours=reference.horizon_hours,
                horizon_steps=reference.horizon_steps,
                event_count=int(np.median([value.event_count for value in valid])),
                valid=True,
                beta_plus_w_m2_k=beta_plus,
                beta_minus_w_m2_k=beta_minus,
                up_flex_wh_m2_k=reference.horizon_hours * beta_plus,
                down_flex_wh_m2_k=reference.horizon_hours * beta_minus,
            )
        )
    return result


def _coverage_rows(
    truth_rows: list[dict],
    scenario_rows: list[dict],
    quantiles: Sequence[float],
) -> list[dict]:
    if not scenario_rows:
        return []
    truth = pd.DataFrame(truth_rows)
    scenarios = pd.DataFrame(scenario_rows)
    rows: list[dict] = []
    for horizon in sorted(truth["horizon_hours"].unique()):
        for direction, metric in (
            ("up", "up_flex_wh_m2_k"),
            ("down", "down_flex_wh_m2_k"),
        ):
            truth_part = truth[(truth["horizon_hours"] == horizon) & truth["valid"]]
            scenario_part = scenarios[
                (scenarios["horizon_hours"] == horizon) & scenarios["valid"]
            ]
            pairs: list[tuple[float, np.ndarray]] = []
            for item in truth_part.itertuples(index=False):
                values = scenario_part.loc[
                    scenario_part["profile_id"] == item.profile_id,
                    metric,
                ].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    pairs.append((float(getattr(item, metric)), values))
            for quantile in quantiles:
                hits = [truth_value <= np.quantile(samples, quantile) for truth_value, samples in pairs]
                if hits:
                    empirical = float(np.mean(hits))
                    rows.append(
                        {
                            "horizon_hours": float(horizon),
                            "direction": direction,
                            "quantile": float(quantile),
                            "empirical_coverage": empirical,
                            "calibration_error": empirical - float(quantile),
                            "profile_count": len(hits),
                        }
                    )
    return rows


def _write_coverage_plot(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Upward KPI", "Downward KPI"))
    colors = ["#1769aa", "#2e7d32", "#ef6c00", "#8e24aa"]
    for column, direction in ((1, "up"), (2, "down")):
        part = frame[frame["direction"] == direction]
        for color_index, horizon in enumerate(sorted(part["horizon_hours"].unique())):
            values = part[part["horizon_hours"] == horizon]
            figure.add_trace(
                go.Scatter(
                    x=values["quantile"],
                    y=values["empirical_coverage"],
                    mode="lines+markers",
                    name=f"{direction} {horizon:g} h",
                    line={"color": colors[color_index % len(colors)]},
                ),
                row=1,
                col=column,
            )
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"color": "black", "dash": "dash"},
                name="ideal",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(title_text="Nominal quantile", range=[0, 1])
    figure.update_yaxes(title_text="Empirical coverage", range=[0, 1])
    figure.update_layout(template="plotly_white", height=620, width=1300, title="Flexibility KPI coverage")
    figure.write_html(path)


def _write_flexibility_comparison_plot(
    path: Path,
    truth_rows: list[dict],
    prediction_rows: list[dict],
) -> None:
    simulated = pd.DataFrame(truth_rows).drop(columns=["signal"], errors="ignore")
    emulated = pd.DataFrame(prediction_rows).drop(columns=["signal"], errors="ignore")
    keys = ["profile_id", "horizon_hours", "horizon_steps"]
    simulated = simulated.rename(
        columns={
            column: f"sim_{column}"
            for column in simulated.columns
            if column not in keys
        }
    )
    emulated = emulated.rename(
        columns={
            column: f"emu_{column}"
            for column in emulated.columns
            if column not in keys
        }
    )
    frame = simulated.merge(emulated, on=keys, how="inner")
    frame["valid"] = frame["sim_valid"] & frame["emu_valid"]
    frame = frame[frame["valid"]].copy()
    if frame.empty:
        return
    for direction in ("up", "down"):
        metric = f"{direction}_flex_wh_m2_k"
        frame[f"diff_{metric}"] = frame[f"emu_{metric}"] - frame[f"sim_{metric}"]

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Upward energy KPI: emulated vs simulated",
            "Downward energy KPI: emulated vs simulated",
            "Profile-level energy KPI error",
            "Mean flexibility-duration curve",
        ),
    )
    colors = ["#1769aa", "#2e7d32", "#ef6c00", "#8e24aa", "#00838f"]
    for color_index, horizon in enumerate(sorted(frame["horizon_hours"].unique())):
        part = frame[frame["horizon_hours"] == horizon]
        color = colors[color_index % len(colors)]
        for direction, column in (("up", 1), ("down", 2)):
            metric = f"{direction}_flex_wh_m2_k"
            figure.add_trace(
                go.Scatter(
                    x=part[f"sim_{metric}"],
                    y=part[f"emu_{metric}"],
                    mode="markers",
                    marker={"size": 7, "opacity": 0.7, "color": color},
                    name=f"H={horizon:g} h",
                    legendgroup=f"horizon_{horizon}",
                    showlegend=direction == "up",
                    text=part["profile_id"].astype(str),
                ),
                row=1,
                col=column,
            )
    for direction, column in (("up", 1), ("down", 2)):
        metric = f"{direction}_flex_wh_m2_k"
        values = np.concatenate(
            [
                frame[f"sim_{metric}"].to_numpy(dtype=float),
                frame[f"emu_{metric}"].to_numpy(dtype=float),
            ]
        )
        values = values[np.isfinite(values)]
        if values.size:
            low, high = float(np.min(values)), float(np.max(values))
            figure.add_trace(
                go.Scatter(
                    x=[low, high],
                    y=[low, high],
                    mode="lines",
                    line={"color": "black", "dash": "dash"},
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
    for direction, color in (("up", "#1769aa"), ("down", "#d32f2f")):
        metric = f"{direction}_flex_wh_m2_k"
        figure.add_trace(
            go.Box(
                x=frame["horizon_hours"],
                y=frame[f"diff_{metric}"],
                name=f"{direction} error",
                marker_color=color,
                boxmean=True,
            ),
            row=2,
            col=1,
        )
        grouped = frame.groupby("horizon_hours", sort=True)
        for signal, dash in (("sim", "solid"), ("emu", "dash")):
            means = grouped[f"{signal}_{metric}"].mean()
            figure.add_trace(
                go.Scatter(
                    x=means.index,
                    y=means.values,
                    mode="lines+markers",
                    name=f"{signal} {direction}",
                    line={"color": color, "dash": dash},
                ),
                row=2,
                col=2,
            )
    figure.update_xaxes(title_text="simulated Wh/(m2 K)", row=1, col=1)
    figure.update_yaxes(title_text="emulated Wh/(m2 K)", row=1, col=1)
    figure.update_xaxes(title_text="simulated Wh/(m2 K)", row=1, col=2)
    figure.update_yaxes(title_text="emulated Wh/(m2 K)", row=1, col=2)
    figure.update_xaxes(title_text="duration H [h]", row=2, col=1)
    figure.update_yaxes(title_text="emulated - simulated Wh/(m2 K)", row=2, col=1)
    figure.update_xaxes(title_text="duration H [h]", row=2, col=2)
    figure.update_yaxes(title_text="mean Wh/(m2 K)", row=2, col=2)
    figure.update_layout(
        title="Setpoint event-study flexibility comparison",
        template="plotly_white",
        height=940,
        width=1450,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    figure.write_html(path)


def _write_profile_plot(path: Path, result, *, model_name: str) -> None:
    channel_names = (
        ["Indoor temperature"]
        if result.target.shape[-1] == 1
        else ["Indoor temperature", "Delivered room heat", "HP electric power"]
    )
    units = ["degC"] if len(channel_names) == 1 else ["degC", "W/m2", "W/m2"]
    figure = make_subplots(
        rows=len(channel_names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=channel_names,
    )
    for channel, (name, unit) in enumerate(zip(channel_names, units), start=1):
        index = channel - 1
        figure.add_trace(
            go.Scatter(
                x=result.datetime,
                y=result.target[:, index],
                mode="lines",
                name=f"Simulated {name}",
                line={"color": "#1769aa", "width": 1.5},
            ),
            row=channel,
            col=1,
        )
        if result.scenarios is not None:
            lower = np.quantile(result.scenarios[:, :, index], 0.025, axis=0)
            upper = np.quantile(result.scenarios[:, :, index], 0.975, axis=0)
            figure.add_trace(
                go.Scatter(
                    x=result.datetime,
                    y=lower,
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=channel,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=result.datetime,
                    y=upper,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(211,47,47,0.14)",
                    line={"width": 0},
                    name=f"95% {name}",
                    hoverinfo="skip",
                ),
                row=channel,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=result.datetime,
                y=result.mean[:, index],
                mode="lines",
                name=f"Emulated {name}",
                line={"color": "#d32f2f", "width": 1.5},
            ),
            row=channel,
            col=1,
        )
        figure.update_yaxes(title_text=unit, row=channel, col=1)
    figure.update_xaxes(title_text="Time", row=len(channel_names), col=1)
    figure.update_layout(
        title=f"{model_name}, profile {result.profile_id}",
        template="plotly_white",
        height=450 if len(channel_names) == 1 else 1050,
        width=1500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    figure.write_html(path)


def score_artifact(
    artifact_dir: Path,
    *,
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    profile_source: ProfileSource = "test",
    profile_ids: Sequence[int] | None = None,
    max_profiles: int | None = None,
    kpi_mode: KpiMode = "mean",
    num_scenarios: int = 100,
    hp_scenario_mode: str = "bernoulli",
    ventilation_rollout_mode: str = "recorded",
    horizons_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    dt_hours: float = 0.25,
    min_setpoint_change: float = 0.05,
    min_events: int = 20,
    kpi_estimator: str = "direct_ratio",
    controls: str = "full",
    seed: int = 13,
    num_profile_plots: int = 5,
    output_suffix: str = "",
) -> Path:
    """Reload a selected model and score complete autonomous trajectories."""
    artifact = load_artifact(artifact_dir)
    if kpi_mode == "scenario_average" and not artifact.spec.probabilistic:
        raise ValueError("scenario_average requires a probabilistic artifact")
    profiles = load_profiles(
        artifact,
        dataset_path=dataset_path,
        source=profile_source,
        max_profiles=max_profiles,
        profile_ids=profile_ids,
    )
    destination = output_dir or artifact.artifact_dir / "scores"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    event_config = EventStudyConfig(
        horizons_hours=horizons_hours,
        horizon_steps=tuple(max(1, int(round(value / dt_hours))) for value in horizons_hours),
        dt_hours=dt_hours,
        min_setpoint_change=min_setpoint_change,
        min_events=min_events,
        controls=controls,  # type: ignore[arg-type]
        estimator=kpi_estimator,  # type: ignore[arg-type]
    )
    metric_rows: list[dict] = []
    truth_kpi_rows: list[dict] = []
    prediction_kpi_rows: list[dict] = []
    scenario_kpi_rows: list[dict] = []
    key = jax.random.PRNGKey(seed)
    aggregate_sum_squared: np.ndarray | None = None
    aggregate_sum_absolute: np.ndarray | None = None
    aggregate_sum_error: np.ndarray | None = None
    aggregate_count = 0
    scenario_count = (
        num_scenarios
        if artifact.spec.probabilistic and kpi_mode == "scenario_average"
        else 0
    )

    for profile_index, profile in enumerate(profiles):
        key, profile_key = jax.random.split(key)
        result = predict_profile(
            artifact,
            profile,
            key=profile_key,
            num_scenarios=scenario_count,
            hp_scenario_mode=hp_scenario_mode,
            ventilation_rollout_mode=ventilation_rollout_mode,
        )
        metric_rows.append({"profile_id": profile.profile_id, **_regression_metrics(result.mean, result.target)})
        if profile_index < num_profile_plots:
            _write_profile_plot(
                destination
                / _output_name(
                    f"profile_{profile_index + 1:02d}_{profile.profile_id}.html",
                    output_suffix,
                ),
                result,
                model_name=artifact.spec.name,
            )
        error = np.asarray(result.mean, dtype=np.float64) - np.asarray(result.target, dtype=np.float64)
        sum_squared = np.sum(error**2, axis=0)
        sum_absolute = np.sum(np.abs(error), axis=0)
        sum_error = np.sum(error, axis=0)
        aggregate_sum_squared = sum_squared if aggregate_sum_squared is None else aggregate_sum_squared + sum_squared
        aggregate_sum_absolute = sum_absolute if aggregate_sum_absolute is None else aggregate_sum_absolute + sum_absolute
        aggregate_sum_error = sum_error if aggregate_sum_error is None else aggregate_sum_error + sum_error
        aggregate_count += error.shape[0]
        if artifact.spec.task != "closed_loop_hp":
            continue
        assert isinstance(profile, ClosedLoopProfile)
        simulated = estimate_trace_flexibility(profile, result.target, event_config)
        truth_kpi_rows.extend({"signal": "simulation", **asdict(value)} for value in simulated)
        if kpi_mode == "mean":
            predicted = estimate_trace_flexibility(profile, result.mean, event_config)
        else:
            assert result.scenarios is not None
            scenario_kpis = [
                estimate_trace_flexibility(profile, trace, event_config)
                for trace in result.scenarios
            ]
            predicted = _average_kpis(scenario_kpis)
            for scenario_index, scenario_values in enumerate(scenario_kpis):
                scenario_kpi_rows.extend(
                    {"scenario_index": scenario_index, **asdict(value)}
                    for value in scenario_values
                )
        prediction_kpi_rows.extend({"signal": "emulation", **asdict(value)} for value in predicted)
        if result.scenarios is not None and kpi_mode == "mean":
            for scenario_index, trace in enumerate(result.scenarios):
                scenario_values = estimate_trace_flexibility(profile, trace, event_config)
                scenario_kpi_rows.extend(
                    {"scenario_index": scenario_index, **asdict(value)}
                    for value in scenario_values
                )

    if aggregate_sum_squared is None or aggregate_sum_absolute is None or aggregate_sum_error is None:
        raise ValueError("No profiles were scored")
    aggregate_metrics: dict[str, float] = {}
    for index, name in enumerate(_metric_names(len(aggregate_sum_squared))):
        aggregate_metrics[f"{name}_rmse"] = float(
            np.sqrt(aggregate_sum_squared[index] / aggregate_count)
        )
        aggregate_metrics[f"{name}_mae"] = float(
            aggregate_sum_absolute[index] / aggregate_count
        )
        aggregate_metrics[f"{name}_bias"] = float(
            aggregate_sum_error[index] / aggregate_count
        )
    pd.DataFrame(metric_rows).to_csv(
        destination / _output_name("profile_metrics.csv", output_suffix),
        index=False,
    )
    if truth_kpi_rows:
        pd.DataFrame(truth_kpi_rows + prediction_kpi_rows).to_csv(
            destination / _output_name("flexibility_kpis.csv", output_suffix),
            index=False,
        )
        _write_flexibility_comparison_plot(
            destination
            / _output_name("flexibility_event_study_comparison.html", output_suffix),
            truth_kpi_rows,
            prediction_kpi_rows,
        )
    coverage_rows = _coverage_rows(
        truth_kpi_rows,
        scenario_kpi_rows,
        np.linspace(0.05, 0.95, 19),
    )
    if scenario_kpi_rows:
        pd.DataFrame(scenario_kpi_rows).to_csv(
            destination / _output_name("flexibility_kpi_scenarios.csv", output_suffix),
            index=False,
        )
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(
            destination / _output_name("flexibility_kpi_coverage.csv", output_suffix),
            index=False,
        )
        _write_coverage_plot(
            destination / _output_name("flexibility_kpi_coverage.html", output_suffix),
            coverage_rows,
        )
    summary = {
        "model_name": artifact.spec.name,
        "task": artifact.spec.task,
        "profile_source": "explicit" if profile_ids is not None else profile_source,
        "profile_count": len(profiles),
        "kpi_mode": kpi_mode,
        "kpi_estimator": event_config.estimator,
        "kpi_controls": event_config.controls,
        "kpi_controls_applied": (
            event_config.estimator == "regression" and event_config.controls != "none"
        ),
        "output_suffix": output_suffix,
        "num_scenarios": scenario_count,
        "ventilation_rollout_mode": ventilation_rollout_mode,
        "metrics": aggregate_metrics,
    }
    (destination / _output_name("summary.json", output_suffix)).write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(f"saved_scores={destination}")
    print(" ".join(f"{key}={value:.4f}" for key, value in aggregate_metrics.items()))
    return destination
