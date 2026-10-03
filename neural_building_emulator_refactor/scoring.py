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

from neural_building_emulator.columns import SETPOINT_TIMESERIES_COLUMN
from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator.flexibility_event_study_kpis import (
    EventStudyConfig,
    FlexibilityKpi,
    TemporalTraceCalibrationAccumulator,
    _temporal_calibration_summary,
    _write_temporal_calibration_html,
    estimate_trace_flexibility,
)

from .artifacts import load_artifact
from .fast_kpis import FastKpiConfig, trajectory_metrics
from .pel_dynamics_calibration import (
    PelDynamicsCalibrationAccumulator,
    write_pel_dynamics_dashboard,
)
from .prediction import predict_profile
from .profiles import ProfileSource, load_profiles
from .qroom_temporal_calibration_diagnostics import (
    BinaryReliabilityAccumulator,
    ConditionalRankAccumulator,
)

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


def _metric_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    result: dict[str, dict[str, float]] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column == "profile_id":
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            result[column] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
                "max": float(np.max(values)),
            }
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


def _write_kpi_outputs(
    destination: Path,
    *,
    truth_rows: list[dict],
    prediction_rows: list[dict],
    scenario_rows: list[dict],
    output_suffix: str,
) -> list[dict]:
    if truth_rows:
        pd.DataFrame(truth_rows + prediction_rows).to_csv(
            destination / _output_name("flexibility_kpis.csv", output_suffix),
            index=False,
        )
        _write_flexibility_comparison_plot(
            destination
            / _output_name("flexibility_event_study_comparison.html", output_suffix),
            truth_rows,
            prediction_rows,
        )
    coverage_rows = _coverage_rows(
        truth_rows,
        scenario_rows,
        np.linspace(0.05, 0.95, 19),
    )
    if scenario_rows:
        pd.DataFrame(scenario_rows).to_csv(
            destination
            / _output_name("flexibility_kpi_scenarios.csv", output_suffix),
            index=False,
        )
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(
            destination
            / _output_name("flexibility_kpi_coverage.csv", output_suffix),
            index=False,
        )
        _write_coverage_plot(
            destination
            / _output_name("flexibility_kpi_coverage.html", output_suffix),
            coverage_rows,
        )
    return coverage_rows


def _scenario_extreme_traces(
    scenarios: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest and worst complete scenario for each output."""
    scenario_values = np.asarray(scenarios, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if scenario_values.ndim != 3 or target_values.ndim != 2:
        raise ValueError(
            "Expected scenarios [scenario, time, output] and target [time, output]"
        )
    if scenario_values.shape[1:] != target_values.shape:
        raise ValueError("Scenario and target time/output dimensions do not match")
    rmse = np.sqrt(
        np.nanmean((scenario_values - target_values[np.newaxis, :, :]) ** 2, axis=1)
    )
    if not np.all(np.any(np.isfinite(rmse), axis=0)):
        raise ValueError("Cannot select extreme traces from non-finite scenarios")
    best = np.nanargmin(rmse, axis=0)
    worst = np.nanargmax(rmse, axis=0)
    outputs = np.arange(target_values.shape[1])
    return (
        scenario_values[best, :, outputs].T,
        scenario_values[worst, :, outputs].T,
    )


def _write_profile_plot(
    path: Path,
    result,
    *,
    model_name: str,
    setpoint: np.ndarray | None = None,
    policy_trace: np.ndarray | None = None,
    policy_label: str = "Policy command",
) -> None:
    channel_names = (
        ["Indoor temperature"]
        if result.target.shape[-1] == 1
        else ["Indoor temperature", "Delivered room heat", "HP electric power"]
    )
    units = ["degC"] if len(channel_names) == 1 else ["degC", "W/m2", "W/m2"]
    row_count = len(channel_names) + int(policy_trace is not None)
    subplot_titles = [*channel_names]
    if policy_trace is not None:
        subplot_titles.append(policy_label)
    figure = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=subplot_titles,
    )
    closest = worst = None
    if result.scenarios is not None:
        closest, worst = _scenario_extreme_traces(result.scenarios, result.target)
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
                    legendgroup="interval",
                    showlegend=channel == 1,
                    hoverinfo="skip",
                ),
                row=channel,
                col=1,
            )
            assert closest is not None and worst is not None
            figure.add_trace(
                go.Scattergl(
                    x=result.datetime,
                    y=closest[:, index],
                    mode="lines",
                    name="Closest annual scenario",
                    legendgroup="closest",
                    line={"color": "#2e7d32", "width": 1.1, "dash": "dash"},
                    showlegend=channel == 1,
                ),
                row=channel,
                col=1,
            )
            figure.add_trace(
                go.Scattergl(
                    x=result.datetime,
                    y=worst[:, index],
                    mode="lines",
                    name="Worst annual scenario",
                    legendgroup="worst",
                    line={"color": "#7b1fa2", "width": 1.1, "dash": "dot"},
                    showlegend=channel == 1,
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
        if channel == 1 and setpoint is not None:
            figure.add_trace(
                go.Scattergl(
                    x=result.datetime,
                    y=np.asarray(setpoint),
                    mode="lines",
                    name="Heating setpoint",
                    line={"color": "#616161", "width": 1.0, "dash": "dot"},
                ),
                row=channel,
                col=1,
            )
        figure.update_yaxes(title_text=unit, row=channel, col=1)
    if policy_trace is not None:
        figure.add_trace(
            go.Scattergl(
                x=result.datetime,
                y=np.asarray(policy_trace),
                mode="lines",
                name=policy_label,
                line={"color": "#f57c00", "width": 1.2},
                fill="tozeroy",
            ),
            row=row_count,
            col=1,
        )
        figure.update_yaxes(title_text="0/1", range=[-0.05, 1.05], row=row_count, col=1)
    figure.update_xaxes(title_text="Time", row=row_count, col=1)
    figure.update_layout(
        title=f"{model_name}, profile {result.profile_id}",
        template="plotly_white",
        height=(450 if len(channel_names) == 1 else 1050) + (220 if policy_trace is not None else 0),
        width=1500,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        hoverlabel={"namelength": -1, "align": "left"},
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
    profile_plot_ids: Sequence[int] | None = None,
    output_suffix: str = "",
    evaluate_direct_f: bool = False,
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
    direct_f_config = (
        EventStudyConfig(
            horizons_hours=horizons_hours,
            horizon_steps=event_config.horizon_steps,
            dt_hours=dt_hours,
            min_setpoint_change=min_setpoint_change,
            min_events=min_events,
            controls="none",
            estimator="direct_ratio",
        )
        if evaluate_direct_f and kpi_estimator != "direct_ratio"
        else None
    )
    metric_rows: list[dict] = []
    truth_kpi_rows: list[dict] = []
    prediction_kpi_rows: list[dict] = []
    scenario_kpi_rows: list[dict] = []
    direct_f_truth_rows: list[dict] = []
    direct_f_prediction_rows: list[dict] = []
    direct_f_scenario_rows: list[dict] = []
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
    temporal_calibration = (
        TemporalTraceCalibrationAccumulator(
            tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
        )
        if artifact.spec.task == "closed_loop_hp" and scenario_count > 0
        else None
    )
    saved_metadata = (
        artifact.legacy_artifact.metadata
        if artifact.backend == "legacy" and artifact.legacy_artifact is not None
        else artifact.metadata
    )
    input_columns = list(saved_metadata.get("input_columns", ()))
    availability_index = (
        input_columns.index("space_heating_available")
        if "space_heating_available" in input_columns
        else None
    )
    setpoint_index = (
        input_columns.index(SETPOINT_TIMESERIES_COLUMN)
        if SETPOINT_TIMESERIES_COLUMN in input_columns
        else 0
    )
    pel_dynamics = (
        PelDynamicsCalibrationAccumulator(
            dt_hours=dt_hours,
            setpoint_threshold_c=min_setpoint_change,
        )
        if temporal_calibration is not None
        else None
    )
    qroom_occurrence = (
        BinaryReliabilityAccumulator.create()
        if temporal_calibration is not None
        else None
    )
    qroom_positive_rank = (
        ConditionalRankAccumulator.create()
        if temporal_calibration is not None
        else None
    )
    fast_config = FastKpiConfig(dt_hours=dt_hours)
    target_scale = np.asarray(artifact.scalers.target.scale, dtype=np.float64)
    selected_plot_ids = (
        None if profile_plot_ids is None else {int(value) for value in profile_plot_ids}
    )
    plotted_profiles = 0

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
        metric_rows.append(
            {
                "profile_id": profile.profile_id,
                **_regression_metrics(result.mean, result.target),
                **trajectory_metrics(
                    result.mean,
                    result.target,
                    target_scale,
                    fast_config,
                ),
            }
        )
        should_plot = (
            profile_index < num_profile_plots
            if selected_plot_ids is None
            else profile.profile_id in selected_plot_ids
        )
        if should_plot and plotted_profiles < num_profile_plots:
            plotted_profiles += 1
            _write_profile_plot(
                destination
                / _output_name(
                    f"profile_{plotted_profiles:02d}_{profile.profile_id}.html",
                    output_suffix,
                ),
                result,
                model_name=artifact.spec.name,
                setpoint=(
                    np.asarray(profile.inputs[:, setpoint_index])
                    if artifact.spec.task == "closed_loop_hp"
                    else None
                ),
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
        if temporal_calibration is not None:
            assert result.scenarios is not None
            temporal_masks = np.ones_like(result.target, dtype=bool)
            heating_available = np.ones(result.target.shape[0], dtype=bool)
            if availability_index is not None:
                heating_available = (
                    profile.inputs[:, availability_index] > np.float32(0.5)
                )
                temporal_masks[:, 1] = heating_available
                temporal_masks[:, 2] = heating_available
            temporal_calibration.add(
                result.target,
                result.scenarios,
                masks=temporal_masks,
            )
            assert qroom_occurrence is not None
            assert qroom_positive_rank is not None
            qroom_truth = np.asarray(result.target[:, 1])
            qroom_scenarios = np.asarray(result.scenarios[:, :, 1])
            qroom_threshold = 0.05
            qroom_occurrence.add(
                np.mean(qroom_scenarios > qroom_threshold, axis=0)[heating_available],
                (qroom_truth > qroom_threshold)[heating_available],
            )
            qroom_positive_rank.add(
                qroom_truth[heating_available],
                qroom_scenarios[:, heating_available],
                qroom_threshold,
            )
            assert pel_dynamics is not None
            current_temperature = np.concatenate(
                [
                    np.asarray(profile.initial_temperature).reshape(-1)[:1],
                    np.asarray(result.target[:-1, 0]),
                ]
            )
            pel_dynamics.add(
                truth=np.asarray(result.target[:, 2]),
                scenarios=np.asarray(result.scenarios[:, :, 2]),
                setpoint=np.asarray(profile.inputs[:, setpoint_index]),
                current_temperature=current_temperature,
                available=heating_available,
            )
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
        if direct_f_config is not None:
            direct_simulated = estimate_trace_flexibility(
                profile,
                result.target,
                direct_f_config,
            )
            direct_f_truth_rows.extend(
                {"signal": "simulation", **asdict(value)}
                for value in direct_simulated
            )
            if kpi_mode == "mean":
                direct_predicted = estimate_trace_flexibility(
                    profile,
                    result.mean,
                    direct_f_config,
                )
            else:
                assert result.scenarios is not None
                direct_scenario_kpis = [
                    estimate_trace_flexibility(profile, trace, direct_f_config)
                    for trace in result.scenarios
                ]
                direct_predicted = _average_kpis(direct_scenario_kpis)
                for scenario_index, scenario_values in enumerate(
                    direct_scenario_kpis
                ):
                    direct_f_scenario_rows.extend(
                        {"scenario_index": scenario_index, **asdict(value)}
                        for value in scenario_values
                    )
            direct_f_prediction_rows.extend(
                {"signal": "emulation", **asdict(value)}
                for value in direct_predicted
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
    coverage_rows = _write_kpi_outputs(
        destination,
        truth_rows=truth_kpi_rows,
        prediction_rows=prediction_kpi_rows,
        scenario_rows=scenario_kpi_rows,
        output_suffix=output_suffix,
    )
    direct_f_coverage_rows = _write_kpi_outputs(
        destination,
        truth_rows=direct_f_truth_rows,
        prediction_rows=direct_f_prediction_rows,
        scenario_rows=direct_f_scenario_rows,
        output_suffix=f"{output_suffix}_f",
    ) if direct_f_config is not None else []
    temporal_calibration_rows = (
        temporal_calibration.reliability_rows()
        if temporal_calibration is not None
        else []
    )
    temporal_histogram_rows = (
        temporal_calibration.histogram_rows()
        if temporal_calibration is not None
        else []
    )
    temporal_summary: list[dict] = []
    temporal_score_rows = (
        temporal_calibration.probabilistic_score_rows()
        if temporal_calibration is not None
        else []
    )
    if temporal_calibration_rows:
        temporal_calibration_frame = pd.DataFrame(temporal_calibration_rows)
        temporal_histogram_frame = pd.DataFrame(temporal_histogram_rows)
        temporal_summary = _temporal_calibration_summary(
            temporal_calibration_frame
        )
        score_by_signal = {
            str(row["signal"]): row for row in temporal_score_rows
        }
        for row in temporal_summary:
            row.update(
                {
                    key: value
                    for key, value in score_by_signal.get(str(row["signal"]), {}).items()
                    if key.startswith("mean_")
                }
            )
        temporal_calibration_frame.to_csv(
            destination
            / _output_name("temporal_trace_calibration.csv", output_suffix),
            index=False,
        )
        temporal_histogram_frame.to_csv(
            destination
            / _output_name("temporal_trace_rank_histogram.csv", output_suffix),
            index=False,
        )
        _write_temporal_calibration_html(
            destination
            / _output_name("temporal_trace_calibration.html", output_suffix),
            temporal_calibration_frame,
            temporal_histogram_frame,
            title=(
                "Temporal trace distribution calibration"
                f"<br><sup>{artifact.spec.name}, scenarios={scenario_count}, "
                "tie-aware randomized ensemble ranks</sup>"
            ),
        )
    if temporal_score_rows:
        pd.DataFrame(temporal_score_rows).to_csv(
            destination
            / _output_name("temporal_trace_probabilistic_scores.csv", output_suffix),
            index=False,
        )
    pel_dynamics_summary: dict[str, float] = {}
    if pel_dynamics is not None:
        transition_frame = pel_dynamics.reliability_frame()
        conditional_frame = pel_dynamics.conditional_frame()
        conditional_summary_frame = pel_dynamics.conditional_summary_frame()
        run_duration_frame = pel_dynamics.run_duration_frame()
        acf_frame = pel_dynamics.acf_frame()
        annual_energy_frame = pel_dynamics.annual_energy_frame()
        pel_dynamics_summary = pel_dynamics.summary()
        frames = (
            ("pel_transition_reliability.csv", transition_frame),
            ("pel_conditional_calibration.csv", conditional_frame),
            ("pel_conditional_calibration_summary.csv", conditional_summary_frame),
            ("pel_run_duration_distribution.csv", run_duration_frame),
            ("pel_mode_autocorrelation.csv", acf_frame),
            ("pel_annual_energy_calibration.csv", annual_energy_frame),
        )
        for filename, frame in frames:
            if not frame.empty:
                frame.to_csv(
                    destination / _output_name(filename, output_suffix),
                    index=False,
                )
        write_pel_dynamics_dashboard(
            destination
            / _output_name("pel_dynamics_calibration.html", output_suffix),
            reliability=transition_frame,
            conditional_summary=conditional_summary_frame,
            run_duration=run_duration_frame,
            acf=acf_frame,
            annual_energy=annual_energy_frame,
            title=(
                "HP switching and conditional calibration"
                f"<br><sup>{artifact.spec.name}, scenarios={scenario_count}</sup>"
            ),
        )
    qroom_calibration_summary: dict[str, float] = {}
    if qroom_occurrence is not None and qroom_positive_rank is not None:
        occurrence = qroom_occurrence.metrics()
        magnitude = qroom_positive_rank.metrics()
        qroom_calibration_summary = {
            "qroom_occurrence_brier": occurrence["brier_score"],
            "qroom_occurrence_log_loss": occurrence["log_loss"],
            "qroom_occurrence_ece": occurrence["expected_calibration_error"],
            "qroom_positive_rank_calibration_mae": magnitude[
                "mean_abs_calibration_error"
            ],
            "qroom_positive_rank_calibration_max": magnitude[
                "max_abs_calibration_error"
            ],
            "qroom_positive_central_90_coverage": magnitude[
                "central_90_coverage"
            ],
        }
        occurrence_rows = qroom_occurrence.rows()
        if occurrence_rows:
            pd.DataFrame(occurrence_rows).to_csv(
                destination
                / _output_name("qroom_occurrence_reliability.csv", output_suffix),
                index=False,
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
        "direct_f_evaluated": direct_f_config is not None or kpi_estimator == "direct_ratio",
        "direct_f_output_suffix": (
            f"{output_suffix}_f" if direct_f_config is not None else output_suffix
        ),
        "direct_f_coverage_row_count": len(direct_f_coverage_rows),
        "output_suffix": output_suffix,
        "num_scenarios": scenario_count,
        "ventilation_rollout_mode": ventilation_rollout_mode,
        "metrics": aggregate_metrics,
        "profile_metric_summary": _metric_summary(metric_rows),
        "temporal_trace_calibration": temporal_summary,
        "pel_dynamics": pel_dynamics_summary,
        "qroom_calibration": qroom_calibration_summary,
    }
    (destination / _output_name("summary.json", output_suffix)).write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(f"saved_scores={destination}")
    print(" ".join(f"{key}={value:.4f}" for key, value in aggregate_metrics.items()))
    return destination
