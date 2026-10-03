"""Zero-shot evaluation of a trained emulator under an HP force-off policy.

The structured counterfactual is applied at the HP generation boundary:

    Pel_SH[t] = (1 - force_off[t]) * Pel_model[t]

The room-heat request and thermal states are not hard-masked. Consequently,
stored heat can continue to reach the room after the compressor is disabled.
This script compares that rollout with the same model run while ignoring the
policy. An optional unstructured artifact can be included as a baseline. New
mixed-policy LSTM artifacts receive force-off as an ordinary exogenous input;
older artifacts retain their original input schema and do not receive it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    DATETIME_COLUMN,
    PROFILE_ID_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
)
from neural_building_emulator.data import ClosedLoopProfile, read_dataset_frame
from neural_building_emulator.dataset_adapter import (
    HP_FORCE_OFF_EFFECTIVE_COLUMN,
    HP_FORCE_OFF_REQUESTED_COLUMN,
)
from neural_building_emulator.flexibility_event_study_kpis import (
    TemporalTraceCalibrationAccumulator,
    _write_temporal_calibration_html,
)

from .artifacts import LoadedArtifact, load_artifact
from .intervention_flexibility import (
    InterventionFlexibilityKpi,
    estimate_force_off_flexibility_curve,
)
from .prediction import PredictionResult, predict_profile
from .profiles import load_profiles
from .scoring import _write_profile_plot as _write_probabilistic_profile_plot

CHANNELS = (
    ("Tin", "degC"),
    ("Qroom", "W/m2"),
    ("Pel_SH", "W/m2"),
)
POLICY_COLUMNS = {
    "requested": HP_FORCE_OFF_REQUESTED_COLUMN,
    "effective": HP_FORCE_OFF_EFFECTIVE_COLUMN,
}
COLORS = {
    "simulation": "#1769aa",
    "structured_gated": "#2e7d32",
    "structured_ungated": "#d32f2f",
    "unstructured": "#7b1fa2",
    "setpoint": "#616161",
    "force_off": "#f57c00",
}


def _post_release_mask(force_off: np.ndarray, steps: int) -> np.ndarray:
    active = np.asarray(force_off, dtype=bool)
    releases = np.flatnonzero(np.r_[False, active[:-1] & ~active[1:]])
    mask = np.zeros_like(active)
    for index in releases:
        mask[index : min(index + steps, len(mask))] = True
    return mask & ~active


def _artifact_metadata(artifact: LoadedArtifact) -> dict:
    if artifact.backend == "legacy" and artifact.legacy_artifact is not None:
        return artifact.legacy_artifact.metadata
    return artifact.metadata


def _policy_traces(
    dataset_path: Path,
    profiles: Sequence[ClosedLoopProfile],
    policy_column: str,
) -> dict[int, np.ndarray]:
    profile_ids = [profile.profile_id for profile in profiles]
    frame = read_dataset_frame(
        dataset_path,
        columns=[DATETIME_COLUMN, PROFILE_ID_COLUMN, policy_column],
        profile_ids=profile_ids,
    )
    traces: dict[int, np.ndarray] = {}
    profiles_by_id = {profile.profile_id: profile for profile in profiles}
    for profile_id, group in frame.groupby(PROFILE_ID_COLUMN, sort=False):
        profile_id = int(profile_id)
        values = (
            group.sort_values(DATETIME_COLUMN)[policy_column]
            .to_numpy(dtype=np.float32)
        )
        profile = profiles_by_id[profile_id]
        if values.shape == (len(profile.inputs) + 1,):
            values = values[1:]
        if values.shape != (len(profile.inputs),):
            raise ValueError(
                f"Profile {profile_id} policy trace has shape {values.shape}; "
                f"expected {(len(profile.inputs),)}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Profile {profile_id} has non-finite {policy_column}")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"Profile {profile_id} has {policy_column} outside [0, 1]")
        traces[profile_id] = values
    missing = sorted(set(profile_ids).difference(traces))
    if missing:
        raise ValueError(f"Missing force-off traces for profiles {missing}")
    return traces


def _metric_rows(
    profile: ClosedLoopProfile,
    prediction: np.ndarray,
    force_off: np.ndarray,
    heating_available: np.ndarray,
    *,
    model: str,
    setpoint_index: int,
    dt_hours: float,
) -> list[dict[str, float | int | str]]:
    target = np.asarray(profile.targets, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    masks = {
        "all": np.ones(len(force_off), dtype=bool),
        "forced_off": (force_off > 0.5) & heating_available,
        "unrestricted": (force_off <= 0.5) & heating_available,
    }
    rows: list[dict[str, float | int | str]] = []
    for period, mask in masks.items():
        if not np.any(mask):
            continue
        for channel, (signal, _) in enumerate(CHANNELS):
            error = prediction[mask, channel] - target[mask, channel]
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "model": model,
                    "period": period,
                    "signal": signal,
                    "count": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                }
            )

    setpoint = np.asarray(profile.inputs[:, setpoint_index], dtype=np.float64)
    off = masks["forced_off"]
    rows.extend(
        [
            {
                "profile_id": profile.profile_id,
                "model": model,
                "period": "annual",
                "signal": "Pel_energy",
                "count": len(force_off),
                "rmse": np.nan,
                "mae": float(dt_hours * np.sum(prediction[:, 2]) / 1000.0),
                "bias": float(
                    dt_hours * np.sum(prediction[:, 2] - target[:, 2]) / 1000.0
                ),
            },
            {
                "profile_id": profile.profile_id,
                "model": model,
                "period": "forced_off",
                "signal": "Pel_leakage",
                "count": int(off.sum()),
                "rmse": float(np.max(prediction[off, 2])) if np.any(off) else np.nan,
                "mae": float(np.mean(prediction[off, 2])) if np.any(off) else np.nan,
                "bias": float(np.mean(target[off, 2])) if np.any(off) else np.nan,
            },
            {
                "profile_id": profile.profile_id,
                "model": model,
                "period": "forced_off",
                "signal": "comfort_error",
                "count": int(off.sum()),
                "rmse": (
                    float(np.sqrt(np.mean((prediction[off, 0] - setpoint[off]) ** 2)))
                    if np.any(off)
                    else np.nan
                ),
                "mae": (
                    float(np.mean(np.abs(prediction[off, 0] - setpoint[off])))
                    if np.any(off)
                    else np.nan
                ),
                "bias": (
                    float(np.mean(prediction[off, 0] - setpoint[off]))
                    if np.any(off)
                    else np.nan
                ),
            },
        ]
    )
    return rows


def _event_rows(
    profile: ClosedLoopProfile,
    traces: dict[str, np.ndarray],
    force_off: np.ndarray,
    heating_available: np.ndarray,
    *,
    setpoint_index: int,
    dt_hours: float,
    horizon_hours: float,
) -> list[dict[str, float | int | str]]:
    horizon = max(1, int(round(horizon_hours / dt_hours)))
    starts = np.flatnonzero(
        (force_off > 0.5)
        & heating_available
        & np.concatenate([[False], force_off[:-1] <= 0.5])
    )
    starts = starts[starts + horizon <= len(force_off)]
    setpoint = np.asarray(profile.inputs[:, setpoint_index], dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    for event_index, start in enumerate(starts):
        for model, values in traces.items():
            derived = (
                values[:, 0] - setpoint,
                values[:, 1],
                values[:, 2],
            )
            for signal, series in zip(("Tin_minus_Tset", "Qroom", "Pel_SH"), derived):
                for lag in range(horizon):
                    rows.append(
                        {
                            "profile_id": profile.profile_id,
                            "event_index": event_index,
                            "model": model,
                            "signal": signal,
                            "lag_hours": lag * dt_hours,
                            "value": float(series[start + lag]),
                        }
                    )
    return rows


def _kpi_row(
    kpi: InterventionFlexibilityKpi,
    *,
    model: str,
    signal: str,
    scenario_index: int | None = None,
) -> dict[str, float | int | bool | str | None]:
    return {
        "profile_id": kpi.profile_id,
        "model": model,
        "signal": signal,
        "scenario_index": scenario_index,
        "intervention_type": kpi.intervention_type,
        "direction": "force_off",
        "estimator": kpi.estimator,
        "horizon_hours": kpi.horizon_hours,
        "horizon_steps": kpi.horizon_steps,
        "event_count": kpi.event_count,
        "transition_count": kpi.transition_count,
        "valid": kpi.valid,
        "response_gain_w_m2_per_unit": kpi.response_gain_w_m2_per_unit,
        "flexibility_wh_m2_per_unit": kpi.flexibility_wh_m2_per_unit,
        "unit": "Wh/(m2 intervention)",
    }


def _profile_force_off_kpis(
    profile: ClosedLoopProfile,
    predictions: dict[str, PredictionResult],
    force_off: np.ndarray,
    heating_available: np.ndarray,
    *,
    input_columns: Sequence[str],
    horizons_hours: Sequence[float],
    dt_hours: float,
    min_events: int,
) -> tuple[list[dict], list[dict]]:
    point_rows: list[dict] = []
    scenario_rows: list[dict] = []
    truth = estimate_force_off_flexibility_curve(
        profile,
        profile.targets,
        force_off,
        heating_available,
        input_columns=input_columns,
        horizons_hours=horizons_hours,
        dt_hours=dt_hours,
        min_events=min_events,
    )
    for model, prediction in predictions.items():
        point_rows.extend(
            _kpi_row(kpi, model=model, signal="simulation") for kpi in truth
        )
        if prediction.scenarios is None:
            emulated = estimate_force_off_flexibility_curve(
                profile,
                prediction.mean,
                force_off,
                heating_available,
                input_columns=input_columns,
                horizons_hours=horizons_hours,
                dt_hours=dt_hours,
                min_events=min_events,
            )
            point_rows.extend(
                _kpi_row(kpi, model=model, signal="emulation") for kpi in emulated
            )
            continue
        by_key: dict[tuple[str, float], list[InterventionFlexibilityKpi]] = {}
        for scenario_index, trace in enumerate(prediction.scenarios):
            estimates = estimate_force_off_flexibility_curve(
                profile,
                trace,
                force_off,
                heating_available,
                input_columns=input_columns,
                horizons_hours=horizons_hours,
                dt_hours=dt_hours,
                min_events=min_events,
            )
            for kpi in estimates:
                scenario_rows.append(
                    _kpi_row(
                        kpi,
                        model=model,
                        signal="emulation",
                        scenario_index=scenario_index,
                    )
                )
                if kpi.valid:
                    by_key.setdefault((kpi.estimator, kpi.horizon_hours), []).append(kpi)
        for truth_kpi in truth:
            values = by_key.get((truth_kpi.estimator, truth_kpi.horizon_hours), [])
            if values:
                gain = float(np.mean([value.response_gain_w_m2_per_unit for value in values]))
                point = InterventionFlexibilityKpi(
                    profile_id=profile.profile_id,
                    intervention_type="force_off",
                    estimator=truth_kpi.estimator,
                    horizon_hours=truth_kpi.horizon_hours,
                    horizon_steps=truth_kpi.horizon_steps,
                    event_count=int(np.median([value.event_count for value in values])),
                    transition_count=int(
                        np.median([value.transition_count for value in values])
                    ),
                    valid=True,
                    response_gain_w_m2_per_unit=gain,
                    flexibility_wh_m2_per_unit=float(truth_kpi.horizon_hours * gain),
                )
            else:
                point = InterventionFlexibilityKpi(
                    profile_id=profile.profile_id,
                    intervention_type="force_off",
                    estimator=truth_kpi.estimator,
                    horizon_hours=truth_kpi.horizon_hours,
                    horizon_steps=truth_kpi.horizon_steps,
                    event_count=0,
                    transition_count=0,
                    valid=False,
                    response_gain_w_m2_per_unit=float("nan"),
                    flexibility_wh_m2_per_unit=float("nan"),
                )
            point_rows.append(_kpi_row(point, model=model, signal="emulation"))
    return point_rows, scenario_rows


def _force_off_kpi_coverage(
    point: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    if point.empty or scenarios.empty:
        return pd.DataFrame()
    truth = point[(point["signal"] == "simulation") & point["valid"].astype(bool)][
        ["profile_id", "model", "estimator", "horizon_hours", "flexibility_wh_m2_per_unit"]
    ].rename(columns={"flexibility_wh_m2_per_unit": "truth"})
    samples = scenarios[scenarios["valid"].astype(bool)].merge(
        truth,
        on=["profile_id", "model", "estimator", "horizon_hours"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict] = []
    quantiles = np.linspace(0.05, 0.95, 19)
    for (model, estimator, horizon), part in samples.groupby(
        ["model", "estimator", "horizon_hours"], sort=False
    ):
        grouped = list(part.groupby("profile_id", sort=False))
        for quantile in quantiles:
            covered: list[bool] = []
            for _, profile_part in grouped:
                predicted_quantile = float(
                    np.quantile(
                        profile_part["flexibility_wh_m2_per_unit"].to_numpy(float),
                        quantile,
                    )
                )
                covered.append(float(profile_part["truth"].iloc[0]) <= predicted_quantile)
            rows.append(
                {
                    "model": model,
                    "intervention_type": "force_off",
                    "direction": "force_off",
                    "estimator": estimator,
                    "horizon_hours": float(horizon),
                    "quantile": float(quantile),
                    "empirical_coverage": float(np.mean(covered)),
                    "profile_count": len(covered),
                }
            )
    return pd.DataFrame(rows)


def _write_force_off_kpi_coverage(path: Path, coverage: pd.DataFrame) -> None:
    if coverage.empty:
        return
    estimators = ("direct_ratio", "regression")
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Direct F(force_off)", "Regression-adjusted F(force_off)"),
    )
    for column, estimator in enumerate(estimators, start=1):
        subset = coverage[coverage["estimator"] == estimator]
        for (model, horizon), part in subset.groupby(
            ["model", "horizon_hours"], sort=False
        ):
            figure.add_trace(
                go.Scatter(
                    x=part["quantile"],
                    y=part["empirical_coverage"],
                    mode="lines+markers",
                    name=f"{model}, H={horizon:g} h",
                    legendgroup=f"{model}:{horizon}",
                    line={"color": COLORS.get(str(model), "#455a64")},
                    showlegend=column == 1,
                    hovertemplate=(
                        "model=%{fullData.name}<br>nominal=%{x:.2f}"
                        "<br>empirical=%{y:.2f}<extra></extra>"
                    ),
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
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="Nominal quantile", range=[0, 1], row=1, col=column)
        figure.update_yaxes(title_text="Empirical rank CDF", range=[0, 1], row=1, col=column)
    figure.update_layout(
        title="Force-off intervention flexibility reliability",
        template="plotly_white",
        height=650,
        width=1450,
        legend={"orientation": "h", "y": -0.18},
        hoverlabel={"namelength": -1, "align": "left"},
    )
    figure.write_html(path)


def _write_force_off_comparison_plot(
    path: Path,
    profile: ClosedLoopProfile,
    force_off: np.ndarray,
    gated: PredictionResult,
    ungated: PredictionResult,
    unstructured: PredictionResult | None,
    *,
    setpoint_index: int,
) -> None:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.34, 0.25, 0.25, 0.16],
        subplot_titles=("Indoor temperature", "Zone heating power", "HP SH electric power", "Force-off command"),
    )
    datetime = np.asarray(profile.datetime)
    setpoint = np.asarray(profile.inputs[:, setpoint_index])
    for channel, (_, unit) in enumerate(CHANNELS):
        row = channel + 1
        figure.add_trace(
            go.Scatter(x=datetime, y=gated.target[:, channel], name="EnergyPlus", legendgroup="simulation", line={"color": COLORS["simulation"], "width": 1.5}),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=datetime, y=gated.mean[:, channel], name="Structured + force-off gate", legendgroup="structured_gated", line={"color": COLORS["structured_gated"], "width": 1.4}),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(x=datetime, y=ungated.mean[:, channel], name="Structured, policy ignored", legendgroup="structured_ungated", line={"color": COLORS["structured_ungated"], "width": 1.1}),
            row=row,
            col=1,
        )
        if unstructured is not None:
            figure.add_trace(
                go.Scatter(x=datetime, y=unstructured.mean[:, channel], name="Unstructured LSTM", legendgroup="unstructured", line={"color": COLORS["unstructured"], "width": 1.0}),
                row=row,
                col=1,
            )
        if channel == 0:
            figure.add_trace(
                go.Scatter(x=datetime, y=setpoint, name="Tset", legendgroup="setpoint", line={"color": COLORS["setpoint"], "width": 1.0, "dash": "dot"}),
                row=row,
                col=1,
            )
        figure.update_yaxes(title_text=unit, row=row, col=1)
    figure.add_trace(
        go.Scatter(x=datetime, y=force_off, name="Force off", legendgroup="force_off", line={"color": COLORS["force_off"], "width": 1.2}, fill="tozeroy"),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="0/1", range=[-0.05, 1.05], row=4, col=1)
    figure.update_xaxes(title_text="Time", row=4, col=1)
    figure.update_layout(
        title=f"Force-off policy transfer, profile {profile.profile_id}",
        template="plotly_white",
        height=1250,
        width=1550,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
    )
    figure.write_html(path)


def _write_dashboard(path: Path, metrics: pd.DataFrame, events: pd.DataFrame) -> None:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Tin RMSE",
            "Qroom RMSE",
            "Pel_SH RMSE",
            "Force-off Tin - Tset response",
            "Force-off Qroom response",
            "Force-off Pel_SH response",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )
    models = list(dict.fromkeys(metrics["model"].tolist()))
    for col, signal in enumerate(("Tin", "Qroom", "Pel_SH"), start=1):
        part = metrics[(metrics["signal"] == signal) & (metrics["period"].isin(["all", "forced_off"]))]
        summary = part.groupby(["model", "period"], sort=False)["rmse"].mean().reset_index()
        for model in models:
            values = summary[summary["model"] == model]
            figure.add_trace(
                go.Bar(
                    x=values["period"],
                    y=values["rmse"],
                    name=model,
                    legendgroup=model,
                    marker_color=COLORS.get(model, "#455a64"),
                    showlegend=col == 1,
                    hovertemplate="model=%{fullData.name}<br>period=%{x}<br>RMSE=%{y:.4f}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    if not events.empty:
        event_mean = events.groupby(["model", "signal", "lag_hours"], sort=False)["value"].mean().reset_index()
        for col, signal in enumerate(("Tin_minus_Tset", "Qroom", "Pel_SH"), start=1):
            for model in dict.fromkeys(event_mean["model"].tolist()):
                values = event_mean[(event_mean["model"] == model) & (event_mean["signal"] == signal)]
                figure.add_trace(
                    go.Scatter(
                        x=values["lag_hours"],
                        y=values["value"],
                        mode="lines",
                        name=model,
                        legendgroup=model,
                        line={"color": COLORS.get(model, "#455a64")},
                        showlegend=False,
                        hovertemplate="model=%{fullData.name}<br>lag=%{x:.2f} h<br>value=%{y:.4f}<extra></extra>",
                    ),
                    row=2,
                    col=col,
                )
            figure.update_xaxes(title_text="Hours after force-off onset", row=2, col=col)
    figure.update_layout(
        title="Zero-shot HP force-off policy transfer",
        template="plotly_white",
        barmode="group",
        height=900,
        width=1600,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "right", "x": 1.0},
    )
    figure.write_html(path)


def evaluate_force_off_policy(
    artifact_dir: Path,
    *,
    dataset_path: Path,
    output_dir: Path,
    profile_source: str = "test",
    profile_ids: Sequence[int] | None = None,
    max_profiles: int | None = None,
    policy_signal: str = "requested",
    num_eval_particles: int = 16,
    num_scenarios: int = 0,
    hp_scenario_mode: str = "bernoulli",
    ventilation_rollout_mode: str = "eplus_rule",
    unstructured_artifact_dir: Path | None = None,
    dt_hours: float = 0.25,
    event_horizon_hours: float = 12.0,
    kpi_horizons_hours: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    kpi_min_events: int = 20,
    num_profile_plots: int = 5,
    profile_plot_ids: Sequence[int] | None = None,
    evaluate_ungated: bool = True,
    seed: int = 13,
) -> Path:
    artifact = load_artifact(artifact_dir)
    if artifact.spec.name != "closed_loop_hp_contracting_probabilistic":
        raise ValueError(
            "Force-off structural gating currently requires "
            "closed_loop_hp_contracting_probabilistic"
        )
    profiles = load_profiles(
        artifact,
        dataset_path=dataset_path,
        source=profile_source,  # type: ignore[arg-type]
        max_profiles=max_profiles,
        profile_ids=profile_ids,
    )
    if not all(isinstance(profile, ClosedLoopProfile) for profile in profiles):
        raise TypeError("Force-off evaluation requires closed-loop profiles")
    closed_loop_profiles = list(profiles)  # type: ignore[arg-type]
    metadata = _artifact_metadata(artifact)
    input_columns = list(metadata["input_columns"])
    setpoint_index = input_columns.index(SETPOINT_TIMESERIES_COLUMN)
    availability_index = input_columns.index("space_heating_available")
    policy_column = POLICY_COLUMNS[policy_signal]
    structured_force_off_index = (
        input_columns.index(HP_FORCE_OFF_REQUESTED_COLUMN)
        if HP_FORCE_OFF_REQUESTED_COLUMN in input_columns
        else -1
    )
    force_off_by_id = _policy_traces(dataset_path, closed_loop_profiles, policy_column)

    unstructured_artifact = None
    unstructured_profiles: dict[int, ClosedLoopProfile] = {}
    if unstructured_artifact_dir is not None:
        unstructured_artifact = load_artifact(unstructured_artifact_dir)
        if "lstm" not in unstructured_artifact.spec.name:
            raise ValueError("--unstructured-artifact-dir must point to an LSTM artifact")
        loaded = load_profiles(
            unstructured_artifact,
            dataset_path=dataset_path,
            profile_ids=[profile.profile_id for profile in closed_loop_profiles],
        )
        unstructured_profiles = {
            profile.profile_id: profile
            for profile in loaded
            if isinstance(profile, ClosedLoopProfile)
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    event_rows: list[dict] = []
    kpi_rows: list[dict] = []
    kpi_scenario_rows: list[dict] = []
    key = jax.random.PRNGKey(seed)
    scenario_count = num_scenarios if num_scenarios > 0 else 0
    temporal_accumulators: dict[str, TemporalTraceCalibrationAccumulator] = {}
    regime_temporal_accumulators: dict[
        tuple[str, str], TemporalTraceCalibrationAccumulator
    ] = {}
    if scenario_count > 0:
        quantiles = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
        temporal_accumulators["structured_gated"] = (
            TemporalTraceCalibrationAccumulator(quantiles)
        )
        if unstructured_artifact is not None and unstructured_artifact.spec.probabilistic:
            temporal_accumulators["unstructured"] = (
                TemporalTraceCalibrationAccumulator(quantiles)
            )
    selected_plot_ids = (
        None if profile_plot_ids is None else {int(value) for value in profile_plot_ids}
    )
    plotted_profiles = 0
    probabilistic_trace_dir = output_dir / "probabilistic_traces"
    for profile_index, profile in enumerate(closed_loop_profiles):
        key, profile_key = jax.random.split(key)
        force_off = force_off_by_id[profile.profile_id]
        force_off_active = np.asarray(force_off > np.float32(0.5))
        setpoint = np.asarray(profile.inputs[:, setpoint_index])
        heating_available = np.asarray(
            profile.inputs[:, availability_index] > np.float32(0.5)
        )
        dates = pd.DatetimeIndex(pd.to_datetime(profile.datetime))
        winter = np.asarray(dates.month.isin((11, 12, 1, 2, 3)))
        regime_masks = {
            "winter": winter,
            "heating_available_no_lockout": heating_available & ~force_off_active,
            "active_lockout": heating_available & force_off_active,
            "winter_active_lockout": winter & heating_available & force_off_active,
            "post_release_0_3h": heating_available
            & _post_release_mask(force_off_active, 12),
            "post_release_0_12h": heating_available
            & _post_release_mask(force_off_active, 48),
            "post_release_0_24h": heating_available
            & _post_release_mask(force_off_active, 96),
        }
        generation_available = np.float32(1.0) - force_off
        gated = predict_profile(
            artifact,
            profile,
            key=profile_key,
            num_scenarios=scenario_count,
            num_eval_particles=num_eval_particles,
            hp_scenario_mode=hp_scenario_mode,
            ventilation_rollout_mode=ventilation_rollout_mode,
            hp_generation_availability=generation_available,
        )
        predictions: dict[str, PredictionResult] = {
            "structured_gated": gated,
        }
        ungated = None
        if evaluate_ungated:
            ungated_profile = profile
            if structured_force_off_index >= 0:
                ungated_inputs = np.asarray(profile.inputs).copy()
                ungated_inputs[:, structured_force_off_index] = np.float32(0.0)
                ungated_profile = replace(profile, inputs=ungated_inputs)
            ungated = predict_profile(
                artifact,
                ungated_profile,
                key=profile_key,
                num_scenarios=scenario_count,
                num_eval_particles=num_eval_particles,
                hp_scenario_mode=hp_scenario_mode,
                ventilation_rollout_mode=ventilation_rollout_mode,
            )
            predictions["structured_ungated"] = ungated
        unstructured_result = None
        if unstructured_artifact is not None:
            unstructured_result = predict_profile(
                unstructured_artifact,
                unstructured_profiles[profile.profile_id],
                key=profile_key,
                num_scenarios=(scenario_count if unstructured_artifact.spec.probabilistic else 0),
                num_eval_particles=num_eval_particles,
                hp_scenario_mode=hp_scenario_mode,
                ventilation_rollout_mode=ventilation_rollout_mode,
            )
            predictions["unstructured"] = unstructured_result

        temporal_masks = np.ones_like(gated.target, dtype=bool)
        temporal_masks[:, 1] = heating_available
        temporal_masks[:, 2] = heating_available
        for model_name, accumulator in temporal_accumulators.items():
            result = predictions[model_name]
            if result.scenarios is None:
                continue
            accumulator.add(
                result.target,
                result.scenarios,
                masks=temporal_masks,
            )
            for regime_name, regime_mask in regime_masks.items():
                key_name = (model_name, regime_name)
                regime_accumulator = regime_temporal_accumulators.setdefault(
                    key_name,
                    TemporalTraceCalibrationAccumulator(quantiles),
                )
                regime_accumulator.add(
                    result.target,
                    result.scenarios,
                    masks=np.repeat(regime_mask[:, np.newaxis], 3, axis=1),
                )

        traces = {"simulation": np.asarray(profile.targets)}
        traces.update({name: np.asarray(result.mean) for name, result in predictions.items()})
        for name, result in predictions.items():
            metric_rows.extend(
                _metric_rows(
                    profile,
                    result.mean,
                    force_off,
                    heating_available,
                    model=name,
                    setpoint_index=setpoint_index,
                    dt_hours=dt_hours,
                )
            )
        event_rows.extend(
            _event_rows(
                profile,
                traces,
                force_off,
                heating_available,
                setpoint_index=setpoint_index,
                dt_hours=dt_hours,
                horizon_hours=event_horizon_hours,
            )
        )
        profile_kpis, profile_kpi_scenarios = _profile_force_off_kpis(
            profile,
            predictions,
            force_off,
            heating_available,
            input_columns=input_columns,
            horizons_hours=kpi_horizons_hours,
            dt_hours=dt_hours,
            min_events=kpi_min_events,
        )
        kpi_rows.extend(profile_kpis)
        kpi_scenario_rows.extend(profile_kpi_scenarios)
        should_plot = (
            profile_index < num_profile_plots
            if selected_plot_ids is None
            else profile.profile_id in selected_plot_ids
        )
        if should_plot and plotted_profiles < num_profile_plots and ungated is not None:
            plotted_profiles += 1
            _write_force_off_comparison_plot(
                output_dir / f"force_off_profile_{plotted_profiles:02d}_{profile.profile_id}.html",
                profile,
                force_off,
                gated,
                ungated,
                unstructured_result,
                setpoint_index=setpoint_index,
            )
            probabilistic_trace_dir.mkdir(parents=True, exist_ok=True)
            for model_name in temporal_accumulators:
                result = predictions[model_name]
                _write_probabilistic_profile_plot(
                    probabilistic_trace_dir
                    / (
                        f"force_off_profile_{plotted_profiles:02d}_"
                        f"{profile.profile_id}_{model_name}.html"
                    ),
                    result,
                    model_name=model_name,
                    setpoint=setpoint,
                    policy_trace=force_off,
                    policy_label="HP force-off request",
                )
        print(
            f"evaluated_profile={profile.profile_id} "
            f"force_off_fraction={float(np.mean(force_off)):.4f} "
            f"active_heating_force_off_fraction="
            f"{float(np.mean(force_off[heating_available])):.4f}"
        )

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    kpis = pd.DataFrame(kpi_rows)
    kpi_scenarios = pd.DataFrame(kpi_scenario_rows)
    kpi_coverage = _force_off_kpi_coverage(kpis, kpi_scenarios)
    metrics.to_csv(output_dir / "force_off_profile_metrics.csv", index=False)
    events.to_csv(output_dir / "force_off_event_responses.csv", index=False)
    kpis.to_csv(output_dir / "force_off_flexibility_kpis.csv", index=False)
    kpi_scenarios.to_csv(
        output_dir / "force_off_flexibility_kpi_scenarios.csv", index=False
    )
    kpi_coverage.to_csv(
        output_dir / "force_off_flexibility_kpi_coverage.csv", index=False
    )
    temporal_rows: list[dict] = []
    temporal_histogram_rows: list[dict] = []
    temporal_score_rows: list[dict] = []
    for model_name, accumulator in temporal_accumulators.items():
        reliability = pd.DataFrame(accumulator.reliability_rows())
        histogram = pd.DataFrame(accumulator.histogram_rows())
        scores = pd.DataFrame(accumulator.probabilistic_score_rows())
        if reliability.empty or histogram.empty:
            continue
        reliability.insert(0, "model", model_name)
        histogram.insert(0, "model", model_name)
        temporal_rows.extend(reliability.to_dict("records"))
        temporal_histogram_rows.extend(histogram.to_dict("records"))
        if not scores.empty:
            scores.insert(0, "model", model_name)
            temporal_score_rows.extend(scores.to_dict("records"))
        _write_temporal_calibration_html(
            output_dir / f"temporal_trace_calibration_{model_name}.html",
            reliability,
            histogram,
            title=(
                "Force-off temporal trace calibration"
                f"<br><sup>{model_name}, scenarios={scenario_count}</sup>"
            ),
        )
    pd.DataFrame(temporal_rows).to_csv(
        output_dir / "temporal_trace_calibration.csv", index=False
    )
    pd.DataFrame(temporal_histogram_rows).to_csv(
        output_dir / "temporal_trace_rank_histogram.csv", index=False
    )
    pd.DataFrame(temporal_score_rows).to_csv(
        output_dir / "temporal_trace_probabilistic_scores.csv", index=False
    )
    regime_temporal_rows: list[dict] = []
    regime_histogram_rows: list[dict] = []
    regime_score_rows: list[dict] = []
    for (model_name, regime_name), accumulator in regime_temporal_accumulators.items():
        for row in accumulator.reliability_rows():
            regime_temporal_rows.append(
                {"model": model_name, "regime": regime_name, **row}
            )
        for row in accumulator.histogram_rows():
            regime_histogram_rows.append(
                {"model": model_name, "regime": regime_name, **row}
            )
        for row in accumulator.probabilistic_score_rows():
            regime_score_rows.append(
                {"model": model_name, "regime": regime_name, **row}
            )
    pd.DataFrame(regime_temporal_rows).to_csv(
        output_dir / "temporal_trace_calibration_by_regime.csv", index=False
    )
    pd.DataFrame(regime_histogram_rows).to_csv(
        output_dir / "temporal_trace_rank_histogram_by_regime.csv", index=False
    )
    pd.DataFrame(regime_score_rows).to_csv(
        output_dir / "temporal_trace_probabilistic_scores_by_regime.csv", index=False
    )
    aggregate = (
        metrics.groupby(["model", "period", "signal"], sort=False)
        .agg(
            profile_count=("profile_id", "nunique"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "force_off_aggregate_metrics.csv", index=False)
    _write_dashboard(output_dir / "force_off_policy_dashboard.html", metrics, events)
    _write_force_off_kpi_coverage(
        output_dir / "force_off_flexibility_kpi_coverage.html",
        kpi_coverage,
    )
    summary = {
        "artifact_dir": str(artifact_dir),
        "dataset_path": str(dataset_path),
        "policy_signal": policy_signal,
        "profile_source": profile_source,
        "profile_count": len(closed_loop_profiles),
        "num_eval_particles": num_eval_particles,
        "num_scenarios": num_scenarios,
        "hp_scenario_mode": hp_scenario_mode,
        "effective_hp_rollout_mode": (
            hp_scenario_mode if num_scenarios > 0 else "expected"
        ),
        "ventilation_rollout_mode": ventilation_rollout_mode,
        "evaluate_ungated": evaluate_ungated,
        "kpi_horizons_hours": [float(value) for value in kpi_horizons_hours],
        "kpi_min_events": int(kpi_min_events),
        "kpi_definition": {
            "direct_ratio": (
                "H * sum(P_pre - P_post at persistent force-off onsets) / "
                "sum(binary force-off dose)"
            ),
            "regression": (
                "negative onset-treatment coefficient from post-minus-pre power, "
                "using stable no-intervention windows and pre-event/exogenous controls"
            ),
            "unit": "Wh/(m2 intervention)",
        },
        "mean_force_off_fraction": float(
            np.mean([np.mean(force_off_by_id[p.profile_id]) for p in closed_loop_profiles])
        ),
        "mean_active_heating_force_off_fraction": float(
            np.mean(
                [
                    np.mean(
                        force_off_by_id[profile.profile_id][
                            profile.inputs[:, availability_index] > np.float32(0.5)
                        ]
                    )
                    for profile in closed_loop_profiles
                ]
            )
        ),
        "aggregate_metrics": aggregate.to_dict(orient="records"),
    }
    (output_dir / "force_off_policy_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot transfer to an EnergyPlus HP force-off policy."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/control=force_off"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-source", choices=("train", "test", "selected"), default="test")
    parser.add_argument("--profile-id", type=int, action="append", dest="profile_ids")
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--policy-signal", choices=tuple(POLICY_COLUMNS), default="requested")
    parser.add_argument("--num-eval-particles", type=int, default=16)
    parser.add_argument("--num-scenarios", type=int, default=0)
    parser.add_argument("--hp-scenario-mode", choices=("bernoulli", "expected"), default="bernoulli")
    parser.add_argument("--ventilation-rollout-mode", choices=("recorded", "eplus_rule"), default="eplus_rule")
    parser.add_argument("--unstructured-artifact-dir", type=Path)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--event-horizon-hours", type=float, default=12.0)
    parser.add_argument(
        "--kpi-horizons-hours",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0, 3.0),
    )
    parser.add_argument("--kpi-min-events", type=int, default=20)
    parser.add_argument("--num-profile-plots", type=int, default=5)
    parser.add_argument(
        "--skip-ungated",
        action="store_true",
        help="Skip the redundant no-policy counterfactual and score only gated rollouts.",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate_force_off_policy(
        args.artifact_dir,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        profile_source=args.profile_source,
        profile_ids=args.profile_ids,
        max_profiles=args.max_profiles,
        policy_signal=args.policy_signal,
        num_eval_particles=args.num_eval_particles,
        num_scenarios=args.num_scenarios,
        hp_scenario_mode=args.hp_scenario_mode,
        ventilation_rollout_mode=args.ventilation_rollout_mode,
        unstructured_artifact_dir=args.unstructured_artifact_dir,
        dt_hours=args.dt_hours,
        event_horizon_hours=args.event_horizon_hours,
        kpi_horizons_hours=args.kpi_horizons_hours,
        kpi_min_events=args.kpi_min_events,
        num_profile_plots=args.num_profile_plots,
        evaluate_ungated=not args.skip_ungated,
        seed=args.seed,
    )
    print(f"saved_force_off_policy_evaluation={output}")


if __name__ == "__main__":
    main()
