"""Compare drift, intervention response, and spread on saved HP model test profiles."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Sequence

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator.flexibility_event_study_kpis import (
    EventStudyConfig,
    regression_data_for_trace,
)

from .artifacts import load_artifact
from .prediction import PredictionResult, predict_profile
from .profiles import load_profiles, saved_profile_ids

SIGNALS = ("Tin [C]", "Qroom [W/m2]", "Pel_SH [W/m2]")
AGE_EDGES_DAYS = (0, 10, 30, 90, 180, 366)
MODEL_COLORS = {"probabilistic_lstm": "#b0443d", "probabilistic_structured": "#176c78"}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _interval_means(values: np.ndarray, events: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equal-length means immediately before and after each event."""
    values = np.asarray(values, dtype=np.float64)
    totals = np.concatenate(
        [np.zeros((*values.shape[:-1], 1), dtype=np.float64), np.cumsum(values, axis=-1)],
        axis=-1,
    )
    before = (totals[..., events] - totals[..., events - steps]) / steps
    after = (totals[..., events + steps] - totals[..., events]) / steps
    return before, after


def _slope(change: np.ndarray, setpoint_change: np.ndarray) -> float:
    denominator = float(np.sum(setpoint_change**2))
    return float(np.sum(change * setpoint_change) / denominator) if denominator > 0 else float("nan")


def horizon_rows(model: str, result: PredictionResult, *, dt_hours: float = 0.25) -> list[dict]:
    assert result.scenarios is not None
    rows = []
    age_days = np.arange(len(result.target)) * dt_hours / 24.0
    for left, right in zip(AGE_EDGES_DAYS[:-1], AGE_EDGES_DAYS[1:]):
        select = (age_days >= left) & (age_days < right)
        if not np.any(select):
            continue
        for signal_index, signal in enumerate(SIGNALS):
            truth = result.target[select, signal_index]
            mean = result.mean[select, signal_index]
            scenarios = result.scenarios[:, select, signal_index]
            lo, hi = np.quantile(scenarios, [0.05, 0.95], axis=0)
            rows.append({
                "model": model,
                "profile_id": result.profile_id,
                "signal": signal,
                "age_from_initialization_days": f"{left}-{right}",
                "age_start_days": left,
                "age_end_days": right,
                "steps": int(np.sum(select)),
                "rmse": _rmse(mean - truth),
                "bias": float(np.mean(mean - truth)),
                "coverage_90": float(np.mean((truth >= lo) & (truth <= hi))),
                "mean_interval_width": float(np.mean(hi - lo)),
            })
    return rows


def context_rows(
    model: str,
    profile: ClosedLoopProfile,
    result: PredictionResult,
    input_columns: Sequence[str],
    *,
    dt_hours: float = 0.25,
) -> Iterator[dict]:
    """Check whether errors and spread fail under identifiable operating conditions."""
    assert result.scenarios is not None
    inputs = profile.inputs
    column_index = {name: index for index, name in enumerate(input_columns)}
    availability = inputs[:, column_index["space_heating_available"]]
    solar = inputs[:, column_index["Environment, Site Global Horizontal Solar Radiation Rate per Area"]]
    ventilation = inputs[:, column_index["FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate"]]
    cases = {
        "SH available": availability > 0.5,
        "SH unavailable": availability <= 0.5,
        "observed Pel active": result.target[:, 2] > 0.5,
        "observed Pel inactive": result.target[:, 2] <= 0.5,
        "high solar (>100 W/m2)": solar > 100.0,
        "low solar (<=100 W/m2)": solar <= 100.0,
        "ventilation active": ventilation > 1e-6,
        "ventilation inactive": ventilation <= 1e-6,
    }
    age_days = np.arange(len(inputs)) * dt_hours / 24.0
    for phase, age_select in (("full year", np.ones(len(inputs), dtype=bool)),
                              ("days 90-180", (age_days >= 90) & (age_days < 180))):
        for condition, select in cases.items():
            select = select & age_select
            if np.sum(select) < 48:
                continue
            for signal_index, signal in enumerate(SIGNALS):
                truth = result.target[select, signal_index]
                mean = result.mean[select, signal_index]
                scenarios = result.scenarios[:, select, signal_index]
                lo, hi = np.quantile(scenarios, [0.05, 0.95], axis=0)
                yield {
                    "model": model,
                    "profile_id": profile.profile_id,
                    "phase": phase,
                    "condition": condition,
                    "signal": signal,
                    "steps": int(np.sum(select)),
                    "rmse": _rmse(mean - truth),
                    "bias": float(np.mean(mean - truth)),
                    "coverage_90": float(np.mean((truth >= lo) & (truth <= hi))),
                    "mean_interval_width": float(np.mean(hi - lo)),
                    "negative_scenario_fraction": float(np.mean(scenarios < -1e-6)),
                }


def intervention_rows(
    model: str,
    profile: ClosedLoopProfile,
    result: PredictionResult,
    *,
    dt_hours: float = 0.25,
    min_setpoint_change: float = 0.05,
) -> list[dict]:
    assert result.scenarios is not None
    rows = []
    for horizon_hours in (0.5, 1.0, 2.0, 3.0):
        steps = int(round(horizon_hours / dt_hours))
        config = EventStudyConfig(
            horizons_hours=(horizon_hours,),
            horizon_steps=(steps,),
            dt_hours=dt_hours,
            min_setpoint_change=min_setpoint_change,
            min_events=0,
            controls="none",
            estimator="direct_ratio",
        )
        event_data = regression_data_for_trace(profile, result.target, horizon_steps=steps, config=config)
        events, setpoint_change = event_data.events, event_data.deltas
        if events.size == 0:
            continue
        for signal_index, signal in enumerate(SIGNALS):
            _, truth_post = _interval_means(result.target[:, signal_index], events, steps)
            truth_pre, _ = _interval_means(result.target[:, signal_index], events, steps)
            truth_change = truth_post - truth_pre
            mean_pre, mean_post = _interval_means(result.mean[:, signal_index], events, steps)
            mean_change = mean_post - mean_pre
            scenario_pre, scenario_post = _interval_means(
                result.scenarios[:, :, signal_index], events, steps
            )
            scenario_change = scenario_post - scenario_pre
            lo, hi = np.quantile(scenario_change, [0.05, 0.95], axis=0)
            correlation = (
                float(np.corrcoef(truth_change, mean_change)[0, 1])
                if np.std(truth_change) > 1e-8 and np.std(mean_change) > 1e-8
                else float("nan")
            )
            for direction, select in (("up", setpoint_change > 0), ("down", setpoint_change < 0)):
                if not np.any(select):
                    continue
                rows.append({
                    "model": model,
                    "profile_id": profile.profile_id,
                    "signal": signal,
                    "horizon_hours": horizon_hours,
                    "direction": direction,
                    "events": int(np.sum(select)),
                    "response_rmse": _rmse(mean_change[select] - truth_change[select]),
                    "response_bias": float(np.mean(mean_change[select] - truth_change[select])),
                    "response_correlation": correlation,
                    "truth_gain_per_k": _slope(truth_change[select], setpoint_change[select]),
                    "pred_gain_per_k": _slope(mean_change[select], setpoint_change[select]),
                    "response_coverage_90": float(np.mean(
                        (truth_change[select] >= lo[select]) & (truth_change[select] <= hi[select])
                    )),
                    "response_interval_width": float(np.mean(hi[select] - lo[select])),
                })
    return rows


def restart_rows(
    model: str,
    artifact,
    profile: ClosedLoopProfile,
    annual: PredictionResult,
    *,
    seed: int,
    num_scenarios: int,
    dt_hours: float = 0.25,
) -> list[dict]:
    """Compare the annual trace with a fresh 90-day rollout at the same calendar dates."""
    rows = []
    steps_per_day = int(round(24.0 / dt_hours))
    for start_day in (90, 180, 270):
        start = start_day * steps_per_day
        stop = min(start + 90 * steps_per_day, len(profile.inputs))
        if stop - start < 30 * steps_per_day:
            continue
        segment = replace(
            profile,
            datetime=profile.datetime[start:stop],
            inputs=profile.inputs[start:stop],
            targets=profile.targets[start:stop],
            initial_temperature=profile.initial_temperature[start:stop],
        )
        restarted = predict_profile(
            artifact,
            segment,
            key=jax.random.PRNGKey(seed + start_day),
            num_scenarios=num_scenarios,
            ventilation_rollout_mode="eplus_rule",
        )
        for horizon_days in (10, 30, 90):
            count = min(horizon_days * steps_per_day, stop - start)
            for signal_index, signal in enumerate(SIGNALS):
                truth = segment.targets[:count, signal_index]
                continuous = annual.mean[start:start + count, signal_index]
                fresh = restarted.mean[:count, signal_index]
                rows.append({
                    "model": model,
                    "profile_id": profile.profile_id,
                    "signal": signal,
                    "start_day": start_day,
                    "horizon_days": horizon_days,
                    "continuous_rmse": _rmse(continuous - truth),
                    "restarted_rmse": _rmse(fresh - truth),
                    "continuous_bias": float(np.mean(continuous - truth)),
                    "restarted_bias": float(np.mean(fresh - truth)),
                })
    return rows


def _summaries(
    horizon: pd.DataFrame,
    intervention: pd.DataFrame,
    restart: pd.DataFrame,
    context: pd.DataFrame,
) -> dict:
    result = {
        "horizon": horizon.groupby(["model", "signal", "age_start_days", "age_end_days"], sort=False)
        [["rmse", "bias", "coverage_90", "mean_interval_width"]].mean().reset_index().to_dict("records"),
        "intervention": intervention.groupby(["model", "signal", "horizon_hours", "direction"], sort=False)
        [["response_rmse", "response_bias", "response_coverage_90", "truth_gain_per_k", "pred_gain_per_k"]]
        .mean().reset_index().to_dict("records"),
        "restart": restart.groupby(["model", "signal", "horizon_days"], sort=False)
        [["continuous_rmse", "restarted_rmse", "continuous_bias", "restarted_bias"]]
        .mean().reset_index().to_dict("records") if not restart.empty else [],
        "context": context.groupby(["model", "signal", "phase", "condition"], sort=False)
        [["rmse", "bias", "coverage_90", "mean_interval_width", "negative_scenario_fraction"]]
        .mean().reset_index().to_dict("records"),
    }
    return result


def _write_dashboard(path: Path, summary: dict) -> None:
    figure = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Error versus time since initialization", "90% trace coverage",
                        "3 h setpoint-to-P response", "Restarted versus continuous Tin"),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    horizon = pd.DataFrame(summary["horizon"])
    intervention = pd.DataFrame(summary["intervention"])
    restart = pd.DataFrame(summary["restart"])
    for model, color in MODEL_COLORS.items():
        tin = horizon[(horizon.model == model) & (horizon.signal == SIGNALS[0])]
        if not tin.empty:
            ages = (tin.age_start_days.to_numpy() + tin.age_end_days.to_numpy()) / 2
            figure.add_trace(go.Scatter(x=ages, y=tin.rmse, name=model, mode="lines+markers",
                                        line={"color": color}, legendgroup=model), row=1, col=1)
            figure.add_trace(go.Scatter(x=ages, y=tin.coverage_90, name=model, mode="lines+markers",
                                        line={"color": color}, legendgroup=model, showlegend=False), row=1, col=2)
        pel = intervention[(intervention.model == model) & (intervention.signal == SIGNALS[2])
                           & (intervention.horizon_hours == 3.0)]
        if not pel.empty:
            for _, item in pel.iterrows():
                figure.add_trace(go.Bar(x=[item.direction], y=[item.pred_gain_per_k], name=model,
                                        marker_color=color, legendgroup=model, showlegend=False,
                                        hovertemplate=f"{model}<br>{item.direction}: %{{y:.2f}} W/m2/K<extra></extra>"),
                                 row=2, col=1)
                figure.add_trace(go.Scatter(x=[item.direction], y=[item.truth_gain_per_k], mode="markers",
                                            marker={"symbol": "diamond", "size": 13, "color": "#222"},
                                            showlegend=False, hovertemplate="EnergyPlus: %{y:.2f} W/m2/K<extra></extra>"),
                                 row=2, col=1)
        if not restart.empty:
            reset_tin = restart[(restart.model == model) & (restart.signal == SIGNALS[0])]
        else:
            reset_tin = pd.DataFrame()
        if not reset_tin.empty:
            figure.add_trace(go.Scatter(x=reset_tin.horizon_days, y=reset_tin.continuous_rmse,
                                        name=f"{model}: continuous", mode="lines+markers",
                                        line={"color": color}, legendgroup=model, showlegend=False), row=2, col=2)
            figure.add_trace(go.Scatter(x=reset_tin.horizon_days, y=reset_tin.restarted_rmse,
                                        name=f"{model}: restarted", mode="lines+markers",
                                        line={"color": color, "dash": "dash"}, legendgroup=model,
                                        showlegend=False), row=2, col=2)
    figure.add_hline(y=0.9, line_dash="dot", line_color="#555", row=1, col=2)
    figure.update_xaxes(title_text="Days after initialization", row=1, col=1)
    figure.update_xaxes(title_text="Days after initialization", row=1, col=2)
    figure.update_xaxes(title_text="Setpoint shock direction", row=2, col=1)
    figure.update_xaxes(title_text="Restart rollout horizon [days]", row=2, col=2)
    figure.update_yaxes(title_text="Tin RMSE [C]", row=1, col=1)
    figure.update_yaxes(title_text="Empirical coverage", row=1, col=2)
    figure.update_yaxes(title_text="Pel gain [W/m2/K]", row=2, col=1)
    figure.update_yaxes(title_text="Tin RMSE [C]", row=2, col=2)
    figure.update_layout(template="plotly_white", height=900, width=1400,
                         title="Held-out autonomous rollout diagnosis", barmode="group",
                         legend={"orientation": "h", "y": 1.07}, margin={"t": 110})
    figure.write_html(path)


def run_diagnostic(
    benchmark_dir: Path,
    output_dir: Path,
    *,
    max_profiles: int = 20,
    num_scenarios: int = 24,
    restart_profiles: int = 6,
    seed: int = 13,
) -> Path:
    paths = json.loads((benchmark_dir / "artifact_paths.json").read_text())
    artifacts = {
        model: load_artifact(Path(paths[model]))
        for model in MODEL_COLORS
    }
    ids = set(saved_profile_ids(artifacts["probabilistic_lstm"], "test"))
    ids &= set(saved_profile_ids(artifacts["probabilistic_structured"], "test"))
    ordered_ids = sorted(ids)
    if max_profiles <= 0 or num_scenarios <= 0 or restart_profiles < 0:
        raise ValueError("max_profiles and num_scenarios must be positive; restart_profiles cannot be negative")
    selected_ids = [ordered_ids[index] for index in np.linspace(
        0, len(ordered_ids) - 1, min(max_profiles, len(ordered_ids)), dtype=int
    )]
    if not selected_ids:
        raise ValueError("The two saved artifacts do not share held-out test profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_data, intervention_data, restart_data, context_data = [], [], [], []
    restart_indices = set(np.linspace(
        0, len(selected_ids) - 1, min(restart_profiles, len(selected_ids)), dtype=int
    )) if restart_profiles else set()
    for model, artifact in artifacts.items():
        profiles = load_profiles(artifact, profile_ids=selected_ids)
        saved_metadata = (
            artifact.legacy_artifact.metadata
            if artifact.backend == "legacy" and artifact.legacy_artifact is not None
            else artifact.metadata
        )
        input_columns = tuple(saved_metadata["input_columns"])
        if len(profiles) != len(selected_ids):
            raise ValueError(f"{model}: missing test profiles in the saved dataset")
        for index, profile in enumerate(profiles):
            print(f"{model} profile={profile.profile_id} ({index + 1}/{len(profiles)})", flush=True)
            prediction = predict_profile(
                artifact, profile,
                key=jax.random.PRNGKey(seed + index),
                num_scenarios=num_scenarios,
                ventilation_rollout_mode="eplus_rule",
            )
            horizon_data.extend(horizon_rows(model, prediction))
            intervention_data.extend(intervention_rows(model, profile, prediction))
            context_data.extend(context_rows(model, profile, prediction, input_columns))
            if index in restart_indices:
                restart_data.extend(restart_rows(
                    model, artifact, profile, prediction,
                    seed=seed + index, num_scenarios=num_scenarios,
                ))
    horizon = pd.DataFrame(horizon_data)
    intervention = pd.DataFrame(intervention_data)
    restart = pd.DataFrame(restart_data)
    context = pd.DataFrame(context_data)
    horizon.to_csv(output_dir / "horizon_metrics.csv", index=False)
    intervention.to_csv(output_dir / "intervention_metrics.csv", index=False)
    restart.to_csv(output_dir / "restart_metrics.csv", index=False)
    context.to_csv(output_dir / "context_metrics.csv", index=False)
    summary = _summaries(horizon, intervention, restart, context)
    summary["study"] = {
        "benchmark_dir": str(benchmark_dir),
        "test_profile_ids": selected_ids,
        "num_scenarios": num_scenarios,
        "restart_profiles": restart_profiles,
        "ventilation_rollout_mode": "eplus_rule",
        "note": "Restart resets observed Tin but not the latent HP/buffer state; compare as a drift diagnostic, not a causal estimate.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    _write_dashboard(output_dir / "rollout_failure_dashboard.html", summary)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(
        "output/neural_building_emulator_refactor/calibrated_four_model_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "output/neural_building_emulator_refactor/rollout_failure_diagnosis"))
    parser.add_argument("--max-profiles", type=int, default=20)
    parser.add_argument("--num-scenarios", type=int, default=24)
    parser.add_argument("--restart-profiles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    destination = run_diagnostic(
        args.benchmark_dir, args.output_dir, max_profiles=args.max_profiles,
        num_scenarios=args.num_scenarios, restart_profiles=args.restart_profiles,
        seed=args.seed,
    )
    print(f"saved_diagnostic={destination / 'rollout_failure_dashboard.html'}")


if __name__ == "__main__":
    main()
