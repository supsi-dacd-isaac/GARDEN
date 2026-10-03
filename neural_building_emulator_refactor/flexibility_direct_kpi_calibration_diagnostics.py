"""Diagnose calibration of annual direct-ratio flexibility KPIs.

For each profile and horizon, this script computes the annual direct estimator

    F_H^+ = H sum(delta_P_H | delta_Tset > 0) / sum(delta_Tset | delta_Tset > 0)
    F_H^- = H sum(delta_P_H | delta_Tset < 0) / sum(delta_Tset | delta_Tset < 0)

for the EnergyPlus trace and every linked emulator scenario. Unlike the
regression estimator, this is a ratio of annual directional event totals and
does not fit nuisance-control coefficients.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from neural_building_emulator.flexibility_event_study_kpis import (
    _config_value,
    _load_closed_loop_profiles,
    _resolve_horizons,
    _sample_scenario_traces,
    _saved_profile_ids,
    _select_profiles,
    _setpoint_events,
)
from neural_building_emulator.model_io import load_training_artifact

from .pel_temporal_calibration_diagnostics import (
    DEFAULT_QUANTILES,
    _cdf_sums,
    _rank_intervals,
    _rank_metrics,
    _trim_indices,
)


DIRECTIONS = ("up", "down")


def _direct_f_values(
    setpoint: np.ndarray,
    traces: np.ndarray,
    *,
    horizons_hours: tuple[float, ...],
    horizon_steps: tuple[int, ...],
    min_setpoint_change: float,
    min_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return direct F values [trace, horizon, direction] and event counts."""
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim != 3 or traces.shape[2] < 3:
        raise ValueError("traces must have shape [trace, time, 3]")
    if traces.shape[1] != len(setpoint):
        raise ValueError("trace and setpoint lengths do not match")

    all_events, _ = _setpoint_events(
        np.asarray(setpoint, dtype=np.float64),
        min_setpoint_change=min_setpoint_change,
    )
    power = traces[:, :, 2]
    prefix = np.concatenate(
        [np.zeros((power.shape[0], 1), dtype=np.float64), np.cumsum(power, axis=1)],
        axis=1,
    )
    values = np.full((traces.shape[0], len(horizon_steps), 2), np.nan, dtype=np.float64)
    event_counts = np.zeros(len(horizon_steps), dtype=np.int64)

    for horizon_index, (horizon_hours, steps) in enumerate(
        zip(horizons_hours, horizon_steps)
    ):
        events = all_events[(all_events >= steps) & (all_events + steps <= len(setpoint))]
        event_counts[horizon_index] = len(events)
        if len(events) < min_events:
            continue
        deltas = setpoint[events] - setpoint[events - 1]
        up = deltas > 0.0
        down = deltas < 0.0
        up_denominator = float(np.sum(deltas[up]))
        down_denominator = float(np.sum(deltas[down]))
        if up_denominator <= 1e-12 or down_denominator >= -1e-12:
            continue

        pre = (prefix[:, events] - prefix[:, events - steps]) / np.float64(steps)
        post = (prefix[:, events + steps] - prefix[:, events]) / np.float64(steps)
        response = post - pre
        values[:, horizon_index, 0] = (
            horizon_hours * np.sum(response[:, up], axis=1) / up_denominator
        )
        values[:, horizon_index, 1] = (
            horizon_hours * np.sum(response[:, down], axis=1) / down_denominator
        )
    return values, event_counts


def _curve_metrics(rows: pd.DataFrame) -> dict[str, float]:
    empirical = rows.sort_values("quantile")["empirical_cdf"].to_numpy(dtype=float)
    return _rank_metrics(empirical * float(rows["observation_count"].iloc[0]), int(rows["observation_count"].iloc[0]))


def _write_dashboard(
    path: Path,
    metric_df: pd.DataFrame,
    building_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Upward direct-F reliability",
            "Downward direct-F reliability",
            "Central 90% coverage by horizon",
            "Building misses vs HP duty cycle",
            "Scenario F error vs ensemble atypicality",
            "All-F sensitivity",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.09,
    )
    palette = ("#636efa", "#00cc96", "#ef553b", "#ab63fa")
    for column, direction in ((1, "up"), (2, "down")):
        direction_part = metric_df[metric_df["direction"] == direction]
        for color, (horizon, part) in zip(
            palette, direction_part.groupby("horizon_hours", sort=True)
        ):
            fig.add_trace(
                go.Scatter(
                    x=part["quantile"],
                    y=part["empirical_cdf"],
                    mode="lines+markers",
                    name=f"{direction}, H={horizon:g}h",
                    line=dict(color=color, width=2),
                    legendgroup=f"{direction}_{horizon}",
                ),
                row=1,
                col=column,
            )
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="ideal",
                showlegend=(direction == "up"),
            ),
            row=1,
            col=column,
        )

    coverage = (
        metric_df.groupby(["horizon_hours", "direction"], as_index=False)["central_90_coverage"]
        .first()
    )
    for direction, color in (("up", "#1f77b4"), ("down", "#d62728")):
        part = coverage[coverage["direction"] == direction]
        fig.add_trace(
            go.Bar(
                x=part["horizon_hours"],
                y=part["central_90_coverage"],
                name=direction,
                marker_color=color,
            ),
            row=1,
            col=3,
        )
    fig.add_hline(y=0.9, line_color="black", line_dash="dash", row=1, col=3)

    fig.add_trace(
        go.Scatter(
            x=building_df["truth_pel_positive_fraction"],
            y=building_df["central_90_miss_fraction"],
            mode="markers",
            marker=dict(
                color=building_df["mean_absolute_rank_deviation"],
                colorscale="Viridis",
                colorbar=dict(title="mean |rank-.5|", x=0.30, y=0.18, len=0.30),
                size=8,
            ),
            text=building_df["profile_id"].astype(str),
            hovertemplate=(
                "profile=%{text}<br>HP duty=%{x:.3f}<br>90% misses=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    plotted = trace_df
    if len(plotted) > 25_000:
        plotted = plotted.sample(25_000, random_state=13)
    fig.add_trace(
        go.Scattergl(
            x=plotted["absolute_distance_from_scenario_median"],
            y=plotted["absolute_error"],
            mode="markers",
            marker=dict(color="rgba(99,110,250,0.22)", size=4),
            text=plotted["profile_id"].astype(str),
            hovertemplate=(
                "profile=%{text}<br>distance from median=%{x:.3f}"
                "<br>truth error=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    colors = {
        "all buildings/traces": "#636efa",
        "exclude worst 10% buildings": "#00cc96",
        "trim 5% traces by median distance": "#ab63fa",
        "oracle trim 5% traces by truth error": "#ef553b",
    }
    for label, part in sensitivity_df.groupby("curve", sort=False):
        fig.add_trace(
            go.Scatter(
                x=part["quantile"],
                y=part["empirical_cdf"],
                mode="lines+markers",
                name=label,
                line=dict(color=colors.get(str(label)), width=2),
            ),
            row=2,
            col=3,
        )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="ideal aggregate",
        ),
        row=2,
        col=3,
    )

    fig.update_xaxes(title_text="nominal quantile", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="empirical rank CDF", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="nominal quantile", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="empirical rank CDF", range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text="horizon H [h]", row=1, col=3)
    fig.update_yaxes(title_text="empirical central-90 coverage", range=[0, 1], row=1, col=3)
    fig.update_xaxes(title_text="true Pel active fraction", row=2, col=1)
    fig.update_yaxes(title_text="fraction of 8 F metrics outside central 90%", row=2, col=1)
    fig.update_xaxes(title_text="|scenario F - scenario median F|", row=2, col=2)
    fig.update_yaxes(title_text="|scenario F - simulated F|", row=2, col=2)
    fig.update_xaxes(title_text="nominal quantile", range=[0, 1], row=2, col=3)
    fig.update_yaxes(title_text="empirical rank CDF", range=[0, 1], row=2, col=3)
    fig.update_layout(
        title=(
            "Annual direct-ratio flexibility calibration attribution<br>"
            "<sup>F uses annual directional event totals; no nuisance regression is fitted</sup>"
        ),
        template="plotly_white",
        height=980,
        width=1650,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def run_diagnostics(args: argparse.Namespace) -> None:
    if args.num_scenarios < 3:
        raise ValueError("--num-scenarios must be at least 3")
    if not 0.0 <= args.trim_fraction < 0.5:
        raise ValueError("--trim-fraction must be in [0, 0.5)")

    overrides = None
    if args.prob_hp_emission_mode != "artifact":
        overrides = {"prob_hp_emission_mode": args.prob_hp_emission_mode}
    artifact = load_training_artifact(args.artifact_dir, train_config_overrides=overrides)
    metadata = artifact.metadata
    if str(metadata.get("model_kind")) not in (
        "closed_loop_hp_probabilistic",
        "closed_loop_hp_contracting_probabilistic",
    ):
        raise ValueError("Direct-F calibration requires a probabilistic closed-loop artifact")

    dataset_path = args.dataset or Path(
        _config_value(metadata, "dataset_path", "neural_building_emulator/tessin_results.parquet")
    )
    horizons_hours, horizon_steps = _resolve_horizons(args.horizons_hours, args.dt_hours)
    profile_ids = _select_profiles(
        _saved_profile_ids(metadata, args.profile_source),
        max_profiles=args.max_profiles,
        seed=args.seed,
    )
    input_columns = list(metadata.get("input_columns", ()))
    availability_index = (
        input_columns.index(SPACE_HEATING_AVAILABILITY_COLUMN)
        if SPACE_HEATING_AVAILABILITY_COLUMN in input_columns
        else None
    )
    profiles = _load_closed_loop_profiles(
        dataset_path,
        profile_ids,
        include_space_heating_availability=availability_index is not None,
        include_internal_gains=INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in input_columns,
        include_dhw_request=DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN in input_columns,
        hp_power_area_normalization=str(
            _config_value(metadata, "hp_power_area_normalization", "zone_floor_area")
        ),
        metadata_columns=tuple(metadata.get("metadata_columns", ())) or None,
    )

    rng_key = jax.random.PRNGKey(args.seed)
    records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for profile_index, profile in enumerate(profiles, start=1):
        rng_key, profile_key = jax.random.split(rng_key)
        scenarios = _sample_scenario_traces(
            artifact,
            profile,
            key=profile_key,
            num_scenarios=args.num_scenarios,
            hp_scenario_mode=args.prob_hp_scenario_mode,
            ventilation_rollout_mode=args.ventilation_rollout_mode,
        )
        truth_values, event_counts = _direct_f_values(
            profile.inputs[:, 0],
            profile.targets[np.newaxis, :, :],
            horizons_hours=horizons_hours,
            horizon_steps=horizon_steps,
            min_setpoint_change=args.min_setpoint_change,
            min_events=args.min_events,
        )
        scenario_values, _ = _direct_f_values(
            profile.inputs[:, 0],
            scenarios,
            horizons_hours=horizons_hours,
            horizon_steps=horizon_steps,
            min_setpoint_change=args.min_setpoint_change,
            min_events=args.min_events,
        )
        available = (
            profile.inputs[:, availability_index] > 0.5
            if availability_index is not None
            else np.ones(profile.targets.shape[0], dtype=bool)
        )
        true_duty = float(
            np.mean(profile.targets[available, 2] > args.active_power_threshold_w_m2)
        )

        for horizon_index, horizon in enumerate(horizons_hours):
            for direction_index, direction in enumerate(DIRECTIONS):
                truth = float(truth_values[0, horizon_index, direction_index])
                samples = scenario_values[:, horizon_index, direction_index]
                samples = samples[np.isfinite(samples)]
                if not np.isfinite(truth) or len(samples) < 3:
                    continue
                lower, upper = _rank_intervals(
                    np.asarray([truth]), samples[:, np.newaxis]
                )
                rank = float(0.5 * (lower[0] + upper[0]))
                median = float(np.median(samples))
                scale = float(max(np.std(samples), 1e-8))
                records.append(
                    {
                        "profile_id": int(profile.profile_id),
                        "horizon_hours": float(horizon),
                        "direction": direction,
                        "event_count": int(event_counts[horizon_index]),
                        "truth_f_wh_m2_k": truth,
                        "scenario_mean_f_wh_m2_k": float(np.mean(samples)),
                        "scenario_median_f_wh_m2_k": median,
                        "scenario_std_f_wh_m2_k": scale,
                        "scenario_p05_f_wh_m2_k": float(np.quantile(samples, 0.05)),
                        "scenario_p95_f_wh_m2_k": float(np.quantile(samples, 0.95)),
                        "rank": rank,
                        "central_90_hit": bool(0.05 <= rank <= 0.95),
                        "median_error_wh_m2_k": median - truth,
                        "standardized_median_error": (median - truth) / scale,
                        "truth_pel_positive_fraction": true_duty,
                    }
                )
                for scenario_index, value in enumerate(samples):
                    trace_records.append(
                        {
                            "profile_id": int(profile.profile_id),
                            "scenario_index": scenario_index,
                            "horizon_hours": float(horizon),
                            "direction": direction,
                            "truth_f_wh_m2_k": truth,
                            "scenario_f_wh_m2_k": float(value),
                            "error": float(value - truth),
                            "absolute_error": float(abs(value - truth)),
                            "absolute_distance_from_scenario_median": float(abs(value - median)),
                        }
                    )
        elapsed_minutes = (time.perf_counter() - started) / 60.0
        print(
            f"profile={profile_index}/{len(profiles)} id={profile.profile_id} "
            f"elapsed_minutes={elapsed_minutes:.1f}"
        )

    record_df = pd.DataFrame(records)
    trace_df = pd.DataFrame(trace_records)
    if record_df.empty or trace_df.empty:
        raise ValueError("No valid direct-F estimates were produced")

    metric_rows: list[dict[str, Any]] = []
    for (horizon, direction), part in record_df.groupby(
        ["horizon_hours", "direction"], sort=True
    ):
        ranks = part["rank"].to_numpy(dtype=float)
        for quantile in DEFAULT_QUANTILES:
            empirical = float(np.mean(ranks <= quantile))
            metric_rows.append(
                {
                    "horizon_hours": float(horizon),
                    "direction": str(direction),
                    "quantile": float(quantile),
                    "empirical_cdf": empirical,
                    "calibration_error": empirical - float(quantile),
                    "observation_count": int(len(ranks)),
                }
            )
    metric_df = pd.DataFrame(metric_rows)
    metric_summaries: list[dict[str, Any]] = []
    for (horizon, direction), part in metric_df.groupby(
        ["horizon_hours", "direction"], sort=True
    ):
        metrics = _curve_metrics(part)
        metric_summaries.append(
            {"horizon_hours": horizon, "direction": direction, **metrics}
        )
        mask = (metric_df["horizon_hours"] == horizon) & (
            metric_df["direction"] == direction
        )
        for key, value in metrics.items():
            metric_df.loc[mask, key] = value

    building_df = (
        record_df.groupby("profile_id", as_index=False)
        .agg(
            metric_count=("rank", "size"),
            mean_rank=("rank", "mean"),
            mean_absolute_rank_deviation=("rank", lambda x: float(np.mean(np.abs(x - 0.5)))),
            central_90_hit_fraction=("central_90_hit", "mean"),
            mean_absolute_standardized_median_error=(
                "standardized_median_error", lambda x: float(np.mean(np.abs(x)))
            ),
            truth_pel_positive_fraction=("truth_pel_positive_fraction", "first"),
        )
    )
    building_df["central_90_miss_fraction"] = 1.0 - building_df["central_90_hit_fraction"]

    all_rank_cdf = np.asarray(
        [np.mean(record_df["rank"] <= quantile) for quantile in DEFAULT_QUANTILES]
    )
    worst_count = max(1, int(np.ceil(len(building_df) * 0.10)))
    worst_ids = set(
        building_df.nlargest(worst_count, "mean_absolute_rank_deviation")["profile_id"]
    )
    kept = record_df[~record_df["profile_id"].isin(worst_ids)]
    kept_rank_cdf = np.asarray(
        [np.mean(kept["rank"] <= quantile) for quantile in DEFAULT_QUANTILES]
    )

    trimmed_ranks: list[float] = []
    oracle_ranks: list[float] = []
    for _, part in trace_df.groupby(
        ["profile_id", "horizon_hours", "direction"], sort=False
    ):
        truth = float(part["truth_f_wh_m2_k"].iloc[0])
        samples = part["scenario_f_wh_m2_k"].to_numpy(dtype=float)
        median_distance = part["absolute_distance_from_scenario_median"].to_numpy(dtype=float)
        truth_error = part["absolute_error"].to_numpy(dtype=float)
        for destination, metric in (
            (trimmed_ranks, median_distance),
            (oracle_ranks, truth_error),
        ):
            indices = _trim_indices(metric, args.trim_fraction)
            lower, upper = _rank_intervals(
                np.asarray([truth]), samples[indices, np.newaxis]
            )
            destination.append(float(0.5 * (lower[0] + upper[0])))

    sensitivity_curves = {
        "all buildings/traces": all_rank_cdf,
        "exclude worst 10% buildings": kept_rank_cdf,
        "trim 5% traces by median distance": np.asarray(
            [np.mean(np.asarray(trimmed_ranks) <= q) for q in DEFAULT_QUANTILES]
        ),
        "oracle trim 5% traces by truth error": np.asarray(
            [np.mean(np.asarray(oracle_ranks) <= q) for q in DEFAULT_QUANTILES]
        ),
    }
    sensitivity_rows: list[dict[str, Any]] = []
    for label, empirical in sensitivity_curves.items():
        for quantile, value in zip(DEFAULT_QUANTILES, empirical):
            sensitivity_rows.append(
                {
                    "curve": label,
                    "quantile": float(quantile),
                    "empirical_cdf": float(value),
                    "calibration_error": float(value - quantile),
                    "observation_count": int(len(record_df)),
                }
            )
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "direct_f_calibration_by_building_metric.csv"
    building_path = output_dir / "direct_f_calibration_by_building.csv"
    trace_path = output_dir / "direct_f_calibration_by_sampled_trace.csv"
    curve_path = output_dir / "direct_f_calibration_curves.csv"
    sensitivity_path = output_dir / "direct_f_calibration_sensitivity.csv"
    dashboard_path = output_dir / "direct_f_calibration_attribution.html"
    record_df.to_csv(record_path, index=False)
    building_df.to_csv(building_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    metric_df.to_csv(curve_path, index=False)
    sensitivity_df.to_csv(sensitivity_path, index=False)
    _write_dashboard(dashboard_path, metric_df, building_df, trace_df, sensitivity_df)

    sensitivity_summary = {
        label: _rank_metrics(
            values * len(record_df), len(record_df)
        )
        for label, values in sensitivity_curves.items()
    }
    trace_distance = trace_df["absolute_distance_from_scenario_median"].to_numpy(dtype=float)
    top_count = max(1, int(np.ceil(len(trace_distance) * 0.05)))
    squared = np.sort(trace_distance**2)[::-1]
    summary = {
        "estimator": "direct_ratio_annual_directional_event_totals",
        "artifact_dir": str(args.artifact_dir),
        "dataset_path": str(dataset_path),
        "profile_source": args.profile_source,
        "profile_count": int(building_df.shape[0]),
        "num_scenarios": int(args.num_scenarios),
        "horizons_hours": list(horizons_hours),
        "metric_calibration": metric_summaries,
        "aggregate_sensitivity": sensitivity_summary,
        "building_distribution": {
            "median_central_90_hit_fraction_across_8_metrics": float(
                building_df["central_90_hit_fraction"].median()
            ),
            "fraction_buildings_with_all_metrics_in_central_90": float(
                np.mean(building_df["central_90_hit_fraction"] == 1.0)
            ),
            "fraction_buildings_with_half_or_more_metrics_outside_central_90": float(
                np.mean(building_df["central_90_miss_fraction"] >= 0.5)
            ),
        },
        "sampled_trace_distribution": {
            "median_absolute_f_error": float(trace_df["absolute_error"].median()),
            "p95_absolute_f_error": float(trace_df["absolute_error"].quantile(0.95)),
            "top_5_percent_share_of_squared_distance_from_scenario_median": float(
                np.sum(squared[:top_count]) / max(np.sum(squared), np.finfo(float).eps)
            ),
        },
        "worst_buildings": building_df.nlargest(
            min(10, len(building_df)), "mean_absolute_rank_deviation"
        ).to_dict(orient="records"),
    }
    summary_path = output_dir / "direct_f_calibration_attribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved_dashboard={dashboard_path}")
    print(f"saved_building_metric_rows={record_path}")
    print(f"saved_trace_rows={trace_path}")
    print(f"saved_summary={summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-source", choices=("test", "train", "selected"), default="test")
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--horizons-hours", default="0.5,1,2,3")
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--active-power-threshold-w-m2", type=float, default=0.05)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument(
        "--prob-hp-scenario-mode", choices=("expected", "bernoulli"), default="bernoulli"
    )
    parser.add_argument(
        "--prob-hp-emission-mode",
        choices=("artifact", "bounded", "legacy_lognormal_mean"),
        default="artifact",
    )
    parser.add_argument(
        "--ventilation-rollout-mode", choices=("recorded", "eplus_rule"), default="eplus_rule"
    )
    return parser.parse_args()


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
