"""Attribute Pel temporal-calibration errors to buildings or sampled traces.

This diagnostic reruns the same linked full-profile scenario sampler used by
``flexibility_event_study_kpis`` and decomposes its pointwise Pel_SH ensemble
ranks along two axes:

* per-building rank calibration and ensemble-mean error;
* per-scenario trajectory error, bias, activation rate, and annual energy.

The trimmed curves are diagnostic counterfactuals, not deployable calibration
estimates: one removes traces farthest from the ensemble median, while the
explicitly labelled oracle curve removes traces with the largest truth RMSE.
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
    _sample_scenario_traces,
    _saved_profile_ids,
    _select_profiles,
)
from neural_building_emulator.model_io import load_training_artifact


DEFAULT_QUANTILES = np.asarray(
    (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    dtype=np.float64,
)
HISTOGRAM_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def _rank_intervals(truth: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tie-aware randomized-rank intervals for one output channel."""
    less = np.sum(samples < truth[np.newaxis, :], axis=0)
    equal = np.sum(samples == truth[np.newaxis, :], axis=0)
    denominator = np.float64(samples.shape[0] + 1)
    lower = less.astype(np.float64) / denominator
    upper = (less + equal + 1).astype(np.float64) / denominator
    return lower, upper


def _cdf_sums(
    lower: np.ndarray,
    upper: np.ndarray,
    quantiles: np.ndarray = DEFAULT_QUANTILES,
) -> np.ndarray:
    width = upper - lower
    conditional = np.clip(
        (quantiles[:, np.newaxis] - lower[np.newaxis, :]) / width[np.newaxis, :],
        0.0,
        1.0,
    )
    return np.sum(conditional, axis=1)


def _histogram_sums(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    width = upper - lower
    bin_lower = HISTOGRAM_EDGES[:-1, np.newaxis]
    bin_upper = HISTOGRAM_EDGES[1:, np.newaxis]
    overlap = np.maximum(
        0.0,
        np.minimum(bin_upper, upper[np.newaxis, :])
        - np.maximum(bin_lower, lower[np.newaxis, :]),
    )
    return np.sum(overlap / width[np.newaxis, :], axis=1)


def _rank_metrics(cdf_sum: np.ndarray, count: int) -> dict[str, float]:
    empirical = cdf_sum / np.float64(count)
    error = empirical - DEFAULT_QUANTILES
    lower = float(empirical[np.flatnonzero(DEFAULT_QUANTILES == 0.05)[0]])
    upper = float(empirical[np.flatnonzero(DEFAULT_QUANTILES == 0.95)[0]])
    return {
        "central_90_coverage": upper - lower,
        "mean_abs_calibration_error": float(np.mean(np.abs(error))),
        "max_abs_calibration_error": float(np.max(np.abs(error))),
        "cdf_at_0_05": lower,
        "cdf_at_0_50": float(empirical[np.flatnonzero(DEFAULT_QUANTILES == 0.5)[0]]),
        "cdf_at_0_95": upper,
    }


def _rank_midpoint(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return 0.5 * (lower + upper)


def _finite_pel(
    truth: np.ndarray,
    scenarios: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        available
        & np.isfinite(truth)
        & np.all(np.isfinite(scenarios), axis=0)
    )
    if not np.any(valid):
        raise ValueError("No finite Pel observations remain in the space-heating-available scope")
    return truth[valid].astype(np.float64), scenarios[:, valid].astype(np.float64)


def _trim_indices(metric: np.ndarray, trim_fraction: float) -> np.ndarray:
    keep_count = max(2, int(np.ceil(metric.size * (1.0 - trim_fraction))))
    return np.argsort(metric)[:keep_count]


def _pooled_curve_rows(
    label: str,
    cdf_sum: np.ndarray,
    count: int,
) -> list[dict[str, Any]]:
    empirical = cdf_sum / np.float64(count)
    return [
        {
            "curve": label,
            "quantile": float(quantile),
            "empirical_cdf": float(value),
            "calibration_error": float(value - quantile),
            "observation_count": int(count),
        }
        for quantile, value in zip(DEFAULT_QUANTILES, empirical)
    ]


def _write_dashboard(
    path: Path,
    building_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    histogram_df: pd.DataFrame,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Central 90% coverage by building",
            "Rank location vs ensemble-mean bias",
            "Building calibration error vs true annual energy",
            "Sampled-trace error vs ensemble atypicality",
            "Observed annual-energy ranks",
            "Pooled reliability sensitivity",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.09,
    )

    ordered = building_df.sort_values("central_90_coverage").reset_index(drop=True)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(ordered)),
            y=ordered["central_90_coverage"],
            mode="markers",
            marker=dict(
                color=ordered["mean_abs_calibration_error"],
                colorscale="Viridis",
                colorbar=dict(title="building<br>calibration<br>error", x=0.30, y=0.82, len=0.30),
                size=7,
            ),
            text=ordered["profile_id"].astype(str),
            hovertemplate="profile=%{text}<br>coverage=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.9, line_dash="dash", line_color="black", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=building_df["mean_rank"],
            y=building_df["ensemble_mean_bias_w_m2"],
            mode="markers",
            marker=dict(
                color=building_df["mean_abs_calibration_error"],
                colorscale="Viridis",
                showscale=False,
                size=8,
            ),
            text=building_df["profile_id"].astype(str),
            hovertemplate=(
                "profile=%{text}<br>mean rank=%{x:.3f}"
                "<br>mean-prediction bias=%{y:.2f} W/m2<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", row=1, col=2)
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=building_df["truth_energy_wh_m2"],
            y=building_df["mean_abs_calibration_error"],
            mode="markers",
            marker=dict(color=building_df["central_90_coverage"], colorscale="RdYlGn", size=8),
            text=building_df["profile_id"].astype(str),
            hovertemplate=(
                "profile=%{text}<br>true energy=%{x:.1f} Wh/m2"
                "<br>calibration error=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    plotted_traces = trace_df
    if len(plotted_traces) > 20_000:
        plotted_traces = plotted_traces.sample(20_000, random_state=13)
    fig.add_trace(
        go.Scattergl(
            x=plotted_traces["rmse_from_ensemble_median_w_m2"],
            y=plotted_traces["rmse_to_truth_w_m2"],
            mode="markers",
            marker=dict(color="rgba(99,110,250,0.25)", size=4),
            text=plotted_traces["profile_id"].astype(str),
            hovertemplate=(
                "profile=%{text}<br>distance from median=%{x:.2f}"
                "<br>truth RMSE=%{y:.2f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=building_df["truth_energy_rank"],
            xbins=dict(start=0.0, end=1.0, size=0.1),
            histnorm="probability",
            marker_color="#ef553b",
            name="energy ranks",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=0.1, line_dash="dash", line_color="black", row=2, col=2)

    colors = {
        "all traces/buildings": "#636efa",
        "exclude worst 10% buildings": "#00cc96",
        "trim 5% traces by median distance": "#ab63fa",
        "oracle trim 5% traces by truth RMSE": "#ef553b",
    }
    for label, part in curve_df.groupby("curve", sort=False):
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
            name="ideal",
        ),
        row=2,
        col=3,
    )

    fig.update_xaxes(title_text="buildings sorted by coverage", row=1, col=1)
    fig.update_yaxes(title_text="empirical coverage", row=1, col=1)
    fig.update_xaxes(title_text="mean randomized rank", row=1, col=2)
    fig.update_yaxes(title_text="predicted - observed [W/m2]", row=1, col=2)
    fig.update_xaxes(title_text="true annual SH electricity [Wh/m2]", row=1, col=3)
    fig.update_yaxes(title_text="mean absolute calibration error", row=1, col=3)
    fig.update_xaxes(title_text="trace-to-median RMSE [W/m2]", row=2, col=1)
    fig.update_yaxes(title_text="trace-to-truth RMSE [W/m2]", row=2, col=1)
    fig.update_xaxes(title_text="truth rank in scenario energy distribution", row=2, col=2)
    fig.update_yaxes(title_text="building fraction", row=2, col=2)
    fig.update_xaxes(title_text="nominal quantile", range=[0.0, 1.0], row=2, col=3)
    fig.update_yaxes(title_text="empirical rank CDF", range=[0.0, 1.0], row=2, col=3)
    fig.update_layout(
        title=(
            "Pel_SH temporal-calibration attribution<br>"
            "<sup>Pointwise ranks use space-heating-available timestamps only; "
            "trimmed curves are diagnostic counterfactuals</sup>"
        ),
        template="plotly_white",
        height=980,
        width=1650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
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
        raise ValueError("Pel calibration attribution requires a probabilistic closed-loop artifact")

    dataset_path = args.dataset or Path(
        _config_value(metadata, "dataset_path", "neural_building_emulator/tessin_results.parquet")
    )
    candidate_ids = _saved_profile_ids(metadata, args.profile_source)
    profile_ids = _select_profiles(candidate_ids, max_profiles=args.max_profiles, seed=args.seed)
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng_key = jax.random.PRNGKey(args.seed)
    building_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    per_building_cdf: list[np.ndarray] = []
    per_building_hist: list[np.ndarray] = []
    per_building_counts: list[int] = []
    trimmed_cdf = np.zeros_like(DEFAULT_QUANTILES)
    trimmed_count = 0
    oracle_cdf = np.zeros_like(DEFAULT_QUANTILES)
    oracle_count = 0
    started = time.perf_counter()

    for index, profile in enumerate(profiles, start=1):
        rng_key, profile_key = jax.random.split(rng_key)
        scenarios = _sample_scenario_traces(
            artifact,
            profile,
            key=profile_key,
            num_scenarios=args.num_scenarios,
            hp_scenario_mode=args.prob_hp_scenario_mode,
            ventilation_rollout_mode=args.ventilation_rollout_mode,
        )
        available = (
            profile.inputs[:, availability_index] > np.float32(0.5)
            if availability_index is not None
            else np.ones(profile.targets.shape[0], dtype=bool)
        )
        truth, samples = _finite_pel(profile.targets[:, 2], scenarios[:, :, 2], available)
        lower, upper = _rank_intervals(truth, samples)
        cdf_sum = _cdf_sums(lower, upper)
        histogram_sum = _histogram_sums(lower, upper)
        count = int(truth.size)
        rank_metric = _rank_metrics(cdf_sum, count)
        rank_midpoint = _rank_midpoint(lower, upper)

        ensemble_mean = np.mean(samples, axis=0)
        ensemble_median = np.median(samples, axis=0)
        scenario_errors = samples - truth[np.newaxis, :]
        scenario_rmse = np.sqrt(np.mean(scenario_errors**2, axis=1))
        scenario_mae = np.mean(np.abs(scenario_errors), axis=1)
        scenario_bias = np.mean(scenario_errors, axis=1)
        median_distance = np.sqrt(np.mean((samples - ensemble_median[np.newaxis, :]) ** 2, axis=1))
        scenario_energy = args.dt_hours * np.sum(samples, axis=1)
        truth_energy = float(args.dt_hours * np.sum(truth))
        truth_energy_lower, truth_energy_upper = _rank_intervals(
            np.asarray([truth_energy]), scenario_energy[:, np.newaxis]
        )

        trim_by_median = _trim_indices(median_distance, args.trim_fraction)
        trim_lower, trim_upper = _rank_intervals(truth, samples[trim_by_median])
        trimmed_cdf += _cdf_sums(trim_lower, trim_upper)
        trimmed_count += count

        trim_by_truth = _trim_indices(scenario_rmse, args.trim_fraction)
        oracle_lower, oracle_upper = _rank_intervals(truth, samples[trim_by_truth])
        oracle_cdf += _cdf_sums(oracle_lower, oracle_upper)
        oracle_count += count

        building_rows.append(
            {
                "profile_id": int(profile.profile_id),
                "observation_count": count,
                **rank_metric,
                "mean_rank": float(np.mean(rank_midpoint)),
                "low_rank_fraction": float(np.mean(rank_midpoint < 0.05)),
                "high_rank_fraction": float(np.mean(rank_midpoint > 0.95)),
                "truth_mean_w_m2": float(np.mean(truth)),
                "truth_positive_fraction": float(np.mean(truth > args.active_power_threshold_w_m2)),
                "truth_energy_wh_m2": truth_energy,
                "truth_energy_rank": float(0.5 * (truth_energy_lower[0] + truth_energy_upper[0])),
                "ensemble_mean_rmse_w_m2": float(np.sqrt(np.mean((ensemble_mean - truth) ** 2))),
                "ensemble_mean_mae_w_m2": float(np.mean(np.abs(ensemble_mean - truth))),
                "ensemble_mean_bias_w_m2": float(np.mean(ensemble_mean - truth)),
                "mean_pointwise_scenario_std_w_m2": float(np.mean(np.std(samples, axis=0))),
                "scenario_rmse_p05_w_m2": float(np.quantile(scenario_rmse, 0.05)),
                "scenario_rmse_median_w_m2": float(np.median(scenario_rmse)),
                "scenario_rmse_p95_w_m2": float(np.quantile(scenario_rmse, 0.95)),
                "scenario_rmse_max_w_m2": float(np.max(scenario_rmse)),
                "scenario_energy_p05_wh_m2": float(np.quantile(scenario_energy, 0.05)),
                "scenario_energy_median_wh_m2": float(np.median(scenario_energy)),
                "scenario_energy_p95_wh_m2": float(np.quantile(scenario_energy, 0.95)),
            }
        )
        for scenario_index in range(samples.shape[0]):
            trace_rows.append(
                {
                    "profile_id": int(profile.profile_id),
                    "scenario_index": scenario_index,
                    "rmse_to_truth_w_m2": float(scenario_rmse[scenario_index]),
                    "mae_to_truth_w_m2": float(scenario_mae[scenario_index]),
                    "bias_to_truth_w_m2": float(scenario_bias[scenario_index]),
                    "rmse_from_ensemble_median_w_m2": float(median_distance[scenario_index]),
                    "energy_wh_m2": float(scenario_energy[scenario_index]),
                    "energy_error_wh_m2": float(scenario_energy[scenario_index] - truth_energy),
                    "active_fraction": float(
                        np.mean(samples[scenario_index] > args.active_power_threshold_w_m2)
                    ),
                }
            )
        per_building_cdf.append(cdf_sum)
        per_building_hist.append(histogram_sum)
        per_building_counts.append(count)
        elapsed_minutes = (time.perf_counter() - started) / 60.0
        print(
            f"profile={index}/{len(profiles)} id={profile.profile_id} "
            f"pel_coverage90={rank_metric['central_90_coverage']:.3f} "
            f"pel_calibration_mae={rank_metric['mean_abs_calibration_error']:.3f} "
            f"elapsed_minutes={elapsed_minutes:.1f}"
        )

    building_df = pd.DataFrame(building_rows)
    trace_df = pd.DataFrame(trace_rows)
    cdf_matrix = np.stack(per_building_cdf)
    histogram_matrix = np.stack(per_building_hist)
    counts = np.asarray(per_building_counts, dtype=np.int64)
    all_cdf = np.sum(cdf_matrix, axis=0)
    all_histogram = np.sum(histogram_matrix, axis=0)
    all_count = int(np.sum(counts))

    worst_count = max(1, int(np.ceil(len(building_df) * 0.10)))
    keep_buildings = np.argsort(building_df["mean_abs_calibration_error"].to_numpy())[:-worst_count]
    kept_cdf = np.sum(cdf_matrix[keep_buildings], axis=0)
    kept_count = int(np.sum(counts[keep_buildings]))

    curve_rows: list[dict[str, Any]] = []
    curve_rows.extend(_pooled_curve_rows("all traces/buildings", all_cdf, all_count))
    curve_rows.extend(
        _pooled_curve_rows("exclude worst 10% buildings", kept_cdf, kept_count)
    )
    curve_rows.extend(
        _pooled_curve_rows("trim 5% traces by median distance", trimmed_cdf, trimmed_count)
    )
    curve_rows.extend(
        _pooled_curve_rows("oracle trim 5% traces by truth RMSE", oracle_cdf, oracle_count)
    )
    curve_df = pd.DataFrame(curve_rows)

    histogram_df = pd.DataFrame(
        {
            "bin_left": HISTOGRAM_EDGES[:-1],
            "bin_right": HISTOGRAM_EDGES[1:],
            "probability": all_histogram / np.float64(all_count),
            "expected_probability": 0.1,
        }
    )

    building_path = output_dir / "pel_calibration_by_building.csv"
    trace_path = output_dir / "pel_calibration_by_sampled_trace.csv"
    curve_path = output_dir / "pel_calibration_sensitivity.csv"
    histogram_path = output_dir / "pel_rank_histogram.csv"
    dashboard_path = output_dir / "pel_temporal_calibration_attribution.html"
    building_df.to_csv(building_path, index=False)
    trace_df.to_csv(trace_path, index=False)
    curve_df.to_csv(curve_path, index=False)
    histogram_df.to_csv(histogram_path, index=False)
    _write_dashboard(dashboard_path, building_df, trace_df, curve_df, histogram_df)

    curve_summary = {
        label: _rank_metrics(
            part["empirical_cdf"].to_numpy(dtype=np.float64)
            * int(part["observation_count"].iloc[0]),
            int(part["observation_count"].iloc[0]),
        )
        for label, part in curve_df.groupby("curve", sort=False)
    }
    ordered_error = np.sort(building_df["mean_abs_calibration_error"].to_numpy())[::-1]
    top_decile_share = float(
        np.sum(ordered_error[:worst_count]) / max(np.sum(ordered_error), np.finfo(float).eps)
    )
    trace_distance = trace_df["rmse_from_ensemble_median_w_m2"].to_numpy(dtype=float)
    top_trace_count = max(1, int(np.ceil(trace_distance.size * 0.05)))
    sorted_distance_squared = np.sort(trace_distance**2)[::-1]
    summary = {
        "artifact_dir": str(args.artifact_dir),
        "dataset_path": str(dataset_path),
        "profile_source": args.profile_source,
        "profile_count": int(len(building_df)),
        "num_scenarios": int(args.num_scenarios),
        "prob_hp_scenario_mode": args.prob_hp_scenario_mode,
        "prob_hp_emission_mode": str(
            _config_value(metadata, "prob_hp_emission_mode", "legacy_lognormal_mean")
            if args.prob_hp_emission_mode == "artifact"
            else args.prob_hp_emission_mode
        ),
        "ventilation_rollout_mode": args.ventilation_rollout_mode,
        "curve_metrics": curve_summary,
        "building_distribution": {
            "median_central_90_coverage": float(building_df["central_90_coverage"].median()),
            "fraction_central_90_below_0_85": float(
                np.mean(building_df["central_90_coverage"] < 0.85)
            ),
            "fraction_central_90_above_0_95": float(
                np.mean(building_df["central_90_coverage"] > 0.95)
            ),
            "median_calibration_mae": float(
                building_df["mean_abs_calibration_error"].median()
            ),
            "top_10_percent_share_of_building_calibration_mae": top_decile_share,
        },
        "sampled_trace_distribution": {
            "median_rmse_to_truth_w_m2": float(trace_df["rmse_to_truth_w_m2"].median()),
            "p95_rmse_to_truth_w_m2": float(trace_df["rmse_to_truth_w_m2"].quantile(0.95)),
            "max_rmse_to_truth_w_m2": float(trace_df["rmse_to_truth_w_m2"].max()),
            "top_5_percent_share_of_trace_to_median_squared_distance": float(
                np.sum(sorted_distance_squared[:top_trace_count])
                / max(np.sum(sorted_distance_squared), np.finfo(float).eps)
            ),
        },
        "worst_buildings": building_df.nlargest(
            min(10, len(building_df)), "mean_abs_calibration_error"
        )[
            [
                "profile_id",
                "mean_abs_calibration_error",
                "central_90_coverage",
                "mean_rank",
                "ensemble_mean_bias_w_m2",
                "truth_energy_wh_m2",
                "truth_energy_rank",
            ]
        ].to_dict(orient="records"),
    }
    summary_path = output_dir / "pel_temporal_calibration_attribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"saved_dashboard={dashboard_path}")
    print(f"saved_building_metrics={building_path}")
    print(f"saved_trace_metrics={trace_path}")
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
    parser.add_argument("--dt-hours", type=float, default=0.25)
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
