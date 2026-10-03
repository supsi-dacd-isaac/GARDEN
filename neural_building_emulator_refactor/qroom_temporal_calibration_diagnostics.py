"""Decompose Qroom calibration into occurrence, positive magnitude, and persistence.

The ordinary ensemble-rank diagnostic mixes the atom at zero with the
continuous positive Qroom distribution.  This diagnostic instead evaluates:

* occurrence: P(Qroom > threshold), using a binary reliability diagram;
* magnitude: Qroom | Qroom > threshold, using tie-aware ensemble ranks;
* persistence: active-state transition probabilities and run durations.

Linked annual scenarios are generated with the same rollout helper used by
``flexibility_event_study_kpis``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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

from .pel_temporal_calibration_diagnostics import (
    DEFAULT_QUANTILES,
    HISTOGRAM_EDGES,
)


RELIABILITY_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


@dataclass
class BinaryReliabilityAccumulator:
    count: np.ndarray
    predicted_sum: np.ndarray
    observed_sum: np.ndarray
    brier_sum: float = 0.0
    log_loss_sum: float = 0.0
    total_count: int = 0

    @classmethod
    def create(cls) -> "BinaryReliabilityAccumulator":
        bins = len(RELIABILITY_EDGES) - 1
        return cls(
            count=np.zeros(bins, dtype=np.int64),
            predicted_sum=np.zeros(bins, dtype=np.float64),
            observed_sum=np.zeros(bins, dtype=np.float64),
        )

    def add(self, probability: np.ndarray, observed: np.ndarray) -> None:
        probability = np.asarray(probability, dtype=np.float64)
        observed = np.asarray(observed, dtype=np.float64)
        valid = np.isfinite(probability) & np.isfinite(observed)
        probability = np.clip(probability[valid], 0.0, 1.0)
        observed = observed[valid]
        if probability.size == 0:
            return
        bins = np.minimum(
            np.searchsorted(RELIABILITY_EDGES, probability, side="right") - 1,
            len(RELIABILITY_EDGES) - 2,
        )
        bins = np.maximum(bins, 0)
        self.count += np.bincount(bins, minlength=len(self.count))
        self.predicted_sum += np.bincount(
            bins, weights=probability, minlength=len(self.count)
        )
        self.observed_sum += np.bincount(
            bins, weights=observed, minlength=len(self.count)
        )
        self.brier_sum += float(np.sum((probability - observed) ** 2))
        clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
        self.log_loss_sum += float(
            -np.sum(observed * np.log(clipped) + (1.0 - observed) * np.log1p(-clipped))
        )
        self.total_count += int(probability.size)

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, count in enumerate(self.count):
            if count == 0:
                continue
            rows.append(
                {
                    "bin_left": float(RELIABILITY_EDGES[index]),
                    "bin_right": float(RELIABILITY_EDGES[index + 1]),
                    "predicted_probability": float(self.predicted_sum[index] / count),
                    "observed_frequency": float(self.observed_sum[index] / count),
                    "observation_count": int(count),
                }
            )
        return rows

    def metrics(self) -> dict[str, float]:
        if self.total_count == 0:
            return {
                "brier_score": float("nan"),
                "log_loss": float("nan"),
                "expected_calibration_error": float("nan"),
            }
        populated = self.count > 0
        mean_predicted = self.predicted_sum[populated] / self.count[populated]
        observed_frequency = self.observed_sum[populated] / self.count[populated]
        ece = np.sum(
            self.count[populated] * np.abs(mean_predicted - observed_frequency)
        ) / self.total_count
        return {
            "brier_score": float(self.brier_sum / self.total_count),
            "log_loss": float(self.log_loss_sum / self.total_count),
            "expected_calibration_error": float(ece),
        }


@dataclass
class ConditionalRankAccumulator:
    cdf_sum: np.ndarray
    histogram_sum: np.ndarray
    count: int = 0
    insufficient_scenario_count: int = 0

    @classmethod
    def create(cls) -> "ConditionalRankAccumulator":
        return cls(
            cdf_sum=np.zeros_like(DEFAULT_QUANTILES),
            histogram_sum=np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.float64),
        )

    def add(self, truth: np.ndarray, scenarios: np.ndarray, threshold: float) -> None:
        truth = np.asarray(truth, dtype=np.float64)
        scenarios = np.asarray(scenarios, dtype=np.float64)
        for time_index in np.flatnonzero(truth > threshold):
            active_samples = scenarios[:, time_index]
            active_samples = active_samples[
                np.isfinite(active_samples) & (active_samples > threshold)
            ]
            if active_samples.size == 0:
                self.insufficient_scenario_count += 1
                continue
            observed = truth[time_index]
            less = int(np.sum(active_samples < observed))
            equal = int(np.sum(active_samples == observed))
            denominator = float(active_samples.size + 1)
            lower = less / denominator
            upper = (less + equal + 1) / denominator
            width = upper - lower
            self.cdf_sum += np.clip((DEFAULT_QUANTILES - lower) / width, 0.0, 1.0)
            overlap = np.maximum(
                0.0,
                np.minimum(HISTOGRAM_EDGES[1:], upper)
                - np.maximum(HISTOGRAM_EDGES[:-1], lower),
            )
            self.histogram_sum += overlap / width
            self.count += 1

    def metrics(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "mean_abs_calibration_error": float("nan"),
                "max_abs_calibration_error": float("nan"),
                "central_90_coverage": float("nan"),
            }
        empirical = self.cdf_sum / self.count
        error = empirical - DEFAULT_QUANTILES
        lower = float(empirical[np.flatnonzero(DEFAULT_QUANTILES == 0.05)[0]])
        upper = float(empirical[np.flatnonzero(DEFAULT_QUANTILES == 0.95)[0]])
        return {
            "mean_abs_calibration_error": float(np.mean(np.abs(error))),
            "max_abs_calibration_error": float(np.max(np.abs(error))),
            "central_90_coverage": upper - lower,
        }


def _transition_probability(active: np.ndarray, previous: bool) -> float:
    active = np.asarray(active, dtype=bool)
    if active.size < 2:
        return float("nan")
    source = active[:-1] == previous
    if not np.any(source):
        return float("nan")
    return float(np.mean(active[1:][source]))


def _positive_run_lengths(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    padded = np.concatenate(([False], active, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return (stops - starts).astype(np.float64)


def _conditional_mean_prediction(
    scenarios: np.ndarray,
    truth_active: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    truth_indices: list[int] = []
    for time_index in np.flatnonzero(truth_active):
        active = scenarios[:, time_index]
        active = active[np.isfinite(active) & (active > threshold)]
        if active.size:
            values.append(float(np.mean(active)))
            truth_indices.append(int(time_index))
    return np.asarray(values), np.asarray(truth_indices, dtype=np.int64)


def _write_dashboard(
    path: Path,
    reliability_df: pd.DataFrame,
    magnitude_reliability_df: pd.DataFrame,
    magnitude_histogram_df: pd.DataFrame,
    building_df: pd.DataFrame,
    *,
    threshold: float,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Qroom on/off reliability",
            "Positive-Qroom magnitude reliability",
            "Positive-Qroom rank histogram",
            "Occurrence calibration by building",
            "Active persistence",
            "Positive-Qroom magnitude error",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.09,
    )

    fig.add_trace(
        go.Scatter(
            x=reliability_df["predicted_probability"],
            y=reliability_df["observed_frequency"],
            mode="lines+markers",
            marker=dict(
                size=np.clip(np.sqrt(reliability_df["observation_count"]) / 3.0, 6, 22),
                color="#2e7d32",
            ),
            name="Q active",
            customdata=reliability_df["observation_count"],
            hovertemplate=(
                "predicted=%{x:.3f}<br>observed=%{y:.3f}"
                "<br>n=%{customdata}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="ideal",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=magnitude_reliability_df["quantile"],
            y=magnitude_reliability_df["empirical_cdf"],
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2),
            name="Q magnitude | active",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=magnitude_histogram_df["bin_center"],
            y=magnitude_histogram_df["probability"],
            width=magnitude_histogram_df["bin_right"]
            - magnitude_histogram_df["bin_left"],
            marker_color="#ff7f0e",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_hline(y=0.1, line_color="black", line_dash="dash", row=1, col=3)

    fig.add_trace(
        go.Scatter(
            x=building_df["truth_active_fraction"],
            y=building_df["predicted_active_fraction"],
            mode="markers",
            marker=dict(
                color=building_df["occurrence_brier_score"],
                colorscale="Viridis",
                colorbar=dict(title="Brier", x=0.30, y=0.18, len=0.28),
                size=8,
            ),
            text=building_df["profile_id"].astype(str),
            hovertemplate=(
                "profile=%{text}<br>true active=%{x:.3f}"
                "<br>predicted active=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=building_df["truth_p_on_given_on"],
            y=building_df["scenario_mean_p_on_given_on"],
            mode="markers",
            marker=dict(color="#9467bd", size=8),
            text=building_df["profile_id"].astype(str),
            name="P(on|on)",
            hovertemplate="profile=%{text}<br>true=%{x:.3f}<br>scenario=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=building_df["truth_p_on_given_off"],
            y=building_df["scenario_mean_p_on_given_off"],
            mode="markers",
            marker=dict(color="#17becf", size=8),
            text=building_df["profile_id"].astype(str),
            name="P(on|off)",
            hovertemplate="profile=%{text}<br>true=%{x:.3f}<br>scenario=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=building_df["truth_active_mean_w_m2"],
            y=building_df["conditional_mean_prediction_w_m2"],
            mode="markers",
            marker=dict(
                color=building_df["conditional_magnitude_bias_w_m2"],
                colorscale="RdBu",
                colorbar=dict(title="bias<br>W/m2", x=1.02, y=0.18, len=0.28),
                size=8,
            ),
            text=building_df["profile_id"].astype(str),
            hovertemplate=(
                "profile=%{text}<br>true active mean=%{x:.2f}"
                "<br>predicted active mean=%{y:.2f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    finite_magnitude = np.concatenate(
        [
            building_df["truth_active_mean_w_m2"].to_numpy(dtype=float),
            building_df["conditional_mean_prediction_w_m2"].to_numpy(dtype=float),
        ]
    )
    finite_magnitude = finite_magnitude[np.isfinite(finite_magnitude)]
    magnitude_limit = float(np.max(finite_magnitude)) if finite_magnitude.size else 1.0
    fig.add_trace(
        go.Scatter(
            x=[0.0, magnitude_limit],
            y=[0.0, magnitude_limit],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    fig.update_xaxes(title_text="predicted P(Q > threshold)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="observed active frequency", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="nominal conditional quantile", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="empirical rank CDF", range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text="conditional ensemble rank", range=[0, 1], row=1, col=3)
    fig.update_yaxes(title_text="probability", row=1, col=3)
    fig.update_xaxes(title_text="true active fraction", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="mean predicted probability", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="true transition probability", range=[0, 1], row=2, col=2)
    fig.update_yaxes(title_text="scenario mean transition probability", range=[0, 1], row=2, col=2)
    fig.update_xaxes(title_text="true E[Q | active] [W/m2]", row=2, col=3)
    fig.update_yaxes(title_text="predicted E[Q | active] [W/m2]", row=2, col=3)
    fig.update_layout(
        title=(
            "Qroom calibration decomposition<br>"
            f"<sup>active means Qroom &gt; {threshold:g} W/m2; "
            "space-heating-available timestamps only</sup>"
        ),
        template="plotly_white",
        height=980,
        width=1650,
        bargap=0.04,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def run_diagnostics(args: argparse.Namespace) -> None:
    if args.num_scenarios < 3:
        raise ValueError("--num-scenarios must be at least 3")
    if args.qroom_threshold_w_m2 < 0.0:
        raise ValueError("--qroom-threshold-w-m2 must be non-negative")

    overrides = None
    if args.prob_hp_emission_mode != "artifact":
        overrides = {"prob_hp_emission_mode": args.prob_hp_emission_mode}
    artifact = load_training_artifact(args.artifact_dir, train_config_overrides=overrides)
    metadata = artifact.metadata
    if str(metadata.get("model_kind")) not in (
        "closed_loop_hp_probabilistic",
        "closed_loop_hp_contracting_probabilistic",
    ):
        raise ValueError("Qroom calibration requires a probabilistic closed-loop artifact")

    dataset_path = args.dataset or Path(
        _config_value(
            metadata,
            "dataset_path",
            "neural_building_emulator/tessin_results.parquet",
        )
    )
    candidate_ids = _saved_profile_ids(metadata, args.profile_source)
    profile_ids = _select_profiles(
        candidate_ids, max_profiles=args.max_profiles, seed=args.seed
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

    occurrence = BinaryReliabilityAccumulator.create()
    conditional_magnitude = ConditionalRankAccumulator.create()
    building_rows: list[dict[str, Any]] = []
    rng_key = jax.random.PRNGKey(args.seed)
    started = time.perf_counter()
    threshold = float(args.qroom_threshold_w_m2)

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
        truth = np.asarray(profile.targets[:, 1], dtype=np.float64)
        samples = np.asarray(scenarios[:, :, 1], dtype=np.float64)
        available = (
            profile.inputs[:, availability_index] > np.float32(0.5)
            if availability_index is not None
            else np.ones(truth.shape, dtype=bool)
        )
        valid = available & np.isfinite(truth) & np.all(np.isfinite(samples), axis=0)
        truth = truth[valid]
        samples = samples[:, valid]
        truth_active = truth > threshold
        sample_active = samples > threshold
        probability = np.mean(sample_active, axis=0)

        profile_occurrence = BinaryReliabilityAccumulator.create()
        profile_occurrence.add(probability, truth_active)
        occurrence.add(probability, truth_active)
        profile_magnitude = ConditionalRankAccumulator.create()
        profile_magnitude.add(truth, samples, threshold)
        conditional_magnitude.add(truth, samples, threshold)

        conditional_mean, conditional_indices = _conditional_mean_prediction(
            samples, truth_active, threshold
        )
        conditional_truth = truth[conditional_indices]
        conditional_errors = conditional_mean - conditional_truth
        truth_runs = _positive_run_lengths(truth_active)
        scenario_runs = [
            _positive_run_lengths(sample_active[scenario_index])
            for scenario_index in range(sample_active.shape[0])
        ]
        scenario_mean_run_steps = np.asarray(
            [np.mean(runs) if runs.size else 0.0 for runs in scenario_runs]
        )
        scenario_p_on_on = np.asarray(
            [_transition_probability(active, True) for active in sample_active]
        )
        scenario_p_on_off = np.asarray(
            [_transition_probability(active, False) for active in sample_active]
        )
        building_rows.append(
            {
                "profile_id": int(profile.profile_id),
                "observation_count": int(truth.size),
                "truth_active_fraction": float(np.mean(truth_active)),
                "predicted_active_fraction": float(np.mean(probability)),
                "active_fraction_bias": float(np.mean(probability) - np.mean(truth_active)),
                **{
                    f"occurrence_{key}": value
                    for key, value in profile_occurrence.metrics().items()
                },
                **{
                    f"conditional_magnitude_{key}": value
                    for key, value in profile_magnitude.metrics().items()
                },
                "conditional_magnitude_observation_count": int(profile_magnitude.count),
                "conditional_no_active_scenario_count": int(
                    profile_magnitude.insufficient_scenario_count
                ),
                "truth_active_mean_w_m2": float(np.mean(truth[truth_active]))
                if np.any(truth_active)
                else float("nan"),
                "conditional_mean_prediction_w_m2": float(np.mean(conditional_mean))
                if conditional_mean.size
                else float("nan"),
                "conditional_magnitude_rmse_w_m2": float(
                    np.sqrt(np.mean(conditional_errors**2))
                )
                if conditional_errors.size
                else float("nan"),
                "conditional_magnitude_bias_w_m2": float(np.mean(conditional_errors))
                if conditional_errors.size
                else float("nan"),
                "truth_p_on_given_on": _transition_probability(truth_active, True),
                "scenario_mean_p_on_given_on": float(np.nanmean(scenario_p_on_on)),
                "truth_p_on_given_off": _transition_probability(truth_active, False),
                "scenario_mean_p_on_given_off": float(np.nanmean(scenario_p_on_off)),
                "truth_mean_active_run_hours": float(
                    args.dt_hours * np.mean(truth_runs)
                )
                if truth_runs.size
                else 0.0,
                "scenario_mean_active_run_hours": float(
                    args.dt_hours * np.mean(scenario_mean_run_steps)
                ),
            }
        )
        elapsed_minutes = (time.perf_counter() - started) / 60.0
        print(
            f"profile={index}/{len(profiles)} id={profile.profile_id} "
            f"q_active_true={np.mean(truth_active):.3f} "
            f"q_active_pred={np.mean(probability):.3f} "
            f"brier={profile_occurrence.metrics()['brier_score']:.4f} "
            f"elapsed_minutes={elapsed_minutes:.1f}"
        )

    reliability_df = pd.DataFrame(occurrence.rows())
    magnitude_reliability_df = pd.DataFrame(
        {
            "quantile": DEFAULT_QUANTILES,
            "empirical_cdf": conditional_magnitude.cdf_sum
            / max(conditional_magnitude.count, 1),
        }
    )
    magnitude_reliability_df["calibration_error"] = (
        magnitude_reliability_df["empirical_cdf"]
        - magnitude_reliability_df["quantile"]
    )
    magnitude_histogram_df = pd.DataFrame(
        {
            "bin_left": HISTOGRAM_EDGES[:-1],
            "bin_right": HISTOGRAM_EDGES[1:],
            "bin_center": 0.5 * (HISTOGRAM_EDGES[:-1] + HISTOGRAM_EDGES[1:]),
            "probability": conditional_magnitude.histogram_sum
            / max(conditional_magnitude.count, 1),
            "expected_probability": 1.0 / (len(HISTOGRAM_EDGES) - 1),
        }
    )
    building_df = pd.DataFrame(building_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = output_dir / "qroom_occurrence_reliability.csv"
    magnitude_path = output_dir / "qroom_positive_magnitude_reliability.csv"
    histogram_path = output_dir / "qroom_positive_magnitude_rank_histogram.csv"
    building_path = output_dir / "qroom_calibration_by_building.csv"
    dashboard_path = output_dir / "qroom_calibration_decomposition.html"
    summary_path = output_dir / "qroom_calibration_decomposition_summary.json"
    reliability_df.to_csv(reliability_path, index=False)
    magnitude_reliability_df.to_csv(magnitude_path, index=False)
    magnitude_histogram_df.to_csv(histogram_path, index=False)
    building_df.to_csv(building_path, index=False)
    _write_dashboard(
        dashboard_path,
        reliability_df,
        magnitude_reliability_df,
        magnitude_histogram_df,
        building_df,
        threshold=threshold,
    )

    occurrence_metrics = occurrence.metrics()
    conditional_metrics = conditional_magnitude.metrics()
    summary = {
        "artifact_dir": str(args.artifact_dir),
        "dataset_path": str(dataset_path),
        "profile_source": args.profile_source,
        "profile_count": int(len(building_df)),
        "num_scenarios": int(args.num_scenarios),
        "qroom_threshold_w_m2": threshold,
        "scope": "space-heating-available timestamps",
        "occurrence": {
            **occurrence_metrics,
            "observation_count": int(occurrence.total_count),
            "truth_active_fraction": float(
                np.average(
                    building_df["truth_active_fraction"],
                    weights=building_df["observation_count"],
                )
            ),
            "predicted_active_fraction": float(
                np.average(
                    building_df["predicted_active_fraction"],
                    weights=building_df["observation_count"],
                )
            ),
        },
        "positive_magnitude": {
            **conditional_metrics,
            "observation_count": int(conditional_magnitude.count),
            "truth_active_without_active_scenario_count": int(
                conditional_magnitude.insufficient_scenario_count
            ),
            "mean_building_bias_w_m2": float(
                building_df["conditional_magnitude_bias_w_m2"].mean()
            ),
            "median_building_rmse_w_m2": float(
                building_df["conditional_magnitude_rmse_w_m2"].median()
            ),
        },
        "persistence": {
            "mean_truth_p_on_given_on": float(
                building_df["truth_p_on_given_on"].mean()
            ),
            "mean_scenario_p_on_given_on": float(
                building_df["scenario_mean_p_on_given_on"].mean()
            ),
            "mean_truth_p_on_given_off": float(
                building_df["truth_p_on_given_off"].mean()
            ),
            "mean_scenario_p_on_given_off": float(
                building_df["scenario_mean_p_on_given_off"].mean()
            ),
            "mean_truth_active_run_hours": float(
                building_df["truth_mean_active_run_hours"].mean()
            ),
            "mean_scenario_active_run_hours": float(
                building_df["scenario_mean_active_run_hours"].mean()
            ),
        },
        "worst_occurrence_buildings": building_df.nlargest(
            min(10, len(building_df)), "occurrence_brier_score"
        )[
            [
                "profile_id",
                "occurrence_brier_score",
                "occurrence_expected_calibration_error",
                "truth_active_fraction",
                "predicted_active_fraction",
            ]
        ].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved_dashboard={dashboard_path}")
    print(f"saved_building_metrics={building_path}")
    print(f"saved_summary={summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile-source", choices=("test", "train", "selected"), default="test"
    )
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--qroom-threshold-w-m2", type=float, default=0.05)
    parser.add_argument(
        "--prob-hp-scenario-mode",
        choices=("expected", "bernoulli"),
        default="bernoulli",
    )
    parser.add_argument(
        "--prob-hp-emission-mode",
        choices=("artifact", "bounded", "legacy_lognormal_mean"),
        default="artifact",
    )
    parser.add_argument(
        "--ventilation-rollout-mode",
        choices=("recorded", "eplus_rule"),
        default="eplus_rule",
    )
    return parser.parse_args()


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
