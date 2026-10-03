"""Online diagnostics for probabilistic heat-pump switching trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


QUANTILES = np.linspace(0.05, 0.95, 19, dtype=np.float64)
RELIABILITY_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)
RUN_DURATION_EDGES_HOURS = np.asarray(
    (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0, np.inf),
    dtype=np.float64,
)
ACF_LAGS = (1, 4, 12)


@dataclass
class _BinaryReliability:
    counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(RELIABILITY_EDGES) - 1, dtype=np.int64)
    )
    predicted_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(RELIABILITY_EDGES) - 1, dtype=np.float64)
    )
    observed_sums: np.ndarray = field(
        default_factory=lambda: np.zeros(len(RELIABILITY_EDGES) - 1, dtype=np.float64)
    )
    brier_sum: float = 0.0
    total_count: int = 0

    def add(self, probability: np.ndarray, observed: np.ndarray) -> None:
        probability = np.asarray(probability, dtype=np.float64)
        observed = np.asarray(observed, dtype=np.float64)
        valid = np.isfinite(probability) & np.isfinite(observed)
        probability = np.clip(probability[valid], 0.0, 1.0)
        observed = observed[valid]
        if not probability.size:
            return
        bins = np.clip(
            np.searchsorted(RELIABILITY_EDGES, probability, side="right") - 1,
            0,
            len(self.counts) - 1,
        )
        self.counts += np.bincount(bins, minlength=len(self.counts))
        self.predicted_sums += np.bincount(
            bins, weights=probability, minlength=len(self.counts)
        )
        self.observed_sums += np.bincount(
            bins, weights=observed, minlength=len(self.counts)
        )
        self.brier_sum += float(np.sum((probability - observed) ** 2))
        self.total_count += int(probability.size)

    def rows(self, transition: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, count in enumerate(self.counts):
            if count == 0:
                continue
            rows.append(
                {
                    "transition": transition,
                    "bin_left": float(RELIABILITY_EDGES[index]),
                    "bin_right": float(RELIABILITY_EDGES[index + 1]),
                    "predicted_probability": float(self.predicted_sums[index] / count),
                    "observed_frequency": float(self.observed_sums[index] / count),
                    "observation_count": int(count),
                }
            )
        return rows

    def metrics(self) -> dict[str, float]:
        if self.total_count == 0:
            return {"brier": float("nan"), "ece": float("nan")}
        populated = self.counts > 0
        predicted = self.predicted_sums[populated] / self.counts[populated]
        observed = self.observed_sums[populated] / self.counts[populated]
        ece = np.sum(self.counts[populated] * np.abs(predicted - observed))
        return {
            "brier": float(self.brier_sum / self.total_count),
            "ece": float(ece / self.total_count),
        }


@dataclass
class _RankCalibration:
    cdf_sum: np.ndarray = field(default_factory=lambda: np.zeros_like(QUANTILES))
    count: int = 0

    def add(self, truth: np.ndarray, samples: np.ndarray, mask: np.ndarray) -> None:
        valid = (
            np.asarray(mask, dtype=bool)
            & np.isfinite(truth)
            & np.all(np.isfinite(samples), axis=0)
        )
        if not np.any(valid):
            return
        observed = np.asarray(truth, dtype=np.float64)[valid]
        ensemble = np.asarray(samples, dtype=np.float64)[:, valid]
        less = np.sum(ensemble < observed[np.newaxis, :], axis=0)
        equal = np.sum(ensemble == observed[np.newaxis, :], axis=0)
        denominator = np.float64(ensemble.shape[0] + 1)
        lower = less / denominator
        upper = (less + equal + 1) / denominator
        self.cdf_sum += np.sum(
            np.clip(
                (QUANTILES[:, np.newaxis] - lower[np.newaxis, :])
                / (upper - lower)[np.newaxis, :],
                0.0,
                1.0,
            ),
            axis=1,
        )
        self.count += int(np.sum(valid))

    def rows(self, *, condition: str, stratum: str) -> list[dict[str, Any]]:
        if self.count == 0:
            return []
        empirical = self.cdf_sum / np.float64(self.count)
        return [
            {
                "condition": condition,
                "stratum": stratum,
                "quantile": float(quantile),
                "empirical_coverage": float(value),
                "calibration_error": float(value - quantile),
                "abs_calibration_error": float(abs(value - quantile)),
                "observation_count": int(self.count),
            }
            for quantile, value in zip(QUANTILES, empirical)
        ]


def _run_lengths(active: np.ndarray, available: np.ndarray) -> np.ndarray:
    values = np.asarray(active, dtype=bool) & np.asarray(available, dtype=bool)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(np.float64)


def _binary_acf(active: np.ndarray, available: np.ndarray, lag: int) -> float:
    if lag >= len(active):
        return float("nan")
    valid = available[:-lag] & available[lag:]
    left = np.asarray(active[:-lag], dtype=np.float64)[valid]
    right = np.asarray(active[lag:], dtype=np.float64)[valid]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _shock_state(
    setpoint: np.ndarray,
    *,
    threshold_c: float,
    dt_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.concatenate(([0.0], np.diff(np.asarray(setpoint, dtype=np.float64))))
    age = np.full(delta.shape, np.inf, dtype=np.float64)
    direction = np.zeros(delta.shape, dtype=np.int8)
    current_age = np.inf
    current_direction = 0
    for index, change in enumerate(delta):
        if abs(change) > threshold_c:
            current_age = 0.0
            current_direction = 1 if change > 0.0 else -1
        elif np.isfinite(current_age):
            current_age += dt_hours
        age[index] = current_age
        direction[index] = current_direction if current_age < 3.0 else 0
    return age, direction


class PelDynamicsCalibrationAccumulator:
    """Aggregate HP transition, persistence, and conditional rank diagnostics."""

    def __init__(
        self,
        *,
        dt_hours: float = 0.25,
        active_threshold_w_m2: float = 1e-6,
        setpoint_threshold_c: float = 0.05,
    ) -> None:
        self.dt_hours = float(dt_hours)
        self.active_threshold_w_m2 = float(active_threshold_w_m2)
        self.setpoint_threshold_c = float(setpoint_threshold_c)
        self.start = _BinaryReliability()
        self.stay_on = _BinaryReliability()
        self.conditional: dict[tuple[str, str], _RankCalibration] = {}
        self.annual_energy = _RankCalibration()
        self.truth_run_hist = np.zeros(len(RUN_DURATION_EDGES_HOURS) - 1, dtype=np.int64)
        self.scenario_run_hist = np.zeros_like(self.truth_run_hist)
        self.truth_run_sum_hours = 0.0
        self.truth_run_count = 0
        self.scenario_run_sum_hours = 0.0
        self.scenario_run_count = 0
        self.acf_rows: list[dict[str, Any]] = []
        self.profile_count = 0

    def _conditional_add(
        self,
        condition: str,
        strata: dict[str, np.ndarray],
        truth: np.ndarray,
        scenarios: np.ndarray,
        available: np.ndarray,
    ) -> None:
        for stratum, mask in strata.items():
            accumulator = self.conditional.setdefault(
                (condition, stratum), _RankCalibration()
            )
            accumulator.add(truth, scenarios, available & mask)

    def add(
        self,
        *,
        truth: np.ndarray,
        scenarios: np.ndarray,
        setpoint: np.ndarray,
        current_temperature: np.ndarray,
        available: np.ndarray,
    ) -> None:
        truth = np.asarray(truth, dtype=np.float64)
        scenarios = np.asarray(scenarios, dtype=np.float64)
        available = np.asarray(available, dtype=bool)
        truth_active = truth > self.active_threshold_w_m2
        scenario_active = scenarios > self.active_threshold_w_m2
        transition_valid = available[1:] & available[:-1]
        truth_previous = truth_active[:-1]
        truth_current = truth_active[1:]
        scenario_previous = scenario_active[:, :-1]
        scenario_current = scenario_active[:, 1:]

        off_count = np.sum(~scenario_previous, axis=0)
        start_probability = np.divide(
            np.sum((~scenario_previous) & scenario_current, axis=0),
            off_count,
            out=np.full(off_count.shape, np.nan, dtype=np.float64),
            where=off_count > 0,
        )
        start_scope = transition_valid & (~truth_previous)
        self.start.add(start_probability[start_scope], truth_current[start_scope])

        on_count = np.sum(scenario_previous, axis=0)
        stay_probability = np.divide(
            np.sum(scenario_previous & scenario_current, axis=0),
            on_count,
            out=np.full(on_count.shape, np.nan, dtype=np.float64),
            where=on_count > 0,
        )
        stay_scope = transition_valid & truth_previous
        self.stay_on.add(stay_probability[stay_scope], truth_current[stay_scope])

        truth_runs = self.dt_hours * _run_lengths(truth_active, available)
        if truth_runs.size:
            self.truth_run_hist += np.histogram(
                truth_runs, bins=RUN_DURATION_EDGES_HOURS
            )[0]
            self.truth_run_sum_hours += float(np.sum(truth_runs))
            self.truth_run_count += int(truth_runs.size)
        for trace_active in scenario_active:
            runs = self.dt_hours * _run_lengths(trace_active, available)
            if runs.size:
                self.scenario_run_hist += np.histogram(
                    runs, bins=RUN_DURATION_EDGES_HOURS
                )[0]
                self.scenario_run_sum_hours += float(np.sum(runs))
                self.scenario_run_count += int(runs.size)

        for lag in ACF_LAGS:
            truth_acf = _binary_acf(truth_active, available, lag)
            scenario_acf = np.asarray(
                [_binary_acf(values, available, lag) for values in scenario_active]
            )
            self.acf_rows.append(
                {
                    "profile_index": self.profile_count,
                    "lag_steps": lag,
                    "lag_hours": lag * self.dt_hours,
                    "truth_acf": truth_acf,
                    "scenario_mean_acf": float(np.nanmean(scenario_acf))
                    if np.any(np.isfinite(scenario_acf))
                    else float("nan"),
                }
            )

        age, direction = _shock_state(
            setpoint,
            threshold_c=self.setpoint_threshold_c,
            dt_hours=self.dt_hours,
        )
        age_strata = {
            "0-0.5h": age < 0.5,
            "0.5-1h": (age >= 0.5) & (age < 1.0),
            "1-2h": (age >= 1.0) & (age < 2.0),
            "2-3h": (age >= 2.0) & (age < 3.0),
            ">=3h/no_recent": age >= 3.0,
        }
        direction_strata = {
            "recent_down": direction < 0,
            "recent_up": direction > 0,
            "no_recent": direction == 0,
        }
        previous_mode = np.concatenate(([False], truth_active[:-1]))
        previously_off = ~previous_mode
        previously_on = previous_mode.copy()
        previously_off[0] = False
        previously_on[0] = False
        previous_mode_strata = {
            "previously_off": previously_off,
            "previously_on": previously_on,
        }
        deficit = np.asarray(setpoint) - np.asarray(current_temperature)
        deficit_strata = {
            "<=-0.5C": deficit <= -0.5,
            "-0.5..0C": (deficit > -0.5) & (deficit <= 0.0),
            "0..0.5C": (deficit > 0.0) & (deficit <= 0.5),
            "0.5..1C": (deficit > 0.5) & (deficit <= 1.0),
            ">1C": deficit > 1.0,
        }
        self._conditional_add("shock_age", age_strata, truth, scenarios, available)
        self._conditional_add(
            "shock_direction", direction_strata, truth, scenarios, available
        )
        self._conditional_add(
            "previous_observed_mode",
            previous_mode_strata,
            truth,
            scenarios,
            available,
        )
        self._conditional_add(
            "thermostat_deficit", deficit_strata, truth, scenarios, available
        )

        truth_energy = np.asarray([self.dt_hours * np.sum(truth[available])])
        scenario_energy = self.dt_hours * np.sum(scenarios[:, available], axis=1)
        self.annual_energy.add(
            truth_energy,
            scenario_energy[:, np.newaxis],
            np.asarray([True]),
        )
        self.profile_count += 1

    def reliability_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.start.rows("start_given_observed_off")
            + self.stay_on.rows("stay_on_given_observed_on")
        )

    def conditional_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (condition, stratum), accumulator in self.conditional.items():
            rows.extend(accumulator.rows(condition=condition, stratum=stratum))
        return pd.DataFrame(rows)

    def conditional_summary_frame(self) -> pd.DataFrame:
        frame = self.conditional_frame()
        if frame.empty:
            return frame
        return (
            frame.groupby(["condition", "stratum"], as_index=False)
            .agg(
                calibration_mae=("abs_calibration_error", "mean"),
                calibration_max=("abs_calibration_error", "max"),
                observation_count=("observation_count", "max"),
            )
        )

    def run_duration_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for signal, counts in (
            ("simulation", self.truth_run_hist),
            ("emulation", self.scenario_run_hist),
        ):
            total = max(int(np.sum(counts)), 1)
            for index, count in enumerate(counts):
                upper = RUN_DURATION_EDGES_HOURS[index + 1]
                rows.append(
                    {
                        "signal": signal,
                        "bin_left_hours": float(RUN_DURATION_EDGES_HOURS[index]),
                        "bin_right_hours": float(upper),
                        "bin_label": (
                            f">={RUN_DURATION_EDGES_HOURS[index]:g}"
                            if np.isinf(upper)
                            else f"{RUN_DURATION_EDGES_HOURS[index]:g}-{upper:g}"
                        ),
                        "probability": float(count / total),
                        "run_count": int(count),
                    }
                )
        return pd.DataFrame(rows)

    def acf_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.acf_rows)
        if frame.empty:
            return frame
        return (
            frame.groupby(["lag_steps", "lag_hours"], as_index=False)
            .agg(
                truth_acf=("truth_acf", "mean"),
                scenario_mean_acf=("scenario_mean_acf", "mean"),
            )
        )

    def annual_energy_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.annual_energy.rows(condition="annual_energy", stratum="all_buildings")
        )

    def summary(self) -> dict[str, float]:
        start = self.start.metrics()
        stay = self.stay_on.metrics()
        truth_mean_run = self.truth_run_sum_hours / max(self.truth_run_count, 1)
        scenario_mean_run = self.scenario_run_sum_hours / max(
            self.scenario_run_count, 1
        )
        truth_hist = self.truth_run_hist / max(np.sum(self.truth_run_hist), 1)
        scenario_hist = self.scenario_run_hist / max(
            np.sum(self.scenario_run_hist), 1
        )
        acf = self.acf_frame()
        acf_error = (
            acf["scenario_mean_acf"].to_numpy() - acf["truth_acf"].to_numpy()
            if not acf.empty
            else np.asarray([], dtype=np.float64)
        )
        conditional = self.conditional_summary_frame()
        annual = self.annual_energy_frame()
        return {
            "pel_start_brier": start["brier"],
            "pel_start_ece": start["ece"],
            "pel_stay_on_brier": stay["brier"],
            "pel_stay_on_ece": stay["ece"],
            "pel_truth_mean_run_hours": float(truth_mean_run),
            "pel_scenario_mean_run_hours": float(scenario_mean_run),
            "pel_mean_run_duration_abs_error_hours": float(
                abs(scenario_mean_run - truth_mean_run)
            ),
            "pel_run_duration_total_variation": float(
                0.5 * np.sum(np.abs(scenario_hist - truth_hist))
            ),
            "pel_mode_acf_rmse": float(np.sqrt(np.nanmean(acf_error**2)))
            if acf_error.size
            else float("nan"),
            "pel_conditional_calibration_macro_mae": float(
                conditional["calibration_mae"].mean()
            )
            if not conditional.empty
            else float("nan"),
            "pel_conditional_calibration_worst_mae": float(
                conditional["calibration_mae"].max()
            )
            if not conditional.empty
            else float("nan"),
            "pel_annual_energy_calibration_mae": float(
                annual["abs_calibration_error"].mean()
            )
            if not annual.empty
            else float("nan"),
        }


def write_pel_dynamics_dashboard(
    path: Path,
    *,
    reliability: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    run_duration: pd.DataFrame,
    acf: pd.DataFrame,
    annual_energy: pd.DataFrame,
    title: str,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Start reliability after observed off state",
            "Persistence reliability after observed on state",
            "Conditional Pel rank calibration",
            "Active-run duration distribution",
            "HP-mode autocorrelation",
            "Annual Pel energy calibration",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )
    for column, transition in enumerate(
        ("start_given_observed_off", "stay_on_given_observed_on"), start=1
    ):
        part = reliability[reliability["transition"] == transition]
        figure.add_trace(
            go.Scatter(
                x=part["predicted_probability"],
                y=part["observed_frequency"],
                mode="lines+markers",
                name=transition,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line={"color": "black", "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    if not conditional_summary.empty:
        labels = conditional_summary["condition"] + ": " + conditional_summary["stratum"]
        figure.add_trace(
            go.Bar(
                x=labels,
                y=conditional_summary["calibration_mae"],
                marker_color="#8e24aa",
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    for signal, color in (("simulation", "#1769aa"), ("emulation", "#d32f2f")):
        part = run_duration[run_duration["signal"] == signal]
        figure.add_trace(
            go.Scatter(
                x=part["bin_label"], y=part["probability"],
                mode="lines+markers", name=f"{signal} run duration",
                line={"color": color},
            ),
            row=2,
            col=1,
        )
    if not acf.empty:
        figure.add_trace(
            go.Scatter(
                x=acf["lag_hours"], y=acf["truth_acf"],
                mode="lines+markers", name="simulated mode ACF",
                line={"color": "#1769aa"},
            ),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=acf["lag_hours"], y=acf["scenario_mean_acf"],
                mode="lines+markers", name="emulated mode ACF",
                line={"color": "#d32f2f"},
            ),
            row=2,
            col=2,
        )
    if not annual_energy.empty:
        figure.add_trace(
            go.Scatter(
                x=annual_energy["quantile"],
                y=annual_energy["empirical_coverage"],
                mode="lines+markers", name="annual Pel energy",
            ),
            row=2,
            col=3,
        )
        figure.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line={"color": "black", "dash": "dash"},
                showlegend=False,
            ),
            row=2,
            col=3,
        )
    figure.update_xaxes(title_text="predicted probability", range=[0, 1], row=1, col=1)
    figure.update_yaxes(title_text="observed frequency", range=[0, 1], row=1, col=1)
    figure.update_xaxes(title_text="predicted probability", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="observed frequency", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="mean |rank CDF error|", row=1, col=3)
    figure.update_xaxes(title_text="active-run duration [h]", row=2, col=1)
    figure.update_yaxes(title_text="probability", row=2, col=1)
    figure.update_xaxes(title_text="lag [h]", row=2, col=2)
    figure.update_yaxes(title_text="mode autocorrelation", row=2, col=2)
    figure.update_xaxes(title_text="nominal quantile", range=[0, 1], row=2, col=3)
    figure.update_yaxes(title_text="empirical rank CDF", range=[0, 1], row=2, col=3)
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=980,
        width=1650,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)
