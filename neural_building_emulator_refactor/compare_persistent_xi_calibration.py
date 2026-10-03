"""Compare matched all-path and restricted persistent-xi experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


SETPOINT_DIR = "flexibility_event_study_scenarios_regressor"
FORCE_OFF_DIR = "force_off_policy_evaluation"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _coverage_metrics(
    path: Path,
    group_columns: list[str],
    *,
    filters: dict[str, object] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column, value in (filters or {}).items():
        frame = frame[frame[column] == value]
    if "calibration_error" not in frame:
        frame["calibration_error"] = frame["empirical_coverage"] - frame["quantile"]
    rows: list[dict] = []
    for keys, values in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["calibration_mae"] = float(np.mean(np.abs(values["calibration_error"])))
        low = values[np.isclose(values["quantile"], 0.05)]
        high = values[np.isclose(values["quantile"], 0.95)]
        if not low.empty and not high.empty:
            central_90 = float(high.iloc[0]["empirical_coverage"] - low.iloc[0]["empirical_coverage"])
            row["central_90_coverage"] = central_90
            row["central_90_error"] = abs(central_90 - 0.90)
        rows.append(row)
    return pd.DataFrame(rows)


def _event_response_errors(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keys = ["profile_id", "event_index", "signal", "lag_hours"]
    truth = frame[frame["model"] == "simulation"][keys + ["value"]].rename(
        columns={"value": "truth"}
    )
    prediction = frame[frame["model"] == "structured_gated"][keys + ["value"]].rename(
        columns={"value": "prediction"}
    )
    merged = truth.merge(prediction, on=keys, validate="one_to_one")
    merged["error"] = merged["prediction"] - merged["truth"]
    return (
        merged.groupby("signal", sort=False)["error"]
        .agg(
            rmse=lambda values: float(np.sqrt(np.mean(np.square(values)))),
            mae=lambda values: float(np.mean(np.abs(values))),
            bias="mean",
        )
        .reset_index()
    )


def _collect(root: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    setpoint = root / SETPOINT_DIR
    force_off = root / FORCE_OFF_DIR
    regime_force_off = root / f"{FORCE_OFF_DIR}_regimes"
    if not regime_force_off.exists():
        regime_force_off = force_off
    summary = _read_json(setpoint / "summary.json")
    for name, value in summary["metrics"].items():
        values[f"setpoint/point/{name}"] = float(value)
    for row in summary.get("temporal_trace_calibration", []):
        signal = row["signal"]
        for name in (
            "mean_crps",
            "mean_abs_calibration_error",
            "max_abs_calibration_error",
            "empirical_central_90_coverage",
            "mean_central_90_width",
        ):
            if name in row:
                values[f"setpoint/trace/{signal}/{name}"] = float(row[name])
    for name, value in summary.get("pel_dynamics", {}).items():
        values[f"setpoint/pel_dynamics/{name}"] = float(value)
    for name, value in summary.get("qroom_calibration", {}).items():
        values[f"setpoint/qroom/{name}"] = float(value)

    for suffix, estimator in (("", "regression"), ("_f", "direct_f")):
        coverage = _coverage_metrics(
            setpoint / f"flexibility_kpi_coverage{suffix}.csv",
            ["horizon_hours", "direction"],
        )
        for row in coverage.itertuples(index=False):
            prefix = f"setpoint/{estimator}/{row.direction}/{row.horizon_hours:g}h"
            values[f"{prefix}/calibration_mae"] = float(row.calibration_mae)
            values[f"{prefix}/central_90_error"] = float(row.central_90_error)

    force_summary = _read_json(force_off / "force_off_policy_summary.json")
    for row in force_summary["aggregate_metrics"]:
        if row["model"] != "structured_gated":
            continue
        prefix = f"force_off/point/{row['period']}/{row['signal']}"
        for name in ("mean_rmse", "mean_mae", "mean_bias"):
            value = row.get(name)
            if value is not None and np.isfinite(value):
                values[f"{prefix}/{name}"] = float(value)

    temporal_scores = pd.read_csv(force_off / "temporal_trace_probabilistic_scores.csv")
    temporal_reliability = pd.read_csv(force_off / "temporal_trace_calibration.csv")
    temporal_reliability["abs_error"] = np.abs(
        temporal_reliability["empirical_coverage"] - temporal_reliability["quantile"]
    )
    for row in temporal_scores[temporal_scores["model"] == "structured_gated"].itertuples(index=False):
        prefix = f"force_off/trace/all/{row.signal}"
        values[f"{prefix}/mean_crps"] = float(row.mean_crps)
        values[f"{prefix}/mean_central_90_width"] = float(row.mean_central_90_width)
        reliability = temporal_reliability[
            (temporal_reliability["model"] == "structured_gated")
            & (temporal_reliability["signal"] == row.signal)
        ]
        values[f"{prefix}/calibration_mae"] = float(reliability["abs_error"].mean())
        low = reliability[np.isclose(reliability["quantile"], 0.05)]
        high = reliability[np.isclose(reliability["quantile"], 0.95)]
        if not low.empty and not high.empty:
            values[f"{prefix}/central_90_coverage"] = float(
                high.iloc[0]["empirical_coverage"] - low.iloc[0]["empirical_coverage"]
            )

    regime_score_path = regime_force_off / "temporal_trace_probabilistic_scores_by_regime.csv"
    regime_reliability_path = regime_force_off / "temporal_trace_calibration_by_regime.csv"
    if regime_score_path.exists() and regime_reliability_path.exists():
        scores = pd.read_csv(regime_score_path)
        reliability = pd.read_csv(regime_reliability_path)
        reliability["abs_error"] = np.abs(
            reliability["empirical_coverage"] - reliability["quantile"]
        )
        for row in scores[scores["model"] == "structured_gated"].itertuples(index=False):
            prefix = f"force_off/trace/{row.regime}/{row.signal}"
            values[f"{prefix}/mean_crps"] = float(row.mean_crps)
            values[f"{prefix}/mean_central_90_width"] = float(row.mean_central_90_width)
            selected = reliability[
                (reliability["model"] == "structured_gated")
                & (reliability["regime"] == row.regime)
                & (reliability["signal"] == row.signal)
            ]
            values[f"{prefix}/calibration_mae"] = float(selected["abs_error"].mean())
            low = selected[np.isclose(selected["quantile"], 0.05)]
            high = selected[np.isclose(selected["quantile"], 0.95)]
            if not low.empty and not high.empty:
                values[f"{prefix}/central_90_coverage"] = float(
                    high.iloc[0]["empirical_coverage"]
                    - low.iloc[0]["empirical_coverage"]
                )

    force_coverage = _coverage_metrics(
        force_off / "force_off_flexibility_kpi_coverage.csv",
        ["estimator", "horizon_hours"],
        filters={"model": "structured_gated"},
    )
    for row in force_coverage.itertuples(index=False):
        prefix = f"force_off/{row.estimator}/{row.horizon_hours:g}h"
        values[f"{prefix}/calibration_mae"] = float(row.calibration_mae)
        values[f"{prefix}/central_90_error"] = float(row.central_90_error)
    for row in _event_response_errors(force_off / "force_off_event_responses.csv").itertuples(index=False):
        values[f"force_off/event/{row.signal}/rmse"] = float(row.rmse)
        values[f"force_off/event/{row.signal}/mae"] = float(row.mae)
        values[f"force_off/event/{row.signal}/abs_bias"] = abs(float(row.bias))
    return values


def _write_dashboard(path: Path, comparison: pd.DataFrame) -> None:
    display = comparison.copy()
    display["change_percent"] = np.where(
        np.abs(display["baseline"]) > 1e-12,
        100.0 * display["delta"] / np.abs(display["baseline"]),
        np.nan,
    )
    figure = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Metric", "All-path xi", "Thermal-only xi", "Delta", "Change"],
                    "fill_color": "#263238",
                    "font": {"color": "white", "size": 13},
                    "align": "left",
                },
                cells={
                    "values": [
                        display["metric"],
                        display["baseline"].map(lambda value: f"{value:.5g}"),
                        display["candidate"].map(lambda value: f"{value:.5g}"),
                        display["delta"].map(lambda value: f"{value:+.5g}"),
                        display["change_percent"].map(
                            lambda value: "" if not np.isfinite(value) else f"{value:+.1f}%"
                        ),
                    ],
                    "fill_color": "#ffffff",
                    "align": "left",
                    "height": 24,
                },
            )
        ]
    )
    figure.update_layout(
        title="Persistent-xi pathway calibration comparison",
        height=max(700, 130 + 24 * len(display)),
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    figure.write_html(path)


def run(baseline: Path, candidate: Path, output_dir: Path) -> Path:
    baseline_values = _collect(baseline)
    candidate_values = _collect(candidate)
    common = sorted(set(baseline_values).intersection(candidate_values))
    comparison = pd.DataFrame(
        {
            "metric": common,
            "baseline": [baseline_values[name] for name in common],
            "candidate": [candidate_values[name] for name in common],
        }
    )
    comparison["delta"] = comparison["candidate"] - comparison["baseline"]
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "persistent_xi_calibration_comparison.csv", index=False)
    _write_dashboard(output_dir / "persistent_xi_calibration_comparison.html", comparison)
    summary = {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "common_metric_count": len(common),
    }
    (output_dir / "persistent_xi_calibration_comparison.json").write_text(
        json.dumps(summary, indent=2)
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(args.baseline, args.candidate, args.output_dir)
    print(f"saved_comparison={output}")


if __name__ == "__main__":
    main()
