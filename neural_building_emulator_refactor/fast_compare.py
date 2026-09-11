"""Compare saved artifacts with the lightweight ablation KPI scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .artifacts import load_artifact
from .fast_kpis import FastKpiConfig, evaluate_fast_kpis
from .profiles import saved_profile_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast comparison of multiple saved artifacts.")
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-source", choices=("train", "test", "selected"), default="test")
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--num-eval-particles", type=int, default=4)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--event-horizon-hours", type=float, default=3.0)
    parser.add_argument("--min-setpoint-change-c", type=float, default=0.05)
    parser.add_argument("--acf-lags", type=int, nargs="+", default=(1, 2, 4, 8, 12, 24, 48, 96))
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _flatten_summary(summary: dict) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "model_name": summary["model_name"],
        "task": summary["task"],
        "profile_count": int(summary["profile_count"]),
        "num_eval_particles": int(summary["num_eval_particles"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
    }
    for metric, statistics in summary["metrics"].items():
        for statistic, value in statistics.items():
            row[f"{metric}_{statistic}"] = float(value)
    return row


def main() -> None:
    args = parse_args()
    if len(args.artifact_dir) < 2:
        raise ValueError("Pass --artifact-dir at least twice for a comparison")
    artifacts = [load_artifact(path) for path in args.artifact_dir]
    tasks = {artifact.spec.task for artifact in artifacts}
    if len(tasks) != 1:
        raise ValueError("Fast ablation comparisons require artifacts from the same task")
    expected_ids = saved_profile_ids(artifacts[0], args.profile_source)[: args.max_profiles]
    for artifact in artifacts[1:]:
        actual_ids = saved_profile_ids(artifact, args.profile_source)[: args.max_profiles]
        if actual_ids != expected_ids:
            raise ValueError(
                "Artifacts do not contain the same ordered profile split. "
                "Use identical selection, split, and seed for an ablation."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = FastKpiConfig(
        dt_hours=args.dt_hours,
        event_horizon_hours=args.event_horizon_hours,
        min_setpoint_change_c=args.min_setpoint_change_c,
        acf_lags=tuple(args.acf_lags),
    )
    rows: list[dict[str, float | str | int]] = []
    for index, artifact_dir in enumerate(args.artifact_dir, start=1):
        artifact = artifacts[index - 1]
        destination = args.output_dir / f"{index:02d}_{artifact.spec.name}"
        evaluate_fast_kpis(
            artifact_dir,
            dataset_path=args.dataset,
            output_dir=destination,
            profile_source=args.profile_source,
            max_profiles=args.max_profiles,
            num_eval_particles=args.num_eval_particles,
            config=config,
            seed=args.seed,
        )
        summary = json.loads((destination / "fast_kpi_summary.json").read_text())
        row = _flatten_summary(summary)
        row["artifact_dir"] = str(artifact_dir)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    path = args.output_dir / "fast_ablation_comparison.csv"
    comparison.to_csv(path, index=False)
    headline = [
        column
        for column in (
            "model_name",
            "total_nrmse_mean",
            "temperature_increment_spectral_js_mean",
            "pel_increment_spectral_js_mean",
            "flex_event_delta_p_nrmse_mean",
            "flex_up_energy_gain_abs_error_wh_m2_k_mean",
            "flex_down_energy_gain_abs_error_wh_m2_k_mean",
            "elapsed_seconds",
        )
        if column in comparison
    ]
    print(comparison[headline].to_string(index=False))
    print(f"saved_fast_comparison={path}")


if __name__ == "__main__":
    main()
