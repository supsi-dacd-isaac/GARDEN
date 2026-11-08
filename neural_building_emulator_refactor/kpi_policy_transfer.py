"""Evaluate frozen KPI predictors on a different setpoint-control dataset.

This is a transfer test, not a refit.  It reuses the probabilistic structured
and unstructured artifacts plus the fitted LightGBM quantile models from an
existing calibrated benchmark.  The original calibration/test building IDs
are retained, while annual trajectories and KPI targets are read from the
new dataset.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from neural_building_emulator.columns import (
    CLOSED_LOOP_METADATA_COLUMNS,
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)

from .artifacts import load_artifact
from .calibrated_model_benchmark import (
    BenchmarkSplit,
    KPI_KINDS,
    _emulator_distribution,
    _event_config,
    _normal_distribution_rows,
    _write_temporal_trace_comparison,
    _write_calibration_dashboard,
    _write_flexibility_point_scatter,
)
from .metadata_flexibility_lightgbm import (
    TargetSpec,
    _artifact_metadata,
    _metadata_value,
    extract_metadata_kpis,
)
from .probabilistic_calibration import calibrate_distribution_table
from .scoring import score_artifact


DEFAULT_BENCHMARK_DIR = Path(
    "output/neural_building_emulator_refactor/"
    "calibrated_four_model_benchmark_mixed_policy_hp_sampling_1000"
)
DEFAULT_DATASET = Path("data/control=setpoint")
DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/"
    "poor_excitation_transfer_best_1000"
)
METHODS = ("probabilistic_structured", "probabilistic_lstm")
STAGES = ("score", "lightgbm", "calibrate", "report")


def _load_source(
    benchmark_dir: Path,
) -> tuple[dict[str, Any], BenchmarkSplit, dict[str, Path]]:
    resolved = json.loads((benchmark_dir / "resolved_recipe.json").read_text())
    recipe = resolved["recipe"]
    split = BenchmarkSplit.from_json(resolved["split"])
    stored_paths = json.loads((benchmark_dir / "artifact_paths.json").read_text())
    artifacts = {
        method: Path(stored_paths[method]).resolve()
        for method in METHODS
    }
    for method, path in artifacts.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {method} artifact: {path}")
    return recipe, split, artifacts


def _score_models(
    *,
    benchmark_dir: Path,
    dataset: Path,
    output_dir: Path,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    artifacts: dict[str, Path],
    num_scenarios: int,
    num_profile_plots: int,
    resume: bool,
) -> None:
    evaluation = recipe["evaluation"]
    for method in METHODS:
        for split_name, profile_ids in (
            ("calibration", split.calibration_ids),
            ("test", split.test_ids),
        ):
            plot_ids = (
                tuple(int(value) for value in profile_ids[:num_profile_plots])
                if split_name == "test"
                else ()
            )
            destination = output_dir / "scoring" / split_name / method
            destination.mkdir(parents=True, exist_ok=True)
            request = {
                "source_benchmark": str(benchmark_dir.resolve()),
                "artifact_dir": str(artifacts[method]),
                "dataset": str(dataset.resolve()),
                "profile_ids": list(profile_ids),
                "num_scenarios": int(num_scenarios),
                "hp_scenario_mode": "bernoulli",
                "ventilation_rollout_mode": str(
                    evaluation["ventilation_rollout_mode"]
                ),
                "horizons_hours": list(evaluation["horizons_hours"]),
                "min_setpoint_change": float(
                    evaluation["min_setpoint_change"]
                ),
                "min_events": int(evaluation["min_events"]),
                "controls": str(evaluation["controls"]),
                "profile_plot_ids": list(plot_ids),
                "seed": int(recipe["seed"])
                + (0 if split_name == "calibration" else 1000),
            }
            request_path = destination / "transfer_request.json"
            summary_path = destination / "summary.json"
            if (
                resume
                and summary_path.exists()
                and request_path.exists()
                and json.loads(request_path.read_text()) == request
            ):
                print(f"transfer_score={method}:{split_name} action=reuse")
                continue
            print(
                f"transfer_score={method}:{split_name} action=run "
                f"profiles={len(profile_ids)} scenarios={num_scenarios}"
            )
            score_artifact(
                artifacts[method],
                dataset_path=dataset,
                output_dir=destination,
                profile_ids=profile_ids,
                kpi_mode="scenario_average",
                num_scenarios=num_scenarios,
                hp_scenario_mode="bernoulli",
                ventilation_rollout_mode=str(
                    evaluation["ventilation_rollout_mode"]
                ),
                horizons_hours=tuple(
                    float(value) for value in evaluation["horizons_hours"]
                ),
                min_setpoint_change=float(
                    evaluation["min_setpoint_change"]
                ),
                min_events=int(evaluation["min_events"]),
                kpi_estimator="regression",
                controls=str(evaluation["controls"]),
                seed=request["seed"],
                num_profile_plots=len(plot_ids),
                profile_plot_ids=plot_ids or None,
                evaluate_direct_f=True,
            )
            request_path.write_text(
                json.dumps(request, indent=2, sort_keys=True) + "\n"
            )


def _target_table(
    *,
    source_artifact: Path,
    source_manifest: dict[str, Any],
    dataset: Path,
    output_dir: Path,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    kpi_kind: str,
    resume: bool,
) -> tuple[pd.DataFrame, tuple[TargetSpec, ...], tuple[str, ...]]:
    destination = output_dir / "lightgbm" / kpi_kind
    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / "metadata_targets.csv"
    metadata_columns = tuple(
        source_manifest.get("metadata_columns", CLOSED_LOOP_METADATA_COLUMNS)
    )
    artifact = load_artifact(source_artifact)
    metadata = _artifact_metadata(artifact)
    input_columns = tuple(metadata.get("input_columns", ()))
    event_config, targets = _event_config(recipe, kpi_kind)
    request = {
        "dataset": str(dataset.resolve()),
        "calibration_ids": list(split.calibration_ids),
        "test_ids": list(split.test_ids),
        "metadata_columns": list(metadata_columns),
        "event_config": {
            "horizons_hours": list(event_config.horizons_hours),
            "horizon_steps": list(event_config.horizon_steps),
            "dt_hours": event_config.dt_hours,
            "min_setpoint_change": event_config.min_setpoint_change,
            "min_events": event_config.min_events,
            "controls": event_config.controls,
            "estimator": event_config.estimator,
        },
    }
    request_path = destination / "metadata_target_request.json"
    if (
        resume
        and table_path.exists()
        and request_path.exists()
        and json.loads(request_path.read_text()) == request
    ):
        return pd.read_csv(table_path), targets, metadata_columns
    table = extract_metadata_kpis(
        dataset,
        train_ids=split.calibration_ids,
        test_ids=split.test_ids,
        metadata_columns=metadata_columns,
        include_space_heating_availability=(
            SPACE_HEATING_AVAILABILITY_COLUMN in input_columns
        ),
        include_internal_gains=(
            INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in input_columns
        ),
        include_dhw_request=(
            DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN in input_columns
        ),
        hp_power_area_normalization=str(
            _metadata_value(
                metadata,
                "hp_power_area_normalization",
                "building_heated_area",
            )
        ),
        event_config=event_config,
    )
    calibration_ids = set(int(value) for value in split.calibration_ids)
    table["split"] = np.where(
        table["profile_id"].astype(int).isin(calibration_ids),
        "calibration",
        "test",
    )
    table["policy_type"] = "setpoint_modulation_poor_excitation"
    table.to_csv(table_path, index=False)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    return table, targets, metadata_columns


def _predict_lightgbm(
    *,
    benchmark_dir: Path,
    dataset: Path,
    output_dir: Path,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    source_artifact: Path,
    num_scenarios: int,
    resume: bool,
) -> None:
    for kpi_kind in KPI_KINDS:
        source_dir = benchmark_dir / "lightgbm" / kpi_kind
        manifest = json.loads((source_dir / "model_manifest.json").read_text())
        destination = output_dir / "lightgbm" / kpi_kind
        calibration_path = destination / "distribution_calibration.csv"
        test_path = destination / "distribution_test.csv"
        prediction_request = {
            "source_models": str(source_dir.resolve()),
            "dataset": str(dataset.resolve()),
            "calibration_ids": list(split.calibration_ids),
            "test_ids": list(split.test_ids),
            "num_scenarios": int(num_scenarios),
        }
        prediction_request_path = destination / "prediction_request.json"
        if (
            resume
            and calibration_path.exists()
            and test_path.exists()
            and prediction_request_path.exists()
            and json.loads(prediction_request_path.read_text())
            == prediction_request
        ):
            print(f"transfer_lightgbm={kpi_kind} action=reuse")
            continue
        table, targets, metadata_columns = _target_table(
            source_artifact=source_artifact,
            source_manifest=manifest,
            dataset=dataset,
            output_dir=output_dir,
            recipe=recipe,
            split=split,
            kpi_kind=kpi_kind,
            resume=resume,
        )
        target_by_key = {target.key: target for target in targets}
        rows: dict[str, list[dict[str, Any]]] = {
            "calibration": [],
            "test": [],
        }
        for target_key, target_manifest in manifest["models"].items():
            if target_key not in target_by_key:
                raise ValueError(
                    f"Saved LightGBM target {target_key!r} is absent from "
                    f"the {kpi_kind} target definition"
                )
            target = target_by_key[target_key]
            predictions: dict[str, dict[float, np.ndarray]] = {
                "calibration": {},
                "test": {},
            }
            for quantile in (0.16, 0.5, 0.84):
                model_entry = target_manifest[f"q{quantile:g}"]
                booster = lgb.Booster(
                    model_file=str(source_dir / "models" / model_entry["model_file"])
                )
                for split_name in predictions:
                    frame = table[table["split"].eq(split_name)]
                    predictions[split_name][quantile] = booster.predict(
                        frame.loc[:, metadata_columns]
                    )
            scale_floor = max(
                1e-3,
                0.01 * float(target_manifest["training_target_std"]),
            )
            for split_name in predictions:
                frame = table[table["split"].eq(split_name)]
                rows[split_name].extend(
                    _normal_distribution_rows(
                        frame,
                        target=target,
                        predictions=predictions[split_name],
                        method="lightgbm",
                        kpi_kind=kpi_kind,
                        sample_count=num_scenarios,
                        scale_floor=scale_floor,
                    )
                )
        pd.DataFrame(rows["calibration"]).to_csv(calibration_path, index=False)
        pd.DataFrame(rows["test"]).to_csv(test_path, index=False)
        prediction_request_path.write_text(
            json.dumps(prediction_request, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"transfer_lightgbm={kpi_kind} action=predicted "
            f"calibration_rows={len(rows['calibration'])} "
            f"test_rows={len(rows['test'])}"
        )


def _distribution_frames(
    output_dir: Path,
    *,
    split_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method in METHODS:
        score_dir = output_dir / "scoring" / split_name / method
        for kpi_kind in KPI_KINDS:
            frames.append(
                _emulator_distribution(
                    score_dir,
                    method=method,
                    kpi_kind=kpi_kind,
                )
            )
    for kpi_kind in KPI_KINDS:
        frames.append(
            pd.read_csv(
                output_dir
                / "lightgbm"
                / kpi_kind
                / f"distribution_{split_name}.csv"
            )
        )
    return pd.concat(frames, ignore_index=True)


def _calibrate(
    *,
    output_dir: Path,
    recipe: dict[str, Any],
) -> None:
    destination = output_dir / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    calibration = _distribution_frames(output_dir, split_name="calibration")
    test = _distribution_frames(output_dir, split_name="test")
    calibration.to_csv(
        destination / "raw_calibration_distributions.csv", index=False
    )
    test.to_csv(destination / "raw_test_distributions.csv", index=False)
    calibrated, metrics, reliability, buildings = calibrate_distribution_table(
        calibration,
        test,
        minimum_scale=float(recipe["calibration"]["minimum_scale"]),
        minimum_scale_fraction=float(
            recipe["calibration"]["minimum_scale_fraction"]
        ),
    )
    calibrated.to_csv(destination / "rns_test_distributions.csv", index=False)
    metrics.to_csv(destination / "distribution_metrics.csv", index=False)
    reliability.to_csv(destination / "distribution_reliability.csv", index=False)
    buildings.to_csv(destination / "distribution_by_building.csv", index=False)
    point_columns = [
        "method",
        "kpi_kind",
        "horizon_hours",
        "direction",
        "count",
        "rmse",
        "nrmse",
        "mae",
        "bias",
        "r2",
        "spearman",
        "kendall",
        "top_10pct_overlap",
    ]
    metrics[metrics["calibration"].eq("raw")][point_columns].to_csv(
        destination / "point_metrics.csv", index=False
    )
    print(
        f"transfer_calibration groups={len(metrics)} "
        f"test_building_rows={len(buildings)}"
    )


def _html_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    numeric = formatted.select_dtypes(include=[np.number]).columns
    formatted[numeric] = formatted[numeric].round(4)
    return formatted.to_html(index=False, border=0, classes="metrics")


def _write_report(
    *,
    benchmark_dir: Path,
    dataset: Path,
    output_dir: Path,
    split: BenchmarkSplit,
    num_scenarios: int,
) -> Path:
    destination = output_dir / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    calibration_dir = output_dir / "calibration"
    metrics = pd.read_csv(calibration_dir / "distribution_metrics.csv")
    reliability = pd.read_csv(calibration_dir / "distribution_reliability.csv")
    buildings = pd.read_csv(calibration_dir / "distribution_by_building.csv")
    calibration_plot = destination / "kpi_calibration.html"
    _write_calibration_dashboard(
        calibration_plot,
        metrics,
        reliability,
        kpi_kinds=("f", "regression_based"),
        subplot_titles=("Direct F reliability", "Regression-based reliability"),
        title="Poor-excitation transfer: raw and RNS KPI calibration",
        unit="Wh/(m2 K)",
    )
    scatter_paths: dict[str, Path] = {}
    for kpi_kind in KPI_KINDS:
        path = destination / f"flexibility_point_scatter_{kpi_kind}.html"
        if _write_flexibility_point_scatter(
            path,
            buildings,
            kpi_kind=kpi_kind,
        ):
            scatter_paths[kpi_kind] = path
    macro_columns = [
        "rmse",
        "nrmse",
        "mae",
        "bias",
        "r2",
        "spearman",
        "kendall",
        "top_10pct_overlap",
        "crps",
        "calibration_mae",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "width_90",
    ]
    macro = (
        metrics.groupby(["method", "kpi_kind", "calibration"], as_index=False)[
            macro_columns
        ]
        .mean()
        .sort_values(["kpi_kind", "calibration", "method"])
    )
    macro.to_csv(destination / "macro_metrics.csv", index=False)
    detail = metrics.sort_values(
        ["kpi_kind", "horizon_hours", "direction", "calibration", "method"]
    )
    signal_rows: list[dict[str, Any]] = []
    temporal_frames: list[pd.DataFrame] = []
    rank_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    diagnostic_filenames = (
        "pel_transition_reliability.csv",
        "pel_conditional_calibration.csv",
        "pel_conditional_calibration_summary.csv",
        "pel_mode_autocorrelation.csv",
        "pel_run_duration_distribution.csv",
        "pel_annual_energy_calibration.csv",
        "qroom_occurrence_reliability.csv",
    )
    diagnostic_frames: dict[str, list[pd.DataFrame]] = {
        filename: [] for filename in diagnostic_filenames
    }
    profile_links: list[str] = []
    for method in METHODS:
        score_dir = output_dir / "scoring" / "test" / method
        summary = json.loads((score_dir / "summary.json").read_text())
        signal_rows.append({"method": method, **summary["metrics"]})
        for filename, target in (
            ("temporal_trace_calibration.csv", temporal_frames),
            ("temporal_trace_rank_histogram.csv", rank_frames),
            ("temporal_trace_probabilistic_scores.csv", score_frames),
        ):
            source = score_dir / filename
            if source.exists():
                frame = pd.read_csv(source)
                frame.insert(0, "arm", method)
                target.append(frame)
        for filename, target in diagnostic_frames.items():
            source = score_dir / filename
            if source.exists():
                frame = pd.read_csv(source)
                frame.insert(0, "arm", method)
                target.append(frame)
        for profile_path in sorted(score_dir.glob("profile_*.html")):
            relative = Path("..") / profile_path.relative_to(output_dir)
            profile_links.append(
                f'<li><a href="{escape(str(relative))}">'
                f'{escape(method)}: {escape(profile_path.stem)}</a></li>'
            )
    pd.DataFrame(signal_rows).to_csv(
        destination / "annual_signal_metrics.csv", index=False
    )
    temporal = (
        pd.concat(temporal_frames, ignore_index=True)
        if temporal_frames
        else pd.DataFrame()
    )
    ranks = (
        pd.concat(rank_frames, ignore_index=True)
        if rank_frames
        else pd.DataFrame()
    )
    temporal_scores = (
        pd.concat(score_frames, ignore_index=True)
        if score_frames
        else pd.DataFrame()
    )
    temporal.to_csv(destination / "single_signal_reliability.csv", index=False)
    ranks.to_csv(destination / "single_signal_rank_histogram.csv", index=False)
    temporal_scores.to_csv(
        destination / "single_signal_probabilistic_scores.csv", index=False
    )
    temporal_plot = destination / "temporal_trace_calibration.html"
    _write_temporal_trace_comparison(
        temporal_plot,
        temporal,
        ranks,
        model_column="arm",
        title="Poor-excitation temporal trace calibration",
    )
    for filename, frames in diagnostic_frames.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined.to_csv(destination / filename, index=False)
    links = [
        '<li><a href="kpi_calibration.html">KPI reliability and CRPS</a></li>',
        '<li><a href="../calibration/distribution_metrics.csv">All metrics (CSV)</a></li>',
        '<li><a href="macro_metrics.csv">Macro-averaged metrics (CSV)</a></li>',
        '<li><a href="../calibration/distribution_by_building.csv">Per-building distributions (CSV)</a></li>',
        '<li><a href="temporal_trace_calibration.html">Tin/Qroom/Pel temporal calibration</a></li>',
        '<li><a href="annual_signal_metrics.csv">Annual signal metrics (CSV)</a></li>',
        '<li><a href="single_signal_probabilistic_scores.csv">Single-signal probabilistic scores (CSV)</a></li>',
    ]
    for kpi_kind, path in scatter_paths.items():
        links.append(
            f'<li><a href="{escape(path.name)}">{escape(kpi_kind)} parity plots</a></li>'
        )
    report_path = destination / "transfer_dashboard.html"
    report_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Poor-excitation KPI transfer</title>"
        "<style>body{font:15px system-ui;margin:30px;color:#263238}"
        "h1,h2{color:#243b5a}.metrics{border-collapse:collapse;font-size:12px}"
        ".metrics th,.metrics td{border:1px solid #d4dce5;padding:5px 7px}"
        ".metrics th{background:#e9eef5;position:sticky;top:0}"
        ".table-wrap{overflow:auto;max-height:700px;margin-bottom:30px}"
        "li{margin:7px 0}a{color:#1769aa}</style></head><body>"
        "<h1>Poor-excitation KPI transfer</h1>"
        f"<p>Frozen source benchmark: <code>{escape(str(benchmark_dir))}</code><br>"
        f"Target dataset: <code>{escape(str(dataset))}</code><br>"
        f"Calibration/test buildings: {len(split.calibration_ids)}/{len(split.test_ids)}; "
        f"annual scenarios per probabilistic prediction: {num_scenarios}.</p>"
        "<p>RNS is fitted only on the target-policy calibration buildings. "
        "All raw predictions come from models fitted on the source benchmark; "
        "none is retrained on this dataset.</p>"
        "<h2>Outputs</h2><ul>"
        + "".join(links)
        + "</ul><h2>Annual example traces</h2><ul>"
        + ("".join(profile_links) if profile_links else "<li>No profile plots requested.</li>")
        + "</ul><h2>Macro metrics</h2><div class=\"table-wrap\">"
        + _html_table(macro)
        + "</div><h2>All horizons and directions</h2><div class=\"table-wrap\">"
        + _html_table(detail)
        + "</div></body></html>"
    )
    summary = {
        "source_benchmark": str(benchmark_dir.resolve()),
        "target_dataset": str(dataset.resolve()),
        "calibration_profiles": len(split.calibration_ids),
        "test_profiles": len(split.test_ids),
        "num_scenarios": num_scenarios,
        "methods": ["lightgbm", *METHODS],
        "kpi_kinds": list(KPI_KINDS),
        "macro_metrics": macro.replace({np.nan: None}).to_dict(orient="records"),
    }
    (destination / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-scenarios", type=int, default=32)
    parser.add_argument("--num-profile-plots", type=int, default=3)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", *STAGES),
        default=("all",),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.num_scenarios < 2:
        raise ValueError("--num-scenarios must be at least 2")
    if args.num_profile_plots < 0:
        raise ValueError("--num-profile-plots cannot be negative")
    benchmark_dir = args.benchmark_dir.resolve()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe, split, artifacts = _load_source(benchmark_dir)
    recipe = json.loads(json.dumps(recipe))
    recipe["dataset_path"] = str(dataset)
    recipe["evaluation"]["num_scenarios"] = int(args.num_scenarios)
    stages = set(STAGES if "all" in args.stages else args.stages)
    if "score" in stages:
        _score_models(
            benchmark_dir=benchmark_dir,
            dataset=dataset,
            output_dir=output_dir,
            recipe=recipe,
            split=split,
            artifacts=artifacts,
            num_scenarios=args.num_scenarios,
            num_profile_plots=args.num_profile_plots,
            resume=args.resume,
        )
    if "lightgbm" in stages:
        _predict_lightgbm(
            benchmark_dir=benchmark_dir,
            dataset=dataset,
            output_dir=output_dir,
            recipe=recipe,
            split=split,
            source_artifact=artifacts["probabilistic_structured"],
            num_scenarios=args.num_scenarios,
            resume=args.resume,
        )
    if "calibrate" in stages:
        _calibrate(output_dir=output_dir, recipe=recipe)
    report_path = output_dir / "reports" / "transfer_dashboard.html"
    if "report" in stages:
        report_path = _write_report(
            benchmark_dir=benchmark_dir,
            dataset=dataset,
            output_dir=output_dir,
            split=split,
            num_scenarios=args.num_scenarios,
        )
    return report_path


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
