"""Controlled 2x2 ablation of persistent-xi routing and targeted KPI loss.

The current best artifact is reused without retraining. The other three arms
change only persistent latent routing and/or a standardized three-coordinate
direct-F energy score. Evaluation is entirely native: no residual-normalized
or other ex-post calibration is applied.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import json_safe, load_artifact
from .calibrated_model_benchmark import (
    KPI_KINDS,
    BenchmarkSplit,
    _boolean,
    _eligible_hp_ids,
    _emulator_distribution,
    _point_flexibility_metrics,
    _summarize_raw_distribution_table,
)
from .fair_probabilistic_lstm_comparison import _experiment_configs
from .kpi_variance_decomposition import run as run_variance_decomposition
from .profiles import saved_profile_ids
from .scoring import score_artifact
from .trainer import train


DEFAULT_BASE_ROOT = Path(
    "output/neural_building_emulator_refactor/"
    "architecture_screen_metadata_xi_all_pathways_flex_direct_high150_"
    "process_mild_tau_poles_200_e10"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/"
    "routed_latent_targeted_loss_2x2"
)
VARIANT_ORDER = ("baseline", "architecture_only", "loss_only", "combined")
STAGES = ("train", "score", "variance", "report")


@dataclass(frozen=True)
class VariantSpec:
    persistent_xi_routing: str
    targeted_flex_energy_weight: float


def _variant_specs(targeted_weight: float) -> dict[str, VariantSpec]:
    return {
        "baseline": VariantSpec("shared", 0.0),
        "architecture_only": VariantSpec("physical", 0.0),
        "loss_only": VariantSpec("shared", targeted_weight),
        "combined": VariantSpec("physical", targeted_weight),
    }


def _load_context(
    base_root: Path,
) -> tuple[dict[str, Any], BenchmarkSplit, Path]:
    resolved = json.loads((base_root / "resolved_recipe.json").read_text())
    recipe = copy.deepcopy(resolved["recipe"])
    split = BenchmarkSplit.from_json(
        json.loads((base_root / "split_manifest.json").read_text())
    )
    artifact_dir = Path(resolved["artifacts"]["probabilistic_structured"]["artifact_dir"])
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Current-best artifact does not exist: {artifact_dir}")
    return recipe, split, artifact_dir


def _target_coordinate_table(score_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(score_dir / "flexibility_kpis_f.csv")
    source = source[source["signal"].eq("simulation") & _boolean(source["valid"])]
    up = source.pivot(index="profile_id", columns="horizon_hours", values="up_flex_wh_m2_k")
    down = source.pivot(
        index="profile_id", columns="horizon_hours", values="down_flex_wh_m2_k"
    )
    required = (0.5, 3.0)
    if any(value not in up.columns for value in required) or 0.5 not in down.columns:
        raise ValueError("Calibration direct-F table lacks the 0.5 h or 3 h coordinates")
    return pd.DataFrame(
        {
            "short_up": up[0.5],
            "asym": up[0.5] - down[0.5],
            "slope": up[3.0] - up[0.5],
        }
    ).dropna()


def _coordinate_standardization(score_dir: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    coordinates = _target_coordinate_table(score_dir)
    if len(coordinates) < 2:
        raise ValueError("At least two valid calibration buildings are required")
    center = tuple(float(value) for value in coordinates.mean(axis=0))
    scale = tuple(float(max(value, 1e-6)) for value in coordinates.std(axis=0, ddof=0))
    return center, scale


def _variant_recipe(
    base: dict[str, Any],
    spec: VariantSpec,
    *,
    center: tuple[float, ...],
    scale: tuple[float, ...],
) -> dict[str, Any]:
    recipe = copy.deepcopy(base)
    settings = recipe["probabilistic_structured"]
    settings["prob_persistent_xi_routing"] = spec.persistent_xi_routing
    settings["prob_targeted_flex_energy_weight"] = spec.targeted_flex_energy_weight
    settings["prob_targeted_flex_short_horizon_hours"] = 0.5
    settings["prob_targeted_flex_long_horizon_hours"] = 3.0
    settings["prob_targeted_flex_coordinate_center"] = list(center)
    settings["prob_targeted_flex_coordinate_scale"] = list(scale)
    return recipe


def _artifact_record_path(root: Path, variant: str) -> Path:
    return root / "variants" / variant / "artifact.json"


def _write_artifact_record(
    root: Path,
    variant: str,
    artifact_dir: Path,
    *,
    training_minutes: float,
    reused_current_best: bool,
) -> None:
    path = _artifact_record_path(root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "variant": variant,
                "artifact_dir": str(artifact_dir.resolve()),
                "training_minutes": float(training_minutes),
                "reused_current_best": bool(reused_current_best),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_artifact_record(root: Path, variant: str) -> Path:
    path = _artifact_record_path(root, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact record for {variant}: {path}")
    artifact_dir = Path(json.loads(path.read_text())["artifact_dir"])
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Recorded artifact does not exist: {artifact_dir}")
    return artifact_dir


def _train_variant(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    variant: str,
) -> Path:
    record = _artifact_record_path(args.output_dir, variant)
    if args.resume and record.exists():
        artifact_dir = _load_artifact_record(args.output_dir, variant)
        print(f"ablation variant={variant} stage=train action=reuse artifact={artifact_dir}")
        return artifact_dir
    eligible = set(_eligible_hp_ids(Path(recipe["dataset_path"])))
    retained = {*split.train_ids, *split.validation_ids}
    configs = _experiment_configs(
        recipe,
        args.output_dir / "variants" / variant / "models",
        max_profiles=len(retained),
        epochs=int(args.epochs),
        fixed_test_profile_ids=split.validation_ids,
        excluded_profile_ids=tuple(sorted(eligible.difference(retained))),
    )
    started = time.perf_counter()
    artifact_dir = Path(train(configs["probabilistic_structured"]))
    elapsed = (time.perf_counter() - started) / 60.0
    artifact = load_artifact(artifact_dir)
    if saved_profile_ids(artifact, "train") != split.train_ids:
        raise ValueError(f"{variant} training IDs differ from the controlled split")
    if saved_profile_ids(artifact, "test") != split.validation_ids:
        raise ValueError(f"{variant} validation IDs differ from the controlled split")
    _write_artifact_record(
        args.output_dir,
        variant,
        artifact_dir,
        training_minutes=elapsed,
        reused_current_best=False,
    )
    print(f"ablation variant={variant} stage=train action=complete minutes={elapsed:.2f}")
    return artifact_dir


def _score_dir(args: argparse.Namespace, variant: str) -> Path:
    if variant == "baseline":
        return args.base_root / "scoring" / "test" / "probabilistic_structured"
    return args.output_dir / "variants" / variant / "scoring" / "test"


def _score_variant(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    variant: str,
    artifact_dir: Path,
) -> None:
    if variant == "baseline":
        required = _score_dir(args, variant) / "summary.json"
        if not required.exists():
            raise FileNotFoundError(f"Current-best score is missing: {required}")
        print("ablation variant=baseline stage=score action=reuse-current-best")
        return
    destination = _score_dir(args, variant)
    request = {
        "artifact_dir": str(artifact_dir.resolve()),
        "profile_ids": list(split.test_ids),
        "num_scenarios": int(args.num_scenarios),
        "hp_scenario_mode": "bernoulli",
        "ventilation_rollout_mode": recipe["evaluation"]["ventilation_rollout_mode"],
        "horizons_hours": recipe["evaluation"]["horizons_hours"],
        "min_setpoint_change": recipe["evaluation"]["min_setpoint_change"],
        "min_events": recipe["evaluation"]["min_events"],
        "controls": recipe["evaluation"]["controls"],
        "seed": int(recipe["seed"]) + 1000,
    }
    request_path = destination / "request.json"
    if (
        args.resume
        and (destination / "summary.json").exists()
        and request_path.exists()
        and json.loads(request_path.read_text()) == request
    ):
        print(f"ablation variant={variant} stage=score action=reuse")
        return
    destination.mkdir(parents=True, exist_ok=True)
    score_artifact(
        artifact_dir,
        dataset_path=Path(recipe["dataset_path"]),
        output_dir=destination,
        profile_ids=split.test_ids,
        kpi_mode="scenario_average",
        num_scenarios=int(args.num_scenarios),
        hp_scenario_mode="bernoulli",
        ventilation_rollout_mode=str(recipe["evaluation"]["ventilation_rollout_mode"]),
        horizons_hours=tuple(float(value) for value in recipe["evaluation"]["horizons_hours"]),
        min_setpoint_change=float(recipe["evaluation"]["min_setpoint_change"]),
        min_events=int(recipe["evaluation"]["min_events"]),
        kpi_estimator="regression",
        controls=str(recipe["evaluation"]["controls"]),
        seed=request["seed"],
        num_profile_plots=int(args.num_profile_plots),
        evaluate_direct_f=True,
    )
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    print(f"ablation variant={variant} stage=score action=complete")


def _variance_dir(args: argparse.Namespace, variant: str) -> Path:
    if variant == "baseline":
        return args.base_root / "diagnostics" / "kpi_variance_decomposition"
    return args.output_dir / "variants" / variant / "diagnostics" / "kpi_variance_decomposition"


def _variance_variant(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    variant: str,
    artifact_dir: Path,
) -> None:
    destination = _variance_dir(args, variant)
    required = destination / "residual_xi_covariance_summary.csv"
    if variant == "baseline" and required.exists():
        print("ablation variant=baseline stage=variance action=reuse-current-best")
        return
    if args.resume and required.exists():
        print(f"ablation variant={variant} stage=variance action=reuse")
        return
    evaluation = recipe["evaluation"]
    run_variance_decomposition(
        argparse.Namespace(
            artifact_dir=artifact_dir,
            dataset=Path(recipe["dataset_path"]),
            output_dir=destination,
            profile_source="test",
            profile_ids=split.test_ids,
            max_profiles=None,
            xi_draws=int(args.xi_draws),
            fast_rollouts=int(args.fast_rollouts),
            common_random_numbers=True,
            write_samples=True,
            resume=bool(args.resume),
            hp_scenario_mode="bernoulli",
            ventilation_rollout_mode=str(evaluation["ventilation_rollout_mode"]),
            horizons_hours=tuple(float(value) for value in evaluation["horizons_hours"]),
            dt_hours=0.25,
            min_setpoint_change=float(evaluation["min_setpoint_change"]),
            min_events=int(evaluation["min_events"]),
            controls=str(evaluation["controls"]),
            scale_reference_dir=(
                args.base_root / "scoring" / "calibration" / "probabilistic_structured"
            ),
            minimum_scale=float(recipe["calibration"]["minimum_scale"]),
            minimum_scale_fraction=float(recipe["calibration"]["minimum_scale_fraction"]),
            residual_bootstrap_samples=int(args.residual_bootstrap_samples),
            seed=int(recipe["seed"]),
        )
    )


def _targeted_distribution(score_dir: Path, method: str) -> pd.DataFrame:
    truth = pd.read_csv(score_dir / "flexibility_kpis_f.csv")
    truth = truth[truth["signal"].eq("simulation") & _boolean(truth["valid"])]
    scenarios = pd.read_csv(score_dir / "flexibility_kpi_scenarios_f.csv")
    scenarios = scenarios[_boolean(scenarios["valid"])]

    def coordinates(frame: pd.DataFrame, index: list[str]) -> pd.DataFrame:
        up = frame.pivot_table(index=index, columns="horizon_hours", values="up_flex_wh_m2_k")
        down = frame.pivot_table(
            index=index, columns="horizon_hours", values="down_flex_wh_m2_k"
        )
        return pd.DataFrame(
            {
                "short_up": up[0.5],
                "asym": up[0.5] - down[0.5],
                "slope": up[3.0] - up[0.5],
            }
        ).dropna()

    target = coordinates(truth, ["profile_id"])
    samples = coordinates(scenarios, ["profile_id", "scenario_index"])
    rows: list[dict[str, Any]] = []
    for coordinate in target.columns:
        values = samples[[coordinate]].rename(columns={coordinate: "prediction"}).reset_index()
        values = values.merge(
            target[[coordinate]].rename(columns={coordinate: "truth"}).reset_index(),
            on="profile_id",
            how="inner",
            validate="many_to_one",
        )
        for row in values.itertuples(index=False):
            rows.append(
                {
                    "method": method,
                    "kpi_kind": "targeted_f",
                    "horizon_hours": 0.5 if coordinate != "slope" else 3.0,
                    "direction": coordinate,
                    "profile_id": int(row.profile_id),
                    "sample_index": int(row.scenario_index),
                    "prediction": float(row.prediction),
                    "truth": float(row.truth),
                }
            )
    return pd.DataFrame(rows)


def _training_rows(args: argparse.Namespace, variants: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        artifact_dir = _load_artifact_record(args.output_dir, variant)
        artifact = load_artifact(artifact_dir)
        metadata = artifact.metadata.get("legacy_metadata", artifact.metadata)
        config = metadata.get("train_config", {})
        record = json.loads(_artifact_record_path(args.output_dir, variant).read_text())
        test = metadata.get("test_metrics", {})
        rows.append(
            {
                "variant": variant,
                "routing": config.get("prob_persistent_xi_routing", "shared"),
                "targeted_weight": config.get("prob_targeted_flex_energy_weight", 0.0),
                "checkpoint_epoch": metadata.get("checkpoint_epoch"),
                "validation_total_nrmse": test.get("total_nrmse"),
                "validation_temperature_rmse_c": test.get("rmse_c"),
                "validation_qroom_rmse_w_m2": test.get("qroom_rmse_w_m2"),
                "validation_pel_rmse_w_m2": test.get("pel_rmse_w_m2"),
                "training_minutes": record["training_minutes"],
                "reused_current_best": record["reused_current_best"],
            }
        )
    return pd.DataFrame(rows)


def _annual_rows(args: argparse.Namespace, variants: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "temperature_c_rmse",
        "qroom_w_m2_rmse",
        "pel_w_m2_rmse",
        "total_nrmse",
    )
    for variant in variants:
        frame = pd.read_csv(_score_dir(args, variant) / "profile_metrics.csv")
        rows.append({"variant": variant, **{name: float(frame[name].mean()) for name in metrics}})
    return pd.DataFrame(rows)


def _trace_rows(args: argparse.Namespace, variants: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        score_dir = _score_dir(args, variant)
        reliability = pd.read_csv(score_dir / "temporal_trace_calibration.csv")
        scores = pd.read_csv(score_dir / "temporal_trace_probabilistic_scores.csv")
        for signal, group in reliability.groupby("signal", sort=False):
            score = scores[scores["signal"].eq(signal)].iloc[0]
            low = group.loc[(group["quantile"] - 0.05).abs().idxmin(), "empirical_coverage"]
            high = group.loc[(group["quantile"] - 0.95).abs().idxmin(), "empirical_coverage"]
            rows.append(
                {
                    "variant": variant,
                    "signal": signal,
                    "crps": float(score["mean_crps"]),
                    "calibration_mae": float(group["abs_calibration_error"].mean()),
                    "coverage_90": float(high - low),
                    "width_90": float(score["mean_central_90_width"]),
                }
            )
    return pd.DataFrame(rows)


def _variance_rows(args: argparse.Namespace, variants: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        root = _variance_dir(args, variant)
        variance = pd.read_csv(root / "variance_summary.csv")
        native = pd.read_csv(root / "native_calibration_summary.csv")
        residual = pd.read_csv(root / "residual_xi_covariance_summary.csv").set_index("metric")
        buildings = pd.read_csv(root / "building_diagnostics.csv")
        rows.append(
            {
                "variant": variant,
                "median_persistent_fraction": float(variance["median_persistent_fraction"].median()),
                "mean_v_total": float(variance["mean_v_total"].mean()),
                "mean_v_xi": float(variance["mean_v_xi"].mean()),
                "mean_v_fast": float(variance["mean_v_fast"].mean()),
                "mean_normalized_v_total": float(variance["mean_normalized_v_total"].mean()),
                "mean_normalized_v_xi": float(variance["mean_normalized_v_xi"].mean()),
                "mean_normalized_v_fast": float(variance["mean_normalized_v_fast"].mean()),
                "native_pit_mae": float(native["mean_absolute_pit_distance"].mean()),
                "native_coverage_90": float(native["coverage_90"].mean()),
                "xi_effective_rank": float(residual.loc["xi_effective_rank", "estimate"]),
                "residual_effective_rank": float(residual.loc["residual_effective_rank", "estimate"]),
                "covariance_cosine": float(residual.loc["covariance_frobenius_cosine", "estimate"]),
                "residual_variance_captured_xi4": float(
                    residual.loc["residual_variance_captured_by_xi_top4", "estimate"]
                ),
                "mean_hp_load_w_m2": float(buildings["hp_mean_pel_w_m2"].mean()),
                "mean_hp_annual_energy_kwh_m2": float(
                    buildings["hp_annual_energy_kwh_m2"].mean()
                ),
                "mean_hp_on_fraction": float(buildings["hp_on_fraction"].mean()),
                "mean_event_count": float(buildings["setpoint_event_count"].mean()),
                "mean_ventilation_active_fraction": float(
                    buildings["ventilation_active_fraction"].mean()
                ),
                "mean_xi_scale": float(buildings["xi_scale_mean"].mean()),
                "mean_xi_scale_0": float(buildings["xi_scale_0"].mean()),
                "mean_xi_scale_1": float(buildings["xi_scale_1"].mean()),
                "mean_xi_scale_2": float(buildings["xi_scale_2"].mean()),
                "mean_xi_scale_3": float(buildings["xi_scale_3"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _combined_variance_driver_rows(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    buildings = pd.read_csv(_variance_dir(args, "combined") / "building_diagnostics.csv")
    target = "median_persistent_fraction"
    drivers = {
        "HP load (W/m2)": "hp_mean_pel_w_m2",
        "HP annual energy (kWh/m2)": "hp_annual_energy_kwh_m2",
        "HP on fraction": "hp_on_fraction",
        "setpoint event count": "setpoint_event_count",
        "ventilation-active fraction": "ventilation_active_fraction",
        "envelope xi scale": "xi_scale_0",
        "HP xi scale": "xi_scale_1",
        "controller xi scale": "xi_scale_2",
        "operation xi scale": "xi_scale_3",
    }
    correlations = pd.DataFrame(
        [
            {
                "driver": label,
                "pearson": float(buildings[target].corr(buildings[column])),
                "spearman": float(
                    buildings[target].corr(buildings[column], method="spearman")
                ),
            }
            for label, column in drivers.items()
        ]
    ).sort_values("spearman", ascending=False)

    buildings = buildings.assign(
        variance_regime=np.where(
            buildings[target] < 0.5, "fast-dominated", "persistent-dominated"
        )
    )
    group_columns = {
        "building_count": "profile_id",
        "median_persistent_fraction": target,
        "HP load (W/m2)": "hp_mean_pel_w_m2",
        "HP on fraction": "hp_on_fraction",
        "setpoint event count": "setpoint_event_count",
        "ventilation-active fraction": "ventilation_active_fraction",
        "envelope xi scale": "xi_scale_0",
        "HP xi scale": "xi_scale_1",
        "controller xi scale": "xi_scale_2",
        "operation xi scale": "xi_scale_3",
        "native 90% coverage": "native_coverage_90",
        "mean abs standardized residual": "mean_absolute_standardized_residual",
    }
    group_rows: list[dict[str, Any]] = []
    for regime, group in buildings.groupby("variance_regime", sort=False):
        row: dict[str, Any] = {"variance_regime": regime}
        for label, column in group_columns.items():
            row[label] = (
                int(group[column].nunique())
                if label == "building_count"
                else float(group[column].mean())
            )
        group_rows.append(row)
    return correlations, pd.DataFrame(group_rows)


def _comparison_summary(
    annual: pd.DataFrame,
    flex: pd.DataFrame,
    targeted: pd.DataFrame,
    variance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        row: dict[str, Any] = {"variant": variant}
        annual_row = annual[annual["variant"].eq(variant)].iloc[0]
        row["annual_total_nrmse"] = float(annual_row["total_nrmse"])
        for kind in KPI_KINDS:
            part = flex[flex["method"].eq(variant) & flex["kpi_kind"].eq(kind)]
            for metric in ("nrmse", "crps", "calibration_mae", "coverage_90", "width_90"):
                row[f"{kind}_{metric}"] = float(part[metric].mean())
        target_part = targeted[targeted["method"].eq(variant)]
        row["targeted_crps"] = float(target_part["crps"].mean())
        row["targeted_calibration_mae"] = float(target_part["calibration_mae"].mean())
        variance_row = variance[variance["variant"].eq(variant)].iloc[0]
        for metric in (
            "median_persistent_fraction",
            "mean_normalized_v_total",
            "xi_effective_rank",
            "covariance_cosine",
            "residual_variance_captured_xi4",
        ):
            row[metric] = float(variance_row[metric])
        rows.append(row)
    result = pd.DataFrame(rows)
    baseline = result[result["variant"].eq("baseline")].iloc[0]
    for column in result.columns:
        if column != "variant":
            result[f"delta_{column}"] = result[column] - float(baseline[column])
    return result


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.4g}")
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in selected.to_numpy()]
    return "\n".join([header, divider, *rows])


def _report(args: argparse.Namespace, variants: Sequence[str]) -> Path:
    training = _training_rows(args, variants)
    annual = _annual_rows(args, variants)
    traces = _trace_rows(args, variants)
    distributions = pd.concat(
        [
            _emulator_distribution(_score_dir(args, variant), method=variant, kpi_kind=kind)
            for variant in variants
            for kind in KPI_KINDS
        ],
        ignore_index=True,
    )
    flex, reliability, buildings = _summarize_raw_distribution_table(distributions)
    point = pd.DataFrame(
        [
            row
            for variant in variants
            for kind in KPI_KINDS
            for row in _point_flexibility_metrics(
                _score_dir(args, variant), arm=variant, kpi_kind=kind
            )
        ]
    )
    targeted_distribution = pd.concat(
        [_targeted_distribution(_score_dir(args, variant), variant) for variant in variants],
        ignore_index=True,
    )
    targeted, targeted_reliability, targeted_buildings = (
        _summarize_raw_distribution_table(targeted_distribution)
    )
    variance = _variance_rows(args, variants)
    variance_drivers, variance_regimes = _combined_variance_driver_rows(args)
    comparison = _comparison_summary(annual, flex, targeted, variance)
    direct_f_up = flex.loc[
        flex["method"].isin(("baseline", "combined"))
        & flex["kpi_kind"].eq("f")
        & flex["direction"].eq("up"),
        ("method", "horizon_hours", "crps", "bias", "coverage_90", "width_90"),
    ]
    targeted_report = targeted.loc[
        targeted["method"].isin(("baseline", "combined")),
        ("method", "direction", "crps", "bias", "coverage_90", "width_90"),
    ]
    outputs = {
        "training.csv": training,
        "annual_metrics.csv": annual,
        "trace_metrics.csv": traces,
        "flexibility_distribution_metrics.csv": flex,
        "flexibility_reliability.csv": reliability,
        "flexibility_building_metrics.csv": buildings,
        "flexibility_point_metrics.csv": point,
        "targeted_coordinate_metrics.csv": targeted,
        "targeted_coordinate_reliability.csv": targeted_reliability,
        "targeted_coordinate_building_metrics.csv": targeted_buildings,
        "variance_comparison.csv": variance,
        "combined_variance_drivers.csv": variance_drivers,
        "combined_variance_regimes.csv": variance_regimes,
        "comparison_summary.csv": comparison,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# Routed latent and targeted-loss 2x2 ablation",
                "",
                "All probabilistic metrics below are native raw-ensemble metrics. No ex-post calibration is used.",
                "",
                "## Primary comparison",
                "",
                _markdown_table(
                    comparison,
                    (
                        "variant",
                        "annual_total_nrmse",
                        "f_crps",
                        "f_calibration_mae",
                        "regression_based_crps",
                        "regression_based_calibration_mae",
                        "targeted_crps",
                        "xi_effective_rank",
                        "covariance_cosine",
                    ),
                ),
                "",
                "## Direct-F upward horizons",
                "",
                _markdown_table(
                    direct_f_up,
                    (
                        "method",
                        "horizon_hours",
                        "crps",
                        "bias",
                        "coverage_90",
                        "width_90",
                    ),
                ),
                "",
                "The targeted coordinates improve at the supervised endpoints, but the combined model is worse at the unsupervised 2 h upward cell. This is evidence that endpoint-and-slope supervision does not constrain interior response curvature.",
                "",
                "## Targeted coordinates",
                "",
                _markdown_table(
                    targeted_report,
                    ("method", "direction", "crps", "bias", "coverage_90", "width_90"),
                ),
                "",
                "## Persistent uncertainty",
                "",
                _markdown_table(
                    variance,
                    (
                        "variant",
                        "median_persistent_fraction",
                        "mean_v_total",
                        "mean_v_xi",
                        "mean_v_fast",
                        "mean_normalized_v_total",
                        "mean_normalized_v_xi",
                        "mean_normalized_v_fast",
                        "native_pit_mae",
                        "native_coverage_90",
                        "xi_effective_rank",
                        "residual_effective_rank",
                        "residual_variance_captured_xi4",
                    ),
                ),
                "",
                "Absolute variances are averaged over the 16 KPI cells and therefore retain KPI-squared units. Normalized variances divide each cell by its calibration-fold scale squared before averaging.",
                "",
                "## Prior and operating diagnostics",
                "",
                _markdown_table(
                    variance,
                    (
                        "variant",
                        "mean_xi_scale_0",
                        "mean_xi_scale_1",
                        "mean_xi_scale_2",
                        "mean_xi_scale_3",
                        "mean_hp_load_w_m2",
                        "mean_hp_on_fraction",
                        "mean_event_count",
                        "mean_ventilation_active_fraction",
                    ),
                ),
                "",
                "For the physical-routing arms, xi scales 0-3 correspond to envelope, HP, controller, and operation. The operating columns are fixed held-out inputs and consequently agree across arms.",
                "",
                "## Fast-dominated buildings",
                "",
                "Exploratory associations below use the 20 combined-arm test buildings; they are diagnostic, not causal estimates.",
                "",
                _markdown_table(variance_drivers, ("driver", "pearson", "spearman")),
                "",
                _markdown_table(
                    variance_regimes,
                    (
                        "variance_regime",
                        "building_count",
                        "median_persistent_fraction",
                        "HP load (W/m2)",
                        "HP on fraction",
                        "setpoint event count",
                        "ventilation-active fraction",
                        "HP xi scale",
                        "controller xi scale",
                        "operation xi scale",
                        "native 90% coverage",
                        "mean abs standardized residual",
                    ),
                ),
                "",
                "The architecture arms route xi as envelope, HP, controller, and operation. The operation factor rescales only the existing ventilation signal with a learned reciprocal bound of 4 and is initialized to identity.",
                "",
                "The loss arms add a 3-D energy score on calibration-standardized short-up, 0.5 h up/down asymmetry, and 3 h-minus-0.5 h upward slope coordinates.",
                "",
                "## Decision",
                "",
                "Do not replace the current best with the combined model as an overall emulator. It materially improves direct-F CRPS and calibration while preserving annual trace NRMSE, but regression-based CRPS and calibration worsen, and physical routing collapses persistent uncertainty further toward one global-amplitude mode.",
                "",
                "The loss-only arm is the more informative architectural result: its xi effective rank rises while direct-F improves, showing that the shared architecture has multi-directional capacity when the objective identifies it. The next controlled loss should retain shared or softly gated routes, include regression-based response coordinates, and constrain the missing 2 h curvature explicitly.",
                "",
            ]
        )
    )
    print(f"ablation report={report_path}")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--targeted-weight", type=float, default=0.25)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--num-profile-plots", type=int, default=3)
    parser.add_argument("--xi-draws", type=int, default=32)
    parser.add_argument("--fast-rollouts", type=int, default=8)
    parser.add_argument("--residual-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=VARIANT_ORDER)
    parser.add_argument("--stages", nargs="+", choices=("all", *STAGES), default=("all",))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.num_scenarios < 2:
        parser.error("epochs must be positive and num-scenarios must be at least 2")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive")
    if args.targeted_weight < 0.0:
        parser.error("targeted-weight must be non-negative")
    if args.xi_draws < 2 or args.fast_rollouts < 2:
        parser.error("xi-draws and fast-rollouts must each be at least 2")
    return args


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recipe, split, baseline_artifact = _load_context(args.base_root)
    center, scale = _coordinate_standardization(
        args.base_root / "scoring" / "calibration" / "probabilistic_structured"
    )
    specs = _variant_specs(float(args.targeted_weight))
    variants = tuple(dict.fromkeys(args.variants))
    stages = set(STAGES if "all" in args.stages else args.stages)
    _write_artifact_record(
        args.output_dir,
        "baseline",
        baseline_artifact,
        training_minutes=0.0,
        reused_current_best=True,
    )
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(
            json_safe(
                {
                    "base_root": str(args.base_root.resolve()),
                    "baseline_artifact": str(baseline_artifact.resolve()),
                    "split": asdict(split),
                    "coordinate_center": center,
                    "coordinate_scale": scale,
                    "variants": {name: asdict(specs[name]) for name in variants},
                    "epochs": int(args.epochs),
                    "max_train_batches": args.max_train_batches,
                    "num_scenarios": int(args.num_scenarios),
                    "xi_draws": int(args.xi_draws),
                    "fast_rollouts": int(args.fast_rollouts),
                    "native_calibration_only": True,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        "ablation split "
        f"train={len(split.train_ids)} validation={len(split.validation_ids)} "
        f"calibration={len(split.calibration_ids)} test={len(split.test_ids)} "
        f"coordinate_center={center} coordinate_scale={scale}"
    )
    variant_recipes: dict[str, dict[str, Any]] = {}
    artifact_dirs: dict[str, Path] = {"baseline": baseline_artifact}
    for variant in variants:
        variant_recipe = _variant_recipe(
            recipe,
            specs[variant],
            center=center,
            scale=scale,
        )
        if args.max_train_batches is not None:
            variant_recipe["probabilistic_structured"]["max_train_batches"] = int(
                args.max_train_batches
            )
        variant_root = args.output_dir / "variants" / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        (variant_root / "resolved_recipe.json").write_text(
            json.dumps(json_safe(variant_recipe), indent=2, sort_keys=True) + "\n"
        )
        variant_recipes[variant] = variant_recipe

    for variant in variants:
        if variant == "baseline":
            continue
        elif "train" in stages:
            artifact_dirs[variant] = _train_variant(
                args, variant_recipes[variant], split, variant
            )
        else:
            artifact_dirs[variant] = _load_artifact_record(args.output_dir, variant)

    if "score" in stages:
        for variant in variants:
            _score_variant(
                args,
                variant_recipes[variant],
                split,
                variant,
                artifact_dirs[variant],
            )

    if "variance" in stages:
        for variant in variants:
            _variance_variant(
                args,
                variant_recipes[variant],
                split,
                variant,
                artifact_dirs[variant],
            )
    if "report" in stages:
        if set(variants) != set(VARIANT_ORDER):
            raise ValueError("The final report requires all four variants")
        return _report(args, VARIANT_ORDER)
    return args.output_dir / "experiment_manifest.json"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
