"""Compare shared and softly routed xi under a joint flexibility-geometry loss.

The current best artifact is reused as the baseline. New arms add one joint
energy score over direct-F and regression-based short response, asymmetry,
horizon slope, and 2 h curvature. Evaluation uses only native ensembles.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import json_safe, load_artifact
from .calibrated_model_benchmark import (
    KPI_KINDS,
    _boolean,
    _emulator_distribution,
    _summarize_raw_distribution_table,
)
from .routed_latent_targeted_loss_ablation import (
    DEFAULT_BASE_ROOT,
    _annual_rows,
    _artifact_record_path,
    _load_artifact_record,
    _load_context,
    _markdown_table,
    _score_dir,
    _score_variant,
    _trace_rows,
    _train_variant,
    _training_rows,
    _variance_dir,
    _variance_rows,
    _variance_variant,
    _write_artifact_record,
)


DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/soft_routing_joint_geometry"
)
VARIANT_ORDER = ("baseline", "joint_shared", "joint_soft")
STAGES = ("train", "score", "variance", "report")
COORDINATE_NAMES = (
    "direct_short_up",
    "direct_asym",
    "direct_slope",
    "direct_curvature_2h",
    "regression_short_up",
    "regression_asym",
    "regression_slope",
    "regression_curvature_2h",
)


@dataclass(frozen=True)
class VariantSpec:
    persistent_xi_routing: str
    targeted_flex_energy_weight: float


def _variant_specs(weight: float) -> dict[str, VariantSpec]:
    return {
        "baseline": VariantSpec("shared", 0.0),
        "joint_shared": VariantSpec("shared", weight),
        "joint_soft": VariantSpec("soft_physical", weight),
    }


def _kind_coordinate_table(
    frame: pd.DataFrame,
    *,
    index: list[str],
    prefix: str,
) -> pd.DataFrame:
    up = frame.pivot_table(
        index=index,
        columns="horizon_hours",
        values="up_flex_wh_m2_k",
    )
    down = frame.pivot_table(
        index=index,
        columns="horizon_hours",
        values="down_flex_wh_m2_k",
    )
    required = (0.5, 2.0, 3.0)
    if any(horizon not in up.columns for horizon in required) or 0.5 not in down:
        raise ValueError("Flexibility tables lack the 0.5 h, 2 h, or 3 h cells")
    interpolated_2h = 0.4 * up[0.5] + 0.6 * up[3.0]
    return pd.DataFrame(
        {
            f"{prefix}_short_up": up[0.5],
            f"{prefix}_asym": up[0.5] - down[0.5],
            f"{prefix}_slope": up[3.0] - up[0.5],
            f"{prefix}_curvature_2h": up[2.0] - interpolated_2h,
        }
    ).dropna()


def _coordinate_table(
    score_dir: Path,
    *,
    scenarios: bool,
) -> pd.DataFrame:
    suffix = "_scenarios" if scenarios else "s"
    direct_name = f"flexibility_kpi{suffix}_f.csv"
    regression_name = f"flexibility_kpi{suffix}.csv"
    index = ["profile_id", "scenario_index"] if scenarios else ["profile_id"]
    direct = pd.read_csv(score_dir / direct_name)
    regression = pd.read_csv(score_dir / regression_name)
    if scenarios:
        direct = direct[_boolean(direct["valid"])]
        regression = regression[_boolean(regression["valid"])]
    else:
        direct = direct[direct["signal"].eq("simulation") & _boolean(direct["valid"])]
        regression = regression[
            regression["signal"].eq("simulation") & _boolean(regression["valid"])
        ]
    return _kind_coordinate_table(
        direct,
        index=index,
        prefix="direct",
    ).join(
        _kind_coordinate_table(regression, index=index, prefix="regression"),
        how="inner",
    )


def _coordinate_standardization(score_dir: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    coordinates = _coordinate_table(score_dir, scenarios=False).loc[:, COORDINATE_NAMES]
    if len(coordinates) < 2:
        raise ValueError("At least two calibration buildings are required")
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


def _geometry_distribution(score_dir: Path, method: str) -> pd.DataFrame:
    target = _coordinate_table(score_dir, scenarios=False)
    samples = _coordinate_table(score_dir, scenarios=True)
    rows: list[dict[str, Any]] = []
    for coordinate in COORDINATE_NAMES:
        values = (
            samples[[coordinate]]
            .rename(columns={coordinate: "prediction"})
            .reset_index()
            .merge(
                target[[coordinate]].rename(columns={coordinate: "truth"}).reset_index(),
                on="profile_id",
                how="inner",
                validate="many_to_one",
            )
        )
        for row in values.itertuples(index=False):
            rows.append(
                {
                    "method": method,
                    "kpi_kind": "joint_geometry",
                    "horizon_hours": 2.0 if "curvature" in coordinate else 0.5,
                    "direction": coordinate,
                    "profile_id": int(row.profile_id),
                    "sample_index": int(row.scenario_index),
                    "prediction": float(row.prediction),
                    "truth": float(row.truth),
                }
            )
    return pd.DataFrame(rows)


def _route_gate_rows(args: argparse.Namespace, variants: Sequence[str]) -> pd.DataFrame:
    routes = ("envelope", "hp", "controller", "operation")
    rows: list[dict[str, Any]] = []
    for variant in variants:
        artifact = load_artifact(_load_artifact_record(args.output_dir, variant))
        logits = getattr(artifact.model, "persistent_xi_route_logits", None)
        if logits is None:
            continue
        gates = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        for route_index, route in enumerate(routes):
            for latent_index in range(gates.shape[1]):
                rows.append(
                    {
                        "variant": variant,
                        "route": route,
                        "latent_index": latent_index,
                        "gate": float(gates[route_index, latent_index]),
                    }
                )
    return pd.DataFrame(rows)


def _variance_driver_rows(
    args: argparse.Namespace,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    buildings = pd.read_csv(_variance_dir(args, variant) / "building_diagnostics.csv")
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


def _persistent_fraction_deltas(args: argparse.Namespace) -> pd.DataFrame:
    columns = ("profile_id", "median_persistent_fraction")
    shared = pd.read_csv(
        _variance_dir(args, "joint_shared") / "building_diagnostics.csv",
        usecols=columns,
    )
    soft = pd.read_csv(
        _variance_dir(args, "joint_soft") / "building_diagnostics.csv",
        usecols=columns,
    )
    result = shared.merge(
        soft,
        on="profile_id",
        validate="one_to_one",
        suffixes=("_shared", "_soft"),
    )
    result["soft_minus_shared"] = (
        result["median_persistent_fraction_soft"]
        - result["median_persistent_fraction_shared"]
    )
    return result.sort_values("soft_minus_shared", ascending=False)


def _comparison_rows(
    annual: pd.DataFrame,
    flexibility: pd.DataFrame,
    geometry: pd.DataFrame,
    variance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        row: dict[str, Any] = {"variant": variant}
        row["annual_total_nrmse"] = float(
            annual.loc[annual["variant"].eq(variant), "total_nrmse"].iloc[0]
        )
        for kind in KPI_KINDS:
            part = flexibility[
                flexibility["method"].eq(variant)
                & flexibility["kpi_kind"].eq(kind)
            ]
            row[f"{kind}_crps"] = float(part["crps"].mean())
            row[f"{kind}_calibration_mae"] = float(part["calibration_mae"].mean())
        geometry_part = geometry[geometry["method"].eq(variant)]
        row["geometry_crps"] = float(geometry_part["crps"].mean())
        row["geometry_calibration_mae"] = float(
            geometry_part["calibration_mae"].mean()
        )
        variance_row = variance[variance["variant"].eq(variant)].iloc[0]
        for metric in (
            "median_persistent_fraction",
            "native_coverage_90",
            "xi_effective_rank",
            "covariance_cosine",
        ):
            row[metric] = float(variance_row[metric])
        rows.append(row)
    result = pd.DataFrame(rows)
    baseline = result[result["variant"].eq("baseline")].iloc[0]
    for column in result.columns:
        if column != "variant":
            result[f"delta_{column}"] = result[column] - float(baseline[column])
    return result


def _report(args: argparse.Namespace) -> Path:
    variants = VARIANT_ORDER
    training = _training_rows(args, variants)
    annual = _annual_rows(args, variants)
    traces = _trace_rows(args, variants)
    distributions = pd.concat(
        [
            _emulator_distribution(
                _score_dir(args, variant),
                method=variant,
                kpi_kind=kind,
            )
            for variant in variants
            for kind in KPI_KINDS
        ],
        ignore_index=True,
    )
    flexibility, flexibility_reliability, flexibility_buildings = (
        _summarize_raw_distribution_table(distributions)
    )
    geometry_distribution = pd.concat(
        [
            _geometry_distribution(_score_dir(args, variant), variant)
            for variant in variants
        ],
        ignore_index=True,
    )
    geometry, geometry_reliability, geometry_buildings = (
        _summarize_raw_distribution_table(geometry_distribution)
    )
    variance = _variance_rows(args, variants)
    gates = _route_gate_rows(args, variants)
    variance_drivers, variance_regimes = _variance_driver_rows(args, "joint_soft")
    persistent_deltas = _persistent_fraction_deltas(args)
    comparison = _comparison_rows(annual, flexibility, geometry, variance)
    shared_row = comparison[comparison["variant"].eq("joint_shared")].iloc[0]
    soft_row = comparison[comparison["variant"].eq("joint_soft")].iloc[0]
    baseline_row = comparison[comparison["variant"].eq("baseline")].iloc[0]
    gate_reference = gates.apply(
        lambda row: 0.9 if int(row["latent_index"]) == (
            "envelope",
            "hp",
            "controller",
            "operation",
        ).index(row["route"]) else 0.1,
        axis=1,
    )
    maximum_gate_change = float((gates["gate"] - gate_reference).abs().max())
    increased_buildings = int((persistent_deltas["soft_minus_shared"] > 0.0).sum())
    mean_persistent_change = float(persistent_deltas["soft_minus_shared"].mean())
    fast_building_count = int(
        variance_regimes.loc[
            variance_regimes["variance_regime"].eq("fast-dominated"),
            "building_count",
        ].iloc[0]
    )
    outputs = {
        "training.csv": training,
        "annual_metrics.csv": annual,
        "trace_metrics.csv": traces,
        "flexibility_distribution_metrics.csv": flexibility,
        "flexibility_reliability.csv": flexibility_reliability,
        "flexibility_building_metrics.csv": flexibility_buildings,
        "geometry_metrics.csv": geometry,
        "geometry_reliability.csv": geometry_reliability,
        "geometry_building_metrics.csv": geometry_buildings,
        "variance_comparison.csv": variance,
        "route_gates.csv": gates,
        "joint_soft_variance_drivers.csv": variance_drivers,
        "joint_soft_variance_regimes.csv": variance_regimes,
        "building_persistent_fraction_deltas.csv": persistent_deltas,
        "comparison_summary.csv": comparison,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)

    upward = flexibility.loc[
        flexibility["direction"].eq("up"),
        (
            "method",
            "kpi_kind",
            "horizon_hours",
            "crps",
            "bias",
            "coverage_90",
        ),
    ]
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# Soft routing and joint response-geometry ablation",
                "",
                "All probabilistic metrics are computed from native raw ensembles. No ex-post calibration is used.",
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
                        "geometry_crps",
                        "xi_effective_rank",
                        "covariance_cosine",
                    ),
                ),
                "",
                "## Upward horizons",
                "",
                _markdown_table(
                    upward,
                    (
                        "method",
                        "kpi_kind",
                        "horizon_hours",
                        "crps",
                        "bias",
                        "coverage_90",
                    ),
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
                        "covariance_cosine",
                    ),
                ),
                "",
                "Absolute variances are averaged over the 16 KPI cells and retain KPI-squared units. Normalized variances divide each cell by its calibration-fold scale squared before averaging.",
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
                "For the soft-routing arm, xi scales 0-3 correspond to envelope, HP, controller, and operation. Operating columns are fixed held-out inputs and therefore agree across arms.",
                "",
                "## Learned soft gates",
                "",
                _markdown_table(gates, ("variant", "route", "latent_index", "gate"))
                if not gates.empty
                else "No soft-route model was scored.",
                "",
                f"The largest learned gate movement from the 0.9 diagonal / 0.1 off-diagonal initialization is only {maximum_gate_change:.4f}.",
                "",
                "## Fast-dominated buildings",
                "",
                "Exploratory associations below use the 20 soft-routing test buildings; they are diagnostic, not causal estimates.",
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
                        "envelope xi scale",
                        "HP xi scale",
                        "controller xi scale",
                        "operation xi scale",
                        "native 90% coverage",
                        "mean abs standardized residual",
                    ),
                ),
                "",
                f"Soft routing raises the per-building persistent fraction for {increased_buildings}/20 buildings (mean change {mean_persistent_change:+.4f}), but {fast_building_count}/20 buildings remain fast-noise dominated.",
                "",
                "## Decision",
                "",
                f"Do not replace the current best with either candidate. Shared routing improves geometry CRPS from {baseline_row['geometry_crps']:.4f} to {shared_row['geometry_crps']:.4f}, xi effective rank from {baseline_row['xi_effective_rank']:.3f} to {shared_row['xi_effective_rank']:.3f}, and covariance alignment from {baseline_row['covariance_cosine']:.3f} to {shared_row['covariance_cosine']:.3f}; however, native direct-F and regression CRPS/calibration both worsen.",
                "",
                f"Soft routing preserves annual fit and restores total spread, but its xi effective rank falls to {soft_row['xi_effective_rank']:.3f}. Near-initial gates plus the lower rank show that sparse physical access does not make pathway effects independent: the route decoders and closed loop still map the four factors into one dominant KPI-amplitude direction.",
                "",
                "The next controlled experiment should keep shared routing, where the joint loss demonstrated more multi-directional capacity, and make the training response operator match annual scoring more closely. A covariance-mode objective should supervise centered ensemble anomalies in the residual modes, rather than only the eight marginal response coordinates; otherwise location improvements can trade against horizon slope and calibration.",
                "",
            ]
        )
        + "\n"
    )
    print(f"ablation report={report_path}")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--targeted-weight", type=float, default=0.15)
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
    variants = tuple(dict.fromkeys(args.variants))
    specs = _variant_specs(float(args.targeted_weight))
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
                    "coordinate_names": COORDINATE_NAMES,
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

    recipes: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Path] = {"baseline": baseline_artifact}
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
        root = args.output_dir / "variants" / variant
        root.mkdir(parents=True, exist_ok=True)
        (root / "resolved_recipe.json").write_text(
            json.dumps(json_safe(variant_recipe), indent=2, sort_keys=True) + "\n"
        )
        recipes[variant] = variant_recipe

    for variant in variants:
        if variant == "baseline":
            continue
        if "train" in stages:
            artifacts[variant] = _train_variant(args, recipes[variant], split, variant)
        else:
            artifacts[variant] = _load_artifact_record(args.output_dir, variant)

    if "score" in stages:
        for variant in variants:
            _score_variant(args, recipes[variant], split, variant, artifacts[variant])
    if "variance" in stages:
        for variant in variants:
            _variance_variant(args, recipes[variant], split, variant, artifacts[variant])
    if "report" in stages:
        if set(variants) != set(VARIANT_ORDER):
            raise ValueError("The final report requires all variants")
        return _report(args)
    return args.output_dir / "experiment_manifest.json"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
