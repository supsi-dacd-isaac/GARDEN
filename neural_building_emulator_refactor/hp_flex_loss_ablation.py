"""Sequential ablation of HP training mode and flexibility-aware scores.

The experiment keeps the probabilistic structured architecture fixed and trains
the requested variants on one shared building split, including:

* ``expected_event``: expected HP power with the joint event-response score;
* ``hard_event``: hard HP forward samples with expected-path gradients and the
  same event-response score;
* ``hard_event_w2``: the same hard HP path with event-response weight increased
  from 0.5 to 2.0;
* ``hard_event_w2_direct_f``: event-response weight 2.0 plus a
  trajectory-consistent direct-F CRPS with weight 2.5;
* ``hard_event_w2_direct_f_grouped``: replace ordinary minibatches with grouped
  seasonal windows sharing persistent ``xi``;
* ``hard_event_w2_direct_f_aux``: retain independent ordinary minibatches and
  add a grouped flex-only update every four steps;
* ``hard_direct_f``: hard HP forward samples with trajectory-consistent CRPS on
  the direct-ratio flexibility KPI.

Each arm is scored on autonomous annual setpoint and force-off rollouts. The
script then fits residual-normalized calibration on the held-out calibration
buildings and writes a consolidated HTML dashboard and machine-readable tables.
All expensive stages are resumable.
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .artifacts import json_safe, load_artifact
from .calibrated_model_benchmark import (
    KPI_KINDS,
    BenchmarkSplit,
    _eligible_hp_ids,
    _emulator_distribution,
    _force_off_dataset_path,
    _force_off_emulator_distribution,
    _load_or_create_split,
    _point_flexibility_metrics,
    _summarize_raw_distribution_table,
)
from .fair_probabilistic_lstm_comparison import _experiment_configs
from .force_off_policy_evaluation import evaluate_force_off_policy
from .probabilistic_calibration import calibrate_distribution_table
from .profiles import saved_profile_ids
from .scoring import score_artifact
from .trainer import train


DEFAULT_CONFIG = (
    Path(__file__).parent
    / "configs"
    / "calibrated_four_model_benchmark_metadata_fixed_three_energy_scores_both.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/hp_flex_loss_ablation_200"
)
STAGES = ("split", "train", "score", "force_off", "report")
VARIANT_ORDER = (
    "expected_event",
    "hard_event",
    "hard_event_w2",
    "hard_event_w2_direct_f",
    "hard_event_w2_direct_f_grouped",
    "hard_event_w2_direct_f_aux",
    "hard_direct_f",
)
VARIANT_COLORS = {
    "expected_event": "#1769aa",
    "hard_event": "#d95f02",
    "hard_event_w2": "#7b3294",
    "hard_event_w2_direct_f": "#c51b7d",
    "hard_event_w2_direct_f_grouped": "#00838f",
    "hard_event_w2_direct_f_aux": "#5e35b1",
    "hard_direct_f": "#16846b",
}
VARIANT_LABELS = {
    "expected_event": "Expected HP + event ES",
    "hard_event": "Hard HP + event ES",
    "hard_event_w2": "Hard HP + event ES (weight 2)",
    "hard_event_w2_direct_f": "Hard HP + event ES (2) + direct-F CRPS (2.5)",
    "hard_event_w2_direct_f_grouped": (
        "Hard HP + event ES (2) + pooled direct-F CRPS (4 seasonal windows)"
    ),
    "hard_event_w2_direct_f_aux": (
        "Hard HP + event ES (2) + auxiliary pooled direct-F CRPS"
    ),
    "hard_direct_f": "Hard HP + direct-F CRPS",
}


@dataclass(frozen=True)
class VariantSpec:
    hp_training_mode: str
    event_energy_weight: float
    flex_kpi_crps_weight: float
    flex_kpi_crps_estimator: str = "direct_ratio"
    flex_kpi_crps_min_events: int = 8
    flex_kpi_crps_windows_per_profile: int = 1
    flex_kpi_crps_aux_every_steps: int = 0
    flex_kpi_crps_aux_buildings: int = 4


VARIANT_SPECS = {
    "expected_event": VariantSpec(
        hp_training_mode="expected",
        event_energy_weight=0.5,
        flex_kpi_crps_weight=0.0,
    ),
    "hard_event": VariantSpec(
        hp_training_mode="hard_expected_grad",
        event_energy_weight=0.5,
        flex_kpi_crps_weight=0.0,
    ),
    "hard_event_w2": VariantSpec(
        hp_training_mode="hard_expected_grad",
        event_energy_weight=2.0,
        flex_kpi_crps_weight=0.0,
    ),
    "hard_event_w2_direct_f": VariantSpec(
        hp_training_mode="hard_expected_grad",
        event_energy_weight=2.0,
        flex_kpi_crps_weight=2.5,
    ),
    "hard_event_w2_direct_f_grouped": VariantSpec(
        hp_training_mode="hard_expected_grad",
        event_energy_weight=2.0,
        flex_kpi_crps_weight=2.5,
        flex_kpi_crps_windows_per_profile=4,
    ),
    "hard_event_w2_direct_f_aux": VariantSpec(
        hp_training_mode="hard_expected_grad",
        event_energy_weight=2.0,
        flex_kpi_crps_weight=2.5,
        flex_kpi_crps_windows_per_profile=4,
        flex_kpi_crps_aux_every_steps=4,
        flex_kpi_crps_aux_buildings=4,
    ),
    "hard_direct_f": VariantSpec(
        hp_training_mode="hard_expected_grad",
        event_energy_weight=0.0,
        flex_kpi_crps_weight=0.1,
    ),
}


def _variant_recipe(base: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one fully resolved recipe while changing only ablated settings."""
    if name not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant {name!r}")
    recipe = copy.deepcopy(base)
    settings = recipe["probabilistic_structured"]
    spec = VARIANT_SPECS[name]

    # Freeze the stable architecture used by the latest 1000-building run.
    settings["contracting_temperature_feedback_gradient_scale"] = 0.05
    settings["contracting_metadata_fixed_state_update"] = "linear"
    settings["prob_hp_scenario_mode"] = "bernoulli"
    settings["prob_hp_training_mode"] = spec.hp_training_mode
    settings["prob_event_energy_weight"] = spec.event_energy_weight
    settings["prob_flex_kpi_crps_weight"] = spec.flex_kpi_crps_weight
    settings["prob_flex_kpi_crps_estimator"] = spec.flex_kpi_crps_estimator
    settings["prob_flex_kpi_crps_min_events"] = spec.flex_kpi_crps_min_events
    settings["prob_flex_kpi_crps_windows_per_profile"] = (
        spec.flex_kpi_crps_windows_per_profile
    )
    settings["prob_flex_kpi_crps_aux_every_steps"] = (
        spec.flex_kpi_crps_aux_every_steps
    )
    settings["prob_flex_kpi_crps_aux_buildings"] = (
        spec.flex_kpi_crps_aux_buildings
    )
    settings["prob_flex_kpi_crps_horizons_hours"] = [0.5, 1.0, 2.0, 3.0]
    settings["prob_flex_kpi_crps_setpoint_threshold_c"] = 0.05
    return recipe


def _resolved_base(args: argparse.Namespace) -> dict[str, Any]:
    recipe = json.loads(args.config.read_text())
    recipe["max_profiles"] = int(args.max_profiles)
    recipe["optimizer"]["epochs"] = int(args.epochs)
    recipe["evaluation"]["num_scenarios"] = int(args.num_scenarios)
    recipe["evaluation"]["num_profile_plots"] = int(args.num_profile_plots)
    recipe["evaluation"]["num_policy_profile_plots"] = int(
        args.num_profile_plots
    )
    recipe.setdefault("force_off_evaluation", {})["raw_only"] = True
    recipe.setdefault("poor_excitation_evaluation", {})["enabled"] = False
    return recipe


def _shared_split(
    args: argparse.Namespace,
    recipe: dict[str, Any],
) -> BenchmarkSplit:
    return _load_or_create_split(
        args.output_dir / "split_manifest.json",
        dataset_path=Path(recipe["dataset_path"]),
        additional_dataset_paths=tuple(
            Path(path) for path in recipe.get("additional_dataset_paths", ())
        ),
        max_profiles=int(recipe["max_profiles"]),
        seed=int(recipe["seed"]),
        split_config=recipe["split"],
        resume=args.resume,
    )


def _artifact_record_path(root: Path, variant: str) -> Path:
    return root / "variants" / variant / "artifact.json"


def _load_artifact_record(root: Path, variant: str) -> Path:
    path = _artifact_record_path(root, variant)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained artifact recorded for {variant!r}; run the train stage first"
        )
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
    record_path = _artifact_record_path(args.output_dir, variant)
    if args.resume and record_path.exists():
        artifact_dir = _load_artifact_record(args.output_dir, variant)
        print(f"ablation_variant={variant} stage=train action=reuse artifact={artifact_dir}")
        return artifact_dir

    variant_root = args.output_dir / "variants" / variant
    model_root = variant_root / "models"
    eligible = set(_eligible_hp_ids(Path(recipe["dataset_path"])))
    excluded = tuple(
        sorted(eligible.difference({*split.train_ids, *split.validation_ids}))
    )
    configs = _experiment_configs(
        recipe,
        model_root,
        max_profiles=len(split.train_ids) + len(split.validation_ids),
        epochs=int(args.epochs),
        fixed_test_profile_ids=split.validation_ids,
        excluded_profile_ids=excluded,
    )
    started = time.perf_counter()
    artifact_dir = Path(train(configs["probabilistic_structured"]))
    elapsed_minutes = (time.perf_counter() - started) / 60.0

    artifact = load_artifact(artifact_dir)
    if saved_profile_ids(artifact, "train") != split.train_ids:
        raise ValueError(f"{variant} training IDs differ from the shared split")
    if saved_profile_ids(artifact, "test") != split.validation_ids:
        raise ValueError(f"{variant} validation IDs differ from the shared split")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "artifact_dir": str(artifact_dir),
                "training_minutes": elapsed_minutes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"ablation_variant={variant} stage=train action=complete "
        f"minutes={elapsed_minutes:.2f}"
    )
    return artifact_dir


def _score_request(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    artifact_dir: Path,
    profile_ids: Sequence[int],
    split_name: str,
) -> dict[str, Any]:
    evaluation = recipe["evaluation"]
    return {
        "artifact_dir": str(artifact_dir.resolve()),
        "profile_ids": [int(value) for value in profile_ids],
        "split_name": split_name,
        "num_scenarios": int(args.num_scenarios),
        "hp_scenario_mode": "bernoulli",
        "ventilation_rollout_mode": str(evaluation["ventilation_rollout_mode"]),
        "horizons_hours": [float(value) for value in evaluation["horizons_hours"]],
        "min_setpoint_change": float(evaluation["min_setpoint_change"]),
        "min_events": int(evaluation["min_events"]),
        "controls": str(evaluation["controls"]),
        "seed": int(recipe["seed"]) + (0 if split_name == "calibration" else 1000),
        "num_profile_plots": int(args.num_profile_plots) if split_name == "test" else 0,
    }


def _score_variant(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    variant: str,
    artifact_dir: Path,
) -> None:
    for split_name, profile_ids in (
        ("calibration", split.calibration_ids),
        ("test", split.test_ids),
    ):
        destination = args.output_dir / "variants" / variant / "scoring" / split_name
        request = _score_request(
            args,
            recipe,
            artifact_dir,
            profile_ids,
            split_name,
        )
        request_path = destination / "request.json"
        summary_path = destination / "summary.json"
        if (
            args.resume
            and summary_path.exists()
            and request_path.exists()
            and json.loads(request_path.read_text()) == request
        ):
            print(
                f"ablation_variant={variant} stage=score:{split_name} action=reuse"
            )
            continue
        destination.mkdir(parents=True, exist_ok=True)
        score_artifact(
            artifact_dir,
            dataset_path=Path(recipe["dataset_path"]),
            output_dir=destination,
            profile_ids=profile_ids,
            kpi_mode="scenario_average",
            num_scenarios=int(args.num_scenarios),
            hp_scenario_mode="bernoulli",
            ventilation_rollout_mode=str(
                recipe["evaluation"]["ventilation_rollout_mode"]
            ),
            horizons_hours=tuple(
                float(value) for value in recipe["evaluation"]["horizons_hours"]
            ),
            min_setpoint_change=float(
                recipe["evaluation"]["min_setpoint_change"]
            ),
            min_events=int(recipe["evaluation"]["min_events"]),
            kpi_estimator="regression",
            controls=str(recipe["evaluation"]["controls"]),
            seed=request["seed"],
            num_profile_plots=request["num_profile_plots"],
            evaluate_direct_f=True,
        )
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        print(
            f"ablation_variant={variant} stage=score:{split_name} action=complete"
        )


def _score_force_off_variant(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    variant: str,
    artifact_dir: Path,
) -> None:
    dataset_path = _force_off_dataset_path(recipe)
    if dataset_path is None:
        print(
            f"ablation_variant={variant} stage=force_off action=skip "
            "reason=no_force_off_dataset"
        )
        return
    destination = args.output_dir / "variants" / variant / "force_off" / "test"
    request = {
        "artifact_dir": str(artifact_dir.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "profile_ids": [int(value) for value in split.test_ids],
        "num_scenarios": int(args.num_scenarios),
        "num_eval_particles": int(
            recipe["probabilistic_structured"].get("prob_eval_particles", 16)
        ),
        "hp_scenario_mode": "bernoulli",
        "ventilation_rollout_mode": str(
            recipe["evaluation"]["ventilation_rollout_mode"]
        ),
        "kpi_horizons_hours": [
            float(value) for value in recipe["evaluation"]["horizons_hours"]
        ],
        "kpi_min_events": int(recipe["evaluation"]["min_events"]),
        "seed": int(recipe["seed"]) + 2000,
    }
    request_path = destination / "request.json"
    summary_path = destination / "force_off_policy_summary.json"
    if (
        args.resume
        and summary_path.exists()
        and request_path.exists()
        and json.loads(request_path.read_text()) == request
    ):
        print(f"ablation_variant={variant} stage=force_off action=reuse")
        return
    destination.mkdir(parents=True, exist_ok=True)
    evaluate_force_off_policy(
        artifact_dir,
        dataset_path=dataset_path,
        output_dir=destination,
        profile_ids=split.test_ids,
        num_eval_particles=request["num_eval_particles"],
        num_scenarios=int(args.num_scenarios),
        hp_scenario_mode="bernoulli",
        ventilation_rollout_mode=request["ventilation_rollout_mode"],
        kpi_horizons_hours=request["kpi_horizons_hours"],
        kpi_min_events=request["kpi_min_events"],
        num_profile_plots=int(args.num_profile_plots),
        evaluate_ungated=False,
        seed=request["seed"],
    )
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    print(f"ablation_variant={variant} stage=force_off action=complete")


def _legacy_metadata(artifact_dir: Path) -> dict[str, Any]:
    metadata = load_artifact(artifact_dir).metadata
    return dict(metadata.get("legacy_metadata", metadata))


def _training_row(root: Path, variant: str, artifact_dir: Path) -> dict[str, Any]:
    metadata = _legacy_metadata(artifact_dir)
    record = json.loads(_artifact_record_path(root, variant).read_text())
    diagnostics = list(artifact_dir.parent.glob("epoch_diagnostics*.csv"))
    max_grad = np.nan
    last_grad = np.nan
    max_transition_grad = np.nan
    if diagnostics:
        frame = pd.read_csv(diagnostics[-1])
        if "max_grad_norm" in frame:
            max_grad = float(frame["max_grad_norm"].max())
            last_grad = float(frame["max_grad_norm"].iloc[-1])
        if "max_grad_transition" in frame:
            max_transition_grad = float(frame["max_grad_transition"].max())
    test_metrics = metadata.get("test_metrics", {})
    return {
        "variant": variant,
        "checkpoint_epoch": metadata.get("checkpoint_epoch"),
        "checkpoint_metric": metadata.get("checkpoint_metric"),
        "checkpoint_metric_value": metadata.get("checkpoint_metric_value"),
        "test_total_nrmse": test_metrics.get("total_nrmse"),
        "test_temperature_rmse_c": test_metrics.get("rmse_c"),
        "test_qroom_rmse_w_m2": test_metrics.get("qroom_rmse_w_m2"),
        "test_pel_rmse_w_m2": test_metrics.get("pel_rmse_w_m2"),
        "max_grad_norm": max_grad,
        "last_grad_norm": last_grad,
        "max_transition_grad": max_transition_grad,
        "training_minutes": record.get("training_minutes"),
    }


def _annual_rows(root: Path, variants: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        path = root / "variants" / variant / "scoring" / "test" / "profile_metrics.csv"
        frame = pd.read_csv(path)
        row: dict[str, Any] = {"variant": variant, "profile_count": len(frame)}
        for column in (
            "temperature_c_rmse",
            "temperature_c_mae",
            "temperature_c_bias",
            "qroom_w_m2_rmse",
            "qroom_w_m2_mae",
            "qroom_w_m2_bias",
            "pel_w_m2_rmse",
            "pel_w_m2_mae",
            "pel_w_m2_bias",
            "temperature_nrmse",
            "qroom_nrmse",
            "pel_nrmse",
            "total_nrmse",
            "temperature_increment_spectral_js",
            "qroom_increment_spectral_js",
            "pel_increment_spectral_js",
        ):
            row[column] = float(frame[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _trace_rows(root: Path, variants: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        score_dir = root / "variants" / variant / "scoring" / "test"
        reliability = pd.read_csv(score_dir / "temporal_trace_calibration.csv")
        scores = pd.read_csv(score_dir / "temporal_trace_probabilistic_scores.csv")
        for signal, group in reliability.groupby("signal", sort=False):
            score = scores[scores["signal"].eq(signal)].iloc[0]
            lo = group.loc[
                (group["quantile"] - 0.05).abs().idxmin(), "empirical_coverage"
            ]
            hi = group.loc[
                (group["quantile"] - 0.95).abs().idxmin(), "empirical_coverage"
            ]
            rows.append(
                {
                    "variant": variant,
                    "signal": signal,
                    "rank_calibration_mae": float(
                        group["abs_calibration_error"].mean()
                    ),
                    "central_90_coverage": float(hi - lo),
                    "mean_crps": float(score["mean_crps"]),
                    "mean_central_90_width": float(
                        score["mean_central_90_width"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _flexibility_tables(
    root: Path,
    variants: Sequence[str],
    recipe: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    point_rows: list[dict[str, Any]] = []
    for variant in variants:
        for kpi_kind in KPI_KINDS:
            calibration_frames.append(
                _emulator_distribution(
                    root / "variants" / variant / "scoring" / "calibration",
                    method=variant,
                    kpi_kind=kpi_kind,
                )
            )
            test_frames.append(
                _emulator_distribution(
                    root / "variants" / variant / "scoring" / "test",
                    method=variant,
                    kpi_kind=kpi_kind,
                )
            )
            point_rows.extend(
                _point_flexibility_metrics(
                    root / "variants" / variant / "scoring" / "test",
                    arm=variant,
                    kpi_kind=kpi_kind,
                )
            )
    _, metrics, reliability, _ = calibrate_distribution_table(
        pd.concat(calibration_frames, ignore_index=True),
        pd.concat(test_frames, ignore_index=True),
        minimum_scale=float(recipe["calibration"]["minimum_scale"]),
        minimum_scale_fraction=float(
            recipe["calibration"]["minimum_scale_fraction"]
        ),
    )
    return metrics, reliability, pd.DataFrame(point_rows)


def _force_off_tables(
    root: Path,
    variants: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_rows: list[pd.DataFrame] = []
    distributions: list[pd.DataFrame] = []
    for variant in variants:
        score_dir = root / "variants" / variant / "force_off" / "test"
        aggregate_path = score_dir / "force_off_aggregate_metrics.csv"
        if not aggregate_path.exists():
            continue
        aggregate = pd.read_csv(aggregate_path)
        aggregate = aggregate[aggregate["model"].eq("structured_gated")].copy()
        aggregate.insert(0, "variant", variant)
        signal_rows.append(aggregate)
        for estimator, kpi_kind in (
            ("direct_ratio", "force_off_f"),
            ("regression", "force_off_regression"),
        ):
            distributions.append(
                _force_off_emulator_distribution(
                    score_dir,
                    method=variant,
                    source_model="structured_gated",
                    estimator=estimator,
                    kpi_kind=kpi_kind,
                )
            )
    signals = pd.concat(signal_rows, ignore_index=True) if signal_rows else pd.DataFrame()
    if not distributions:
        return signals, pd.DataFrame()
    metrics, _, _ = _summarize_raw_distribution_table(
        pd.concat(distributions, ignore_index=True)
    )
    return signals, metrics


def _macro_flexibility(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "nrmse",
        "rmse",
        "mae",
        "bias",
        "crps",
        "calibration_mae",
        "coverage_90",
        "width_90",
    ]
    return (
        metrics.groupby(
            ["method", "kpi_kind", "calibration"],
            as_index=False,
            sort=False,
        )[columns]
        .mean()
        .rename(columns={"method": "variant"})
    )


def _rank_components(
    annual: pd.DataFrame,
    traces: pd.DataFrame,
    flex_distribution: pd.DataFrame,
    points: pd.DataFrame,
    force_flex: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an auditable balanced-rank assessment without mixing metric units."""
    rows: list[dict[str, Any]] = []

    def add(
        frame: pd.DataFrame,
        *,
        domain: str,
        metric: str,
        groups: Sequence[str],
        variant_column: str,
    ) -> None:
        required = {variant_column, metric, *groups}
        if frame.empty or not required.issubset(frame.columns):
            return
        selected = frame[[variant_column, *groups, metric]].dropna().copy()
        if selected.empty:
            return
        group_key: str | list[str] = list(groups)
        grouped = (
            selected.groupby(group_key, dropna=False, sort=False)
            if groups
            else [((), selected)]
        )
        for key, group in grouped:
            if group[variant_column].nunique() < 2:
                continue
            keys = key if isinstance(key, tuple) else (key,)
            comparison = ", ".join(
                f"{name}={value}" for name, value in zip(groups, keys)
            ) or "overall"
            ranked = group.assign(rank=group[metric].rank(method="average"))
            for row in ranked.itertuples(index=False):
                rows.append(
                    {
                        "variant": str(getattr(row, variant_column)),
                        "domain": domain,
                        "metric": metric,
                        "comparison": comparison,
                        "value": float(getattr(row, metric)),
                        "rank": float(row.rank),
                    }
                )

    add(
        annual,
        domain="annual_point_accuracy",
        metric="total_nrmse",
        groups=(),
        variant_column="variant",
    )
    for metric in ("rank_calibration_mae", "mean_crps"):
        add(
            traces,
            domain="temporal_probabilistic_quality",
            metric=metric,
            groups=("signal",),
            variant_column="variant",
        )
    add(
        points,
        domain="flexibility_point_accuracy",
        metric="nrmse",
        groups=("kpi_kind", "horizon_hours", "direction"),
        variant_column="method",
    )
    raw_flex = flex_distribution[
        flex_distribution.get("calibration", pd.Series(dtype=str)).eq("raw")
    ]
    for metric in ("crps", "calibration_mae"):
        add(
            raw_flex,
            domain="flexibility_probabilistic_quality",
            metric=metric,
            groups=("kpi_kind", "horizon_hours", "direction"),
            variant_column="method",
        )
    add(
        force_flex,
        domain="force_off_probabilistic_quality",
        metric="crps",
        groups=("kpi_kind", "horizon_hours", "direction"),
        variant_column="method",
    )

    components = pd.DataFrame(rows)
    if components.empty:
        return components, pd.DataFrame(
            columns=["variant", "balanced_mean_rank", "domain_wins"]
        )
    domain_scores = (
        components.groupby(["variant", "domain"], as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "domain_mean_rank"})
    )
    domain_minimum = domain_scores.groupby("domain")["domain_mean_rank"].transform(
        "min"
    )
    domain_scores["domain_win"] = np.isclose(
        domain_scores["domain_mean_rank"], domain_minimum
    )
    assessment = (
        domain_scores.groupby("variant", as_index=False)
        .agg(
            balanced_mean_rank=("domain_mean_rank", "mean"),
            domain_wins=("domain_win", "sum"),
            domains_scored=("domain", "nunique"),
        )
        .sort_values(["balanced_mean_rank", "variant"])
    )
    return components, assessment


def _write_dashboard(
    path: Path,
    training: pd.DataFrame,
    annual: pd.DataFrame,
    traces: pd.DataFrame,
    flex: pd.DataFrame,
    reliability: pd.DataFrame,
    points: pd.DataFrame,
    force_flex: pd.DataFrame,
    assessment: pd.DataFrame,
) -> None:
    figure = make_subplots(
        rows=4,
        cols=3,
        specs=[
            [{"type": "table", "colspan": 3}, None, None],
            [{}, {}, {}],
            [{}, {}, {}],
            [{}, {}, {}],
        ],
        subplot_titles=(
            "Training and selected checkpoint",
            "Annual autonomous NRMSE",
            "Trace rank calibration",
            "Trace CRPS",
            "Flexibility point NRMSE",
            "Flexibility CRPS",
            "Flexibility calibration error",
            "Flexibility 90% coverage",
            "Force-off flexibility CRPS",
            "Direct-F RNS reliability",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        row_heights=(0.16, 0.27, 0.27, 0.30),
    )
    table_columns = [
        "variant",
        "checkpoint_epoch",
        "test_total_nrmse",
        "max_grad_norm",
        "training_minutes",
    ]
    display = training[table_columns].merge(
        assessment[["variant", "balanced_mean_rank"]],
        on="variant",
        how="left",
    )
    table_columns.append("balanced_mean_rank")
    display["variant"] = display["variant"].map(VARIANT_LABELS)
    figure.add_trace(
        go.Table(
            header={"values": table_columns, "fill_color": "#e8eef5"},
            cells={"values": [display[column] for column in table_columns]},
        ),
        row=1,
        col=1,
    )

    for variant in VARIANT_ORDER:
        annual_part = annual[annual["variant"].eq(variant)]
        if annual_part.empty:
            continue
        row = annual_part.iloc[0]
        common = {
            "name": VARIANT_LABELS[variant],
            "legendgroup": variant,
            "marker_color": VARIANT_COLORS[variant],
        }
        figure.add_trace(
            go.Bar(
                x=["Tin", "Qroom", "Pel"],
                y=[
                    row["temperature_nrmse"],
                    row["qroom_nrmse"],
                    row["pel_nrmse"],
                ],
                **common,
            ),
            row=2,
            col=1,
        )
        trace_part = traces[traces["variant"].eq(variant)]
        figure.add_trace(
            go.Bar(
                x=trace_part["signal"],
                y=trace_part["rank_calibration_mae"],
                showlegend=False,
                **{key: value for key, value in common.items() if key != "name"},
                name=VARIANT_LABELS[variant],
            ),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Bar(
                x=trace_part["signal"],
                y=trace_part["mean_crps"],
                showlegend=False,
                **{key: value for key, value in common.items() if key != "name"},
                name=VARIANT_LABELS[variant],
            ),
            row=2,
            col=3,
        )

    point_macro = (
        points.groupby(["method", "kpi_kind"], as_index=False)["nrmse"].mean()
        if not points.empty
        else pd.DataFrame()
    )
    for variant in VARIANT_ORDER:
        color = VARIANT_COLORS[variant]
        point_part = point_macro[point_macro["method"].eq(variant)]
        figure.add_trace(
            go.Bar(
                x=point_part["kpi_kind"],
                y=point_part["nrmse"],
                name=VARIANT_LABELS[variant],
                legendgroup=variant,
                marker_color=color,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        for calibration, pattern in (("raw", ""), ("rns", "/")):
            part = flex[
                flex["variant"].eq(variant)
                & flex["calibration"].eq(calibration)
            ]
            labels = [f"{kind} {calibration}" for kind in part["kpi_kind"]]
            marker = {"color": color, "pattern": {"shape": pattern}}
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=part["crps"],
                    name=f"{VARIANT_LABELS[variant]} {calibration}",
                    legendgroup=variant,
                    marker=marker,
                    showlegend=False,
                ),
                row=3,
                col=2,
            )
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=part["calibration_mae"],
                    name=f"{VARIANT_LABELS[variant]} {calibration}",
                    legendgroup=variant,
                    marker=marker,
                    showlegend=False,
                ),
                row=3,
                col=3,
            )
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=part["coverage_90"],
                    name=f"{VARIANT_LABELS[variant]} {calibration}",
                    legendgroup=variant,
                    marker=marker,
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

        force_part = (
            force_flex[
                force_flex["method"].eq(variant)
                & force_flex["kpi_kind"].eq("force_off_f")
            ]
            if {"method", "kpi_kind"}.issubset(force_flex.columns)
            else pd.DataFrame()
        )
        if not force_part.empty:
            figure.add_trace(
                go.Bar(
                    x=[VARIANT_LABELS[variant]],
                    y=[force_part["crps"].mean()],
                    marker_color=color,
                    name=VARIANT_LABELS[variant],
                    legendgroup=variant,
                    showlegend=False,
                ),
                row=4,
                col=2,
            )

        rel = reliability[
            reliability["method"].eq(variant)
            & reliability["kpi_kind"].eq("f")
            & reliability["calibration"].eq("rns")
        ]
        if not rel.empty:
            rel = rel.groupby("quantile", as_index=False)["empirical_coverage"].mean()
            figure.add_trace(
                go.Scatter(
                    x=rel["quantile"],
                    y=rel["empirical_coverage"],
                    mode="lines+markers",
                    line={"color": color},
                    name=VARIANT_LABELS[variant],
                    legendgroup=variant,
                    showlegend=False,
                ),
                row=4,
                col=3,
            )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": "black", "dash": "dash"},
            showlegend=False,
        ),
        row=4,
        col=3,
    )
    figure.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0.9,
        y1=0.9,
        xref="x7 domain",
        yref="y7",
        line={"color": "black", "dash": "dash"},
    )
    figure.update_yaxes(title_text="NRMSE", row=2, col=1)
    figure.update_yaxes(title_text="Mean |rank CDF error|", row=2, col=2)
    figure.update_yaxes(title_text="CRPS", row=2, col=3)
    figure.update_yaxes(title_text="NRMSE", row=3, col=1)
    figure.update_yaxes(title_text="CRPS", row=3, col=2)
    figure.update_yaxes(title_text="Mean |coverage - nominal|", row=3, col=3)
    figure.update_yaxes(title_text="Coverage", range=[0, 1.05], row=4, col=1)
    figure.update_yaxes(title_text="CRPS", row=4, col=2)
    figure.update_xaxes(title_text="Nominal quantile", range=[0, 1], row=4, col=3)
    figure.update_yaxes(title_text="Empirical coverage", range=[0, 1], row=4, col=3)
    figure.update_layout(
        title="HP sampling and flexibility-loss ablation",
        template="plotly_white",
        barmode="group",
        height=1800,
        width=1700,
        legend={"orientation": "h", "y": -0.10},
        hoverlabel={"namelength": -1, "align": "left"},
        margin={"b": 180},
    )
    figure.write_html(path)


def _best_label(frame: pd.DataFrame, column: str) -> str:
    valid = frame.dropna(subset=[column])
    if valid.empty:
        return "not available"
    row = valid.loc[valid[column].idxmin()]
    return f"{VARIANT_LABELS.get(str(row['variant']), row['variant'])} ({row[column]:.4f})"


def _write_summary(
    path: Path,
    training: pd.DataFrame,
    annual: pd.DataFrame,
    traces: pd.DataFrame,
    flex: pd.DataFrame,
    assessment: pd.DataFrame,
) -> None:
    trace_macro = traces.groupby("variant", as_index=False)[
        "rank_calibration_mae"
    ].mean()
    flex_rns = flex[flex["calibration"].eq("rns")]
    direct = flex_rns[flex_rns["kpi_kind"].eq("f")]
    regression = flex_rns[flex_rns["kpi_kind"].eq("regression_based")]
    balanced = (
        "not available"
        if assessment.empty
        else (
            f"{VARIANT_LABELS.get(str(assessment.iloc[0]['variant']), assessment.iloc[0]['variant'])} "
            f"(mean rank {assessment.iloc[0]['balanced_mean_rank']:.3f}; "
            f"{int(assessment.iloc[0]['domain_wins'])} domain wins)"
        )
    )
    lines = [
        "# HP Sampling And Flexibility-Loss Ablation",
        "",
        "All variants use the same architecture, building split, optimizer, and evaluation seeds.",
        "",
        "## Automatic Winners",
        "",
        f"- Best balanced rank across evaluation domains: {balanced}",
        f"- Lowest selected validation NRMSE: {_best_label(training, 'test_total_nrmse')}",
        f"- Lowest annual Tin RMSE: {_best_label(annual.rename(columns={'temperature_c_rmse': 'metric'}), 'metric')}",
        f"- Lowest mean trace rank-calibration error: {_best_label(trace_macro.rename(columns={'rank_calibration_mae': 'metric'}), 'metric')}",
        f"- Lowest direct-F RNS CRPS: {_best_label(direct.rename(columns={'crps': 'metric'}), 'metric')}",
        f"- Lowest regression-KPI RNS CRPS: {_best_label(regression.rename(columns={'crps': 'metric'}), 'metric')}",
        "",
        "The balanced rank gives every evaluation domain equal weight and never mixes raw metric units. It is a screening recommendation, not a significance test; inspect the domain tables before promoting a model.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _report(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    variants: Sequence[str],
) -> Path:
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        variant: _load_artifact_record(args.output_dir, variant)
        for variant in variants
    }
    training = pd.DataFrame(
        [
            _training_row(args.output_dir, variant, artifact_dir)
            for variant, artifact_dir in artifacts.items()
        ]
    )
    annual = _annual_rows(args.output_dir, variants)
    traces = _trace_rows(args.output_dir, variants)
    distribution, reliability, points = _flexibility_tables(
        args.output_dir,
        variants,
        recipe,
    )
    flex = _macro_flexibility(distribution)
    force_signals, force_flex = _force_off_tables(args.output_dir, variants)
    assessment_components, assessment = _rank_components(
        annual,
        traces,
        distribution,
        points,
        force_flex,
    )

    tables = {
        "training_summary.csv": training,
        "annual_signal_metrics.csv": annual,
        "temporal_trace_metrics.csv": traces,
        "flexibility_distribution_metrics.csv": distribution,
        "flexibility_distribution_macro.csv": flex,
        "flexibility_reliability.csv": reliability,
        "flexibility_point_metrics.csv": points,
        "force_off_signal_metrics.csv": force_signals,
        "force_off_flexibility_metrics.csv": force_flex,
        "automatic_assessment_components.csv": assessment_components,
        "automatic_assessment.csv": assessment,
    }
    for name, frame in tables.items():
        frame.to_csv(report_dir / name, index=False)
    dashboard = report_dir / "comparison_dashboard.html"
    _write_dashboard(
        dashboard,
        training,
        annual,
        traces,
        flex,
        reliability,
        points,
        force_flex,
        assessment,
    )
    _write_summary(
        report_dir / "comparison_summary.md",
        training,
        annual,
        traces,
        flex,
        assessment,
    )
    manifest = {
        "variants": list(variants),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "tables": list(tables),
        "dashboard": dashboard.name,
        "summary": "comparison_summary.md",
    }
    (report_dir / "report_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    print(f"saved_hp_flex_ablation_dashboard={dashboard}")
    return dashboard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-profiles", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-scenarios", type=int, default=50)
    parser.add_argument("--num-profile-plots", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=VARIANT_ORDER,
    )
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
    if args.max_profiles < 12:
        raise ValueError("--max-profiles must be at least 12")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.num_scenarios < 2:
        raise ValueError("--num-scenarios must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = _resolved_base(args)
    split = _shared_split(args, base)
    stages = set(STAGES if "all" in args.stages else args.stages)
    variants = tuple(dict.fromkeys(args.variants))
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(
            json_safe(
                {
                    "source_config": str(args.config),
                    "base_recipe": base,
                    "split": asdict(split),
                    "variants": {
                        name: asdict(VARIANT_SPECS[name]) for name in variants
                    },
                    "stages": list(args.stages),
                    "num_scenarios": int(args.num_scenarios),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"ablation_split train={len(split.train_ids)} "
        f"validation={len(split.validation_ids)} "
        f"calibration={len(split.calibration_ids)} test={len(split.test_ids)} "
        f"fingerprint={split.fingerprint}"
    )
    if stages == {"split"}:
        return args.output_dir / "split_manifest.json"

    for variant in variants:
        recipe = _variant_recipe(base, variant)
        variant_root = args.output_dir / "variants" / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        (variant_root / "resolved_recipe.json").write_text(
            json.dumps(json_safe(recipe), indent=2, sort_keys=True) + "\n"
        )
        if "train" in stages:
            artifact_dir = _train_variant(args, recipe, split, variant)
        else:
            artifact_dir = _load_artifact_record(args.output_dir, variant)
        if "score" in stages:
            _score_variant(args, recipe, split, variant, artifact_dir)
        if "force_off" in stages:
            _score_force_off_variant(args, recipe, split, variant, artifact_dir)

    if "report" in stages:
        return _report(args, base, variants)
    return args.output_dir / "experiment_manifest.json"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
