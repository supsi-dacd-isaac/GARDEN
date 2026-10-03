"""End-to-end calibrated benchmark for structured and vanilla HP emulators.

The command freezes one building-level train/validation/calibration/test split,
trains four matched models, scores autonomous annual rollouts, fits metadata-only
LightGBM models for both flexibility KPIs, calibrates scalar KPI distributions,
and writes one consolidated report. Expensive stages are resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from plotly.subplots import make_subplots
from scipy.stats import norm

from neural_building_emulator.columns import (
    CLOSED_LOOP_METADATA_COLUMNS,
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    HP_FORCE_OFF_REQUESTED_COLUMN,
    HP_SIZE_BINDING_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    PROFILE_ID_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from neural_building_emulator.data import read_dataset_frame
from neural_building_emulator.flexibility_event_study_kpis import EventStudyConfig

from .artifacts import json_safe, load_artifact
from .fair_probabilistic_lstm_comparison import (
    _check_matching_models,
    _check_matching_splits,
    _experiment_configs,
    _tree_parameter_count,
)
from .force_off_policy_evaluation import evaluate_force_off_policy
from .force_off_policy_evaluation import _policy_traces as _force_off_policy_traces
from .intervention_flexibility import estimate_force_off_flexibility_curve
from .metadata_flexibility_lightgbm import (
    TargetSpec,
    _artifact_metadata,
    _metadata_value,
    _resolve_targets,
    extract_metadata_kpis,
    regression_metrics,
)
from .probabilistic_calibration import calibrate_distribution_table
from .profiles import load_profiles, saved_profile_ids
from .scoring import score_artifact
from .trainer import train


DEFAULT_CONFIG = (
    Path(__file__).parent / "configs" / "calibrated_four_model_benchmark.json"
)
ARM_ORDER = (
    "deterministic_structured",
    "deterministic_lstm",
    "probabilistic_structured",
    "probabilistic_lstm",
)
PROBABILISTIC_ARMS = ("probabilistic_structured", "probabilistic_lstm")
KPI_KINDS = ("regression_based", "f")
FORCE_OFF_KPI_KINDS = ("force_off_regression", "force_off_f")
STAGES = (
    "split",
    "train",
    "score",
    "lightgbm",
    "force_off",
    "calibrate",
    "report",
)
MODEL_COLORS = {
    "deterministic_structured": "#1769aa",
    "deterministic_lstm": "#d95f02",
    "probabilistic_structured": "#16846b",
    "probabilistic_lstm": "#9b4fa6",
    "lightgbm": "#606b78",
}
TEMPORAL_MODEL_COLORS = {
    **MODEL_COLORS,
    "structured_gated": MODEL_COLORS["probabilistic_structured"],
    "unstructured": MODEL_COLORS["probabilistic_lstm"],
}


def _model_color(method: str) -> str:
    return TEMPORAL_MODEL_COLORS.get(method, "#606b78")


def _policy_profile_ids(
    test_ids: Sequence[int],
    *,
    policy: str,
    count: int,
) -> tuple[int, ...]:
    """Choose reproducible policy examples, disjoint when the split permits."""
    values = tuple(int(value) for value in test_ids)
    if count <= 0 or not values:
        return ()
    if policy == "setpoint":
        return values[:count]
    if policy != "force_off":
        raise ValueError(f"Unknown policy {policy!r}")
    offset = count if len(values) >= 2 * count else max(0, len(values) - count)
    return values[offset : offset + count]


@dataclass(frozen=True)
class BenchmarkSplit:
    dataset_path: str
    seed: int
    requested_max_profiles: int | None
    eligible_count: int
    selected_ids: tuple[int, ...]
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    calibration_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    fingerprint: str
    additional_dataset_paths: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> "BenchmarkSplit":
        converted = dict(values)
        for name in (
            "selected_ids",
            "train_ids",
            "validation_ids",
            "calibration_ids",
            "test_ids",
        ):
            converted[name] = tuple(int(value) for value in converted[name])
        converted["additional_dataset_paths"] = tuple(
            str(value) for value in converted.get("additional_dataset_paths", ())
        )
        return cls(**converted)


@dataclass(frozen=True)
class ForceOffTargetSpec:
    estimator: str
    horizon_hours: float
    horizon_steps: int
    direction: str = "force_off"

    @property
    def key(self) -> str:
        horizon = f"{self.horizon_hours:g}".replace(".", "p")
        return f"force_off_{self.estimator}_{horizon}h_wh_m2_intervention"


def _eligible_hp_ids(dataset_path: Path) -> tuple[int, ...]:
    frame = read_dataset_frame(
        dataset_path,
        columns=[PROFILE_ID_COLUMN, "hp_ref_capacity_W", HP_SIZE_BINDING_COLUMN],
    )
    frame[PROFILE_ID_COLUMN] = frame[PROFILE_ID_COLUMN].astype(int)
    first = frame.groupby(PROFILE_ID_COLUMN, sort=True)[
        ["hp_ref_capacity_W", HP_SIZE_BINDING_COLUMN]
    ].first()
    return tuple(
        int(profile_id)
        for profile_id, row in first.iterrows()
        if pd.notna(row["hp_ref_capacity_W"])
        and float(row["hp_ref_capacity_W"]) > 0.0
        and str(row[HP_SIZE_BINDING_COLUMN]).strip().upper() == "SH"
    )


def _split_count(total: int, fraction: float, minimum: int) -> int:
    return max(minimum, int(round(total * fraction)))


def create_benchmark_split(
    dataset_path: Path,
    *,
    additional_dataset_paths: Sequence[Path] = (),
    max_profiles: int | None,
    seed: int,
    validation_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
) -> BenchmarkSplit:
    fractions = (validation_fraction, calibration_fraction, test_fraction)
    if any(value <= 0.0 for value in fractions) or sum(fractions) >= 1.0:
        raise ValueError("Split fractions must be positive and sum to less than one")
    eligible_ids = set(_eligible_hp_ids(dataset_path))
    for additional_path in additional_dataset_paths:
        eligible_ids.intersection_update(_eligible_hp_ids(Path(additional_path)))
    eligible = np.asarray(sorted(eligible_ids), dtype=np.int64)
    if max_profiles is not None:
        if max_profiles < 12:
            raise ValueError("The calibrated benchmark requires at least 12 profiles")
        eligible_selected = min(max_profiles, len(eligible))
    else:
        eligible_selected = len(eligible)
    if eligible_selected < 12:
        raise ValueError("The dataset contains fewer than 12 eligible SH-HP profiles")
    selected = np.random.default_rng(seed).permutation(eligible)[:eligible_selected]
    n_validation = _split_count(len(selected), validation_fraction, 1)
    n_calibration = _split_count(len(selected), calibration_fraction, 3)
    n_test = _split_count(len(selected), test_fraction, 3)
    if n_validation + n_calibration + n_test >= len(selected):
        raise ValueError("Split fractions leave no training profiles")
    validation = selected[:n_validation]
    calibration = selected[n_validation : n_validation + n_calibration]
    test = selected[n_validation + n_calibration : n_validation + n_calibration + n_test]
    training = selected[n_validation + n_calibration + n_test :]
    payload = {
        "dataset_path": str(dataset_path.resolve()),
        "additional_dataset_paths": [
            str(Path(path).resolve()) for path in additional_dataset_paths
        ],
        "seed": seed,
        "selected_ids": sorted(int(value) for value in selected),
        "train_ids": sorted(int(value) for value in training),
        "validation_ids": sorted(int(value) for value in validation),
        "calibration_ids": sorted(int(value) for value in calibration),
        "test_ids": sorted(int(value) for value in test),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return BenchmarkSplit(
        dataset_path=payload["dataset_path"],
        seed=seed,
        requested_max_profiles=max_profiles,
        eligible_count=len(eligible),
        selected_ids=tuple(payload["selected_ids"]),
        train_ids=tuple(payload["train_ids"]),
        validation_ids=tuple(payload["validation_ids"]),
        calibration_ids=tuple(payload["calibration_ids"]),
        test_ids=tuple(payload["test_ids"]),
        fingerprint=fingerprint,
        additional_dataset_paths=tuple(payload["additional_dataset_paths"]),
    )


def _load_or_create_split(
    path: Path,
    *,
    dataset_path: Path,
    additional_dataset_paths: Sequence[Path] = (),
    max_profiles: int | None,
    seed: int,
    split_config: dict[str, Any],
    resume: bool,
) -> BenchmarkSplit:
    if resume and path.exists():
        result = BenchmarkSplit.from_json(json.loads(path.read_text()))
        if Path(result.dataset_path).resolve() != dataset_path.resolve():
            raise ValueError("Saved split manifest belongs to a different dataset")
        resolved_additional = tuple(
            str(Path(value).resolve()) for value in additional_dataset_paths
        )
        if result.additional_dataset_paths != resolved_additional:
            raise ValueError(
                "Saved split manifest belongs to a different set of additional "
                "policy datasets; use a new output directory or --no-resume"
            )
        if result.seed != seed or result.requested_max_profiles != max_profiles:
            raise ValueError(
                "Saved split manifest seed/max_profiles differs from this command; "
                "use a new output directory or --no-resume"
            )
        return result
    result = create_benchmark_split(
        dataset_path,
        additional_dataset_paths=additional_dataset_paths,
        max_profiles=max_profiles,
        seed=seed,
        validation_fraction=float(split_config["validation_fraction"]),
        calibration_fraction=float(split_config["calibration_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
    )
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return result


def _artifact_arguments(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "deterministic_structured": args.deterministic_structured_artifact_dir,
        "deterministic_lstm": args.deterministic_lstm_artifact_dir,
        "probabilistic_structured": args.probabilistic_structured_artifact_dir,
        "probabilistic_lstm": args.probabilistic_lstm_artifact_dir,
    }


def _train_models(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
) -> tuple[dict[str, Path], dict[str, float]]:
    model_root = args.output_dir / "models"
    excluded = tuple(
        sorted(set(_eligible_hp_ids(Path(recipe["dataset_path"]))).difference(
            {*split.train_ids, *split.validation_ids}
        ))
    )
    configs = _experiment_configs(
        recipe,
        model_root,
        max_profiles=len(split.train_ids) + len(split.validation_ids),
        epochs=args.epochs,
        fixed_test_profile_ids=split.validation_ids,
        excluded_profile_ids=excluded,
    )
    supplied = _artifact_arguments(args)
    requested_retrains = set(args.train_arms or ())
    paths_file = args.output_dir / "artifact_paths.json"
    saved_paths = json.loads(paths_file.read_text()) if args.resume and paths_file.exists() else {}
    artifact_dirs: dict[str, Path] = {}
    previous_times_path = args.output_dir / "training_times.json"
    previous_minutes = (
        json.loads(previous_times_path.read_text())
        if args.resume and previous_times_path.exists()
        else {}
    )
    durations: dict[str, float] = {}
    for arm in ARM_ORDER:
        if arm in requested_retrains:
            if supplied[arm] is not None:
                raise ValueError(
                    f"Cannot both retrain {arm!r} and supply its artifact path"
                )
            started = time.perf_counter()
            artifact_dirs[arm] = Path(train(configs[arm]))
            durations[arm] = time.perf_counter() - started
            saved_paths[arm] = str(artifact_dirs[arm])
            paths_file.write_text(
                json.dumps(saved_paths, indent=2, sort_keys=True) + "\n"
            )
            print(f"benchmark_model={arm} action=retrain")
            continue
        selected = supplied[arm]
        if selected is None and arm in saved_paths and Path(saved_paths[arm]).exists():
            selected = Path(saved_paths[arm])
        if selected is not None:
            artifact_dirs[arm] = Path(selected)
            durations[arm] = 60.0 * float(previous_minutes.get(arm, 0.0))
            print(f"benchmark_model={arm} action=reuse artifact={selected}")
            continue
        started = time.perf_counter()
        artifact_dirs[arm] = Path(train(configs[arm]))
        durations[arm] = time.perf_counter() - started
        saved_paths[arm] = str(artifact_dirs[arm])
        paths_file.write_text(json.dumps(saved_paths, indent=2, sort_keys=True) + "\n")
    _check_matching_models(artifact_dirs, configs)
    _check_matching_splits(artifact_dirs)
    for arm, path in artifact_dirs.items():
        artifact = load_artifact(path)
        if saved_profile_ids(artifact, "train") != split.train_ids:
            raise ValueError(f"{arm} training IDs do not match split_manifest.json")
        if saved_profile_ids(artifact, "test") != split.validation_ids:
            raise ValueError(f"{arm} validation IDs do not match split_manifest.json")
    paths_file.write_text(
        json.dumps({key: str(value) for key, value in artifact_dirs.items()}, indent=2, sort_keys=True)
        + "\n"
    )
    (args.output_dir / "training_times.json").write_text(
        json.dumps({key: value / 60.0 for key, value in durations.items()}, indent=2) + "\n"
    )
    return artifact_dirs, durations


def _load_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths_file = args.output_dir / "artifact_paths.json"
    values = json.loads(paths_file.read_text()) if paths_file.exists() else {}
    for arm, path in _artifact_arguments(args).items():
        if path is not None:
            values[arm] = str(path)
    missing = [arm for arm in ARM_ORDER if arm not in values]
    if missing:
        raise ValueError(f"Missing trained artifacts for: {', '.join(missing)}")
    return {arm: Path(values[arm]) for arm in ARM_ORDER}


def _score_models(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    artifact_dirs: dict[str, Path],
) -> None:
    evaluation = recipe["evaluation"]
    profile_plot_count = int(evaluation.get("num_policy_profile_plots", 3))
    setpoint_plot_ids = _policy_profile_ids(
        split.test_ids,
        policy="setpoint",
        count=profile_plot_count,
    )
    for arm, artifact_dir in artifact_dirs.items():
        artifact = load_artifact(artifact_dir)
        destinations = [("test", split.test_ids)]
        if artifact.spec.probabilistic:
            destinations.insert(0, ("calibration", split.calibration_ids))
        for split_name, profile_ids in destinations:
            destination = args.output_dir / "scoring" / split_name / arm
            summary_path = destination / "summary.json"
            plot_ids: tuple[int, ...] | None = (
                setpoint_plot_ids
                if split_name == "test" and artifact.spec.probabilistic
                else None
            )
            profile_plot_count = (
                len(plot_ids)
                if plot_ids is not None
                else (
                    int(evaluation["num_profile_plots"])
                    if split_name == "test"
                    else 0
                )
            )
            request = {
                "artifact_dir": str(artifact_dir.resolve()),
                "profile_ids": list(profile_ids),
                "probabilistic": artifact.spec.probabilistic,
                "kpi_mode": "scenario_average" if artifact.spec.probabilistic else "mean",
                "num_scenarios": int(evaluation["num_scenarios"]),
                "hp_scenario_mode": "bernoulli",
                "ventilation_rollout_mode": str(evaluation["ventilation_rollout_mode"]),
                "horizons_hours": list(evaluation["horizons_hours"]),
                "min_setpoint_change": float(evaluation["min_setpoint_change"]),
                "min_events": int(evaluation["min_events"]),
                "kpi_estimator": "regression",
                "controls": str(evaluation["controls"]),
                "evaluate_direct_f": True,
                "seed": int(recipe["seed"]) + (0 if split_name == "calibration" else 1000),
                "num_profile_plots": profile_plot_count,
            }
            if plot_ids is not None:
                request["profile_plot_ids"] = list(plot_ids)
            request_path = destination / "score_request.json"
            request_matches = (
                request_path.exists()
                and json.loads(request_path.read_text()) == request
            )
            if args.resume and summary_path.exists() and request_matches:
                print(f"benchmark_score={arm}:{split_name} action=reuse")
                continue
            for stale_plot in destination.glob("profile_*.html"):
                stale_plot.unlink()
            score_artifact(
                artifact_dir,
                dataset_path=Path(recipe["dataset_path"]),
                output_dir=destination,
                profile_ids=profile_ids,
                kpi_mode="scenario_average" if artifact.spec.probabilistic else "mean",
                num_scenarios=(int(evaluation["num_scenarios"]) if artifact.spec.probabilistic else 0),
                hp_scenario_mode="bernoulli",
                ventilation_rollout_mode=str(evaluation["ventilation_rollout_mode"]),
                horizons_hours=tuple(float(value) for value in evaluation["horizons_hours"]),
                min_setpoint_change=float(evaluation["min_setpoint_change"]),
                min_events=int(evaluation["min_events"]),
                kpi_estimator="regression",
                controls=str(evaluation["controls"]),
                seed=int(recipe["seed"]) + (0 if split_name == "calibration" else 1000),
                num_profile_plots=profile_plot_count,
                profile_plot_ids=plot_ids,
                evaluate_direct_f=True,
            )
            request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")


def _force_off_dataset_path(recipe: dict[str, Any]) -> Path | None:
    configured = recipe.get("force_off_evaluation", {}).get("dataset_path")
    if configured:
        return Path(configured)
    for value in recipe.get("additional_dataset_paths", ()):
        path = Path(value)
        if "force_off" in path.name.lower() or "force_off" in str(path).lower():
            return path
    return None


def _score_force_off_policy(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    artifact_dirs: dict[str, Path],
) -> None:
    dataset_path = _force_off_dataset_path(recipe)
    if dataset_path is None:
        print("benchmark_force_off=skipped reason=no_force_off_dataset")
        return
    evaluation = recipe["evaluation"]
    force_off_config = recipe.get("force_off_evaluation", {})
    profile_plot_count = int(evaluation.get("num_policy_profile_plots", 3))
    force_off_plot_ids = _policy_profile_ids(
        split.test_ids,
        policy="force_off",
        count=profile_plot_count,
    )
    for split_name, profile_ids in (
        ("calibration", split.calibration_ids),
        ("test", split.test_ids),
    ):
        output_dir = args.output_dir / "force_off_policy" / split_name
        request = {
            "dataset_path": str(dataset_path.resolve()),
            "structured_artifact": str(
                artifact_dirs["probabilistic_structured"].resolve()
            ),
            "unstructured_artifact": str(
                artifact_dirs["probabilistic_lstm"].resolve()
            ),
            "profile_ids": list(profile_ids),
            "num_eval_particles": int(
                recipe["probabilistic_structured"].get("prob_eval_particles", 16)
            ),
            "num_scenarios": int(evaluation["num_scenarios"]),
            "hp_scenario_mode": str(
                force_off_config.get("hp_scenario_mode", "bernoulli")
            ),
            "ventilation_rollout_mode": str(evaluation["ventilation_rollout_mode"]),
            "event_horizon_hours": float(
                force_off_config.get("event_horizon_hours", 12.0)
            ),
            "kpi_horizons_hours": [
                float(value) for value in evaluation["horizons_hours"]
            ],
            "kpi_min_events": int(evaluation["min_events"]),
            "num_profile_plots": (
                len(force_off_plot_ids) if split_name == "test" else 0
            ),
            "seed": int(recipe["seed"]) + (2000 if split_name == "test" else 1000),
        }
        if split_name == "test":
            request["profile_plot_ids"] = list(force_off_plot_ids)
        request_path = output_dir / "request.json"
        summary_path = output_dir / "force_off_policy_summary.json"
        request_matches = (
            request_path.exists() and json.loads(request_path.read_text()) == request
        )
        if args.resume and summary_path.exists() and request_matches:
            print(f"benchmark_force_off={split_name} action=reuse")
            continue
        for stale_plot in output_dir.glob("force_off_profile_*.html"):
            stale_plot.unlink()
        probabilistic_trace_dir = output_dir / "probabilistic_traces"
        if probabilistic_trace_dir.exists():
            for stale_plot in probabilistic_trace_dir.glob("force_off_profile_*.html"):
                stale_plot.unlink()
        evaluate_force_off_policy(
            artifact_dirs["probabilistic_structured"],
            dataset_path=dataset_path,
            output_dir=output_dir,
            profile_ids=profile_ids,
            num_eval_particles=request["num_eval_particles"],
            num_scenarios=request["num_scenarios"],
            hp_scenario_mode=request["hp_scenario_mode"],
            ventilation_rollout_mode=request["ventilation_rollout_mode"],
            unstructured_artifact_dir=artifact_dirs["probabilistic_lstm"],
            event_horizon_hours=request["event_horizon_hours"],
            kpi_horizons_hours=request["kpi_horizons_hours"],
            kpi_min_events=request["kpi_min_events"],
            num_profile_plots=request["num_profile_plots"],
            profile_plot_ids=request.get("profile_plot_ids"),
            seed=request["seed"],
        )
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")


def _event_config(recipe: dict[str, Any], kpi_kind: str) -> tuple[EventStudyConfig, tuple[TargetSpec, ...]]:
    evaluation = recipe["evaluation"]
    base, targets = _resolve_targets(evaluation["horizons_hours"], 0.25)
    return (
        EventStudyConfig(
            horizons_hours=base.horizons_hours,
            horizon_steps=base.horizon_steps,
            dt_hours=base.dt_hours,
            min_setpoint_change=float(evaluation["min_setpoint_change"]),
            min_events=int(evaluation["min_events"]),
            controls=("none" if kpi_kind == "f" else str(evaluation["controls"])),
            estimator=("direct_ratio" if kpi_kind == "f" else "regression"),
        ),
        targets,
    )


def _metadata_target_table(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    artifact_dir: Path,
    kpi_kind: str,
) -> tuple[pd.DataFrame, tuple[TargetSpec, ...], tuple[str, ...]]:
    destination = args.output_dir / "lightgbm" / kpi_kind
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "metadata_targets.csv"
    artifact = load_artifact(artifact_dir)
    metadata = _artifact_metadata(artifact)
    metadata_columns = tuple(metadata.get("metadata_columns", CLOSED_LOOP_METADATA_COLUMNS))
    input_columns = tuple(metadata.get("input_columns", ()))
    event_config, targets = _event_config(recipe, kpi_kind)
    request = {
        "dataset_path": str(Path(recipe["dataset_path"]).resolve()),
        "split_fingerprint": split.fingerprint,
        "reference_artifact": str(artifact_dir.resolve()),
        "metadata_columns": list(metadata_columns),
        "event_config": json_safe(asdict(event_config)),
        "input_columns": list(input_columns),
        "hp_power_area_normalization": str(
            _metadata_value(metadata, "hp_power_area_normalization", "building_heated_area")
        ),
    }
    request_path = destination / "metadata_target_request.json"
    request_matches = (
        request_path.exists() and json.loads(request_path.read_text()) == request
    )
    if args.resume and path.exists() and request_matches:
        return pd.read_csv(path), targets, metadata_columns
    table = extract_metadata_kpis(
        Path(recipe["dataset_path"]),
        train_ids=(*split.train_ids, *split.validation_ids, *split.calibration_ids),
        test_ids=split.test_ids,
        metadata_columns=metadata_columns,
        include_space_heating_availability=SPACE_HEATING_AVAILABILITY_COLUMN in input_columns,
        include_internal_gains=INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in input_columns,
        include_dhw_request=DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN in input_columns,
        hp_power_area_normalization=str(
            _metadata_value(metadata, "hp_power_area_normalization", "building_heated_area")
        ),
        event_config=event_config,
    )
    table["policy_type"] = "setpoint_modulation"
    split_lookup = {
        **{value: "train" for value in split.train_ids},
        **{value: "validation" for value in split.validation_ids},
        **{value: "calibration" for value in split.calibration_ids},
        **{value: "test" for value in split.test_ids},
    }
    table["split"] = table["profile_id"].map(split_lookup)
    table.to_csv(path, index=False)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    return table, targets, metadata_columns


def _force_off_metadata_target_table(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, tuple[ForceOffTargetSpec, ...], tuple[str, ...]]:
    dataset_path = _force_off_dataset_path(recipe)
    if dataset_path is None:
        return pd.DataFrame(), (), ()
    destination = args.output_dir / "lightgbm" / "force_off"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "metadata_targets.csv"
    artifact = load_artifact(artifact_dir)
    metadata = _artifact_metadata(artifact)
    metadata_columns = tuple(metadata.get("metadata_columns", CLOSED_LOOP_METADATA_COLUMNS))
    input_columns = tuple(metadata.get("input_columns", ()))
    horizons = tuple(float(value) for value in recipe["evaluation"]["horizons_hours"])
    targets = tuple(
        ForceOffTargetSpec(
            estimator=estimator,
            horizon_hours=horizon,
            horizon_steps=max(1, int(round(horizon / 0.25))),
        )
        for estimator in ("direct_ratio", "regression")
        for horizon in horizons
    )
    request = {
        "dataset_path": str(dataset_path.resolve()),
        "split_fingerprint": split.fingerprint,
        "reference_artifact": str(artifact_dir.resolve()),
        "metadata_columns": list(metadata_columns),
        "input_columns": list(input_columns),
        "targets": [asdict(target) for target in targets],
        "min_events": int(recipe["evaluation"]["min_events"]),
    }
    request_path = destination / "metadata_target_request.json"
    request_matches = (
        request_path.exists() and json.loads(request_path.read_text()) == request
    )
    if args.resume and path.exists() and request_matches:
        return pd.read_csv(path), targets, metadata_columns
    profile_ids = (
        *split.train_ids,
        *split.validation_ids,
        *split.calibration_ids,
        *split.test_ids,
    )
    profiles = load_profiles(
        artifact,
        dataset_path=dataset_path,
        profile_ids=profile_ids,
    )
    force_off_by_id = _force_off_policy_traces(
        dataset_path,
        profiles,
        HP_FORCE_OFF_REQUESTED_COLUMN,
    )
    availability_index = input_columns.index(SPACE_HEATING_AVAILABILITY_COLUMN)
    split_lookup = {
        **{value: "train" for value in split.train_ids},
        **{value: "validation" for value in split.validation_ids},
        **{value: "calibration" for value in split.calibration_ids},
        **{value: "test" for value in split.test_ids},
    }
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        row: dict[str, Any] = {
            "profile_id": int(profile.profile_id),
            "split": split_lookup[int(profile.profile_id)],
            **{
                column: float(value)
                for column, value in zip(metadata_columns, profile.metadata)
            },
        }
        curve = estimate_force_off_flexibility_curve(
            profile,
            profile.targets,
            force_off_by_id[profile.profile_id],
            profile.inputs[:, availability_index] > np.float32(0.5),
            input_columns=input_columns,
            horizons_hours=horizons,
            dt_hours=0.25,
            min_events=int(recipe["evaluation"]["min_events"]),
        )
        values = {
            (kpi.estimator, kpi.horizon_hours): (
                kpi.flexibility_wh_m2_per_unit if kpi.valid else np.nan
            )
            for kpi in curve
        }
        for target in targets:
            row[target.key] = values.get(
                (target.estimator, target.horizon_hours), np.nan
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    table["policy_type"] = "force_off"
    table.to_csv(path, index=False)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    return table, targets, metadata_columns


def _lgbm_parameters(config: dict[str, Any], seed: int, *, quantile: float, estimators: int) -> dict[str, Any]:
    return {
        "objective": "quantile",
        "alpha": quantile,
        "n_estimators": estimators,
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_child_samples": int(config["min_child_samples"]),
        "subsample": float(config["subsample"]),
        "subsample_freq": 1,
        "colsample_bytree": float(config["colsample_bytree"]),
        "reg_alpha": float(config["reg_alpha"]),
        "reg_lambda": float(config["reg_lambda"]),
        "random_state": seed,
        "n_jobs": int(config["n_jobs"]),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def _fit_quantile_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    target: TargetSpec | ForceOffTargetSpec,
    feature_columns: Sequence[str],
    quantile: float,
    config: dict[str, Any],
    seed: int,
) -> tuple[LGBMRegressor, int]:
    train_valid = np.isfinite(train_frame[target.key])
    validation_valid = np.isfinite(validation_frame[target.key])
    train_values = train_frame.loc[train_valid]
    validation_values = validation_frame.loc[validation_valid]
    if len(train_values) < 8 or len(validation_values) < 2:
        raise ValueError(f"Not enough valid buildings to fit {target.key}")
    maximum = int(config["max_estimators"])
    probe = LGBMRegressor(**_lgbm_parameters(config, seed, quantile=quantile, estimators=maximum))
    probe.fit(
        train_values.loc[:, feature_columns],
        train_values[target.key],
        eval_set=[(validation_values.loc[:, feature_columns], validation_values[target.key])],
        callbacks=[
            lgb.early_stopping(int(config["early_stopping_rounds"]), verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best = max(1, int(probe.best_iteration_ or maximum))
    combined = pd.concat([train_values, validation_values], ignore_index=True)
    model = LGBMRegressor(**_lgbm_parameters(config, seed, quantile=quantile, estimators=best))
    model.fit(combined.loc[:, feature_columns], combined[target.key])
    return model, best


def _normal_distribution_rows(
    frame: pd.DataFrame,
    *,
    target: TargetSpec | ForceOffTargetSpec,
    predictions: dict[float, np.ndarray],
    method: str,
    kpi_kind: str,
    sample_count: int,
    scale_floor: float,
) -> list[dict[str, Any]]:
    median = predictions[0.5]
    scale = np.maximum((predictions[0.84] - predictions[0.16]) / 2.0, scale_floor)
    probabilities = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    standard = norm.ppf(probabilities)
    rows: list[dict[str, Any]] = []
    truth = frame[target.key].to_numpy(float)
    for row_index, profile_id in enumerate(frame["profile_id"].astype(int)):
        if not np.isfinite(truth[row_index]):
            continue
        values = median[row_index] + scale[row_index] * standard
        rows.extend(
            {
                "method": method,
                "kpi_kind": kpi_kind,
                "horizon_hours": target.horizon_hours,
                "direction": target.direction,
                "profile_id": int(profile_id),
                "truth": float(truth[row_index]),
                "sample_index": int(sample_index),
                "prediction": float(value),
            }
            for sample_index, value in enumerate(values)
        )
    return rows


def _fit_lightgbm(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    reference_artifact: Path,
) -> None:
    config = recipe["lightgbm"]
    required_quantiles = tuple(float(value) for value in config["quantiles"])
    if required_quantiles != (0.16, 0.5, 0.84):
        raise ValueError("The RNS benchmark currently requires LightGBM quantiles 0.16, 0.5, 0.84")
    for kpi_kind in KPI_KINDS:
        destination = args.output_dir / "lightgbm" / kpi_kind
        calibration_path = destination / "distribution_calibration.csv"
        test_path = destination / "distribution_test.csv"
        request = {
            "split_fingerprint": split.fingerprint,
            "reference_artifact": str(reference_artifact.resolve()),
            "kpi_kind": kpi_kind,
            "lightgbm": config,
            "evaluation": recipe["evaluation"],
            "seed": int(recipe["seed"]),
        }
        request_path = destination / "fit_request.json"
        request_matches = (
            request_path.exists() and json.loads(request_path.read_text()) == request
        )
        if (
            args.resume
            and calibration_path.exists()
            and test_path.exists()
            and request_matches
        ):
            print(f"benchmark_lightgbm={kpi_kind} action=reuse")
            continue
        table, targets, metadata_columns = _metadata_target_table(
            args, recipe, split, reference_artifact, kpi_kind
        )
        frames = {
            name: table[table["split"] == name].copy()
            for name in ("train", "validation", "calibration", "test")
        }
        models_dir = destination / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        distribution_rows = {"calibration": [], "test": []}
        manifest: dict[str, Any] = {}
        for target_index, target in enumerate(targets):
            predictions: dict[str, dict[float, np.ndarray]] = {
                "calibration": {}, "test": {}
            }
            target_manifest: dict[str, Any] = {}
            for quantile_index, quantile in enumerate(required_quantiles):
                model, best = _fit_quantile_model(
                    frames["train"],
                    frames["validation"],
                    target=target,
                    feature_columns=metadata_columns,
                    quantile=quantile,
                    config=config,
                    seed=int(recipe["seed"]) + 100 * target_index + quantile_index,
                )
                filename = f"{target.key}_q{quantile:g}.txt"
                model.booster_.save_model(models_dir / filename)
                target_manifest[f"q{quantile:g}"] = {
                    "model_file": filename,
                    "best_iteration": best,
                }
                for split_name in predictions:
                    predictions[split_name][quantile] = model.predict(
                        frames[split_name].loc[:, metadata_columns]
                    )
            train_scale = float(np.nanstd(frames["train"][target.key].to_numpy(float)))
            scale_floor = max(1e-3, 0.01 * train_scale)
            for split_name in predictions:
                distribution_rows[split_name].extend(
                    _normal_distribution_rows(
                        frames[split_name],
                        target=target,
                        predictions=predictions[split_name],
                        method="lightgbm",
                        kpi_kind=kpi_kind,
                        sample_count=int(recipe["evaluation"]["num_scenarios"]),
                        scale_floor=scale_floor,
                    )
                )
            target_manifest["training_target_std"] = train_scale
            manifest[target.key] = target_manifest
        pd.DataFrame(distribution_rows["calibration"]).to_csv(calibration_path, index=False)
        pd.DataFrame(distribution_rows["test"]).to_csv(test_path, index=False)
        (destination / "model_manifest.json").write_text(
            json.dumps(
                {
                    "metadata_columns": list(metadata_columns),
                    "policy_type": "setpoint_modulation",
                    "policy_conditioning": (
                        "separate LightGBM fit for this policy/KPI; policy is "
                        "constant within the fit and is therefore not an input feature"
                    ),
                    "models": manifest,
                    "distribution_assumption": (
                        "Gaussian location-scale from LightGBM q16/q50/q84 before RNS calibration"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    _fit_force_off_lightgbm(
        args,
        recipe,
        split,
        reference_artifact,
        required_quantiles=required_quantiles,
    )


def _fit_force_off_lightgbm(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    split: BenchmarkSplit,
    reference_artifact: Path,
    *,
    required_quantiles: tuple[float, ...],
) -> None:
    if _force_off_dataset_path(recipe) is None:
        return
    config = recipe["lightgbm"]
    table, all_targets, metadata_columns = _force_off_metadata_target_table(
        args,
        recipe,
        split,
        reference_artifact,
    )
    frames = {
        name: table[table["split"] == name].copy()
        for name in ("train", "validation", "calibration", "test")
    }
    for estimator, kpi_kind in (
        ("direct_ratio", "force_off_f"),
        ("regression", "force_off_regression"),
    ):
        targets = tuple(
            target for target in all_targets if target.estimator == estimator
        )
        destination = args.output_dir / "lightgbm" / kpi_kind
        calibration_path = destination / "distribution_calibration.csv"
        test_path = destination / "distribution_test.csv"
        request = {
            "split_fingerprint": split.fingerprint,
            "reference_artifact": str(reference_artifact.resolve()),
            "kpi_kind": kpi_kind,
            "lightgbm": config,
            "evaluation": recipe["evaluation"],
            "seed": int(recipe["seed"]),
        }
        request_path = destination / "fit_request.json"
        request_matches = (
            request_path.exists() and json.loads(request_path.read_text()) == request
        )
        if (
            args.resume
            and calibration_path.exists()
            and test_path.exists()
            and request_matches
        ):
            print(f"benchmark_lightgbm={kpi_kind} action=reuse")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        models_dir = destination / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        distribution_rows: dict[str, list[dict[str, Any]]] = {
            "calibration": [],
            "test": [],
        }
        manifest: dict[str, Any] = {}
        for target_index, target in enumerate(targets):
            predictions: dict[str, dict[float, np.ndarray]] = {
                "calibration": {},
                "test": {},
            }
            target_manifest: dict[str, Any] = {}
            for quantile_index, quantile in enumerate(required_quantiles):
                model, best = _fit_quantile_model(
                    frames["train"],
                    frames["validation"],
                    target=target,
                    feature_columns=metadata_columns,
                    quantile=quantile,
                    config=config,
                    seed=(
                        int(recipe["seed"])
                        + 10_000
                        + 100 * target_index
                        + quantile_index
                    ),
                )
                filename = f"{target.key}_q{quantile:g}.txt"
                model.booster_.save_model(models_dir / filename)
                target_manifest[f"q{quantile:g}"] = {
                    "model_file": filename,
                    "best_iteration": best,
                }
                for split_name in predictions:
                    predictions[split_name][quantile] = model.predict(
                        frames[split_name].loc[:, metadata_columns]
                    )
            train_scale = float(
                np.nanstd(frames["train"][target.key].to_numpy(float))
            )
            scale_floor = max(1e-3, 0.01 * train_scale)
            for split_name in predictions:
                distribution_rows[split_name].extend(
                    _normal_distribution_rows(
                        frames[split_name],
                        target=target,
                        predictions=predictions[split_name],
                        method="lightgbm",
                        kpi_kind=kpi_kind,
                        sample_count=int(recipe["evaluation"]["num_scenarios"]),
                        scale_floor=scale_floor,
                    )
                )
            target_manifest["training_target_std"] = train_scale
            manifest[target.key] = target_manifest
        pd.DataFrame(distribution_rows["calibration"]).to_csv(
            calibration_path, index=False
        )
        pd.DataFrame(distribution_rows["test"]).to_csv(test_path, index=False)
        (destination / "model_manifest.json").write_text(
            json.dumps(
                {
                    "metadata_columns": list(metadata_columns),
                    "policy_type": "force_off",
                    "policy_conditioning": (
                        "separate LightGBM fit for this policy/KPI; policy is "
                        "constant within the fit and is therefore not an input feature"
                    ),
                    "models": manifest,
                    "distribution_assumption": (
                        "Gaussian location-scale from LightGBM q16/q50/q84 "
                        "before RNS calibration"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(("true", "1", "yes"))


def _emulator_distribution(score_dir: Path, *, method: str, kpi_kind: str) -> pd.DataFrame:
    suffix = "_f" if kpi_kind == "f" else ""
    truth_path = score_dir / f"flexibility_kpis{suffix}.csv"
    scenario_path = score_dir / f"flexibility_kpi_scenarios{suffix}.csv"
    truth = pd.read_csv(truth_path)
    scenarios = pd.read_csv(scenario_path)
    truth = truth[(truth["signal"] == "simulation") & _boolean(truth["valid"])]
    scenarios = scenarios[_boolean(scenarios["valid"])]
    keys = ["profile_id", "horizon_hours"]
    rows: list[pd.DataFrame] = []
    for direction in ("up", "down"):
        metric = f"{direction}_flex_wh_m2_k"
        target = truth[[*keys, metric]].rename(columns={metric: "truth"})
        values = scenarios[[*keys, "scenario_index", metric]].rename(
            columns={metric: "prediction", "scenario_index": "sample_index"}
        )
        values = values.merge(target, on=keys, how="inner", validate="many_to_one")
        values.insert(0, "method", method)
        values.insert(1, "kpi_kind", kpi_kind)
        values.insert(3, "direction", direction)
        rows.append(values)
    return pd.concat(rows, ignore_index=True)


def _force_off_emulator_distribution(
    score_dir: Path,
    *,
    method: str,
    source_model: str,
    estimator: str,
    kpi_kind: str,
) -> pd.DataFrame:
    truth = pd.read_csv(score_dir / "force_off_flexibility_kpis.csv")
    scenarios = pd.read_csv(
        score_dir / "force_off_flexibility_kpi_scenarios.csv"
    )
    truth = truth[
        truth["model"].eq(source_model)
        & truth["signal"].eq("simulation")
        & truth["estimator"].eq(estimator)
        & _boolean(truth["valid"])
    ]
    scenarios = scenarios[
        scenarios["model"].eq(source_model)
        & scenarios["signal"].eq("emulation")
        & scenarios["estimator"].eq(estimator)
        & _boolean(scenarios["valid"])
    ]
    keys = ["profile_id", "horizon_hours"]
    target = truth[[*keys, "flexibility_wh_m2_per_unit"]].rename(
        columns={"flexibility_wh_m2_per_unit": "truth"}
    )
    target = target.drop_duplicates(keys)
    values = scenarios[
        [*keys, "scenario_index", "flexibility_wh_m2_per_unit"]
    ].rename(
        columns={
            "scenario_index": "sample_index",
            "flexibility_wh_m2_per_unit": "prediction",
        }
    )
    values = values.merge(target, on=keys, how="inner", validate="many_to_one")
    values.insert(0, "method", method)
    values.insert(1, "kpi_kind", kpi_kind)
    values.insert(3, "direction", "force_off")
    return values


def _calibrate(args: argparse.Namespace, recipe: dict[str, Any]) -> None:
    calibration_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for kpi_kind in KPI_KINDS:
        calibration_frames.append(
            pd.read_csv(args.output_dir / "lightgbm" / kpi_kind / "distribution_calibration.csv")
        )
        test_frames.append(
            pd.read_csv(args.output_dir / "lightgbm" / kpi_kind / "distribution_test.csv")
        )
        for arm in PROBABILISTIC_ARMS:
            calibration_frames.append(
                _emulator_distribution(
                    args.output_dir / "scoring" / "calibration" / arm,
                    method=arm,
                    kpi_kind=kpi_kind,
                )
            )
            test_frames.append(
                _emulator_distribution(
                    args.output_dir / "scoring" / "test" / arm,
                    method=arm,
                    kpi_kind=kpi_kind,
                )
            )
    if _force_off_dataset_path(recipe) is not None:
        force_off_directories = {
            "calibration": args.output_dir / "force_off_policy" / "calibration",
            "test": args.output_dir / "force_off_policy" / "test",
        }
        missing = [
            str(directory)
            for directory in force_off_directories.values()
            if not (directory / "force_off_flexibility_kpi_scenarios.csv").exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Force-off calibration requires policy scoring first. Missing: "
                + ", ".join(missing)
            )
        for estimator, kpi_kind in (
            ("direct_ratio", "force_off_f"),
            ("regression", "force_off_regression"),
        ):
            calibration_frames.append(
                pd.read_csv(
                    args.output_dir
                    / "lightgbm"
                    / kpi_kind
                    / "distribution_calibration.csv"
                )
            )
            test_frames.append(
                pd.read_csv(
                    args.output_dir
                    / "lightgbm"
                    / kpi_kind
                    / "distribution_test.csv"
                )
            )
            for method, source_model in (
                ("probabilistic_structured", "structured_gated"),
                ("probabilistic_lstm", "unstructured"),
            ):
                calibration_frames.append(
                    _force_off_emulator_distribution(
                        force_off_directories["calibration"],
                        method=method,
                        source_model=source_model,
                        estimator=estimator,
                        kpi_kind=kpi_kind,
                    )
                )
                test_frames.append(
                    _force_off_emulator_distribution(
                        force_off_directories["test"],
                        method=method,
                        source_model=source_model,
                        estimator=estimator,
                        kpi_kind=kpi_kind,
                    )
                )
    calibration = pd.concat(calibration_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    destination = args.output_dir / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(destination / "raw_calibration_distributions.csv", index=False)
    test.to_csv(destination / "raw_test_distributions.csv", index=False)
    calibrated, metrics, reliability, buildings = calibrate_distribution_table(
        calibration,
        test,
        minimum_scale=float(recipe["calibration"]["minimum_scale"]),
        minimum_scale_fraction=float(recipe["calibration"]["minimum_scale_fraction"]),
    )
    calibrated.to_csv(destination / "rns_test_distributions.csv", index=False)
    metrics.to_csv(destination / "distribution_metrics.csv", index=False)
    reliability.to_csv(destination / "distribution_reliability.csv", index=False)
    buildings.to_csv(destination / "distribution_by_building.csv", index=False)
    _write_calibration_dashboard(
        destination / "kpi_calibration.html",
        metrics[metrics["kpi_kind"].isin(KPI_KINDS)],
        reliability[reliability["kpi_kind"].isin(KPI_KINDS)],
        kpi_kinds=("f", "regression_based"),
        subplot_titles=("Direct F reliability", "Regression-based reliability"),
        title="Raw and residual-normalized setpoint-flexibility calibration",
        unit="Wh/(m2 K)",
    )
    force_metrics = metrics[metrics["kpi_kind"].isin(FORCE_OFF_KPI_KINDS)]
    if not force_metrics.empty:
        _write_calibration_dashboard(
            destination / "force_off_kpi_calibration.html",
            force_metrics,
            reliability[reliability["kpi_kind"].isin(FORCE_OFF_KPI_KINDS)],
            kpi_kinds=("force_off_f", "force_off_regression"),
            subplot_titles=(
                "Direct F(force_off) reliability",
                "Regression-adjusted F(force_off) reliability",
            ),
            title="Raw and residual-normalized force-off flexibility calibration",
            unit="Wh/(m2 intervention)",
        )


def _write_calibration_dashboard(
    path: Path,
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    kpi_kinds: tuple[str, str],
    subplot_titles: tuple[str, str],
    title: str,
    unit: str,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            *subplot_titles,
            "Calibration error",
            "CRPS and sharpness",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.14,
    )
    for column, kpi_kind in enumerate(kpi_kinds, start=1):
        for (method, calibration), part in reliability[reliability["kpi_kind"] == kpi_kind].groupby(
            ["method", "calibration"], sort=False
        ):
            grouped = part.groupby("quantile", as_index=False)["empirical_coverage"].mean()
            figure.add_trace(
                go.Scatter(
                    x=grouped["quantile"],
                    y=grouped["empirical_coverage"],
                    mode="lines+markers",
                    name=f"{method} {calibration}",
                    legendgroup=f"{method}:{calibration}",
                    line={"color": _model_color(str(method)), "dash": "solid" if calibration == "raw" else "dot"},
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": "black", "dash": "dash"}, showlegend=False),
            row=1,
            col=column,
        )
    labels = metrics.apply(
        lambda row: f"{row['method']} {row['kpi_kind']} {row['calibration']}", axis=1
    )
    bar_colors = [_model_color(str(method)) for method in metrics["method"]]
    figure.add_trace(go.Bar(x=labels, y=metrics["calibration_mae"], marker_color=bar_colors, name="calibration MAE", hovertemplate="%{x}<br>Calibration MAE: %{y:.3f}<extra></extra>"), row=2, col=1)
    figure.add_trace(go.Bar(x=labels, y=metrics["crps"], marker_color=bar_colors, name="CRPS", hovertemplate="%{x}<br>CRPS: %{y:.3f}<extra></extra>"), row=2, col=2)
    figure.add_trace(
        go.Bar(x=labels, y=metrics["width_90"], marker_color=bar_colors, opacity=0.5, name="90% interval width", hovertemplate="%{x}<br>90% width: %{y:.3f}<extra></extra>"),
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="Nominal quantile", range=[0, 1], row=1)
    figure.update_yaxes(title_text="Empirical rank CDF", range=[0, 1], row=1)
    figure.update_yaxes(title_text="Mean |coverage - nominal|", row=2, col=1)
    figure.update_yaxes(title_text=unit, row=2, col=2)
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=1050,
        width=1600,
        legend={"orientation": "h", "y": -0.22},
        margin={"b": 230},
        hoverlabel={"namelength": -1, "align": "left"},
    )
    figure.write_html(path)


def _point_flexibility_metrics(
    score_dir: Path,
    *,
    arm: str,
    kpi_kind: str,
) -> list[dict[str, Any]]:
    suffix = "_f" if kpi_kind == "f" else ""
    frame = pd.read_csv(score_dir / f"flexibility_kpis{suffix}.csv")
    frame = frame[_boolean(frame["valid"])]
    rows: list[dict[str, Any]] = []
    for horizon in sorted(frame["horizon_hours"].unique()):
        for direction in ("up", "down"):
            metric = f"{direction}_flex_wh_m2_k"
            part = frame[np.isclose(frame["horizon_hours"], horizon)]
            paired = part.pivot_table(index="profile_id", columns="signal", values=metric, aggfunc="first").dropna()
            values = regression_metrics(
                paired["simulation"].to_numpy(float),
                paired["emulation"].to_numpy(float),
                train_scale=float(np.std(paired["simulation"].to_numpy(float))),
            )
            rows.append(
                {
                    "method": arm,
                    "kpi_kind": kpi_kind,
                    "horizon_hours": float(horizon),
                    "direction": direction,
                    **values,
                }
            )
    return rows


def _setpoint_intervention_table(
    score_dir: Path,
    *,
    arm: str,
    kpi_kind: str,
    scenarios: bool,
) -> pd.DataFrame:
    suffix = "_f" if kpi_kind == "f" else ""
    stem = "flexibility_kpi_scenarios" if scenarios else "flexibility_kpis"
    path = score_dir / f"{stem}{suffix}.csv"
    if not path.exists():
        return pd.DataFrame()
    source = pd.read_csv(path)
    estimator = "direct_ratio" if kpi_kind == "f" else "regression"
    frames: list[pd.DataFrame] = []
    for direction, intervention_type, metric in (
        ("up", "tset_up", "up_flex_wh_m2_k"),
        ("down", "tset_down", "down_flex_wh_m2_k"),
    ):
        keep = [
            column
            for column in (
                "profile_id",
                "scenario_index",
                "signal",
                "horizon_hours",
                "horizon_steps",
                "event_count",
                "valid",
            )
            if column in source.columns
        ]
        frame = source[[*keep, metric]].rename(
            columns={metric: "flexibility_value"}
        )
        frame.insert(1, "model", arm)
        if scenarios and "signal" not in frame:
            frame["signal"] = "emulation"
        frame["intervention_type"] = intervention_type
        frame["direction"] = direction
        frame["estimator"] = estimator
        frame["kpi_kind"] = kpi_kind
        frame["unit"] = "Wh/(m2 K)"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _unified_intervention_tables(
    args: argparse.Namespace,
    artifact_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_frames: list[pd.DataFrame] = []
    scenario_frames: list[pd.DataFrame] = []
    for arm in artifact_dirs:
        score_dir = args.output_dir / "scoring" / "test" / arm
        for kpi_kind in KPI_KINDS:
            point_frames.append(
                _setpoint_intervention_table(
                    score_dir,
                    arm=arm,
                    kpi_kind=kpi_kind,
                    scenarios=False,
                )
            )
            scenario_frames.append(
                _setpoint_intervention_table(
                    score_dir,
                    arm=arm,
                    kpi_kind=kpi_kind,
                    scenarios=True,
                )
            )
    force_off_dir = args.output_dir / "force_off_policy" / "test"
    force_point_path = force_off_dir / "force_off_flexibility_kpis.csv"
    force_scenario_path = force_off_dir / "force_off_flexibility_kpi_scenarios.csv"
    model_map = {
        "structured_gated": "probabilistic_structured",
        "unstructured": "probabilistic_lstm",
    }
    if force_point_path.exists():
        force = pd.read_csv(force_point_path)
        force["model"] = force["model"].replace(model_map)
        force = force.rename(
            columns={"flexibility_wh_m2_per_unit": "flexibility_value"}
        )
        force["kpi_kind"] = np.where(
            force["estimator"].eq("direct_ratio"),
            "force_off_f",
            "force_off_regression",
        )
        point_frames.append(force)
    if force_scenario_path.exists():
        force = pd.read_csv(force_scenario_path)
        force["model"] = force["model"].replace(model_map)
        force = force.rename(
            columns={"flexibility_wh_m2_per_unit": "flexibility_value"}
        )
        force["kpi_kind"] = np.where(
            force["estimator"].eq("direct_ratio"),
            "force_off_f",
            "force_off_regression",
        )
        scenario_frames.append(force)
    nonempty_points = [frame for frame in point_frames if not frame.empty]
    nonempty_scenarios = [frame for frame in scenario_frames if not frame.empty]
    point = (
        pd.concat(nonempty_points, ignore_index=True, sort=False)
        if nonempty_points
        else pd.DataFrame()
    )
    scenarios = (
        pd.concat(nonempty_scenarios, ignore_index=True, sort=False)
        if nonempty_scenarios
        else pd.DataFrame()
    )
    return point, scenarios


def _point_flexibility_by_building(
    score_dir: Path,
    *,
    arm: str,
    kpi_kind: str,
) -> pd.DataFrame:
    suffix = "_f" if kpi_kind == "f" else ""
    frame = pd.read_csv(score_dir / f"flexibility_kpis{suffix}.csv")
    frame = frame[_boolean(frame["valid"])]
    rows: list[pd.DataFrame] = []
    for direction in ("up", "down"):
        metric = f"{direction}_flex_wh_m2_k"
        paired = frame.pivot_table(
            index=["profile_id", "horizon_hours"],
            columns="signal",
            values=metric,
            aggfunc="first",
        ).dropna(subset=["simulation", "emulation"])
        paired = paired.reset_index().rename(
            columns={"simulation": "truth", "emulation": "prediction"}
        )
        paired.insert(0, "method", arm)
        paired.insert(1, "calibration", "none")
        paired.insert(2, "kpi_kind", kpi_kind)
        paired.insert(4, "direction", direction)
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)


def _paired_bootstrap_difference(
    target: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 5000,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction_a = np.asarray(prediction_a, dtype=np.float64)
    prediction_b = np.asarray(prediction_b, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(prediction_a) & np.isfinite(prediction_b)
    target = target[finite]
    prediction_a = prediction_a[finite]
    prediction_b = prediction_b[finite]
    if not len(target):
        return {
            name: float("nan")
            for name in (
                "count",
                "rmse_difference_a_minus_b",
                "rmse_difference_ci_low",
                "rmse_difference_ci_high",
                "probability_a_lower_rmse",
                "mae_difference_a_minus_b",
                "mae_difference_ci_low",
                "mae_difference_ci_high",
                "probability_a_lower_mae",
            )
        }
    error_a = prediction_a - target
    error_b = prediction_b - target
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(target), size=(bootstrap_samples, len(target)))
    bootstrap_error_a = error_a[indices]
    bootstrap_error_b = error_b[indices]
    rmse_bootstrap = np.sqrt(np.mean(bootstrap_error_a**2, axis=1)) - np.sqrt(
        np.mean(bootstrap_error_b**2, axis=1)
    )
    mae_bootstrap = np.mean(np.abs(bootstrap_error_a), axis=1) - np.mean(
        np.abs(bootstrap_error_b), axis=1
    )
    return {
        "count": float(len(target)),
        "rmse_difference_a_minus_b": float(
            np.sqrt(np.mean(error_a**2)) - np.sqrt(np.mean(error_b**2))
        ),
        "rmse_difference_ci_low": float(np.quantile(rmse_bootstrap, 0.025)),
        "rmse_difference_ci_high": float(np.quantile(rmse_bootstrap, 0.975)),
        "probability_a_lower_rmse": float(np.mean(rmse_bootstrap < 0.0)),
        "mae_difference_a_minus_b": float(
            np.mean(np.abs(error_a)) - np.mean(np.abs(error_b))
        ),
        "mae_difference_ci_low": float(np.quantile(mae_bootstrap, 0.025)),
        "mae_difference_ci_high": float(np.quantile(mae_bootstrap, 0.975)),
        "probability_a_lower_mae": float(np.mean(mae_bootstrap < 0.0)),
    }


def _paired_flexibility_comparisons(
    predictions: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    comparisons = (
        (("deterministic_structured", "none"), ("deterministic_lstm", "none")),
        (("probabilistic_structured", "raw"), ("probabilistic_lstm", "raw")),
        (("probabilistic_structured", "rns"), ("probabilistic_lstm", "rns")),
        (("lightgbm", "raw"), ("probabilistic_structured", "raw")),
        (("lightgbm", "rns"), ("probabilistic_structured", "rns")),
        (("lightgbm", "raw"), ("probabilistic_lstm", "raw")),
        (("lightgbm", "rns"), ("probabilistic_lstm", "rns")),
    )
    rows: list[dict[str, Any]] = []
    group_columns = ["kpi_kind", "horizon_hours", "direction"]
    for comparison_index, ((method_a, calibration_a), (method_b, calibration_b)) in enumerate(comparisons):
        left = predictions[
            predictions["method"].eq(method_a)
            & predictions["calibration"].eq(calibration_a)
        ]
        right = predictions[
            predictions["method"].eq(method_b)
            & predictions["calibration"].eq(calibration_b)
        ]
        paired = left.merge(
            right,
            on=["profile_id", *group_columns],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        for group_index, (keys, part) in enumerate(paired.groupby(group_columns, sort=False)):
            truth = 0.5 * (
                part["truth_a"].to_numpy(float) + part["truth_b"].to_numpy(float)
            )
            values = _paired_bootstrap_difference(
                truth,
                part["prediction_a"].to_numpy(float),
                part["prediction_b"].to_numpy(float),
                seed=seed + 1000 * comparison_index + group_index,
            )
            rows.append(
                {
                    "method_a": method_a,
                    "calibration_a": calibration_a,
                    "method_b": method_b,
                    "calibration_b": calibration_b,
                    "kpi_kind": keys[0],
                    "horizon_hours": float(keys[1]),
                    "direction": keys[2],
                    **values,
                }
            )
    return pd.DataFrame(rows)


def _write_paired_comparison_dashboard(path: Path, paired: pd.DataFrame) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Regression upward",
            "Regression downward",
            "Direct F upward",
            "Direct F downward",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )
    for row_index, kpi_kind in enumerate(("regression_based", "f"), start=1):
        for column_index, direction in enumerate(("up", "down"), start=1):
            part = paired[
                paired["kpi_kind"].eq(kpi_kind)
                & paired["direction"].eq(direction)
            ]
            for keys, values in part.groupby(
                ["method_a", "calibration_a", "method_b", "calibration_b"],
                sort=False,
            ):
                method_a, calibration_a, method_b, calibration_b = keys
                values = values.sort_values("horizon_hours")
                center = values["rmse_difference_a_minus_b"].to_numpy(float)
                figure.add_trace(
                    go.Scatter(
                        x=values["horizon_hours"],
                        y=center,
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": values["rmse_difference_ci_high"].to_numpy(float) - center,
                            "arrayminus": center - values["rmse_difference_ci_low"].to_numpy(float),
                        },
                        mode="lines+markers",
                        name=(
                            f"{method_a} ({calibration_a}) - "
                            f"{method_b} ({calibration_b})"
                        ),
                        legendgroup=f"{method_a}:{calibration_a}:{method_b}:{calibration_b}",
                        line={"color": _model_color(str(method_a)), "dash": "solid" if calibration_a in ("none", "raw") else "dot"},
                        showlegend=row_index == 1 and column_index == 1,
                    ),
                    row=row_index,
                    col=column_index,
                )
            figure.add_hline(
                y=0.0,
                line={"color": "black", "dash": "dash", "width": 1},
                row=row_index,
                col=column_index,
            )
    figure.update_xaxes(title_text="Horizon [h]")
    figure.update_yaxes(title_text="Paired RMSE difference [Wh/(m2 K)]")
    figure.update_layout(
        title="Paired held-out flexibility comparisons (negative favors first model)",
        template="plotly_white",
        height=1000,
        width=1600,
        legend={"orientation": "v", "x": 1.01, "y": 1.0},
        margin={"r": 430},
        hoverlabel={"namelength": -1, "align": "left"},
    )
    figure.write_html(path)


def _write_temporal_trace_comparison(
    path: Path,
    reliability: pd.DataFrame,
    histogram: pd.DataFrame,
    *,
    model_column: str,
    title: str,
) -> None:
    if reliability.empty or histogram.empty:
        return
    signals = ("Tin", "Qroom", "Pel_SH")
    scopes = {
        str(row.signal): str(row.scope)
        for row in reliability[["signal", "scope"]].drop_duplicates().itertuples()
    }
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            *(f"{signal} reliability<br><sup>{scopes.get(signal, '')}</sup>" for signal in signals),
            *(f"{signal} ensemble-rank histogram" for signal in signals),
        ],
        vertical_spacing=0.14,
    )
    models = tuple(dict.fromkeys(reliability[model_column].astype(str)))
    for column, signal in enumerate(signals, start=1):
        for model in models:
            curve = reliability[
                reliability[model_column].astype(str).eq(model)
                & reliability["signal"].eq(signal)
            ].sort_values("quantile")
            ranks = histogram[
                histogram[model_column].astype(str).eq(model)
                & histogram["signal"].eq(signal)
            ].sort_values("bin_center")
            color = _model_color(model)
            if not curve.empty:
                figure.add_trace(
                    go.Scatter(
                        x=curve["quantile"],
                        y=curve["empirical_coverage"],
                        mode="lines+markers",
                        name=model,
                        legendgroup=model,
                        line={"color": color, "width": 2},
                        showlegend=column == 1,
                        hovertemplate=(
                            "model=%{fullData.name}<br>nominal=%{x:.2f}"
                            "<br>empirical=%{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
            if not ranks.empty:
                figure.add_trace(
                    go.Bar(
                        x=ranks["bin_center"],
                        y=ranks["probability"],
                        width=0.038,
                        offsetgroup=model,
                        name=f"{model} ranks",
                        legendgroup=model,
                        marker_color=color,
                        opacity=0.78,
                        showlegend=False,
                        hovertemplate=(
                            "model=%{fullData.name}<br>rank bin=%{x:.2f}"
                            "<br>probability=%{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=column,
                )
        figure.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line={"color": "black", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        expected = 0.1
        signal_ranks = histogram[histogram["signal"].eq(signal)]
        if not signal_ranks.empty and "expected_probability" in signal_ranks:
            expected = float(signal_ranks["expected_probability"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[expected, expected],
                mode="lines",
                line={"color": "black", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=column,
        )
        figure.update_xaxes(
            title_text="Nominal quantile", range=[0.0, 1.0], row=1, col=column
        )
        figure.update_yaxes(
            title_text="Empirical rank CDF", range=[0.0, 1.0], row=1, col=column
        )
        figure.update_xaxes(
            title_text="Randomized ensemble rank",
            range=[0.0, 1.0],
            row=2,
            col=column,
        )
        figure.update_yaxes(title_text="Probability", rangemode="tozero", row=2, col=column)
    figure.update_layout(
        title=title,
        template="plotly_white",
        barmode="group",
        height=920,
        width=1600,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        hoverlabel={"namelength": -1, "align": "left"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def _write_full_year_trace_index(
    path: Path,
    traces: Sequence[dict[str, str]],
) -> None:
    rows = []
    for trace in traces:
        href = "../" + escape(trace["path"], quote=True)
        rows.append(
            "<tr>"
            f"<td>{escape(trace['policy'])}</td>"
            f"<td>{escape(trace['model'])}</td>"
            f"<td>{escape(trace['profile_id'])}</td>"
            f"<td><a href=\"{href}\">Open annual trace</a></td>"
            "</tr>"
        )
    path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Probabilistic full-year traces</title>"
        "<style>body{font:16px system-ui;margin:32px;color:#263238}"
        "table{border-collapse:collapse;min-width:760px}"
        "th,td{padding:10px 14px;border:1px solid #cfd8dc;text-align:left}"
        "th{background:#e8eef5}a{color:#1769aa}</style></head><body>"
        "<h1>Probabilistic full-year traces</h1>"
        "<p>Each model/profile plot contains the simulated trace, scenario mean, "
        "95% interval, and closest/worst annual scenarios for Tin, Qroom, and Pel.</p>"
        "<table><thead><tr><th>Policy</th><th>Model</th><th>Profile</th>"
        "<th>Plot</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def _collect_reports(args: argparse.Namespace, artifact_dirs: dict[str, Path]) -> None:
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    training_times_path = args.output_dir / "training_times.json"
    training_times = json.loads(training_times_path.read_text()) if training_times_path.exists() else {}
    signal_rows: list[dict[str, Any]] = []
    temporal_frames: list[pd.DataFrame] = []
    temporal_rank_frames: list[pd.DataFrame] = []
    temporal_score_frames: list[pd.DataFrame] = []
    dynamics_rows: list[dict[str, Any]] = []
    flexibility_rows: list[dict[str, Any]] = []
    flexibility_building_frames: list[pd.DataFrame] = []
    detailed_reliability: dict[str, list[pd.DataFrame]] = {
        filename: []
        for filename in (
            "pel_transition_reliability.csv",
            "pel_conditional_calibration.csv",
            "pel_conditional_calibration_summary.csv",
            "pel_mode_autocorrelation.csv",
            "pel_run_duration_distribution.csv",
            "pel_annual_energy_calibration.csv",
            "qroom_occurrence_reliability.csv",
        )
    }
    for arm, artifact_dir in artifact_dirs.items():
        score_dir = args.output_dir / "scoring" / "test" / arm
        summary = json.loads((score_dir / "summary.json").read_text())
        signal_rows.append(
            {
                "arm": arm,
                "family": "unstructured_lstm" if "lstm" in arm else "structured",
                "probabilistic": load_artifact(artifact_dir).spec.probabilistic,
                **summary["metrics"],
                "parameter_count": _tree_parameter_count(artifact_dir),
                "training_minutes": float(training_times.get(arm, 0.0)),
            }
        )
        profile_summary = summary.get("profile_metric_summary", {})
        dynamics_rows.append(
            {
                "arm": arm,
                **summary.get("pel_dynamics", {}),
                **summary.get("qroom_calibration", {}),
                **{
                    f"{name}_{statistic}": value
                    for name, values in profile_summary.items()
                    for statistic, value in values.items()
                    if name.endswith(("acf_mae", "increment_spectral_js", "wasserstein_norm", "nrmse"))
                },
            }
        )
        temporal_path = score_dir / "temporal_trace_calibration.csv"
        if temporal_path.exists():
            frame = pd.read_csv(temporal_path)
            frame.insert(0, "arm", arm)
            temporal_frames.append(frame)
        temporal_rank_path = score_dir / "temporal_trace_rank_histogram.csv"
        if temporal_rank_path.exists():
            frame = pd.read_csv(temporal_rank_path)
            frame.insert(0, "arm", arm)
            temporal_rank_frames.append(frame)
        temporal_score_path = score_dir / "temporal_trace_probabilistic_scores.csv"
        if temporal_score_path.exists():
            frame = pd.read_csv(temporal_score_path)
            frame.insert(0, "arm", arm)
            temporal_score_frames.append(frame)
        for filename, frames in detailed_reliability.items():
            source = score_dir / filename
            if source.exists():
                frame = pd.read_csv(source)
                frame.insert(0, "arm", arm)
                frames.append(frame)
        for kind in KPI_KINDS:
            flexibility_rows.extend(_point_flexibility_metrics(score_dir, arm=arm, kpi_kind=kind))
            flexibility_building_frames.append(
                _point_flexibility_by_building(score_dir, arm=arm, kpi_kind=kind)
            )
    signals = pd.DataFrame(signal_rows)
    dynamics = pd.DataFrame(dynamics_rows)
    temporal = pd.concat(temporal_frames, ignore_index=True) if temporal_frames else pd.DataFrame()
    flexibility = pd.DataFrame(flexibility_rows)
    distribution = pd.read_csv(args.output_dir / "calibration" / "distribution_metrics.csv")
    reliability = pd.read_csv(args.output_dir / "calibration" / "distribution_reliability.csv")
    distribution_buildings = pd.read_csv(
        args.output_dir / "calibration" / "distribution_by_building.csv"
    )
    lightgbm_point = distribution[
        distribution["method"].eq("lightgbm")
        & distribution["calibration"].eq("raw")
        & distribution["kpi_kind"].isin(KPI_KINDS)
    ].copy()
    if not lightgbm_point.empty:
        flexibility = pd.concat(
            [
                flexibility,
                lightgbm_point[["method", "kpi_kind", "horizon_hours", "direction", "count", "rmse", "nrmse", "mae", "bias", "r2", "spearman"]],
            ],
            ignore_index=True,
        )
    signals.to_csv(report_dir / "annual_signal_metrics.csv", index=False)
    dynamics.to_csv(report_dir / "temporal_and_dynamics_metrics.csv", index=False)
    temporal.to_csv(report_dir / "single_signal_reliability.csv", index=False)
    temporal_ranks = (
        pd.concat(temporal_rank_frames, ignore_index=True)
        if temporal_rank_frames
        else pd.DataFrame()
    )
    temporal_ranks.to_csv(report_dir / "single_signal_rank_histogram.csv", index=False)
    temporal_scores = (
        pd.concat(temporal_score_frames, ignore_index=True)
        if temporal_score_frames
        else pd.DataFrame()
    )
    temporal_scores.to_csv(
        report_dir / "single_signal_probabilistic_scores.csv", index=False
    )
    probabilistic_temporal = temporal[
        temporal["arm"].isin(PROBABILISTIC_ARMS)
    ] if not temporal.empty else temporal
    probabilistic_ranks = temporal_ranks[
        temporal_ranks["arm"].isin(PROBABILISTIC_ARMS)
    ] if not temporal_ranks.empty else temporal_ranks
    _write_temporal_trace_comparison(
        report_dir / "setpoint_temporal_trace_calibration.html",
        probabilistic_temporal,
        probabilistic_ranks,
        model_column="arm",
        title="Setpoint-policy temporal trace calibration",
    )
    force_off_test_dir = args.output_dir / "force_off_policy" / "test"
    force_temporal_path = force_off_test_dir / "temporal_trace_calibration.csv"
    force_rank_path = force_off_test_dir / "temporal_trace_rank_histogram.csv"
    if force_temporal_path.exists() and force_rank_path.exists():
        _write_temporal_trace_comparison(
            report_dir / "force_off_temporal_trace_calibration.html",
            pd.read_csv(force_temporal_path),
            pd.read_csv(force_rank_path),
            model_column="model",
            title="Force-off-policy temporal trace calibration",
        )
    for filename, frames in detailed_reliability.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined.to_csv(report_dir / filename, index=False)
    flexibility.to_csv(report_dir / "flexibility_point_metrics.csv", index=False)
    distribution_points = distribution_buildings[
        [
            "method",
            "calibration",
            "kpi_kind",
            "horizon_hours",
            "direction",
            "profile_id",
            "truth",
            "mean",
        ]
    ].rename(columns={"mean": "prediction"})
    flexibility_by_building = pd.concat(
        [*flexibility_building_frames, distribution_points], ignore_index=True
    )
    flexibility_by_building.to_csv(
        report_dir / "flexibility_point_by_building.csv", index=False
    )
    resolved = json.loads((args.output_dir / "resolved_recipe.json").read_text())
    paired = _paired_flexibility_comparisons(
        flexibility_by_building,
        seed=int(resolved["recipe"]["seed"]),
    )
    paired.to_csv(report_dir / "paired_flexibility_comparisons.csv", index=False)
    _write_paired_comparison_dashboard(
        report_dir / "paired_flexibility_comparisons.html",
        paired,
    )
    distribution.to_csv(report_dir / "flexibility_distribution_metrics.csv", index=False)
    reliability.to_csv(report_dir / "flexibility_distribution_reliability.csv", index=False)
    intervention_point, intervention_scenarios = _unified_intervention_tables(
        args,
        artifact_dirs,
    )
    intervention_point.to_csv(
        report_dir / "intervention_flexibility_kpis.csv", index=False
    )
    intervention_scenarios.to_csv(
        report_dir / "intervention_flexibility_kpi_scenarios.csv", index=False
    )
    dashboard = report_dir / "benchmark_dashboard.html"
    _write_benchmark_dashboard(
        dashboard,
        signals,
        dynamics,
        temporal,
        flexibility,
        distribution[distribution["kpi_kind"].isin(KPI_KINDS)],
    )
    table_paths = {
        "annual_signal_metrics": "annual_signal_metrics.csv",
        "temporal_and_dynamics_metrics": "temporal_and_dynamics_metrics.csv",
        "single_signal_reliability": "single_signal_reliability.csv",
        "single_signal_rank_histogram": "single_signal_rank_histogram.csv",
        "single_signal_probabilistic_scores": (
            "single_signal_probabilistic_scores.csv"
        ),
        "setpoint_temporal_trace_calibration": (
            "setpoint_temporal_trace_calibration.html"
        ),
        "flexibility_point_metrics": "flexibility_point_metrics.csv",
        "flexibility_point_by_building": "flexibility_point_by_building.csv",
        "paired_flexibility_comparisons": "paired_flexibility_comparisons.csv",
        "paired_flexibility_dashboard": "paired_flexibility_comparisons.html",
        "flexibility_distribution_metrics": "flexibility_distribution_metrics.csv",
        "flexibility_distribution_reliability": (
            "flexibility_distribution_reliability.csv"
        ),
        "setpoint_kpi_calibration_dashboard": "../calibration/kpi_calibration.html",
        "intervention_flexibility_kpis": "intervention_flexibility_kpis.csv",
        "intervention_flexibility_kpi_scenarios": (
            "intervention_flexibility_kpi_scenarios.csv"
        ),
        **{filename.removesuffix(".csv"): filename for filename in detailed_reliability},
    }
    if (args.output_dir / "calibration" / "force_off_kpi_calibration.html").exists():
        table_paths["force_off_kpi_calibration_dashboard"] = (
            "../calibration/force_off_kpi_calibration.html"
        )
    if (report_dir / "force_off_temporal_trace_calibration.html").exists():
        table_paths["force_off_temporal_trace_calibration"] = (
            "force_off_temporal_trace_calibration.html"
        )
    trace_manifest: list[dict[str, str]] = []
    for arm in PROBABILISTIC_ARMS:
        for path in sorted(
            (args.output_dir / "scoring" / "test" / arm).glob("profile_*.html")
        ):
            trace_manifest.append(
                {
                    "policy": "setpoint_modulation",
                    "model": arm,
                    "profile_id": path.stem.rsplit("_", 1)[-1],
                    "path": str(path.relative_to(args.output_dir)),
                }
            )
    for path in sorted(
        (force_off_test_dir / "probabilistic_traces").glob("force_off_profile_*.html")
    ):
        model = "structured_gated" if "structured_gated" in path.stem else "unstructured"
        profile_stub = path.stem.removesuffix(f"_{model}")
        trace_manifest.append(
            {
                "policy": "force_off",
                "model": model,
                "profile_id": profile_stub.rsplit("_", 1)[-1],
                "path": str(path.relative_to(args.output_dir)),
            }
        )
    (report_dir / "full_year_trace_manifest.json").write_text(
        json.dumps(trace_manifest, indent=2) + "\n"
    )
    _write_full_year_trace_index(
        report_dir / "full_year_traces.html",
        trace_manifest,
    )
    table_paths["full_year_trace_manifest"] = "full_year_trace_manifest.json"
    table_paths["full_year_traces"] = "full_year_traces.html"
    summary = {
        "artifacts": {arm: str(path) for arm, path in artifact_dirs.items()},
        "tables": table_paths,
    }
    (report_dir / "benchmark_summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")
    print(f"saved_benchmark_dashboard={dashboard}")


def _write_benchmark_dashboard(
    path: Path,
    signals: pd.DataFrame,
    dynamics: pd.DataFrame,
    temporal: pd.DataFrame,
    flexibility: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    figure = make_subplots(
        rows=3,
        cols=3,
        specs=[[{"type": "table", "colspan": 3}, None, None], [{}, {}, {}], [{}, {}, {}]],
        subplot_titles=(
            "Annual autonomous-rollout summary",
            "Signal RMSE",
            "Temporal structure",
            "Flexibility point NRMSE",
            "Single-signal reliability",
            "Flexibility calibration",
            "Flexibility CRPS",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    table_columns = [
        "arm", "temperature_c_rmse", "qroom_w_m2_rmse", "pel_w_m2_rmse",
        "parameter_count", "training_minutes",
    ]
    display = signals[table_columns].copy()
    figure.add_trace(
        go.Table(
            header={"values": table_columns, "fill_color": "#e8eef5"},
            cells={"values": [display[column] for column in table_columns]},
        ),
        row=1,
        col=1,
    )
    for row in signals.itertuples(index=False):
        figure.add_trace(
            go.Bar(
                x=["Tin [C]", "Qroom [W/m2]", "Pel [W/m2]"],
                y=[row.temperature_c_rmse, row.qroom_w_m2_rmse, row.pel_w_m2_rmse],
                name=row.arm,
                marker_color=_model_color(row.arm),
            ),
            row=2,
            col=1,
        )
    temporal_columns = [column for column in dynamics if column.endswith("increment_spectral_js_mean")]
    for row in dynamics.itertuples(index=False):
        figure.add_trace(
            go.Bar(
                x=[value.removesuffix("_increment_spectral_js_mean") for value in temporal_columns],
                y=[getattr(row, value) for value in temporal_columns],
                name=row.arm,
                showlegend=False,
                marker_color=_model_color(row.arm),
            ),
            row=2,
            col=2,
        )
    for (method, kind), part in flexibility.groupby(["method", "kpi_kind"], sort=False):
        figure.add_trace(
            go.Scatter(
                x=[f"{row.direction} {row.horizon_hours:g}h" for row in part.itertuples()],
                y=part["nrmse"],
                mode="lines+markers",
                name=f"{method} {kind}",
                line={"color": _model_color(str(method)), "dash": "solid" if kind == "f" else "dot"},
                showlegend=False,
            ),
            row=2,
            col=3,
        )
    if not temporal.empty:
        trace_summary = temporal.groupby(["arm", "signal"], as_index=False)["abs_calibration_error"].mean()
        for arm, part in trace_summary.groupby("arm", sort=False):
            figure.add_trace(go.Bar(x=part["signal"], y=part["abs_calibration_error"], name=arm, marker_color=_model_color(str(arm)), showlegend=False), row=3, col=1)
    calibration_summary = distribution.groupby(["method", "calibration"], as_index=False)[["calibration_mae", "crps"]].mean()
    for calibration, dash in (("raw", "solid"), ("rns", "dot")):
        part = calibration_summary[calibration_summary["calibration"] == calibration]
        colors = [_model_color(str(method)) for method in part["method"]]
        pattern = "" if calibration == "raw" else "/"
        figure.add_trace(go.Bar(x=part["method"], y=part["calibration_mae"], marker={"color": colors, "pattern": {"shape": pattern}}, name=f"{calibration} calibration", hovertemplate="%{x}<br>Calibration MAE: %{y:.3f}<extra></extra>"), row=3, col=2)
        figure.add_trace(go.Bar(x=part["method"], y=part["crps"], marker={"color": colors, "pattern": {"shape": pattern}}, name=f"{calibration} CRPS", opacity=0.75, hovertemplate="%{x}<br>CRPS: %{y:.3f}<extra></extra>"), row=3, col=3)
    figure.update_yaxes(title_text="RMSE", row=2, col=1)
    figure.update_yaxes(title_text="Spectral JS distance", row=2, col=2)
    figure.update_yaxes(title_text="NRMSE", row=2, col=3)
    figure.update_yaxes(title_text="Mean |rank CDF error|", row=3, col=1)
    figure.update_yaxes(title_text="Mean |coverage - nominal|", row=3, col=2)
    figure.update_yaxes(title_text="Wh/(m2 K)", row=3, col=3)
    figure.update_layout(
        title="Calibrated structured versus unstructured building-emulator benchmark",
        template="plotly_white",
        height=1500,
        width=1750,
        barmode="group",
        legend={"orientation": "h", "y": -0.15},
        margin={"b": 220},
        hoverlabel={"namelength": -1, "align": "left"},
    )
    figure.write_html(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/neural_building_emulator_refactor/calibrated_four_model_benchmark"),
    )
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--probabilistic-structured-legacy-config-json",
        type=Path,
        help=(
            "Metadata or a JSON object containing legacy TrainConfig overrides for "
            "only the probabilistic structured arm. Dataset, split, output path, "
            "and epoch budget remain controlled by this benchmark."
        ),
    )
    parser.add_argument(
        "--train-arms",
        nargs="+",
        choices=ARM_ORDER,
        help=(
            "Retrain only these arms during the train stage. Other arms are "
            "reused from artifact_paths.json or explicit artifact arguments."
        ),
    )
    parser.add_argument(
        "--additional-dataset",
        dest="additional_datasets",
        type=Path,
        action="append",
        help=(
            "Additional policy dataset used by every emulator training arm. "
            "May be repeated. Randomized-setpoint KPIs use the primary dataset; "
            "a force-off dataset also enables force-off KPI scoring, LightGBM, "
            "and calibration."
        ),
    )
    parser.add_argument(
        "--prob-hp-training-mode",
        choices=("expected", "straight_through", "hard_expected_grad"),
        help="Override only the probabilistic structured HP training emission mode.",
    )
    parser.add_argument(
        "--contracting-temperature-feedback-gradient-scale",
        type=float,
        help=(
            "Override the temperature-feedback gradient scale in both structured "
            "arms. Use 0.05 to reproduce the current stable Option C recipe."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stages", nargs="+", choices=("all", *STAGES), default=("all",))
    parser.add_argument("--deterministic-structured-artifact-dir", type=Path)
    parser.add_argument("--deterministic-lstm-artifact-dir", type=Path)
    parser.add_argument("--probabilistic-structured-artifact-dir", type=Path)
    parser.add_argument("--probabilistic-lstm-artifact-dir", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    recipe = json.loads(args.config.read_text())
    if args.probabilistic_structured_legacy_config_json is not None:
        imported = json.loads(
            args.probabilistic_structured_legacy_config_json.read_text()
        )
        if not isinstance(imported, dict):
            raise ValueError(
                "--probabilistic-structured-legacy-config-json must contain a JSON object"
            )
        imported = imported.get("train_config", imported)
        if not isinstance(imported, dict):
            raise ValueError("Imported structured train_config must be a JSON object")
        recipe["probabilistic_structured"].update(imported)
    if args.max_profiles is not None:
        recipe["max_profiles"] = args.max_profiles
    if args.epochs is not None:
        recipe["optimizer"]["epochs"] = args.epochs
    if args.prob_hp_training_mode is not None:
        recipe["probabilistic_structured"]["prob_hp_training_mode"] = args.prob_hp_training_mode
    if args.contracting_temperature_feedback_gradient_scale is not None:
        for section in ("deterministic_structured", "probabilistic_structured"):
            recipe[section]["contracting_temperature_feedback_gradient_scale"] = (
                args.contracting_temperature_feedback_gradient_scale
            )
    recipe["dataset_path"] = str(Path(recipe["dataset_path"]))
    if args.additional_datasets is not None:
        recipe["additional_dataset_paths"] = [
            str(path) for path in args.additional_datasets
        ]
    else:
        recipe["additional_dataset_paths"] = [
            str(path) for path in recipe.get("additional_dataset_paths", ())
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = set(STAGES if "all" in args.stages else args.stages)
    split = _load_or_create_split(
        args.output_dir / "split_manifest.json",
        dataset_path=Path(recipe["dataset_path"]),
        additional_dataset_paths=tuple(
            Path(path) for path in recipe["additional_dataset_paths"]
        ),
        max_profiles=recipe.get("max_profiles"),
        seed=int(recipe["seed"]),
        split_config=recipe["split"],
        resume=args.resume,
    )
    (args.output_dir / "resolved_recipe.json").write_text(
        json.dumps({"source": str(args.config), "recipe": recipe, "split": asdict(split)}, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"benchmark_split train={len(split.train_ids)} validation={len(split.validation_ids)} "
        f"calibration={len(split.calibration_ids)} test={len(split.test_ids)} "
        f"fingerprint={split.fingerprint}"
    )
    if stages == {"split"}:
        return args.output_dir / "split_manifest.json"
    if "train" in stages:
        artifact_dirs, _ = _train_models(args, recipe, split)
    else:
        artifact_dirs = _load_artifact_paths(args)
    if "score" in stages:
        _score_models(args, recipe, split, artifact_dirs)
    if "lightgbm" in stages:
        _fit_lightgbm(args, recipe, split, artifact_dirs["probabilistic_structured"])
    if "force_off" in stages:
        _score_force_off_policy(args, recipe, split, artifact_dirs)
    if "calibrate" in stages:
        _calibrate(args, recipe)
    if "report" in stages:
        _collect_reports(args, artifact_dirs)
    return args.output_dir / "reports" / "benchmark_dashboard.html"


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
