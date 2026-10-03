"""Predict building-level flexibility KPIs directly from static metadata.

This is a ceiling/bypass benchmark for the closed-loop emulator.  Targets are
estimated from the EnergyPlus trajectories on training buildings only, while
the regressors receive exactly the static metadata fields used by the neural
emulators.  Complete held-out buildings are used for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from neural_building_emulator.columns import (
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    METADATA_COLUMNS,
    PROFILE_ID_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    closed_loop_required_columns,
)
from neural_building_emulator.data import (
    ClosedLoopProfile,
    read_dataset_frame,
    to_closed_loop_profiles,
)
from neural_building_emulator.dataset_adapter import is_entity_profile_dataset
from neural_building_emulator.flexibility_event_study_kpis import (
    EventStudyConfig,
    estimate_trace_flexibility,
)

from .artifacts import LoadedArtifact, load_artifact
from .profiles import artifact_value, saved_profile_ids
from .scoring import score_artifact


Direction = Literal["up", "down"]
KpiKind = Literal["f", "regression_based"]

KPI_ESTIMATORS: dict[KpiKind, str] = {
    "f": "direct_ratio",
    "regression_based": "regression",
}
KPI_ALIASES: dict[str, KpiKind] = {
    "f": "f",
    "direct_ratio": "f",
    "regression_based": "regression_based",
    "regression": "regression_based",
}
RESERVED_COMPARISON_METHODS = {
    "true",
    "train_mean",
    "lightgbm",
    "emulator",
}


def _normalize_kpi_kinds(values: Sequence[str]) -> tuple[KpiKind, ...]:
    normalized: list[KpiKind] = []
    for value in values:
        try:
            kind = KPI_ALIASES[value]
        except KeyError as exc:
            choices = ", ".join(sorted(KPI_ALIASES))
            raise ValueError(f"Unknown KPI {value!r}; expected one of: {choices}") from exc
        if kind not in normalized:
            normalized.append(kind)
    if not normalized:
        raise ValueError("At least one KPI must be selected")
    return tuple(normalized)


def _resolve_emulator_kpi_csvs(
    values: Sequence[str | Path],
    kpi_kinds: Sequence[KpiKind],
) -> dict[KpiKind, Path]:
    mapped: dict[KpiKind, Path] = {}
    bare: list[Path] = []
    for raw_value in values:
        value = str(raw_value)
        name, separator, path_value = value.partition("=")
        if separator and name in KPI_ALIASES:
            kind = KPI_ALIASES[name]
            if kind not in kpi_kinds:
                raise ValueError(
                    f"Emulator KPI CSV was supplied for unselected KPI {kind!r}"
                )
            if kind in mapped:
                raise ValueError(f"Duplicate emulator KPI CSV for {kind!r}")
            mapped[kind] = Path(path_value)
        else:
            bare.append(Path(value))
    if bare:
        if len(bare) != 1 or len(kpi_kinds) != 1:
            raise ValueError(
                "Bare --emulator-kpi-csv paths are accepted only for a single KPI. "
                "For multiple KPIs use f=PATH and regression_based=PATH."
            )
        kind = kpi_kinds[0]
        if kind in mapped:
            raise ValueError(f"Duplicate emulator KPI CSV for {kind!r}")
        mapped[kind] = bare[0]
    return mapped


def _resolve_additional_emulator_artifacts(
    values: Sequence[str],
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        label, separator, path_value = value.partition("=")
        if not separator or not label or not path_value:
            raise ValueError(
                "--additional-emulator-artifact must use the form NAME=PATH"
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(
                f"Invalid additional emulator label {label!r}; use letters, digits, "
                "dots, dashes, or underscores"
            )
        if label in RESERVED_COMPARISON_METHODS:
            raise ValueError(f"Reserved additional emulator label {label!r}")
        if label in artifacts:
            raise ValueError(f"Duplicate additional emulator label {label!r}")
        artifacts[label] = Path(path_value)
    return artifacts


def _validate_comparison_artifact(
    reference: LoadedArtifact,
    candidate: LoadedArtifact,
    *,
    label: str,
) -> None:
    if candidate.spec.task != "closed_loop_hp":
        raise ValueError(
            f"Additional emulator {label!r} has task {candidate.spec.task!r}; "
            "expected 'closed_loop_hp'"
        )
    reference_ids = set(saved_profile_ids(reference, "test"))
    candidate_ids = set(saved_profile_ids(candidate, "test"))
    if candidate_ids != reference_ids:
        raise ValueError(
            f"Additional emulator {label!r} does not use the reference artifact's "
            "test-building split"
        )
    reference_normalization = str(
        artifact_value(reference, "hp_power_area_normalization")
        or "zone_floor_area"
    )
    candidate_normalization = str(
        artifact_value(candidate, "hp_power_area_normalization")
        or "zone_floor_area"
    )
    if candidate_normalization != reference_normalization:
        raise ValueError(
            f"Additional emulator {label!r} uses HP power normalization "
            f"{candidate_normalization!r}, but the reference uses "
            f"{reference_normalization!r}"
        )


def _resolve_dataset_path(
    artifact: LoadedArtifact,
    override: Path | None,
) -> tuple[Path, str]:
    if override is not None:
        return Path(override), "override"
    inferred = artifact_value(artifact, "dataset_path")
    if not inferred:
        raise ValueError(
            "The artifact does not record dataset_path; pass --dataset explicitly"
        )
    return Path(inferred), "artifact"


def _automatic_emulator_scoring_plan(
    kpi_kinds: Sequence[KpiKind],
    output_dir: Path,
) -> tuple[str, bool, dict[KpiKind, Path]]:
    """Plan one artifact rollout that supplies all requested emulator KPI tables."""
    selected = tuple(kpi_kinds)
    if "regression_based" in selected:
        primary_estimator = "regression"
        evaluate_direct_f = "f" in selected
        paths: dict[KpiKind, Path] = {
            "regression_based": output_dir / "flexibility_kpis.csv"
        }
        if evaluate_direct_f:
            paths["f"] = output_dir / "flexibility_kpis_f.csv"
        return primary_estimator, evaluate_direct_f, paths
    return "direct_ratio", False, {"f": output_dir / "flexibility_kpis.csv"}


def _generate_emulator_kpi_csvs(
    args: argparse.Namespace,
    *,
    artifact_dir: Path,
    artifact: LoadedArtifact,
    dataset_path: Path,
    kpi_kinds: Sequence[KpiKind],
    output_dir: Path,
) -> dict[KpiKind, Path]:
    """Roll out the artifact once and derive every selected KPI from those traces."""
    estimator, evaluate_direct_f, paths = _automatic_emulator_scoring_plan(
        kpi_kinds,
        output_dir,
    )
    if artifact.spec.probabilistic and args.emulator_num_scenarios < 1:
        raise ValueError("--emulator-num-scenarios must be positive")
    if args.emulator_max_profiles is not None and args.emulator_max_profiles < 1:
        raise ValueError("--emulator-max-profiles must be positive")
    kpi_mode = "scenario_average" if artifact.spec.probabilistic else "mean"
    scenario_count = args.emulator_num_scenarios if artifact.spec.probabilistic else 0
    print(
        "emulator_kpi_csv=auto_generate "
        f"mode={kpi_mode} scenarios={scenario_count} "
        f"hp_scenario_mode={args.emulator_hp_scenario_mode} "
        f"ventilation_rollout_mode={args.emulator_ventilation_rollout_mode} "
        f"output={output_dir}"
    )
    score_artifact(
        artifact_dir,
        dataset_path=dataset_path,
        output_dir=output_dir,
        profile_source="test",
        max_profiles=args.emulator_max_profiles,
        kpi_mode=kpi_mode,
        num_scenarios=args.emulator_num_scenarios,
        hp_scenario_mode=args.emulator_hp_scenario_mode,
        ventilation_rollout_mode=args.emulator_ventilation_rollout_mode,
        horizons_hours=tuple(args.horizons_hours),
        dt_hours=args.dt_hours,
        min_setpoint_change=args.min_setpoint_change,
        min_events=args.min_events,
        kpi_estimator=estimator,
        controls=args.controls,
        seed=args.seed,
        num_profile_plots=args.emulator_num_profile_plots,
        evaluate_direct_f=evaluate_direct_f,
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Automatic emulator scoring did not produce the expected KPI tables: "
            + ", ".join(missing)
        )
    return paths


@dataclass(frozen=True)
class TargetSpec:
    direction: Direction
    horizon_hours: float
    horizon_steps: int

    @property
    def key(self) -> str:
        horizon = f"{self.horizon_hours:g}".replace(".", "p")
        return f"{self.direction}_flex_{horizon}h_wh_m2_k"

    @property
    def display_name(self) -> str:
        return f"{self.direction}, H={self.horizon_hours:g} h"


@dataclass(frozen=True)
class LightGBMConfig:
    learning_rate: float = 0.03
    max_estimators: int = 2000
    early_stopping_rounds: int = 100
    validation_fraction: float = 0.2
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: int = -1


def _artifact_metadata(artifact: LoadedArtifact) -> dict[str, Any]:
    if artifact.backend == "legacy":
        if artifact.legacy_artifact is None:
            raise ValueError("Legacy artifact is missing its loaded training artifact")
        return artifact.legacy_artifact.metadata
    return artifact.metadata


def _metadata_value(metadata: dict[str, Any], name: str, default: Any) -> Any:
    if name in metadata:
        return metadata[name]
    train_config = metadata.get("train_config", {})
    if name in train_config:
        return train_config[name]
    return metadata.get("experiment_config", {}).get(name, default)


def _resolve_targets(
    horizons_hours: Sequence[float],
    dt_hours: float,
) -> tuple[EventStudyConfig, tuple[TargetSpec, ...]]:
    unique: dict[int, float] = {}
    for horizon in horizons_hours:
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("All horizons must be finite and positive")
        steps = max(1, int(round(float(horizon) / dt_hours)))
        unique.setdefault(steps, steps * dt_hours)
    effective_horizons = tuple(unique.values())
    horizon_steps = tuple(unique.keys())
    targets = tuple(
        TargetSpec(direction, horizon, steps)
        for horizon, steps in zip(effective_horizons, horizon_steps)
        for direction in ("up", "down")
    )
    config = EventStudyConfig(
        horizons_hours=effective_horizons,
        horizon_steps=horizon_steps,
        dt_hours=dt_hours,
        min_setpoint_change=0.05,
        min_events=20,
        controls="full",
    )
    return config, targets


def _parquet_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_dir():
        files = sorted(dataset_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {dataset_path}")
        return files
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    return [dataset_path]


def _profiles_in_file(
    dataset_path: Path,
    wanted_ids: set[int],
    *,
    metadata_columns: Sequence[str],
    include_space_heating_availability: bool,
    include_internal_gains: bool,
    include_dhw_request: bool,
    hp_power_area_normalization: str,
) -> list[ClosedLoopProfile]:
    frame = read_dataset_frame(
        dataset_path,
        columns=closed_loop_required_columns(
            metadata_columns,
            include_internal_gains=include_internal_gains,
            include_dhw_request=include_dhw_request,
        ),
        profile_ids=sorted(wanted_ids),
    )
    frame[PROFILE_ID_COLUMN] = frame[PROFILE_ID_COLUMN].astype(int)
    frame = frame[frame[PROFILE_ID_COLUMN].isin(wanted_ids)]
    if frame.empty:
        return []
    return to_closed_loop_profiles(
        frame,
        include_space_heating_availability=include_space_heating_availability,
        include_internal_gains=include_internal_gains,
        include_dhw_request=include_dhw_request,
        hp_power_area_normalization=hp_power_area_normalization,  # type: ignore[arg-type]
        metadata_columns=metadata_columns,
    )


def _kpi_row(
    profile: ClosedLoopProfile,
    split: str,
    event_config: EventStudyConfig,
    metadata_columns: Sequence[str] = METADATA_COLUMNS,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "profile_id": int(profile.profile_id),
        "split": split,
    }
    row.update(
        {
            column: float(value)
            for column, value in zip(metadata_columns, np.asarray(profile.metadata).reshape(-1))
        }
    )
    for kpi in estimate_trace_flexibility(profile, profile.targets, event_config):
        horizon = f"{kpi.horizon_hours:g}".replace(".", "p")
        row[f"event_count_{horizon}h"] = int(kpi.event_count)
        row[f"valid_{horizon}h"] = bool(kpi.valid)
        row[f"up_flex_{horizon}h_wh_m2_k"] = float(kpi.up_flex_wh_m2_k)
        row[f"down_flex_{horizon}h_wh_m2_k"] = float(kpi.down_flex_wh_m2_k)
    return row


def extract_metadata_kpis(
    dataset_path: Path,
    *,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
    metadata_columns: Sequence[str] = METADATA_COLUMNS,
    include_space_heating_availability: bool,
    include_internal_gains: bool,
    include_dhw_request: bool,
    hp_power_area_normalization: str,
    event_config: EventStudyConfig,
) -> pd.DataFrame:
    """Extract one metadata/KPI row per building without retaining all profiles."""
    train_set = set(int(value) for value in train_ids)
    test_set = set(int(value) for value in test_ids)
    wanted = train_set | test_set
    rows: list[dict[str, Any]] = []
    found: set[int] = set()
    if is_entity_profile_dataset(dataset_path):
        sorted_ids = sorted(wanted)
        batch_size = 16
        work_items = [
            (dataset_path, set(sorted_ids[start : start + batch_size]))
            for start in range(0, len(sorted_ids), batch_size)
        ]
    else:
        work_items = [(path, wanted) for path in _parquet_files(dataset_path)]
    for file_index, (source_path, source_ids) in enumerate(work_items, start=1):
        profiles = _profiles_in_file(
            source_path,
            source_ids,
            metadata_columns=metadata_columns,
            include_space_heating_availability=include_space_heating_availability,
            include_internal_gains=include_internal_gains,
            include_dhw_request=include_dhw_request,
            hp_power_area_normalization=hp_power_area_normalization,
        )
        for profile in profiles:
            if profile.profile_id in found:
                raise ValueError(
                    f"Profile {profile.profile_id} spans multiple parquet files; "
                    "metadata KPI extraction requires complete profiles per file."
                )
            found.add(profile.profile_id)
            split = "train" if profile.profile_id in train_set else "test"
            rows.append(_kpi_row(profile, split, event_config, metadata_columns))
        print(
            f"metadata_flex_extract_batch={file_index}/{len(work_items)} "
            f"profiles_complete={len(rows)}/{len(wanted)}"
        )
    missing = sorted(wanted.difference(found))
    if missing:
        raise ValueError(f"Dataset is missing {len(missing)} saved profile ids: {missing[:10]}")
    frame = pd.DataFrame(rows)
    order = {int(value): index for index, value in enumerate((*train_ids, *test_ids))}
    frame["_order"] = frame["profile_id"].map(order)
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _model_parameters(config: LightGBMConfig, seed: int, n_estimators: int) -> dict[str, Any]:
    return {
        "objective": "regression_l2",
        "n_estimators": int(n_estimators),
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "subsample_freq": 1,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "random_state": seed,
        "n_jobs": config.n_jobs,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def fit_lightgbm_target(
    train_frame: pd.DataFrame,
    target: TargetSpec,
    *,
    feature_columns: Sequence[str] = METADATA_COLUMNS,
    config: LightGBMConfig,
    seed: int,
) -> tuple[LGBMRegressor, int]:
    valid = np.isfinite(train_frame[target.key].to_numpy(dtype=np.float64))
    frame = train_frame.loc[valid]
    if len(frame) < 20:
        raise ValueError(f"Only {len(frame)} valid training buildings for {target.display_name}")
    fit_frame, validation_frame = train_test_split(
        frame,
        test_size=config.validation_fraction,
        random_state=seed,
    )
    probe = LGBMRegressor(**_model_parameters(config, seed, config.max_estimators))
    probe.fit(
        fit_frame.loc[:, feature_columns],
        fit_frame[target.key],
        eval_set=[
            (
                validation_frame.loc[:, feature_columns],
                validation_frame[target.key],
            )
        ],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = max(1, int(probe.best_iteration_ or config.max_estimators))
    model = LGBMRegressor(**_model_parameters(config, seed, best_iteration))
    model.fit(frame.loc[:, feature_columns], frame[target.key])
    return model, best_iteration


def _finite_pair(target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(prediction)
    return target[finite], prediction[finite]


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    train_scale: float,
) -> dict[str, float]:
    target, prediction = _finite_pair(target, prediction)
    if not len(target):
        return {name: float("nan") for name in ("count", "rmse", "nrmse", "mae", "bias", "r2", "spearman")}
    correlation = (
        spearmanr(target, prediction).statistic
        if len(target) >= 3 and np.std(target) > 1e-12 and np.std(prediction) > 1e-12
        else float("nan")
    )
    return {
        "count": float(len(target)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "nrmse": float(np.sqrt(mean_squared_error(target, prediction)) / max(train_scale, 1e-8)),
        "mae": float(mean_absolute_error(target, prediction)),
        "bias": float(np.mean(prediction - target)),
        "r2": float(r2_score(target, prediction)) if len(target) >= 2 else float("nan"),
        "spearman": float(correlation),
    }


def paired_error_comparison(
    target: np.ndarray,
    lightgbm_prediction: np.ndarray,
    emulator_prediction: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 5000,
) -> dict[str, float]:
    """Paired bootstrap of LightGBM-minus-emulator held-out errors."""
    target = np.asarray(target, dtype=np.float64)
    lightgbm_prediction = np.asarray(lightgbm_prediction, dtype=np.float64)
    emulator_prediction = np.asarray(emulator_prediction, dtype=np.float64)
    finite = (
        np.isfinite(target)
        & np.isfinite(lightgbm_prediction)
        & np.isfinite(emulator_prediction)
    )
    target = target[finite]
    lightgbm_prediction = lightgbm_prediction[finite]
    emulator_prediction = emulator_prediction[finite]
    if not len(target):
        return {
            "count": 0.0,
            "rmse_difference": float("nan"),
            "rmse_difference_ci_low": float("nan"),
            "rmse_difference_ci_high": float("nan"),
            "probability_lightgbm_lower_rmse": float("nan"),
            "mae_difference": float("nan"),
            "mae_difference_ci_low": float("nan"),
            "mae_difference_ci_high": float("nan"),
            "probability_lightgbm_lower_mae": float("nan"),
        }

    def errors(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        true = target[indices]
        lightgbm_error = lightgbm_prediction[indices] - true
        emulator_error = emulator_prediction[indices] - true
        rmse_difference = np.sqrt(np.mean(lightgbm_error**2, axis=1)) - np.sqrt(
            np.mean(emulator_error**2, axis=1)
        )
        mae_difference = np.mean(np.abs(lightgbm_error), axis=1) - np.mean(
            np.abs(emulator_error), axis=1
        )
        return rmse_difference, mae_difference

    lightgbm_error = lightgbm_prediction - target
    emulator_error = emulator_prediction - target
    rmse_difference = float(
        np.sqrt(np.mean(lightgbm_error**2)) - np.sqrt(np.mean(emulator_error**2))
    )
    mae_difference = float(np.mean(np.abs(lightgbm_error)) - np.mean(np.abs(emulator_error)))
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(target), size=(bootstrap_samples, len(target)))
    bootstrap_rmse, bootstrap_mae = errors(sample_indices)
    return {
        "count": float(len(target)),
        "rmse_difference": rmse_difference,
        "rmse_difference_ci_low": float(np.quantile(bootstrap_rmse, 0.025)),
        "rmse_difference_ci_high": float(np.quantile(bootstrap_rmse, 0.975)),
        "probability_lightgbm_lower_rmse": float(np.mean(bootstrap_rmse < 0.0)),
        "mae_difference": mae_difference,
        "mae_difference_ci_low": float(np.quantile(bootstrap_mae, 0.025)),
        "mae_difference_ci_high": float(np.quantile(bootstrap_mae, 0.975)),
        "probability_lightgbm_lower_mae": float(np.mean(bootstrap_mae < 0.0)),
    }


def _load_emulator_predictions(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    legacy_required = {
        "profile_id",
        "horizon_hours",
        "sim_up_flex_wh_m2_k",
        "sim_down_flex_wh_m2_k",
        "emu_up_flex_wh_m2_k",
        "emu_down_flex_wh_m2_k",
    }
    if legacy_required.issubset(frame.columns):
        frame["profile_id"] = frame["profile_id"].astype(int)
        return frame

    scorer_required = {
        "signal",
        "profile_id",
        "horizon_hours",
        "up_flex_wh_m2_k",
        "down_flex_wh_m2_k",
    }
    if scorer_required.issubset(frame.columns):
        keys = ["profile_id", "horizon_hours"]
        value_columns = ["up_flex_wh_m2_k", "down_flex_wh_m2_k"]
        parts: list[pd.DataFrame] = []
        for signal, prefix in (("simulation", "sim"), ("emulation", "emu")):
            part = frame.loc[frame["signal"].eq(signal), [*keys, *value_columns]].copy()
            if part.empty:
                raise ValueError(f"Emulator KPI CSV has no {signal!r} rows")
            part = part.rename(
                columns={column: f"{prefix}_{column}" for column in value_columns}
            )
            parts.append(part)
        frame = parts[0].merge(parts[1], on=keys, how="inner", validate="one_to_one")
        frame["profile_id"] = frame["profile_id"].astype(int)
        return frame

    missing_legacy = sorted(legacy_required.difference(frame.columns))
    missing_scorer = sorted(scorer_required.difference(frame.columns))
    raise ValueError(
        "Emulator KPI CSV matches neither supported schema. "
        f"Legacy columns missing: {missing_legacy}; scorer columns missing: {missing_scorer}"
    )


def _emulator_target_frame(
    emulator: pd.DataFrame,
    target: TargetSpec,
    *,
    method: str = "emulator",
) -> pd.DataFrame:
    if emulator.empty:
        return emulator
    rows = emulator[np.isclose(emulator["horizon_hours"], target.horizon_hours)].copy()
    truth_column = f"{method}__csv_truth"
    return rows.loc[
        :,
        [
            "profile_id",
            f"sim_{target.direction}_flex_wh_m2_k",
            f"emu_{target.direction}_flex_wh_m2_k",
        ],
    ].rename(
        columns={
            f"sim_{target.direction}_flex_wh_m2_k": truth_column,
            f"emu_{target.direction}_flex_wh_m2_k": method,
        }
    )


def _write_scatter_dashboard(
    path: Path,
    predictions: pd.DataFrame,
    targets: Sequence[TargetSpec],
    emulator_methods: Sequence[str],
) -> None:
    horizons = sorted({target.horizon_hours for target in targets})
    figure = make_subplots(
        rows=len(horizons),
        cols=2,
        subplot_titles=[
            TargetSpec(direction, horizon, 0).display_name
            for horizon in horizons
            for direction in ("up", "down")
        ],
        horizontal_spacing=0.08,
        vertical_spacing=max(0.04, 0.12 / len(horizons)),
    )
    palette = ("#d95f02", "#6a3d9a", "#1f78b4", "#e31a1c", "#ff7f00")
    colors = {"lightgbm": "#16846b", "train_mean": "#757575"}
    colors.update(
        {
            method: palette[index % len(palette)]
            for index, method in enumerate(emulator_methods)
        }
    )
    for row, horizon in enumerate(horizons, start=1):
        for col, direction in enumerate(("up", "down"), start=1):
            target = next(
                item for item in targets
                if item.horizon_hours == horizon and item.direction == direction
            )
            frame = predictions[predictions["target_key"] == target.key]
            values = [frame["true"].to_numpy(), frame["lightgbm"].to_numpy()]
            values.extend(frame[method].to_numpy() for method in emulator_methods)
            finite = np.concatenate(values)
            finite = finite[np.isfinite(finite)]
            low = float(np.min(finite)) if len(finite) else 0.0
            high = float(np.max(finite)) if len(finite) else 1.0
            figure.add_trace(
                go.Scatter(
                    x=[low, high],
                    y=[low, high],
                    mode="lines",
                    line={"color": "black", "dash": "dash"},
                    name="ideal",
                    showlegend=row == 1 and col == 1,
                ),
                row=row,
                col=col,
            )
            methods = ["lightgbm", *emulator_methods]
            for method in methods:
                figure.add_trace(
                    go.Scattergl(
                        x=frame["true"],
                        y=frame[method],
                        mode="markers",
                        marker={"color": colors[method], "opacity": 0.55, "size": 6},
                        text=frame["profile_id"].astype(str),
                        hovertemplate=(
                            "profile=%{text}<br>true=%{x:.3f}<br>"
                            + method
                            + "=%{y:.3f}<extra></extra>"
                        ),
                        name=method,
                        legendgroup=method,
                        showlegend=row == 1 and col == 1,
                    ),
                    row=row,
                    col=col,
                )
            figure.update_xaxes(title_text="EnergyPlus Wh/(m2 K)", row=row, col=col)
            figure.update_yaxes(title_text="prediction Wh/(m2 K)", row=row, col=col)
    figure.update_layout(
        title="Metadata-only flexibility prediction on held-out buildings",
        template="plotly_white",
        height=max(700, 430 * len(horizons)),
        width=1400,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
    )
    figure.write_html(path)


def _write_score_dashboard(
    path: Path,
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    feature_columns: Sequence[str] = METADATA_COLUMNS,
) -> None:
    methods = list(dict.fromkeys(metrics["method"].tolist()))
    target_order = list(dict.fromkeys(metrics["target_display"].tolist()))
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Held-out normalized RMSE", "Normalized LightGBM gain importance"),
        specs=[[{"type": "xy"}], [{"type": "heatmap"}]],
        row_heights=[0.45, 0.55],
        vertical_spacing=0.16,
    )
    colors = {"lightgbm": "#16846b", "emulator": "#d95f02", "train_mean": "#757575"}
    for method in methods:
        frame = metrics[metrics["method"] == method].set_index("target_display").reindex(target_order)
        figure.add_trace(
            go.Bar(
                x=target_order,
                y=frame["nrmse"],
                name=method,
                marker_color=colors.get(method),
                text=frame["nrmse"].map(lambda value: f"{value:.3f}"),
                textposition="outside",
            ),
            row=1,
            col=1,
        )
    matrix = (
        importance.pivot(index="metadata", columns="target_display", values="gain_fraction")
        .reindex(index=feature_columns, columns=target_order)
    )
    figure.add_trace(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            colorscale="Viridis",
            colorbar={"title": "gain fraction"},
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="RMSE / training-target std", row=1, col=1)
    figure.update_layout(
        title="Direct metadata-to-flexibility benchmark",
        template="plotly_white",
        height=1100,
        width=1450,
        barmode="group",
        margin={"b": 170, "l": 180, "t": 100, "r": 70},
    )
    figure.write_html(path)


def _run_single_benchmark(
    args: argparse.Namespace,
    *,
    artifact: LoadedArtifact,
    dataset_path: Path,
    output_dir: Path,
    kpi_kind: KpiKind,
    kpi_estimator: str,
    emulator_kpi_csvs: dict[str, Path],
) -> Path:
    metadata = _artifact_metadata(artifact)
    metadata_columns = tuple(metadata.get("metadata_columns", METADATA_COLUMNS))
    if not metadata_columns:
        metadata_columns = tuple(METADATA_COLUMNS)
    train_ids = saved_profile_ids(artifact, "train")
    test_ids = saved_profile_ids(artifact, "test")
    if not train_ids or not test_ids:
        raise ValueError("Artifact must contain non-empty saved train and test profile IDs")

    event_config, targets = _resolve_targets(args.horizons_hours, args.dt_hours)
    event_config = EventStudyConfig(
        horizons_hours=event_config.horizons_hours,
        horizon_steps=event_config.horizon_steps,
        dt_hours=event_config.dt_hours,
        min_setpoint_change=args.min_setpoint_change,
        min_events=args.min_events,
        controls=args.controls,
        estimator=kpi_estimator,
    )
    include_availability = SPACE_HEATING_AVAILABILITY_COLUMN in metadata.get("input_columns", [])
    include_internal_gains = (
        INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in metadata.get("input_columns", [])
    )
    include_dhw_request = (
        DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN in metadata.get("input_columns", [])
    )
    hp_power_area_normalization = str(
        _metadata_value(metadata, "hp_power_area_normalization", "zone_floor_area")
    )
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    kpi_frame = extract_metadata_kpis(
        dataset_path,
        train_ids=train_ids,
        test_ids=test_ids,
        metadata_columns=metadata_columns,
        include_space_heating_availability=include_availability,
        include_internal_gains=include_internal_gains,
        include_dhw_request=include_dhw_request,
        hp_power_area_normalization=hp_power_area_normalization,
        event_config=event_config,
    )
    kpi_frame.to_csv(output_dir / "metadata_flexibility_targets.csv", index=False)
    train_frame = kpi_frame[kpi_frame["split"] == "train"].copy()
    test_frame = kpi_frame[kpi_frame["split"] == "test"].copy()
    emulators = {
        method: _load_emulator_predictions(path)
        for method, path in emulator_kpi_csvs.items()
    }
    lgbm_config = LightGBMConfig(
        learning_rate=args.learning_rate,
        max_estimators=args.max_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        validation_fraction=args.validation_fraction,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        n_jobs=args.n_jobs,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    model_manifest: dict[str, Any] = {}
    truth_mismatch: dict[str, float] = {}
    for target_index, target in enumerate(targets):
        model, best_iteration = fit_lightgbm_target(
            train_frame,
            target,
            feature_columns=metadata_columns,
            config=lgbm_config,
            seed=args.seed + target_index,
        )
        model_file = model_dir / f"{target.key}.txt"
        model.booster_.save_model(model_file)
        model_manifest[target.key] = {
            "direction": target.direction,
            "horizon_hours": target.horizon_hours,
            "horizon_steps": target.horizon_steps,
            "best_iteration": best_iteration,
            "model_file": str(model_file.relative_to(output_dir)),
        }

        true = test_frame[target.key].to_numpy(dtype=np.float64)
        lightgbm_prediction = model.predict(test_frame.loc[:, metadata_columns])
        train_values = train_frame[target.key].to_numpy(dtype=np.float64)
        train_values = train_values[np.isfinite(train_values)]
        train_mean = float(np.mean(train_values))
        train_scale = float(np.std(train_values))
        target_predictions = pd.DataFrame(
            {
                "profile_id": test_frame["profile_id"].astype(int).to_numpy(),
                "target_key": target.key,
                "target_display": target.display_name,
                "direction": target.direction,
                "horizon_hours": target.horizon_hours,
                "true": true,
                "lightgbm": lightgbm_prediction,
                "train_mean": np.full(len(test_frame), train_mean),
            }
        )
        emulator_methods: list[str] = []
        for emulator_index, (method, emulator) in enumerate(emulators.items()):
            emulator_target = _emulator_target_frame(
                emulator,
                target,
                method=method,
            )
            if emulator_target.empty:
                continue
            truth_column = f"{method}__csv_truth"
            target_predictions = target_predictions.merge(
                emulator_target,
                on="profile_id",
                how="left",
                validate="one_to_one",
            )
            mismatch = np.abs(
                target_predictions["true"].to_numpy(dtype=np.float64)
                - target_predictions[truth_column].to_numpy(dtype=np.float64)
            )
            truth_mismatch[f"{method}:{target.key}"] = float(np.nanmax(mismatch))
            target_predictions = target_predictions.drop(columns=truth_column)
            emulator_methods.append(method)
            comparison_rows.append(
                {
                    "emulator_model": method,
                    "target_key": target.key,
                    "target_display": target.display_name,
                    "direction": target.direction,
                    "horizon_hours": target.horizon_hours,
                    **paired_error_comparison(
                        target_predictions["true"].to_numpy(),
                        target_predictions["lightgbm"].to_numpy(),
                        target_predictions[method].to_numpy(),
                        seed=(
                            args.seed
                            + 1000
                            + target_index
                            + 100 * emulator_index
                        ),
                    ),
                }
            )
        prediction_rows.extend(target_predictions.to_dict(orient="records"))

        metric_start = len(metric_rows)
        methods = ["train_mean", "lightgbm"]
        methods.extend(emulator_methods)
        for method in methods:
            values = regression_metrics(
                target_predictions["true"].to_numpy(),
                target_predictions[method].to_numpy(),
                train_scale=train_scale,
            )
            metric_rows.append(
                {
                    "target_key": target.key,
                    "target_display": target.display_name,
                    "direction": target.direction,
                    "horizon_hours": target.horizon_hours,
                    "method": method,
                    "train_target_mean": train_mean,
                    "train_target_std": train_scale,
                    "best_iteration": best_iteration if method == "lightgbm" else np.nan,
                    **values,
                }
            )

        gains = model.booster_.feature_importance(importance_type="gain").astype(np.float64)
        gain_total = float(np.sum(gains))
        fractions = gains / gain_total if gain_total > 0.0 else np.zeros_like(gains)
        for column, gain, fraction in zip(metadata_columns, gains, fractions):
            importance_rows.append(
                {
                    "target_key": target.key,
                    "target_display": target.display_name,
                    "metadata": column,
                    "gain": float(gain),
                    "gain_fraction": float(fraction),
                }
            )
        print(
            f"metadata_flex_model={target.key} best_iteration={best_iteration} "
            f"test_nrmse={metric_rows[metric_start + 1]['nrmse']:.4f}"
        )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    importance = pd.DataFrame(importance_rows)
    comparison = pd.DataFrame(comparison_rows)
    predictions.to_csv(output_dir / "metadata_flexibility_predictions.csv", index=False)
    metrics.to_csv(output_dir / "metadata_flexibility_metrics.csv", index=False)
    importance.to_csv(output_dir / "metadata_flexibility_feature_importance.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(
            output_dir / "metadata_flexibility_emulator_comparison.csv",
            index=False,
        )
    emulator_methods = [method for method in emulators if method in predictions.columns]
    _write_scatter_dashboard(
        output_dir / "metadata_flexibility_scatter.html",
        predictions,
        targets,
        emulator_methods,
    )
    _write_score_dashboard(
        output_dir / "metadata_flexibility_dashboard.html",
        metrics,
        importance,
        metadata_columns,
    )
    summary = {
        "artifact_dir": str(args.artifact_dir),
        "dataset": str(dataset_path),
        "kpi_kind": kpi_kind,
        "train_profiles": len(train_ids),
        "test_profiles": len(test_ids),
        "metadata_columns": list(metadata_columns),
        "metadata_encoding": "raw numeric values; LightGBM handles missing values; no trajectory inputs",
        "hp_power_area_normalization": hp_power_area_normalization,
        "event_study": {
            "horizons_hours": list(event_config.horizons_hours),
            "horizon_steps": list(event_config.horizon_steps),
            "dt_hours": event_config.dt_hours,
            "min_setpoint_change": event_config.min_setpoint_change,
            "min_events": event_config.min_events,
            "kpi_estimator": event_config.estimator,
            "controls": event_config.controls,
            "controls_applied": (
                event_config.estimator == "regression" and event_config.controls != "none"
            ),
        },
        "lightgbm": asdict(lgbm_config),
        "models": model_manifest,
        "emulator_kpi_csv": (
            str(emulator_kpi_csvs["emulator"])
            if "emulator" in emulator_kpi_csvs
            else None
        ),
        "emulator_kpi_csvs": {
            method: str(path) for method, path in emulator_kpi_csvs.items()
        },
        "emulator_truth_max_abs_mismatch": truth_mismatch,
        "metrics": metric_rows,
        "paired_lightgbm_minus_emulator": comparison_rows,
    }
    (output_dir / "metadata_flexibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"saved_metadata_flexibility_benchmark={output_dir} "
        f"kpi={kpi_kind} estimator={kpi_estimator}"
    )
    return output_dir


def _write_multi_kpi_dashboard(path: Path, metrics: pd.DataFrame) -> None:
    figure = go.Figure()
    colors = {
        "train_mean": "#757575",
        "lightgbm": "#16846b",
        "emulator": "#d95f02",
    }
    for (kpi_kind, method), frame in metrics.groupby(["kpi_kind", "method"]):
        figure.add_trace(
            go.Bar(
                x=[f"{kpi_kind}: {value}" for value in frame["target_display"]],
                y=frame["nrmse"],
                name=f"{kpi_kind}: {method}",
                marker_color=colors.get(str(method)),
            )
        )
    figure.update_layout(
        title="Metadata-to-flexibility benchmark across KPI definitions",
        template="plotly_white",
        width=1500,
        height=760,
        barmode="group",
        xaxis_title="KPI and response horizon",
        yaxis_title="RMSE / training-target standard deviation",
        legend={"orientation": "h", "y": -0.3},
        margin={"b": 230},
    )
    figure.write_html(path)


def run_benchmark(args: argparse.Namespace) -> Path:
    artifact = load_artifact(args.artifact_dir)
    if artifact.spec.task != "closed_loop_hp":
        raise ValueError("Metadata flexibility prediction requires a closed-loop HP artifact")
    dataset_path, dataset_source = _resolve_dataset_path(artifact, args.dataset)
    if args.dataset is None:
        print(f"dataset=inferred_from_artifact path={dataset_path}")
    else:
        print(f"dataset=explicit_override path={dataset_path}")

    explicit_kpis = args.kpis is not None
    kpi_kinds = _normalize_kpi_kinds(
        args.kpis if explicit_kpis else (args.kpi_estimator,)
    )
    root_output = Path(args.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    emulator_paths = _resolve_emulator_kpi_csvs(
        args.emulator_kpi_csv,
        kpi_kinds,
    )
    emulator_kpi_sources: dict[str, str] = {}
    if not args.emulator_kpi_csv:
        emulator_paths = _generate_emulator_kpi_csvs(
            args,
            artifact_dir=args.artifact_dir,
            artifact=artifact,
            dataset_path=dataset_path,
            kpi_kinds=kpi_kinds,
            output_dir=root_output / "emulator_scoring",
        )
        emulator_kpi_sources["emulator"] = "generated"
    else:
        emulator_kpi_sources["emulator"] = "explicit"

    emulator_paths_by_kpi: dict[KpiKind, dict[str, Path]] = {
        kind: ({"emulator": emulator_paths[kind]} if kind in emulator_paths else {})
        for kind in kpi_kinds
    }
    additional_artifact_dirs = _resolve_additional_emulator_artifacts(
        args.additional_emulator_artifact
    )
    for label, artifact_dir in additional_artifact_dirs.items():
        additional_artifact = load_artifact(artifact_dir)
        _validate_comparison_artifact(
            artifact,
            additional_artifact,
            label=label,
        )
        additional_paths = _generate_emulator_kpi_csvs(
            args,
            artifact_dir=artifact_dir,
            artifact=additional_artifact,
            dataset_path=dataset_path,
            kpi_kinds=kpi_kinds,
            output_dir=root_output / "additional_emulator_scoring" / label,
        )
        for kind, path in additional_paths.items():
            emulator_paths_by_kpi[kind][label] = path
        emulator_kpi_sources[label] = "generated"

    result_dirs: dict[KpiKind, Path] = {}
    for kind in kpi_kinds:
        output_dir = root_output / kind if explicit_kpis else root_output
        result_dirs[kind] = _run_single_benchmark(
            args,
            artifact=artifact,
            dataset_path=dataset_path,
            output_dir=output_dir,
            kpi_kind=kind,
            kpi_estimator=KPI_ESTIMATORS[kind],
            emulator_kpi_csvs=emulator_paths_by_kpi[kind],
        )

    if explicit_kpis:
        metric_frames: list[pd.DataFrame] = []
        comparison_frames: list[pd.DataFrame] = []
        for kind, result_dir in result_dirs.items():
            metrics = pd.read_csv(result_dir / "metadata_flexibility_metrics.csv")
            metrics.insert(0, "kpi_kind", kind)
            metric_frames.append(metrics)
            comparison_path = result_dir / "metadata_flexibility_emulator_comparison.csv"
            if comparison_path.exists():
                comparison = pd.read_csv(comparison_path)
                comparison.insert(0, "kpi_kind", kind)
                comparison_frames.append(comparison)
        combined_metrics = pd.concat(metric_frames, ignore_index=True)
        combined_metrics.to_csv(
            root_output / "metadata_flexibility_all_metrics.csv", index=False
        )
        if comparison_frames:
            pd.concat(comparison_frames, ignore_index=True).to_csv(
                root_output / "metadata_flexibility_all_emulator_comparisons.csv",
                index=False,
            )
        dashboard = root_output / "metadata_flexibility_all_kpis.html"
        _write_multi_kpi_dashboard(dashboard, combined_metrics)
        (root_output / "metadata_flexibility_all_kpis_summary.json").write_text(
            json.dumps(
                {
                    "artifact_dir": str(args.artifact_dir),
                    "dataset": str(dataset_path),
                    "dataset_source": dataset_source,
                    "kpis": list(kpi_kinds),
                    "result_directories": {
                        kind: str(path) for kind, path in result_dirs.items()
                    },
                    "emulator_kpi_csvs": {
                        kind: {
                            method: str(path)
                            for method, path in method_paths.items()
                        }
                        for kind, method_paths in emulator_paths_by_kpi.items()
                    },
                    "emulator_kpi_sources": emulator_kpi_sources,
                    "additional_emulator_artifacts": {
                        label: str(path)
                        for label, path in additional_artifact_dirs.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"saved_metadata_flexibility_multi_kpi_dashboard={dashboard}")
    return root_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM regressors from emulator metadata to simulated flexibility KPIs."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional dataset override; defaults to dataset_path saved in the artifact.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--emulator-kpi-csv",
        action="append",
        default=[],
        metavar="[KPI=]PATH",
        help=(
            "Optional precomputed emulator KPI table. For --kpis with multiple "
            "entries, repeat as f=PATH and regression_based=PATH. A bare path "
            "remains valid for a single KPI. If omitted, the saved emulator is "
            "rolled out automatically on its test split."
        ),
    )
    parser.add_argument(
        "--emulator-num-scenarios",
        type=int,
        default=100,
        help="Linked full-profile scenarios used for automatic probabilistic scoring.",
    )
    parser.add_argument(
        "--additional-emulator-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Additional saved closed-loop emulator to score on the reference test "
            "split and include in every comparison. Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--emulator-hp-scenario-mode",
        choices=("bernoulli", "expected"),
        default="bernoulli",
        help="HP emission used when automatically rolling out a probabilistic artifact.",
    )
    parser.add_argument(
        "--emulator-ventilation-rollout-mode",
        choices=("recorded", "eplus_rule"),
        default="recorded",
        help="Ventilation input used during automatic full-profile emulator rollouts.",
    )
    parser.add_argument(
        "--emulator-max-profiles",
        type=int,
        help="Optional test-profile cap for automatic scoring; defaults to the full test split.",
    )
    parser.add_argument(
        "--emulator-num-profile-plots",
        type=int,
        default=0,
        help="Number of annual trace plots written by the automatic scorer.",
    )
    parser.add_argument(
        "--kpis",
        nargs="+",
        choices=tuple(KPI_ALIASES),
        help=(
            "Evaluate multiple KPI definitions in one run. Canonical names are "
            "'f' and 'regression_based'; direct_ratio/regression are accepted aliases."
        ),
    )
    parser.add_argument("--horizons-hours", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument(
        "--kpi-estimator",
        choices=["direct_ratio", "regression"],
        default="direct_ratio",
    )
    parser.add_argument("--controls", choices=["none", "weather", "full"], default="full")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
