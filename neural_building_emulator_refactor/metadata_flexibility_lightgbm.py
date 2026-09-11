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
from .profiles import saved_profile_ids


Direction = Literal["up", "down"]


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


def _emulator_target_frame(emulator: pd.DataFrame, target: TargetSpec) -> pd.DataFrame:
    if emulator.empty:
        return emulator
    rows = emulator[np.isclose(emulator["horizon_hours"], target.horizon_hours)].copy()
    return rows.loc[
        :,
        [
            "profile_id",
            f"sim_{target.direction}_flex_wh_m2_k",
            f"emu_{target.direction}_flex_wh_m2_k",
        ],
    ].rename(
        columns={
            f"sim_{target.direction}_flex_wh_m2_k": "emulator_csv_truth",
            f"emu_{target.direction}_flex_wh_m2_k": "emulator",
        }
    )


def _write_scatter_dashboard(
    path: Path,
    predictions: pd.DataFrame,
    targets: Sequence[TargetSpec],
    has_emulator: bool,
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
    colors = {"lightgbm": "#16846b", "emulator": "#d95f02", "train_mean": "#757575"}
    for row, horizon in enumerate(horizons, start=1):
        for col, direction in enumerate(("up", "down"), start=1):
            target = next(
                item for item in targets
                if item.horizon_hours == horizon and item.direction == direction
            )
            frame = predictions[predictions["target_key"] == target.key]
            values = [frame["true"].to_numpy(), frame["lightgbm"].to_numpy()]
            if has_emulator:
                values.append(frame["emulator"].to_numpy())
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
            methods = ["lightgbm", "emulator"] if has_emulator else ["lightgbm"]
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


def run_benchmark(args: argparse.Namespace) -> Path:
    artifact = load_artifact(args.artifact_dir)
    if artifact.spec.task != "closed_loop_hp":
        raise ValueError("Metadata flexibility prediction requires a closed-loop HP artifact")
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
        estimator=args.kpi_estimator,
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
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    kpi_frame = extract_metadata_kpis(
        Path(args.dataset),
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
    emulator = _load_emulator_predictions(args.emulator_kpi_csv)
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
        emulator_target = _emulator_target_frame(emulator, target)
        if not emulator_target.empty:
            target_predictions = target_predictions.merge(
                emulator_target,
                on="profile_id",
                how="left",
                validate="one_to_one",
            )
            mismatch = np.abs(
                target_predictions["true"].to_numpy(dtype=np.float64)
                - target_predictions["emulator_csv_truth"].to_numpy(dtype=np.float64)
            )
            truth_mismatch[target.key] = float(np.nanmax(mismatch))
            target_predictions = target_predictions.drop(columns="emulator_csv_truth")
            comparison_rows.append(
                {
                    "target_key": target.key,
                    "target_display": target.display_name,
                    "direction": target.direction,
                    "horizon_hours": target.horizon_hours,
                    **paired_error_comparison(
                        target_predictions["true"].to_numpy(),
                        target_predictions["lightgbm"].to_numpy(),
                        target_predictions["emulator"].to_numpy(),
                        seed=args.seed + 1000 + target_index,
                    ),
                }
            )
        prediction_rows.extend(target_predictions.to_dict(orient="records"))

        methods = ["train_mean", "lightgbm"]
        if "emulator" in target_predictions:
            methods.append("emulator")
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
            f"test_nrmse={metric_rows[-1 if methods[-1] == 'lightgbm' else -2]['nrmse']:.4f}"
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
    has_emulator = "emulator" in predictions.columns
    _write_scatter_dashboard(
        output_dir / "metadata_flexibility_scatter.html",
        predictions,
        targets,
        has_emulator,
    )
    _write_score_dashboard(
        output_dir / "metadata_flexibility_dashboard.html",
        metrics,
        importance,
        metadata_columns,
    )
    summary = {
        "artifact_dir": str(args.artifact_dir),
        "dataset": str(args.dataset),
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
        "emulator_kpi_csv": str(args.emulator_kpi_csv) if args.emulator_kpi_csv else None,
        "emulator_truth_max_abs_mismatch": truth_mismatch,
        "metrics": metric_rows,
        "paired_lightgbm_minus_emulator": comparison_rows,
    }
    (output_dir / "metadata_flexibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"saved_metadata_flexibility_benchmark={output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM regressors from emulator metadata to simulated flexibility KPIs."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--emulator-kpi-csv", type=Path)
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
