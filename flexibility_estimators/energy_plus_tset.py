"""Setpoint-shock flexibility KPIs for EnergyPlus Tset-controlled exports.

Mirrors the event-study definition in
``neural_building_emulator/flexibility_event_study_kpis.py``:

    Delta P_H = mean(P[t:t+H]) - mean(P[t-H:t])
    Delta P_H = alpha + beta_+ max(Delta T_set, 0) + beta_- min(Delta T_set, 0)
                + controls

Reported KPIs (power in W/m2 of heated area):

    up_flex   = H * beta_+   [Wh/(m2 K)]
    down_flex = H * beta_-   [Wh/(m2 K)]

Expected layout under --data-dir (default: data/control=setpoint)::

    egid=<id>/
      timeseries.parquet
      simulation_metadata.json

Uses FL0 thermostat / Tin / ventilation, building-level weather, and
technical-room HP electric power (DHW and summer SH lockout zeroed, matching
the closed-loop loader).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "control=setpoint"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "energy_plus_tset_plots"

DT_HOURS = 0.25
ControlsMode = Literal["none", "weather", "full"]

SETPOINT_COL = "Zone Thermostat Heating Setpoint Temperature"
TIN_COL = "Zone Air Temperature"
VENT_COL = "Zone Ventilation Standard Density Volume Flow Rate"
OUTDOOR_COL = "Environment, Site Outdoor Air Drybulb Temperature"
SOLAR_COL = "Environment, Site Global Horizontal Solar Radiation Rate per Area"
PEL_COL = "heat_pump_electric_power"
DHW_COL = "hp_mode_is_dhw"

TS_COLUMNS = [
    "datetime",
    "zone",
    SETPOINT_COL,
    TIN_COL,
    VENT_COL,
    OUTDOOR_COL,
    SOLAR_COL,
    PEL_COL,
    DHW_COL,
]

FEATURE_COLS = [
    "buildingType",
    "totalFloors",
    "constructionPeriod",
    "wallsRenovationPeriod",
    "floorsRenovationPeriod",
    "roofRenovationPeriod",
    "windowsRenovationPeriod",
    "dwellingNumber",
    "shSetpoint",
    "floor_area",
    "volume",
    "envelope_area",
    "window_area",
    "surface_to_volume_ratio",
    "thermal_mass_class",
    "hp_ref_capacity_W",
    "hp_ref_cop",
    "SH_design_cap_W",
    "DHW_recovery_W",
    "dhwVolume_m3",
    "shVolume_m3",
    "m2",
    "resolved_ach_h_1",
]


@dataclass(frozen=True)
class FlexibilityKpi:
    egid: int
    horizon_hours: float
    horizon_steps: int
    event_count: int
    valid: bool
    beta_plus_w_m2_k: float
    beta_minus_w_m2_k: float
    up_flex_wh_m2_k: float
    down_flex_wh_m2_k: float


def _egid_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith("egid=")
    )


def _load_metadata(building_dir: Path) -> dict:
    meta_path = building_dir / "simulation_metadata.json"
    payload = json.loads(meta_path.read_text())
    values = dict(payload.get("gardenCompatibility", {}).get("values", {}))
    if "egid" not in values:
        values["egid"] = int(building_dir.name.split("=", 1)[1])
    meter = payload.get("meterSummary", {})
    values["annual_house_load_kwh"] = meter.get("total_house_load (kWh)")
    infiltration = payload.get("resolved_infiltration", {})
    values["resolved_ach_h_1"] = infiltration.get("resolved_ach_h_1")
    return values


def _is_space_heating_hp(meta: dict) -> bool:
    binding = str(meta.get("hp_size_binding", "")).strip().upper()
    capacity = meta.get("hp_ref_capacity_W")
    try:
        capacity_ok = capacity is not None and float(capacity) > 0.0
    except (TypeError, ValueError):
        capacity_ok = False
    return binding == "SH" and capacity_ok


def _space_heating_availability(datetimes: pd.DatetimeIndex) -> np.ndarray:
    """EnergyPlus May 15 through September 30 SH lockout (0 during lockout)."""
    month = datetimes.month.to_numpy()
    day = datetimes.day.to_numpy()
    summer_lockout = (
        ((month == 5) & (day >= 15))
        | ((month > 5) & (month < 9))
        | (month == 9)
    )
    return (~summer_lockout).astype(np.float64)


def _calendar_features(datetimes: pd.DatetimeIndex) -> np.ndarray:
    hour = datetimes.hour.to_numpy(dtype=np.float64) + datetimes.minute.to_numpy(
        dtype=np.float64
    ) / 60.0
    day_of_year = datetimes.dayofyear.to_numpy(dtype=np.float64)
    hour_angle = 2.0 * np.pi * hour / 24.0
    year_angle = 2.0 * np.pi * (day_of_year - 1.0) / 365.0
    return np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(year_angle),
            np.cos(year_angle),
        ]
    )


def _standardized_controls(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    values = values.astype(np.float64, copy=True)
    values = np.where(np.isfinite(values), values, np.nan)
    means = np.nanmean(values, axis=0)
    values = np.where(np.isfinite(values), values, means)
    scales = np.nanstd(values, axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    return (values - means) / scales


def _window_mean(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.mean(values[int(start) : int(end)], axis=0) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )


def _previous_event_deltas(
    all_events: np.ndarray,
    all_deltas: np.ndarray,
    events: np.ndarray,
) -> np.ndarray:
    if len(events) == 0:
        return np.empty(0, dtype=np.float64)
    previous_by_event: dict[int, float] = {}
    previous = 0.0
    for event, delta in zip(all_events, all_deltas):
        previous_by_event[int(event)] = float(previous)
        previous = float(delta)
    return np.asarray(
        [previous_by_event.get(int(event), 0.0) for event in events],
        dtype=np.float64,
    )


def _invalid_kpi(egid: int, horizon_hours: float, horizon_steps: int, event_count: int) -> FlexibilityKpi:
    return FlexibilityKpi(
        egid=egid,
        horizon_hours=float(horizon_hours),
        horizon_steps=int(horizon_steps),
        event_count=int(event_count),
        valid=False,
        beta_plus_w_m2_k=float("nan"),
        beta_minus_w_m2_k=float("nan"),
        up_flex_wh_m2_k=float("nan"),
        down_flex_wh_m2_k=float("nan"),
    )


def estimate_event_study_kpi(
    *,
    egid: int,
    setpoint: np.ndarray,
    tin: np.ndarray,
    pel_wm2: np.ndarray,
    disturbances: np.ndarray,
    exogenous_tail: np.ndarray,
    horizon_hours: float,
    horizon_steps: int,
    min_setpoint_change: float,
    min_events: int,
    controls: ControlsMode,
) -> FlexibilityKpi:
    """OLS event-study beta for one building and one horizon."""
    setpoint = np.asarray(setpoint, dtype=np.float64)
    deltas_all = setpoint[1:] - setpoint[:-1]
    all_events = np.flatnonzero(np.abs(deltas_all) >= min_setpoint_change) + 1
    all_deltas = setpoint[all_events] - setpoint[all_events - 1]

    events = all_events[
        (all_events >= horizon_steps) & (all_events + horizon_steps <= len(setpoint))
    ]
    if len(events) == 0:
        return _invalid_kpi(egid, horizon_hours, horizon_steps, event_count=0)

    deltas = setpoint[events] - setpoint[events - 1]
    pre_starts = events - horizon_steps
    pre_ends = events
    post_starts = events
    post_ends = events + horizon_steps
    power = np.asarray(pel_wm2, dtype=np.float64)
    response = _window_mean(power, post_starts, post_ends) - _window_mean(
        power, pre_starts, pre_ends
    )

    columns = [
        np.ones_like(deltas, dtype=np.float64),
        np.maximum(deltas, 0.0),
        np.minimum(deltas, 0.0),
    ]

    if controls != "none":
        disturbances = np.asarray(disturbances, dtype=np.float64)
        exogenous_tail = np.asarray(exogenous_tail, dtype=np.float64)
        disturbances_calendar = np.column_stack(
            [disturbances[events], exogenous_tail[events]]
        )
        pre_disturbance = _window_mean(disturbances, pre_starts, pre_ends)
        post_disturbance = _window_mean(disturbances, post_starts, post_ends)
        disturbance_delta = post_disturbance - pre_disturbance
        control_blocks = [disturbances_calendar, disturbance_delta]
        if controls == "full":
            pre_power = _window_mean(power, pre_starts, pre_ends)
            pre_temperature = tin[events]
            pre_setpoint = setpoint[events - 1]
            previous_delta = _previous_event_deltas(all_events, all_deltas, events)
            control_blocks.insert(
                0,
                np.column_stack(
                    [pre_power, pre_temperature, pre_setpoint, previous_delta]
                ),
            )
        columns.append(_standardized_controls(np.column_stack(control_blocks)))

    design = np.column_stack(columns).astype(np.float64)
    event_count = int(len(response))
    if event_count < min_events or event_count <= design.shape[1]:
        return _invalid_kpi(egid, horizon_hours, horizon_steps, event_count=event_count)

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    beta_plus = float(coefficients[1])
    beta_minus = float(coefficients[2])
    return FlexibilityKpi(
        egid=int(egid),
        horizon_hours=float(horizon_hours),
        horizon_steps=int(horizon_steps),
        event_count=event_count,
        valid=True,
        beta_plus_w_m2_k=beta_plus,
        beta_minus_w_m2_k=beta_minus,
        up_flex_wh_m2_k=float(horizon_hours * beta_plus),
        down_flex_wh_m2_k=float(horizon_hours * beta_minus),
    )


def load_building_series(building_dir: Path, meta: dict) -> tuple[pd.DataFrame, float] | None:
    """Return aligned FL0 series with Pel in W/m2, or None if unusable."""
    floor_area = float(meta.get("floor_area", np.nan))
    total_floors = float(meta.get("totalFloors", np.nan))
    if not np.isfinite(floor_area) or floor_area <= 0.0:
        return None
    if not np.isfinite(total_floors) or total_floors <= 0.0:
        return None
    heated_area = floor_area * total_floors

    frame = pd.read_parquet(building_dir / "timeseries.parquet", columns=TS_COLUMNS)
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    fl0 = (
        frame.loc[
            frame["zone"] == "fl0_thz0",
            ["datetime", SETPOINT_COL, TIN_COL, VENT_COL],
        ]
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .set_index("datetime")
    )
    building = (
        frame.loc[
            frame["zone"] == "building_total",
            ["datetime", OUTDOOR_COL, SOLAR_COL],
        ]
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .set_index("datetime")
    )
    technical = (
        frame.loc[
            frame["zone"] == "technical_room",
            ["datetime", PEL_COL, DHW_COL],
        ]
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .set_index("datetime")
    )

    aligned = pd.concat(
        {
            "setpoint": fl0[SETPOINT_COL],
            "tin": fl0[TIN_COL],
            "vent": fl0[VENT_COL],
            "tout": building[OUTDOOR_COL],
            "solar": building[SOLAR_COL],
            "pel_w": technical[PEL_COL],
            "dhw": technical[DHW_COL],
        },
        axis=1,
    ).sort_index()
    aligned = aligned.dropna()
    if len(aligned) < 8:
        return None

    availability = _space_heating_availability(aligned.index)
    dhw = aligned["dhw"].to_numpy(dtype=np.float64) > 0.5
    pel_w = np.maximum(aligned["pel_w"].to_numpy(dtype=np.float64), 0.0)
    pel_wm2 = pel_w / heated_area
    pel_wm2 = np.where(dhw | (availability < 0.5), 0.0, pel_wm2)

    out = aligned.copy()
    out["pel_wm2"] = pel_wm2
    out["sh_available"] = availability
    calendar = _calendar_features(aligned.index)
    out["hour_sin"] = calendar[:, 0]
    out["hour_cos"] = calendar[:, 1]
    out["year_sin"] = calendar[:, 2]
    out["year_cos"] = calendar[:, 3]
    return out, heated_area


def estimate_building_kpis(
    series: pd.DataFrame,
    *,
    egid: int,
    horizons_hours: tuple[float, ...],
    dt_hours: float,
    min_setpoint_change: float,
    min_events: int,
    controls: ControlsMode,
) -> list[FlexibilityKpi]:
    setpoint = series["setpoint"].to_numpy(dtype=np.float64)
    tin = series["tin"].to_numpy(dtype=np.float64)
    pel_wm2 = series["pel_wm2"].to_numpy(dtype=np.float64)
    disturbances = series[["tout", "solar", "vent"]].to_numpy(dtype=np.float64)
    exogenous_tail = series[
        ["sh_available", "hour_sin", "hour_cos", "year_sin", "year_cos"]
    ].to_numpy(dtype=np.float64)

    kpis: list[FlexibilityKpi] = []
    for horizon_hours in horizons_hours:
        horizon_steps = int(round(horizon_hours / dt_hours))
        if horizon_steps < 1:
            kpis.append(_invalid_kpi(egid, horizon_hours, horizon_steps, event_count=0))
            continue
        effective_hours = float(horizon_steps * dt_hours)
        kpis.append(
            estimate_event_study_kpi(
                egid=egid,
                setpoint=setpoint,
                tin=tin,
                pel_wm2=pel_wm2,
                disturbances=disturbances,
                exogenous_tail=exogenous_tail,
                horizon_hours=effective_hours,
                horizon_steps=horizon_steps,
                min_setpoint_change=min_setpoint_change,
                min_events=min_events,
                controls=controls,
            )
        )
    return kpis


def collect_kpis(
    data_dir: Path,
    *,
    max_buildings: int | None,
    horizons_hours: tuple[float, ...],
    dt_hours: float,
    min_setpoint_change: float,
    min_events: int,
    controls: ControlsMode,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (meta indexed by egid, long KPI table)."""
    dirs = _egid_dirs(data_dir)
    if max_buildings is not None:
        dirs = dirs[: max(0, int(max_buildings))]

    meta_rows: list[dict] = []
    kpi_rows: list[dict] = []

    for index, building_dir in enumerate(dirs, start=1):
        meta = _load_metadata(building_dir)
        if not _is_space_heating_hp(meta):
            continue
        egid = int(meta["egid"])
        loaded = load_building_series(building_dir, meta)
        if loaded is None:
            continue
        series, heated_area = loaded
        annual_hp_kwh = float((series["pel_w"].clip(lower=0.0) * DT_HOURS / 1000.0).sum())

        meta_rows.append(
            {
                **meta,
                "egid": egid,
                "annual_consumption": annual_hp_kwh,
                "m2": heated_area,
                "U": float(meta.get("surface_to_volume_ratio", np.nan)),
            }
        )
        for kpi in estimate_building_kpis(
            series,
            egid=egid,
            horizons_hours=horizons_hours,
            dt_hours=dt_hours,
            min_setpoint_change=min_setpoint_change,
            min_events=min_events,
            controls=controls,
        ):
            kpi_rows.append(
                {
                    "egid": kpi.egid,
                    "horizon_hours": kpi.horizon_hours,
                    "horizon_steps": kpi.horizon_steps,
                    "event_count": kpi.event_count,
                    "valid": kpi.valid,
                    "beta_plus_w_m2_k": kpi.beta_plus_w_m2_k,
                    "beta_minus_w_m2_k": kpi.beta_minus_w_m2_k,
                    "up_flex_wh_m2_k": kpi.up_flex_wh_m2_k,
                    "down_flex_wh_m2_k": kpi.down_flex_wh_m2_k,
                }
            )

        if index % 50 == 0 or index == len(dirs):
            print(f"processed {index}/{len(dirs)} folders; kept {len(meta_rows)} HP buildings")

    if not meta_rows:
        raise ValueError(f"No space-heating HP buildings found under {data_dir}")

    meta = pd.DataFrame(meta_rows).set_index("egid").sort_index()
    kpis = pd.DataFrame(kpi_rows)
    return meta, kpis


def _save_or_show(fig: plt.Figure, output_dir: Path | None, name: str) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _kpi_distribution_plot(
    kpis: pd.DataFrame,
    *,
    horizon_hours: float,
    output_dir: Path | None,
    filename: str,
) -> None:
    """Histogram of up/down flex, matching ``energy_plus_turnoff`` KPI dist style."""
    fig, ax = plt.subplots(1, 1, layout="constrained")
    kpis[["up_flex_wh_m2_k", "down_flex_wh_m2_k"]].plot.hist(alpha=0.5, bins=50, ax=ax)
    ax.set_xlabel(rf"Event-study flexibility KPI @ {horizon_hours:g}h [Wh/(m²·K)]")
    ax.set_title(rf"$H\cdot\beta$  ($H={horizon_hours:g}\,\mathrm{{h}}$)")
    ax.legend([r"up_flex $H\cdot\beta_+$", r"down_flex $H\cdot\beta_-$"])
    ax.spines[["top", "right"]].set_visible(False)
    _save_or_show(fig, output_dir, filename)


def _pred_vs_true_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output_dir: Path | None,
    filename: str,
) -> None:
    """Hold-out scatter with identity line, matching ``energy_plus_turnoff``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, layout="constrained")
    ax.scatter(y_true, y_pred, alpha=0.8)
    lims = [
        float(np.nanmin([np.min(y_true), np.min(y_pred), 0.0])),
        float(np.nanmax([np.max(y_true), np.max(y_pred)])),
    ]
    ax.plot(lims, lims, c="k", ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    _save_or_show(fig, output_dir, filename)


def _parse_horizons(raw: str, dt_hours: float) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--horizons-hours must contain positive values")
    steps = [max(1, int(round(h / dt_hours))) for h in values]
    effective = [float(step * dt_hours) for step in steps]
    # Deduplicate by step count while preserving order.
    seen: set[int] = set()
    out: list[float] = []
    for step, hours in zip(steps, effective):
        if step in seen:
            continue
        seen.add(step)
        out.append(hours)
    return tuple(out)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    if len(y_true) > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "corr": corr,
        "pred_std": float(np.std(y_pred)),
        "y_std": float(np.std(y_true)),
    }


def _baseline_lgbm_params(n_train: int, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": max(1, min(20, n_train // 5)),
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
        "verbosity": -1,
    }


def _suggest_lgbm_params(trial: optuna.Trial, *, n_train: int, seed: int) -> dict[str, Any]:
    max_min_child = max(5, min(50, max(5, n_train // 3)))
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, max_min_child),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": seed,
        "verbosity": -1,
    }


def _cv_mae(
    params: dict[str, Any],
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    n_folds: int,
    seed: int,
) -> float:
    n_folds = min(n_folds, len(x_tr))
    if n_folds < 2:
        model = LGBMRegressor(**params).fit(x_tr, y_tr)
        pred = model.predict(x_tr)
        return float(np.mean(np.abs(pred - y_tr.to_numpy())))

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, val_idx in folds.split(x_tr):
        model = LGBMRegressor(**params).fit(x_tr.iloc[train_idx], y_tr.iloc[train_idx])
        pred = model.predict(x_tr.iloc[val_idx])
        scores.append(float(np.mean(np.abs(pred - y_tr.iloc[val_idx].to_numpy()))))
    return float(np.mean(scores))


def _optuna_search(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    n_trials: int,
    n_folds: int,
    seed: int,
    label: str = "",
) -> tuple[dict[str, Any], float]:
    """Tune LightGBM on the training split only; return best params and CV MAE."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_lgbm_params(trial, n_train=len(x_tr), seed=seed)
        return _cv_mae(params, x_tr, y_tr, n_folds=n_folds, seed=seed)

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            print(
                f"  Optuna [{label}] trial {trial.number + 1}/{n_trials}: "
                f"state={trial.state.name}",
                flush=True,
            )
            return
        best = study.best_value
        marker = " *" if trial.value <= best + 1e-15 else ""
        print(
            f"  Optuna [{label}] trial {trial.number + 1}/{n_trials}: "
            f"cv_mae={trial.value:.4f}  best={best:.4f}{marker}",
            flush=True,
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # Enqueue a near-baseline configuration so the search always sees the default.
    baseline = _baseline_lgbm_params(len(x_tr), seed)
    study.enqueue_trial(
        {
            "n_estimators": int(baseline["n_estimators"]),
            "learning_rate": float(baseline["learning_rate"]),
            "num_leaves": int(baseline["num_leaves"]),
            "min_child_samples": int(baseline["min_child_samples"]),
            "subsample": float(baseline["subsample"]),
            "colsample_bytree": float(baseline["colsample_bytree"]),
            "reg_alpha": 1e-8,
            "reg_lambda": 1e-8,
            "max_depth": 12,
            "min_split_gain": 0.0,
        }
    )
    print(
        f"  Optuna [{label}]: starting {n_trials} trials "
        f"({n_folds}-fold CV MAE on train, n={len(x_tr)})",
        flush=True,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_log_trial],
    )
    best_params = dict(study.best_params)
    best_params["random_state"] = seed
    best_params["verbosity"] = -1
    return best_params, float(study.best_value)


def _fit_lgbm_holdout(
    meta: pd.DataFrame,
    target: pd.Series,
    *,
    train_ratio: float,
    seed: int,
    title: str,
    xlabel: str,
    output_dir: Path | None,
    filename_prefix: str,
    optuna_trials: int = 50,
    cv_folds: int = 5,
) -> dict[str, Any] | None:
    available = [column for column in FEATURE_COLS if column in meta.columns]
    features = meta.loc[target.index, available].apply(pd.to_numeric, errors="coerce")
    valid = features.notna().all(axis=1) & target.notna()
    features = features.loc[valid]
    y = target.loc[valid]

    nunique = features.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"dropping constant metadata columns: {constant_cols}")
        features = features.drop(columns=constant_cols)

    if len(y) < 4:
        print(f"skip LightGBM ({title}): need at least 4 buildings")
        return None

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_train = max(2, int(train_ratio * len(y)))
    n_train = min(n_train, len(y) - 1)
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    x_tr, x_te = features.iloc[train_idx], features.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    y_te_np = y_te.to_numpy(dtype=np.float64)

    baseline_params = _baseline_lgbm_params(len(x_tr), seed)
    baseline_model = LGBMRegressor(**baseline_params).fit(x_tr, y_tr)
    baseline_pred = baseline_model.predict(x_te)
    baseline_metrics = _regression_metrics(y_te_np, baseline_pred)
    baseline_cv = _cv_mae(
        baseline_params, x_tr, y_tr, n_folds=cv_folds, seed=seed
    )
    print(
        f"LightGBM baseline [{title}]: "
        f"n_train={len(x_tr)} n_test={len(x_te)} "
        f"holdout_mae={baseline_metrics['mae']:.4f} "
        f"holdout_rmse={baseline_metrics['rmse']:.4f} "
        f"holdout_r2={baseline_metrics['r2']:.4f} "
        f"holdout_corr={baseline_metrics['corr']:.4f} "
        f"cv_mae={baseline_cv:.4f}"
    )

    if optuna_trials <= 0:
        model = baseline_model
        pred = baseline_pred
        tuned_metrics = baseline_metrics
        best_params = baseline_params
        best_cv = baseline_cv
        comparison: dict[str, Any] = {
            "target": title,
            "n_train": len(x_tr),
            "n_test": len(x_te),
            "optuna_trials": 0,
            "baseline": {**baseline_metrics, "cv_mae": baseline_cv, "params": baseline_params},
            "optuna": None,
            "improvement": None,
        }
    else:
        best_params, best_cv = _optuna_search(
            x_tr,
            y_tr,
            n_trials=optuna_trials,
            n_folds=cv_folds,
            seed=seed,
            label=filename_prefix,
        )
        model = LGBMRegressor(**best_params).fit(x_tr, y_tr)
        pred = model.predict(x_te)
        tuned_metrics = _regression_metrics(y_te_np, pred)

        def _rel_improvement(base: float, tuned: float, *, higher_is_better: bool) -> float:
            if not np.isfinite(base) or abs(base) < 1e-12:
                return float("nan")
            if higher_is_better:
                return float((tuned - base) / abs(base))
            return float((base - tuned) / abs(base))

        improvement = {
            "holdout_mae": _rel_improvement(
                baseline_metrics["mae"], tuned_metrics["mae"], higher_is_better=False
            ),
            "holdout_rmse": _rel_improvement(
                baseline_metrics["rmse"], tuned_metrics["rmse"], higher_is_better=False
            ),
            "holdout_r2": _rel_improvement(
                baseline_metrics["r2"], tuned_metrics["r2"], higher_is_better=True
            ),
            "holdout_corr": _rel_improvement(
                baseline_metrics["corr"], tuned_metrics["corr"], higher_is_better=True
            ),
            "cv_mae": _rel_improvement(baseline_cv, best_cv, higher_is_better=False),
        }
        print(
            f"LightGBM Optuna [{title}]: "
            f"trials={optuna_trials} folds={cv_folds} "
            f"cv_mae={best_cv:.4f} "
            f"holdout_mae={tuned_metrics['mae']:.4f} "
            f"holdout_rmse={tuned_metrics['rmse']:.4f} "
            f"holdout_r2={tuned_metrics['r2']:.4f} "
            f"holdout_corr={tuned_metrics['corr']:.4f}"
        )
        print(
            f"  vs baseline: "
            f"mae {improvement['holdout_mae']:+.1%} "
            f"rmse {improvement['holdout_rmse']:+.1%} "
            f"r2 {improvement['holdout_r2']:+.1%} "
            f"corr {improvement['holdout_corr']:+.1%} "
            f"cv_mae {improvement['cv_mae']:+.1%} "
            f"(positive = better)"
        )
        print(f"  best_params={ {k: best_params[k] for k in best_params if k not in {'random_state', 'verbosity'}} }")
        comparison = {
            "target": title,
            "n_train": len(x_tr),
            "n_test": len(x_te),
            "optuna_trials": optuna_trials,
            "cv_folds": cv_folds,
            "baseline": {
                **baseline_metrics,
                "cv_mae": baseline_cv,
                "params": baseline_params,
            },
            "optuna": {
                **tuned_metrics,
                "cv_mae": best_cv,
                "params": best_params,
            },
            "improvement_fraction": improvement,
        }

    # Side-by-side hold-out scatter: baseline vs tuned.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, y_hat, subtitle, metrics in (
        (axes[0], baseline_pred, "baseline", baseline_metrics),
        (axes[1], pred, "Optuna" if optuna_trials > 0 else "baseline", tuned_metrics),
    ):
        ax.scatter(y_te_np, y_hat, alpha=0.8)
        lims = [
            float(np.nanmin([y_te_np.min(), y_hat.min()])),
            float(np.nanmax([y_te_np.max(), y_hat.max()])),
        ]
        ax.plot(lims, lims, c="k", ls="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Predicted")
        ax.set_title(
            f"{subtitle}\n"
            f"MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}  corr={metrics['corr']:.3f}"
        )
    fig.suptitle(title)
    _save_or_show(fig, output_dir, f"{filename_prefix}_lightgbm_predictions.png")

    # Turnoff-style single scatter: predicted vs true KPI on the hold-out set.
    _pred_vs_true_scatter(
        y_te_np,
        pred,
        xlabel=xlabel,
        ylabel="Predicted",
        title=title,
        output_dir=output_dir,
        filename=f"{filename_prefix}_pred_vs_true.png",
    )

    if optuna_trials > 0 and comparison.get("improvement_fraction") is not None:
        imp = comparison["improvement_fraction"]
        labels = ["MAE", "RMSE", "R²", "corr", "CV MAE"]
        values = [
            100.0 * imp["holdout_mae"],
            100.0 * imp["holdout_rmse"],
            100.0 * imp["holdout_r2"],
            100.0 * imp["holdout_corr"],
            100.0 * imp["cv_mae"],
        ]
        fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(6, 4))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
        ax.bar(labels, values, color=colors)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylabel("% improvement vs baseline")
        ax.set_title(f"Optuna vs baseline — {title}")
        _save_or_show(fig, output_dir, f"{filename_prefix}_optuna_vs_baseline.png")

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_te)
        plt.close("all")
        shap.summary_plot(shap_values, x_te, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        _save_or_show(fig, output_dir, f"{filename_prefix}_shap_summary.png")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"skip SHAP summary ({title}): {exc}")

    if output_dir is not None:
        out_json = output_dir / f"{filename_prefix}_lgbm_metrics.json"
        out_json.write_text(json.dumps(comparison, indent=2, default=str))
        print(f"wrote {out_json}")
    return comparison


def load_cached_kpis(kpis_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload meta + long KPI table written by a previous run."""
    frame = pd.read_csv(kpis_csv)
    kpi_cols = [
        "egid",
        "horizon_hours",
        "horizon_steps",
        "event_count",
        "valid",
        "beta_plus_w_m2_k",
        "beta_minus_w_m2_k",
        "up_flex_wh_m2_k",
        "down_flex_wh_m2_k",
    ]
    missing = [column for column in kpi_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"{kpis_csv} is missing columns: {missing}")
    kpis = frame.loc[:, kpi_cols].copy()
    if kpis["valid"].dtype == object:
        kpis["valid"] = kpis["valid"].astype(str).str.lower().isin(["true", "1"])
    else:
        kpis["valid"] = kpis["valid"].astype(bool)
    meta_cols = [column for column in frame.columns if column not in kpi_cols or column == "egid"]
    meta = (
        frame.loc[:, meta_cols]
        .drop_duplicates(subset=["egid"])
        .set_index("egid")
        .sort_index()
    )
    return meta, kpis


def run_analysis(
    data_dir: Path,
    *,
    max_buildings: int | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    horizons_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    primary_horizon_hours: float = 1.0,
    controls: ControlsMode = "full",
    min_setpoint_change: float = 0.05,
    min_events: int = 20,
    dt_hours: float = DT_HOURS,
    train_ratio: float = 0.8,
    seed: int = 0,
    optuna_trials: int = 50,
    cv_folds: int = 5,
    kpis_csv: Path | None = None,
) -> None:
    if kpis_csv is not None:
        meta, kpis = load_cached_kpis(kpis_csv)
        print(f"loaded cached KPIs from {kpis_csv}")
    else:
        meta, kpis = collect_kpis(
            data_dir,
            max_buildings=max_buildings,
            horizons_hours=horizons_hours,
            dt_hours=dt_hours,
            min_setpoint_change=min_setpoint_change,
            min_events=min_events,
            controls=controls,
        )
    print(
        f"buildings={len(meta)} kpi_rows={len(kpis)} "
        f"controls={controls} horizons={sorted(kpis['horizon_hours'].unique().tolist())}"
    )

    if output_dir is not None and kpis_csv is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        kpis_path = output_dir / "setpoint_flex_kpis.csv"
        meta_out = meta.reset_index().merge(kpis, on="egid", how="left")
        meta_out.to_csv(kpis_path, index=False)
        print(f"wrote {kpis_path}")

    # ---------------------------------------------------------------------------
    # Metadata diagnostic
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, layout="constrained")
    scatter = ax.scatter(
        meta["annual_consumption"],
        meta["m2"],
        c=meta["U"],
        cmap="viridis",
        alpha=0.8,
    )
    fig.colorbar(scatter, ax=ax, label="surface_to_volume_ratio [1/m]")
    ax.set_xlabel("Annual HP Consumption (kWh)")
    ax.set_ylabel("Heated Area (m²)")
    ax.set_title("Annual HP Consumption vs Heated Area colored by S/V")
    _save_or_show(fig, output_dir, "01_annual_consumption_vs_area.png")

    # ---------------------------------------------------------------------------
    # KPI distributions by horizon
    # ---------------------------------------------------------------------------
    valid = kpis.loc[kpis["valid"]].copy()
    if valid.empty:
        print("no valid event-study KPIs; check min-events / setpoint variation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for horizon, group in valid.groupby("horizon_hours"):
        axes[0].hist(
            group["up_flex_wh_m2_k"],
            bins=40,
            alpha=0.45,
            label=f"H={horizon:g}h",
        )
        axes[1].hist(
            group["down_flex_wh_m2_k"],
            bins=40,
            alpha=0.45,
            label=f"H={horizon:g}h",
        )
    axes[0].set_xlabel(r"up_flex $H\cdot\beta_+$ [Wh/(m²·K)]")
    axes[1].set_xlabel(r"down_flex $H\cdot\beta_-$ [Wh/(m²·K)]")
    axes[0].set_title("Upward setpoint flexibility")
    axes[1].set_title("Downward setpoint flexibility")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    _save_or_show(fig, output_dir, "02_flex_kpi_histograms.png")

    # Primary horizon slice for scatter / LGBM.
    primary = float(
        min(valid["horizon_hours"].unique(), key=lambda h: abs(float(h) - primary_horizon_hours))
    )
    primary_kpis = valid.loc[np.isclose(valid["horizon_hours"], primary)].set_index("egid")
    print(
        f"primary horizon={primary:g}h: "
        f"n_valid={len(primary_kpis)} "
        f"up_flex mean={primary_kpis['up_flex_wh_m2_k'].mean():.4f} "
        f"down_flex mean={primary_kpis['down_flex_wh_m2_k'].mean():.4f}"
    )
    print(
        "up_flex [Wh/(m²·K)]: "
        f"mean={primary_kpis['up_flex_wh_m2_k'].mean():.4f} "
        f"std={primary_kpis['up_flex_wh_m2_k'].std():.4f} "
        f"min={primary_kpis['up_flex_wh_m2_k'].min():.4f} "
        f"max={primary_kpis['up_flex_wh_m2_k'].max():.4f}"
    )
    print(
        "down_flex [Wh/(m²·K)]: "
        f"mean={primary_kpis['down_flex_wh_m2_k'].mean():.4f} "
        f"std={primary_kpis['down_flex_wh_m2_k'].std():.4f} "
        f"min={primary_kpis['down_flex_wh_m2_k'].min():.4f} "
        f"max={primary_kpis['down_flex_wh_m2_k'].max():.4f}"
    )

    # Turnoff-style overlapping KPI distribution at the primary horizon.
    _kpi_distribution_plot(
        primary_kpis,
        horizon_hours=primary,
        output_dir=output_dir,
        filename="02b_primary_flex_kpi_distribution.png",
    )

    joined = meta.join(primary_kpis, how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].scatter(joined["m2"], joined["up_flex_wh_m2_k"], alpha=0.7)
    axes[0].set_xlabel("Heated area (m²)")
    axes[0].set_ylabel(rf"up_flex @ {primary:g}h [Wh/(m²·K)]")
    axes[1].scatter(joined["m2"], joined["down_flex_wh_m2_k"], alpha=0.7)
    axes[1].set_xlabel("Heated area (m²)")
    axes[1].set_ylabel(rf"down_flex @ {primary:g}h [Wh/(m²·K)]")
    fig.suptitle(f"Event-study KPIs vs heated area (controls={controls})")
    _save_or_show(fig, output_dir, "03_flex_vs_area.png")

    # Event counts
    fig, ax = plt.subplots(1, 1, layout="constrained")
    sns.boxplot(
        data=valid,
        x="horizon_hours",
        y="event_count",
        ax=ax,
    )
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Setpoint-jump events used in OLS")
    ax.set_title("Event counts by horizon")
    _save_or_show(fig, output_dir, "04_event_counts.png")

    # LightGBM: predict primary-horizon up/down flex from static metadata.
    comparisons: list[dict[str, Any]] = []
    down_cmp = _fit_lgbm_holdout(
        meta,
        primary_kpis["down_flex_wh_m2_k"],
        train_ratio=train_ratio,
        seed=seed,
        title=rf"LightGBM hold-out: down_flex @ {primary:g}h",
        xlabel=rf"True down_flex [Wh/(m²·K)] @ {primary:g}h",
        output_dir=output_dir,
        filename_prefix="05_down_flex",
        optuna_trials=optuna_trials,
        cv_folds=cv_folds,
    )
    if down_cmp is not None:
        comparisons.append(down_cmp)
    up_cmp = _fit_lgbm_holdout(
        meta,
        primary_kpis["up_flex_wh_m2_k"],
        train_ratio=train_ratio,
        seed=seed + 1,
        title=rf"LightGBM hold-out: up_flex @ {primary:g}h",
        xlabel=rf"True up_flex [Wh/(m²·K)] @ {primary:g}h",
        output_dir=output_dir,
        filename_prefix="06_up_flex",
        optuna_trials=optuna_trials,
        cv_folds=cv_folds,
    )
    if up_cmp is not None:
        comparisons.append(up_cmp)

    if output_dir is not None and comparisons:
        summary_path = output_dir / "lgbm_optuna_summary.json"
        summary_path.write_text(json.dumps(comparisons, indent=2, default=str))
        print(f"wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing egid=*/timeseries.parquet exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG figures and KPI CSV.",
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=None,
        help="Optional cap on the number of egid folders to load.",
    )
    parser.add_argument(
        "--kpis-csv",
        type=Path,
        default=None,
        help="Reuse a previously written setpoint_flex_kpis.csv (skips parquet reload).",
    )
    parser.add_argument(
        "--horizons-hours",
        type=str,
        default="0.5,1,2,3",
        help="Comma-separated activation durations H in hours.",
    )
    parser.add_argument(
        "--primary-horizon-hours",
        type=float,
        default=1.0,
        help="Horizon used for LightGBM targets and primary scatter plots.",
    )
    parser.add_argument(
        "--controls",
        choices=("none", "weather", "full"),
        default="full",
        help="Event-study control set (same semantics as flexibility_event_study_kpis).",
    )
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--dt-hours", type=float, default=DT_HOURS)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Optuna trials for LightGBM HPO on the train split (0 disables search).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="K-fold CV folds used as the Optuna objective on the train split.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only show figures; do not write PNGs/CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons_hours, args.dt_hours)
    output_dir = None if args.no_save else args.output_dir
    run_analysis(
        args.data_dir,
        max_buildings=args.max_buildings,
        output_dir=output_dir,
        horizons_hours=horizons,
        primary_horizon_hours=args.primary_horizon_hours,
        controls=args.controls,
        min_setpoint_change=args.min_setpoint_change,
        min_events=args.min_events,
        dt_hours=args.dt_hours,
        train_ratio=args.train_ratio,
        seed=args.seed,
        optuna_trials=args.optuna_trials,
        cv_folds=args.cv_folds,
        kpis_csv=args.kpis_csv,
    )


if __name__ == "__main__":
    main()
