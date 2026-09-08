"""Estimate setpoint-shock flexibility KPIs from saved closed-loop emulator runs.

For every thermostat setpoint discontinuity at event time t, and for every
activation duration H, this script computes

    delta_p_H = mean(P[t : t + H]) - mean(P[t - H : t])

By default, it reports the directional ratio-of-sums energy KPI

    F_H^+ = H * sum(delta_p_H | delta_tset > 0)
                  / sum(delta_tset | delta_tset > 0)

    F_H^- = H * sum(delta_p_H | delta_tset < 0)
                  / sum(delta_tset | delta_tset < 0).

The previous windowed regression estimator remains available with
``--kpi-estimator regression``. It regresses the event response on the setpoint
jump:

    delta_p_H = alpha_H
              + beta_plus_H  * max(delta_tset, 0)
              + beta_minus_H * min(delta_tset, 0)
              + controls
              + error.

The reported energy-flexibility gains are

    up_flex_H   = H * beta_plus_H
    down_flex_H = H * beta_minus_H

in Wh/(m2 K). With the `min(delta_tset, 0)` convention, a positive
down_flex_H means a -1 K setpoint change reduces electric energy by that many
Wh/m2 over duration H.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    PROFILE_ID_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    closed_loop_required_columns,
)
from .data import (
    ClosedLoopProfile,
    DEFAULT_DATASET_PATH,
    read_dataset_frame,
    to_closed_loop_profiles,
)
from .model_io import load_training_artifact
from .models import (
    ClosedLoopHPEmulator,
    ContractingClosedLoopHPEmulator,
    ProbabilisticClosedLoopHPEmulator,
    ProbabilisticContractingClosedLoopHPEmulator,
)
from .train import (
    predict_closed_loop_full_profile,
    predict_probabilistic_closed_loop_full_profile,
    sample_probabilistic_closed_loop_full_profile_scenarios,
)

ProfileSource = Literal["test", "train", "selected"]
EmulationKpiMode = Literal["mean", "scenario_average"]
ControlsMode = Literal["none", "weather", "full"]
KpiEstimator = Literal["direct_ratio", "regression"]
ProbHpScenarioMode = Literal["expected", "bernoulli"]
ProbHpEmissionOverride = Literal["artifact", "bounded", "legacy_lognormal_mean"]


@dataclass(frozen=True)
class EventStudyConfig:
    horizons_hours: tuple[float, ...]
    horizon_steps: tuple[int, ...]
    dt_hours: float
    min_setpoint_change: float
    min_events: int
    controls: ControlsMode
    estimator: KpiEstimator = "regression"


@dataclass(frozen=True)
class RegressionData:
    design: np.ndarray
    response: np.ndarray
    events: np.ndarray
    deltas: np.ndarray


@dataclass(frozen=True)
class FlexibilityKpi:
    profile_id: int
    horizon_hours: float
    horizon_steps: int
    event_count: int
    valid: bool
    beta_plus_w_m2_k: float
    beta_minus_w_m2_k: float
    up_flex_wh_m2_k: float
    down_flex_wh_m2_k: float


class OLSAccumulator:
    """Collect pooled sufficient statistics for either KPI estimator."""

    def __init__(self) -> None:
        self.xtx: np.ndarray | None = None
        self.xty: np.ndarray | None = None
        self.event_count = 0
        self.up_response_sum = 0.0
        self.up_delta_sum = 0.0
        self.down_response_sum = 0.0
        self.down_delta_sum = 0.0

    def add(self, regression_data: RegressionData) -> None:
        design = regression_data.design
        response = regression_data.response
        if len(response) == 0:
            return
        xtx = design.T @ design
        xty = design.T @ response
        if self.xtx is None:
            self.xtx = xtx
            self.xty = xty
        else:
            self.xtx = self.xtx + xtx
            self.xty = self.xty + xty
        up = regression_data.deltas > 0.0
        down = regression_data.deltas < 0.0
        self.up_response_sum += float(np.sum(response[up]))
        self.up_delta_sum += float(np.sum(regression_data.deltas[up]))
        self.down_response_sum += float(np.sum(response[down]))
        self.down_delta_sum += float(np.sum(regression_data.deltas[down]))
        self.event_count += int(len(response))

    def estimate(
        self,
        *,
        profile_id: int,
        horizon_hours: float,
        horizon_steps: int,
        min_events: int,
        estimator: KpiEstimator = "regression",
    ) -> FlexibilityKpi:
        if estimator == "direct_ratio":
            return kpi_from_directional_sums(
                profile_id=profile_id,
                horizon_hours=horizon_hours,
                horizon_steps=horizon_steps,
                event_count=self.event_count,
                min_events=min_events,
                up_response_sum=self.up_response_sum,
                up_delta_sum=self.up_delta_sum,
                down_response_sum=self.down_response_sum,
                down_delta_sum=self.down_delta_sum,
            )
        if self.xtx is None or self.xty is None:
            return invalid_kpi(profile_id, horizon_hours, horizon_steps, event_count=0)
        if self.event_count < min_events or self.event_count <= self.xtx.shape[0]:
            return invalid_kpi(profile_id, horizon_hours, horizon_steps, event_count=self.event_count)
        coefficients = np.linalg.pinv(self.xtx) @ self.xty
        return kpi_from_coefficients(
            profile_id=profile_id,
            horizon_hours=horizon_hours,
            horizon_steps=horizon_steps,
            event_count=self.event_count,
            coefficients=coefficients,
        )


def invalid_kpi(
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    *,
    event_count: int,
) -> FlexibilityKpi:
    return FlexibilityKpi(
        profile_id=profile_id,
        horizon_hours=float(horizon_hours),
        horizon_steps=int(horizon_steps),
        event_count=int(event_count),
        valid=False,
        beta_plus_w_m2_k=float("nan"),
        beta_minus_w_m2_k=float("nan"),
        up_flex_wh_m2_k=float("nan"),
        down_flex_wh_m2_k=float("nan"),
    )


def kpi_from_coefficients(
    *,
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    event_count: int,
    coefficients: np.ndarray,
) -> FlexibilityKpi:
    beta_plus = float(coefficients[1])
    beta_minus = float(coefficients[2])
    return FlexibilityKpi(
        profile_id=int(profile_id),
        horizon_hours=float(horizon_hours),
        horizon_steps=int(horizon_steps),
        event_count=int(event_count),
        valid=True,
        beta_plus_w_m2_k=beta_plus,
        beta_minus_w_m2_k=beta_minus,
        up_flex_wh_m2_k=float(horizon_hours * beta_plus),
        down_flex_wh_m2_k=float(horizon_hours * beta_minus),
    )


def kpi_from_directional_sums(
    *,
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    event_count: int,
    min_events: int,
    up_response_sum: float,
    up_delta_sum: float,
    down_response_sum: float,
    down_delta_sum: float,
) -> FlexibilityKpi:
    if (
        event_count < min_events
        or not np.isfinite(up_response_sum)
        or not np.isfinite(down_response_sum)
        or not np.isfinite(up_delta_sum)
        or not np.isfinite(down_delta_sum)
        or up_delta_sum <= 1e-12
        or down_delta_sum >= -1e-12
    ):
        return invalid_kpi(
            profile_id,
            horizon_hours,
            horizon_steps,
            event_count=event_count,
        )
    beta_plus = float(up_response_sum / up_delta_sum)
    beta_minus = float(down_response_sum / down_delta_sum)
    return FlexibilityKpi(
        profile_id=int(profile_id),
        horizon_hours=float(horizon_hours),
        horizon_steps=int(horizon_steps),
        event_count=int(event_count),
        valid=True,
        beta_plus_w_m2_k=beta_plus,
        beta_minus_w_m2_k=beta_minus,
        up_flex_wh_m2_k=float(horizon_hours * beta_plus),
        down_flex_wh_m2_k=float(horizon_hours * beta_minus),
    )


def estimate_direct_kpi_from_regression_data(
    *,
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    data: RegressionData,
    min_events: int,
) -> FlexibilityKpi:
    up = data.deltas > 0.0
    down = data.deltas < 0.0
    return kpi_from_directional_sums(
        profile_id=profile_id,
        horizon_hours=horizon_hours,
        horizon_steps=horizon_steps,
        event_count=int(len(data.response)),
        min_events=min_events,
        up_response_sum=float(np.sum(data.response[up])),
        up_delta_sum=float(np.sum(data.deltas[up])),
        down_response_sum=float(np.sum(data.response[down])),
        down_delta_sum=float(np.sum(data.deltas[down])),
    )


def estimate_kpi_from_regression_data(
    *,
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    data: RegressionData,
    min_events: int,
) -> FlexibilityKpi:
    event_count = int(len(data.response))
    if event_count < min_events or event_count <= data.design.shape[1]:
        return invalid_kpi(
            profile_id,
            horizon_hours,
            horizon_steps,
            event_count=event_count,
        )
    coefficients, *_ = np.linalg.lstsq(data.design, data.response, rcond=None)
    return kpi_from_coefficients(
        profile_id=profile_id,
        horizon_hours=horizon_hours,
        horizon_steps=horizon_steps,
        event_count=event_count,
        coefficients=coefficients,
    )


def _read_parquet(path: Path, *, columns: Iterable[str], profile_ids: list[int]) -> pd.DataFrame:
    return read_dataset_frame(
        path,
        columns=list(columns),
        profile_ids=profile_ids,
    )


def _config_value(metadata: dict[str, Any], name: str, default: Any) -> Any:
    return metadata.get("train_config", {}).get(name, default)


def _saved_profile_ids(metadata: dict[str, Any], source: ProfileSource) -> list[int]:
    key = f"{source}_ids"
    ids = [int(value) for value in metadata.get(key, [])]
    if not ids:
        raise ValueError(f"The artifact metadata does not contain saved {key}.")
    return ids


def _select_profiles(profile_ids: list[int], *, max_profiles: int | None, seed: int) -> list[int]:
    if max_profiles is None or max_profiles >= len(profile_ids):
        return list(profile_ids)
    if max_profiles < 1:
        raise ValueError("--max-profiles must be positive or omitted")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(profile_ids), size=max_profiles, replace=False))
    return [profile_ids[int(index)] for index in indices]


def _parse_horizons_hours(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--horizons-hours must contain at least one positive value")
    if any(value <= 0.0 for value in values):
        raise ValueError("--horizons-hours values must be positive")
    return tuple(sorted(set(values)))


def _parse_quantiles(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--coverage-quantiles must contain at least one value")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("--coverage-quantiles values must be strictly between 0 and 1")
    return tuple(sorted(set(values)))


def _resolve_horizons(raw: str, dt_hours: float) -> tuple[tuple[float, ...], tuple[int, ...]]:
    horizons = _parse_horizons_hours(raw)
    steps = tuple(int(round(horizon / dt_hours)) for horizon in horizons)
    if any(step < 1 for step in steps):
        raise ValueError("Every horizon must contain at least one timestep")
    effective = tuple(float(step * dt_hours) for step in steps)
    unique: dict[int, float] = {}
    for step, horizon in zip(steps, effective):
        unique.setdefault(step, horizon)
    return tuple(unique.values()), tuple(unique.keys())


def _current_temperature_from_trace(trace: np.ndarray, initial_temperature: np.ndarray) -> np.ndarray:
    initial = float(np.asarray(initial_temperature).reshape(-1)[0])
    return np.concatenate([[initial], trace[:-1, 0].astype(np.float64)])


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


def _setpoint_events(setpoint: np.ndarray, *, min_setpoint_change: float) -> tuple[np.ndarray, np.ndarray]:
    setpoint = setpoint.astype(np.float64)
    deltas = setpoint[1:] - setpoint[:-1]
    events = np.flatnonzero(np.abs(deltas) >= min_setpoint_change) + 1
    event_deltas = setpoint[events] - setpoint[events - 1]
    return events.astype(np.int64), event_deltas.astype(np.float64)


def _previous_event_deltas(all_events: np.ndarray, all_deltas: np.ndarray, events: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return np.empty(0, dtype=np.float64)
    previous_by_event: dict[int, float] = {}
    previous = 0.0
    for event, delta in zip(all_events, all_deltas):
        previous_by_event[int(event)] = float(previous)
        previous = float(delta)
    return np.asarray([previous_by_event.get(int(event), 0.0) for event in events], dtype=np.float64)


def _window_mean(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    widths = ends - starts
    if len(starts) and np.all(widths == widths[0]) and widths[0] > 0:
        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape=int(widths[0]),
            axis=0,
        )
        return np.asarray(np.mean(windows[starts], axis=-1), dtype=np.float64)
    return np.asarray(
        [np.mean(values[int(start) : int(end)], axis=0) for start, end in zip(starts, ends)],
        dtype=np.float64,
    )


def regression_data_for_trace(
    profile: ClosedLoopProfile,
    trace: np.ndarray,
    *,
    horizon_steps: int,
    config: EventStudyConfig,
) -> RegressionData:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] < 3:
        raise ValueError("closed-loop traces must have shape [steps, 3] = [Tin, Qroom, Pel_SH]")
    if trace.shape[0] != profile.inputs.shape[0]:
        raise ValueError(
            f"Profile {profile.profile_id}: trace length {trace.shape[0]} does not match "
            f"input length {profile.inputs.shape[0]}"
        )

    setpoint = profile.inputs[:, 0].astype(np.float64)
    all_events, all_deltas = _setpoint_events(
        setpoint,
        min_setpoint_change=config.min_setpoint_change,
    )
    events = all_events[
        (all_events >= horizon_steps) & (all_events + horizon_steps <= len(setpoint))
    ]
    deltas = setpoint[events] - setpoint[events - 1]

    if len(events) == 0:
        return RegressionData(
            design=np.empty((0, 3), dtype=np.float64),
            response=np.empty(0, dtype=np.float64),
            events=events,
            deltas=deltas,
        )

    power = trace[:, 2].astype(np.float64)
    pre_starts = events - horizon_steps
    pre_ends = events
    post_starts = events
    post_ends = events + horizon_steps
    pre_power = _window_mean(power, pre_starts, pre_ends)
    post_power = _window_mean(power, post_starts, post_ends)
    response = post_power - pre_power

    dset_plus = np.maximum(deltas, 0.0)
    dset_minus = np.minimum(deltas, 0.0)
    columns = [
        np.ones_like(deltas, dtype=np.float64),
        dset_plus,
        dset_minus,
    ]

    if config.estimator == "regression" and config.controls != "none":
        disturbances_calendar = profile.inputs[events, 1:].astype(np.float64)
        disturbance_values = profile.inputs[:, 1:4].astype(np.float64)
        pre_disturbance = _window_mean(disturbance_values, pre_starts, pre_ends)
        post_disturbance = _window_mean(disturbance_values, post_starts, post_ends)
        disturbance_delta = post_disturbance - pre_disturbance
        controls = [disturbances_calendar, disturbance_delta]
        if config.controls == "full":
            current_temperature = _current_temperature_from_trace(trace, profile.initial_temperature)
            pre_temperature = current_temperature[events]
            pre_setpoint = setpoint[events - 1]
            previous_delta = _previous_event_deltas(all_events, all_deltas, events)
            controls.insert(
                0,
                np.column_stack(
                    [
                        pre_power,
                        pre_temperature,
                        pre_setpoint,
                        previous_delta,
                    ]
                ),
            )
        columns.append(_standardized_controls(np.column_stack(controls)))

    design = np.column_stack(columns).astype(np.float64)
    return RegressionData(
        design=design,
        response=response.astype(np.float64),
        events=events.astype(np.int64),
        deltas=deltas.astype(np.float64),
    )


def estimate_trace_flexibility(
    profile: ClosedLoopProfile,
    trace: np.ndarray,
    config: EventStudyConfig,
) -> list[FlexibilityKpi]:
    """Estimate windowed upward/downward flexibility for one profile trace."""
    kpis: list[FlexibilityKpi] = []
    for horizon_hours, horizon_steps in zip(config.horizons_hours, config.horizon_steps):
        data = regression_data_for_trace(
            profile,
            trace,
            horizon_steps=horizon_steps,
            config=config,
        )
        estimate = (
            estimate_direct_kpi_from_regression_data
            if config.estimator == "direct_ratio"
            else estimate_kpi_from_regression_data
        )
        kpis.append(
            estimate(
                profile_id=profile.profile_id,
                horizon_hours=horizon_hours,
                horizon_steps=horizon_steps,
                data=data,
                min_events=config.min_events,
            )
        )
    return kpis


def _mean_kpis(
    kpis_by_scenario: list[list[FlexibilityKpi]],
    *,
    profile_id: int,
    config: EventStudyConfig,
) -> list[FlexibilityKpi]:
    by_horizon: list[FlexibilityKpi] = []
    for horizon_index, (horizon_hours, horizon_steps) in enumerate(
        zip(config.horizons_hours, config.horizon_steps)
    ):
        valid = [items[horizon_index] for items in kpis_by_scenario if items[horizon_index].valid]
        if not valid:
            by_horizon.append(
                invalid_kpi(profile_id, horizon_hours, horizon_steps, event_count=0)
            )
            continue
        beta_plus = float(np.nanmean([kpi.beta_plus_w_m2_k for kpi in valid]))
        beta_minus = float(np.nanmean([kpi.beta_minus_w_m2_k for kpi in valid]))
        by_horizon.append(
            FlexibilityKpi(
                profile_id=profile_id,
                horizon_hours=horizon_hours,
                horizon_steps=horizon_steps,
                event_count=int(np.nanmedian([kpi.event_count for kpi in valid])),
                valid=True,
                beta_plus_w_m2_k=beta_plus,
                beta_minus_w_m2_k=beta_minus,
                up_flex_wh_m2_k=float(horizon_hours * beta_plus),
                down_flex_wh_m2_k=float(horizon_hours * beta_minus),
            )
        )
    return by_horizon


def _kpi_to_row(prefix: str, kpi: FlexibilityKpi) -> dict[str, float | int | bool]:
    return {
        f"{prefix}_event_count": int(kpi.event_count),
        f"{prefix}_valid": bool(kpi.valid),
        f"{prefix}_beta_plus_w_m2_k": float(kpi.beta_plus_w_m2_k),
        f"{prefix}_beta_minus_w_m2_k": float(kpi.beta_minus_w_m2_k),
        f"{prefix}_up_flex_wh_m2_k": float(kpi.up_flex_wh_m2_k),
        f"{prefix}_down_flex_wh_m2_k": float(kpi.down_flex_wh_m2_k),
    }


def _load_closed_loop_profiles(
    dataset_path: Path,
    profile_ids: list[int],
    *,
    include_space_heating_availability: bool,
    include_internal_gains: bool,
    hp_power_area_normalization: str = "zone_floor_area",
    metadata_columns: Sequence[str] | None = None,
) -> list[ClosedLoopProfile]:
    df = _read_parquet(
        dataset_path,
        columns=closed_loop_required_columns(
            None if metadata_columns is None else tuple(metadata_columns),
            include_internal_gains=include_internal_gains,
        ),
        profile_ids=profile_ids,
    )
    profiles = to_closed_loop_profiles(
        df,
        include_space_heating_availability=include_space_heating_availability,
        include_internal_gains=include_internal_gains,
        hp_power_area_normalization=hp_power_area_normalization,  # type: ignore[arg-type]
        metadata_columns=metadata_columns,
    )
    profile_order = {profile_id: index for index, profile_id in enumerate(profile_ids)}
    profiles = [profile for profile in profiles if profile.profile_id in profile_order]
    profiles.sort(key=lambda profile: profile_order[profile.profile_id])
    if not profiles:
        raise ValueError("No requested profiles were found in the dataset.")
    missing = sorted(set(profile_ids).difference(profile.profile_id for profile in profiles))
    if missing:
        print(f"warning=missing_profiles count={len(missing)} ids={missing[:10]}")
    return profiles


def _predict_mean_trace(
    artifact: Any,
    profile: ClosedLoopProfile,
    *,
    key: jax.Array,
    num_particles: int,
) -> np.ndarray:
    model = artifact.model
    model_kind = str(artifact.metadata["model_kind"])
    if model_kind in ("closed_loop_hp", "closed_loop_hp_contracting"):
        if not isinstance(model, (ClosedLoopHPEmulator, ContractingClosedLoopHPEmulator)):
            raise TypeError(f"Loaded model has unexpected type {type(model)!r}")
        return predict_closed_loop_full_profile(model, profile, artifact.scalers)
    if model_kind in ("closed_loop_hp_probabilistic", "closed_loop_hp_contracting_probabilistic"):
        if not isinstance(model, (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator)):
            raise TypeError(f"Loaded model has unexpected type {type(model)!r}")
        return predict_probabilistic_closed_loop_full_profile(
            model,
            profile,
            artifact.scalers,
            key=key,
            num_particles=num_particles,
        )
    raise ValueError(f"Unsupported model_kind for flexibility KPIs: {model_kind!r}")


def _sample_scenario_traces(
    artifact: Any,
    profile: ClosedLoopProfile,
    *,
    key: jax.Array,
    num_scenarios: int,
    hp_scenario_mode: ProbHpScenarioMode,
) -> np.ndarray:
    model = artifact.model
    model_kind = str(artifact.metadata["model_kind"])
    if model_kind not in ("closed_loop_hp_probabilistic", "closed_loop_hp_contracting_probabilistic"):
        raise ValueError("--emulation-kpi-mode scenario_average requires a probabilistic closed-loop model.")
    if not isinstance(model, (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator)):
        raise TypeError(f"Loaded model has unexpected type {type(model)!r}")
    return sample_probabilistic_closed_loop_full_profile_scenarios(
        model,
        profile,
        artifact.scalers,
        key=key,
        num_particles=num_scenarios,
        hp_scenario_mode=hp_scenario_mode,
    )


def _empty_pooled_accumulators(config: EventStudyConfig) -> dict[float, OLSAccumulator]:
    return {horizon_hours: OLSAccumulator() for horizon_hours in config.horizons_hours}


def _add_trace_to_pooled(
    accumulators: dict[float, OLSAccumulator],
    profile: ClosedLoopProfile,
    trace: np.ndarray,
    config: EventStudyConfig,
) -> None:
    for horizon_hours, horizon_steps in zip(config.horizons_hours, config.horizon_steps):
        data = regression_data_for_trace(
            profile,
            trace,
            horizon_steps=horizon_steps,
            config=config,
        )
        accumulators[horizon_hours].add(data)


def _estimate_pooled_rows(
    accumulators: dict[float, OLSAccumulator],
    *,
    signal: str,
    config: EventStudyConfig,
    scenario_index: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_hours, horizon_steps in zip(config.horizons_hours, config.horizon_steps):
        kpi = accumulators[horizon_hours].estimate(
            profile_id=-1,
            horizon_hours=horizon_hours,
            horizon_steps=horizon_steps,
            min_events=config.min_events,
            estimator=config.estimator,
        )
        rows.append(
            {
                "signal": signal,
                "scenario_index": scenario_index,
                "horizon_hours": horizon_hours,
                "horizon_steps": horizon_steps,
                "event_count": kpi.event_count,
                "valid": bool(kpi.valid),
                "beta_plus_w_m2_k": kpi.beta_plus_w_m2_k,
                "beta_minus_w_m2_k": kpi.beta_minus_w_m2_k,
                "up_flex_wh_m2_k": kpi.up_flex_wh_m2_k,
                "down_flex_wh_m2_k": kpi.down_flex_wh_m2_k,
            }
        )
    return rows


def _scenario_average_pooled_rows(
    scenario_rows: list[dict[str, Any]],
    config: EventStudyConfig,
) -> list[dict[str, Any]]:
    if not scenario_rows:
        return []
    df = pd.DataFrame(scenario_rows)
    valid = df[df["valid"]].copy()
    rows: list[dict[str, Any]] = []
    for horizon_hours, horizon_steps in zip(config.horizons_hours, config.horizon_steps):
        part = valid[valid["horizon_hours"] == horizon_hours]
        if part.empty:
            rows.append(
                {
                    "signal": "emulation",
                    "scenario_index": None,
                    "horizon_hours": horizon_hours,
                    "horizon_steps": horizon_steps,
                    "event_count": 0,
                    "valid": False,
                    "beta_plus_w_m2_k": float("nan"),
                    "beta_minus_w_m2_k": float("nan"),
                    "up_flex_wh_m2_k": float("nan"),
                    "down_flex_wh_m2_k": float("nan"),
                    "beta_plus_std_w_m2_k": float("nan"),
                    "beta_minus_std_w_m2_k": float("nan"),
                    "up_flex_std_wh_m2_k": float("nan"),
                    "down_flex_std_wh_m2_k": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "signal": "emulation",
                "scenario_index": None,
                "horizon_hours": horizon_hours,
                "horizon_steps": horizon_steps,
                "event_count": int(part["event_count"].median()),
                "valid": True,
                "beta_plus_w_m2_k": float(part["beta_plus_w_m2_k"].mean()),
                "beta_minus_w_m2_k": float(part["beta_minus_w_m2_k"].mean()),
                "up_flex_wh_m2_k": float(part["up_flex_wh_m2_k"].mean()),
                "down_flex_wh_m2_k": float(part["down_flex_wh_m2_k"].mean()),
                "beta_plus_std_w_m2_k": float(part["beta_plus_w_m2_k"].std(ddof=0)),
                "beta_minus_std_w_m2_k": float(part["beta_minus_w_m2_k"].std(ddof=0)),
                "up_flex_std_wh_m2_k": float(part["up_flex_wh_m2_k"].std(ddof=0)),
                "down_flex_std_wh_m2_k": float(part["down_flex_wh_m2_k"].std(ddof=0)),
            }
        )
    return rows


def _signal_summary_by_horizon(profile_df: pd.DataFrame, metric: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for horizon, part in profile_df.groupby("horizon_hours", sort=True):
        simulated = part[f"sim_{metric}"].to_numpy(dtype=float)
        emulated = part[f"emu_{metric}"].to_numpy(dtype=float)
        valid = part["valid"].to_numpy(dtype=bool) & np.isfinite(simulated) & np.isfinite(emulated)
        if int(np.sum(valid)) == 0:
            rows.append(
                {
                    "horizon_hours": float(horizon),
                    "valid_profiles": 0.0,
                    "simulated_mean": float("nan"),
                    "emulated_mean": float("nan"),
                    "bias_emulated_minus_simulated": float("nan"),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "pearson_corr": float("nan"),
                }
            )
            continue
        sim = simulated[valid].astype(np.float64)
        emu = emulated[valid].astype(np.float64)
        diff = emu - sim
        if len(sim) > 1 and np.std(sim) > 1e-12 and np.std(emu) > 1e-12:
            corr = float(np.corrcoef(sim, emu)[0, 1])
        else:
            corr = float("nan")
        rows.append(
            {
                "horizon_hours": float(horizon),
                "valid_profiles": float(len(sim)),
                "simulated_mean": float(np.mean(sim)),
                "emulated_mean": float(np.mean(emu)),
                "bias_emulated_minus_simulated": float(np.mean(diff)),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "pearson_corr": corr,
            }
        )
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_html(
    path: Path,
    profile_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    *,
    title: str,
) -> None:
    valid = profile_df[profile_df["valid"]].copy()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Upward energy KPI: emulated vs simulated",
            "Downward energy KPI: emulated vs simulated",
            "Profile-level energy KPI error",
            "Pooled duration curve",
        ],
    )

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]
    horizons = sorted(valid["horizon_hours"].unique())
    for horizon_index, horizon in enumerate(horizons):
        part = valid[valid["horizon_hours"] == horizon]
        color = palette[horizon_index % len(palette)]
        for direction, row, col in [("up", 1, 1), ("down", 1, 2)]:
            column = f"{direction}_flex_wh_m2_k"
            fig.add_trace(
                go.Scatter(
                    x=part[f"sim_{column}"],
                    y=part[f"emu_{column}"],
                    mode="markers",
                    name=f"H={horizon:g}h",
                    legendgroup=f"horizon_{horizon}",
                    marker=dict(size=8, opacity=0.7, color=color),
                    text=part["profile_id"].astype(str),
                    showlegend=(direction == "up"),
                ),
                row=row,
                col=col,
            )

    for direction, row, col in [("up", 1, 1), ("down", 1, 2)]:
        sim_values = valid[f"sim_{direction}_flex_wh_m2_k"].to_numpy(dtype=float)
        emu_values = valid[f"emu_{direction}_flex_wh_m2_k"].to_numpy(dtype=float)
        finite = np.concatenate([sim_values, emu_values])
        finite = finite[np.isfinite(finite)]
        if len(finite):
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    name="1:1",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    for direction, color in [("up", "#1f77b4"), ("down", "#d62728")]:
        fig.add_trace(
            go.Box(
                x=valid["horizon_hours"],
                y=valid[f"diff_{direction}_flex_wh_m2_k"],
                name=f"{direction} error",
                marker_color=color,
                boxmean=True,
            ),
            row=2,
            col=1,
        )

    pooled_valid = pooled_df[pooled_df["valid"]].copy()
    for signal, dash in [("simulation", "solid"), ("emulation", "dash")]:
        part = pooled_valid[pooled_valid["signal"] == signal]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["horizon_hours"],
                y=part["up_flex_wh_m2_k"],
                mode="lines+markers",
                name=f"{signal} up",
                line=dict(color="#1f77b4", dash=dash),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=part["horizon_hours"],
                y=part["down_flex_wh_m2_k"],
                mode="lines+markers",
                name=f"{signal} down",
                line=dict(color="#d62728", dash=dash),
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="simulated Wh/(m2 K)", row=1, col=1)
    fig.update_yaxes(title_text="emulated Wh/(m2 K)", row=1, col=1)
    fig.update_xaxes(title_text="simulated Wh/(m2 K)", row=1, col=2)
    fig.update_yaxes(title_text="emulated Wh/(m2 K)", row=1, col=2)
    fig.update_xaxes(title_text="duration H [h]", row=2, col=1)
    fig.update_yaxes(title_text="emulated - simulated Wh/(m2 K)", row=2, col=1)
    fig.update_xaxes(title_text="duration H [h]", row=2, col=2)
    fig.update_yaxes(title_text="pooled Wh/(m2 K)", row=2, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=940,
        width=1450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def _coverage_rows(
    profile_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    quantiles: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if profile_df.empty or scenario_df.empty:
        return rows

    valid_profile_df = profile_df[profile_df["valid"]].copy()
    valid_scenario_df = scenario_df[scenario_df["valid"]].copy()
    for horizon, profile_part in valid_profile_df.groupby("horizon_hours", sort=True):
        scenario_part = valid_scenario_df[valid_scenario_df["horizon_hours"] == horizon]
        if scenario_part.empty:
            continue
        for direction, metric in [
            ("up", "up_flex_wh_m2_k"),
            ("down", "down_flex_wh_m2_k"),
        ]:
            truth_by_profile = {
                int(row.profile_id): float(getattr(row, f"sim_{metric}"))
                for row in profile_part.itertuples(index=False)
                if np.isfinite(float(getattr(row, f"sim_{metric}")))
            }
            profiles = sorted(set(truth_by_profile).intersection(scenario_part["profile_id"].astype(int)))
            if not profiles:
                continue

            samples_by_profile: dict[int, np.ndarray] = {}
            pit_values: list[float] = []
            for profile_id in profiles:
                values = scenario_part.loc[
                    scenario_part["profile_id"].astype(int) == profile_id,
                    metric,
                ].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                samples_by_profile[profile_id] = values
                pit_values.append(float(np.mean(values <= truth_by_profile[profile_id])))

            if not samples_by_profile:
                continue

            for quantile in quantiles:
                hits = []
                for profile_id, samples in samples_by_profile.items():
                    predicted_quantile = float(np.quantile(samples, quantile))
                    hits.append(float(truth_by_profile[profile_id] <= predicted_quantile))
                empirical_coverage = float(np.mean(hits))
                rows.append(
                    {
                        "horizon_hours": float(horizon),
                        "direction": direction,
                        "metric": metric,
                        "quantile": float(quantile),
                        "empirical_coverage": empirical_coverage,
                        "calibration_error": empirical_coverage - float(quantile),
                        "abs_calibration_error": abs(empirical_coverage - float(quantile)),
                        "profile_count": int(len(hits)),
                        "mean_pit": float(np.mean(pit_values)),
                        "median_pit": float(np.median(pit_values)),
                        "min_scenarios_per_profile": int(
                            min(len(values) for values in samples_by_profile.values())
                        ),
                        "max_scenarios_per_profile": int(
                            max(len(values) for values in samples_by_profile.values())
                        ),
                    }
                )
    return rows


def _coverage_summary(coverage_df: pd.DataFrame) -> list[dict[str, Any]]:
    if coverage_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (horizon, direction), part in coverage_df.groupby(["horizon_hours", "direction"], sort=True):
        rows.append(
            {
                "horizon_hours": float(horizon),
                "direction": str(direction),
                "mean_abs_calibration_error": float(part["abs_calibration_error"].mean()),
                "max_abs_calibration_error": float(part["abs_calibration_error"].max()),
                "mean_pit": float(part["mean_pit"].mean()),
                "profile_count": int(part["profile_count"].max()),
                "min_scenarios_per_profile": int(part["min_scenarios_per_profile"].min()),
            }
        )
    return rows


def _write_coverage_html(path: Path, coverage_df: pd.DataFrame, *, title: str) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Upward KPI coverage",
            "Downward KPI coverage",
        ],
    )
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]
    for col, direction in [(1, "up"), (2, "down")]:
        part = coverage_df[coverage_df["direction"] == direction]
        horizons = sorted(part["horizon_hours"].unique())
        for horizon_index, horizon in enumerate(horizons):
            horizon_part = part[part["horizon_hours"] == horizon]
            fig.add_trace(
                go.Scatter(
                    x=horizon_part["quantile"],
                    y=horizon_part["empirical_coverage"],
                    mode="lines+markers",
                    name=f"{direction} H={horizon:g}h",
                    line=dict(color=palette[horizon_index % len(palette)]),
                ),
                row=1,
                col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="ideal",
                line=dict(color="black", dash="dash"),
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

    fig.update_xaxes(title_text="nominal quantile", range=[0.0, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="nominal quantile", range=[0.0, 1.0], row=1, col=2)
    fig.update_yaxes(title_text="empirical P(true KPI <= predicted quantile)", range=[0.0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="empirical coverage", range=[0.0, 1.0], row=1, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=650,
        width=1350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def run_analysis(args: argparse.Namespace) -> None:
    if args.min_events < 1:
        raise ValueError("--min-events must be positive")
    if args.min_setpoint_change < 0.0:
        raise ValueError("--min-setpoint-change must be non-negative")
    if args.emulation_kpi_mode == "scenario_average" and args.num_scenarios < 1:
        raise ValueError("--num-scenarios must be positive with scenario_average mode")
    coverage_quantiles = _parse_quantiles(args.coverage_quantiles)

    train_config_overrides: dict[str, Any] | None = None
    if args.prob_hp_emission_mode != "artifact":
        train_config_overrides = {"prob_hp_emission_mode": args.prob_hp_emission_mode}
    artifact = load_training_artifact(
        args.artifact_dir,
        train_config_overrides=train_config_overrides,
    )
    metadata = artifact.metadata
    model_kind = str(metadata["model_kind"])
    if model_kind not in (
        "closed_loop_hp",
        "closed_loop_hp_contracting",
        "closed_loop_hp_probabilistic",
        "closed_loop_hp_contracting_probabilistic",
    ):
        raise ValueError(f"Unsupported model_kind {model_kind!r}; expected a closed-loop HP artifact.")

    dataset_path = args.dataset or Path(_config_value(metadata, "dataset_path", DEFAULT_DATASET_PATH))
    output_dir = args.output_dir or Path("output/neural_building_emulator/flexibility_event_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    dt_hours = float(args.dt_hours if args.dt_hours is not None else _config_value(metadata, "hp_dt_hours", 0.25))
    horizons_hours, horizon_steps = _resolve_horizons(args.horizons_hours, dt_hours)
    config = EventStudyConfig(
        horizons_hours=horizons_hours,
        horizon_steps=horizon_steps,
        dt_hours=dt_hours,
        min_setpoint_change=float(args.min_setpoint_change),
        min_events=int(args.min_events),
        controls=args.controls,
        estimator=args.kpi_estimator,
    )
    print(
        f"kpi_estimator={config.estimator} "
        f"controls={'not_applied' if config.estimator == 'direct_ratio' else config.controls}"
    )

    candidate_ids = _saved_profile_ids(metadata, args.profile_source)
    profile_ids = _select_profiles(candidate_ids, max_profiles=args.max_profiles, seed=args.seed)
    profiles = _load_closed_loop_profiles(
        dataset_path,
        profile_ids,
        include_space_heating_availability=(
            SPACE_HEATING_AVAILABILITY_COLUMN in artifact.metadata.get("input_columns", [])
        ),
        include_internal_gains=(
            INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN
            in artifact.metadata.get("input_columns", [])
        ),
        hp_power_area_normalization=str(
            _config_value(metadata, "hp_power_area_normalization", "zone_floor_area")
        ),
        metadata_columns=tuple(metadata.get("metadata_columns", ())) or None,
    )

    rng_key = jax.random.PRNGKey(int(args.seed))
    profile_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    pooled_simulation = _empty_pooled_accumulators(config)
    pooled_emulation = _empty_pooled_accumulators(config)
    pooled_scenarios = [
        _empty_pooled_accumulators(config)
        for _ in range(int(args.num_scenarios) if args.emulation_kpi_mode == "scenario_average" else 0)
    ]

    prob_eval_particles = int(args.prob_eval_particles or _config_value(metadata, "prob_eval_particles", 16))
    for index, profile in enumerate(profiles, start=1):
        simulated_kpis = estimate_trace_flexibility(profile, profile.targets, config)
        _add_trace_to_pooled(pooled_simulation, profile, profile.targets, config)

        rng_key, profile_key = jax.random.split(rng_key)
        if args.emulation_kpi_mode == "mean":
            emulated_trace = _predict_mean_trace(
                artifact,
                profile,
                key=profile_key,
                num_particles=prob_eval_particles,
            )
            emulated_kpis = estimate_trace_flexibility(profile, emulated_trace, config)
            _add_trace_to_pooled(pooled_emulation, profile, emulated_trace, config)
        else:
            scenarios = _sample_scenario_traces(
                artifact,
                profile,
                key=profile_key,
                num_scenarios=int(args.num_scenarios),
                hp_scenario_mode=args.prob_hp_scenario_mode,
            )
            kpis_by_scenario: list[list[FlexibilityKpi]] = []
            for scenario_index in range(scenarios.shape[0]):
                scenario_trace = scenarios[scenario_index]
                scenario_kpis = estimate_trace_flexibility(profile, scenario_trace, config)
                kpis_by_scenario.append(scenario_kpis)
                _add_trace_to_pooled(
                    pooled_scenarios[scenario_index],
                    profile,
                    scenario_trace,
                    config,
                )
                for kpi in scenario_kpis:
                    scenario_rows.append(
                        {
                            "profile_id": profile.profile_id,
                            "scenario_index": int(scenario_index),
                            "horizon_hours": kpi.horizon_hours,
                            "horizon_steps": kpi.horizon_steps,
                            "event_count": kpi.event_count,
                            "valid": bool(kpi.valid),
                            "beta_plus_w_m2_k": kpi.beta_plus_w_m2_k,
                            "beta_minus_w_m2_k": kpi.beta_minus_w_m2_k,
                            "up_flex_wh_m2_k": kpi.up_flex_wh_m2_k,
                            "down_flex_wh_m2_k": kpi.down_flex_wh_m2_k,
                        }
                    )
            emulated_kpis = _mean_kpis(
                kpis_by_scenario,
                profile_id=profile.profile_id,
                config=config,
            )

        for simulated_kpi, emulated_kpi in zip(simulated_kpis, emulated_kpis):
            row: dict[str, Any] = {
                "profile_id": profile.profile_id,
                "horizon_hours": simulated_kpi.horizon_hours,
                "horizon_steps": simulated_kpi.horizon_steps,
                "valid": bool(simulated_kpi.valid and emulated_kpi.valid),
            }
            row.update(_kpi_to_row("sim", simulated_kpi))
            row.update(_kpi_to_row("emu", emulated_kpi))
            row["diff_beta_plus_w_m2_k"] = row["emu_beta_plus_w_m2_k"] - row["sim_beta_plus_w_m2_k"]
            row["diff_beta_minus_w_m2_k"] = row["emu_beta_minus_w_m2_k"] - row["sim_beta_minus_w_m2_k"]
            row["diff_up_flex_wh_m2_k"] = row["emu_up_flex_wh_m2_k"] - row["sim_up_flex_wh_m2_k"]
            row["diff_down_flex_wh_m2_k"] = row["emu_down_flex_wh_m2_k"] - row["sim_down_flex_wh_m2_k"]
            profile_rows.append(row)

        last_sim = simulated_kpis[-1]
        last_emu = emulated_kpis[-1]
        print(
            f"profile={profile.profile_id} index={index}/{len(profiles)} "
            f"events={last_sim.event_count} "
            f"H={last_sim.horizon_hours:g}h "
            f"sim_up={last_sim.up_flex_wh_m2_k:.4f} "
            f"emu_up={last_emu.up_flex_wh_m2_k:.4f} "
            f"sim_down={last_sim.down_flex_wh_m2_k:.4f} "
            f"emu_down={last_emu.down_flex_wh_m2_k:.4f}"
        )

    pooled_rows.extend(_estimate_pooled_rows(pooled_simulation, signal="simulation", config=config))
    if args.emulation_kpi_mode == "mean":
        pooled_rows.extend(_estimate_pooled_rows(pooled_emulation, signal="emulation", config=config))
    else:
        pooled_scenario_rows: list[dict[str, Any]] = []
        for scenario_index, accumulators in enumerate(pooled_scenarios):
            pooled_scenario_rows.extend(
                _estimate_pooled_rows(
                    accumulators,
                    signal="emulation",
                    config=config,
                    scenario_index=scenario_index,
                )
            )
        pooled_rows.extend(_scenario_average_pooled_rows(pooled_scenario_rows, config))
    profile_df = pd.DataFrame(profile_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    coverage_df = pd.DataFrame(
        _coverage_rows(profile_df, scenario_df, coverage_quantiles)
    )
    coverage_summary = _coverage_summary(coverage_df)

    summary = {
        "artifact_dir": str(args.artifact_dir),
        "dataset_path": str(dataset_path),
        "model_kind": model_kind,
        "profile_source": args.profile_source,
        "requested_profiles": int(len(profile_ids)),
        "profile_count": int(profile_df["profile_id"].nunique()),
        "valid_profile_horizon_count": int(np.sum(profile_df["valid"].to_numpy(dtype=bool))),
        "emulation_kpi_mode": args.emulation_kpi_mode,
        "num_scenarios": int(args.num_scenarios) if args.emulation_kpi_mode == "scenario_average" else 0,
        "prob_hp_scenario_mode": args.prob_hp_scenario_mode,
        "prob_hp_emission_mode": getattr(artifact.model, "hp_emission_mode", None),
        "prob_hp_emission_mode_arg": args.prob_hp_emission_mode,
        "horizons_hours": list(config.horizons_hours),
        "horizon_steps": list(config.horizon_steps),
        "dt_hours": float(config.dt_hours),
        "min_setpoint_change": float(config.min_setpoint_change),
        "kpi_estimator": config.estimator,
        "controls": config.controls,
        "controls_applied": config.estimator == "regression" and config.controls != "none",
        "up_flex_wh_m2_k_by_horizon": _signal_summary_by_horizon(profile_df, "up_flex_wh_m2_k"),
        "down_flex_wh_m2_k_by_horizon": _signal_summary_by_horizon(profile_df, "down_flex_wh_m2_k"),
        "kpi_coverage_by_horizon": coverage_summary,
        "pooled_kpis": pooled_rows,
        "interpretation": (
            "With direct_ratio, F_H is H times the directional ratio of summed post-minus-pre "
            "power responses to summed signed setpoint changes; beta is retained as F_H/H. "
            "With regression, beta is the corresponding controlled OLS coefficient and F_H=H*beta. "
            "A positive downward value means a -1 K shock reduces electrical energy."
        ),
    }

    profile_csv = output_dir / "flexibility_event_study_profile_kpis.csv"
    pooled_csv = output_dir / "flexibility_event_study_pooled_kpis.csv"
    scenario_csv = output_dir / "flexibility_event_study_scenario_kpis.csv"
    coverage_csv = output_dir / "flexibility_event_study_kpi_coverage.csv"
    summary_json = output_dir / "flexibility_event_study_summary.json"
    html_path = output_dir / "flexibility_event_study_comparison.html"
    coverage_html_path = output_dir / "flexibility_event_study_kpi_coverage.html"

    profile_df.to_csv(profile_csv, index=False)
    pooled_df.to_csv(pooled_csv, index=False)
    if not scenario_df.empty:
        scenario_df.to_csv(scenario_csv, index=False)
    if not coverage_df.empty:
        coverage_df.to_csv(coverage_csv, index=False)
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    _write_html(
        html_path,
        profile_df,
        pooled_df,
        title=(
            "Windowed setpoint-shock flexibility KPI comparison"
            f"<br><sup>{model_kind}, {args.emulation_kpi_mode}, "
            f"hp_emission={getattr(artifact.model, 'hp_emission_mode', None)}, "
            f"H={','.join(f'{h:g}' for h in config.horizons_hours)} h, "
            f"estimator={config.estimator}, "
            f"controls={'not applied' if config.estimator == 'direct_ratio' else config.controls}</sup>"
        ),
    )
    if not coverage_df.empty:
        _write_coverage_html(
            coverage_html_path,
            coverage_df,
            title=(
            "Scenario KPI quantile reliability"
            f"<br><sup>{model_kind}, scenarios={args.num_scenarios}, "
            f"hp_emission={getattr(artifact.model, 'hp_emission_mode', None)}, "
            f"H={','.join(f'{h:g}' for h in config.horizons_hours)} h</sup>"
        ),
        )

    print(f"saved_profile_kpis={profile_csv}")
    print(f"saved_pooled_kpis={pooled_csv}")
    if not scenario_df.empty:
        print(f"saved_scenario_kpis={scenario_csv}")
    if not coverage_df.empty:
        print(f"saved_kpi_coverage={coverage_csv}")
        print(f"saved_kpi_coverage_html={coverage_html_path}")
    print(f"saved_summary={summary_json}")
    print(f"saved_html={html_path}")
    print(
        "prob_hp_emission_mode="
        f"{getattr(artifact.model, 'hp_emission_mode', None)} "
        f"arg={args.prob_hp_emission_mode}"
    )
    for horizon in config.horizons_hours:
        part = profile_df[(profile_df["horizon_hours"] == horizon) & profile_df["valid"]]
        if part.empty:
            print(f"H={horizon:g}h valid_profiles=0")
            continue
        print(
            f"H={horizon:g}h "
            f"up_mae={np.mean(np.abs(part['diff_up_flex_wh_m2_k'])):.4f} "
            f"up_bias={np.mean(part['diff_up_flex_wh_m2_k']):.4f} "
            f"down_mae={np.mean(np.abs(part['diff_down_flex_wh_m2_k'])):.4f} "
            f"down_bias={np.mean(part['diff_down_flex_wh_m2_k']):.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--profile-source",
        choices=("test", "train", "selected"),
        default="test",
        help="Saved artifact split used for the KPI comparison.",
    )
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--emulation-kpi-mode",
        choices=("mean", "scenario_average"),
        default="mean",
        help=(
            "mean computes the KPI on one mean emulator rollout; scenario_average "
            "computes the KPI per sampled scenario and averages the KPIs."
        ),
    )
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--prob-eval-particles", type=int, default=None)
    parser.add_argument(
        "--prob-hp-scenario-mode",
        choices=("expected", "bernoulli"),
        default="bernoulli",
    )
    parser.add_argument(
        "--prob-hp-emission-mode",
        choices=("artifact", "bounded", "legacy_lognormal_mean"),
        default="artifact",
        help=(
            "HP active-power emission mode to use when loading probabilistic closed-loop artifacts. "
            "'artifact' uses saved metadata; old artifacts without the field fall back to legacy_lognormal_mean."
        ),
    )
    parser.add_argument(
        "--coverage-quantiles",
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
        help=(
            "Comma-separated nominal quantiles for KPI scenario reliability curves. "
            "Only used with --emulation-kpi-mode scenario_average."
        ),
    )
    parser.add_argument(
        "--horizons-hours",
        default="0.5,1,2,3",
        help="Comma-separated activation durations H in hours.",
    )
    parser.add_argument("--dt-hours", type=float, default=None)
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument(
        "--kpi-estimator",
        choices=("direct_ratio", "regression"),
        default="direct_ratio",
        help=(
            "direct_ratio computes F_H directly from directional ratios of aggregate event "
            "responses (default); regression retains the previous controlled OLS beta estimator."
        ),
    )
    parser.add_argument(
        "--controls",
        choices=("none", "weather", "full"),
        default="full",
        help=(
            "Only used by --kpi-estimator regression. none uses only setpoint jumps; "
            "weather adds event weather/calendar and "
            "post-minus-pre weather-window changes; full also adds pre-window power, "
            "pre-event indoor temperature, previous setpoint, and previous setpoint jump."
        ),
    )
    return parser.parse_args()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
