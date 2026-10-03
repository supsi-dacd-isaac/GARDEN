"""Common event-response flexibility estimators for closed-loop controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from neural_building_emulator.columns import (
    DISTURBANCE_INPUT_COLUMNS,
    HP_FORCE_OFF_REQUESTED_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
)
from neural_building_emulator.data import ClosedLoopProfile

InterventionEstimator = Literal["direct_ratio", "regression"]


@dataclass(frozen=True)
class InterventionFlexibilityKpi:
    profile_id: int
    intervention_type: str
    estimator: InterventionEstimator
    horizon_hours: float
    horizon_steps: int
    event_count: int
    transition_count: int
    valid: bool
    response_gain_w_m2_per_unit: float
    flexibility_wh_m2_per_unit: float


def _invalid_kpi(
    profile_id: int,
    estimator: InterventionEstimator,
    horizon_hours: float,
    horizon_steps: int,
    *,
    event_count: int,
    transition_count: int,
) -> InterventionFlexibilityKpi:
    return InterventionFlexibilityKpi(
        profile_id=int(profile_id),
        intervention_type="force_off",
        estimator=estimator,
        horizon_hours=float(horizon_hours),
        horizon_steps=int(horizon_steps),
        event_count=int(event_count),
        transition_count=int(transition_count),
        valid=False,
        response_gain_w_m2_per_unit=float("nan"),
        flexibility_wh_m2_per_unit=float("nan"),
    )


def _window_mean(values: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(starts) == 0:
        return np.empty((0, *values.shape[1:]), dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=width,
        axis=0,
    )
    return np.asarray(np.mean(windows[starts], axis=-1), dtype=np.float64)


def _standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.reshape(len(values), 0)
    finite = np.where(np.isfinite(values), values, np.nan)
    means = np.nanmean(finite, axis=0)
    finite = np.where(np.isfinite(finite), finite, means)
    scales = np.nanstd(finite, axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    return (finite - means) / scales


def _stable_force_off_transitions(
    force_off: np.ndarray,
    heating_available: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    force_off = np.asarray(force_off, dtype=np.float64)
    available = np.asarray(heating_available, dtype=bool)
    delta = force_off[1:] - force_off[:-1]
    candidates = np.flatnonzero(np.abs(delta) > 0.5) + 1
    events: list[int] = []
    signed_demand_changes: list[float] = []
    for event in candidates:
        if event < horizon_steps or event + horizon_steps > len(force_off):
            continue
        if not available[event]:
            continue
        before = force_off[event - horizon_steps : event]
        after = force_off[event : event + horizon_steps]
        is_onset = bool(np.all(before <= 0.5) and np.all(after > 0.5))
        is_release = bool(np.all(before > 0.5) and np.all(after <= 0.5))
        if not (is_onset or is_release):
            continue
        events.append(int(event))
        # Positive means more heating demand; force-off onset is therefore -1.
        signed_demand_changes.append(float(force_off[event - 1] - force_off[event]))
    return (
        np.asarray(events, dtype=np.int64),
        np.asarray(signed_demand_changes, dtype=np.float64),
    )


def _control_matrix(
    profile: ClosedLoopProfile,
    trace: np.ndarray,
    events: np.ndarray,
    horizon_steps: int,
    input_columns: Sequence[str],
) -> np.ndarray:
    if len(events) == 0:
        return np.empty((0, 0), dtype=np.float64)
    inputs = np.asarray(profile.inputs, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    power = trace[:, 2]
    current_temperature = np.concatenate(
        [
            [float(np.asarray(profile.initial_temperature).reshape(-1)[0])],
            trace[:-1, 0],
        ]
    )
    starts = events - horizon_steps
    pre_power = _window_mean(power, starts, horizon_steps)
    controls: list[np.ndarray] = [
        pre_power[:, None],
        current_temperature[events, None],
    ]
    useful_columns = (
        SETPOINT_TIMESERIES_COLUMN,
        *DISTURBANCE_INPUT_COLUMNS[:3],
        INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    )
    selected = [
        input_columns.index(column)
        for column in useful_columns
        if column in input_columns and column != HP_FORCE_OFF_REQUESTED_COLUMN
    ]
    if selected:
        values = inputs[:, selected]
        controls.append(values[events])
        dynamic_columns = [
            index
            for index, column_index in enumerate(selected)
            if input_columns[column_index]
            in (
                *DISTURBANCE_INPUT_COLUMNS[:3],
                INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
            )
        ]
        if dynamic_columns:
            dynamic = values[:, dynamic_columns]
            pre = _window_mean(dynamic, starts, horizon_steps)
            post = _window_mean(dynamic, events, horizon_steps)
            controls.append(post - pre)
    return _standardized(np.column_stack(controls))


def _untreated_reference_events(
    force_off: np.ndarray,
    heating_available: np.ndarray,
    horizon_steps: int,
    *,
    target_count: int,
) -> np.ndarray:
    """Choose spread-out windows with no force-off transition as controls."""
    force_off = np.asarray(force_off, dtype=np.float64)
    available = np.asarray(heating_available, dtype=bool)
    candidates: list[int] = []
    for event in range(horizon_steps, len(force_off) - horizon_steps + 1):
        if not available[event]:
            continue
        window = force_off[event - horizon_steps : event + horizon_steps]
        if np.all(window <= 0.5):
            candidates.append(event)
    if not candidates or target_count <= 0:
        return np.empty(0, dtype=np.int64)
    # Avoid filling the regression with almost identical overlapping windows.
    spacing = max(1, horizon_steps)
    separated_values: list[int] = []
    for event in candidates:
        if not separated_values or event - separated_values[-1] >= spacing:
            separated_values.append(event)
    separated = np.asarray(separated_values, dtype=np.int64)
    if len(separated) <= target_count:
        return separated
    indices = np.linspace(0, len(separated) - 1, target_count, dtype=np.int64)
    return separated[indices]


def estimate_force_off_flexibility(
    profile: ClosedLoopProfile,
    trace: np.ndarray,
    force_off: np.ndarray,
    heating_available: np.ndarray,
    *,
    input_columns: Sequence[str],
    horizon_hours: float,
    horizon_steps: int,
    estimator: InterventionEstimator,
    min_events: int,
) -> InterventionFlexibilityKpi:
    """Estimate event-local energy shed after a persistent force-off onset.

    The direct estimator is ``H * sum(P_pre - P_post) / sum(force_off dose)``.
    The regression estimator compares force-off onsets with spread-out windows
    where no intervention occurred and residualizes both against pre-event
    state and exogenous controls. Its treatment coefficient is sign-flipped so
    positive values retain the same shed convention as the direct estimator.
    """
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] < 3:
        raise ValueError("trace must have shape [steps, 3] = [Tin, Qroom, Pel_SH]")
    if len(trace) != len(force_off) or len(trace) != len(profile.inputs):
        raise ValueError("trace, force_off, and profile inputs must have equal length")
    events, demand_changes = _stable_force_off_transitions(
        force_off,
        heating_available,
        horizon_steps,
    )
    onset = demand_changes < 0.0
    onset_count = int(np.sum(onset))
    if onset_count < min_events:
        return _invalid_kpi(
            profile.profile_id,
            estimator,
            horizon_hours,
            horizon_steps,
            event_count=onset_count,
            transition_count=len(events),
        )
    power = trace[:, 2]
    pre = _window_mean(power, events - horizon_steps, horizon_steps)
    post = _window_mean(power, events, horizon_steps)
    response = post - pre
    if estimator == "direct_ratio":
        denominator = float(np.sum(-demand_changes[onset]))
        gain = (
            float(np.sum(-response[onset]) / denominator)
            if denominator > 1e-12
            else float("nan")
        )
    else:
        onset_events = events[onset]
        references = _untreated_reference_events(
            force_off,
            heating_available,
            horizon_steps,
            target_count=max(4 * onset_count, min_events),
        )
        regression_events = np.concatenate([onset_events, references])
        treatment = np.concatenate(
            [
                np.ones(len(onset_events), dtype=np.float64),
                np.zeros(len(references), dtype=np.float64),
            ]
        )
        regression_pre = _window_mean(
            power,
            regression_events - horizon_steps,
            horizon_steps,
        )
        regression_post = _window_mean(
            power,
            regression_events,
            horizon_steps,
        )
        regression_response = regression_post - regression_pre
        controls = _control_matrix(
            profile,
            trace,
            regression_events,
            horizon_steps,
            input_columns,
        )
        design = np.column_stack(
            [np.ones(len(regression_events)), treatment, controls]
        )
        if len(regression_response) <= design.shape[1] or len(references) == 0:
            return _invalid_kpi(
                profile.profile_id,
                estimator,
                horizon_hours,
                horizon_steps,
                event_count=onset_count,
                transition_count=len(events),
            )
        coefficients, *_ = np.linalg.lstsq(
            design,
            regression_response,
            rcond=None,
        )
        gain = float(-coefficients[1])
    valid = bool(np.isfinite(gain))
    return InterventionFlexibilityKpi(
        profile_id=int(profile.profile_id),
        intervention_type="force_off",
        estimator=estimator,
        horizon_hours=float(horizon_hours),
        horizon_steps=int(horizon_steps),
        event_count=onset_count,
        transition_count=int(len(events)),
        valid=valid,
        response_gain_w_m2_per_unit=gain,
        flexibility_wh_m2_per_unit=float(horizon_hours * gain),
    )


def estimate_force_off_flexibility_curve(
    profile: ClosedLoopProfile,
    trace: np.ndarray,
    force_off: np.ndarray,
    heating_available: np.ndarray,
    *,
    input_columns: Sequence[str],
    horizons_hours: Sequence[float],
    dt_hours: float,
    min_events: int,
) -> list[InterventionFlexibilityKpi]:
    rows: list[InterventionFlexibilityKpi] = []
    for horizon_hours in horizons_hours:
        horizon_steps = max(1, int(round(float(horizon_hours) / dt_hours)))
        effective_horizon = float(horizon_steps * dt_hours)
        for estimator in ("direct_ratio", "regression"):
            rows.append(
                estimate_force_off_flexibility(
                    profile,
                    trace,
                    force_off,
                    heating_available,
                    input_columns=input_columns,
                    horizon_hours=effective_horizon,
                    horizon_steps=horizon_steps,
                    estimator=estimator,
                    min_events=min_events,
                )
            )
    return rows
