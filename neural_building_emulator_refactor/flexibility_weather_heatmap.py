"""Map normalized thermostat-event responses over outdoor weather conditions.

For each clean setpoint event and response horizon H, this module computes

    g[k, H] = (mean(Pel after) - mean(Pel before)) / delta(Tset)

in W/(m2 K). Upward and downward events are kept separate. Events are first
averaged within each building/weather bin, then those building means are
averaged, so buildings with more events do not dominate the result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    CLOSED_LOOP_INPUT_COLUMNS,
    HP_REF_CAPACITY_COLUMN,
    HP_SIZE_BINDING_COLUMN,
    PROFILE_ID_COLUMN,
    SPACE_HEATING_HP_SIZE_BINDING,
    closed_loop_required_columns,
)
from neural_building_emulator.data import (
    ClosedLoopProfile,
    read_dataset_frame,
    read_profile_ids,
    select_profile_ids,
    to_closed_loop_profiles,
)


DEFAULT_DATASET = Path("data/control=setpoint_identification")
DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/flexibility_weather_heatmap"
)
DEFAULT_TEMPERATURE_BIN_EDGES = (
    -np.inf,
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    np.inf,
)
DEFAULT_IRRADIANCE_BIN_EDGES = (
    -np.inf,
    1.0,
    50.0,
    150.0,
    300.0,
    500.0,
    750.0,
    np.inf,
)


@dataclass(frozen=True)
class WeatherHeatmapConfig:
    dt_hours: float = 0.25
    horizons_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    min_setpoint_change_c: float = 0.05
    weather_window: str = "pre"
    require_space_heating_availability: bool = True
    availability_threshold: float = 0.5
    temperature_bin_edges: tuple[float, ...] = DEFAULT_TEMPERATURE_BIN_EDGES
    irradiance_bin_edges: tuple[float, ...] = DEFAULT_IRRADIANCE_BIN_EDGES
    min_buildings_per_bin: int = 10
    line_dashboard_horizon_hours: float = 3.0
    joint_heatmap_bins: int = 12


def _validate_edges(values: Sequence[float], name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64)
    if len(edges) < 3 or not bool(np.all(np.diff(edges) > 0.0)):
        raise ValueError(f"{name} must contain at least three strictly increasing edges")
    return edges


def _bin_labels(edges: np.ndarray, unit: str) -> list[str]:
    labels: list[str] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if np.isneginf(left):
            labels.append(f"< {right:g} {unit}")
        elif np.isposinf(right):
            labels.append(f">= {left:g} {unit}")
        else:
            labels.append(f"{left:g}-{right:g} {unit}")
    return labels


def _window_means(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def _clean_events(
    setpoint: np.ndarray,
    availability: np.ndarray,
    *,
    horizon_steps: int,
    threshold: float,
    require_availability: bool,
    availability_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    setpoint = np.asarray(setpoint, dtype=np.float64)
    changes = np.abs(np.diff(setpoint)) >= threshold
    candidates = np.flatnonzero(changes) + 1
    valid: list[int] = []
    excluded_boundary = 0
    excluded_nearby_event = 0
    excluded_unavailable = 0
    for event in candidates:
        if event < horizon_steps or event + horizon_steps > len(setpoint):
            excluded_boundary += 1
            continue
        pre = setpoint[event - horizon_steps : event]
        post = setpoint[event : event + horizon_steps]
        if (
            np.max(np.abs(np.diff(pre)), initial=0.0) >= threshold
            or np.max(np.abs(np.diff(post)), initial=0.0) >= threshold
        ):
            excluded_nearby_event += 1
            continue
        if require_availability and not bool(
            np.all(
                availability[event - horizon_steps : event + horizon_steps]
                > availability_threshold
            )
        ):
            excluded_unavailable += 1
            continue
        valid.append(int(event))
    return np.asarray(valid, dtype=np.int64), {
        "candidate_events": int(len(candidates)),
        "valid_events": int(len(valid)),
        "excluded_boundary": excluded_boundary,
        "excluded_nearby_event": excluded_nearby_event,
        "excluded_unavailable": excluded_unavailable,
    }


def _weather_at_events(
    values: np.ndarray,
    events: np.ndarray,
    horizon_steps: int,
    mode: str,
) -> np.ndarray:
    if mode == "event":
        return np.asarray(values, dtype=np.float64)[events]
    if mode == "pre":
        return _window_means(values, events - horizon_steps, events)
    if mode == "centered":
        return _window_means(values, events - horizon_steps, events + horizon_steps)
    raise ValueError("weather_window must be 'event', 'pre', or 'centered'")


def extract_event_responses(
    profile: ClosedLoopProfile,
    config: WeatherHeatmapConfig,
) -> tuple[pd.DataFrame, list[dict[str, int | float]]]:
    """Return one row per valid profile/event/horizon."""
    temperature_edges = _validate_edges(
        config.temperature_bin_edges, "temperature_bin_edges"
    )
    irradiance_edges = _validate_edges(
        config.irradiance_bin_edges, "irradiance_bin_edges"
    )
    setpoint = np.asarray(profile.inputs[:, 0], dtype=np.float64)
    outdoor_temperature = np.asarray(profile.inputs[:, 1], dtype=np.float64)
    irradiance = np.asarray(profile.inputs[:, 2], dtype=np.float64)
    ventilation = np.asarray(profile.inputs[:, 3], dtype=np.float64)
    internal_gain = np.asarray(profile.inputs[:, 4], dtype=np.float64)
    availability_index = CLOSED_LOOP_INPUT_COLUMNS.index("space_heating_available")
    availability = np.asarray(profile.inputs[:, availability_index], dtype=np.float64)
    power = np.asarray(profile.targets[:, 2], dtype=np.float64)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, int | float]] = []

    for horizon_hours in config.horizons_hours:
        horizon_steps = int(round(horizon_hours / config.dt_hours))
        if horizon_steps < 1 or not np.isclose(
            horizon_steps * config.dt_hours, horizon_hours
        ):
            raise ValueError(
                f"Horizon {horizon_hours:g} h is not an integer multiple of "
                f"dt_hours={config.dt_hours:g}"
            )
        events, counts = _clean_events(
            setpoint,
            availability,
            horizon_steps=horizon_steps,
            threshold=config.min_setpoint_change_c,
            require_availability=config.require_space_heating_availability,
            availability_threshold=config.availability_threshold,
        )
        diagnostics.append(
            {
                "profile_id": int(profile.profile_id),
                "horizon_hours": float(horizon_hours),
                **counts,
            }
        )
        if not len(events):
            continue

        deltas = setpoint[events] - setpoint[events - 1]
        pre_power = _window_means(power, events - horizon_steps, events)
        post_power = _window_means(power, events, events + horizon_steps)
        response = (post_power - pre_power) / deltas
        pre_outdoor_temperature = _window_means(
            outdoor_temperature, events - horizon_steps, events
        )
        post_outdoor_temperature = _window_means(
            outdoor_temperature, events, events + horizon_steps
        )
        pre_irradiance = _window_means(
            irradiance, events - horizon_steps, events
        )
        post_irradiance = _window_means(
            irradiance, events, events + horizon_steps
        )
        pre_internal_gain = _window_means(
            internal_gain, events - horizon_steps, events
        )
        post_internal_gain = _window_means(
            internal_gain, events, events + horizon_steps
        )
        pre_ventilation = _window_means(
            ventilation, events - horizon_steps, events
        )
        post_ventilation = _window_means(
            ventilation, events, events + horizon_steps
        )
        event_temperature = _weather_at_events(
            outdoor_temperature, events, horizon_steps, config.weather_window
        )
        event_irradiance = _weather_at_events(
            irradiance, events, horizon_steps, config.weather_window
        )
        temperature_bin = np.digitize(event_temperature, temperature_edges[1:-1])
        irradiance_bin = np.digitize(event_irradiance, irradiance_edges[1:-1])

        for index, event in enumerate(events):
            event_time = (
                pd.Timestamp(profile.datetime[event - 1])
                if event > 0 and event - 1 < len(profile.datetime)
                else pd.NaT
            )
            rows.append(
                {
                    "profile_id": int(profile.profile_id),
                    "horizon_hours": float(horizon_hours),
                    "direction": "up" if deltas[index] > 0.0 else "down",
                    "event_step": int(event),
                    "event_time": event_time,
                    "delta_tset_c": float(deltas[index]),
                    "pre_pel_w_m2": float(pre_power[index]),
                    "post_pel_w_m2": float(post_power[index]),
                    "delta_pel_w_m2": float(post_power[index] - pre_power[index]),
                    "normalized_response_w_m2_k": float(response[index]),
                    "outdoor_temperature_c": float(event_temperature[index]),
                    "irradiance_w_m2": float(event_irradiance[index]),
                    "pre_outdoor_temperature_c": float(
                        pre_outdoor_temperature[index]
                    ),
                    "delta_outdoor_temperature_c": float(
                        post_outdoor_temperature[index]
                        - pre_outdoor_temperature[index]
                    ),
                    "pre_irradiance_w_m2": float(pre_irradiance[index]),
                    "delta_irradiance_w_m2": float(
                        post_irradiance[index] - pre_irradiance[index]
                    ),
                    "delta_internal_gain_w_m2": float(
                        post_internal_gain[index] - pre_internal_gain[index]
                    ),
                    "delta_ventilation_m3_s": float(
                        post_ventilation[index] - pre_ventilation[index]
                    ),
                    "temperature_bin": int(temperature_bin[index]),
                    "irradiance_bin": int(irradiance_bin[index]),
                }
            )
    return pd.DataFrame(rows), diagnostics


def building_bin_means(events: pd.DataFrame) -> pd.DataFrame:
    """Average repeated events before combining buildings."""
    columns = [
        "profile_id",
        "horizon_hours",
        "direction",
        "temperature_bin",
        "irradiance_bin",
    ]
    if events.empty:
        return pd.DataFrame(
            columns=[*columns, "building_mean_response_w_m2_k", "event_count"]
        )
    return (
        events.groupby(columns, observed=True)["normalized_response_w_m2_k"]
        .agg(building_mean_response_w_m2_k="mean", event_count="size")
        .reset_index()
    )


def aggregate_building_bins(building_means: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bin responses with equal weight for every represented building."""
    columns = [
        "horizon_hours",
        "direction",
        "temperature_bin",
        "irradiance_bin",
    ]
    if building_means.empty:
        return pd.DataFrame(
            columns=[
                *columns,
                "mean_response_w_m2_k",
                "std_between_buildings_w_m2_k",
                "building_count",
                "event_count",
                "sem_w_m2_k",
            ]
        )
    aggregate = (
        building_means.groupby(columns, observed=True)
        .agg(
            mean_response_w_m2_k=("building_mean_response_w_m2_k", "mean"),
            std_between_buildings_w_m2_k=("building_mean_response_w_m2_k", "std"),
            building_count=("profile_id", "nunique"),
            event_count=("event_count", "sum"),
        )
        .reset_index()
    )
    aggregate["sem_w_m2_k"] = (
        aggregate["std_between_buildings_w_m2_k"]
        / np.sqrt(aggregate["building_count"].clip(lower=1))
    )
    return aggregate


def temperature_setpoint_building_means(events: pd.DataFrame) -> pd.DataFrame:
    """Average repeated events at fixed temperature bin and setpoint jump."""
    columns = [
        "profile_id",
        "horizon_hours",
        "direction",
        "temperature_bin",
        "delta_tset_c",
    ]
    if events.empty:
        return pd.DataFrame(
            columns=[
                *columns,
                "building_mean_delta_pel_w_m2",
                "building_mean_gain_w_m2_k",
                "event_count",
            ]
        )
    values = events.copy()
    # The excitation uses half-kelvin levels. Rounding prevents floating-point
    # representations of the same commanded jump from creating separate groups.
    values["delta_tset_c"] = values["delta_tset_c"].round(6)
    return (
        values.groupby(columns, observed=True)
        .agg(
            building_mean_delta_pel_w_m2=("delta_pel_w_m2", "mean"),
            building_mean_gain_w_m2_k=("normalized_response_w_m2_k", "mean"),
            event_count=("delta_pel_w_m2", "size"),
        )
        .reset_index()
    )


def aggregate_temperature_setpoint_response(building_means: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon_hours",
        "direction",
        "temperature_bin",
        "delta_tset_c",
    ]
    if building_means.empty:
        return pd.DataFrame(
            columns=[
                *columns,
                "mean_delta_pel_w_m2",
                "std_delta_pel_between_buildings_w_m2",
                "mean_gain_w_m2_k",
                "std_gain_between_buildings_w_m2_k",
                "building_count",
                "event_count",
                "sem_delta_pel_w_m2",
                "sem_gain_w_m2_k",
            ]
        )
    aggregate = (
        building_means.groupby(columns, observed=True)
        .agg(
            mean_delta_pel_w_m2=("building_mean_delta_pel_w_m2", "mean"),
            std_delta_pel_between_buildings_w_m2=(
                "building_mean_delta_pel_w_m2",
                "std",
            ),
            mean_gain_w_m2_k=("building_mean_gain_w_m2_k", "mean"),
            std_gain_between_buildings_w_m2_k=(
                "building_mean_gain_w_m2_k",
                "std",
            ),
            building_count=("profile_id", "nunique"),
            event_count=("event_count", "sum"),
        )
        .reset_index()
    )
    denominator = np.sqrt(aggregate["building_count"].clip(lower=1))
    aggregate["sem_delta_pel_w_m2"] = (
        aggregate["std_delta_pel_between_buildings_w_m2"] / denominator
    )
    aggregate["sem_gain_w_m2_k"] = (
        aggregate["std_gain_between_buildings_w_m2_k"] / denominator
    )
    return aggregate


def _joint_heatmap_edges(
    config: WeatherHeatmapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    finite_temperature_edges = np.asarray(
        [value for value in config.temperature_bin_edges if np.isfinite(value)],
        dtype=np.float64,
    )
    if len(finite_temperature_edges) < 2:
        raise ValueError("temperature_bin_edges need at least two finite values")
    internal_temperature_edges = np.linspace(
        finite_temperature_edges[0],
        finite_temperature_edges[-1],
        config.joint_heatmap_bins - 1,
    )
    temperature_edges = np.concatenate(
        ([-np.inf], internal_temperature_edges, [np.inf])
    )
    setpoint_edges = np.linspace(-4.0, 4.0, config.joint_heatmap_bins + 1)
    return temperature_edges, setpoint_edges


def joint_heatmap_building_means(
    events: pd.DataFrame,
    config: WeatherHeatmapConfig,
) -> pd.DataFrame:
    """Create per-building cells for the signed Tout-by-delta-Tset heatmap."""
    columns = ["profile_id", "temperature_bin_12", "setpoint_bin_12"]
    if events.empty:
        return pd.DataFrame(
            columns=[
                *columns,
                "building_mean_delta_pel_w_m2",
                "building_mean_gain_w_m2_k",
                "event_count",
            ]
        )
    temperature_edges, setpoint_edges = _joint_heatmap_edges(config)
    values = events[
        np.isclose(
            events["horizon_hours"],
            config.line_dashboard_horizon_hours,
        )
    ].copy()
    if values.empty:
        return pd.DataFrame(columns=columns)
    values["temperature_bin_12"] = np.digitize(
        values["outdoor_temperature_c"], temperature_edges[1:-1]
    )
    values["setpoint_bin_12"] = np.digitize(
        values["delta_tset_c"], setpoint_edges[1:-1]
    )
    values["temperature_bin_12"] = values["temperature_bin_12"].clip(
        0, config.joint_heatmap_bins - 1
    )
    values["setpoint_bin_12"] = values["setpoint_bin_12"].clip(
        0, config.joint_heatmap_bins - 1
    )
    return (
        values.groupby(columns, observed=True)
        .agg(
            building_mean_delta_pel_w_m2=("delta_pel_w_m2", "mean"),
            building_mean_gain_w_m2_k=("normalized_response_w_m2_k", "mean"),
            event_count=("delta_pel_w_m2", "size"),
        )
        .reset_index()
    )


def aggregate_joint_heatmap(building_means: pd.DataFrame) -> pd.DataFrame:
    columns = ["temperature_bin_12", "setpoint_bin_12"]
    if building_means.empty:
        return pd.DataFrame(
            columns=[
                *columns,
                "mean_delta_pel_w_m2",
                "mean_gain_w_m2_k",
                "building_count",
                "event_count",
            ]
        )
    return (
        building_means.groupby(columns, observed=True)
        .agg(
            mean_delta_pel_w_m2=("building_mean_delta_pel_w_m2", "mean"),
            mean_gain_w_m2_k=("building_mean_gain_w_m2_k", "mean"),
            building_count=("profile_id", "nunique"),
            event_count=("event_count", "sum"),
        )
        .reset_index()
    )


def _setpoint_linearity_summary(
    aggregate: pd.DataFrame,
    min_buildings: int,
) -> pd.DataFrame:
    """Fit delta Pel = intercept + slope * delta Tset in each Tout bin."""
    rows: list[dict[str, float | str | int]] = []
    group_columns = ["horizon_hours", "direction", "temperature_bin"]
    for keys, group in aggregate.groupby(group_columns, observed=True):
        group = group[group["building_count"] >= min_buildings]
        if len(group) < 3:
            continue
        x = group["delta_tset_c"].to_numpy(dtype=np.float64)
        y = group["mean_delta_pel_w_m2"].to_numpy(dtype=np.float64)
        weights = group["building_count"].to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(group)), x])
        root_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * root_weights[:, None], y * root_weights, rcond=None
        )[0]
        prediction = design @ coefficients
        mean = float(np.average(y, weights=weights))
        total = float(np.sum(weights * (y - mean) ** 2))
        residual = float(np.sum(weights * (y - prediction) ** 2))
        horizon, direction, temperature_bin = keys
        rows.append(
            {
                "horizon_hours": float(horizon),
                "direction": str(direction),
                "temperature_bin": int(temperature_bin),
                "setpoint_jump_levels": int(len(group)),
                "intercept_w_m2": float(coefficients[0]),
                "slope_w_m2_k": float(coefficients[1]),
                "linear_r2": 1.0 - residual / total if total > 1e-12 else np.nan,
                "weighted_rmse_w_m2": float(
                    np.sqrt(residual / max(float(np.sum(weights)), 1.0))
                ),
            }
        )
    return pd.DataFrame(rows)


def _linear_departure_summary(
    aggregate: pd.DataFrame,
    temperature_edges: np.ndarray,
    irradiance_edges: np.ndarray,
    min_buildings: int,
) -> pd.DataFrame:
    temperature_centers = np.asarray(
        [
            (
                right - 2.5
                if np.isneginf(left)
                else left + 2.5
                if np.isposinf(right)
                else (left + right) / 2
            )
            for left, right in zip(temperature_edges[:-1], temperature_edges[1:])
        ]
    )
    irradiance_centers = np.asarray(
        [
            (
                0.0
                if np.isneginf(left)
                else left + 125.0
                if np.isposinf(right)
                else (left + right) / 2
            )
            for left, right in zip(irradiance_edges[:-1], irradiance_edges[1:])
        ]
    )
    rows: list[dict[str, float | str | int]] = []
    for (horizon, direction), group in aggregate.groupby(
        ["horizon_hours", "direction"], observed=True
    ):
        group = group[group["building_count"] >= min_buildings]
        if len(group) < 4:
            continue
        y = group["mean_response_w_m2_k"].to_numpy(dtype=np.float64)
        t = temperature_centers[group["temperature_bin"].to_numpy(dtype=int)]
        solar = irradiance_centers[group["irradiance_bin"].to_numpy(dtype=int)]
        design = np.column_stack([np.ones(len(group)), t, solar])
        weights = group["building_count"].to_numpy(dtype=np.float64)
        root_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * root_weights[:, None], y * root_weights, rcond=None
        )[0]
        prediction = design @ coefficients
        residual = y - prediction
        weighted_mean = float(np.average(y, weights=weights))
        total = float(np.average((y - weighted_mean) ** 2, weights=weights))
        residual_variance = float(np.average(residual**2, weights=weights))
        rows.append(
            {
                "horizon_hours": float(horizon),
                "direction": str(direction),
                "supported_bin_count": int(len(group)),
                "linear_plane_r2": (
                    1.0 - residual_variance / total if total > 1e-12 else np.nan
                ),
                "linear_departure_rmse_w_m2_k": float(np.sqrt(residual_variance)),
                "response_between_bin_std_w_m2_k": float(np.sqrt(total)),
                "normalized_linear_departure": (
                    float(np.sqrt(residual_variance / total)) if total > 1e-12 else np.nan
                ),
                "supported_response_min_w_m2_k": float(np.min(y)),
                "supported_response_max_w_m2_k": float(np.max(y)),
            }
        )
    return pd.DataFrame(rows)


def _matrix(
    group: pd.DataFrame,
    column: str,
    n_temperature_bins: int,
    n_irradiance_bins: int,
) -> np.ndarray:
    result = np.full((n_temperature_bins, n_irradiance_bins), np.nan)
    for row in group.itertuples(index=False):
        result[int(row.temperature_bin), int(row.irradiance_bin)] = float(
            getattr(row, column)
        )
    return result


def write_heatmap(
    aggregate: pd.DataFrame,
    config: WeatherHeatmapConfig,
    path: Path,
) -> None:
    temperature_edges = _validate_edges(
        config.temperature_bin_edges, "temperature_bin_edges"
    )
    irradiance_edges = _validate_edges(
        config.irradiance_bin_edges, "irradiance_bin_edges"
    )
    temperature_labels = _bin_labels(temperature_edges, "C")
    irradiance_labels = _bin_labels(irradiance_edges, "W/m2")
    horizons = tuple(config.horizons_hours)
    titles = [
        f"H={horizon:g} h, {direction} setpoint"
        for horizon in horizons
        for direction in ("upward", "downward")
    ]
    figure = make_subplots(
        rows=len(horizons),
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.11,
        vertical_spacing=0.08,
    )
    for row_index, horizon in enumerate(horizons, start=1):
        pair = aggregate[
            aggregate["horizon_hours"].eq(horizon)
            & aggregate["building_count"].ge(config.min_buildings_per_bin)
        ]
        pair_values = pair["mean_response_w_m2_k"].to_numpy(dtype=np.float64)
        pair_values = pair_values[np.isfinite(pair_values)]
        if len(pair_values):
            low, high = np.quantile(pair_values, [0.02, 0.98])
            if low < 0.0:
                bound = max(abs(float(low)), abs(float(high)), 1e-6)
                zmin, zmax, colorscale, zmid = -bound, bound, "RdBu_r", 0.0
            else:
                zmin, zmax, colorscale, zmid = float(low), max(float(high), float(low) + 1e-6), "Viridis", None
        else:
            zmin, zmax, colorscale, zmid = 0.0, 1.0, "Viridis", None

        for column_index, direction in enumerate(("up", "down"), start=1):
            group = aggregate[
                aggregate["horizon_hours"].eq(horizon)
                & aggregate["direction"].eq(direction)
            ]
            response = _matrix(
                group,
                "mean_response_w_m2_k",
                len(temperature_labels),
                len(irradiance_labels),
            )
            building_count = _matrix(
                group,
                "building_count",
                len(temperature_labels),
                len(irradiance_labels),
            )
            event_count = _matrix(
                group,
                "event_count",
                len(temperature_labels),
                len(irradiance_labels),
            )
            sem = _matrix(
                group,
                "sem_w_m2_k",
                len(temperature_labels),
                len(irradiance_labels),
            )
            supported = building_count >= config.min_buildings_per_bin
            response = np.where(supported, response, np.nan)
            text = np.empty(response.shape, dtype=object)
            for y_index in range(response.shape[0]):
                for x_index in range(response.shape[1]):
                    text[y_index, x_index] = (
                        f"{response[y_index, x_index]:.2f}<br>n={int(building_count[y_index, x_index])}"
                        if np.isfinite(response[y_index, x_index])
                        else ""
                    )
            customdata = np.stack([building_count, event_count, sem], axis=-1)
            figure.add_trace(
                go.Heatmap(
                    x=irradiance_labels,
                    y=temperature_labels,
                    z=response,
                    zmin=zmin,
                    zmax=zmax,
                    zmid=zmid,
                    colorscale=colorscale,
                    text=text,
                    texttemplate="%{text}",
                    customdata=customdata,
                    hovertemplate=(
                        "Tout=%{y}<br>Irradiance=%{x}<br>"
                        "response=%{z:.3f} W/(m2 K)<br>"
                        "buildings=%{customdata[0]:.0f}<br>"
                        "events=%{customdata[1]:.0f}<br>"
                        "between-building SEM=%{customdata[2]:.3f}<extra></extra>"
                    ),
                    showscale=column_index == 2,
                    colorbar=(
                        {
                            "title": "W/(m2 K)",
                            "len": 0.19,
                            "y": 1.0 - (row_index - 0.5) / len(horizons),
                        }
                        if column_index == 2
                        else None
                    ),
                ),
                row=row_index,
                col=column_index,
            )
            figure.update_xaxes(
                title_text="Pre-event global horizontal irradiance"
                if row_index == len(horizons)
                else None,
                tickangle=30,
                row=row_index,
                col=column_index,
            )
            if column_index == 1:
                figure.update_yaxes(
                    title_text="Pre-event outdoor temperature",
                    row=row_index,
                    col=column_index,
                )

    figure.update_layout(
        title={
            "text": (
                "Normalized HP response to setpoint events by weather"
                "<br><sup>Cell value: building-equal mean of "
                "(post Pel - pre Pel) / delta Tset; n is number of buildings. "
                f"Bins with fewer than {config.min_buildings_per_bin} buildings are hidden.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1500,
        height=max(520, 390 * len(horizons)),
        margin={"l": 110, "r": 120, "t": 120, "b": 90},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def write_setpoint_response_scatter(
    aggregate: pd.DataFrame,
    config: WeatherHeatmapConfig,
    path: Path,
) -> None:
    """Plot raw power responses against signed setpoint jumps by Tout bin."""
    temperature_edges = _validate_edges(
        config.temperature_bin_edges, "temperature_bin_edges"
    )
    temperature_labels = _bin_labels(temperature_edges, "C")
    colors = (
        "#264653",
        "#287271",
        "#2a9d8f",
        "#7fba73",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#c44536",
        "#6d597a",
    )
    horizons = tuple(config.horizons_hours)
    figure = make_subplots(
        rows=len(horizons),
        cols=1,
        subplot_titles=[f"H={horizon:g} h" for horizon in horizons],
        vertical_spacing=0.08,
    )
    for row_index, horizon in enumerate(horizons, start=1):
        horizon_values = aggregate[
            aggregate["horizon_hours"].eq(horizon)
            & aggregate["building_count"].ge(config.min_buildings_per_bin)
        ]
        for temperature_bin, label in enumerate(temperature_labels):
            temperature_values = horizon_values[
                horizon_values["temperature_bin"].eq(temperature_bin)
            ]
            for direction in ("down", "up"):
                group = temperature_values[
                    temperature_values["direction"].eq(direction)
                ].sort_values("delta_tset_c")
                if group.empty:
                    continue
                customdata = np.column_stack(
                    [
                        group["mean_gain_w_m2_k"],
                        group["building_count"],
                        group["event_count"],
                    ]
                )
                figure.add_trace(
                    go.Scatter(
                        x=group["delta_tset_c"],
                        y=group["mean_delta_pel_w_m2"],
                        error_y={
                            "type": "data",
                            "array": group["sem_delta_pel_w_m2"],
                            "visible": True,
                            "thickness": 1,
                        },
                        mode="lines+markers",
                        line={"color": colors[temperature_bin % len(colors)], "width": 2},
                        marker={"size": 7},
                        name=label,
                        legendgroup=f"temperature-{temperature_bin}",
                        showlegend=row_index == 1 and direction == "down",
                        customdata=customdata,
                        hovertemplate=(
                            f"Tout={label}<br>"
                            "delta Tset=%{x:.2f} C<br>"
                            "delta mean Pel=%{y:.3f} W/m2<br>"
                            "gain=%{customdata[0]:.3f} W/(m2 K)<br>"
                            "buildings=%{customdata[1]:.0f}<br>"
                            "events=%{customdata[2]:.0f}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=1,
                )
        figure.add_hline(y=0.0, line_color="#77808a", line_dash="dot", row=row_index, col=1)
        figure.add_vline(x=0.0, line_color="#77808a", line_dash="dot", row=row_index, col=1)
        figure.update_yaxes(
            title_text="Post-pre Pel [W/m2]",
            zeroline=False,
            row=row_index,
            col=1,
        )
        if row_index == len(horizons):
            figure.update_xaxes(
                title_text="Signed setpoint change [C]",
                row=row_index,
                col=1,
            )
    figure.update_layout(
        title={
            "text": (
                "HP power response versus setpoint intervention"
                "<br><sup>Building-equal conditional means; error bars are "
                "between-building SEM. Near-straight lines support local linearity "
                "in delta Tset, while changing slopes across colors show Tout dependence.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1300,
        height=max(520, 390 * len(horizons)),
        legend={
            "title": "Pre-event outdoor temperature",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 100, "r": 50, "t": 170, "b": 80},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def write_gain_temperature_scatter(
    aggregate: pd.DataFrame,
    config: WeatherHeatmapConfig,
    path: Path,
) -> None:
    """Plot normalized response over Tout for each intervention magnitude."""
    temperature_edges = _validate_edges(
        config.temperature_bin_edges, "temperature_bin_edges"
    )
    temperature_labels = _bin_labels(temperature_edges, "C")
    horizons = tuple(config.horizons_hours)
    magnitudes = sorted(
        float(value)
        for value in np.unique(np.abs(aggregate["delta_tset_c"].to_numpy(dtype=float)))
        if value > 0.0
    )
    palette = (
        "#277da1",
        "#43aa8b",
        "#90be6d",
        "#f9c74f",
        "#f8961e",
        "#f3722c",
        "#e63946",
        "#6d597a",
    )
    titles = [
        f"H={horizon:g} h, {direction} setpoint"
        for horizon in horizons
        for direction in ("upward", "downward")
    ]
    figure = make_subplots(
        rows=len(horizons),
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.09,
        vertical_spacing=0.08,
    )
    for row_index, horizon in enumerate(horizons, start=1):
        for column_index, direction in enumerate(("up", "down"), start=1):
            values = aggregate[
                aggregate["horizon_hours"].eq(horizon)
                & aggregate["direction"].eq(direction)
                & aggregate["building_count"].ge(config.min_buildings_per_bin)
            ].copy()
            values["magnitude"] = values["delta_tset_c"].abs()
            for magnitude_index, magnitude in enumerate(magnitudes):
                group = values[np.isclose(values["magnitude"], magnitude)].sort_values(
                    "temperature_bin"
                )
                if group.empty:
                    continue
                x = [temperature_labels[int(value)] for value in group["temperature_bin"]]
                customdata = np.column_stack(
                    [
                        group["mean_delta_pel_w_m2"],
                        group["building_count"],
                        group["event_count"],
                    ]
                )
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=group["mean_gain_w_m2_k"],
                        error_y={
                            "type": "data",
                            "array": group["sem_gain_w_m2_k"],
                            "visible": True,
                            "thickness": 1,
                        },
                        mode="lines+markers",
                        line={"color": palette[magnitude_index % len(palette)], "width": 2},
                        marker={"size": 7},
                        name=f"|delta Tset|={magnitude:g} C",
                        legendgroup=f"magnitude-{magnitude:g}",
                        showlegend=row_index == 1 and column_index == 1,
                        customdata=customdata,
                        hovertemplate=(
                            "Tout=%{x}<br>gain=%{y:.3f} W/(m2 K)<br>"
                            "delta mean Pel=%{customdata[0]:.3f} W/m2<br>"
                            "buildings=%{customdata[1]:.0f}<br>"
                            "events=%{customdata[2]:.0f}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                )
            figure.add_hline(
                y=0.0,
                line_color="#77808a",
                line_dash="dot",
                row=row_index,
                col=column_index,
            )
            if column_index == 1:
                figure.update_yaxes(
                    title_text="Normalized response [W/(m2 K)]",
                    row=row_index,
                    col=column_index,
                )
            if row_index == len(horizons):
                figure.update_xaxes(
                    title_text="Pre-event outdoor temperature",
                    tickangle=30,
                    row=row_index,
                    col=column_index,
                )
    figure.update_layout(
        title={
            "text": (
                "Setpoint-normalized HP response over outdoor temperature"
                "<br><sup>Overlap between intervention-magnitude curves supports "
                "linearity in delta Tset; curvature along the x-axis supports a "
                "nonlinear temperature nuisance/interaction model.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1500,
        height=max(520, 390 * len(horizons)),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 100, "r": 50, "t": 160, "b": 90},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def write_three_hour_line_dashboard(
    aggregate: pd.DataFrame,
    config: WeatherHeatmapConfig,
    path: Path,
) -> None:
    """Combine the four directional 3 h line diagnostics in one figure."""
    temperature_edges = _validate_edges(
        config.temperature_bin_edges, "temperature_bin_edges"
    )
    temperature_labels = _bin_labels(temperature_edges, "C")
    temperature_colors = (
        "#264653",
        "#287271",
        "#2a9d8f",
        "#7fba73",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#c44536",
        "#6d597a",
    )
    magnitude_colors = (
        "#277da1",
        "#43aa8b",
        "#90be6d",
        "#f9c74f",
        "#f8961e",
        "#f3722c",
        "#e63946",
        "#6d597a",
    )
    horizon = config.line_dashboard_horizon_hours
    values = aggregate[
        np.isclose(aggregate["horizon_hours"], horizon)
        & aggregate["building_count"].ge(config.min_buildings_per_bin)
    ].copy()
    values["magnitude"] = values["delta_tset_c"].abs()
    magnitudes = sorted(float(value) for value in values["magnitude"].unique())
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Upward: raw power response",
            "Downward: raw power response",
            "Upward: raw response over Tout",
            "Downward: raw response over Tout",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )

    for column_index, direction in enumerate(("up", "down"), start=1):
        directional = values[values["direction"].eq(direction)]
        for temperature_bin, label in enumerate(temperature_labels):
            group = directional[
                directional["temperature_bin"].eq(temperature_bin)
            ].sort_values("delta_tset_c")
            if group.empty:
                continue
            customdata = np.column_stack(
                [
                    group["mean_gain_w_m2_k"],
                    group["building_count"],
                    group["event_count"],
                ]
            )
            figure.add_trace(
                go.Scatter(
                    x=group["delta_tset_c"],
                    y=group["mean_delta_pel_w_m2"],
                    error_y={
                        "type": "data",
                        "array": group["sem_delta_pel_w_m2"],
                        "visible": True,
                        "thickness": 1,
                    },
                    mode="lines+markers",
                    line={
                        "color": temperature_colors[
                            temperature_bin % len(temperature_colors)
                        ],
                        "width": 2,
                    },
                    marker={"size": 7},
                    name=f"Tout {label}",
                    legend="legend",
                    legendgroup=f"temperature-{temperature_bin}",
                    showlegend=column_index == 1,
                    customdata=customdata,
                    hovertemplate=(
                        f"Tout={label}<br>"
                        "delta Tset=%{x:.2f} C<br>"
                        "delta mean Pel=%{y:.3f} W/m2<br>"
                        "gain=%{customdata[0]:.3f} W/(m2 K)<br>"
                        "buildings=%{customdata[1]:.0f}<br>"
                        "events=%{customdata[2]:.0f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
        for magnitude_index, magnitude in enumerate(magnitudes):
            group = directional[
                np.isclose(directional["magnitude"], magnitude)
            ].sort_values("temperature_bin")
            if group.empty:
                continue
            x = [temperature_labels[int(value)] for value in group["temperature_bin"]]
            customdata = np.column_stack(
                [
                    group["mean_gain_w_m2_k"],
                    group["building_count"],
                    group["event_count"],
                ]
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=group["mean_delta_pel_w_m2"],
                    error_y={
                        "type": "data",
                        "array": group["sem_delta_pel_w_m2"],
                        "visible": True,
                        "thickness": 1,
                    },
                    mode="lines+markers",
                    line={
                        "color": magnitude_colors[
                            magnitude_index % len(magnitude_colors)
                        ],
                        "width": 2,
                    },
                    marker={"size": 7},
                    name=f"|delta Tset| {magnitude:g} C",
                    legend="legend2",
                    legendgroup=f"magnitude-{magnitude:g}",
                    showlegend=column_index == 1,
                    customdata=customdata,
                    hovertemplate=(
                        "Tout=%{x}<br>delta mean Pel=%{y:.3f} W/m2<br>"
                        "gain=%{customdata[0]:.3f} W/(m2 K)<br>"
                        "buildings=%{customdata[1]:.0f}<br>"
                        "events=%{customdata[2]:.0f}<extra></extra>"
                    ),
                ),
                row=2,
                col=column_index,
            )
        figure.add_hline(
            y=0.0,
            line_color="#77808a",
            line_dash="dot",
            row=1,
            col=column_index,
        )
        figure.add_hline(
            y=0.0,
            line_color="#77808a",
            line_dash="dot",
            row=2,
            col=column_index,
        )
        figure.update_xaxes(
            title_text="Signed setpoint change [C]",
            row=1,
            col=column_index,
        )
        figure.update_xaxes(
            title_text="Pre-event outdoor temperature",
            tickangle=30,
            row=2,
            col=column_index,
        )
        if column_index == 1:
            figure.update_yaxes(
                title_text="Post-pre Pel [W/m2]",
                row=1,
                col=column_index,
            )
            figure.update_yaxes(
                title_text="Post-pre Pel [W/m2]",
                row=2,
                col=column_index,
            )
    figure.update_layout(
        title={
            "text": (
                f"Three-hour thermostat response: linearity in delta Tset and "
                f"nonlinearity in Tout"
                "<br><sup>Building-equal conditional means; error bars are "
                "between-building SEM.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        font={"size": 15},
        width=1750,
        height=1100,
        legend={
            "title": {"text": "Outdoor temperature bins", "font": {"size": 17}},
            "font": {"size": 15},
            "x": 1.01,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
        },
        legend2={
            "title": {"text": "Setpoint jump magnitudes", "font": {"size": 17}},
            "font": {"size": 15},
            "x": 1.01,
            "y": 0.46,
            "xanchor": "left",
            "yanchor": "top",
        },
        margin={"l": 110, "r": 330, "t": 130, "b": 100},
    )
    figure.update_annotations(font={"size": 19})
    figure.update_xaxes(title_font={"size": 17}, tickfont={"size": 14})
    figure.update_yaxes(title_font={"size": 17}, tickfont={"size": 14})
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def write_temperature_setpoint_heatmap(
    aggregate: pd.DataFrame,
    config: WeatherHeatmapConfig,
    path: Path,
) -> None:
    """Plot raw 3 h power response over 12x12 Tout/setpoint-change bins."""
    temperature_edges, setpoint_edges = _joint_heatmap_edges(config)
    temperature_labels = _bin_labels(temperature_edges, "C")
    setpoint_labels = _bin_labels(setpoint_edges, "C")
    bins = config.joint_heatmap_bins
    response = np.full((bins, bins), np.nan)
    gain = np.full((bins, bins), np.nan)
    building_count = np.full((bins, bins), np.nan)
    event_count = np.full((bins, bins), np.nan)
    for row in aggregate.itertuples(index=False):
        y_index = int(row.setpoint_bin_12)
        x_index = int(row.temperature_bin_12)
        response[y_index, x_index] = float(row.mean_delta_pel_w_m2)
        gain[y_index, x_index] = float(row.mean_gain_w_m2_k)
        building_count[y_index, x_index] = float(row.building_count)
        event_count[y_index, x_index] = float(row.event_count)
    supported = building_count >= config.min_buildings_per_bin
    response = np.where(supported, response, np.nan)
    finite = response[np.isfinite(response)]
    bound = (
        max(float(np.quantile(np.abs(finite), 0.98)), 1e-6)
        if len(finite)
        else 1.0
    )
    customdata = np.stack([gain, building_count, event_count], axis=-1)
    text = np.empty(response.shape, dtype=object)
    for y_index in range(bins):
        for x_index in range(bins):
            text[y_index, x_index] = (
                f"{response[y_index, x_index]:.2f}"
                if np.isfinite(response[y_index, x_index])
                else ""
            )
    figure = go.Figure(
        go.Heatmap(
            x=temperature_labels,
            y=setpoint_labels,
            z=response,
            zmin=-bound,
            zmax=bound,
            zmid=0.0,
            colorscale="RdBu_r",
            text=text,
            texttemplate="%{text}",
            customdata=customdata,
            colorbar={"title": "Delta Pel<br>[W/m2]"},
            hovertemplate=(
                "Tout=%{x}<br>delta Tset=%{y}<br>"
                "delta mean Pel=%{z:.3f} W/m2<br>"
                "normalized gain=%{customdata[0]:.3f} W/(m2 K)<br>"
                "buildings=%{customdata[1]:.0f}<br>"
                "events=%{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": (
                f"Three-hour HP response over Tout and setpoint intervention"
                f"<br><sup>{bins} x {bins} bins; cell color is the building-equal "
                "mean post-minus-pre electric power. Unsupported cells are hidden.</sup>"
            ),
            "x": 0.5,
        },
        xaxis_title="Pre-event outdoor temperature",
        yaxis_title="Signed setpoint change",
        template="plotly_white",
        width=1350,
        height=900,
        margin={"l": 120, "r": 130, "t": 120, "b": 110},
    )
    figure.update_xaxes(tickangle=30)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def _select_space_heating_profiles(
    dataset: Path,
    *,
    max_profiles: int | None,
    selection: str,
    seed: int,
) -> tuple[tuple[int, ...], dict[str, int]]:
    all_ids = read_profile_ids(dataset)
    metadata = read_dataset_frame(
        dataset,
        columns=[PROFILE_ID_COLUMN, HP_REF_CAPACITY_COLUMN, HP_SIZE_BINDING_COLUMN],
        profile_ids=all_ids,
    )
    metadata = metadata.groupby(PROFILE_ID_COLUMN, observed=True).first().reset_index()
    capacity = pd.to_numeric(metadata[HP_REF_CAPACITY_COLUMN], errors="coerce")
    binding = metadata[HP_SIZE_BINDING_COLUMN].astype(str).str.strip().str.upper()
    keep = capacity.gt(0.0) & binding.eq(SPACE_HEATING_HP_SIZE_BINDING)
    eligible = tuple(sorted(metadata.loc[keep, PROFILE_ID_COLUMN].astype(int)))
    selected = select_profile_ids(
        eligible,
        max_profiles=max_profiles,
        seed=seed,
        strategy=selection,
    )
    return selected, {
        "all_profiles": int(len(all_ids)),
        "space_heating_hp_profiles": int(len(eligible)),
        "excluded_non_space_heating_profiles": int(len(all_ids) - len(eligible)),
        "selected_profiles": int(len(selected)),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    started = pd.Timestamp.now()
    dataset = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = WeatherHeatmapConfig(
        dt_hours=args.dt_hours,
        horizons_hours=tuple(args.horizons_hours),
        min_setpoint_change_c=args.min_setpoint_change_c,
        weather_window=args.weather_window,
        require_space_heating_availability=not args.include_unavailable_events,
        availability_threshold=args.availability_threshold,
        temperature_bin_edges=tuple(args.temperature_bin_edges),
        irradiance_bin_edges=tuple(args.irradiance_bin_edges),
        min_buildings_per_bin=args.min_buildings_per_bin,
        line_dashboard_horizon_hours=args.line_dashboard_horizon_hours,
        joint_heatmap_bins=args.joint_heatmap_bins,
    )
    selected_ids, selection_summary = _select_space_heating_profiles(
        dataset,
        max_profiles=args.max_profiles,
        selection=args.profile_selection,
        seed=args.seed,
    )
    print(
        "profile_filter=hp_ref_capacity_W>0,hp_size_binding=SH "
        f"kept={selection_summary['space_heating_hp_profiles']} "
        f"excluded={selection_summary['excluded_non_space_heating_profiles']} "
        f"selected={selection_summary['selected_profiles']}"
    )

    building_frames: list[pd.DataFrame] = []
    temperature_setpoint_frames: list[pd.DataFrame] = []
    joint_heatmap_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, int | float]] = []
    columns = closed_loop_required_columns()
    for batch_start in range(0, len(selected_ids), args.read_batch_size):
        batch_ids = selected_ids[batch_start : batch_start + args.read_batch_size]
        frame = read_dataset_frame(dataset, columns=columns, profile_ids=batch_ids)
        profiles = to_closed_loop_profiles(
            frame,
            include_space_heating_availability=True,
            include_internal_gains=True,
            hp_power_area_normalization="building_heated_area",
        )
        for profile in profiles:
            event_frame, diagnostics = extract_event_responses(profile, config)
            if not event_frame.empty:
                building_frames.append(building_bin_means(event_frame))
                temperature_setpoint_frames.append(
                    temperature_setpoint_building_means(event_frame)
                )
                joint_heatmap_frames.append(
                    joint_heatmap_building_means(event_frame, config)
                )
            diagnostic_rows.extend(diagnostics)
        completed = min(batch_start + len(batch_ids), len(selected_ids))
        print(f"processed_profiles={completed}/{len(selected_ids)}", flush=True)

    building_means = (
        pd.concat(building_frames, ignore_index=True)
        if building_frames
        else building_bin_means(pd.DataFrame())
    )
    aggregate = aggregate_building_bins(building_means)
    temperature_setpoint_building = (
        pd.concat(temperature_setpoint_frames, ignore_index=True)
        if temperature_setpoint_frames
        else temperature_setpoint_building_means(pd.DataFrame())
    )
    temperature_setpoint_aggregate = aggregate_temperature_setpoint_response(
        temperature_setpoint_building
    )
    joint_heatmap_building = (
        pd.concat(joint_heatmap_frames, ignore_index=True)
        if joint_heatmap_frames
        else joint_heatmap_building_means(pd.DataFrame(), config)
    )
    joint_heatmap_aggregate = aggregate_joint_heatmap(joint_heatmap_building)
    diagnostics = pd.DataFrame(diagnostic_rows)
    temperature_edges = _validate_edges(
        config.temperature_bin_edges, "temperature_bin_edges"
    )
    irradiance_edges = _validate_edges(
        config.irradiance_bin_edges, "irradiance_bin_edges"
    )
    nonlinearity = _linear_departure_summary(
        aggregate,
        temperature_edges,
        irradiance_edges,
        config.min_buildings_per_bin,
    )
    setpoint_linearity = _setpoint_linearity_summary(
        temperature_setpoint_aggregate,
        config.min_buildings_per_bin,
    )

    building_means.to_csv(output_dir / "building_bin_responses.csv", index=False)
    aggregate.to_csv(output_dir / "weather_bin_response_summary.csv", index=False)
    temperature_setpoint_aggregate.to_csv(
        output_dir / "temperature_setpoint_response_summary.csv", index=False
    )
    joint_heatmap_aggregate.to_csv(
        output_dir / "temperature_setpoint_heatmap_summary.csv", index=False
    )
    diagnostics.to_csv(output_dir / "event_filter_diagnostics.csv", index=False)
    nonlinearity.to_csv(output_dir / "weather_linearity_summary.csv", index=False)
    setpoint_linearity.to_csv(output_dir / "setpoint_linearity_summary.csv", index=False)
    heatmap_path = output_dir / "flexibility_weather_response_heatmaps.html"
    write_heatmap(aggregate, config, heatmap_path)
    response_scatter_path = output_dir / "delta_p_vs_delta_tset_by_temperature.html"
    write_setpoint_response_scatter(
        temperature_setpoint_aggregate,
        config,
        response_scatter_path,
    )
    gain_scatter_path = output_dir / "gain_vs_temperature_by_setpoint_step.html"
    write_gain_temperature_scatter(
        temperature_setpoint_aggregate,
        config,
        gain_scatter_path,
    )
    line_dashboard_path = output_dir / "three_hour_response_dashboard.html"
    write_three_hour_line_dashboard(
        temperature_setpoint_aggregate,
        config,
        line_dashboard_path,
    )
    temperature_setpoint_heatmap_path = (
        output_dir / "three_hour_temperature_setpoint_heatmap.html"
    )
    write_temperature_setpoint_heatmap(
        joint_heatmap_aggregate,
        config,
        temperature_setpoint_heatmap_path,
    )

    diagnostic_totals = (
        diagnostics.drop(columns=["profile_id", "horizon_hours"]).sum().astype(int).to_dict()
        if not diagnostics.empty
        else {}
    )
    summary: dict[str, object] = {
        "dataset": str(dataset),
        "estimand": "(mean Pel post - mean Pel pre) / delta Tset",
        "units": "W/(m2 K)",
        "power_area": "whole-building heated area",
        "aggregation": "event mean within building/bin, then equal-weight building mean",
        "selection": selection_summary,
        "config": asdict(config),
        "event_filter_totals_across_horizons": diagnostic_totals,
        "elapsed_seconds": float((pd.Timestamp.now() - started).total_seconds()),
        "outputs": {
            "heatmap": str(heatmap_path),
            "setpoint_response_scatter": str(response_scatter_path),
            "gain_temperature_scatter": str(gain_scatter_path),
            "three_hour_line_dashboard": str(line_dashboard_path),
            "three_hour_temperature_setpoint_heatmap": str(
                temperature_setpoint_heatmap_path
            ),
            "aggregate_csv": str(output_dir / "weather_bin_response_summary.csv"),
            "building_bin_csv": str(output_dir / "building_bin_responses.csv"),
            "linearity_csv": str(output_dir / "weather_linearity_summary.csv"),
            "temperature_setpoint_csv": str(
                output_dir / "temperature_setpoint_response_summary.csv"
            ),
            "setpoint_linearity_csv": str(
                output_dir / "setpoint_linearity_summary.csv"
            ),
            "temperature_setpoint_heatmap_csv": str(
                output_dir / "temperature_setpoint_heatmap_summary.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(f"saved_heatmap={heatmap_path}")
    print(f"saved_summary={output_dir / 'summary.json'}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot building-equal normalized HP setpoint responses over binned "
            "outdoor temperature and irradiance."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--profile-selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--read-batch-size", type=int, default=8)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument(
        "--horizons-hours", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0]
    )
    parser.add_argument("--min-setpoint-change-c", type=float, default=0.05)
    parser.add_argument(
        "--weather-window", choices=("event", "pre", "centered"), default="pre"
    )
    parser.add_argument("--include-unavailable-events", action="store_true")
    parser.add_argument("--availability-threshold", type=float, default=0.5)
    parser.add_argument("--min-buildings-per-bin", type=int, default=10)
    parser.add_argument("--line-dashboard-horizon-hours", type=float, default=3.0)
    parser.add_argument("--joint-heatmap-bins", type=int, default=12)
    parser.add_argument(
        "--temperature-bin-edges",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEMPERATURE_BIN_EDGES),
    )
    parser.add_argument(
        "--irradiance-bin-edges",
        type=float,
        nargs="+",
        default=list(DEFAULT_IRRADIANCE_BIN_EDGES),
    )
    args = parser.parse_args(argv)
    if args.read_batch_size < 1:
        parser.error("--read-batch-size must be positive")
    if args.min_buildings_per_bin < 1:
        parser.error("--min-buildings-per-bin must be positive")
    if args.joint_heatmap_bins < 2:
        parser.error("--joint-heatmap-bins must be at least 2")
    if not any(
        np.isclose(args.line_dashboard_horizon_hours, horizon)
        for horizon in args.horizons_hours
    ):
        parser.error(
            "--line-dashboard-horizon-hours must be included in --horizons-hours"
        )
    return args


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
