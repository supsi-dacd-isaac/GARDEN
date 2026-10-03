"""EnergyPlus-compatible natural-window ventilation for autonomous rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd


VentilationRolloutMode = Literal["recorded", "eplus_rule"]

DEFAULT_WINDOW_VENTILATION_ACH = 2.5
DEFAULT_HEATING_SEASON_OFFSET_C = 3.0
DEFAULT_NO_HEATING_SEASON_OFFSET_C = 1.0
DEFAULT_MINIMUM_INDOOR_OUTDOOR_DELTA_C = 2.0


@dataclass(frozen=True)
class EPlusVentilationSchedule:
    """Exogenous parts of the EnergyPlus window-ventilation controller."""

    time_available: np.ndarray
    minimum_indoor_temperature_c: np.ndarray
    nominal_flow_m3_s: float
    minimum_indoor_outdoor_delta_c: float


@dataclass(frozen=True)
class EPlusVentilationRollout:
    """Schedule and scaling needed to generate ventilation from predicted Tin."""

    schedule: EPlusVentilationSchedule
    ventilation_input_index: int
    outdoor_input_index: int
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    temperature_mean_c: float
    temperature_scale_c: float

    def apply(self, input_t: jnp.ndarray, temperature_scaled_t: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
        return replace_with_eplus_ventilation(
            input_t,
            temperature_scaled_t,
            time_available_t=jnp.asarray(self.schedule.time_available)[index],
            minimum_indoor_temperature_c_t=jnp.asarray(self.schedule.minimum_indoor_temperature_c)[index],
            nominal_flow_m3_s=self.schedule.nominal_flow_m3_s,
            minimum_indoor_outdoor_delta_c=self.schedule.minimum_indoor_outdoor_delta_c,
            ventilation_input_index=self.ventilation_input_index,
            outdoor_input_index=self.outdoor_input_index,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            temperature_mean_c=self.temperature_mean_c,
            temperature_scale_c=self.temperature_scale_c,
        )


def _metadata_value(
    metadata: np.ndarray,
    metadata_columns: Sequence[str],
    name: str,
) -> float:
    try:
        index = list(metadata_columns).index(name)
    except ValueError as exc:
        raise ValueError(
            f"EnergyPlus ventilation rollout requires metadata column {name!r}"
        ) from exc
    value = float(np.asarray(metadata).reshape(-1)[index])
    if not np.isfinite(value):
        raise ValueError(f"EnergyPlus ventilation rollout received non-finite {name!r}")
    return value


def build_eplus_ventilation_schedule(
    datetimes: np.ndarray,
    metadata: np.ndarray,
    metadata_columns: Sequence[str],
    *,
    nominal_ach: float = DEFAULT_WINDOW_VENTILATION_ACH,
    heating_season_offset_c: float = DEFAULT_HEATING_SEASON_OFFSET_C,
    no_heating_season_offset_c: float = DEFAULT_NO_HEATING_SEASON_OFFSET_C,
    minimum_indoor_outdoor_delta_c: float = DEFAULT_MINIMUM_INDOOR_OUTDOOR_DELTA_C,
) -> EPlusVentilationSchedule:
    """Build the non-state-dependent inputs of the idf-builder ventilation rule.

    The IDF schedule is available overnight. Against the interval timestamps in
    the exported data, ``time < 08:00 or time >= 20:00`` reproduces its boundary
    convention: 08:00 is off and 20:00 is on.
    """

    if nominal_ach <= 0.0:
        raise ValueError("nominal_ach must be positive")
    datetime = pd.DatetimeIndex(pd.to_datetime(np.asarray(datetimes)))
    if datetime.isna().any():
        raise ValueError("EnergyPlus ventilation rollout requires valid datetimes")
    minute_of_day = datetime.hour.to_numpy() * 60 + datetime.minute.to_numpy()
    time_available = (minute_of_day < 8 * 60) | (minute_of_day >= 20 * 60)

    month = datetime.month.to_numpy()
    day = datetime.day.to_numpy()
    heating_season = (
        (month < 5)
        | (month > 9)
        | ((month == 5) & (day <= 15))
        | ((month == 9) & (day >= 30))
    )
    nominal_setpoint_c = _metadata_value(metadata, metadata_columns, "shSetpoint")
    zone_volume_m3 = _metadata_value(metadata, metadata_columns, "volume")
    if zone_volume_m3 <= 0.0:
        raise ValueError("EnergyPlus ventilation rollout requires positive zone volume")
    offsets = np.where(
        heating_season,
        np.float32(heating_season_offset_c),
        np.float32(no_heating_season_offset_c),
    )
    return EPlusVentilationSchedule(
        time_available=time_available.astype(np.float32),
        minimum_indoor_temperature_c=(nominal_setpoint_c + offsets).astype(np.float32),
        nominal_flow_m3_s=float(nominal_ach * zone_volume_m3 / 3600.0),
        minimum_indoor_outdoor_delta_c=float(minimum_indoor_outdoor_delta_c),
    )


def replace_with_eplus_ventilation(
    input_t: jnp.ndarray,
    temperature_scaled_t: jnp.ndarray,
    *,
    time_available_t: jnp.ndarray,
    minimum_indoor_temperature_c_t: jnp.ndarray,
    nominal_flow_m3_s: jnp.ndarray,
    minimum_indoor_outdoor_delta_c: jnp.ndarray,
    ventilation_input_index: int,
    outdoor_input_index: int,
    input_mean: Sequence[float],
    input_scale: Sequence[float],
    temperature_mean_c: float,
    temperature_scale_c: float,
) -> jnp.ndarray:
    """Replace a scaled ventilation input using the current rollout state."""

    dtype = input_t.dtype
    indoor_c = (
        temperature_scaled_t * jnp.asarray(temperature_scale_c, dtype=dtype)
        + jnp.asarray(temperature_mean_c, dtype=dtype)
    )
    outdoor_c = (
        input_t[outdoor_input_index]
        * jnp.asarray(input_scale[outdoor_input_index], dtype=dtype)
        + jnp.asarray(input_mean[outdoor_input_index], dtype=dtype)
    )
    gate = (
        (time_available_t > 0.5)
        & (indoor_c >= minimum_indoor_temperature_c_t)
        & (indoor_c - outdoor_c >= minimum_indoor_outdoor_delta_c)
    )
    flow_m3_s = jnp.where(gate, nominal_flow_m3_s, jnp.asarray(0.0, dtype=dtype))
    scale = jnp.maximum(
        jnp.abs(jnp.asarray(input_scale[ventilation_input_index], dtype=dtype)),
        jnp.asarray(1e-6, dtype=dtype),
    )
    flow_scaled = (
        flow_m3_s - jnp.asarray(input_mean[ventilation_input_index], dtype=dtype)
    ) / scale
    return input_t.at[ventilation_input_index].set(flow_scaled)
