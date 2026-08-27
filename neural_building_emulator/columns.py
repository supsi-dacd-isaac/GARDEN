"""Column names and training-mode definitions for the Tessin emulator dataset."""

from __future__ import annotations

from typing import Literal

HeatingMode = Literal["zone_thermal", "heating_electric"]
HeatInputNormalization = Literal["raw", "per_floor_area"]
InputFeatureMode = Literal["base", "heating_regime"]

DATETIME_COLUMN = "datetime"
PROFILE_ID_COLUMN = "egid"
TARGET_COLUMN = "FL0_THZ0, Zone Air Temperature"
SETPOINT_TIMESERIES_COLUMN = "FL0_THZ0, Zone Thermostat Heating Setpoint Temperature"
SPACE_HEATING_AVAILABILITY_COLUMN = "space_heating_available"
HP_MODE_IS_DHW_COLUMN = "hp_mode_is_dhw"
HP_SIZE_BINDING_COLUMN = "hp_size_binding"
SPACE_HEATING_HP_SIZE_BINDING = "SH"

HEATING_INPUT_COLUMNS: dict[str, str] = {
    "zone_thermal": "zone_thermal_heating_power",
    "heating_electric": "heat_pump_electric_power",
}

ZONE_THERMAL_HEATING_POWER_COLUMN = HEATING_INPUT_COLUMNS["zone_thermal"]
HEAT_PUMP_ELECTRIC_POWER_COLUMN = HEATING_INPUT_COLUMNS["heating_electric"]

OPTION_TO_HEATING_MODE: dict[str, HeatingMode] = {
    "A": "zone_thermal",
    "B": "heating_electric",
}

DISTURBANCE_COLUMNS = [
    "Environment, Site Outdoor Air Drybulb Temperature",
    "Environment, Site Global Horizontal Solar Radiation Rate per Area",
    "FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate",
]

CLOSED_LOOP_CALENDAR_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

CLOSED_LOOP_INPUT_COLUMNS = [
    SETPOINT_TIMESERIES_COLUMN,
    *DISTURBANCE_COLUMNS,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    *CLOSED_LOOP_CALENDAR_COLUMNS,
]

CLOSED_LOOP_TARGET_COLUMNS = [
    "indoor_temperature_c",
    "zone_thermal_heating_power_sh_w_m2",
    "heat_pump_electric_power_sh_w_m2",
]

METADATA_COLUMNS = [
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
]


def normalize_heating_mode(mode: str) -> HeatingMode:
    """Accept option labels or explicit names and return the canonical mode."""
    normalized = mode.strip()
    normalized = OPTION_TO_HEATING_MODE.get(normalized.upper(), normalized).lower()
    if normalized not in HEATING_INPUT_COLUMNS:
        valid = ", ".join([*OPTION_TO_HEATING_MODE, *HEATING_INPUT_COLUMNS])
        raise ValueError(f"Unknown heating mode {mode!r}. Expected one of: {valid}")
    return normalized  # type: ignore[return-value]


def source_input_columns(heating_mode: str) -> list[str]:
    """Return raw parquet input columns: selected heat signal plus disturbances."""
    mode = normalize_heating_mode(heating_mode)
    return [HEATING_INPUT_COLUMNS[mode], *DISTURBANCE_COLUMNS]


def input_columns(
    heating_mode: str,
    heat_input_normalization: HeatInputNormalization = "raw",
    input_feature_mode: InputFeatureMode = "base",
    heating_regime_window_steps: int = 96 * 7,
) -> list[str]:
    """Return model input names after optional heat-channel normalization."""
    raw_columns = source_input_columns(heating_mode)
    if heat_input_normalization == "raw":
        columns = raw_columns
    elif heat_input_normalization == "per_floor_area":
        columns = [f"{raw_columns[0]}_per_m2", *raw_columns[1:]]
    else:
        raise ValueError("heat_input_normalization must be 'raw' or 'per_floor_area'")

    if input_feature_mode == "base":
        return columns
    if input_feature_mode == "heating_regime":
        heat_column = columns[0]
        return [
            *columns,
            f"{heat_column}_is_on",
            f"{heat_column}_recently_on_{heating_regime_window_steps}",
        ]
    raise ValueError("input_feature_mode must be 'base' or 'heating_regime'")


def required_columns(heating_mode: str) -> list[str]:
    """Return the full parquet column subset needed for a training split."""
    return [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        *source_input_columns(heating_mode),
        *METADATA_COLUMNS,
    ]


def closed_loop_required_columns() -> list[str]:
    """Return the parquet columns needed by the closed-loop HP emulator."""
    return [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        SETPOINT_TIMESERIES_COLUMN,
        *DISTURBANCE_COLUMNS,
        ZONE_THERMAL_HEATING_POWER_COLUMN,
        HEAT_PUMP_ELECTRIC_POWER_COLUMN,
        HP_MODE_IS_DHW_COLUMN,
        *METADATA_COLUMNS,
    ]
