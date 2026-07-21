"""Column names and training-mode definitions for the Tessin emulator dataset."""

from __future__ import annotations

from typing import Literal

HeatingMode = Literal["zone_thermal", "heating_electric"]
HeatInputNormalization = Literal["raw", "per_floor_area"]

DATETIME_COLUMN = "datetime"
PROFILE_ID_COLUMN = "egid"
TARGET_COLUMN = "FL0_THZ0, Zone Air Temperature"

HEATING_INPUT_COLUMNS: dict[str, str] = {
    "zone_thermal": "zone_thermal_heating_power",
    "heating_electric": "heat_pump_electric_power",
}

OPTION_TO_HEATING_MODE: dict[str, HeatingMode] = {
    "A": "zone_thermal",
    "B": "heating_electric",
}

DISTURBANCE_COLUMNS = [
    "Environment, Site Outdoor Air Drybulb Temperature",
    "Environment, Site Global Horizontal Solar Radiation Rate per Area",
    "FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate",
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
) -> list[str]:
    """Return model input names after optional heat-channel normalization."""
    raw_columns = source_input_columns(heating_mode)
    if heat_input_normalization == "raw":
        return raw_columns
    if heat_input_normalization == "per_floor_area":
        return [f"{raw_columns[0]}_per_m2", *raw_columns[1:]]
    raise ValueError("heat_input_normalization must be 'raw' or 'per_floor_area'")


def required_columns(heating_mode: str) -> list[str]:
    """Return the full parquet column subset needed for a training split."""
    return [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        *source_input_columns(heating_mode),
        *METADATA_COLUMNS,
    ]
