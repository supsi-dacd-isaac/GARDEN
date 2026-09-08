"""Column names and training-mode definitions for the Tessin emulator dataset."""

from __future__ import annotations

from typing import Literal

HeatingMode = Literal["zone_thermal", "heating_electric"]
HeatInputNormalization = Literal["raw", "per_floor_area"]
InputFeatureMode = Literal["base", "heating_regime"]
HPPowerAreaNormalization = Literal["zone_floor_area", "building_heated_area"]

DATETIME_COLUMN = "datetime"
PROFILE_ID_COLUMN = "egid"
TARGET_COLUMN = "FL0_THZ0, Zone Air Temperature"
SETPOINT_TIMESERIES_COLUMN = "FL0_THZ0, Zone Thermostat Heating Setpoint Temperature"
SPACE_HEATING_AVAILABILITY_COLUMN = "space_heating_available"
HP_MODE_IS_DHW_COLUMN = "hp_mode_is_dhw"
HP_SIZE_BINDING_COLUMN = "hp_size_binding"
SPACE_HEATING_HP_SIZE_BINDING = "SH"
HP_REF_CAPACITY_COLUMN = "hp_ref_capacity_W"
HP_REF_COP_COLUMN = "hp_ref_cop"
HP_MODEL_NAME_COLUMN = "hp_model_name"
SH_DESIGN_CAPACITY_COLUMN = "SH_design_cap_W"
SH_VOLUME_COLUMN = "shVolume_m3"
INTERNAL_GAIN_COLUMN = "input_non_people_internal_gain_net_kw"
OCCUPANTS_PRESENT_COLUMN = "input_occupants_present"
INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN = (
    "input_non_people_internal_gain_net_w_per_m2"
)
OCCUPANTS_PER_FLOOR_AREA_COLUMN = "input_occupants_present_per_m2"

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
    INTERNAL_GAIN_COLUMN,
    OCCUPANTS_PRESENT_COLUMN,
]

DISTURBANCE_INPUT_COLUMNS = [
    *DISTURBANCE_COLUMNS[:3],
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    OCCUPANTS_PER_FLOOR_AREA_COLUMN,
]

CLOSED_LOOP_CALENDAR_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

CLOSED_LOOP_INPUT_COLUMNS = [
    SETPOINT_TIMESERIES_COLUMN,
    *DISTURBANCE_INPUT_COLUMNS,
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

# Closed-loop HP models need equipment descriptors in addition to the envelope
# metadata shared with Q-to-T models. Powers and buffer volume are normalized by
# the whole heated area when profiles are materialized.
HP_MODEL_NAME_CATEGORIES = (
    "aerotop_g07_14m",
    "aerotop_t35r",
)
HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN = "hp_ref_capacity_w_per_m2_heated_area"
SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN = "sh_design_cap_w_per_m2_heated_area"
SH_VOLUME_PER_HEATED_AREA_COLUMN = "sh_volume_m3_per_m2_heated_area"
HP_MODEL_NAME_ONE_HOT_COLUMNS = [
    f"hp_model_name__{name}" for name in HP_MODEL_NAME_CATEGORIES
]
CLOSED_LOOP_HP_METADATA_SOURCE_COLUMNS = [
    HP_REF_CAPACITY_COLUMN,
    SH_DESIGN_CAPACITY_COLUMN,
    SH_VOLUME_COLUMN,
    HP_REF_COP_COLUMN,
    HP_MODEL_NAME_COLUMN,
]
CLOSED_LOOP_METADATA_COLUMNS = [
    *METADATA_COLUMNS,
    HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN,
    SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN,
    SH_VOLUME_PER_HEATED_AREA_COLUMN,
    HP_REF_COP_COLUMN,
    *HP_MODEL_NAME_ONE_HOT_COLUMNS,
]


def normalize_heating_mode(mode: str) -> HeatingMode:
    """Accept option labels or explicit names and return the canonical mode."""
    normalized = mode.strip()
    normalized = OPTION_TO_HEATING_MODE.get(normalized.upper(), normalized).lower()
    if normalized not in HEATING_INPUT_COLUMNS:
        valid = ", ".join([*OPTION_TO_HEATING_MODE, *HEATING_INPUT_COLUMNS])
        raise ValueError(f"Unknown heating mode {mode!r}. Expected one of: {valid}")
    return normalized  # type: ignore[return-value]


def source_input_columns(
    heating_mode: str,
    *,
    include_internal_gains: bool = True,
) -> list[str]:
    """Return raw parquet input columns: selected heat signal plus disturbances."""
    mode = normalize_heating_mode(heating_mode)
    disturbances = (
        DISTURBANCE_COLUMNS
        if include_internal_gains
        else DISTURBANCE_COLUMNS[:3]
    )
    return [HEATING_INPUT_COLUMNS[mode], *disturbances]


def input_columns(
    heating_mode: str,
    heat_input_normalization: HeatInputNormalization = "raw",
    input_feature_mode: InputFeatureMode = "base",
    heating_regime_window_steps: int = 96 * 7,
    *,
    include_internal_gains: bool = True,
) -> list[str]:
    """Return model input names after optional heat-channel normalization."""
    raw_columns = source_input_columns(
        heating_mode,
        include_internal_gains=include_internal_gains,
    )
    if heat_input_normalization == "raw":
        heat_column = raw_columns[0]
    elif heat_input_normalization == "per_floor_area":
        heat_column = f"{raw_columns[0]}_per_m2"
    else:
        raise ValueError("heat_input_normalization must be 'raw' or 'per_floor_area'")
    disturbance_columns = (
        DISTURBANCE_INPUT_COLUMNS
        if include_internal_gains
        else DISTURBANCE_INPUT_COLUMNS[:3]
    )
    columns = [heat_column, *disturbance_columns]

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


def required_columns(
    heating_mode: str,
    *,
    include_internal_gains: bool = True,
) -> list[str]:
    """Return the full parquet column subset needed for a training split."""
    return [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        *source_input_columns(
            heating_mode,
            include_internal_gains=include_internal_gains,
        ),
        *METADATA_COLUMNS,
    ]


def closed_loop_required_columns(
    metadata_columns: list[str] | tuple[str, ...] | None = None,
    *,
    include_internal_gains: bool = True,
) -> list[str]:
    """Return the parquet columns needed by the closed-loop HP emulator."""
    resolved_metadata = METADATA_COLUMNS if metadata_columns is None else metadata_columns
    requires_hp_metadata = any(
        column not in METADATA_COLUMNS for column in resolved_metadata
    )
    return [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        SETPOINT_TIMESERIES_COLUMN,
        *(
            DISTURBANCE_COLUMNS
            if include_internal_gains
            else DISTURBANCE_COLUMNS[:3]
        ),
        ZONE_THERMAL_HEATING_POWER_COLUMN,
        HEAT_PUMP_ELECTRIC_POWER_COLUMN,
        HP_MODE_IS_DHW_COLUMN,
        SPACE_HEATING_AVAILABILITY_COLUMN,
        *METADATA_COLUMNS,
        *(CLOSED_LOOP_HP_METADATA_SOURCE_COLUMNS if requires_hp_metadata else ()),
    ]
