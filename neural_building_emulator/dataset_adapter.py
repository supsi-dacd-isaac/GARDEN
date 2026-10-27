"""Adapter for entity-oriented EnergyPlus simulation result folders.

The training code uses one canonical, wide row per timestamp and building. New
simulation exports instead store one row per timestamp and entity (building,
zone, or technical room), with static values in ``simulation_metadata.json``.
This module bridges those representations without changing the legacy parquet
reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .columns import (
    BUILDING_THERMAL_HEATING_POWER_COLUMN,
    SH_STORAGE_CHARGE_COLUMN,
    SH_DESIGN_SUPPLY_COLUMN,
    DATETIME_COLUMN,
    DHW_MIXED_WATER_COLUMN,
    HEAT_PUMP_ELECTRIC_POWER_COLUMN,
    HP_MODE_IS_DHW_COLUMN,
    HP_FORCE_OFF_EFFECTIVE_COLUMN,
    HP_FORCE_OFF_REQUESTED_COLUMN,
    INTERNAL_GAIN_COLUMN,
    OCCUPANTS_PRESENT_COLUMN,
    PROFILE_ID_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    TARGET_COLUMN,
    ZONE_THERMAL_HEATING_POWER_COLUMN,
)

ENTITY_TIMESERIES_FILENAME = "timeseries.parquet"
ENTITY_METADATA_FILENAME = "simulation_metadata.json"
MODELED_ZONE = "fl0_thz0"
BUILDING_ENTITY = "building_total"
TECHNICAL_ROOM_ENTITY = "technical_room"

OUTDOOR_TEMPERATURE_COLUMN = (
    "Environment, Site Outdoor Air Drybulb Temperature"
)
SOLAR_IRRADIANCE_COLUMN = (
    "Environment, Site Global Horizontal Solar Radiation Rate per Area"
)
VENTILATION_COLUMN = "FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate"
SOURCE_VENTILATION_COLUMN = "Zone Ventilation Standard Density Volume Flow Rate"
HP_EFFECTIVE_MODE_COLUMN = "hp_effective_mode"

# canonical column -> (preferred entity, source column)
_DYNAMIC_SOURCES: dict[str, tuple[str, str]] = {
    SH_STORAGE_CHARGE_COLUMN: (TECHNICAL_ROOM_ENTITY, SH_STORAGE_CHARGE_COLUMN),
    BUILDING_THERMAL_HEATING_POWER_COLUMN: ("*", ZONE_THERMAL_HEATING_POWER_COLUMN),
    TARGET_COLUMN: (MODELED_ZONE, "Zone Air Temperature"),
    SETPOINT_TIMESERIES_COLUMN: (
        MODELED_ZONE,
        "Zone Thermostat Heating Setpoint Temperature",
    ),
    ZONE_THERMAL_HEATING_POWER_COLUMN: (
        MODELED_ZONE,
        ZONE_THERMAL_HEATING_POWER_COLUMN,
    ),
    HEAT_PUMP_ELECTRIC_POWER_COLUMN: (
        TECHNICAL_ROOM_ENTITY,
        HEAT_PUMP_ELECTRIC_POWER_COLUMN,
    ),
    HP_MODE_IS_DHW_COLUMN: (TECHNICAL_ROOM_ENTITY, HP_MODE_IS_DHW_COLUMN),
    HP_EFFECTIVE_MODE_COLUMN: (TECHNICAL_ROOM_ENTITY, HP_EFFECTIVE_MODE_COLUMN),
    INTERNAL_GAIN_COLUMN: (MODELED_ZONE, INTERNAL_GAIN_COLUMN),
    OCCUPANTS_PRESENT_COLUMN: (MODELED_ZONE, OCCUPANTS_PRESENT_COLUMN),
    OUTDOOR_TEMPERATURE_COLUMN: (BUILDING_ENTITY, OUTDOOR_TEMPERATURE_COLUMN),
    SOLAR_IRRADIANCE_COLUMN: (BUILDING_ENTITY, SOLAR_IRRADIANCE_COLUMN),
    VENTILATION_COLUMN: (MODELED_ZONE, SOURCE_VENTILATION_COLUMN),
    HP_FORCE_OFF_REQUESTED_COLUMN: (
        TECHNICAL_ROOM_ENTITY,
        HP_FORCE_OFF_REQUESTED_COLUMN,
    ),
    HP_FORCE_OFF_EFFECTIVE_COLUMN: (
        TECHNICAL_ROOM_ENTITY,
        HP_FORCE_OFF_EFFECTIVE_COLUMN,
    ),
    SPACE_HEATING_AVAILABILITY_COLUMN: (
        BUILDING_ENTITY,
        SPACE_HEATING_AVAILABILITY_COLUMN,
    ),
    DHW_MIXED_WATER_COLUMN: ("*", DHW_MIXED_WATER_COLUMN),
}


def _entity_space_heating_availability(timestamps: pd.Series) -> pd.Series:
    """Fallback for schema-v4 exports that omitted their availability column.

    These exports apply the seasonal lockout from May 16 through September 29;
    September 30 is the first enabled day in the timestamped results.
    """
    datetimes = pd.to_datetime(timestamps)
    month = datetimes.dt.month
    day = datetimes.dt.day
    summer_lockout = (
        ((month == 5) & (day >= 16))
        | ((month > 5) & (month < 9))
        | ((month == 9) & (day <= 29))
    )
    return (~summer_lockout).astype("float32")


def entity_profile_directories(dataset_path: Path) -> list[Path]:
    """Return sorted ``egid=*`` profile folders from a new-format dataset."""
    return sorted(
        path
        for path in dataset_path.glob("egid=*")
        if path.is_dir() and (path / ENTITY_TIMESERIES_FILENAME).is_file()
    )


def is_entity_profile_dataset(dataset_path: Path) -> bool:
    """Whether ``dataset_path`` follows the entity-oriented folder schema."""
    return dataset_path.is_dir() and bool(entity_profile_directories(dataset_path))


def profile_id_from_directory(profile_dir: Path) -> int:
    try:
        return int(profile_dir.name.split("=", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid profile directory name: {profile_dir.name!r}") from exc


def read_entity_profile_ids(dataset_path: Path) -> tuple[int, ...]:
    return tuple(
        profile_id_from_directory(path)
        for path in entity_profile_directories(dataset_path)
    )


def _metadata_values(profile_dir: Path) -> dict[str, object]:
    metadata_path = profile_dir / ENTITY_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing simulation metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    try:
        values = metadata["gardenCompatibility"]["values"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{metadata_path} has no gardenCompatibility.values mapping"
        ) from exc
    if not isinstance(values, dict):
        raise ValueError(f"{metadata_path} gardenCompatibility.values must be an object")
    values = dict(values)
    supply = metadata.get("sizing", {}).get("sh_supply_C")
    if supply is not None:
        values[SH_DESIGN_SUPPLY_COLUMN] = float(supply)
    return values


def read_entity_dataset_columns(dataset_path: Path) -> tuple[str, ...]:
    """Return canonical columns exposed by an entity-oriented dataset."""
    profile_dirs = entity_profile_directories(dataset_path)
    if not profile_dirs:
        raise FileNotFoundError(f"No egid=*/timeseries.parquet files found in {dataset_path}")
    metadata_columns = tuple(_metadata_values(profile_dirs[0]))
    return tuple(
        dict.fromkeys(
            (
                DATETIME_COLUMN,
                PROFILE_ID_COLUMN,
                *_DYNAMIC_SOURCES,
                *metadata_columns,
            )
        )
    )


def _profile_directories_by_id(dataset_path: Path) -> dict[int, Path]:
    return {
        profile_id_from_directory(path): path
        for path in entity_profile_directories(dataset_path)
    }


def _read_dynamic_profile(
    profile_dir: Path,
    requested_columns: Sequence[str],
) -> pd.DataFrame:
    requested_dynamic = [
        column for column in requested_columns if column in _DYNAMIC_SOURCES
    ]
    parquet_path = profile_dir / ENTITY_TIMESERIES_FILENAME
    available_source_columns = set(pq.read_schema(parquet_path).names)
    requested_from_parquet = [
        column
        for column in requested_dynamic
        if _DYNAMIC_SOURCES[column][1] in available_source_columns
    ]
    source_columns = tuple(
        dict.fromkeys(
            (
                DATETIME_COLUMN,
                "zone",
                *(_DYNAMIC_SOURCES[column][1] for column in requested_from_parquet),
            )
        )
    )
    raw = pd.read_parquet(
        parquet_path,
        columns=list(source_columns),
    )
    if raw.empty:
        raise ValueError(f"Empty timeseries parquet: {profile_dir / ENTITY_TIMESERIES_FILENAME}")

    timestamps = (
        raw.loc[raw["zone"].eq(MODELED_ZONE), DATETIME_COLUMN]
        .drop_duplicates()
        .sort_values()
    )
    if timestamps.empty:
        raise ValueError(
            f"{profile_dir / ENTITY_TIMESERIES_FILENAME} has no {MODELED_ZONE!r} rows"
        )
    frame = pd.DataFrame({DATETIME_COLUMN: timestamps.to_numpy()})
    timestamp_index = pd.Index(frame[DATETIME_COLUMN])

    for canonical_column in requested_dynamic:
        entity, source_column = _DYNAMIC_SOURCES[canonical_column]
        if source_column not in available_source_columns:
            if canonical_column in (
                DHW_MIXED_WATER_COLUMN,
                HP_FORCE_OFF_REQUESTED_COLUMN,
                HP_FORCE_OFF_EFFECTIVE_COLUMN,
            ):
                frame[canonical_column] = np.float32(0.0)
                continue
            if canonical_column == SPACE_HEATING_AVAILABILITY_COLUMN:
                frame[canonical_column] = _entity_space_heating_availability(
                    frame[DATETIME_COLUMN]
                ).to_numpy()
                continue
            raise ValueError(
                f"{parquet_path} has no source column {source_column!r} for "
                f"{canonical_column!r}"
            )
        if canonical_column == BUILDING_THERMAL_HEATING_POWER_COLUMN:
            values = raw.loc[raw["zone"].str.match(r"^fl\d+_thz\d+$"),
                             [DATETIME_COLUMN, "zone", source_column]]
            if values.empty or values.duplicated([DATETIME_COLUMN, "zone"]).any():
                raise ValueError(f"Invalid whole-building zone heat rows in {parquet_path}")
            floors = values["zone"].str.extract(r"^fl(\d+)_", expand=False).astype(int).unique()
            expected_floors = int(_metadata_values(profile_dir)["totalFloors"])
            if set(floors) != set(range(expected_floors)):
                raise ValueError(f"Missing or unexpected heated floors in {parquet_path}")
            counts = values.groupby(DATETIME_COLUMN)[source_column].count()
            expected = values["zone"].nunique()
            if not bool((counts == expected).all()):
                raise ValueError(f"Incomplete zone heat observations in {parquet_path}")
            series = values.groupby(DATETIME_COLUMN)[source_column].sum(min_count=expected).reindex(timestamp_index)
            if series.isna().any():
                raise ValueError(f"Unaligned whole-building heat in {parquet_path}")
            frame[canonical_column] = series.to_numpy()
            continue
        if canonical_column == DHW_MIXED_WATER_COLUMN:
            values = raw.loc[
                raw["zone"].ne(BUILDING_ENTITY)
                & raw["zone"].ne(TECHNICAL_ROOM_ENTITY),
                [DATETIME_COLUMN, source_column],
            ].dropna(subset=[source_column])
            if values.empty:
                frame[canonical_column] = np.float32(0.0)
                continue
            series = (
                values.groupby(DATETIME_COLUMN, sort=True)[source_column]
                .sum()
                .reindex(timestamp_index)
            )
            if series.isna().any():
                missing = int(series.isna().sum())
                raise ValueError(
                    f"Cannot align {missing} values for {canonical_column!r} in "
                    f"{profile_dir / ENTITY_TIMESERIES_FILENAME}"
                )
            frame[canonical_column] = series.to_numpy()
            continue
        values = raw.loc[
            raw["zone"].eq(entity),
            [DATETIME_COLUMN, source_column],
        ].dropna(subset=[source_column])
        if values.empty:
            raise ValueError(
                f"{profile_dir / ENTITY_TIMESERIES_FILENAME} has no values for "
                f"{source_column!r} on entity {entity!r}"
            )
        if values[DATETIME_COLUMN].duplicated().any():
            raise ValueError(
                f"Duplicate timestamps for {source_column!r} on entity {entity!r} "
                f"in {profile_dir / ENTITY_TIMESERIES_FILENAME}"
            )
        series = values.set_index(DATETIME_COLUMN)[source_column].reindex(timestamp_index)
        if series.isna().any():
            missing = int(series.isna().sum())
            raise ValueError(
                f"Cannot align {missing} values for {canonical_column!r} in "
                f"{profile_dir / ENTITY_TIMESERIES_FILENAME}"
            )
        frame[canonical_column] = series.to_numpy()
    return frame


def read_entity_dataset(
    dataset_path: Path,
    *,
    columns: Sequence[str],
    profile_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Materialize selected new-format profiles as the canonical wide schema."""
    requested_columns = tuple(dict.fromkeys(columns))
    directories = _profile_directories_by_id(dataset_path)
    selected_ids = (
        tuple(sorted(directories))
        if profile_ids is None
        else tuple(int(profile_id) for profile_id in profile_ids)
    )
    missing_profiles = sorted(set(selected_ids).difference(directories))
    if missing_profiles:
        raise ValueError(f"Profile ids are not available in {dataset_path}: {missing_profiles}")

    available = set(read_entity_dataset_columns(dataset_path))
    missing_columns = [column for column in requested_columns if column not in available]
    if missing_columns:
        raise ValueError(
            "Entity-oriented dataset does not expose columns: "
            + ", ".join(repr(column) for column in missing_columns)
        )

    needs_timeseries = any(column in _DYNAMIC_SOURCES for column in requested_columns)
    frames: list[pd.DataFrame] = []
    for profile_id in selected_ids:
        profile_dir = directories[profile_id]
        metadata = _metadata_values(profile_dir)
        if needs_timeseries:
            frame = _read_dynamic_profile(profile_dir, requested_columns)
        else:
            frame = pd.DataFrame(index=pd.RangeIndex(1))
        for column in requested_columns:
            if column == PROFILE_ID_COLUMN:
                frame[column] = profile_id
            elif column in metadata:
                frame[column] = metadata[column]
            elif column == DATETIME_COLUMN or column in _DYNAMIC_SOURCES:
                continue
        frames.append(frame.loc[:, requested_columns])

    if not frames:
        return pd.DataFrame(columns=requested_columns)
    return pd.concat(frames, ignore_index=True)
