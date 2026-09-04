"""Data loading, meter-level splitting, and sequence windowing.

The important invariant is that train/test splits happen over complete profiles
(`egid`s), never over timesteps. This avoids leakage from the same simulated
building appearing in both training and evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .columns import (
    CLOSED_LOOP_METADATA_COLUMNS,
    CLOSED_LOOP_INPUT_COLUMNS,
    CLOSED_LOOP_TARGET_COLUMNS,
    HP_MODEL_NAME_CATEGORIES,
    HP_MODEL_NAME_COLUMN,
    HP_MODEL_NAME_ONE_HOT_COLUMNS,
    HP_REF_CAPACITY_COLUMN,
    HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN,
    HP_REF_COP_COLUMN,
    SH_DESIGN_CAPACITY_COLUMN,
    SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN,
    SH_VOLUME_COLUMN,
    SH_VOLUME_PER_HEATED_AREA_COLUMN,
    DATETIME_COLUMN,
    DISTURBANCE_COLUMNS,
    HeatInputNormalization,
    HEAT_PUMP_ELECTRIC_POWER_COLUMN,
    HEATING_INPUT_COLUMNS,
    HPPowerAreaNormalization,
    HP_MODE_IS_DHW_COLUMN,
    HP_SIZE_BINDING_COLUMN,
    InputFeatureMode,
    METADATA_COLUMNS,
    PROFILE_ID_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    SPACE_HEATING_HP_SIZE_BINDING,
    TARGET_COLUMN,
    ZONE_THERMAL_HEATING_POWER_COLUMN,
    closed_loop_required_columns,
    input_columns,
    normalize_heating_mode,
    required_columns,
    source_input_columns,
)

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "tessin_results.parquet"
    / "variable_setpoints"
)
WindowTargetAlignment = Literal["same_time", "next_step"]


@dataclass(frozen=True)
class SplitConfig:
    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    max_profiles: int | None = 10
    test_fraction: float = 0.2
    seed: int = 13
    profile_selection: str = "random"  # "random" or "first"
    fixed_test_profile_ids: tuple[int, ...] = ()
    excluded_profile_ids: tuple[int, ...] = ()
    profile_id_column: str = PROFILE_ID_COLUMN
    heat_input_normalization: HeatInputNormalization = "per_floor_area"
    hp_power_area_normalization: HPPowerAreaNormalization = "building_heated_area"
    input_feature_mode: InputFeatureMode = "base"
    heating_regime_window_steps: int = 96 * 7


@dataclass(frozen=True)
class WindowConfig:
    sequence_length: int = 96
    stride: int = 96
    target_alignment: WindowTargetAlignment = "same_time"


@dataclass(frozen=True)
class BuildingDatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    selected_ids: tuple[int, ...]
    heating_mode: str
    heat_input_normalization: HeatInputNormalization
    hp_power_area_normalization: HPPowerAreaNormalization
    input_feature_mode: InputFeatureMode
    heating_regime_window_steps: int
    input_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]


@dataclass(frozen=True)
class ClosedLoopDatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    selected_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    dropped_ids: tuple[int, ...]
    dropped_non_hp_ids: tuple[int, ...]
    dropped_non_sh_hp_ids: tuple[int, ...]
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    hp_power_area_normalization: HPPowerAreaNormalization = "building_heated_area"
    heating_mode: str = "closed_loop_hp"
    heat_input_normalization: HeatInputNormalization = "per_floor_area"
    input_feature_mode: InputFeatureMode = "base"
    heating_regime_window_steps: int = 0


@dataclass(frozen=True)
class BuildingProfile:
    profile_id: int
    datetime: np.ndarray
    metadata: np.ndarray
    inputs: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class WindowedArrays:
    profile_ids: np.ndarray
    start_indices: np.ndarray
    metadata: np.ndarray
    inputs: np.ndarray
    targets: np.ndarray
    initial_temperature: np.ndarray


@dataclass(frozen=True)
class ClosedLoopProfile:
    profile_id: int
    datetime: np.ndarray
    metadata: np.ndarray
    inputs: np.ndarray
    targets: np.ndarray
    initial_temperature: np.ndarray


@dataclass(frozen=True)
class ClosedLoopWindowedArrays:
    profile_ids: np.ndarray
    start_indices: np.ndarray
    metadata: np.ndarray
    inputs: np.ndarray
    targets: np.ndarray
    initial_temperature: np.ndarray


def _read_parquet(path: Path, *, columns: Sequence[str], filters=None) -> pd.DataFrame:
    read_path: Path | list[Path]
    if path.is_dir():
        read_path = sorted(path.glob("*.parquet"))
        if not read_path:
            raise FileNotFoundError(f"No parquet files found in {path}")
    else:
        read_path = path
    try:
        return pd.read_parquet(read_path, columns=list(columns), filters=filters)
    except ImportError as exc:
        raise ImportError(
            "Reading the emulator parquet dataset requires pyarrow or fastparquet. "
            "Install the project dependencies, or run `uv add pyarrow`."
        ) from exc


def _parquet_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        return files
    return [path]


def read_parquet_columns(path: Path) -> tuple[str, ...]:
    """Read column names from the first parquet part without loading row data."""
    files = _parquet_files(path)
    return tuple(pq.read_schema(files[0]).names)


def read_profile_ids(dataset_path: Path, profile_id_column: str = PROFILE_ID_COLUMN) -> tuple[int, ...]:
    ids = _read_parquet(dataset_path, columns=[profile_id_column])[profile_id_column]
    ids = ids.dropna().astype(int).unique()
    return tuple(sorted(int(profile_id) for profile_id in ids))


def select_profile_ids(
    profile_ids: Sequence[int],
    *,
    max_profiles: int | None,
    seed: int,
    strategy: str,
) -> tuple[int, ...]:
    ids = np.asarray(sorted(int(profile_id) for profile_id in profile_ids), dtype=np.int64)
    if max_profiles is None or max_profiles >= len(ids):
        return tuple(int(profile_id) for profile_id in ids)
    if max_profiles < 1:
        raise ValueError("max_profiles must be positive or None")

    if strategy == "first":
        selected = ids[:max_profiles]
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        selected = rng.permutation(ids)[:max_profiles]
    else:
        raise ValueError("profile_selection must be 'random' or 'first'")
    return tuple(sorted(int(profile_id) for profile_id in selected))


def split_profile_ids(
    profile_ids: Sequence[int],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    ids = np.asarray(sorted(int(profile_id) for profile_id in profile_ids), dtype=np.int64)
    if len(ids) == 0:
        raise ValueError("Cannot split an empty profile set")
    if len(ids) == 1 or test_fraction == 0:
        return (int(ids[0]),), ()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    n_test = min(n_test, len(ids) - 1)

    test_ids = tuple(sorted(int(profile_id) for profile_id in shuffled[:n_test]))
    train_ids = tuple(sorted(int(profile_id) for profile_id in shuffled[n_test:]))
    return train_ids, test_ids


def select_and_split_profile_ids(
    profile_ids: Sequence[int],
    *,
    max_profiles: int | None,
    test_fraction: float,
    seed: int,
    strategy: str,
    fixed_test_profile_ids: Sequence[int] = (),
    excluded_profile_ids: Sequence[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Select profiles with an optional fixed holdout shared across experiments."""
    excluded = set(int(profile_id) for profile_id in excluded_profile_ids)
    available_ids = tuple(
        int(profile_id) for profile_id in profile_ids if int(profile_id) not in excluded
    )
    if not available_ids:
        raise ValueError("No profiles remain after applying excluded_profile_ids")
    if not fixed_test_profile_ids:
        selected_ids = select_profile_ids(
            available_ids,
            max_profiles=max_profiles,
            seed=seed,
            strategy=strategy,
        )
        train_ids, test_ids = split_profile_ids(
            selected_ids,
            test_fraction=test_fraction,
            seed=seed + 1,
        )
        return selected_ids, train_ids, test_ids

    available = tuple(sorted(int(profile_id) for profile_id in available_ids))
    available_set = set(available)
    test_ids = tuple(sorted(set(int(profile_id) for profile_id in fixed_test_profile_ids)))
    missing = tuple(sorted(set(test_ids).difference(available_set)))
    if missing:
        raise ValueError(f"Fixed test profile ids are not available: {missing}")
    if max_profiles is not None and max_profiles <= len(test_ids):
        raise ValueError(
            "max_profiles must exceed the number of fixed test profiles so at least "
            "one training profile remains"
        )
    train_candidates = tuple(profile_id for profile_id in available if profile_id not in test_ids)
    max_train_profiles = (
        None if max_profiles is None else max_profiles - len(test_ids)
    )
    train_ids = select_profile_ids(
        train_candidates,
        max_profiles=max_train_profiles,
        seed=seed,
        strategy=strategy,
    )
    selected_ids = tuple(sorted((*train_ids, *test_ids)))
    return selected_ids, train_ids, test_ids


def load_result_splits(config: SplitConfig, heating_mode: str = "zone_thermal") -> BuildingDatasetSplits:
    """Load a configurable subset and split it by complete `egid` profiles."""
    mode = normalize_heating_mode(heating_mode)
    dataset_path = Path(config.dataset_path)
    all_ids = read_profile_ids(dataset_path, config.profile_id_column)
    selected_ids, train_ids, test_ids = select_and_split_profile_ids(
        all_ids,
        max_profiles=config.max_profiles,
        test_fraction=config.test_fraction,
        seed=config.seed,
        strategy=config.profile_selection,
        fixed_test_profile_ids=config.fixed_test_profile_ids,
        excluded_profile_ids=config.excluded_profile_ids,
    )

    columns = required_columns(mode)
    filters = [(config.profile_id_column, "in", list(selected_ids))]
    try:
        df = _read_parquet(dataset_path, columns=columns, filters=filters)
    except (ValueError, NotImplementedError):
        df = _read_parquet(dataset_path, columns=columns)
        df = df[df[config.profile_id_column].isin(selected_ids)]

    df = df[df[config.profile_id_column].isin(selected_ids)].copy()
    df[config.profile_id_column] = df[config.profile_id_column].astype(int)
    df = df.sort_values([config.profile_id_column, DATETIME_COLUMN]).reset_index(drop=True)

    train = df[df[config.profile_id_column].isin(train_ids)].reset_index(drop=True)
    test = df[df[config.profile_id_column].isin(test_ids)].reset_index(drop=True)

    return BuildingDatasetSplits(
        train=train,
        test=test,
        train_ids=train_ids,
        test_ids=test_ids,
        selected_ids=selected_ids,
        heating_mode=mode,
        heat_input_normalization=config.heat_input_normalization,
        hp_power_area_normalization=config.hp_power_area_normalization,
        input_feature_mode=config.input_feature_mode,
        heating_regime_window_steps=config.heating_regime_window_steps,
        input_columns=tuple(
            input_columns(
                mode,
                config.heat_input_normalization,
                config.input_feature_mode,
                config.heating_regime_window_steps,
            )
        ),
        metadata_columns=tuple(METADATA_COLUMNS),
    )


def _check_required_columns(
    dataset_path: Path,
    required: Sequence[str],
    *,
    context: str,
) -> None:
    available = set(read_parquet_columns(dataset_path))
    missing = [column for column in required if column not in available]
    if missing:
        formatted = "\n  - ".join(missing)
        raise ValueError(
            f"{context} requires missing parquet columns:\n  - {formatted}\n"
            "Regenerate the EnergyPlus simulations with the updated output template."
        )


def load_closed_loop_result_splits(config: SplitConfig) -> ClosedLoopDatasetSplits:
    """Load the profile split required by the closed-loop HP emulator."""
    dataset_path = Path(config.dataset_path)
    columns = closed_loop_required_columns(CLOSED_LOOP_METADATA_COLUMNS)
    _check_required_columns(
        dataset_path,
        columns,
        context="--model-kind closed_loop_hp",
    )

    hp_filter_columns = [config.profile_id_column, "hp_ref_capacity_W", HP_SIZE_BINDING_COLUMN]
    hp_filter_df = _read_parquet(dataset_path, columns=hp_filter_columns)
    hp_filter_df[config.profile_id_column] = hp_filter_df[config.profile_id_column].astype(int)
    hp_by_profile = hp_filter_df.groupby(config.profile_id_column)[
        ["hp_ref_capacity_W", HP_SIZE_BINDING_COLUMN]
    ].first()
    positive_capacity_ids = tuple(
        sorted(
            int(profile_id)
            for profile_id, row in hp_by_profile.iterrows()
            if pd.notna(row["hp_ref_capacity_W"]) and float(row["hp_ref_capacity_W"]) > 0.0
        )
    )
    hp_ids = tuple(
        sorted(
            int(profile_id)
            for profile_id, row in hp_by_profile.iterrows()
            if (
                pd.notna(row["hp_ref_capacity_W"])
                and float(row["hp_ref_capacity_W"]) > 0.0
                and str(row[HP_SIZE_BINDING_COLUMN]).strip().upper()
                == SPACE_HEATING_HP_SIZE_BINDING
            )
        )
    )
    all_ids = tuple(sorted(int(profile_id) for profile_id in hp_by_profile.index))
    dropped_non_hp_ids = tuple(sorted(set(all_ids).difference(positive_capacity_ids)))
    dropped_non_sh_hp_ids = tuple(sorted(set(positive_capacity_ids).difference(hp_ids)))
    dropped_ids = tuple(sorted(set(all_ids).difference(hp_ids)))
    if not hp_ids:
        raise ValueError(
            "--model-kind closed_loop_hp found no HP profiles. "
            "Expected finite positive hp_ref_capacity_W and hp_size_binding='SH' "
            "for at least one profile."
        )
    selected_ids, train_ids, test_ids = select_and_split_profile_ids(
        hp_ids,
        max_profiles=config.max_profiles,
        test_fraction=config.test_fraction,
        seed=config.seed,
        strategy=config.profile_selection,
        fixed_test_profile_ids=config.fixed_test_profile_ids,
        excluded_profile_ids=config.excluded_profile_ids,
    )

    filters = [(config.profile_id_column, "in", list(selected_ids))]
    try:
        df = _read_parquet(dataset_path, columns=columns, filters=filters)
    except (ValueError, NotImplementedError):
        df = _read_parquet(dataset_path, columns=columns)
        df = df[df[config.profile_id_column].isin(selected_ids)]

    df = df[df[config.profile_id_column].isin(selected_ids)].copy()
    df[config.profile_id_column] = df[config.profile_id_column].astype(int)
    df = df.sort_values([config.profile_id_column, DATETIME_COLUMN]).reset_index(drop=True)

    train = df[df[config.profile_id_column].isin(train_ids)].reset_index(drop=True)
    test = df[df[config.profile_id_column].isin(test_ids)].reset_index(drop=True)

    return ClosedLoopDatasetSplits(
        train=train,
        test=test,
        train_ids=train_ids,
        test_ids=test_ids,
        selected_ids=selected_ids,
        candidate_ids=hp_ids,
        dropped_ids=dropped_ids,
        dropped_non_hp_ids=dropped_non_hp_ids,
        dropped_non_sh_hp_ids=dropped_non_sh_hp_ids,
        input_columns=tuple(CLOSED_LOOP_INPUT_COLUMNS),
        target_columns=tuple(CLOSED_LOOP_TARGET_COLUMNS),
        metadata_columns=tuple(CLOSED_LOOP_METADATA_COLUMNS),
        hp_power_area_normalization=config.hp_power_area_normalization,
    )


def _validate_heat_input_normalization(heat_input_normalization: str) -> HeatInputNormalization:
    if heat_input_normalization not in ("raw", "per_floor_area"):
        raise ValueError("heat_input_normalization must be 'raw' or 'per_floor_area'")
    return heat_input_normalization  # type: ignore[return-value]


def _validate_input_feature_mode(input_feature_mode: str) -> InputFeatureMode:
    if input_feature_mode not in ("base", "heating_regime"):
        raise ValueError("input_feature_mode must be 'base' or 'heating_regime'")
    return input_feature_mode  # type: ignore[return-value]


def _next_non_leap_year(year: int) -> int:
    candidate = year + 1
    while pd.Timestamp(year=candidate, month=1, day=1).is_leap_year:
        candidate += 1
    return candidate


def _has_missing_leap_day_gap(datetimes: pd.Series) -> bool:
    if len(datetimes) < 2:
        return False
    previous = datetimes.iloc[:-1].reset_index(drop=True)
    following = datetimes.iloc[1:].reset_index(drop=True)
    gaps = following - previous
    leap_gap = (
        (previous.dt.month == 2)
        & (previous.dt.day == 28)
        & (following.dt.month == 3)
        & (following.dt.day == 1)
        & (gaps > pd.Timedelta(hours=1))
    )
    return bool(leap_gap.any())


def _simulation_calendar_datetimes(datetime_values: np.ndarray) -> np.ndarray:
    """Return timestamps on a calendar consistent with the simulated time base.

    Some EnergyPlus output folders contain 365-day simulations labeled with
    leap-year timestamps, e.g. 2020-02-28 23:45 followed by 2020-03-01 00:00.
    Relabeling those profiles to the next non-leap year removes the artificial
    plotting gap and keeps day-of-year features aligned with the 365-day
    simulation index.
    """
    datetimes = pd.to_datetime(pd.Series(datetime_values))
    if datetimes.empty or datetimes.isna().any():
        return datetimes.to_numpy()

    years = datetimes.dt.year.unique()
    if len(years) != 1:
        return datetimes.to_numpy()

    year = int(years[0])
    is_leap_year = bool(pd.Timestamp(year=year, month=1, day=1).is_leap_year)
    has_feb29 = bool(((datetimes.dt.month == 2) & (datetimes.dt.day == 29)).any())
    if not is_leap_year or has_feb29 or not _has_missing_leap_day_gap(datetimes):
        return datetimes.to_numpy()

    target_year = _next_non_leap_year(year)
    remapped = datetimes.map(lambda value: value.replace(year=target_year))
    return remapped.to_numpy()


def _heating_regime_features(
    heat: np.ndarray,
    *,
    heat_on_threshold: float,
    window_steps: int,
) -> np.ndarray:
    if window_steps < 1:
        raise ValueError("heating_regime_window_steps must be positive")
    heat_on = (heat > np.float32(heat_on_threshold)).astype(np.float32)
    recently_on = (
        pd.Series(heat_on)
        .rolling(window=window_steps, min_periods=1)
        .max()
        .to_numpy(dtype=np.float32)
    )
    return np.column_stack([heat_on, recently_on]).astype(np.float32)


def _calendar_features(datetime_values: np.ndarray) -> np.ndarray:
    datetimes = pd.to_datetime(pd.Series(_simulation_calendar_datetimes(datetime_values)))
    hour = (
        datetimes.dt.hour.to_numpy(dtype=np.float32)
        + datetimes.dt.minute.to_numpy(dtype=np.float32) / np.float32(60.0)
    )
    day_of_year = datetimes.dt.dayofyear.to_numpy(dtype=np.float32)
    hour_angle = np.float32(2.0 * np.pi) * hour / np.float32(24.0)
    year_angle = np.float32(2.0 * np.pi) * (day_of_year - np.float32(1.0)) / np.float32(365.0)
    return np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(year_angle),
            np.cos(year_angle),
        ]
    ).astype(np.float32)


def _space_heating_availability(datetime_values: np.ndarray) -> np.ndarray:
    """Reproduce the EnergyPlus May 15 through September 30 SH lockout."""
    datetimes = pd.to_datetime(pd.Series(_simulation_calendar_datetimes(datetime_values)))
    month = datetimes.dt.month.to_numpy()
    day = datetimes.dt.day.to_numpy()
    summer_lockout = (
        ((month == 5) & (day >= 15))
        | ((month > 5) & (month < 9))
        | (month == 9)
    )
    return (~summer_lockout).astype(np.float32)


def _positive_heating_signal(values: np.ndarray, *, profile_id: int, column: str) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    negative = values < np.float32(-1e-3)
    if bool(np.any(negative)):
        min_value = float(np.min(values))
        raise ValueError(
            f"Profile {profile_id} has negative values in {column!r} down to {min_value:.6g}; "
            "closed_loop_hp expects non-negative heating powers."
        )
    return np.maximum(values, np.float32(0.0))


def to_closed_loop_profiles(
    df: pd.DataFrame,
    *,
    include_space_heating_availability: bool = True,
    hp_power_area_normalization: HPPowerAreaNormalization = "building_heated_area",
    metadata_columns: Sequence[str] | None = None,
) -> list[ClosedLoopProfile]:
    """Convert a dataframe split into profile arrays for closed-loop HP training.

    The alignment is explicit: exogenous inputs and power targets are taken at t,
    while the temperature target is Tin[t+1]. The initial temperature is Tin[t].
    """
    metadata_columns = tuple(
        METADATA_COLUMNS if metadata_columns is None else metadata_columns
    )
    profiles: list[ClosedLoopProfile] = []
    for profile_id, group in df.groupby(PROFILE_ID_COLUMN, sort=True):
        group = group.sort_values(DATETIME_COLUMN).reset_index(drop=True)
        if len(group) < 2:
            continue

        profile_datetime = _simulation_calendar_datetimes(group.loc[:, DATETIME_COLUMN].to_numpy())
        floor_area = float(group.loc[:, "floor_area"].iloc[0])
        if not np.isfinite(floor_area) or floor_area <= 0.0:
            raise ValueError(
                f"Profile {profile_id} has invalid floor_area={floor_area!r}; "
                "cannot build W/m2 closed-loop targets."
            )

        total_floors = float(group.loc[:, "totalFloors"].iloc[0])
        if not np.isfinite(total_floors) or total_floors <= 0.0:
            raise ValueError(
                f"Profile {profile_id} has invalid totalFloors={total_floors!r}; "
                "cannot normalize whole-building HP power."
            )
        heated_area = floor_area * total_floors
        if hp_power_area_normalization == "zone_floor_area":
            hp_power_area = floor_area
        elif hp_power_area_normalization == "building_heated_area":
            hp_power_area = heated_area
        else:
            raise ValueError(
                "hp_power_area_normalization must be 'zone_floor_area' or "
                "'building_heated_area'"
            )

        row = group.iloc[0]
        derived_metadata: dict[str, float] = {}
        if any(column not in METADATA_COLUMNS for column in metadata_columns):
            hp_model_name = str(row.get(HP_MODEL_NAME_COLUMN, "")).strip().lower()
            if hp_model_name not in HP_MODEL_NAME_CATEGORIES:
                raise ValueError(
                    f"Profile {profile_id} has unsupported hp_model_name={hp_model_name!r}; "
                    f"expected one of {HP_MODEL_NAME_CATEGORIES}"
                )
            derived_metadata = {
                HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN: (
                    float(row[HP_REF_CAPACITY_COLUMN]) / heated_area
                ),
                SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN: (
                    float(row[SH_DESIGN_CAPACITY_COLUMN]) / heated_area
                ),
                SH_VOLUME_PER_HEATED_AREA_COLUMN: (
                    float(row[SH_VOLUME_COLUMN]) / heated_area
                ),
                HP_REF_COP_COLUMN: float(row[HP_REF_COP_COLUMN]),
                **{
                    column: float(hp_model_name == category)
                    for category, column in zip(
                        HP_MODEL_NAME_CATEGORIES,
                        HP_MODEL_NAME_ONE_HOT_COLUMNS,
                    )
                },
            }
        metadata_values = [
            derived_metadata[column] if column in derived_metadata else row[column]
            for column in metadata_columns
        ]
        metadata = np.asarray(metadata_values, dtype=np.float32)
        if not bool(np.all(np.isfinite(metadata))):
            raise ValueError(
                f"Profile {profile_id} has non-finite closed-loop metadata values"
            )

        temperature = group.loc[:, TARGET_COLUMN].to_numpy(dtype=np.float32)
        setpoint = group.loc[:, SETPOINT_TIMESERIES_COLUMN].to_numpy(dtype=np.float32)
        disturbances = group.loc[:, DISTURBANCE_COLUMNS].to_numpy(dtype=np.float32)
        availability = _space_heating_availability(profile_datetime)
        calendar = _calendar_features(profile_datetime)

        q_room = group.loc[:, ZONE_THERMAL_HEATING_POWER_COLUMN].to_numpy(dtype=np.float32)
        p_el = group.loc[:, HEAT_PUMP_ELECTRIC_POWER_COLUMN].to_numpy(dtype=np.float32)
        dhw_mode = group.loc[:, HP_MODE_IS_DHW_COLUMN].to_numpy(dtype=np.float32) > np.float32(0.5)
        if include_space_heating_availability:
            space_heating_available = availability > np.float32(0.5)
            q_room = np.where(space_heating_available, q_room, np.float32(0.0))
            p_el = np.where(
                dhw_mode | ~space_heating_available,
                np.float32(0.0),
                p_el,
            )
        else:
            p_el = np.where(dhw_mode, np.float32(0.0), p_el)

        q_room = _positive_heating_signal(
            q_room,
            profile_id=int(profile_id),
            column=ZONE_THERMAL_HEATING_POWER_COLUMN,
        )
        p_el = _positive_heating_signal(
            p_el,
            profile_id=int(profile_id),
            column=HEAT_PUMP_ELECTRIC_POWER_COLUMN,
        )
        q_room = q_room / np.float32(floor_area)
        p_el = p_el / np.float32(hp_power_area)

        input_parts: list[np.ndarray] = [setpoint, disturbances]
        if include_space_heating_availability:
            input_parts.append(availability)
        input_parts.append(calendar)
        inputs = np.column_stack(input_parts).astype(np.float32)
        targets = np.column_stack([temperature[1:], q_room[:-1], p_el[:-1]]).astype(np.float32)
        profiles.append(
            ClosedLoopProfile(
                profile_id=int(profile_id),
                datetime=profile_datetime[1:],
                metadata=metadata,
                inputs=inputs[:-1],
                targets=targets,
                initial_temperature=temperature[:-1, np.newaxis].astype(np.float32),
            )
        )
    return profiles


def to_profiles(
    df: pd.DataFrame,
    heating_mode: str,
    heat_input_normalization: HeatInputNormalization = "per_floor_area",
    input_feature_mode: InputFeatureMode = "base",
    heating_regime_window_steps: int = 96 * 7,
    heat_on_threshold: float = 1e-6,
    hp_power_area_normalization: HPPowerAreaNormalization = "building_heated_area",
) -> list[BuildingProfile]:
    """Convert a dataframe split into per-building arrays."""
    mode = normalize_heating_mode(heating_mode)
    normalization = _validate_heat_input_normalization(heat_input_normalization)
    feature_mode = _validate_input_feature_mode(input_feature_mode)
    in_cols = source_input_columns(mode)
    profiles: list[BuildingProfile] = []

    for profile_id, group in df.groupby(PROFILE_ID_COLUMN, sort=True):
        group = group.sort_values(DATETIME_COLUMN)
        profile_datetime = _simulation_calendar_datetimes(group.loc[:, DATETIME_COLUMN].to_numpy())
        metadata = group.loc[:, METADATA_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
        inputs = group.loc[:, in_cols].to_numpy(dtype=np.float32)
        if normalization == "per_floor_area":
            floor_area = float(group.loc[:, "floor_area"].iloc[0])
            if not np.isfinite(floor_area) or floor_area <= 0.0:
                raise ValueError(
                    f"Profile {profile_id} has invalid floor_area={floor_area!r}; "
                    "cannot normalize heat input to W/m2."
                )
            heat_input_area = floor_area
            if mode == "heating_electric":
                total_floors = float(group.loc[:, "totalFloors"].iloc[0])
                if not np.isfinite(total_floors) or total_floors <= 0.0:
                    raise ValueError(
                        f"Profile {profile_id} has invalid totalFloors={total_floors!r}; "
                        "cannot normalize whole-building HP power."
                    )
                if hp_power_area_normalization == "building_heated_area":
                    heat_input_area *= total_floors
                elif hp_power_area_normalization != "zone_floor_area":
                    raise ValueError(
                        "hp_power_area_normalization must be 'zone_floor_area' or "
                        "'building_heated_area'"
                    )
            inputs[:, 0] = inputs[:, 0] / heat_input_area
        if feature_mode == "heating_regime":
            inputs = np.column_stack(
                [
                    inputs,
                    _heating_regime_features(
                        inputs[:, 0],
                        heat_on_threshold=heat_on_threshold,
                        window_steps=heating_regime_window_steps,
                    ),
                ]
            ).astype(np.float32)
        target = group.loc[:, [TARGET_COLUMN]].to_numpy(dtype=np.float32)
        profiles.append(
            BuildingProfile(
                profile_id=int(profile_id),
                datetime=profile_datetime,
                metadata=metadata,
                inputs=inputs,
                target=target,
            )
        )
    return profiles


def _window_starts(last_start: int, stride: int, start_offset: int) -> list[int]:
    if not 0 <= start_offset < stride:
        raise ValueError("start_offset must be in [0, stride)")
    base_starts = range(0, last_start + 1, stride)
    return [min(start + start_offset, last_start) for start in base_starts]


def make_windows(
    profiles: Iterable[BuildingProfile],
    config: WindowConfig,
    *,
    start_offset: int = 0,
) -> WindowedArrays:
    """Slice full profiles into fixed-length rollout windows."""
    if config.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if config.stride < 1:
        raise ValueError("stride must be positive")
    if config.target_alignment not in ("same_time", "next_step"):
        raise ValueError("target_alignment must be 'same_time' or 'next_step'")

    profile_ids: list[int] = []
    start_indices: list[int] = []
    metadata: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    initial_temperature: list[np.ndarray] = []

    for profile in profiles:
        n_steps = profile.target.shape[0]
        target_offset = 1 if config.target_alignment == "next_step" else 0
        last_start = n_steps - config.sequence_length - target_offset
        if last_start < 0:
            continue
        for start in _window_starts(last_start, config.stride, start_offset):
            end = start + config.sequence_length
            target_start = start + target_offset
            target_end = end + target_offset
            profile_ids.append(profile.profile_id)
            start_indices.append(start)
            metadata.append(profile.metadata)
            inputs.append(profile.inputs[start:end])
            targets.append(profile.target[target_start:target_end])
            initial_temperature.append(profile.target[start])

    if not inputs:
        raise ValueError("No windows were created; reduce sequence_length or load longer profiles")

    return WindowedArrays(
        profile_ids=np.asarray(profile_ids, dtype=np.int64),
        start_indices=np.asarray(start_indices, dtype=np.int64),
        metadata=np.stack(metadata).astype(np.float32),
        inputs=np.stack(inputs).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        initial_temperature=np.stack(initial_temperature).astype(np.float32),
    )


def make_closed_loop_windows(
    profiles: Iterable[ClosedLoopProfile],
    config: WindowConfig,
    *,
    start_offset: int = 0,
) -> ClosedLoopWindowedArrays:
    """Slice closed-loop profiles into fixed-length rollout windows."""
    if config.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if config.stride < 1:
        raise ValueError("stride must be positive")

    profile_ids: list[int] = []
    start_indices: list[int] = []
    metadata: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    initial_temperature: list[np.ndarray] = []

    for profile in profiles:
        n_steps = profile.targets.shape[0]
        last_start = n_steps - config.sequence_length
        if last_start < 0:
            continue
        for start in _window_starts(last_start, config.stride, start_offset):
            end = start + config.sequence_length
            profile_ids.append(profile.profile_id)
            start_indices.append(start)
            metadata.append(profile.metadata)
            inputs.append(profile.inputs[start:end])
            targets.append(profile.targets[start:end])
            initial_temperature.append(profile.initial_temperature[start])

    if not inputs:
        raise ValueError("No windows were created; reduce sequence_length or load longer profiles")

    return ClosedLoopWindowedArrays(
        profile_ids=np.asarray(profile_ids, dtype=np.int64),
        start_indices=np.asarray(start_indices, dtype=np.int64),
        metadata=np.stack(metadata).astype(np.float32),
        inputs=np.stack(inputs).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        initial_temperature=np.stack(initial_temperature).astype(np.float32),
    )


def _format_ids(ids: Sequence[int]) -> str:
    if not ids:
        return "[]"
    preview = ", ".join(str(profile_id) for profile_id in ids[:8])
    suffix = "" if len(ids) <= 8 else f", ... ({len(ids)} total)"
    return f"[{preview}{suffix}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect emulator train/test profile splits.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-profiles", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--profile-selection", choices=("random", "first"), default="random")
    parser.add_argument("--heating-mode", choices=(*HEATING_INPUT_COLUMNS.keys(), "A", "B"), default="A")
    parser.add_argument(
        "--heat-input-normalization",
        choices=("raw", "per_floor_area"),
        default="per_floor_area",
        help="Use the selected heat input as raw W or divide it by floor_area to W/m2.",
    )
    parser.add_argument(
        "--hp-power-area-normalization",
        choices=("building_heated_area", "zone_floor_area"),
        default="building_heated_area",
    )
    parser.add_argument("--input-feature-mode", choices=("base", "heating_regime"), default="base")
    parser.add_argument("--heating-regime-window-steps", type=int, default=96 * 7)
    parser.add_argument("--heat-on-threshold", type=float, default=1e-6)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--target-alignment", choices=("same_time", "next_step"), default="same_time")
    args = parser.parse_args()

    split_config = SplitConfig(
        dataset_path=args.dataset,
        max_profiles=args.max_profiles,
        test_fraction=args.test_fraction,
        seed=args.seed,
        profile_selection=args.profile_selection,
        heat_input_normalization=args.heat_input_normalization,
        hp_power_area_normalization=args.hp_power_area_normalization,
        input_feature_mode=args.input_feature_mode,
        heating_regime_window_steps=args.heating_regime_window_steps,
    )
    splits = load_result_splits(split_config, heating_mode=args.heating_mode)
    window_config = WindowConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        target_alignment=args.target_alignment,
    )
    train_windows = make_windows(
        to_profiles(
            splits.train,
            splits.heating_mode,
            splits.heat_input_normalization,
            splits.input_feature_mode,
            splits.heating_regime_window_steps,
            args.heat_on_threshold,
            splits.hp_power_area_normalization,
        ),
        window_config,
    )
    test_windows = None
    if splits.test_ids:
        test_windows = make_windows(
            to_profiles(
                splits.test,
                splits.heating_mode,
                splits.heat_input_normalization,
                splits.input_feature_mode,
                splits.heating_regime_window_steps,
                args.heat_on_threshold,
                splits.hp_power_area_normalization,
            ),
            window_config,
        )

    print(f"heating_mode={splits.heating_mode}")
    print(f"heat_input_normalization={splits.heat_input_normalization}")
    if splits.heating_mode == "heating_electric":
        print(f"hp_power_area_normalization={splits.hp_power_area_normalization}")
    print(
        "input_feature_mode="
        f"{splits.input_feature_mode} "
        f"heating_regime_window_steps={splits.heating_regime_window_steps}"
    )
    print(f"target_alignment={args.target_alignment}")
    print(f"input_columns={list(splits.input_columns)}")
    print(f"selected_ids={_format_ids(splits.selected_ids)}")
    print(f"train_ids={_format_ids(splits.train_ids)} rows={len(splits.train)} windows={len(train_windows.profile_ids)}")
    print(
        f"test_ids={_format_ids(splits.test_ids)} rows={len(splits.test)} "
        f"windows={0 if test_windows is None else len(test_windows.profile_ids)}"
    )
    overlap = set(splits.train_ids).intersection(splits.test_ids)
    print(f"train_test_profile_overlap={sorted(overlap)}")


if __name__ == "__main__":
    main()
