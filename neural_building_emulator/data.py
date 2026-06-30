"""Data loading, meter-level splitting, and sequence windowing.

The important invariant is that train/test splits happen over complete profiles
(`egid`s), never over timesteps. This avoids leakage from the same simulated
building appearing in both training and evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .columns import (
    DATETIME_COLUMN,
    HEATING_INPUT_COLUMNS,
    METADATA_COLUMNS,
    PROFILE_ID_COLUMN,
    TARGET_COLUMN,
    input_columns,
    normalize_heating_mode,
    required_columns,
)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "tessin_results.parquet"


@dataclass(frozen=True)
class SplitConfig:
    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    max_profiles: int | None = 10
    test_fraction: float = 0.2
    seed: int = 13
    profile_selection: str = "random"  # "random" or "first"
    profile_id_column: str = PROFILE_ID_COLUMN


@dataclass(frozen=True)
class WindowConfig:
    sequence_length: int = 96
    stride: int = 96


@dataclass(frozen=True)
class BuildingDatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    train_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    selected_ids: tuple[int, ...]
    heating_mode: str
    input_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]


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


def _read_parquet(path: Path, *, columns: Sequence[str], filters=None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns), filters=filters)
    except ImportError as exc:
        raise ImportError(
            "Reading the emulator parquet dataset requires pyarrow or fastparquet. "
            "Install the project dependencies, or run `uv add pyarrow`."
        ) from exc


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


def load_result_splits(config: SplitConfig, heating_mode: str = "zone_thermal") -> BuildingDatasetSplits:
    """Load a configurable subset and split it by complete `egid` profiles."""
    mode = normalize_heating_mode(heating_mode)
    dataset_path = Path(config.dataset_path)
    all_ids = read_profile_ids(dataset_path, config.profile_id_column)
    selected_ids = select_profile_ids(
        all_ids,
        max_profiles=config.max_profiles,
        seed=config.seed,
        strategy=config.profile_selection,
    )
    train_ids, test_ids = split_profile_ids(
        selected_ids,
        test_fraction=config.test_fraction,
        seed=config.seed + 1,
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
        input_columns=tuple(input_columns(mode)),
        metadata_columns=tuple(METADATA_COLUMNS),
    )


def to_profiles(df: pd.DataFrame, heating_mode: str) -> list[BuildingProfile]:
    """Convert a dataframe split into per-building arrays."""
    mode = normalize_heating_mode(heating_mode)
    in_cols = input_columns(mode)
    profiles: list[BuildingProfile] = []

    for profile_id, group in df.groupby(PROFILE_ID_COLUMN, sort=True):
        group = group.sort_values(DATETIME_COLUMN)
        metadata = group.loc[:, METADATA_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
        inputs = group.loc[:, in_cols].to_numpy(dtype=np.float32)
        target = group.loc[:, [TARGET_COLUMN]].to_numpy(dtype=np.float32)
        profiles.append(
            BuildingProfile(
                profile_id=int(profile_id),
                datetime=group.loc[:, DATETIME_COLUMN].to_numpy(),
                metadata=metadata,
                inputs=inputs,
                target=target,
            )
        )
    return profiles


def make_windows(profiles: Iterable[BuildingProfile], config: WindowConfig) -> WindowedArrays:
    """Slice full profiles into fixed-length rollout windows."""
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
        n_steps = profile.target.shape[0]
        last_start = n_steps - config.sequence_length
        if last_start < 0:
            continue
        for start in range(0, last_start + 1, config.stride):
            end = start + config.sequence_length
            profile_ids.append(profile.profile_id)
            start_indices.append(start)
            metadata.append(profile.metadata)
            inputs.append(profile.inputs[start:end])
            targets.append(profile.target[start:end])
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
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    args = parser.parse_args()

    split_config = SplitConfig(
        dataset_path=args.dataset,
        max_profiles=args.max_profiles,
        test_fraction=args.test_fraction,
        seed=args.seed,
        profile_selection=args.profile_selection,
    )
    splits = load_result_splits(split_config, heating_mode=args.heating_mode)
    window_config = WindowConfig(sequence_length=args.sequence_length, stride=args.stride)
    train_windows = make_windows(to_profiles(splits.train, splits.heating_mode), window_config)
    test_windows = None
    if splits.test_ids:
        test_windows = make_windows(to_profiles(splits.test, splits.heating_mode), window_config)

    print(f"heating_mode={splits.heating_mode}")
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
