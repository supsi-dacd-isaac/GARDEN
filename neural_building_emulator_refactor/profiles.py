"""Load complete profiles referenced by a saved comparison artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

from neural_building_emulator.columns import (
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    PROFILE_ID_COLUMN,
    closed_loop_required_columns,
    required_columns,
)
from neural_building_emulator.data import (
    BuildingProfile,
    ClosedLoopProfile,
    read_dataset_frame,
    to_closed_loop_profiles,
    to_profiles,
)

from .artifacts import LoadedArtifact

ProfileSource = Literal["train", "test", "selected"]


def artifact_value(artifact: LoadedArtifact, name: str, default=None):
    if artifact.backend == "legacy":
        assert artifact.legacy_artifact is not None
        legacy_metadata = artifact.legacy_artifact.metadata
        if name in legacy_metadata:
            return legacy_metadata[name]
        return legacy_metadata.get("train_config", {}).get(name, default)
    experiment = artifact.metadata.get("experiment_config", {})
    return experiment.get(name, default)


def saved_profile_ids(artifact: LoadedArtifact, source: ProfileSource) -> tuple[int, ...]:
    metadata = (
        artifact.legacy_artifact.metadata
        if artifact.backend == "legacy" and artifact.legacy_artifact is not None
        else artifact.metadata
    )
    key = {"train": "train_ids", "test": "test_ids", "selected": "selected_ids"}[source]
    return tuple(int(value) for value in metadata.get(key, []))


def _read_profiles(
    dataset_path: Path,
    *,
    columns: Sequence[str],
    profile_ids: Sequence[int],
) -> pd.DataFrame:
    frame = read_dataset_frame(
        dataset_path,
        columns=columns,
        profile_ids=profile_ids,
    )
    return frame[frame[PROFILE_ID_COLUMN].isin(profile_ids)].copy()


def load_profiles(
    artifact: LoadedArtifact,
    *,
    dataset_path: Path | None = None,
    source: ProfileSource = "test",
    max_profiles: int | None = None,
    profile_ids: Sequence[int] | None = None,
) -> list[BuildingProfile] | list[ClosedLoopProfile]:
    """Load complete profiles in the exact saved split, preserving its order."""
    ids = (
        [int(profile_id) for profile_id in profile_ids]
        if profile_ids is not None
        else list(saved_profile_ids(artifact, source))
    )
    if max_profiles is not None:
        ids = ids[:max_profiles]
    if not ids:
        if profile_ids is not None:
            raise ValueError("profile_ids must contain at least one profile")
        raise ValueError(f"Artifact contains no {source} profile ids")
    path = Path(dataset_path or artifact_value(artifact, "dataset_path"))
    if artifact.spec.task == "closed_loop_hp":
        saved_metadata = (
            artifact.legacy_artifact.metadata
            if artifact.backend == "legacy" and artifact.legacy_artifact is not None
            else artifact.metadata
        )
        metadata_columns = tuple(saved_metadata.get("metadata_columns", ()))
        input_columns = (
            artifact.legacy_artifact.metadata.get("input_columns", [])
            if artifact.backend == "legacy" and artifact.legacy_artifact is not None
            else artifact.metadata.get("input_columns", [])
        )
        include_internal_gains = (
            INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in input_columns
        )
        include_dhw_request = DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN in input_columns
        frame = _read_profiles(
            path,
            columns=closed_loop_required_columns(
                metadata_columns or None,
                include_internal_gains=include_internal_gains,
                include_dhw_request=include_dhw_request,
            ),
            profile_ids=ids,
        )
        include_availability = "space_heating_available" in input_columns
        profiles = to_closed_loop_profiles(
            frame,
            include_space_heating_availability=include_availability,
            include_internal_gains=include_internal_gains,
            include_dhw_request=include_dhw_request,
            hp_power_area_normalization=str(
                artifact_value(
                    artifact,
                    "hp_power_area_normalization",
                    "zone_floor_area",
                )
            ),  # type: ignore[arg-type]
            metadata_columns=metadata_columns or None,
        )
    else:
        heating_mode = str(artifact_value(artifact, "heating_mode", "zone_thermal"))
        normalization = str(
            artifact_value(artifact, "heat_input_normalization", "per_floor_area")
        )
        input_feature_mode = str(artifact_value(artifact, "input_feature_mode", "base"))
        regime_steps = int(artifact_value(artifact, "heating_regime_window_steps", 96 * 7))
        heat_threshold = float(artifact_value(artifact, "heat_on_threshold", 1e-6))
        input_columns = (
            artifact.legacy_artifact.metadata.get("input_columns", [])
            if artifact.backend == "legacy" and artifact.legacy_artifact is not None
            else artifact.metadata.get("input_columns", [])
        )
        include_internal_gains = (
            INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in input_columns
        )
        frame = _read_profiles(
            path,
            columns=required_columns(
                heating_mode,
                include_internal_gains=include_internal_gains,
            ),
            profile_ids=ids,
        )
        profiles = to_profiles(
            frame,
            heating_mode,
            normalization,  # type: ignore[arg-type]
            input_feature_mode,  # type: ignore[arg-type]
            regime_steps,
            heat_threshold,
            str(
                artifact_value(
                    artifact,
                    "hp_power_area_normalization",
                    "zone_floor_area",
                )
            ),  # type: ignore[arg-type]
            include_internal_gains=include_internal_gains,
        )
    order = {profile_id: index for index, profile_id in enumerate(ids)}
    profiles = [profile for profile in profiles if profile.profile_id in order]
    profiles.sort(key=lambda profile: order[profile.profile_id])
    return profiles
