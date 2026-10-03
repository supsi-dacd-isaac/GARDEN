"""Regression tests for legacy and entity-oriented emulator datasets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from neural_building_emulator.columns import (
    CLOSED_LOOP_METADATA_COLUMNS,
    DATETIME_COLUMN,
    DHW_MIXED_WATER_COLUMN,
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    DISTURBANCE_COLUMNS,
    HEAT_PUMP_ELECTRIC_POWER_COLUMN,
    HP_MODE_IS_DHW_COLUMN,
    HP_FORCE_OFF_REQUESTED_COLUMN,
    HP_MODEL_NAME_COLUMN,
    HP_REF_CAPACITY_COLUMN,
    HP_REF_COP_COLUMN,
    HP_SIZE_BINDING_COLUMN,
    INTERNAL_GAIN_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    METADATA_COLUMNS,
    OCCUPANTS_PER_FLOOR_AREA_COLUMN,
    OCCUPANTS_PRESENT_COLUMN,
    PROFILE_ID_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
    SH_DESIGN_CAPACITY_COLUMN,
    SH_VOLUME_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    TARGET_COLUMN,
    ZONE_THERMAL_HEATING_POWER_COLUMN,
    closed_loop_required_columns,
    required_columns,
)
from neural_building_emulator.data import (
    SplitConfig,
    load_closed_loop_result_splits,
    load_result_splits,
    read_dataset_frame,
    read_profile_ids,
    to_closed_loop_profiles,
    to_profiles,
)


def _metadata_values(profile_id: int) -> dict[str, object]:
    values: dict[str, object] = {
        PROFILE_ID_COLUMN: profile_id,
        "buildingType": 0.0,
        "totalFloors": 2,
        "constructionPeriod": 2005.5,
        "wallsRenovationPeriod": 2005.5,
        "floorsRenovationPeriod": 2005.5,
        "roofRenovationPeriod": 2005.5,
        "windowsRenovationPeriod": 2005.5,
        "dwellingNumber": 1,
        "shSetpoint": 21.0,
        "floor_area": 50.0,
        "volume": 250.0,
        "envelope_area": 200.0,
        "window_area": 20.0,
        "surface_to_volume_ratio": 0.8,
        "thermal_mass_class": 8,
        HP_REF_CAPACITY_COLUMN: 5000.0,
        SH_DESIGN_CAPACITY_COLUMN: 4500.0,
        SH_VOLUME_COLUMN: 0.1,
        HP_REF_COP_COLUMN: 4.0,
        HP_MODEL_NAME_COLUMN: "aerotop_g07_14m",
        HP_SIZE_BINDING_COLUMN: "SH",
    }
    return values


def _write_entity_profile(root: Path, profile_id: int = 123) -> Path:
    profile_dir = root / f"egid={profile_id}"
    profile_dir.mkdir(parents=True)
    timestamps = pd.date_range("2023-01-01", periods=4, freq="15min")
    rows: list[dict[str, object]] = []
    for zone in ("building_total", "fl0_thz0", "fl1_thz0", "technical_room"):
        for index, timestamp in enumerate(timestamps):
            row: dict[str, object] = {
                "datetime": timestamp,
                "zone": zone,
                "Zone Thermostat Heating Setpoint Temperature": np.nan,
                "Zone Air Temperature": np.nan,
                ZONE_THERMAL_HEATING_POWER_COLUMN: np.nan,
                HEAT_PUMP_ELECTRIC_POWER_COLUMN: np.nan,
                HP_MODE_IS_DHW_COLUMN: np.nan,
                DHW_MIXED_WATER_COLUMN: np.nan,
                "hp_effective_mode": np.nan,
                INTERNAL_GAIN_COLUMN: np.nan,
                OCCUPANTS_PRESENT_COLUMN: np.nan,
                DISTURBANCE_COLUMNS[0]: np.nan,
                DISTURBANCE_COLUMNS[1]: np.nan,
                "Zone Ventilation Standard Density Volume Flow Rate": np.nan,
                "hp_force_off_requested": np.nan,
                "hp_force_off_effective": np.nan,
            }
            if zone == "building_total":
                row[DISTURBANCE_COLUMNS[0]] = 5.0 + index
                row[DISTURBANCE_COLUMNS[1]] = 100.0 * index
                row[HP_MODE_IS_DHW_COLUMN] = 0.0
                row["hp_effective_mode"] = 1.0
                row["hp_force_off_requested"] = 0.0
                row["hp_force_off_effective"] = 0.0
            elif zone == "fl0_thz0":
                row["Zone Thermostat Heating Setpoint Temperature"] = 21.0
                row["Zone Air Temperature"] = 20.0 + index
                row[ZONE_THERMAL_HEATING_POWER_COLUMN] = 500.0 + index
                row[INTERNAL_GAIN_COLUMN] = 0.05 + 0.01 * index
                row[OCCUPANTS_PRESENT_COLUMN] = 1.0
                row[DHW_MIXED_WATER_COLUMN] = 1.0 + index
                row["Zone Ventilation Standard Density Volume Flow Rate"] = 0.01
            elif zone == "fl1_thz0":
                row["Zone Thermostat Heating Setpoint Temperature"] = 22.0
                row["Zone Air Temperature"] = 30.0 + index
                row[ZONE_THERMAL_HEATING_POWER_COLUMN] = 1500.0 + index
                row[INTERNAL_GAIN_COLUMN] = 0.15 + 0.01 * index
                row[OCCUPANTS_PRESENT_COLUMN] = 2.0
                row[DHW_MIXED_WATER_COLUMN] = 2.0 * (1.0 + index)
                row["Zone Ventilation Standard Density Volume Flow Rate"] = 0.02
            else:
                row[HEAT_PUMP_ELECTRIC_POWER_COLUMN] = 200.0 + index
                row[HP_MODE_IS_DHW_COLUMN] = 0.0
                row["hp_effective_mode"] = 1.0
                row["hp_force_off_requested"] = 0.0
                row["hp_force_off_effective"] = 0.0
            rows.append(row)
    pd.DataFrame(rows).to_parquet(profile_dir / "timeseries.parquet", index=False)
    metadata = {"gardenCompatibility": {"values": _metadata_values(profile_id)}}
    (profile_dir / "simulation_metadata.json").write_text(json.dumps(metadata))
    return profile_dir


class EntityDatasetAdapterTests(unittest.TestCase):
    def test_synthesizes_entity_export_availability_at_season_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_dir = _write_entity_profile(root)
            path = profile_dir / "timeseries.parquet"
            frame = pd.read_parquet(path)
            replacement = pd.to_datetime(
                [
                    "2023-05-15 23:45",
                    "2023-05-16 00:00",
                    "2023-09-29 23:45",
                    "2023-09-30 00:00",
                ]
            )
            for zone in frame["zone"].unique():
                frame.loc[frame["zone"].eq(zone), "datetime"] = replacement.to_numpy()
            frame.to_parquet(path, index=False)

            loaded = read_dataset_frame(
                root,
                columns=[
                    DATETIME_COLUMN,
                    PROFILE_ID_COLUMN,
                    SPACE_HEATING_AVAILABILITY_COLUMN,
                ],
                profile_ids=[123],
            )
            np.testing.assert_array_equal(
                loaded[SPACE_HEATING_AVAILABILITY_COLUMN],
                np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )

    def test_prefers_exported_availability_over_calendar_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_dir = _write_entity_profile(root)
            path = profile_dir / "timeseries.parquet"
            frame = pd.read_parquet(path)
            frame[SPACE_HEATING_AVAILABILITY_COLUMN] = np.nan
            building_rows = frame["zone"].eq("building_total")
            frame.loc[building_rows, SPACE_HEATING_AVAILABILITY_COLUMN] = [
                0.0,
                1.0,
                0.0,
                1.0,
            ]
            frame.to_parquet(path, index=False)

            loaded = read_dataset_frame(
                root,
                columns=[
                    DATETIME_COLUMN,
                    PROFILE_ID_COLUMN,
                    SPACE_HEATING_AVAILABILITY_COLUMN,
                ],
                profile_ids=[123],
            )
            np.testing.assert_array_equal(
                loaded[SPACE_HEATING_AVAILABILITY_COLUMN],
                np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            )

    def test_maps_first_zone_and_technical_room_to_canonical_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_entity_profile(root)
            frame = read_dataset_frame(
                root,
                columns=closed_loop_required_columns(CLOSED_LOOP_METADATA_COLUMNS),
                profile_ids=[123],
            )
            self.assertEqual(frame.shape[0], 4)
            np.testing.assert_allclose(frame[TARGET_COLUMN], [20.0, 21.0, 22.0, 23.0])
            np.testing.assert_allclose(
                frame[HEAT_PUMP_ELECTRIC_POWER_COLUMN],
                [200.0, 201.0, 202.0, 203.0],
            )
            np.testing.assert_allclose(frame[INTERNAL_GAIN_COLUMN], [0.05, 0.06, 0.07, 0.08])
            np.testing.assert_allclose(frame[DHW_MIXED_WATER_COLUMN], [3.0, 6.0, 9.0, 12.0])
            self.assertEqual(read_profile_ids(root), (123,))

    def test_new_gain_channels_are_normalized_and_enter_both_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_entity_profile(root)
            config = SplitConfig(
                dataset_path=root,
                max_profiles=1,
                test_fraction=0.0,
                profile_selection="first",
            )

            q_splits = load_result_splits(config, "zone_thermal")
            q_profile = to_profiles(q_splits.train, "zone_thermal")[0]
            self.assertIn(INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN, q_splits.input_columns)
            self.assertIn(OCCUPANTS_PER_FLOOR_AREA_COLUMN, q_splits.input_columns)
            np.testing.assert_allclose(q_profile.inputs[0, -2:], [1.0, 0.02])

            closed_splits = load_closed_loop_result_splits(config)
            self.assertIn(DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN, closed_splits.input_columns)
            closed_profile = to_closed_loop_profiles(
                closed_splits.train,
                metadata_columns=closed_splits.metadata_columns,
            )[0]
            gain_index = closed_splits.input_columns.index(
                INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN
            )
            occupancy_index = closed_splits.input_columns.index(
                OCCUPANTS_PER_FLOOR_AREA_COLUMN
            )
            dhw_index = closed_splits.input_columns.index(
                DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN
            )
            self.assertAlmostEqual(float(closed_profile.inputs[0, gain_index]), 1.2)
            self.assertAlmostEqual(float(closed_profile.inputs[0, occupancy_index]), 0.02)
            self.assertAlmostEqual(float(closed_profile.inputs[0, dhw_index]), 0.24)

    def test_mixed_policy_datasets_keep_building_split_and_add_force_off_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary = root / "ordinary"
            force_off = root / "force_off"
            _write_entity_profile(ordinary)
            force_profile = _write_entity_profile(force_off)
            path = force_profile / "timeseries.parquet"
            frame = pd.read_parquet(path)
            technical = frame["zone"].eq("technical_room")
            frame.loc[technical, HP_FORCE_OFF_REQUESTED_COLUMN] = [0.0, 1.0, 1.0, 0.0]
            frame.to_parquet(path, index=False)

            splits = load_closed_loop_result_splits(
                SplitConfig(
                    dataset_path=ordinary,
                    additional_dataset_paths=(force_off,),
                    max_profiles=1,
                    test_fraction=0.0,
                    profile_selection="first",
                )
            )
            force_index = splits.input_columns.index(HP_FORCE_OFF_REQUESTED_COLUMN)
            profiles = to_closed_loop_profiles(
                splits.train,
                include_hp_force_off=True,
                metadata_columns=splits.metadata_columns,
            )

            self.assertEqual(splits.train_ids, (123,))
            self.assertEqual(len(profiles), 2)
            np.testing.assert_array_equal(profiles[0].inputs[:, force_index], 0.0)
            np.testing.assert_array_equal(
                profiles[1].inputs[:, force_index],
                np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            )

    def test_legacy_wide_dataset_without_gain_columns_uses_zero_forcing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.parquet"
            values = _metadata_values(123)
            timestamps = pd.date_range("2023-01-01", periods=4, freq="15min")
            frame = pd.DataFrame(
                {
                    DATETIME_COLUMN: timestamps,
                    PROFILE_ID_COLUMN: np.full(4, 123),
                    TARGET_COLUMN: np.arange(4, dtype=float) + 20.0,
                    ZONE_THERMAL_HEATING_POWER_COLUMN: np.full(4, 500.0),
                    DISTURBANCE_COLUMNS[0]: np.full(4, 5.0),
                    DISTURBANCE_COLUMNS[1]: np.zeros(4),
                    DISTURBANCE_COLUMNS[2]: np.zeros(4),
                    **{column: [values[column]] * 4 for column in METADATA_COLUMNS},
                }
            )
            frame.to_parquet(path, index=False)
            loaded = read_dataset_frame(
                path,
                columns=required_columns("zone_thermal"),
                profile_ids=[123],
            )
            self.assertTrue((loaded[INTERNAL_GAIN_COLUMN] == 0.0).all())
            self.assertTrue((loaded[OCCUPANTS_PRESENT_COLUMN] == 0.0).all())
            profile = to_profiles(loaded, "zone_thermal")[0]
            np.testing.assert_array_equal(profile.inputs[:, -2:], 0.0)


if __name__ == "__main__":
    unittest.main()
