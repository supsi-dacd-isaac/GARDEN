"""Optional plant export channels without training or simulator dependencies."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from neural_building_emulator.columns import (
    BUILDING_THERMAL_HEATING_POWER_COLUMN,
    CLOSED_LOOP_METADATA_COLUMNS,
    HEAT_PUMP_ELECTRIC_POWER_COLUMN,
    HP_MODE_IS_DHW_COLUMN,
    PLANT_METADATA_COLUMNS,
    PLANT_TARGET_SOURCE_COLUMNS,
    SH_STORAGE_CHARGE_COLUMN,
    closed_loop_required_columns,
)
from neural_building_emulator.data import (
    WindowConfig, make_closed_loop_windows, read_dataset_frame, to_closed_loop_profiles,
)
from neural_building_emulator.scaling import (
    fit_window_scalers, transform_closed_loop_profiles, transform_closed_loop_windows,
)
from neural_building_emulator.test_dataset_adapter import _write_entity_profile


class DatasetPlantChannelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.folder = _write_entity_profile(self.root)
        self.parquet = self.folder / "timeseries.parquet"
        raw = pd.read_parquet(self.parquet)
        raw[SH_STORAGE_CHARGE_COLUMN] = np.where(raw.zone == "technical_room", 3., np.nan)
        raw.to_parquet(self.parquet, index=False)

    def frame(self):
        return read_dataset_frame(
            self.root,
            columns=closed_loop_required_columns(CLOSED_LOOP_METADATA_COLUMNS)
            + PLANT_TARGET_SOURCE_COLUMNS,
            profile_ids=[123],
        )

    def test_energy_boundary_normalization_and_interval_alignment(self):
        frame = self.frame()
        np.testing.assert_allclose(
            frame[BUILDING_THERMAL_HEATING_POWER_COLUMN], [2000, 2002, 2004, 2006],
        )
        profile = to_closed_loop_profiles(frame, metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)[0]
        legacy = to_closed_loop_profiles(
            frame.drop(columns=PLANT_TARGET_SOURCE_COLUMNS),
            metadata_columns=CLOSED_LOOP_METADATA_COLUMNS,
        )[0]
        self.assertEqual(profile.plant_targets.shape, (3, 3))
        np.testing.assert_allclose(
            profile.plant_targets, [[30, 20.02, 0], [30, 20.04, 0], [30, 20.06, 0]],
            rtol=1e-6,
        )
        np.testing.assert_array_equal(profile.inputs, legacy.inputs)
        np.testing.assert_array_equal(profile.targets, legacy.targets)
        self.assertIsNone(legacy.plant_targets)

    def test_plant_channels_remain_in_physical_units_when_scaling(self):
        profile = to_closed_loop_profiles(self.frame(), metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)[0]
        windows = make_closed_loop_windows([profile], WindowConfig(sequence_length=2, stride=1))
        scalers = fit_window_scalers(windows)
        np.testing.assert_array_equal(
            transform_closed_loop_windows(windows, scalers).plant_targets, windows.plant_targets,
        )
        np.testing.assert_array_equal(
            transform_closed_loop_profiles([profile], scalers)[0].plant_targets,
            profile.plant_targets,
        )
        with self.assertRaisesRegex(ValueError, "Cannot mix"):
            make_closed_loop_windows(
                [profile, replace(profile, plant_targets=None)], WindowConfig(2, 1),
            )

    def test_whole_building_heat_requires_all_floor_observations(self):
        original = pd.read_parquet(self.parquet)
        floor_rows = original.index[original.zone == "fl1_thz0"]
        for removed in (floor_rows[:1], floor_rows):
            with self.subTest(removed=len(removed)):
                original.drop(removed).to_parquet(self.parquet, index=False)
                with self.assertRaises(ValueError):
                    self.frame()

    def test_partial_or_nonfinite_plant_targets_fail(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "requires both"):
            to_closed_loop_profiles(frame.drop(columns=[SH_STORAGE_CHARGE_COLUMN]))
        frame.loc[1, SH_STORAGE_CHARGE_COLUMN] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite plant targets"):
            to_closed_loop_profiles(frame)

    def test_latched_dhw_mode_without_electricity_is_not_active_dhw(self):
        frame = self.frame()
        frame[HP_MODE_IS_DHW_COLUMN] = [1., 1., 1., 0.]
        frame[HEAT_PUMP_ELECTRIC_POWER_COLUMN] = [0., 0., 200., 300.]
        profile = to_closed_loop_profiles(frame)[0]
        np.testing.assert_array_equal(profile.plant_targets[:, 2], [0., 1., 0.])

    def test_optional_plant_metadata_uses_whole_building_area(self):
        path = self.folder / "simulation_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["gardenCompatibility"]["values"].update(dhwVolume_m3=.2, DHW_recovery_W=1000.)
        metadata["sizing"] = {"sh_supply_C": 45.}
        path.write_text(json.dumps(metadata))
        columns = [*CLOSED_LOOP_METADATA_COLUMNS, *PLANT_METADATA_COLUMNS]
        frame = read_dataset_frame(
            self.root, columns=closed_loop_required_columns(columns), profile_ids=[123],
        )
        profile = to_closed_loop_profiles(frame, metadata_columns=columns)[0]
        np.testing.assert_allclose(profile.metadata[-3:], [.002, 10., 45.])


if __name__ == "__main__":
    unittest.main()
