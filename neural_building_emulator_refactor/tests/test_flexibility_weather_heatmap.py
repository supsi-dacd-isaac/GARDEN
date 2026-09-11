from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from neural_building_emulator.columns import CLOSED_LOOP_INPUT_COLUMNS
from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator_refactor.flexibility_weather_heatmap import (
    WeatherHeatmapConfig,
    aggregate_building_bins,
    aggregate_temperature_setpoint_response,
    building_bin_means,
    extract_event_responses,
    temperature_setpoint_building_means,
)


class FlexibilityWeatherHeatmapTests(unittest.TestCase):
    def test_extracts_signed_normalized_up_and_down_responses(self) -> None:
        steps = 64
        inputs = np.zeros((steps, len(CLOSED_LOOP_INPUT_COLUMNS)), dtype=np.float32)
        inputs[:, 0] = 20.0
        inputs[16:40, 0] = 21.0
        inputs[:, 1] = 5.0
        inputs[:, 2] = 100.0
        inputs[:, CLOSED_LOOP_INPUT_COLUMNS.index("space_heating_available")] = 1.0
        targets = np.zeros((steps, 3), dtype=np.float32)
        targets[:, 2] = 2.0
        targets[16:20, 2] = 4.0
        targets[40:44, 2] = 1.0
        datetimes = pd.date_range("2021-01-01", periods=steps, freq="15min").to_numpy()
        profile = ClosedLoopProfile(
            profile_id=7,
            datetime=datetimes,
            metadata=np.zeros(1, dtype=np.float32),
            inputs=inputs,
            targets=targets,
            initial_temperature=np.full((steps, 1), 20.0, dtype=np.float32),
        )
        config = WeatherHeatmapConfig(
            horizons_hours=(1.0,),
            min_buildings_per_bin=1,
        )

        events, diagnostics = extract_event_responses(profile, config)

        self.assertEqual(len(events), 2)
        self.assertEqual(diagnostics[0]["valid_events"], 2)
        by_direction = events.set_index("direction")
        self.assertAlmostEqual(
            by_direction.loc["up", "normalized_response_w_m2_k"], 2.0
        )
        self.assertAlmostEqual(
            by_direction.loc["down", "normalized_response_w_m2_k"], 1.0
        )

    def test_aggregation_weights_buildings_not_events(self) -> None:
        events = pd.DataFrame(
            {
                "profile_id": [1, 1, 1, 2],
                "horizon_hours": [3.0] * 4,
                "direction": ["up"] * 4,
                "temperature_bin": [2] * 4,
                "irradiance_bin": [1] * 4,
                "normalized_response_w_m2_k": [10.0, 10.0, 10.0, 0.0],
            }
        )

        aggregate = aggregate_building_bins(building_bin_means(events))

        self.assertAlmostEqual(aggregate.loc[0, "mean_response_w_m2_k"], 5.0)
        self.assertEqual(aggregate.loc[0, "building_count"], 2)
        self.assertEqual(aggregate.loc[0, "event_count"], 4)

        setpoint_aggregate = aggregate_temperature_setpoint_response(
            temperature_setpoint_building_means(
                events.assign(
                    delta_tset_c=[1.0] * 4,
                    delta_pel_w_m2=[10.0, 10.0, 10.0, 0.0],
                )
            )
        )
        self.assertAlmostEqual(setpoint_aggregate.loc[0, "mean_delta_pel_w_m2"], 5.0)
        self.assertEqual(setpoint_aggregate.loc[0, "building_count"], 2)


if __name__ == "__main__":
    unittest.main()
