from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from neural_building_emulator.columns import METADATA_COLUMNS
from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator.flexibility_event_study_kpis import EventStudyConfig
from neural_building_emulator_refactor.metadata_flexibility_lightgbm import (
    _kpi_row,
    _load_emulator_predictions,
    _resolve_targets,
    paired_error_comparison,
)


class MetadataFlexibilityLightGBMTests(unittest.TestCase):
    def test_target_resolution_deduplicates_equivalent_discrete_horizons(self) -> None:
        config, targets = _resolve_targets((0.5, 0.51, 1.0), 0.25)

        self.assertEqual(config.horizon_steps, (2, 4))
        self.assertEqual(len(targets), 4)
        self.assertEqual(targets[0].key, "up_flex_0p5h_wh_m2_k")
        self.assertEqual(targets[-1].key, "down_flex_1h_wh_m2_k")

    def test_simulated_kpi_extraction_recovers_known_setpoint_gain(self) -> None:
        block_steps = 12
        block_count = 12
        setpoint = np.repeat(
            np.asarray([20.0, 21.0] * (block_count // 2), dtype=np.float32),
            block_steps,
        )
        steps = len(setpoint)
        inputs = np.zeros((steps, 9), dtype=np.float32)
        inputs[:, 0] = setpoint
        pel = 5.0 + 2.0 * (setpoint - 20.0)
        targets = np.column_stack(
            [
                np.full(steps, 20.0, dtype=np.float32),
                np.zeros(steps, dtype=np.float32),
                pel,
            ]
        )
        profile = ClosedLoopProfile(
            profile_id=7,
            datetime=np.arange(steps),
            metadata=np.zeros(len(METADATA_COLUMNS), dtype=np.float32),
            inputs=inputs,
            targets=targets,
            initial_temperature=np.full((steps, 1), 20.0, dtype=np.float32),
        )
        config = EventStudyConfig(
            horizons_hours=(3.0,),
            horizon_steps=(block_steps,),
            dt_hours=0.25,
            min_setpoint_change=0.05,
            min_events=4,
            controls="none",
        )

        row = _kpi_row(profile, "train", config)

        self.assertAlmostEqual(row["up_flex_3h_wh_m2_k"], 6.0, places=5)
        self.assertAlmostEqual(row["down_flex_3h_wh_m2_k"], 6.0, places=5)

    def test_paired_bootstrap_detects_uniformly_better_predictions(self) -> None:
        target = np.linspace(0.0, 10.0, 50)
        comparison = paired_error_comparison(
            target,
            target,
            target + 1.0,
            seed=13,
            bootstrap_samples=500,
        )

        self.assertLess(comparison["rmse_difference"], 0.0)
        self.assertEqual(comparison["probability_lightgbm_lower_rmse"], 1.0)
        self.assertEqual(comparison["probability_lightgbm_lower_mae"], 1.0)

    def test_refactor_scorer_kpi_table_is_converted_to_comparison_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "flexibility_kpis.csv"
            pd.DataFrame(
                {
                    "signal": ["simulation", "emulation"],
                    "profile_id": [7, 7],
                    "horizon_hours": [3.0, 3.0],
                    "up_flex_wh_m2_k": [2.0, 1.5],
                    "down_flex_wh_m2_k": [3.0, 2.5],
                }
            ).to_csv(path, index=False)

            converted = _load_emulator_predictions(path)

        self.assertEqual(converted.loc[0, "profile_id"], 7)
        self.assertEqual(converted.loc[0, "sim_up_flex_wh_m2_k"], 2.0)
        self.assertEqual(converted.loc[0, "emu_down_flex_wh_m2_k"], 2.5)


if __name__ == "__main__":
    unittest.main()
