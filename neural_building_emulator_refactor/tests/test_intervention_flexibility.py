from __future__ import annotations

import unittest

import numpy as np

from neural_building_emulator.columns import (
    DISTURBANCE_INPUT_COLUMNS,
    HP_FORCE_OFF_REQUESTED_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
)
from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator_refactor.intervention_flexibility import (
    estimate_force_off_flexibility,
)


class InterventionFlexibilityTests(unittest.TestCase):
    def test_force_off_direct_and_regression_use_positive_shed_convention(self) -> None:
        force_off = np.tile(
            np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
            20,
        )
        power = np.where(force_off > 0.5, 5.0, 10.0).astype(np.float32)
        steps = len(force_off)
        inputs = np.column_stack(
            [
                np.full(steps, 21.0, dtype=np.float32),
                np.linspace(-5.0, 5.0, steps, dtype=np.float32),
                force_off,
            ]
        )
        trace = np.column_stack(
            [
                np.full(steps, 20.0, dtype=np.float32),
                np.full(steps, 15.0, dtype=np.float32),
                power,
            ]
        )
        profile = ClosedLoopProfile(
            profile_id=1,
            datetime=np.arange(steps),
            metadata=np.asarray([0.0], dtype=np.float32),
            inputs=inputs,
            initial_temperature=np.asarray([20.0], dtype=np.float32),
            targets=trace,
        )
        columns = (
            SETPOINT_TIMESERIES_COLUMN,
            DISTURBANCE_INPUT_COLUMNS[0],
            HP_FORCE_OFF_REQUESTED_COLUMN,
        )
        direct = estimate_force_off_flexibility(
            profile,
            trace,
            force_off,
            np.ones(steps, dtype=bool),
            input_columns=columns,
            horizon_hours=0.5,
            horizon_steps=2,
            estimator="direct_ratio",
            min_events=5,
        )
        regression = estimate_force_off_flexibility(
            profile,
            trace,
            force_off,
            np.ones(steps, dtype=bool),
            input_columns=columns,
            horizon_hours=0.5,
            horizon_steps=2,
            estimator="regression",
            min_events=5,
        )
        self.assertTrue(direct.valid)
        self.assertTrue(regression.valid)
        self.assertAlmostEqual(direct.response_gain_w_m2_per_unit, 5.0, places=5)
        self.assertAlmostEqual(direct.flexibility_wh_m2_per_unit, 2.5, places=5)
        self.assertAlmostEqual(regression.response_gain_w_m2_per_unit, 5.0, places=4)

    def test_short_force_off_pulses_are_not_counted_for_long_horizon(self) -> None:
        force_off = np.asarray([0, 0, 1, 0, 0, 0], dtype=np.float32)
        trace = np.zeros((len(force_off), 3), dtype=np.float32)
        profile = ClosedLoopProfile(
            profile_id=1,
            datetime=np.arange(len(force_off)),
            metadata=np.asarray([0.0], dtype=np.float32),
            inputs=np.column_stack(
                [np.full(len(force_off), 21.0), force_off]
            ).astype(np.float32),
            initial_temperature=np.asarray([20.0], dtype=np.float32),
            targets=trace,
        )
        result = estimate_force_off_flexibility(
            profile,
            trace,
            force_off,
            np.ones(len(force_off), dtype=bool),
            input_columns=(SETPOINT_TIMESERIES_COLUMN, HP_FORCE_OFF_REQUESTED_COLUMN),
            horizon_hours=0.5,
            horizon_steps=2,
            estimator="direct_ratio",
            min_events=1,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.event_count, 0)


if __name__ == "__main__":
    unittest.main()
