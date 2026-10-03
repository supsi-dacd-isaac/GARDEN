from __future__ import annotations

import unittest

import numpy as np

from neural_building_emulator_refactor.pel_dynamics_calibration import (
    PelDynamicsCalibrationAccumulator,
    _run_lengths,
    _shock_state,
)


class PelDynamicsCalibrationTests(unittest.TestCase):
    def test_run_lengths_respect_availability_gaps(self) -> None:
        active = np.asarray([0, 1, 1, 1, 1, 0], dtype=bool)
        available = np.asarray([1, 1, 1, 0, 1, 1], dtype=bool)
        np.testing.assert_array_equal(
            _run_lengths(active, available),
            np.asarray([2.0, 1.0]),
        )

    def test_shock_state_tracks_direction_for_three_hours(self) -> None:
        setpoint = np.full(20, 20.0)
        setpoint[2:] = 21.0
        age, direction = _shock_state(
            setpoint,
            threshold_c=0.05,
            dt_hours=0.25,
        )
        self.assertEqual(age[2], 0.0)
        self.assertEqual(direction[2], 1)
        self.assertEqual(direction[13], 1)
        self.assertEqual(direction[14], 0)

    def test_accumulator_produces_all_metric_families(self) -> None:
        steps = 64
        truth = np.zeros(steps)
        truth[4:8] = 5.0
        truth[20:27] = 7.0
        scenarios = np.stack(
            [
                values
                for shift in (0, 1, 2)
                for values in (
                    np.roll(truth, shift),
                    np.maximum(np.roll(truth, -shift), 0.0),
                )
            ]
        )
        setpoint = np.full(steps, 20.0)
        setpoint[16:32] = 21.0
        temperature = np.full(steps, 19.5)
        accumulator = PelDynamicsCalibrationAccumulator()
        accumulator.add(
            truth=truth,
            scenarios=scenarios,
            setpoint=setpoint,
            current_temperature=temperature,
            available=np.ones(steps, dtype=bool),
        )

        self.assertFalse(accumulator.reliability_frame().empty)
        self.assertFalse(accumulator.conditional_summary_frame().empty)
        self.assertFalse(accumulator.run_duration_frame().empty)
        self.assertFalse(accumulator.acf_frame().empty)
        self.assertFalse(accumulator.annual_energy_frame().empty)
        self.assertTrue(np.isfinite(accumulator.summary()["pel_start_brier"]))


if __name__ == "__main__":
    unittest.main()
