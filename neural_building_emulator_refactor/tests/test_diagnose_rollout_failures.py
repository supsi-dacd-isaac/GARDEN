from __future__ import annotations

import unittest

import numpy as np

from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator_refactor.diagnose_rollout_failures import (
    _interval_means,
    context_rows,
    intervention_rows,
)
from neural_building_emulator_refactor.prediction import PredictionResult


class RolloutFailureDiagnosticTests(unittest.TestCase):
    def test_equal_length_event_windows_are_aligned(self) -> None:
        values = np.arange(8, dtype=float)
        before, after = _interval_means(values, np.array([3]), 2)
        self.assertAlmostEqual(float(before[0]), 1.5)
        self.assertAlmostEqual(float(after[0]), 3.5)
        particles = np.stack([values, values + 10])
        particle_before, particle_after = _interval_means(particles, np.array([3]), 2)
        np.testing.assert_allclose(particle_after - particle_before, [[2], [2]])

    def test_event_gain_and_scenario_coverage(self) -> None:
        inputs = np.zeros((16, 3), dtype=np.float32)
        inputs[:, 0] = 20.0
        inputs[5:, 0] = 21.0
        targets = np.zeros((16, 3), dtype=np.float32)
        targets[5:, 2] = 2.0
        profile = ClosedLoopProfile(
            profile_id=7,
            datetime=np.arange(16),
            metadata=np.zeros(1, dtype=np.float32),
            inputs=inputs,
            targets=targets,
            initial_temperature=np.full((16, 1), 20.0, dtype=np.float32),
        )
        prediction = PredictionResult(
            profile_id=7,
            datetime=np.arange(16),
            target=targets,
            mean=targets.copy(),
            scenarios=np.stack([targets, targets]),
        )
        row = next(item for item in intervention_rows("test", profile, prediction)
                   if item["signal"] == "Pel_SH [W/m2]" and item["horizon_hours"] == 0.5)
        self.assertEqual(row["events"], 1)
        self.assertAlmostEqual(row["truth_gain_per_k"], 2.0)
        self.assertAlmostEqual(row["pred_gain_per_k"], 2.0)
        self.assertAlmostEqual(row["response_rmse"], 0.0)
        self.assertAlmostEqual(row["response_coverage_90"], 1.0)

    def test_context_uses_named_input_columns(self) -> None:
        columns = (
            "space_heating_available",
            "FL0_THZ0, Zone Ventilation Standard Density Volume Flow Rate",
            "Environment, Site Global Horizontal Solar Radiation Rate per Area",
        )
        inputs = np.zeros((96, 3), dtype=np.float32)
        inputs[:48, 0] = 1.0
        inputs[:48, 2] = 200.0
        targets = np.zeros((96, 3), dtype=np.float32)
        scenarios = np.stack([targets.copy(), targets.copy()])
        scenarios[:, :48, 2] = -1.0
        profile = ClosedLoopProfile(7, np.arange(96), np.zeros(1), inputs, targets,
                                    np.zeros((96, 1)))
        prediction = PredictionResult(7, np.arange(96), targets, targets.copy(), scenarios)
        rows = list(context_rows("test", profile, prediction, columns))
        active_pel = next(row for row in rows if row["phase"] == "full year"
                          and row["condition"] == "SH available"
                          and row["signal"] == "Pel_SH [W/m2]")
        self.assertAlmostEqual(active_pel["negative_scenario_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
