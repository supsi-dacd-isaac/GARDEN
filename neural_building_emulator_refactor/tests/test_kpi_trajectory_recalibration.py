from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.kpi_trajectory_recalibration import (
    density_ratio_log_weights,
    ecc_reorder,
    fit_emos,
    stabilized_density_ratio_weights,
    weighted_crps,
)
from neural_building_emulator_refactor.probabilistic_calibration import ensemble_crps


class KpiTrajectoryRecalibrationTests(unittest.TestCase):
    def test_weighted_crps_matches_empirical_crps_for_uniform_weights(self) -> None:
        values = np.asarray([0.0, 1.0, 1.0, 3.0])
        weights = np.full(len(values), 1.0 / len(values))
        self.assertAlmostEqual(
            weighted_crps(values, weights, 1.5),
            ensemble_crps(values, 1.5),
        )

    def test_ecc_preserves_calibrated_marginal_and_raw_ranks(self) -> None:
        common = {
            "method": "emulator",
            "kpi_kind": "f",
            "horizon_hours": 0.5,
            "direction": "up",
            "profile_id": 1,
            "truth": 2.0,
        }
        raw = pd.DataFrame(
            [
                {**common, "sample_index": 0, "prediction": 3.0},
                {**common, "sample_index": 1, "prediction": 1.0},
                {**common, "sample_index": 2, "prediction": 2.0},
            ]
        )
        calibrated = pd.DataFrame(
            [
                {**common, "sample_index": 0, "prediction": 10.0},
                {**common, "sample_index": 1, "prediction": 20.0},
                {**common, "sample_index": 2, "prediction": 30.0},
            ]
        )
        reordered = ecc_reorder(raw, calibrated).sort_values("sample_index")
        np.testing.assert_allclose(
            reordered["prediction"].to_numpy(),
            np.asarray([30.0, 10.0, 20.0]),
        )
        np.testing.assert_allclose(
            np.sort(reordered["prediction"]),
            np.sort(calibrated["prediction"]),
        )

    def test_density_ratio_tempering_respects_ess_floor(self) -> None:
        log_ratio = np.asarray([20.0, 0.0, -5.0, -10.0, -20.0])
        weights, effective_alpha = stabilized_density_ratio_weights(
            log_ratio,
            alpha=1.0,
            minimum_ess=3.0,
        )
        self.assertLess(effective_alpha, 1.0)
        self.assertGreaterEqual(1.0 / np.sum(weights**2), 3.0 - 1e-8)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_emos_fit_and_density_ratio_are_finite(self) -> None:
        rng = np.random.default_rng(7)
        raw_location = rng.normal(size=(14, 2))
        particles = raw_location[:, None, :] + rng.normal(
            scale=0.35,
            size=(14, 30, 2),
        )
        truth = 0.25 + 1.1 * raw_location + rng.normal(scale=0.15, size=(14, 2))
        model = fit_emos(particles, truth)
        ratios = density_ratio_log_weights(particles[0], model)
        self.assertEqual(ratios.shape, (30,))
        self.assertTrue(np.all(np.isfinite(ratios)))
        self.assertTrue(np.all(model.variance_intercept > 0.0))
        self.assertTrue(np.all(model.ensemble_variance_coefficient > 0.0))


if __name__ == "__main__":
    unittest.main()
