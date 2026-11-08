import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.plant_rollout_diagnostics import (
    distribution_metrics, empirical_crps, energy_scores, event_metrics,
)


class PlantRolloutDiagnosticsTests(unittest.TestCase):
    def test_crps_matches_pairwise_formula(self):
        samples = np.random.default_rng(13).normal(size=(7, 11))
        truth = np.linspace(-1, 1, 11)
        expected = np.abs(samples - truth).mean(axis=0) - .5 * np.abs(samples[:, None] - samples[None]).mean(axis=(0, 1))
        np.testing.assert_allclose(empirical_crps(samples, truth), expected)
        self.assertEqual(distribution_metrics(np.ones(5), 1.)["pit_mid"], .5)
        with self.assertRaises(ValueError):
            empirical_crps(np.array([np.nan]), 1.)

    def test_block_energy_units_and_untruncated_annual_energy(self):
        observed = np.ones(101) * 2
        predicted = np.ones((3, 101)) * 3
        scores = energy_scores(predicted, observed)
        self.assertEqual(scores["4h_crps"], 4.)
        self.assertEqual(scores["24h_crps"], 24.)
        self.assertEqual(scores["annual_crps"], 25.25)

    def test_event_raw_row_alignment_and_dhw_event_probability(self):
        paths = np.zeros((3, 20, 3))
        paths[:, :, 0] = 21.
        paths[:, 8, 0] = [20., 21., 19.]
        dhw = np.zeros((3, 20))
        dhw[:2, 8] = 1
        events = pd.DataFrame([dict(building=1, event=2, split="test", start=7, release=9, end=13,
                                    drop_c=1., dhw_first_hour=True, initially_heating_near_setpoint=True,
                                    minimum_hour_from_release=0.)])
        result = dict(predictions=paths, energy=np.ones((3, 20)) * 10,
                      dhw_active=dhw, dhw_debt=np.ones((3, 20)))
        rows = event_metrics(events, result, dict(limited=np.zeros((3, 20), dtype=bool)))
        self.assertAlmostEqual(rows.iloc[0].mean_particle_drop_c, 1.)
        self.assertAlmostEqual(rows.iloc[0].drop_crps, 2 / 9)
        self.assertAlmostEqual(rows.iloc[0].dhw_first_hour_probability, 2 / 3)
        self.assertAlmostEqual(rows.iloc[0].dhw_first_hour_brier, 1 / 9)
        self.assertEqual(rows.iloc[0].drop_coverage90, 1.)


if __name__ == "__main__":
    unittest.main()
