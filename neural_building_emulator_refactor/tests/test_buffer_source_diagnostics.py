import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.buffer_source_diagnostics import (
    block_source_error, charging_checks, conversion_window_checks, fit_source_probe, hourly_source_rows, source_features,
)


class BufferSourceDiagnosticsTests(unittest.TestCase):
    def test_checks_flag_heat_without_electricity_and_during_lockout(self):
        checks = charging_checks([0, 3, 5, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
        self.assertEqual(checks["charge_during_off_wh_m2"], 1.25)
        self.assertEqual(checks["charge_during_dhw_wh_m2"], .5)
        self.assertEqual(checks["charge_without_sh_electric_wh_m2"], 1.75)

    def test_checks_reject_missing_or_misaligned_signals(self):
        with self.assertRaises(ValueError):
            charging_checks([np.nan], [1], [0], [0])
        with self.assertRaises(ValueError):
            charging_checks([1, 2], [1], [0, 0], [0, 0])

    def test_features_do_not_use_simulator_state(self):
        metadata = {"sizing": {"sh_supply_C": 50}, "gardenCompatibility": {"values": {"hp_ref_cop": 4}}}
        x, names = source_features(np.array([0, 10]), metadata, "weather_design")
        self.assertEqual(x.shape, (2, len(names)))
        np.testing.assert_allclose(x[:, :2], [[1, 0], [1, 1]])

    def test_hourly_aggregation_uses_power_weighted_features(self):
        signals = pd.DataFrame({"tout": [0, 10, 0, 10], "pel_sh_w_m2": [2, 6, 2, 6],
                                "charge_w_m2": [6, 24, 6, 24]})
        design, target, _, _ = hourly_source_rows(signals, {}, "weather")
        np.testing.assert_allclose(design, [[4, 3]])
        np.testing.assert_allclose(target, [15])

    def test_block_conversion_avoids_instantaneous_transfer_mismatch(self):
        checks = conversion_window_checks(np.array([0, 40, 0, 40]), np.array([10, 2, 10, 2]), 1)
        self.assertEqual(checks["outside_1_8_fraction"], 1)
        averaged = conversion_window_checks(np.array([0, 40, 0, 40]), np.array([10, 2, 10, 2]), 4)
        self.assertEqual(averaged["outside_1_8_fraction"], 0)

    def test_probe_recovers_source_conversion_and_clips_effective_cop(self):
        design = np.array([[2, 0], [3, 3], [4, 8]], dtype=float)
        truth = design @ np.array([3, .5])
        record = (design, truth, 1., ["intercept", "outdoor"])
        coefficients = fit_source_probe([record], 0.)
        np.testing.assert_allclose(coefficients, [3, .5], atol=1e-12)
        self.assertLess(block_source_error(record, coefficients)["relative_rmse"], 1e-12)
        capped = (design, design[:, 0] * 8, 1., [])
        self.assertEqual(block_source_error(capped, np.array([20., 0]))["relative_rmse"], 0)


if __name__ == "__main__":
    unittest.main()
