import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.buffer_recharge_diagnostics import (
    DHW, PEL, QROOM, REQUEST, SETPOINT, TIN, VENT, drop_features, event_rows, intervals,
)


class BufferRechargeDiagnosticsTests(unittest.TestCase):
    def test_intervals_include_boundaries(self):
        self.assertEqual(intervals(np.array([1, 1, 0, 1])), [(0, 2), (3, 4)])
        self.assertEqual(intervals(np.zeros(4)), [])

    def test_particle_drops_not_drop_of_average(self):
        paths = np.full((2, 16), 21.)
        paths[0, 6], paths[1, 7] = 19., 19.
        self.assertEqual(drop_features(paths.mean(0), 4, 8, 12)["drop_c"], 1.)
        np.testing.assert_array_equal(drop_features(paths, 4, 8, 12)["drop_c"], [2., 2.])

    def test_events_keep_successes_and_do_not_call_idle_dhw_recovery(self):
        frame = pd.DataFrame(index=pd.date_range("2023-01-01", periods=160, freq="15min"))
        for name, value in ((TIN, 21.), (QROOM, 20.), (SETPOINT, 21.), (VENT, 0.),
                            (REQUEST, 0.), (DHW, 0.), (PEL, 1000.), ("tout", 5.), ("dhw_l", 1.)):
            frame[name] = value
        frame.iloc[100:108, frame.columns.get_loc(REQUEST)] = 1
        frame.iloc[108:110, frame.columns.get_loc(DHW)] = 1
        frame.iloc[108:110, frame.columns.get_loc(PEL)] = 0
        rows = event_rows(frame, 1, "test")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["drop_c"], 0)
        self.assertFalse(rows[0]["dhw_first_hour"])
        self.assertEqual(rows[0]["sh_restart_delay_h"], .5)

    def test_post_release_drop_is_separate_from_lockout_drop(self):
        temperature = np.full(16, 21.)
        temperature[6:9] = [20.8, 20.6, 19.]
        result = drop_features(temperature, 4, 8, 12)
        self.assertAlmostEqual(result["lockout_drop_c"], .4)
        self.assertAlmostEqual(result["post_release_extra_drop_c"], 1.6)
        self.assertAlmostEqual(result["drop_c"], 2.)
        self.assertEqual(result["minimum_hour_from_release"], 0.)

    def test_paired_comparison_rejects_missing_events(self):
        from neural_building_emulator_refactor.buffer_recharge_report import paired_bootstrap
        raw = pd.DataFrame({"building": [1, 1], "event": [1, 2]})
        candidate = pd.DataFrame({"building": [1], "event": [1]})
        with self.assertRaisesRegex(ValueError, "identical events"):
            paired_bootstrap(raw, candidate)

    def test_paired_bootstrap_resamples_buildings_with_equal_weight(self):
        from neural_building_emulator_refactor.buffer_recharge_report import paired_bootstrap
        raw = pd.DataFrame({"building": [1, 1, 1, 1, 2, 2], "event": [1, 2, 3, 4, 1, 2],
                            "dhw_first_hour": [True, False, True, False, True, False],
                            "drop_crps": [0.] * 6, "brier_gt_0.5": [0.] * 6,
                            "drop_coverage90": [False] * 6})
        candidate = raw.copy()
        candidate["drop_crps"] = [1.] * 4 + [3.] * 2
        results = paired_bootstrap(raw, candidate)
        overall = next(row for row in results if row["regime"] == "all" and row["metric"] == "drop_crps")
        self.assertEqual(overall["buildings"], 2)
        self.assertEqual(overall["difference"], 2.)


if __name__ == "__main__":
    unittest.main()
