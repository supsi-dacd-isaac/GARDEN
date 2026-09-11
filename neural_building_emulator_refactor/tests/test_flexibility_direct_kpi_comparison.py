from __future__ import annotations

import unittest

import pandas as pd

from neural_building_emulator_refactor.flexibility_direct_kpi_comparison import (
    _direct_ratios,
)


class DirectFlexibilityKpiTests(unittest.TestCase):
    def test_ratio_of_sums_preserves_positive_downward_interpretation(self) -> None:
        events = pd.DataFrame(
            {
                "profile_id": [7, 7, 7, 7],
                "direction": ["up", "up", "down", "down"],
                "delta_tset_c": [1.0, 2.0, -1.0, -2.0],
                "delta_pel_w_m2": [0.5, 2.0, -0.2, -1.8],
            }
        )

        result = _direct_ratios(events, horizon_hours=3.0).iloc[0]

        self.assertAlmostEqual(result["direct_flex_wh_m2_k_up"], 2.5)
        self.assertAlmostEqual(result["direct_flex_wh_m2_k_down"], 2.0)
        self.assertEqual(result["event_count_up"], 2)
        self.assertEqual(result["event_count_down"], 2)


if __name__ == "__main__":
    unittest.main()
