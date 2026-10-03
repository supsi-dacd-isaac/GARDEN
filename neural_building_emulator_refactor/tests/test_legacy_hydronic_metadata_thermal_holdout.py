from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.legacy_hydronic_metadata_thermal_holdout import (
    _contiguous_events,
    _event_summary,
    _metadata_capacity_model,
)


class LegacyHydronicMetadataThermalHoldoutTests(unittest.TestCase):
    def test_metadata_capacity_starts_at_requested_value_and_is_bounded(self) -> None:
        model = _metadata_capacity_model(
            None,  # type: ignore[arg-type]
            metadata_dim=3,
            hidden_dim=4,
            depth=2,
            minimum_hours=2.0,
            maximum_hours=16.0,
            initial_hours=8.0,
            seed=13,
        )

        initial = float(model.capacity_hours(np.zeros(3, dtype=np.float32)))

        self.assertAlmostEqual(initial, 8.0, places=5)
        self.assertGreater(initial, model.minimum_hours)
        self.assertLess(initial, model.maximum_hours)

    def test_contiguous_events_returns_exclusive_stops(self) -> None:
        mask = np.asarray([False, True, True, False, True, False])

        self.assertEqual(_contiguous_events(mask), [(1, 3), (4, 5)])

    def test_event_summary_reports_material_temperature_drops(self) -> None:
        rows = []
        for observed, predicted, pre_pel in (
            (-0.6, -0.4, 2.0),
            (-0.2, -0.1, 0.0),
        ):
            row = {"observed_pel_pre_mean": pre_pel}
            for response in ("event_delta", "rebound"):
                for signal in ("tin", "qroom", "pel"):
                    row[f"observed_{signal}_{response}"] = (
                        observed if signal == "tin" else 1.0
                    )
                    row[f"predicted_{signal}_{response}"] = (
                        predicted if signal == "tin" else 0.8
                    )
            rows.append(row)

        summary = _event_summary(pd.DataFrame(rows))
        material = summary[
            summary.selection.eq("tin_drop_gt_0.5_c")
            & summary.signal.eq("tin")
            & summary.response.eq("event_delta")
        ].iloc[0]

        self.assertEqual(material.event_count, 1)
        self.assertAlmostEqual(material.bias, 0.2)


if __name__ == "__main__":
    unittest.main()
