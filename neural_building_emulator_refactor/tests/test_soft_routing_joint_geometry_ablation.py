from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.soft_routing_joint_geometry_ablation import (
    _coordinate_standardization,
    _variant_recipe,
    _variant_specs,
)


class SoftRoutingJointGeometryAblationTests(unittest.TestCase):
    def test_variants_compare_shared_and_soft_routes_under_the_same_loss(self) -> None:
        specs = _variant_specs(0.15)
        self.assertEqual(specs["baseline"].persistent_xi_routing, "shared")
        self.assertEqual(specs["baseline"].targeted_flex_energy_weight, 0.0)
        self.assertEqual(specs["joint_shared"].persistent_xi_routing, "shared")
        self.assertEqual(specs["joint_soft"].persistent_xi_routing, "soft_physical")
        self.assertEqual(specs["joint_shared"].targeted_flex_energy_weight, 0.15)
        self.assertEqual(specs["joint_soft"].targeted_flex_energy_weight, 0.15)

    def test_coordinate_standardization_contains_both_kpi_definitions(self) -> None:
        rows = []
        values = {
            1: {0.5: (2.0, 1.0), 2.0: (5.0, 2.0), 3.0: (7.0, 4.0)},
            2: {0.5: (4.0, 1.0), 2.0: (5.0, 2.0), 3.0: (8.0, 3.0)},
        }
        for profile_id, horizons in values.items():
            for horizon, (up, down) in horizons.items():
                rows.append(
                    {
                        "signal": "simulation",
                        "profile_id": profile_id,
                        "horizon_hours": horizon,
                        "valid": True,
                        "up_flex_wh_m2_k": up,
                        "down_flex_wh_m2_k": down,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(rows)
            frame.to_csv(root / "flexibility_kpis_f.csv", index=False)
            regression = frame.copy()
            regression["up_flex_wh_m2_k"] *= 2.0
            regression["down_flex_wh_m2_k"] *= 2.0
            regression.to_csv(root / "flexibility_kpis.csv", index=False)
            center, scale = _coordinate_standardization(root)

        self.assertEqual(len(center), 8)
        self.assertEqual(len(scale), 8)
        np.testing.assert_allclose(center[4:], 2.0 * np.asarray(center[:4]))
        np.testing.assert_allclose(scale[4:], 2.0 * np.asarray(scale[:4]))

    def test_variant_recipe_changes_only_joint_ablation_settings(self) -> None:
        base = {"probabilistic_structured": {"hidden_dim": 64}}
        spec = _variant_specs(0.15)["joint_soft"]
        candidate = _variant_recipe(
            base,
            spec,
            center=tuple(float(value) for value in range(8)),
            scale=(1.0,) * 8,
        )

        self.assertEqual(base, {"probabilistic_structured": {"hidden_dim": 64}})
        settings = candidate["probabilistic_structured"]
        self.assertEqual(settings["hidden_dim"], 64)
        self.assertEqual(settings["prob_persistent_xi_routing"], "soft_physical")
        self.assertEqual(settings["prob_targeted_flex_energy_weight"], 0.15)
        self.assertEqual(len(settings["prob_targeted_flex_coordinate_center"]), 8)
        self.assertEqual(len(settings["prob_targeted_flex_coordinate_scale"]), 8)


if __name__ == "__main__":
    unittest.main()
