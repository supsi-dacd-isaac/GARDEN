from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.routed_latent_targeted_loss_ablation import (
    VariantSpec,
    _coordinate_standardization,
    _variant_recipe,
    _variant_specs,
)


class RoutedLatentTargetedLossAblationTests(unittest.TestCase):
    def test_variant_specs_form_requested_two_by_two(self) -> None:
        specs = _variant_specs(0.25)
        self.assertEqual(specs["baseline"], VariantSpec("shared", 0.0))
        self.assertEqual(specs["architecture_only"], VariantSpec("physical", 0.0))
        self.assertEqual(specs["loss_only"], VariantSpec("shared", 0.25))
        self.assertEqual(specs["combined"], VariantSpec("physical", 0.25))

    def test_coordinate_standardization_uses_calibration_truth(self) -> None:
        rows = []
        values = {
            1: {0.5: (2.0, 1.0), 3.0: (7.0, 4.0)},
            2: {0.5: (4.0, 1.0), 3.0: (8.0, 3.0)},
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
            path = Path(directory)
            pd.DataFrame(rows).to_csv(path / "flexibility_kpis_f.csv", index=False)
            center, scale = _coordinate_standardization(path)

        np.testing.assert_allclose(center, (3.0, 2.0, 4.5))
        np.testing.assert_allclose(scale, (1.0, 1.0, 0.5))

    def test_variant_recipe_changes_only_ablation_settings(self) -> None:
        base = {"probabilistic_structured": {"hidden_dim": 64}}
        candidate = _variant_recipe(
            base,
            VariantSpec("physical", 0.25),
            center=(1.0, 2.0, 3.0),
            scale=(4.0, 5.0, 6.0),
        )

        self.assertEqual(base, {"probabilistic_structured": {"hidden_dim": 64}})
        settings = candidate["probabilistic_structured"]
        self.assertEqual(settings["hidden_dim"], 64)
        self.assertEqual(settings["prob_persistent_xi_routing"], "physical")
        self.assertEqual(settings["prob_targeted_flex_energy_weight"], 0.25)
        self.assertEqual(settings["prob_targeted_flex_coordinate_center"], [1.0, 2.0, 3.0])
        self.assertEqual(settings["prob_targeted_flex_coordinate_scale"], [4.0, 5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
