import copy
import unittest

import pandas as pd

from neural_building_emulator_refactor.generic_response_experiment import response_recipe
from neural_building_emulator_refactor.generic_response_report import (
    controlled_config_changes, native_scores, verify_matched_events,
)
from neural_building_emulator_refactor.train import LEGACY_OPTION_FIELDS


class GenericResponseTests(unittest.TestCase):
    def test_recipe_only_changes_objective(self):
        original = {"probabilistic_structured": {
            "hp_energy_cap_wh_m2": 270.0, "prob_latent_dim": 4,
            "prob_flex_kpi_crps_weight": 1.5, "hp_buffer_capacity_mode": "global",
        }, "optimizer": {"epochs": 10}, "evaluation": {"num_scenarios": 100}}
        saved = copy.deepcopy(original)
        candidate = response_recipe(original)
        self.assertEqual(original, saved)
        self.assertEqual(candidate["optimizer"], original["optimizer"])
        self.assertEqual(candidate["evaluation"], original["evaluation"])
        for key, value in original["probabilistic_structured"].items():
            self.assertEqual(candidate["probabilistic_structured"][key], value)
        added = set(candidate["probabilistic_structured"]) - set(original["probabilistic_structured"])
        self.assertTrue(all(key.startswith(("prob_event_energy_", "prob_block_energy_")) for key in added))

    def test_new_option_is_forwarded(self):
        self.assertEqual(LEGACY_OPTION_FIELDS["prob_event_energy_include_force_off"],
                         "prob_event_energy_include_force_off")

    def test_matching_rejects_missing_or_conflicting_events(self):
        frame = pd.DataFrame({"building": [1, 1], "event": [2, 3], "observed_drop_c": [0., 1.],
                              "initially_heating_near_setpoint": [True, True],
                              "dhw_first_hour": [False, True]})
        verify_matched_events(frame, frame.copy())
        with self.assertRaises(ValueError):
            verify_matched_events(frame, frame.iloc[:1])
        changed = frame.copy()
        changed.loc[0, "observed_drop_c"] = 2.
        with self.assertRaises(AssertionError):
            verify_matched_events(frame, changed)

    def test_report_preserves_native_probabilities(self):
        frame = pd.DataFrame({
            "method": ["baseline"] * 3, "kpi_kind": ["setpoint_f"] * 3,
            "profile_id": [1] * 3, "horizon_hours": [0.5] * 3,
            "direction": ["up"] * 3, "prediction": [-1., 0., 1.], "truth": [0.] * 3,
        })
        summary, _, _, _, pit = native_scores(frame)
        self.assertAlmostEqual(summary.crps.iloc[0], 2 / 9)
        self.assertEqual(pit.pit.iloc[0], 0.5)
        self.assertEqual(summary.minimum_particles.iloc[0], 3)
        self.assertEqual(summary.coverage_90.iloc[0], 1.)

    def test_report_rejects_architecture_or_split_changes(self):
        base = {"train_ids": [1], "test_ids": [2], "metadata_columns": ["area"],
                "input_columns": ["setpoint"], "train_config": {"prob_latent_dim": 4}}
        candidate = copy.deepcopy(base)
        candidate["train_config"]["prob_event_energy_include_force_off"] = True
        self.assertEqual(len(controlled_config_changes(base, candidate)), 1)
        candidate["train_config"]["prob_latent_dim"] = 5
        with self.assertRaises(ValueError):
            controlled_config_changes(base, candidate)
        candidate = copy.deepcopy(base)
        candidate["test_ids"] = [3]
        with self.assertRaises(ValueError):
            controlled_config_changes(base, candidate)


if __name__ == "__main__":
    unittest.main()
