from __future__ import annotations

import json
import unittest

import pandas as pd

from neural_building_emulator_refactor.hp_flex_loss_ablation import (
    DEFAULT_CONFIG,
    VARIANT_ORDER,
    _rank_components,
    _variant_recipe,
    parse_args,
)


class HpFlexLossAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = json.loads(DEFAULT_CONFIG.read_text())

    def test_expected_event_variant(self) -> None:
        recipe = _variant_recipe(self.base, "expected_event")
        settings = recipe["probabilistic_structured"]
        self.assertEqual(settings["prob_hp_training_mode"], "expected")
        self.assertEqual(settings["prob_event_energy_weight"], 0.5)
        self.assertEqual(settings["prob_flex_kpi_crps_weight"], 0.0)

    def test_hard_event_changes_only_sampling_mode(self) -> None:
        expected = _variant_recipe(self.base, "expected_event")
        hard = _variant_recipe(self.base, "hard_event")
        expected_settings = expected["probabilistic_structured"]
        hard_settings = hard["probabilistic_structured"]
        differing = {
            key
            for key in set(expected_settings).union(hard_settings)
            if expected_settings.get(key) != hard_settings.get(key)
        }
        self.assertEqual(differing, {"prob_hp_training_mode"})
        self.assertEqual(
            hard_settings["prob_hp_training_mode"], "hard_expected_grad"
        )

    def test_hard_direct_f_replaces_event_score(self) -> None:
        recipe = _variant_recipe(self.base, "hard_direct_f")
        settings = recipe["probabilistic_structured"]
        self.assertEqual(
            settings["prob_hp_training_mode"], "hard_expected_grad"
        )
        self.assertEqual(settings["prob_event_energy_weight"], 0.0)
        self.assertEqual(settings["prob_flex_kpi_crps_weight"], 0.1)
        self.assertEqual(settings["prob_flex_kpi_crps_estimator"], "direct_ratio")
        self.assertEqual(settings["prob_flex_kpi_crps_min_events"], 8)
        self.assertEqual(
            settings["prob_flex_kpi_crps_horizons_hours"],
            [0.5, 1.0, 2.0, 3.0],
        )

    def test_hard_event_weight_two_changes_only_event_weight(self) -> None:
        baseline = _variant_recipe(self.base, "hard_event")[
            "probabilistic_structured"
        ]
        stronger = _variant_recipe(self.base, "hard_event_w2")[
            "probabilistic_structured"
        ]
        differing = {
            key
            for key in set(baseline).union(stronger)
            if baseline.get(key) != stronger.get(key)
        }
        self.assertEqual(differing, {"prob_event_energy_weight"})
        self.assertEqual(stronger["prob_event_energy_weight"], 2.0)

    def test_combined_variant_only_adds_direct_f_to_weight_two(self) -> None:
        event_only = _variant_recipe(self.base, "hard_event_w2")[
            "probabilistic_structured"
        ]
        combined = _variant_recipe(self.base, "hard_event_w2_direct_f")[
            "probabilistic_structured"
        ]
        differing = {
            key
            for key in set(event_only).union(combined)
            if event_only.get(key) != combined.get(key)
        }
        self.assertEqual(differing, {"prob_flex_kpi_crps_weight"})
        self.assertEqual(combined["prob_event_energy_weight"], 2.0)
        self.assertEqual(combined["prob_flex_kpi_crps_weight"], 2.5)
        self.assertEqual(
            combined["prob_flex_kpi_crps_estimator"], "direct_ratio"
        )

    def test_grouped_variant_only_adds_shared_seasonal_windows(self) -> None:
        local = _variant_recipe(self.base, "hard_event_w2_direct_f")[
            "probabilistic_structured"
        ]
        grouped = _variant_recipe(
            self.base,
            "hard_event_w2_direct_f_grouped",
        )["probabilistic_structured"]
        differing = {
            key
            for key in set(local).union(grouped)
            if local.get(key) != grouped.get(key)
        }
        self.assertEqual(
            differing,
            {"prob_flex_kpi_crps_windows_per_profile"},
        )
        self.assertEqual(grouped["prob_flex_kpi_crps_windows_per_profile"], 4)

    def test_aux_variant_keeps_main_batches_independent(self) -> None:
        local = _variant_recipe(self.base, "hard_event_w2_direct_f")[
            "probabilistic_structured"
        ]
        auxiliary = _variant_recipe(
            self.base,
            "hard_event_w2_direct_f_aux",
        )["probabilistic_structured"]
        differing = {
            key
            for key in set(local).union(auxiliary)
            if local.get(key) != auxiliary.get(key)
        }
        self.assertEqual(
            differing,
            {
                "prob_flex_kpi_crps_windows_per_profile",
                "prob_flex_kpi_crps_aux_every_steps",
            },
        )
        self.assertEqual(auxiliary["prob_flex_kpi_crps_windows_per_profile"], 4)
        self.assertEqual(auxiliary["prob_flex_kpi_crps_aux_every_steps"], 4)
        self.assertEqual(auxiliary["prob_flex_kpi_crps_aux_buildings"], 4)

    def test_all_variants_share_stability_settings(self) -> None:
        for name in VARIANT_ORDER:
            settings = _variant_recipe(self.base, name)[
                "probabilistic_structured"
            ]
            self.assertEqual(
                settings["contracting_temperature_feedback_gradient_scale"],
                0.05,
            )
            self.assertEqual(
                settings["contracting_metadata_fixed_state_update"], "linear"
            )
            self.assertEqual(settings["prob_hp_scenario_mode"], "bernoulli")

    def test_default_cli_is_focused_preliminary_study(self) -> None:
        args = parse_args([])
        self.assertEqual(args.max_profiles, 200)
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.num_scenarios, 50)
        self.assertEqual(tuple(args.variants), VARIANT_ORDER)

    def test_automatic_assessment_balances_domains(self) -> None:
        ordered_values = [float(index + 1) for index in range(len(VARIANT_ORDER))]
        annual = pd.DataFrame(
            {
                "variant": list(VARIANT_ORDER),
                "total_nrmse": ordered_values,
            }
        )
        traces = pd.DataFrame(
            {
                "variant": list(VARIANT_ORDER),
                "signal": ["Tin"] * len(VARIANT_ORDER),
                "rank_calibration_mae": [value / 10.0 for value in ordered_values],
                "mean_crps": [value / 10.0 for value in ordered_values],
            }
        )
        empty = pd.DataFrame()
        components, assessment = _rank_components(
            annual,
            traces,
            empty,
            empty,
            empty,
        )
        self.assertFalse(components.empty)
        self.assertEqual(assessment.iloc[0]["variant"], "expected_event")
        self.assertEqual(assessment.iloc[0]["balanced_mean_rank"], 1.0)


if __name__ == "__main__":
    unittest.main()
