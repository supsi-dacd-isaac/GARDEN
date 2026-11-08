import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.plant_accounting_report import (
    comparison_conclusions, paired_crps_bootstrap, recovery_event_diagnostics,
    storage_support_diagnostics,
)
from neural_building_emulator_refactor.plant_accounting_experiment import recipe_for_variant


class PlantAccountingReportTests(unittest.TestCase):
    def test_storage_support_uses_whole_building_capacity_and_observed_subsets(self):
        annual = pd.DataFrame(dict(building=[1, 2], capacity_wh_m2=[10., 20.]))
        observed = pd.DataFrame(dict(
            building=[1, 2, 1], initially_heating_near_setpoint=[True, True, False],
            dhw_first_hour=[False, True, False], cooling_before_recharge=[False, True, False],
            drop_gt_0p5c=[False, True, False], withdrawal_to_cooling_or_censor_wh_m2=[5., 30., 200.],
            water_capacity_wh_m2_k=[1., 2., 1.],
        ))
        result = storage_support_diagnostics(annual, observed).set_index("subset")
        self.assertEqual(result.loc["initially_heating", "events"], 2)
        self.assertEqual(result.loc["initially_heating", "fraction_withdrawal_above_capacity"], .5)
        self.assertEqual(result.loc["initially_heating", "median_withdrawal_capacity_ratio"], 1.)
        self.assertEqual(result.loc["initially_heating", "median_capacity_equivalent_water_band_k"], 10.)
        self.assertEqual(result.loc["no_dhw_no_cooling", "fraction_withdrawal_above_capacity"], 0.)
        self.assertEqual(result.loc["dhw_drop_gt_0p5c", "fraction_withdrawal_above_capacity"], 1.)
        annual.loc[0, "capacity_wh_m2"] = 0.
        with self.assertRaises(ValueError):
            storage_support_diagnostics(annual, observed)

    def test_recovery_diagnostics_preserve_outcome_and_heating_subsets(self):
        events = pd.DataFrame(dict(
            building=[1, 1, 2, 2], initially_heating_near_setpoint=[True, True, False, False],
            dhw_first_hour=[True, False, True, False],
            dhw_first_hour_probability=[.9, .1, .7, .3],
            observed_drop_c=[1., 0., 1., 0.], **{"p_drop_gt_0.5": [.8, .2, .6, .4]},
        ))
        result = recovery_event_diagnostics(events).set_index(["subset", "regime"])
        self.assertEqual(result.loc[("all_eligible", "all"), "events"], 4)
        self.assertAlmostEqual(result.loc[("all_eligible", "all"), "recovery_brier"], .05)
        self.assertAlmostEqual(result.loc[("initially_heating", "dhw"), "predicted_recovery_probability"], .9)
        self.assertAlmostEqual(result.loc[("initially_heating", "no_dhw"), "predicted_recovery_probability"], .1)
        self.assertEqual(result.loc[("initially_heating", "no_dhw"), "observed_recovery_frequency"], 0.)
        scored = recovery_event_diagnostics(events, training_events=events).set_index(["subset", "regime"])
        self.assertAlmostEqual(scored.loc[("all_eligible", "all"), "training_recovery_frequency"], .5)
        self.assertAlmostEqual(scored.loc[("all_eligible", "all"), "recovery_brier_skill"], .8)
        self.assertAlmostEqual(scored.loc[("initially_heating", "no_dhw"), "training_climatology_brier"], .25)
        self.assertTrue(recovery_event_diagnostics(events.drop(columns="dhw_first_hour_probability")).empty)
        events.loc[0, "dhw_first_hour_probability"] = np.nan
        with self.assertRaises(ValueError):
            recovery_event_diagnostics(events)

    def test_conclusions_require_all_four_fully_matched_kpi_groups(self):
        kinds = ["setpoint_f", "setpoint_regression", "force_off_f", "force_off_regression"]
        summary = pd.DataFrame(dict(method="accounting", kpi_kind=kind) for kind in kinds)
        contrasts = pd.DataFrame(dict(method="accounting", reference="baseline", kpi_kind=kind,
            matched_cells=80, baseline_cells=80, candidate_cells=80, matched_buildings=20,
            lower95=.1, upper95=.3) for kind in kinds)
        status = [dict(method="accounting", setpoint_scored=True, force_off_scored=True,
                       expected_test_buildings=20, diagnostics_complete=False)]
        result = comparison_conclusions(summary, contrasts, status).iloc[0]
        self.assertEqual(result.decision, "do_not_promote")
        self.assertIn("audit remains incomplete", result.conclusion)
        self.assertEqual(comparison_conclusions(summary, contrasts.iloc[:-1], status).iloc[0].decision, "incomplete")
        contrasts.loc[0, "matched_cells"] = 79
        self.assertEqual(comparison_conclusions(summary, contrasts, status).iloc[0].decision, "incomplete")
        contrasts.loc[0, "matched_cells"] = 80
        contrasts.loc[0, "lower95"] = -.1
        self.assertEqual(comparison_conclusions(summary, contrasts, status).iloc[0].decision, "review")

    def test_paired_comparison_preserves_building_and_cell_keys(self):
        rows = []
        for building in (1, 2):
            for horizon in (.5, 1., 2., 3.):
                for method, error in (("baseline", 2.), ("candidate", 1.)):
                    rows.append(dict(method=method, kpi_kind="force_off_f", profile_id=building,
                                     horizon_hours=horizon, direction="force_off", truth=3., crps=error))
        frame = pd.DataFrame(rows)
        result = paired_crps_bootstrap(frame).iloc[0]
        self.assertEqual(result.matched_buildings, 2)
        self.assertEqual(result.matched_cells, 8)
        np.testing.assert_allclose(result[["crps_difference", "lower95", "upper95"]].to_numpy(float), -1.)
        frame.loc[frame.method == "candidate", "truth"] = 4.
        with self.assertRaises(AssertionError):
            paired_crps_bootstrap(frame)

    def test_recipe_does_not_mutate_baseline(self):
        base = {"probabilistic_structured": {"hp_buffer_capacity_mode": "global"}}
        accounting = recipe_for_variant(base, "accounting")["probabilistic_structured"]
        physical = recipe_for_variant(base, "physical")["probabilistic_structured"]
        dhw = recipe_for_variant(base, "physical_dhw")["probabilistic_structured"]
        self.assertEqual(base, {"probabilistic_structured": {"hp_buffer_capacity_mode": "global"}})
        self.assertTrue(accounting["hp_whole_building_accounting"])
        self.assertTrue(physical["hp_plant_metadata"])
        self.assertTrue(dhw["prob_dhw_recovery_state"])
        self.assertEqual(physical["hp_buffer_delta_t_usable_init_k"], dhw["hp_buffer_delta_t_usable_init_k"])


if __name__ == "__main__":
    unittest.main()
