import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.dhw_source_diagnostics import summarize_building


class DHWSourceDiagnosticsTests(unittest.TestCase):
    def test_source_power_and_draw_units_with_latched_inactive_mode(self):
        dates = pd.date_range("2023-01-01", periods=4, freq="15min")
        plant = pd.DataFrame(dict(
            zone="technical_room", datetime=dates, hp_mode_is_dhw=[0, 1, 1, 1],
            heat_pump_electric_power=[0., 0., 1000., 1000.],
            thermal_storage_charge_dhw_kw=[0., 0., 2., 4.], input_dhw_mixed_water_l=np.nan,
        ))
        floor = pd.DataFrame(dict(zone="fl0_thz0", datetime=dates, input_dhw_mixed_water_l=[999., 1., 1., 1.]))
        metadata = dict(
            gardenCompatibility=dict(values=dict(DHW_recovery_W=1000.)),
            resolved_plant_design=dict(dhw_tank_source_capacity_w=1000., dhw_resistance_capacity_w=200.),
        )
        row = summarize_building(pd.concat([plant, floor], ignore_index=True), metadata)
        self.assertEqual(row["annual_charge_wh"], 1500.)
        self.assertEqual(row["annual_mixed_water_l"], 3.)
        self.assertEqual(row["mean_active_charge_w"], 3000.)
        self.assertEqual(row["mean_active_charge_over_nominal"], 3.)
        self.assertEqual(row["active_fraction_above_surrogate_max"], 1.)
        self.assertAlmostEqual(row["observed_active_fraction"], 2 / 3)
        self.assertEqual(row["charge_energy_without_active_fraction"], 0.)


if __name__ == "__main__":
    unittest.main()
