"""Dataset-design checks independent of trained artifacts and benchmark outputs."""

import subprocess
import sys
import unittest

import numpy as np

from neural_building_emulator_refactor.policy_kpi_identifiability import (
    directional_identifiability,
)


def design_for(deltas, *nuisance):
    return np.column_stack(
        [np.ones(len(deltas)), np.maximum(deltas, 0), np.minimum(deltas, 0), *nuisance]
    )


class PolicyKpiIdentifiabilityTests(unittest.TestCase):
    def test_binary_policy_loses_both_directions_with_pre_setpoint(self):
        deltas = np.tile([-1., 1.], 20)
        result = directional_identifiability(design_for(deltas))
        self.assertEqual(result["directional_rank_after_controls"], 1)
        pre_setpoint = np.where(deltas > 0, 20., 21.)
        result = directional_identifiability(design_for(deltas, pre_setpoint))
        self.assertEqual(result["directional_rank_after_controls"], 0)

    def test_multilevel_policy_retains_directions_with_constant_nuisance(self):
        deltas = np.tile([-2., -1., 1., 2.], 10)
        result = directional_identifiability(design_for(deltas, np.zeros(40)))
        self.assertEqual(result["directional_rank_after_controls"], 2)
        self.assertLess(result["design_rank"], result["columns"])
        self.assertGreater(result["directional_singular_min"], 0)

    def test_relative_rank_is_invariant_to_change_of_setpoint_units(self):
        deltas = np.tile([-2., -1., 1., 2.], 10)
        reference = directional_identifiability(design_for(deltas))
        scaled = directional_identifiability(design_for(deltas * 100))
        self.assertEqual(reference["directional_rank_after_controls"],
                         scaled["directional_rank_after_controls"])
        self.assertAlmostEqual(reference["up_residual_fraction"], scaled["up_residual_fraction"])

    def test_no_excitation_has_zero_rank(self):
        result = directional_identifiability(design_for(np.zeros(10)))
        self.assertEqual(result["directional_rank_after_controls"], 0)

    def test_invalid_designs_fail_explicitly(self):
        for design in (np.ones(4), np.ones((2, 3)), np.ones((10, 2)),
                       np.full((10, 3), np.nan), np.full((10, 3), np.inf)):
            with self.subTest(shape=design.shape), self.assertRaises(ValueError):
                directional_identifiability(design)
        for tolerance in (0, -1, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                directional_identifiability(np.ones((10, 3)), tolerance)

    def test_import_does_not_load_experimental_pipeline_or_jax(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; "
             "from neural_building_emulator_refactor.policy_kpi_identifiability "
             "import directional_identifiability; "
             "assert 'jax' not in sys.modules; "
             "assert 'neural_building_emulator_refactor.matched_policy_transfer' "
             "not in sys.modules"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
