import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.matched_policy_transfer import (
    CELL, KPI_KINDS, METHODS, calibrated_branches, digest, flatten, freeze,
    validate_distribution, verify_hashes, write_json, report,
)
from neural_building_emulator_refactor.policy_kpi_identifiability import directional_identifiability
from neural_building_emulator_refactor.matched_policy_transfer_review import run as review


def ensemble(ids):
    rows = []
    for method in METHODS:
        for kind in KPI_KINDS:
            for profile in ids:
                for horizon in (.5, 1., 2., 3.):
                    for direction in ("up", "down"):
                        for sample in range(20):
                            # Opposing ranks exercise ECC, rather than an already sorted copula.
                            value = sample if direction == "up" else 19 - sample
                            rows.append(dict(method=method, kpi_kind=kind, profile_id=profile,
                                             horizon_hours=horizon, direction=direction,
                                             sample_index=sample, prediction=value / 10 + horizon,
                                             truth=horizon + .2 * profile))
    return pd.DataFrame(rows)


class MatchedPolicyTransferTest(unittest.TestCase):
    def test_complete_and_matching_cells_required(self):
        frame = ensemble([10, 11])
        self.assertEqual(len(validate_distribution(frame, [10, 11], 20)), 4)
        for broken in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]),
                       frame[frame.method != "lightgbm"]):
            with self.assertRaises(ValueError):
                validate_distribution(broken, [10, 11], 20)
        broken = frame.copy()
        broken.loc[broken.method == "lightgbm", "truth"] += 1
        with self.assertRaises(AssertionError):
            validate_distribution(broken, [10, 11], 20)

    def test_label_independence_and_ecc_marginals(self):
        source, target, test = ensemble([1, 2, 3]), ensemble([1, 2, 3]), ensemble([10])
        target["truth"] += 8
        settings = dict(minimum_scale=.001, minimum_scale_fraction=.01)
        result = calibrated_branches(source, target, test, settings)
        changed_test = test.copy()
        changed_test["truth"] += 1000
        truth_only = calibrated_branches(source, target, changed_test, settings)
        for label in result:
            np.testing.assert_array_equal(result[label].prediction, truth_only[label].prediction)
        changed_target = target.copy()
        changed_target["truth"] += 5
        altered = calibrated_branches(source, changed_target, changed_test, settings)
        np.testing.assert_array_equal(result["source_rns"].prediction, altered["source_rns"].prediction)
        self.assertFalse(np.array_equal(result["target_rns"].prediction, altered["target_rns"].prediction))
        for label in ("source_rns", "target_rns"):
            left = result[label].groupby(["method", *CELL]).prediction.apply(lambda x: sorted(x))
            right = result[label + "_ecc"].groupby(["method", *CELL]).prediction.apply(lambda x: sorted(x))
            self.assertEqual(left.to_dict(), right.to_dict())
            self.assertFalse(np.array_equal(result[label].prediction, result[label + "_ecc"].prediction))

    def test_no_overlap_or_missing_calibration_groups(self):
        source, test = ensemble([1, 2, 3]), ensemble([10])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            calibrated_branches(source, test, test, {})
        with self.assertRaisesRegex(ValueError, "group mismatch"):
            calibrated_branches(source[source.direction == "up"], source, test, {})

    def test_immutable_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            freeze(path, {"seed": 13})
            freeze(path, {"seed": 13})
            with self.assertRaises(ValueError):
                freeze(path, {"seed": 14})
            verify_hashes({str(path): digest(path)})
            with self.assertRaises(ValueError):
                verify_hashes({str(path): "changed"})

    def test_nested_domain_audit(self):
        self.assertEqual(flatten({"sizing": {"temperature": 55}, "a": [1, 2]}),
                         {"sizing.temperature": 55, "a": [1, 2]})

    def test_binary_policy_cannot_identify_both_directions(self):
        deltas = np.tile([-1., 1.], 20)
        directions = np.column_stack([np.ones(40), np.maximum(deltas, 0), np.minimum(deltas, 0)])
        self.assertEqual(directional_identifiability(directions)["directional_rank_after_controls"], 1)
        pre_setpoint = np.where(deltas > 0, 20., 21.)
        full = np.column_stack([directions, pre_setpoint - pre_setpoint.mean()])
        self.assertEqual(directional_identifiability(full)["directional_rank_after_controls"], 0)

    def test_varied_policy_can_identify_directions_with_constant_nuisance(self):
        deltas = np.tile([-2., -1., 1., 2.], 10)
        design = np.column_stack([np.ones(40), np.maximum(deltas, 0), np.minimum(deltas, 0), np.zeros(40)])
        result = directional_identifiability(design)
        self.assertEqual(result["directional_rank_after_controls"], 2)
        self.assertLess(result["design_rank"], result["columns"])

    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = SimpleNamespace(output=output)
            split = SimpleNamespace(calibration_ids=(1, 2, 3), test_ids=(10,))
            recipe = dict(seed=13, evaluation={"num_scenarios": 20},
                          calibration=dict(minimum_scale=.001, minimum_scale_fraction=.01))
            write_json(output / "request.json", {"hashes": {}})
            for method in METHODS[:-1]:
                destination = output / "scoring/test" / method
                destination.mkdir(parents=True)
                write_json(destination / "summary.json", {
                    "metrics": {"temperature_c_rmse": .5},
                    "temporal_trace_calibration": [dict(signal=signal, scope="all", mean_crps=.1,
                        empirical_central_90_coverage=.9, mean_central_90_width=1.)
                        for signal in ("Tin", "Qroom", "Pel_SH")],
                })
            pd.DataFrame([dict(fold="test", field="inputs.building.dhwSetpoint", buildings=1,
                               source_example="60", target_example="55")]).to_csv(
                                   output / "domain_difference_summary.csv", index=False)
            pd.DataFrame([dict(fold="test", profile_id=10, field="inputs.building.dhwSetpoint",
                               source="60", target="55")]).to_csv(output / "domain_differences.csv", index=False)
            pd.DataFrame([dict(field="weather", fraction_different=0., max_abs_diff=0.)]).to_csv(
                output / "deterministic_input_audit.csv", index=False)
            audit = pd.DataFrame([dict(dataset=dataset, fold=fold, controls=controls,
                                       min_directional_rank=2 if dataset == "source" else 0)
                                  for dataset in ("source", "tariff") for fold in ("calibration", "test")
                                  for controls in ("none", "full")])
            audit.to_csv(output / "kpi_identifiability_summary.csv", index=False)
            audit.to_csv(output / "kpi_identifiability.csv", index=False)
            frames = {"calibration": ensemble([1, 2, 3]), "test": ensemble([10])}
            module = "neural_building_emulator_refactor.matched_policy_transfer"
            with patch(module + ".load_source", side_effect=lambda _, fold: frames[fold]), \
                 patch(module + ".load_target", side_effect=lambda _, fold: frames[fold]), \
                 patch("builtins.print"):
                report(args, recipe, split)
                review(output)
            self.assertIn("not uniquely identified", (output / "reports/review.html").read_text())
            self.assertTrue((output / "review_completion.json").exists())
            self.assertTrue((output / "reports/direct_f_reliability.html").exists())


if __name__ == "__main__":
    unittest.main()
