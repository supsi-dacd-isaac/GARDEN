"""The two arms must retain fold membership and share all original losses."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from neural_building_emulator.columns import CLOSED_LOOP_TARGET_COLUMNS
from neural_building_emulator_refactor.tank_temperature_experiment import FOLDS, subset, variant_recipe, validate_report_artifacts
from neural_building_emulator_refactor.tank_temperature_monitor import epoch_protocol, extension_decision


def artifact_stub():
    metadata = dict(train_ids=[1], test_ids=[2], metadata_columns=["volume"],
        input_columns=["setpoint"], model_kind="closed_loop_hp_contracting_probabilistic",
        target_alignment="same_time", target_mode="absolute")
    scaler = lambda size: SimpleNamespace(mean=np.zeros(size), scale=np.ones(size))
    return SimpleNamespace(legacy_artifact=SimpleNamespace(metadata=metadata),
        model=SimpleNamespace(temperature_target_index=0, qroom_target_index=1, pel_target_index=2),
        scalers=SimpleNamespace(metadata=scaler(1), inputs=scaler(1), target=scaler(3)))


class TankExperimentTests(unittest.TestCase):
    def test_full_cohort_preserves_every_original_fold(self):
        source = {key: list(range(i*1000, i*1000+n)) for i, (key, n) in enumerate(zip(FOLDS, (701,100,99,100)))}
        source["selected_ids"] = sorted(x for key in FOLDS for x in source[key])
        folds, ids = subset(source, max_profiles=1000)
        self.assertEqual(ids, source["selected_ids"])
        for key in FOLDS:
            self.assertEqual(list(folds[key]), source[key])
        source["test_ids"][0] = source["train_ids"][0]
        with self.assertRaises(ValueError):
            subset(source, max_profiles=1000)

    def test_gate_only_uses_validation_and_extends_for_joint_progress(self):
        protocol = epoch_protocol(dict(validation_ids=[11, 12], test_ids=[13], calibration_ids=[14]))
        self.assertEqual(protocol["profile_ids"], [11, 12])
        before = pd.DataFrame([dict(group=g, policy=p, metric=m, crps=x)
            for g, m, x in [("trace", "Tin", 1.), ("trace", "Qroom", 5.), ("kpi", "f", 3.)]
            for p in ["setpoint_excitation_v2", "force_off_excitation_v2"]])
        after = before.copy()
        after["crps"] *= .97
        self.assertTrue(extension_decision(before, after, protocol)["extend_to_15"])
        after["crps"] = before.crps * .995
        self.assertFalse(extension_decision(before, after, protocol)["extend_to_15"])
        after["crps"] = before.crps * np.where(before.group.eq("trace"), .90, 1.03)
        self.assertFalse(extension_decision(before, after, protocol)["extend_to_15"])

    def test_gate_rejects_missing_nonfinite_or_duplicate_scores(self):
        protocol = epoch_protocol(dict(validation_ids=[1]))
        before = pd.DataFrame([dict(group=g, policy="source", metric="score", crps=1.) for g in ["trace", "kpi"]])
        for after in [before.iloc[:1], pd.concat([before, before]), before.assign(crps=np.nan)]:
            with self.assertRaises(ValueError):
                extension_decision(before, after, protocol)

    def test_legacy_target_schema_does_not_require_explicit_columns(self):
        before, after = artifact_stub(), artifact_stub()
        validate_report_artifacts([before, after])
        after.legacy_artifact.metadata["target_columns"] = list(CLOSED_LOOP_TARGET_COLUMNS)
        validate_report_artifacts([before, after])

    def test_target_schema_still_rejects_mismatches(self):
        for failure in ("columns", "indices", "dimensions", "unsupported"):
            with self.subTest(failure=failure):
                before, after = artifact_stub(), artifact_stub()
                if failure == "columns":
                    after.legacy_artifact.metadata["target_columns"] = list(reversed(CLOSED_LOOP_TARGET_COLUMNS))
                elif failure == "indices":
                    after.model.qroom_target_index = 2
                elif failure == "dimensions":
                    after.scalers.target.mean = np.zeros(4)
                else:
                    for a in (before, after):
                        a.legacy_artifact.metadata["model_kind"] = "unknown"
                with self.assertRaises(ValueError):
                    validate_report_artifacts([before, after])

    def test_report_rejects_changed_split_or_scalers(self):
        before, after = artifact_stub(), artifact_stub()
        after.legacy_artifact.metadata["test_ids"] = [3]
        with self.assertRaises(ValueError):
            validate_report_artifacts([before, after])
        after = artifact_stub()
        after.scalers.target.scale[1] = 2.
        with self.assertRaises(AssertionError):
            validate_report_artifacts([before, after])

    def test_subset_preserves_folds_and_is_reproducible(self):
        source = {key: list(range(i*1000, i*1000+n)) for i, (key, n) in enumerate(zip(FOLDS, (701,100,99,100)))}
        folds, ids = subset(source)
        self.assertEqual(subset(source), (folds, ids))
        self.assertEqual(len(ids), 200)
        self.assertEqual(len(set(ids)), 200)
        self.assertEqual([len(folds[k]) for k in FOLDS], [140,20,20,20])
        for key in FOLDS:
            self.assertTrue(set(folds[key]).issubset(source[key]))

    def test_only_candidate_state_and_loss_differ(self):
        base = dict(probabilistic_structured=dict(prob_flex_kpi_crps_weight=1.5, hp_energy_cap_hours=.5))
        baseline = variant_recipe(base, "baseline")["probabilistic_structured"]
        candidate = variant_recipe(base, "two_tanks")["probabilistic_structured"]
        self.assertTrue(baseline["hp_dhw_metadata"])
        self.assertTrue(candidate["hp_dhw_metadata"])
        differences = {key for key in baseline if baseline[key] != candidate[key]}
        self.assertEqual(differences, {"prob_tank_temperature_state", "prob_tank_temperature_loss_weight"})
        self.assertNotIn("hp_dhw_metadata", base["probabilistic_structured"])
        with self.assertRaises(ValueError):
            variant_recipe(base, "typo")


if __name__ == "__main__":
    unittest.main()
