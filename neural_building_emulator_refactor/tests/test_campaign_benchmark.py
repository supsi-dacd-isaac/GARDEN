from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.campaign_benchmark import (
    METHODS, POLICIES, VARIANTS, job_plan, resolve_campaign, validate_distributions, report,
)
from neural_building_emulator.data import (
    SplitConfig, load_closed_loop_result_splits, load_closed_loop_training_profiles,
    make_closed_loop_windows, to_closed_loop_profiles, WindowConfig,
)
from neural_building_emulator.test_dataset_adapter import _write_entity_profile
from neural_building_emulator.columns import SH_STORAGE_CHARGE_COLUMN


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def campaign(self):
        source = dict(seed=20261028, requested_max_profiles=1000, selected_ids=[1, 2, 3, 4, 5],
                      train_ids=[1, 2], validation_ids=[3], calibration_ids=[4], test_ids=[5])
        (self.root / "split_manifest.json").write_text(json.dumps(source))
        (self.root / "campaign_manifest.json").write_text('{}')
        for policy in POLICIES:
            for building in (1, 3, 4, 5):
                folder = self.root / "data" / ("control=" + policy) / f"egid={building}"
                folder.mkdir(parents=True)
                for name in ("complete.json", "simulation_metadata.json", "timeseries.parquet"):
                    (folder / name).write_text('{}')
        return dict(max_profiles=1000, seed=13, deterministic_structured={})

    def test_preserves_folds_and_separates_training_seed(self):
        base = self.campaign()
        with patch("neural_building_emulator_refactor.calibrated_model_benchmark._eligible_hp_ids", return_value=(1, 3, 4, 5)):
            recipe, split = resolve_campaign(base, self.root, self.root / "output", excluded_ids=[2])
            _, repeated = resolve_campaign(base, self.root, self.root / "output", excluded_ids=[2])
        self.assertEqual(split.train_ids, (1,))
        self.assertEqual(split.calibration_ids, (4,))
        self.assertEqual(split.seed, 20261028)
        self.assertEqual(recipe["seed"], 13)
        self.assertEqual(split, repeated)
        self.assertIn("setpoint_excitation", recipe["dataset_path"])
        self.assertEqual(len(recipe["additional_dataset_paths"]), 1)
        self.assertIn("force_off_excitation", recipe["additional_dataset_paths"][0])
        self.assertEqual(recipe["campaign"]["excluded_missing_ids"], [2])

    def test_missing_buildings_are_never_silently_resplit(self):
        base = self.campaign()
        with self.assertRaisesRegex(ValueError, "explicit exclusions"):
            resolve_campaign(base, self.root, self.root / "output")
        source = json.loads((self.root / "split_manifest.json").read_text())
        source["test_ids"] += [1]
        (self.root / "split_manifest.json").write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            resolve_campaign(base, self.root, self.root / "output", excluded_ids=[2])

    def test_only_source_policies_have_calibration_jobs(self):
        args = SimpleNamespace(output_dir=self.root, campaign_root=self.root, current_best_root=self.root,
                               epochs=15, num_scenarios=100, num_profile_plots=5, exclude_missing_profile_ids=[2])
        jobs = job_plan(args)
        for policy in POLICIES[2:]:
            self.assertFalse(any(policy in name and "calibration" in name for name, _ in jobs))
            self.assertTrue(any(name == "lightgbm_" + policy for name, _ in jobs))
        self.assertEqual([name for name, _ in jobs if name.startswith("train_")], ["train_" + v for v in VARIANTS])
        self.assertEqual(sum("score_" in name for name, _ in jobs), 18)

    def test_streamed_profiles_match_materialized_policies_and_plant_targets(self):
        paths = [self.root / str(index) for index in (0, 1)]
        for index, path in enumerate(paths):
            for building in (123, 456, 789):
                folder = _write_entity_profile(path, building)
                parquet = folder / "timeseries.parquet"
                frame = pd.read_parquet(parquet)
                frame[SH_STORAGE_CHARGE_COLUMN] = np.where(frame.zone.eq("technical_room"), 2. + index, np.nan)
                frame.loc[frame.zone.eq("fl0_thz0"), "Zone Air Temperature"] += index
                frame.to_parquet(parquet, index=False)
        config = SplitConfig(dataset_path=paths[0], additional_dataset_paths=(paths[1],), max_profiles=3,
                             fixed_test_profile_ids=(456,), include_plant_targets=True)
        expected = load_closed_loop_result_splits(config)
        actual, train, validation = load_closed_loop_training_profiles(config, batch_size=1)
        self.assertTrue(actual.train.empty and actual.test.empty)
        self.assertEqual(actual.train_ids, expected.train_ids)
        for name, profiles in (("train", train), ("test", validation)):
            reference = to_closed_loop_profiles(getattr(expected, name), include_hp_force_off=True,
                metadata_columns=expected.metadata_columns, hp_power_area_normalization=expected.hp_power_area_normalization)
            self.assertEqual(len(reference), len(profiles))
            for before, after in zip(reference, profiles):
                self.assertEqual(before.profile_id, after.profile_id)
                for field in ("datetime", "metadata", "inputs", "targets", "initial_temperature", "plant_targets"):
                    np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
            before = make_closed_loop_windows(reference, WindowConfig(2, 1))
            after = make_closed_loop_windows(profiles, WindowConfig(2, 1))
            for field in asdict(before):
                np.testing.assert_array_equal(getattr(before, field), getattr(after, field))

    @staticmethod
    def samples(ids):
        return pd.DataFrame([dict(method=method, kpi_kind=kind, profile_id=b, horizon_hours=h,
            direction=direction, truth=float(b + h), sample_index=s, prediction=b + h + (s - 10) / 10)
            for method in METHODS for kind in ("regression_based", "f", "force_off_regression", "force_off_f")
            for b in ids for h in (.5, 1., 2., 3.)
            for direction in (("force_off",) if kind.startswith("force_off") else ("up", "down"))
            for s in range(20)])

    def test_report_and_fairness_checks_cover_all_models_and_policy_families(self):
        calibration, test = self.samples([1, 2, 3, 4]), self.samples([5, 6])
        validate_distributions(test, [5, 6], 20, (.5, 1., 2., 3.))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_distributions(test.iloc[1:], [5, 6], 20, (.5, 1., 2., 3.))
        args = SimpleNamespace(policy=POLICIES[1], output_dir=self.root)
        recipe = dict(evaluation=dict(num_scenarios=20, horizons_hours=[.5, 1., 2., 3.]),
                      calibration=dict(minimum_scale=.001, minimum_scale_fraction=.01))
        split = SimpleNamespace(calibration_ids=(1, 2, 3, 4), test_ids=(5, 6))
        with patch("neural_building_emulator_refactor.campaign_benchmark.distributions", side_effect=[calibration, test]):
            report(args, recipe, split)
        joint = pd.read_csv(self.root / "reports/joint_distribution_metrics.csv")
        self.assertEqual(set(joint.dimension), {8, 16})
        self.assertEqual(set(joint.method), set(METHODS))
        self.assertTrue((self.root / "reports/index.html").exists())


if __name__ == "__main__":
    unittest.main()
