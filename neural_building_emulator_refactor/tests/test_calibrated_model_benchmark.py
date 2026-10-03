from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import jax.numpy as jnp

from neural_building_emulator.ventilation import (
    EPlusVentilationRollout,
    EPlusVentilationSchedule,
)

from neural_building_emulator_refactor.calibrated_model_benchmark import (
    DEFAULT_CONFIG,
    _force_off_emulator_distribution,
    _force_off_dataset_path,
    _paired_bootstrap_difference,
    _policy_profile_ids,
    create_benchmark_split,
    parse_args,
)
from neural_building_emulator_refactor.fair_probabilistic_lstm_comparison import (
    _experiment_configs,
)
from neural_building_emulator_refactor.probabilistic_calibration import (
    calibrate_distribution_table,
    ensemble_crps,
)
from neural_building_emulator_refactor.scoring import _scenario_extreme_traces


class CalibratedBenchmarkTests(unittest.TestCase):
    def test_extreme_trace_selection_is_per_output_and_annual(self) -> None:
        target = np.zeros((2, 2), dtype=float)
        scenarios = np.asarray(
            [
                [[0.0, 4.0], [0.0, 4.0]],
                [[3.0, 0.0], [3.0, 0.0]],
            ]
        )
        closest, worst = _scenario_extreme_traces(scenarios, target)
        np.testing.assert_allclose(closest, np.zeros((2, 2)))
        np.testing.assert_allclose(worst, np.asarray([[3.0, 4.0], [3.0, 4.0]]))

    def test_policy_profile_examples_are_disjoint_when_possible(self) -> None:
        test_ids = tuple(range(10, 20))
        setpoint = _policy_profile_ids(
            test_ids, policy="setpoint", count=3
        )
        force_off = _policy_profile_ids(
            test_ids, policy="force_off", count=3
        )
        self.assertEqual(setpoint, (10, 11, 12))
        self.assertEqual(force_off, (13, 14, 15))
        self.assertFalse(set(setpoint).intersection(force_off))

    def test_policy_profile_examples_fall_back_for_small_test_split(self) -> None:
        test_ids = (10, 11, 12, 13)
        self.assertEqual(
            _policy_profile_ids(test_ids, policy="force_off", count=3),
            (11, 12, 13),
        )

    def test_hp_training_mode_override_is_exposed(self) -> None:
        args = parse_args(["--prob-hp-training-mode", "hard_expected_grad"])
        self.assertEqual(args.prob_hp_training_mode, "hard_expected_grad")

    def test_additional_policy_dataset_override_is_exposed(self) -> None:
        args = parse_args(
            [
                "--additional-dataset",
                "force_off_a",
                "--additional-dataset",
                "force_off_b",
            ]
        )
        self.assertEqual(
            args.additional_datasets,
            [Path("force_off_a"), Path("force_off_b")],
        )

    def test_feedback_gradient_scale_override_is_exposed(self) -> None:
        args = parse_args(
            ["--contracting-temperature-feedback-gradient-scale", "0.05"]
        )
        self.assertEqual(
            args.contracting_temperature_feedback_gradient_scale,
            0.05,
        )

    def test_force_off_dataset_is_inferred_from_additional_policies(self) -> None:
        self.assertEqual(
            _force_off_dataset_path(
                {
                    "additional_dataset_paths": [
                        "data/control=other",
                        "data/control=force_off",
                    ]
                }
            ),
            Path("data/control=force_off"),
        )

    def test_force_off_distribution_uses_matching_model_and_estimator(self) -> None:
        columns = {
            "profile_id": [7, 7, 7, 7],
            "model": ["structured_gated"] * 4,
            "signal": ["simulation", "simulation", "emulation", "emulation"],
            "scenario_index": [np.nan, np.nan, 0, 1],
            "intervention_type": ["force_off"] * 4,
            "direction": ["force_off"] * 4,
            "estimator": ["direct_ratio"] * 4,
            "horizon_hours": [1.0] * 4,
            "valid": [True] * 4,
            "flexibility_wh_m2_per_unit": [3.0, 3.0, 2.0, 4.0],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory)
            point = pd.DataFrame(columns).iloc[:2]
            scenarios = pd.DataFrame(columns).iloc[2:]
            point.to_csv(path / "force_off_flexibility_kpis.csv", index=False)
            scenarios.to_csv(
                path / "force_off_flexibility_kpi_scenarios.csv", index=False
            )
            result = _force_off_emulator_distribution(
                path,
                method="probabilistic_structured",
                source_model="structured_gated",
                estimator="direct_ratio",
                kpi_kind="force_off_f",
            )
        self.assertEqual(result["truth"].tolist(), [3.0, 3.0])
        self.assertEqual(result["prediction"].tolist(), [2.0, 4.0])
        self.assertEqual(set(result["direction"]), {"force_off"})

    @patch(
        "neural_building_emulator_refactor.calibrated_model_benchmark._eligible_hp_ids",
        return_value=tuple(range(100)),
    )
    def test_four_way_split_is_disjoint_and_reproducible(self, _eligible) -> None:
        first = create_benchmark_split(
            Path("dataset"),
            max_profiles=50,
            seed=13,
            validation_fraction=0.1,
            calibration_fraction=0.1,
            test_fraction=0.1,
        )
        second = create_benchmark_split(
            Path("dataset"),
            max_profiles=50,
            seed=13,
            validation_fraction=0.1,
            calibration_fraction=0.1,
            test_fraction=0.1,
        )
        self.assertEqual(first, second)
        groups = (
            set(first.train_ids),
            set(first.validation_ids),
            set(first.calibration_ids),
            set(first.test_ids),
        )
        self.assertEqual(sum(map(len, groups)), 50)
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                self.assertFalse(left.intersection(right))

    @patch(
        "neural_building_emulator_refactor.calibrated_model_benchmark._eligible_hp_ids",
        side_effect=(tuple(range(20)), tuple(range(5, 20))),
    )
    def test_split_uses_profiles_shared_by_all_policy_datasets(self, _eligible) -> None:
        split = create_benchmark_split(
            Path("setpoint"),
            additional_dataset_paths=(Path("force_off"),),
            max_profiles=12,
            seed=13,
            validation_fraction=0.1,
            calibration_fraction=0.2,
            test_fraction=0.2,
        )
        self.assertEqual(len(split.selected_ids), 12)
        self.assertTrue(set(split.selected_ids).issubset(set(range(5, 20))))

    def test_four_model_configs_share_fixed_validation_and_exclusions(self) -> None:
        recipe = json.loads(DEFAULT_CONFIG.read_text())
        self.assertEqual(recipe["evaluation"]["ventilation_rollout_mode"], "eplus_rule")
        with TemporaryDirectory() as directory:
            configs = _experiment_configs(
                recipe,
                Path(directory),
                max_profiles=14,
                epochs=1,
                fixed_test_profile_ids=(10, 11),
                excluded_profile_ids=(12, 13),
            )
        for config in configs.values():
            self.assertEqual(config.fixed_test_profile_ids, (10, 11))
            self.assertEqual(config.excluded_profile_ids, (12, 13))

    def test_four_model_configs_share_additional_policy_datasets(self) -> None:
        recipe = json.loads(DEFAULT_CONFIG.read_text())
        recipe["additional_dataset_paths"] = ["force_off_a", "force_off_b"]
        with TemporaryDirectory() as directory:
            configs = _experiment_configs(
                recipe,
                Path(directory),
                max_profiles=14,
                epochs=1,
            )
        for config in configs.values():
            self.assertEqual(
                config.additional_dataset_paths,
                (Path("force_off_a"), Path("force_off_b")),
            )

    def test_eplus_ventilation_depends_on_predicted_temperature(self) -> None:
        rollout = EPlusVentilationRollout(
            schedule=EPlusVentilationSchedule(
                time_available=np.array([1.0], dtype=np.float32),
                minimum_indoor_temperature_c=np.array([23.0], dtype=np.float32),
                nominal_flow_m3_s=0.2,
                minimum_indoor_outdoor_delta_c=2.0,
            ),
            ventilation_input_index=2,
            outdoor_input_index=1,
            input_mean=(0.0, 0.0, 0.0),
            input_scale=(1.0, 1.0, 1.0),
            temperature_mean_c=0.0,
            temperature_scale_c=1.0,
        )
        input_t = jnp.array([0.0, 18.0, 0.5])
        self.assertAlmostEqual(float(rollout.apply(input_t, jnp.array(24.0), jnp.array(0))[2]), 0.2)
        self.assertAlmostEqual(float(rollout.apply(input_t, jnp.array(22.0), jnp.array(0))[2]), 0.0)

    def test_empirical_crps_matches_two_member_example(self) -> None:
        self.assertAlmostEqual(ensemble_crps([0.0, 1.0], 0.5), 0.25)

    def test_rns_corrects_location_bias_without_quantile_crossing(self) -> None:
        calibration_rows = []
        test_rows = []
        for profile_id in range(20):
            truth = float(profile_id) / 5.0
            for sample_index, offset in enumerate(np.linspace(-0.5, 0.5, 41)):
                common = {
                    "method": "emulator",
                    "kpi_kind": "f",
                    "horizon_hours": 3.0,
                    "direction": "up",
                    "profile_id": profile_id,
                    "truth": truth,
                    "sample_index": sample_index,
                    "prediction": truth - 1.0 + float(offset),
                }
                calibration_rows.append(common)
                test_rows.append({**common, "profile_id": profile_id + 100})
        calibrated, metrics, _, _ = calibrate_distribution_table(
            pd.DataFrame(calibration_rows),
            pd.DataFrame(test_rows),
        )
        raw = metrics[metrics["calibration"] == "raw"].iloc[0]
        corrected = metrics[metrics["calibration"] == "rns"].iloc[0]
        self.assertLess(corrected["rmse"], raw["rmse"])
        for _, group in calibrated.groupby("profile_id"):
            self.assertTrue(np.all(np.diff(group["prediction"].to_numpy()) >= 0.0))

    def test_paired_bootstrap_difference_has_declared_sign(self) -> None:
        target = np.arange(20, dtype=float)
        result = _paired_bootstrap_difference(
            target,
            target + 0.1,
            target + 1.0,
            seed=13,
            bootstrap_samples=200,
        )
        self.assertLess(result["rmse_difference_a_minus_b"], 0.0)
        self.assertEqual(result["probability_a_lower_rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
