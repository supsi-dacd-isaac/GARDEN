from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import jax.numpy as jnp

from neural_building_emulator.ventilation import (
    EPlusVentilationRollout,
    EPlusVentilationSchedule,
)

from neural_building_emulator_refactor.calibrated_model_benchmark import (
    ARM_ORDER,
    BenchmarkSplit,
    DEFAULT_CONFIG,
    _artifact_training_configuration,
    _apply_probabilistic_structured_buffer_capacity_override,
    _apply_probabilistic_structured_hp_energy_cap_override,
    _force_off_emulator_distribution,
    _force_off_dataset_path,
    _force_off_raw_only,
    _flexibility_calibration_summary,
    _paired_bootstrap_difference,
    _policy_profile_ids,
    _resolved_recipe_payload,
    _summarize_raw_distribution_table,
    _write_flexibility_point_scatter,
    create_benchmark_split,
    parse_args,
)
from neural_building_emulator_refactor.fair_probabilistic_lstm_comparison import (
    _experiment_configs,
)
from neural_building_emulator_refactor.probabilistic_calibration import (
    calibrate_distribution_table,
    ecc_reorder,
    ensemble_crps,
    summarize_joint_distribution_tables,
)
from neural_building_emulator_refactor.scoring import _scenario_extreme_traces


class CalibratedBenchmarkTests(unittest.TestCase):
    def test_artifact_configuration_is_authoritative_in_resolved_recipe(self) -> None:
        artifact = SimpleNamespace(
            backend="legacy",
            artifact_dir=Path("selected"),
            spec=SimpleNamespace(name="structured", probabilistic=True),
            metadata={},
            legacy_artifact=SimpleNamespace(
                metadata={
                    "train_config": {"prob_flex_kpi_crps_weight": 1.5},
                    "checkpoint_epoch": 10,
                    "checkpoint_metric": "test_total_nrmse",
                    "checkpoint_metric_value": 0.5,
                    "train_ids": [1],
                    "test_ids": [2],
                }
            ),
        )
        configuration, source = _artifact_training_configuration(artifact)
        self.assertEqual(configuration["prob_flex_kpi_crps_weight"], 1.5)
        self.assertEqual(source, "legacy_metadata.train_config")

        requested = {
            "seed": 13,
            **{arm: {"prob_flex_kpi_crps_weight": 0.0} for arm in ARM_ORDER},
        }
        split = BenchmarkSplit(
            dataset_path="dataset",
            seed=13,
            requested_max_profiles=2,
            eligible_count=2,
            selected_ids=(1, 2),
            train_ids=(1,),
            validation_ids=(2,),
            calibration_ids=(),
            test_ids=(),
            fingerprint="test",
        )
        with patch(
            "neural_building_emulator_refactor.calibrated_model_benchmark.load_artifact",
            return_value=artifact,
        ):
            payload = _resolved_recipe_payload(
                source=Path("recipe.json"),
                requested_recipe=requested,
                split=split,
                artifact_dirs={arm: Path(arm) for arm in ARM_ORDER},
            )

        self.assertEqual(payload["resolution_status"], "artifact_configs_authoritative")
        self.assertEqual(
            payload["recipe"]["probabilistic_structured"][
                "prob_flex_kpi_crps_weight"
            ],
            1.5,
        )
        self.assertEqual(
            payload["requested_recipe"]["probabilistic_structured"][
                "prob_flex_kpi_crps_weight"
            ],
            0.0,
        )
        self.assertEqual(
            payload["artifacts"]["probabilistic_structured"][
                "configuration_source"
            ],
            "legacy_metadata.train_config",
        )

    def test_flexibility_point_scatter_uses_raw_building_predictions(self) -> None:
        rows = []
        for method, offset in (
            ("lightgbm", 0.1),
            ("probabilistic_lstm", 0.2),
            ("probabilistic_structured", -0.1),
        ):
            for direction in ("up", "down"):
                rows.append(
                    {
                        "method": method,
                        "kpi_kind": "f",
                        "horizon_hours": 3.0,
                        "direction": direction,
                        "calibration": "raw",
                        "profile_id": 7,
                        "truth": 2.0,
                        "mean": 2.0 + offset,
                    }
                )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scatter.html"
            written = _write_flexibility_point_scatter(
                path,
                pd.DataFrame(rows),
                kpi_kind="f",
            )
            html = path.read_text()
        self.assertTrue(written)
        self.assertIn("Direct F: held-out building point predictions", html)
        self.assertIn("LightGBM", html)
        self.assertIn("Unstructured LSTM", html)
        self.assertIn("Structured", html)

    def test_flexibility_calibration_summary_keeps_both_kpis_separate(self) -> None:
        rows = []
        for kpi_kind, offset in (("f", 0.0), ("regression_based", 1.0)):
            for horizon, value in ((1.0, 0.1), (3.0, 0.3)):
                rows.append(
                    {
                        "method": "probabilistic_structured",
                        "calibration": "raw",
                        "kpi_kind": kpi_kind,
                        "horizon_hours": horizon,
                        "direction": "up",
                        "calibration_mae": offset + value,
                        "crps": offset + 2.0 * value,
                        "coverage_90": 0.8 + value / 10.0,
                        "width_90": offset + 3.0 * value,
                    }
                )
        summary = _flexibility_calibration_summary(pd.DataFrame(rows))
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(float(summary.iloc[0]["f_calibration_mae"]), 0.2)
        self.assertAlmostEqual(
            float(summary.iloc[0]["regression_based_calibration_mae"]),
            1.2,
        )

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

    def test_structured_buffer_capacity_override_is_exposed(self) -> None:
        args = parse_args(
            ["--probabilistic-structured-hp-energy-cap-hours", "0.5"]
        )
        self.assertEqual(
            args.probabilistic_structured_hp_energy_cap_hours,
            0.5,
        )
        recipe = {
            "probabilistic_structured": {
                "hp_energy_cap_hours": 8.0,
                "hp_energy_cap_wh_m2": 1862.0,
            }
        }
        _apply_probabilistic_structured_hp_energy_cap_override(
            recipe,
            args.probabilistic_structured_hp_energy_cap_hours,
        )
        self.assertEqual(
            recipe["probabilistic_structured"]["hp_energy_cap_hours"],
            0.5,
        )
        self.assertEqual(
            recipe["probabilistic_structured"]["hp_energy_cap_wh_m2"],
            0.0,
        )

    def test_physical_buffer_capacity_override_is_exposed(self) -> None:
        args = parse_args(
            [
                "--probabilistic-structured-buffer-capacity-mode",
                "physical_volume",
                "--probabilistic-structured-buffer-delta-t-usable-init-k",
                "5",
                "--probabilistic-structured-buffer-delta-t-min-k",
                "1",
                "--probabilistic-structured-buffer-delta-t-max-k",
                "15",
            ]
        )
        recipe = {
            "probabilistic_structured": {
                "hp_energy_cap_wh_m2": 1862.0,
            }
        }
        _apply_probabilistic_structured_buffer_capacity_override(
            recipe,
            capacity_mode=args.probabilistic_structured_buffer_capacity_mode,
            delta_t_usable_init_k=(
                args.probabilistic_structured_buffer_delta_t_usable_init_k
            ),
            delta_t_min_k=(
                args.probabilistic_structured_buffer_delta_t_min_k
            ),
            delta_t_max_k=(
                args.probabilistic_structured_buffer_delta_t_max_k
            ),
        )

        section = recipe["probabilistic_structured"]
        self.assertEqual(section["hp_buffer_capacity_mode"], "physical_volume")
        self.assertEqual(section["hp_energy_cap_wh_m2"], 0.0)
        self.assertEqual(section["hp_buffer_delta_t_usable_init_k"], 5.0)
        self.assertEqual(section["hp_buffer_delta_t_min_k"], 1.0)
        self.assertEqual(section["hp_buffer_delta_t_max_k"], 15.0)

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

    def test_force_off_is_raw_only_by_default(self) -> None:
        self.assertTrue(_force_off_raw_only({}))
        self.assertTrue(_force_off_raw_only({"force_off_evaluation": None}))
        self.assertFalse(
            _force_off_raw_only(
                {"force_off_evaluation": {"raw_only": False}}
            )
        )

    def test_force_off_raw_only_cli_override_is_exposed(self) -> None:
        self.assertTrue(parse_args(["--force-off-raw-only"]).force_off_raw_only)
        self.assertFalse(
            parse_args(["--no-force-off-raw-only"]).force_off_raw_only
        )

    def test_poor_excitation_overrides_are_exposed(self) -> None:
        args = parse_args(
            [
                "--evaluate-poor-excitation",
                "--poor-excitation-dataset",
                "poor_setpoints",
                "--poor-excitation-num-scenarios",
                "32",
                "--poor-excitation-num-profile-plots",
                "2",
            ]
        )
        self.assertTrue(args.evaluate_poor_excitation)
        self.assertEqual(args.poor_excitation_dataset, Path("poor_setpoints"))
        self.assertEqual(args.poor_excitation_num_scenarios, 32)
        self.assertEqual(args.poor_excitation_num_profile_plots, 2)

    def test_raw_distribution_summary_does_not_require_calibration_split(self) -> None:
        frame = pd.DataFrame(
            {
                "method": ["emulator"] * 4,
                "kpi_kind": ["force_off_f"] * 4,
                "horizon_hours": [1.0] * 4,
                "direction": ["force_off"] * 4,
                "profile_id": [1, 1, 2, 2],
                "truth": [1.0, 1.0, 3.0, 3.0],
                "prediction": [0.0, 2.0, 2.0, 4.0],
            }
        )
        metrics, reliability, buildings = _summarize_raw_distribution_table(frame)
        self.assertEqual(metrics["calibration"].tolist(), ["raw"])
        self.assertEqual(int(metrics.iloc[0]["count"]), 2)
        self.assertEqual(buildings["profile_id"].tolist(), [1, 2])
        self.assertFalse(reliability.empty)

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

    def test_rns_ecc_preserves_raw_joint_ranks(self) -> None:
        raw_rows = []
        rns_rows = []
        for profile_id in range(4):
            for direction, raw_values in (
                ("up", (-1.0, 1.0)),
                ("down", (2.0, -2.0)),
            ):
                truth = float(profile_id + (direction == "down"))
                for sample_index, raw_value in enumerate(raw_values):
                    common = {
                        "method": "emulator",
                        "kpi_kind": "f",
                        "horizon_hours": 1.0,
                        "direction": direction,
                        "profile_id": profile_id,
                        "truth": truth,
                        "sample_index": sample_index,
                    }
                    raw_rows.append({**common, "prediction": truth + raw_value})
                    rns_rows.append(
                        {
                            **common,
                            "prediction": truth + (-0.5 if sample_index == 0 else 0.5),
                        }
                    )
        raw = pd.DataFrame(raw_rows)
        rns = pd.DataFrame(rns_rows)
        ecc = ecc_reorder(raw, rns)
        scores = summarize_joint_distribution_tables(
            raw,
            {"raw": raw, "rns": rns, "rns_ecc": ecc},
            kpi_kinds=("f",),
        ).set_index("calibration")
        self.assertGreater(float(scores.loc["rns", "raw_rank_copula_rmse"]), 0.0)
        self.assertAlmostEqual(
            float(scores.loc["rns_ecc", "raw_rank_copula_rmse"]),
            0.0,
        )

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
