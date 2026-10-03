from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from neural_building_emulator.columns import METADATA_COLUMNS
from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator.flexibility_event_study_kpis import EventStudyConfig
from neural_building_emulator_refactor.metadata_flexibility_lightgbm import (
    _automatic_emulator_scoring_plan,
    _generate_emulator_kpi_csvs,
    _kpi_row,
    _load_emulator_predictions,
    _normalize_kpi_kinds,
    _resolve_additional_emulator_artifacts,
    _resolve_dataset_path,
    _resolve_emulator_kpi_csvs,
    _resolve_targets,
    paired_error_comparison,
)


class MetadataFlexibilityLightGBMTests(unittest.TestCase):
    def test_dataset_path_defaults_to_artifact_training_dataset(self) -> None:
        artifact = SimpleNamespace(
            backend="lstm",
            metadata={"experiment_config": {"dataset_path": "saved/dataset"}},
        )
        path, source = _resolve_dataset_path(artifact, None)
        self.assertEqual(path, Path("saved/dataset"))
        self.assertEqual(source, "artifact")

        path, source = _resolve_dataset_path(artifact, Path("override/dataset"))
        self.assertEqual(path, Path("override/dataset"))
        self.assertEqual(source, "override")

    def test_multiple_kpi_names_and_aliases_are_normalized_and_deduplicated(self) -> None:
        self.assertEqual(
            _normalize_kpi_kinds(("f", "regression", "direct_ratio")),
            ("f", "regression_based"),
        )

    def test_multiple_emulator_tables_require_explicit_kpi_labels(self) -> None:
        resolved = _resolve_emulator_kpi_csvs(
            ("f=direct.csv", "regression_based=regression.csv"),
            ("f", "regression_based"),
        )
        self.assertEqual(resolved["f"], Path("direct.csv"))
        self.assertEqual(
            resolved["regression_based"], Path("regression.csv")
        )
        with self.assertRaisesRegex(ValueError, "Bare"):
            _resolve_emulator_kpi_csvs(
                ("ambiguous.csv",),
                ("f", "regression_based"),
            )

    def test_additional_emulator_artifacts_require_unique_safe_labels(self) -> None:
        resolved = _resolve_additional_emulator_artifacts(
            ("vanilla_lstm=artifacts/lstm", "other-model=artifacts/other")
        )
        self.assertEqual(resolved["vanilla_lstm"], Path("artifacts/lstm"))
        with self.assertRaisesRegex(ValueError, "NAME=PATH"):
            _resolve_additional_emulator_artifacts(("artifacts/lstm",))
        with self.assertRaisesRegex(ValueError, "Reserved"):
            _resolve_additional_emulator_artifacts(("lightgbm=artifacts/lstm",))

    def test_automatic_emulator_scoring_reuses_one_regression_rollout(self) -> None:
        output = Path("generated")
        estimator, also_direct_f, paths = _automatic_emulator_scoring_plan(
            ("f", "regression_based"),
            output,
        )

        self.assertEqual(estimator, "regression")
        self.assertTrue(also_direct_f)
        self.assertEqual(paths["regression_based"], output / "flexibility_kpis.csv")
        self.assertEqual(paths["f"], output / "flexibility_kpis_f.csv")

    def test_automatic_emulator_scoring_uses_direct_ratio_for_f_only(self) -> None:
        output = Path("generated")
        estimator, also_direct_f, paths = _automatic_emulator_scoring_plan(
            ("f",),
            output,
        )

        self.assertEqual(estimator, "direct_ratio")
        self.assertFalse(also_direct_f)
        self.assertEqual(paths, {"f": output / "flexibility_kpis.csv"})

    @patch(
        "neural_building_emulator_refactor.metadata_flexibility_lightgbm.score_artifact"
    )
    def test_automatic_probabilistic_scoring_calls_artifact_scorer_once(
        self,
        scorer,
    ) -> None:
        def write_outputs(*_args, **kwargs) -> None:
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True)
            (output / "flexibility_kpis.csv").write_text("signal\n")
            (output / "flexibility_kpis_f.csv").write_text("signal\n")

        scorer.side_effect = write_outputs
        artifact = SimpleNamespace(spec=SimpleNamespace(probabilistic=True))
        args = SimpleNamespace(
            artifact_dir=Path("artifact"),
            emulator_num_scenarios=37,
            emulator_hp_scenario_mode="bernoulli",
            emulator_ventilation_rollout_mode="recorded",
            emulator_max_profiles=None,
            emulator_num_profile_plots=0,
            horizons_hours=[0.5, 1.0, 2.0, 3.0],
            dt_hours=0.25,
            min_setpoint_change=0.05,
            min_events=20,
            controls="full",
            seed=13,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "scoring"
            paths = _generate_emulator_kpi_csvs(
                args,
                artifact_dir=Path("artifact"),
                artifact=artifact,
                dataset_path=Path("dataset"),
                kpi_kinds=("f", "regression_based"),
                output_dir=output,
            )

        scorer.assert_called_once()
        call = scorer.call_args.kwargs
        self.assertEqual(call["kpi_mode"], "scenario_average")
        self.assertEqual(call["num_scenarios"], 37)
        self.assertEqual(call["kpi_estimator"], "regression")
        self.assertTrue(call["evaluate_direct_f"])
        self.assertEqual(set(paths), {"f", "regression_based"})

    def test_target_resolution_deduplicates_equivalent_discrete_horizons(self) -> None:
        config, targets = _resolve_targets((0.5, 0.51, 1.0), 0.25)

        self.assertEqual(config.horizon_steps, (2, 4))
        self.assertEqual(len(targets), 4)
        self.assertEqual(targets[0].key, "up_flex_0p5h_wh_m2_k")
        self.assertEqual(targets[-1].key, "down_flex_1h_wh_m2_k")

    def test_simulated_kpi_extraction_recovers_known_setpoint_gain(self) -> None:
        block_steps = 12
        block_count = 12
        setpoint = np.repeat(
            np.asarray([20.0, 21.0] * (block_count // 2), dtype=np.float32),
            block_steps,
        )
        steps = len(setpoint)
        inputs = np.zeros((steps, 9), dtype=np.float32)
        inputs[:, 0] = setpoint
        pel = 5.0 + 2.0 * (setpoint - 20.0)
        targets = np.column_stack(
            [
                np.full(steps, 20.0, dtype=np.float32),
                np.zeros(steps, dtype=np.float32),
                pel,
            ]
        )
        profile = ClosedLoopProfile(
            profile_id=7,
            datetime=np.arange(steps),
            metadata=np.zeros(len(METADATA_COLUMNS), dtype=np.float32),
            inputs=inputs,
            targets=targets,
            initial_temperature=np.full((steps, 1), 20.0, dtype=np.float32),
        )
        config = EventStudyConfig(
            horizons_hours=(3.0,),
            horizon_steps=(block_steps,),
            dt_hours=0.25,
            min_setpoint_change=0.05,
            min_events=4,
            controls="none",
        )

        row = _kpi_row(profile, "train", config)

        self.assertAlmostEqual(row["up_flex_3h_wh_m2_k"], 6.0, places=5)
        self.assertAlmostEqual(row["down_flex_3h_wh_m2_k"], 6.0, places=5)

    def test_paired_bootstrap_detects_uniformly_better_predictions(self) -> None:
        target = np.linspace(0.0, 10.0, 50)
        comparison = paired_error_comparison(
            target,
            target,
            target + 1.0,
            seed=13,
            bootstrap_samples=500,
        )

        self.assertLess(comparison["rmse_difference"], 0.0)
        self.assertEqual(comparison["probability_lightgbm_lower_rmse"], 1.0)
        self.assertEqual(comparison["probability_lightgbm_lower_mae"], 1.0)

    def test_refactor_scorer_kpi_table_is_converted_to_comparison_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "flexibility_kpis.csv"
            pd.DataFrame(
                {
                    "signal": ["simulation", "emulation"],
                    "profile_id": [7, 7],
                    "horizon_hours": [3.0, 3.0],
                    "up_flex_wh_m2_k": [2.0, 1.5],
                    "down_flex_wh_m2_k": [3.0, 2.5],
                }
            ).to_csv(path, index=False)

            converted = _load_emulator_predictions(path)

        self.assertEqual(converted.loc[0, "profile_id"], 7)
        self.assertEqual(converted.loc[0, "sim_up_flex_wh_m2_k"], 2.0)
        self.assertEqual(converted.loc[0, "emu_down_flex_wh_m2_k"], 2.5)


if __name__ == "__main__":
    unittest.main()
