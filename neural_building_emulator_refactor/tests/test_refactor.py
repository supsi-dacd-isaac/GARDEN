from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd

from neural_building_emulator.data import ClosedLoopProfile, select_and_split_profile_ids
from neural_building_emulator.scaling import StandardScaler, WindowScalers
from neural_building_emulator.train import (
    checkpoint_metric_value,
    resolve_checkpoint_metric,
)
from neural_building_emulator_refactor.artifacts import load_artifact, save_lstm_artifact
from neural_building_emulator_refactor.config import ExperimentConfig, LSTMConfig
from neural_building_emulator_refactor.deterministic_scaling_study import (
    MODEL_CLOSED_LOOP,
    MODEL_Q_TO_T,
    _recommendations,
    _study_runs,
)
from neural_building_emulator_refactor.fast_kpis import (
    FastKpiConfig,
    increment_spectral_js_distance,
    raw_flexibility_metrics,
    trajectory_metrics,
)
from neural_building_emulator_refactor.models import AutoregressiveLSTM
from neural_building_emulator_refactor.optuna_deterministic_study import (
    ARCHITECTURE_FAMILIES,
    BASELINE_PARAMS,
    _aggregate_runs,
    _common_params,
    _enqueue_architecture_quotas,
    _objective_components,
    _suggest_params,
    _trial_overrides,
    _write_weight_sensitivity,
)
from neural_building_emulator_refactor.registry import MODEL_REGISTRY, get_model_spec
from neural_building_emulator_refactor.legacy import build_legacy_config
from neural_building_emulator_refactor.train import (
    _run_post_training_flexibility,
    _validate_post_training_flexibility,
    config_from_args,
    parse_args,
)


class RegistryTests(unittest.TestCase):
    def test_original_six_and_two_lstm_models_are_registered(self) -> None:
        self.assertEqual(
            set(MODEL_REGISTRY),
            {
                "q_to_t_deterministic_ss",
                "q_to_t_probabilistic_ss",
                "closed_loop_hp_deterministic_ss",
                "closed_loop_hp_probabilistic_ss",
                "closed_loop_hp_contracting_deterministic",
                "closed_loop_hp_contracting_probabilistic",
                "q_to_t_lstm",
                "closed_loop_hp_lstm",
            },
        )

    def test_exposed_legacy_command_builds_expected_config(self) -> None:
        args = parse_args(
            [
                "--dataset", "data.parquet",
                "--max-profiles", "1000",
                "--test-fraction", "0.2",
                "--epochs", "15",
                "--batch-size", "32",
                "--learning-rate", "5e-4",
                "--input-encoder-dim", "8",
                "--input-encoder-hidden-dim", "64",
                "--input-encoder-depth", "2",
                "--sequence-length", "960",
                "--stride", "2024",
                "--rotate-window-starts",
                "--state-dim", "8",
                "--hidden-dim", "64",
                "--depth", "3",
                "--target-mode", "absolute",
                "--model-kind", "closed_loop_hp_contracting_probabilistic",
                "--contracting-gamma", "0.99",
                "--contracting-state-bound", "5",
                "--contracting-temperature-scale", "8",
                "--contracting-temperature-delta-max-c", "2",
                "--contracting-temperature-update", "leaky_equilibrium",
                "--contracting-transition-conditioning", "exogenous_additive_feedback",
                "--contracting-additive-feedback-gain-bound", "0.75",
                "--bptt-truncate-steps", "32",
                "--hp-thermostat-demand-mode", "monotone",
                "--prob-particles", "24",
                "--prob-eval-particles", "16",
                "--prob-plot-particles", "100",
                "--prob-latent-dim", "4",
                "--prob-process-noise", "heteroscedastic",
                "--prob-process-noise-init", "-4",
                "--prob-hp-emission-mode", "bounded",
                "--prob-hp-training-mode", "expected",
                "--prob-hp-activation-model", "independent",
                "--prob-hp-scenario-mode", "bernoulli",
                "--prob-softopt-weight", "2",
                "--prob-variogram-weight", "1",
                "--prob-variogram-lags", "1", "2", "4", "8",
                "--prob-variogram-output-weights", "1", "0", "0",
                "--prob-variogram-constant-setpoint-only",
                "--prob-variogram-setpoint-threshold-c", "0.05",
                "--prob-ires-weight", "1",
                "--hp-inactive-leakage-weight", "0",
                "--hp-cop-cap", "8",
                "--gradient-clip-norm", "1",
                "--skip-grad-norm-above", "1e3",
                "--train-eval-max-windows", "64",
                "--full-train-eval-every-epochs", "0",
                "--num-window-plots", "0",
                "--num-full-profile-plots", "5",
                "--save-model",
                "--output-dir", "output/test",
                "--seed", "13",
                "--contracting-q-to-t-mode", "positive_leaky",
                "--contracting-q-to-t-time-constants-hours", "1", "24",
                "--contracting-q-to-t-gain-min-c-per-w-m2", "0.01",
                "--contracting-q-to-t-gain-max-c-per-w-m2", "2",
                "--log-update-diagnostics",
            ]
        )
        experiment = config_from_args(args)
        config = build_legacy_config(
            experiment,
            get_model_spec(experiment.model_name),
        )
        self.assertEqual(experiment.model_name, "closed_loop_hp_contracting_probabilistic")
        self.assertEqual(config.max_profiles, 1000)
        self.assertEqual(config.learning_rate, 5e-4)
        self.assertEqual(config.prob_particles, 24)
        self.assertEqual(
            config.contracting_transition_conditioning,
            "exogenous_additive_feedback",
        )
        self.assertEqual(config.contracting_additive_feedback_gain_bound, 0.75)
        self.assertEqual(config.contracting_q_to_t_mode, "positive_leaky")
        self.assertEqual(config.contracting_q_to_t_time_constants_hours, (1.0, 24.0))
        self.assertTrue(config.prob_variogram_constant_setpoint_only)
        self.assertTrue(config.log_update_diagnostics)

    def test_post_training_flexibility_preset_uses_requested_defaults(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "data.parquet",
                "--output-dir",
                "output/test",
                "--model",
                "closed_loop_hp_contracting_probabilistic",
                "--evaluate-flexibility-after-training",
            ]
        )
        config = config_from_args(args)
        _validate_post_training_flexibility(args, config)
        artifact_dir = Path("output/test/artifacts/model/selected")

        with patch(
            "neural_building_emulator.flexibility_event_study_kpis.run_analysis"
        ) as run_analysis:
            output_dir = _run_post_training_flexibility(args, config, artifact_dir)

        evaluation_args = run_analysis.call_args.args[0]
        self.assertEqual(evaluation_args.artifact_dir, artifact_dir)
        self.assertEqual(evaluation_args.profile_source, "test")
        self.assertEqual(evaluation_args.max_profiles, 100)
        self.assertEqual(evaluation_args.emulation_kpi_mode, "scenario_average")
        self.assertEqual(evaluation_args.num_scenarios, 100)
        self.assertEqual(evaluation_args.prob_hp_scenario_mode, "bernoulli")
        self.assertEqual(evaluation_args.prob_hp_emission_mode, "artifact")
        self.assertEqual(evaluation_args.horizons_hours, "0.5,1,2,3")
        self.assertEqual(evaluation_args.controls, "full")
        self.assertEqual(evaluation_args.min_setpoint_change, 0.05)
        self.assertEqual(evaluation_args.min_events, 20)
        self.assertEqual(evaluation_args.seed, 13)
        self.assertEqual(evaluation_args.kpi_estimator, "regression")
        self.assertEqual(evaluation_args.ventilation_rollout_mode, "eplus_rule")
        self.assertEqual(
            output_dir,
            Path("output/test/flexibility_event_study_scenarios_regressor"),
        )

    def test_post_training_flexibility_rejects_incompatible_model(self) -> None:
        args = parse_args(
            [
                "--dataset",
                "data.parquet",
                "--model",
                "q_to_t_probabilistic_ss",
                "--evaluate-flexibility-after-training",
            ]
        )
        with self.assertRaisesRegex(ValueError, "currently requires"):
            _validate_post_training_flexibility(args, config_from_args(args))


class LSTMTests(unittest.TestCase):
    def _model(self, output_dim: int) -> AutoregressiveLSTM:
        return AutoregressiveLSTM(
            metadata_dim=3,
            input_dim=4,
            output_dim=output_dim,
            hidden_dim=8,
            metadata_hidden_dim=8,
            metadata_depth=1,
            bptt_truncate_steps=0,
            key=jax.random.PRNGKey(3),
        )

    def test_q_to_t_shape(self) -> None:
        prediction = self._model(1)(jnp.zeros(3), jnp.zeros((12, 4)), jnp.zeros(1))
        self.assertEqual(prediction.shape, (12, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(prediction))))

    def test_closed_loop_shape(self) -> None:
        prediction = self._model(3)(jnp.zeros(3), jnp.zeros((12, 4)), jnp.zeros(3))
        self.assertEqual(prediction.shape, (12, 3))

    def test_artifact_round_trip(self) -> None:
        model = self._model(1)
        scaler = StandardScaler(
            mean=np.zeros(1, dtype=np.float32),
            scale=np.ones(1, dtype=np.float32),
        )
        scalers = WindowScalers(
            metadata=StandardScaler(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
            inputs=StandardScaler(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)),
            target=scaler,
        )
        config = ExperimentConfig(
            model_name="q_to_t_lstm",
            lstm=LSTMConfig(
                hidden_dim=8,
                metadata_hidden_dim=8,
                metadata_depth=1,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            save_lstm_artifact(
                path,
                model=model,
                scalers=scalers,
                spec=get_model_spec("q_to_t_lstm"),
                experiment_config=config,
                input_columns=["a", "b", "c", "d"],
                metadata_columns=["x", "y", "z"],
                target_columns=["temperature"],
                selected_ids=[1, 2],
                train_ids=[1],
                test_ids=[2],
                checkpoint_epoch=1,
                checkpoint_metric_value=1.0,
                train_metrics={},
                test_metrics={},
            )
            loaded = load_artifact(path)
            expected = model(jnp.zeros(3), jnp.zeros((5, 4)), jnp.zeros(1))
            actual = loaded.model(jnp.zeros(3), jnp.zeros((5, 4)), jnp.zeros(1))
            np.testing.assert_allclose(actual, expected)


class FastKpiTests(unittest.TestCase):
    def test_identical_trajectory_has_zero_distances(self) -> None:
        time = np.arange(96 * 4, dtype=np.float64)
        target = np.column_stack(
            [
                20.0 + np.sin(2.0 * np.pi * time / 96.0),
                np.maximum(0.0, 10.0 * np.sin(2.0 * np.pi * time / 24.0)),
                np.maximum(0.0, 4.0 * np.sin(2.0 * np.pi * time / 24.0)),
            ]
        )
        metrics = trajectory_metrics(target, target, np.ones(3), FastKpiConfig())
        self.assertEqual(metrics["total_nrmse"], 0.0)
        self.assertAlmostEqual(metrics["temperature_acf_mae"], 0.0)
        self.assertAlmostEqual(metrics["pel_increment_spectral_js"], 0.0)

    def test_spectral_distance_detects_wrong_frequency(self) -> None:
        time = np.arange(96 * 14, dtype=np.float64)
        slow = np.sin(2.0 * np.pi * time / 96.0)
        fast = np.sin(2.0 * np.pi * time / 8.0)
        self.assertGreater(increment_spectral_js_distance(fast, slow, 0.25), 0.5)

    def test_raw_flexibility_recovers_three_hour_gain(self) -> None:
        steps = 96
        setpoint = np.full(steps, 20.0)
        setpoint[24:48] = 21.0
        setpoint[48:72] = 20.0
        setpoint[72:] = 21.0
        power = 5.0 * (setpoint - 20.0)
        inputs = np.zeros((steps, 8), dtype=np.float32)
        inputs[:, 0] = setpoint
        targets = np.column_stack([np.full(steps, 20.0), power * 2.0, power]).astype(np.float32)
        profile = ClosedLoopProfile(
            profile_id=1,
            datetime=np.arange(steps),
            metadata=np.zeros(2),
            inputs=inputs,
            targets=targets,
            initial_temperature=np.asarray([20.0]),
        )
        metrics, _ = raw_flexibility_metrics(
            profile, targets, targets, 1.0, FastKpiConfig(event_horizon_hours=3.0)
        )
        self.assertAlmostEqual(metrics["sim_up_power_gain_w_m2_k"], 5.0)
        self.assertAlmostEqual(metrics["sim_down_power_gain_w_m2_k"], 5.0)
        self.assertAlmostEqual(metrics["flex_event_delta_p_rmse_w_m2"], 0.0)


class ScalingStudyTests(unittest.TestCase):
    def test_fixed_holdout_is_identical_and_training_sets_are_nested(self) -> None:
        profile_ids = tuple(range(100))
        fixed = (90, 91, 92, 93, 94)
        selected_small, train_small, test_small = select_and_split_profile_ids(
            profile_ids,
            max_profiles=20,
            test_fraction=0.2,
            seed=13,
            strategy="random",
            fixed_test_profile_ids=fixed,
        )
        selected_large, train_large, test_large = select_and_split_profile_ids(
            profile_ids,
            max_profiles=50,
            test_fraction=0.2,
            seed=13,
            strategy="random",
            fixed_test_profile_ids=fixed,
        )
        self.assertEqual(test_small, fixed)
        self.assertEqual(test_large, fixed)
        self.assertEqual(len(selected_small), 20)
        self.assertEqual(len(selected_large), 50)
        self.assertTrue(set(train_small).issubset(train_large))

    def test_excluded_profiles_never_enter_selected_split(self) -> None:
        selected, train_ids, test_ids = select_and_split_profile_ids(
            tuple(range(30)),
            max_profiles=20,
            test_fraction=0.2,
            seed=13,
            strategy="random",
            fixed_test_profile_ids=(0, 1, 2),
            excluded_profile_ids=(3, 4, 5),
        )
        self.assertEqual(test_ids, (0, 1, 2))
        self.assertEqual(len(selected), 20)
        self.assertFalse(set(selected).intersection((3, 4, 5)))
        self.assertFalse(set(train_ids).intersection((0, 1, 2, 3, 4, 5)))


class OptunaDeterministicStudyTests(unittest.TestCase):
    def test_closed_loop_auto_checkpoint_uses_equal_channel_nrmse(self) -> None:
        metric = resolve_checkpoint_metric(
            "auto",
            has_test_windows=True,
            prefer_total_nrmse=True,
        )
        self.assertEqual(metric, "test_total_nrmse")
        value = checkpoint_metric_value(
            metric,
            {"total_nrmse": 0.8},
            {"total_nrmse": 0.6},
        )
        self.assertAlmostEqual(value, 0.6)

    def test_q_to_t_auto_checkpoint_remains_temperature_rmse(self) -> None:
        metric = resolve_checkpoint_metric("auto", has_test_windows=True)
        self.assertEqual(metric, "test_rmse_c")
        value = checkpoint_metric_value(
            metric,
            {"rmse_c": 1.2},
            {"rmse_c": 0.9},
        )
        self.assertAlmostEqual(value, 0.9)

    def test_baseline_normalization_produces_unit_objective(self) -> None:
        metrics = {
            "total_nrmse_mean": 0.8,
            "temperature_acf_mae_mean": 0.2,
            "temperature_increment_spectral_js_mean": 0.4,
            "flex_event_delta_p_nrmse_mean": 0.6,
        }
        components = _objective_components(
            metrics,
            {
                "trajectory": 0.8,
                "temperature_acf": 0.2,
                "temperature_spectrum": 0.4,
                "flexibility": 0.6,
            },
            {"trajectory": 0.5, "temporal": 0.2, "flexibility": 0.3},
        )
        self.assertAlmostEqual(components["trajectory_ratio"], 1.0)
        self.assertAlmostEqual(components["temporal_ratio"], 1.0)
        self.assertAlmostEqual(components["flexibility_ratio"], 1.0)
        self.assertAlmostEqual(components["objective"], 1.0)

    def test_trial_overrides_keep_validation_and_final_test_disjoint(self) -> None:
        overrides = _trial_overrides(
            {
                "state_dim": 8,
                "controller_state_dim": 2,
                "hidden_dim": 64,
                "encoder_dim": 8,
                "controller_calendar_features": True,
                "thermostat_demand_mode": "monotone",
                "temperature_delta_max_c": 2.0,
                "q_to_t_mode": "positive_leaky",
                "qroom_loss_weight": 1.0,
                "pel_loss_weight": 1.0,
                "hp_mode_loss_weight": 0.1,
                "q_to_t_time_constants": "1,24",
                "q_to_t_gain_max": 2.0,
            },
            [10, 11],
            [20, 21],
            64,
        )
        self.assertEqual(overrides["fixed_test_profile_ids"], [10, 11])
        self.assertEqual(overrides["excluded_profile_ids"], [20, 21])
        self.assertFalse(
            set(overrides["fixed_test_profile_ids"]).intersection(
                overrides["excluded_profile_ids"]
            )
        )

    def test_architecture_quota_enqueues_equal_family_coverage(self) -> None:
        study = optuna.create_study()
        study.enqueue_trial(BASELINE_PARAMS, user_attrs={"is_baseline": True})
        queued = _enqueue_architecture_quotas(study, 3)
        self.assertEqual(len(queued), len(ARCHITECTURE_FAMILIES) * 3 - 1)
        counts: dict[tuple[object, ...], int] = {}
        keys = (
            "thermostat_demand_mode",
            "q_to_t_mode",
            "controller_calendar_features",
        )
        for trial in study.trials:
            fixed = trial.system_attrs["fixed_params"]
            family = tuple(fixed[key] for key in keys)
            counts[family] = counts.get(family, 0) + 1
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(len(counts), len(ARCHITECTURE_FAMILIES))
        self.assertEqual(_enqueue_architecture_quotas(study, 3), [])

    def test_fixed_learning_rate_is_recorded_and_excluded_from_importance(self) -> None:
        study = optuna.create_study()

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, fixed_learning_rate=8.5e-4)
            return float(params["state_dim"])

        study.optimize(objective, n_trials=2)
        for trial in study.trials:
            self.assertEqual(trial.params["learning_rate"], 8.5e-4)
        self.assertNotIn("learning_rate", _common_params(study))

    def test_confirmation_aggregation_reports_seed_mean_and_spread(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate": "baseline",
                    "search_trial": 0,
                    "seed": 13,
                    "artifact_dir": "a",
                    "objective": 1.0,
                    "temperature_nrmse_mean": 0.5,
                },
                {
                    "candidate": "baseline",
                    "search_trial": 0,
                    "seed": 29,
                    "artifact_dir": "b",
                    "objective": 1.0,
                    "temperature_nrmse_mean": 0.7,
                },
                {
                    "candidate": "trial_0004",
                    "search_trial": 4,
                    "seed": 13,
                    "artifact_dir": "c",
                    "objective": 0.8,
                    "temperature_nrmse_mean": 0.4,
                },
                {
                    "candidate": "trial_0004",
                    "search_trial": 4,
                    "seed": 29,
                    "artifact_dir": "d",
                    "objective": 1.0,
                    "temperature_nrmse_mean": 0.6,
                },
            ]
        )
        summary = _aggregate_runs(frame)
        winner = summary.iloc[0]
        self.assertEqual(winner["candidate"], "trial_0004")
        self.assertAlmostEqual(winner["objective_mean"], 0.9)
        self.assertGreater(winner["objective_std"], 0.0)

    def test_weight_sensitivity_writes_csv_and_interactive_html(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "trial": 0,
                    "architecture": "monotone | positive_leaky | calendar=True",
                    "trajectory_ratio": 1.0,
                    "temporal_ratio": 1.0,
                    "flexibility_ratio": 1.0,
                },
                {
                    "trial": 1,
                    "architecture": "unconstrained | unconstrained | calendar=False",
                    "trajectory_ratio": 0.8,
                    "temporal_ratio": 1.1,
                    "flexibility_ratio": 1.2,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            html_path = Path(directory) / "weights.html"
            csv_path = Path(directory) / "weights.csv"
            _write_weight_sensitivity(
                frame,
                html_path,
                csv_path,
                {"trajectory": 0.5, "temporal": 0.2, "flexibility": 0.3},
            )
            self.assertTrue(html_path.exists())
            written = pd.read_csv(csv_path)
            self.assertIn("objective_selected", written.columns)
            self.assertIn("rank_flexibility_heavy", written.columns)


class ScalingDesignTests(unittest.TestCase):
    def test_compact_design_has_seven_runs_per_model(self) -> None:
        args = argparse.Namespace(
            models=(MODEL_Q_TO_T, MODEL_CLOSED_LOOP),
            profile_counts=(100, 300, 1000),
            strides=(2024, 4048, 8096),
            design="compact",
        )
        runs = _study_runs(args, {MODEL_Q_TO_T: 1000, MODEL_CLOSED_LOOP: 963})
        self.assertEqual(len(runs), 14)
        closed = [run for run in runs if run.model_name == MODEL_CLOSED_LOOP]
        pairs = [(run.profile_count, run.stride) for run in closed]
        self.assertIn((300, 2024), pairs)
        self.assertIn((300, 8096), pairs)
        self.assertIn((963, 2024), pairs)
        self.assertIn((963, 4048), pairs)
        self.assertIn((963, 8096), pairs)

    def test_closed_loop_recommendation_preserves_flexibility_quality(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model_name": MODEL_CLOSED_LOOP,
                    "profile_count": 963,
                    "stride": 2024,
                    "training_wall_minutes": 6.0,
                    "total_nrmse_mean": 0.55,
                    "flex_event_delta_p_nrmse_mean": 0.37,
                    "flex_up_energy_gain_abs_error_wh_m2_k_mean": 8.2,
                    "flex_down_energy_gain_abs_error_wh_m2_k_mean": 4.7,
                },
                {
                    "model_name": MODEL_CLOSED_LOOP,
                    "profile_count": 300,
                    "stride": 2024,
                    "training_wall_minutes": 2.0,
                    "total_nrmse_mean": 0.59,
                    "flex_event_delta_p_nrmse_mean": 0.42,
                    "flex_up_energy_gain_abs_error_wh_m2_k_mean": 13.2,
                    "flex_down_energy_gain_abs_error_wh_m2_k_mean": 9.3,
                },
                {
                    "model_name": MODEL_CLOSED_LOOP,
                    "profile_count": 963,
                    "stride": 8096,
                    "training_wall_minutes": 3.0,
                    "total_nrmse_mean": 0.59,
                    "flex_event_delta_p_nrmse_mean": 0.39,
                    "flex_up_energy_gain_abs_error_wh_m2_k_mean": 9.1,
                    "flex_down_energy_gain_abs_error_wh_m2_k_mean": 5.6,
                },
            ]
        )
        recommendation = _recommendations(frame, tolerance=0.10).iloc[0]
        self.assertEqual(int(recommendation["recommended_profile_count"]), 963)
        self.assertEqual(int(recommendation["recommended_stride"]), 8096)


if __name__ == "__main__":
    unittest.main()
