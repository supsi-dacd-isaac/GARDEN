from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import json
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.train import flexibility_direct_ratio_crps
from neural_building_emulator_refactor.config import ExperimentConfig, OptimizerConfig, ProbabilisticLSTMConfig
from neural_building_emulator_refactor.models.lstm import (
    HurdleProbabilisticAutoregressiveLSTM, VanillaProbabilisticAutoregressiveLSTM,
)
from neural_building_emulator_refactor.probabilistic_lstm_trainer import (
    _flex_loss, _sample_batch_with_aux, _make_train_step, build_trajectory_energy_score_context,
    _validate_flex_training_windows,
)
from neural_building_emulator_refactor.trainer import _build_optimizer
from neural_building_emulator_refactor.lstm_hurdle_ablation import (
    _current_best_base, _variant_recipe, parse_args,
)


def small_model(hurdle=False, noise="heteroscedastic"):
    common = dict(metadata_dim=3, input_dim=5, hidden_dim=8, metadata_hidden_dim=8,
                  metadata_depth=1, latent_dim=2, process_noise_mode=noise,
                  process_noise_init=-3., bptt_truncate_steps=4,
                  setpoint_input_index=0, input_scale=(1.,) * 5,
                  key=jax.random.PRNGKey(91))
    if hurdle:
        return HurdleProbabilisticAutoregressiveLSTM(
            **common, activation_model="asymmetric_markov", availability_input_index=3,
            force_off_input_index=4, input_mean=(0.,) * 5,
            target_mean=(0.,) * 3, target_scale=(1.,) * 3,
        )
    return VanillaProbabilisticAutoregressiveLSTM(**common)


class CurrentBestLSTMTests(unittest.TestCase):
    def test_hard_forward_matches_bernoulli_and_has_switching_gradients(self):
        model = small_model(hurdle=True)
        inputs = jnp.zeros((16, 5)).at[:, 3].set(1.).at[8:, 4].set(1.)
        kwargs = dict(key=jax.random.PRNGKey(94), num_particles=8, sample_process_noise=True)
        hard = model.sample_with_aux(jnp.zeros(3), inputs, jnp.zeros(3),
                                     hp_scenario_mode="hard_expected_grad", **kwargs)[0]
        sampled = model.sample_with_aux(jnp.zeros(3), inputs, jnp.zeros(3),
                                        hp_scenario_mode="bernoulli", **kwargs)[0]
        np.testing.assert_array_equal(hard, sampled)
        np.testing.assert_array_equal(hard[:, 8:, 2], 0.)

        def power(m):
            return m.sample_with_aux(jnp.zeros(3), inputs, jnp.zeros(3),
                                     hp_scenario_mode="hard_expected_grad", **kwargs)[0][..., 2].sum()
        grad = eqx.filter_grad(power)(model)
        self.assertTrue(np.isfinite(np.asarray(grad.readout.bias)).all())
        self.assertGreater(float(jnp.linalg.norm(grad.readout.bias[2:4])), 0.)

    def test_grouped_latents_are_shared_but_fast_noise_is_not(self):
        inputs = jnp.zeros((4, 16, 5))
        kwargs = dict(num_particles=3, hp_scenario_mode="expected", persistent_xi_group_size=2)
        outputs, _ = _sample_batch_with_aux(small_model(noise="none"), jnp.zeros((4, 3)),
                                            inputs, jnp.zeros((4, 3)), jax.random.PRNGKey(90), **kwargs)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[2], outputs[3])
        self.assertFalse(np.allclose(outputs[0], outputs[2]))
        noisy, _ = _sample_batch_with_aux(small_model(), jnp.zeros((4, 3)), inputs,
                                         jnp.zeros((4, 3)), jax.random.PRNGKey(90), **kwargs)
        self.assertFalse(np.allclose(noisy[0], noisy[1]))
        with self.assertRaises(ValueError):
            _sample_batch_with_aux(small_model(), jnp.zeros((3, 3)), inputs[:3],
                                   jnp.zeros((3, 3)), jax.random.PRNGKey(90), **kwargs)

    def test_shared_kpi_loss_equals_structured_function(self):
        settings = ProbabilisticLSTMConfig(flex_kpi_crps_weight=1.5,
                                          flex_kpi_crps_windows_per_profile=2,
                                          flex_kpi_crps_min_events=1,
                                          flex_kpi_crps_horizons_hours=(0.5, 1.))
        inputs = np.zeros((2, 32, 5), dtype=np.float32)
        inputs[:, :, 0] = (np.arange(32) // 4) % 2
        targets = np.zeros((2, 32, 3), dtype=np.float32)
        targets[:, :, 2] = 1. + 2. * inputs[:, :, 0]
        predictions = jnp.asarray(np.repeat(targets[:, None], 3, axis=1))
        context = build_trajectory_energy_score_context(settings, targets=targets, inputs=inputs,
                                                         setpoint_input_scale=1., sequence_length=32)
        windows = SimpleNamespace(inputs=inputs, profile_ids=np.array([7, 7]))
        self.assertEqual(_validate_flex_training_windows(windows, settings, context), (1, 1))
        with self.assertRaises(ValueError):
            _validate_flex_training_windows(SimpleNamespace(inputs=np.zeros_like(inputs),
                                                             profile_ids=windows.profile_ids), settings, context)
        # The shared training CRPS smooths absolute values with sqrt(x*x + 1e-6).
        self.assertAlmostEqual(float(_flex_loss(predictions, targets, inputs, settings, context)),
                               1.5 * .0005, places=6)
        predictions = predictions.at[..., 2].multiply(0.5)
        expected = 1.5 * flexibility_direct_ratio_crps(
            predictions, jnp.asarray(targets), jnp.asarray(inputs), horizon_steps=(2, 4),
            horizon_hours=(0.5, 1.), setpoint_delta_threshold=.05, min_events=1,
            target_channel=2, windows_per_profile=2,
        )
        actual = _flex_loss(predictions, jnp.asarray(targets), jnp.asarray(inputs), settings, context)
        self.assertGreater(float(actual), 0.)
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_both_grouped_training_steps_are_finite(self):
        inputs = np.zeros((2, 32, 5), dtype=np.float32)
        inputs[:, :, 0] = (np.arange(32) // 4) % 2
        inputs[:, :, 3] = 1.
        targets = np.zeros((2, 32, 3), dtype=np.float32)
        targets[:, :, 2] = 1. + inputs[:, :, 0]
        for hurdle in (False, True):
            model = small_model(hurdle=hurdle)
            settings = ProbabilisticLSTMConfig(
                particles=3, hp_training_mode="hard_expected_grad", flex_kpi_crps_weight=1.5,
                flex_kpi_crps_min_events=1, flex_kpi_crps_windows_per_profile=2,
                flex_kpi_crps_horizons_hours=(0.5, 1.), process_regularization_weight=.1,
                variogram_output_weights=(1., 1., 0.), setpoint_shock_timescales_hours=(),
            )
            config = ExperimentConfig(model_name="closed_loop_hp_hurdle_markov_probabilistic_lstm"
                                      if hurdle else "closed_loop_hp_vanilla_probabilistic_lstm",
                                      optimizer=OptimizerConfig(batch_size=2), probabilistic_lstm=settings)
            config.validate()
            context = build_trajectory_energy_score_context(settings, targets=targets, inputs=inputs,
                                                             setpoint_input_scale=1., sequence_length=32)
            optimizer = _build_optimizer(config)
            state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
            step = _make_train_step(optimizer, config, vanilla=not hurdle, score_context=context)
            _, _, loss, components, grad_norm, accepted = step(
                model, state, jnp.zeros((2, 3)), jnp.asarray(inputs), jnp.zeros((2, 3)),
                jnp.asarray(targets), jax.random.PRNGKey(120),
            )
            self.assertTrue(np.isfinite(float(loss)))
            self.assertTrue(np.isfinite(np.asarray(components)).all())
            self.assertGreater(float(components[-2]), 0.)
            self.assertTrue(np.isfinite(float(grad_norm)))
            self.assertTrue(bool(accepted))

    def test_current_best_recipe_is_authoritative_and_preserves_requested_budget(self):
        source = {"probabilistic_lstm": {}, "optimizer": {"epochs": 5, "batch_size": 32},
                  "evaluation": {}}
        config = {
            "hp_energy_cap_hours": .5, "prob_hp_activation_model": "asymmetric_markov",
            "prob_flex_kpi_crps_estimator": "direct_ratio", "prob_flex_kpi_crps_aux_every_steps": 0,
            "prob_flex_kpi_crps_weight": 1.5, "prob_flex_kpi_crps_windows_per_profile": 4,
            "prob_hp_training_mode": "hard_expected_grad", "prob_physics_weight": .1,
            "dataset_path": "train", "additional_dataset_paths": ["force_off"],
            "sequence_length": 960, "stride": 2024, "rotate_window_starts": True,
            "seed": 13, "hp_power_area_normalization": "building_heated_area", "epochs": 10,
            "batch_size": 32, "bptt_truncate_steps": 32, "skip_grad_norm_above": 1000.,
            "hp_mode_loss_weight": .1, "hp_active_power_nll_weight": 1., "hp_inactive_leakage_weight": .1,
            "closed_loop_trajectory_output_weights": [1., 1., 1.],
        }
        args = parse_args(["--current-best-root", "unused", "--epochs", "15", "--max-profiles", "1000"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "metadata_closed_loop_hp_contracting_probabilistic.json").write_text(json.dumps({"train_config": config}))
            with patch("neural_building_emulator_refactor.routed_latent_targeted_loss_ablation._load_context",
                       return_value=(source, None, path)):
                recipe = _current_best_base(args)
        self.assertEqual(recipe["optimizer"]["epochs"], 15)
        self.assertEqual(recipe["max_profiles"], 1000)
        self.assertEqual(recipe["probabilistic_structured"]["hp_energy_cap_hours"], .5)
        self.assertNotIn("epochs", recipe["probabilistic_structured"])
        self.assertNotIn("dataset_path", recipe["probabilistic_structured"])
        vanilla = _variant_recipe(recipe, "continuous")["probabilistic_lstm"]
        markov = _variant_recipe(recipe, "hurdle_markov")["probabilistic_lstm"]
        self.assertEqual(vanilla["flex_kpi_crps_weight"], 1.5)
        self.assertEqual(markov["flex_kpi_crps_windows_per_profile"], 4)
        self.assertEqual(markov["hp_training_mode"], "hard_expected_grad")
        self.assertEqual(markov["hp_active_power_nll_weight"], 1.)
        self.assertEqual(vanilla["hp_active_power_nll_weight"], 0.)


if __name__ == "__main__":
    unittest.main()
