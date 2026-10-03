"""Focused checks for the sequential Q-to-T response experiment."""

from __future__ import annotations

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.models import (
    ContractingClosedLoopHPEmulator,
    ProbabilisticContractingClosedLoopHPEmulator,
)
from neural_building_emulator.train import (
    PROB_CLOSED_LOOP_GRADIENT_GROUP_NAMES,
    _array_tree_global_norm,
    _probabilistic_closed_loop_gradient_group_norms,
)
from neural_building_emulator_refactor.q_to_t_response_experiments import (
    _promote,
    _spaced,
    _temperature_step_metrics,
)


def _kwargs() -> dict:
    return {
        "metadata_dim": 3,
        "input_dim": 9,
        "state_dim": 3,
        "hidden_dim": 8,
        "depth": 2,
        "temperature_delta_max_c": 2.0,
        "temperature_update_mode": "leaky_equilibrium",
        "q_to_t_mode": "positive_leaky",
        "q_to_t_time_constants_hours": (0.25, 1.0),
        "q_to_t_gain_min_c_per_w_m2": 0.01,
        "q_to_t_gain_max_c_per_w_m2": 2.0,
        "hp_pel_cap_w_m2": 20.0,
        "hp_qroom_cap_w_m2": 100.0,
        "hp_energy_cap_wh_m2": 500.0,
        "input_mean": (0.0,) * 9,
        "input_scale": (1.0,) * 9,
        "target_mean": (20.0, 5.0, 2.0),
        "target_scale": (2.0, 10.0, 4.0),
        "availability_input_index": 8,
    }


class QToTResponseExperimentTests(unittest.TestCase):
    def test_gradient_groups_partition_all_trainable_parameters(self) -> None:
        kwargs = _kwargs()
        kwargs["temperature_update_mode"] = "alpha_bounded_equilibrium"
        kwargs["input_encoder_dim"] = 4
        kwargs["input_encoder_hidden_dim"] = 8
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **kwargs,
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="heteroscedastic",
            key=jax.random.PRNGKey(7),
        )
        arrays = eqx.filter(model, eqx.is_inexact_array)
        gradients = jax.tree_util.tree_map(
            lambda value: None if value is None else jnp.ones_like(value), arrays
        )
        grouped = np.asarray(
            _probabilistic_closed_loop_gradient_group_norms(gradients)
        )
        total = float(_array_tree_global_norm(gradients))

        self.assertEqual(grouped.shape, (len(PROB_CLOSED_LOOP_GRADIENT_GROUP_NAMES),))
        np.testing.assert_allclose(np.sum(grouped**2), total**2, rtol=2e-6)
        for name in (
            "alpha", "temperature_equilibrium", "transition", "q_response",
            "hp", "process_noise", "initial_state", "input_encoder",
        ):
            self.assertGreater(
                grouped[PROB_CLOSED_LOOP_GRADIENT_GROUP_NAMES.index(name)], 0.0
            )

    def test_fixed_alpha_removes_alpha_network(self) -> None:
        kwargs = _kwargs()
        kwargs.update(
            temperature_update_mode="alpha_bounded_equilibrium",
            temperature_alpha_cap=0.5,
            temperature_fixed_alpha=0.05,
        )
        deterministic = ContractingClosedLoopHPEmulator(
            **kwargs,
            key=jax.random.PRNGKey(8),
        )
        probabilistic = ProbabilisticContractingClosedLoopHPEmulator(
            **kwargs,
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            key=jax.random.PRNGKey(9),
        )

        self.assertIsNone(deterministic.temperature_alpha_net)
        self.assertIsNone(probabilistic.temperature_alpha_net)
        self.assertAlmostEqual(
            float(deterministic.temperature_alpha(jnp.zeros((3,)))), 0.05
        )
        self.assertAlmostEqual(
            float(probabilistic.temperature_alpha(jnp.zeros((3,)), jnp.zeros((2,)))),
            0.05,
        )

    def test_metadata_conditioned_alpha_is_particle_invariant(self) -> None:
        kwargs = _kwargs()
        kwargs.update(
            temperature_update_mode="alpha_bounded_equilibrium",
            temperature_alpha_cap=0.5,
            temperature_alpha_conditioning="metadata",
        )
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **kwargs,
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            key=jax.random.PRNGKey(10),
        )
        metadata = jnp.asarray([0.2, -0.4, 0.7])
        alpha_a = model.temperature_alpha(metadata, jnp.asarray([2.0, -3.0]))
        alpha_b = model.temperature_alpha(metadata, jnp.asarray([-8.0, 5.0]))

        self.assertEqual(model.temperature_alpha_conditioning, "metadata")
        self.assertAlmostEqual(float(alpha_a), float(alpha_b), places=7)
        self.assertGreater(float(alpha_a), 0.0)
        self.assertLessEqual(float(alpha_a), 0.5)

    def test_temperature_step_metrics_measure_limit_usage(self) -> None:
        metrics = _temperature_step_metrics(
            np.asarray([[20.5, 21.5, 23.4], [19.0, 17.2, 15.21]]),
            20.0,
            2.0,
            prefix="step",
        )
        self.assertAlmostEqual(metrics["step_max_abs_c"], 1.99)
        self.assertAlmostEqual(metrics["step_fraction_ge_50pct_limit"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["step_fraction_ge_90pct_limit"], 3.0 / 6.0)
        self.assertAlmostEqual(metrics["step_fraction_ge_99pct_limit"], 1.0 / 6.0)

    def test_oracle_changes_thermal_path_and_preserves_default(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **_kwargs(), controller_state_dim=2, latent_dim=2,
            process_noise_mode="none", key=jax.random.PRNGKey(7),
        )
        horizon = 8
        inputs = jnp.zeros((horizon, 9), dtype=jnp.float32).at[:, 8].set(1.0)
        common = dict(
            key=jax.random.PRNGKey(20), num_particles=2,
            sample_process_noise=False, hp_scenario_mode="expected",
        )
        ordinary = model.sample(jnp.zeros((3,)), inputs, jnp.zeros((1,)), **common)
        unchanged = model.sample(
            jnp.zeros((3,)), inputs, jnp.zeros((1,)),
            thermal_qroom_reference=None, **common,
        )
        oracle = model.sample(
            jnp.zeros((3,)), inputs, jnp.zeros((1,)),
            thermal_qroom_reference=jnp.full((horizon,), 80.0), **common,
        )
        np.testing.assert_array_equal(ordinary, unchanged)
        self.assertEqual(oracle.shape, (2, horizon, 3))
        self.assertFalse(np.allclose(np.asarray(ordinary[:, :, 0]), np.asarray(oracle[:, :, 0])))
        self.assertTrue(np.all(np.isfinite(np.asarray(oracle))))
        with self.assertRaisesRegex(ValueError, "shape"):
            model.sample(
                jnp.zeros((3,)), inputs, jnp.zeros((1,)),
                thermal_qroom_reference=jnp.ones((horizon - 1,)), **common,
            )

    def test_alpha_bounded_starts_near_tenth_not_half(self) -> None:
        kwargs = _kwargs()
        kwargs["temperature_update_mode"] = "alpha_bounded_equilibrium"
        kwargs["temperature_alpha_cap"] = 0.5
        deterministic = ContractingClosedLoopHPEmulator(
            **kwargs, key=jax.random.PRNGKey(11),
        )
        probabilistic = ProbabilisticContractingClosedLoopHPEmulator(
            **kwargs, controller_state_dim=2, latent_dim=2,
            process_noise_mode="none", key=jax.random.PRNGKey(12),
        )
        deterministic_alpha = float(deterministic.temperature_alpha(jnp.zeros((3,))))
        probabilistic_alpha = float(probabilistic.temperature_alpha(
            jnp.zeros((3,)), jnp.zeros((2,))
        ))
        self.assertGreater(deterministic_alpha, 0.05)
        self.assertLess(deterministic_alpha, 0.15)
        self.assertGreater(probabilistic_alpha, 0.05)
        self.assertLess(probabilistic_alpha, 0.15)
        self.assertLessEqual(deterministic_alpha, 0.5)
        self.assertLessEqual(probabilistic_alpha, 0.5)

    def test_promotion_uses_response_and_annual_safety_checks(self) -> None:
        scores = []
        for label, annual, response in (
            ("A_leaky_2c", 1.0, 0.5),
            ("C_alpha_bounded_2c", 1.03, 0.42),
        ):
            scores.extend([
                {"label": label, "horizon_hours": 0.0, "tin_rmse_c": annual,
                 "qroom_rmse_w_m2": 10.0, "pel_rmse_w_m2": 5.0},
                {"label": label, "horizon_hours": 0.5,
                 "tin_response_rmse_c": response},
            ])
        diagnostics = [
            {"arm": "A_leaky_2c", "skipped_update_batches": 0},
            {"arm": "C_alpha_bounded_2c", "skipped_update_batches": 0},
        ]
        promoted, checks = _promote(scores, diagnostics)
        self.assertTrue(promoted)
        self.assertTrue(all(checks.values()))
        scores[2]["tin_rmse_c"] = 1.2
        self.assertFalse(_promote(scores, diagnostics)[0])
        self.assertEqual(_spaced([1, 2, 3, 4, 5], 3), [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
