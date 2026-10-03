"""Tests for the opt-in structural positive Qroom-to-temperature path."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.models import (
    ContractingClosedLoopHPEmulator,
    ProbabilisticContractingClosedLoopHPEmulator,
)


def _common_kwargs() -> dict[str, object]:
    return {
        "metadata_dim": 3,
        "input_dim": 9,
        "state_dim": 4,
        "hidden_dim": 8,
        "depth": 2,
        "temperature_delta_max_c": 2.0,
        "temperature_update_mode": "leaky_equilibrium",
        "q_to_t_mode": "positive_leaky",
        "q_to_t_time_constants_hours": (1.0, 24.0),
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


class StructuralQToTResponseTest(unittest.TestCase):
    def test_exogenous_transition_conditioning_removes_temperature_feedback(self) -> None:
        kwargs = _common_kwargs()
        state_feedback = ContractingClosedLoopHPEmulator(
            **kwargs,
            transition_conditioning="state_feedback",
            key=jax.random.PRNGKey(20),
        )
        exogenous = ContractingClosedLoopHPEmulator(
            **kwargs,
            transition_conditioning="exogenous",
            key=jax.random.PRNGKey(20),
        )
        metadata = jnp.zeros((3,), dtype=jnp.float32)
        input_t = jnp.zeros((9,), dtype=jnp.float32)
        cold_forcing = exogenous.thermal_forcing(
            input_t, jnp.asarray(0.0), jnp.asarray(-1.0)
        )
        warm_forcing = exogenous.thermal_forcing(
            input_t, jnp.asarray(0.0), jnp.asarray(1.0)
        )

        cold_matrix, cold_bias = exogenous.transition_matrix_and_bias(
            metadata, cold_forcing
        )
        warm_matrix, warm_bias = exogenous.transition_matrix_and_bias(
            metadata, warm_forcing
        )
        np.testing.assert_allclose(cold_matrix, warm_matrix, atol=1e-7)
        np.testing.assert_allclose(cold_bias, warm_bias, atol=1e-7)

        feedback_cold, feedback_cold_bias = state_feedback.transition_matrix_and_bias(
            metadata, cold_forcing
        )
        feedback_warm, feedback_warm_bias = state_feedback.transition_matrix_and_bias(
            metadata, warm_forcing
        )
        self.assertFalse(
            np.allclose(feedback_cold, feedback_warm)
            and np.allclose(feedback_cold_bias, feedback_warm_bias)
        )

    def test_probabilistic_exogenous_transition_conditioning_is_particle_persistent(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **_common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            transition_conditioning="exogenous",
            key=jax.random.PRNGKey(21),
        )
        metadata = jnp.zeros((3,), dtype=jnp.float32)
        xi = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
        input_t = jnp.zeros((9,), dtype=jnp.float32)
        cold = model.thermal_forcing(input_t, jnp.asarray(0.0), jnp.asarray(-1.0))
        warm = model.thermal_forcing(input_t, jnp.asarray(0.0), jnp.asarray(1.0))
        cold_matrix, cold_bias = model.transition_matrix_and_bias(metadata, xi, cold)
        warm_matrix, warm_bias = model.transition_matrix_and_bias(metadata, xi, warm)
        np.testing.assert_allclose(cold_matrix, warm_matrix, atol=1e-7)
        np.testing.assert_allclose(cold_bias, warm_bias, atol=1e-7)

    def test_additive_feedback_keeps_matrix_exogenous_and_bias_temperature_sensitive(self) -> None:
        model = ContractingClosedLoopHPEmulator(
            **_common_kwargs(),
            input_encoder_dim=5,
            additive_feedback_gain_bound=0.75,
            transition_conditioning="exogenous_additive_feedback",
            key=jax.random.PRNGKey(22),
        )
        metadata = jnp.zeros((3,), dtype=jnp.float32)
        input_t = jnp.zeros((9,), dtype=jnp.float32)
        cold = model.thermal_forcing(input_t, jnp.asarray(0.0), jnp.asarray(-1.0))
        warm = model.thermal_forcing(input_t, jnp.asarray(0.0), jnp.asarray(1.0))

        cold_matrix, cold_bias = model.transition_matrix_and_bias(metadata, cold)
        warm_matrix, warm_bias = model.transition_matrix_and_bias(metadata, warm)

        self.assertIsNotNone(model.transition_forcing_encoder)
        np.testing.assert_allclose(cold_matrix, warm_matrix, atol=1e-7)
        self.assertFalse(np.allclose(cold_bias, warm_bias))

    def test_probabilistic_additive_feedback_separates_matrix_and_temperature_feedback(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **_common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            input_encoder_dim=5,
            process_noise_mode="none",
            additive_feedback_gain_bound=0.75,
            transition_conditioning="exogenous_additive_feedback",
            key=jax.random.PRNGKey(23),
        )
        metadata = jnp.zeros((3,), dtype=jnp.float32)
        xi = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
        input_t = jnp.zeros((9,), dtype=jnp.float32)
        cold = model.thermal_forcing(input_t, jnp.asarray(0.0), jnp.asarray(-1.0))
        warm = model.thermal_forcing(input_t, jnp.asarray(0.0), jnp.asarray(1.0))

        cold_matrix, cold_bias = model.transition_matrix_and_bias(metadata, xi, cold)
        warm_matrix, warm_bias = model.transition_matrix_and_bias(metadata, xi, warm)

        self.assertIsNotNone(model.transition_forcing_encoder)
        np.testing.assert_allclose(cold_matrix, warm_matrix, atol=1e-7)
        self.assertFalse(np.allclose(cold_bias, warm_bias))

    def test_legacy_transition_modes_keep_their_parameter_trees(self) -> None:
        kwargs = _common_kwargs()
        state_feedback = ContractingClosedLoopHPEmulator(
            **kwargs,
            transition_conditioning="state_feedback",
            key=jax.random.PRNGKey(24),
        )
        exogenous = ContractingClosedLoopHPEmulator(
            **kwargs,
            transition_conditioning="exogenous",
            key=jax.random.PRNGKey(24),
        )

        feedback_leaves = jax.tree_util.tree_leaves(state_feedback)
        exogenous_leaves = jax.tree_util.tree_leaves(exogenous)
        self.assertEqual(len(feedback_leaves), len(exogenous_leaves))
        for feedback_leaf, exogenous_leaf in zip(feedback_leaves, exogenous_leaves):
            np.testing.assert_array_equal(feedback_leaf, exogenous_leaf)

    def test_requires_equilibrium_temperature_update(self) -> None:
        kwargs = _common_kwargs()
        kwargs["temperature_update_mode"] = "absolute"
        with self.assertRaisesRegex(ValueError, "bounded_equilibrium"):
            ContractingClosedLoopHPEmulator(
                **kwargs,
                key=jax.random.PRNGKey(0),
            )

    def test_bounded_equilibrium_has_hard_step_cap_and_no_alpha(self) -> None:
        kwargs = _common_kwargs()
        kwargs["temperature_update_mode"] = "bounded_equilibrium"
        deterministic = ContractingClosedLoopHPEmulator(
            **kwargs,
            key=jax.random.PRNGKey(10),
        )
        probabilistic = ProbabilisticContractingClosedLoopHPEmulator(
            **kwargs,
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            key=jax.random.PRNGKey(11),
        )

        self.assertIsNone(deterministic.temperature_alpha_net)
        self.assertIsNone(probabilistic.temperature_alpha_net)
        previous = jnp.asarray(0.0, dtype=jnp.float32)
        outputs = []
        for model in (deterministic, probabilistic):
            low_alpha = model._temperature_output(
                jnp.asarray(0.0),
                previous,
                jnp.asarray(0.0),
                jnp.asarray(100.0),
            )
            high_alpha = model._temperature_output(
                jnp.asarray(0.0),
                previous,
                jnp.asarray(1.0),
                jnp.asarray(100.0),
            )
            np.testing.assert_allclose(low_alpha, high_alpha, atol=1e-7)
            physical_step_c = float(low_alpha) * float(model.target_scale[0])
            self.assertGreater(physical_step_c, 0.0)
            self.assertLessEqual(physical_step_c, 2.0 + 1e-6)
            outputs.append(float(low_alpha))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7)

    def test_bounded_equilibrium_requires_positive_step_cap(self) -> None:
        kwargs = _common_kwargs()
        kwargs["temperature_update_mode"] = "bounded_equilibrium"
        kwargs["temperature_delta_max_c"] = 0.0
        with self.assertRaisesRegex(ValueError, "temperature_delta_max_c > 0"):
            ContractingClosedLoopHPEmulator(
                **kwargs,
                key=jax.random.PRNGKey(12),
            )

    def test_alpha_bounded_equilibrium_controls_rate_but_not_cap(self) -> None:
        kwargs = _common_kwargs()
        kwargs["temperature_update_mode"] = "alpha_bounded_equilibrium"
        deterministic = ContractingClosedLoopHPEmulator(
            **kwargs,
            key=jax.random.PRNGKey(13),
        )
        probabilistic = ProbabilisticContractingClosedLoopHPEmulator(
            **kwargs,
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            key=jax.random.PRNGKey(14),
        )

        self.assertIsNotNone(deterministic.temperature_alpha_net)
        self.assertIsNotNone(probabilistic.temperature_alpha_net)
        deterministic_alpha = float(
            deterministic.temperature_alpha(jnp.zeros((3,), dtype=jnp.float32))
        )
        probabilistic_alpha = float(
            probabilistic.temperature_alpha(
                jnp.zeros((3,), dtype=jnp.float32),
                jnp.zeros((2,), dtype=jnp.float32),
            )
        )
        self.assertGreater(deterministic_alpha, 0.0)
        self.assertLess(deterministic_alpha, 1.0)
        self.assertGreater(probabilistic_alpha, 0.0)
        self.assertLess(probabilistic_alpha, 1.0)

        for model in (deterministic, probabilistic):
            slow = model._temperature_output(
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(0.1),
                jnp.asarray(100.0),
            )
            fast = model._temperature_output(
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(0.9),
                jnp.asarray(100.0),
            )
            self.assertGreater(float(fast), float(slow))
            self.assertLessEqual(
                float(fast) * float(model.target_scale[0]),
                2.0 + 1e-6,
            )

    def test_deterministic_path_is_separate_positive_and_leaky(self) -> None:
        model = ContractingClosedLoopHPEmulator(
            **_common_kwargs(),
            key=jax.random.PRNGKey(1),
        )
        metadata = jnp.zeros((3,), dtype=jnp.float32)
        input_t = jnp.zeros((9,), dtype=jnp.float32)
        temperature_t = jnp.asarray(0.0, dtype=jnp.float32)

        low_forcing = model.thermal_forcing(input_t, jnp.asarray(-3.0), temperature_t)
        high_forcing = model.thermal_forcing(input_t, jnp.asarray(7.0), temperature_t)
        np.testing.assert_allclose(low_forcing, high_forcing)
        self.assertEqual(model.thermal_forcing_dim, 10)

        total_gain, weights = model.q_response_parameters(metadata)
        self.assertGreaterEqual(float(total_gain), 0.01)
        self.assertLessEqual(float(total_gain), 2.0)
        np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, atol=1e-6)
        self.assertTrue(bool(jnp.all(weights > 0.0)))

        heat0 = jnp.zeros((2,), dtype=jnp.float32)
        heat_low = model.next_q_response_state(metadata, heat0, jnp.asarray(0.0))
        heat_high = model.next_q_response_state(metadata, heat0, jnp.asarray(10.0))
        self.assertTrue(bool(jnp.all(heat_high > heat_low)))

        hotter = model.decode_temperature(
            jnp.zeros((4,), dtype=jnp.float32),
            temperature_t,
            jnp.asarray(0.1),
            heat_high,
        )
        cooler = model.decode_temperature(
            jnp.zeros((4,), dtype=jnp.float32),
            temperature_t,
            jnp.asarray(0.1),
            heat_low,
        )
        self.assertGreater(float(hotter), float(cooler))

    def test_probabilistic_path_is_carried_per_particle_and_in_stability_state(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **_common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            key=jax.random.PRNGKey(2),
        )
        predictions, aux = model.sample_with_aux(
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((6, 9), dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            key=jax.random.PRNGKey(3),
            num_particles=3,
            hp_scenario_mode="expected",
        )
        self.assertEqual(predictions.shape, (3, 6, 3))
        self.assertEqual(len(aux), 21)
        self.assertEqual(aux[-1].shape, (3, 6, 2))
        self.assertEqual(model.stability_state_dim(), model.state_dim + 2 + 2)

        packed = model.pack_stability_state(
            jnp.zeros((model.state_dim,), dtype=jnp.float32),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.zeros((model.controller_state_dim,), dtype=jnp.float32),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray([0.2, 0.3], dtype=jnp.float32),
        )
        unpacked = model.unpack_stability_state(packed)
        np.testing.assert_allclose(unpacked[-1], jnp.asarray([0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
