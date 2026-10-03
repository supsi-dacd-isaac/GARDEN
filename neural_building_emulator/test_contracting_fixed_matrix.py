"""Tests for trajectory-persistent contracting thermal dynamics."""

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
    closed_loop_stability_penalty,
    local_jacobian_spectral_norm,
)


def common_kwargs() -> dict[str, object]:
    return {
        "metadata_dim": 3,
        "input_dim": 9,
        "state_dim": 4,
        "hidden_dim": 8,
        "depth": 2,
        "hp_cop_cap": 8.0,
        "hp_pel_cap_w_m2": 80.0,
        "hp_qroom_cap_w_m2": 120.0,
        "hp_energy_cap_wh_m2": 400.0,
        "input_mean": (0.0,) * 9,
        "input_scale": (1.0,) * 9,
        "target_mean": (20.0, 10.0, 5.0),
        "target_scale": (2.0, 4.0, 3.0),
        "availability_input_index": 4,
        "bptt_truncate_steps": 0,
    }


class FixedMatrixTests(unittest.TestCase):
    def test_deterministic_monotone_thermostat_counterfactual(self) -> None:
        model = ContractingClosedLoopHPEmulator(
            **common_kwargs(),
            thermostat_demand_mode="monotone",
            key=jax.random.PRNGKey(30),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        temperature = jnp.asarray(0.0, dtype=jnp.float32)  # 20 degC in physical units.
        base_input = jnp.zeros((9,), dtype=jnp.float32).at[4].set(1.0)

        def evaluate(gap_c: float) -> np.ndarray:
            input_t = base_input.at[0].set(20.0 + gap_c)
            return np.asarray(
                model.decode_hp(
                    metadata,
                    model.encode_input(input_t),
                    input_t,
                    jnp.asarray(100.0, dtype=jnp.float32),
                    temperature,
                )
            )

        low = evaluate(-1.0)
        high = evaluate(2.0)
        # pi, delivered Q, expected Pel, E_next, available power, requested Q.
        # Only latent emitter demand is structurally monotone. Delivered heat
        # and compressor power also depend on available buffer energy/cycling.
        self.assertGreater(float(high[5]), float(low[5]))
        np.testing.assert_allclose(
            np.asarray(model.encode_input(base_input.at[0].set(19.0))),
            np.asarray(model.encode_input(base_input.at[0].set(22.0))),
            atol=0.0,
        )

    def test_causal_monotone_demand_uses_temperature_only_through_gap(self) -> None:
        model = ContractingClosedLoopHPEmulator(
            **common_kwargs(),
            thermostat_demand_mode="causal_monotone",
            key=jax.random.PRNGKey(34),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        base_input = jnp.zeros((9,), dtype=jnp.float32).at[4].set(1.0)

        def requested_qroom(
            temperature_scaled: float,
            setpoint_c: float,
            energy_wh_m2: float,
        ) -> float:
            input_t = base_input.at[0].set(setpoint_c)
            result = model.decode_hp(
                metadata,
                model.encode_input(input_t),
                input_t,
                jnp.asarray(energy_wh_m2, dtype=jnp.float32),
                jnp.asarray(temperature_scaled, dtype=jnp.float32),
            )
            return float(result[5])

        # Both cases have Tset-Tin = 2 C. Neither absolute Tin nor buffer
        # energy may alter the causal demand parameters.
        low_state = requested_qroom(0.0, 22.0, 10.0)
        high_state = requested_qroom(1.0, 24.0, 300.0)
        np.testing.assert_allclose(low_state, high_state, atol=1e-6)

        larger_gap = requested_qroom(0.0, 24.0, 10.0)
        self.assertGreater(larger_gap, low_state)

    def test_probabilistic_monotone_thermostat_counterfactual(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            thermostat_demand_mode="monotone",
            key=jax.random.PRNGKey(31),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        xi = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
        temperature = jnp.asarray(0.0, dtype=jnp.float32)
        base_input = jnp.zeros((9,), dtype=jnp.float32).at[4].set(1.0)
        hp_power_history, hp_mode_history = model.initial_hp_history(jnp.float32)

        def evaluate(gap_c: float) -> np.ndarray:
            input_t = base_input.at[0].set(20.0 + gap_c)
            result = model.decode_hp(
                metadata,
                xi,
                model.encode_input(input_t),
                input_t,
                jnp.asarray(100.0, dtype=jnp.float32),
                temperature,
                jnp.asarray(0.5, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.zeros((2,), dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                hp_power_history,
                hp_mode_history,
                jnp.zeros((0,), dtype=jnp.float32),
                jnp.zeros((2,), dtype=jnp.float32),
                hp_scenario_mode="expected",
                hp_concrete_temperature=0.5,
            )
            return np.asarray(result[:9])

        low = evaluate(-1.0)
        high = evaluate(2.0)
        # pi, COP, delivered Q, E_next, available power, requested Q, mode, active Pel, Pel.
        # Only latent emitter demand is structurally monotone. HP activation is
        # intentionally unrestricted so it can reproduce cycling and buffering.
        self.assertGreater(float(high[5]), float(low[5]))

    def test_force_off_gate_stops_generation_but_preserves_buffer_draw(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            thermostat_demand_mode="monotone",
            key=jax.random.PRNGKey(32),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        xi = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
        input_t = jnp.zeros((9,), dtype=jnp.float32).at[0].set(24.0).at[4].set(1.0)
        hp_power_history, hp_mode_history = model.initial_hp_history(jnp.float32)

        def evaluate(generation_available: float) -> np.ndarray:
            result = model.decode_hp(
                metadata,
                xi,
                model.encode_input(input_t),
                input_t,
                jnp.asarray(100.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.zeros((2,), dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                hp_power_history,
                hp_mode_history,
                jnp.zeros((0,), dtype=jnp.float32),
                jnp.zeros((2,), dtype=jnp.float32),
                hp_scenario_mode="expected",
                hp_concrete_temperature=0.5,
                hp_generation_available_t=generation_available,
            )
            return np.asarray(result[:9])

        enabled = evaluate(1.0)
        forced_off = evaluate(0.0)
        self.assertGreater(float(enabled[8]), 0.0)
        self.assertEqual(float(forced_off[0]), 0.0)
        self.assertEqual(float(forced_off[6]), 0.0)
        self.assertEqual(float(forced_off[8]), 0.0)
        self.assertGreater(float(forced_off[2]), 0.0)
        self.assertGreater(float(forced_off[5]), 0.0)

    def test_force_off_input_is_a_generation_only_gate(self) -> None:
        kwargs = {
            **common_kwargs(),
            "hp_force_off_input_index": 5,
            "hp_controller_masked_input_indices": (5,),
            "thermal_dynamics_masked_input_indices": (5,),
        }
        model = ContractingClosedLoopHPEmulator(
            **kwargs,
            thermostat_demand_mode="monotone",
            key=jax.random.PRNGKey(33),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        base_input = (
            jnp.zeros((9,), dtype=jnp.float32)
            .at[0].set(24.0)
            .at[4].set(1.0)
        )

        def evaluate(force_off: float) -> np.ndarray:
            input_t = base_input.at[5].set(force_off)
            return np.asarray(
                model.decode_hp(
                    metadata,
                    model.encode_input(input_t),
                    input_t,
                    jnp.asarray(100.0, dtype=jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                )
            )

        enabled = evaluate(0.0)
        forced_off = evaluate(1.0)
        # Force-off suppresses compressor probability/power, but neither the
        # requested emitter heat nor buffer discharge.
        self.assertGreater(float(enabled[2]), 0.0)
        self.assertEqual(float(forced_off[0]), 0.0)
        self.assertEqual(float(forced_off[2]), 0.0)
        self.assertGreater(float(forced_off[1]), 0.0)
        np.testing.assert_allclose(forced_off[5], enabled[5], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(model.encode_input(base_input.at[5].set(0.0))),
            np.asarray(model.encode_input(base_input.at[5].set(1.0))),
            atol=0.0,
        )

    def test_calendar_mask_only_changes_hp_encoder_input(self) -> None:
        masked_indices = (5, 6, 7, 8)
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            input_encoder_dim=5,
            process_noise_mode="none",
            hp_controller_masked_input_indices=masked_indices,
            key=jax.random.PRNGKey(0),
        )
        input_a = jnp.arange(9, dtype=jnp.float32)
        input_b = input_a.at[jnp.asarray(masked_indices)].set(
            jnp.asarray([100.0, -100.0, 50.0, -50.0], dtype=jnp.float32)
        )

        np.testing.assert_allclose(
            np.asarray(model.encode_input(input_a)),
            np.asarray(model.encode_input(input_b)),
            atol=0.0,
        )
        forcing_a = model.thermal_forcing(input_a, jnp.asarray(0.0), jnp.asarray(0.0))
        forcing_b = model.thermal_forcing(input_b, jnp.asarray(0.0), jnp.asarray(0.0))
        self.assertGreater(float(jnp.max(jnp.abs(forcing_a - forcing_b))), 1.0)

    def test_deterministic_matrix_uses_forcing_and_remains_bounded(self) -> None:
        model = ContractingClosedLoopHPEmulator(
            **common_kwargs(),
            contraction_gamma=0.91,
            key=jax.random.PRNGKey(1),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        forcing_a = jnp.zeros((model.thermal_forcing_dim,), dtype=jnp.float32)
        forcing_b = jnp.ones((model.thermal_forcing_dim,), dtype=jnp.float32)
        matrix_a, _ = model.transition_matrix_and_bias(metadata, forcing_a)
        matrix_b, _ = model.transition_matrix_and_bias(metadata, forcing_b)

        self.assertGreater(float(jnp.max(jnp.abs(matrix_a - matrix_b))), 1e-6)
        self.assertLessEqual(float(jnp.linalg.norm(matrix_a, ord=2)), 0.91 + 1e-6)
        self.assertLessEqual(float(jnp.linalg.norm(matrix_b, ord=2)), 0.91 + 1e-6)

    def test_probabilistic_matrix_uses_forcing_by_default_and_remains_bounded(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            contraction_gamma=0.93,
            key=jax.random.PRNGKey(2),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        xi = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
        forcing_a = jnp.zeros((model.thermal_forcing_dim,), dtype=jnp.float32)
        forcing_b = jnp.ones((model.thermal_forcing_dim,), dtype=jnp.float32)
        matrix_a, _ = model.transition_matrix_and_bias(metadata, xi, forcing_a)
        matrix_b, _ = model.transition_matrix_and_bias(metadata, xi, forcing_b)

        self.assertGreater(float(jnp.max(jnp.abs(matrix_a - matrix_b))), 1e-6)
        self.assertLessEqual(float(jnp.linalg.norm(matrix_a, ord=2)), 0.93 + 1e-6)
        self.assertLessEqual(float(jnp.linalg.norm(matrix_b, ord=2)), 0.93 + 1e-6)

    def test_diagnostic_fixed_matrix_is_forcing_independent_and_bounded(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            contraction_gamma=0.93,
            transition_matrix_mode="fixed",
            key=jax.random.PRNGKey(21),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        xi = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
        forcing_a = jnp.zeros((model.thermal_forcing_dim,), dtype=jnp.float32)
        forcing_b = jnp.ones((model.thermal_forcing_dim,), dtype=jnp.float32)
        matrix_a, _ = model.transition_matrix_and_bias(metadata, xi, forcing_a)
        matrix_b, _ = model.transition_matrix_and_bias(metadata, xi, forcing_b)

        np.testing.assert_allclose(np.asarray(matrix_a), np.asarray(matrix_b), atol=0.0)
        self.assertLessEqual(float(jnp.linalg.norm(matrix_a, ord=2)), 0.93 + 1e-6)

    def test_stability_state_contains_only_active_hp_memory(self) -> None:
        independent = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            hp_activation_model="independent",
            key=jax.random.PRNGKey(3),
        )
        persistent = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            hp_activation_model="persistent_markov",
            key=jax.random.PRNGKey(4),
        )
        history = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="none",
            hp_activation_model="power_history",
            hp_history_timescales_hours=(0.5, 3.0),
            key=jax.random.PRNGKey(5),
        )

        self.assertEqual(independent.stability_state_dim(), 6)
        self.assertEqual(persistent.stability_state_dim(), 9)
        self.assertEqual(history.stability_state_dim(), 12)


class JacobianPenaltyTests(unittest.TestCase):
    def test_power_iteration_matches_exact_spectral_norm(self) -> None:
        matrix = jnp.asarray([[0.7, 0.2], [-0.1, 0.4]], dtype=jnp.float32)
        state = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
        direction = jnp.asarray([0.8, -0.6], dtype=jnp.float32)
        step_fn = lambda value: matrix @ value
        exact = local_jacobian_spectral_norm(
            step_fn,
            state,
            direction,
            method="exact_svd",
            power_iterations=1,
        )
        estimated = local_jacobian_spectral_norm(
            step_fn,
            state,
            direction,
            method="power_iteration",
            power_iterations=12,
        )
        np.testing.assert_allclose(float(estimated), float(exact), rtol=1e-4, atol=1e-5)

    def test_stability_penalty_has_finite_parameter_gradient(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="heteroscedastic",
            hp_activation_model="independent",
            key=jax.random.PRNGKey(6),
        )
        metadata = jnp.zeros((1, 3), dtype=jnp.float32)
        inputs = jnp.zeros((1, 4, 9), dtype=jnp.float32).at[:, :, 4].set(1.0)
        initial_temperature = jnp.zeros((1, 1), dtype=jnp.float32)
        rollout_key = jax.random.PRNGKey(7)
        stability_key = jax.random.PRNGKey(8)

        def objective(candidate):
            keys = jax.random.split(rollout_key, metadata.shape[0])
            _, aux = jax.vmap(
                lambda row_metadata, row_inputs, row_temperature, row_key: candidate.sample_with_aux(
                    row_metadata,
                    row_inputs,
                    row_temperature,
                    key=row_key,
                    num_particles=2,
                    sample_process_noise=True,
                    hp_scenario_mode="expected",
                )
            )(metadata, inputs, initial_temperature, keys)
            return closed_loop_stability_penalty(
                candidate,
                metadata,
                inputs,
                aux,
                stability_key,
                gamma=0.8,
                num_samples=1,
                aggregation="max",
                method="power_iteration",
                power_iterations=2,
            )

        value, gradients = eqx.filter_value_and_grad(objective)(model)
        self.assertTrue(np.isfinite(float(value)))
        gradient_leaves = [
            np.asarray(leaf)
            for leaf in jax.tree_util.tree_leaves(gradients)
            if eqx.is_array(leaf)
        ]
        self.assertTrue(gradient_leaves)
        self.assertTrue(all(np.isfinite(leaf).all() for leaf in gradient_leaves))


class MetadataFixedThermalArchitectureTests(unittest.TestCase):
    def test_deterministic_rollout_is_finite_and_matrix_is_stable(self) -> None:
        model = ContractingClosedLoopHPEmulator(
            **common_kwargs(),
            thermal_architecture="metadata_fixed_ss",
            q_to_t_mode="unconstrained",
            temperature_output_scale=5.0,
            key=jax.random.PRNGKey(40),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        inputs = jnp.zeros((16, 9), dtype=jnp.float32).at[:, 4].set(1.0)
        predictions = model(metadata, inputs, jnp.asarray([0.0], dtype=jnp.float32))
        rho, _, _, _, _ = model.metadata_fixed_parameters(metadata)

        self.assertEqual(predictions.shape, (16, 3))
        self.assertTrue(np.isfinite(np.asarray(predictions)).all())
        self.assertTrue(np.all(np.asarray(rho) > 0.0))
        self.assertTrue(np.all(np.asarray(rho) < model.contraction_gamma))

    def test_probabilistic_rollout_and_gradient_are_finite(self) -> None:
        model = ProbabilisticContractingClosedLoopHPEmulator(
            **common_kwargs(),
            controller_state_dim=2,
            latent_dim=2,
            process_noise_mode="heteroscedastic",
            thermal_architecture="metadata_fixed_ss",
            q_to_t_mode="unconstrained",
            temperature_output_scale=5.0,
            key=jax.random.PRNGKey(41),
        )
        metadata = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        inputs = jnp.zeros((12, 9), dtype=jnp.float32).at[:, 4].set(1.0)
        initial_temperature = jnp.asarray([0.0], dtype=jnp.float32)

        def objective(candidate):
            samples = candidate.sample(
                metadata,
                inputs,
                initial_temperature,
                key=jax.random.PRNGKey(42),
                num_particles=3,
                sample_process_noise=True,
                hp_scenario_mode="bernoulli",
            )
            return jnp.mean(samples**2)

        value, gradients = eqx.filter_value_and_grad(objective)(model)
        samples = model.sample(
            metadata,
            inputs,
            initial_temperature,
            key=jax.random.PRNGKey(43),
            num_particles=3,
            sample_process_noise=True,
            hp_scenario_mode="bernoulli",
        )
        xi = jnp.asarray([0.3, -0.2], dtype=jnp.float32)
        rho, _, _, _, _ = model.metadata_fixed_parameters(metadata, xi)

        self.assertEqual(samples.shape, (3, 12, 3))
        self.assertTrue(np.isfinite(np.asarray(samples)).all())
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.asarray(rho) > 0.0))
        self.assertTrue(np.all(np.asarray(rho) < model.contraction_gamma))
        gradient_leaves = [
            np.asarray(leaf)
            for leaf in jax.tree_util.tree_leaves(gradients)
            if eqx.is_array(leaf)
        ]
        self.assertTrue(gradient_leaves)
        self.assertTrue(all(np.isfinite(leaf).all() for leaf in gradient_leaves))


if __name__ == "__main__":
    unittest.main()
