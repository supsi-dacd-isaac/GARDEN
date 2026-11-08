"""Focused tests for the optional trajectory actor-critic training backend."""

from __future__ import annotations

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.test_prob_closed_loop_grad_stability import (
    make_tiny_model,
    synthetic_batch,
)
from neural_building_emulator.train import (
    _array_tree_all_finite,
    energy_score,
    trajectory_actor_critic_loss_fn,
)
from neural_building_emulator.trajectory_actor_critic import (
    ParticleContinuationCritic,
    energy_score_per_profile,
    energy_score_prefixes_per_profile,
    stop_gradient_module,
)


class TrajectoryActorCriticTests(unittest.TestCase):
    def test_profile_and_prefix_scores_match_training_energy_score(self) -> None:
        key = jax.random.PRNGKey(1)
        prediction_key, target_key = jax.random.split(key)
        predictions = jax.random.normal(prediction_key, (3, 4, 9, 3))
        targets = jax.random.normal(target_key, (3, 9, 3))
        weights = jnp.asarray([1.5, 0.75, 0.75], dtype=predictions.dtype)

        profile_scores = energy_score_per_profile(predictions, targets, weights)
        prefix_scores = energy_score_prefixes_per_profile(
            predictions,
            targets,
            weights,
        )

        np.testing.assert_allclose(
            np.asarray(jnp.mean(profile_scores)),
            np.asarray(
                energy_score(
                    predictions,
                    targets,
                    output_weights=(1.5, 0.75, 0.75),
                )
            ),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(prefix_scores[:, -1]),
            np.asarray(profile_scores),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_critic_is_particle_permutation_invariant(self) -> None:
        critic = ParticleContinuationCritic(
            6,
            5,
            hidden_dim=12,
            depth=2,
            key=jax.random.PRNGKey(2),
        )
        particles = jax.random.normal(jax.random.PRNGKey(3), (7, 6))
        context = jax.random.normal(jax.random.PRNGKey(4), (5,))
        value = critic(particles, context)
        permuted_value = critic(particles[jnp.asarray([5, 1, 6, 0, 3, 2, 4])], context)
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(permuted_value),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_frozen_critic_keeps_input_derivative_only(self) -> None:
        critic = ParticleContinuationCritic(
            4,
            3,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(5),
        )
        particles = jax.random.normal(jax.random.PRNGKey(6), (3, 4))
        context = jax.random.normal(jax.random.PRNGKey(7), (3,))
        frozen = stop_gradient_module(critic)
        input_gradient = jax.grad(lambda value: frozen(value, context))(particles)
        parameter_gradient = eqx.filter_grad(
            lambda candidate: stop_gradient_module(candidate)(particles, context)
        )(critic)

        self.assertTrue(np.isfinite(np.asarray(input_gradient)).all())
        self.assertGreater(float(jnp.linalg.norm(input_gradient)), 0.0)
        leaves = [
            leaf
            for leaf in jax.tree.leaves(parameter_gradient)
            if eqx.is_array(leaf)
        ]
        self.assertTrue(leaves)
        self.assertTrue(all(np.allclose(np.asarray(leaf), 0.0) for leaf in leaves))

    def test_score_function_loss_and_gradients_are_finite(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(8),
            bptt_truncate_steps=4,
            hp_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, targets = synthetic_batch(
            batch_size=2,
            horizon=12,
            metadata_dim=model.metadata_dim,
            input_dim=model.input_dim,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(9),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        critic = ParticleContinuationCritic(
            model.state_dim + model.latent_dim + 7,
            model.metadata_dim + model.input_dim + 7,
            hidden_dim=16,
            depth=2,
            key=jax.random.PRNGKey(10),
        )

        (loss, diagnostics), grads = trajectory_actor_critic_loss_fn(
            (model, critic),
            metadata,
            inputs,
            initial_temperature,
            targets,
            jax.random.PRNGKey(11),
            3,
            "score_function",
            jnp.asarray(0.5, dtype=jnp.float32),
            (1.0, 1.0, 1.0),
            0.0,
            (1, 2),
            0,
            (1.0, 1.0, 1.0),
            False,
            0.05,
            jnp.ones((3,), dtype=jnp.float32),
            0.0,
            (2, 4),
            0,
            (1.0, 1.0, 1.0),
            0.05,
            jnp.ones((6,), dtype=jnp.float32),
            4,
            1.0,
            1.0,
            1.0,
        )

        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(_array_tree_all_finite(diagnostics)))
        self.assertTrue(bool(_array_tree_all_finite(grads)))
        self.assertGreater(float(jnp.abs(diagnostics[4])), 0.0)

    def test_score_function_forward_matches_bernoulli_forward(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(12),
            hp_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=12,
            metadata_dim=model.metadata_dim,
            input_dim=model.input_dim,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(13),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        rollout_key = jax.random.PRNGKey(14)
        bernoulli_predictions, bernoulli_aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=rollout_key,
            num_particles=3,
            hp_scenario_mode="bernoulli",
        )
        score_predictions, score_aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=rollout_key,
            num_particles=3,
            hp_scenario_mode="score_function",
        )
        np.testing.assert_array_equal(
            np.asarray(score_predictions),
            np.asarray(bernoulli_predictions),
        )
        np.testing.assert_array_equal(
            np.asarray(score_aux[5]),
            np.asarray(bernoulli_aux[5]),
        )


if __name__ == "__main__":
    unittest.main()
