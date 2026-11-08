"""Tests for the differentiable event-study coefficient CRPS."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.data import ClosedLoopWindowedArrays
from neural_building_emulator.train import (
    flexibility_coefficient_crps,
    flexibility_direct_ratio_crps,
    grouped_profile_minibatches,
    predict_probabilistic_closed_loop_batch_with_aux,
    targeted_flexibility_energy_score,
)


def synthetic_event_study_batch(
    *,
    particle_slopes: tuple[float, ...],
    target_slope: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size = 2
    horizon = 96
    input_dim = 9
    levels = jnp.asarray(
        [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.25] * 3,
        dtype=jnp.float32,
    )
    setpoint = jnp.tile(jnp.repeat(levels, 4)[jnp.newaxis, :], (batch_size, 1))
    time = jnp.arange(horizon, dtype=jnp.float32)
    inputs = jnp.zeros((batch_size, horizon, input_dim), dtype=jnp.float32)
    inputs = inputs.at[:, :, 0].set(setpoint)
    inputs = inputs.at[:, :, 1].set(jnp.sin(time / 10.0))
    inputs = inputs.at[:, :, 2].set(jnp.maximum(jnp.sin(time / 8.0), 0.0))
    inputs = inputs.at[:, :, 5].set(jnp.sin(time / 4.0))
    inputs = inputs.at[:, :, 6].set(jnp.cos(time / 4.0))

    targets = jnp.zeros((batch_size, horizon, 3), dtype=jnp.float32)
    targets = targets.at[:, :, 0].set(0.2 * setpoint)
    targets = targets.at[:, :, 2].set(target_slope * setpoint)

    predictions = jnp.broadcast_to(
        targets[:, jnp.newaxis, :, :],
        (batch_size, len(particle_slopes), horizon, 3),
    )
    slopes = jnp.asarray(particle_slopes, dtype=jnp.float32)
    predictions = predictions.at[:, :, :, 2].set(
        slopes[jnp.newaxis, :, jnp.newaxis] * setpoint[:, jnp.newaxis, :]
    )
    initial_temperature = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    return predictions, targets, inputs, initial_temperature


def score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    controls: str,
    min_events: int = 8,
) -> jnp.ndarray:
    return flexibility_coefficient_crps(
        predictions,
        targets,
        inputs,
        initial_temperature,
        horizon_steps=(2, 4),
        horizon_hours=(0.5, 1.0),
        setpoint_delta_threshold=0.05,
        min_events=min_events,
        ridge=1e-3,
        controls=controls,  # type: ignore[arg-type]
    )


def direct_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    *,
    min_events: int = 8,
) -> jnp.ndarray:
    del initial_temperature
    return flexibility_direct_ratio_crps(
        predictions,
        targets,
        inputs,
        horizon_steps=(2, 4),
        horizon_hours=(0.5, 1.0),
        setpoint_delta_threshold=0.05,
        min_events=min_events,
    )


class FlexibilityCoefficientCrpsTests(unittest.TestCase):
    def test_wrong_setpoint_sensitivity_scores_worse_than_truth(self) -> None:
        perfect = synthetic_event_study_batch(particle_slopes=(2.0, 2.0, 2.0))
        biased = synthetic_event_study_batch(particle_slopes=(3.0, 3.0, 3.0))
        perfect_score = float(score(*perfect, controls="none"))
        biased_score = float(score(*biased, controls="none"))
        self.assertLess(perfect_score, 1e-3)
        self.assertGreater(biased_score, perfect_score + 0.2)

    def test_full_control_score_has_finite_nonzero_gradient(self) -> None:
        batch = synthetic_event_study_batch(particle_slopes=(2.5, 3.0, 3.5))
        predictions, targets, inputs, initial_temperature = batch
        score_fn = lambda value: score(
            value,
            targets,
            inputs,
            initial_temperature,
            controls="full",
        )
        value, gradient = jax.value_and_grad(score_fn)(predictions)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-6)

    def test_windows_without_enough_events_are_ignored(self) -> None:
        batch = synthetic_event_study_batch(particle_slopes=(3.0, 3.0, 3.0))
        value = score(*batch, controls="full", min_events=100)
        self.assertEqual(float(value), 0.0)


class FlexibilityDirectRatioCrpsTests(unittest.TestCase):
    def test_wrong_direct_flexibility_scores_worse_than_truth(self) -> None:
        perfect = synthetic_event_study_batch(particle_slopes=(2.0, 2.0, 2.0))
        biased = synthetic_event_study_batch(particle_slopes=(3.0, 3.0, 3.0))

        perfect_score = float(direct_score(*perfect))
        biased_score = float(direct_score(*biased))

        self.assertLess(perfect_score, 1e-3)
        self.assertGreater(biased_score, perfect_score + 0.2)

    def test_direct_flexibility_score_has_finite_nonzero_gradient(self) -> None:
        predictions, targets, inputs, initial_temperature = synthetic_event_study_batch(
            particle_slopes=(2.5, 3.0, 3.5)
        )
        score_fn = lambda value: direct_score(
            value,
            targets,
            inputs,
            initial_temperature,
        )

        value, gradient = jax.value_and_grad(score_fn)(predictions)

        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-6)

    def test_direct_windows_without_enough_events_are_ignored(self) -> None:
        batch = synthetic_event_study_batch(particle_slopes=(3.0, 3.0, 3.0))
        value = direct_score(*batch, min_events=100)
        self.assertEqual(float(value), 0.0)

    def test_targeted_response_score_penalizes_wrong_geometry(self) -> None:
        perfect = synthetic_event_study_batch(particle_slopes=(2.0, 2.0, 2.0))
        biased = synthetic_event_study_batch(particle_slopes=(2.5, 3.0, 3.5))

        def targeted(batch) -> jnp.ndarray:
            predictions, targets, inputs, _ = batch
            return targeted_flexibility_energy_score(
                predictions,
                targets,
                inputs,
                short_horizon_steps=2,
                short_horizon_hours=0.5,
                long_horizon_steps=4,
                long_horizon_hours=1.0,
                setpoint_delta_threshold=0.05,
                min_events=8,
                coordinate_center=(1.0, 0.0, 1.0),
                coordinate_scale=(1.0, 0.5, 1.0),
                physical_kpi_scale=1.0,
            )

        perfect_score = float(targeted(perfect))
        biased_score = float(targeted(biased))

        self.assertLess(perfect_score, 1e-3)
        self.assertGreater(biased_score, perfect_score + 0.2)

    def test_targeted_response_score_has_finite_nonzero_gradient(self) -> None:
        predictions, targets, inputs, _ = synthetic_event_study_batch(
            particle_slopes=(2.5, 3.0, 3.5)
        )

        def score_fn(value: jnp.ndarray) -> jnp.ndarray:
            return targeted_flexibility_energy_score(
                value,
                targets,
                inputs,
                short_horizon_steps=2,
                short_horizon_hours=0.5,
                long_horizon_steps=4,
                long_horizon_hours=1.0,
                setpoint_delta_threshold=0.05,
                min_events=8,
                coordinate_center=(0.0, 0.0, 0.0),
                coordinate_scale=(1.0, 1.0, 1.0),
                physical_kpi_scale=1.0,
            )

        value, gradient = jax.value_and_grad(score_fn)(predictions)

        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-6)

    def test_joint_response_geometry_scores_direct_and_regression_coordinates(
        self,
    ) -> None:
        perfect = synthetic_event_study_batch(particle_slopes=(2.0, 2.0, 2.0))
        biased = synthetic_event_study_batch(particle_slopes=(2.5, 3.0, 3.5))

        def joint_score(batch) -> jnp.ndarray:
            predictions, targets, inputs, initial_temperature = batch
            return targeted_flexibility_energy_score(
                predictions,
                targets,
                inputs,
                short_horizon_steps=2,
                short_horizon_hours=0.5,
                middle_horizon_steps=3,
                middle_horizon_hours=0.75,
                long_horizon_steps=4,
                long_horizon_hours=1.0,
                setpoint_delta_threshold=0.05,
                min_events=8,
                coordinate_center=(0.0,) * 8,
                coordinate_scale=(1.0,) * 8,
                physical_kpi_scale=1.0,
                initial_temperature=initial_temperature,
                regression_controls="none",
            )

        perfect_score = float(joint_score(perfect))
        biased_score = float(joint_score(biased))
        value, gradient = jax.value_and_grad(
            lambda prediction: joint_score(
                (prediction, biased[1], biased[2], biased[3])
            )
        )(biased[0])

        self.assertLess(perfect_score, 1e-3)
        self.assertGreater(biased_score, perfect_score + 0.2)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-6)

    def test_zero_threshold_does_not_count_unchanged_timesteps_as_events(self) -> None:
        predictions = jnp.zeros((1, 3, 16, 3), dtype=jnp.float32)
        targets = jnp.zeros((1, 16, 3), dtype=jnp.float32)
        inputs = jnp.zeros((1, 16, 9), dtype=jnp.float32)
        value = flexibility_direct_ratio_crps(
            predictions,
            targets,
            inputs,
            horizon_steps=(2,),
            horizon_hours=(0.5,),
            setpoint_delta_threshold=0.0,
            min_events=1,
        )
        self.assertEqual(float(value), 0.0)

    def test_grouped_windows_pool_to_the_same_ratio_for_identical_windows(self) -> None:
        predictions, targets, inputs, _ = synthetic_event_study_batch(
            particle_slopes=(2.5, 3.0, 3.5)
        )
        independent = flexibility_direct_ratio_crps(
            predictions,
            targets,
            inputs,
            horizon_steps=(2, 4),
            horizon_hours=(0.5, 1.0),
            setpoint_delta_threshold=0.05,
            min_events=8,
            windows_per_profile=1,
        )
        grouped = flexibility_direct_ratio_crps(
            predictions,
            targets,
            inputs,
            horizon_steps=(2, 4),
            horizon_hours=(0.5, 1.0),
            setpoint_delta_threshold=0.05,
            min_events=8,
            windows_per_profile=2,
        )
        self.assertAlmostEqual(float(grouped), float(independent), places=6)

    def test_grouped_rollout_reuses_xi_but_not_window_keys(self) -> None:
        class DummyModel:
            latent_dim = 2

            def sample_with_aux(
                self,
                metadata,
                inputs,
                initial_temperature,
                *,
                key,
                num_particles,
                persistent_xi=None,
                **kwargs,
            ):
                del metadata, initial_temperature, kwargs
                marker = jax.random.normal(key, (num_particles, inputs.shape[0], 1))
                predictions = jnp.broadcast_to(marker, (num_particles, inputs.shape[0], 3))
                return predictions, (persistent_xi, marker)

        metadata = jnp.zeros((4, 1), dtype=jnp.float32)
        inputs = jnp.zeros((4, 8, 2), dtype=jnp.float32)
        initial_temperature = jnp.zeros((4, 1), dtype=jnp.float32)
        _, aux = predict_probabilistic_closed_loop_batch_with_aux(
            DummyModel(),
            metadata,
            inputs,
            initial_temperature,
            key=jax.random.PRNGKey(7),
            num_particles=3,
            sample_process_noise=True,
            hp_scenario_mode="bernoulli",
            persistent_xi_group_size=2,
        )
        xi, local_marker = aux
        np.testing.assert_array_equal(np.asarray(xi[0]), np.asarray(xi[1]))
        np.testing.assert_array_equal(np.asarray(xi[2]), np.asarray(xi[3]))
        self.assertFalse(np.array_equal(np.asarray(xi[0]), np.asarray(xi[2])))
        self.assertFalse(
            np.array_equal(np.asarray(local_marker[0]), np.asarray(local_marker[1]))
        )

    def test_grouped_sampler_does_not_mix_policy_trajectories(self) -> None:
        policy_marker = np.concatenate(
            (np.zeros((4,), dtype=np.float32), np.ones((4,), dtype=np.float32))
        )
        windows = ClosedLoopWindowedArrays(
            profile_ids=np.full((8,), 7, dtype=np.int64),
            start_indices=np.tile(np.arange(4, dtype=np.int64), 2),
            metadata=np.zeros((8, 1), dtype=np.float32),
            inputs=policy_marker[:, None, None],
            targets=np.zeros((8, 1, 3), dtype=np.float32),
            initial_temperature=np.zeros((8,), dtype=np.float32),
        )

        batches = list(
            grouped_profile_minibatches(
                windows,
                batch_size=4,
                windows_per_profile=4,
                rng=np.random.default_rng(3),
                shuffle=False,
            )
        )

        self.assertEqual(len(batches), 2)
        for _, inputs, _, _ in batches:
            self.assertEqual(np.unique(inputs[:, 0, 0]).size, 1)

    def test_grouped_sampler_can_select_only_setpoint_policy(self) -> None:
        inputs = np.zeros((8, 3, 2), dtype=np.float32)
        inputs[:4, :, 0] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        inputs[4:, :, 1] = 1.0
        windows = ClosedLoopWindowedArrays(
            profile_ids=np.full((8,), 7, dtype=np.int64),
            start_indices=np.tile(np.arange(4, dtype=np.int64), 2),
            metadata=np.zeros((8, 1), dtype=np.float32),
            inputs=inputs,
            targets=np.zeros((8, 3, 3), dtype=np.float32),
            initial_temperature=np.zeros((8,), dtype=np.float32),
        )

        batches = list(
            grouped_profile_minibatches(
                windows,
                batch_size=4,
                windows_per_profile=4,
                rng=np.random.default_rng(3),
                shuffle=False,
                require_setpoint_events=True,
                setpoint_delta_threshold=0.05,
            )
        )

        self.assertEqual(len(batches), 2)
        for _, selected_inputs, _, _ in batches:
            np.testing.assert_array_equal(selected_inputs[:, :, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
