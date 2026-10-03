"""Tests for the differentiable event-study coefficient CRPS."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.train import (
    flexibility_coefficient_crps,
    flexibility_direct_ratio_crps,
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


if __name__ == "__main__":
    unittest.main()
