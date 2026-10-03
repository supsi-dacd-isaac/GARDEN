"""Focused invariants for the building-local causal hybrid HP emulator."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from .models.causal_hybrid_hp import CausalHybridHPEmulator


def make_model(**overrides) -> CausalHybridHPEmulator:
    kwargs = {
        "input_mean": (20.0, 5.0, 100.0, 0.1, 1.0, 0.0, 1.0, 0.0, 1.0),
        "input_scale": (2.0, 10.0, 100.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0),
        "temperature_mean": 20.0,
        "temperature_scale": 2.0,
        "history_dim": 104,
        "key": jax.random.PRNGKey(0),
    }
    kwargs.update(overrides)
    return CausalHybridHPEmulator(**kwargs)


class CausalHybridHPTest(unittest.TestCase):
    def test_positive_hydronic_convolution_conserves_available_energy(self) -> None:
        model = make_model()
        stored = jnp.asarray([2.0, 3.0, 4.0])
        source_power = jnp.asarray(10.0)
        next_stored, q_room = model._hydronic_step(stored, source_power)
        available_energy = float(jnp.sum(stored) + model.dt_hours * source_power)
        accounted = float(jnp.sum(next_stored) + model.dt_hours * q_room)
        self.assertGreaterEqual(float(q_room), 0.0)
        self.assertTrue(bool(jnp.all(next_stored >= 0.0)))
        self.assertLessEqual(accounted, available_energy + 1e-5)

    def test_rc_transition_is_passive_and_has_no_setpoint_argument(self) -> None:
        model = make_model()
        temperatures = jnp.asarray([20.0, 19.0, 18.0])
        first = model._rc_step(
            temperatures,
            outdoor_temperature=jnp.asarray(5.0),
            solar=jnp.asarray(0.0),
            ventilation=jnp.asarray(0.0),
            q_room=jnp.asarray(0.0),
        )
        second = model._rc_step(
            temperatures,
            outdoor_temperature=jnp.asarray(5.0),
            solar=jnp.asarray(0.0),
            ventilation=jnp.asarray(0.0),
            q_room=jnp.asarray(0.0),
        )
        np.testing.assert_allclose(first, second)
        self.assertTrue(bool(jnp.all(first <= 20.0 + 1e-5)))
        self.assertTrue(bool(jnp.all(first >= 5.0 - 1e-5)))

    def test_rollout_shapes_and_bounds(self) -> None:
        model = make_model()
        horizon = 24
        inputs = jnp.tile(
            jnp.asarray([20.0, 5.0, 100.0, 0.1, 1.0, 0.0, 1.0, 0.0, 1.0]),
            (horizon, 1),
        )
        distribution = jnp.zeros((2, model.max_dwell_steps + 1)).at[0, 0].set(1.0)
        rollout = model(
            inputs,
            jnp.asarray(20.0),
            jnp.zeros((104,)),
            distribution,
            jnp.zeros((horizon,), dtype=jnp.int32),
            jnp.arange(horizon, dtype=jnp.int32),
            jnp.arange(horizon) % model.control_interval_steps == 0,
        )
        self.assertEqual(rollout.prediction.shape, (horizon, 3))
        self.assertEqual(rollout.duration_logits.shape, (horizon, 13))
        self.assertEqual(rollout.tau_power_mean.shape, (horizon, 4))
        self.assertTrue(bool(jnp.all(jnp.isfinite(rollout.prediction))))
        self.assertTrue(bool(jnp.all(rollout.prediction[:, 1:] >= 0.0)))
        self.assertTrue(bool(jnp.all((rollout.mode_probability >= 0.0) & (rollout.mode_probability <= 1.0))))


if __name__ == "__main__":
    unittest.main()
