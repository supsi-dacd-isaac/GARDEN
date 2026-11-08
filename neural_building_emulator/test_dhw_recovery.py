"""Causality, lockout, persistence and inference-only inputs for DHW recovery."""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.models.dhw_recovery import DHWRecovery
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator
from neural_building_emulator.test_plant_supervision import model_kwargs


class DHWRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.controller = DHWRecovery(2, 2, key=jax.random.PRNGKey(1))

    def rollout(self, draw, available, mode="expected", controller=None):
        return (controller or self.controller).rollout(
            jnp.zeros(2), jnp.zeros(2), jnp.asarray(draw, jnp.float32), jnp.asarray(available, jnp.float32),
            jnp.full(len(draw), .5), volume_m3_m2=jnp.asarray(.001), recovery_w_m2=jnp.asarray(10.),
            dt_hours=.25, scenario_mode=mode)

    def test_lockout_stops_recovery_but_preserves_demand_debt(self):
        draw = np.full(24, .1)
        available = np.r_[np.zeros(12), np.ones(12)]
        probability, active, debt = self.rollout(draw, available)
        np.testing.assert_array_equal(active[:12], 0)
        np.testing.assert_array_equal(probability[:12], 0)
        self.assertGreater(float(debt[12]), float(debt[0]))
        self.assertGreater(float(probability[12]), .95)
        self.assertLess(float(debt[-1]), float(debt[12]))

    def test_future_demand_does_not_change_past(self):
        first = self.rollout(np.zeros(24), np.ones(24))
        second = self.rollout(np.r_[np.zeros(12), np.ones(12)], np.ones(24))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a[:12], b[:12])

    def test_hard_mode_is_binary_and_gradients_are_finite(self):
        _, active, debt = self.rollout(np.full(24, .2), np.ones(24), "hard_expected_grad")
        self.assertTrue(np.isin(np.asarray(active), [0., 1.]).all())
        self.assertTrue((np.asarray(debt) >= 0).all())
        gradient = eqx.filter_grad(lambda c: self.rollout(np.full(24, .2), np.ones(24), "hard_expected_grad", c)[0].sum())(self.controller)
        self.assertTrue(all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(gradient)))

    def test_rollout_uses_only_metadata_and_known_inputs(self):
        options = model_kwargs()
        options.update(metadata_dim=2, metadata_mean=(.001, 10.), metadata_scale=(.001, 5.),
                       dhw_recovery_enabled=True, dhw_request_input_index=6,
                       dhw_volume_metadata_index=0, dhw_recovery_metadata_index=1,
                       hp_force_off_input_index=7)
        model = ProbabilisticContractingClosedLoopHPEmulator(**options)
        inputs = jnp.zeros((24, 9)).at[:, 6].set(.1).at[:12, 7].set(1.).at[:, 8].set(1.)
        predictions, aux = model.sample_with_aux(jnp.zeros(2), inputs, jnp.zeros(1),
                                                  key=jax.random.PRNGKey(7), num_particles=3,
                                                  hp_scenario_mode="bernoulli")
        self.assertEqual(predictions.shape, (3, 24, 3))
        self.assertEqual(aux[-3].shape, (3, 24))
        np.testing.assert_array_equal(aux[-2][:, :12], 0.)
        physical_power = predictions[..., 2] * options["target_scale"][2] + options["target_mean"][2]
        np.testing.assert_allclose(np.asarray(physical_power)[np.asarray(aux[-2]) > .5], 0., atol=1e-6)


if __name__ == "__main__":
    unittest.main()
