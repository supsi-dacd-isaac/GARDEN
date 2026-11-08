import unittest

import numpy as np

from neural_building_emulator_refactor.buffer_energy_diagnostics import (
    audit_bucket, replay_bucket, withdrawal_before_drop,
)


class BufferEnergyDiagnosticsTests(unittest.TestCase):
    def test_replay_preserves_overflow_before_delivery(self):
        energy, q = replay_bucket(np.array([[20., 0.]]), np.array([[20., 20.]]),
                                  np.array([10.]), np.array([1.]), 10.)
        np.testing.assert_allclose(energy, [[10., 5.]])
        np.testing.assert_allclose(q, [[20., 20.]])

    def test_audit_recovers_overflow_loss_and_conservation(self):
        energy = np.array([[10., 4.]])
        generated = np.array([[20., 0.]])
        requested = np.array([[20., 20.]])
        available = np.array([[36., 14.4]])
        delivered = np.minimum(requested, available)
        result = audit_bucket(energy, generated, requested, delivered, available, 10.)
        np.testing.assert_allclose(result["overflow"], [[5., 0.]])
        np.testing.assert_allclose(result["loss"], [[1., .4]])
        np.testing.assert_allclose(result["transition_residual"], 0, atol=1e-10)
        np.testing.assert_allclose(result["residual"], 0, atol=1e-10)
        np.testing.assert_array_equal(result["limited"], [[False, True]])

    def test_signed_source_exchange_cannot_make_negative_delivery(self):
        energy, q = replay_bucket(np.array([[-20., 0.]]), np.array([[2., 2.]]),
                                  np.array([1.]), np.array([1.]), 10.)
        np.testing.assert_allclose(energy, [[1., 0.]])
        np.testing.assert_allclose(q, 0)

    def test_audit_post_delivery_cap_and_zone_area_conversion(self):
        result = audit_bucket(
            np.array([[0., 10.]]), np.array([[100., 0.]]),
            np.array([[16., 40.]]), np.array([[16., 28.8]]),
            np.array([[80., 28.8]]), 10., zone_heat_ratio=np.array([.8]),
            cap_before_delivery=False)
        np.testing.assert_allclose(result["overflow"], [[10., 0.]])
        np.testing.assert_allclose(result["loss"], [[0., 1.]])
        np.testing.assert_allclose(result["residual"], 0, atol=1e-10)
        np.testing.assert_allclose(result["transition_residual"], 0, atol=1e-10)
        np.testing.assert_array_equal(result["limited"], [[False, True]])

    def test_audit_rejects_invalid_ratio_and_capacity(self):
        x = np.ones((2, 3))
        for ratio, cap in ((0., 1.), (np.nan, 1.), (1., 0.)):
            with self.assertRaises(ValueError):
                audit_bucket(x, x, x, x, x, cap, zone_heat_ratio=ratio)

    def test_post_delivery_cap_allows_within_interval_throughput(self):
        _, old = replay_bucket(np.array([[80.]]), np.array([[60.]]), np.array([0.]), np.ones(1), 10.)
        _, after = replay_bucket(np.array([[80.]]), np.array([[60.]]), np.array([0.]), np.ones(1), 10.,
                                 cap_before_delivery=False)
        np.testing.assert_allclose(old, [[40.]])
        np.testing.assert_allclose(after, [[60.]])

    def test_withdrawal_is_censored_at_recharge(self):
        temperature = np.full(16, 21.)
        temperature[10] = 19.
        heat = np.full(16, 20.)
        generation = np.full(16, 20.)
        generation[4:9] = 0
        result = withdrawal_before_drop(temperature, heat, generation, 6, 8, 12)
        self.assertFalse(result["cooling_before_recharge"])
        self.assertEqual(result["withdrawal_start"], 4)
        self.assertEqual(result["withdrawal_to_cooling_or_censor_wh_m2"], 25.)

    def test_withdrawal_to_drop_includes_preinterruption_depletion(self):
        temperature = np.full(16, 21.)
        temperature[8] = 20.
        heat = np.full(16, 20.)
        generation = np.full(16, 20.)
        generation[4:12] = 0
        result = withdrawal_before_drop(temperature, heat, generation, 6, 10, 12)
        self.assertTrue(result["cooling_before_recharge"])
        self.assertEqual(result["no_charge_age_at_onset_h"], .5)
        self.assertEqual(result["withdrawal_to_cooling_or_censor_wh_m2"], 25.)


if __name__ == "__main__":
    unittest.main()
