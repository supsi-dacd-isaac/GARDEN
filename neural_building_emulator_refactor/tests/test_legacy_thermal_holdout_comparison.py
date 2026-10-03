from __future__ import annotations

import unittest
from argparse import Namespace

import jax.numpy as jnp
import numpy as np

from neural_building_emulator_refactor.conditional_state_space_sufficiency import (
    INPUT_NAMES,
    Scales,
)
from neural_building_emulator_refactor.legacy_thermal_holdout_comparison import (
    LegacyScales,
    _initial_model,
    _legacy_rollout,
    _online_mean_std,
)


class LegacyThermalHoldoutComparisonTests(unittest.TestCase):
    def test_online_mean_std_matches_numpy(self) -> None:
        arrays = (
            np.asarray([[1.0, 3.0], [2.0, 5.0]], dtype=np.float32),
            np.asarray([[7.0, 9.0]], dtype=np.float32),
        )
        mean, std = _online_mean_std(arrays)
        expected = np.concatenate(arrays)
        np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(std, expected.std(axis=0), rtol=1e-6)

    def test_legacy_observed_q_rollout_is_finite(self) -> None:
        metadata_dim = 4
        scales = LegacyScales(
            shared=Scales(
                metadata_mean=np.zeros(metadata_dim, dtype=np.float32),
                metadata_std=np.ones(metadata_dim, dtype=np.float32),
                exogenous_mean=np.zeros(10, dtype=np.float32),
                exogenous_std=np.ones(10, dtype=np.float32),
                temperature_mean=21.0,
                temperature_std=2.0,
                qroom_scale=10.0,
                pel_scale=5.0,
            ),
            input_mean=np.zeros(len(INPUT_NAMES), dtype=np.float32),
            input_std=np.ones(len(INPUT_NAMES), dtype=np.float32),
            qroom_mean=5.0,
            qroom_std=4.0,
            pel_mean=2.0,
            pel_std=1.5,
        )
        args = Namespace(
            state_dim=3,
            hidden_dim=8,
            depth=2,
            input_encoder_dim=3,
            input_encoder_hidden_dim=8,
            input_encoder_depth=2,
            gamma=0.99,
            state_bound=5.0,
            temperature_scale=4.0,
            temperature_delta_max_c=2.0,
            temperature_alpha_cap=0.25,
            temperature_feedback_gradient_scale=0.05,
            additive_feedback_gain_bound=1.0,
            dt_hours=0.25,
            bptt_truncate_steps=4,
            q_to_t_time_constants_hours=(0.25, 1.0, 4.0),
            q_to_t_gain_min=0.01,
            q_to_t_gain_max=2.0,
            seed=13,
        )
        model = _initial_model(scales, metadata_dim=metadata_dim, args=args)
        prediction = _legacy_rollout(
            model,
            jnp.zeros((metadata_dim,)),
            jnp.zeros((12, len(INPUT_NAMES))),
            jnp.ones((12,)) * 5.0,
            jnp.asarray(21.0),
        )
        self.assertEqual(prediction.shape, (12,))
        self.assertTrue(np.all(np.isfinite(np.asarray(prediction))))


if __name__ == "__main__":
    unittest.main()
