"""Tests for blockwise closed-loop oracle diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from neural_building_emulator.blockwise_oracle_diagnostics import (
    _causal_history,
    hp_oracle_features,
    qroom_oracle_features,
    thermal_static_features,
)
from neural_building_emulator.data import ClosedLoopProfile


def _profile(steps: int = 32) -> ClosedLoopProfile:
    inputs = np.zeros((steps, 9), dtype=np.float32)
    inputs[:, 0] = 20.0
    inputs[:, 1] = np.linspace(0.0, 10.0, steps)
    targets = np.zeros((steps, 3), dtype=np.float32)
    targets[:, 0] = 20.0 + np.linspace(0.0, 1.0, steps)
    targets[:, 1] = np.arange(steps) % 5
    targets[:, 2] = np.arange(steps) % 7
    return ClosedLoopProfile(
        profile_id=1,
        datetime=np.arange(steps),
        metadata=np.empty((0,), dtype=np.float32),
        inputs=inputs,
        targets=targets,
        initial_temperature=targets[:, :1] - 0.01,
    )


class BlockwiseOracleTests(unittest.TestCase):
    def test_history_is_strictly_causal(self) -> None:
        values = np.arange(10, dtype=np.float64)
        original = _causal_history(values, (2, 4))
        modified_values = values.copy()
        modified_values[6:] = 1000.0
        modified = _causal_history(modified_values, (2, 4))

        np.testing.assert_allclose(original[:7], modified[:7])
        self.assertEqual(original[5, 0], values[4])

    def test_oracle_feature_shapes(self) -> None:
        profile = _profile()
        hp = hp_oracle_features(profile, history_steps=(2, 4), heat_on_threshold=1e-6)
        q_pel = qroom_oracle_features(
            profile,
            history_steps=(2, 4),
            include_qroom_history=False,
        )
        q_history = qroom_oracle_features(
            profile,
            history_steps=(2, 4),
            include_qroom_history=True,
        )
        thermal = thermal_static_features(profile, history_steps=(2, 4))

        self.assertEqual(hp.shape[0], 32)
        self.assertEqual(q_pel.shape[0], 32)
        self.assertGreater(q_history.shape[1], q_pel.shape[1])
        self.assertEqual(thermal.shape[0], 32)


if __name__ == "__main__":
    unittest.main()
