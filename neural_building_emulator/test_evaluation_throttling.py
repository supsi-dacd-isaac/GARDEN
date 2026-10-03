"""Tests for bounded-memory evaluation and fixed train-evaluation subsets."""

from __future__ import annotations

import unittest

import numpy as np

from neural_building_emulator.data import ClosedLoopWindowedArrays, WindowedArrays
from neural_building_emulator.metrics import regression_metrics
from neural_building_emulator.train import (
    _RegressionAccumulator,
    _TemperatureDiagnosticsAccumulator,
    fixed_evaluation_subset,
)


def _windows(window_type: type[WindowedArrays] | type[ClosedLoopWindowedArrays]):
    n_windows = 20
    return window_type(
        profile_ids=np.arange(n_windows, dtype=np.int64) + 100,
        start_indices=np.arange(n_windows, dtype=np.int64) * 96,
        metadata=np.arange(n_windows * 2, dtype=np.float32).reshape(n_windows, 2),
        inputs=np.arange(n_windows * 4 * 3, dtype=np.float32).reshape(n_windows, 4, 3),
        targets=np.arange(n_windows * 4, dtype=np.float32).reshape(n_windows, 4),
        initial_temperature=np.arange(n_windows, dtype=np.float32).reshape(n_windows, 1),
    )


class RegressionAccumulatorTests(unittest.TestCase):
    def test_streaming_metrics_match_concatenated_metrics(self) -> None:
        rng = np.random.default_rng(12)
        target = rng.normal(size=(17, 11, 3))
        prediction = target + rng.normal(scale=0.4, size=target.shape)
        accumulator = _RegressionAccumulator()
        for start in range(0, target.shape[0], 5):
            accumulator.update(prediction[start : start + 5], target[start : start + 5])

        expected = regression_metrics(prediction, target)
        actual = accumulator.metrics()
        self.assertAlmostEqual(actual["rmse"], expected.rmse, places=12)
        self.assertAlmostEqual(actual["mae"], expected.mae, places=12)
        self.assertAlmostEqual(actual["nmae"], expected.nmae, places=12)
        self.assertAlmostEqual(actual["bias"], expected.bias, places=12)

    def test_temperature_diagnostics_stream_across_batches(self) -> None:
        accumulator = _TemperatureDiagnosticsAccumulator()
        accumulator.update(
            np.asarray([[20.0, 21.0], [18.0, 19.0]]),
            np.asarray([[20.0, 20.0], [20.0, 20.0]]),
            np.asarray([1, 2]),
            np.asarray([0, 96]),
        )
        accumulator.update(
            np.asarray([[22.0, 27.0]]),
            np.asarray([[21.0, 20.0]]),
            np.asarray([3]),
            np.asarray([192]),
        )

        metrics = accumulator.as_dict()
        self.assertEqual(metrics["max_abs_error_c"], 7.0)
        self.assertEqual(metrics["max_abs_error_profile_id"], 3.0)
        self.assertEqual(metrics["max_abs_error_start"], 192.0)
        self.assertEqual(metrics["max_abs_error_step"], 1.0)
        self.assertEqual(metrics["min_pred_c"], 18.0)
        self.assertEqual(metrics["max_pred_c"], 27.0)


class FixedEvaluationSubsetTests(unittest.TestCase):
    def test_subset_is_reproducible_and_slices_every_field(self) -> None:
        windows = _windows(ClosedLoopWindowedArrays)
        first = fixed_evaluation_subset(windows, max_windows=7, seed=41)
        second = fixed_evaluation_subset(windows, max_windows=7, seed=41)

        self.assertIsInstance(first, ClosedLoopWindowedArrays)
        self.assertEqual(first.targets.shape[0], 7)
        np.testing.assert_array_equal(first.profile_ids, second.profile_ids)
        selected = first.profile_ids - 100
        np.testing.assert_array_equal(first.metadata, windows.metadata[selected])
        np.testing.assert_array_equal(first.inputs, windows.inputs[selected])
        np.testing.assert_array_equal(first.targets, windows.targets[selected])
        np.testing.assert_array_equal(first.initial_temperature, windows.initial_temperature[selected])

    def test_zero_limit_keeps_all_windows(self) -> None:
        windows = _windows(WindowedArrays)
        subset = fixed_evaluation_subset(windows, max_windows=0, seed=41)
        self.assertIs(subset, windows)


if __name__ == "__main__":
    unittest.main()
