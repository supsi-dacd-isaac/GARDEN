from __future__ import annotations

import numpy as np

from neural_building_emulator_refactor.qroom_temporal_calibration_diagnostics import (
    BinaryReliabilityAccumulator,
    ConditionalRankAccumulator,
    _positive_run_lengths,
    _transition_probability,
)


def test_binary_reliability_is_exact_for_calibrated_bins() -> None:
    accumulator = BinaryReliabilityAccumulator.create()
    accumulator.add(
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
    )
    metrics = accumulator.metrics()
    assert metrics["brier_score"] == 0.0
    assert metrics["expected_calibration_error"] == 0.0


def test_conditional_rank_ignores_inactive_scenario_marks() -> None:
    truth = np.asarray([0.0, 2.0])
    scenarios = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 3.0],
        ]
    )
    accumulator = ConditionalRankAccumulator.create()
    accumulator.add(truth, scenarios, threshold=0.05)
    assert accumulator.count == 1
    assert accumulator.insufficient_scenario_count == 0
    np.testing.assert_allclose(accumulator.cdf_sum[5], 0.5)


def test_persistence_helpers() -> None:
    active = np.asarray([False, True, True, False, True, True, True, False])
    np.testing.assert_array_equal(_positive_run_lengths(active), np.asarray([2.0, 3.0]))
    assert _transition_probability(active, True) == 3.0 / 5.0
    assert _transition_probability(active, False) == 2.0 / 2.0
