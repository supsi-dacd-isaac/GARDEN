from __future__ import annotations

import unittest

import numpy as np

from neural_building_emulator.flexibility_event_study_kpis import (
    OLSAccumulator,
    RegressionData,
    TemporalTraceCalibrationAccumulator,
    _window_mean,
    estimate_direct_kpi_from_regression_data,
    estimate_kpi_from_regression_data,
)


class FlexibilityEventStudyEstimatorTests(unittest.TestCase):
    def test_temporal_calibration_randomizes_zero_mass_ties_analytically(self) -> None:
        quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
        accumulator = TemporalTraceCalibrationAccumulator(quantiles)
        truth = np.zeros((8, 3), dtype=np.float64)
        scenarios = np.zeros((20, 8, 3), dtype=np.float64)

        accumulator.add(truth, scenarios)

        reliability = accumulator.reliability_rows()
        for row in reliability:
            self.assertAlmostEqual(row["empirical_coverage"], row["quantile"])
        histogram = accumulator.histogram_rows()
        for row in histogram:
            self.assertAlmostEqual(row["probability"], 0.1)

    def test_temporal_calibration_canonicalizes_near_zero_power_atoms(self) -> None:
        quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
        accumulator = TemporalTraceCalibrationAccumulator(quantiles)
        truth = np.zeros((8, 3), dtype=np.float64)
        scenarios = np.zeros((20, 8, 3), dtype=np.float64)
        scenarios[:, :, 1:] = np.float64(2.3841858e-7)

        accumulator.add(truth, scenarios)

        reliability = accumulator.reliability_rows()
        power_rows = [row for row in reliability if row["signal"] != "Tin"]
        for row in power_rows:
            self.assertAlmostEqual(row["empirical_coverage"], row["quantile"])

    def test_temporal_calibration_respects_per_signal_masks(self) -> None:
        accumulator = TemporalTraceCalibrationAccumulator((0.5,))
        truth = np.zeros((6, 3), dtype=np.float64)
        scenarios = np.zeros((4, 6, 3), dtype=np.float64)
        masks = np.ones((6, 3), dtype=bool)
        masks[:2, 1] = False
        masks[:4, 2] = False

        accumulator.add(truth, scenarios, masks=masks)

        counts = {
            row["signal"]: row["observation_count"]
            for row in accumulator.reliability_rows()
        }
        self.assertEqual(counts, {"Tin": 6, "Qroom": 4, "Pel_SH": 2})

    def test_temporal_probabilistic_scores_use_exact_ensemble_crps(self) -> None:
        accumulator = TemporalTraceCalibrationAccumulator((0.05, 0.5, 0.95))
        truth = np.full((3, 3), 0.5, dtype=np.float64)
        scenarios = np.stack(
            [np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)]
        )

        accumulator.add(truth, scenarios)

        scores = {row["signal"]: row for row in accumulator.probabilistic_score_rows()}
        self.assertAlmostEqual(scores["Tin"]["mean_crps"], 0.25)
        self.assertAlmostEqual(scores["Tin"]["mean_central_90_width"], 0.9)
        self.assertEqual(scores["Tin"]["observation_count"], 3)

    def test_vectorized_window_means_match_independent_reductions(self) -> None:
        rng = np.random.default_rng(13)
        starts = np.asarray([0, 3, 9, 17])
        ends = starts + 4
        for values in (
            rng.normal(size=24),
            rng.normal(size=(24, 3)),
            rng.normal(size=24).astype(np.float32),
        ):
            expected = np.asarray(
                [np.mean(values[start:end], axis=0) for start, end in zip(starts, ends)],
                dtype=np.float64,
            )
            actual = _window_mean(values, starts, ends)
            np.testing.assert_array_equal(actual, expected)

    def test_direct_ratio_uses_ratio_of_directional_sums(self) -> None:
        deltas = np.asarray([1.0, 2.0, -1.0, -3.0])
        response = np.asarray([2.0, 6.0, -4.0, -8.0])
        design = np.column_stack(
            [np.ones(4), np.maximum(deltas, 0.0), np.minimum(deltas, 0.0)]
        )
        data = RegressionData(
            design=design,
            response=response,
            events=np.arange(4),
            deltas=deltas,
        )

        result = estimate_direct_kpi_from_regression_data(
            profile_id=7,
            horizon_hours=3.0,
            horizon_steps=12,
            data=data,
            min_events=4,
        )

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.beta_plus_w_m2_k, 8.0 / 3.0)
        self.assertAlmostEqual(result.beta_minus_w_m2_k, 3.0)
        self.assertAlmostEqual(result.up_flex_wh_m2_k, 8.0)
        self.assertAlmostEqual(result.down_flex_wh_m2_k, 9.0)

    def test_regression_estimator_remains_available(self) -> None:
        deltas = np.asarray([1.0, 2.0, -1.0, -2.0, 3.0, -3.0])
        design = np.column_stack(
            [np.ones(6), np.maximum(deltas, 0.0), np.minimum(deltas, 0.0)]
        )
        response = 0.5 + 2.0 * design[:, 1] + 4.0 * design[:, 2]
        data = RegressionData(
            design=design,
            response=response,
            events=np.arange(6),
            deltas=deltas,
        )

        result = estimate_kpi_from_regression_data(
            profile_id=8,
            horizon_hours=2.0,
            horizon_steps=8,
            data=data,
            min_events=4,
        )

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.beta_plus_w_m2_k, 2.0)
        self.assertAlmostEqual(result.beta_minus_w_m2_k, 4.0)
        self.assertAlmostEqual(result.up_flex_wh_m2_k, 4.0)
        self.assertAlmostEqual(result.down_flex_wh_m2_k, 8.0)

    def test_pooled_direct_ratio_aggregates_events_before_dividing(self) -> None:
        accumulator = OLSAccumulator()
        for deltas, response in (
            (np.asarray([1.0, -1.0]), np.asarray([1.0, -2.0])),
            (np.asarray([3.0, -2.0]), np.asarray([9.0, -8.0])),
        ):
            design = np.column_stack(
                [np.ones(2), np.maximum(deltas, 0.0), np.minimum(deltas, 0.0)]
            )
            accumulator.add(
                RegressionData(
                    design=design,
                    response=response,
                    events=np.arange(2),
                    deltas=deltas,
                )
            )

        result = accumulator.estimate(
            profile_id=-1,
            horizon_hours=2.0,
            horizon_steps=8,
            min_events=4,
            estimator="direct_ratio",
        )

        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.beta_plus_w_m2_k, 2.5)
        self.assertAlmostEqual(result.beta_minus_w_m2_k, 10.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
