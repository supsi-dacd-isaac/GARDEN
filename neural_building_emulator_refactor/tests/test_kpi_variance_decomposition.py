from __future__ import annotations

import unittest

import numpy as np

from neural_building_emulator_refactor.kpi_variance_decomposition import (
    KpiCoordinate,
    _decomposition_rows,
    _native_calibration_rows,
    _spectrum_metrics,
    decompose_nested_samples,
)


class KpiVarianceDecompositionTest(unittest.TestCase):
    def test_scalar_decomposition_closes_exactly(self) -> None:
        xi_effect = np.asarray([-2.0, -0.5, 1.0, 3.0])[:, None, None]
        fast_effect = np.asarray([-1.0, 0.0, 2.0])[None, :, None]
        samples = 4.0 + xi_effect + fast_effect

        result = decompose_nested_samples(samples)

        np.testing.assert_allclose(result.total, result.persistent_xi + result.fast)
        self.assertGreater(float(result.persistent_xi[0, 0]), 0.0)
        self.assertGreater(float(result.fast[0, 0]), 0.0)

    def test_vector_covariance_handles_xi_dependent_fast_noise(self) -> None:
        xi = np.asarray([-1.5, -0.5, 0.5, 1.5])[:, None]
        fast = np.asarray([-1.0, 0.0, 1.0])[None, :]
        first = xi + (1.0 + 0.3 * xi) * fast
        second = -2.0 * xi + (0.5 - 0.2 * xi) * fast
        samples = np.stack((first, second), axis=-1)

        result = decompose_nested_samples(samples)

        np.testing.assert_allclose(
            result.total,
            result.persistent_xi + result.fast,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertLess(float(result.persistent_xi[0, 1]), 0.0)

    def test_invalid_kpi_is_excluded_without_pooling_other_coordinates(self) -> None:
        samples = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
        samples[:, :, 1] = np.nan
        coordinates = (
            KpiCoordinate("f", 2.0, "up"),
            KpiCoordinate("f", 2.0, "down"),
        )

        variance, covariance, _, valid_indices = _decomposition_rows(
            123,
            coordinates,
            samples,
        )

        self.assertEqual(valid_indices.tolist(), [0])
        self.assertTrue(variance[0]["valid"])
        self.assertFalse(variance[1]["valid"])
        self.assertEqual(len(covariance), 1)
        self.assertEqual(covariance[0]["left_kpi_label"], "f:up:2h")

    def test_variance_is_normalized_by_fixed_building_truth(self) -> None:
        samples = np.asarray(
            [
                [[1.0], [3.0]],
                [[3.0], [5.0]],
            ]
        )
        coordinates = (KpiCoordinate("f", 2.0, "up"),)

        variance, _, _, _ = _decomposition_rows(
            123,
            coordinates,
            samples,
            truth=np.asarray([2.0]),
        )

        self.assertEqual(variance[0]["normalization_scale"], 2.0)
        self.assertAlmostEqual(
            variance[0]["normalized_v_total"],
            variance[0]["v_total"] / 4.0,
        )
        self.assertAlmostEqual(
            variance[0]["normalized_v_xi"] + variance[0]["normalized_v_fast"],
            variance[0]["normalized_v_total"],
        )

    def test_native_calibration_reports_midrank_and_central_coverage(self) -> None:
        samples = np.asarray(
            [
                [[0.0], [1.0]],
                [[2.0], [3.0]],
            ]
        )
        coordinates = (KpiCoordinate("f", 1.0, "down"),)

        row = _native_calibration_rows(
            7,
            coordinates,
            samples,
            np.asarray([1.5]),
        )[0]

        self.assertTrue(row["valid"])
        self.assertEqual(row["sample_count"], 4)
        self.assertAlmostEqual(row["pit"], 0.5)
        self.assertAlmostEqual(row["ensemble_median"], 1.5)
        self.assertTrue(row["coverage_90"])

    def test_rejects_unbalanced_or_nonfinite_input(self) -> None:
        with self.assertRaises(ValueError):
            decompose_nested_samples(np.ones((1, 4, 2)))
        with self.assertRaises(ValueError):
            decompose_nested_samples(np.asarray([[[np.nan]], [[1.0]]]))

    def test_spectrum_metrics_report_effective_dimension(self) -> None:
        metrics = _spectrum_metrics(np.eye(2))

        self.assertAlmostEqual(metrics["top1_fraction"], 0.5)
        self.assertAlmostEqual(metrics["top4_fraction"], 1.0)
        self.assertAlmostEqual(metrics["effective_rank"], 2.0)


if __name__ == "__main__":
    unittest.main()
