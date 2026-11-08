from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from neural_building_emulator_refactor.residual_covariance_diagnostics import (
    analyze_residual_covariance,
    oas_covariance,
)


class ResidualCovarianceDiagnosticsTest(unittest.TestCase):
    def test_oas_covariance_is_finite_symmetric_and_positive_semidefinite(self) -> None:
        values = np.asarray(
            [
                [1.0, 0.0, -1.0],
                [2.0, 1.0, -2.0],
                [-1.0, 2.0, 1.0],
                [0.0, -1.0, 0.5],
            ]
        )

        covariance, shrinkage = oas_covariance(values)

        self.assertTrue(np.all(np.isfinite(covariance)))
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-12)
        self.assertGreaterEqual(shrinkage, 0.0)
        self.assertLessEqual(shrinkage, 1.0)

    def test_detects_residual_modes_missing_from_one_factor_xi(self) -> None:
        rng = np.random.default_rng(19)
        labels = ("f:up:1h", "f:down:1h", "f:up:2h", "f:down:2h")
        native_rows: list[dict] = []
        covariance_rows: list[dict] = []
        xi_loading = np.asarray([1.0, 1.0, 1.0, 1.0])
        xi_covariance = 0.04 * np.outer(xi_loading, xi_loading)
        for profile_id in range(30):
            shared = rng.normal()
            horizon_slope = rng.normal()
            up_down = rng.normal()
            residual = (
                shared * np.asarray([0.5, 0.5, 0.5, 0.5])
                + horizon_slope * np.asarray([-1.0, -1.0, 1.0, 1.0])
                + up_down * np.asarray([1.0, -1.0, 1.0, -1.0])
            )
            for index, label in enumerate(labels):
                native_rows.append(
                    {
                        "profile_id": profile_id,
                        "kpi_label": label,
                        "valid": True,
                        "truth": float(residual[index]),
                        "ensemble_median": 0.0,
                        "median_bias": float(-residual[index]),
                        "width_50": 1.349,
                    }
                )
                for right_index, right_label in enumerate(labels):
                    covariance_rows.append(
                        {
                            "profile_id": profile_id,
                            "left_kpi_label": label,
                            "right_kpi_label": right_label,
                            "cov_xi": float(xi_covariance[index, right_index]),
                        }
                    )

        result = analyze_residual_covariance(
            pd.DataFrame(native_rows),
            pd.DataFrame(covariance_rows),
            bootstrap_samples=100,
            seed=7,
        )
        comparison = result.comparison.set_index("metric")

        self.assertGreater(
            comparison.loc["effective_rank_difference", "estimate"],
            1.0,
        )
        self.assertGreater(
            comparison.loc["effective_rank_difference", "bootstrap_ci_low"],
            0.0,
        )
        self.assertLess(
            comparison.loc[
                "residual_variance_captured_by_xi_top1", "estimate"
            ],
            0.5,
        )
        self.assertEqual(len(result.standardized_residuals), 30)


if __name__ == "__main__":
    unittest.main()
