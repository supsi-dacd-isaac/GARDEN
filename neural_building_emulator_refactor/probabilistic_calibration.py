"""Distribution calibration and diagnostics shared by emulators and LightGBM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


DEFAULT_QUANTILES = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
GROUP_COLUMNS = ("method", "kpi_kind", "horizon_hours", "direction")


@dataclass(frozen=True)
class ResidualNormalizedCalibrator:
    residuals: np.ndarray
    scale_floor: float

    def samples(self, location: float, scale: float, count: int) -> np.ndarray:
        quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
        residual_quantiles = np.quantile(self.residuals, quantiles)
        return location + max(scale, self.scale_floor) * residual_quantiles


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    return array[np.isfinite(array)]


def distribution_location_scale(
    samples: Sequence[float] | np.ndarray,
    *,
    scale_floor: float,
) -> tuple[float, float]:
    values = _finite(samples)
    if not len(values):
        return float("nan"), float("nan")
    location = float(np.median(values))
    scale = float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349)
    return location, max(scale, scale_floor)


def ensemble_crps(samples: Sequence[float] | np.ndarray, truth: float) -> float:
    """Exact empirical CRPS in O(K log K), including duplicate particles."""
    values = np.sort(_finite(samples))
    if not len(values) or not np.isfinite(truth):
        return float("nan")
    count = len(values)
    first = float(np.mean(np.abs(values - truth)))
    coefficients = 2.0 * np.arange(1, count + 1) - count - 1.0
    half_pairwise = float(np.sum(coefficients * values) / (count * count))
    return first - half_pairwise


def fit_residual_normalized_calibrator(
    frame: pd.DataFrame,
    *,
    minimum_scale: float = 1e-3,
    minimum_scale_fraction: float = 0.01,
) -> ResidualNormalizedCalibrator:
    """Fit signed standardized residuals from one row per scenario and building."""
    required = {"profile_id", "truth", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Calibration frame is missing columns: {sorted(missing)}")
    target = frame.groupby("profile_id", sort=False)["truth"].first().to_numpy(float)
    target = target[np.isfinite(target)]
    target_scale = float(np.std(target)) if len(target) else 0.0
    floor = max(float(minimum_scale), minimum_scale_fraction * target_scale)
    residuals: list[float] = []
    for _, group in frame.groupby("profile_id", sort=False):
        truth = float(group["truth"].iloc[0])
        location, scale = distribution_location_scale(
            group["prediction"].to_numpy(float),
            scale_floor=floor,
        )
        if np.isfinite(truth) and np.isfinite(location) and np.isfinite(scale):
            residuals.append((truth - location) / scale)
    values = _finite(residuals)
    if len(values) < 3:
        raise ValueError(
            "Residual-normalized calibration requires at least three valid buildings"
        )
    return ResidualNormalizedCalibrator(values, floor)


def _safe_correlation(kind: str, truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 3 or np.std(truth) < 1e-12 or np.std(prediction) < 1e-12:
        return float("nan")
    result = spearmanr(truth, prediction) if kind == "spearman" else kendalltau(truth, prediction)
    return float(result.statistic)


def _top_fraction_overlap(truth: np.ndarray, prediction: np.ndarray, fraction: float = 0.1) -> float:
    if not len(truth):
        return float("nan")
    count = max(1, int(np.ceil(fraction * len(truth))))
    true_top = set(np.argsort(truth)[-count:])
    predicted_top = set(np.argsort(prediction)[-count:])
    return float(len(true_top.intersection(predicted_top)) / count)


def summarize_distributions(
    frame: pd.DataFrame,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    train_scale: float | None = None,
) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    """Return aggregate metrics, reliability rows, and per-building summaries."""
    building_rows: list[dict[str, float]] = []
    grouped_samples: list[tuple[float, np.ndarray]] = []
    for profile_id, group in frame.groupby("profile_id", sort=False):
        truth = float(group["truth"].iloc[0])
        samples = _finite(group["prediction"].to_numpy(float))
        if not np.isfinite(truth) or not len(samples):
            continue
        location = float(np.median(samples))
        grouped_samples.append((truth, samples))
        building_rows.append(
            {
                "profile_id": int(profile_id),
                "truth": truth,
                "median": location,
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "q025": float(np.quantile(samples, 0.025)),
                "q10": float(np.quantile(samples, 0.10)),
                "q90": float(np.quantile(samples, 0.90)),
                "q975": float(np.quantile(samples, 0.975)),
                "crps": ensemble_crps(samples, truth),
            }
        )
    buildings = pd.DataFrame(building_rows)
    if buildings.empty:
        empty_metrics = {
            key: float("nan")
            for key in (
                "count", "rmse", "nrmse", "mae", "bias", "r2", "spearman",
                "kendall", "top_10pct_overlap", "crps", "coverage_50", "width_50",
                "coverage_80", "width_80", "coverage_90", "width_90",
                "coverage_95", "width_95", "calibration_mae",
            )
        }
        return empty_metrics, [], []
    truth = buildings["truth"].to_numpy(float)
    prediction = buildings["median"].to_numpy(float)
    error = prediction - truth
    scale = max(
        float(train_scale) if train_scale is not None else float(np.std(truth)),
        1e-8,
    )
    metrics = {
        "count": float(len(buildings)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nrmse": float(np.sqrt(np.mean(error**2)) / scale),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / max(np.sum((truth - np.mean(truth)) ** 2), 1e-12)),
        "spearman": _safe_correlation("spearman", truth, prediction),
        "kendall": _safe_correlation("kendall", truth, prediction),
        "top_10pct_overlap": _top_fraction_overlap(truth, prediction),
        "crps": float(buildings["crps"].mean()),
    }
    reliability: list[dict[str, float]] = []
    for quantile in quantiles:
        empirical = float(
            np.mean([truth_value <= np.quantile(samples, quantile) for truth_value, samples in grouped_samples])
        )
        reliability.append(
            {
                "quantile": float(quantile),
                "empirical_coverage": empirical,
                "calibration_error": empirical - float(quantile),
            }
        )
    metrics["calibration_mae"] = float(
        np.mean([abs(row["calibration_error"]) for row in reliability])
    )
    for nominal, low, high in (
        (50, 0.25, 0.75),
        (80, 0.10, 0.90),
        (90, 0.05, 0.95),
        (95, 0.025, 0.975),
    ):
        lower = np.asarray([np.quantile(samples, low) for _, samples in grouped_samples])
        upper = np.asarray([np.quantile(samples, high) for _, samples in grouped_samples])
        metrics[f"coverage_{nominal}"] = float(np.mean((truth >= lower) & (truth <= upper)))
        metrics[f"width_{nominal}"] = float(np.mean(upper - lower))
    return metrics, reliability, buildings.to_dict(orient="records")


def calibrate_distribution_table(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_columns: Sequence[str] = GROUP_COLUMNS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    minimum_scale: float = 1e-3,
    minimum_scale_fraction: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate each method/KPI/horizon/direction and score raw and calibrated samples."""
    required = {*group_columns, "profile_id", "truth", "prediction"}
    for name, frame in (("calibration", calibration), ("test", test)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} distribution table is missing: {sorted(missing)}")
    calibrated_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    building_rows: list[dict[str, object]] = []
    calibration_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): value
        for key, value in calibration.groupby(list(group_columns), dropna=False, sort=False)
    }
    for key, test_group in test.groupby(list(group_columns), dropna=False, sort=False):
        normalized_key = tuple(key if isinstance(key, tuple) else (key,))
        if normalized_key not in calibration_groups:
            continue
        calibration_group = calibration_groups[normalized_key]
        calibrator = fit_residual_normalized_calibrator(
            calibration_group,
            minimum_scale=minimum_scale,
            minimum_scale_fraction=minimum_scale_fraction,
        )
        target_scale = float(
            np.std(calibration_group.groupby("profile_id")["truth"].first().to_numpy(float))
        )
        labels = dict(zip(group_columns, normalized_key))
        raw_metrics, raw_reliability, raw_buildings = summarize_distributions(
            test_group,
            quantiles=quantiles,
            train_scale=target_scale,
        )
        metric_rows.append({**labels, "calibration": "raw", **raw_metrics})
        reliability_rows.extend(
            {**labels, "calibration": "raw", **row} for row in raw_reliability
        )
        building_rows.extend(
            {**labels, "calibration": "raw", **row} for row in raw_buildings
        )

        generated: list[pd.DataFrame] = []
        for profile_id, profile_group in test_group.groupby("profile_id", sort=False):
            truth = float(profile_group["truth"].iloc[0])
            raw_samples = profile_group["prediction"].to_numpy(float)
            location, scale = distribution_location_scale(
                raw_samples,
                scale_floor=calibrator.scale_floor,
            )
            count = max(20, len(_finite(raw_samples)))
            samples = calibrator.samples(location, scale, count)
            generated.append(
                pd.DataFrame(
                    {
                        **{column: labels[column] for column in group_columns},
                        "profile_id": int(profile_id),
                        "truth": truth,
                        "sample_index": np.arange(count),
                        "prediction": samples,
                    }
                )
            )
        calibrated = pd.concat(generated, ignore_index=True)
        calibrated_frames.append(calibrated)
        calibrated_metrics, calibrated_reliability, calibrated_buildings = summarize_distributions(
            calibrated,
            quantiles=quantiles,
            train_scale=target_scale,
        )
        metric_rows.append({**labels, "calibration": "rns", **calibrated_metrics})
        reliability_rows.extend(
            {**labels, "calibration": "rns", **row}
            for row in calibrated_reliability
        )
        building_rows.extend(
            {**labels, "calibration": "rns", **row}
            for row in calibrated_buildings
        )
    calibrated_table = (
        pd.concat(calibrated_frames, ignore_index=True)
        if calibrated_frames
        else pd.DataFrame(columns=list(required) + ["sample_index"])
    )
    return (
        calibrated_table,
        pd.DataFrame(metric_rows),
        pd.DataFrame(reliability_rows),
        pd.DataFrame(building_rows),
    )
