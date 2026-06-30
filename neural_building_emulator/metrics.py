"""Evaluation metrics for emulator rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    nmae: float
    bias: float


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> RegressionMetrics:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    error = pred - true
    rmse = float(math.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.mean(np.abs(true)))
    nmae = float(mae / denominator) if denominator > 1e-12 else float("nan")
    bias = float(np.mean(error))
    return RegressionMetrics(rmse=rmse, mae=mae, nmae=nmae, bias=bias)
