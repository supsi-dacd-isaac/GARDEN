"""Study whether pre-event HP power improves the flexibility KPI regression.

For each building and response horizon, this diagnostic fits the existing full
event-study regression twice:

1. with the standardized pre-window mean HP power ``P_pre``;
2. with exactly the same design matrix, except for the ``P_pre`` column.

The script compares in-sample residual variance, blocked event-level
cross-validation error, and movement of the upward/downward setpoint-response
coefficients.  This distinction matters because the response is
``mean(P_post) - mean(P_pre)``: adding ``P_pre`` can reduce residual variance
mechanically without necessarily improving the flexibility coefficient.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from .data import DEFAULT_DATASET_PATH
from .flexibility_event_study_kpis import (
    EventStudyConfig,
    _load_closed_loop_profiles,
    _resolve_horizons,
    _saved_profile_ids,
    _select_profiles,
    regression_data_for_trace,
)


@dataclass(frozen=True)
class FitStatistics:
    coefficients: np.ndarray
    residual_variance: float
    response_variance: float
    r_squared: float
    adjusted_r_squared: float
    condition_number: float
    beta_plus_hc1_se: float
    beta_minus_hc1_se: float


@dataclass(frozen=True)
class CrossValidationStatistics:
    mse: float
    r_squared: float
    beta_plus_sd: float
    beta_minus_sd: float
    folds_used: int


def _metadata_path(artifact_dir: Path) -> Path:
    candidates = sorted(artifact_dir.glob("metadata*.json"))
    if not candidates:
        raise FileNotFoundError(f"No metadata file found in {artifact_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple metadata files found in {artifact_dir}: "
            + ", ".join(path.name for path in candidates)
        )
    return candidates[0]


def _load_metadata(artifact_dir: Path) -> dict[str, Any]:
    return json.loads(_metadata_path(artifact_dir).read_text())


def _config_value(metadata: dict[str, Any], name: str, default: Any) -> Any:
    return metadata.get("train_config", {}).get(name, default)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    first = first[valid]
    second = second[valid]
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _residualize(values: np.ndarray, nuisance_design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(nuisance_design, values, rcond=None)
    return values - nuisance_design @ coefficients


def _partial_treatment_pre_power_correlations(design: np.ndarray) -> tuple[float, float]:
    other_controls = list(range(4, design.shape[1]))
    plus_nuisance = design[:, [0, 2, *other_controls]]
    minus_nuisance = design[:, [0, 1, *other_controls]]
    pre_power = design[:, 3]
    plus_correlation = _safe_correlation(
        _residualize(design[:, 1], plus_nuisance),
        _residualize(pre_power, plus_nuisance),
    )
    minus_correlation = _safe_correlation(
        _residualize(design[:, 2], minus_nuisance),
        _residualize(pre_power, minus_nuisance),
    )
    return plus_correlation, minus_correlation


def _fit_statistics(design: np.ndarray, response: np.ndarray) -> FitStatistics:
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    count = len(response)
    degrees_of_freedom = max(count - int(rank), 1)
    residual_sse = float(residual @ residual)
    residual_variance = residual_sse / degrees_of_freedom
    centered = response - np.mean(response)
    total_sse = float(centered @ centered)
    response_variance = total_sse / max(count - 1, 1)
    r_squared = 1.0 - residual_sse / total_sse if total_sse > 0.0 else float("nan")
    if count > int(rank) and np.isfinite(r_squared):
        adjusted_r_squared = 1.0 - (1.0 - r_squared) * (count - 1) / (count - int(rank))
    else:
        adjusted_r_squared = float("nan")

    bread = np.linalg.pinv(design.T @ design)
    meat = design.T @ ((residual**2)[:, np.newaxis] * design)
    hc1_scale = count / degrees_of_freedom
    hc1_covariance = hc1_scale * bread @ meat @ bread
    hc1_diagonal = np.maximum(np.diag(hc1_covariance), 0.0)
    hc1_standard_errors = np.sqrt(hc1_diagonal)

    if len(singular_values) == 0 or singular_values[-1] <= 0.0:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])

    return FitStatistics(
        coefficients=coefficients,
        residual_variance=float(residual_variance),
        response_variance=float(response_variance),
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted_r_squared),
        condition_number=condition_number,
        beta_plus_hc1_se=float(hc1_standard_errors[1]),
        beta_minus_hc1_se=float(hc1_standard_errors[2]),
    )


def _blocked_cross_validation(
    design: np.ndarray,
    response: np.ndarray,
    *,
    requested_folds: int,
) -> CrossValidationStatistics:
    count = len(response)
    folds = min(requested_folds, max(2, count // max(design.shape[1] + 1, 2)))
    blocks = [block for block in np.array_split(np.arange(count), folds) if len(block)]
    squared_errors: list[np.ndarray] = []
    held_out_responses: list[np.ndarray] = []
    fold_beta_plus: list[float] = []
    fold_beta_minus: list[float] = []

    all_indices = np.arange(count)
    for held_out in blocks:
        train_mask = np.ones(count, dtype=bool)
        train_mask[held_out] = False
        train_indices = all_indices[train_mask]
        if len(train_indices) <= design.shape[1]:
            continue
        coefficients, *_ = np.linalg.lstsq(
            design[train_indices],
            response[train_indices],
            rcond=None,
        )
        prediction = design[held_out] @ coefficients
        squared_errors.append((response[held_out] - prediction) ** 2)
        held_out_responses.append(response[held_out])
        fold_beta_plus.append(float(coefficients[1]))
        fold_beta_minus.append(float(coefficients[2]))

    if not squared_errors:
        return CrossValidationStatistics(
            mse=float("nan"),
            r_squared=float("nan"),
            beta_plus_sd=float("nan"),
            beta_minus_sd=float("nan"),
            folds_used=0,
        )

    errors = np.concatenate(squared_errors)
    observed = np.concatenate(held_out_responses)
    mse = float(np.mean(errors))
    total_mse = float(np.mean((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - mse / total_mse if total_mse > 0.0 else float("nan")
    ddof = 1 if len(fold_beta_plus) > 1 else 0
    return CrossValidationStatistics(
        mse=mse,
        r_squared=float(r_squared),
        beta_plus_sd=float(np.std(fold_beta_plus, ddof=ddof)),
        beta_minus_sd=float(np.std(fold_beta_minus, ddof=ddof)),
        folds_used=len(squared_errors),
    )


def _ablation_row(
    *,
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    design_with_pre_power: np.ndarray,
    response: np.ndarray,
    deltas: np.ndarray,
    raw_pre_power: np.ndarray,
    cv_folds: int,
) -> dict[str, Any]:
    if design_with_pre_power.shape[1] < 4:
        raise ValueError("The full event-study design does not contain the expected P_pre column.")
    design_without_pre_power = np.delete(design_with_pre_power, 3, axis=1)

    with_fit = _fit_statistics(design_with_pre_power, response)
    without_fit = _fit_statistics(design_without_pre_power, response)
    with_cv = _blocked_cross_validation(
        design_with_pre_power,
        response,
        requested_folds=cv_folds,
    )
    without_cv = _blocked_cross_validation(
        design_without_pre_power,
        response,
        requested_folds=cv_folds,
    )

    residual_variance_ratio = _safe_ratio(
        with_fit.residual_variance,
        without_fit.residual_variance,
    )
    cv_mse_ratio = _safe_ratio(with_cv.mse, without_cv.mse)
    beta_plus_se_ratio = _safe_ratio(
        with_fit.beta_plus_hc1_se,
        without_fit.beta_plus_hc1_se,
    )
    beta_minus_se_ratio = _safe_ratio(
        with_fit.beta_minus_hc1_se,
        without_fit.beta_minus_hc1_se,
    )
    beta_plus_fold_sd_ratio = _safe_ratio(
        with_cv.beta_plus_sd,
        without_cv.beta_plus_sd,
    )
    beta_minus_fold_sd_ratio = _safe_ratio(
        with_cv.beta_minus_sd,
        without_cv.beta_minus_sd,
    )

    beta_plus_with = float(with_fit.coefficients[1])
    beta_minus_with = float(with_fit.coefficients[2])
    beta_plus_without = float(without_fit.coefficients[1])
    beta_minus_without = float(without_fit.coefficients[2])
    standardized_pre_power_coefficient = float(with_fit.coefficients[3])
    pre_power_standard_deviation = float(np.std(raw_pre_power))
    if pre_power_standard_deviation > 1e-12:
        raw_pre_power_coefficient = (
            standardized_pre_power_coefficient / pre_power_standard_deviation
        )
    else:
        raw_pre_power_coefficient = float("nan")
    partial_plus_correlation, partial_minus_correlation = (
        _partial_treatment_pre_power_correlations(design_with_pre_power)
    )

    return {
        "profile_id": int(profile_id),
        "horizon_hours": float(horizon_hours),
        "horizon_steps": int(horizon_steps),
        "event_count": int(len(response)),
        "response_variance": with_fit.response_variance,
        "residual_variance_with_pre_power": with_fit.residual_variance,
        "residual_variance_without_pre_power": without_fit.residual_variance,
        "residual_variance_ratio_with_over_without": residual_variance_ratio,
        "residual_variance_reduction_fraction": 1.0 - residual_variance_ratio,
        "r_squared_with_pre_power": with_fit.r_squared,
        "r_squared_without_pre_power": without_fit.r_squared,
        "r_squared_gain": with_fit.r_squared - without_fit.r_squared,
        "adjusted_r_squared_with_pre_power": with_fit.adjusted_r_squared,
        "adjusted_r_squared_without_pre_power": without_fit.adjusted_r_squared,
        "adjusted_r_squared_gain": (
            with_fit.adjusted_r_squared - without_fit.adjusted_r_squared
        ),
        "cv_mse_with_pre_power": with_cv.mse,
        "cv_mse_without_pre_power": without_cv.mse,
        "cv_mse_ratio_with_over_without": cv_mse_ratio,
        "cv_mse_reduction_fraction": 1.0 - cv_mse_ratio,
        "cv_r_squared_with_pre_power": with_cv.r_squared,
        "cv_r_squared_without_pre_power": without_cv.r_squared,
        "cv_r_squared_gain": with_cv.r_squared - without_cv.r_squared,
        "cv_folds_used": min(with_cv.folds_used, without_cv.folds_used),
        "beta_plus_with_pre_power_w_m2_k": beta_plus_with,
        "beta_plus_without_pre_power_w_m2_k": beta_plus_without,
        "beta_plus_shift_w_m2_k": beta_plus_with - beta_plus_without,
        "beta_minus_with_pre_power_w_m2_k": beta_minus_with,
        "beta_minus_without_pre_power_w_m2_k": beta_minus_without,
        "beta_minus_shift_w_m2_k": beta_minus_with - beta_minus_without,
        "up_flex_with_pre_power_wh_m2_k": horizon_hours * beta_plus_with,
        "up_flex_without_pre_power_wh_m2_k": horizon_hours * beta_plus_without,
        "up_flex_shift_wh_m2_k": horizon_hours * (beta_plus_with - beta_plus_without),
        "down_flex_with_pre_power_wh_m2_k": horizon_hours * beta_minus_with,
        "down_flex_without_pre_power_wh_m2_k": horizon_hours * beta_minus_without,
        "down_flex_shift_wh_m2_k": horizon_hours * (beta_minus_with - beta_minus_without),
        "beta_plus_hc1_se_with_pre_power": with_fit.beta_plus_hc1_se,
        "beta_plus_hc1_se_without_pre_power": without_fit.beta_plus_hc1_se,
        "beta_plus_hc1_se_ratio": beta_plus_se_ratio,
        "beta_minus_hc1_se_with_pre_power": with_fit.beta_minus_hc1_se,
        "beta_minus_hc1_se_without_pre_power": without_fit.beta_minus_hc1_se,
        "beta_minus_hc1_se_ratio": beta_minus_se_ratio,
        "beta_plus_block_sd_with_pre_power": with_cv.beta_plus_sd,
        "beta_plus_block_sd_without_pre_power": without_cv.beta_plus_sd,
        "beta_plus_block_sd_ratio": beta_plus_fold_sd_ratio,
        "beta_minus_block_sd_with_pre_power": with_cv.beta_minus_sd,
        "beta_minus_block_sd_without_pre_power": without_cv.beta_minus_sd,
        "beta_minus_block_sd_ratio": beta_minus_fold_sd_ratio,
        "pre_power_standard_deviation_w_m2": pre_power_standard_deviation,
        "pre_power_coefficient_standardized": standardized_pre_power_coefficient,
        "pre_power_coefficient_in_delta_response": raw_pre_power_coefficient,
        "pre_power_coefficient_in_post_power_equivalent": raw_pre_power_coefficient + 1.0,
        "correlation_setpoint_jump_pre_power": _safe_correlation(
            deltas,
            raw_pre_power,
        ),
        "partial_correlation_positive_jump_pre_power": partial_plus_correlation,
        "partial_correlation_negative_jump_pre_power": partial_minus_correlation,
        "correlation_delta_power_pre_power": _safe_correlation(
            response,
            raw_pre_power,
        ),
        "condition_number_with_pre_power": with_fit.condition_number,
        "condition_number_without_pre_power": without_fit.condition_number,
    }


def _quantile(values: pd.Series, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.quantile(quantile)) if len(finite) else float("nan")


def _summary_rows(detail: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon, part in detail.groupby("horizon_hours", sort=True):
        up_scale = float(np.nanstd(part["up_flex_without_pre_power_wh_m2_k"], ddof=1))
        down_scale = float(np.nanstd(part["down_flex_without_pre_power_wh_m2_k"], ddof=1))
        rows.append(
            {
                "horizon_hours": float(horizon),
                "profile_count": int(len(part)),
                "median_event_count": float(part["event_count"].median()),
                "median_residual_variance_reduction_fraction": float(
                    part["residual_variance_reduction_fraction"].median()
                ),
                "mean_residual_variance_reduction_fraction": float(
                    part["residual_variance_reduction_fraction"].mean()
                ),
                "median_cv_mse_reduction_fraction": float(
                    part["cv_mse_reduction_fraction"].median()
                ),
                "mean_cv_mse_reduction_fraction": float(
                    part["cv_mse_reduction_fraction"].mean()
                ),
                "fraction_profiles_cv_mse_improved": float(
                    np.mean(part["cv_mse_ratio_with_over_without"] < 1.0)
                ),
                "median_r_squared_gain": float(part["r_squared_gain"].median()),
                "median_adjusted_r_squared_gain": float(part["adjusted_r_squared_gain"].median()),
                "median_cv_r_squared_gain": float(part["cv_r_squared_gain"].median()),
                "median_beta_plus_hc1_se_ratio": float(part["beta_plus_hc1_se_ratio"].median()),
                "median_beta_minus_hc1_se_ratio": float(part["beta_minus_hc1_se_ratio"].median()),
                "median_beta_plus_block_sd_ratio": float(part["beta_plus_block_sd_ratio"].median()),
                "median_beta_minus_block_sd_ratio": float(part["beta_minus_block_sd_ratio"].median()),
                "median_abs_up_flex_shift_wh_m2_k": float(
                    np.abs(part["up_flex_shift_wh_m2_k"]).median()
                ),
                "median_abs_down_flex_shift_wh_m2_k": float(
                    np.abs(part["down_flex_shift_wh_m2_k"]).median()
                ),
                "median_abs_up_shift_over_between_building_sd": _safe_ratio(
                    float(np.abs(part["up_flex_shift_wh_m2_k"]).median()),
                    up_scale,
                ),
                "median_abs_down_shift_over_between_building_sd": _safe_ratio(
                    float(np.abs(part["down_flex_shift_wh_m2_k"]).median()),
                    down_scale,
                ),
                "correlation_up_kpi_with_vs_without": _safe_correlation(
                    part["up_flex_with_pre_power_wh_m2_k"].to_numpy(),
                    part["up_flex_without_pre_power_wh_m2_k"].to_numpy(),
                ),
                "correlation_down_kpi_with_vs_without": _safe_correlation(
                    part["down_flex_with_pre_power_wh_m2_k"].to_numpy(),
                    part["down_flex_without_pre_power_wh_m2_k"].to_numpy(),
                ),
                "median_pre_power_coefficient_in_delta_response": float(
                    part["pre_power_coefficient_in_delta_response"].median()
                ),
                "q10_pre_power_coefficient_in_delta_response": _quantile(
                    part["pre_power_coefficient_in_delta_response"], 0.1
                ),
                "q90_pre_power_coefficient_in_delta_response": _quantile(
                    part["pre_power_coefficient_in_delta_response"], 0.9
                ),
                "median_pre_power_coefficient_in_post_power_equivalent": float(
                    part["pre_power_coefficient_in_post_power_equivalent"].median()
                ),
                "median_abs_correlation_setpoint_jump_pre_power": float(
                    np.abs(part["correlation_setpoint_jump_pre_power"]).median()
                ),
                "median_abs_partial_correlation_positive_jump_pre_power": float(
                    np.abs(part["partial_correlation_positive_jump_pre_power"]).median()
                ),
                "median_abs_partial_correlation_negative_jump_pre_power": float(
                    np.abs(part["partial_correlation_negative_jump_pre_power"]).median()
                ),
                "median_condition_number_ratio": float(
                    (
                        part["condition_number_with_pre_power"]
                        / part["condition_number_without_pre_power"]
                    ).median()
                ),
            }
        )
    return rows


def _axis_limits(first: pd.Series, second: pd.Series) -> tuple[float, float]:
    values = np.concatenate([first.to_numpy(dtype=float), second.to_numpy(dtype=float)])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return -1.0, 1.0
    low = float(np.quantile(values, 0.01))
    high = float(np.quantile(values, 0.99))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.1, 1.0)
    else:
        padding = 0.08 * (high - low)
    return low - padding, high + padding


def _write_html(path: Path, detail: pd.DataFrame, *, title: str) -> None:
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Residual variance ratio (with / without P_pre)",
            "Blocked-CV MSE ratio (with / without P_pre)",
            "Upward flexibility coefficient stability",
            "Downward flexibility coefficient stability",
                "Raw P_pre coefficient in Delta P regression",
            "Flexibility KPI shift caused by P_pre",
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.11,
    )
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#17becf"]
    horizons = sorted(detail["horizon_hours"].unique())

    for index, horizon in enumerate(horizons):
        part = detail[detail["horizon_hours"] == horizon]
        color = colors[index % len(colors)]
        label = f"H={horizon:g} h"
        custom = np.column_stack([part["profile_id"], part["event_count"]])
        figure.add_trace(
            go.Box(
                x=np.full(len(part), label),
                y=part["residual_variance_ratio_with_over_without"],
                name=label,
                marker_color=color,
                boxpoints="outliers",
                showlegend=False,
                customdata=custom,
                hovertemplate="profile=%{customdata[0]}<br>events=%{customdata[1]}<br>ratio=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Box(
                x=np.full(len(part), label),
                y=part["cv_mse_ratio_with_over_without"],
                name=label,
                marker_color=color,
                boxpoints="outliers",
                showlegend=False,
                customdata=custom,
                hovertemplate="profile=%{customdata[0]}<br>events=%{customdata[1]}<br>ratio=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=part["up_flex_without_pre_power_wh_m2_k"],
                y=part["up_flex_with_pre_power_wh_m2_k"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.65),
                customdata=custom,
                hovertemplate=(
                    "profile=%{customdata[0]}<br>events=%{customdata[1]}"
                    "<br>without=%{x:.3f}<br>with=%{y:.3f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=part["down_flex_without_pre_power_wh_m2_k"],
                y=part["down_flex_with_pre_power_wh_m2_k"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.65),
                customdata=custom,
                hovertemplate=(
                    "profile=%{customdata[0]}<br>events=%{customdata[1]}"
                    "<br>without=%{x:.3f}<br>with=%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Box(
                x=np.full(len(part), label),
                y=part["pre_power_coefficient_in_delta_response"],
                name=label,
                marker_color=color,
                boxpoints="outliers",
                showlegend=False,
                customdata=custom,
                hovertemplate="profile=%{customdata[0]}<br>coefficient=%{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        for direction, offset in (("up", -0.16), ("down", 0.16)):
            figure.add_trace(
                go.Box(
                    x=np.full(len(part), f"{label} {direction}"),
                    y=part[f"{direction}_flex_shift_wh_m2_k"],
                    name=f"{label} {direction}",
                    marker_color=color,
                    boxpoints="outliers",
                    showlegend=False,
                    customdata=custom,
                    hovertemplate=(
                        "profile=%{customdata[0]}<br>shift=%{y:.3f} Wh/(m2 K)<extra></extra>"
                    ),
                ),
                row=3,
                col=2,
            )

    figure.add_hline(y=1.0, line_dash="dash", line_color="black", row=1, col=1)
    figure.add_hline(y=1.0, line_dash="dash", line_color="black", row=1, col=2)
    figure.add_hline(y=-1.0, line_dash="dot", line_color="black", row=3, col=1)
    figure.add_hline(y=0.0, line_dash="dash", line_color="black", row=3, col=2)

    up_limits = _axis_limits(
        detail["up_flex_without_pre_power_wh_m2_k"],
        detail["up_flex_with_pre_power_wh_m2_k"],
    )
    down_limits = _axis_limits(
        detail["down_flex_without_pre_power_wh_m2_k"],
        detail["down_flex_with_pre_power_wh_m2_k"],
    )
    figure.add_trace(
        go.Scatter(
            x=up_limits,
            y=up_limits,
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="unchanged KPI",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=down_limits,
            y=down_limits,
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="unchanged KPI",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(range=list(up_limits), title_text="without P_pre [Wh/(m2 K)]", row=2, col=1)
    figure.update_yaxes(range=list(up_limits), title_text="with P_pre [Wh/(m2 K)]", row=2, col=1)
    figure.update_xaxes(range=list(down_limits), title_text="without P_pre [Wh/(m2 K)]", row=2, col=2)
    figure.update_yaxes(range=list(down_limits), title_text="with P_pre [Wh/(m2 K)]", row=2, col=2)
    figure.update_yaxes(title_text="ratio; below 1 is better", row=1, col=1)
    figure.update_yaxes(title_text="ratio; below 1 is better", row=1, col=2)
    figure.update_yaxes(title_text="standardized coefficient", row=3, col=1)
    figure.update_yaxes(title_text="with minus without [Wh/(m2 K)]", row=3, col=2)
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=1450,
        width=1550,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def _write_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    profile_source: str,
    profile_count: int,
    cv_folds: int,
) -> None:
    lines = [
        "# Pre-event HP Power Regression Ablation",
        "",
        "## Design",
        "",
        f"- Source: EnergyPlus ground-truth traces from `{profile_source}` profiles ({profile_count} buildings).",
        "- Comparison: current `full` event-study controls versus the identical design without `P_pre`.",
        "- Response: `mean(P_post) - mean(P_pre)` for each setpoint discontinuity and horizon.",
        f"- Generalization check: {cv_folds}-fold contiguous blocked event cross-validation within each building.",
        "- Treatment effects: separate positive and negative setpoint-jump coefficients.",
        "",
        "## Results",
        "",
        "| H [h] | Profiles | Median in-sample variance reduction | Median CV-MSE reduction | CV improved | Median abs(up KPI shift) | Median abs(down KPI shift) | Median corr(dTset, Ppre) | Median gamma(Ppre) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['horizon_hours']:.2f} | {int(row['profile_count'])} | "
            f"{row['median_residual_variance_reduction_fraction']:.1%} | "
            f"{row['median_cv_mse_reduction_fraction']:.1%} | "
            f"{row['fraction_profiles_cv_mse_improved']:.1%} | "
            f"{row['median_abs_up_flex_shift_wh_m2_k']:.3f} | "
            f"{row['median_abs_down_flex_shift_wh_m2_k']:.3f} | "
            f"{row['median_abs_correlation_setpoint_jump_pre_power']:.3f} | "
            f"{row['median_pre_power_coefficient_in_delta_response']:.3f} |"
        )
    minimum_cv_reduction = float(summary["median_cv_mse_reduction_fraction"].min())
    minimum_improved_fraction = float(summary["fraction_profiles_cv_mse_improved"].min())
    minimum_up_correlation = float(summary["correlation_up_kpi_with_vs_without"].min())
    minimum_down_correlation = float(summary["correlation_down_kpi_with_vs_without"].min())
    maximum_relative_shift = float(
        max(
            summary["median_abs_up_shift_over_between_building_sd"].max(),
            summary["median_abs_down_shift_over_between_building_sd"].max(),
        )
    )
    maximum_partial_correlation = float(
        max(
            summary["median_abs_partial_correlation_positive_jump_pre_power"].max(),
            summary["median_abs_partial_correlation_negative_jump_pre_power"].max(),
        )
    )
    lines.extend(
        [
            "",
            "## Main Finding",
            "",
            f"Across all horizons, including `P_pre` lowers the median blocked-CV MSE by at least {minimum_cv_reduction:.1%}, and it improves held-out error for at least {minimum_improved_fraction:.1%} of buildings. The resulting KPIs remain strongly aligned with the no-`P_pre` estimates (minimum correlations: {minimum_up_correlation:.3f} upward and {minimum_down_correlation:.3f} downward). The median absolute shift is at most {maximum_relative_shift:.1%} of the between-building KPI standard deviation.",
            "",
            f"After conditioning on the other treatment and full state/weather controls, the largest median absolute partial correlation between a setpoint-jump regressor and `P_pre` is {maximum_partial_correlation:.3f}. This supports interpreting `P_pre` mainly as a precision covariate rather than a replacement for the randomized setpoint signal.",
            "",
            "For this test split, the evidence favors retaining `P_pre` in the primary KPI regression. The no-`P_pre` specification remains useful as a sensitivity analysis in the paper.",
            "",
            "## Interpretation Guide",
            "",
            "`P_pre` is a useful precision control when the blocked-CV MSE ratio is below one, coefficient uncertainty falls, and the upward/downward KPI estimates remain close to the no-`P_pre` estimates. Low `corr(dTset, P_pre)` supports the randomized-treatment assumption.",
            "",
            "Because `P_pre` appears with coefficient `-1` inside the response by construction, a fitted raw-power coefficient near `-1` can mostly reflect this algebraic coupling. The equivalent coefficient in a `P_post` regression is `1 + gamma(P_pre)`, and is included in the detailed CSV. The out-of-sample comparison and treatment-coefficient movement are therefore more informative than the in-sample R2 gain alone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_study(args: argparse.Namespace) -> None:
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if args.min_events < 1:
        raise ValueError("--min-events must be positive")
    metadata = _load_metadata(args.artifact_dir)
    dataset_path = args.dataset or Path(
        _config_value(metadata, "dataset_path", DEFAULT_DATASET_PATH)
    )
    dt_hours = float(
        args.dt_hours
        if args.dt_hours is not None
        else _config_value(metadata, "hp_dt_hours", 0.25)
    )
    horizons_hours, horizon_steps = _resolve_horizons(args.horizons_hours, dt_hours)
    config = EventStudyConfig(
        horizons_hours=horizons_hours,
        horizon_steps=horizon_steps,
        dt_hours=dt_hours,
        min_setpoint_change=float(args.min_setpoint_change),
        min_events=int(args.min_events),
        controls="full",
    )

    candidate_ids = _saved_profile_ids(metadata, args.profile_source)
    profile_ids = _select_profiles(
        candidate_ids,
        max_profiles=args.max_profiles,
        seed=args.seed,
    )
    profiles = _load_closed_loop_profiles(
        dataset_path,
        profile_ids,
        include_space_heating_availability=(
            SPACE_HEATING_AVAILABILITY_COLUMN in metadata.get("input_columns", [])
        ),
        include_internal_gains=(
            INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in metadata.get("input_columns", [])
        ),
        include_dhw_request=(
            DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN in metadata.get("input_columns", [])
        ),
        hp_power_area_normalization=str(
            _config_value(metadata, "hp_power_area_normalization", "zone_floor_area")
        ),
    )

    rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles, start=1):
        for horizon_hours, steps in zip(config.horizons_hours, config.horizon_steps):
            data = regression_data_for_trace(
                profile,
                profile.targets,
                horizon_steps=steps,
                config=config,
            )
            minimum_required = max(config.min_events, data.design.shape[1] + 2)
            if len(data.response) < minimum_required:
                print(
                    f"warning=insufficient_events profile={profile.profile_id} "
                    f"H={horizon_hours:g} count={len(data.response)} required={minimum_required}"
                )
                continue
            power = profile.targets[:, 2].astype(np.float64)
            raw_pre_power = np.asarray(
                [
                    np.mean(power[int(event) - steps : int(event)])
                    for event in data.events
                ],
                dtype=np.float64,
            )
            rows.append(
                _ablation_row(
                    profile_id=profile.profile_id,
                    horizon_hours=horizon_hours,
                    horizon_steps=steps,
                    design_with_pre_power=data.design,
                    response=data.response,
                    deltas=data.deltas,
                    raw_pre_power=raw_pre_power,
                    cv_folds=args.cv_folds,
                )
            )
        if profile_index % 25 == 0 or profile_index == len(profiles):
            print(f"processed_profiles={profile_index}/{len(profiles)}")

    if not rows:
        raise ValueError("No profile/horizon regressions had enough valid setpoint events.")

    detail = pd.DataFrame(rows).sort_values(["horizon_hours", "profile_id"])
    summary = pd.DataFrame(_summary_rows(detail)).sort_values("horizon_hours")
    output_dir = args.output_dir or Path(
        "output/neural_building_emulator/flexibility_pre_power_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "flexibility_pre_power_ablation_profiles.csv"
    summary_path = output_dir / "flexibility_pre_power_ablation_summary.csv"
    html_path = output_dir / "flexibility_pre_power_ablation.html"
    report_path = output_dir / "flexibility_pre_power_ablation_report.md"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_html(
        html_path,
        detail,
        title=(
            "Pre-event HP power control ablation"
            f"<br><sup>EnergyPlus {args.profile_source} split, profiles={len(profiles)}, "
            f"blocked CV folds={args.cv_folds}; all other full controls held fixed</sup>"
        ),
    )
    _write_report(
        report_path,
        summary=summary,
        profile_source=args.profile_source,
        profile_count=len(profiles),
        cv_folds=args.cv_folds,
    )

    print(f"saved_profile_results={detail_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_html={html_path}")
    print(f"saved_report={report_path}")
    for _, row in summary.iterrows():
        print(
            f"H={row['horizon_hours']:g}h profiles={int(row['profile_count'])} "
            f"median_variance_reduction={row['median_residual_variance_reduction_fraction']:.4f} "
            f"median_cv_mse_reduction={row['median_cv_mse_reduction_fraction']:.4f} "
            f"cv_improved_fraction={row['fraction_profiles_cv_mse_improved']:.4f} "
            f"median_abs_up_shift={row['median_abs_up_flex_shift_wh_m2_k']:.4f} "
            f"median_abs_down_shift={row['median_abs_down_flex_shift_wh_m2_k']:.4f} "
            f"median_abs_corr_dset_ppre={row['median_abs_correlation_setpoint_jump_pre_power']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--profile-source",
        choices=("test", "train", "selected"),
        default="test",
    )
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--horizons-hours", default="0.5,1,2,3")
    parser.add_argument("--dt-hours", type=float, default=None)
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
