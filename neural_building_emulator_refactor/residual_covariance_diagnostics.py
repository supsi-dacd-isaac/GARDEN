"""Compare held-out KPI residual structure with persistent-latent covariance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class ResidualCovarianceResult:
    native_calibration: pd.DataFrame
    standardized_residuals: pd.DataFrame
    location_bias: pd.DataFrame
    covariance: pd.DataFrame
    modes: pd.DataFrame
    comparison: pd.DataFrame
    building_xi_spectrum: pd.DataFrame
    residual_covariance: np.ndarray
    persistent_xi_covariance: np.ndarray
    coordinate_labels: tuple[str, ...]
    scale_floor_source: str


def oas_covariance(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Oracle-approximating shrinkage covariance toward a scaled identity."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("values must have shape [building, coordinate]")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    centered = values - np.mean(values, axis=0, keepdims=True)
    empirical = centered.T @ centered / values.shape[0]
    coordinate_count = values.shape[1]
    mu = float(np.trace(empirical) / coordinate_count)
    alpha = float(np.mean(empirical**2))
    denominator = (values.shape[0] + 1.0) * (
        alpha - mu**2 / coordinate_count
    )
    shrinkage = (
        1.0
        if denominator <= 1e-15
        else min((alpha + mu**2) / denominator, 1.0)
    )
    shrunk = (1.0 - shrinkage) * empirical
    shrunk.flat[:: coordinate_count + 1] += shrinkage * mu
    return 0.5 * (shrunk + shrunk.T), float(shrinkage)


def _spectrum_metrics(covariance: np.ndarray) -> dict[str, float]:
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    total = float(np.sum(eigenvalues))
    if total <= 1e-15:
        return {
            "top1_fraction": float("nan"),
            "top2_fraction": float("nan"),
            "top4_fraction": float("nan"),
            "effective_rank": float("nan"),
        }
    fractions = eigenvalues / total
    positive = fractions[fractions > 1e-15]
    return {
        "top1_fraction": float(fractions[0]),
        "top2_fraction": float(np.sum(fractions[:2])),
        "top4_fraction": float(np.sum(fractions[:4])),
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
    }


def _correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(scale, scale)
    return np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 1e-15,
    )


def _comparison_metrics(
    residual_covariance: np.ndarray,
    xi_covariance: np.ndarray,
) -> dict[str, float]:
    residual_metrics = _spectrum_metrics(residual_covariance)
    xi_metrics = _spectrum_metrics(xi_covariance)
    denominator = float(
        np.linalg.norm(residual_covariance) * np.linalg.norm(xi_covariance)
    )
    result = {
        **{f"residual_{name}": value for name, value in residual_metrics.items()},
        **{f"xi_{name}": value for name, value in xi_metrics.items()},
        "effective_rank_difference": (
            residual_metrics["effective_rank"] - xi_metrics["effective_rank"]
        ),
        "covariance_frobenius_cosine": (
            float(np.sum(residual_covariance * xi_covariance) / denominator)
            if denominator > 1e-15
            else float("nan")
        ),
    }
    residual_trace = float(np.trace(residual_covariance))
    _, eigenvectors = np.linalg.eigh(xi_covariance)
    eigenvectors = eigenvectors[:, ::-1]
    for count in (1, 2, 4):
        basis = eigenvectors[:, : min(count, eigenvectors.shape[1])]
        projection = basis @ basis.T
        captured = float(np.trace(projection @ residual_covariance))
        result[f"residual_variance_captured_by_xi_top{count}"] = (
            captured / residual_trace if residual_trace > 1e-15 else float("nan")
        )
    return result


def _simulation_truths(path: Path, kpi_kind: str) -> pd.DataFrame:
    suffix = "_f" if kpi_kind == "f" else ""
    source = path / f"flexibility_kpis{suffix}.csv"
    if not source.exists():
        return pd.DataFrame(columns=("profile_id", "kpi_label", "truth"))
    frame = pd.read_csv(source)
    valid = frame["valid"] if frame["valid"].dtype == bool else (
        frame["valid"].astype(str).str.lower().isin(("true", "1", "yes"))
    )
    frame = frame[(frame["signal"] == "simulation") & valid]
    rows: list[dict] = []
    for row in frame.itertuples(index=False):
        horizon = float(row.horizon_hours)
        horizon_label = f"{horizon:g}h"
        for direction, column in (
            ("up", "up_flex_wh_m2_k"),
            ("down", "down_flex_wh_m2_k"),
        ):
            rows.append(
                {
                    "profile_id": int(row.profile_id),
                    "kpi_label": f"{kpi_kind}:{direction}:{horizon_label}",
                    "truth": float(getattr(row, column)),
                }
            )
    return pd.DataFrame(rows)


def _scale_floors(
    native: pd.DataFrame,
    *,
    scale_reference_dir: Path | None,
    minimum_scale: float,
    minimum_scale_fraction: float,
) -> tuple[dict[str, float], str]:
    reference_parts = []
    if scale_reference_dir is not None:
        reference_parts = [
            _simulation_truths(scale_reference_dir, kind)
            for kind in ("regression_based", "f")
        ]
        reference_parts = [part for part in reference_parts if not part.empty]
    if reference_parts:
        reference = pd.concat(reference_parts, ignore_index=True)
        source = f"calibration targets: {scale_reference_dir.resolve()}"
    else:
        reference = native[["profile_id", "kpi_label", "truth"]].copy()
        source = "held-out targets fallback"
    target_scale = reference.groupby("kpi_label", sort=False)["truth"].std(ddof=0)
    return (
        {
            label: max(
                float(minimum_scale),
                float(minimum_scale_fraction) * float(scale),
            )
            for label, scale in target_scale.items()
        },
        source,
    )


def _mode_rows(
    component: str,
    covariance: np.ndarray,
    coordinate_labels: Sequence[str],
) -> list[dict]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(np.sum(eigenvalues))
    rows: list[dict] = []
    for mode, eigenvalue in enumerate(eigenvalues, start=1):
        fraction = float(eigenvalue / total) if total > 1e-15 else float("nan")
        cumulative = (
            float(np.sum(eigenvalues[:mode]) / total)
            if total > 1e-15
            else float("nan")
        )
        loading = eigenvectors[:, mode - 1]
        if loading[np.argmax(np.abs(loading))] < 0.0:
            loading = -loading
        for label, value in zip(coordinate_labels, loading):
            rows.append(
                {
                    "component": component,
                    "mode": mode,
                    "eigenvalue": float(eigenvalue),
                    "variance_fraction": fraction,
                    "cumulative_fraction": cumulative,
                    "kpi_label": label,
                    "loading": float(value),
                }
            )
    return rows


def analyze_residual_covariance(
    native_calibration: pd.DataFrame,
    covariance_decomposition: pd.DataFrame,
    *,
    scale_reference_dir: Path | None = None,
    minimum_scale: float = 1e-3,
    minimum_scale_fraction: float = 0.01,
    bootstrap_samples: int = 2000,
    seed: int = 13,
) -> ResidualCovarianceResult:
    """Compare standardized held-out residual covariance with persistent xi."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    native = native_calibration[native_calibration["valid"].astype(bool)].copy()
    labels = tuple(dict.fromkeys(native["kpi_label"].astype(str)))
    floors, floor_source = _scale_floors(
        native,
        scale_reference_dir=scale_reference_dir,
        minimum_scale=minimum_scale,
        minimum_scale_fraction=minimum_scale_fraction,
    )
    native["rns_raw_scale"] = native["width_50"] / 1.349
    native["rns_scale_floor"] = native["kpi_label"].map(floors)
    native["rns_scale"] = np.maximum(
        native["rns_raw_scale"],
        native["rns_scale_floor"],
    )
    native["standardized_residual"] = (
        native["truth"] - native["ensemble_median"]
    ) / native["rns_scale"]

    residual_wide = native.pivot(
        index="profile_id",
        columns="kpi_label",
        values="standardized_residual",
    ).reindex(columns=labels)
    scale_wide = native.pivot(
        index="profile_id",
        columns="kpi_label",
        values="rns_scale",
    ).reindex(columns=labels)
    complete = residual_wide.notna().all(axis=1) & scale_wide.notna().all(axis=1)
    residual_wide = residual_wide.loc[complete]
    scale_wide = scale_wide.loc[complete]
    if len(residual_wide) < 3:
        raise ValueError("Residual covariance requires at least three complete buildings")

    xi_matrices: list[np.ndarray] = []
    kept_ids: list[int] = []
    xi_spectrum_rows: list[dict] = []
    for profile_id in residual_wide.index:
        part = covariance_decomposition[
            covariance_decomposition["profile_id"].astype(int) == int(profile_id)
        ]
        matrix = part.pivot(
            index="left_kpi_label",
            columns="right_kpi_label",
            values="cov_xi",
        ).reindex(index=labels, columns=labels)
        if matrix.isna().any().any():
            continue
        scale = scale_wide.loc[profile_id].to_numpy(dtype=float)
        standardized = matrix.to_numpy(dtype=float) / np.outer(scale, scale)
        standardized = 0.5 * (standardized + standardized.T)
        xi_matrices.append(standardized)
        kept_ids.append(int(profile_id))
        xi_spectrum_rows.append(
            {
                "profile_id": int(profile_id),
                **_spectrum_metrics(standardized),
            }
        )
    if len(kept_ids) < 3:
        raise ValueError(
            "Residual covariance requires at least three buildings with complete "
            "persistent-xi covariance matrices"
        )
    residual_wide = residual_wide.loc[kept_ids]
    scale_wide = scale_wide.loc[kept_ids]
    residuals = residual_wide.to_numpy(dtype=float)
    xi_stack = np.stack(xi_matrices, axis=0)
    residual_covariance, shrinkage = oas_covariance(residuals)
    xi_covariance = np.mean(xi_stack, axis=0)
    estimates = _comparison_metrics(residual_covariance, xi_covariance)
    estimates["residual_oas_shrinkage"] = shrinkage
    estimates["building_count"] = float(len(residuals))

    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {name: [] for name in estimates}
    mean_bootstrap = np.empty(
        (bootstrap_samples, residuals.shape[1]),
        dtype=np.float64,
    )
    for draw in range(bootstrap_samples):
        indices = rng.integers(0, len(residuals), size=len(residuals))
        residual_draw, draw_shrinkage = oas_covariance(residuals[indices])
        xi_draw = np.mean(xi_stack[indices], axis=0)
        metrics = _comparison_metrics(residual_draw, xi_draw)
        metrics["residual_oas_shrinkage"] = draw_shrinkage
        metrics["building_count"] = float(len(residuals))
        for name, value in metrics.items():
            bootstrap[name].append(value)
        mean_bootstrap[draw] = np.mean(residuals[indices], axis=0)

    comparison_rows = []
    for name, estimate in estimates.items():
        values = np.asarray(bootstrap[name], dtype=np.float64)
        comparison_rows.append(
            {
                "metric": name,
                "estimate": float(estimate),
                "bootstrap_median": float(np.nanmedian(values)),
                "bootstrap_ci_low": float(np.nanquantile(values, 0.025)),
                "bootstrap_ci_high": float(np.nanquantile(values, 0.975)),
            }
        )

    raw_error = native.pivot(
        index="profile_id",
        columns="kpi_label",
        values="median_bias",
    ).reindex(index=kept_ids, columns=labels)
    mean_residual = np.mean(residuals, axis=0)
    location_rows = []
    for index, label in enumerate(labels):
        location_rows.append(
            {
                "kpi_label": label,
                "mean_standardized_residual": float(mean_residual[index]),
                "bootstrap_ci_low": float(
                    np.quantile(mean_bootstrap[:, index], 0.025)
                ),
                "bootstrap_ci_high": float(
                    np.quantile(mean_bootstrap[:, index], 0.975)
                ),
                "mean_raw_prediction_minus_truth": float(raw_error[label].mean()),
            }
        )

    residual_correlation = _correlation(residual_covariance)
    xi_correlation = _correlation(xi_covariance)
    covariance_rows = []
    for left, left_label in enumerate(labels):
        for right, right_label in enumerate(labels):
            covariance_rows.append(
                {
                    "left_kpi_label": left_label,
                    "right_kpi_label": right_label,
                    "residual_covariance": float(residual_covariance[left, right]),
                    "persistent_xi_covariance": float(xi_covariance[left, right]),
                    "residual_correlation": float(residual_correlation[left, right]),
                    "persistent_xi_correlation": float(xi_correlation[left, right]),
                }
            )
    modes = pd.DataFrame(
        [
            *_mode_rows("residual_shrinkage", residual_covariance, labels),
            *_mode_rows("persistent_xi_mean", xi_covariance, labels),
        ]
    )
    standardized_residuals = residual_wide.reset_index()
    return ResidualCovarianceResult(
        native_calibration=native_calibration.merge(
            native[
                [
                    "profile_id",
                    "kpi_label",
                    "rns_raw_scale",
                    "rns_scale_floor",
                    "rns_scale",
                    "standardized_residual",
                ]
            ],
            on=["profile_id", "kpi_label"],
            how="left",
        ),
        standardized_residuals=standardized_residuals,
        location_bias=pd.DataFrame(location_rows),
        covariance=pd.DataFrame(covariance_rows),
        modes=modes,
        comparison=pd.DataFrame(comparison_rows),
        building_xi_spectrum=pd.DataFrame(xi_spectrum_rows),
        residual_covariance=residual_covariance,
        persistent_xi_covariance=xi_covariance,
        coordinate_labels=labels,
        scale_floor_source=floor_source,
    )


def write_residual_covariance_dashboard(
    result: ResidualCovarianceResult,
    path: Path,
) -> None:
    labels = list(result.coordinate_labels)
    residual_correlation = _correlation(result.residual_covariance)
    xi_correlation = _correlation(result.persistent_xi_covariance)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Shrinkage residual correlation",
            "Mean persistent-xi correlation",
            "Standardized location bias",
            "Covariance eigenvalue fractions",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )
    for column, matrix in ((1, residual_correlation), (2, xi_correlation)):
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                zmin=-1.0,
                zmax=1.0,
                colorscale="RdBu",
                reversescale=True,
                showscale=column == 2,
                hovertemplate="%{y}<br>%{x}<br>corr=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    bias = result.location_bias
    figure.add_trace(
        go.Bar(
            x=bias["kpi_label"],
            y=bias["mean_standardized_residual"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": bias["bootstrap_ci_high"] - bias["mean_standardized_residual"],
                "arrayminus": bias["mean_standardized_residual"] - bias["bootstrap_ci_low"],
            },
            marker_color="#1769aa",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for component, color in (
        ("residual_shrinkage", "#d95f02"),
        ("persistent_xi_mean", "#16846b"),
    ):
        part = result.modes[result.modes["component"] == component]
        spectrum = part.groupby("mode", sort=True)["variance_fraction"].first()
        figure.add_trace(
            go.Bar(
                x=spectrum.index,
                y=spectrum.values,
                name=component,
                marker_color=color,
            ),
            row=2,
            col=2,
        )
    figure.add_hline(y=0.0, line_color="black", line_width=1, row=2, col=1)
    figure.update_xaxes(tickangle=45, row=2, col=1)
    figure.update_xaxes(title_text="mode", row=2, col=2)
    figure.update_yaxes(title_text="mean standardized residual", row=2, col=1)
    figure.update_yaxes(title_text="variance fraction", row=2, col=2)
    figure.update_layout(
        title="Held-out residual covariance versus persistent-latent covariance",
        barmode="group",
        height=1250,
        width=1800,
        margin={"b": 220},
        hoverlabel={"namelength": -1},
    )
    figure.write_html(path, include_plotlyjs="cdn")
