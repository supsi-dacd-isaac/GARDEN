"""Compare KPI recalibration while preserving complete emulator trajectories.

The benchmark keeps the selected emulator frozen.  It compares its raw KPI
ensemble with residual-normalized scores (RNS), RNS with ensemble copula
coupling (RNS-ECC), and density-ratio weights attached to whole annual paths.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import multivariate_normal, rankdata
from sklearn.covariance import OAS

from neural_building_emulator.columns import (
    SETPOINT_TIMESERIES_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from neural_building_emulator.data import ClosedLoopProfile

from .artifacts import json_safe, load_artifact
from .calibrated_model_benchmark import _emulator_distribution
from .prediction import predict_profile
from .probabilistic_calibration import calibrate_distribution_table, ecc_reorder
from .profiles import load_profiles
from .routed_latent_targeted_loss_ablation import _markdown_table


DEFAULT_BASE_ROOT = Path(
    "output/neural_building_emulator_refactor/"
    "architecture_screen_metadata_xi_all_pathways_flex_direct_high150_"
    "process_mild_tau_poles_200_e10"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/kpi_trajectory_recalibration"
)
FOLDS = ("calibration", "test")
KPI_KINDS = ("regression_based", "f")
CORE_COLUMNS = (
    "log_room_energy",
    "log_electric_energy",
    "log_pel_peak_p99",
    "flex_index_asinh",
    "log_comfort_degree_hours",
)
RESPONSE_COLUMNS = (
    "log_room_energy",
    "log_electric_energy",
    "log_pel_peak_p99",
    "log_comfort_degree_hours",
    "direct_short_up",
    "direct_asym",
    "direct_slope",
)
METHOD_ORDER = (
    "raw",
    "rns",
    "rns_ecc",
    "trajectory_reweight",
    "trajectory_reweight_response7",
)
QUANTILES = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))


@dataclass(frozen=True)
class EmosModel:
    center: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    ensemble_mean_coefficient: np.ndarray
    variance_intercept: np.ndarray
    ensemble_variance_coefficient: np.ndarray
    residual_correlation: np.ndarray
    optimization_success: tuple[bool, ...]

    def parameters(self, columns: Sequence[str] = CORE_COLUMNS) -> dict[str, Any]:
        return {
            "core_columns": list(columns),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "intercept": self.intercept.tolist(),
            "ensemble_mean_coefficient": self.ensemble_mean_coefficient.tolist(),
            "variance_intercept": self.variance_intercept.tolist(),
            "ensemble_variance_coefficient": (
                self.ensemble_variance_coefficient.tolist()
            ),
            "residual_correlation": self.residual_correlation.tolist(),
            "optimization_success": list(self.optimization_success),
        }


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    return array[np.isfinite(array)]


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array >= 0.0)
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    if not len(values_array) or float(np.sum(weights_array)) <= 0.0:
        return float("nan")
    order = np.argsort(values_array, kind="stable")
    values_array = values_array[order]
    weights_array = weights_array[order]
    weights_array /= np.sum(weights_array)
    positions = np.cumsum(weights_array) - 0.5 * weights_array
    return float(
        np.interp(
            float(quantile),
            np.concatenate(([0.0], positions, [1.0])),
            np.concatenate(([values_array[0]], values_array, [values_array[-1]])),
        )
    )


def weighted_crps(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    truth: float,
) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array >= 0.0)
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    if not len(values_array) or not np.isfinite(truth) or np.sum(weights_array) <= 0.0:
        return float("nan")
    weights_array /= np.sum(weights_array)
    first = float(np.sum(weights_array * np.abs(values_array - truth)))
    pairwise = np.abs(values_array[:, None] - values_array[None, :])
    second = 0.5 * float(np.sum(weights_array[:, None] * weights_array[None, :] * pairwise))
    return first - second


def weighted_energy_score(
    values: np.ndarray,
    weights: np.ndarray,
    truth: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    weights = weights / np.sum(weights)
    first = float(np.sum(weights * np.linalg.norm(values - truth[None, :], axis=1)))
    pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    second = 0.5 * float(np.sum(weights[:, None] * weights[None, :] * pairwise))
    return first - second


def weighted_variogram_score(
    values: np.ndarray,
    weights: np.ndarray,
    truth: np.ndarray,
    *,
    power: float = 0.5,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    weights = weights / np.sum(weights)
    row, column = np.triu_indices(values.shape[1], k=1)
    observed = np.abs(truth[row] - truth[column]) ** power
    ensemble = np.sum(
        weights[:, None] * np.abs(values[:, row] - values[:, column]) ** power,
        axis=0,
    )
    return float(np.mean((observed - ensemble) ** 2))


def _weighted_correlation(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    mean = np.sum(weights[:, None] * values, axis=0)
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.outer(std, std)
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def _rank_correlation(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ranks = np.column_stack(
        [rankdata(values[:, index], method="average") for index in range(values.shape[1])]
    )
    return _weighted_correlation(ranks, weights)


def stabilized_density_ratio_weights(
    log_ratio: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    minimum_ess: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(log_ratio, dtype=np.float64)
    values = np.where(np.isfinite(values), values, -np.inf)
    if not np.any(np.isfinite(values)):
        values = np.zeros_like(values)

    def weights_for(effective_alpha: float) -> np.ndarray:
        logits = effective_alpha * values
        finite = np.isfinite(logits)
        if not np.any(finite):
            return np.full(len(values), 1.0 / len(values))
        logits = np.where(finite, logits - np.max(logits[finite]), -np.inf)
        weights = np.exp(np.clip(logits, -700.0, 0.0))
        return weights / np.sum(weights)

    requested = max(float(alpha), 0.0)
    weights = weights_for(requested)
    ess = 1.0 / float(np.sum(weights**2))
    if ess >= minimum_ess or requested == 0.0:
        return weights, requested
    low, high = 0.0, requested
    for _ in range(50):
        middle = 0.5 * (low + high)
        candidate = weights_for(middle)
        candidate_ess = 1.0 / float(np.sum(candidate**2))
        if candidate_ess >= minimum_ess:
            low = middle
        else:
            high = middle
    return weights_for(low), low


def _gaussian_crps(truth: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(std, 1e-6)
    z = (truth - mean) / std
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return std * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * density - 1.0 / np.sqrt(np.pi))


def _positive_definite_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-8))
    correlation = covariance / np.outer(std, std)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    correlation = (eigenvectors * np.maximum(eigenvalues, 1e-4)) @ eigenvectors.T
    std = np.sqrt(np.diag(correlation))
    correlation /= np.outer(std, std)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def fit_emos(particles: np.ndarray, truth: np.ndarray) -> EmosModel:
    """Fit exchangeable-member Gaussian EMOS independently by coordinate."""
    particles = np.asarray(particles, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    raw_mean = np.mean(particles, axis=1)
    center = np.mean(raw_mean, axis=0)
    scale = np.maximum(
        np.std(np.concatenate((raw_mean, truth), axis=0), axis=0),
        1e-3,
    )
    standardized_particles = (particles - center[None, None, :]) / scale[None, None, :]
    standardized_truth = (truth - center[None, :]) / scale[None, :]
    ensemble_mean = np.mean(standardized_particles, axis=1)
    ensemble_variance = np.var(standardized_particles, axis=1)
    dimension = particles.shape[2]
    intercept = np.zeros(dimension)
    mean_coefficient = np.ones(dimension)
    variance_intercept = np.full(dimension, 0.05)
    variance_coefficient = np.ones(dimension)
    success: list[bool] = []
    for index in range(dimension):
        target = standardized_truth[:, index]
        raw_location = ensemble_mean[:, index]
        raw_variance = ensemble_variance[:, index]

        def objective(parameters: np.ndarray) -> float:
            a, b, log_c, log_d = parameters
            c = np.exp(log_c)
            d = np.exp(log_d)
            location = a + b * raw_location
            std = np.sqrt(c + d * raw_variance + 1e-8)
            regularization = 1e-3 * (a**2 + (b - 1.0) ** 2 + log_d**2)
            return float(np.mean(_gaussian_crps(target, location, std)) + regularization)

        initial_error = target - raw_location
        initial = np.asarray(
            [
                float(np.mean(initial_error)),
                1.0,
                float(np.log(max(np.var(initial_error) * 0.25, 1e-3))),
                0.0,
            ]
        )
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=((-5.0, 5.0), (-2.0, 3.0), (-9.0, 4.0), (-9.0, 4.0)),
        )
        parameters = result.x if np.all(np.isfinite(result.x)) else initial
        intercept[index] = parameters[0]
        mean_coefficient[index] = parameters[1]
        variance_intercept[index] = np.exp(parameters[2])
        variance_coefficient[index] = np.exp(parameters[3])
        success.append(bool(result.success))

    location, std = _emos_location_scale_from_standardized(
        ensemble_mean,
        ensemble_variance,
        intercept,
        mean_coefficient,
        variance_intercept,
        variance_coefficient,
    )
    residuals = (standardized_truth - location) / std
    residual_covariance = OAS(store_precision=False).fit(residuals).covariance_
    return EmosModel(
        center=center,
        scale=scale,
        intercept=intercept,
        ensemble_mean_coefficient=mean_coefficient,
        variance_intercept=variance_intercept,
        ensemble_variance_coefficient=variance_coefficient,
        residual_correlation=_positive_definite_correlation(residual_covariance),
        optimization_success=tuple(success),
    )


def _emos_location_scale_from_standardized(
    ensemble_mean: np.ndarray,
    ensemble_variance: np.ndarray,
    intercept: np.ndarray,
    mean_coefficient: np.ndarray,
    variance_intercept: np.ndarray,
    variance_coefficient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    location = intercept + mean_coefficient * ensemble_mean
    std = np.sqrt(
        variance_intercept + variance_coefficient * ensemble_variance + 1e-8
    )
    return location, std


def density_ratio_log_weights(particles: np.ndarray, model: EmosModel) -> np.ndarray:
    particles = np.asarray(particles, dtype=np.float64)
    standardized = (particles - model.center[None, :]) / model.scale[None, :]
    ensemble_mean = np.mean(standardized, axis=0)
    ensemble_variance = np.var(standardized, axis=0)
    location, std = _emos_location_scale_from_standardized(
        ensemble_mean,
        ensemble_variance,
        model.intercept,
        model.ensemble_mean_coefficient,
        model.variance_intercept,
        model.ensemble_variance_coefficient,
    )
    raw_covariance = OAS(store_precision=False).fit(standardized).covariance_
    raw_covariance = raw_covariance + np.eye(raw_covariance.shape[0]) * 1e-6
    calibrated_covariance = (
        std[:, None] * model.residual_correlation * std[None, :]
        + np.eye(len(std)) * 1e-6
    )
    calibrated = multivariate_normal.logpdf(
        standardized,
        mean=location,
        cov=calibrated_covariance,
        allow_singular=False,
    )
    raw = multivariate_normal.logpdf(
        standardized,
        mean=ensemble_mean,
        cov=raw_covariance,
        allow_singular=False,
    )
    return np.asarray(calibrated - raw, dtype=np.float64)


def _artifact_input_columns(artifact: Any) -> list[str]:
    metadata = (
        artifact.legacy_artifact.metadata
        if artifact.backend == "legacy" and artifact.legacy_artifact is not None
        else artifact.metadata
    )
    return list(metadata.get("input_columns", ()))


def _path_signature(
    trace: np.ndarray,
    profile: ClosedLoopProfile,
    *,
    input_columns: Sequence[str],
    dt_hours: float,
) -> dict[str, float]:
    trace = np.asarray(trace, dtype=np.float64)
    qroom = np.maximum(trace[:, 1], 0.0)
    pel = np.maximum(trace[:, 2], 0.0)
    setpoint_index = input_columns.index(SETPOINT_TIMESERIES_COLUMN)
    setpoint = np.asarray(profile.inputs[:, setpoint_index], dtype=np.float64)
    if SPACE_HEATING_AVAILABILITY_COLUMN in input_columns:
        availability_index = input_columns.index(SPACE_HEATING_AVAILABILITY_COLUMN)
        active = np.asarray(profile.inputs[:, availability_index] > 0.5)
    else:
        active = np.ones(len(trace), dtype=bool)
    active_pel = pel[active]
    return {
        "room_energy_kwh_m2": float(np.sum(qroom) * dt_hours / 1000.0),
        "electric_energy_kwh_m2": float(np.sum(pel) * dt_hours / 1000.0),
        "pel_peak_p99_w_m2": float(np.quantile(active_pel, 0.99)) if len(active_pel) else 0.0,
        "comfort_degree_hours": float(
            np.sum(np.maximum(setpoint - trace[:, 0], 0.0) * active) * dt_hours
        ),
    }


def _extract_path_signatures(
    *,
    base_root: Path,
    fold: str,
    output_dir: Path,
    dt_hours: float,
    resume: bool,
) -> tuple[Path, Path]:
    particle_path = output_dir / f"annual_path_signatures_{fold}.csv"
    truth_path = output_dir / f"annual_truth_signatures_{fold}.csv"
    if resume and particle_path.exists() and truth_path.exists():
        return particle_path, truth_path
    score_dir = base_root / "scoring" / fold / "probabilistic_structured"
    request = json.loads((score_dir / "score_request.json").read_text())
    artifact = load_artifact(Path(request["artifact_dir"]))
    profiles = load_profiles(
        artifact,
        profile_ids=tuple(int(value) for value in request["profile_ids"]),
    )
    input_columns = _artifact_input_columns(artifact)
    key = jax.random.PRNGKey(int(request["seed"]))
    particle_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(profiles):
        if not isinstance(profile, ClosedLoopProfile):
            raise TypeError("Trajectory recalibration requires closed-loop profiles")
        key, profile_key = jax.random.split(key)
        result = predict_profile(
            artifact,
            profile,
            key=profile_key,
            num_scenarios=int(request["num_scenarios"]),
            hp_scenario_mode=str(request["hp_scenario_mode"]),
            ventilation_rollout_mode=str(request["ventilation_rollout_mode"]),
        )
        if result.scenarios is None:
            raise ValueError("Probabilistic scoring did not return annual scenarios")
        truth_rows.append(
            {
                "profile_id": int(profile.profile_id),
                **_path_signature(
                    np.asarray(result.target),
                    profile,
                    input_columns=input_columns,
                    dt_hours=dt_hours,
                ),
            }
        )
        for scenario_index, trace in enumerate(np.asarray(result.scenarios)):
            particle_rows.append(
                {
                    "profile_id": int(profile.profile_id),
                    "scenario_index": int(scenario_index),
                    **_path_signature(
                        trace,
                        profile,
                        input_columns=input_columns,
                        dt_hours=dt_hours,
                    ),
                }
            )
        print(
            f"path_signatures fold={fold} profile={profile.profile_id} "
            f"index={profile_index + 1}/{len(profiles)} particles={len(result.scenarios)}"
        )
    pd.DataFrame(particle_rows).to_csv(particle_path, index=False)
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    return particle_path, truth_path


def _raw_distribution(base_root: Path, fold: str) -> pd.DataFrame:
    score_dir = base_root / "scoring" / fold / "probabilistic_structured"
    return pd.concat(
        [
            _emulator_distribution(score_dir, method="emulator", kpi_kind=kind)
            for kind in KPI_KINDS
        ],
        ignore_index=True,
    )


def _flex_index(distribution: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = distribution.copy()
    frame["beta_like"] = frame["prediction"] / frame["horizon_hours"]
    particles = (
        frame.groupby(["profile_id", "sample_index"], as_index=False)["beta_like"]
        .mean()
        .rename(columns={"sample_index": "scenario_index", "beta_like": "flex_index"})
    )
    truth_frame = frame.drop_duplicates(
        ["profile_id", "kpi_kind", "horizon_hours", "direction"]
    ).copy()
    truth_frame["truth_beta_like"] = truth_frame["truth"] / truth_frame["horizon_hours"]
    truth = (
        truth_frame.groupby("profile_id", as_index=False)["truth_beta_like"]
        .mean()
        .rename(columns={"truth_beta_like": "flex_index"})
    )
    return particles, truth


def _core_tables(
    annual_particles: pd.DataFrame,
    annual_truth: pd.DataFrame,
    distribution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flex_particles, flex_truth = _flex_index(distribution)
    particles = annual_particles.merge(
        flex_particles,
        on=["profile_id", "scenario_index"],
        validate="one_to_one",
    )
    truth = annual_truth.merge(flex_truth, on="profile_id", validate="one_to_one")
    for frame in (particles, truth):
        frame["log_room_energy"] = np.log1p(frame["room_energy_kwh_m2"])
        frame["log_electric_energy"] = np.log1p(frame["electric_energy_kwh_m2"])
        frame["log_pel_peak_p99"] = np.log1p(frame["pel_peak_p99_w_m2"])
        frame["flex_index_asinh"] = np.arcsinh(frame["flex_index"])
        frame["log_comfort_degree_hours"] = np.log1p(frame["comfort_degree_hours"])
    return particles, truth


def _core_arrays(
    particles: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    columns: Sequence[str] = CORE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile_ids = truth["profile_id"].to_numpy(int)
    particle_arrays: list[np.ndarray] = []
    truth_arrays: list[np.ndarray] = []
    expected_count: int | None = None
    for row in truth.itertuples(index=False):
        group = particles[particles["profile_id"].eq(int(row.profile_id))].sort_values(
            "scenario_index"
        )
        if expected_count is None:
            expected_count = len(group)
        if len(group) != expected_count:
            raise ValueError("Every building must have the same number of particles")
        particle_arrays.append(group.loc[:, columns].to_numpy(float))
        truth_arrays.append(np.asarray([getattr(row, column) for column in columns]))
    return profile_ids, np.stack(particle_arrays), np.stack(truth_arrays)


def _response_tables(
    core_particles: pd.DataFrame,
    core_truth: pd.DataFrame,
    distribution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct = distribution[distribution["kpi_kind"].eq("f")].copy()
    particle_pivot = direct.pivot(
        index=["profile_id", "sample_index"],
        columns=["horizon_hours", "direction"],
        values="prediction",
    )
    truth_pivot = direct.drop_duplicates(
        ["profile_id", "horizon_hours", "direction"]
    ).pivot(
        index="profile_id",
        columns=["horizon_hours", "direction"],
        values="truth",
    )

    def coordinates(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "direct_short_up": np.arcsinh(frame[(0.5, "up")]),
                "direct_asym": np.arcsinh(
                    frame[(0.5, "up")] - frame[(0.5, "down")]
                ),
                "direct_slope": np.arcsinh(
                    frame[(3.0, "up")] - frame[(0.5, "up")]
                ),
            },
            index=frame.index,
        ).reset_index()

    particle_coordinates = coordinates(particle_pivot).rename(
        columns={"sample_index": "scenario_index"}
    )
    truth_coordinates = coordinates(truth_pivot)
    particles = core_particles.merge(
        particle_coordinates,
        on=["profile_id", "scenario_index"],
        validate="one_to_one",
    )
    truth = core_truth.merge(
        truth_coordinates,
        on="profile_id",
        validate="one_to_one",
    )
    return particles, truth


def _select_alpha(
    particles: np.ndarray,
    truth: np.ndarray,
    *,
    alpha_grid: Sequence[float],
    minimum_ess_fraction: float,
) -> tuple[float, pd.DataFrame]:
    score_rows: list[dict[str, float]] = []
    fold_scores = {float(alpha): [] for alpha in alpha_grid}
    for held_out in range(len(particles)):
        keep = np.arange(len(particles)) != held_out
        model = fit_emos(particles[keep], truth[keep])
        log_ratio = density_ratio_log_weights(particles[held_out], model)
        standardized_particles = (
            particles[held_out] - model.center[None, :]
        ) / model.scale[None, :]
        standardized_truth = (truth[held_out] - model.center) / model.scale
        minimum_ess = minimum_ess_fraction * particles.shape[1]
        for alpha in alpha_grid:
            weights, effective_alpha = stabilized_density_ratio_weights(
                log_ratio,
                alpha=float(alpha),
                minimum_ess=minimum_ess,
            )
            score = weighted_energy_score(
                standardized_particles,
                weights,
                standardized_truth,
            )
            fold_scores[float(alpha)].append(score)
            score_rows.append(
                {
                    "held_out_index": float(held_out),
                    "alpha": float(alpha),
                    "effective_alpha": float(effective_alpha),
                    "energy_score": float(score),
                    "ess": float(1.0 / np.sum(weights**2)),
                }
            )
    means = {alpha: float(np.mean(scores)) for alpha, scores in fold_scores.items()}
    selected = min(means, key=lambda value: (means[value], value))
    summary = pd.DataFrame(score_rows)
    summary["selected"] = summary["alpha"].eq(selected)
    return float(selected), summary


def _weight_rows(
    profile_ids: np.ndarray,
    particles: np.ndarray,
    model: EmosModel,
    *,
    alpha: float,
    minimum_ess_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile_id, values in zip(profile_ids, particles, strict=True):
        log_ratio = density_ratio_log_weights(values, model)
        weights, effective_alpha = stabilized_density_ratio_weights(
            log_ratio,
            alpha=alpha,
            minimum_ess=minimum_ess_fraction * len(values),
        )
        ess = float(1.0 / np.sum(weights**2))
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-300))))
        for scenario_index, (weight, ratio) in enumerate(zip(weights, log_ratio, strict=True)):
            rows.append(
                {
                    "profile_id": int(profile_id),
                    "scenario_index": int(scenario_index),
                    "weight": float(weight),
                    "log_density_ratio": float(ratio),
                    "selected_alpha": float(alpha),
                    "effective_alpha": float(effective_alpha),
                    "ess": ess,
                    "normalized_entropy": entropy / np.log(len(weights)),
                    "max_weight": float(np.max(weights)),
                }
            )
    return pd.DataFrame(rows)


def _wide_distribution(
    distribution: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    frame = distribution.copy()
    frame["cell"] = frame.apply(
        lambda row: (
            f"{row['kpi_kind']}|{float(row['horizon_hours']):g}|{row['direction']}"
        ),
        axis=1,
    )
    cells = sorted(frame["cell"].unique())
    cell_metadata = frame[["cell", "kpi_kind", "horizon_hours", "direction"]].drop_duplicates()
    values = frame.pivot(
        index=["profile_id", "sample_index"],
        columns="cell",
        values="prediction",
    ).reindex(columns=cells)
    truth = frame.drop_duplicates(["profile_id", "cell"]).pivot(
        index="profile_id",
        columns="cell",
        values="truth",
    ).reindex(columns=cells)
    profile_ids = truth.index.to_numpy(int)
    arrays: list[np.ndarray] = []
    expected_count: int | None = None
    for profile_id in profile_ids:
        group = values.loc[profile_id]
        if expected_count is None:
            expected_count = len(group)
        if len(group) != expected_count:
            raise ValueError("Every KPI vector must contain the same number of particles")
        arrays.append(group.to_numpy(float))
    return profile_ids, np.stack(arrays), truth.to_numpy(float), cells, cell_metadata


def _weights_by_profile(
    profile_ids: np.ndarray,
    particle_count: int,
    frame: pd.DataFrame | None,
) -> np.ndarray:
    if frame is None:
        return np.full((len(profile_ids), particle_count), 1.0 / particle_count)
    rows: list[np.ndarray] = []
    for profile_id in profile_ids:
        group = frame[frame["profile_id"].eq(int(profile_id))].sort_values("scenario_index")
        if len(group) != particle_count:
            raise ValueError(f"Missing trajectory weights for profile {profile_id}")
        values = group["weight"].to_numpy(float)
        rows.append(values / np.sum(values))
    return np.stack(rows)


def _calibration_cell_scales(calibration: pd.DataFrame, cells: Sequence[str]) -> np.ndarray:
    frame = calibration.copy()
    frame["cell"] = frame.apply(
        lambda row: (
            f"{row['kpi_kind']}|{float(row['horizon_hours']):g}|{row['direction']}"
        ),
        axis=1,
    )
    truth = frame.drop_duplicates(["profile_id", "cell"]).pivot(
        index="profile_id", columns="cell", values="truth"
    ).reindex(columns=cells)
    return np.maximum(np.std(truth.to_numpy(float), axis=0), 1e-3)


def _marginal_metrics(
    *,
    method: str,
    profile_ids: np.ndarray,
    values: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    cells: Sequence[str],
    cell_metadata: pd.DataFrame,
    calibration_scales: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metadata = cell_metadata.set_index("cell")
    for cell_index, cell in enumerate(cells):
        medians = np.asarray(
            [
                weighted_quantile(values[index, :, cell_index], weights[index], 0.5)
                for index in range(len(profile_ids))
            ]
        )
        errors = medians - truth[:, cell_index]
        crps = np.asarray(
            [
                weighted_crps(
                    values[index, :, cell_index],
                    weights[index],
                    truth[index, cell_index],
                )
                for index in range(len(profile_ids))
            ]
        )
        reliability_errors = []
        for quantile in QUANTILES:
            predicted = np.asarray(
                [
                    weighted_quantile(
                        values[index, :, cell_index], weights[index], quantile
                    )
                    for index in range(len(profile_ids))
                ]
            )
            reliability_errors.append(
                float(np.mean(truth[:, cell_index] <= predicted) - quantile)
            )
        q05 = np.asarray(
            [weighted_quantile(values[i, :, cell_index], weights[i], 0.05) for i in range(len(profile_ids))]
        )
        q95 = np.asarray(
            [weighted_quantile(values[i, :, cell_index], weights[i], 0.95) for i in range(len(profile_ids))]
        )
        labels = metadata.loc[cell]
        rows.append(
            {
                "method": method,
                "cell": cell,
                "kpi_kind": labels["kpi_kind"],
                "horizon_hours": float(labels["horizon_hours"]),
                "direction": labels["direction"],
                "count": int(len(profile_ids)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "nrmse": float(np.sqrt(np.mean(errors**2)) / calibration_scales[cell_index]),
                "mae": float(np.mean(np.abs(errors))),
                "bias": float(np.mean(errors)),
                "crps": float(np.mean(crps)),
                "calibration_mae": float(np.mean(np.abs(reliability_errors))),
                "coverage_90": float(
                    np.mean(
                        (truth[:, cell_index] >= q05)
                        & (truth[:, cell_index] <= q95)
                    )
                ),
                "width_90": float(np.mean(q95 - q05)),
            }
        )
    return pd.DataFrame(rows)


def _joint_metrics(
    *,
    method: str,
    values: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
    raw_values: np.ndarray,
) -> dict[str, Any]:
    energy_scores: list[float] = []
    variogram_scores: list[float] = []
    copula_errors: list[float] = []
    for index in range(len(values)):
        standardized = values[index] / scales[None, :]
        standardized_truth = truth[index] / scales
        energy_scores.append(
            weighted_energy_score(standardized, weights[index], standardized_truth)
        )
        variogram_scores.append(
            weighted_variogram_score(standardized, weights[index], standardized_truth)
        )
        reference = _rank_correlation(
            raw_values[index],
            np.full(raw_values.shape[1], 1.0 / raw_values.shape[1]),
        )
        candidate = _rank_correlation(values[index], weights[index])
        row, column = np.triu_indices(values.shape[2], k=1)
        copula_errors.append(
            float(np.sqrt(np.mean((candidate[row, column] - reference[row, column]) ** 2)))
        )
    ess = 1.0 / np.sum(weights**2, axis=1)
    return {
        "method": method,
        "energy_score_16d": float(np.mean(energy_scores)),
        "variogram_score_16d": float(np.mean(variogram_scores)),
        "raw_rank_copula_rmse": float(np.mean(copula_errors)),
        "mean_ess": float(np.mean(ess)),
        "min_ess": float(np.min(ess)),
        "mean_max_weight": float(np.mean(np.max(weights, axis=1))),
    }


def _building_scores(
    *,
    method: str,
    profile_ids: np.ndarray,
    values: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
    cells: Sequence[str],
    raw_values: np.ndarray,
) -> pd.DataFrame:
    kind_indices = {
        kind: np.asarray([index for index, cell in enumerate(cells) if cell.startswith(f"{kind}|")])
        for kind in KPI_KINDS
    }
    rows: list[dict[str, Any]] = []
    for profile_index, profile_id in enumerate(profile_ids):
        row: dict[str, Any] = {
            "method": method,
            "profile_id": int(profile_id),
        }
        for kind, indices in kind_indices.items():
            row[f"crps_{kind}"] = float(
                np.mean(
                    [
                        weighted_crps(
                            values[profile_index, :, cell_index],
                            weights[profile_index],
                            truth[profile_index, cell_index],
                        )
                        for cell_index in indices
                    ]
                )
            )
        standardized = values[profile_index] / scales[None, :]
        standardized_truth = truth[profile_index] / scales
        row["energy_score_16d"] = weighted_energy_score(
            standardized,
            weights[profile_index],
            standardized_truth,
        )
        row["variogram_score_16d"] = weighted_variogram_score(
            standardized,
            weights[profile_index],
            standardized_truth,
        )
        reference = _rank_correlation(
            raw_values[profile_index],
            np.full(raw_values.shape[1], 1.0 / raw_values.shape[1]),
        )
        candidate = _rank_correlation(values[profile_index], weights[profile_index])
        upper = np.triu_indices(values.shape[2], k=1)
        row["raw_rank_copula_rmse"] = float(
            np.sqrt(np.mean((candidate[upper] - reference[upper]) ** 2))
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_method_differences(
    building_metrics: pd.DataFrame,
    *,
    samples: int = 5000,
    seed: int = 13,
) -> pd.DataFrame:
    metrics = (
        "crps_f",
        "crps_regression_based",
        "energy_score_16d",
        "variogram_score_16d",
    )
    raw = building_metrics[building_metrics["method"].eq("raw")].set_index("profile_id")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method == "raw":
            continue
        candidate = building_metrics[building_metrics["method"].eq(method)].set_index(
            "profile_id"
        )
        candidate = candidate.reindex(raw.index)
        for metric in metrics:
            difference = candidate[metric].to_numpy(float) - raw[metric].to_numpy(float)
            indices = rng.integers(0, len(difference), size=(samples, len(difference)))
            bootstrap = np.mean(difference[indices], axis=1)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean_difference_vs_raw": float(np.mean(difference)),
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                    "probability_improvement": float(np.mean(bootstrap < 0.0)),
                    "building_count": int(len(difference)),
                }
            )
    return pd.DataFrame(rows)


def _signature_energy_summary(
    *,
    calibration_truth: np.ndarray,
    test_particles: np.ndarray,
    test_truth: np.ndarray,
    test_weights: pd.DataFrame,
    profile_ids: np.ndarray,
) -> tuple[float, float]:
    scale = np.maximum(np.std(calibration_truth, axis=0), 1e-3)
    uniform = np.full(test_particles.shape[1], 1.0 / test_particles.shape[1])
    weights = _weights_by_profile(profile_ids, test_particles.shape[1], test_weights)
    raw_scores = []
    weighted_scores = []
    for index in range(len(test_particles)):
        values = test_particles[index] / scale[None, :]
        truth = test_truth[index] / scale
        raw_scores.append(weighted_energy_score(values, uniform, truth))
        weighted_scores.append(weighted_energy_score(values, weights[index], truth))
    return float(np.mean(raw_scores)), float(np.mean(weighted_scores))


def _signature_marginal_rows(
    *,
    method: str,
    columns: Sequence[str],
    profile_ids: np.ndarray,
    test_particles: np.ndarray,
    test_truth: np.ndarray,
    test_weights: pd.DataFrame,
) -> pd.DataFrame:
    uniform = np.full(test_particles.shape[1], 1.0 / test_particles.shape[1])
    weights = _weights_by_profile(profile_ids, test_particles.shape[1], test_weights)
    rows: list[dict[str, Any]] = []
    for coordinate, column in enumerate(columns):
        raw = np.mean(
            [
                weighted_crps(
                    test_particles[index, :, coordinate],
                    uniform,
                    test_truth[index, coordinate],
                )
                for index in range(len(profile_ids))
            ]
        )
        weighted = np.mean(
            [
                weighted_crps(
                    test_particles[index, :, coordinate],
                    weights[index],
                    test_truth[index, coordinate],
                )
                for index in range(len(profile_ids))
            ]
        )
        rows.append(
            {
                "method": method,
                "coordinate": column,
                "raw_crps": float(raw),
                "weighted_crps": float(weighted),
                "delta_crps": float(weighted - raw),
            }
        )
    return pd.DataFrame(rows)


def _evaluate(
    *,
    calibration_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    test_rns: pd.DataFrame,
    test_ecc: pd.DataFrame,
    trajectory_weights: pd.DataFrame,
    response_trajectory_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = {
        "raw": (test_raw, None),
        "rns": (test_rns, None),
        "rns_ecc": (test_ecc, None),
        "trajectory_reweight": (test_raw, trajectory_weights),
        "trajectory_reweight_response7": (test_raw, response_trajectory_weights),
    }
    raw_profile_ids, raw_values, raw_truth, cells, cell_metadata = _wide_distribution(test_raw)
    calibration_scales = _calibration_cell_scales(calibration_raw, cells)
    marginal_frames: list[pd.DataFrame] = []
    joint_rows: list[dict[str, Any]] = []
    building_frames: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        distribution, weight_frame = methods[method]
        profile_ids, values, truth, method_cells, _ = _wide_distribution(distribution)
        if not np.array_equal(profile_ids, raw_profile_ids) or method_cells != cells:
            raise ValueError(f"{method} does not align with the raw KPI vectors")
        weights = _weights_by_profile(profile_ids, values.shape[1], weight_frame)
        marginal_frames.append(
            _marginal_metrics(
                method=method,
                profile_ids=profile_ids,
                values=values,
                truth=truth,
                weights=weights,
                cells=cells,
                cell_metadata=cell_metadata,
                calibration_scales=calibration_scales,
            )
        )
        joint_rows.append(
            _joint_metrics(
                method=method,
                values=values,
                truth=truth,
                weights=weights,
                scales=calibration_scales,
                raw_values=raw_values,
            )
        )
        building_frames.append(
            _building_scores(
                method=method,
                profile_ids=profile_ids,
                values=values,
                truth=truth,
                weights=weights,
                scales=calibration_scales,
                cells=cells,
                raw_values=raw_values,
            )
        )
    marginal = pd.concat(marginal_frames, ignore_index=True)
    summary = (
        marginal.groupby(["method", "kpi_kind"], as_index=False)[
            ["nrmse", "crps", "calibration_mae", "coverage_90", "width_90"]
        ]
        .mean()
    )
    return marginal, summary, pd.DataFrame(joint_rows), pd.concat(
        building_frames, ignore_index=True
    )


def _report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    joint: pd.DataFrame,
    schemes: pd.DataFrame,
    signature_marginals: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> Path:
    comparison = summary.pivot(index="method", columns="kpi_kind")
    comparison.columns = [f"{metric}_{kind}" for metric, kind in comparison.columns]
    comparison = comparison.reset_index().merge(joint, on="method", validate="one_to_one")
    comparison["order"] = comparison["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )
    comparison = comparison.sort_values("order").drop(columns="order")
    report_path = output_dir / "REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# KPI trajectory recalibration benchmark",
                "",
                "The emulator checkpoint and every annual path are frozen. RNS and RNS-ECC alter KPI values only; trajectory reweighting changes only the probability attached to each complete path.",
                "",
                "## Test comparison",
                "",
                _markdown_table(
                    comparison,
                    (
                        "method",
                        "crps_f",
                        "calibration_mae_f",
                        "crps_regression_based",
                        "calibration_mae_regression_based",
                        "energy_score_16d",
                        "variogram_score_16d",
                        "raw_rank_copula_rmse",
                        "mean_ess",
                    ),
                ),
                "",
                "RNS and RNS-ECC have identical calibrated marginals by construction. RNS-ECC differs only by assigning each calibrated quantile to the raw particle with the corresponding marginal rank.",
                "",
                "## Paired bootstrap",
                "",
                "Differences are method minus raw over test buildings; negative values are improvements. Intervals are paired 95% building-bootstrap intervals.",
                "",
                _markdown_table(
                    bootstrap,
                    (
                        "method",
                        "metric",
                        "mean_difference_vs_raw",
                        "ci_low",
                        "ci_high",
                        "probability_improvement",
                    ),
                ),
                "",
                "## Trajectory weights",
                "",
                "Density-ratio tempering is selected independently for each signature using leave-one-building-out signature energy score on the calibration fold. Both methods enforce ESS >= 20/100.",
                "",
                _markdown_table(
                    schemes,
                    (
                        "method",
                        "signature_dimension",
                        "selected_alpha",
                        "loo_energy_raw",
                        "loo_energy_selected",
                        "test_signature_energy_raw",
                        "test_signature_energy_weighted",
                        "test_mean_ess",
                        "test_mean_max_weight",
                    ),
                ),
                "",
                "The 5-D signature is `[log Eroom, log Eel, log P99, flexibility index, log comfort]`. The 7-D rescue replaces the scalar flexibility index with direct-F short-up, 0.5 h asymmetry, and 3 h-minus-0.5 h slope.",
                "",
                "## Signature marginals",
                "",
                _markdown_table(
                    signature_marginals,
                    ("method", "coordinate", "raw_crps", "weighted_crps", "delta_crps"),
                ),
                "",
                "## Coherence",
                "",
                "- `raw`: complete physical trajectories, equal weights.",
                "- `rns`: calibrated scalar marginals; shared quantile indices impose a comonotone KPI dependence pattern.",
                "- `rns_ecc`: calibrated scalar marginals with the raw KPI rank copula; no corresponding physical trajectory is guaranteed.",
                "- `trajectory_reweight`: unchanged complete trajectories weighted from the proposed 5-D annual signature.",
                "- `trajectory_reweight_response7`: unchanged complete trajectories weighted from the flexibility-aware 7-D signature.",
                "",
                "## Decision",
                "",
                "RNS-ECC is the best KPI postprocessor in this experiment. It retains the RNS marginal gains, exactly preserves the raw rank copula in the scored KPI vector, and materially improves both 16-D energy and variogram scores.",
                "",
                "Do not use either trajectory-weighting fit as the default flexibility calibrator. The proposed 5-D weights improve their aggregate signature mainly through comfort and peak power while worsening energy and flexibility marginals. The flexibility-aware 7-D weights improve short response, asymmetry, and the variogram score, but do not improve aggregate F or regression CRPS and place every test building at the ESS floor.",
                "",
                "The result rejects the strong form of the wrong-probabilities-only hypothesis for flexibility: the raw path support does not contain enough trajectories that are simultaneously well located across all KPI cells. Reweighting can only redistribute existing support, so it trades one bias against another. RNS-ECC succeeds because it is allowed to move marginal KPI values, but it does not produce calibrated physical trajectories.",
                "",
            ]
        )
        + "\n"
    )
    return report_path


def _fit_weighting_scheme(
    *,
    method: str,
    stem: str,
    columns: Sequence[str],
    calibration_ids: np.ndarray,
    calibration_particles: np.ndarray,
    calibration_truth: np.ndarray,
    test_ids: np.ndarray,
    test_particles: np.ndarray,
    test_truth: np.ndarray,
    output_dir: Path,
    alpha_grid: Sequence[float],
    minimum_ess_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    selected_alpha, cv = _select_alpha(
        calibration_particles,
        calibration_truth,
        alpha_grid=alpha_grid,
        minimum_ess_fraction=minimum_ess_fraction,
    )
    cv.insert(0, "method", method)
    model = fit_emos(calibration_particles, calibration_truth)
    calibration_weights = _weight_rows(
        calibration_ids,
        calibration_particles,
        model,
        alpha=selected_alpha,
        minimum_ess_fraction=minimum_ess_fraction,
    )
    test_weights = _weight_rows(
        test_ids,
        test_particles,
        model,
        alpha=selected_alpha,
        minimum_ess_fraction=minimum_ess_fraction,
    )
    cv.to_csv(output_dir / f"density_ratio_alpha_cv_{stem}.csv", index=False)
    calibration_weights.to_csv(
        output_dir / f"trajectory_weights_calibration_{stem}.csv", index=False
    )
    test_weights.to_csv(
        output_dir / f"trajectory_weights_test_{stem}.csv", index=False
    )
    (output_dir / f"emos_model_{stem}.json").write_text(
        json.dumps(
            json_safe(
                {
                    **model.parameters(columns),
                    "selected_alpha": selected_alpha,
                    "minimum_ess_fraction": minimum_ess_fraction,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    raw_test_energy, weighted_test_energy = _signature_energy_summary(
        calibration_truth=calibration_truth,
        test_particles=test_particles,
        test_truth=test_truth,
        test_weights=test_weights,
        profile_ids=test_ids,
    )
    selected_cv = cv[cv["selected"]]
    raw_cv = cv[cv["alpha"].eq(0.0)]
    buildings = test_weights.drop_duplicates("profile_id")
    diagnostics = {
        "method": method,
        "signature_dimension": int(len(columns)),
        "selected_alpha": float(selected_alpha),
        "loo_energy_raw": float(raw_cv["energy_score"].mean()),
        "loo_energy_selected": float(selected_cv["energy_score"].mean()),
        "test_signature_energy_raw": raw_test_energy,
        "test_signature_energy_weighted": weighted_test_energy,
        "test_mean_ess": float(buildings["ess"].mean()),
        "test_min_ess": float(buildings["ess"].min()),
        "test_mean_max_weight": float(buildings["max_weight"].mean()),
        "test_mean_effective_alpha": float(buildings["effective_alpha"].mean()),
    }
    marginal_rows = _signature_marginal_rows(
        method=method,
        columns=columns,
        profile_ids=test_ids,
        test_particles=test_particles,
        test_truth=test_truth,
        test_weights=test_weights,
    )
    return test_weights, cv, diagnostics, marginal_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--minimum-ess-fraction", type=float, default=0.2)
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.dt_hours <= 0.0:
        parser.error("dt-hours must be positive")
    if not 0.0 < args.minimum_ess_fraction <= 1.0:
        parser.error("minimum-ess-fraction must be in (0, 1]")
    if not args.alpha_grid or any(value < 0.0 for value in args.alpha_grid):
        parser.error("alpha-grid must contain nonnegative values")
    return args


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_distributions = {fold: _raw_distribution(args.base_root, fold) for fold in FOLDS}
    core_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    response_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in FOLDS:
        particle_path, truth_path = _extract_path_signatures(
            base_root=args.base_root,
            fold=fold,
            output_dir=args.output_dir,
            dt_hours=float(args.dt_hours),
            resume=bool(args.resume),
        )
        core_tables[fold] = _core_tables(
            pd.read_csv(particle_path),
            pd.read_csv(truth_path),
            raw_distributions[fold],
        )
        core_tables[fold][0].to_csv(
            args.output_dir / f"core_kpi_particles_{fold}.csv", index=False
        )
        core_tables[fold][1].to_csv(
            args.output_dir / f"core_kpi_truth_{fold}.csv", index=False
        )
        response_tables[fold] = _response_tables(
            *core_tables[fold],
            raw_distributions[fold],
        )
        response_tables[fold][0].to_csv(
            args.output_dir / f"response7_kpi_particles_{fold}.csv", index=False
        )
        response_tables[fold][1].to_csv(
            args.output_dir / f"response7_kpi_truth_{fold}.csv", index=False
        )

    calibration_ids, calibration_particles, calibration_truth = _core_arrays(
        *core_tables["calibration"]
    )
    test_ids, test_particles, test_truth = _core_arrays(*core_tables["test"])
    response_calibration_ids, response_calibration_particles, response_calibration_truth = (
        _core_arrays(*response_tables["calibration"], columns=RESPONSE_COLUMNS)
    )
    response_test_ids, response_test_particles, response_test_truth = _core_arrays(
        *response_tables["test"], columns=RESPONSE_COLUMNS
    )
    if not np.array_equal(calibration_ids, response_calibration_ids) or not np.array_equal(
        test_ids, response_test_ids
    ):
        raise ValueError("Annual and response signatures use different building orders")

    test_weights, annual_cv, annual_diagnostics, annual_marginals = _fit_weighting_scheme(
        method="trajectory_reweight",
        stem="annual5",
        columns=CORE_COLUMNS,
        calibration_ids=calibration_ids,
        calibration_particles=calibration_particles,
        calibration_truth=calibration_truth,
        test_ids=test_ids,
        test_particles=test_particles,
        test_truth=test_truth,
        output_dir=args.output_dir,
        alpha_grid=tuple(float(value) for value in args.alpha_grid),
        minimum_ess_fraction=float(args.minimum_ess_fraction),
    )
    (
        response_test_weights,
        response_cv,
        response_diagnostics,
        response_marginals,
    ) = _fit_weighting_scheme(
        method="trajectory_reweight_response7",
        stem="response7",
        columns=RESPONSE_COLUMNS,
        calibration_ids=response_calibration_ids,
        calibration_particles=response_calibration_particles,
        calibration_truth=response_calibration_truth,
        test_ids=response_test_ids,
        test_particles=response_test_particles,
        test_truth=response_test_truth,
        output_dir=args.output_dir,
        alpha_grid=tuple(float(value) for value in args.alpha_grid),
        minimum_ess_fraction=float(args.minimum_ess_fraction),
    )
    cv = pd.concat((annual_cv, response_cv), ignore_index=True)
    schemes = pd.DataFrame((annual_diagnostics, response_diagnostics))
    signature_marginals = pd.concat(
        (annual_marginals, response_marginals), ignore_index=True
    )
    cv.to_csv(args.output_dir / "density_ratio_alpha_cv.csv", index=False)
    schemes.to_csv(args.output_dir / "trajectory_weighting_diagnostics.csv", index=False)
    signature_marginals.to_csv(
        args.output_dir / "signature_marginal_metrics.csv", index=False
    )
    test_weights.to_csv(args.output_dir / "trajectory_weights_test.csv", index=False)
    pd.read_csv(
        args.output_dir / "trajectory_weights_calibration_annual5.csv"
    ).to_csv(args.output_dir / "trajectory_weights_calibration.csv", index=False)
    (args.output_dir / "emos_model.json").write_text(
        (args.output_dir / "emos_model_annual5.json").read_text()
    )

    rns_test, _, _, _ = calibrate_distribution_table(
        raw_distributions["calibration"],
        raw_distributions["test"],
    )
    rns_ecc_test = ecc_reorder(raw_distributions["test"], rns_test)
    rns_test.to_csv(args.output_dir / "rns_test_distribution.csv", index=False)
    rns_ecc_test.to_csv(args.output_dir / "rns_ecc_test_distribution.csv", index=False)
    marginal, summary, joint, building_metrics = _evaluate(
        calibration_raw=raw_distributions["calibration"],
        test_raw=raw_distributions["test"],
        test_rns=rns_test,
        test_ecc=rns_ecc_test,
        trajectory_weights=test_weights,
        response_trajectory_weights=response_test_weights,
    )
    marginal.to_csv(args.output_dir / "marginal_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "marginal_summary.csv", index=False)
    joint.to_csv(args.output_dir / "joint_metrics.csv", index=False)
    building_metrics.to_csv(args.output_dir / "building_metrics.csv", index=False)
    bootstrap = _bootstrap_method_differences(building_metrics)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    report = _report(
        output_dir=args.output_dir,
        summary=summary,
        joint=joint,
        schemes=schemes,
        signature_marginals=signature_marginals,
        bootstrap=bootstrap,
    )
    request = {
        "base_root": str(args.base_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "dt_hours": float(args.dt_hours),
        "minimum_ess_fraction": float(args.minimum_ess_fraction),
        "alpha_grid": [float(value) for value in args.alpha_grid],
        "core_columns": list(CORE_COLUMNS),
        "response_columns": list(RESPONSE_COLUMNS),
        "selected_alpha": {
            row["method"]: row["selected_alpha"] for row in schemes.to_dict(orient="records")
        },
        "methods": list(METHOD_ORDER),
        "paths_modified": False,
    }
    (args.output_dir / "request.json").write_text(
        json.dumps(json_safe(request), indent=2, sort_keys=True)
    )
    print(f"trajectory_recalibration_report={report}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
