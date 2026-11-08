"""Nested Monte Carlo decomposition of annual flexibility-KPI uncertainty.

For each fixed building, the diagnostic crosses persistent annual latent draws
with repeated draws of every other stochastic source in the emulator. It then
applies the finite-sample covariance identity

    Cov(K) = Cov_xi(E_fast[K | xi]) + E_xi(Cov_fast(K | xi)).

The variance decomposition itself does not use the EnergyPlus target. Companion
native-calibration and normalization columns compare the raw ensemble with that
fixed target. Buildings are never pooled before the components are estimated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.columns import (
    DISTURBANCE_INPUT_COLUMNS,
    HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
    SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN,
    SH_VOLUME_PER_HEATED_AREA_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator.flexibility_event_study_kpis import (
    EventStudyConfig,
    estimate_trace_flexibility,
)
from neural_building_emulator.models import (
    ProbabilisticContractingClosedLoopHPEmulator,
)
from neural_building_emulator.scaling import inverse_target
from neural_building_emulator.train import (
    _probabilistic_ventilation_rollout_kwargs,
)
from neural_building_emulator.ventilation import build_eplus_ventilation_schedule

from .artifacts import LoadedArtifact, json_safe, load_artifact
from .profiles import load_profiles
from .residual_covariance_diagnostics import (
    analyze_residual_covariance,
    write_residual_covariance_dashboard,
)


@dataclass(frozen=True)
class KpiCoordinate:
    kpi_kind: str
    horizon_hours: float
    direction: str

    @property
    def label(self) -> str:
        horizon = f"{self.horizon_hours:g}h"
        return f"{self.kpi_kind}:{self.direction}:{horizon}"


@dataclass(frozen=True)
class CovarianceDecomposition:
    total: np.ndarray
    persistent_xi: np.ndarray
    fast: np.ndarray

    @property
    def closure(self) -> np.ndarray:
        return self.total - self.persistent_xi - self.fast


def _population_covariance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("values must have shape [draw, coordinate]")
    centered = values - np.mean(values, axis=0, keepdims=True)
    return centered.T @ centered / values.shape[0]


def decompose_nested_samples(samples: np.ndarray) -> CovarianceDecomposition:
    """Decompose a finite balanced ``[xi, fast, KPI]`` sample exactly.

    Population rather than unbiased sample covariances are intentional. They
    make the empirical ANOVA identity exact for the finite crossed design. The
    components converge to the law-of-total-covariance terms as the two draw
    counts increase.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("samples must have shape [xi_draw, fast_draw, coordinate]")
    if values.shape[0] < 2 or values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError("samples require at least 2 xi draws, 2 fast draws, and 1 KPI")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    conditional_means = np.mean(values, axis=1)
    persistent_xi = _population_covariance(conditional_means)
    fast = np.mean(
        np.stack([_population_covariance(row) for row in values], axis=0),
        axis=0,
    )
    total = _population_covariance(values.reshape(-1, values.shape[-1]))
    return CovarianceDecomposition(
        total=total,
        persistent_xi=persistent_xi,
        fast=fast,
    )


@eqx.filter_jit
def _sample_fixed_xi_normalized(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    persistent_xi: jnp.ndarray,
    key: jax.Array,
    *,
    hp_scenario_mode: str,
    ventilation_kwargs: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    predictions, _ = model.sample_with_aux(
        metadata,
        inputs,
        initial_temperature,
        key=key,
        num_particles=persistent_xi.shape[0],
        persistent_xi=persistent_xi,
        sample_process_noise=True,
        hp_scenario_mode=hp_scenario_mode,
        **ventilation_kwargs,
    )
    return predictions


def _fixed_xi_scenarios(
    artifact: LoadedArtifact,
    profile: ClosedLoopProfile,
    base_xi: np.ndarray,
    *,
    fast_rollouts: int,
    key: jax.Array,
    hp_scenario_mode: str,
    ventilation_rollout_mode: str,
) -> np.ndarray:
    model = artifact.model
    if not isinstance(model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError("KPI variance decomposition requires the structured probabilistic model")
    base_xi = np.asarray(base_xi, dtype=np.float32)
    if base_xi.shape != (model.latent_dim,):
        raise ValueError(f"base_xi must have shape ({model.latent_dim},)")
    metadata = jnp.asarray(artifact.scalers.metadata.transform(profile.metadata))
    inputs = jnp.asarray(artifact.scalers.inputs.transform(profile.inputs))
    target_mean = np.asarray(artifact.scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(artifact.scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = jnp.asarray(
        ((profile.initial_temperature[0] - target_mean[:1]) / target_scale[:1]).astype(
            np.float32
        )
    )
    saved_metadata = artifact.legacy_artifact.metadata if artifact.legacy_artifact else artifact.metadata
    ventilation_kwargs = _probabilistic_ventilation_rollout_kwargs(
        profile,
        mode=ventilation_rollout_mode,  # type: ignore[arg-type]
        metadata_columns=tuple(saved_metadata.get("metadata_columns", ())) or None,
    )
    dynamic_ventilation_kwargs = {
        name: jnp.asarray(value) for name, value in ventilation_kwargs.items()
    }
    repeated_xi = jnp.broadcast_to(
        jnp.asarray(base_xi),
        (fast_rollouts, model.latent_dim),
    )
    normalized = _sample_fixed_xi_normalized(
        model,
        metadata,
        inputs,
        initial_temperature,
        repeated_xi,
        key,
        hp_scenario_mode=hp_scenario_mode,
        ventilation_kwargs=dynamic_ventilation_kwargs,
    )
    return inverse_target(np.asarray(normalized), artifact.scalers)


def _kpi_configs(
    horizons_hours: tuple[float, ...],
    *,
    dt_hours: float,
    min_setpoint_change: float,
    min_events: int,
    controls: str,
) -> tuple[tuple[str, EventStudyConfig], ...]:
    horizon_steps = tuple(max(1, int(round(value / dt_hours))) for value in horizons_hours)
    common = {
        "horizons_hours": horizons_hours,
        "horizon_steps": horizon_steps,
        "dt_hours": dt_hours,
        "min_setpoint_change": min_setpoint_change,
        "min_events": min_events,
    }
    return (
        (
            "regression_based",
            EventStudyConfig(
                **common,
                controls=controls,  # type: ignore[arg-type]
                estimator="regression",
            ),
        ),
        (
            "f",
            EventStudyConfig(
                **common,
                controls="none",
                estimator="direct_ratio",
            ),
        ),
    )


def _coordinates(configs: tuple[tuple[str, EventStudyConfig], ...]) -> tuple[KpiCoordinate, ...]:
    return tuple(
        KpiCoordinate(kind, float(horizon), direction)
        for kind, config in configs
        for horizon in config.horizons_hours
        for direction in ("up", "down")
    )


def _trace_kpi_vector(
    profile: ClosedLoopProfile,
    trace: np.ndarray,
    configs: tuple[tuple[str, EventStudyConfig], ...],
) -> np.ndarray:
    values: list[float] = []
    for _, config in configs:
        for kpi in estimate_trace_flexibility(profile, trace, config):
            values.extend(
                (
                    kpi.up_flex_wh_m2_k if kpi.valid else float("nan"),
                    kpi.down_flex_wh_m2_k if kpi.valid else float("nan"),
                )
            )
    return np.asarray(values, dtype=np.float64)


def _kpi_normalization_scale(truth: float) -> float:
    """Use the fixed building's KPI magnitude for dimensionless dispersion."""
    return max(abs(float(truth)), 1e-6) if np.isfinite(truth) else float("nan")


def _native_calibration_rows(
    profile_id: int,
    coordinates: tuple[KpiCoordinate, ...],
    samples: np.ndarray,
    truth: np.ndarray,
) -> list[dict]:
    """Score the raw nested ensemble against the fixed simulator trajectory."""
    truth = np.asarray(truth, dtype=np.float64)
    if truth.shape != (len(coordinates),):
        raise ValueError("truth must contain one value per KPI coordinate")
    rows: list[dict] = []
    for index, coordinate in enumerate(coordinates):
        values = np.asarray(samples[:, :, index], dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        target = float(truth[index])
        common = {
            "profile_id": int(profile_id),
            **asdict(coordinate),
            "kpi_label": coordinate.label,
            "truth": target,
            "sample_count": int(values.size),
        }
        if values.size < 2 or not np.isfinite(target):
            rows.append({**common, "valid": False})
            continue
        scale = _kpi_normalization_scale(target)
        mean = float(np.mean(values))
        median = float(np.median(values))
        less = float(np.mean(values < target))
        equal = float(np.mean(values == target))
        row = {
            **common,
            "valid": True,
            "normalization_scale": scale,
            "ensemble_mean": mean,
            "ensemble_median": median,
            "ensemble_std": float(np.std(values)),
            "mean_bias": mean - target,
            "median_bias": median - target,
            "absolute_median_error": abs(median - target),
            "normalized_absolute_median_error": abs(median - target) / scale,
            "pit": less + 0.5 * equal,
            "absolute_pit_distance_from_median": abs(less + 0.5 * equal - 0.5),
        }
        for level in (50, 80, 90, 95):
            tail = (1.0 - level / 100.0) / 2.0
            lower, upper = np.quantile(values, (tail, 1.0 - tail))
            row[f"lower_{level}"] = float(lower)
            row[f"upper_{level}"] = float(upper)
            row[f"width_{level}"] = float(upper - lower)
            row[f"coverage_{level}"] = bool(lower <= target <= upper)
        rows.append(row)
    return rows


def _saved_artifact_metadata(artifact: LoadedArtifact) -> dict:
    if artifact.legacy_artifact is not None:
        return artifact.legacy_artifact.metadata
    return artifact.metadata


def _column_value(
    values: np.ndarray,
    columns: tuple[str, ...],
    name: str,
) -> np.ndarray | None:
    try:
        index = columns.index(name)
    except ValueError:
        return None
    return np.asarray(values[:, index], dtype=np.float64)


def _metadata_scalar(
    profile: ClosedLoopProfile,
    metadata_columns: tuple[str, ...],
    name: str,
) -> float:
    try:
        index = metadata_columns.index(name)
    except ValueError:
        return float("nan")
    return float(np.asarray(profile.metadata).reshape(-1)[index])


def _latent_prior_diagnostics(
    artifact: LoadedArtifact,
    profile: ClosedLoopProfile,
) -> dict[str, float]:
    model = artifact.model
    assert isinstance(model, ProbabilisticContractingClosedLoopHPEmulator)
    metadata = jnp.asarray(artifact.scalers.metadata.transform(profile.metadata))
    zeros = jnp.zeros((model.latent_dim,), dtype=metadata.dtype)
    mean = np.asarray(model.condition_xi(metadata, zeros), dtype=np.float64)
    scales = np.empty((model.latent_dim,), dtype=np.float64)
    for index in range(model.latent_dim):
        basis = zeros.at[index].set(1.0)
        conditioned = np.asarray(model.condition_xi(metadata, basis), dtype=np.float64)
        scales[index] = conditioned[index] - mean[index]
    result = {
        "xi_mean_l2": float(np.linalg.norm(mean)),
        "xi_scale_mean": float(np.mean(scales)),
        "xi_scale_min": float(np.min(scales)),
        "xi_scale_max": float(np.max(scales)),
    }
    result.update({f"xi_mean_{index}": float(value) for index, value in enumerate(mean)})
    result.update(
        {f"xi_scale_{index}": float(value) for index, value in enumerate(scales)}
    )
    return result


def _operating_diagnostics(
    artifact: LoadedArtifact,
    profile: ClosedLoopProfile,
    *,
    dt_hours: float,
    min_setpoint_change: float,
    ventilation_rollout_mode: str,
) -> dict[str, float | int | str]:
    metadata = _saved_artifact_metadata(artifact)
    input_columns = tuple(str(value) for value in metadata.get("input_columns", ()))
    metadata_columns = tuple(
        str(value) for value in metadata.get("metadata_columns", ())
    )
    pel = np.asarray(profile.targets[:, 2], dtype=np.float64)
    positive_pel = np.maximum(pel, 0.0)
    setpoint = _column_value(profile.inputs, input_columns, SETPOINT_TIMESERIES_COLUMN)
    if setpoint is None:
        raise ValueError("Artifact input schema does not contain the heating setpoint")
    setpoint_delta = np.diff(setpoint, prepend=setpoint[0])
    up_events = setpoint_delta >= min_setpoint_change
    down_events = setpoint_delta <= -min_setpoint_change

    recorded_ventilation = _column_value(
        profile.inputs,
        input_columns,
        DISTURBANCE_INPUT_COLUMNS[2],
    )
    recorded_fraction = (
        float(np.mean(recorded_ventilation > 1e-8))
        if recorded_ventilation is not None
        else float("nan")
    )
    outdoor = _column_value(
        profile.inputs,
        input_columns,
        DISTURBANCE_INPUT_COLUMNS[0],
    )
    eplus_fraction = float("nan")
    if outdoor is not None and metadata_columns:
        schedule = build_eplus_ventilation_schedule(
            profile.datetime,
            profile.metadata,
            metadata_columns,
        )
        indoor = np.asarray(profile.targets[:, 0], dtype=np.float64)
        eplus_active = (
            (np.asarray(schedule.time_available) > 0.5)
            & (indoor >= np.asarray(schedule.minimum_indoor_temperature_c))
            & (indoor - outdoor >= schedule.minimum_indoor_outdoor_delta_c)
        )
        eplus_fraction = float(np.mean(eplus_active))
    selected_ventilation_fraction = (
        eplus_fraction if ventilation_rollout_mode == "eplus_rule" else recorded_fraction
    )
    availability = _column_value(
        profile.inputs,
        input_columns,
        SPACE_HEATING_AVAILABILITY_COLUMN,
    )
    return {
        "profile_id": int(profile.profile_id),
        "time_steps": int(len(pel)),
        "duration_hours": float(len(pel) * dt_hours),
        "hp_mean_pel_w_m2": float(np.mean(positive_pel)),
        "hp_std_pel_w_m2": float(np.std(positive_pel)),
        "hp_p95_pel_w_m2": float(np.quantile(positive_pel, 0.95)),
        "hp_max_pel_w_m2": float(np.max(positive_pel)),
        "hp_annual_energy_kwh_m2": float(np.sum(positive_pel) * dt_hours / 1000.0),
        "hp_on_fraction": float(np.mean(positive_pel > 1e-6)),
        "space_heating_available_fraction": (
            float(np.mean(availability > 0.5))
            if availability is not None
            else float("nan")
        ),
        "setpoint_event_count": int(np.sum(up_events | down_events)),
        "setpoint_up_event_count": int(np.sum(up_events)),
        "setpoint_down_event_count": int(np.sum(down_events)),
        "setpoint_mean_absolute_event_c": (
            float(np.mean(np.abs(setpoint_delta[up_events | down_events])))
            if np.any(up_events | down_events)
            else 0.0
        ),
        "ventilation_rollout_mode": ventilation_rollout_mode,
        "ventilation_active_fraction": selected_ventilation_fraction,
        "recorded_ventilation_active_fraction": recorded_fraction,
        "eplus_rule_truth_ventilation_active_fraction": eplus_fraction,
        "hp_ref_capacity_w_m2": _metadata_scalar(
            profile,
            metadata_columns,
            HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN,
        ),
        "sh_design_capacity_w_m2": _metadata_scalar(
            profile,
            metadata_columns,
            SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN,
        ),
        "sh_volume_m3_m2": _metadata_scalar(
            profile,
            metadata_columns,
            SH_VOLUME_PER_HEATED_AREA_COLUMN,
        ),
    }


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-15 else float("nan")


def _safe_correlation(covariance: np.ndarray, left: int, right: int) -> float:
    denominator = float(np.sqrt(max(covariance[left, left], 0.0) * max(covariance[right, right], 0.0)))
    return float(covariance[left, right] / denominator) if denominator > 1e-15 else float("nan")


def _spectrum_metrics(matrix: np.ndarray) -> dict[str, float]:
    eigenvalues = np.maximum(np.linalg.eigvalsh(matrix), 0.0)[::-1]
    total = float(np.sum(eigenvalues))
    if total <= 1e-15:
        return {
            "top1_fraction": float("nan"),
            "top4_fraction": float("nan"),
            "effective_rank": float("nan"),
        }
    proportions = eigenvalues / total
    positive = proportions[proportions > 1e-15]
    return {
        "top1_fraction": float(proportions[0]),
        "top4_fraction": float(np.sum(proportions[:4])),
        "effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
    }


def _covariance_spectrum_rows(
    profile_id: int,
    decomposition: CovarianceDecomposition,
) -> list[dict]:
    rows: list[dict] = []
    for component, covariance in (
        ("total", decomposition.total),
        ("persistent_xi", decomposition.persistent_xi),
        ("fast", decomposition.fast),
    ):
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-30))
        correlation = covariance / np.outer(standard_deviation, standard_deviation)
        rows.append(
            {
                "profile_id": int(profile_id),
                "component": component,
                "coordinate_count": int(covariance.shape[0]),
                **{
                    f"covariance_{name}": value
                    for name, value in _spectrum_metrics(covariance).items()
                },
                **{
                    f"correlation_{name}": value
                    for name, value in _spectrum_metrics(correlation).items()
                },
            }
        )
    return rows


def _decomposition_rows(
    profile_id: int,
    coordinates: tuple[KpiCoordinate, ...],
    samples: np.ndarray,
    truth: np.ndarray | None = None,
) -> tuple[list[dict], list[dict], CovarianceDecomposition | None, np.ndarray]:
    truth_values = (
        np.full((len(coordinates),), np.nan, dtype=np.float64)
        if truth is None
        else np.asarray(truth, dtype=np.float64)
    )
    if truth_values.shape != (len(coordinates),):
        raise ValueError("truth must contain one value per KPI coordinate")
    valid = np.all(np.isfinite(samples), axis=(0, 1))
    valid_indices = np.flatnonzero(valid)
    decomposition = (
        decompose_nested_samples(samples[:, :, valid]) if np.any(valid) else None
    )
    index_in_valid = {int(original): offset for offset, original in enumerate(valid_indices)}
    variance_rows: list[dict] = []
    for index, coordinate in enumerate(coordinates):
        target = float(truth_values[index])
        normalization_scale = _kpi_normalization_scale(target)
        common = {
            "profile_id": int(profile_id),
            **asdict(coordinate),
            "kpi_label": coordinate.label,
            "truth": target,
            "normalization_scale": normalization_scale,
        }
        if index not in index_in_valid or decomposition is None:
            variance_rows.append(
                {
                    **common,
                    "valid": False,
                    "valid_rollouts": int(np.sum(np.isfinite(samples[:, :, index]))),
                    "v_total": float("nan"),
                    "v_xi": float("nan"),
                    "v_fast": float("nan"),
                    "std_total": float("nan"),
                    "std_xi": float("nan"),
                    "std_fast": float("nan"),
                    "normalized_v_total": float("nan"),
                    "normalized_v_xi": float("nan"),
                    "normalized_v_fast": float("nan"),
                    "persistent_fraction": float("nan"),
                    "fast_fraction": float("nan"),
                    "closure_error": float("nan"),
                    "relative_closure_error": float("nan"),
                }
            )
            continue
        offset = index_in_valid[index]
        v_total = float(decomposition.total[offset, offset])
        v_xi = float(decomposition.persistent_xi[offset, offset])
        v_fast = float(decomposition.fast[offset, offset])
        closure = float(decomposition.closure[offset, offset])
        scale_squared = normalization_scale**2
        variance_rows.append(
            {
                **common,
                "valid": True,
                "valid_rollouts": int(samples.shape[0] * samples.shape[1]),
                "v_total": v_total,
                "v_xi": v_xi,
                "v_fast": v_fast,
                "std_total": float(np.sqrt(max(v_total, 0.0))),
                "std_xi": float(np.sqrt(max(v_xi, 0.0))),
                "std_fast": float(np.sqrt(max(v_fast, 0.0))),
                "normalized_v_total": _safe_fraction(v_total, scale_squared),
                "normalized_v_xi": _safe_fraction(v_xi, scale_squared),
                "normalized_v_fast": _safe_fraction(v_fast, scale_squared),
                "persistent_fraction": _safe_fraction(v_xi, v_total),
                "fast_fraction": _safe_fraction(v_fast, v_total),
                "closure_error": closure,
                "relative_closure_error": _safe_fraction(abs(closure), v_total),
            }
        )

    covariance_rows: list[dict] = []
    if decomposition is not None:
        valid_coordinates = tuple(coordinates[index] for index in valid_indices)
        for left, left_coordinate in enumerate(valid_coordinates):
            for right, right_coordinate in enumerate(valid_coordinates):
                covariance_rows.append(
                    {
                        "profile_id": int(profile_id),
                        "left_kpi_label": left_coordinate.label,
                        "left_kpi_kind": left_coordinate.kpi_kind,
                        "left_horizon_hours": left_coordinate.horizon_hours,
                        "left_direction": left_coordinate.direction,
                        "right_kpi_label": right_coordinate.label,
                        "right_kpi_kind": right_coordinate.kpi_kind,
                        "right_horizon_hours": right_coordinate.horizon_hours,
                        "right_direction": right_coordinate.direction,
                        "cov_total": float(decomposition.total[left, right]),
                        "cov_xi": float(decomposition.persistent_xi[left, right]),
                        "cov_fast": float(decomposition.fast[left, right]),
                        "corr_total": _safe_correlation(decomposition.total, left, right),
                        "corr_xi": _safe_correlation(decomposition.persistent_xi, left, right),
                        "corr_fast": _safe_correlation(decomposition.fast, left, right),
                        "closure_error": float(decomposition.closure[left, right]),
                    }
                )
    return variance_rows, covariance_rows, decomposition, valid_indices


def _source_inventory(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    *,
    hp_scenario_mode: str,
) -> dict:
    return {
        "persistent_xi": {
            "dimension": int(model.latent_dim),
            "metadata_conditioned": bool(getattr(model, "metadata_conditioned_xi", False)),
            "pathways": list(model.persistent_xi_pathways),
        },
        "fast_group": {
            "thermal_process_innovations": model.process_noise_mode != "none",
            "colored_latent_timescales_hours": list(
                getattr(model, "colored_latent_timescales_hours", ())
            ),
            "hp_start_stop_draws": hp_scenario_mode == "bernoulli",
            "qroom_start_stop_draws": model.qroom_activation_model != "continuous",
            "active_power_innovations": hp_scenario_mode == "bernoulli",
            "controller_innovations": bool(
                model.hp_activation_model == "persistent_markov"
                and model.hp_controller_noise_scale > 0.0
            ),
            "hp_internal_annual_latent": bool(
                model.hp_activation_model == "persistent_markov"
                and model.hp_persistent_latent_dim > 0
            ),
            "hydronic_innovations": bool(model.hydronic_innovation_enabled),
        },
    }


def _building_diagnostic_row(
    artifact: LoadedArtifact,
    profile: ClosedLoopProfile,
    variance_rows: list[dict],
    calibration_rows: list[dict],
    *,
    dt_hours: float,
    min_setpoint_change: float,
    ventilation_rollout_mode: str,
) -> dict:
    valid_variance = [row for row in variance_rows if bool(row["valid"])]
    valid_calibration = [row for row in calibration_rows if bool(row["valid"])]
    v_total = float(np.sum([row["v_total"] for row in valid_variance]))
    v_xi = float(np.sum([row["v_xi"] for row in valid_variance]))
    v_fast = float(np.sum([row["v_fast"] for row in valid_variance]))
    normalized_total = float(
        np.sum([row["normalized_v_total"] for row in valid_variance])
    )
    normalized_xi = float(
        np.sum([row["normalized_v_xi"] for row in valid_variance])
    )
    normalized_fast = float(
        np.sum([row["normalized_v_fast"] for row in valid_variance])
    )
    result = {
        **_operating_diagnostics(
            artifact,
            profile,
            dt_hours=dt_hours,
            min_setpoint_change=min_setpoint_change,
            ventilation_rollout_mode=ventilation_rollout_mode,
        ),
        **_latent_prior_diagnostics(artifact, profile),
        "valid_kpi_count": int(len(valid_variance)),
        "sum_v_total": v_total,
        "sum_v_xi": v_xi,
        "sum_v_fast": v_fast,
        "sum_normalized_v_total": normalized_total,
        "sum_normalized_v_xi": normalized_xi,
        "sum_normalized_v_fast": normalized_fast,
        "median_persistent_fraction": (
            float(np.median([row["persistent_fraction"] for row in valid_variance]))
            if valid_variance
            else float("nan")
        ),
        "variance_weighted_persistent_fraction": _safe_fraction(v_xi, v_total),
        "variance_weighted_fast_fraction": _safe_fraction(v_fast, v_total),
        "native_mean_pit": (
            float(np.mean([row["pit"] for row in valid_calibration]))
            if valid_calibration
            else float("nan")
        ),
        "native_mean_absolute_pit_distance": (
            float(
                np.mean(
                    [
                        row["absolute_pit_distance_from_median"]
                        for row in valid_calibration
                    ]
                )
            )
            if valid_calibration
            else float("nan")
        ),
        "native_mean_normalized_absolute_median_error": (
            float(
                np.mean(
                    [
                        row["normalized_absolute_median_error"]
                        for row in valid_calibration
                    ]
                )
            )
            if valid_calibration
            else float("nan")
        ),
    }
    for level in (50, 80, 90, 95):
        result[f"native_coverage_{level}"] = (
            float(np.mean([row[f"coverage_{level}"] for row in valid_calibration]))
            if valid_calibration
            else float("nan")
        )
    return result


def _write_dashboard(
    variance: pd.DataFrame,
    covariance: pd.DataFrame,
    building_diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    valid = variance[variance["valid"].astype(bool)].copy()
    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Persistent-latent variance share",
            "Fast-randomness variance share",
            "Relative closure error",
            "Total KPI correlation",
            "Persistent-latent correlation",
            "Fast-randomness correlation",
            "Latent scale and persistent share",
            "Persistent share and native PIT",
            "HP load and normalized fast variance",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for index, coordinate in enumerate(dict.fromkeys(valid["kpi_label"])):
        part = valid[valid["kpi_label"] == coordinate]
        color = "#1769aa" if part["kpi_kind"].iloc[0] == "regression_based" else "#d95f02"
        figure.add_trace(
            go.Box(
                x=[coordinate] * len(part),
                y=part["persistent_fraction"],
                name=coordinate,
                marker_color=color,
                showlegend=False,
                boxpoints="all",
                jitter=0.25,
                pointpos=0.0,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Box(
                x=[coordinate] * len(part),
                y=part["fast_fraction"],
                name=coordinate,
                marker_color=color,
                showlegend=False,
                boxpoints="all",
                jitter=0.25,
                pointpos=0.0,
            ),
            row=1,
            col=2,
        )
    closure = valid.groupby("kpi_label", sort=False)["relative_closure_error"].max()
    figure.add_trace(
        go.Bar(x=closure.index, y=closure.values, marker_color="#606b78", showlegend=False),
        row=1,
        col=3,
    )
    if not covariance.empty:
        profile_id = int(covariance["profile_id"].iloc[0])
        selected = covariance[covariance["profile_id"] == profile_id]
        labels = list(dict.fromkeys(selected["left_kpi_label"]))
        for column, metric in enumerate(("corr_total", "corr_xi", "corr_fast"), start=1):
            matrix = selected.pivot(
                index="left_kpi_label",
                columns="right_kpi_label",
                values=metric,
            ).reindex(index=labels, columns=labels)
            figure.add_trace(
                go.Heatmap(
                    z=matrix.to_numpy(dtype=float),
                    x=labels,
                    y=labels,
                    zmin=-1.0,
                    zmax=1.0,
                    colorscale="RdBu",
                    reversescale=True,
                    colorbar={"title": "corr", "len": 0.35, "y": 0.2},
                    showscale=column == 3,
                    hovertemplate="%{y}<br>%{x}<br>corr=%{z:.3f}<extra></extra>",
                ),
                row=2,
                col=column,
            )
        figure.layout.annotations[3].text += f" (building {profile_id})"
        figure.layout.annotations[4].text += f" (building {profile_id})"
        figure.layout.annotations[5].text += f" (building {profile_id})"
    if not building_diagnostics.empty:
        hover = building_diagnostics["profile_id"].astype(str)
        figure.add_trace(
            go.Scatter(
                x=building_diagnostics["xi_scale_mean"],
                y=building_diagnostics["variance_weighted_persistent_fraction"],
                mode="markers",
                marker={
                    "size": 10,
                    "color": building_diagnostics["hp_mean_pel_w_m2"],
                    "colorscale": "Viridis",
                    "showscale": False,
                },
                text=hover,
                showlegend=False,
                hovertemplate=(
                    "building=%{text}<br>mean xi scale=%{x:.3f}"
                    "<br>persistent share=%{y:.3f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=building_diagnostics["variance_weighted_persistent_fraction"],
                y=building_diagnostics["native_mean_absolute_pit_distance"],
                mode="markers",
                marker={
                    "size": 10,
                    "color": building_diagnostics["native_coverage_90"],
                    "colorscale": "Cividis",
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "showscale": False,
                },
                text=hover,
                showlegend=False,
                hovertemplate=(
                    "building=%{text}<br>persistent share=%{x:.3f}"
                    "<br>mean |PIT-0.5|=%{y:.3f}<extra></extra>"
                ),
            ),
            row=3,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=building_diagnostics["hp_mean_pel_w_m2"],
                y=building_diagnostics["sum_normalized_v_fast"],
                mode="markers",
                marker={
                    "size": 10,
                    "color": building_diagnostics["ventilation_active_fraction"],
                    "colorscale": "Plasma",
                    "showscale": False,
                },
                text=hover,
                showlegend=False,
                hovertemplate=(
                    "building=%{text}<br>mean Pel=%{x:.3f} W/m2"
                    "<br>sum normalized fast variance=%{y:.3f}<extra></extra>"
                ),
            ),
            row=3,
            col=3,
        )
    figure.update_yaxes(range=[0.0, 1.05], row=1, col=1)
    figure.update_yaxes(range=[0.0, 1.05], row=1, col=2)
    figure.update_yaxes(range=[0.0, 1.05], row=3, col=1)
    figure.update_xaxes(title_text="Mean conditioned xi scale", row=3, col=1)
    figure.update_yaxes(title_text="Persistent variance share", row=3, col=1)
    figure.update_xaxes(title_text="Persistent variance share", row=3, col=2)
    figure.update_yaxes(title_text="Mean |PIT - 0.5|", row=3, col=2)
    figure.update_xaxes(title_text="Mean simulator Pel [W/m2]", row=3, col=3)
    figure.update_yaxes(title_text="Normalized fast variance", row=3, col=3)
    figure.update_xaxes(tickangle=45, row=1)
    figure.update_layout(
        title="Per-building annual KPI uncertainty decomposition",
        height=1650,
        width=1900,
        hoverlabel={"namelength": -1},
        margin={"b": 220},
    )
    figure.write_html(path, include_plotlyjs="cdn")


def _load_cached_profile(
    path: Path,
    *,
    xi_draws: int,
    fast_rollouts: int,
    latent_dim: int,
    coordinates: tuple[KpiCoordinate, ...],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as cached:
        base_xi = np.asarray(cached["base_xi"], dtype=np.float32)
        samples = np.asarray(cached["samples"], dtype=np.float64)
        labels = tuple(str(value) for value in cached["coordinate_labels"])
    expected_xi_shape = (xi_draws, latent_dim)
    expected_sample_shape = (xi_draws, fast_rollouts, len(coordinates))
    expected_labels = tuple(value.label for value in coordinates)
    if base_xi.shape != expected_xi_shape:
        raise ValueError(
            f"Cached base_xi in {path} has shape {base_xi.shape}; "
            f"expected {expected_xi_shape}"
        )
    if samples.shape != expected_sample_shape:
        raise ValueError(
            f"Cached samples in {path} have shape {samples.shape}; "
            f"expected {expected_sample_shape}"
        )
    if labels != expected_labels:
        raise ValueError(f"Cached KPI coordinates in {path} do not match this request")
    return base_xi, samples


def run(args: argparse.Namespace) -> Path:
    artifact = load_artifact(args.artifact_dir)
    model = artifact.model
    if not isinstance(model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError("Artifact must contain ProbabilisticContractingClosedLoopHPEmulator")
    if model.latent_dim < 1:
        raise ValueError("The artifact has no persistent annual latent")
    profiles = load_profiles(
        artifact,
        dataset_path=args.dataset,
        source=args.profile_source,
        max_profiles=args.max_profiles,
        profile_ids=args.profile_ids,
    )
    if not all(isinstance(profile, ClosedLoopProfile) for profile in profiles):
        raise TypeError("KPI variance decomposition requires closed-loop profiles")
    horizons = tuple(float(value) for value in args.horizons_hours)
    configs = _kpi_configs(
        horizons,
        dt_hours=args.dt_hours,
        min_setpoint_change=args.min_setpoint_change,
        min_events=args.min_events,
        controls=args.controls,
    )
    coordinates = _coordinates(configs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "dataset": str(Path(args.dataset).resolve()) if args.dataset is not None else None,
        "profile_source": args.profile_source,
        "profile_ids": [int(profile.profile_id) for profile in profiles],
        "xi_draws": int(args.xi_draws),
        "fast_rollouts": int(args.fast_rollouts),
        "common_random_numbers": bool(args.common_random_numbers),
        "hp_scenario_mode": args.hp_scenario_mode,
        "ventilation_rollout_mode": args.ventilation_rollout_mode,
        "horizons_hours": list(horizons),
        "dt_hours": float(args.dt_hours),
        "min_setpoint_change": float(args.min_setpoint_change),
        "min_events": int(args.min_events),
        "controls": args.controls,
        "seed": int(args.seed),
        "coordinates": [asdict(value) for value in coordinates],
    }
    request_path = args.output_dir / "request.json"
    if request_path.exists() and getattr(args, "resume", True):
        previous_request = json.loads(request_path.read_text())
        if previous_request != json_safe(request):
            raise ValueError(
                f"Existing decomposition request differs at {request_path}; "
                "use --no-resume or a different output directory"
            )
    request_path.write_text(json.dumps(json_safe(request), indent=2, sort_keys=True) + "\n")
    cache_dir = args.output_dir / "per_building"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict] = []
    mean_rows: list[dict] = []
    variance_rows: list[dict] = []
    covariance_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    calibration_rows: list[dict] = []
    building_rows: list[dict] = []
    root_key = jax.random.PRNGKey(args.seed)
    for profile_index, profile in enumerate(profiles):
        assert isinstance(profile, ClosedLoopProfile)
        cache_path = cache_dir / f"profile_{int(profile.profile_id)}.npz"
        if cache_path.exists() and getattr(args, "resume", True):
            xi_draws, samples = _load_cached_profile(
                cache_path,
                xi_draws=args.xi_draws,
                fast_rollouts=args.fast_rollouts,
                latent_dim=model.latent_dim,
                coordinates=coordinates,
            )
            print(
                f"variance_decomposition profile={profile.profile_id} "
                f"index={profile_index + 1}/{len(profiles)} resumed=true",
                flush=True,
            )
        else:
            profile_key = jax.random.fold_in(root_key, int(profile.profile_id))
            xi_key, fast_key = jax.random.split(profile_key)
            xi_draws = np.asarray(
                jax.random.normal(
                    xi_key,
                    (args.xi_draws, model.latent_dim),
                    dtype=jnp.float32,
                )
            )
            samples = np.full(
                (args.xi_draws, args.fast_rollouts, len(coordinates)),
                np.nan,
                dtype=np.float64,
            )
            for xi_index, base_xi in enumerate(xi_draws):
                rollout_key = (
                    fast_key
                    if args.common_random_numbers
                    else jax.random.fold_in(fast_key, xi_index)
                )
                scenarios = _fixed_xi_scenarios(
                    artifact,
                    profile,
                    base_xi,
                    fast_rollouts=args.fast_rollouts,
                    key=rollout_key,
                    hp_scenario_mode=args.hp_scenario_mode,
                    ventilation_rollout_mode=args.ventilation_rollout_mode,
                )
                for fast_index, trace in enumerate(scenarios):
                    samples[xi_index, fast_index] = _trace_kpi_vector(
                        profile,
                        trace,
                        configs,
                    )
            np.savez_compressed(
                cache_path,
                base_xi=xi_draws,
                samples=samples,
                coordinate_labels=np.asarray(
                    [value.label for value in coordinates],
                    dtype=str,
                ),
            )
        finite_counts = np.sum(np.isfinite(samples), axis=1)
        conditional_means = np.divide(
            np.nansum(samples, axis=1),
            finite_counts,
            out=np.full((args.xi_draws, len(coordinates)), np.nan),
            where=finite_counts > 0,
        )
        for xi_index, base_xi in enumerate(xi_draws):
            xi_values = {f"base_xi_{index}": float(value) for index, value in enumerate(base_xi)}
            for coordinate_index, coordinate in enumerate(coordinates):
                mean_rows.append(
                    {
                        "profile_id": int(profile.profile_id),
                        "xi_index": xi_index,
                        **xi_values,
                        **asdict(coordinate),
                        "kpi_label": coordinate.label,
                        "conditional_mean": float(conditional_means[xi_index, coordinate_index]),
                    }
                )
                if args.write_samples:
                    for fast_index in range(args.fast_rollouts):
                        sample_rows.append(
                            {
                                "profile_id": int(profile.profile_id),
                                "xi_index": xi_index,
                                "fast_index": fast_index,
                                **xi_values,
                                **asdict(coordinate),
                                "kpi_label": coordinate.label,
                                "value": float(samples[xi_index, fast_index, coordinate_index]),
                            }
                        )
        truth = _trace_kpi_vector(profile, profile.targets, configs)
        profile_calibration = _native_calibration_rows(
            profile.profile_id,
            coordinates,
            samples,
            truth,
        )
        profile_variance, profile_covariance, profile_decomposition, _ = _decomposition_rows(
            profile.profile_id,
            coordinates,
            samples,
            truth,
        )
        variance_rows.extend(profile_variance)
        covariance_rows.extend(profile_covariance)
        calibration_rows.extend(profile_calibration)
        building_rows.append(
            _building_diagnostic_row(
                artifact,
                profile,
                profile_variance,
                profile_calibration,
                dt_hours=args.dt_hours,
                min_setpoint_change=args.min_setpoint_change,
                ventilation_rollout_mode=args.ventilation_rollout_mode,
            )
        )
        if profile_decomposition is not None:
            spectrum_rows.extend(
                _covariance_spectrum_rows(
                    profile.profile_id,
                    profile_decomposition,
                )
            )
        profile_valid = [row for row in profile_variance if row["valid"]]
        median_share = float(
            np.nanmedian([row["persistent_fraction"] for row in profile_valid])
        ) if profile_valid else float("nan")
        print(
            f"variance_decomposition profile={profile.profile_id} "
            f"index={profile_index + 1}/{len(profiles)} "
            f"valid_kpis={len(profile_valid)}/{len(coordinates)} "
            f"median_persistent_fraction={median_share:.3f}",
            flush=True,
        )

    variance = pd.DataFrame(variance_rows)
    covariance = pd.DataFrame(covariance_rows)
    covariance_spectrum = pd.DataFrame(spectrum_rows)
    conditional_means = pd.DataFrame(mean_rows)
    native_calibration = pd.DataFrame(calibration_rows)
    building_diagnostics = pd.DataFrame(building_rows)
    residual_analysis = analyze_residual_covariance(
        native_calibration,
        covariance,
        scale_reference_dir=getattr(args, "scale_reference_dir", None),
        minimum_scale=float(getattr(args, "minimum_scale", 1e-3)),
        minimum_scale_fraction=float(
            getattr(args, "minimum_scale_fraction", 0.01)
        ),
        bootstrap_samples=int(getattr(args, "residual_bootstrap_samples", 2000)),
        seed=int(args.seed),
    )
    native_calibration = residual_analysis.native_calibration
    residual_by_building = (
        native_calibration[native_calibration["valid"].astype(bool)]
        .groupby("profile_id", as_index=False)
        .agg(
            mean_standardized_residual=("standardized_residual", "mean"),
            mean_absolute_standardized_residual=(
                "standardized_residual",
                lambda values: float(np.mean(np.abs(values))),
            ),
        )
    )
    building_diagnostics = building_diagnostics.merge(
        residual_by_building,
        on="profile_id",
        how="left",
    )
    variance.to_csv(args.output_dir / "variance_decomposition.csv", index=False)
    covariance.to_csv(args.output_dir / "covariance_decomposition.csv", index=False)
    covariance_spectrum.to_csv(
        args.output_dir / "covariance_spectrum.csv",
        index=False,
    )
    conditional_means.to_csv(args.output_dir / "conditional_means.csv", index=False)
    native_calibration.to_csv(
        args.output_dir / "native_calibration.csv",
        index=False,
    )
    building_diagnostics.to_csv(
        args.output_dir / "building_diagnostics.csv",
        index=False,
    )
    residual_analysis.standardized_residuals.to_csv(
        args.output_dir / "standardized_residual_vectors.csv",
        index=False,
    )
    residual_analysis.location_bias.to_csv(
        args.output_dir / "residual_location_bias.csv",
        index=False,
    )
    residual_analysis.covariance.to_csv(
        args.output_dir / "residual_xi_covariance_comparison.csv",
        index=False,
    )
    residual_analysis.modes.to_csv(
        args.output_dir / "residual_xi_covariance_modes.csv",
        index=False,
    )
    residual_analysis.comparison.to_csv(
        args.output_dir / "residual_xi_covariance_summary.csv",
        index=False,
    )
    residual_analysis.building_xi_spectrum.to_csv(
        args.output_dir / "standardized_xi_spectrum_by_building.csv",
        index=False,
    )
    if args.write_samples:
        pd.DataFrame(sample_rows).to_csv(args.output_dir / "nested_kpi_samples.csv", index=False)

    valid = variance[variance["valid"].astype(bool)]
    aggregate = (
        valid.groupby(
            ["kpi_kind", "horizon_hours", "direction", "kpi_label"],
            as_index=False,
            sort=False,
        )
        .agg(
            building_count=("profile_id", "nunique"),
            median_persistent_fraction=("persistent_fraction", "median"),
            mean_persistent_fraction=("persistent_fraction", "mean"),
            median_fast_fraction=("fast_fraction", "median"),
            mean_fast_fraction=("fast_fraction", "mean"),
            mean_v_total=("v_total", "mean"),
            mean_v_xi=("v_xi", "mean"),
            mean_v_fast=("v_fast", "mean"),
            mean_normalized_v_total=("normalized_v_total", "mean"),
            mean_normalized_v_xi=("normalized_v_xi", "mean"),
            mean_normalized_v_fast=("normalized_v_fast", "mean"),
            max_relative_closure_error=("relative_closure_error", "max"),
        )
        if not valid.empty
        else pd.DataFrame()
    )
    aggregate.to_csv(args.output_dir / "variance_summary.csv", index=False)
    valid_calibration = native_calibration[
        native_calibration["valid"].astype(bool)
    ].copy()
    calibration_summary = (
        valid_calibration.groupby(
            ["kpi_kind", "horizon_hours", "direction", "kpi_label"],
            as_index=False,
            sort=False,
        ).agg(
            building_count=("profile_id", "nunique"),
            mean_pit=("pit", "mean"),
            median_pit=("pit", "median"),
            mean_absolute_pit_distance=(
                "absolute_pit_distance_from_median",
                "mean",
            ),
            coverage_50=("coverage_50", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_90=("coverage_90", "mean"),
            coverage_95=("coverage_95", "mean"),
            median_normalized_absolute_median_error=(
                "normalized_absolute_median_error",
                "median",
            ),
        )
        if not valid_calibration.empty
        else pd.DataFrame()
    )
    calibration_summary.to_csv(
        args.output_dir / "native_calibration_summary.csv",
        index=False,
    )
    spectrum_summary = covariance_spectrum.groupby(
        "component",
        as_index=False,
        sort=False,
    ).agg(
        building_count=("profile_id", "nunique"),
        median_covariance_top1_fraction=("covariance_top1_fraction", "median"),
        median_covariance_effective_rank=("covariance_effective_rank", "median"),
        median_correlation_top1_fraction=("correlation_top1_fraction", "median"),
        median_correlation_top4_fraction=("correlation_top4_fraction", "median"),
        median_correlation_effective_rank=("correlation_effective_rank", "median"),
    )
    spectrum_summary.to_csv(
        args.output_dir / "covariance_spectrum_summary.csv",
        index=False,
    )
    dashboard_path = args.output_dir / "variance_decomposition.html"
    _write_dashboard(variance, covariance, building_diagnostics, dashboard_path)
    residual_dashboard_path = args.output_dir / "residual_covariance.html"
    write_residual_covariance_dashboard(
        residual_analysis,
        residual_dashboard_path,
    )

    summary = {
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "dataset": str(Path(args.dataset).resolve()) if args.dataset is not None else None,
        "profile_source": args.profile_source,
        "profile_ids": [int(profile.profile_id) for profile in profiles],
        "xi_draws": int(args.xi_draws),
        "fast_rollouts": int(args.fast_rollouts),
        "common_random_numbers": bool(args.common_random_numbers),
        "hp_scenario_mode": args.hp_scenario_mode,
        "ventilation_rollout_mode": args.ventilation_rollout_mode,
        "coordinates": [asdict(value) for value in coordinates],
        "uncertainty_sources": _source_inventory(
            model,
            hp_scenario_mode=args.hp_scenario_mode,
        ),
        "valid_building_kpi_cells": int(len(valid)),
        "aggregate_diagnostics": {
            "mean_persistent_fraction": (
                float(valid["persistent_fraction"].mean()) if not valid.empty else None
            ),
            "median_persistent_fraction": (
                float(valid["persistent_fraction"].median()) if not valid.empty else None
            ),
            "variance_weighted_persistent_fraction": (
                float(valid["v_xi"].sum() / valid["v_total"].sum())
                if not valid.empty and float(valid["v_total"].sum()) > 1e-15
                else None
            ),
            "mean_native_absolute_pit_distance": (
                float(valid_calibration["absolute_pit_distance_from_median"].mean())
                if not valid_calibration.empty
                else None
            ),
            "native_coverage_90": (
                float(valid_calibration["coverage_90"].mean())
                if not valid_calibration.empty
                else None
            ),
        },
        "variance_normalization": (
            "Each KPI variance is divided by max(abs(simulator KPI), 1e-6)^2 "
            "for the same fixed building and coordinate."
        ),
        "residual_covariance": {
            "definition": (
                "OAS shrinkage covariance across held-out standardized residual "
                "vectors, compared with the mean persistent-xi covariance after "
                "using the same per-building RNS scales."
            ),
            "scale_floor_source": residual_analysis.scale_floor_source,
            "bootstrap_samples": int(
                getattr(args, "residual_bootstrap_samples", 2000)
            ),
            "comparison": {
                str(row.metric): float(row.estimate)
                for row in residual_analysis.comparison.itertuples(index=False)
            },
        },
        "maximum_relative_closure_error": (
            float(valid["relative_closure_error"].max()) if not valid.empty else None
        ),
        "files": {
            "variance": "variance_decomposition.csv",
            "covariance": "covariance_decomposition.csv",
            "covariance_spectrum": "covariance_spectrum.csv",
            "covariance_spectrum_summary": "covariance_spectrum_summary.csv",
            "conditional_means": "conditional_means.csv",
            "samples": "nested_kpi_samples.csv" if args.write_samples else None,
            "aggregate": "variance_summary.csv",
            "native_calibration": "native_calibration.csv",
            "native_calibration_summary": "native_calibration_summary.csv",
            "building_diagnostics": "building_diagnostics.csv",
            "dashboard": dashboard_path.name,
            "standardized_residual_vectors": "standardized_residual_vectors.csv",
            "residual_location_bias": "residual_location_bias.csv",
            "residual_xi_covariance_comparison": (
                "residual_xi_covariance_comparison.csv"
            ),
            "residual_xi_covariance_modes": "residual_xi_covariance_modes.csv",
            "residual_xi_covariance_summary": "residual_xi_covariance_summary.csv",
            "standardized_xi_spectrum_by_building": (
                "standardized_xi_spectrum_by_building.csv"
            ),
            "residual_covariance_dashboard": residual_dashboard_path.name,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    print(f"saved_variance_decomposition={dashboard_path}")
    return dashboard_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile-source",
        choices=("train", "test", "selected"),
        default="test",
    )
    parser.add_argument("--profile-ids", type=int, nargs="+")
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--xi-draws", type=int, default=32)
    parser.add_argument("--fast-rollouts", type=int, default=8)
    parser.add_argument(
        "--common-random-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hp-scenario-mode",
        choices=("bernoulli", "expected"),
        default="bernoulli",
    )
    parser.add_argument(
        "--ventilation-rollout-mode",
        choices=("recorded", "eplus_rule"),
        default="eplus_rule",
    )
    parser.add_argument("--horizons-hours", type=float, nargs="+", default=(0.5, 1.0, 2.0, 3.0))
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--controls", choices=("none", "weather", "full"), default="full")
    parser.add_argument("--scale-reference-dir", type=Path)
    parser.add_argument("--minimum-scale", type=float, default=1e-3)
    parser.add_argument("--minimum-scale-fraction", type=float, default=0.01)
    parser.add_argument("--residual-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    if not args.artifact_dir.exists():
        parser.error(f"--artifact-dir does not exist: {args.artifact_dir}")
    if args.dataset is not None and not args.dataset.exists():
        parser.error(f"--dataset does not exist: {args.dataset}")
    if args.xi_draws < 2:
        parser.error("--xi-draws must be at least 2")
    if args.fast_rollouts < 2:
        parser.error("--fast-rollouts must be at least 2")
    if args.max_profiles is not None and args.max_profiles < 1:
        parser.error("--max-profiles must be positive")
    if args.dt_hours <= 0.0:
        parser.error("--dt-hours must be positive")
    if any(value <= 0.0 for value in args.horizons_hours):
        parser.error("--horizons-hours values must be positive")
    if args.minimum_scale <= 0.0:
        parser.error("--minimum-scale must be positive")
    if args.minimum_scale_fraction < 0.0:
        parser.error("--minimum-scale-fraction cannot be negative")
    if args.residual_bootstrap_samples < 1:
        parser.error("--residual-bootstrap-samples must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
