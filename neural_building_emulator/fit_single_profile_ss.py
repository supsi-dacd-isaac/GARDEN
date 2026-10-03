"""Fit direct stable state-space models to one building profile.

This is an intentionally local diagnostic: no metadata-conditioned generator,
no train/test profile split, no probabilistic particles. The goal is to answer
whether the state-space equation class can fit one known full-year trajectory
before we ask a global neural model to learn it for many buildings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    DISTURBANCE_COLUMNS,
    HEATING_INPUT_COLUMNS,
    InputFeatureMode,
    METADATA_COLUMNS,
    PROFILE_ID_COLUMN,
    TARGET_COLUMN,
    input_columns,
    normalize_heating_mode,
    required_columns,
)
from .data import DEFAULT_DATASET_PATH, BuildingProfile, WindowTargetAlignment, _read_parquet, to_profiles
from .models.schur import SchurMode, simba_schur_matrix, spectral_radius

FeatureMode = Literal["raw", "engineered"]
ExperimentSuite = Literal["first4", "custom"]
OutputTiming = Literal["pre_update", "post_update"]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    state_dim: int
    zero_d: bool
    feature_mode: FeatureMode
    output_timing: OutputTiming


@dataclass(frozen=True)
class PreparedProfile:
    profile: BuildingProfile
    datetime: np.ndarray
    raw_inputs: np.ndarray
    raw_input_names: tuple[str, ...]
    target_c: np.ndarray
    target_norm: np.ndarray
    target_mean: float
    target_scale: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: tuple[str, ...]
    setpoint_c: float
    heat_on_threshold: float
    target_alignment: WindowTargetAlignment
    input_feature_mode: InputFeatureMode
    fit_start: str | None
    fit_end: str | None


@dataclass(frozen=True)
class FitResult:
    spec: ExperimentSpec
    params: dict[str, jax.Array]
    prediction_c: np.ndarray
    residual_c: np.ndarray
    metrics: dict[str, float | str | int]
    html_path: Path


def _safe_scale(values: np.ndarray, *, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    scale = np.nanstd(values, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, 1.0).astype(np.float32)
    return mean, scale


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    a_masked = a[mask]
    b_masked = b[mask]
    if float(np.std(a_masked)) == 0.0 or float(np.std(b_masked)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_masked, b_masked)[0, 1])


def _feature_names(raw_names: Sequence[str], feature_mode: FeatureMode) -> tuple[str, ...]:
    if feature_mode == "raw":
        return tuple(raw_names)
    return (
        *raw_names,
        "outdoor_minus_predicted_indoor_temperature",
        "setpoint_minus_predicted_indoor_temperature",
        "heat_on",
    )


def _physical_features_for_scaling(
    raw_inputs: np.ndarray,
    target_c: np.ndarray,
    *,
    feature_mode: FeatureMode,
    setpoint_c: float,
    heat_on_threshold: float,
) -> np.ndarray:
    if feature_mode == "raw":
        return raw_inputs.astype(np.float32)
    heat_on = (raw_inputs[:, 0] > heat_on_threshold).astype(np.float32)
    return np.column_stack(
        [
            raw_inputs,
            raw_inputs[:, 1] - target_c,
            np.full_like(target_c, setpoint_c) - target_c,
            heat_on,
        ]
    ).astype(np.float32)


def _normalized_step_features(
    raw_t: jnp.ndarray,
    y_t_norm: jnp.ndarray,
    *,
    feature_mode: FeatureMode,
    feature_mean: jnp.ndarray,
    feature_scale: jnp.ndarray,
    target_mean: float,
    target_scale: float,
    setpoint_c: float,
    heat_on_threshold: float,
) -> jnp.ndarray:
    if feature_mode == "raw":
        physical = raw_t
    else:
        y_t_c = y_t_norm * target_scale + target_mean
        heat_on = (raw_t[0] > heat_on_threshold).astype(raw_t.dtype)
        physical = jnp.concatenate(
            [
                raw_t,
                jnp.asarray([raw_t[1] - y_t_c, setpoint_c - y_t_c, heat_on], dtype=raw_t.dtype),
            ],
            axis=0,
        )
    return (physical - feature_mean) / feature_scale


def load_single_profile(
    dataset_path: Path,
    *,
    profile_id: int,
    heating_mode: str,
    heat_input_normalization: str,
    input_feature_mode: InputFeatureMode,
    heating_regime_window_steps: int,
    heat_on_threshold: float,
) -> BuildingProfile:
    mode = normalize_heating_mode(heating_mode)
    columns = required_columns(mode)
    filters = [(PROFILE_ID_COLUMN, "=", int(profile_id))]
    try:
        df = _read_parquet(dataset_path, columns=columns, filters=filters)
    except (ValueError, NotImplementedError):
        df = _read_parquet(dataset_path, columns=columns)
        df = df[df[PROFILE_ID_COLUMN] == int(profile_id)]
    df = df[df[PROFILE_ID_COLUMN] == int(profile_id)].copy()
    if df.empty:
        raise ValueError(f"Profile {profile_id} not found in {dataset_path}")
    profiles = to_profiles(
        df,
        mode,
        heat_input_normalization,  # type: ignore[arg-type]
        input_feature_mode=input_feature_mode,
        heating_regime_window_steps=heating_regime_window_steps,
        heat_on_threshold=heat_on_threshold,
    )
    if len(profiles) != 1:
        raise ValueError(f"Expected one profile for {profile_id}, got {len(profiles)}")
    return profiles[0]


def prepare_profile(
    profile: BuildingProfile,
    *,
    heating_mode: str,
    heat_input_normalization: str,
    feature_mode: FeatureMode,
    heat_on_threshold: float,
    target_alignment: WindowTargetAlignment,
    input_feature_mode: InputFeatureMode,
    heating_regime_window_steps: int,
    fit_start: str | None,
    fit_end: str | None,
) -> PreparedProfile:
    if target_alignment == "same_time":
        raw_inputs = profile.inputs.astype(np.float32)
        target_c = profile.target[:, 0].astype(np.float32)
        datetimes = profile.datetime
    elif target_alignment == "next_step":
        if profile.target.shape[0] < 2:
            raise ValueError("Profile needs at least two timestamps for next_step target alignment")
        raw_inputs = profile.inputs[:-1].astype(np.float32)
        target_c = profile.target[1:, 0].astype(np.float32)
        datetimes = profile.datetime[1:]
    else:
        raise ValueError("target_alignment must be 'same_time' or 'next_step'")

    if fit_start is not None or fit_end is not None:
        timestamps = pd.to_datetime(datetimes)
        mask = np.ones(len(timestamps), dtype=bool)
        if fit_start is not None:
            mask &= np.asarray(timestamps >= pd.Timestamp(fit_start))
        if fit_end is not None:
            mask &= np.asarray(timestamps <= pd.Timestamp(fit_end))
        if int(mask.sum()) < 2:
            raise ValueError(
                f"Selected period fit_start={fit_start!r}, fit_end={fit_end!r} "
                f"contains only {int(mask.sum())} samples"
            )
        raw_inputs = raw_inputs[mask]
        target_c = target_c[mask]
        datetimes = datetimes[mask]

    target_mean = float(np.mean(target_c))
    target_scale = float(np.std(target_c))
    if not np.isfinite(target_scale) or target_scale < 1e-6:
        target_scale = 1.0
    target_norm = ((target_c - target_mean) / target_scale).astype(np.float32)
    setpoint_index = METADATA_COLUMNS.index("shSetpoint")
    setpoint_c = float(profile.metadata[setpoint_index])

    raw_names = input_columns(
        heating_mode,
        heat_input_normalization,  # type: ignore[arg-type]
        input_feature_mode=input_feature_mode,
        heating_regime_window_steps=heating_regime_window_steps,
    )
    feature_values = _physical_features_for_scaling(
        raw_inputs,
        target_c,
        feature_mode=feature_mode,
        setpoint_c=setpoint_c,
        heat_on_threshold=heat_on_threshold,
    )
    feature_mean, feature_scale = _safe_scale(feature_values)
    return PreparedProfile(
        profile=profile,
        datetime=datetimes,
        raw_inputs=raw_inputs,
        raw_input_names=tuple(raw_names),
        target_c=target_c,
        target_norm=target_norm,
        target_mean=target_mean,
        target_scale=target_scale,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_names=_feature_names(raw_names, feature_mode),
        setpoint_c=setpoint_c,
        heat_on_threshold=heat_on_threshold,
        target_alignment=target_alignment,
        input_feature_mode=input_feature_mode,
        fit_start=fit_start,
        fit_end=fit_end,
    )


def init_params(
    *,
    key: jax.Array,
    state_dim: int,
    feature_dim: int,
    init_scale: float,
) -> dict[str, jax.Array]:
    keys = jax.random.split(key, 8)
    return {
        "raw_w": init_scale * jax.random.normal(keys[0], ((2 * state_dim) ** 2,)),
        "raw_v": init_scale * jax.random.normal(keys[1], (state_dim**2,)),
        "b_matrix": init_scale * jax.random.normal(keys[2], (state_dim, feature_dim)),
        "c_vector": init_scale * jax.random.normal(keys[3], (state_dim,)),
        "d_vector": init_scale * jax.random.normal(keys[4], (feature_dim,)),
        "state_bias": jnp.zeros((state_dim,), dtype=jnp.float32),
        "output_bias": jnp.asarray(0.0, dtype=jnp.float32),
        "x0": init_scale * jax.random.normal(keys[5], (state_dim,)),
    }


def state_matrix(
    params: dict[str, jax.Array],
    *,
    state_dim: int,
    schur_gamma: float,
    pf_lambda_min: float,
    schur_mode: SchurMode,
) -> jnp.ndarray:
    return simba_schur_matrix(
        params["raw_w"],
        params["raw_v"],
        state_dim,
        gamma=schur_gamma,
        pf_lambda_min=pf_lambda_min,
        mode=schur_mode,
    )


def rollout_normalized(
    params: dict[str, jax.Array],
    raw_inputs: jnp.ndarray,
    *,
    spec: ExperimentSpec,
    feature_mean: jnp.ndarray,
    feature_scale: jnp.ndarray,
    target_mean: float,
    target_scale: float,
    setpoint_c: float,
    heat_on_threshold: float,
    schur_gamma: float,
    pf_lambda_min: float,
    schur_mode: SchurMode,
) -> jnp.ndarray:
    a_matrix = state_matrix(
        params,
        state_dim=spec.state_dim,
        schur_gamma=schur_gamma,
        pf_lambda_min=pf_lambda_min,
        schur_mode=schur_mode,
    )

    def step(x_t: jnp.ndarray, raw_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        y_state = jnp.dot(params["c_vector"], x_t) + params["output_bias"]
        features_t = _normalized_step_features(
            raw_t,
            y_state,
            feature_mode=spec.feature_mode,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            setpoint_c=setpoint_c,
            heat_on_threshold=heat_on_threshold,
        )
        direct = 0.0 if spec.zero_d else jnp.dot(params["d_vector"], features_t)
        x_next = a_matrix @ x_t + params["b_matrix"] @ features_t + params["state_bias"]
        if spec.output_timing == "pre_update":
            y_t = y_state + direct
        elif spec.output_timing == "post_update":
            y_t = jnp.dot(params["c_vector"], x_next) + params["output_bias"] + direct
        else:
            raise ValueError(f"Unknown output_timing {spec.output_timing!r}")
        return x_next, y_t

    _, outputs = jax.lax.scan(jax.checkpoint(step), params["x0"], raw_inputs)
    return outputs


def make_train_step(
    *,
    spec: ExperimentSpec,
    optimizer: optax.GradientTransformation,
    ridge: float,
    schur_gamma: float,
    pf_lambda_min: float,
    schur_mode: SchurMode,
    target_mean: float,
    target_scale: float,
    setpoint_c: float,
    heat_on_threshold: float,
):
    def loss_fn(
        params: dict[str, jax.Array],
        raw_inputs: jnp.ndarray,
        target_norm: jnp.ndarray,
        feature_mean: jnp.ndarray,
        feature_scale: jnp.ndarray,
    ) -> jnp.ndarray:
        prediction = rollout_normalized(
            params,
            raw_inputs,
            spec=spec,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            setpoint_c=setpoint_c,
            heat_on_threshold=heat_on_threshold,
            schur_gamma=schur_gamma,
            pf_lambda_min=pf_lambda_min,
            schur_mode=schur_mode,
        )
        mse = jnp.mean((prediction - target_norm) ** 2)
        if ridge == 0.0:
            return mse
        penalty = (
            jnp.mean(params["b_matrix"] ** 2)
            + jnp.mean(params["c_vector"] ** 2)
            + jnp.mean(params["state_bias"] ** 2)
            + params["output_bias"] ** 2
        )
        if not spec.zero_d:
            penalty = penalty + jnp.mean(params["d_vector"] ** 2)
        return mse + ridge * penalty

    @jax.jit
    def train_step(
        params: dict[str, jax.Array],
        opt_state: optax.OptState,
        raw_inputs: jnp.ndarray,
        target_norm: jnp.ndarray,
        feature_mean: jnp.ndarray,
        feature_scale: jnp.ndarray,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, raw_inputs, target_norm, feature_mean, feature_scale)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step, jax.jit(loss_fn)


def compute_metrics(
    target_c: np.ndarray,
    prediction_c: np.ndarray,
    raw_inputs: np.ndarray,
    *,
    spec: ExperimentSpec,
    feature_dim: int,
    rho: float,
    final_loss: float,
    setpoint_c: float,
    target_alignment: WindowTargetAlignment,
    input_feature_mode: InputFeatureMode,
) -> dict[str, float | str | int]:
    residual = prediction_c - target_c
    error = prediction_c - target_c
    d_true = np.diff(target_c)
    d_pred = np.diff(prediction_c)
    target_range = float(np.max(target_c) - np.min(target_c))
    if target_range <= 1e-6:
        target_range = 1.0
    metrics: dict[str, float | str | int] = {
        "experiment": spec.name,
        "state_dim": spec.state_dim,
        "feature_mode": spec.feature_mode,
        "output_timing": spec.output_timing,
        "target_alignment": target_alignment,
        "input_feature_mode": input_feature_mode,
        "feature_dim": feature_dim,
        "zero_d": int(spec.zero_d),
        "rho": rho,
        "final_loss_norm_mse": final_loss,
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "mae_c": float(np.mean(np.abs(error))),
        "bias_c": float(np.mean(error)),
        "nmae": float(np.mean(np.abs(error)) / target_range),
        "dt_rmse_c": float(np.sqrt(np.mean((d_pred - d_true) ** 2))),
        "dt_corr": _corr(d_true, d_pred),
        "residual_corr_heat": _corr(residual, raw_inputs[:, 0]),
        "residual_corr_outdoor": _corr(residual, raw_inputs[:, 1]),
        "residual_corr_solar": _corr(residual, raw_inputs[:, 2]),
        "residual_corr_vent": _corr(residual, raw_inputs[:, 3]),
        "residual_corr_setpoint_deficit": _corr(residual, setpoint_c - target_c),
    }
    return metrics


def impulse_responses(
    params: dict[str, jax.Array],
    *,
    spec: ExperimentSpec,
    horizon: int,
    feature_dim: int,
    target_scale: float,
    schur_gamma: float,
    pf_lambda_min: float,
    schur_mode: SchurMode,
) -> np.ndarray:
    a_matrix = state_matrix(
        params,
        state_dim=spec.state_dim,
        schur_gamma=schur_gamma,
        pf_lambda_min=pf_lambda_min,
        schur_mode=schur_mode,
    )
    b_matrix = params["b_matrix"]
    c_vector = params["c_vector"]
    d_vector = jnp.zeros_like(params["d_vector"]) if spec.zero_d else params["d_vector"]

    responses = []
    for feature_index in range(feature_dim):
        state = jnp.zeros((spec.state_dim,), dtype=jnp.float32)
        pulse = jnp.zeros((feature_dim,), dtype=jnp.float32).at[feature_index].set(1.0)
        feature_response = []
        y0 = jnp.dot(d_vector, pulse)
        feature_response.append(y0)
        state = b_matrix @ pulse
        for _ in range(1, horizon):
            feature_response.append(jnp.dot(c_vector, state))
            state = a_matrix @ state
        responses.append(jnp.stack(feature_response) * target_scale)
    return np.asarray(jnp.stack(responses, axis=1))


def save_experiment_html(
    path: Path,
    *,
    prepared: PreparedProfile,
    result: FitResult,
    impulse: np.ndarray,
) -> None:
    spec = result.spec
    target = prepared.target_c
    pred = result.prediction_c
    residual = result.residual_c
    raw_inputs = prepared.raw_inputs
    datetimes = prepared.datetime
    dt_true = np.diff(target)
    dt_pred = np.diff(pred)
    heat_label = prepared.feature_names[0]
    tset_deficit = prepared.setpoint_c - target

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Full-year in-sample rollout",
            "Residual and heat input",
            "dT/dt: fitted vs simulated",
            "Residual vs heating",
            "Residual vs setpoint deficit",
            "Feature impulse responses",
        ),
        specs=[[{}, {"secondary_y": True}], [{}, {}], [{}, {}]],
        horizontal_spacing=0.08,
        vertical_spacing=0.11,
    )
    fig.add_trace(go.Scattergl(x=datetimes, y=target, mode="lines", name="simulated"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=datetimes, y=pred, mode="lines", name=spec.name), row=1, col=1)
    fig.add_trace(
        go.Scattergl(x=datetimes, y=residual, mode="lines", name="residual", line={"color": "#d62728"}),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scattergl(
            x=datetimes,
            y=raw_inputs[:, 0],
            mode="lines",
            name=heat_label,
            line={"color": "#444", "width": 1},
            opacity=0.35,
        ),
        row=1,
        col=2,
        secondary_y=True,
    )
    min_dt = float(np.nanmin([np.min(dt_true), np.min(dt_pred)]))
    max_dt = float(np.nanmax([np.max(dt_true), np.max(dt_pred)]))
    fig.add_trace(
        go.Scattergl(
            x=dt_true,
            y=dt_pred,
            mode="markers",
            marker={"size": 3, "opacity": 0.35},
            name="dT points",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[min_dt, max_dt], y=[min_dt, max_dt], mode="lines", name="identity", showlegend=False),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=raw_inputs[:, 0],
            y=residual,
            mode="markers",
            marker={"size": 3, "opacity": 0.28, "color": raw_inputs[:, 1], "colorscale": "RdBu"},
            name="residual vs heat",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scattergl(
            x=tset_deficit,
            y=residual,
            mode="markers",
            marker={"size": 3, "opacity": 0.28, "color": raw_inputs[:, 0], "colorscale": "Viridis"},
            name="residual vs setpoint deficit",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    horizon = np.arange(impulse.shape[0]) * 0.25
    for index, name in enumerate(prepared.feature_names):
        fig.add_trace(
            go.Scatter(x=horizon, y=impulse[:, index], mode="lines", name=f"impulse: {name}"),
            row=3,
            col=2,
        )

    metric_text = (
        f"RMSE={result.metrics['rmse_c']:.3f} degC, "
        f"MAE={result.metrics['mae_c']:.3f} degC, "
        f"bias={result.metrics['bias_c']:.3f} degC, "
        f"dT corr={result.metrics['dt_corr']:.3f}, "
        f"rho={result.metrics['rho']:.5f}"
    )
    fig.update_layout(
        title=f"{spec.name}, profile {prepared.profile.profile_id}<br><sup>{metric_text}</sup>",
        template="plotly_white",
        height=1250,
        width=1800,
        hovermode="closest",
    )
    fig.update_yaxes(title_text="Tin [degC]", row=1, col=1)
    fig.update_yaxes(title_text="residual [degC]", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text=heat_label, row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="simulated dT [degC/step]", row=2, col=1)
    fig.update_yaxes(title_text="fitted dT [degC/step]", row=2, col=1)
    fig.update_xaxes(title_text=heat_label, row=2, col=2)
    fig.update_yaxes(title_text="residual [degC]", row=2, col=2)
    fig.update_xaxes(title_text="Tset - Tin simulated [degC]", row=3, col=1)
    fig.update_yaxes(title_text="residual [degC]", row=3, col=1)
    fig.update_xaxes(title_text="time after unit feature pulse [h]", row=3, col=2)
    fig.update_yaxes(title_text="Tin response [degC]", row=3, col=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True)


def fit_experiment(
    spec: ExperimentSpec,
    profile: BuildingProfile,
    *,
    heating_mode: str,
    heat_input_normalization: str,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    gradient_clip_norm: float,
    ridge: float,
    seed: int,
    init_scale: float,
    report_every: int,
    schur_gamma: float,
    pf_lambda_min: float,
    schur_mode: SchurMode,
    impulse_horizon: int,
    heat_on_threshold: float,
    target_alignment: WindowTargetAlignment,
    input_feature_mode: InputFeatureMode,
    heating_regime_window_steps: int,
    fit_start: str | None,
    fit_end: str | None,
) -> tuple[FitResult, PreparedProfile]:
    if spec.feature_mode == "engineered" and not spec.zero_d:
        raise ValueError("engineered feedback features require zero_d=True")
    prepared = prepare_profile(
        profile,
        heating_mode=heating_mode,
        heat_input_normalization=heat_input_normalization,
        feature_mode=spec.feature_mode,
        heat_on_threshold=heat_on_threshold,
        target_alignment=target_alignment,
        input_feature_mode=input_feature_mode,
        heating_regime_window_steps=heating_regime_window_steps,
        fit_start=fit_start,
        fit_end=fit_end,
    )
    raw_inputs = jnp.asarray(prepared.raw_inputs)
    target_norm = jnp.asarray(prepared.target_norm)
    feature_mean = jnp.asarray(prepared.feature_mean)
    feature_scale = jnp.asarray(prepared.feature_scale)
    feature_dim = len(prepared.feature_names)

    params = init_params(
        key=jax.random.PRNGKey(seed),
        state_dim=spec.state_dim,
        feature_dim=feature_dim,
        init_scale=init_scale,
    )
    if spec.zero_d:
        params["d_vector"] = jnp.zeros_like(params["d_vector"])

    optimizer = optax.chain(optax.clip_by_global_norm(gradient_clip_norm), optax.adam(learning_rate))
    opt_state = optimizer.init(params)
    train_step, loss_fn = make_train_step(
        spec=spec,
        optimizer=optimizer,
        ridge=ridge,
        schur_gamma=schur_gamma,
        pf_lambda_min=pf_lambda_min,
        schur_mode=schur_mode,
        target_mean=prepared.target_mean,
        target_scale=prepared.target_scale,
        setpoint_c=prepared.setpoint_c,
        heat_on_threshold=prepared.heat_on_threshold,
    )

    final_loss = float("nan")
    best_loss = float("inf")
    best_epoch = 0
    best_params = params
    for epoch in range(1, epochs + 1):
        params, opt_state, loss = train_step(params, opt_state, raw_inputs, target_norm, feature_mean, feature_scale)
        final_loss = float(loss)
        if not np.isfinite(final_loss):
            raise FloatingPointError(f"{spec.name} produced non-finite loss at epoch {epoch}: {final_loss}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(lambda leaf: leaf.copy(), params)
        if report_every > 0 and (epoch == 1 or epoch % report_every == 0 or epoch == epochs):
            print(
                f"experiment={spec.name} epoch={epoch:04d} "
                f"loss={final_loss:.6f} best_loss={best_loss:.6f} best_epoch={best_epoch}"
            )

    params = best_params
    final_loss = best_loss

    prediction_norm = np.asarray(
        rollout_normalized(
            params,
            raw_inputs,
            spec=spec,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=prepared.target_mean,
            target_scale=prepared.target_scale,
            setpoint_c=prepared.setpoint_c,
            heat_on_threshold=prepared.heat_on_threshold,
            schur_gamma=schur_gamma,
            pf_lambda_min=pf_lambda_min,
            schur_mode=schur_mode,
        )
    )
    prediction_c = prediction_norm * prepared.target_scale + prepared.target_mean
    residual_c = prediction_c - prepared.target_c
    rho = float(spectral_radius(state_matrix(params, state_dim=spec.state_dim, schur_gamma=schur_gamma, pf_lambda_min=pf_lambda_min, schur_mode=schur_mode)))
    metrics = compute_metrics(
        prepared.target_c,
        prediction_c,
        prepared.raw_inputs,
        spec=spec,
        feature_dim=feature_dim,
        rho=rho,
        final_loss=final_loss,
        setpoint_c=prepared.setpoint_c,
        target_alignment=prepared.target_alignment,
        input_feature_mode=prepared.input_feature_mode,
    )
    metrics["selected_epoch"] = best_epoch
    impulse = impulse_responses(
        params,
        spec=spec,
        horizon=impulse_horizon,
        feature_dim=feature_dim,
        target_scale=prepared.target_scale,
        schur_gamma=schur_gamma,
        pf_lambda_min=pf_lambda_min,
        schur_mode=schur_mode,
    )
    html_path = output_dir / f"single_profile_ss_{prepared.profile.profile_id}_{spec.name}.html"
    result = FitResult(
        spec=spec,
        params=params,
        prediction_c=prediction_c,
        residual_c=residual_c,
        metrics=metrics,
        html_path=html_path,
    )
    save_experiment_html(html_path, prepared=prepared, result=result, impulse=impulse)
    return result, prepared


def build_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    suffix = f"{args.output_timing}_{args.target_alignment}_{args.input_feature_mode}"
    if args.experiment_suite == "custom":
        return [
            ExperimentSpec(
                name=f"custom_{args.feature_mode}_{'zeroD' if args.zero_d else 'D'}_n{args.state_dim}_{suffix}",
                state_dim=args.state_dim,
                zero_d=bool(args.zero_d),
                feature_mode=args.feature_mode,
                output_timing=args.output_timing,
            )
        ]

    experiments = [
        ExperimentSpec(
            name=f"01_raw_D_n{args.state_dim}_{suffix}",
            state_dim=args.state_dim,
            zero_d=False,
            feature_mode="raw",
            output_timing=args.output_timing,
        ),
        ExperimentSpec(
            name=f"02_raw_zeroD_n{args.state_dim}_{suffix}",
            state_dim=args.state_dim,
            zero_d=True,
            feature_mode="raw",
            output_timing=args.output_timing,
        ),
    ]
    for dim in args.scan_state_dims:
        if int(dim) == int(args.state_dim):
            continue
        experiments.append(
            ExperimentSpec(
                name=f"03_state_scan_raw_D_n{int(dim)}_{suffix}",
                state_dim=int(dim),
                zero_d=False,
                feature_mode="raw",
                output_timing=args.output_timing,
            )
        )
    experiments.append(
        ExperimentSpec(
            name=f"04_engineered_zeroD_n{args.state_dim}_{suffix}",
            state_dim=args.state_dim,
            zero_d=True,
            feature_mode="engineered",
            output_timing=args.output_timing,
        )
    )
    return experiments


def save_summary_html(
    path: Path,
    *,
    prepared: PreparedProfile,
    results: Sequence[FitResult],
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Full-year fits", "RMSE by experiment"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scattergl(x=prepared.datetime, y=prepared.target_c, mode="lines", name="simulated"),
        row=1,
        col=1,
    )
    for result in results:
        fig.add_trace(
            go.Scattergl(
                x=prepared.datetime,
                y=result.prediction_c,
                mode="lines",
                name=f"{result.spec.name} RMSE={result.metrics['rmse_c']:.2f}",
                opacity=0.82,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=[result.spec.name for result in results],
            y=[result.metrics["rmse_c"] for result in results],
            name="RMSE [degC]",
            marker={"color": "#1f77b4"},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"Single-profile SS summary, profile {prepared.profile.profile_id}",
        template="plotly_white",
        height=900,
        width=1700,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Tin [degC]", row=1, col=1)
    fig.update_yaxes(title_text="RMSE [degC]", row=2, col=1)
    fig.update_xaxes(tickangle=25, row=2, col=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True)


def save_predictions_csv(path: Path, prepared: PreparedProfile, results: Sequence[FitResult]) -> None:
    columns = {
        "datetime": prepared.datetime,
        "simulated_temperature_c": prepared.target_c,
        "heat_input": prepared.raw_inputs[:, 0],
        "outdoor_temperature_c": prepared.raw_inputs[:, 1],
        "solar": prepared.raw_inputs[:, 2],
        "ventilation": prepared.raw_inputs[:, 3],
    }
    for index, name in enumerate(prepared.raw_input_names[4:], start=4):
        columns[f"input_{index}_{name}"] = prepared.raw_inputs[:, index]
    frame = pd.DataFrame(columns)
    for result in results:
        frame[f"{result.spec.name}_prediction_c"] = result.prediction_c
        frame[f"{result.spec.name}_residual_c"] = result.residual_c
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--profile-id", type=int, default=745930)
    parser.add_argument("--heating-mode", choices=(*HEATING_INPUT_COLUMNS.keys(), "A", "B"), default="zone_thermal")
    parser.add_argument(
        "--heat-input-normalization",
        choices=("raw", "per_floor_area"),
        default="per_floor_area",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator_single_profile"))
    parser.add_argument("--experiment-suite", choices=("first4", "custom"), default="first4")
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--scan-state-dims", nargs="+", type=int, default=[1, 2, 8, 12])
    parser.add_argument("--feature-mode", choices=("raw", "engineered"), default="raw")
    parser.add_argument("--fit-start", type=str, default=None, help="Optional inclusive start timestamp for fitting.")
    parser.add_argument("--fit-end", type=str, default=None, help="Optional inclusive end timestamp for fitting.")
    parser.add_argument(
        "--input-feature-mode",
        choices=("base", "heating_regime"),
        default="base",
        help="Append heat-on and recent-heat-on regime indicators to the raw input vector.",
    )
    parser.add_argument(
        "--heating-regime-window-steps",
        type=int,
        default=96 * 7,
        help="Rolling window used for the recent-heat-on input feature.",
    )
    parser.add_argument(
        "--target-alignment",
        choices=("same_time", "next_step"),
        default="same_time",
        help="Use same-time targets or predict T[t+1] from u[t].",
    )
    parser.add_argument(
        "--output-timing",
        choices=("pre_update", "post_update"),
        default="pre_update",
        help=(
            "Whether y[t] is emitted before or after applying u[t] to the state. "
            "EnergyPlus end-of-interval timestamps may be better represented by post_update."
        ),
    )
    parser.add_argument("--zero-d", action="store_true")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-scale", type=float, default=1e-2)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--schur-gamma", type=float, default=0.999)
    parser.add_argument("--pf-lambda-min", type=float, default=0.0)
    parser.add_argument("--schur-mode", choices=("dense", "near_identity", "pf"), default="near_identity")
    parser.add_argument("--impulse-horizon", type=int, default=96)
    parser.add_argument(
        "--heat-on-threshold",
        type=float,
        default=1e-6,
        help="Threshold, in the selected heat input unit, for the engineered heat_on feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = normalize_heating_mode(args.heating_mode)
    if args.state_dim < 1:
        raise ValueError("--state-dim must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.heating_regime_window_steps < 1:
        raise ValueError("--heating-regime-window-steps must be positive")
    if args.experiment_suite == "custom" and args.feature_mode == "engineered" and not args.zero_d:
        raise ValueError("--feature-mode engineered requires --zero-d")

    profile = load_single_profile(
        args.dataset,
        profile_id=args.profile_id,
        heating_mode=mode,
        heat_input_normalization=args.heat_input_normalization,
        input_feature_mode=args.input_feature_mode,
        heating_regime_window_steps=args.heating_regime_window_steps,
        heat_on_threshold=args.heat_on_threshold,
    )
    output_dir = args.output_dir / f"profile_{profile.profile_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"profile_id={profile.profile_id} steps={profile.target.shape[0]} "
        f"heating_mode={mode} heat_input_normalization={args.heat_input_normalization} "
        f"target_alignment={args.target_alignment} output_timing={args.output_timing} "
        f"input_feature_mode={args.input_feature_mode}"
    )
    print(
        "heating_regime_window_steps="
        f"{args.heating_regime_window_steps} heat_on_threshold={args.heat_on_threshold}"
    )
    if args.fit_start is not None or args.fit_end is not None:
        print(f"fit_period_start={args.fit_start} fit_period_end={args.fit_end}")
    print(
        "input_columns="
        f"{input_columns(mode, args.heat_input_normalization, args.input_feature_mode, args.heating_regime_window_steps)}"
    )
    print(f"disturbance_columns={DISTURBANCE_COLUMNS}")
    experiments = build_experiments(args)
    print(f"experiments={[experiment.name for experiment in experiments]}")

    results: list[FitResult] = []
    prepared_for_summary: PreparedProfile | None = None
    for experiment_index, spec in enumerate(experiments):
        print(f"starting_experiment={spec.name}")
        result, prepared = fit_experiment(
            spec,
            profile,
            heating_mode=mode,
            heat_input_normalization=args.heat_input_normalization,
            output_dir=output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_clip_norm=args.gradient_clip_norm,
            ridge=args.ridge,
            seed=args.seed + 1000 * experiment_index,
            init_scale=args.init_scale,
            report_every=args.report_every,
            schur_gamma=args.schur_gamma,
            pf_lambda_min=args.pf_lambda_min,
            schur_mode=args.schur_mode,
            impulse_horizon=args.impulse_horizon,
            heat_on_threshold=args.heat_on_threshold,
            target_alignment=args.target_alignment,
            input_feature_mode=args.input_feature_mode,
            heating_regime_window_steps=args.heating_regime_window_steps,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
        )
        results.append(result)
        prepared_for_summary = prepared_for_summary or prepared
        print(
            f"finished_experiment={spec.name} "
            f"rmse_c={result.metrics['rmse_c']:.4f} "
            f"mae_c={result.metrics['mae_c']:.4f} "
            f"bias_c={result.metrics['bias_c']:.4f} "
            f"dt_corr={result.metrics['dt_corr']:.4f} "
            f"rho={result.metrics['rho']:.5f} "
            f"html={result.html_path}"
        )

    assert prepared_for_summary is not None
    summary = pd.DataFrame([result.metrics for result in results]).sort_values("rmse_c")
    summary_path = output_dir / f"single_profile_ss_{profile.profile_id}_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary_html = output_dir / f"single_profile_ss_{profile.profile_id}_summary.html"
    save_summary_html(summary_html, prepared=prepared_for_summary, results=results)
    predictions_path = output_dir / f"single_profile_ss_{profile.profile_id}_predictions.csv"
    save_predictions_csv(predictions_path, prepared_for_summary, results)
    print(f"saved_summary_csv={summary_path}")
    print(f"saved_summary_html={summary_html}")
    print(f"saved_predictions_csv={predictions_path}")
    print("best_experiments_by_rmse=")
    print(summary.loc[:, ["experiment", "rmse_c", "mae_c", "bias_c", "dt_corr", "rho"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
