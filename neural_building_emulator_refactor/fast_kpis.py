"""Cheap, interpretable diagnostics for autonomous emulator trajectories."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch
from scipy.stats import wasserstein_distance

from neural_building_emulator.data import ClosedLoopProfile

from .artifacts import load_artifact
from .prediction import predict_profile
from .profiles import ProfileSource, load_profiles


CHANNEL_NAMES = ("temperature", "qroom", "pel")


@dataclass(frozen=True)
class FastKpiConfig:
    dt_hours: float = 0.25
    event_horizon_hours: float = 3.0
    min_setpoint_change_c: float = 0.05
    acf_lags: tuple[int, ...] = (1, 2, 4, 8, 12, 24, 48, 96)
    relative_power_floor_w_m2: float = 0.1

    @property
    def event_horizon_steps(self) -> int:
        return max(1, int(round(self.event_horizon_hours / self.dt_hours)))


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _acf(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    centered = values - np.mean(values) if len(values) else values
    variance = float(np.mean(centered**2)) if len(centered) else 0.0
    result = np.full(len(lags), np.nan, dtype=np.float64)
    if variance < 1e-12:
        return result
    for index, lag in enumerate(lags):
        if 0 < lag < len(centered):
            result[index] = float(np.mean(centered[lag:] * centered[:-lag]) / variance)
    return result


def acf_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    lags: Sequence[int],
) -> float:
    """Mean absolute ACF difference, with a unit penalty for one-sided constants."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_constant = np.nanstd(prediction) < 1e-12
    target_constant = np.nanstd(target) < 1e-12
    if pred_constant and target_constant:
        return 0.0
    if pred_constant != target_constant:
        return 1.0
    pred_acf = _acf(prediction, lags)
    target_acf = _acf(target, lags)
    valid = np.isfinite(pred_acf) & np.isfinite(target_acf)
    return float(np.mean(np.abs(pred_acf[valid] - target_acf[valid]))) if np.any(valid) else 0.0


def _normalized_increment_psd(values: np.ndarray, dt_hours: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    increments = np.diff(values)
    if len(increments) < 4 or np.std(increments) < 1e-12:
        return np.empty(0), np.empty(0)
    segment = min(len(increments), 96 * 14)
    frequencies, density = welch(
        increments,
        fs=1.0 / dt_hours,
        nperseg=segment,
        detrend="constant",
        scaling="spectrum",
    )
    keep = frequencies > 0.0
    frequencies = frequencies[keep]
    density = np.maximum(density[keep], 0.0)
    total = float(np.sum(density))
    if total < 1e-20:
        return np.empty(0), np.empty(0)
    return frequencies, density / total


def increment_spectral_js_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    dt_hours: float,
) -> float:
    """Normalized Jensen-Shannon distance between first-difference spectra."""
    pred_frequency, pred_psd = _normalized_increment_psd(prediction, dt_hours)
    target_frequency, target_psd = _normalized_increment_psd(target, dt_hours)
    if not len(pred_psd) and not len(target_psd):
        return 0.0
    if not len(pred_psd) or not len(target_psd):
        return 1.0
    if len(pred_psd) != len(target_psd) or not np.allclose(pred_frequency, target_frequency):
        pred_psd = np.interp(target_frequency, pred_frequency, pred_psd, left=0.0, right=0.0)
        pred_psd = pred_psd / max(float(np.sum(pred_psd)), 1e-20)
    epsilon = 1e-15
    pred_psd = np.maximum(pred_psd, epsilon)
    target_psd = np.maximum(target_psd, epsilon)
    pred_psd /= np.sum(pred_psd)
    target_psd /= np.sum(target_psd)
    midpoint = 0.5 * (pred_psd + target_psd)
    divergence = 0.5 * np.sum(pred_psd * np.log(pred_psd / midpoint))
    divergence += 0.5 * np.sum(target_psd * np.log(target_psd / midpoint))
    return float(np.sqrt(max(divergence, 0.0) / np.log(2.0)))


def _increment_spectral_bands(values: np.ndarray, dt_hours: float) -> dict[str, float]:
    frequencies, density = _normalized_increment_psd(values, dt_hours)
    if not len(density):
        return {name: 0.0 for name in ("fast", "intraday", "daily", "slow")}
    masks = {
        "fast": frequencies >= 1.0,
        "intraday": (frequencies >= 1.0 / 6.0) & (frequencies < 1.0),
        "daily": (frequencies >= 1.0 / 36.0) & (frequencies < 1.0 / 6.0),
        "slow": frequencies < 1.0 / 36.0,
    }
    return {name: float(np.sum(density[mask])) for name, mask in masks.items()}


def trajectory_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    config: FastKpiConfig,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have the same [time, channel] shape")
    scale = np.asarray(target_scale, dtype=np.float64).reshape(-1)[: target.shape[1]]
    scale = np.maximum(scale, 1e-6)
    error = prediction - target
    result = {
        "total_nrmse": float(np.sqrt(np.mean((error / scale[None, :]) ** 2))),
    }
    for index in range(target.shape[1]):
        name = CHANNEL_NAMES[index]
        channel_error = error[:, index]
        target_std = max(float(np.std(target[:, index])), 1e-6)
        result.update(
            {
                f"{name}_rmse": float(np.sqrt(np.mean(channel_error**2))),
                f"{name}_nrmse": float(np.sqrt(np.mean(channel_error**2)) / scale[index]),
                f"{name}_profile_std_nrmse": float(
                    np.sqrt(np.mean(channel_error**2)) / target_std
                ),
                f"{name}_mae": float(np.mean(np.abs(channel_error))),
                f"{name}_bias": float(np.mean(channel_error)),
                f"{name}_wasserstein_norm": float(
                    wasserstein_distance(target[:, index], prediction[:, index]) / scale[index]
                ),
                f"{name}_acf_mae": acf_distance(
                    prediction[:, index], target[:, index], config.acf_lags
                ),
                f"{name}_increment_spectral_js": increment_spectral_js_distance(
                    prediction[:, index], target[:, index], config.dt_hours
                ),
            }
        )
    return result


def _clean_setpoint_events(
    setpoint: np.ndarray,
    horizon_steps: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    setpoint = np.asarray(setpoint, dtype=np.float64)
    candidates = np.flatnonzero(np.abs(np.diff(setpoint)) >= threshold) + 1
    valid: list[int] = []
    for event in candidates:
        if event < horizon_steps or event + horizon_steps > len(setpoint):
            continue
        pre = setpoint[event - horizon_steps : event]
        post = setpoint[event : event + horizon_steps]
        if np.max(np.abs(np.diff(pre)), initial=0.0) >= threshold:
            continue
        if np.max(np.abs(np.diff(post)), initial=0.0) >= threshold:
            continue
        valid.append(int(event))
    events = np.asarray(valid, dtype=np.int64)
    deltas = setpoint[events] - setpoint[events - 1] if len(events) else np.empty(0)
    return events, deltas


def _window_response(power: np.ndarray, events: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    power = np.asarray(power, dtype=np.float64)
    pre = np.asarray([np.mean(power[event - steps : event]) for event in events])
    post = np.asarray([np.mean(power[event : event + steps]) for event in events])
    return pre, post, post - pre


def _directional_event_summary(
    prefix: str,
    deltas: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    response: np.ndarray,
    horizon_hours: float,
    relative_floor: float,
) -> dict[str, float]:
    if not len(response):
        return {
            f"{prefix}_event_count": 0.0,
            f"{prefix}_power_gain_w_m2_k": float("nan"),
            f"{prefix}_energy_gain_wh_m2_k": float("nan"),
            f"{prefix}_correct_sign_fraction": float("nan"),
            f"{prefix}_symmetric_relative_power_change": float("nan"),
        }
    gain = response / deltas
    symmetric_relative = 2.0 * response / (
        np.abs(post) + np.abs(pre) + relative_floor
    )
    power_gain = float(np.mean(gain))
    return {
        f"{prefix}_event_count": float(len(response)),
        f"{prefix}_power_gain_w_m2_k": power_gain,
        f"{prefix}_energy_gain_wh_m2_k": horizon_hours * power_gain,
        f"{prefix}_correct_sign_fraction": float(np.mean(response * deltas >= 0.0)),
        f"{prefix}_symmetric_relative_power_change": float(np.mean(symmetric_relative)),
    }


def raw_flexibility_metrics(
    profile: ClosedLoopProfile,
    prediction: np.ndarray,
    target: np.ndarray,
    pel_scale: float,
    config: FastKpiConfig,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compare paired pre/post HP-power responses without fitting a regression."""
    steps = config.event_horizon_steps
    events, setpoint_delta = _clean_setpoint_events(
        profile.inputs[:, 0], steps, config.min_setpoint_change_c
    )
    true_pre, true_post, true_response = _window_response(target[:, 2], events, steps)
    pred_pre, pred_post, pred_response = _window_response(prediction[:, 2], events, steps)
    event_error = pred_response - true_response
    metrics: dict[str, float] = {
        "flex_event_count": float(len(events)),
        "flex_event_delta_p_rmse_w_m2": (
            float(np.sqrt(np.mean(event_error**2))) if len(events) else float("nan")
        ),
        "flex_event_delta_p_nrmse": (
            float(np.sqrt(np.mean(event_error**2)) / max(float(pel_scale), 1e-6))
            if len(events)
            else float("nan")
        ),
        "flex_event_delta_p_mae_w_m2": (
            float(np.mean(np.abs(event_error))) if len(events) else float("nan")
        ),
        "flex_event_delta_p_correlation": _safe_correlation(true_response, pred_response),
        "flex_event_response_sign_agreement": (
            float(np.mean(np.sign(true_response) == np.sign(pred_response)))
            if len(events)
            else float("nan")
        ),
    }
    event_rows: list[dict[str, float]] = []
    for index, event in enumerate(events):
        event_rows.append(
            {
                "profile_id": float(profile.profile_id),
                "event_step": float(event),
                "delta_tset_c": float(setpoint_delta[index]),
                "sim_pre_pel_w_m2": float(true_pre[index]),
                "sim_post_pel_w_m2": float(true_post[index]),
                "sim_delta_pel_w_m2": float(true_response[index]),
                "emu_pre_pel_w_m2": float(pred_pre[index]),
                "emu_post_pel_w_m2": float(pred_post[index]),
                "emu_delta_pel_w_m2": float(pred_response[index]),
            }
        )
    for direction, mask in (
        ("up", setpoint_delta > 0.0),
        ("down", setpoint_delta < 0.0),
    ):
        simulated = _directional_event_summary(
            f"sim_{direction}",
            setpoint_delta[mask],
            true_pre[mask],
            true_post[mask],
            true_response[mask],
            config.event_horizon_hours,
            config.relative_power_floor_w_m2,
        )
        emulated = _directional_event_summary(
            f"emu_{direction}",
            setpoint_delta[mask],
            pred_pre[mask],
            pred_post[mask],
            pred_response[mask],
            config.event_horizon_hours,
            config.relative_power_floor_w_m2,
        )
        metrics.update(simulated)
        metrics.update(emulated)
        sim_gain = simulated[f"sim_{direction}_energy_gain_wh_m2_k"]
        emu_gain = emulated[f"emu_{direction}_energy_gain_wh_m2_k"]
        metrics[f"flex_{direction}_energy_gain_abs_error_wh_m2_k"] = abs(emu_gain - sim_gain)
    return metrics, event_rows


def profile_characteristics(
    profile_id: int,
    target: np.ndarray,
    config: FastKpiConfig,
    flex_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    result: dict[str, float] = {"profile_id": float(profile_id)}
    for index in range(target.shape[1]):
        name = CHANNEL_NAMES[index]
        values = target[:, index]
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
        result[f"{name}_q90_range"] = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        acf_values = _acf(values, (1, 4, 12, 96))
        for lag, value in zip((1, 4, 12, 96), acf_values):
            result[f"{name}_acf_{lag}"] = _safe_float(value)
        for band, fraction in _increment_spectral_bands(values, config.dt_hours).items():
            result[f"{name}_increment_band_{band}"] = fraction
    if target.shape[1] >= 3:
        result["pel_active_fraction"] = float(np.mean(target[:, 2] > 0.1))
        result["qroom_active_fraction"] = float(np.mean(target[:, 1] > 0.1))
    if flex_metrics:
        for name in (
            "sim_up_energy_gain_wh_m2_k",
            "sim_down_energy_gain_wh_m2_k",
            "sim_up_correct_sign_fraction",
            "sim_down_correct_sign_fraction",
            "sim_up_symmetric_relative_power_change",
            "sim_down_symmetric_relative_power_change",
        ):
            result[name] = flex_metrics.get(name, float("nan"))
    return result


def _numeric_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column == "profile_id":
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values):
            summary[column] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
            }
    return summary


def _write_quality_dashboard(path: Path, frame: pd.DataFrame, *, closed_loop: bool) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Normalized trajectory error",
            "Temporal-structure distance",
            "Marginal-distribution distance",
            "Raw 3 h flexibility gain",
        ),
    )
    channels = CHANNEL_NAMES if closed_loop else CHANNEL_NAMES[:1]
    for channel in channels:
        figure.add_trace(
            go.Box(y=frame[f"{channel}_nrmse"], name=channel, boxmean=True), row=1, col=1
        )
        figure.add_trace(
            go.Box(y=frame[f"{channel}_acf_mae"], name=f"{channel} ACF", boxmean=True),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Box(
                y=frame[f"{channel}_increment_spectral_js"],
                name=f"{channel} spectrum",
                boxmean=True,
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Box(
                y=frame[f"{channel}_wasserstein_norm"], name=channel, boxmean=True
            ),
            row=2,
            col=1,
        )
    if closed_loop:
        for direction, color in (("up", "#d32f2f"), ("down", "#1769aa")):
            figure.add_trace(
                go.Scatter(
                    x=frame[f"sim_{direction}_energy_gain_wh_m2_k"],
                    y=frame[f"emu_{direction}_energy_gain_wh_m2_k"],
                    mode="markers",
                    name=direction,
                    marker={"color": color, "opacity": 0.7},
                    text=frame["profile_id"].astype(str),
                ),
                row=2,
                col=2,
            )
    figure.update_yaxes(title_text="RMSE / training scale", row=1, col=1)
    figure.update_yaxes(title_text="distance (0 is ideal)", row=1, col=2)
    figure.update_yaxes(title_text="Wasserstein / training scale", row=2, col=1)
    figure.update_xaxes(title_text="simulated Wh/(m2 K)", row=2, col=2)
    figure.update_yaxes(title_text="emulated Wh/(m2 K)", row=2, col=2)
    figure.update_layout(
        title="Fast ablation scorecard",
        template="plotly_white",
        height=900,
        width=1450,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    figure.write_html(path)


def _write_characteristic_heatmap(path: Path, frame: pd.DataFrame) -> None:
    candidates = [
        "temperature_std",
        "temperature_acf_96",
        "temperature_increment_band_fast",
        "qroom_std",
        "qroom_active_fraction",
        "qroom_increment_band_fast",
        "pel_std",
        "pel_active_fraction",
        "pel_increment_band_fast",
        "sim_up_energy_gain_wh_m2_k",
        "sim_down_energy_gain_wh_m2_k",
    ]
    columns = [column for column in candidates if column in frame and frame[column].notna().sum() >= 3]
    if len(columns) < 2:
        return
    correlation = frame[columns].corr(method="spearman")
    figure = go.Figure(
        go.Heatmap(
            z=correlation.to_numpy(),
            x=columns,
            y=columns,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "Spearman"},
        )
    )
    figure.update_layout(
        title="Do the profile descriptors measure distinct behavior?",
        template="plotly_white",
        height=900,
        width=1200,
        margin={"l": 260, "b": 240, "r": 40, "t": 90},
    )
    figure.write_html(path)


def _write_event_response_plot(path: Path, frame: pd.DataFrame, seed: int) -> None:
    if frame.empty:
        return
    if len(frame) > 20000:
        frame = frame.sample(20000, random_state=seed)
    colors = np.where(frame["delta_tset_c"] > 0.0, "#d32f2f", "#1769aa")
    values = np.concatenate(
        [frame["sim_delta_pel_w_m2"].to_numpy(), frame["emu_delta_pel_w_m2"].to_numpy()]
    )
    finite = values[np.isfinite(values)]
    low, high = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (-1.0, 1.0)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=frame["sim_delta_pel_w_m2"],
            y=frame["emu_delta_pel_w_m2"],
            mode="markers",
            marker={"color": colors, "opacity": 0.35, "size": 5},
            text=frame["profile_id"].astype(int).astype(str),
            name="setpoint events",
        )
    )
    figure.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines", line={"dash": "dash", "color": "black"}, name="ideal")
    )
    figure.update_layout(
        title="Paired-window HP response at thermostat interventions",
        template="plotly_white",
        height=750,
        width=1000,
        xaxis_title="Simulated post-minus-pre Pel [W/m2]",
        yaxis_title="Emulated post-minus-pre Pel [W/m2]",
    )
    figure.write_html(path)


def evaluate_fast_kpis(
    artifact_dir: Path,
    *,
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    profile_source: ProfileSource = "test",
    profile_ids: Sequence[int] | None = None,
    max_profiles: int | None = 100,
    num_eval_particles: int = 4,
    target_scale_override: np.ndarray | None = None,
    config: FastKpiConfig = FastKpiConfig(),
    seed: int = 13,
) -> Path:
    """Evaluate mean trajectories only; no scenario KPI regressions are fitted."""
    started_at = time.perf_counter()
    artifact = load_artifact(artifact_dir)
    profiles = load_profiles(
        artifact,
        dataset_path=dataset_path,
        source=profile_source,
        max_profiles=max_profiles,
        profile_ids=profile_ids,
    )
    destination = Path(output_dir or artifact.artifact_dir / "fast_kpis")
    destination.mkdir(parents=True, exist_ok=True)
    target_scale = np.asarray(
        artifact.scalers.target.scale
        if target_scale_override is None
        else target_scale_override,
        dtype=np.float64,
    )
    quality_rows: list[dict[str, float]] = []
    characteristic_rows: list[dict[str, float]] = []
    event_rows: list[dict[str, float]] = []
    key = jax.random.PRNGKey(seed)
    for profile_index, profile in enumerate(profiles, start=1):
        key, profile_key = jax.random.split(key)
        result = predict_profile(
            artifact,
            profile,
            key=profile_key,
            num_eval_particles=num_eval_particles,
        )
        row: dict[str, float] = {"profile_id": float(profile.profile_id)}
        row.update(trajectory_metrics(result.mean, result.target, target_scale, config))
        flex: dict[str, float] | None = None
        if artifact.spec.task == "closed_loop_hp":
            assert isinstance(profile, ClosedLoopProfile)
            flex, events = raw_flexibility_metrics(
                profile,
                result.mean,
                result.target,
                target_scale[2],
                config,
            )
            row.update(flex)
            event_rows.extend(events)
        quality_rows.append(row)
        characteristic_rows.append(
            profile_characteristics(profile.profile_id, result.target, config, flex)
        )
        print(f"fast_kpi_profile={profile_index}/{len(profiles)} profile_id={profile.profile_id}")

    quality = pd.DataFrame(quality_rows)
    characteristics = pd.DataFrame(characteristic_rows)
    events = pd.DataFrame(event_rows)
    quality.to_csv(destination / "fast_profile_kpis.csv", index=False)
    characteristics.to_csv(destination / "simulated_profile_characteristics.csv", index=False)
    if not events.empty:
        events.to_csv(destination / "raw_flexibility_events.csv", index=False)
    _write_quality_dashboard(
        destination / "fast_kpi_dashboard.html",
        quality,
        closed_loop=artifact.spec.task == "closed_loop_hp",
    )
    _write_characteristic_heatmap(
        destination / "simulated_profile_characteristics.html", characteristics
    )
    _write_event_response_plot(destination / "raw_flexibility_event_response.html", events, seed)
    summary = {
        "model_name": artifact.spec.name,
        "task": artifact.spec.task,
        "profile_source": "explicit" if profile_ids is not None else profile_source,
        "profile_count": len(profiles),
        "num_eval_particles": num_eval_particles if artifact.spec.probabilistic else 0,
        "normalization_scale": target_scale.tolist(),
        "normalization_source": (
            "artifact_training_scaler"
            if target_scale_override is None
            else "fixed_holdout"
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
        "configuration": {
            "dt_hours": config.dt_hours,
            "event_horizon_hours": config.event_horizon_hours,
            "min_setpoint_change_c": config.min_setpoint_change_c,
            "acf_lags": list(config.acf_lags),
        },
        "metrics": _numeric_summary(quality),
    }
    (destination / "fast_kpi_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    headline = summary["metrics"].get("total_nrmse", {})
    print(f"saved_fast_kpis={destination}")
    if headline:
        print(
            f"total_nrmse_mean={headline['mean']:.4f} "
            f"total_nrmse_median={headline['median']:.4f}"
        )
    print(f"fast_kpi_elapsed_seconds={summary['elapsed_seconds']:.2f}")
    return destination
