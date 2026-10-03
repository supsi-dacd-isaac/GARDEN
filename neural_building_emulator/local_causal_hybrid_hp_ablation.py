"""Building-local ablation for the causal semi-Markov HP emulator.

This script is deliberately separate from ``train.py``.  Each building gets a
single best-fit parameter set, the first half-year is used for fitting, and the
second half-year is one autonomous rollout initialized from observed pre-split
history.  This tests the model class without metadata conditioning.

Example:

    .venv/bin/python -m neural_building_emulator.local_causal_hybrid_hp_ablation \
        --dataset neural_building_emulator/tessin_results.parquet/all_hp \
        --num-profiles 3 --epochs 6
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import DEFAULT_DATASET_PATH, ClosedLoopProfile
from .flexibility_event_study_kpis import EventStudyConfig, estimate_trace_flexibility
from .local_closed_loop_hp_ablation import load_profiles
from .models.causal_hybrid_hp import CausalHybridHPEmulator, CausalHybridRollout


@dataclass(frozen=True)
class HybridVariant:
    name: str
    history_hours: float = 48.0
    use_history_encoder: bool = True
    use_dwell_age: bool = True
    hydronic_mode: str = "convolution"
    rc_nodes: int = 3
    use_event_losses: bool = True


DEFAULT_VARIANTS = (
    HybridVariant("full_48h"),
    HybridVariant("history_24h", history_hours=24.0),
    HybridVariant("history_72h", history_hours=72.0),
    HybridVariant("no_state_inference", use_history_encoder=False),
    HybridVariant("markov_no_dwell_age", use_dwell_age=False),
    HybridVariant("instant_hydronics", hydronic_mode="instant"),
    HybridVariant("one_node_rc", rc_nodes=1),
    HybridVariant("no_event_heads", use_event_losses=False),
)

LOSS_NAMES = (
    "temperature_mse",
    "qroom_mse",
    "semi_markov_nll",
    "duration_nll",
    "interval_energy_nll",
    "event_power_nll",
    "causal_tau_nll",
)


@dataclass(frozen=True)
class ProfileData:
    profile: ClosedLoopProfile
    split: int
    mode: np.ndarray
    age: np.ndarray
    switch: np.ndarray
    interval_start: np.ndarray
    duration: np.ndarray
    interval_energy: np.ndarray
    event_power: np.ndarray
    tau_power: np.ndarray
    tau_temperature: np.ndarray
    event_mask: np.ndarray
    observation_mean: np.ndarray
    observation_scale: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray
    temperature_mean: float
    temperature_scale: float
    qroom_scale: float
    pel_scale: float
    interval_energy_scale: float
    tau_power_scale: float
    tau_temperature_scale: float


class WindowArrays(eqx.Module):
    inputs: jax.Array
    initial_temperature: jax.Array
    history_summary: jax.Array
    initial_mode_distribution: jax.Array
    observed_mode: jax.Array
    observed_age: jax.Array
    observed_switch: jax.Array
    interval_start: jax.Array
    duration: jax.Array
    interval_energy: jax.Array
    event_power: jax.Array
    tau_power: jax.Array
    tau_temperature: jax.Array
    event_mask: jax.Array
    targets: jax.Array


def _safe_scale(values: np.ndarray) -> float:
    value = float(np.nanstd(values))
    return value if np.isfinite(value) and value > 1e-5 else 1.0


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3 or np.std(left[mask]) < 1e-10 or np.std(right[mask]) < 1e-10:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def trace_event_response_metrics(
    inputs: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    horizon_steps: int,
    threshold_c: float,
) -> dict[str, float]:
    """Match the legacy local baseline's pre/post event-response metric."""
    setpoint = inputs[:, 0]
    delta = np.diff(setpoint, prepend=setpoint[0])
    events = np.flatnonzero(np.abs(delta) >= threshold_c)
    events = events[
        (events >= horizon_steps) & (events + horizon_steps <= target.shape[0])
    ]
    result: dict[str, float] = {"test_event_count": float(events.size)}
    for channel, name in ((1, "qroom"), (2, "pel")):
        if events.size == 0:
            result[f"test_event_{name}_response_rmse_w_m2"] = float("nan")
            result[f"test_event_{name}_response_bias_w_m2"] = float("nan")
            result[f"test_event_{name}_response_corr"] = float("nan")
            continue
        true_response = np.asarray(
            [
                np.mean(target[index:index + horizon_steps, channel])
                - np.mean(target[index - horizon_steps:index, channel])
                for index in events
            ]
        )
        predicted_response = np.asarray(
            [
                np.mean(prediction[index:index + horizon_steps, channel])
                - np.mean(prediction[index - horizon_steps:index, channel])
                for index in events
            ]
        )
        result[f"test_event_{name}_response_rmse_w_m2"] = _rmse(
            predicted_response, true_response
        )
        result[f"test_event_{name}_response_bias_w_m2"] = float(
            np.mean(predicted_response - true_response)
        )
        result[f"test_event_{name}_response_corr"] = _corr(
            predicted_response, true_response
        )
    return result


def observed_state_features(profile: ClosedLoopProfile) -> np.ndarray:
    """Observable history ending before an autonomous rollout starts."""
    return np.column_stack(
        [
            profile.initial_temperature[:, 0],
            profile.inputs[:, 0],
            profile.targets[:, 2],
            profile.targets[:, 1],
            profile.inputs[:, 1],
            profile.inputs[:, 2],
            profile.inputs[:, 3],
            profile.inputs[:, 4],
        ]
    ).astype(np.float32)


def history_summary(
    observations: np.ndarray,
    *,
    end: int,
    history_steps: int,
    mean: np.ndarray,
    scale: np.ndarray,
    dt_hours: float,
) -> np.ndarray:
    start = max(0, end - history_steps)
    values = (observations[start:end] - mean) / scale
    if values.shape[0] == 0:
        values = np.zeros((1, observations.shape[1]), dtype=np.float32)
    pools: list[np.ndarray] = [values[-1]]
    for hours in (1.0, 6.0, 24.0, history_steps * dt_hours):
        count = max(1, min(values.shape[0], int(round(hours / dt_hours))))
        recent = values[-count:]
        pools.extend([np.mean(recent, axis=0), np.std(recent, axis=0), recent[-1] - recent[0]])
    return np.concatenate(pools).astype(np.float32)


def dwell_age(mode: np.ndarray) -> np.ndarray:
    age = np.zeros(mode.shape, dtype=np.int32)
    for index in range(1, mode.size):
        age[index] = age[index - 1] + 1 if mode[index] == mode[index - 1] else 0
    return age


def make_event_targets(
    profile: ClosedLoopProfile,
    horizon_steps: tuple[int, ...],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = profile.targets.shape[0]
    count = len(horizon_steps)
    event_power = np.zeros((length, count), dtype=np.float32)
    tau_power = np.zeros_like(event_power)
    tau_temperature = np.zeros_like(event_power)
    mask = np.zeros_like(event_power, dtype=bool)
    setpoint = profile.inputs[:, 0]
    delta = np.diff(setpoint, prepend=setpoint[0])
    temperature_current = profile.initial_temperature[:, 0]
    pel = profile.targets[:, 2]
    for index in np.flatnonzero(np.abs(delta) >= threshold):
        for horizon_index, steps in enumerate(horizon_steps):
            if index - steps < 0 or index + steps > length:
                continue
            pre_power = float(np.mean(pel[index - steps:index]))
            post_power = float(np.mean(pel[index:index + steps]))
            pre_temperature = float(np.mean(temperature_current[index - steps:index]))
            post_temperature = float(np.mean(profile.targets[index:index + steps, 0]))
            event_power[index, horizon_index] = post_power
            tau_power[index, horizon_index] = (post_power - pre_power) / float(delta[index])
            tau_temperature[index, horizon_index] = (
                post_temperature - pre_temperature
            ) / float(delta[index])
            mask[index, horizon_index] = True
    return event_power, tau_power, tau_temperature, mask


def prepare_profile(profile: ClosedLoopProfile, args: argparse.Namespace) -> ProfileData:
    length = profile.targets.shape[0]
    split = int(round(length * args.train_fraction))
    split -= split % args.control_interval_steps
    if split < args.sequence_length or length - split < args.sequence_length:
        raise ValueError(f"Profile {profile.profile_id} is too short for the requested split")
    mode = (profile.targets[:, 2] > args.heat_on_threshold_w_m2).astype(np.int32)
    age = dwell_age(mode)
    switch = np.zeros_like(mode, dtype=np.float32)
    switch[:-1] = (mode[1:] != mode[:-1]).astype(np.float32)
    interval_start = (np.arange(length) % args.control_interval_steps == 0)
    duration = np.zeros((length,), dtype=np.int32)
    interval_energy = np.zeros((length,), dtype=np.float32)
    for start in np.flatnonzero(interval_start):
        end = min(start + args.control_interval_steps, length)
        duration[start] = int(np.sum(mode[start:end]))
        interval_energy[start] = float(np.sum(profile.targets[start:end, 2]) * args.dt_hours)
    event_power, tau_power, tau_temperature, event_mask = make_event_targets(
        profile,
        args.horizon_steps,
        args.setpoint_change_threshold_c,
    )
    observations = observed_state_features(profile)
    observation_mean = np.mean(observations[:split], axis=0)
    observation_scale = np.std(observations[:split], axis=0)
    observation_scale = np.where(observation_scale > 1e-5, observation_scale, 1.0)
    input_mean = np.mean(profile.inputs[:split], axis=0)
    input_scale = np.std(profile.inputs[:split], axis=0)
    input_scale = np.where(input_scale > 1e-5, input_scale, 1.0)
    train_event_mask = event_mask[:split]
    return ProfileData(
        profile=profile,
        split=split,
        mode=mode,
        age=age,
        switch=switch,
        interval_start=interval_start,
        duration=duration,
        interval_energy=interval_energy,
        event_power=event_power,
        tau_power=tau_power,
        tau_temperature=tau_temperature,
        event_mask=event_mask,
        observation_mean=observation_mean.astype(np.float32),
        observation_scale=observation_scale.astype(np.float32),
        input_mean=input_mean.astype(np.float32),
        input_scale=input_scale.astype(np.float32),
        temperature_mean=float(np.mean(profile.initial_temperature[:split, 0])),
        temperature_scale=_safe_scale(profile.initial_temperature[:split, 0]),
        qroom_scale=_safe_scale(profile.targets[:split, 1]),
        pel_scale=_safe_scale(profile.targets[:split, 2]),
        interval_energy_scale=_safe_scale(interval_energy[:split][interval_start[:split]]),
        tau_power_scale=_safe_scale(tau_power[:split][train_event_mask]),
        tau_temperature_scale=_safe_scale(tau_temperature[:split][train_event_mask]),
    )


def initial_mode_distribution(
    data: ProfileData,
    start: int,
    *,
    max_dwell_steps: int,
    use_history: bool,
) -> np.ndarray:
    distribution = np.zeros((2, max_dwell_steps + 1), dtype=np.float32)
    if use_history and start > 0:
        distribution[data.mode[start - 1], min(data.age[start - 1], max_dwell_steps)] = 1.0
    else:
        distribution[0, 0] = 1.0
    return distribution


def make_windows(
    data: ProfileData,
    variant: HybridVariant,
    args: argparse.Namespace,
    *,
    starts: Sequence[int],
) -> WindowArrays:
    profile = data.profile
    observations = observed_state_features(profile)
    history_steps = int(round(variant.history_hours / args.dt_hours))
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "inputs",
            "initial_temperature",
            "history_summary",
            "initial_mode_distribution",
            "observed_mode",
            "observed_age",
            "observed_switch",
            "interval_start",
            "duration",
            "interval_energy",
            "event_power",
            "tau_power",
            "tau_temperature",
            "event_mask",
            "targets",
        )
    }
    for start in starts:
        end = start + args.sequence_length
        fields["inputs"].append(profile.inputs[start:end])
        fields["initial_temperature"].append(profile.initial_temperature[start, 0])
        fields["history_summary"].append(
            history_summary(
                observations,
                end=start,
                history_steps=history_steps,
                mean=data.observation_mean,
                scale=data.observation_scale,
                dt_hours=args.dt_hours,
            )
        )
        fields["initial_mode_distribution"].append(
            initial_mode_distribution(
                data,
                start,
                max_dwell_steps=args.max_dwell_steps,
                use_history=variant.use_history_encoder,
            )
        )
        fields["observed_mode"].append(data.mode[start:end])
        fields["observed_age"].append(data.age[start:end])
        fields["observed_switch"].append(data.switch[start:end])
        fields["interval_start"].append(data.interval_start[start:end])
        fields["duration"].append(data.duration[start:end])
        fields["interval_energy"].append(data.interval_energy[start:end])
        fields["event_power"].append(data.event_power[start:end])
        fields["tau_power"].append(data.tau_power[start:end])
        fields["tau_temperature"].append(data.tau_temperature[start:end])
        fields["event_mask"].append(data.event_mask[start:end])
        fields["targets"].append(profile.targets[start:end])
    return WindowArrays(**{name: jnp.asarray(values) for name, values in fields.items()})


def training_starts(data: ProfileData, variant: HybridVariant, args: argparse.Namespace) -> np.ndarray:
    history_steps = int(round(variant.history_hours / args.dt_hours))
    first = max(args.control_interval_steps, history_steps)
    first += (-first) % args.control_interval_steps
    stride = max(args.control_interval_steps, args.stride)
    stride += (-stride) % args.control_interval_steps
    last = data.split - args.sequence_length
    starts = np.arange(first, last + 1, stride, dtype=np.int32)
    if starts.size == 0:
        raise ValueError(
            f"No training windows for profile {data.profile.profile_id}; reduce history/sequence"
        )
    return starts


def build_model(
    data: ProfileData,
    variant: HybridVariant,
    args: argparse.Namespace,
    *,
    key: jax.Array,
) -> CausalHybridHPEmulator:
    summary_dim = observed_state_features(data.profile).shape[1] * 13
    train_pel = data.profile.targets[: data.split, 2]
    train_q = data.profile.targets[: data.split, 1]
    event_power_cap = max(args.cap_factor * float(np.max(train_pel)), 1.0)
    energy_cap = max(
        args.cap_factor
        * args.control_interval_steps
        * args.dt_hours
        * float(np.max(train_pel)),
        1.0,
    )
    return CausalHybridHPEmulator(
        input_mean=tuple(float(value) for value in data.input_mean),
        input_scale=tuple(float(value) for value in data.input_scale),
        temperature_mean=data.temperature_mean,
        temperature_scale=data.temperature_scale,
        history_dim=summary_dim,
        controller_dim=args.controller_dim,
        rc_nodes=variant.rc_nodes,
        hydronic_modes=args.hydronic_modes,
        max_dwell_steps=args.max_dwell_steps,
        control_interval_steps=args.control_interval_steps,
        horizon_steps=args.horizon_steps,
        dt_hours=args.dt_hours,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        energy_cap_wh_m2=energy_cap,
        event_power_cap_w_m2=event_power_cap,
        hydronic_mode=variant.hydronic_mode,  # type: ignore[arg-type]
        use_history_encoder=variant.use_history_encoder,
        use_dwell_age=variant.use_dwell_age,
        key=key,
    )


def batch_rollout(model: CausalHybridHPEmulator, batch: WindowArrays) -> CausalHybridRollout:
    return jax.vmap(model)(
        jnp.asarray(batch.inputs),
        jnp.asarray(batch.initial_temperature),
        jnp.asarray(batch.history_summary),
        jnp.asarray(batch.initial_mode_distribution),
        jnp.asarray(batch.observed_mode),
        jnp.asarray(batch.observed_age),
        jnp.asarray(batch.interval_start),
    )


def gaussian_nll(
    target: jnp.ndarray,
    mean: jnp.ndarray,
    scale: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    scale = jnp.maximum(scale, 1e-4)
    value = 0.5 * ((target - mean) / scale) ** 2 + jnp.log(scale)
    return jnp.sum(jnp.where(mask, value, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)


def forward_window_mean(values: jnp.ndarray, steps: int) -> jnp.ndarray:
    """Mean of ``values[t:t+steps]`` with zero padding at invalid final rows."""
    cumulative = jnp.concatenate(
        [jnp.zeros((*values.shape[:-1], 1), dtype=values.dtype), jnp.cumsum(values, axis=-1)],
        axis=-1,
    )
    valid = (cumulative[..., steps:] - cumulative[..., :-steps]) / float(steps)
    return jnp.pad(valid, [(0, 0)] * (valid.ndim - 1) + [(0, steps - 1)])


def backward_window_mean(values: jnp.ndarray, steps: int) -> jnp.ndarray:
    """Mean of ``values[t-steps:t]`` with zero padding at invalid initial rows."""
    forward = forward_window_mean(values, steps)
    return jnp.pad(forward[..., : values.shape[-1] - steps], [(0, 0)] * (values.ndim - 1) + [(steps, 0)])


def make_train_step(
    variant: HybridVariant,
    data: ProfileData,
    args: argparse.Namespace,
    optimizer: optax.GradientTransformation,
):
    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(model: CausalHybridHPEmulator, batch: WindowArrays):
        rollout = batch_rollout(model, batch)
        targets = jnp.asarray(batch.targets)
        temperature_mse = jnp.mean(
            ((rollout.prediction[..., 0] - targets[..., 0]) / data.temperature_scale) ** 2
        )
        qroom_mse = jnp.mean(
            ((rollout.prediction[..., 1] - targets[..., 1]) / data.qroom_scale) ** 2
        )
        hazard = jnp.clip(rollout.transition_probability_observed, 1e-6, 1.0 - 1e-6)
        switch = jnp.asarray(batch.observed_switch)
        semi_markov_nll = -jnp.mean(switch * jnp.log(hazard) + (1.0 - switch) * jnp.log1p(-hazard))

        interval_mask = jnp.asarray(batch.interval_start)
        duration_target = jnp.asarray(batch.duration)
        duration_log_probability = jax.nn.log_softmax(rollout.duration_logits, axis=-1)
        selected_duration = jnp.take_along_axis(
            duration_log_probability,
            duration_target[..., None],
            axis=-1,
        )[..., 0]
        duration_nll = -jnp.sum(jnp.where(interval_mask, selected_duration, 0.0)) / jnp.maximum(
            jnp.sum(interval_mask), 1.0
        )
        energy_target = jnp.asarray(batch.interval_energy)
        active_interval = interval_mask & (energy_target > args.heat_on_threshold_w_m2 * args.dt_hours)
        interval_energy_nll = gaussian_nll(
            jnp.log1p(energy_target),
            rollout.active_energy_log_mean,
            rollout.active_energy_log_scale,
            active_interval,
        )
        # Couple the interval emission to the autonomous path. This is an
        # integral target, not a 15-minute compressor-phase loss.
        interval_count = rollout.prediction.shape[1] // args.control_interval_steps
        predicted_interval_energy = args.dt_hours * jnp.sum(
            rollout.prediction[..., 2].reshape(
                rollout.prediction.shape[0],
                interval_count,
                args.control_interval_steps,
            ),
            axis=-1,
        )
        target_interval_energy = jnp.asarray(batch.interval_energy)[:, :: args.control_interval_steps]
        interval_energy_path_mse = jnp.mean(
            ((predicted_interval_energy - target_interval_energy) / data.interval_energy_scale) ** 2
        )
        interval_energy_nll = interval_energy_nll + interval_energy_path_mse

        event_mask = jnp.asarray(batch.event_mask)
        if variant.use_event_losses:
            event_power_nll = gaussian_nll(
                jnp.asarray(batch.event_power),
                rollout.event_power_mean,
                rollout.event_power_scale,
                event_mask,
            )
            tau_power_nll = gaussian_nll(
                jnp.asarray(batch.tau_power),
                rollout.tau_power_mean,
                rollout.tau_power_scale,
                event_mask,
            )
            tau_temperature_nll = gaussian_nll(
                jnp.asarray(batch.tau_temperature),
                rollout.tau_temperature_mean,
                rollout.tau_temperature_scale,
                event_mask,
            )
            predicted_pel = rollout.prediction[..., 2]
            predicted_temperature_current = jnp.concatenate(
                [jnp.asarray(batch.initial_temperature)[:, None], rollout.prediction[:, :-1, 0]],
                axis=1,
            )
            setpoint = jnp.asarray(batch.inputs)[..., 0]
            setpoint_delta = jnp.concatenate(
                [jnp.zeros((setpoint.shape[0], 1), dtype=setpoint.dtype), jnp.diff(setpoint, axis=1)],
                axis=1,
            )
            trace_power_mse = jnp.asarray(0.0)
            trace_tau_power_mse = jnp.asarray(0.0)
            trace_tau_temperature_mse = jnp.asarray(0.0)
            for horizon_index, steps in enumerate(args.horizon_steps):
                local_valid = jnp.zeros((predicted_pel.shape[1],), dtype=bool)
                local_valid = local_valid.at[steps : predicted_pel.shape[1] - steps + 1].set(True)
                mask = event_mask[..., horizon_index] & local_valid[None, :]
                post_power = forward_window_mean(predicted_pel, steps)
                pre_power = backward_window_mean(predicted_pel, steps)
                post_temperature = forward_window_mean(rollout.prediction[..., 0], steps)
                pre_temperature = backward_window_mean(predicted_temperature_current, steps)
                safe_delta = jnp.where(jnp.abs(setpoint_delta) > 1e-5, setpoint_delta, 1.0)
                trace_tau_power = (post_power - pre_power) / safe_delta
                trace_tau_temperature = (post_temperature - pre_temperature) / safe_delta
                denominator = jnp.maximum(jnp.sum(mask), 1.0)
                trace_power_mse = trace_power_mse + jnp.sum(
                    jnp.where(
                        mask,
                        ((post_power - jnp.asarray(batch.event_power)[..., horizon_index]) / data.pel_scale) ** 2,
                        0.0,
                    )
                ) / denominator
                trace_tau_power_mse = trace_tau_power_mse + jnp.sum(
                    jnp.where(
                        mask,
                        ((trace_tau_power - jnp.asarray(batch.tau_power)[..., horizon_index]) / data.tau_power_scale) ** 2,
                        0.0,
                    )
                ) / denominator
                trace_tau_temperature_mse = trace_tau_temperature_mse + jnp.sum(
                    jnp.where(
                        mask,
                        ((trace_tau_temperature - jnp.asarray(batch.tau_temperature)[..., horizon_index]) / data.tau_temperature_scale) ** 2,
                        0.0,
                    )
                ) / denominator
            horizon_count = float(len(args.horizon_steps))
            event_power_nll = event_power_nll + trace_power_mse / horizon_count
            causal_tau_nll = 0.5 * (tau_power_nll + tau_temperature_nll) + 0.5 * (
                trace_tau_power_mse + trace_tau_temperature_mse
            ) / horizon_count
        else:
            event_power_nll = jnp.asarray(0.0)
            causal_tau_nll = jnp.asarray(0.0)
        components = jnp.stack(
            [
                temperature_mse,
                qroom_mse,
                semi_markov_nll,
                duration_nll,
                interval_energy_nll,
                event_power_nll,
                causal_tau_nll,
            ]
        )
        weights = jnp.asarray(
            [
                args.temperature_weight,
                args.qroom_weight,
                args.semi_markov_weight,
                args.duration_weight,
                args.interval_energy_weight,
                args.event_power_weight,
                args.causal_tau_weight,
            ]
        )
        return jnp.sum(weights * components), components

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        (loss, components), grads = loss_fn(model, batch)
        grad_norm = optax.global_norm(grads)
        updates, candidate_opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        candidate_model = eqx.apply_updates(model, updates)
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        finite = finite & jax.tree_util.tree_reduce(
            jnp.logical_and,
            jax.tree_util.tree_map(
                lambda value: jnp.all(jnp.isfinite(value)) if eqx.is_array(value) else True,
                candidate_model,
            ),
            initializer=jnp.asarray(True),
        )
        model = jax.tree_util.tree_map(
            lambda candidate, current: jnp.where(finite, candidate, current)
            if eqx.is_array(candidate)
            else candidate,
            candidate_model,
            model,
        )
        opt_state = jax.tree_util.tree_map(
            lambda candidate, current: jnp.where(finite, candidate, current)
            if eqx.is_array(candidate)
            else candidate,
            candidate_opt_state,
            opt_state,
        )
        return model, opt_state, loss, components, grad_norm, finite

    return train_step


def take_batch(windows: WindowArrays, indices: np.ndarray) -> WindowArrays:
    return jax.tree_util.tree_map(lambda value: value[indices], windows)


def fit_model(
    model: CausalHybridHPEmulator,
    data: ProfileData,
    variant: HybridVariant,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[CausalHybridHPEmulator, dict[str, float]]:
    optimizer = optax.chain(optax.clip_by_global_norm(args.gradient_clip_norm), optax.adam(args.learning_rate))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_step = make_train_step(variant, data, args, optimizer)
    starts = training_starts(data, variant, args)
    windows = make_windows(data, variant, args, starts=starts)
    rng = np.random.default_rng(seed)
    totals = np.zeros((len(LOSS_NAMES),), dtype=np.float64)
    updates = 0
    skipped = 0
    max_grad = 0.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        order = rng.permutation(starts.size)
        for offset in range(0, starts.size, args.batch_size):
            indices = order[offset:offset + args.batch_size]
            if indices.size == 0:
                continue
            batch = take_batch(windows, indices)
            model, opt_state, loss, components, grad_norm, finite = train_step(
                model,
                opt_state,
                batch,
            )
            grad_value = float(np.asarray(grad_norm))
            max_grad = max(max_grad, grad_value if np.isfinite(grad_value) else np.inf)
            if not bool(np.asarray(finite)):
                skipped += 1
                continue
            totals += np.asarray(components)
            updates += 1
    stats = {
        "runtime_seconds": time.perf_counter() - started,
        "max_grad_norm": max_grad,
        "updates": float(updates),
        "skipped_updates": float(skipped),
    }
    stats.update(
        {f"train_{name}": float(totals[index] / max(updates, 1)) for index, name in enumerate(LOSS_NAMES)}
    )
    return model, stats


def autonomous_test_rollout(
    model: CausalHybridHPEmulator,
    data: ProfileData,
    variant: HybridVariant,
    args: argparse.Namespace,
) -> CausalHybridRollout:
    start = data.split
    profile = data.profile
    observations = observed_state_features(profile)
    summary = history_summary(
        observations,
        end=start,
        history_steps=int(round(variant.history_hours / args.dt_hours)),
        mean=data.observation_mean,
        scale=data.observation_scale,
        dt_hours=args.dt_hours,
    )
    distribution = initial_mode_distribution(
        data,
        start,
        max_dwell_steps=args.max_dwell_steps,
        use_history=variant.use_history_encoder,
    )
    # observed_mode/age select hazards only for the diagnostic likelihood.  They
    # never enter mode propagation or any predicted output.
    return model(
        jnp.asarray(profile.inputs[start:]),
        jnp.asarray(profile.initial_temperature[start, 0]),
        jnp.asarray(summary),
        jnp.asarray(distribution),
        jnp.asarray(data.mode[start:]),
        jnp.asarray(data.age[start:]),
        jnp.asarray(data.interval_start[start:]),
    )


def evaluate(
    model: CausalHybridHPEmulator,
    rollout: CausalHybridRollout,
    data: ProfileData,
    args: argparse.Namespace,
) -> dict[str, float]:
    start = data.split
    prediction = np.asarray(rollout.prediction)
    target = data.profile.targets[start:]
    result = {
        "test_rmse_c": _rmse(prediction[:, 0], target[:, 0]),
        "test_qroom_rmse_w_m2": _rmse(prediction[:, 1], target[:, 1]),
        "test_pel_rmse_w_m2": _rmse(prediction[:, 2], target[:, 2]),
        "test_mode_brier": float(np.mean((np.asarray(rollout.mode_probability) - data.mode[start:]) ** 2)),
        "test_delta_pel_corr": _corr(np.diff(prediction[:, 2]), np.diff(target[:, 2])),
    }
    result.update(
        trace_event_response_metrics(
            data.profile.inputs[start:],
            target,
            prediction,
            horizon_steps=int(round(3.0 / args.dt_hours)),
            threshold_c=args.setpoint_change_threshold_c,
        )
    )
    for days in (1, 7, 30):
        steps = min(int(round(24.0 * days / args.dt_hours)), target.shape[0])
        result[f"test_rmse_c_first_{days}d"] = _rmse(
            prediction[:steps, 0], target[:steps, 0]
        )
        result[f"test_qroom_rmse_w_m2_first_{days}d"] = _rmse(
            prediction[:steps, 1], target[:steps, 1]
        )
        result[f"test_pel_rmse_w_m2_first_{days}d"] = _rmse(
            prediction[:steps, 2], target[:steps, 2]
        )
    event_mask = data.event_mask[start:]
    for horizon_index, steps in enumerate(args.horizon_steps):
        hours = steps * args.dt_hours
        mask = event_mask[:, horizon_index]
        if bool(mask.any()):
            result[f"test_event_power_rmse_{hours:g}h"] = _rmse(
                np.asarray(rollout.event_power_mean)[mask, horizon_index],
                data.event_power[start:][mask, horizon_index],
            )
            result[f"test_tau_power_rmse_{hours:g}h"] = _rmse(
                np.asarray(rollout.tau_power_mean)[mask, horizon_index],
                data.tau_power[start:][mask, horizon_index],
            )
            result[f"test_tau_temperature_rmse_{hours:g}h"] = _rmse(
                np.asarray(rollout.tau_temperature_mean)[mask, horizon_index],
                data.tau_temperature[start:][mask, horizon_index],
            )
            lower = np.asarray(rollout.tau_power_mean)[mask, horizon_index] - 1.96 * np.asarray(
                rollout.tau_power_scale
            )[mask, horizon_index]
            upper = np.asarray(rollout.tau_power_mean)[mask, horizon_index] + 1.96 * np.asarray(
                rollout.tau_power_scale
            )[mask, horizon_index]
            truth = data.tau_power[start:][mask, horizon_index]
            result[f"test_tau_power_95_coverage_{hours:g}h"] = float(
                np.mean((truth >= lower) & (truth <= upper))
            )
        else:
            result[f"test_event_power_rmse_{hours:g}h"] = float("nan")
            result[f"test_tau_power_rmse_{hours:g}h"] = float("nan")
            result[f"test_tau_temperature_rmse_{hours:g}h"] = float("nan")
            result[f"test_tau_power_95_coverage_{hours:g}h"] = float("nan")

    test_profile = ClosedLoopProfile(
        profile_id=data.profile.profile_id,
        datetime=data.profile.datetime[start:],
        metadata=data.profile.metadata,
        inputs=data.profile.inputs[start:],
        targets=target,
        initial_temperature=data.profile.initial_temperature[start:],
    )
    event_config = EventStudyConfig(
        horizons_hours=(3.0,),
        horizon_steps=(int(round(3.0 / args.dt_hours)),),
        dt_hours=args.dt_hours,
        min_setpoint_change=args.setpoint_change_threshold_c,
        min_events=args.min_events,
        controls="full",
    )
    true_flex = estimate_trace_flexibility(test_profile, target, event_config)[0]
    pred_flex = estimate_trace_flexibility(test_profile, prediction, event_config)[0]
    result.update(
        {
            "test_flex_up_true_wh_m2_k": true_flex.up_flex_wh_m2_k,
            "test_flex_up_pred_wh_m2_k": pred_flex.up_flex_wh_m2_k,
            "test_flex_down_true_wh_m2_k": true_flex.down_flex_wh_m2_k,
            "test_flex_down_pred_wh_m2_k": pred_flex.down_flex_wh_m2_k,
            "test_flex_up_abs_error_wh_m2_k": abs(pred_flex.up_flex_wh_m2_k - true_flex.up_flex_wh_m2_k),
            "test_flex_down_abs_error_wh_m2_k": abs(pred_flex.down_flex_wh_m2_k - true_flex.down_flex_wh_m2_k),
        }
    )
    source_energy = np.sum(
        np.asarray(jax.vmap(model.cop)(jnp.asarray(data.profile.inputs[start:, 1])))
        * prediction[:, 2]
        * args.dt_hours
    )
    result["test_qroom_to_cop_pel_energy_ratio"] = float(
        np.sum(prediction[:, 1]) * args.dt_hours / max(float(source_energy), 1e-8)
    )
    return result


def save_plot(
    data: ProfileData,
    prediction: np.ndarray,
    variant: HybridVariant,
    output_dir: Path,
) -> Path:
    start = data.split
    target = data.profile.targets[start:]
    datetime = data.profile.datetime[start:]
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Indoor temperature", "Delivered heat", "HP electric power"),
    )
    labels = ((0, "degC"), (1, "W/m2"), (2, "W/m2"))
    for row, (channel, unit) in enumerate(labels, start=1):
        figure.add_trace(go.Scatter(x=datetime, y=target[:, channel], name="Simulated"), row=row, col=1)
        figure.add_trace(
            go.Scatter(x=datetime, y=prediction[:, channel], name="Autonomous", line={"color": "#d62728"}),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=unit, row=row, col=1)
    figure.add_trace(
        go.Scatter(x=datetime, y=data.profile.inputs[start:, 0], name="Tset", line={"dash": "dash"}),
        row=1,
        col=1,
    )
    figure.update_layout(
        title=f"Causal hybrid HP, profile {data.profile.profile_id}, {variant.name}",
        height=1050,
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"profile_{data.profile.profile_id}_{variant.name}.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def save_summary(runs: pd.DataFrame, output_dir: Path) -> Path:
    metrics = (
        "test_rmse_c",
        "test_qroom_rmse_w_m2",
        "test_pel_rmse_w_m2",
        "test_flex_up_abs_error_wh_m2_k",
        "test_flex_down_abs_error_wh_m2_k",
        "test_tau_power_rmse_3h",
    )
    means = runs.groupby("variant", as_index=False)[list(metrics)].mean(numeric_only=True)
    figure = make_subplots(rows=2, cols=3, subplot_titles=metrics)
    for index, metric in enumerate(metrics):
        figure.add_trace(
            go.Bar(x=means["variant"], y=means[metric], showlegend=False),
            row=index // 3 + 1,
            col=index % 3 + 1,
        )
        figure.update_xaxes(tickangle=35, row=index // 3 + 1, col=index % 3 + 1)
    figure.update_layout(title="Causal hybrid building-local ablation", height=900)
    path = output_dir / "causal_hybrid_ablation_summary.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def selected_variants(names: Sequence[str]) -> tuple[HybridVariant, ...]:
    by_name = {variant.name: variant for variant in DEFAULT_VARIANTS}
    if not names:
        return DEFAULT_VARIANTS
    unknown = sorted(set(names).difference(by_name))
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {sorted(by_name)}")
    return tuple(by_name[name] for name in names)


def run_ablation(args: argparse.Namespace) -> None:
    args.horizon_steps = tuple(int(round(hours / args.dt_hours)) for hours in args.horizons_hours)
    profiles = [prepare_profile(profile, args) for profile in load_profiles(args)]
    variants = selected_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "profiles": [data.profile.profile_id for data in profiles],
        "variants": [asdict(variant) for variant in variants],
        "test_protocol": "first-half fit; one history initialization; autonomous second-half rollout",
        "pel_pointwise_rmse_in_training": False,
        "setpoint_thermal_path": "HP -> COP*Pel -> positive hydronics -> passive RC only",
    }
    (args.output_dir / "causal_hybrid_ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    rows: list[dict[str, Any]] = []
    best_by_profile: dict[int, tuple[float, np.ndarray, HybridVariant]] = {}
    for variant in variants:
        print(f"starting variant={variant.name} settings={asdict(variant)}")
        for profile_index, data in enumerate(profiles):
            # Pair initial weights and data order across variants. Models with
            # matching shapes now differ only in the ablated mechanism.
            seed = args.seed + 10_000 * profile_index
            model = build_model(data, variant, args, key=jax.random.PRNGKey(seed))
            model, train_stats = fit_model(model, data, variant, args, seed=seed)
            rollout = autonomous_test_rollout(model, data, variant, args)
            metrics = evaluate(model, rollout, data, args)
            row = {
                "variant": variant.name,
                "profile_id": data.profile.profile_id,
                **asdict(variant),
                **train_stats,
                **metrics,
            }
            rows.append(row)
            score = metrics["test_rmse_c"] + 0.05 * metrics["test_qroom_rmse_w_m2"]
            if data.profile.profile_id not in best_by_profile or score < best_by_profile[data.profile.profile_id][0]:
                best_by_profile[data.profile.profile_id] = (
                    score,
                    np.asarray(rollout.prediction),
                    variant,
                )
            pd.DataFrame(rows).to_csv(args.output_dir / "causal_hybrid_ablation_runs.csv", index=False)
            print(
                f"finished variant={variant.name} profile={data.profile.profile_id} "
                f"T={metrics['test_rmse_c']:.3f} Q={metrics['test_qroom_rmse_w_m2']:.3f} "
                f"Pel={metrics['test_pel_rmse_w_m2']:.3f} "
                f"flex_up={metrics['test_flex_up_abs_error_wh_m2_k']:.3f} "
                f"grad={train_stats['max_grad_norm']:.3g}"
            )
    runs = pd.DataFrame(rows)
    runs_path = args.output_dir / "causal_hybrid_ablation_runs.csv"
    runs.to_csv(runs_path, index=False)
    means_path = args.output_dir / "causal_hybrid_ablation_means.csv"
    runs.groupby("variant", as_index=False).mean(numeric_only=True).to_csv(means_path, index=False)
    summary_path = save_summary(runs, args.output_dir)
    for data in profiles:
        _, prediction, variant = best_by_profile[data.profile.profile_id]
        path = save_plot(data, prediction, variant, args.output_dir / "best_profile_plots")
        print(f"saved_profile_plot={path}")
    print(f"saved_runs={runs_path}")
    print(f"saved_means={means_path}")
    print(f"saved_summary={summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator/local_causal_hybrid_ablation"))
    parser.add_argument("--profile-ids", type=int, nargs="*", default=())
    parser.add_argument("--num-profiles", type=int, default=3)
    parser.add_argument("--profile-selection", choices=("random", "first"), default="random")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--variants", nargs="*", default=())
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--sequence-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--control-interval-steps", type=int, default=12)
    parser.add_argument("--horizons-hours", type=float, nargs="+", default=(0.5, 1.0, 2.0, 3.0))
    parser.add_argument("--max-dwell-steps", type=int, default=96)
    parser.add_argument("--controller-dim", type=int, default=8)
    parser.add_argument("--hydronic-modes", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--cap-factor", type=float, default=1.5)
    parser.add_argument("--heat-on-threshold-w-m2", type=float, default=1e-3)
    parser.add_argument("--setpoint-change-threshold-c", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--temperature-weight", type=float, default=1.0)
    parser.add_argument("--qroom-weight", type=float, default=1.0)
    parser.add_argument("--semi-markov-weight", type=float, default=0.2)
    parser.add_argument("--duration-weight", type=float, default=0.2)
    parser.add_argument("--interval-energy-weight", type=float, default=0.5)
    parser.add_argument("--event-power-weight", type=float, default=0.5)
    parser.add_argument("--causal-tau-weight", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    if args.sequence_length % args.control_interval_steps:
        parser.error("--sequence-length must be a multiple of --control-interval-steps")
    if any(hours <= 0 or hours > args.control_interval_steps * args.dt_hours for hours in args.horizons_hours):
        parser.error("event horizons must be positive and no longer than the control interval")
    return args


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
