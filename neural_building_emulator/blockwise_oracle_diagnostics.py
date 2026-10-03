"""Blockwise oracle diagnostics for the closed-loop HP emulator.

Each building is fitted independently on the first part of its annual profile.
Ground-truth upstream signals are then supplied to one block at a time:

* HP oracle: observed indoor temperature and causal observed Pel history -> Pel;
* hydronic oracle: observed Pel and its history -> Qroom;
* thermal oracle: observed Qroom -> next indoor temperature.

These are deliberately teacher-forced diagnostics, not deployable emulators.
They test whether each conditional relationship is identifiable before the
controller, buffer, and thermal recurrences are coupled together.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import DEFAULT_DATASET_PATH, ClosedLoopProfile
from .local_closed_loop_hp_ablation import load_profiles, prepare_profiles


DEFAULT_SOURCE_MANIFEST = Path(
    "output/neural_building_emulator/local_hp_ablation_focused_30/"
    "local_ablation_manifest.json"
)


@dataclass(frozen=True)
class FittedRegressor:
    model: HistGradientBoostingRegressor | None
    constant: float
    log_target: bool
    cap: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            prediction = np.full(features.shape[0], self.constant, dtype=np.float64)
        else:
            prediction = np.asarray(self.model.predict(features), dtype=np.float64)
        if self.log_target:
            prediction = np.expm1(prediction)
        return np.clip(prediction, 0.0, self.cap)


@dataclass(frozen=True)
class FittedClassifier:
    model: HistGradientBoostingClassifier | None
    probability: float

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(features.shape[0], self.probability, dtype=np.float64)
        return np.asarray(self.model.predict_proba(features)[:, 1], dtype=np.float64)


@dataclass(frozen=True)
class PhysicalRidge:
    coefficients: np.ndarray
    intercept: float

    def predict_one(self, features: np.ndarray) -> float:
        return float(self.intercept + np.dot(self.coefficients, features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.intercept + features @ self.coefficients


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3 or np.std(left[mask]) < 1e-12 or np.std(right[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _causal_history(values: np.ndarray, window_steps: Sequence[int]) -> np.ndarray:
    """Observed history ending at t-1; the current value is never included."""
    values = np.asarray(values, dtype=np.float64)
    series = pd.Series(values)
    columns = [series.shift(1).fillna(values[0]).to_numpy(dtype=np.float64)]
    for steps in window_steps:
        history = series.shift(1).rolling(int(steps), min_periods=1).mean()
        columns.append(history.fillna(values[0]).to_numpy(dtype=np.float64))
    return np.column_stack(columns)


def _history_steps(hours: Sequence[float], dt_hours: float) -> tuple[int, ...]:
    return tuple(sorted(set(max(1, int(round(value / dt_hours))) for value in hours)))


def _base_state_features(profile: ClosedLoopProfile) -> np.ndarray:
    inputs = np.asarray(profile.inputs, dtype=np.float64)
    temperature = np.asarray(profile.initial_temperature[:, 0], dtype=np.float64)
    setpoint = inputs[:, 0]
    outdoor = inputs[:, 1]
    setpoint_change = np.diff(setpoint, prepend=setpoint[0])
    return np.column_stack(
        [
            inputs,
            temperature,
            setpoint - temperature,
            outdoor - temperature,
            setpoint_change,
        ]
    )


def hp_oracle_features(
    profile: ClosedLoopProfile,
    *,
    history_steps: Sequence[int],
    heat_on_threshold: float,
) -> np.ndarray:
    pel = np.asarray(profile.targets[:, 2], dtype=np.float64)
    mode = (pel > heat_on_threshold).astype(np.float64)
    return np.column_stack(
        [
            _base_state_features(profile),
            _causal_history(pel, history_steps),
            _causal_history(mode, history_steps),
        ]
    )


def qroom_oracle_features(
    profile: ClosedLoopProfile,
    *,
    history_steps: Sequence[int],
    include_qroom_history: bool,
) -> np.ndarray:
    pel = np.asarray(profile.targets[:, 2], dtype=np.float64)
    features = [
        _base_state_features(profile),
        pel[:, np.newaxis],
        _causal_history(pel, history_steps),
    ]
    if include_qroom_history:
        qroom = np.asarray(profile.targets[:, 1], dtype=np.float64)
        features.append(_causal_history(qroom, history_steps))
    return np.column_stack(features)


def thermal_static_features(
    profile: ClosedLoopProfile,
    *,
    history_steps: Sequence[int],
) -> np.ndarray:
    inputs = np.asarray(profile.inputs, dtype=np.float64)
    qroom = np.asarray(profile.targets[:, 1], dtype=np.float64)
    outdoor = inputs[:, 1]
    solar = inputs[:, 2]
    # Tset is intentionally excluded. It may affect Tin only through observed Qroom.
    return np.column_stack(
        [
            inputs[:, 1:],
            qroom,
            _causal_history(qroom, history_steps),
            _causal_history(outdoor, history_steps),
            _causal_history(solar, history_steps),
        ]
    )


def thermal_features(static: np.ndarray, outdoor: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    return np.column_stack([static, temperature, outdoor - temperature])


def _fit_regressor(
    features: np.ndarray,
    target: np.ndarray,
    *,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    seed: int,
    log_target: bool = True,
    cap_factor: float = 1.5,
) -> FittedRegressor:
    target = np.asarray(target, dtype=np.float64)
    cap = max(float(np.max(target)) * cap_factor, 1.0)
    transformed = np.log1p(target) if log_target else target
    if np.std(transformed) < 1e-10:
        return FittedRegressor(None, float(transformed[0]), log_target, cap)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(features, transformed)
    return FittedRegressor(model, float(np.mean(transformed)), log_target, cap)


def _fit_classifier(
    features: np.ndarray,
    target: np.ndarray,
    *,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    seed: int,
) -> FittedClassifier:
    target = np.asarray(target, dtype=np.int32)
    probability = float(np.mean(target))
    if np.unique(target).size < 2:
        return FittedClassifier(None, probability)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(features, target)
    return FittedClassifier(model, probability)


def _fit_physical_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> PhysicalRidge:
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=alpha).fit(scaler.transform(features), target)
    coefficients = np.asarray(model.coef_, dtype=np.float64) / scaler.scale_
    intercept = float(model.intercept_ - np.dot(coefficients, scaler.mean_))
    return PhysicalRidge(coefficients, intercept)


def _event_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    setpoint: np.ndarray,
    *,
    start: int,
    horizon_steps: int,
    threshold_c: float,
) -> dict[str, float]:
    setpoint_delta = np.diff(setpoint, prepend=setpoint[0])
    candidates = np.flatnonzero(np.abs(setpoint_delta) >= threshold_c)
    events = candidates[
        (candidates - horizon_steps >= start)
        & (candidates + horizon_steps <= len(target))
    ]
    if events.size == 0:
        return {"count": 0.0, "rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "corr": float("nan")}
    true_response = np.asarray(
        [
            np.mean(target[index : index + horizon_steps])
            - np.mean(target[index - horizon_steps : index])
            for index in events
        ]
    )
    predicted_response = np.asarray(
        [
            np.mean(prediction[index : index + horizon_steps])
            - np.mean(prediction[index - horizon_steps : index])
            for index in events
        ]
    )
    return {
        "count": float(events.size),
        "rmse": _rmse(predicted_response, true_response),
        "mae": _mae(predicted_response, true_response),
        "bias": float(np.mean(predicted_response - true_response)),
        "corr": _corr(predicted_response, true_response),
    }


def _add_event_metrics(row: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value


def fit_profile(
    profile: ClosedLoopProfile,
    *,
    train_fraction: float,
    history_steps: Sequence[int],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_steps = profile.targets.shape[0]
    split = int(round(n_steps * train_fraction))
    train = slice(0, split)
    test = slice(split, n_steps)
    target = np.asarray(profile.targets, dtype=np.float64)
    temperature_current = np.asarray(profile.initial_temperature[:, 0], dtype=np.float64)
    setpoint = np.asarray(profile.inputs[:, 0], dtype=np.float64)
    outdoor = np.asarray(profile.inputs[:, 1], dtype=np.float64)
    qroom = target[:, 1]
    pel = target[:, 2]
    mode = (pel > args.heat_on_threshold_w_m2).astype(np.int32)

    hp_features = hp_oracle_features(
        profile,
        history_steps=history_steps,
        heat_on_threshold=args.heat_on_threshold_w_m2,
    )
    mode_model = _fit_classifier(
        hp_features[train],
        mode[train],
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        seed=seed,
    )
    active_train = mode[train] > 0
    active_model = _fit_regressor(
        hp_features[train][active_train],
        pel[train][active_train],
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        seed=seed + 1,
    )
    hp_probability = np.clip(mode_model.predict_probability(hp_features), 1e-6, 1.0 - 1e-6)
    hp_active_power = active_model.predict(hp_features)
    hp_expected_power = hp_probability * hp_active_power
    hp_oracle_mode_power = mode.astype(np.float64) * hp_active_power

    q_pel_features = qroom_oracle_features(
        profile,
        history_steps=history_steps,
        include_qroom_history=False,
    )
    q_history_features = qroom_oracle_features(
        profile,
        history_steps=history_steps,
        include_qroom_history=True,
    )
    q_pel_model = _fit_regressor(
        q_pel_features[train],
        qroom[train],
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        seed=seed + 2,
    )
    q_history_model = _fit_regressor(
        q_history_features[train],
        qroom[train],
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        seed=seed + 3,
    )
    q_from_pel = q_pel_model.predict(q_pel_features)
    q_from_observed_history = q_history_model.predict(q_history_features)

    thermal_static = thermal_static_features(profile, history_steps=history_steps)
    thermal_teacher_features = thermal_features(thermal_static, outdoor, temperature_current)
    thermal_tree = _fit_regressor(
        thermal_teacher_features[train],
        target[train, 0],
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        seed=seed + 4,
        log_target=False,
        cap_factor=1.5,
    )
    thermal_teacher_prediction = thermal_tree.predict(thermal_teacher_features)
    thermal_ridge = _fit_physical_ridge(
        thermal_teacher_features[train],
        target[train, 0],
        alpha=args.thermal_ridge,
    )
    thermal_ridge_teacher = thermal_ridge.predict(thermal_teacher_features)
    thermal_recursive = np.full(n_steps, np.nan, dtype=np.float64)
    temperature_t = float(temperature_current[split])
    for index in range(split, n_steps):
        feature = np.concatenate(
            [thermal_static[index], [temperature_t, outdoor[index] - temperature_t]]
        )
        temperature_t = thermal_ridge.predict_one(feature)
        if not np.isfinite(temperature_t) or abs(temperature_t) > args.recursive_temperature_guard_c:
            break
        thermal_recursive[index] = temperature_t

    train_mode_rate = float(np.mean(mode[train]))
    train_active_mean = float(np.mean(pel[train][active_train]))
    constant_hp = np.full(n_steps, train_mode_rate * train_active_mean)
    previous_mode = np.concatenate([[mode[0]], mode[:-1]])
    previous_mode_probability = previous_mode.astype(np.float64)
    q_train_mean = float(np.mean(qroom[train]))
    q_constant = np.full(n_steps, q_train_mean)
    temperature_persistence = temperature_current.copy()
    temperature_setpoint = setpoint.copy()

    finite_recursive = np.isfinite(thermal_recursive[test])
    recursive_coverage = float(np.mean(finite_recursive))
    if bool(finite_recursive.any()):
        recursive_rmse = _rmse(thermal_recursive[test][finite_recursive], target[test, 0][finite_recursive])
        recursive_mae = _mae(thermal_recursive[test][finite_recursive], target[test, 0][finite_recursive])
        recursive_bias = float(np.mean(thermal_recursive[test][finite_recursive] - target[test, 0][finite_recursive]))
    else:
        recursive_rmse = recursive_mae = recursive_bias = float("nan")

    active_test = mode[test] > 0
    row: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "steps": n_steps,
        "split_index": split,
        "train_mode_rate": train_mode_rate,
        "test_mode_rate": float(np.mean(mode[test])),
        "hp_mode_brier": float(np.mean((hp_probability[test] - mode[test]) ** 2)),
        "hp_mode_log_loss": float(
            -np.mean(
                mode[test] * np.log(hp_probability[test])
                + (1 - mode[test]) * np.log1p(-hp_probability[test])
            )
        ),
        "hp_previous_mode_brier": float(np.mean((previous_mode_probability[test] - mode[test]) ** 2)),
        "hp_constant_mode_brier": float(np.mean((train_mode_rate - mode[test]) ** 2)),
        "hp_expected_pel_rmse_w_m2": _rmse(hp_expected_power[test], pel[test]),
        "hp_expected_pel_mae_w_m2": _mae(hp_expected_power[test], pel[test]),
        "hp_constant_pel_rmse_w_m2": _rmse(constant_hp[test], pel[test]),
        "hp_zero_pel_rmse_w_m2": _rmse(np.zeros_like(pel[test]), pel[test]),
        "hp_oracle_mode_pel_rmse_w_m2": _rmse(hp_oracle_mode_power[test], pel[test]),
        "hp_active_pel_rmse_w_m2": _rmse(hp_active_power[test][active_test], pel[test][active_test]),
        "hp_active_train_mean_rmse_w_m2": _rmse(
            np.full(int(active_test.sum()), train_active_mean), pel[test][active_test]
        ),
        "hp_delta_pel_corr": _corr(np.diff(hp_expected_power[test]), np.diff(pel[test])),
        "q_pel_history_rmse_w_m2": _rmse(q_from_pel[test], qroom[test]),
        "q_pel_history_mae_w_m2": _mae(q_from_pel[test], qroom[test]),
        "q_observed_history_rmse_w_m2": _rmse(q_from_observed_history[test], qroom[test]),
        "q_observed_history_mae_w_m2": _mae(q_from_observed_history[test], qroom[test]),
        "q_constant_rmse_w_m2": _rmse(q_constant[test], qroom[test]),
        "q_zero_rmse_w_m2": _rmse(np.zeros_like(qroom[test]), qroom[test]),
        "q_pel_history_delta_corr": _corr(np.diff(q_from_pel[test]), np.diff(qroom[test])),
        "q_observed_history_delta_corr": _corr(
            np.diff(q_from_observed_history[test]), np.diff(qroom[test])
        ),
        "thermal_tree_one_step_rmse_c": _rmse(thermal_teacher_prediction[test], target[test, 0]),
        "thermal_tree_one_step_mae_c": _mae(thermal_teacher_prediction[test], target[test, 0]),
        "thermal_ridge_one_step_rmse_c": _rmse(thermal_ridge_teacher[test], target[test, 0]),
        "thermal_ridge_recursive_rmse_c": recursive_rmse,
        "thermal_ridge_recursive_mae_c": recursive_mae,
        "thermal_ridge_recursive_bias_c": recursive_bias,
        "thermal_ridge_recursive_coverage": recursive_coverage,
        "thermal_persistence_one_step_rmse_c": _rmse(temperature_persistence[test], target[test, 0]),
        "thermal_setpoint_rmse_c": _rmse(temperature_setpoint[test], target[test, 0]),
        "thermal_ridge_current_temperature_coefficient": float(thermal_ridge.coefficients[-2]),
    }
    horizon_steps = max(1, int(round(args.event_horizon_hours / args.dt_hours)))
    for prefix, actual, prediction in (
        ("hp_event", pel, hp_expected_power),
        ("hp_oracle_mode_event", pel, hp_oracle_mode_power),
        ("hp_constant_event", pel, constant_hp),
        ("q_pel_event", qroom, q_from_pel),
        ("q_observed_history_event", qroom, q_from_observed_history),
        ("q_constant_event", qroom, q_constant),
    ):
        _add_event_metrics(
            row,
            prefix,
            _event_metrics(
                actual,
                prediction,
                setpoint,
                start=split,
                horizon_steps=horizon_steps,
                threshold_c=args.setpoint_change_threshold_c,
            ),
        )

    traces = {
        "hp_probability": hp_probability,
        "hp_expected_power": hp_expected_power,
        "hp_oracle_mode_power": hp_oracle_mode_power,
        "q_from_pel": q_from_pel,
        "q_from_observed_history": q_from_observed_history,
        "thermal_tree_teacher": thermal_teacher_prediction,
        "thermal_ridge_recursive": thermal_recursive,
    }
    return row, traces


def save_profile_plot(
    path: Path,
    profile: ClosedLoopProfile,
    row: dict[str, Any],
    traces: dict[str, np.ndarray],
) -> None:
    split = int(row["split_index"])
    times = profile.datetime[split:]
    target = profile.targets[split:]
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=("Thermal oracle", "HP electric-power oracle", "Hydronic oracle", "HP mode oracle"),
    )
    figure.add_trace(go.Scattergl(x=times, y=target[:, 0], name="Tin simulated", mode="lines"), row=1, col=1)
    figure.add_trace(go.Scattergl(x=times, y=traces["thermal_tree_teacher"][split:], name="Tin one-step oracle", mode="lines"), row=1, col=1)
    figure.add_trace(go.Scattergl(x=times, y=traces["thermal_ridge_recursive"][split:], name="Tin recursive ridge", mode="lines"), row=1, col=1)
    figure.add_trace(go.Scattergl(x=times, y=target[:, 2], name="Pel simulated", mode="lines"), row=2, col=1)
    figure.add_trace(go.Scattergl(x=times, y=traces["hp_expected_power"][split:], name="Pel expected oracle", mode="lines"), row=2, col=1)
    figure.add_trace(go.Scattergl(x=times, y=traces["hp_oracle_mode_power"][split:], name="Pel true-mode oracle", mode="lines"), row=2, col=1)
    figure.add_trace(go.Scattergl(x=times, y=target[:, 1], name="Qroom simulated", mode="lines"), row=3, col=1)
    figure.add_trace(go.Scattergl(x=times, y=traces["q_from_pel"][split:], name="Q from Pel history", mode="lines"), row=3, col=1)
    figure.add_trace(go.Scattergl(x=times, y=traces["q_from_observed_history"][split:], name="Q with observed Q history", mode="lines"), row=3, col=1)
    figure.add_trace(go.Scattergl(x=times, y=(target[:, 2] > 1e-6).astype(float), name="HP mode simulated", mode="lines"), row=4, col=1)
    figure.add_trace(go.Scattergl(x=times, y=traces["hp_probability"][split:], name="HP mode probability", mode="lines"), row=4, col=1)
    figure.add_vline(x=times[0], line_dash="dash", line_color="#666")
    figure.update_yaxes(title_text="degC", row=1, col=1)
    figure.update_yaxes(title_text="W/m2", row=2, col=1)
    figure.update_yaxes(title_text="W/m2", row=3, col=1)
    figure.update_yaxes(title_text="probability", range=[-0.05, 1.05], row=4, col=1)
    figure.update_layout(
        title=(
            f"Blockwise oracle diagnostics, profile {profile.profile_id}<br>"
            f"HP RMSE={row['hp_expected_pel_rmse_w_m2']:.2f} W/m2, "
            f"Q(P history) RMSE={row['q_pel_history_rmse_w_m2']:.2f} W/m2, "
            f"Tin recursive RMSE={row['thermal_ridge_recursive_rmse_c']:.2f} C"
        ),
        height=1250,
        template="plotly_white",
        hovermode="x unified",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn")


def save_summary(path: Path, runs: pd.DataFrame) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "HP prediction errors",
            "Hydronic prediction errors",
            "Thermal prediction errors",
            "Setpoint-event response errors",
        ),
    )
    panels = (
        (1, 1, [
            ("HP expected", "hp_expected_pel_rmse_w_m2"),
            ("HP true mode", "hp_oracle_mode_pel_rmse_w_m2"),
            ("HP constant", "hp_constant_pel_rmse_w_m2"),
            ("Active oracle", "hp_active_pel_rmse_w_m2"),
            ("Active mean", "hp_active_train_mean_rmse_w_m2"),
        ]),
        (1, 2, [
            ("Q from Pel", "q_pel_history_rmse_w_m2"),
            ("Q + observed history", "q_observed_history_rmse_w_m2"),
            ("Q constant", "q_constant_rmse_w_m2"),
            ("Q zero", "q_zero_rmse_w_m2"),
        ]),
        (2, 1, [
            ("Tin tree one-step", "thermal_tree_one_step_rmse_c"),
            ("Tin ridge one-step", "thermal_ridge_one_step_rmse_c"),
            ("Tin ridge recursive", "thermal_ridge_recursive_rmse_c"),
            ("Tin persistence", "thermal_persistence_one_step_rmse_c"),
            ("Tin setpoint", "thermal_setpoint_rmse_c"),
        ]),
        (2, 2, [
            ("Pel oracle", "hp_event_rmse"),
            ("Pel true mode", "hp_oracle_mode_event_rmse"),
            ("Pel constant", "hp_constant_event_rmse"),
            ("Q from Pel", "q_pel_event_rmse"),
            ("Q + history", "q_observed_history_event_rmse"),
            ("Q constant", "q_constant_event_rmse"),
        ]),
    )
    for row, col, metrics in panels:
        for label, metric in metrics:
            figure.add_trace(
                go.Box(y=runs[metric], name=label, boxmean=True, boxpoints="outliers"),
                row=row,
                col=col,
            )
    figure.update_yaxes(title_text="RMSE [W/m2]", row=1, col=1)
    figure.update_yaxes(title_text="RMSE [W/m2]", row=1, col=2)
    figure.update_yaxes(title_text="RMSE [degC]", row=2, col=1)
    figure.update_yaxes(title_text="response RMSE [W/m2]", row=2, col=2)
    figure.update_layout(
        title="Building-specific blockwise oracle diagnostics",
        height=950,
        template="plotly_white",
        showlegend=False,
    )
    figure.write_html(path, include_plotlyjs="cdn")


def load_profile_ids(args: argparse.Namespace) -> list[int]:
    if args.profile_ids:
        return sorted(set(int(value) for value in args.profile_ids))
    if args.source_manifest is None or not args.source_manifest.exists():
        raise FileNotFoundError(
            "Pass --profile-ids or an existing --source-manifest containing profile_ids"
        )
    payload = json.loads(args.source_manifest.read_text())
    return [int(value) for value in payload["profile_ids"]]


def run_diagnostics(args: argparse.Namespace) -> None:
    profile_ids = load_profile_ids(args)
    profiles = load_profiles(SimpleNamespace(profile_ids=profile_ids, dataset=args.dataset))
    prepared = prepare_profiles(profiles, train_fraction=args.train_fraction)
    local_profiles = [value.local for value in prepared]
    history_steps = _history_steps(args.history_hours, args.dt_hours)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    plot_payloads: list[tuple[float, ClosedLoopProfile, dict[str, Any], dict[str, np.ndarray]]] = []
    for index, profile in enumerate(local_profiles):
        print(f"fitting_oracles profile={profile.profile_id} ({index + 1}/{len(local_profiles)})")
        row, traces = fit_profile(
            profile,
            train_fraction=args.train_fraction,
            history_steps=history_steps,
            args=args,
            seed=args.seed + index * 10,
        )
        rows.append(row)
        score = float(row["thermal_ridge_recursive_rmse_c"])
        plot_payloads.append((score, profile, row, traces))
        print(
            f"profile={profile.profile_id} hp_rmse={row['hp_expected_pel_rmse_w_m2']:.3f} "
            f"q_pel_rmse={row['q_pel_history_rmse_w_m2']:.3f} "
            f"q_history_rmse={row['q_observed_history_rmse_w_m2']:.3f} "
            f"thermal_one_step_rmse={row['thermal_tree_one_step_rmse_c']:.3f} "
            f"thermal_recursive_rmse={row['thermal_ridge_recursive_rmse_c']:.3f}"
        )

    runs = pd.DataFrame(rows).sort_values("profile_id").reset_index(drop=True)
    runs_path = args.output_dir / "blockwise_oracle_runs.csv"
    runs.to_csv(runs_path, index=False)
    numeric = runs.select_dtypes(include=[np.number]).drop(columns=["profile_id", "steps", "split_index"])
    summary = numeric.agg(["mean", "median", "std"]).T.reset_index(names="metric")
    summary_path = args.output_dir / "blockwise_oracle_summary.csv"
    summary.to_csv(summary_path, index=False)
    html_path = args.output_dir / "blockwise_oracle_summary.html"
    save_summary(html_path, runs)

    plot_dir = args.output_dir / "profile_plots"
    finite_payloads = [value for value in plot_payloads if np.isfinite(value[0])]
    finite_payloads.sort(key=lambda value: value[0], reverse=True)
    for _, profile, row, traces in finite_payloads[: args.num_profile_plots]:
        save_profile_plot(
            plot_dir / f"blockwise_oracle_{profile.profile_id}.html",
            profile,
            row,
            traces,
        )
    manifest = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "profile_ids": profile_ids,
        "history_steps": list(history_steps),
        "oracle_semantics": {
            "hp": "observed Tin and observed causal Pel/mode history",
            "qroom_pel": "observed current and causal Pel history",
            "qroom_observed_history": "observed Pel plus observed causal Qroom history",
            "thermal": "observed Qroom; one-step uses observed Tin, recursive uses predicted Tin",
        },
    }
    (args.output_dir / "blockwise_oracle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(f"saved_blockwise_runs={runs_path}")
    print(f"saved_blockwise_summary={summary_path}")
    print(f"saved_blockwise_plot={html_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--profile-ids", type=int, nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator/blockwise_oracles"))
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--history-hours", type=float, nargs="+", default=(0.5, 1.0, 3.0, 6.0))
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--heat-on-threshold-w-m2", type=float, default=1e-6)
    parser.add_argument("--setpoint-change-threshold-c", type=float, default=0.05)
    parser.add_argument("--event-horizon-hours", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--thermal-ridge", type=float, default=10.0)
    parser.add_argument("--recursive-temperature-guard-c", type=float, default=1000.0)
    parser.add_argument("--num-profile-plots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be in (0, 1)")
    if args.dt_hours <= 0.0 or any(value <= 0.0 for value in args.history_hours):
        parser.error("--dt-hours and --history-hours must be positive")
    if args.max_iter < 1 or args.max_leaf_nodes < 2 or args.min_samples_leaf < 1:
        parser.error("invalid gradient-boosting complexity")
    return args


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
