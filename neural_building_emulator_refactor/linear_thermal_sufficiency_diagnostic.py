"""Test whether observed Qroom is sufficient for a simple autonomous Tin model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import lsq_linear

from .deterministic_two_building_temporal_ablation import _temporal_segments
from .single_building_architecture_search import (
    INTERNAL_INDEX,
    OCCUPANCY_INDEX,
    SOLAR_INDEX,
    TOUT_INDEX,
    VENT_INDEX,
    _load_policy_profile,
)


FEATURE_NAMES = (
    "intercept",
    "Tout_minus_Tin",
    "Qroom",
    "solar",
    "internal_gain",
    "occupancy",
    "vent_times_Tout_minus_Tin",
)


def _features(inputs: np.ndarray, temperature: np.ndarray, qroom: np.ndarray) -> np.ndarray:
    envelope_gap = inputs[:, TOUT_INDEX] - temperature
    return np.column_stack(
        (
            np.ones(len(inputs)),
            envelope_gap,
            qroom,
            inputs[:, SOLAR_INDEX],
            inputs[:, INTERNAL_INDEX],
            inputs[:, OCCUPANCY_INDEX],
            inputs[:, VENT_INDEX] * envelope_gap,
        )
    ).astype(np.float64)


def _fit(
    inputs: np.ndarray,
    current_temperature: np.ndarray,
    next_temperature: np.ndarray,
    qroom: np.ndarray,
    *,
    ridge: float,
    constrained: bool,
) -> tuple[np.ndarray, np.ndarray]:
    design = _features(inputs, current_temperature, qroom)
    scales = np.maximum(np.sqrt(np.mean(np.square(design), axis=0)), 1e-8)
    scales[0] = 1.0
    scaled = design / scales
    target = np.asarray(next_temperature - current_temperature, dtype=np.float64)
    penalty = np.sqrt(ridge) * np.eye(scaled.shape[1])
    penalty[0, 0] = 0.0
    augmented_design = np.vstack((scaled, penalty))
    augmented_target = np.concatenate((target, np.zeros(scaled.shape[1])))
    if constrained:
        lower = np.zeros(scaled.shape[1])
        upper = np.full(scaled.shape[1], np.inf)
        lower[0] = -np.inf
        result = lsq_linear(
            augmented_design,
            augmented_target,
            bounds=(lower, upper),
            lsmr_tol="auto",
        )
        scaled_coefficients = result.x
    else:
        scaled_coefficients = np.linalg.lstsq(
            augmented_design, augmented_target, rcond=None
        )[0]
    return scaled_coefficients / scales, scales


def _rollout(
    coefficients: np.ndarray,
    inputs: np.ndarray,
    qroom: np.ndarray,
    initial_temperature: float,
    *,
    delta_cap_c: float,
) -> np.ndarray:
    temperature = float(initial_temperature)
    output = np.empty(len(inputs), dtype=np.float64)
    for step in range(len(inputs)):
        design = _features(
            inputs[step : step + 1],
            np.asarray([temperature]),
            qroom[step : step + 1],
        )[0]
        delta = float(np.clip(design @ coefficients, -delta_cap_c, delta_cap_c))
        temperature = float(np.clip(temperature + delta, 5.0, 40.0))
        output[step] = temperature
    return output


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_profile = _load_policy_profile(args.dataset, args.profile_id, policy=args.policy)
    profile = policy_profile.profile
    train, test = _temporal_segments(policy_profile, args.train_fraction)

    train_inputs = np.asarray(profile.inputs[train.start : train.end])
    train_targets = np.asarray(profile.targets[train.start : train.end])
    train_current = np.asarray(profile.initial_temperature[train.start : train.end, 0])
    test_inputs = np.asarray(profile.inputs[test.start : test.end])
    test_targets = np.asarray(profile.targets[test.start : test.end])
    test_current = np.asarray(profile.initial_temperature[test.start : test.end, 0])

    traces: dict[str, np.ndarray] = {}
    metrics: dict[str, object] = {
        "profile_id": args.profile_id,
        "policy": args.policy,
        "train_start": train.start,
        "train_end": train.end,
        "test_start": test.start,
        "test_end": test.end,
    }
    for name, constrained in (("positive_rc", True), ("unconstrained_arx", False)):
        coefficients, _ = _fit(
            train_inputs,
            train_current,
            train_targets[:, 0],
            train_targets[:, 1],
            ridge=args.ridge,
            constrained=constrained,
        )
        one_step_delta = _features(test_inputs, test_current, test_targets[:, 1]) @ coefficients
        one_step = test_current + np.clip(
            one_step_delta, -args.temperature_delta_cap, args.temperature_delta_cap
        )
        autonomous = _rollout(
            coefficients,
            test_inputs,
            test_targets[:, 1],
            float(profile.initial_temperature[test.start, 0]),
            delta_cap_c=args.temperature_delta_cap,
        )
        traces[name] = autonomous
        metrics[name] = {
            "one_step_temperature_rmse_c": _rmse(one_step, test_targets[:, 0]),
            "autonomous_temperature_rmse_c": _rmse(autonomous, test_targets[:, 0]),
            "autonomous_temperature_bias_c": float(np.mean(autonomous - test_targets[:, 0])),
            "coefficients": dict(zip(FEATURE_NAMES, (float(v) for v in coefficients), strict=True)),
        }

    datetime = np.asarray(profile.datetime[test.start : test.end])
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Autonomous Tin with observed Qroom", "Tin error"),
    )
    figure.add_trace(
        go.Scatter(x=datetime, y=test_targets[:, 0], name="EnergyPlus", line={"color": "#202124", "width": 2}),
        row=1,
        col=1,
    )
    colors = {"positive_rc": "#187b78", "unconstrained_arx": "#d17837"}
    for name, prediction in traces.items():
        figure.add_trace(
            go.Scatter(x=datetime, y=prediction, name=name, line={"color": colors[name], "width": 1.2}),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=prediction - test_targets[:, 0],
                name=f"{name} error",
                legendgroup=name,
                showlegend=False,
                line={"color": colors[name], "width": 1.0},
            ),
            row=2,
            col=1,
        )
    figure.add_hline(y=0.0, line={"color": "#666", "dash": "dot"}, row=2, col=1)
    figure.update_yaxes(title_text="Tin [C]", row=1, col=1)
    figure.update_yaxes(title_text="Error [C]", row=2, col=1)
    figure.update_layout(
        title=f"Linear thermal sufficiency: {args.policy} / {args.profile_id}",
        height=850,
        hovermode="x unified",
        template="plotly_white",
    )
    html_path = args.output_dir / "linear_thermal_sufficiency.html"
    json_path = args.output_dir / "linear_thermal_sufficiency_metrics.json"
    figure.write_html(html_path, include_plotlyjs="cdn")
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"saved_linear_thermal_sufficiency={html_path}", flush=True)
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile-id", type=int, required=True)
    parser.add_argument("--policy", choices=("setpoint", "force_off"), required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--temperature-delta-cap", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must lie strictly between zero and one")
    if args.ridge < 0.0 or args.temperature_delta_cap <= 0.0:
        parser.error("--ridge must be non-negative and --temperature-delta-cap positive")
    return args


if __name__ == "__main__":
    run(parse_args())
