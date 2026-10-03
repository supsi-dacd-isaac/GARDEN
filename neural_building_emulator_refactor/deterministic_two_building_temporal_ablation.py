"""Fast deterministic model-class test on two policy-specific buildings.

Each candidate is fitted locally, without metadata conditioning. The first
half of each annual trajectory is used for training and the second half is one
autonomous test rollout. One building supplies setpoint-identification data;
another supplies force-off data. This is deliberately a small architecture
screen, not a replacement for global-model validation.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .single_building_architecture_search import (
    HP_CHOICES,
    HYDRONIC_CHOICES,
    THERMAL_CHOICES,
    FORCE_OFF_INDEX,
    Architecture,
    PolicyProfile,
    Segment,
    _caps,
    _evaluate_candidate,
    _fit_scales,
    _load_policy_profile,
    _train_candidate,
    _training_windows,
)


COLORS = {
    "hyd=buffer__thermal=bounded_mlp": "#68757d",
    "hyd=buffer__thermal=passive_rc": "#d17837",
    "hyd=leaky_emitter__thermal=bounded_mlp": "#187b78",
    "hyd=buffer__thermal=state_aware_mlp": "#3d5a80",
    "hyd=leaky_emitter__thermal=state_aware_mlp": "#8b5e9f",
    "hyd=buffer__thermal=setpoint_error_ssm": "#d1495b",
    "hyd=leaky_emitter__thermal=setpoint_error_ssm": "#edae49",
    "hyd=positive_convolution__thermal=stable_rc": "#2a9d8f",
    "hyd=positive_convolution__thermal=stable_rc_narx": "#e76f51",
    "hyd=leaky_emitter__thermal=passive_rc": "#6c55a3",
    "hyd=causal_filter__thermal=bounded_mlp": "#355ea8",
    "hyd=causal_filter__thermal=passive_rc": "#bf4d5d",
}
SIGNALS = ((0, "Tin", "C"), (1, "Qroom", "W/m2"), (2, "Pel_SH", "W/m2"))


def _label(architecture: Architecture) -> str:
    return f"hyd={architecture.hydronic}__thermal={architecture.thermal}"


def _temporal_segments(policy_profile: PolicyProfile, train_fraction: float) -> list[Segment]:
    total = int(policy_profile.profile.targets.shape[0])
    split = int(round(total * train_fraction))
    steps_per_day = 96
    split = max(steps_per_day, min(total - steps_per_day, split))
    split -= split % steps_per_day
    return [
        Segment(policy_profile.policy, 0, split, False),
        Segment(policy_profile.policy, split, total, True),
    ]


def _severe_force_off_metrics(trace: dict[str, Any]) -> dict[str, float]:
    inputs = np.asarray(trace["inputs"])
    target = np.asarray(trace["target"])
    prediction = np.asarray(trace["prediction"])
    force = inputs[:, FORCE_OFF_INDEX] > 0.5
    starts = np.flatnonzero(force & ~np.r_[False, force[:-1]])
    pre_steps, post_steps = 4, 16
    starts = starts[(starts >= pre_steps) & (starts + post_steps <= len(force))]
    true_drop = np.asarray(
        [
            target[start : start + post_steps, 0].min()
            - target[start - pre_steps : start, 0].mean()
            for start in starts
        ]
    )
    severe = true_drop < -0.5
    starts = starts[severe]
    true_drop = true_drop[severe]
    if not starts.size:
        return {
            "severe_force_off_events": 0.0,
            "severe_force_off_true_drop_c": float("nan"),
            "severe_force_off_predicted_drop_c": float("nan"),
            "severe_force_off_drop_rmse_c": float("nan"),
            "severe_force_off_qroom_response_bias_w_m2": float("nan"),
        }
    predicted_drop = np.asarray(
        [
            prediction[start : start + post_steps, 0].min()
            - prediction[start - pre_steps : start, 0].mean()
            for start in starts
        ]
    )

    def response(values: np.ndarray, channel: int) -> np.ndarray:
        return np.asarray(
            [
                values[start : start + post_steps, channel].mean()
                - values[start - pre_steps : start, channel].mean()
                for start in starts
            ]
        )

    return {
        "severe_force_off_events": float(starts.size),
        "severe_force_off_true_drop_c": float(np.mean(true_drop)),
        "severe_force_off_predicted_drop_c": float(np.mean(predicted_drop)),
        "severe_force_off_drop_rmse_c": float(
            np.sqrt(np.mean(np.square(predicted_drop - true_drop)))
        ),
        "severe_force_off_qroom_response_bias_w_m2": float(
            np.mean(response(prediction, 1) - response(target, 1))
        ),
    }


def _fit_case(
    policy: str,
    dataset: Path,
    profile_id: int,
    architectures: Sequence[Architecture],
    args: argparse.Namespace,
    objective_variants: Sequence[tuple[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    policy_profile = _load_policy_profile(dataset, profile_id, policy=policy)
    segments = _temporal_segments(policy_profile, args.train_fraction)
    input_scale, target_scale = _fit_scales((policy_profile,), segments)
    caps = _caps((policy_profile,), segments)
    train_inputs, train_targets, train_initials, _ = _training_windows(
        (policy_profile,),
        segments,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    variants = objective_variants or (
        ("default", float(getattr(args, "event_tin_loss_weight", 0.0))),
    )
    print(
        f"case={policy} profile={profile_id} train_steps={segments[0].end} "
        f"test_steps={segments[1].end - segments[1].start} "
        f"train_windows={len(train_inputs)} candidates="
        f"{len(architectures) * len(variants)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    traces: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    total_candidates = len(architectures) * len(variants)
    candidate_index = 0
    for architecture in architectures:
        for objective_name, event_tin_loss_weight in variants:
            candidate_index += 1
            candidate_args = copy.copy(args)
            candidate_args.event_tin_loss_weight = event_tin_loss_weight
            print(
                f"starting case={policy} candidate={candidate_index}/"
                f"{total_candidates} architecture={_label(architecture)} "
                f"objective={objective_name} "
                f"event_tin_loss_weight={event_tin_loss_weight}",
                flush=True,
            )
            params, train_metrics = _train_candidate(
                architecture,
                train_inputs,
                train_targets,
                train_initials,
                args=candidate_args,
                input_scale=input_scale,
                target_scale=target_scale,
                caps=caps,
                seed=args.seed,
            )
            test_metrics, candidate_traces = _evaluate_candidate(
                params,
                architecture,
                (policy_profile,),
                segments,
                args=candidate_args,
                input_scale=input_scale,
                target_scale=target_scale,
                caps=caps,
            )
            trace = candidate_traces[0]
            event_metrics = (
                _severe_force_off_metrics(trace)
                if policy == "force_off"
                else {}
            )
            base_label = (
                architecture.name
                if getattr(args, "include_hp_in_architecture_label", False)
                else _label(architecture)
            )
            label = (
                f"{base_label}__objective={objective_name}"
                if objective_variants is not None
                else base_label
            )
            row = {
                "policy": policy,
                "profile_id": profile_id,
                "architecture": label,
                "objective": objective_name,
                "event_tin_loss_weight": event_tin_loss_weight,
                **asdict(architecture),
                **train_metrics,
                **test_metrics,
                **event_metrics,
            }
            rows.append(row)
            traces[label] = trace
            print(
                f"finished case={policy} architecture={label} "
                f"Tin_RMSE={row['temperature_rmse_c']:.4f} "
                f"Q_RMSE={row['qroom_rmse_w_m2']:.4f} "
                f"Pel_RMSE={row['pel_rmse_w_m2']:.4f} "
                f"capacity_h={row['hydronic_capacity_hours']:.3f} "
                f"capacity_wh_m2={row['hydronic_capacity_wh_m2']:.2f} "
                f"minutes={(time.perf_counter() - started) / 60.0:.2f}",
                flush=True,
            )
    context = {
        "split_index": segments[0].end,
        "train_windows": int(len(train_inputs)),
        "input_scale": {"mean": input_scale.mean.tolist(), "std": input_scale.std.tolist()},
        "target_scale": {"mean": target_scale.mean.tolist(), "std": target_scale.std.tolist()},
        "caps": {"pel_w_m2": caps[0], "qroom_w_m2": caps[1], "energy_wh_m2": caps[2]},
    }
    return rows, traces, context


def _write_trace_dashboard(
    traces: dict[str, dict[str, dict[str, Any]]],
    profile_ids: dict[str, int],
    output_path: Path,
) -> None:
    policies = ("setpoint", "force_off")
    figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes="columns",
        subplot_titles=tuple(
            f"{policy} ({profile_ids[policy]}): {signal}"
            for _, signal, _ in SIGNALS
            for policy in policies
        ),
    )
    for column, policy in enumerate(policies, start=1):
        policy_traces = traces[policy]
        reference = next(iter(policy_traces.values()))
        for row, (channel, signal, unit) in enumerate(SIGNALS, start=1):
            figure.add_trace(
                go.Scattergl(
                    x=reference["datetime"],
                    y=reference["target"][:, channel],
                    name="EnergyPlus",
                    legendgroup="truth",
                    line={"color": "#15191c", "width": 1.5},
                    showlegend=column == 1 and row == 1,
                ),
                row=row,
                col=column,
            )
            for label, trace in policy_traces.items():
                figure.add_trace(
                    go.Scattergl(
                        x=trace["datetime"],
                        y=trace["prediction"][:, channel],
                        name=label,
                        legendgroup=label,
                        line={"color": COLORS.get(label, "#777"), "width": 1.0},
                        showlegend=column == 1 and row == 1,
                        hovertemplate=(
                            f"model={label}<br>time=%{{x}}<br>{signal}=%{{y:.4f}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=column,
                )
            if column == 1:
                figure.update_yaxes(title_text=unit, row=row, col=column)
    figure.update_layout(
        title="Building-local temporal holdout: autonomous July-December rollouts",
        template="plotly_white",
        height=1150,
        width=1800,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
        legend={"orientation": "h", "y": -0.06},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def _aligned_events(
    trace: dict[str, Any],
    policy: str,
    *,
    pre_steps: int = 4,
    post_steps: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.asarray(trace["inputs"])
    if policy == "force_off":
        treatment = inputs[:, FORCE_OFF_INDEX]
        starts = np.flatnonzero(np.diff(treatment, prepend=treatment[0]) > 0.5)
        signs = np.ones(starts.shape, dtype=np.float32)
    else:
        delta = np.diff(inputs[:, 0], prepend=inputs[0, 0])
        starts = np.flatnonzero(np.abs(delta) >= 0.05)
        signs = np.sign(delta[starts]).astype(np.float32)
    valid = (starts >= pre_steps) & (starts + post_steps <= len(inputs))
    starts, signs = starts[valid], signs[valid]

    def collect(values: np.ndarray) -> np.ndarray:
        rows = []
        for start, sign in zip(starts, signs):
            segment = values[start - pre_steps : start + post_steps]
            baseline = np.mean(values[start - pre_steps : start], axis=0)
            rows.append((segment - baseline) * sign)
        return np.stack(rows) if rows else np.empty((0, pre_steps + post_steps, 3))

    return starts, collect(np.asarray(trace["target"])), collect(np.asarray(trace["prediction"]))


def _write_event_dashboard(
    traces: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
    dt_hours: float,
) -> None:
    policies = ("setpoint", "force_off")
    pre_steps, post_steps = 4, 24
    hours = np.arange(-pre_steps, post_steps) * dt_hours
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=tuple(
            f"{policy}: aligned {signal} response"
            for _, signal, _ in SIGNALS
            for policy in policies
        ),
    )
    for column, policy in enumerate(policies, start=1):
        policy_traces = traces[policy]
        reference = next(iter(policy_traces.values()))
        _, truth_events, _ = _aligned_events(
            reference, policy, pre_steps=pre_steps, post_steps=post_steps
        )
        for row, (channel, signal, unit) in enumerate(SIGNALS, start=1):
            if truth_events.size:
                figure.add_trace(
                    go.Scatter(
                        x=hours,
                        y=np.mean(truth_events[:, :, channel], axis=0),
                        name="EnergyPlus",
                        legendgroup="truth",
                        line={"color": "#15191c", "width": 2.0},
                        showlegend=column == 1 and row == 1,
                    ),
                    row=row,
                    col=column,
                )
            for label, trace in policy_traces.items():
                starts, _, predicted_events = _aligned_events(
                    trace, policy, pre_steps=pre_steps, post_steps=post_steps
                )
                if not predicted_events.size:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=hours,
                        y=np.mean(predicted_events[:, :, channel], axis=0),
                        name=label,
                        legendgroup=label,
                        line={"color": COLORS.get(label, "#777"), "width": 1.4},
                        showlegend=column == 1 and row == 1,
                        customdata=np.full(hours.shape, len(starts)),
                        hovertemplate=(
                            f"model={label}<br>hours=%{{x:.2f}}<br>response=%{{y:.4f}}"
                            "<br>events=%{customdata}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=column,
                )
            figure.add_vline(x=0.0, line={"color": "#777", "dash": "dot"}, row=row, col=column)
            if column == 1:
                figure.update_yaxes(title_text=unit, row=row, col=column)
    figure.update_layout(
        title="Held-out intervention responses (setpoint responses oriented upward)",
        template="plotly_white",
        height=1100,
        width=1700,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
        legend={"orientation": "h", "y": -0.06},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def _write_metric_dashboard(results: pd.DataFrame, output_path: Path) -> None:
    panels = (
        ("temperature_rmse_c", "Tin RMSE [C]"),
        ("qroom_rmse_w_m2", "Qroom RMSE [W/m2]"),
        ("pel_rmse_w_m2", "Pel RMSE [W/m2]"),
        ("selection_score", "Composite score"),
        ("force_off_temperature_response_rmse", "Force-off Tin response RMSE [C]"),
        ("severe_force_off_drop_rmse_c", "Severe force-off drop RMSE [C]"),
    )
    figure = make_subplots(rows=2, cols=3, subplot_titles=tuple(title for _, title in panels))
    for index, (metric, _) in enumerate(panels):
        row, column = divmod(index, 3)
        for architecture, group in results.groupby("architecture", sort=False):
            figure.add_trace(
                go.Bar(
                    x=group["policy"],
                    y=group[metric] if metric in group else np.full(len(group), np.nan),
                    name=architecture,
                    legendgroup=architecture,
                    marker_color=COLORS.get(architecture, "#777"),
                    showlegend=index == 0,
                    hovertemplate=(
                        f"model={architecture}<br>policy=%{{x}}<br>{metric}=%{{y:.4f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row + 1,
                col=column + 1,
            )
    figure.update_layout(
        title="Fast deterministic two-building temporal ablation",
        template="plotly_white",
        barmode="group",
        height=900,
        width=1750,
        hoverlabel={"namelength": -1},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.architectures:
        architectures = []
        for spec in args.architectures:
            values = spec.split(":")
            if len(values) != 3:
                raise ValueError(
                    "Architecture specifications must use hp:hydronic:thermal"
                )
            architectures.append(Architecture(*values))
    else:
        architectures = [
            Architecture(hp, hydronic, thermal)
            for hp, hydronic, thermal in itertools.product(
                args.hp_architectures,
                args.hydronic_architectures,
                args.thermal_architectures,
            )
        ]
    cases = (
        ("setpoint", args.setpoint_dataset, args.setpoint_profile_id),
        ("force_off", args.force_off_dataset, args.force_off_profile_id),
    )
    all_rows: list[dict[str, Any]] = []
    all_traces: dict[str, dict[str, dict[str, Any]]] = {}
    contexts: dict[str, Any] = {}
    started = time.perf_counter()
    for policy, dataset, profile_id in cases:
        rows, traces, context = _fit_case(
            policy, dataset, profile_id, architectures, args
        )
        all_rows.extend(rows)
        all_traces[policy] = traces
        contexts[policy] = context
    results = pd.DataFrame(all_rows)
    results.to_csv(args.output_dir / "temporal_ablation_metrics.csv", index=False)
    _write_trace_dashboard(
        all_traces,
        {
            "setpoint": args.setpoint_profile_id,
            "force_off": args.force_off_profile_id,
        },
        args.output_dir / "temporal_holdout_traces.html",
    )
    _write_event_dashboard(
        all_traces,
        args.output_dir / "intervention_response_comparison.html",
        args.dt_hours,
    )
    _write_metric_dashboard(
        results,
        args.output_dir / "temporal_ablation_dashboard.html",
    )
    manifest = {
        "metadata_conditioning": False,
        "temporal_split": "first_fraction_train_second_fraction_test",
        "profile_ids": {
            "setpoint": args.setpoint_profile_id,
            "force_off": args.force_off_profile_id,
        },
        "datasets": {
            "setpoint": str(args.setpoint_dataset),
            "force_off": str(args.force_off_dataset),
        },
        "architectures": [asdict(value) for value in architectures],
        "contexts": contexts,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "temporal_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(
        f"saved_temporal_ablation={args.output_dir} "
        f"elapsed_minutes={manifest['elapsed_minutes']:.2f}",
        flush=True,
    )
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setpoint-dataset",
        type=Path,
        default=Path("data/control=setpoint_identification_dhw_fixed"),
    )
    parser.add_argument(
        "--force-off-dataset",
        type=Path,
        default=Path("data/control=force_off"),
    )
    parser.add_argument("--setpoint-profile-id", type=int, default=192033312)
    parser.add_argument("--force-off-profile-id", type=int, default=11132764)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--dt-hours", type=float, default=0.25)
    parser.add_argument("--temperature-delta-cap", type=float, default=2.0)
    parser.add_argument(
        "--hp-recovery-emergency-threshold-c",
        type=float,
        default=1.0,
    )
    parser.add_argument("--hydronic-capacity-min-hours", type=float, default=0.25)
    parser.add_argument("--hydronic-capacity-max-hours", type=float, default=24.0)
    parser.add_argument("--hydronic-capacity-init-hours", type=float, default=2.0)
    parser.add_argument("--mode-loss-weight", type=float, default=0.1)
    parser.add_argument("--event-horizon-hours", type=float, default=3.0)
    parser.add_argument("--event-tin-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--event-tin-loss-horizon-hours", type=float, default=3.0
    )
    parser.add_argument(
        "--event-tin-loss-setpoint-threshold-c", type=float, default=0.05
    )
    parser.add_argument("--event-score-weight", type=float, default=0.5)
    parser.add_argument("--warmup-hours", type=float, default=168.0)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=(),
        metavar="HP:HYDRONIC:THERMAL",
        help="Exact candidate triples; when supplied these replace the factorial options.",
    )
    parser.add_argument(
        "--hp-architectures",
        nargs="+",
        choices=HP_CHOICES,
        default=("hurdle",),
    )
    parser.add_argument(
        "--hydronic-architectures",
        nargs="+",
        choices=HYDRONIC_CHOICES,
        default=("buffer", "leaky_emitter"),
    )
    parser.add_argument(
        "--thermal-architectures",
        nargs="+",
        choices=THERMAL_CHOICES,
        default=("bounded_mlp", "passive_rc"),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "deterministic_two_building_temporal_ablation"
        ),
    )
    args = parser.parse_args()
    if not 0.1 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction must lie in [0.1, 0.9]")
    if args.epochs < 1 or args.batch_size < 1 or args.sequence_length < 2:
        parser.error("epochs, batch size and sequence length must be positive")
    if args.stride < 1 or args.warmup_hours < 0.0:
        parser.error("stride must be positive and warmup hours non-negative")
    if args.hp_recovery_emergency_threshold_c < 0.0:
        parser.error("--hp-recovery-emergency-threshold-c must be non-negative")
    if not (
        0.0 < args.hydronic_capacity_min_hours
        < args.hydronic_capacity_init_hours
        < args.hydronic_capacity_max_hours
    ):
        parser.error(
            "hydronic capacity bounds must satisfy 0 < min < init < max"
        )
    for spec in args.architectures:
        values = spec.split(":")
        if len(values) != 3:
            parser.error(
                f"Invalid --architectures value {spec!r}; expected hp:hydronic:thermal"
            )
        hp, hydronic, thermal = values
        if hp not in HP_CHOICES or hydronic not in HYDRONIC_CHOICES or thermal not in THERMAL_CHOICES:
            parser.error(f"Unsupported architecture specification: {spec}")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
