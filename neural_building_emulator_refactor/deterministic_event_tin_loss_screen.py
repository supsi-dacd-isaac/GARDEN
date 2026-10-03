"""Paired screen for the event-aligned Tin response training objective.

The HP controller and hydronic buffer are held fixed at the legacy setting.
The factorial crosses legacy versus causal-Q thermal dynamics with trajectory-
only versus event-aligned Tin training. Each model is initialized identically,
fitted on January-June, and rolled autonomously over July-December.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .deterministic_paired_architecture_screen import _selected_profile_ids
from .deterministic_two_building_temporal_ablation import (
    COLORS,
    _fit_case,
    _write_trace_dashboard,
)
from .single_building_architecture_search import Architecture


LEGACY = Architecture("hurdle", "buffer", "bounded_mlp")
CAUSAL_Q = Architecture("hurdle", "buffer", "stable_rc_narx_positive_q")
ARCHITECTURES = (LEGACY, CAUSAL_Q)
OBJECTIVES = (("trajectory_only", 0.0), ("event_aligned", 1.0))

MODEL_NAMES = {
    ("bounded_mlp", "trajectory_only"): "Legacy / trajectory only",
    ("bounded_mlp", "event_aligned"): "Legacy / event aligned",
    (
        "stable_rc_narx_positive_q",
        "trajectory_only",
    ): "Causal Q / trajectory only",
    (
        "stable_rc_narx_positive_q",
        "event_aligned",
    ): "Causal Q / event aligned",
}
MODEL_ORDER = tuple(MODEL_NAMES.values())
BASELINE = MODEL_NAMES[("bounded_mlp", "trajectory_only")]
MODEL_COLORS = {
    "Legacy / trajectory only": "#68757d",
    "Legacy / event aligned": "#8b5e9f",
    "Causal Q / trajectory only": "#355ea8",
    "Causal Q / event aligned": "#e76f51",
}

for architecture in ARCHITECTURES:
    for objective, _ in OBJECTIVES:
        display = MODEL_NAMES[(architecture.thermal, objective)]
        COLORS[f"{architecture.name}__objective={objective}"] = MODEL_COLORS[
            display
        ]


def _prepare_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    results["model"] = [
        MODEL_NAMES[(thermal, objective)]
        for thermal, objective in zip(results.thermal, results.objective)
    ]
    results["temperature_abs_bias_c"] = results.temperature_bias_c.abs()
    for signal in ("qroom", "pel"):
        ratio = results[f"{signal}_energy_ratio"].clip(lower=1e-8)
        results[f"{signal}_energy_log_error"] = np.abs(np.log(ratio))
    return results


METRICS = {
    "temperature_rmse_c": "Tin RMSE [C]",
    "qroom_rmse_w_m2": "Qroom RMSE [W/m2]",
    "pel_rmse_w_m2": "Pel RMSE [W/m2]",
    "temperature_abs_bias_c": "Absolute Tin bias [C]",
    "qroom_energy_log_error": "|log Qroom energy ratio|",
    "pel_energy_log_error": "|log Pel energy ratio|",
    "setpoint_temperature_response_rmse": "Setpoint Tin response RMSE [C]",
    "force_off_temperature_response_rmse": "Force-off Tin response RMSE [C]",
    "severe_force_off_drop_rmse_c": "Severe force-off drop RMSE [C]",
}


def _summarize(results: pd.DataFrame) -> dict[str, Any]:
    aggregates: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for policy in ("setpoint", "force_off"):
        policy_rows = results[results.policy == policy]
        for metric, title in METRICS.items():
            for model in MODEL_ORDER:
                values = policy_rows.loc[
                    policy_rows.model == model, metric
                ].dropna()
                if len(values):
                    aggregates.append(
                        {
                            "policy": policy,
                            "metric": metric,
                            "title": title,
                            "model": model,
                            "count": int(len(values)),
                            "mean": float(values.mean()),
                            "median": float(values.median()),
                            "p95": float(values.quantile(0.95)),
                        }
                    )
            wide = policy_rows.pivot(
                index="profile_id", columns="model", values=metric
            )
            for model in MODEL_ORDER[1:]:
                comparison = wide[[BASELINE, model]].dropna()
                if not len(comparison):
                    continue
                delta = comparison[model] - comparison[BASELINE]
                paired.append(
                    {
                        "policy": policy,
                        "metric": metric,
                        "title": title,
                        "candidate": model,
                        "count": int(len(delta)),
                        "candidate_minus_baseline_mean": float(delta.mean()),
                        "candidate_minus_baseline_median": float(delta.median()),
                        "candidate_win_fraction": float(np.mean(delta < 0.0)),
                    }
                )
    return {"aggregates": aggregates, "paired_comparisons": paired}


def _write_dashboard(results: pd.DataFrame, output_path: Path) -> None:
    panels = tuple(METRICS.items())[:8]
    figure = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=tuple(title for _, title in panels),
    )
    for index, (metric, _) in enumerate(panels):
        row, column = divmod(index, 4)
        for policy in ("setpoint", "force_off"):
            for model in MODEL_ORDER:
                group = results[
                    (results.policy == policy) & (results.model == model)
                ]
                figure.add_trace(
                    go.Box(
                        x=np.full(len(group), policy),
                        y=group[metric],
                        name=model,
                        legendgroup=model,
                        marker_color=MODEL_COLORS[model],
                        boxpoints="all",
                        jitter=0.25,
                        pointpos=0.0,
                        showlegend=index == 0 and policy == "setpoint",
                        customdata=group.profile_id.astype(str),
                        hovertemplate=(
                            f"model={model}<br>policy={policy}"
                            "<br>EGID=%{customdata}<br>value=%{y:.4f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row + 1,
                    col=column + 1,
                )
    figure.update_layout(
        title="Event-aligned Tin objective: paired deterministic comparison",
        template="plotly_white",
        boxmode="group",
        height=900,
        width=2100,
        hoverlabel={"namelength": -1},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def _write_difference_dashboard(
    results: pd.DataFrame, output_path: Path
) -> None:
    panels = (
        ("temperature_rmse_c", "Delta Tin RMSE [C]"),
        ("qroom_rmse_w_m2", "Delta Qroom RMSE [W/m2]"),
        ("pel_rmse_w_m2", "Delta Pel RMSE [W/m2]"),
        ("qroom_energy_log_error", "Delta Qroom energy error"),
        ("setpoint_temperature_response_rmse", "Delta setpoint Tin response"),
        ("severe_force_off_drop_rmse_c", "Delta severe force-off drop"),
    )
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=tuple(title for _, title in panels),
    )
    for index, (metric, _) in enumerate(panels):
        row, column = divmod(index, 3)
        for model in MODEL_ORDER[1:]:
            for policy, symbol in (
                ("setpoint", "circle"),
                ("force_off", "diamond"),
            ):
                wide = results[results.policy == policy].pivot(
                    index="profile_id", columns="model", values=metric
                )
                comparison = wide[[BASELINE, model]].dropna()
                delta = comparison[model] - comparison[BASELINE]
                figure.add_trace(
                    go.Scatter(
                        x=np.full(len(delta), policy),
                        y=delta,
                        mode="markers",
                        name=model,
                        legendgroup=model,
                        marker={
                            "color": MODEL_COLORS[model],
                            "symbol": symbol,
                            "opacity": 0.7,
                        },
                        showlegend=index == 0 and policy == "setpoint",
                        customdata=delta.index.astype(str),
                        hovertemplate=(
                            f"model={model}<br>policy={policy}"
                            "<br>EGID=%{customdata}"
                            "<br>candidate-baseline=%{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=row + 1,
                    col=column + 1,
                )
        figure.add_hline(
            y=0.0,
            line={"color": "#555", "dash": "dot"},
            row=row + 1,
            col=column + 1,
        )
    figure.update_layout(
        title="Paired differences from legacy trajectory-only baseline",
        template="plotly_white",
        height=850,
        width=1800,
        hoverlabel={"namelength": -1},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.include_hp_in_architecture_label = True
    selected = _selected_profile_ids(args)
    rows: list[dict[str, Any]] = []
    focal_traces: dict[str, dict[str, dict[str, Any]]] = {}
    started = time.perf_counter()
    cases = (
        ("setpoint", args.setpoint_dataset, args.setpoint_focus_profile_id),
        ("force_off", args.force_off_dataset, args.force_off_focus_profile_id),
    )
    total = 2 * args.profiles_per_policy
    completed = 0
    for policy, dataset, focus_id in cases:
        for profile_id in selected[policy]:
            completed += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / max(completed - 1, 1) * (total - completed) / 60.0
            print(
                f"event_loss_screen profile={completed}/{total} "
                f"policy={policy} egid={profile_id} eta_minutes={eta:.1f}",
                flush=True,
            )
            case_rows, traces, _ = _fit_case(
                policy,
                dataset,
                int(profile_id),
                ARCHITECTURES,
                args,
                objective_variants=OBJECTIVES,
            )
            rows.extend(case_rows)
            if int(profile_id) == int(focus_id):
                focal_traces[policy] = traces

    results = _prepare_results(rows)
    results.to_csv(args.output_dir / "event_tin_loss_metrics.csv", index=False)
    summary = _summarize(results)
    summary.update(
        {
            "selected_profile_ids": selected,
            "architectures": [value.name for value in ARCHITECTURES],
            "objectives": [
                {"name": name, "event_tin_loss_weight": weight}
                for name, weight in OBJECTIVES
            ],
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    (args.output_dir / "event_tin_loss_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    _write_dashboard(results, args.output_dir / "event_tin_loss_dashboard.html")
    _write_difference_dashboard(
        results, args.output_dir / "event_tin_loss_differences.html"
    )
    if set(focal_traces) == {"setpoint", "force_off"}:
        _write_trace_dashboard(
            focal_traces,
            {
                "setpoint": args.setpoint_focus_profile_id,
                "force_off": args.force_off_focus_profile_id,
            },
            args.output_dir / "event_tin_loss_focal_traces.html",
        )
    print(
        f"saved_event_tin_loss_screen={args.output_dir} "
        f"elapsed_minutes={summary['elapsed_minutes']:.2f}",
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
    parser.add_argument("--setpoint-focus-profile-id", type=int, default=192033312)
    parser.add_argument("--force-off-focus-profile-id", type=int, default=11132764)
    parser.add_argument("--profiles-per-policy", type=int, default=20)
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
    parser.add_argument("--hp-recovery-emergency-threshold-c", type=float, default=1.0)
    parser.add_argument("--hydronic-capacity-min-hours", type=float, default=0.25)
    parser.add_argument("--hydronic-capacity-max-hours", type=float, default=24.0)
    parser.add_argument("--hydronic-capacity-init-hours", type=float, default=2.0)
    parser.add_argument("--mode-loss-weight", type=float, default=0.1)
    parser.add_argument("--event-horizon-hours", type=float, default=3.0)
    parser.add_argument("--event-score-weight", type=float, default=0.5)
    parser.add_argument("--event-tin-loss-horizon-hours", type=float, default=3.0)
    parser.add_argument(
        "--event-tin-loss-setpoint-threshold-c", type=float, default=0.05
    )
    parser.add_argument("--warmup-hours", type=float, default=168.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "deterministic_event_tin_loss_screen_20"
        ),
    )
    args = parser.parse_args()
    if args.profiles_per_policy < 2:
        parser.error("--profiles-per-policy must be at least 2")
    if not 0.1 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction must lie in [0.1, 0.9]")
    if args.event_tin_loss_horizon_hours <= 0.0:
        parser.error("event Tin loss horizon must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
