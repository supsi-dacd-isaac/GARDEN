"""Paired deterministic factorial screen for closed-loop HP architectures.

Each architecture is fitted independently to the first half of each building
trajectory and evaluated autonomously on the second half. The same buildings,
splits, initialization, and optimizer settings are used for all candidates.

The default 2x2 factorial keeps the learned energy buffer fixed and crosses
the legacy versus recovery-debt HP controller with the legacy versus causal
RC-NARX thermal block. This separates controller recovery from a thermal block
where Qroom has an explicit positive fast/slow response and cannot enter the
free neural residual.
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

from neural_building_emulator.dataset_adapter import read_entity_profile_ids

from .deterministic_two_building_temporal_ablation import (
    COLORS,
    _fit_case,
    _write_trace_dashboard,
)
from .single_building_architecture_search import Architecture


LEGACY = Architecture("hurdle", "buffer", "bounded_mlp")
THERMAL_ONLY = Architecture(
    "hurdle",
    "buffer",
    "stable_rc_narx_positive_q",
)
HP_HYDRONIC_ONLY = Architecture(
    "recovery_hurdle",
    "buffer",
    "bounded_mlp",
)
RECOVERY_NARX = Architecture(
    "recovery_hurdle",
    "buffer",
    "stable_rc_narx_positive_q",
)
ARCHITECTURES = (LEGACY, THERMAL_ONLY, HP_HYDRONIC_ONLY, RECOVERY_NARX)
MODEL_NAMES = {
    LEGACY.name: "Legacy",
    THERMAL_ONLY.name: "Legacy HP/buffer + causal RC-NARX",
    HP_HYDRONIC_ONLY.name: "Recovery debt + legacy thermal",
    RECOVERY_NARX.name: "Recovery debt + causal RC-NARX",
}
LEGACY_NAME = MODEL_NAMES[LEGACY.name]
FULL_CANDIDATE_NAME = MODEL_NAMES[RECOVERY_NARX.name]
MODEL_ORDER = tuple(MODEL_NAMES[architecture.name] for architecture in ARCHITECTURES)
MODEL_COLORS = {
    "Legacy": COLORS["hyd=buffer__thermal=bounded_mlp"],
    "Legacy HP/buffer + causal RC-NARX": "#355ea8",
    "Recovery debt + legacy thermal": "#2a9d8f",
    "Recovery debt + causal RC-NARX": "#e76f51",
}

# The shared full-profile renderer keys colors by hydronic/thermal labels.
COLORS.update(
    {
        LEGACY.name: MODEL_COLORS["Legacy"],
        THERMAL_ONLY.name: MODEL_COLORS[
            "Legacy HP/buffer + causal RC-NARX"
        ],
        HP_HYDRONIC_ONLY.name: MODEL_COLORS[
            "Recovery debt + legacy thermal"
        ],
        RECOVERY_NARX.name: MODEL_COLORS[
            "Recovery debt + causal RC-NARX"
        ],
    }
)


def _selected_profile_ids(args: argparse.Namespace) -> dict[str, list[int]]:
    setpoint_ids = set(
        map(int, read_entity_profile_ids(args.setpoint_dataset))
    )
    force_off_ids = set(
        map(int, read_entity_profile_ids(args.force_off_dataset))
    )
    if args.setpoint_focus_profile_id not in setpoint_ids:
        raise ValueError("Setpoint focus profile is absent from its dataset")
    if args.force_off_focus_profile_id not in force_off_ids:
        raise ValueError("Force-off focus profile is absent from its dataset")

    excluded = {
        args.setpoint_focus_profile_id,
        args.force_off_focus_profile_id,
    }
    common = np.asarray(
        sorted((setpoint_ids & force_off_ids) - excluded), dtype=np.int64
    )
    shared_count = max(0, args.profiles_per_policy - 1)
    if common.size < shared_count:
        raise ValueError(
            f"Only {common.size} non-focus profiles are shared; "
            f"need {shared_count}"
        )
    rng = np.random.default_rng(args.seed)
    shared = (
        rng.choice(common, size=shared_count, replace=False).astype(int).tolist()
        if shared_count
        else []
    )
    return {
        "setpoint": [args.setpoint_focus_profile_id, *shared],
        "force_off": [args.force_off_focus_profile_id, *shared],
    }


def _prepare_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    architecture_names = (
        "hp="
        + results["hp"].astype(str)
        + "__hyd="
        + results["hydronic"].astype(str)
        + "__thermal="
        + results["thermal"].astype(str)
    )
    results["model"] = architecture_names.map(MODEL_NAMES)
    if results["model"].isna().any():
        missing = sorted(architecture_names[results["model"].isna()].unique())
        raise ValueError(f"Missing display names for architectures: {missing}")
    results["temperature_abs_bias_c"] = results["temperature_bias_c"].abs()
    for signal in ("qroom", "pel"):
        ratio = results[f"{signal}_energy_ratio"].clip(lower=1e-8)
        results[f"{signal}_energy_log_error"] = np.abs(np.log(ratio))
    return results


def _paired_summary(results: pd.DataFrame) -> dict[str, Any]:
    metrics = {
        "temperature_rmse_c": "Tin RMSE",
        "qroom_rmse_w_m2": "Qroom RMSE",
        "pel_rmse_w_m2": "Pel RMSE",
        "temperature_abs_bias_c": "absolute Tin bias",
        "qroom_energy_log_error": "Qroom energy error",
        "pel_energy_log_error": "Pel energy error",
        "setpoint_temperature_response_rmse": "setpoint Tin response RMSE",
        "force_off_temperature_response_rmse": "force-off Tin response RMSE",
        "severe_force_off_drop_rmse_c": "severe force-off drop RMSE",
    }
    aggregates: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for policy in ("setpoint", "force_off"):
        policy_rows = results[results["policy"] == policy]
        for metric, title in metrics.items():
            if metric not in policy_rows:
                continue
            for model, group in policy_rows.groupby("model", sort=False):
                values = group[metric].dropna().to_numpy(dtype=float)
                if not values.size:
                    continue
                aggregates.append(
                    {
                        "policy": policy,
                        "metric": metric,
                        "title": title,
                        "model": model,
                        "count": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "p95": float(np.quantile(values, 0.95)),
                    }
                )
            wide = policy_rows.pivot(
                index="profile_id", columns="model", values=metric
            )
            for model in MODEL_ORDER[1:]:
                if LEGACY_NAME not in wide or model not in wide:
                    continue
                comparison = wide[[LEGACY_NAME, model]].dropna()
                if not len(comparison):
                    continue
                difference = comparison[model] - comparison[LEGACY_NAME]
                paired.append(
                    {
                        "policy": policy,
                        "metric": metric,
                        "title": title,
                        "candidate": model,
                        "count": int(len(difference)),
                        "candidate_minus_legacy_mean": float(difference.mean()),
                        "candidate_minus_legacy_median": float(
                            difference.median()
                        ),
                        "candidate_win_fraction": float(
                            np.mean(difference < 0.0)
                        ),
                    }
                )

    aggregate_frame = pd.DataFrame(aggregates)

    def median(policy: str, metric: str, model: str) -> float:
        row = aggregate_frame[
            (aggregate_frame.policy == policy)
            & (aggregate_frame.metric == metric)
            & (aggregate_frame.model == model)
        ]
        return float(row.iloc[0]["median"]) if len(row) else float("nan")

    def p95(policy: str, metric: str, model: str) -> float:
        row = aggregate_frame[
            (aggregate_frame.policy == policy)
            & (aggregate_frame.metric == metric)
            & (aggregate_frame.model == model)
        ]
        return float(row.iloc[0]["p95"]) if len(row) else float("nan")

    candidate_assessments: dict[str, Any] = {}
    for model in MODEL_ORDER[1:]:
        checks: dict[str, bool] = {}
        checks["tin_median_better_both_policies"] = all(
            median(policy, "temperature_rmse_c", model)
            < median(policy, "temperature_rmse_c", LEGACY_NAME)
            for policy in ("setpoint", "force_off")
        )
        checks["tin_p95_not_worse_both_policies"] = all(
            p95(policy, "temperature_rmse_c", model)
            <= p95(policy, "temperature_rmse_c", LEGACY_NAME)
            for policy in ("setpoint", "force_off")
        )
        checks["median_energy_error_not_worse"] = all(
            median(policy, metric, model)
            <= median(policy, metric, LEGACY_NAME)
            for policy in ("setpoint", "force_off")
            for metric in ("qroom_energy_log_error", "pel_energy_log_error")
        )
        checks["catastrophic_tin_fraction_not_worse"] = all(
            float(
                np.mean(
                    results[
                        (results.policy == policy)
                        & (results.model == model)
                    ].temperature_rmse_c
                    > 3.0
                )
            )
            <= float(
                np.mean(
                    results[
                        (results.policy == policy)
                        & (results.model == LEGACY_NAME)
                    ].temperature_rmse_c
                    > 3.0
                )
            )
            for policy in ("setpoint", "force_off")
        )
        checks["policy_response_not_worse"] = (
            median(
                "setpoint",
                "setpoint_temperature_response_rmse",
                model,
            )
            <= median(
                "setpoint",
                "setpoint_temperature_response_rmse",
                LEGACY_NAME,
            )
            and median(
                "force_off",
                "force_off_temperature_response_rmse",
                model,
            )
            <= median(
                "force_off",
                "force_off_temperature_response_rmse",
                LEGACY_NAME,
            )
        )
        candidate_assessments[model] = {
            "acceptance_checks": checks,
            "accepted_for_probabilistic_port": bool(all(checks.values())),
        }
    full_assessment = candidate_assessments[FULL_CANDIDATE_NAME]
    return {
        "aggregates": aggregates,
        "paired_comparisons": paired,
        "candidate_assessments": candidate_assessments,
        "acceptance_checks": full_assessment["acceptance_checks"],
        "accepted_for_probabilistic_port": full_assessment[
            "accepted_for_probabilistic_port"
        ],
    }


def _write_dashboard(
    results: pd.DataFrame,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    panels = (
        ("temperature_rmse_c", "Tin RMSE [C]"),
        ("qroom_rmse_w_m2", "Qroom RMSE [W/m2]"),
        ("pel_rmse_w_m2", "Pel RMSE [W/m2]"),
        ("temperature_abs_bias_c", "Absolute Tin bias [C]"),
        ("qroom_energy_log_error", "|log Qroom energy ratio|"),
        ("pel_energy_log_error", "|log Pel energy ratio|"),
        (
            "force_off_temperature_response_rmse",
            "Force-off Tin response RMSE [C]",
        ),
        ("severe_force_off_drop_rmse_c", "Severe-drop RMSE [C]"),
    )
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
                        hovertext=group.profile_id.astype(str),
                        hovertemplate=(
                            f"model={model}<br>policy={policy}"
                            "<br>EGID=%{hovertext}<br>value=%{y:.4f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row + 1,
                    col=column + 1,
                )
    accepted = [
        model
        for model, assessment in summary["candidate_assessments"].items()
        if assessment["accepted_for_probabilistic_port"]
    ]
    status = ", ".join(accepted) if accepted else "no candidate passes all checks"
    figure.update_layout(
        title=f"Paired deterministic 2x2 architecture screen: {status}",
        template="plotly_white",
        boxmode="group",
        height=900,
        width=2100,
        hoverlabel={"namelength": -1},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def _write_paired_difference_dashboard(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    panels = (
        ("temperature_rmse_c", "Delta Tin RMSE [C]"),
        ("qroom_rmse_w_m2", "Delta Qroom RMSE [W/m2]"),
        ("pel_rmse_w_m2", "Delta Pel RMSE [W/m2]"),
        ("qroom_energy_log_error", "Delta Q energy error"),
        ("pel_energy_log_error", "Delta Pel energy error"),
        ("temperature_abs_bias_c", "Delta absolute Tin bias [C]"),
    )
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=tuple(title for _, title in panels),
    )
    for index, (metric, _) in enumerate(panels):
        row, column = divmod(index, 3)
        for model in MODEL_ORDER[1:]:
            for policy, symbol in (("setpoint", "circle"), ("force_off", "diamond")):
                group = results[results.policy == policy].pivot(
                    index="profile_id", columns="model", values=metric
                )
                comparison = group[[LEGACY_NAME, model]].dropna()
                delta = comparison[model] - comparison[LEGACY_NAME]
                figure.add_trace(
                    go.Scatter(
                        x=np.full(len(delta), policy),
                        y=delta,
                        mode="markers",
                        name=model,
                        legendgroup=model,
                        marker={
                            "color": MODEL_COLORS[model],
                            "opacity": 0.7,
                            "symbol": symbol,
                        },
                        showlegend=index == 0 and policy == "setpoint",
                        customdata=delta.index.astype(str),
                        hovertemplate=(
                            f"model={model}<br>policy={policy}"
                            "<br>EGID=%{customdata}"
                            "<br>candidate-legacy=%{y:.4f}<extra></extra>"
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
        title="Paired differences against exact legacy; values below zero improve",
        template="plotly_white",
        height=850,
        width=1800,
        hoverlabel={"namelength": -1},
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Both controller variants now share the same hydronic and thermal labels;
    # retain HP kind in focal-trace keys so neither curve is overwritten.
    args.include_hp_in_architecture_label = True
    selected = _selected_profile_ids(args)
    rows: list[dict[str, Any]] = []
    focal_traces: dict[str, dict[str, dict[str, Any]]] = {}
    started = time.perf_counter()
    total = 2 * args.profiles_per_policy
    completed = 0
    cases = (
        (
            "setpoint",
            args.setpoint_dataset,
            args.setpoint_focus_profile_id,
        ),
        (
            "force_off",
            args.force_off_dataset,
            args.force_off_focus_profile_id,
        ),
    )
    for policy, dataset, focus_id in cases:
        for profile_id in selected[policy]:
            completed += 1
            elapsed = time.perf_counter() - started
            eta_minutes = (
                elapsed / max(completed - 1, 1) * (total - completed) / 60.0
            )
            print(
                f"paired_screen profile={completed}/{total} policy={policy} "
                f"egid={profile_id} eta_minutes={eta_minutes:.1f}",
                flush=True,
            )
            case_rows, traces, _ = _fit_case(
                policy,
                dataset,
                int(profile_id),
                ARCHITECTURES,
                args,
            )
            rows.extend(case_rows)
            if int(profile_id) == int(focus_id):
                focal_traces[policy] = traces

    results = _prepare_results(rows)
    results.to_csv(args.output_dir / "paired_building_metrics.csv", index=False)
    summary = _paired_summary(results)
    summary.update(
        {
            "selected_profile_ids": selected,
            "architectures": [architecture.name for architecture in ARCHITECTURES],
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
    )
    (args.output_dir / "paired_screen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    _write_dashboard(
        results,
        summary,
        args.output_dir / "paired_architecture_dashboard.html",
    )
    _write_paired_difference_dashboard(
        results,
        args.output_dir / "paired_difference_dashboard.html",
    )
    if set(focal_traces) == {"setpoint", "force_off"}:
        _write_trace_dashboard(
            focal_traces,
            {
                "setpoint": args.setpoint_focus_profile_id,
                "force_off": args.force_off_focus_profile_id,
            },
            args.output_dir / "focal_profile_traces.html",
        )
    print(
        f"saved_paired_screen={args.output_dir} "
        f"full_candidate_accepted={summary['accepted_for_probabilistic_port']} "
        f"candidate_acceptance="
        f"{json.dumps({name: value['accepted_for_probabilistic_port'] for name, value in summary['candidate_assessments'].items()}, sort_keys=True)} "
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
    parser.add_argument(
        "--hp-recovery-emergency-threshold-c", type=float, default=1.0
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
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "deterministic_causal_q_factorial_screen_20"
        ),
    )
    args = parser.parse_args()
    if args.profiles_per_policy < 2:
        parser.error("--profiles-per-policy must be at least 2")
    if not 0.1 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction must lie in [0.1, 0.9]")
    if args.epochs < 1 or args.batch_size < 1 or args.sequence_length < 2:
        parser.error("epochs, batch size, and sequence length must be positive")
    if args.stride < 1 or args.warmup_hours < 0.0:
        parser.error("stride must be positive and warmup non-negative")
    if args.hp_recovery_emergency_threshold_c < 0.0:
        parser.error("emergency threshold must be non-negative")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
