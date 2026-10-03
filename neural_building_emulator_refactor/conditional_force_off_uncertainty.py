"""Locate probabilistic spread by policy, season and force-off phase.

This analysis complements the source decomposition. It asks where each source
becomes overdispersed, rather than treating annual average width as homogeneous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.model_io import load_training_artifact
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator

from .probabilistic_uncertainty_decomposition import (
    ABLATIONS,
    _controlled_scenarios,
    _load_local_model,
)
from .single_building_exact_probabilistic_reproduction import _load_profile


SELECTED_ABLATIONS = tuple(
    value
    for value in ABLATIONS
    if value.name in {"deterministic_center", "xi_only", "process_hp", "full"}
)


def _post_release_mask(force_off: np.ndarray, steps: int) -> np.ndarray:
    active = np.asarray(force_off > 0.5)
    releases = np.flatnonzero(np.r_[False, active[:-1] & ~active[1:]])
    mask = np.zeros_like(active)
    for index in releases:
        mask[index : min(index + steps, len(mask))] = True
    return mask & ~active


def _regimes(profile, input_columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(profile.datetime))
    winter = np.asarray(dates.month.isin((11, 12, 1, 2, 3)))
    availability = profile.inputs[:, input_columns.index("space_heating_available")] > 0.5
    force_off = profile.inputs[:, input_columns.index("hp_force_off_requested")] > 0.5
    active_lockout = availability & force_off
    return {
        "all": np.ones(len(dates), dtype=bool),
        "winter": winter,
        "non_winter": ~winter,
        "heating_available": availability,
        "heating_available_no_lockout": availability & ~force_off,
        "active_lockout": active_lockout,
        "post_release_0_3h": availability & _post_release_mask(force_off, 12),
        "post_release_0_12h": availability & _post_release_mask(force_off, 48),
        "post_release_0_24h": availability & _post_release_mask(force_off, 96),
        "cold_tout_below_10c": profile.inputs[:, 1] < 10.0,
    }


def _conditional_metrics(
    scenarios: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    if not np.any(mask):
        return {
            "count": 0,
            "temperature_width_mean": float("nan"),
            "temperature_width_p95": float("nan"),
            "temperature_coverage95": float("nan"),
            "temperature_mean_rmse": float("nan"),
        }
    temperature = scenarios[:, :, 0]
    lower = np.quantile(temperature, 0.025, axis=0)
    upper = np.quantile(temperature, 0.975, axis=0)
    mean = np.mean(temperature, axis=0)
    truth = target[:, 0]
    return {
        "count": int(np.sum(mask)),
        "temperature_width_mean": float(np.mean((upper - lower)[mask])),
        "temperature_width_p95": float(np.quantile((upper - lower)[mask], 0.95)),
        "temperature_coverage95": float(
            np.mean((truth[mask] >= lower[mask]) & (truth[mask] <= upper[mask]))
        ),
        "temperature_mean_rmse": float(np.sqrt(np.mean((mean[mask] - truth[mask]) ** 2))),
    }


def _monthly_width_rows(
    scenarios: np.ndarray,
    datetime: np.ndarray,
    *,
    model: str,
    policy: str,
    ablation: str,
) -> list[dict[str, float | str]]:
    dates = pd.DatetimeIndex(pd.to_datetime(datetime))
    width = np.quantile(scenarios[:, :, 0], 0.975, axis=0) - np.quantile(
        scenarios[:, :, 0], 0.025, axis=0
    )
    return [
        {
            "model": model,
            "policy": policy,
            "ablation": ablation,
            "month": month,
            "temperature_width_mean": float(np.mean(width[dates.month == month])),
        }
        for month in range(1, 13)
    ]


def _event_width_rows(
    scenarios: np.ndarray,
    force_off: np.ndarray,
    *,
    model: str,
    ablation: str,
    before_steps: int = 96,
    after_steps: int = 288,
) -> list[dict[str, float | str]]:
    active = force_off > 0.5
    width = np.quantile(scenarios[:, :, 0], 0.975, axis=0) - np.quantile(
        scenarios[:, :, 0], 0.025, axis=0
    )
    rows: list[dict[str, float | str]] = []
    for edge_name, edges in (
        ("lockout_onset", np.flatnonzero(np.r_[active[0], ~active[:-1] & active[1:]])),
        ("lockout_release", np.flatnonzero(np.r_[False, active[:-1] & ~active[1:]])),
    ):
        aligned = []
        for index in edges:
            start, end = index - before_steps, index + after_steps + 1
            if start >= 0 and end <= len(width):
                aligned.append(width[start:end])
        if not aligned:
            continue
        mean_width = np.mean(np.stack(aligned), axis=0)
        for offset, value in enumerate(mean_width, start=-before_steps):
            rows.append(
                {
                    "model": model,
                    "ablation": ablation,
                    "edge": edge_name,
                    "relative_hour": offset * 0.25,
                    "temperature_width_mean": float(value),
                    "events": len(aligned),
                }
            )
    return rows


def _write_dashboard(
    conditional: pd.DataFrame,
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: Path,
) -> Path:
    colors = {
        "deterministic_center": "#616161",
        "process_hp": "#00897b",
        "xi_only": "#f9a825",
        "full": "#c62828",
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Global model: conditional temperature width",
            "Local model: conditional temperature width",
            "Monthly width: global model",
            "Width aligned to force-off edges: global model",
        ),
        vertical_spacing=0.13,
    )
    regime_order = (
        "winter",
        "non_winter",
        "heating_available_no_lockout",
        "active_lockout",
        "post_release_0_3h",
        "post_release_0_12h",
        "post_release_0_24h",
    )
    for col, model in enumerate(("global", "local"), start=1):
        selected = conditional[
            (conditional["model"] == model)
            & (conditional["policy"] == "force_off")
        ]
        for ablation in colors:
            values = selected[selected["ablation"] == ablation].set_index("regime")
            figure.add_trace(
                go.Bar(
                    x=list(regime_order),
                    y=[values.loc[name, "temperature_width_mean"] for name in regime_order],
                    name=ablation,
                    legendgroup=ablation,
                    marker_color=colors[ablation],
                    showlegend=col == 1,
                    hovertemplate=(
                        f"model={model}<br>source={ablation}<br>regime=%{{x}}"
                        "<br>mean width=%{y:.3f} C<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    global_monthly = monthly[monthly["model"] == "global"]
    dash_by_policy = {"setpoint": "solid", "force_off": "dash"}
    for policy, dash in dash_by_policy.items():
        for ablation in ("xi_only", "full"):
            values = global_monthly[
                (global_monthly["policy"] == policy)
                & (global_monthly["ablation"] == ablation)
            ]
            figure.add_trace(
                go.Scatter(
                    x=values["month"],
                    y=values["temperature_width_mean"],
                    mode="lines+markers",
                    line={"color": colors[ablation], "dash": dash},
                    name=f"{policy}: {ablation}",
                    legendgroup=f"{policy}_{ablation}",
                    hovertemplate=(
                        f"policy={policy}<br>source={ablation}<br>month=%{{x}}"
                        "<br>mean width=%{y:.3f} C<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
    global_events = events[events["model"] == "global"]
    edge_dash = {"lockout_onset": "solid", "lockout_release": "dash"}
    for edge, dash in edge_dash.items():
        for ablation in ("xi_only", "full"):
            values = global_events[
                (global_events["edge"] == edge)
                & (global_events["ablation"] == ablation)
            ]
            figure.add_trace(
                go.Scatter(
                    x=values["relative_hour"],
                    y=values["temperature_width_mean"],
                    mode="lines",
                    line={"color": colors[ablation], "dash": dash},
                    name=f"{edge}: {ablation}",
                    hovertemplate=(
                        f"edge={edge}<br>source={ablation}<br>relative hour=%{{x:.1f}}"
                        "<br>mean width=%{y:.3f} C<extra></extra>"
                    ),
                ),
                row=2,
                col=2,
            )
    figure.add_vline(x=0, line_dash="dot", line_color="#424242", row=2, col=2)
    figure.update_yaxes(title_text="Mean 95% width [C]")
    figure.update_xaxes(title_text="Month", row=2, col=1)
    figure.update_xaxes(title_text="Hours from policy edge", row=2, col=2)
    figure.update_layout(
        title="Policy-conditional uncertainty localization",
        barmode="group",
        height=1050,
        width=1750,
        hoverlabel={"namelength": -1},
    )
    path = output_dir / "conditional_force_off_uncertainty.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return path


def run(args: argparse.Namespace) -> None:
    artifact = load_training_artifact(args.artifact_dir)
    if not isinstance(artifact.model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError("Artifact is not a probabilistic contracting closed-loop model")
    metadata_columns = tuple(artifact.metadata["metadata_columns"])
    input_columns = tuple(artifact.metadata["input_columns"])
    profiles = {
        "setpoint": _load_profile(
            args.setpoint_dataset,
            args.profile_id,
            policy="setpoint",
            metadata_columns=metadata_columns,
        ).profile,
        "force_off": _load_profile(
            args.force_off_dataset,
            args.profile_id,
            policy="force_off",
            metadata_columns=metadata_columns,
        ).profile,
    }
    models = {"global": artifact.model}
    if args.local_model is not None:
        models["local"] = _load_local_model(
            args.local_model,
            artifact.metadata,
            artifact.scalers,
        )
    conditional_rows: list[dict] = []
    monthly_rows: list[dict] = []
    event_rows: list[dict] = []
    for model_index, (model_name, model) in enumerate(models.items()):
        for policy_index, (policy, profile) in enumerate(profiles.items()):
            regimes = _regimes(profile, input_columns)
            force_off = profile.inputs[:, input_columns.index("hp_force_off_requested")]
            for ablation in SELECTED_ABLATIONS:
                scenarios = _controlled_scenarios(
                    model,
                    profile,
                    artifact.scalers,
                    metadata_columns,
                    ablation=ablation,
                    num_scenarios=args.num_scenarios,
                    seed=args.seed + model_index * 10_000 + policy_index * 1_000,
                )
                for regime, mask in regimes.items():
                    conditional_rows.append(
                        {
                            "model": model_name,
                            "policy": policy,
                            "ablation": ablation.name,
                            "regime": regime,
                            **_conditional_metrics(scenarios, profile.targets, mask),
                        }
                    )
                monthly_rows.extend(
                    _monthly_width_rows(
                        scenarios,
                        profile.datetime,
                        model=model_name,
                        policy=policy,
                        ablation=ablation.name,
                    )
                )
                if policy == "force_off":
                    event_rows.extend(
                        _event_width_rows(
                            scenarios,
                            force_off,
                            model=model_name,
                            ablation=ablation.name,
                        )
                    )
                winter_width = conditional_rows[-len(regimes)]["temperature_width_mean"]
                print(
                    f"model={model_name} policy={policy} ablation={ablation.name} "
                    f"all_width={winter_width:.3f}",
                    flush=True,
                )
    conditional = pd.DataFrame(conditional_rows)
    monthly = pd.DataFrame(monthly_rows)
    events = pd.DataFrame(event_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditional_path = args.output_dir / "conditional_uncertainty_metrics.csv"
    monthly_path = args.output_dir / "monthly_uncertainty_width.csv"
    events_path = args.output_dir / "force_off_event_aligned_width.csv"
    conditional.to_csv(conditional_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    events.to_csv(events_path, index=False)
    dashboard_path = _write_dashboard(conditional, monthly, events, args.output_dir)
    summary = {
        "profile_id": args.profile_id,
        "num_scenarios": args.num_scenarios,
        "conditional_metrics": conditional_rows,
    }
    summary_path = args.output_dir / "conditional_uncertainty_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved_metrics={conditional_path}")
    print(f"saved_dashboard={dashboard_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "calibrated_four_model_benchmark_force_off_debt_1000/models/"
            "probabilistic_structured/artifacts/"
            "closed_loop_hp_contracting_probabilistic/selected"
        ),
    )
    parser.add_argument(
        "--local-model",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/exact_local_reproduction_11140124/"
            "exact_local_model.eqx"
        ),
    )
    parser.add_argument(
        "--setpoint-dataset",
        type=Path,
        default=Path("data/control=setpoint_identification_dhw_fixed"),
    )
    parser.add_argument("--force-off-dataset", type=Path, default=Path("data/control=force_off"))
    parser.add_argument("--profile-id", type=int, default=11140124)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20015)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/exact_local_reproduction_11140124/"
            "conditional_uncertainty"
        ),
    )
    args = parser.parse_args()
    for name in ("artifact_dir", "setpoint_dataset", "force_off_dataset"):
        if not getattr(args, name).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    if args.local_model is not None and not args.local_model.exists():
        parser.error(f"--local-model does not exist: {args.local_model}")
    if args.num_scenarios < 2:
        parser.error("--num-scenarios must be at least 2")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
