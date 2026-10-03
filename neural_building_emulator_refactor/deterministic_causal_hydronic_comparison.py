"""Compare deterministic room-heat demand and buffer ablations.

The comparison is intentionally narrower than the probabilistic benchmark. It
evaluates complete annual trajectories under both setpoint and force-off
policies, and reports whether delivered room heat is limited by learned demand
or by physically available buffer energy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.models import ContractingClosedLoopHPEmulator
from neural_building_emulator.scaling import inverse_target
from neural_building_emulator.train import _probabilistic_ventilation_rollout_kwargs

from .artifacts import load_artifact
from .profiles import load_profiles


VARIANTS = (
    ("legacy_cap24", "Legacy demand, 24 h buffer"),
    ("causal_cap24", "Causal demand, 24 h buffer"),
    ("causal_cap4", "Causal demand, 4 h buffer"),
    ("legacy_cap1", "Legacy demand, 1 h buffer"),
    ("causal_cap1", "Causal demand, 1 h buffer"),
)
SIGNALS = ("Tin", "Qroom", "Pel_SH")
COLORS = {
    "legacy_cap24": "#68757d",
    "causal_cap24": "#187b78",
    "causal_cap4": "#6c55a3",
    "legacy_cap1": "#d17837",
    "causal_cap1": "#355ea8",
}


def _rollout(artifact, profile) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    model = artifact.model
    if not isinstance(model, ContractingClosedLoopHPEmulator):
        raise TypeError("Expected a deterministic contracting closed-loop HP model")
    metadata = jnp.asarray(artifact.scalers.metadata.transform(profile.metadata))
    inputs = jnp.asarray(artifact.scalers.inputs.transform(profile.inputs))
    target_mean = np.asarray(artifact.scalers.target.mean, dtype=np.float32).reshape(-1)
    target_scale = np.asarray(artifact.scalers.target.scale, dtype=np.float32).reshape(-1)
    initial_temperature = jnp.asarray(
        (
            (np.asarray(profile.initial_temperature).reshape(-1)[:1] - target_mean[:1])
            / target_scale[:1]
        ).astype(np.float32)
    )
    metadata_columns = tuple(artifact.legacy_artifact.metadata["metadata_columns"])
    ventilation_kwargs = _probabilistic_ventilation_rollout_kwargs(
        profile,
        mode="eplus_rule",
        metadata_columns=metadata_columns,
    )
    predictions, aux = model.rollout_with_aux(
        metadata,
        inputs,
        initial_temperature,
        **ventilation_kwargs,
    )
    return inverse_target(np.asarray(predictions), artifact.scalers), tuple(
        np.asarray(value) for value in aux
    )


def _regimes(profile, input_columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(profile.datetime))
    winter = np.asarray(dates.month.isin((11, 12, 1, 2, 3)))
    available = np.ones(len(dates), dtype=bool)
    if "space_heating_available" in input_columns:
        available = profile.inputs[:, input_columns.index("space_heating_available")] > 0.5
    force_off = np.zeros(len(dates), dtype=bool)
    if "hp_force_off_requested" in input_columns:
        force_off = profile.inputs[:, input_columns.index("hp_force_off_requested")] > 0.5
    return {
        "annual": np.ones(len(dates), dtype=bool),
        "winter": winter,
        "winter_heating_available": winter & available,
        "winter_no_lockout": winter & available & ~force_off,
        "winter_active_lockout": winter & available & force_off,
    }


def _profile_rows(
    variant: str,
    policy: str,
    profile,
    prediction: np.ndarray,
    aux: tuple[np.ndarray, ...],
    input_columns: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    # Aux indices are documented by ContractingClosedLoopHPEmulator:
    # pi, Q, Pel, state, ||M||, E, E_next, Qavailable, Qrequest.
    energy = aux[5]
    available_power = aux[7]
    requested_qroom = aux[8]
    rows: list[dict[str, float | int | str]] = []
    for regime, mask in _regimes(profile, input_columns).items():
        if not np.any(mask):
            continue
        error = prediction[mask] - profile.targets[mask]
        row: dict[str, float | int | str] = {
            "variant": variant,
            "policy": policy,
            "profile_id": int(profile.profile_id),
            "regime": regime,
            "count": int(np.sum(mask)),
            "request_limited_count": int(
                np.sum(requested_qroom[mask] <= available_power[mask] + 1e-5)
            ),
            "empty_buffer_count": int(np.sum(energy[mask] <= 1e-4)),
            "energy_sum_wh_m2": float(np.sum(energy[mask])),
            "requested_qroom_sum_w_m2": float(np.sum(requested_qroom[mask])),
            "available_power_sum_w_m2": float(np.sum(available_power[mask])),
            "truth_qroom_sum_w_m2": float(np.sum(profile.targets[mask, 1])),
            "prediction_qroom_sum_w_m2": float(np.sum(prediction[mask, 1])),
        }
        for index, signal in enumerate(SIGNALS):
            row[f"{signal}_sse"] = float(np.sum(np.square(error[:, index])))
            row[f"{signal}_sae"] = float(np.sum(np.abs(error[:, index])))
            row[f"{signal}_error_sum"] = float(np.sum(error[:, index]))
        rows.append(row)
    return rows


def _aggregate(profile_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for (variant, policy, regime), group in profile_rows.groupby(
        ["variant", "policy", "regime"], sort=False
    ):
        count = int(group["count"].sum())
        row: dict[str, float | int | str] = {
            "variant": variant,
            "policy": policy,
            "regime": regime,
            "profile_count": int(group["profile_id"].nunique()),
            "observation_count": count,
            "request_limited_fraction": float(
                group["request_limited_count"].sum() / count
            ),
            "empty_buffer_fraction": float(group["empty_buffer_count"].sum() / count),
            "mean_energy_wh_m2": float(group["energy_sum_wh_m2"].sum() / count),
            "mean_requested_qroom_w_m2": float(
                group["requested_qroom_sum_w_m2"].sum() / count
            ),
            "mean_available_power_w_m2": float(
                group["available_power_sum_w_m2"].sum() / count
            ),
            "truth_qroom_w_m2": float(group["truth_qroom_sum_w_m2"].sum() / count),
            "prediction_qroom_w_m2": float(
                group["prediction_qroom_sum_w_m2"].sum() / count
            ),
        }
        for signal in SIGNALS:
            row[f"{signal}_rmse"] = float(
                np.sqrt(group[f"{signal}_sse"].sum() / count)
            )
            row[f"{signal}_mae"] = float(group[f"{signal}_sae"].sum() / count)
            row[f"{signal}_bias"] = float(
                group[f"{signal}_error_sum"].sum() / count
            )
        records.append(row)
    return pd.DataFrame(records)


def _dashboard(summary: pd.DataFrame, output_path: Path) -> None:
    panels = (
        ("Tin_rmse", "Tin RMSE [C]", "winter_heating_available"),
        ("Tin_bias", "Tin bias [C]", "winter_heating_available"),
        ("Qroom_rmse", "Qroom RMSE [W/m2]", "winter_heating_available"),
        ("Qroom_bias", "Qroom bias [W/m2]", "winter_heating_available"),
        ("request_limited_fraction", "Demand-limited fraction", "winter_heating_available"),
        ("mean_energy_wh_m2", "Mean buffer energy [Wh/m2]", "winter_heating_available"),
    )
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=tuple(title for _, title, _ in panels),
    )
    for panel, (metric, _, regime) in enumerate(panels):
        row, column = divmod(panel, 3)
        selected = summary[summary["regime"] == regime]
        for variant, label in VARIANTS:
            values = selected[selected["variant"] == variant]
            figure.add_trace(
                go.Bar(
                    x=values["policy"],
                    y=values[metric],
                    name=label,
                    legendgroup=variant,
                    marker_color=COLORS[variant],
                    showlegend=panel == 0,
                    hovertemplate=(
                        f"model={label}<br>policy=%{{x}}<br>{metric}=%{{y:.4f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row + 1,
                col=column + 1,
            )
    figure.update_layout(
        title="Deterministic causal hydronic ablation: winter heating regime",
        barmode="group",
        template="plotly_white",
        height=900,
        width=1700,
        hoverlabel={"namelength": -1},
    )
    figure.write_html(output_path)


def _focus_dashboard(
    artifacts: dict[str, object],
    datasets: dict[str, Path],
    profile_id: int,
    output_path: Path,
) -> None:
    profiles: dict[str, object] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes="columns",
        subplot_titles=tuple(
            f"{policy}: {signal}" for signal in SIGNALS for policy in datasets
        ),
    )
    for column, (policy, dataset) in enumerate(datasets.items(), start=1):
        reference_profile = None
        predictions[policy] = {}
        for variant, _ in VARIANTS:
            artifact = artifacts[variant]
            profile = load_profiles(
                artifact,
                dataset_path=dataset,
                profile_ids=[profile_id],
            )[0]
            prediction, _ = _rollout(artifact, profile)
            reference_profile = profile
            predictions[policy][variant] = prediction
            for row, signal in enumerate(SIGNALS, start=1):
                figure.add_trace(
                    go.Scatter(
                        x=profile.datetime,
                        y=prediction[:, row - 1],
                        name=dict(VARIANTS)[variant],
                        legendgroup=variant,
                        line={"color": COLORS[variant], "width": 1.1},
                        showlegend=column == 1 and row == 1,
                    ),
                    row=row,
                    col=column,
                )
        assert reference_profile is not None
        for row, signal in enumerate(SIGNALS, start=1):
            figure.add_trace(
                go.Scatter(
                    x=reference_profile.datetime,
                    y=reference_profile.targets[:, row - 1],
                    name="EnergyPlus",
                    legendgroup="truth",
                    line={"color": "#161b1f", "width": 1.4},
                    showlegend=column == 1 and row == 1,
                ),
                row=row,
                col=column,
            )
        profiles[policy] = reference_profile
    figure.update_yaxes(title_text="Tin [C]", row=1, col=1)
    figure.update_yaxes(title_text="Qroom [W/m2]", row=2, col=1)
    figure.update_yaxes(title_text="Pel_SH [W/m2]", row=3, col=1)
    figure.update_layout(
        title=f"Deterministic annual comparison: profile {profile_id}",
        template="plotly_white",
        height=1150,
        width=1800,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
        legend={"orientation": "h", "y": -0.06},
    )
    figure.write_html(output_path)

    difference_figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes="columns",
        subplot_titles=tuple(
            f"{policy}: delta {signal} vs legacy cap24"
            for signal in SIGNALS
            for policy in datasets
        ),
    )
    for column, policy in enumerate(datasets, start=1):
        profile = profiles[policy]
        baseline = predictions[policy]["legacy_cap24"]
        for variant, label in VARIANTS:
            if variant == "legacy_cap24":
                continue
            difference = predictions[policy][variant] - baseline
            for row, signal in enumerate(SIGNALS, start=1):
                difference_figure.add_trace(
                    go.Scatter(
                        x=profile.datetime,
                        y=difference[:, row - 1],
                        name=label,
                        legendgroup=variant,
                        line={"color": COLORS[variant], "width": 1.1},
                        showlegend=column == 1 and row == 1,
                        hovertemplate=(
                            f"model={label}<br>time=%{{x}}<br>delta {signal}=%{{y:.4f}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=column,
                )
        for row in range(1, 4):
            difference_figure.add_hline(
                y=0.0,
                line={"color": "#7d858a", "width": 0.8, "dash": "dot"},
                row=row,
                col=column,
            )
    difference_figure.update_yaxes(title_text="delta Tin [C]", row=1, col=1)
    difference_figure.update_yaxes(title_text="delta Qroom [W/m2]", row=2, col=1)
    difference_figure.update_yaxes(title_text="delta Pel_SH [W/m2]", row=3, col=1)
    difference_figure.update_layout(
        title=f"Deterministic variant differences: profile {profile_id}",
        template="plotly_white",
        height=1150,
        width=1800,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
        legend={"orientation": "h", "y": -0.06},
    )
    difference_figure.write_html(
        output_path.with_name(f"{output_path.stem}_differences.html")
    )


def run(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        variant: args.root
        / variant
        / "artifacts"
        / "closed_loop_hp_contracting"
        / "selected"
        for variant, _ in VARIANTS
    }
    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing deterministic artifacts: " + ", ".join(missing))
    artifacts = {name: load_artifact(path) for name, path in artifact_paths.items()}
    datasets = {"setpoint": args.setpoint_dataset, "force_off": args.force_off_dataset}
    rows: list[dict[str, float | int | str]] = []
    for variant, _ in VARIANTS:
        artifact = artifacts[variant]
        input_columns = tuple(artifact.legacy_artifact.metadata["input_columns"])
        for policy, dataset in datasets.items():
            profiles = load_profiles(
                artifact,
                dataset_path=dataset,
                source="test",
                max_profiles=args.max_profiles,
            )
            for index, profile in enumerate(profiles, start=1):
                prediction, aux = _rollout(artifact, profile)
                rows.extend(
                    _profile_rows(
                        variant,
                        policy,
                        profile,
                        prediction,
                        aux,
                        input_columns,
                    )
                )
                print(
                    f"variant={variant} policy={policy} "
                    f"profile={index}/{len(profiles)} id={profile.profile_id}",
                    flush=True,
                )
    profile_rows = pd.DataFrame(rows)
    summary = _aggregate(profile_rows)
    profile_rows.to_csv(args.output_dir / "profile_regime_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "aggregate_regime_metrics.csv", index=False)
    _dashboard(summary, args.output_dir / "deterministic_causal_hydronic_ablation.html")
    _focus_dashboard(
        artifacts,
        datasets,
        args.focus_profile_id,
        args.output_dir / f"focus_profile_{args.focus_profile_id}.html",
    )
    compact = summary[
        summary["regime"].isin(("annual", "winter_heating_available"))
    ].to_dict(orient="records")
    (args.output_dir / "summary.json").write_text(json.dumps(compact, indent=2))
    print(f"saved_deterministic_hydronic_comparison={args.output_dir}")
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "deterministic_causal_hydronic_ablation"
        ),
    )
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
    parser.add_argument("--max-profiles", type=int, default=20)
    parser.add_argument("--focus-profile-id", type=int, default=11132764)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.root / "comparison"
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
