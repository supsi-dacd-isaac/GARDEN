"""Diagnose winter bias and uncertainty widening for one fitted building."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.model_io import load_training_artifact
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator

from .conditional_force_off_uncertainty import _post_release_mask
from .probabilistic_uncertainty_decomposition import ABLATIONS, _controlled_scenarios
from .single_building_exact_probabilistic_reproduction import _load_profile


SIGNALS = ("Tin", "Qroom", "Pel_SH")
UNITS = ("degC", "W/m2", "W/m2")
TRACE_ABLATIONS = (
    "deterministic_center",
    "xi_only",
    "process_only",
    "hp_only",
    "full",
)
COLORS = {
    "deterministic_center": "#59676f",
    "xi_only": "#d97706",
    "process_only": "#c43d55",
    "hp_only": "#16858b",
    "full": "#3f6db3",
    "truth": "#151b1f",
    "setpoint": "#7a4eab",
    "Qroom": "#d97706",
    "Pel_SH": "#16858b",
}
HP_AUX_NAMES = (
    "pi",
    "cop",
    "energy_wh_m2",
    "available_power_w_m2",
    "requested_qroom_w_m2",
    "hp_on",
    "pel_active_w_m2",
    "pel_w_m2",
    "log_mu",
    "log_sigma",
)


def _regimes(
    profile,
    input_columns: tuple[str, ...],
) -> dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(profile.datetime))
    winter = np.asarray(dates.month.isin((11, 12, 1, 2, 3)))
    availability = np.asarray(
        profile.inputs[:, input_columns.index("space_heating_available")] > 0.5
    )
    if "hp_force_off_requested" in input_columns:
        force_off = np.asarray(
            profile.inputs[:, input_columns.index("hp_force_off_requested")] > 0.5
        )
    else:
        force_off = np.zeros(len(dates), dtype=bool)
    return {
        "all": np.ones(len(dates), dtype=bool),
        "winter": winter,
        "non_winter": ~winter,
        "heating_available_no_lockout": availability & ~force_off,
        "active_lockout": availability & force_off,
        "winter_active_lockout": winter & availability & force_off,
        "post_release_0_3h": availability & _post_release_mask(force_off, 12),
        "post_release_0_12h": availability & _post_release_mask(force_off, 48),
    }


def _signal_metrics(
    scenarios: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    channel: int,
) -> dict[str, float | int]:
    if not np.any(mask):
        return {
            "observation_count": 0,
            "mean_bias": float("nan"),
            "rmse": float("nan"),
            "mean_width_95": float("nan"),
            "coverage_95": float("nan"),
            "truth_mean": float("nan"),
            "prediction_mean": float("nan"),
        }
    values = scenarios[:, :, channel]
    mean = np.mean(values, axis=0)
    lower = np.quantile(values, 0.025, axis=0)
    upper = np.quantile(values, 0.975, axis=0)
    truth = target[:, channel]
    error = mean - truth
    return {
        "observation_count": int(np.sum(mask)),
        "mean_bias": float(np.mean(error[mask])),
        "rmse": float(np.sqrt(np.mean(np.square(error[mask])))),
        "mean_width_95": float(np.mean((upper - lower)[mask])),
        "coverage_95": float(
            np.mean((truth[mask] >= lower[mask]) & (truth[mask] <= upper[mask]))
        ),
        "truth_mean": float(np.mean(truth[mask])),
        "prediction_mean": float(np.mean(mean[mask])),
    }


def _daily_diagnostics(
    profile,
    scenarios: dict[str, np.ndarray],
    *,
    setpoint_index: int,
    force_off_index: int,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(profile.datetime))
    data: dict[str, np.ndarray] = {
        "truth_tin": np.asarray(profile.targets[:, 0]),
        "truth_qroom": np.asarray(profile.targets[:, 1]),
        "truth_pel": np.asarray(profile.targets[:, 2]),
        "setpoint": np.asarray(profile.inputs[:, setpoint_index]),
        "force_off": (
            np.asarray(profile.inputs[:, force_off_index])
            if force_off_index >= 0
            else np.zeros(len(dates), dtype=np.float32)
        ),
    }
    for name, values in scenarios.items():
        mean = np.mean(values, axis=0)
        data[f"{name}_tin_mean"] = mean[:, 0]
        data[f"{name}_tin_width"] = (
            np.quantile(values[:, :, 0], 0.975, axis=0)
            - np.quantile(values[:, :, 0], 0.025, axis=0)
        )
        data[f"{name}_qroom_mean"] = mean[:, 1]
        data[f"{name}_pel_mean"] = mean[:, 2]
    frame = pd.DataFrame(data, index=dates)
    return frame.resample("1D").mean()


def _select_bad_windows(
    policy: str,
    daily: pd.DataFrame,
    *,
    window_days: int = 7,
    count: int = 5,
) -> list[dict[str, float | str]]:
    rolling = daily.rolling(window_days, min_periods=max(3, window_days // 2)).mean()
    rolling["full_tin_bias"] = rolling["full_tin_mean"] - rolling["truth_tin"]
    rolling["center_tin_bias"] = (
        rolling["deterministic_center_tin_mean"] - rolling["truth_tin"]
    )
    rolling["cold_spread_score"] = rolling["full_tin_width"] * np.maximum(
        -rolling["full_tin_bias"], 0.0
    )
    winter = rolling.index.month.isin((11, 12, 1, 2, 3))
    candidates = rolling.loc[winter].sort_values("cold_spread_score", ascending=False)
    chosen: list[pd.Timestamp] = []
    rows: list[dict[str, float | str]] = []
    for end, values in candidates.iterrows():
        if not np.isfinite(values["cold_spread_score"]):
            continue
        if any(abs((end - previous).days) < window_days for previous in chosen):
            continue
        start = end - pd.Timedelta(days=window_days - 1)
        chosen.append(end)
        row: dict[str, float | str] = {
            "policy": policy,
            "start": str(start.date()),
            "end": str(end.date()),
            "cold_spread_score": float(values["cold_spread_score"]),
            "full_tin_bias_c": float(values["full_tin_bias"]),
            "deterministic_center_tin_bias_c": float(values["center_tin_bias"]),
            "stochastic_mean_shift_c": float(
                values["full_tin_mean"] - values["deterministic_center_tin_mean"]
            ),
            "full_tin_width_95_c": float(values["full_tin_width"]),
            "truth_tin_c": float(values["truth_tin"]),
            "full_tin_c": float(values["full_tin_mean"]),
            "setpoint_c": float(values["setpoint"]),
            "full_qroom_bias_w_m2": float(
                values["full_qroom_mean"] - values["truth_qroom"]
            ),
            "full_pel_bias_w_m2": float(
                values["full_pel_mean"] - values["truth_pel"]
            ),
            "force_off_fraction": float(values["force_off"]),
        }
        for name in ("xi_only", "process_only", "hp_only"):
            row[f"{name}_tin_width_95_c"] = float(values[f"{name}_tin_width"])
            row[f"{name}_tin_bias_c"] = float(
                values[f"{name}_tin_mean"] - values["truth_tin"]
            )
        rows.append(row)
        if len(rows) >= count:
            break
    return rows


def _buffer_diagnostics(
    policy: str,
    profile,
    auxiliary: dict[str, tuple[np.ndarray, ...]],
    bad_windows: list[dict[str, float | str]],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(profile.datetime))
    rows: list[dict[str, float | str]] = []
    windows: list[tuple[str, np.ndarray]] = [
        ("winter", np.asarray(dates.month.isin((11, 12, 1, 2, 3))))
    ]
    for index, window in enumerate(bad_windows, start=1):
        if window["policy"] != policy:
            continue
        start = pd.Timestamp(str(window["start"]))
        end = pd.Timestamp(str(window["end"])) + pd.Timedelta(days=1)
        windows.append(
            (
                f"worst_{index}_{window['start']}_{window['end']}",
                np.asarray((dates >= start) & (dates < end)),
            )
        )

    for source, values in auxiliary.items():
        aux = dict(zip(HP_AUX_NAMES, values))
        requested = aux["requested_qroom_w_m2"]
        available = aux["available_power_w_m2"]
        predicted_qroom = np.minimum(requested, available)
        for window_name, mask in windows:
            if not np.any(mask):
                continue
            selected = (slice(None), mask)
            demand_limited = requested[selected] <= available[selected] + 1e-5
            rows.append(
                {
                    "policy": policy,
                    "source": source,
                    "window": window_name,
                    "observation_count": int(np.sum(mask)),
                    "request_limited_fraction": float(np.mean(demand_limited)),
                    "energy_limited_fraction": float(np.mean(~demand_limited)),
                    "mean_requested_qroom_w_m2": float(np.mean(requested[selected])),
                    "mean_available_power_w_m2": float(np.mean(available[selected])),
                    "mean_predicted_qroom_w_m2": float(np.mean(predicted_qroom[selected])),
                    "mean_energy_wh_m2": float(np.mean(aux["energy_wh_m2"][selected])),
                    "empty_buffer_fraction": float(
                        np.mean(aux["energy_wh_m2"][selected] <= 1e-4)
                    ),
                    "mean_cop": float(np.mean(aux["cop"][selected])),
                    "mean_pi": float(np.mean(aux["pi"][selected])),
                    "mean_hp_on": float(np.mean(aux["hp_on"][selected])),
                    "mean_pel_w_m2": float(np.mean(aux["pel_w_m2"][selected])),
                    "truth_qroom_w_m2": float(np.mean(profile.targets[mask, 1])),
                    "truth_pel_w_m2": float(np.mean(profile.targets[mask, 2])),
                }
            )
    return pd.DataFrame(rows)


def _rolling(values: np.ndarray | pd.Series, points: int = 7) -> np.ndarray:
    return (
        pd.Series(np.asarray(values))
        .rolling(points, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def _write_dashboard(
    profiles: dict[str, object],
    scenarios_by_policy: dict[str, dict[str, np.ndarray]],
    input_columns: tuple[str, ...],
    output_path: Path,
) -> None:
    policies = ("setpoint", "force_off")
    titles: list[str] = []
    for row_title in (
        "Tin: target, full mean and 95% band",
        "Seven-day mean Tin bias by source",
        "Seven-day mean Tin 95% width by source",
        "Full-model seven-day power bias",
    ):
        titles.extend(f"{policy}: {row_title}" for policy in policies)
    figure = make_subplots(
        rows=4,
        cols=2,
        shared_xaxes="columns",
        subplot_titles=tuple(titles),
        vertical_spacing=0.065,
        horizontal_spacing=0.055,
    )
    setpoint_index = input_columns.index(
        "FL0_THZ0, Zone Thermostat Heating Setpoint Temperature"
    )
    for column, policy in enumerate(policies, start=1):
        profile = profiles[policy]
        scenarios = scenarios_by_policy[policy]
        datetime = np.asarray(profile.datetime)
        full = scenarios["full"]
        full_mean = np.mean(full, axis=0)
        lower = np.quantile(full[:, :, 0], 0.025, axis=0)
        upper = np.quantile(full[:, :, 0], 0.975, axis=0)
        figure.add_trace(
            go.Scatter(x=datetime, y=lower, line={"width": 0}, showlegend=False, hoverinfo="skip"),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=upper,
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(63,109,179,0.20)",
                name="Full 95% interval",
                legendgroup="full_interval",
                showlegend=column == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=profile.targets[:, 0],
                name="EnergyPlus Tin",
                legendgroup="truth",
                line={"color": COLORS["truth"], "width": 1.2},
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=full_mean[:, 0],
                name="Full scenario mean",
                legendgroup="full",
                line={"color": COLORS["full"], "width": 1.3},
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=datetime,
                y=profile.inputs[:, setpoint_index],
                name="Tset",
                legendgroup="setpoint",
                line={"color": COLORS["setpoint"], "dash": "dot", "width": 1},
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        truth_tin = np.asarray(profile.targets[:, 0])
        for name in TRACE_ABLATIONS:
            mean = np.mean(scenarios[name][:, :, 0], axis=0)
            bias = _rolling(mean - truth_tin, 96 * 7)
            figure.add_trace(
                go.Scatter(
                    x=datetime,
                    y=bias,
                    name=name,
                    legendgroup=name,
                    line={"color": COLORS[name], "width": 1.2},
                    showlegend=column == 1,
                ),
                row=2,
                col=column,
            )
            width = (
                np.quantile(scenarios[name][:, :, 0], 0.975, axis=0)
                - np.quantile(scenarios[name][:, :, 0], 0.025, axis=0)
            )
            figure.add_trace(
                go.Scatter(
                    x=datetime,
                    y=_rolling(width, 96 * 7),
                    name=name,
                    legendgroup=name,
                    line={"color": COLORS[name], "width": 1.2},
                    showlegend=False,
                ),
                row=3,
                col=column,
            )
        for channel, name in ((1, "Qroom"), (2, "Pel_SH")):
            residual = _rolling(full_mean[:, channel] - profile.targets[:, channel], 96 * 7)
            figure.add_trace(
                go.Scatter(
                    x=datetime,
                    y=residual,
                    name=f"{name} bias",
                    legendgroup=f"{name}_bias",
                    line={"color": COLORS[name], "width": 1.2},
                    showlegend=column == 1,
                ),
                row=4,
                col=column,
            )
    figure.add_hline(y=0, line_dash="dot", line_color="#8a969c", row=2)
    figure.add_hline(y=0, line_dash="dot", line_color="#8a969c", row=4)
    figure.update_yaxes(title_text="Tin [degC]", row=1, col=1)
    figure.update_yaxes(title_text="Bias [degC]", row=2, col=1)
    figure.update_yaxes(title_text="95% width [degC]", row=3, col=1)
    figure.update_yaxes(title_text="Bias [W/m2]", row=4, col=1)
    figure.update_xaxes(title_text="Time", row=4)
    figure.update_layout(
        title="Winter underprediction and uncertainty-source diagnosis",
        template="plotly_white",
        height=1550,
        width=1850,
        hovermode="x unified",
        hoverlabel={"namelength": -1, "align": "left"},
        legend={"orientation": "h", "y": -0.055},
    )
    figure.write_html(output_path)


def run(args: argparse.Namespace) -> Path:
    artifact = load_training_artifact(args.artifact_dir)
    model = artifact.model
    if not isinstance(model, ProbabilisticContractingClosedLoopHPEmulator):
        raise TypeError("Artifact is not a probabilistic contracting closed-loop HP model")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_by_policy: dict[str, dict[str, np.ndarray]] = {}
    auxiliary_by_policy: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    metric_rows: list[dict[str, float | int | str | bool]] = []
    bad_window_rows: list[dict[str, float | str]] = []
    for policy_index, (policy, profile) in enumerate(profiles.items()):
        policy_scenarios: dict[str, np.ndarray] = {}
        policy_auxiliary: dict[str, tuple[np.ndarray, ...]] = {}
        for ablation in ABLATIONS:
            result = _controlled_scenarios(
                model,
                profile,
                artifact.scalers,
                metadata_columns,
                ablation=ablation,
                num_scenarios=args.num_scenarios,
                seed=args.seed + 1000 * policy_index,
                return_hp_aux=ablation.name in {"deterministic_center", "full"},
            )
            if isinstance(result, tuple):
                values, hp_aux = result
                policy_auxiliary[ablation.name] = hp_aux
            else:
                values = result
            policy_scenarios[ablation.name] = values
            regimes = _regimes(profile, input_columns)
            for regime, mask in regimes.items():
                for channel, (signal, unit) in enumerate(zip(SIGNALS, UNITS)):
                    metric_rows.append(
                        {
                            "policy": policy,
                            "ablation": ablation.name,
                            "xi": ablation.xi,
                            "process": ablation.process,
                            "hp": ablation.hp,
                            "regime": regime,
                            "signal": signal,
                            "unit": unit,
                            **_signal_metrics(values, profile.targets, mask, channel),
                        }
                    )
            winter_tin = _signal_metrics(
                values,
                profile.targets,
                regimes["winter"],
                0,
            )
            print(
                f"policy={policy} source={ablation.name} "
                f"winter_Tin_width={winter_tin['mean_width_95']:.3f} "
                f"winter_Tin_bias={winter_tin['mean_bias']:.3f}",
                flush=True,
            )
        scenarios_by_policy[policy] = policy_scenarios
        auxiliary_by_policy[policy] = policy_auxiliary
        setpoint_index = input_columns.index(
            "FL0_THZ0, Zone Thermostat Heating Setpoint Temperature"
        )
        force_off_index = (
            input_columns.index("hp_force_off_requested")
            if "hp_force_off_requested" in input_columns
            else -1
        )
        daily = _daily_diagnostics(
            profile,
            policy_scenarios,
            setpoint_index=setpoint_index,
            force_off_index=force_off_index,
        )
        daily.to_csv(args.output_dir / f"{policy}_daily_diagnostics.csv")
        bad_window_rows.extend(_select_bad_windows(policy, daily))
    metrics = pd.DataFrame(metric_rows)
    bad_windows = pd.DataFrame(bad_window_rows)
    metrics.to_csv(args.output_dir / "source_regime_metrics.csv", index=False)
    bad_windows.to_csv(args.output_dir / "worst_winter_windows.csv", index=False)
    buffer_diagnostics = pd.concat(
        [
            _buffer_diagnostics(
                policy,
                profiles[policy],
                auxiliary_by_policy[policy],
                bad_window_rows,
            )
            for policy in profiles
        ],
        ignore_index=True,
    )
    buffer_diagnostics.to_csv(
        args.output_dir / "buffer_limiter_diagnostics.csv",
        index=False,
    )
    _write_dashboard(
        profiles,
        scenarios_by_policy,
        input_columns,
        args.output_dir / "winter_bias_uncertainty_diagnostics.html",
    )
    summary = {
        "artifact_dir": str(args.artifact_dir),
        "persistent_xi_pathways": list(model.persistent_xi_pathways),
        "profile_id": args.profile_id,
        "num_scenarios": args.num_scenarios,
        "worst_winter_windows": bad_window_rows,
    }
    (args.output_dir / "winter_bias_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"saved_winter_diagnostics={args.output_dir}")
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
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
    parser.add_argument("--profile-id", type=int, default=11132764)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20015)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("artifact_dir", "setpoint_dataset", "force_off_dataset"):
        value = getattr(args, name)
        if not value.exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {value}")
    if args.num_scenarios < 2:
        parser.error("--num-scenarios must be at least 2")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
