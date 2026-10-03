"""Train and score structured models against genuinely vanilla LSTM controls.

The default recipe is frozen from the best structured run immediately before
the experimental Markov Qroom gate. All four arms use the same building split,
raw inputs, rollout windows, optimizer budget, and annual evaluation protocol.
The stochastic vanilla arm has no HP-specific features, heads, or losses.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .artifacts import json_safe, load_artifact
from .config import (
    ExperimentConfig,
    LSTMConfig,
    OptimizerConfig,
    ProbabilisticLSTMConfig,
)
from .profiles import saved_profile_ids
from .registry import get_model_spec
from .scoring import score_artifact
from .trainer import train

DEFAULT_CONFIG = (
    Path(__file__).parent
    / "configs"
    / "fair_structured_vs_probabilistic_lstm.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched deterministic/probabilistic structured-versus-LSTM comparison."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "fair_structured_vs_probabilistic_lstm"
        ),
    )
    parser.add_argument(
        "--structured-artifact-dir",
        "--probabilistic-structured-artifact-dir",
        dest="probabilistic_structured_artifact_dir",
        type=Path,
        help="Reuse a trained probabilistic structured artifact.",
    )
    parser.add_argument(
        "--lstm-artifact-dir",
        "--probabilistic-lstm-artifact-dir",
        dest="probabilistic_lstm_artifact_dir",
        type=Path,
        help="Reuse a trained probabilistic LSTM artifact.",
    )
    parser.add_argument(
        "--deterministic-structured-artifact-dir",
        type=Path,
        help="Reuse a trained deterministic structured artifact.",
    )
    parser.add_argument(
        "--deterministic-lstm-artifact-dir",
        type=Path,
        help="Reuse a trained deterministic LSTM artifact.",
    )
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--evaluation-max-profiles", type=int)
    return parser.parse_args(argv)


def _tuple_fields(cls) -> set[str]:
    return {
        field.name
        for field in fields(cls)
        if field.name
        in {
            "variogram_lags",
            "variogram_output_weights",
            "setpoint_shock_timescales_hours",
            "output_weights",
        }
    }


def _probabilistic_lstm_config(values: dict[str, Any]) -> ProbabilisticLSTMConfig:
    converted = dict(values)
    for name in _tuple_fields(ProbabilisticLSTMConfig):
        if name in converted:
            converted[name] = tuple(converted[name])
    return ProbabilisticLSTMConfig(**converted)


def _lstm_config(values: dict[str, Any]) -> LSTMConfig:
    converted = dict(values)
    converted["output_weights"] = tuple(converted["output_weights"])
    return LSTMConfig(**converted)


def _experiment_configs(
    recipe: dict[str, Any],
    output_dir: Path,
    *,
    max_profiles: int | None,
    epochs: int | None,
    fixed_test_profile_ids: tuple[int, ...] = (),
    excluded_profile_ids: tuple[int, ...] = (),
) -> dict[str, ExperimentConfig]:
    optimizer_values = dict(recipe["optimizer"])
    if epochs is not None:
        optimizer_values["epochs"] = epochs
    common = dict(
        dataset_path=Path(recipe["dataset_path"]),
        additional_dataset_paths=tuple(
            Path(path) for path in recipe.get("additional_dataset_paths", ())
        ),
        max_profiles=(
            max_profiles if max_profiles is not None else int(recipe["max_profiles"])
        ),
        test_fraction=float(recipe["test_fraction"]),
        fixed_test_profile_ids=fixed_test_profile_ids,
        excluded_profile_ids=excluded_profile_ids,
        seed=int(recipe["seed"]),
        sequence_length=int(recipe["sequence_length"]),
        stride=int(recipe["stride"]),
        rotate_window_starts=bool(recipe["rotate_window_starts"]),
        hp_power_area_normalization=str(recipe["hp_power_area_normalization"]),
        optimizer=OptimizerConfig(**optimizer_values),
    )
    deterministic_structured = ExperimentConfig(
        model_name="closed_loop_hp_contracting_deterministic",
        output_dir=output_dir / "deterministic_structured",
        legacy_overrides=dict(recipe["deterministic_structured"]),
        **common,
    )
    deterministic_lstm = ExperimentConfig(
        model_name="closed_loop_hp_lstm",
        output_dir=output_dir / "deterministic_lstm",
        lstm=_lstm_config(recipe["deterministic_lstm"]),
        **common,
    )
    probabilistic_structured = ExperimentConfig(
        model_name="closed_loop_hp_contracting_probabilistic",
        output_dir=output_dir / "probabilistic_structured",
        legacy_overrides=dict(recipe["probabilistic_structured"]),
        **common,
    )
    probabilistic_lstm = ExperimentConfig(
        model_name="closed_loop_hp_vanilla_probabilistic_lstm",
        output_dir=output_dir / "probabilistic_lstm",
        probabilistic_lstm=_probabilistic_lstm_config(
            recipe["probabilistic_lstm"]
        ),
        **common,
    )
    return {
        "deterministic_structured": deterministic_structured,
        "deterministic_lstm": deterministic_lstm,
        "probabilistic_structured": probabilistic_structured,
        "probabilistic_lstm": probabilistic_lstm,
    }


def _tree_parameter_count(artifact_dir: Path) -> int:
    artifact = load_artifact(artifact_dir)
    leaves = eqx.filter(artifact.model, eqx.is_inexact_array)
    return int(
        sum(
            np.asarray(value).size
            for value in jax.tree.leaves(leaves)
            if value is not None
        )
    )


def _check_matching_splits(artifact_dirs: dict[str, Path]) -> None:
    artifacts = {
        arm: load_artifact(path) for arm, path in artifact_dirs.items()
    }
    reference_arm = next(iter(artifacts))
    mismatches: dict[str, dict[str, int]] = {}
    for source in ("selected", "train", "test"):
        reference_ids = saved_profile_ids(artifacts[reference_arm], source)
        for arm, artifact in artifacts.items():
            ids = saved_profile_ids(artifact, source)
            if ids != reference_ids:
                mismatches.setdefault(arm, {})[source] = len(ids)
    if mismatches:
        raise ValueError(
            "Comparison artifacts do not contain the same ordered selected/train/test "
            f"splits. reference={reference_arm} mismatches={mismatches}"
        )


def _check_matching_models(
    artifact_dirs: dict[str, Path],
    configs: dict[str, ExperimentConfig],
) -> None:
    for arm, path in artifact_dirs.items():
        actual = load_artifact(path).spec.name
        expected = configs[arm].model_name
        if actual != expected:
            raise ValueError(
                f"Artifact for {arm!r} contains model {actual!r}; expected "
                f"{expected!r}. The fair probabilistic LSTM arm now requires the "
                "genuinely vanilla model, not the previous semi-structured LSTM."
            )


def _flexibility_error_metrics(
    score_dir: Path,
    *,
    output_suffix: str = "",
    metric_prefix: str = "flex",
) -> dict[str, float]:
    path = score_dir / f"flexibility_kpis{output_suffix}.csv"
    if not path.exists():
        return {}
    values = pd.read_csv(path)
    values = values[values["valid"].astype(bool)]
    index = ["profile_id", "horizon_hours"]
    metrics: dict[str, float] = {}
    for direction, column in (
        ("up", "up_flex_wh_m2_k"),
        ("down", "down_flex_wh_m2_k"),
    ):
        paired = values.pivot_table(
            index=index,
            columns="signal",
            values=column,
            aggfunc="first",
        ).dropna(subset=["simulation", "emulation"])
        if paired.empty:
            continue
        error = paired["emulation"].to_numpy() - paired["simulation"].to_numpy()
        metrics[f"{metric_prefix}_{direction}_rmse_wh_m2_k"] = float(
            np.sqrt(np.mean(error**2))
        )
        metrics[f"{metric_prefix}_{direction}_bias_wh_m2_k"] = float(np.mean(error))
    return metrics


def _write_dashboard(
    path: Path,
    comparison: pd.DataFrame,
    coverage: pd.DataFrame,
    temporal_calibration: pd.DataFrame,
) -> None:
    figure = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"type": "table", "colspan": 3}, None, None],
            [{}, {}, {}],
            [{}, {}, {}],
        ],
        subplot_titles=(
            "Annual point metrics",
            "RMSE by output",
            "Regression and direct-F calibration error",
            "Temporal calibration summary",
            "Tin trace reliability",
            "Qroom trace reliability",
            "Pel_SH trace reliability",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )
    display_columns = [
        "arm",
        "temperature_c_rmse",
        "qroom_w_m2_rmse",
        "pel_w_m2_rmse",
        "flex_up_rmse_wh_m2_k",
        "flex_down_rmse_wh_m2_k",
        "direct_f_up_rmse_wh_m2_k",
        "direct_f_down_rmse_wh_m2_k",
        "temperature_c_bias",
        "parameter_count",
    ]
    display = comparison[display_columns].copy()
    for column in display_columns[1:-1]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    figure.add_trace(
        go.Table(
            header={"values": display_columns, "fill_color": "#e8eef5"},
            cells={"values": [display[column] for column in display_columns]},
        ),
        row=1,
        col=1,
    )
    colors = {
        "deterministic_structured": "#1769aa",
        "deterministic_lstm": "#ef6c00",
        "probabilistic_structured": "#00897b",
        "probabilistic_lstm": "#d32f2f",
    }
    for row in comparison.itertuples(index=False):
        label = str(row.arm)
        figure.add_trace(
            go.Bar(
                x=["Tin [C]", "Qroom [W/m2]", "Pel [W/m2]"],
                y=[
                    row.temperature_c_rmse,
                    row.qroom_w_m2_rmse,
                    row.pel_w_m2_rmse,
                ],
                name=label,
                marker_color=colors[label],
            ),
            row=2,
            col=1,
        )
    if not coverage.empty:
        grouped = (
            coverage.assign(
                absolute_calibration_error=coverage["calibration_error"].abs()
            )
            .groupby(
                ["arm", "kpi_estimator", "direction", "horizon_hours"],
                as_index=False,
            )[
                "absolute_calibration_error"
            ]
            .mean()
        )
        for arm in ("probabilistic_structured", "probabilistic_lstm"):
            for estimator, dash in (("regression", "solid"), ("direct_ratio", "dot")):
                part = grouped[
                    (grouped["arm"] == arm)
                    & (grouped["kpi_estimator"] == estimator)
                ]
                if part.empty:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=[
                            f"{row.direction} {row.horizon_hours:g}h"
                            for row in part.itertuples()
                        ],
                        y=part["absolute_calibration_error"],
                        mode="lines+markers",
                        name=f"{arm} {estimator}",
                        line={"color": colors[arm], "dash": dash},
                    ),
                    row=2,
                    col=2,
                )
    probabilistic_arms = ("probabilistic_structured", "probabilistic_lstm")
    if not temporal_calibration.empty:
        temporal_summary = (
            temporal_calibration.groupby(["arm", "signal"], as_index=False)[
                "abs_calibration_error"
            ]
            .mean()
        )
        for arm in probabilistic_arms:
            part = temporal_summary[temporal_summary["arm"] == arm]
            figure.add_trace(
                go.Bar(
                    x=part["signal"],
                    y=part["abs_calibration_error"],
                    name=f"{arm} trace calibration",
                    marker_color=colors[arm],
                    showlegend=False,
                ),
                row=2,
                col=3,
            )
        for column, signal in enumerate(("Tin", "Qroom", "Pel_SH"), start=1):
            for arm in probabilistic_arms:
                part = temporal_calibration[
                    (temporal_calibration["arm"] == arm)
                    & (temporal_calibration["signal"] == signal)
                ].sort_values("quantile")
                if part.empty:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=part["quantile"],
                        y=part["empirical_coverage"],
                        mode="lines+markers",
                        name=arm,
                        line={"color": colors[arm]},
                        showlegend=False,
                    ),
                    row=3,
                    col=column,
                )
            figure.add_trace(
                go.Scatter(
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    mode="lines",
                    line={"color": "black", "dash": "dash"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=column,
            )
    figure.update_yaxes(title_text="RMSE", row=2, col=1)
    figure.update_yaxes(title_text="Mean |empirical - nominal|", row=2, col=2)
    figure.update_yaxes(title_text="Mean |empirical - nominal|", row=2, col=3)
    for column in range(1, 4):
        figure.update_xaxes(
            title_text="Nominal quantile", range=[0.0, 1.0], row=3, col=column
        )
        figure.update_yaxes(
            title_text="Empirical rank CDF", range=[0.0, 1.0], row=3, col=column
        )
    figure.update_layout(
        title="Fair structured-model versus vanilla LSTM comparison",
        template="plotly_white",
        height=1350,
        width=1650,
        barmode="group",
        legend={"orientation": "h", "y": -0.12},
    )
    figure.write_html(path)


def _write_pel_dynamics_comparison_dashboard(
    path: Path,
    *,
    transition: pd.DataFrame,
    conditional: pd.DataFrame,
    run_duration: pd.DataFrame,
    acf: pd.DataFrame,
    annual_energy: pd.DataFrame,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "HP start reliability",
            "HP stay-on reliability",
            "Conditional Pel rank calibration",
            "Active-run duration distribution",
            "HP-mode autocorrelation",
            "Annual Pel energy calibration",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )
    colors = {
        "probabilistic_structured": "#00897b",
        "probabilistic_lstm": "#d32f2f",
    }
    arms = tuple(colors)
    for column, transition_name in enumerate(
        ("start_given_observed_off", "stay_on_given_observed_on"), start=1
    ):
        for arm in arms:
            part = transition[
                (transition["arm"] == arm)
                & (transition["transition"] == transition_name)
            ]
            figure.add_trace(
                go.Scatter(
                    x=part["predicted_probability"],
                    y=part["observed_frequency"],
                    mode="lines+markers",
                    name=arm,
                    legendgroup=arm,
                    line={"color": colors[arm]},
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line={"color": "black", "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    for arm in arms:
        part = conditional[conditional["arm"] == arm]
        figure.add_trace(
            go.Bar(
                x=part["condition"] + ": " + part["stratum"],
                y=part["calibration_mae"],
                name=arm,
                marker_color=colors[arm],
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    truth = run_duration[
        (run_duration["arm"] == arms[0])
        & (run_duration["signal"] == "simulation")
    ]
    figure.add_trace(
        go.Scatter(
            x=truth["bin_label"], y=truth["probability"],
            mode="lines+markers", name="simulation",
            line={"color": "#1769aa", "width": 3},
        ),
        row=2,
        col=1,
    )
    for arm in arms:
        part = run_duration[
            (run_duration["arm"] == arm)
            & (run_duration["signal"] == "emulation")
        ]
        figure.add_trace(
            go.Scatter(
                x=part["bin_label"], y=part["probability"],
                mode="lines+markers", name=f"{arm} run duration",
                line={"color": colors[arm]},
            ),
            row=2,
            col=1,
        )
    truth_acf = acf[acf["arm"] == arms[0]]
    figure.add_trace(
        go.Scatter(
            x=truth_acf["lag_hours"], y=truth_acf["truth_acf"],
            mode="lines+markers", name="simulation ACF",
            line={"color": "#1769aa", "width": 3},
        ),
        row=2,
        col=2,
    )
    for arm in arms:
        part = acf[acf["arm"] == arm]
        figure.add_trace(
            go.Scatter(
                x=part["lag_hours"], y=part["scenario_mean_acf"],
                mode="lines+markers", name=f"{arm} ACF",
                line={"color": colors[arm]},
            ),
            row=2,
            col=2,
        )
    for arm in arms:
        part = annual_energy[annual_energy["arm"] == arm]
        figure.add_trace(
            go.Scatter(
                x=part["quantile"], y=part["empirical_coverage"],
                mode="lines+markers", name=f"{arm} annual energy",
                line={"color": colors[arm]},
            ),
            row=2,
            col=3,
        )
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line={"color": "black", "dash": "dash"}, showlegend=False,
        ),
        row=2,
        col=3,
    )
    figure.update_xaxes(title_text="predicted probability", range=[0, 1], row=1, col=1)
    figure.update_yaxes(title_text="observed frequency", range=[0, 1], row=1, col=1)
    figure.update_xaxes(title_text="predicted probability", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="observed frequency", range=[0, 1], row=1, col=2)
    figure.update_yaxes(title_text="mean |rank CDF error|", row=1, col=3)
    figure.update_xaxes(title_text="run duration [h]", row=2, col=1)
    figure.update_yaxes(title_text="probability", row=2, col=1)
    figure.update_xaxes(title_text="lag [h]", row=2, col=2)
    figure.update_yaxes(title_text="mode autocorrelation", row=2, col=2)
    figure.update_xaxes(title_text="nominal quantile", range=[0, 1], row=2, col=3)
    figure.update_yaxes(title_text="empirical rank CDF", range=[0, 1], row=2, col=3)
    figure.update_layout(
        title="Probabilistic HP switching and conditional calibration comparison",
        template="plotly_white",
        height=980,
        width=1650,
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    figure.write_html(path)


def run(args: argparse.Namespace) -> Path:
    recipe = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = _experiment_configs(
        recipe,
        args.output_dir,
        max_profiles=args.max_profiles,
        epochs=args.epochs,
    )
    (args.output_dir / "resolved_recipe.json").write_text(
        json.dumps(
            {
                "source_recipe": str(args.config),
                "recipe": recipe,
                "experiments": {
                    arm: json_safe(config) for arm, config in configs.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )

    supplied_artifacts = {
        "deterministic_structured": args.deterministic_structured_artifact_dir,
        "deterministic_lstm": args.deterministic_lstm_artifact_dir,
        "probabilistic_structured": args.probabilistic_structured_artifact_dir,
        "probabilistic_lstm": args.probabilistic_lstm_artifact_dir,
    }
    previous_duration_minutes: dict[str, float] = {}
    previous_comparison_path = args.output_dir / "model_comparison.csv"
    if previous_comparison_path.exists():
        previous_comparison = pd.read_csv(previous_comparison_path)
        if {"arm", "training_minutes"}.issubset(previous_comparison.columns):
            previous_duration_minutes = dict(
                zip(
                    previous_comparison["arm"].astype(str),
                    previous_comparison["training_minutes"].astype(float),
                )
            )
    durations: dict[str, float] = {}
    artifact_dirs: dict[str, Path] = {}
    for arm, config in configs.items():
        supplied = supplied_artifacts[arm]
        if supplied is not None:
            artifact_dirs[arm] = Path(supplied)
            durations[arm] = 60.0 * previous_duration_minutes.get(arm, 0.0)
            continue
        started = time.perf_counter()
        artifact_dirs[arm] = Path(train(config))
        durations[arm] = time.perf_counter() - started
    _check_matching_models(artifact_dirs, configs)
    _check_matching_splits(artifact_dirs)

    evaluation = recipe["evaluation"]
    rows: list[dict[str, Any]] = []
    coverage_frames: list[pd.DataFrame] = []
    temporal_calibration_frames: list[pd.DataFrame] = []
    temporal_histogram_frames: list[pd.DataFrame] = []
    pel_dynamics_frames: dict[str, list[pd.DataFrame]] = {
        "transition": [],
        "conditional": [],
        "run_duration": [],
        "acf": [],
        "annual_energy": [],
    }
    for arm, artifact_dir in artifact_dirs.items():
        artifact = load_artifact(artifact_dir)
        probabilistic = artifact.spec.probabilistic
        score_dir = args.output_dir / "evaluation" / arm
        score_artifact(
            artifact_dir,
            dataset_path=Path(recipe["dataset_path"]),
            output_dir=score_dir,
            profile_source="test",
            max_profiles=args.evaluation_max_profiles,
            kpi_mode="scenario_average" if probabilistic else "mean",
            num_scenarios=(
                int(evaluation["num_scenarios"]) if probabilistic else 0
            ),
            hp_scenario_mode="bernoulli",
            ventilation_rollout_mode=str(evaluation["ventilation_rollout_mode"]),
            horizons_hours=tuple(evaluation["horizons_hours"]),
            min_setpoint_change=float(evaluation["min_setpoint_change"]),
            min_events=int(evaluation["min_events"]),
            kpi_estimator=str(evaluation["kpi_estimator"]),
            controls=str(evaluation["controls"]),
            seed=int(recipe["seed"]),
            num_profile_plots=int(evaluation["num_profile_plots"]),
            evaluate_direct_f=True,
        )
        summary = json.loads((score_dir / "summary.json").read_text())
        rows.append(
            {
                "arm": arm,
                "family": "lstm" if "lstm" in arm else "structured",
                "probabilistic": probabilistic,
                "model_name": summary["model_name"],
                **summary["metrics"],
                **_flexibility_error_metrics(score_dir),
                **_flexibility_error_metrics(
                    score_dir,
                    output_suffix="_f",
                    metric_prefix="direct_f",
                ),
                **summary.get("pel_dynamics", {}),
                "parameter_count": _tree_parameter_count(artifact_dir),
                "training_minutes": durations[arm] / 60.0,
            }
        )
        coverage_path = score_dir / "flexibility_kpi_coverage.csv"
        if coverage_path.exists():
            frame = pd.read_csv(coverage_path)
            frame.insert(0, "arm", arm)
            frame.insert(1, "kpi_estimator", str(evaluation["kpi_estimator"]))
            coverage_frames.append(frame)
        direct_f_coverage_path = score_dir / "flexibility_kpi_coverage_f.csv"
        if direct_f_coverage_path.exists():
            frame = pd.read_csv(direct_f_coverage_path)
            frame.insert(0, "arm", arm)
            frame.insert(1, "kpi_estimator", "direct_ratio")
            coverage_frames.append(frame)
        temporal_path = score_dir / "temporal_trace_calibration.csv"
        if temporal_path.exists():
            frame = pd.read_csv(temporal_path)
            frame.insert(0, "arm", arm)
            temporal_calibration_frames.append(frame)
        histogram_path = score_dir / "temporal_trace_rank_histogram.csv"
        if histogram_path.exists():
            frame = pd.read_csv(histogram_path)
            frame.insert(0, "arm", arm)
            temporal_histogram_frames.append(frame)
        pel_paths = {
            "transition": score_dir / "pel_transition_reliability.csv",
            "conditional": score_dir / "pel_conditional_calibration_summary.csv",
            "run_duration": score_dir / "pel_run_duration_distribution.csv",
            "acf": score_dir / "pel_mode_autocorrelation.csv",
            "annual_energy": score_dir / "pel_annual_energy_calibration.csv",
        }
        for name, path in pel_paths.items():
            if path.exists():
                frame = pd.read_csv(path)
                frame.insert(0, "arm", arm)
                pel_dynamics_frames[name].append(frame)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    coverage = (
        pd.concat(coverage_frames, ignore_index=True)
        if coverage_frames
        else pd.DataFrame()
    )
    if not coverage.empty:
        coverage.to_csv(args.output_dir / "flexibility_coverage_comparison.csv", index=False)
    temporal_calibration = (
        pd.concat(temporal_calibration_frames, ignore_index=True)
        if temporal_calibration_frames
        else pd.DataFrame()
    )
    if not temporal_calibration.empty:
        temporal_calibration.to_csv(
            args.output_dir / "temporal_trace_calibration_comparison.csv",
            index=False,
        )
    if temporal_histogram_frames:
        pd.concat(temporal_histogram_frames, ignore_index=True).to_csv(
            args.output_dir / "temporal_trace_rank_histogram_comparison.csv",
            index=False,
        )
    combined_pel_frames: dict[str, pd.DataFrame] = {}
    for name, frames in pel_dynamics_frames.items():
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined_pel_frames[name] = combined
        if not combined.empty:
            combined.to_csv(
                args.output_dir / f"pel_{name}_comparison.csv",
                index=False,
            )
    dashboard = args.output_dir / "fair_comparison_dashboard.html"
    _write_dashboard(dashboard, comparison, coverage, temporal_calibration)
    if all(not frame.empty for frame in combined_pel_frames.values()):
        _write_pel_dynamics_comparison_dashboard(
            args.output_dir / "pel_dynamics_comparison.html",
            transition=combined_pel_frames["transition"],
            conditional=combined_pel_frames["conditional"],
            run_duration=combined_pel_frames["run_duration"],
            acf=combined_pel_frames["acf"],
            annual_energy=combined_pel_frames["annual_energy"],
        )
    print(comparison.to_string(index=False))
    print(f"saved_dashboard={dashboard}")
    return dashboard


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
