"""Compare flexibility-KPI coverage under three pre-event-state designs.

The same probabilistic yearly emulator scenarios are reused for:

``without_pre_power``
    Full event-study controls except for pre-window HP electric power.
``scenario_pre_power``
    The published/full estimator, using each scenario's own pre-window power
    and pre-event indoor temperature.
``fixed_observed_design``
    The approximation currently used by the differentiable training KPI
    score: every particle response is projected through the design built from
    the observed target trajectory.

This isolates the regression definition from Monte Carlo sampling and does
not retrain the emulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN,
    INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
)
from .data import DEFAULT_DATASET_PATH, ClosedLoopProfile
from .flexibility_event_study_kpis import (
    EventStudyConfig,
    _config_value,
    _coverage_rows,
    _coverage_summary,
    _load_closed_loop_profiles,
    _parse_quantiles,
    _resolve_horizons,
    _sample_scenario_traces,
    _saved_profile_ids,
    _select_profiles,
    regression_data_for_trace,
)
from .model_io import load_training_artifact


VARIANTS = (
    "without_pre_power",
    "scenario_pre_power",
    "fixed_observed_design",
)


def _row_standardize(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=1, keepdims=True)
    scales = np.std(values, axis=1, keepdims=True)
    scales = np.where(scales < 1e-8, 1.0, scales)
    return (values - means) / scales


def _window_means(
    values: np.ndarray,
    *,
    events: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.concatenate(
        [
            np.zeros((*values.shape[:-1], 1), dtype=np.float64),
            np.cumsum(values, axis=-1, dtype=np.float64),
        ],
        axis=-1,
    )
    pre = (
        cumulative[..., events] - cumulative[..., events - horizon_steps]
    ) / horizon_steps
    post = (
        cumulative[..., events + horizon_steps] - cumulative[..., events]
    ) / horizon_steps
    return pre, post


def _fit_coefficients(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return coefficients


def _fit_particle_coefficients(
    designs: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    xtx = np.einsum("kep,keq->kpq", designs, designs)
    xty = np.einsum("kep,ke->kp", designs, responses)
    scale = np.maximum(np.max(np.abs(xtx), axis=(1, 2)), 1.0)
    regularizer = (
        np.finfo(np.float64).eps
        * scale[:, np.newaxis, np.newaxis]
        * np.eye(xtx.shape[-1], dtype=np.float64)[np.newaxis, :, :]
    )
    try:
        return np.linalg.solve(
            xtx + regularizer,
            xty[..., np.newaxis],
        )[..., 0]
    except np.linalg.LinAlgError:
        return np.stack(
            [
                _fit_coefficients(design, response)
                for design, response in zip(designs, responses)
            ],
            axis=0,
        )


def _scenario_regression_quantities(
    profile: ClosedLoopProfile,
    scenarios: np.ndarray,
    *,
    observed_design: np.ndarray,
    events: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario_power = scenarios[:, :, 2].astype(np.float64)
    pre_power, post_power = _window_means(
        scenario_power,
        events=events,
        horizon_steps=horizon_steps,
    )
    response = post_power - pre_power

    # The preceding interval's end temperature is the event's pre-state.
    pre_temperature = scenarios[:, events - 1, 0].astype(np.float64)
    scenario_design = np.broadcast_to(
        observed_design[np.newaxis, :, :],
        (scenarios.shape[0], *observed_design.shape),
    ).copy()
    scenario_design[:, :, 3] = _row_standardize(pre_power)
    scenario_design[:, :, 4] = _row_standardize(pre_temperature)
    scenario_design_without_pre_power = np.delete(scenario_design, 3, axis=2)
    return response, scenario_design, scenario_design_without_pre_power


def _kpi_values(coefficients: np.ndarray, horizon_hours: float) -> tuple[np.ndarray, np.ndarray]:
    return horizon_hours * coefficients[..., 1], horizon_hours * coefficients[..., 2]


def _append_profile_row(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    event_count: int,
    true_coefficients: np.ndarray,
) -> None:
    up, down = _kpi_values(true_coefficients, horizon_hours)
    rows.append(
        {
            "variant": variant,
            "profile_id": int(profile_id),
            "horizon_hours": float(horizon_hours),
            "horizon_steps": int(horizon_steps),
            "valid": True,
            "sim_up_flex_wh_m2_k": float(up),
            "sim_down_flex_wh_m2_k": float(down),
            "event_count": int(event_count),
        }
    )


def _append_scenario_rows(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    profile_id: int,
    horizon_hours: float,
    horizon_steps: int,
    event_count: int,
    scenario_coefficients: np.ndarray,
) -> None:
    up, down = _kpi_values(scenario_coefficients, horizon_hours)
    for scenario_index in range(len(scenario_coefficients)):
        rows.append(
            {
                "variant": variant,
                "profile_id": int(profile_id),
                "scenario_index": int(scenario_index),
                "horizon_hours": float(horizon_hours),
                "horizon_steps": int(horizon_steps),
                "event_count": int(event_count),
                "valid": True,
                "up_flex_wh_m2_k": float(up[scenario_index]),
                "down_flex_wh_m2_k": float(down[scenario_index]),
            }
        )


def _sample_crps(samples: np.ndarray, observation: float) -> float:
    observation_term = float(np.mean(np.abs(samples - observation)))
    pairwise_term = float(
        0.5 * np.mean(np.abs(samples[:, np.newaxis] - samples[np.newaxis, :]))
    )
    return observation_term - pairwise_term


def _distribution_summary(
    profile_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, horizon), profile_part in profile_df.groupby(
        ["variant", "horizon_hours"], sort=True
    ):
        scenario_part = scenario_df[
            (scenario_df["variant"] == variant)
            & (scenario_df["horizon_hours"] == horizon)
        ]
        for direction in ("up", "down"):
            truth_column = f"sim_{direction}_flex_wh_m2_k"
            sample_column = f"{direction}_flex_wh_m2_k"
            pits: list[float] = []
            central_hits: list[float] = []
            widths: list[float] = []
            standard_deviations: list[float] = []
            biases: list[float] = []
            squared_errors: list[float] = []
            crps_values: list[float] = []
            for profile_row in profile_part.itertuples(index=False):
                profile_id = int(profile_row.profile_id)
                truth = float(getattr(profile_row, truth_column))
                samples = scenario_part.loc[
                    scenario_part["profile_id"] == profile_id,
                    sample_column,
                ].to_numpy(dtype=float)
                samples = samples[np.isfinite(samples)]
                if len(samples) == 0 or not np.isfinite(truth):
                    continue
                lower, upper = np.quantile(samples, [0.05, 0.95])
                mean = float(np.mean(samples))
                pits.append(float(np.mean(samples <= truth)))
                central_hits.append(float(lower <= truth <= upper))
                widths.append(float(upper - lower))
                standard_deviations.append(float(np.std(samples, ddof=0)))
                biases.append(mean - truth)
                squared_errors.append((mean - truth) ** 2)
                crps_values.append(_sample_crps(samples, truth))
            rows.append(
                {
                    "variant": variant,
                    "horizon_hours": float(horizon),
                    "direction": direction,
                    "profile_count": len(pits),
                    "mean_pit": float(np.mean(pits)),
                    "median_pit": float(np.median(pits)),
                    "central_90_coverage": float(np.mean(central_hits)),
                    "mean_90_interval_width_wh_m2_k": float(np.mean(widths)),
                    "mean_scenario_std_wh_m2_k": float(np.mean(standard_deviations)),
                    "mean_bias_wh_m2_k": float(np.mean(biases)),
                    "rmse_of_scenario_mean_wh_m2_k": float(np.sqrt(np.mean(squared_errors))),
                    "mean_crps_wh_m2_k": float(np.mean(crps_values)),
                }
            )
    return pd.DataFrame(rows)


def _coverage_for_variants(
    profile_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        rows = _coverage_rows(
            profile_df[profile_df["variant"] == variant],
            scenario_df[scenario_df["variant"] == variant],
            quantiles,
        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame.insert(0, "variant", variant)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _coverage_summary_for_variants(coverage: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        frame = pd.DataFrame(
            _coverage_summary(coverage[coverage["variant"] == variant])
        )
        if not frame.empty:
            frame.insert(0, "variant", variant)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_coverage_html(path: Path, coverage: pd.DataFrame, *, title: str) -> None:
    titles = []
    labels = {
        "without_pre_power": "Without P_pre",
        "scenario_pre_power": "Scenario-specific P_pre",
        "fixed_observed_design": "Fixed observed design",
    }
    for variant in VARIANTS:
        titles.extend([f"{labels[variant]}: upward", f"{labels[variant]}: downward"])
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=titles,
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for row, variant in enumerate(VARIANTS, start=1):
        for col, direction in ((1, "up"), (2, "down")):
            part = coverage[
                (coverage["variant"] == variant)
                & (coverage["direction"] == direction)
            ]
            for horizon_index, horizon in enumerate(sorted(part["horizon_hours"].unique())):
                horizon_part = part[part["horizon_hours"] == horizon]
                figure.add_trace(
                    go.Scatter(
                        x=horizon_part["quantile"],
                        y=horizon_part["empirical_coverage"],
                        mode="lines+markers",
                        name=f"H={horizon:g} h",
                        legendgroup=f"horizon_{horizon}",
                        showlegend=(row == 1 and col == 1),
                        line=dict(color=colors[horizon_index % len(colors)]),
                    ),
                    row=row,
                    col=col,
                )
            figure.add_trace(
                go.Scatter(
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    mode="lines",
                    name="ideal",
                    legendgroup="ideal",
                    showlegend=(row == 1 and col == 1),
                    line=dict(color="black", dash="dash"),
                ),
                row=row,
                col=col,
            )
            figure.update_xaxes(title_text="nominal quantile", range=[0, 1], row=row, col=col)
            figure.update_yaxes(
                title_text="empirical P(true <= quantile)",
                range=[0, 1],
                row=row,
                col=col,
            )
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=1550,
        width=1450,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def _write_report(
    path: Path,
    *,
    coverage_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    profiles: int,
    scenarios: int,
) -> None:
    labels = {
        "without_pre_power": "without P_pre",
        "scenario_pre_power": "scenario-specific P_pre",
        "fixed_observed_design": "fixed observed design",
    }
    lines = [
        "# Flexibility KPI P_pre Coverage Ablation",
        "",
        f"The comparison uses {scenarios} identical probabilistic yearly scenarios for each of {profiles} test buildings.",
        "",
        "## Coverage Summary",
        "",
        "| Variant | H [h] | Direction | Coverage MACE | Mean PIT | Central 90% coverage | Mean scenario SD | Mean CRPS |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    merged = coverage_summary.merge(
        distribution_summary,
        on=["variant", "horizon_hours", "direction", "profile_count"],
        how="left",
    )
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {labels[row.variant]} | {row.horizon_hours:g} | {row.direction} | "
            f"{row.mean_abs_calibration_error:.3f} | {row.mean_pit_x:.3f} | "
            f"{row.central_90_coverage:.3f} | {row.mean_scenario_std_wh_m2_k:.3f} | "
            f"{row.mean_crps_wh_m2_k:.3f} |"
        )
    aggregate = (
        merged.groupby("variant", sort=False)
        .agg(
            coverage_mace=("mean_abs_calibration_error", "mean"),
            mean_pit=("mean_pit_x", "mean"),
            central_90_coverage=("central_90_coverage", "mean"),
            mean_crps=("mean_crps_wh_m2_k", "mean"),
        )
        .reset_index()
    )
    lines.extend(
        [
            "",
            "## Aggregate Comparison",
            "",
            "| Variant | Coverage MACE | Mean PIT | Central 90% coverage | Mean CRPS |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {labels[row.variant]} | {row.coverage_mace:.3f} | "
            f"{row.mean_pit:.3f} | {row.central_90_coverage:.3f} | "
            f"{row.mean_crps:.3f} |"
        )
    lines.extend(
        [
            "",
            "`scenario-specific P_pre` is the current ex-post estimator. `fixed observed design` mirrors the design approximation in the differentiable training KPI score, while `without P_pre` removes only the pre-power column and retains all other controls.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_ablation(args: argparse.Namespace) -> None:
    if args.num_scenarios < 2:
        raise ValueError("--num-scenarios must be at least 2")
    quantiles = _parse_quantiles(args.coverage_quantiles)
    artifact = load_training_artifact(args.artifact_dir)
    metadata = artifact.metadata
    dataset_path = args.dataset or Path(
        _config_value(metadata, "dataset_path", DEFAULT_DATASET_PATH)
    )
    dt_hours = float(
        args.dt_hours
        if args.dt_hours is not None
        else _config_value(metadata, "hp_dt_hours", 0.25)
    )
    horizons_hours, horizon_steps = _resolve_horizons(args.horizons_hours, dt_hours)
    config = EventStudyConfig(
        horizons_hours=horizons_hours,
        horizon_steps=horizon_steps,
        dt_hours=dt_hours,
        min_setpoint_change=float(args.min_setpoint_change),
        min_events=int(args.min_events),
        controls="full",
    )
    candidate_ids = _saved_profile_ids(metadata, args.profile_source)
    profile_ids = _select_profiles(
        candidate_ids,
        max_profiles=args.max_profiles,
        seed=args.seed,
    )
    profiles = _load_closed_loop_profiles(
        dataset_path,
        profile_ids,
        include_space_heating_availability=(
            SPACE_HEATING_AVAILABILITY_COLUMN in metadata.get("input_columns", [])
        ),
        include_internal_gains=(
            INTERNAL_GAIN_PER_FLOOR_AREA_COLUMN in metadata.get("input_columns", [])
        ),
        include_dhw_request=(
            DHW_MIXED_WATER_PER_HEATED_AREA_COLUMN in metadata.get("input_columns", [])
        ),
        hp_power_area_normalization=str(
            _config_value(metadata, "hp_power_area_normalization", "zone_floor_area")
        ),
    )

    profile_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    key = jax.random.PRNGKey(args.seed)
    for profile_index, profile in enumerate(profiles, start=1):
        key, profile_key = jax.random.split(key)
        scenarios = _sample_scenario_traces(
            artifact,
            profile,
            key=profile_key,
            num_scenarios=args.num_scenarios,
            hp_scenario_mode=args.prob_hp_scenario_mode,
        )
        scenarios = np.asarray(scenarios, dtype=np.float64)
        for horizon_hours, steps in zip(config.horizons_hours, config.horizon_steps):
            observed = regression_data_for_trace(
                profile,
                profile.targets,
                horizon_steps=steps,
                config=config,
            )
            if len(observed.response) < max(config.min_events, observed.design.shape[1] + 1):
                continue

            scenario_response, scenario_design, scenario_design_without_pre = (
                _scenario_regression_quantities(
                    profile,
                    scenarios,
                    observed_design=observed.design,
                    events=observed.events,
                    horizon_steps=steps,
                )
            )
            observed_design_without_pre = np.delete(observed.design, 3, axis=1)
            true_full = _fit_coefficients(observed.design, observed.response)
            true_without_pre = _fit_coefficients(
                observed_design_without_pre,
                observed.response,
            )
            scenario_full = _fit_particle_coefficients(
                scenario_design,
                scenario_response,
            )
            scenario_without_pre = _fit_particle_coefficients(
                scenario_design_without_pre,
                scenario_response,
            )
            fixed_designs = np.broadcast_to(
                observed.design[np.newaxis, :, :],
                (args.num_scenarios, *observed.design.shape),
            )
            scenario_fixed = _fit_particle_coefficients(
                fixed_designs,
                scenario_response,
            )

            variant_values = {
                "without_pre_power": (true_without_pre, scenario_without_pre),
                "scenario_pre_power": (true_full, scenario_full),
                "fixed_observed_design": (true_full, scenario_fixed),
            }
            for variant, (true_coefficients, scenario_coefficients) in variant_values.items():
                _append_profile_row(
                    profile_rows,
                    variant=variant,
                    profile_id=profile.profile_id,
                    horizon_hours=horizon_hours,
                    horizon_steps=steps,
                    event_count=len(observed.response),
                    true_coefficients=true_coefficients,
                )
                _append_scenario_rows(
                    scenario_rows,
                    variant=variant,
                    profile_id=profile.profile_id,
                    horizon_hours=horizon_hours,
                    horizon_steps=steps,
                    event_count=len(observed.response),
                    scenario_coefficients=scenario_coefficients,
                )
        if profile_index % 10 == 0 or profile_index == len(profiles):
            print(f"processed_profiles={profile_index}/{len(profiles)}")

    profile_df = pd.DataFrame(profile_rows)
    scenario_df = pd.DataFrame(scenario_rows)
    if profile_df.empty or scenario_df.empty:
        raise ValueError("No valid profile/scenario KPI estimates were produced.")
    coverage_df = _coverage_for_variants(profile_df, scenario_df, quantiles)
    coverage_summary = _coverage_summary_for_variants(coverage_df)
    distribution_summary = _distribution_summary(profile_df, scenario_df)

    output_dir = args.output_dir or Path(
        "output/neural_building_emulator/flexibility_pre_power_coverage_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "flexibility_pre_power_coverage_profile_kpis.csv"
    scenario_path = output_dir / "flexibility_pre_power_coverage_scenario_kpis.csv"
    coverage_path = output_dir / "flexibility_pre_power_coverage_curves.csv"
    coverage_summary_path = output_dir / "flexibility_pre_power_coverage_summary.csv"
    distribution_summary_path = output_dir / "flexibility_pre_power_distribution_summary.csv"
    html_path = output_dir / "flexibility_pre_power_coverage_ablation.html"
    report_path = output_dir / "flexibility_pre_power_coverage_ablation_report.md"
    profile_df.to_csv(profile_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    coverage_summary.to_csv(coverage_summary_path, index=False)
    distribution_summary.to_csv(distribution_summary_path, index=False)
    _write_coverage_html(
        html_path,
        coverage_df,
        title=(
            "Flexibility KPI coverage under P_pre regression definitions"
            f"<br><sup>same {args.num_scenarios} yearly scenarios per building; "
            f"profiles={len(profiles)}, HP mode={args.prob_hp_scenario_mode}</sup>"
        ),
    )
    _write_report(
        report_path,
        coverage_summary=coverage_summary,
        distribution_summary=distribution_summary,
        profiles=len(profiles),
        scenarios=args.num_scenarios,
    )
    print(f"saved_profile_kpis={profile_path}")
    print(f"saved_scenario_kpis={scenario_path}")
    print(f"saved_coverage={coverage_path}")
    print(f"saved_coverage_summary={coverage_summary_path}")
    print(f"saved_distribution_summary={distribution_summary_path}")
    print(f"saved_html={html_path}")
    print(f"saved_report={report_path}")
    for variant in VARIANTS:
        part = coverage_summary[coverage_summary["variant"] == variant]
        distribution_part = distribution_summary[
            distribution_summary["variant"] == variant
        ]
        print(
            f"variant={variant} "
            f"coverage_mace={part['mean_abs_calibration_error'].mean():.4f} "
            f"mean_pit={part['mean_pit'].mean():.4f} "
            f"central_90_coverage={distribution_part['central_90_coverage'].mean():.4f} "
            f"mean_crps={distribution_part['mean_crps_wh_m2_k'].mean():.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--profile-source",
        choices=("test", "train", "selected"),
        default="test",
    )
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--horizons-hours", default="0.5,1,2,3")
    parser.add_argument("--dt-hours", type=float, default=None)
    parser.add_argument("--min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument(
        "--coverage-quantiles",
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
    )
    parser.add_argument(
        "--prob-hp-scenario-mode",
        choices=("expected", "bernoulli"),
        default="bernoulli",
    )
    return parser.parse_args()


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
