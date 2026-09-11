"""Compare separate upward/downward linear and GAM flexibility regressions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .flexibility_joint_gam_study import (
    DEFAULT_DATASET,
    FEATURE_LABELS,
    NUISANCE_FEATURES,
    SMOOTH_FEATURES,
    JointFlexibilityModel,
    _assign_time_block_split,
    _binned_residual,
    _load_events,
    _per_building_metrics,
    fit_joint_model,
    predict,
    prediction_metrics,
)


DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/flexibility_directional_gam_study"
)
DIRECTIONS = ("up", "down")


@dataclass
class DirectionalModels:
    kind: str
    models: dict[str, JointFlexibilityModel]


def _validate_event_data(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "profile_id",
        "event_time",
        "direction",
        "delta_tset_c",
        "delta_pel_w_m2",
        *NUISANCE_FEATURES,
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"Event data is missing columns: {missing}")
    values = events[events["direction"].isin(DIRECTIONS)].copy()
    finite_columns = ["delta_tset_c", "delta_pel_w_m2", *NUISANCE_FEATURES]
    finite = np.all(np.isfinite(values.loc[:, finite_columns]), axis=1)
    return values.loc[finite].reset_index(drop=True)


def _fit_directional(
    frame: pd.DataFrame,
    *,
    kind: str,
    n_knots: int,
    smoothing: dict[str, float],
    max_iterations: int,
) -> DirectionalModels:
    models: dict[str, JointFlexibilityModel] = {}
    for direction in DIRECTIONS:
        directional = frame[frame["direction"].eq(direction)].reset_index(drop=True)
        if directional.empty:
            raise ValueError(f"No {direction!r} events available for fitting")
        models[direction] = fit_joint_model(
            directional,
            kind=kind,
            n_knots=n_knots,
            smoothing=smoothing.get(direction, 0.0),
            max_iterations=max_iterations,
        )
    return DirectionalModels(kind=kind, models=models)


def _predict_directional(models: DirectionalModels, frame: pd.DataFrame) -> np.ndarray:
    result = np.full(len(frame), np.nan, dtype=np.float64)
    for direction in DIRECTIONS:
        mask = frame["direction"].eq(direction).to_numpy()
        if np.any(mask):
            result[mask] = predict(models.models[direction], frame.loc[mask])
    if not np.all(np.isfinite(result)):
        raise ValueError("Directional prediction left non-finite event predictions")
    return result


def _directional_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, dict[str, float]]:
    result = {"combined": prediction_metrics(frame, prediction)}
    for direction in DIRECTIONS:
        mask = frame["direction"].eq(direction).to_numpy()
        result[direction] = prediction_metrics(frame.loc[mask], prediction[mask])
    return result


def _fit_predict_per_building_linear(
    fit_frame: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit the current per-building KPI regression with the matched z vector."""
    prediction = np.full(len(test), np.nan, dtype=np.float64)
    coefficient_rows: list[dict[str, float | int]] = []
    for profile_id, test_building in test.groupby("profile_id", sort=False):
        fit_building = fit_frame[fit_frame["profile_id"].eq(profile_id)]
        if fit_building.empty:
            raise ValueError(f"No fitting events for profile {profile_id}")
        fit_nuisance = fit_building.loc[:, NUISANCE_FEATURES].to_numpy(float)
        nuisance_mean = fit_nuisance.mean(axis=0)
        nuisance_scale = fit_nuisance.std(axis=0)
        nuisance_scale[nuisance_scale < 1e-8] = 1.0

        fit_delta = fit_building["delta_tset_c"].to_numpy(float)
        fit_design = np.column_stack(
            [
                np.ones(len(fit_building)),
                np.maximum(fit_delta, 0.0),
                np.minimum(fit_delta, 0.0),
                (fit_nuisance - nuisance_mean) / nuisance_scale,
            ]
        )
        coefficients, *_ = np.linalg.lstsq(
            fit_design,
            fit_building["delta_pel_w_m2"].to_numpy(float),
            rcond=None,
        )

        test_delta = test_building["delta_tset_c"].to_numpy(float)
        test_nuisance = test_building.loc[:, NUISANCE_FEATURES].to_numpy(float)
        test_design = np.column_stack(
            [
                np.ones(len(test_building)),
                np.maximum(test_delta, 0.0),
                np.minimum(test_delta, 0.0),
                (test_nuisance - nuisance_mean) / nuisance_scale,
            ]
        )
        prediction[test_building.index.to_numpy()] = test_design @ coefficients
        coefficient_rows.append(
            {
                "profile_id": int(profile_id),
                "local_linear_intercept_w_m2": float(coefficients[0]),
                "local_linear_beta_up_w_m2_k": float(coefficients[1]),
                "local_linear_beta_down_w_m2_k": float(coefficients[2]),
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Per-building linear model left non-finite predictions")
    return prediction, pd.DataFrame(coefficient_rows)


def _coefficient_table(
    linear: DirectionalModels,
    gam: DirectionalModels,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for direction in DIRECTIONS:
        beta_column = f"beta_{direction}_w_m2_k"
        linear_values = linear.models[direction].building_coefficients()[
            ["profile_id", "intercept_w_m2", beta_column]
        ].rename(
            columns={
                "intercept_w_m2": f"linear_intercept_{direction}_w_m2",
                beta_column: f"linear_beta_{direction}_w_m2_k",
            }
        )
        gam_values = gam.models[direction].building_coefficients()[
            ["profile_id", "intercept_w_m2", beta_column]
        ].rename(
            columns={
                "intercept_w_m2": f"gam_intercept_{direction}_w_m2",
                beta_column: f"gam_beta_{direction}_w_m2_k",
            }
        )
        tables.append(linear_values.merge(gam_values, on="profile_id", how="inner"))
    return tables[0].merge(tables[1], on="profile_id", how="inner")


def _comparison_dashboard(
    test: pd.DataFrame,
    linear_prediction: np.ndarray,
    gam_prediction: np.ndarray,
    per_building: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
    seed: int,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Observed versus predicted",
            "Upward residual over pre-event Tout",
            "Downward residual over pre-event Tout",
            "Per-building held-out RMSE",
            "Upward beta",
            "Downward beta",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )
    target = test["delta_pel_w_m2"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(test), size=min(25000, len(test)), replace=False)
    candidates = (
        ("Directional linear", linear_prediction, "#2878b5"),
        ("Directional GAM", gam_prediction, "#d84a3a"),
    )
    for name, prediction_values, color in candidates:
        figure.add_trace(
            go.Scattergl(
                x=target[sample],
                y=prediction_values[sample],
                mode="markers",
                marker={"size": 4, "opacity": 0.18, "color": color},
                name=name,
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        residual = prediction_values - target
        for column, direction in ((2, "up"), (3, "down")):
            mask = test["direction"].eq(direction).to_numpy()
            summary = _binned_residual(
                test.loc[mask, "pre_outdoor_temperature_c"].to_numpy(float),
                residual[mask],
            )
            figure.add_trace(
                go.Scatter(
                    x=summary["value"],
                    y=summary["residual"],
                    mode="lines+markers",
                    line={"width": 3, "color": color},
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
    limits = float(
        np.quantile(
            np.abs(np.concatenate([target, linear_prediction, gam_prediction])), 0.995
        )
    )
    figure.add_shape(
        type="line",
        x0=-limits,
        y0=-limits,
        x1=limits,
        y1=limits,
        line={"color": "#5d6570", "dash": "dot"},
        row=1,
        col=1,
    )
    figure.add_hline(y=0.0, line_dash="dot", line_color="#5d6570", row=1, col=2)
    figure.add_hline(y=0.0, line_dash="dot", line_color="#5d6570", row=1, col=3)
    figure.add_trace(
        go.Scattergl(
            x=per_building["linear_rmse_w_m2"],
            y=per_building["gam_rmse_w_m2"],
            mode="markers",
            marker={"size": 6, "opacity": 0.55, "color": "#2a9d8f"},
            text=per_building["profile_id"],
            showlegend=False,
            hovertemplate=(
                "building=%{text}<br>linear=%{x:.3f}<br>"
                "GAM=%{y:.3f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    rmse_limit = float(
        np.quantile(
            np.concatenate(
                [per_building["linear_rmse_w_m2"], per_building["gam_rmse_w_m2"]]
            ),
            0.995,
        )
    )
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=rmse_limit,
        y1=rmse_limit,
        line={"color": "#5d6570", "dash": "dot"},
        row=2,
        col=1,
    )
    for column, direction in ((2, "up"), (3, "down")):
        x = coefficients[f"linear_beta_{direction}_w_m2_k"]
        y = coefficients[f"gam_beta_{direction}_w_m2_k"]
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker={"size": 6, "opacity": 0.55, "color": "#e76f51"},
                text=coefficients["profile_id"],
                showlegend=False,
                hovertemplate=(
                    "building=%{text}<br>linear=%{x:.3f}<br>"
                    "GAM=%{y:.3f}<extra></extra>"
                ),
            ),
            row=2,
            col=column,
        )
        coefficient_limit = float(
            np.quantile(np.abs(np.concatenate([x.to_numpy(), y.to_numpy()])), 0.995)
        )
        figure.add_shape(
            type="line",
            x0=-coefficient_limit,
            y0=-coefficient_limit,
            x1=coefficient_limit,
            y1=coefficient_limit,
            line={"color": "#5d6570", "dash": "dot"},
            row=2,
            col=column,
        )
    figure.update_xaxes(title_text="Observed delta Pel [W/m2]", row=1, col=1)
    figure.update_yaxes(title_text="Predicted delta Pel [W/m2]", row=1, col=1)
    for column in (2, 3):
        figure.update_xaxes(title_text="Pre-event outdoor temperature [C]", row=1, col=column)
        figure.update_yaxes(title_text="Mean residual [W/m2]", row=1, col=column)
    figure.update_xaxes(title_text="Linear RMSE [W/m2]", row=2, col=1)
    figure.update_yaxes(title_text="GAM RMSE [W/m2]", row=2, col=1)
    for column in (2, 3):
        figure.update_xaxes(title_text="Linear beta [W/(m2 K)]", row=2, col=column)
        figure.update_yaxes(title_text="GAM beta [W/(m2 K)]", row=2, col=column)
    figure.update_layout(
        title={
            "text": "Separate upward/downward regressions: linear versus GAM nuisance",
            "x": 0.5,
        },
        template="plotly_white",
        width=1750,
        height=1000,
        font={"size": 14},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.03},
        margin={"l": 90, "r": 50, "t": 120, "b": 80},
    )
    figure.update_annotations(font={"size": 18})
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def _partial_effect_dashboard(
    linear: DirectionalModels,
    gam: DirectionalModels,
    fit_frame: pd.DataFrame,
    path: Path,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=4,
        row_titles=("Upward", "Downward"),
        subplot_titles=tuple(FEATURE_LABELS[name] for name in SMOOTH_FEATURES) * 2,
        horizontal_spacing=0.06,
        vertical_spacing=0.17,
    )
    colors = {"Directional linear": "#2878b5", "Directional GAM": "#d84a3a"}
    for row, direction in enumerate(DIRECTIONS, start=1):
        directional = fit_frame[fit_frame["direction"].eq(direction)]
        reference = {
            feature: float(directional[feature].median())
            for feature in NUISANCE_FEATURES
        }
        for column, feature in enumerate(SMOOTH_FEATURES, start=1):
            grid = np.linspace(
                directional[feature].quantile(0.01),
                directional[feature].quantile(0.99),
                150,
            )
            synthetic = pd.DataFrame(
                {name: np.full(len(grid), value) for name, value in reference.items()}
            )
            synthetic[feature] = grid
            midpoint_frame = pd.DataFrame(
                {name: [value] for name, value in reference.items()}
            )
            for collection, name in (
                (linear, "Directional linear"),
                (gam, "Directional GAM"),
            ):
                model = collection.models[direction]
                effect = (
                    model.transformer.transform(synthetic)
                    @ model.nuisance_coefficients
                )
                midpoint = (
                    model.transformer.transform(midpoint_frame)
                    @ model.nuisance_coefficients
                )
                figure.add_trace(
                    go.Scatter(
                        x=grid,
                        y=effect - float(midpoint[0]),
                        mode="lines",
                        line={"width": 3, "color": colors[name]},
                        name=name,
                        legendgroup=name,
                        showlegend=row == 1 and column == 1,
                    ),
                    row=row,
                    col=column,
                )
            figure.update_xaxes(title_text=FEATURE_LABELS[feature], row=row, col=column)
            if column == 1:
                figure.update_yaxes(
                    title_text="Centered nuisance effect [W/m2]", row=row, col=column
                )
    figure.update_layout(
        title={"text": "Direction-specific exogenous nuisance functions", "x": 0.5},
        template="plotly_white",
        width=1900,
        height=950,
        font={"size": 14},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.04},
        margin={"l": 115, "r": 50, "t": 120, "b": 90},
    )
    figure.update_annotations(font={"size": 17})
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def _coefficient_stability(coefficients: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for direction in DIRECTIONS:
        linear = coefficients[f"linear_beta_{direction}_w_m2_k"].to_numpy(float)
        gam = coefficients[f"gam_beta_{direction}_w_m2_k"].to_numpy(float)
        result[direction] = {
            "linear_gam_correlation": float(np.corrcoef(linear, gam)[0, 1]),
            "mean_absolute_change_w_m2_k": float(np.mean(np.abs(gam - linear))),
            "linear_median_w_m2_k": float(np.median(linear)),
            "gam_median_w_m2_k": float(np.median(gam)),
        }
    return result


def run_study(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.event_data is None:
        events, selection = _load_events(args)
        events = _assign_time_block_split(
            events, block_days=args.holdout_block_days, seed=args.seed
        )
    else:
        events = _validate_event_data(pd.read_parquet(args.event_data))
        if "split" not in events.columns:
            events = _assign_time_block_split(
                events, block_days=args.holdout_block_days, seed=args.seed
            )
        selection = {
            "selected_profiles": int(events["profile_id"].nunique()),
            "source": "precomputed_event_data",
        }
    events.to_parquet(output_dir / "event_regression_data.parquet", index=False)
    train = events[events["split"].eq("train")].reset_index(drop=True)
    validation = events[events["split"].eq("validation")].reset_index(drop=True)
    test = events[events["split"].eq("test")].reset_index(drop=True)
    print(
        f"events train={len(train)} validation={len(validation)} test={len(test)} "
        f"buildings={events['profile_id'].nunique()}",
        flush=True,
    )

    validation_rows: list[dict[str, float | str]] = []
    best_smoothing: dict[str, float] = {}
    for direction in DIRECTIONS:
        direction_train = train[train["direction"].eq(direction)]
        direction_validation = validation[validation["direction"].eq(direction)]
        for smoothing in args.gam_smoothing_grid:
            model = fit_joint_model(
                direction_train,
                kind="gam",
                n_knots=args.gam_n_knots,
                smoothing=smoothing,
                max_iterations=args.solver_max_iterations,
            )
            metrics = prediction_metrics(
                direction_validation, predict(model, direction_validation)
            )
            validation_rows.append(
                {"direction": direction, "smoothing": smoothing, **metrics}
            )
            print(
                f"direction={direction} gam_smoothing={smoothing:g} "
                "validation_building_balanced_rmse="
                f"{metrics['building_balanced_rmse_w_m2']:.6f}",
                flush=True,
            )
        direction_rows = [row for row in validation_rows if row["direction"] == direction]
        best = min(direction_rows, key=lambda row: row["building_balanced_rmse_w_m2"])
        best_smoothing[direction] = float(best["smoothing"])
    pd.DataFrame(validation_rows).to_csv(
        output_dir / "gam_smoothing_validation.csv", index=False
    )

    fit_frame = pd.concat([train, validation], ignore_index=True)
    linear = _fit_directional(
        fit_frame,
        kind="linear",
        n_knots=args.gam_n_knots,
        smoothing={direction: 0.0 for direction in DIRECTIONS},
        max_iterations=args.solver_max_iterations,
    )
    gam = _fit_directional(
        fit_frame,
        kind="gam",
        n_knots=args.gam_n_knots,
        smoothing=best_smoothing,
        max_iterations=args.solver_max_iterations,
    )
    linear_prediction = _predict_directional(linear, test)
    gam_prediction = _predict_directional(gam, test)
    local_linear_prediction, local_linear_coefficients = (
        _fit_predict_per_building_linear(fit_frame, test)
    )
    linear_metrics = _directional_metrics(test, linear_prediction)
    gam_metrics = _directional_metrics(test, gam_prediction)
    local_linear_metrics = _directional_metrics(test, local_linear_prediction)

    per_building = _per_building_metrics(test, linear_prediction, "linear").merge(
        _per_building_metrics(test, gam_prediction, "gam"),
        on=["profile_id", "event_count"],
        how="inner",
    )
    per_building = per_building.merge(
        _per_building_metrics(test, local_linear_prediction, "local_linear"),
        on=["profile_id", "event_count"],
        how="inner",
    )
    per_building["rmse_improvement_w_m2"] = (
        per_building["linear_rmse_w_m2"] - per_building["gam_rmse_w_m2"]
    )
    per_building.to_csv(output_dir / "per_building_test_metrics.csv", index=False)
    coefficients = _coefficient_table(linear, gam).merge(
        local_linear_coefficients, on="profile_id", how="inner"
    )
    coefficients.to_csv(output_dir / "building_flexibility_coefficients.csv", index=False)
    comparison_table = pd.DataFrame(
        [
            {"model": "directional_linear", **linear_metrics["combined"]},
            {"model": "directional_gam", **gam_metrics["combined"]},
            {
                "model": "per_building_linear_same_z",
                **local_linear_metrics["combined"],
            },
        ]
    )
    comparison_table.to_csv(output_dir / "model_comparison.csv", index=False)
    _comparison_dashboard(
        test,
        linear_prediction,
        gam_prediction,
        per_building,
        coefficients,
        output_dir / "directional_linear_vs_gam_dashboard.html",
        args.seed,
    )
    _partial_effect_dashboard(
        linear,
        gam,
        fit_frame,
        output_dir / "directional_nuisance_partial_effects.html",
    )
    linear_combined = linear_metrics["combined"]
    gam_combined = gam_metrics["combined"]
    summary: dict[str, object] = {
        "dataset": str(args.dataset),
        "event_data": str(args.event_data) if args.event_data is not None else None,
        "horizon_hours": args.horizon_hours,
        "specification": {
            "up": "alpha_i_up + beta_i_up * delta_tset + f_up(z)",
            "down": "alpha_i_down + beta_i_down * delta_tset + f_down(z)",
            "smooth_in_gam": list(SMOOTH_FEATURES),
            "linear_in_gam": list(NUISANCE_FEATURES[len(SMOOTH_FEATURES) :]),
            "all_linear_baseline": list(NUISANCE_FEATURES),
        },
        "selection": selection,
        "event_counts": events["split"].value_counts().to_dict(),
        "selected_gam_smoothing": best_smoothing,
        "linear_test": linear_metrics,
        "gam_test": gam_metrics,
        "per_building_linear_same_z_test": local_linear_metrics,
        "combined_relative_improvement": {
            name: (
                (linear_combined[name] - gam_combined[name]) / linear_combined[name]
                if linear_combined[name] != 0.0
                else np.nan
            )
            for name in (
                "rmse_w_m2",
                "mae_w_m2",
                "building_balanced_rmse_w_m2",
                "mean_building_rmse_w_m2",
            )
        },
        "fraction_buildings_with_lower_gam_rmse": float(
            np.mean(per_building["rmse_improvement_w_m2"] > 0.0)
        ),
        "coefficient_stability": _coefficient_stability(coefficients),
        "solver": {
            kind: {
                direction: {
                    "iterations": collection.models[direction].solver_iterations,
                    "condition": collection.models[direction].solver_condition,
                }
                for direction in DIRECTIONS
            }
            for kind, collection in (("linear", linear), ("gam", gam))
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    print(
        json.dumps(
            {"linear_test": linear_metrics, "gam_test": gam_metrics}, indent=2
        )
    )
    print(
        f"saved_dashboard={output_dir / 'directional_linear_vs_gam_dashboard.html'}"
    )
    print(f"saved_summary={output_dir / 'summary.json'}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare separate upward/downward linear and GAM flexibility models."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--event-data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--profile-selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--read-batch-size", type=int, default=8)
    parser.add_argument("--horizon-hours", type=float, default=3.0)
    parser.add_argument("--min-setpoint-change-c", type=float, default=0.05)
    parser.add_argument("--holdout-block-days", type=int, default=14)
    parser.add_argument("--gam-n-knots", type=int, default=7)
    parser.add_argument(
        "--gam-smoothing-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--solver-max-iterations", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.holdout_block_days < 1:
        parser.error("--holdout-block-days must be positive")
    if args.gam_n_knots < 4:
        parser.error("--gam-n-knots must be at least 4")
    return args


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
