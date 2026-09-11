"""Compare direct annual flexibility ratios with directional GAM estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kendalltau, pearsonr, spearmanr

from .flexibility_joint_gam_study import fit_joint_model


DEFAULT_EVENT_DATA = Path(
    "output/neural_building_emulator_refactor/flexibility_directional_gam_study/"
    "event_regression_data.parquet"
)
DEFAULT_DIRECTIONAL_SUMMARY = Path(
    "output/neural_building_emulator_refactor/flexibility_directional_gam_study/"
    "summary.json"
)
DEFAULT_HELD_COEFFICIENTS = Path(
    "output/neural_building_emulator_refactor/flexibility_directional_gam_study/"
    "building_flexibility_coefficients.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/flexibility_direct_kpi_comparison"
)
DIRECTIONS = ("up", "down")


def _direct_ratios(events: pd.DataFrame, horizon_hours: float) -> pd.DataFrame:
    grouped = (
        events.groupby(["profile_id", "direction"], observed=True)
        .agg(
            event_count=("delta_pel_w_m2", "size"),
            sum_delta_pel_w_m2=("delta_pel_w_m2", "sum"),
            sum_delta_tset_c=("delta_tset_c", "sum"),
        )
        .reset_index()
    )
    grouped["direct_beta_w_m2_k"] = (
        grouped["sum_delta_pel_w_m2"] / grouped["sum_delta_tset_c"]
    )
    grouped["direct_flex_wh_m2_k"] = (
        horizon_hours * grouped["direct_beta_w_m2_k"]
    )
    wide = grouped.pivot(index="profile_id", columns="direction")
    wide.columns = [f"{name}_{direction}" for name, direction in wide.columns]
    return wide.reset_index()


def _full_year_gam_coefficients(
    events: pd.DataFrame,
    *,
    horizon_hours: float,
    n_knots: int,
    smoothing: dict[str, float],
    max_iterations: int,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for direction in DIRECTIONS:
        model = fit_joint_model(
            events[events["direction"].eq(direction)].reset_index(drop=True),
            kind="gam",
            n_knots=n_knots,
            smoothing=float(smoothing[direction]),
            max_iterations=max_iterations,
        )
        coefficient = f"beta_{direction}_w_m2_k"
        table = model.building_coefficients()[["profile_id", coefficient]].rename(
            columns={coefficient: f"gam_beta_w_m2_k_{direction}"}
        )
        table[f"gam_flex_wh_m2_k_{direction}"] = (
            horizon_hours * table[f"gam_beta_w_m2_k_{direction}"]
        )
        tables.append(table)
    return tables[0].merge(tables[1], on="profile_id", how="inner")


def _association(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman_rank": float(spearmanr(x, y).statistic),
        "kendall_rank": float(kendalltau(x, y).statistic),
        "mean_absolute_difference_wh_m2_k": float(np.mean(np.abs(x - y))),
        "root_mean_square_difference_wh_m2_k": float(np.sqrt(np.mean((x - y) ** 2))),
        "mean_signed_difference_direct_minus_gam_wh_m2_k": float(np.mean(x - y)),
    }


def _rank_diagnostics(
    values: pd.DataFrame,
    direction: str,
    top_fraction: float,
) -> dict[str, float | int]:
    direct = values[f"direct_flex_wh_m2_k_{direction}"]
    gam = values[f"gam_flex_wh_m2_k_{direction}"]
    direct_rank = direct.rank(method="average", ascending=False)
    gam_rank = gam.rank(method="average", ascending=False)
    count = max(1, int(np.ceil(top_fraction * len(values))))
    direct_top = set(values.loc[direct_rank.le(count), "profile_id"].astype(int))
    gam_top = set(values.loc[gam_rank.le(count), "profile_id"].astype(int))
    return {
        "building_count": int(len(values)),
        "top_count": int(count),
        "top_set_overlap_fraction": float(len(direct_top & gam_top) / count),
        "median_absolute_rank_difference": float(np.median(np.abs(direct_rank - gam_rank))),
        "mean_absolute_rank_difference": float(np.mean(np.abs(direct_rank - gam_rank))),
    }


def _comparison_dashboard(
    values: pd.DataFrame,
    associations: dict[str, dict[str, float]],
    path: Path,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Upward annual flexibility",
            "Downward annual flexibility",
            "Upward building rank",
            "Downward building rank",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )
    colors = {"up": "#d95f02", "down": "#1b7f79"}
    for column, direction in enumerate(DIRECTIONS, start=1):
        direct_column = f"direct_flex_wh_m2_k_{direction}"
        gam_column = f"gam_flex_wh_m2_k_{direction}"
        direct = values[direct_column]
        gam = values[gam_column]
        customdata = np.column_stack(
            [
                values["profile_id"],
                values[f"event_count_{direction}"],
            ]
        )
        figure.add_trace(
            go.Scattergl(
                x=gam,
                y=direct,
                mode="markers",
                marker={"size": 7, "opacity": 0.55, "color": colors[direction]},
                customdata=customdata,
                name=direction.capitalize(),
                showlegend=False,
                hovertemplate=(
                    "building=%{customdata[0]:.0f}<br>"
                    "GAM=%{x:.3f}<br>direct=%{y:.3f}<br>"
                    "events=%{customdata[1]:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        limit_low = float(np.quantile(np.concatenate([gam, direct]), 0.005))
        limit_high = float(np.quantile(np.concatenate([gam, direct]), 0.995))
        figure.add_shape(
            type="line",
            x0=limit_low,
            y0=limit_low,
            x1=limit_high,
            y1=limit_high,
            line={"color": "#66717e", "dash": "dot"},
            row=1,
            col=column,
        )
        figure.add_annotation(
            text=(
                f"Pearson={associations[direction]['pearson']:.3f}<br>"
                f"Spearman={associations[direction]['spearman_rank']:.3f}<br>"
                f"Kendall={associations[direction]['kendall_rank']:.3f}"
            ),
            x=0.03,
            y=0.97,
            xref=f"x{'' if column == 1 else column} domain",
            yref=f"y{'' if column == 1 else column} domain",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="#b8c1cc",
        )
        direct_rank = direct.rank(method="average", ascending=False)
        gam_rank = gam.rank(method="average", ascending=False)
        figure.add_trace(
            go.Scattergl(
                x=gam_rank,
                y=direct_rank,
                mode="markers",
                marker={"size": 7, "opacity": 0.55, "color": colors[direction]},
                customdata=customdata,
                showlegend=False,
                hovertemplate=(
                    "building=%{customdata[0]:.0f}<br>"
                    "GAM rank=%{x:.0f}<br>direct rank=%{y:.0f}<extra></extra>"
                ),
            ),
            row=2,
            col=column,
        )
        figure.add_shape(
            type="line",
            x0=1,
            y0=1,
            x1=len(values),
            y1=len(values),
            line={"color": "#66717e", "dash": "dot"},
            row=2,
            col=column,
        )
        figure.update_xaxes(title_text="Directional GAM [Wh/(m2 K)]", row=1, col=column)
        figure.update_yaxes(title_text="Direct annual ratio [Wh/(m2 K)]", row=1, col=column)
        figure.update_xaxes(title_text="Directional GAM rank", row=2, col=column)
        figure.update_yaxes(title_text="Direct annual-ratio rank", row=2, col=column)
    figure.update_layout(
        title={
            "text": (
                "Direct annual event-response ratio versus directional GAM KPI"
                "<br><sup>Both use the same filtered 3 h setpoint events; rank 1 is most flexible.</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1500,
        height=1000,
        font={"size": 15},
        margin={"l": 105, "r": 45, "t": 125, "b": 85},
    )
    figure.update_annotations(font={"size": 17})
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.event_data)
    events = events[np.isclose(events["horizon_hours"], args.horizon_hours)].copy()
    if events.empty:
        raise ValueError(f"No events found for H={args.horizon_hours:g} h")
    summary = json.loads(Path(args.directional_summary).read_text())
    smoothing = {
        direction: float(summary["selected_gam_smoothing"][direction])
        for direction in DIRECTIONS
    }
    direct = _direct_ratios(events, args.horizon_hours)
    gam = _full_year_gam_coefficients(
        events,
        horizon_hours=args.horizon_hours,
        n_knots=args.gam_n_knots,
        smoothing=smoothing,
        max_iterations=args.solver_max_iterations,
    )
    values = direct.merge(gam, on="profile_id", how="inner")

    held_path = Path(args.held_coefficients)
    if held_path.exists():
        held = pd.read_csv(held_path)
        held_columns = ["profile_id"]
        for direction in DIRECTIONS:
            beta = f"gam_beta_{direction}_w_m2_k"
            held[f"held_gam_flex_wh_m2_k_{direction}"] = args.horizon_hours * held[beta]
            held_columns.append(f"held_gam_flex_wh_m2_k_{direction}")
        values = values.merge(held.loc[:, held_columns], on="profile_id", how="left")

    associations: dict[str, dict[str, float]] = {}
    rank_diagnostics: dict[str, dict[str, float | int]] = {}
    held_associations: dict[str, dict[str, float]] = {}
    for direction in DIRECTIONS:
        direct_values = values[f"direct_flex_wh_m2_k_{direction}"].to_numpy(float)
        gam_values = values[f"gam_flex_wh_m2_k_{direction}"].to_numpy(float)
        associations[direction] = _association(direct_values, gam_values)
        rank_diagnostics[direction] = _rank_diagnostics(
            values, direction, args.top_fraction
        )
        held_column = f"held_gam_flex_wh_m2_k_{direction}"
        if held_column in values:
            finite = np.isfinite(values[held_column])
            held_associations[direction] = _association(
                direct_values[finite], values.loc[finite, held_column].to_numpy(float)
            )
        values[f"direct_rank_{direction}"] = values[
            f"direct_flex_wh_m2_k_{direction}"
        ].rank(method="average", ascending=False)
        values[f"gam_rank_{direction}"] = values[
            f"gam_flex_wh_m2_k_{direction}"
        ].rank(method="average", ascending=False)
        values[f"absolute_rank_difference_{direction}"] = np.abs(
            values[f"direct_rank_{direction}"] - values[f"gam_rank_{direction}"]
        )

    values.to_csv(output_dir / "building_direct_vs_gam_flexibility.csv", index=False)
    dashboard_path = output_dir / "direct_vs_gam_flexibility.html"
    _comparison_dashboard(values, associations, dashboard_path)
    result: dict[str, object] = {
        "event_data": str(args.event_data),
        "horizon_hours": args.horizon_hours,
        "building_count": int(len(values)),
        "event_count": int(len(events)),
        "direct_definition": (
            "H * sum(delta mean Pel) / sum(delta Tset), separately by direction"
        ),
        "gam_definition": (
            "H * beta_direction from separate directional GAMs refitted on all events"
        ),
        "selected_gam_smoothing": smoothing,
        "association_with_full_year_gam": associations,
        "rank_diagnostics": rank_diagnostics,
        "association_with_previous_holdout_fit_gam": held_associations,
        "outputs": {
            "dashboard": str(dashboard_path),
            "building_values": str(
                output_dir / "building_direct_vs_gam_flexibility.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=True)
    )
    print(json.dumps(result, indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct annual flexibility ratios with directional GAM KPIs."
    )
    parser.add_argument("--event-data", type=Path, default=DEFAULT_EVENT_DATA)
    parser.add_argument("--directional-summary", type=Path, default=DEFAULT_DIRECTIONAL_SUMMARY)
    parser.add_argument("--held-coefficients", type=Path, default=DEFAULT_HELD_COEFFICIENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-hours", type=float, default=3.0)
    parser.add_argument("--gam-n-knots", type=int, default=7)
    parser.add_argument("--solver-max-iterations", type=int, default=3000)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    if not 0.0 < args.top_fraction <= 1.0:
        parser.error("--top-fraction must be in (0, 1]")
    return args


def main() -> None:
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
