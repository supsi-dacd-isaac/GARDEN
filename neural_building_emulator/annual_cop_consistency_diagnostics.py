"""Annual Qroom/Pel consistency checks for closed-loop HP training signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    HEAT_PUMP_ELECTRIC_POWER_COLUMN,
    HP_MODE_IS_DHW_COLUMN,
    PROFILE_ID_COLUMN,
    ZONE_THERMAL_HEATING_POWER_COLUMN,
)
from .data import DEFAULT_DATASET_PATH, select_profile_ids

FLOOR_AREA_COLUMN = "floor_area"
TOTAL_FLOORS_COLUMN = "totalFloors"
HP_REF_CAPACITY_COLUMN = "hp_ref_capacity_W"
HP_THERMAL_POWER_COLUMN = "heat_pump_thermal_power"
DEFAULT_DT_HOURS = 0.25


def _parquet_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        return files
    return [path]


def _available_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(_parquet_files(path)[0]).names)
    except Exception:
        return set(pd.read_parquet(_parquet_files(path)[0]).columns)


def _read_dataset(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    read_path: Path | list[Path] = _parquet_files(path)
    if len(read_path) == 1:
        read_path = read_path[0]
    return pd.read_parquet(read_path, columns=list(columns))


def load_hp_signal_frame(dataset_path: Path) -> pd.DataFrame:
    columns = [
        PROFILE_ID_COLUMN,
        FLOOR_AREA_COLUMN,
        TOTAL_FLOORS_COLUMN,
        HP_REF_CAPACITY_COLUMN,
        ZONE_THERMAL_HEATING_POWER_COLUMN,
        HEAT_PUMP_ELECTRIC_POWER_COLUMN,
        HP_MODE_IS_DHW_COLUMN,
    ]
    available = _available_columns(dataset_path)
    missing = sorted(set(columns).difference(available))
    if missing:
        lines = "\n".join(f"- {column}" for column in missing)
        raise ValueError(f"Missing required columns for annual COP diagnostics:\n{lines}")
    if HP_THERMAL_POWER_COLUMN in available:
        columns.append(HP_THERMAL_POWER_COLUMN)

    df = _read_dataset(dataset_path, columns)
    numeric_columns = [
        FLOOR_AREA_COLUMN,
        TOTAL_FLOORS_COLUMN,
        HP_REF_CAPACITY_COLUMN,
        ZONE_THERMAL_HEATING_POWER_COLUMN,
        HEAT_PUMP_ELECTRIC_POWER_COLUMN,
        HP_MODE_IS_DHW_COLUMN,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if HP_THERMAL_POWER_COLUMN in df.columns:
        df[HP_THERMAL_POWER_COLUMN] = pd.to_numeric(
            df[HP_THERMAL_POWER_COLUMN],
            errors="coerce",
        ).fillna(0.0)
    else:
        df[HP_THERMAL_POWER_COLUMN] = np.nan

    df = df[df[HP_REF_CAPACITY_COLUMN].fillna(0.0) > 0.0].copy()
    df = df[df[FLOOR_AREA_COLUMN].fillna(0.0) > 0.0].copy()
    df = df[df[TOTAL_FLOORS_COLUMN].fillna(0.0) > 0.0].copy()
    df[PROFILE_ID_COLUMN] = df[PROFILE_ID_COLUMN].astype(int)
    return df


def add_power_density_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zone_floor_area = df[FLOOR_AREA_COLUMN].astype(float)
    building_heated_area = zone_floor_area * df[TOTAL_FLOORS_COLUMN].astype(float)
    is_dhw = df[HP_MODE_IS_DHW_COLUMN].fillna(0.0) > 0.5

    df["qroom_w_m2"] = (
        df[ZONE_THERMAL_HEATING_POWER_COLUMN].clip(lower=0.0) / zone_floor_area
    )
    df["pel_raw_w_m2"] = (
        df[HEAT_PUMP_ELECTRIC_POWER_COLUMN].clip(lower=0.0) / building_heated_area
    )
    df["pel_sh_w_m2"] = df["pel_raw_w_m2"].where(~is_dhw, 0.0)
    df["hp_thermal_raw_w_m2"] = (
        df[HP_THERMAL_POWER_COLUMN].clip(lower=0.0) / building_heated_area
    )
    df["hp_thermal_sh_w_m2"] = df["hp_thermal_raw_w_m2"].where(~is_dhw, 0.0)
    return df


def summarize_annual_consistency(df: pd.DataFrame, *, dt_hours: float) -> pd.DataFrame:
    records = []
    for profile_id, group in df.groupby(PROFILE_ID_COLUMN, sort=True):
        qroom = group["qroom_w_m2"].to_numpy(dtype=float)
        pel_sh = group["pel_sh_w_m2"].to_numpy(dtype=float)
        pel_raw = group["pel_raw_w_m2"].to_numpy(dtype=float)
        hp_thermal_sh = group["hp_thermal_sh_w_m2"].to_numpy(dtype=float)

        qroom_kwh_m2 = float(np.nansum(qroom) * dt_hours / 1000.0)
        pel_sh_kwh_m2 = float(np.nansum(pel_sh) * dt_hours / 1000.0)
        pel_raw_kwh_m2 = float(np.nansum(pel_raw) * dt_hours / 1000.0)
        hp_thermal_sh_kwh_m2 = float(np.nansum(hp_thermal_sh) * dt_hours / 1000.0)

        qroom_active = qroom > 50.0
        if np.nanstd(qroom) > 0.0 and np.nanstd(pel_sh) > 0.0:
            corr = float(np.corrcoef(qroom, pel_sh)[0, 1])
        else:
            corr = float("nan")

        records.append(
            {
                PROFILE_ID_COLUMN: int(profile_id),
                "n_steps": int(len(group)),
                "floor_area_m2": float(group[FLOOR_AREA_COLUMN].iloc[0]),
                "total_floors": float(group[TOTAL_FLOORS_COLUMN].iloc[0]),
                "building_heated_area_m2": float(
                    group[FLOOR_AREA_COLUMN].iloc[0]
                    * group[TOTAL_FLOORS_COLUMN].iloc[0]
                ),
                "hp_ref_capacity_W": float(group[HP_REF_CAPACITY_COLUMN].iloc[0]),
                "annual_qroom_kwh_m2": qroom_kwh_m2,
                "annual_pel_sh_kwh_m2": pel_sh_kwh_m2,
                "annual_pel_raw_kwh_m2": pel_raw_kwh_m2,
                "annual_hp_thermal_sh_kwh_m2": hp_thermal_sh_kwh_m2,
                "required_cop_qroom_over_pel_sh": qroom_kwh_m2 / pel_sh_kwh_m2
                if pel_sh_kwh_m2 > 1e-9
                else float("nan"),
                "required_cop_qroom_over_pel_raw": qroom_kwh_m2 / pel_raw_kwh_m2
                if pel_raw_kwh_m2 > 1e-9
                else float("nan"),
                "pel_removed_by_dhw_mask_frac": 1.0 - pel_sh_kwh_m2 / pel_raw_kwh_m2
                if pel_raw_kwh_m2 > 1e-9
                else float("nan"),
                "observed_cop_hpthermal_over_pel_sh": hp_thermal_sh_kwh_m2 / pel_sh_kwh_m2
                if pel_sh_kwh_m2 > 1e-9
                else float("nan"),
                "qroom_over_hpthermal": qroom_kwh_m2 / hp_thermal_sh_kwh_m2
                if hp_thermal_sh_kwh_m2 > 1e-9
                else float("nan"),
                "peak_qroom_w_m2": float(np.nanmax(qroom)) if len(qroom) else float("nan"),
                "peak_pel_sh_w_m2": float(np.nanmax(pel_sh)) if len(pel_sh) else float("nan"),
                "qroom_pel_corr": corr,
                "frac_q_gt50_with_zero_pel": float(np.mean(pel_sh[qroom_active] <= 1e-6))
                if np.any(qroom_active)
                else float("nan"),
            }
        )
    return pd.DataFrame(records)


def select_summary_profiles(
    summary: pd.DataFrame,
    *,
    max_profiles: int | None,
    seed: int,
    strategy: str,
) -> pd.DataFrame:
    ids = select_profile_ids(
        summary[PROFILE_ID_COLUMN].astype(int).to_list(),
        max_profiles=max_profiles,
        seed=seed,
        strategy=strategy,
    )
    return summary[summary[PROFILE_ID_COLUMN].isin(ids)].copy()


def aggregate_stats(summary: pd.DataFrame) -> dict[str, float | int]:
    required = summary["required_cop_qroom_over_pel_sh"].replace([np.inf, -np.inf], np.nan).dropna()
    required_raw = (
        summary["required_cop_qroom_over_pel_raw"].replace([np.inf, -np.inf], np.nan).dropna()
    )
    qroom_sum = float(summary["annual_qroom_kwh_m2"].sum())
    pel_sum = float(summary["annual_pel_sh_kwh_m2"].sum())
    pel_raw_sum = float(summary["annual_pel_raw_kwh_m2"].sum())
    hp_thermal_sum = float(summary["annual_hp_thermal_sh_kwh_m2"].sum())
    return {
        "profiles": int(len(summary)),
        "finite_required_cop_profiles": int(len(required)),
        "required_cop_mean": float(required.mean()),
        "required_cop_energy_weighted": qroom_sum / pel_sum if pel_sum > 0.0 else float("nan"),
        "required_cop_median": float(required.median()),
        "required_cop_p90": float(required.quantile(0.90)),
        "required_cop_p95": float(required.quantile(0.95)),
        "required_cop_p99": float(required.quantile(0.99)),
        "required_cop_max": float(required.max()),
        "required_cop_raw_mean": float(required_raw.mean()),
        "required_cop_raw_energy_weighted": qroom_sum / pel_raw_sum
        if pel_raw_sum > 0.0
        else float("nan"),
        "required_cop_raw_median": float(required_raw.median()),
        "required_cop_raw_p95": float(required_raw.quantile(0.95)),
        "required_cop_raw_p99": float(required_raw.quantile(0.99)),
        "required_cop_raw_max": float(required_raw.max()),
        "profiles_required_cop_gt_5": int((required > 5.0).sum()),
        "profiles_required_cop_gt_8": int((required > 8.0).sum()),
        "profiles_required_cop_gt_10": int((required > 10.0).sum()),
        "profiles_required_cop_gt_20": int((required > 20.0).sum()),
        "profiles_required_cop_raw_gt_10": int((required_raw > 10.0).sum()),
        "profiles_required_cop_raw_gt_20": int((required_raw > 20.0).sum()),
        "annual_qroom_sum_kwh_m2": qroom_sum,
        "annual_pel_sh_sum_kwh_m2": pel_sum,
        "annual_pel_raw_sum_kwh_m2": pel_raw_sum,
        "observed_hp_cop_energy_weighted": hp_thermal_sum / pel_sum if pel_sum > 0.0 else float("nan"),
        "qroom_over_hpthermal_energy_weighted": qroom_sum / hp_thermal_sum
        if hp_thermal_sum > 0.0
        else float("nan"),
        "mean_pel_removed_by_dhw_mask_frac": float(summary["pel_removed_by_dhw_mask_frac"].mean()),
        "mean_frac_q_gt50_with_zero_pel": float(summary["frac_q_gt50_with_zero_pel"].mean()),
    }


def write_html(summary: pd.DataFrame, path: Path, *, title: str) -> None:
    required_cop = summary["required_cop_qroom_over_pel_sh"].to_numpy(dtype=float)
    marker_color = np.clip(required_cop, 0.0, 10.0)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Annual Qroom vs Pel_SH",
            "Required COP distribution",
            "Required COP vs observed HP COP",
            "High-Qroom periods with zero Pel_SH",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )

    fig.add_trace(
        go.Scatter(
            x=summary["annual_pel_sh_kwh_m2"],
            y=summary["annual_qroom_kwh_m2"],
            mode="markers",
            marker={
                "color": marker_color,
                "colorscale": "Viridis",
                "size": 8,
                "colorbar": {"title": "required COP", "x": 0.46, "y": 0.78, "len": 0.36},
            },
            text=summary[PROFILE_ID_COLUMN].astype(str),
            customdata=np.column_stack(
                [
                    summary["required_cop_qroom_over_pel_sh"],
                    summary["observed_cop_hpthermal_over_pel_sh"],
                    summary["qroom_pel_corr"],
                ]
            ),
            hovertemplate=(
                "egid=%{text}<br>"
                "Pel_SH=%{x:.2f} kWh/m2<br>"
                "Qroom=%{y:.2f} kWh/m2<br>"
                "required COP=%{customdata[0]:.2f}<br>"
                "observed HP COP=%{customdata[1]:.2f}<br>"
                "corr(Qroom,Pel_SH)=%{customdata[2]:.2f}<extra></extra>"
            ),
            name="profiles",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    max_pel = float(summary["annual_pel_sh_kwh_m2"].max())
    x_line = np.linspace(0.0, max_pel, 100)
    for cop in (1.0, 3.0, 5.0, 8.0):
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=cop * x_line,
                mode="lines",
                line={"dash": "dash", "width": 1},
                name=f"COP={cop:g}",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=summary["required_cop_qroom_over_pel_sh"],
            nbinsx=40,
            name="required COP",
            marker={"color": "#5277c3"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=summary["observed_cop_hpthermal_over_pel_sh"],
            y=summary["required_cop_qroom_over_pel_sh"],
            mode="markers",
            marker={"color": "#dd6b20", "size": 8, "opacity": 0.75},
            text=summary[PROFILE_ID_COLUMN].astype(str),
            hovertemplate=(
                "egid=%{text}<br>"
                "observed HP thermal/Pel=%{x:.2f}<br>"
                "required Qroom/Pel=%{y:.2f}<extra></extra>"
            ),
            name="COP consistency",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 10.0],
            y=[0.0, 10.0],
            mode="lines",
            line={"dash": "dot", "color": "#6b7280", "width": 1},
            name="match",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=summary["required_cop_qroom_over_pel_sh"],
            y=summary["frac_q_gt50_with_zero_pel"],
            mode="markers",
            marker={"color": "#c2410c", "size": 8, "opacity": 0.75},
            text=summary[PROFILE_ID_COLUMN].astype(str),
            customdata=np.column_stack(
                [
                    summary["peak_qroom_w_m2"],
                    summary["peak_pel_sh_w_m2"],
                ]
            ),
            hovertemplate=(
                "egid=%{text}<br>"
                "required COP=%{x:.2f}<br>"
                "frac Qroom>50 with Pel=0=%{y:.2%}<br>"
                "peak Qroom=%{customdata[0]:.1f} W/m2<br>"
                "peak Pel_SH=%{customdata[1]:.1f} W/m2<extra></extra>"
            ),
            name="zero Pel during high Q",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Annual Pel_SH [kWh/m2]", row=1, col=1)
    fig.update_yaxes(title_text="Annual Qroom [kWh/m2]", row=1, col=1)
    fig.update_xaxes(title_text="Required annual COP = Qroom / Pel_SH", row=1, col=2)
    fig.update_yaxes(title_text="profile count", row=1, col=2)
    fig.update_xaxes(title_text="Observed HP thermal / Pel_SH", row=2, col=1)
    fig.update_yaxes(title_text="Required Qroom / Pel_SH", row=2, col=1)
    fig.update_xaxes(title_text="Required annual COP", row=2, col=2)
    fig.update_yaxes(title_text="fraction", tickformat=".0%", row=2, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1450,
        height=950,
        legend={"orientation": "h", "y": 1.05},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def print_stats(label: str, stats: dict[str, float | int]) -> None:
    print(label)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}={value:.6g}")
        else:
            print(f"  {key}={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator_diagnostics"))
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--profile-selection", choices=("random", "first"), default="random")
    parser.add_argument("--dt-hours", type=float, default=DEFAULT_DT_HOURS)
    parser.add_argument(
        "--all-hp",
        action="store_true",
        help="Ignore --max-profiles and diagnose all profiles with positive hp_ref_capacity_W.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dt_hours <= 0.0:
        raise ValueError("--dt-hours must be positive")

    df = add_power_density_columns(load_hp_signal_frame(args.dataset_path))
    all_summary = summarize_annual_consistency(df, dt_hours=args.dt_hours)
    all_path = args.output_dir / "annual_cop_consistency_all_hp_profiles.csv"
    all_summary.to_csv(all_path, index=False)

    if args.all_hp:
        summary = all_summary
        suffix = "all_hp_profiles"
    else:
        if args.max_profiles < 1:
            raise ValueError("--max-profiles must be positive unless --all-hp is used")
        summary = select_summary_profiles(
            all_summary,
            max_profiles=args.max_profiles,
            seed=args.seed,
            strategy=args.profile_selection,
        )
        suffix = f"{len(summary)}_{args.profile_selection}_seed_{args.seed}"

    csv_path = args.output_dir / f"annual_cop_consistency_{suffix}.csv"
    html_path = args.output_dir / f"annual_cop_consistency_{suffix}.html"
    stats_path = args.output_dir / f"annual_cop_consistency_{suffix}_stats.csv"
    summary.to_csv(csv_path, index=False)
    pd.DataFrame([aggregate_stats(summary)]).to_csv(stats_path, index=False)
    write_html(summary, html_path, title=f"Annual Qroom/Pel consistency: {suffix}")

    print(f"saved_csv={csv_path}")
    print(f"saved_stats={stats_path}")
    print(f"saved_html={html_path}")
    print(f"saved_all_hp_csv={all_path}")
    print_stats("selected_profiles", aggregate_stats(summary))
    print_stats("all_hp_profiles", aggregate_stats(all_summary))
    top_columns = [
        PROFILE_ID_COLUMN,
        "annual_qroom_kwh_m2",
        "annual_pel_sh_kwh_m2",
        "annual_pel_raw_kwh_m2",
        "required_cop_qroom_over_pel_sh",
        "required_cop_qroom_over_pel_raw",
        "pel_removed_by_dhw_mask_frac",
        "observed_cop_hpthermal_over_pel_sh",
        "qroom_over_hpthermal",
        "peak_qroom_w_m2",
        "peak_pel_sh_w_m2",
        "frac_q_gt50_with_zero_pel",
    ]
    print("top_required_cop_profiles")
    print(
        all_summary.sort_values("required_cop_qroom_over_pel_sh", ascending=False)
        .head(10)[top_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
