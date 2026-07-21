"""Lagged diagnostics for heat-change events and indoor-temperature response."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import DATETIME_COLUMN, PROFILE_ID_COLUMN, TARGET_COLUMN
from .data import DEFAULT_DATASET_PATH

SETPOINT_COLUMN = "shSetpoint"
FLOOR_AREA_COLUMN = "floor_area"
HP_MODE_IS_DHW_COLUMN = "hp_mode_is_dhw"
HEAT_CANDIDATES = [
    ("zone_thermal_heating_power", "Zone thermal heat"),
    ("heat_pump_thermal_power", "HP thermal power"),
    ("heat_pump_electric_power", "HP electric power"),
]
ChangeDirection = Literal["absolute", "increase", "decrease"]


def _read_dataset(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if path.is_dir():
        read_path: Path | list[Path] = sorted(path.glob("*.parquet"))
    else:
        read_path = path
    if not read_path:
        raise FileNotFoundError(f"No parquet files found in {path}")
    return pd.read_parquet(read_path, columns=list(columns))


def _available_columns(path: Path) -> list[str]:
    if path.is_dir():
        read_path = sorted(path.glob("*.parquet"))[0]
    else:
        read_path = path
    return list(pd.read_parquet(read_path).columns)


def _corr(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([a.rename("a"), b.rename("b")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 4:
        return float("nan")
    if frame["a"].std() == 0.0 or frame["b"].std() == 0.0:
        return float("nan")
    return float(frame["a"].corr(frame["b"]))


def _heat_series(group: pd.DataFrame, column: str, normalize_per_m2: bool) -> pd.Series:
    heat = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
    if column in {"heat_pump_thermal_power", "heat_pump_electric_power"} and HP_MODE_IS_DHW_COLUMN in group.columns:
        is_dhw = pd.to_numeric(group[HP_MODE_IS_DHW_COLUMN], errors="coerce").fillna(0.0) > 0.5
        heat = heat.mask(is_dhw, 0.0)
    if normalize_per_m2:
        floor_area = float(group[FLOOR_AREA_COLUMN].iloc[0])
        if not np.isfinite(floor_area) or floor_area <= 0.0:
            raise ValueError(
                f"Profile {group[PROFILE_ID_COLUMN].iloc[0]} has invalid floor_area={floor_area!r}"
            )
        heat = heat / floor_area
    return heat


def _event_mask(delta_heat: pd.Series, quantile: float, direction: ChangeDirection) -> pd.Series:
    finite = delta_heat.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(False, index=delta_heat.index)

    if direction == "absolute":
        metric = finite.abs()
        threshold = float(metric.quantile(quantile))
        return delta_heat.abs() >= threshold
    if direction == "increase":
        positive = finite[finite > 0.0]
        if positive.empty:
            return pd.Series(False, index=delta_heat.index)
        threshold = float(positive.quantile(quantile))
        return delta_heat >= threshold
    if direction == "decrease":
        negative_magnitude = -finite[finite < 0.0]
        if negative_magnitude.empty:
            return pd.Series(False, index=delta_heat.index)
        threshold = float(negative_magnitude.quantile(quantile))
        return -delta_heat >= threshold
    raise ValueError("direction must be 'absolute', 'increase', or 'decrease'")


def lagged_change_correlations(
    group: pd.DataFrame,
    *,
    heat_column: str,
    label: str,
    max_lag_steps: int,
    delta_steps: int,
    quantile: float,
    direction: ChangeDirection,
    normalize_per_m2: bool,
) -> pd.DataFrame:
    group = group.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    profile_id = int(group[PROFILE_ID_COLUMN].iloc[0])
    temperature = pd.to_numeric(group[TARGET_COLUMN], errors="coerce")
    heat = _heat_series(group, heat_column, normalize_per_m2)
    delta_heat = heat.diff(delta_steps)
    event_mask = _event_mask(delta_heat, quantile, direction)

    records = []
    for lag_steps in range(max_lag_steps + 1):
        delta_temperature = temperature.shift(-(lag_steps + delta_steps)) - temperature.shift(-lag_steps)
        corr = _corr(delta_heat.loc[event_mask], delta_temperature.loc[event_mask])
        records.append(
            {
                "egid": profile_id,
                "candidate": heat_column,
                "candidate_label": label,
                "lag_steps": lag_steps,
                "lag_h": lag_steps / 4.0,
                "corr_delta_heat_delta_temperature": corr,
                "event_count": int(event_mask.sum()),
                "event_fraction": float(event_mask.mean()),
                "delta_heat_threshold": float(delta_heat.loc[event_mask].abs().min())
                if event_mask.any()
                else float("nan"),
                "delta_heat_mean": float(delta_heat.loc[event_mask].mean())
                if event_mask.any()
                else float("nan"),
                "delta_heat_abs_mean": float(delta_heat.loc[event_mask].abs().mean())
                if event_mask.any()
                else float("nan"),
            }
        )
    return pd.DataFrame(records)


def make_summary(
    df: pd.DataFrame,
    *,
    heat_candidates: list[tuple[str, str]],
    max_lag_steps: int,
    delta_steps: int,
    quantile: float,
    direction: ChangeDirection,
    normalize_per_m2: bool,
) -> pd.DataFrame:
    records = []
    for _, group in df.groupby(PROFILE_ID_COLUMN, sort=True):
        for heat_column, label in heat_candidates:
            if heat_column not in group.columns:
                continue
            records.append(
                lagged_change_correlations(
                    group,
                    heat_column=heat_column,
                    label=label,
                    max_lag_steps=max_lag_steps,
                    delta_steps=delta_steps,
                    quantile=quantile,
                    direction=direction,
                    normalize_per_m2=normalize_per_m2,
                )
            )
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, egid), group in summary.groupby(["candidate", "egid"], sort=True):
        valid = group.dropna(subset=["corr_delta_heat_delta_temperature"])
        if valid.empty:
            continue
        best = valid.iloc[valid["corr_delta_heat_delta_temperature"].abs().argmax()]
        rows.append(best)
    return pd.DataFrame(rows)


def write_html(
    summary: pd.DataFrame,
    output_html: Path,
    *,
    quantile: float,
    direction: ChangeDirection,
    delta_steps: int,
    normalize_per_m2: bool,
) -> None:
    candidates = list(summary[["candidate", "candidate_label"]].drop_duplicates().itertuples(index=False, name=None))
    fig = make_subplots(
        rows=len(candidates),
        cols=3,
        subplot_titles=[
            title
            for _, label in candidates
            for title in (
                f"{label}: profile lag correlations",
                "Median and profile spread",
                "Best lag per profile",
            )
        ],
        horizontal_spacing=0.065,
        vertical_spacing=0.11,
    )

    for row, (candidate, label) in enumerate(candidates, start=1):
        candidate_summary = summary.loc[summary["candidate"] == candidate]
        for egid, group in candidate_summary.groupby("egid", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=group["lag_h"],
                    y=group["corr_delta_heat_delta_temperature"],
                    mode="lines",
                    line={"width": 1},
                    opacity=0.32,
                    name=str(egid),
                    legendgroup=str(egid),
                    showlegend=row == 1,
                    hovertemplate=(
                        f"candidate={label}<br>"
                        f"egid={egid}<br>"
                        "lag=%{x:.2f} h<br>"
                        "corr=%{y:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        by_lag = candidate_summary.groupby("lag_h")["corr_delta_heat_delta_temperature"]
        spread = by_lag.quantile([0.25, 0.5, 0.75]).unstack()
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread[0.75],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread[0.25],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.18)",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread[0.5],
                mode="lines",
                line={"width": 3, "color": "#1f77b4"},
                name=f"{label} median",
                showlegend=False,
                hovertemplate="lag=%{x:.2f} h<br>median corr=%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=2,
        )

        best = _best_rows(candidate_summary)
        if not best.empty:
            fig.add_trace(
                go.Scatter(
                    x=best["lag_h"],
                    y=best["corr_delta_heat_delta_temperature"],
                    mode="markers",
                    marker={
                        "size": np.clip(best["event_count"] / 12.0, 5, 18),
                        "color": best["event_fraction"],
                        "colorscale": "Viridis",
                        "showscale": row == 1,
                        "colorbar": {"title": "event frac"} if row == 1 else None,
                    },
                    text=best["egid"],
                    customdata=np.stack(
                        [
                            best["event_count"],
                            best["delta_heat_abs_mean"],
                            best["delta_heat_mean"],
                        ],
                        axis=-1,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "egid=%{text}<br>"
                        "best lag=%{x:.2f} h<br>"
                        "corr=%{y:.3f}<br>"
                        "events=%{customdata[0]}<br>"
                        "|dQ| mean=%{customdata[1]:.3f}<br>"
                        "dQ mean=%{customdata[2]:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=3,
            )

        for col in (1, 2, 3):
            fig.add_trace(
                go.Scatter(
                    x=[0, candidate_summary["lag_h"].max()],
                    y=[0, 0],
                    mode="lines",
                    line={"width": 1, "dash": "dot", "color": "gray"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    for row in range(1, len(candidates) + 1):
        fig.update_xaxes(title_text="lag after heat change [h]", row=row, col=1)
        fig.update_yaxes(title_text="corr(dQ, delayed dTin)", range=[-1, 1], row=row, col=1)
        fig.update_xaxes(title_text="lag after heat change [h]", row=row, col=2)
        fig.update_yaxes(title_text="corr(dQ, delayed dTin)", range=[-1, 1], row=row, col=2)
        fig.update_xaxes(title_text="best lag [h]", row=row, col=3)
        fig.update_yaxes(title_text="best corr", range=[-1, 1], row=row, col=3)

    unit = "W/m2" if normalize_per_m2 else "W"
    fig.update_layout(
        title=(
            "Lagged heat-change diagnostic: "
            f"{direction} dQ events above q{quantile:.2f}, "
            f"dQ unit={unit}, dT window={delta_steps} step(s)"
        ),
        template="plotly_white",
        height=max(650, 430 * len(candidates)),
        width=1850,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot lagged correlation between temporal changes in indoor temperature "
            "and large temporal changes in heating power."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator_diagnostics"))
    parser.add_argument("--max-lag-steps", type=int, default=96, help="Maximum lag in 15-minute steps.")
    parser.add_argument("--delta-steps", type=int, default=1, help="Difference interval in 15-minute steps.")
    parser.add_argument("--change-quantile", type=float, default=0.90)
    parser.add_argument(
        "--change-direction",
        choices=("absolute", "increase", "decrease"),
        default="absolute",
        help="Which heat changes define events. 'increase' isolates turn-on/ramp-up events.",
    )
    parser.add_argument(
        "--raw-watts",
        action="store_true",
        help="Use raw W. By default heat is divided by floor_area before differencing.",
    )
    args = parser.parse_args()

    if args.max_lag_steps < 0:
        raise ValueError("max-lag-steps must be non-negative")
    if args.delta_steps < 1:
        raise ValueError("delta-steps must be positive")
    if not 0.0 < args.change_quantile < 1.0:
        raise ValueError("change-quantile must be in (0, 1)")

    available = _available_columns(args.dataset)
    heat_candidates = [(column, label) for column, label in HEAT_CANDIDATES if column in available]
    required_columns = [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        *(column for column, _ in heat_candidates),
    ]
    optional_columns = [FLOOR_AREA_COLUMN, HP_MODE_IS_DHW_COLUMN, SETPOINT_COLUMN]
    columns = [column for column in [*required_columns, *optional_columns] if column in available]
    normalize_per_m2 = not args.raw_watts
    if normalize_per_m2 and FLOOR_AREA_COLUMN not in columns:
        raise ValueError("floor_area is required for default W/m2 heat-change diagnostics")

    df = _read_dataset(args.dataset, columns)
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
    df = df.sort_values([PROFILE_ID_COLUMN, DATETIME_COLUMN]).reset_index(drop=True)
    summary = make_summary(
        df,
        heat_candidates=heat_candidates,
        max_lag_steps=args.max_lag_steps,
        delta_steps=args.delta_steps,
        quantile=args.change_quantile,
        direction=args.change_direction,
        normalize_per_m2=normalize_per_m2,
    )
    if summary.empty:
        raise ValueError("No heat-change diagnostics could be computed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"heating_signal_lagged_dq_{args.change_direction}_q{int(args.change_quantile * 100):02d}"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    html_path = args.output_dir / f"{stem}.html"
    summary.to_csv(summary_path, index=False)
    write_html(
        summary,
        html_path,
        quantile=args.change_quantile,
        direction=args.change_direction,
        delta_steps=args.delta_steps,
        normalize_per_m2=normalize_per_m2,
    )

    best = _best_rows(summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {html_path}")
    print(
        best.groupby("candidate")[
            ["corr_delta_heat_delta_temperature", "lag_h", "event_count", "event_fraction"]
        ]
        .median()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
