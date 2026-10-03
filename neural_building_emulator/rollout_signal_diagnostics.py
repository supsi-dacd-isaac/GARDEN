"""Diagnostics comparing simulated and emulated heat-response dynamics."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .columns import (
    DATETIME_COLUMN,
    DISTURBANCE_COLUMNS,
    HEATING_INPUT_COLUMNS,
    PROFILE_ID_COLUMN,
    TARGET_COLUMN,
)
from .data import DEFAULT_DATASET_PATH
from .metrics import regression_metrics


def _extract_balanced(text: str, start: int, open_char: str, close_char: str) -> str:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    raise ValueError("Could not extract balanced JSON payload from Plotly HTML")


def _plotly_array(values: object) -> np.ndarray:
    if isinstance(values, dict) and "bdata" in values:
        dtype = np.dtype(values.get("dtype", "f8"))
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=dtype).astype(float)
    return np.asarray(values, dtype=float)


def parse_rollout_html(path: Path) -> pd.DataFrame:
    html = path.read_text()
    start = html.rfind("Plotly.newPlot(")
    if start < 0:
        raise ValueError(f"No Plotly.newPlot call found in {path}")
    data_start = html.find("[", start)
    data = json.loads(_extract_balanced(html, data_start, "[", "]"))

    traces = {trace.get("name", f"trace_{index}"): trace for index, trace in enumerate(data)}
    if "Simulated" not in traces or "Emulated" not in traces:
        raise ValueError(f"Expected Simulated and Emulated traces in {path}")
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(traces["Simulated"]["x"]),
            "simulated": _plotly_array(traces["Simulated"]["y"]),
            "emulated": _plotly_array(traces["Emulated"]["y"]),
        }
    )


def _profile_id_from_path(path: Path) -> int:
    match = re.search(r"_(\d+)(?:_prob)?\.html$", path.name)
    if match is None:
        raise ValueError(f"Could not infer profile id from {path.name}")
    return int(match.group(1))


def _read_raw_profile(dataset_path: Path, profile_id: int, heating_mode: str) -> pd.DataFrame:
    columns = [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        HEATING_INPUT_COLUMNS[heating_mode],
        *DISTURBANCE_COLUMNS,
        "shSetpoint",
        "floor_area",
    ]
    read_path: Path | list[Path]
    if dataset_path.is_dir():
        read_path = sorted(dataset_path.glob("*.parquet"))
    else:
        read_path = dataset_path
    try:
        df = pd.read_parquet(
            read_path,
            columns=columns,
            filters=[(PROFILE_ID_COLUMN, "=", profile_id)],
        )
    except (ValueError, NotImplementedError):
        df = pd.read_parquet(read_path, columns=columns)
        df = df[df[PROFILE_ID_COLUMN].astype(int) == profile_id]
    if df.empty:
        raise ValueError(f"Profile {profile_id} not found in {dataset_path}")
    return df.sort_values(DATETIME_COLUMN).reset_index(drop=True)


def _future_change(values: np.ndarray, *, lag: int, delta: int) -> np.ndarray:
    end = len(values) - lag - delta
    if end <= 1:
        return np.asarray([], dtype=float)
    return values[lag + delta : lag + delta + end] - values[lag : lag + end]


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if np.std(x_valid) < 1e-12 or np.std(y_valid) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _select_events(indices: np.ndarray, min_gap: int) -> np.ndarray:
    selected: list[int] = []
    last = -10**9
    for index in indices:
        if index - last >= min_gap:
            selected.append(int(index))
            last = int(index)
    return np.asarray(selected, dtype=int)


def _event_response(values: np.ndarray, events: np.ndarray, horizon: int) -> np.ndarray:
    responses = []
    for event in events:
        if event + horizon < len(values):
            responses.append(values[event : event + horizon + 1] - values[event])
    if not responses:
        return np.full((horizon + 1,), np.nan, dtype=float)
    return np.nanmean(np.asarray(responses), axis=0)


def _transition_center(df: pd.DataFrame, dq: np.ndarray, delta_steps: int) -> int:
    months = df["datetime_raw"].dt.month.to_numpy()
    shoulder = np.isin(months[: len(dq)], [3, 4, 5, 9, 10, 11])
    error = np.abs(df["error"].to_numpy()[: len(dq)])
    score = np.abs(dq) * error
    score = np.where(shoulder & np.isfinite(score), score, -np.inf)
    if not np.isfinite(score).any() or np.nanmax(score) == -np.inf:
        score = np.abs(dq) * error
    if not np.isfinite(score).any():
        return len(df) // 2
    return int(np.nanargmax(score) + delta_steps // 2)


def analyze_rollout(
    *,
    html_path: Path,
    dataset_path: Path,
    heating_mode: str,
    heat_change_quantile: float,
    delta_steps: int,
    max_lag_steps: int,
    event_horizon_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    profile_id = _profile_id_from_path(html_path)
    rollout = parse_rollout_html(html_path)
    raw = _read_raw_profile(dataset_path, profile_id, heating_mode)
    if len(raw) != len(rollout):
        raise ValueError(f"Length mismatch for {profile_id}: raw={len(raw)} html={len(rollout)}")

    df = pd.DataFrame(
        {
            "datetime": rollout["datetime"],
            "datetime_raw": pd.to_datetime(raw[DATETIME_COLUMN]),
            "profile_id": profile_id,
            "simulated": rollout["simulated"].to_numpy(dtype=float),
            "emulated": rollout["emulated"].to_numpy(dtype=float),
            "q_heat_raw": raw[HEATING_INPUT_COLUMNS[heating_mode]].to_numpy(dtype=float),
            "floor_area": raw["floor_area"].to_numpy(dtype=float),
            "outdoor_temperature": raw[DISTURBANCE_COLUMNS[0]].to_numpy(dtype=float),
            "solar": raw[DISTURBANCE_COLUMNS[1]].to_numpy(dtype=float),
            "ventilation": raw[DISTURBANCE_COLUMNS[2]].to_numpy(dtype=float),
            "setpoint": raw["shSetpoint"].to_numpy(dtype=float),
        }
    )
    df["q_heat_wm2"] = df["q_heat_raw"] / df["floor_area"]
    df["load_proxy"] = np.maximum(df["setpoint"] - df["outdoor_temperature"], 0.0)
    df["error"] = df["emulated"] - df["simulated"]

    q = df["q_heat_wm2"].to_numpy()
    simulated = df["simulated"].to_numpy()
    emulated = df["emulated"].to_numpy()
    dq = q[delta_steps:] - q[:-delta_steps]
    threshold = float(np.nanquantile(np.abs(dq), heat_change_quantile))
    large = np.abs(dq) >= threshold
    positive_events = _select_events(np.flatnonzero(dq >= threshold), event_horizon_steps // 2)
    negative_events = _select_events(np.flatnonzero(dq <= -threshold), event_horizon_steps // 2)
    scatter_lag = min(4, max_lag_steps)
    scatter_n = len(q) - scatter_lag - delta_steps
    df["dq_for_scatter"] = np.nan
    df["sim_dt_for_scatter"] = np.nan
    df["emu_dt_for_scatter"] = np.nan
    df["large_dq_for_scatter"] = False
    if scatter_n > 3:
        df.loc[: scatter_n - 1, "dq_for_scatter"] = dq[:scatter_n]
        df.loc[: scatter_n - 1, "sim_dt_for_scatter"] = _future_change(
            simulated,
            lag=scatter_lag,
            delta=delta_steps,
        )
        df.loc[: scatter_n - 1, "emu_dt_for_scatter"] = _future_change(
            emulated,
            lag=scatter_lag,
            delta=delta_steps,
        )
        df.loc[: scatter_n - 1, "large_dq_for_scatter"] = large[:scatter_n]

    lag_rows = []
    for lag in range(max_lag_steps + 1):
        n = len(q) - lag - delta_steps
        if n <= 3:
            continue
        x = dq[:n]
        event_mask = large[:n]
        sim_change = _future_change(simulated, lag=lag, delta=delta_steps)
        emu_change = _future_change(emulated, lag=lag, delta=delta_steps)
        lag_rows.append(
            {
                "profile_id": profile_id,
                "lag_steps": lag,
                "lag_hours": lag / 4.0,
                "corr_sim_all": _corr(x, sim_change),
                "corr_emu_all": _corr(x, emu_change),
                "corr_sim_large_dq": _corr(x[event_mask], sim_change[event_mask]),
                "corr_emu_large_dq": _corr(x[event_mask], emu_change[event_mask]),
                "n_large_dq": int(event_mask.sum()),
            }
        )
    lag_df = pd.DataFrame(lag_rows)

    horizon_hours = np.arange(event_horizon_steps + 1) / 4.0
    event_df = pd.DataFrame(
        {
            "profile_id": profile_id,
            "horizon_steps": np.arange(event_horizon_steps + 1),
            "horizon_hours": horizon_hours,
            "sim_positive_dq": _event_response(simulated, positive_events, event_horizon_steps),
            "emu_positive_dq": _event_response(emulated, positive_events, event_horizon_steps),
            "sim_negative_dq": _event_response(simulated, negative_events, event_horizon_steps),
            "emu_negative_dq": _event_response(emulated, negative_events, event_horizon_steps),
        }
    )

    metric = regression_metrics(df["emulated"].to_numpy()[:, None], df["simulated"].to_numpy()[:, None])
    best_sim = lag_df.loc[lag_df["corr_sim_large_dq"].abs().idxmax()] if lag_df["corr_sim_large_dq"].notna().any() else None
    best_emu = lag_df.loc[lag_df["corr_emu_large_dq"].abs().idxmax()] if lag_df["corr_emu_large_dq"].notna().any() else None
    one_hour = lag_df.loc[lag_df["lag_steps"] == 4].iloc[0] if (lag_df["lag_steps"] == 4).any() else None
    four_hour = lag_df.loc[lag_df["lag_steps"] == 16].iloc[0] if (lag_df["lag_steps"] == 16).any() else None
    summary = {
        "profile_id": profile_id,
        "html_path": str(html_path),
        "rmse_c": metric.rmse,
        "mae_c": metric.mae,
        "bias_c": metric.bias,
        "q_mean_wm2": float(np.nanmean(q)),
        "q_active_fraction": float(np.nanmean(q > 1e-6)),
        "q_load_corr": _corr(q, df["load_proxy"].to_numpy()),
        "dq_threshold_wm2": threshold,
        "positive_event_count": int(len(positive_events)),
        "negative_event_count": int(len(negative_events)),
        "corr_sim_large_dq_1h": float(one_hour["corr_sim_large_dq"]) if one_hour is not None else np.nan,
        "corr_emu_large_dq_1h": float(one_hour["corr_emu_large_dq"]) if one_hour is not None else np.nan,
        "corr_sim_large_dq_4h": float(four_hour["corr_sim_large_dq"]) if four_hour is not None else np.nan,
        "corr_emu_large_dq_4h": float(four_hour["corr_emu_large_dq"]) if four_hour is not None else np.nan,
        "best_sim_corr": float(best_sim["corr_sim_large_dq"]) if best_sim is not None else np.nan,
        "best_sim_lag_h": float(best_sim["lag_hours"]) if best_sim is not None else np.nan,
        "best_emu_corr": float(best_emu["corr_emu_large_dq"]) if best_emu is not None else np.nan,
        "best_emu_lag_h": float(best_emu["lag_hours"]) if best_emu is not None else np.nan,
        "sim_pos_response_24h": float(event_df["sim_positive_dq"].iloc[-1]),
        "emu_pos_response_24h": float(event_df["emu_positive_dq"].iloc[-1]),
        "sim_neg_response_24h": float(event_df["sim_negative_dq"].iloc[-1]),
        "emu_neg_response_24h": float(event_df["emu_negative_dq"].iloc[-1]),
        "transition_center_index": _transition_center(df, dq, delta_steps),
    }
    return df, lag_df, event_df, summary


def _iter_html_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.glob("test_full_profile_*.html")))
        else:
            result.append(path)
    return result


def write_diagnostic_html(
    *,
    profile_frames: list[pd.DataFrame],
    lag_frames: list[pd.DataFrame],
    event_frames: list[pd.DataFrame],
    summaries: list[dict[str, object]],
    output_path: Path,
    zoom_days: float,
) -> None:
    n_profiles = len(profile_frames)
    fig = make_subplots(
        rows=n_profiles,
        cols=4,
        specs=[[{}, {}, {}, {"secondary_y": True}] for _ in range(n_profiles)],
        subplot_titles=[
            title
            for summary in summaries
            for title in (
                f"{summary['profile_id']} dQ vs future dT",
                f"{summary['profile_id']} lag corr dQ -> future dT",
                f"{summary['profile_id']} average response to large dQ",
                f"{summary['profile_id']} transition zoom",
            )
        ],
        horizontal_spacing=0.055,
        vertical_spacing=0.035,
    )

    for row, (df, lag_df, event_df, summary) in enumerate(
        zip(profile_frames, lag_frames, event_frames, summaries, strict=True),
        start=1,
    ):
        scatter = df[df["large_dq_for_scatter"]].copy()
        if len(scatter) > 1200:
            scatter = scatter.sample(1200, random_state=13)
        fig.add_trace(
            go.Scattergl(
                x=scatter["dq_for_scatter"],
                y=scatter["sim_dt_for_scatter"],
                mode="markers",
                name="sim dT",
                marker={"color": "#1f77b4", "size": 4, "opacity": 0.45},
                legendgroup="sim_scatter",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=scatter["dq_for_scatter"],
                y=scatter["emu_dt_for_scatter"],
                mode="markers",
                name="emu dT",
                marker={"color": "#d62728", "size": 4, "opacity": 0.45},
                legendgroup="emu_scatter",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=0.0, row=row, col=1, line_dash="dash", line_color="#999")
        fig.add_vline(x=0.0, row=row, col=1, line_dash="dash", line_color="#999")

        fig.add_trace(
            go.Scatter(
                x=lag_df["lag_hours"],
                y=lag_df["corr_sim_large_dq"],
                mode="lines",
                name="sim corr",
                line={"color": "#1f77b4"},
                legendgroup="sim_corr",
                showlegend=row == 1,
            ),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=lag_df["lag_hours"],
                y=lag_df["corr_emu_large_dq"],
                mode="lines",
                name="emu corr",
                line={"color": "#d62728"},
                legendgroup="emu_corr",
                showlegend=row == 1,
            ),
            row=row,
            col=2,
        )
        fig.add_hline(y=0.0, row=row, col=2, line_dash="dash", line_color="#999")

        fig.add_trace(
            go.Scatter(
                x=event_df["horizon_hours"],
                y=event_df["sim_positive_dq"],
                mode="lines",
                name="sim after dQ+",
                line={"color": "#1f77b4"},
                legendgroup="sim_pos",
                showlegend=row == 1,
            ),
            row=row,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=event_df["horizon_hours"],
                y=event_df["emu_positive_dq"],
                mode="lines",
                name="emu after dQ+",
                line={"color": "#d62728"},
                legendgroup="emu_pos",
                showlegend=row == 1,
            ),
            row=row,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=event_df["horizon_hours"],
                y=event_df["sim_negative_dq"],
                mode="lines",
                name="sim after dQ-",
                line={"color": "#1f77b4", "dash": "dot"},
                legendgroup="sim_neg",
                showlegend=row == 1,
            ),
            row=row,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=event_df["horizon_hours"],
                y=event_df["emu_negative_dq"],
                mode="lines",
                name="emu after dQ-",
                line={"color": "#d62728", "dash": "dot"},
                legendgroup="emu_neg",
                showlegend=row == 1,
            ),
            row=row,
            col=3,
        )
        fig.add_hline(y=0.0, row=row, col=3, line_dash="dash", line_color="#999")

        center = int(summary["transition_center_index"])
        half_window = int(zoom_days * 96 / 2)
        start = max(0, center - half_window)
        end = min(len(df), center + half_window)
        zoom = df.iloc[start:end]
        fig.add_trace(
            go.Scattergl(
                x=zoom["datetime_raw"],
                y=zoom["simulated"],
                mode="lines",
                name="simulated Tin",
                line={"color": "#1f77b4"},
                legendgroup="sim_temp",
                showlegend=row == 1,
            ),
            row=row,
            col=4,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scattergl(
                x=zoom["datetime_raw"],
                y=zoom["emulated"],
                mode="lines",
                name="emulated Tin",
                line={"color": "#d62728"},
                legendgroup="emu_temp",
                showlegend=row == 1,
            ),
            row=row,
            col=4,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=zoom["datetime_raw"],
                y=zoom["setpoint"],
                mode="lines",
                name="setpoint",
                line={"color": "#444", "dash": "dash"},
                legendgroup="setpoint",
                showlegend=row == 1,
            ),
            row=row,
            col=4,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scattergl(
                x=zoom["datetime_raw"],
                y=zoom["q_heat_wm2"],
                mode="lines",
                name="Q heat W/m2",
                line={"color": "#ff7f0e"},
                legendgroup="q_heat",
                showlegend=row == 1,
            ),
            row=row,
            col=4,
            secondary_y=True,
        )

        fig.update_xaxes(title_text="dQ [W/m2 over interval]", row=row, col=1)
        fig.update_xaxes(title_text="lag [h]", row=row, col=2)
        fig.update_xaxes(title_text="hours after event", row=row, col=3)
        fig.update_yaxes(title_text="future dT [C]", row=row, col=1)
        fig.update_yaxes(title_text="corr", row=row, col=2)
        fig.update_yaxes(title_text="dTin [C]", row=row, col=3)
        fig.update_yaxes(title_text="Tin [C]", row=row, col=4, secondary_y=False)
        fig.update_yaxes(title_text="Q [W/m2]", row=row, col=4, secondary_y=True)

    fig.update_layout(
        title="Rollout heat-response diagnostics: simulated vs emulated",
        template="plotly_white",
        height=max(420 * n_profiles, 700),
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "html_paths",
        nargs="*",
        type=Path,
        default=[Path("output/neural_building_emulator")],
        help="Rollout HTML files or directories containing test_full_profile_*.html.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--heating-mode", choices=tuple(HEATING_INPUT_COLUMNS), default="zone_thermal")
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator_rollout_diagnostics"))
    parser.add_argument("--heat-change-quantile", type=float, default=0.90)
    parser.add_argument("--delta-steps", type=int, default=4, help="Temporal change interval; 4 means one hour.")
    parser.add_argument("--max-lag-steps", type=int, default=96, help="Maximum lag for dQ->dT correlation.")
    parser.add_argument("--event-horizon-steps", type=int, default=96, help="Average event-response horizon.")
    parser.add_argument("--zoom-days", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_paths = _iter_html_paths(args.html_paths)
    if not html_paths:
        raise ValueError("No rollout HTML files found")
    profile_frames = []
    lag_frames = []
    event_frames = []
    summaries = []
    for html_path in html_paths:
        df, lag_df, event_df, summary = analyze_rollout(
            html_path=html_path,
            dataset_path=args.dataset,
            heating_mode=args.heating_mode,
            heat_change_quantile=args.heat_change_quantile,
            delta_steps=args.delta_steps,
            max_lag_steps=args.max_lag_steps,
            event_horizon_steps=args.event_horizon_steps,
        )
        profile_frames.append(df)
        lag_frames.append(lag_df)
        event_frames.append(event_df)
        summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_path = args.output_dir / "rollout_heat_response_summary.csv"
    lag_path = args.output_dir / "rollout_heat_lag_correlations.csv"
    event_path = args.output_dir / "rollout_heat_event_responses.csv"
    html_path = args.output_dir / "rollout_heat_response_diagnostics.html"
    summary_df.to_csv(summary_path, index=False)
    pd.concat(lag_frames, ignore_index=True).to_csv(lag_path, index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(event_path, index=False)
    write_diagnostic_html(
        profile_frames=profile_frames,
        lag_frames=lag_frames,
        event_frames=event_frames,
        summaries=summaries,
        output_path=html_path,
        zoom_days=args.zoom_days,
    )
    print(f"saved_summary={summary_path}")
    print(f"saved_lag_correlations={lag_path}")
    print(f"saved_event_responses={event_path}")
    print(f"saved_diagnostic_html={html_path}")
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
