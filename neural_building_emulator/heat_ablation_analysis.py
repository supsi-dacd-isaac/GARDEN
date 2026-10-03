"""Ex-post heat-input ablation plots for saved emulator artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import jax
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
    required_columns,
)
from .data import DEFAULT_DATASET_PATH, BuildingProfile, _heating_regime_features, to_profiles
from .metrics import regression_metrics
from .model_io import load_training_artifact
from .train import full_profile_arrays, predict_full_profile


def _read_profile(
    dataset_path: Path,
    profile_id: int,
    heating_mode: str,
    heat_input_normalization: str,
    input_feature_mode: str,
    heating_regime_window_steps: int,
    heat_on_threshold: float,
) -> BuildingProfile:
    read_path: Path | list[Path]
    if dataset_path.is_dir():
        read_path = sorted(dataset_path.glob("*.parquet"))
    else:
        read_path = dataset_path
    columns = required_columns(heating_mode)
    try:
        df = pd.read_parquet(
            read_path,
            columns=columns,
            filters=[(PROFILE_ID_COLUMN, "=", profile_id)],
        )
    except (ValueError, NotImplementedError):
        df = pd.read_parquet(read_path, columns=columns)
        df = df[df[PROFILE_ID_COLUMN].astype(int) == int(profile_id)]
    if df.empty:
        raise ValueError(f"Profile {profile_id} not found in {dataset_path}")
    profiles = to_profiles(
        df,
        heating_mode,
        heat_input_normalization,  # type: ignore[arg-type]
        input_feature_mode,  # type: ignore[arg-type]
        heating_regime_window_steps,
        heat_on_threshold,
    )
    if len(profiles) != 1:
        raise ValueError(f"Expected one profile for {profile_id}, found {len(profiles)}")
    return profiles[0]


def _raw_profile_frame(dataset_path: Path, profile_id: int) -> pd.DataFrame:
    columns = [
        DATETIME_COLUMN,
        PROFILE_ID_COLUMN,
        TARGET_COLUMN,
        *HEATING_INPUT_COLUMNS.values(),
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
        df = df[df[PROFILE_ID_COLUMN].astype(int) == int(profile_id)]
    return df.sort_values(DATETIME_COLUMN).reset_index(drop=True)


def _profile_with_heat_scale(
    profile: BuildingProfile,
    scale: float,
    *,
    input_feature_mode: str,
    heating_regime_window_steps: int,
    heat_on_threshold: float,
) -> BuildingProfile:
    inputs = profile.inputs.copy()
    inputs[:, 0] = inputs[:, 0] * np.float32(scale)
    if input_feature_mode == "heating_regime":
        inputs[:, -2:] = _heating_regime_features(
            inputs[:, 0],
            heat_on_threshold=heat_on_threshold,
            window_steps=heating_regime_window_steps,
        )
    return replace(profile, inputs=inputs)


def _label_for_scale(scale: float) -> str:
    return f"Q x {scale:g}"


def save_heat_ablation_plot(
    *,
    artifact_dir: Path,
    dataset_path: Path,
    profile_id: int,
    output_dir: Path,
    heat_scales: Sequence[float],
    num_particles: int,
) -> Path:
    artifact = load_training_artifact(artifact_dir)
    artifact_metadata = artifact.metadata
    heating_mode = artifact_metadata["heating_mode"]
    heat_input_normalization = artifact_metadata["heat_input_normalization"]
    input_feature_mode = artifact_metadata.get("input_feature_mode", "base")
    train_config = artifact_metadata["train_config"]
    heating_regime_window_steps = int(
        artifact_metadata.get(
            "heating_regime_window_steps",
            train_config.get("heating_regime_window_steps", 96 * 7),
        )
    )
    heat_on_threshold = float(train_config.get("heat_on_threshold", 1e-6))
    target_alignment = artifact_metadata.get("target_alignment", train_config.get("target_alignment", "same_time"))
    target_mode = artifact_metadata["target_mode"]
    model_kind = artifact_metadata["model_kind"]

    profile = _read_profile(
        dataset_path,
        profile_id,
        heating_mode,
        heat_input_normalization,
        input_feature_mode,
        heating_regime_window_steps,
        heat_on_threshold,
    )
    raw = _raw_profile_frame(dataset_path, profile_id)
    raw[DATETIME_COLUMN] = pd.to_datetime(raw[DATETIME_COLUMN])
    raw_heat = raw[HEATING_INPUT_COLUMNS[heating_mode]].to_numpy(dtype=np.float32)
    floor_area = float(raw["floor_area"].iloc[0])
    heat_for_plot = raw_heat / floor_area if heat_input_normalization == "per_floor_area" else raw_heat
    heat_units = "W/m2" if heat_input_normalization == "per_floor_area" else "W"

    predictions: dict[float, np.ndarray] = {}
    for index, scale in enumerate(heat_scales):
        scaled_profile = _profile_with_heat_scale(
            profile,
            scale,
            input_feature_mode=input_feature_mode,
            heating_regime_window_steps=heating_regime_window_steps,
            heat_on_threshold=heat_on_threshold,
        )
        prediction = predict_full_profile(
            artifact.model,
            scaled_profile,
            artifact.scalers,
            target_mode,
            target_alignment=target_alignment,  # type: ignore[arg-type]
            model_kind=model_kind,
            key=jax.random.PRNGKey(10_000 + index),
            num_particles=num_particles,
        )
        predictions[float(scale)] = prediction.ravel()

    baseline_scale = 1.0 if 1.0 in predictions else float(heat_scales[0])
    baseline_prediction = predictions[baseline_scale]
    _, aligned_target, aligned_datetime, _ = full_profile_arrays(profile, target_alignment)  # type: ignore[arg-type]
    baseline_metrics = regression_metrics(baseline_prediction[:, None], aligned_target)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "Temperature response to heat-channel ablation",
            "Original forcing signals",
            f"Prediction change relative to {_label_for_scale(baseline_scale)}",
        ),
        row_heights=[0.46, 0.27, 0.27],
    )
    x = raw[DATETIME_COLUMN]
    prediction_x = pd.to_datetime(aligned_datetime)
    fig.add_trace(
        go.Scattergl(
            x=prediction_x,
            y=aligned_target.ravel(),
            mode="lines",
            name="Simulated",
            line={"color": "#1f77b4", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=raw["shSetpoint"],
            mode="lines",
            name="Setpoint",
            line={"color": "#444", "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    for index, (scale, prediction) in enumerate(predictions.items()):
        fig.add_trace(
            go.Scattergl(
                x=prediction_x,
                y=prediction,
                mode="lines",
                name=_label_for_scale(scale),
                line={"color": palette[index % len(palette)]},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=heat_for_plot,
            mode="lines",
            name=f"Q heat [{heat_units}]",
            line={"color": "#d62728"},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=raw[DISTURBANCE_COLUMNS[0]],
            mode="lines",
            name="Outdoor temperature [C]",
            line={"color": "#1f77b4"},
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=raw[DISTURBANCE_COLUMNS[1]] / 20.0,
            mode="lines",
            name="Solar / 20",
            line={"color": "#f28e2b", "dash": "dot"},
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    for index, (scale, prediction) in enumerate(predictions.items()):
        if scale == baseline_scale:
            continue
        fig.add_trace(
            go.Scattergl(
                x=prediction_x,
                y=prediction - baseline_prediction,
                mode="lines",
                name=f"{_label_for_scale(scale)} - {_label_for_scale(baseline_scale)}",
                line={"color": palette[index % len(palette)]},
            ),
            row=3,
            col=1,
        )
    fig.add_hline(y=0.0, row=3, col=1, line_dash="dash", line_color="#888")
    fig.update_yaxes(title_text="Indoor temperature [degC]", row=1, col=1)
    fig.update_yaxes(title_text=f"Q heat [{heat_units}]", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Tout [C] / solar scale", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Temperature delta [degC]", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(
        title=(
            f"Heat ablation, profile {profile_id}, artifact {artifact_dir.name}"
            f"<br><sup>baseline {_label_for_scale(baseline_scale)}: "
            f"RMSE={baseline_metrics.rmse:.3f} degC, MAE={baseline_metrics.mae:.3f} degC, "
            f"bias={baseline_metrics.bias:.3f} degC; model_kind={model_kind}</sup>"
        ),
        template="plotly_white",
        height=1100,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"heat_ablation_profile_{profile_id}.html"
    fig.write_html(output_path, include_plotlyjs=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Saved model artifact directory.")
    parser.add_argument("--dataset", type=Path, default=None, help="Parquet dataset path. Defaults to artifact config dataset.")
    parser.add_argument("--profile-id", action="append", type=int, default=None, help="Profile id to analyze. Can be repeated.")
    parser.add_argument("--profile-ids", nargs="+", type=int, default=None, help="Profile ids to analyze.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator_analysis"))
    parser.add_argument("--heat-scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--num-particles",
        type=int,
        default=64,
        help="Particle count for probabilistic mean predictions; ignored by deterministic models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_training_artifact(args.artifact)
    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = Path(artifact.metadata["train_config"].get("dataset_path") or DEFAULT_DATASET_PATH)
    profile_ids = args.profile_ids or args.profile_id or artifact.metadata.get("test_ids", [])[:1]
    if not profile_ids:
        raise ValueError("No profile id provided and artifact has no test ids.")
    for profile_id in profile_ids:
        output_path = save_heat_ablation_plot(
            artifact_dir=args.artifact,
            dataset_path=dataset_path,
            profile_id=int(profile_id),
            output_dir=args.output_dir,
            heat_scales=args.heat_scales,
            num_particles=args.num_particles,
        )
        print(f"saved_heat_ablation_plot={output_path}")


if __name__ == "__main__":
    main()
