"""Regenerate emulator HTML plots from a saved model artifact without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .columns import (
    PROFILE_ID_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    closed_loop_required_columns,
    required_columns,
)
from .data import (
    DEFAULT_DATASET_PATH,
    WindowConfig,
    make_closed_loop_windows,
    make_windows,
    to_closed_loop_profiles,
    to_profiles,
)
from .model_io import load_training_artifact
from .models import (
    ClosedLoopHPEmulator,
    ContractingClosedLoopHPEmulator,
    MetadataStateSpaceEmulator,
    ProbabilisticClosedLoopHPEmulator,
    ProbabilisticContractingClosedLoopHPEmulator,
    ProbabilisticStableStateSpaceEmulator,
)
from .train import (
    evaluate_closed_loop_full_profiles,
    evaluate_full_profiles,
    evaluate_probabilistic_closed_loop_full_profiles,
    save_closed_loop_prediction_visualizations,
    save_prediction_visualizations,
    save_probabilistic_closed_loop_prediction_visualizations,
)


def _read_parquet(path: Path, *, columns: Iterable[str], profile_ids: list[int]) -> pd.DataFrame:
    filters = [(PROFILE_ID_COLUMN, "in", profile_ids)]
    read_path: Path | list[Path]
    if path.is_dir():
        read_path = sorted(path.glob("*.parquet"))
        if not read_path:
            raise FileNotFoundError(f"No parquet files found in {path}")
    else:
        read_path = path
    try:
        return pd.read_parquet(read_path, columns=list(columns), filters=filters)
    except (ValueError, NotImplementedError):
        df = pd.read_parquet(read_path, columns=list(columns))
        return df[df[PROFILE_ID_COLUMN].isin(profile_ids)].copy()


def _config_value(metadata: dict[str, Any], name: str, default: Any) -> Any:
    return metadata.get("train_config", {}).get(name, default)


def _saved_test_ids(metadata: dict[str, Any]) -> list[int]:
    ids = [int(value) for value in metadata.get("test_ids", [])]
    if not ids:
        raise ValueError("The artifact metadata does not contain saved test_ids; cannot reproduce plots.")
    return ids


def _closed_loop_suffix_and_title(model_kind: str) -> tuple[str, str]:
    if model_kind == "closed_loop_hp_contracting":
        return "_closed_loop_hp_contracting", "contracting closed-loop HP"
    if model_kind == "closed_loop_hp":
        return "_closed_loop_hp", "closed-loop HP"
    if model_kind == "closed_loop_hp_contracting_probabilistic":
        return "_closed_loop_hp_contracting_prob", "probabilistic contractive closed-loop HP"
    if model_kind == "closed_loop_hp_probabilistic":
        return "_closed_loop_hp_prob", "probabilistic closed-loop HP"
    raise ValueError(f"Unsupported closed-loop model kind {model_kind!r}")


def regenerate_closed_loop_plots(
    *,
    artifact_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    num_window_plots: int,
    num_full_profile_plots: int,
    sequence_length: int | None,
    stride: int | None,
    prob_plot_particles: int | None,
    prob_eval_particles: int | None,
    prob_hp_scenario_mode: str | None,
) -> None:
    artifact = load_training_artifact(artifact_dir)
    metadata = artifact.metadata
    model_kind = str(metadata["model_kind"])
    test_ids = _saved_test_ids(metadata)

    df = _read_parquet(
        dataset_path,
        columns=closed_loop_required_columns(),
        profile_ids=test_ids,
    )
    profiles = to_closed_loop_profiles(
        df,
        include_space_heating_availability=(
            SPACE_HEATING_AVAILABILITY_COLUMN in metadata.get("input_columns", [])
        ),
    )
    profiles = [profile for profile in profiles if profile.profile_id in set(test_ids)]
    profiles.sort(key=lambda profile: test_ids.index(profile.profile_id))
    if not profiles:
        raise ValueError("No saved test profiles were found in the dataset.")

    window_config = WindowConfig(
        sequence_length=int(sequence_length or _config_value(metadata, "sequence_length", 96)),
        stride=int(stride or _config_value(metadata, "stride", 96)),
    )
    windows = make_closed_loop_windows(profiles, window_config)
    suffix, title = _closed_loop_suffix_and_title(model_kind)

    if model_kind in ("closed_loop_hp_probabilistic", "closed_loop_hp_contracting_probabilistic"):
        model = artifact.model
        if not isinstance(model, (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator)):
            raise TypeError(f"Loaded model has unexpected type {type(model)!r}")
        eval_particles = int(prob_eval_particles or _config_value(metadata, "prob_eval_particles", 16))
        plot_particles = int(prob_plot_particles or _config_value(metadata, "prob_plot_particles", 100))
        scenario_mode = str(prob_hp_scenario_mode or _config_value(metadata, "prob_hp_scenario_mode", "bernoulli"))
        metrics = evaluate_probabilistic_closed_loop_full_profiles(
            model,
            profiles,
            artifact.scalers,
            num_particles=eval_particles,
        )
        print(
            "full_profile_test "
            f"profiles={int(metrics['profile_count'])} "
            f"rmse_c={metrics['rmse_c']:.4f} "
            f"mae_c={metrics['mae_c']:.4f} "
            f"qroom_rmse_w_m2={metrics['qroom_rmse_w_m2']:.4f} "
            f"pel_rmse_w_m2={metrics['pel_rmse_w_m2']:.4f}"
        )
        paths = save_probabilistic_closed_loop_prediction_visualizations(
            model,
            windows,
            profiles,
            artifact.scalers,
            output_dir,
            num_window_plots,
            num_full_profile_plots,
            num_particles=plot_particles,
            hp_scenario_mode=scenario_mode,  # type: ignore[arg-type]
            filename_suffix=suffix,
            title_label=title,
        )
    else:
        model = artifact.model
        if not isinstance(model, (ClosedLoopHPEmulator, ContractingClosedLoopHPEmulator)):
            raise TypeError(f"Loaded model has unexpected type {type(model)!r}")
        metrics = evaluate_closed_loop_full_profiles(model, profiles, artifact.scalers)
        print(
            "full_profile_test "
            f"profiles={int(metrics['profile_count'])} "
            f"rmse_c={metrics['rmse_c']:.4f} "
            f"mae_c={metrics['mae_c']:.4f} "
            f"qroom_rmse_w_m2={metrics['qroom_rmse_w_m2']:.4f} "
            f"pel_rmse_w_m2={metrics['pel_rmse_w_m2']:.4f}"
        )
        paths = save_closed_loop_prediction_visualizations(
            model,
            windows,
            profiles,
            artifact.scalers,
            output_dir,
            num_window_plots,
            num_full_profile_plots,
            filename_suffix=suffix,
            title_label=title,
        )

    for name, path in paths.items():
        print(f"saved_{name}_plot={path}")


def regenerate_q_to_t_plots(
    *,
    artifact_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    num_window_plots: int,
    num_full_profile_plots: int,
    sequence_length: int | None,
    stride: int | None,
    prob_plot_particles: int | None,
    prob_eval_particles: int | None,
) -> None:
    artifact = load_training_artifact(artifact_dir)
    metadata = artifact.metadata
    model_kind = str(metadata["model_kind"])
    test_ids = _saved_test_ids(metadata)
    heating_mode = str(metadata.get("heating_mode", _config_value(metadata, "heating_mode", "A")))
    heat_input_normalization = str(
        metadata.get(
            "heat_input_normalization",
            _config_value(metadata, "heat_input_normalization", "per_floor_area"),
        )
    )
    input_feature_mode = str(
        metadata.get("input_feature_mode", _config_value(metadata, "input_feature_mode", "base"))
    )
    heating_regime_window_steps = int(
        metadata.get(
            "heating_regime_window_steps",
            _config_value(metadata, "heating_regime_window_steps", 96 * 7),
        )
    )

    df = _read_parquet(
        dataset_path,
        columns=required_columns(heating_mode),
        profile_ids=test_ids,
    )
    profiles = to_profiles(
        df,
        heating_mode=heating_mode,
        heat_input_normalization=heat_input_normalization,  # type: ignore[arg-type]
        input_feature_mode=input_feature_mode,  # type: ignore[arg-type]
        heating_regime_window_steps=heating_regime_window_steps,
        heat_on_threshold=float(_config_value(metadata, "heat_on_threshold", 1e-6)),
    )
    profiles = [profile for profile in profiles if profile.profile_id in set(test_ids)]
    profiles.sort(key=lambda profile: test_ids.index(profile.profile_id))
    if not profiles:
        raise ValueError("No saved test profiles were found in the dataset.")

    window_config = WindowConfig(
        sequence_length=int(sequence_length or _config_value(metadata, "sequence_length", 96)),
        stride=int(stride or _config_value(metadata, "stride", 96)),
        target_alignment=str(_config_value(metadata, "target_alignment", "same_time")),  # type: ignore[arg-type]
    )
    windows = make_windows(profiles, window_config)
    target_mode = str(metadata.get("target_mode", _config_value(metadata, "target_mode", "absolute")))
    eval_particles = int(prob_eval_particles or _config_value(metadata, "prob_eval_particles", 16))
    plot_particles = int(prob_plot_particles or _config_value(metadata, "prob_plot_particles", 100))

    model = artifact.model
    if model_kind == "probabilistic":
        if not isinstance(model, ProbabilisticStableStateSpaceEmulator):
            raise TypeError(f"Loaded model has unexpected type {type(model)!r}")
    elif model_kind == "deterministic":
        if not isinstance(model, MetadataStateSpaceEmulator):
            raise TypeError(f"Loaded model has unexpected type {type(model)!r}")
    else:
        raise ValueError(f"Unsupported Q-to-T model kind {model_kind!r}")

    metrics = evaluate_full_profiles(
        model,
        profiles,
        artifact.scalers,
        target_mode,  # type: ignore[arg-type]
        target_alignment=window_config.target_alignment,
        model_kind=model_kind,  # type: ignore[arg-type]
        num_particles=eval_particles,
    )
    print(
        "full_profile_test "
        f"profiles={int(metrics['profile_count'])} "
        f"rmse_c={metrics['rmse_c']:.4f} "
        f"mae_c={metrics['mae_c']:.4f} "
        f"nmae={metrics['nmae']:.4f} "
        f"bias_c={metrics['bias_c']:.4f}"
    )
    paths = save_prediction_visualizations(
        model,
        windows,
        profiles,
        artifact.scalers,
        output_dir,
        target_mode,  # type: ignore[arg-type]
        num_window_plots,
        num_full_profile_plots,
        target_alignment=window_config.target_alignment,
        model_kind=model_kind,  # type: ignore[arg-type]
        num_particles=eval_particles,
        prob_plot_particles=plot_particles,
    )
    for name, path in paths.items():
        print(f"saved_{name}_plot={path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-window-plots", type=int, default=None)
    parser.add_argument("--num-full-profile-plots", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--prob-plot-particles", type=int, default=None)
    parser.add_argument("--prob-eval-particles", type=int, default=None)
    parser.add_argument(
        "--prob-hp-scenario-mode",
        choices=("expected", "bernoulli"),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_training_artifact(args.artifact_dir)
    metadata = artifact.metadata
    model_kind = str(metadata["model_kind"])
    dataset_path = args.dataset or Path(_config_value(metadata, "dataset_path", DEFAULT_DATASET_PATH))
    output_dir = args.output_dir or Path(_config_value(metadata, "output_dir", "output/neural_building_emulator"))
    num_window_plots = int(args.num_window_plots if args.num_window_plots is not None else _config_value(metadata, "num_window_plots", 1))
    num_full_profile_plots = int(
        args.num_full_profile_plots
        if args.num_full_profile_plots is not None
        else _config_value(metadata, "num_full_profile_plots", 1)
    )

    print(f"loaded_artifact={args.artifact_dir}")
    print(f"model_kind={model_kind}")
    print(f"dataset_path={dataset_path}")
    print(f"output_dir={output_dir}")
    print(f"saved_test_profiles={len(_saved_test_ids(metadata))}")
    print(f"num_window_plots={num_window_plots} num_full_profile_plots={num_full_profile_plots}")

    if model_kind in (
        "closed_loop_hp",
        "closed_loop_hp_contracting",
        "closed_loop_hp_probabilistic",
        "closed_loop_hp_contracting_probabilistic",
    ):
        regenerate_closed_loop_plots(
            artifact_dir=args.artifact_dir,
            dataset_path=dataset_path,
            output_dir=output_dir,
            num_window_plots=num_window_plots,
            num_full_profile_plots=num_full_profile_plots,
            sequence_length=args.sequence_length,
            stride=args.stride,
            prob_plot_particles=args.prob_plot_particles,
            prob_eval_particles=args.prob_eval_particles,
            prob_hp_scenario_mode=args.prob_hp_scenario_mode,
        )
        return

    regenerate_q_to_t_plots(
        artifact_dir=args.artifact_dir,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_window_plots=num_window_plots,
        num_full_profile_plots=num_full_profile_plots,
        sequence_length=args.sequence_length,
        stride=args.stride,
        prob_plot_particles=args.prob_plot_particles,
        prob_eval_particles=args.prob_eval_particles,
    )


if __name__ == "__main__":
    main()
