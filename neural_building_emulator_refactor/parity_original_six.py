"""Exact parity test for the six original state-space model kinds.

Each pair is trained independently from the same seed and configuration:

1. direct call through ``neural_building_emulator.train``;
2. call through the refactored registry adapter.

The selected artifacts are then compared with zero numerical tolerance.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, TextIO

import equinox as eqx
import jax
import numpy as np
import pandas as pd

from neural_building_emulator.model_io import load_training_artifact
from neural_building_emulator.train import run_training

from .artifacts import LoadedArtifact, load_artifact
from .config import ExperimentConfig, OptimizerConfig
from .legacy import build_legacy_config
from .prediction import predict_profile
from .profiles import load_profiles
from .registry import ModelSpec, get_model_spec
from .trainer import train

PARITY_MODELS = (
    "q_to_t_deterministic_ss",
    "q_to_t_probabilistic_ss",
    "closed_loop_hp_deterministic_ss",
    "closed_loop_hp_probabilistic_ss",
    "closed_loop_hp_contracting_deterministic",
    "closed_loop_hp_contracting_probabilistic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("neural_building_emulator/tessin_results.parquet/all_hp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/neural_building_emulator_refactor/original_six_parity"),
    )
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument(
        "--stride",
        type=int,
        default=40000,
        help="Default creates one training window per building, keeping the six-pair test short.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prob-particles", type=int, default=2)
    parser.add_argument("--prob-eval-particles", type=int, default=2)
    parser.add_argument("--prediction-profile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _legacy_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "state_dim": 3,
        "hidden_dim": 8,
        "depth": 1,
        "input_encoder_dim": 4,
        "input_encoder_hidden_dim": 8,
        "input_encoder_depth": 1,
        "target_mode": "absolute",
        "schur_mode": "near_identity",
        "schur_gamma": 0.99,
        "prob_particles": args.prob_particles,
        "prob_eval_particles": args.prob_eval_particles,
        "prob_plot_particles": args.prob_eval_particles,
        "prob_latent_dim": 2,
        "prob_process_noise": "constant",
        "prob_softopt_weight": 0.1,
        "prob_variogram_weight": 0.0,
        "prob_ires_weight": 0.0,
        "prob_flex_kpi_crps_weight": 0.0,
        "bptt_truncate_steps": 0,
        "num_window_plots": 0,
        "num_full_profile_plots": 0,
        "full_train_eval_every_epochs": 0,
        "save_model_every_epochs": 0,
        "log_update_diagnostics": False,
    }


def _experiment(
    args: argparse.Namespace,
    *,
    model_name: str,
    output_dir: Path,
) -> ExperimentConfig:
    return ExperimentConfig(
        model_name=model_name,
        dataset_path=args.dataset,
        output_dir=output_dir,
        max_profiles=args.max_profiles,
        test_fraction=args.test_fraction,
        seed=args.seed,
        sequence_length=args.sequence_length,
        stride=args.stride,
        rotate_window_starts=False,
        heating_mode="zone_thermal",
        heat_input_normalization="per_floor_area",
        optimizer=OptimizerConfig(
            epochs=1,
            batch_size=args.batch_size,
            learning_rate=1e-3,
            gradient_clip_norm=1.0,
            train_eval_max_windows=0,
        ),
        legacy_overrides=_legacy_overrides(args),
    )


def _train_direct(
    experiment: ExperimentConfig,
    spec: ModelSpec,
    log: TextIO,
) -> Path:
    config = build_legacy_config(experiment, spec)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        run_training(config)
    return Path(config.model_checkpoint_dir) / str(config.model_kind) / "selected"


def _train_refactor(experiment: ExperimentConfig, log: TextIO) -> Path:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return train(experiment)


def _array_leaves(value: Any) -> list[np.ndarray]:
    filtered = eqx.filter(value, eqx.is_array)
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(filtered) if leaf is not None]


def _compare_array_lists(
    original: list[np.ndarray],
    refactor: list[np.ndarray],
) -> tuple[bool, float, int]:
    if len(original) != len(refactor):
        return False, float("inf"), abs(len(original) - len(refactor))
    exact = True
    max_abs = 0.0
    mismatched = 0
    for left, right in zip(original, refactor):
        same = left.shape == right.shape and np.array_equal(left, right, equal_nan=True)
        if not same:
            exact = False
            mismatched += 1
            if left.shape != right.shape:
                max_abs = float("inf")
            else:
                difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
                finite = difference[np.isfinite(difference)]
                if finite.size:
                    max_abs = max(max_abs, float(np.max(finite)))
    return exact, max_abs, mismatched


def _metric_payload(artifact) -> dict[str, Any]:
    metadata = artifact.metadata
    return {
        "checkpoint_epoch": metadata.get("checkpoint_epoch"),
        "checkpoint_metric": metadata.get("checkpoint_metric"),
        "checkpoint_metric_value": metadata.get("checkpoint_metric_value"),
        "train_metrics": metadata.get("train_metrics"),
        "test_metrics": metadata.get("test_metrics"),
        "selected_ids": metadata.get("selected_ids"),
        "train_ids": metadata.get("train_ids"),
        "test_ids": metadata.get("test_ids"),
    }


def _prediction_difference(
    original: LoadedArtifact,
    refactor: LoadedArtifact,
    *,
    dataset_path: Path,
    enabled: bool,
) -> tuple[bool, float]:
    if not enabled:
        return True, 0.0
    profiles = load_profiles(
        refactor,
        dataset_path=dataset_path,
        source="test",
        max_profiles=1,
    )
    profile = profiles[0]
    key = jax.random.PRNGKey(991)
    original_prediction = predict_profile(original, profile, key=key).mean
    refactor_prediction = predict_profile(refactor, profile, key=key).mean
    exact = np.array_equal(original_prediction, refactor_prediction, equal_nan=True)
    difference = np.abs(
        original_prediction.astype(np.float64) - refactor_prediction.astype(np.float64)
    )
    finite = difference[np.isfinite(difference)]
    return exact, float(np.max(finite)) if finite.size else 0.0


def _compare_pair(
    original_dir: Path,
    refactor_dir: Path,
    *,
    dataset_path: Path,
    compare_prediction: bool,
) -> dict[str, Any]:
    original_legacy = load_training_artifact(original_dir)
    refactor_legacy = load_training_artifact(refactor_dir)
    model_exact, model_max_abs, mismatched_leaves = _compare_array_lists(
        _array_leaves(original_legacy.model),
        _array_leaves(refactor_legacy.model),
    )
    scaler_exact, scaler_max_abs, mismatched_scalers = _compare_array_lists(
        [
            original_legacy.scalers.metadata.mean,
            original_legacy.scalers.metadata.scale,
            original_legacy.scalers.inputs.mean,
            original_legacy.scalers.inputs.scale,
            original_legacy.scalers.target.mean,
            original_legacy.scalers.target.scale,
        ],
        [
            refactor_legacy.scalers.metadata.mean,
            refactor_legacy.scalers.metadata.scale,
            refactor_legacy.scalers.inputs.mean,
            refactor_legacy.scalers.inputs.scale,
            refactor_legacy.scalers.target.mean,
            refactor_legacy.scalers.target.scale,
        ],
    )
    metrics_exact = _metric_payload(original_legacy) == _metric_payload(refactor_legacy)
    original = load_artifact(original_dir)
    refactor = load_artifact(refactor_dir)
    prediction_exact, prediction_max_abs = _prediction_difference(
        original,
        refactor,
        dataset_path=dataset_path,
        enabled=compare_prediction,
    )
    exact = model_exact and scaler_exact and metrics_exact and prediction_exact
    return {
        "exact": exact,
        "model_exact": model_exact,
        "model_max_abs_difference": model_max_abs,
        "mismatched_model_leaves": mismatched_leaves,
        "scalers_exact": scaler_exact,
        "scaler_max_abs_difference": scaler_max_abs,
        "mismatched_scalers": mismatched_scalers,
        "metrics_and_splits_exact": metrics_exact,
        "prediction_exact": prediction_exact,
        "prediction_max_abs_difference": prediction_max_abs,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    suite_started = time.perf_counter()
    for index, model_name in enumerate(PARITY_MODELS, start=1):
        spec = get_model_spec(model_name)
        pair_dir = args.output_dir / model_name
        original_experiment = _experiment(
            args,
            model_name=model_name,
            output_dir=pair_dir / "original",
        )
        refactor_experiment = dataclasses.replace(
            original_experiment,
            output_dir=pair_dir / "refactor",
        )
        print(f"[{index}/{len(PARITY_MODELS)}] training {model_name} original")
        pair_started = time.perf_counter()
        pair_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (pair_dir / "original.log").open("w") as log:
                original_dir = _train_direct(original_experiment, spec, log)
            print(f"[{index}/{len(PARITY_MODELS)}] training {model_name} refactor")
            with (pair_dir / "refactor.log").open("w") as log:
                refactor_dir = _train_refactor(refactor_experiment, log)
            comparison = _compare_pair(
                original_dir,
                refactor_dir,
                dataset_path=args.dataset,
                compare_prediction=args.prediction_profile,
            )
            row = {
                "model_name": model_name,
                "legacy_model_kind": spec.legacy_model_kind,
                "status": "pass" if comparison["exact"] else "mismatch",
                "elapsed_minutes": (time.perf_counter() - pair_started) / 60.0,
                **comparison,
                "original_artifact": str(original_dir),
                "refactor_artifact": str(refactor_dir),
            }
        except Exception as exc:
            row = {
                "model_name": model_name,
                "legacy_model_kind": spec.legacy_model_kind,
                "status": "error",
                "exact": False,
                "elapsed_minutes": (time.perf_counter() - pair_started) / 60.0,
                "error": repr(exc),
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir / "parity_results.csv", index=False)
        print(
            f"[{index}/{len(PARITY_MODELS)}] {model_name} status={row['status']} "
            f"elapsed_min={row['elapsed_minutes']:.2f}"
        )

    exact = all(bool(row.get("exact")) for row in rows)
    summary = {
        "exact_parity": exact,
        "model_count": len(rows),
        "passed": sum(row.get("status") == "pass" for row in rows),
        "mismatched": sum(row.get("status") == "mismatch" for row in rows),
        "errors": sum(row.get("status") == "error" for row in rows),
        "elapsed_minutes": (time.perf_counter() - suite_started) / 60.0,
        "configuration": {
            "dataset": str(args.dataset),
            "max_profiles": args.max_profiles,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "epochs": 1,
            "sequence_length": args.sequence_length,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "prob_particles": args.prob_particles,
            "prob_eval_particles": args.prob_eval_particles,
        },
    }
    (args.output_dir / "parity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved_parity_results={args.output_dir / 'parity_results.csv'}")
    print(f"exact_parity={exact}")
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
