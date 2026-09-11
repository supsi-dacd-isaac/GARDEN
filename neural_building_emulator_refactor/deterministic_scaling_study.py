"""Runtime/accuracy study over profile count and window stride."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.data import (
    SplitConfig,
    load_closed_loop_result_splits,
    read_profile_ids,
)

from .artifacts import load_artifact
from .fast_kpis import FastKpiConfig, evaluate_fast_kpis
from .profiles import load_profiles


MODEL_Q_TO_T = "q_to_t_deterministic_ss"
MODEL_CLOSED_LOOP = "closed_loop_hp_contracting_deterministic"
MODELS = (MODEL_Q_TO_T, MODEL_CLOSED_LOOP)


Q_TO_T_OVERRIDES = {
    "state_dim": 5,
    "hidden_dim": 32,
    "depth": 3,
    "input_encoder_dim": 8,
    "input_encoder_hidden_dim": 64,
    "input_encoder_depth": 2,
    "schur_mode": "near_identity",
    "schur_gamma": 0.99,
    "target_mode": "absolute",
    "zero_d": True,
    "input_encoder_feedback": "thermal_gaps",
}


CLOSED_LOOP_OVERRIDES = {
    "state_dim": 8,
    "hidden_dim": 64,
    "depth": 3,
    "input_encoder_dim": 8,
    "input_encoder_hidden_dim": 64,
    "input_encoder_depth": 2,
    "contracting_gamma": 0.99,
    "contracting_state_bound": 5.0,
    "contracting_temperature_scale": 8.0,
    "contracting_temperature_delta_max_c": 2.0,
    "contracting_temperature_update": "leaky_equilibrium",
    "contracting_q_to_t_mode": "positive_leaky",
    "contracting_q_to_t_time_constants_hours": [0.25, 1.0, 4.0, 16.0, 24.0, 48.0],
    "contracting_q_to_t_gain_min_c_per_w_m2": 0.01,
    "contracting_q_to_t_gain_max_c_per_w_m2": 2.0,
    "bptt_truncate_steps": 32,
    "hp_thermostat_demand_mode": "monotone",
    "hp_cop_cap": 8.0,
    "hp_inactive_leakage_weight": 0.0,
}


@dataclass(frozen=True)
class StudyRun:
    model_name: str
    profile_count: int
    stride: int
    full_profile_count: int

    @property
    def slug(self) -> str:
        return f"{self.model_name}__profiles_{self.profile_count}__stride_{self.stride}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure deterministic emulator runtime/quality scaling."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("neural_building_emulator/tessin_results.parquet/all_hp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/neural_building_emulator_refactor/deterministic_scaling_study"),
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--profile-counts", type=int, nargs="+", default=(100, 300, 1000))
    parser.add_argument("--strides", type=int, nargs="+", default=(2024, 4048, 8096))
    parser.add_argument("--design", choices=("compact", "full"), default="compact")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--holdout-profiles", type=int, default=20)
    parser.add_argument("--train-eval-max-windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--quality-tolerance", type=float, default=0.10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _candidate_ids(dataset: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    all_ids = read_profile_ids(dataset)
    closed_loop = load_closed_loop_result_splits(
        SplitConfig(dataset_path=dataset, max_profiles=1, test_fraction=0.0)
    )
    return all_ids, closed_loop.candidate_ids


def _fixed_holdout(
    q_to_t_ids: tuple[int, ...],
    closed_loop_ids: tuple[int, ...],
    *,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    common = np.asarray(sorted(set(q_to_t_ids).intersection(closed_loop_ids)), dtype=np.int64)
    if count < 1 or count >= len(common):
        raise ValueError("holdout-profiles must be positive and smaller than the common pool")
    rng = np.random.default_rng(seed + 991)
    return tuple(sorted(int(value) for value in rng.choice(common, size=count, replace=False)))


def _study_runs(
    args: argparse.Namespace,
    full_counts: dict[str, int],
) -> list[StudyRun]:
    requested_counts = tuple(sorted(set(args.profile_counts)))
    strides = tuple(sorted(set(args.strides)))
    anchor_stride = min(strides, key=lambda value: abs(value - 4048))
    runs: list[StudyRun] = []
    for model_name in args.models:
        full_count = full_counts[model_name]
        counts = tuple(sorted(set(min(value, full_count) for value in requested_counts)))
        if args.design == "full":
            pairs = {(count, stride) for count in counts for stride in strides}
        else:
            middle_requested = min(requested_counts, key=lambda value: abs(value - 300))
            middle_count = min(middle_requested, full_count)
            pairs = {(count, anchor_stride) for count in counts}
            pairs.update((middle_count, stride) for stride in strides)
            # Keep the dense full-population reference and compare profile
            # diversity against temporal density at roughly equal window count.
            pairs.add((full_count, min(strides)))
            pairs.add((full_count, max(strides)))
            if middle_count not in counts:
                pairs.add((middle_count, anchor_stride))
        runs.extend(
            StudyRun(model_name, count, stride, full_count)
            for count, stride in sorted(pairs)
        )
    return runs


def _legacy_overrides(run: StudyRun, holdout_ids: tuple[int, ...]) -> dict:
    values = dict(Q_TO_T_OVERRIDES if run.model_name == MODEL_Q_TO_T else CLOSED_LOOP_OVERRIDES)
    values.update(
        fixed_test_profile_ids=list(holdout_ids),
        full_train_eval_every_epochs=0,
        num_window_plots=0,
        num_full_profile_plots=0,
        log_update_diagnostics=False,
    )
    return values


def _run_command(command: list[str], log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def _artifact_from_log(log_text: str) -> Path:
    matches = re.findall(r"selected_artifact=(.+)", log_text)
    if not matches:
        raise RuntimeError("Training log does not contain selected_artifact")
    return Path(matches[-1].strip())


def _training_log_metrics(log_text: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    profile_match = re.search(r"train_profiles=(\d+) test_profiles=(\d+)", log_text)
    if profile_match:
        result["train_profiles"] = int(profile_match.group(1))
        result["test_profiles"] = int(profile_match.group(2))
    train_window_match = re.search(r"train_windows=(\d+)", log_text)
    test_window_match = re.search(r"test_windows=(\d+)", log_text)
    if train_window_match:
        result["train_windows"] = int(train_window_match.group(1))
    if test_window_match:
        result["test_windows"] = int(test_window_match.group(1))
    epoch_minutes = [float(value) for value in re.findall(r"epoch_elapsed_min=([0-9.]+)", log_text)]
    if not epoch_minutes:
        epoch_minutes = [float(value) for value in re.findall(r"elapsed_min=([0-9.]+)", log_text)]
    if epoch_minutes:
        result["mean_epoch_minutes"] = float(np.mean(epoch_minutes))
        result["last_epoch_minutes"] = epoch_minutes[-1]
    return result


def _holdout_scale(artifact_dir: Path, dataset: Path) -> np.ndarray:
    artifact = load_artifact(artifact_dir)
    profiles = load_profiles(artifact, dataset_path=dataset, source="test")
    target = np.concatenate(
        [profile.targets if hasattr(profile, "targets") else profile.target for profile in profiles],
        axis=0,
    )
    scale = np.std(target, axis=0, dtype=np.float64)
    return np.maximum(scale, 1e-6)


def _flatten_fast_summary(summary: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric, statistics in summary["metrics"].items():
        for statistic, value in statistics.items():
            result[f"{metric}_{statistic}"] = float(value)
    return result


def _write_plot(path: Path, frame: pd.DataFrame) -> None:
    models = list(frame["model_name"].drop_duplicates())
    figure = make_subplots(
        rows=2,
        cols=len(models),
        subplot_titles=tuple(
            title
            for model in models
            for title in ()
        ) or None,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )
    colors = {2024: "#1769aa", 4048: "#2e7d32", 8096: "#d32f2f"}
    for column, model_name in enumerate(models, start=1):
        part = frame[frame["model_name"] == model_name]
        for stride in sorted(part["stride"].unique()):
            values = part[part["stride"] == stride].sort_values("train_profiles")
            color = colors.get(int(stride), "#6a1b9a")
            figure.add_trace(
                go.Scatter(
                    x=values["train_profiles"],
                    y=values["training_wall_minutes"],
                    mode="lines+markers",
                    name=f"stride {stride}",
                    legendgroup=f"stride_{stride}",
                    showlegend=column == 1,
                    line={"color": color},
                    marker={"size": 10},
                    text=[f"windows={value}" for value in values["train_windows"]],
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=values["training_wall_minutes"],
                    y=values["total_nrmse_mean"],
                    mode="lines+markers",
                    name=f"stride {stride}",
                    legendgroup=f"stride_{stride}",
                    showlegend=False,
                    line={"color": color},
                    marker={"size": 10},
                    text=[f"profiles={value}" for value in values["train_profiles"]],
                ),
                row=2,
                col=column,
            )
        figure.update_xaxes(title_text="training profiles", row=1, col=column)
        figure.update_yaxes(title_text="training wall time [min]", row=1, col=column)
        figure.update_xaxes(title_text="training wall time [min]", row=2, col=column)
        figure.update_yaxes(title_text="fixed-holdout total NRMSE", row=2, col=column)
        figure.add_annotation(
            x=0.5,
            y=1.08,
            xref=f"x{column} domain" if column > 1 else "x domain",
            yref="paper",
            text=model_name,
            showarrow=False,
            font={"size": 16},
        )
    figure.update_layout(
        title="Deterministic profile-count and stride scaling",
        template="plotly_white",
        height=900,
        width=max(900, 720 * len(models)),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    figure.write_html(path)


def _recommendations(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name, part in frame.groupby("model_name"):
        reference = part.sort_values(["profile_count", "stride"], ascending=[False, True]).iloc[0]
        candidates = part[
            part["total_nrmse_mean"] <= reference["total_nrmse_mean"] * (1.0 + tolerance)
        ].copy()
        if model_name == MODEL_CLOSED_LOOP and "flex_event_delta_p_nrmse_mean" in part:
            flex_tolerance = max(tolerance, 0.20)
            candidates = candidates[
                candidates["flex_event_delta_p_nrmse_mean"]
                <= reference["flex_event_delta_p_nrmse_mean"] * (1.0 + flex_tolerance)
            ]
            for metric in (
                "flex_up_energy_gain_abs_error_wh_m2_k_mean",
                "flex_down_energy_gain_abs_error_wh_m2_k_mean",
            ):
                if metric in candidates:
                    candidates = candidates[
                        candidates[metric] <= reference[metric] * (1.0 + flex_tolerance)
                    ]
        selected = reference if candidates.empty else candidates.sort_values("training_wall_minutes").iloc[0]
        rows.append(
            {
                "model_name": model_name,
                "recommended_profile_count": int(selected["profile_count"]),
                "recommended_stride": int(selected["stride"]),
                "training_wall_minutes": float(selected["training_wall_minutes"]),
                "total_nrmse_mean": float(selected["total_nrmse_mean"]),
                "reference_profile_count": int(reference["profile_count"]),
                "reference_stride": int(reference["stride"]),
                "reference_wall_minutes": float(reference["training_wall_minutes"]),
                "speedup": float(reference["training_wall_minutes"] / selected["training_wall_minutes"]),
            }
        )
    return pd.DataFrame(rows)


def run_study(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    q_ids, closed_ids = _candidate_ids(args.dataset)
    holdout_ids = _fixed_holdout(
        q_ids,
        closed_ids,
        count=args.holdout_profiles,
        seed=args.seed,
    )
    (args.output_dir / "fixed_holdout_profile_ids.json").write_text(
        json.dumps(list(holdout_ids), indent=2)
    )
    full_counts = {MODEL_Q_TO_T: len(q_ids), MODEL_CLOSED_LOOP: len(closed_ids)}
    runs = _study_runs(args, full_counts)
    plan = pd.DataFrame(
        [
            {
                "model_name": run.model_name,
                "profile_count": run.profile_count,
                "stride": run.stride,
                "full_profile_count": run.full_profile_count,
                "temporal_coverage_equivalents": args.epochs * args.sequence_length / run.stride,
            }
            for run in runs
        ]
    )
    plan.to_csv(args.output_dir / "study_plan.csv", index=False)
    print(plan.to_string(index=False))
    if args.dry_run:
        return args.output_dir

    rows: list[dict] = []
    for run_index, run in enumerate(runs, start=1):
        run_dir = args.output_dir / "runs" / run.slug
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "study_result.json"
        if args.resume and result_path.exists():
            print(f"resume_run={run.slug}")
            rows.append(json.loads(result_path.read_text()))
            continue
        overrides = _legacy_overrides(run, holdout_ids)
        config_path = run_dir / "legacy_config.json"
        config_path.write_text(json.dumps(overrides, indent=2, sort_keys=True))
        learning_rate = 1e-3 if run.model_name == MODEL_Q_TO_T else 3e-4
        command = [
            sys.executable,
            "-m",
            "neural_building_emulator_refactor.train",
            "--model",
            run.model_name,
            "--dataset",
            str(args.dataset),
            "--output-dir",
            str(run_dir),
            "--max-profiles",
            str(run.profile_count),
            "--test-fraction",
            "0.1",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(learning_rate),
            "--sequence-length",
            str(args.sequence_length),
            "--stride",
            str(run.stride),
            "--rotate-window-starts",
            "--train-eval-max-windows",
            str(args.train_eval_max_windows),
            "--legacy-config-json",
            str(config_path),
            "--seed",
            str(args.seed),
        ]
        print(f"study_run={run_index}/{len(runs)} slug={run.slug}")
        return_code, wall_seconds = _run_command(command, run_dir / "training.log")
        if return_code != 0:
            raise RuntimeError(f"Training failed for {run.slug}; see {run_dir / 'training.log'}")
        log_text = (run_dir / "training.log").read_text()
        artifact_dir = _artifact_from_log(log_text)
        scale = _holdout_scale(artifact_dir, args.dataset)
        score_dir = run_dir / "fast_kpis"
        evaluate_fast_kpis(
            artifact_dir,
            dataset_path=args.dataset,
            output_dir=score_dir,
            profile_source="test",
            max_profiles=args.holdout_profiles,
            num_eval_particles=1,
            target_scale_override=scale,
            config=FastKpiConfig(),
            seed=args.seed,
        )
        fast_summary = json.loads((score_dir / "fast_kpi_summary.json").read_text())
        result: dict = {
            "model_name": run.model_name,
            "profile_count": run.profile_count,
            "stride": run.stride,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "temporal_coverage_equivalents": args.epochs * args.sequence_length / run.stride,
            "training_wall_minutes": wall_seconds / 60.0,
            "artifact_dir": str(artifact_dir),
            **_training_log_metrics(log_text),
            **_flatten_fast_summary(fast_summary),
        }
        if "train_windows" in result:
            result["optimizer_steps_estimate"] = math.ceil(
                result["train_windows"] / args.batch_size
            ) * args.epochs
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output_dir / "scaling_results_partial.csv", index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "scaling_results.csv", index=False)
    _write_plot(args.output_dir / "deterministic_scaling_study.html", frame)
    recommendations = _recommendations(frame, args.quality_tolerance)
    recommendations.to_csv(args.output_dir / "recommendations.csv", index=False)
    print(recommendations.to_string(index=False))
    print(f"saved_scaling_study={args.output_dir}")
    return args.output_dir


def main() -> None:
    args = parse_args()
    run_study(args)


if __name__ == "__main__":
    main()
