"""Run short ablations for the probabilistic contracting closed-loop HP emulator.

The script trains several small comparable variants, evaluates the saved best
checkpoint with the event-study flexibility KPI script, and writes compact CSV
summaries of prediction error, gradient diagnostics, and KPI calibration.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_KIND = "closed_loop_hp_contracting_probabilistic"
DEFAULT_ABLATION_DATASET = Path(__file__).resolve().parent / "tessin_results.parquet" / "all_hp"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    temperature_update: str
    temperature_delta_max_c: float
    variogram_constant_setpoint_only: bool = True
    prob_softopt_weight: float = 1.0
    gradient_clip_norm: float = 1.0


VARIANTS: dict[str, VariantSpec] = {
    "absolute": VariantSpec(
        name="absolute",
        description="Legacy absolute bounded temperature readout; no per-step jump cap.",
        temperature_update="auto",
        temperature_delta_max_c=0.0,
    ),
    "legacy_delta_2c": VariantSpec(
        name="legacy_delta_2c",
        description="Legacy additive bounded-delta readout with a 2 degC physical jump cap.",
        temperature_update="auto",
        temperature_delta_max_c=2.0,
    ),
    "leaky_2c": VariantSpec(
        name="leaky_2c",
        description="New rollout-constant-alpha leaky equilibrium temperature update with a 2 degC jump cap.",
        temperature_update="leaky_equilibrium",
        temperature_delta_max_c=2.0,
    ),
    "leaky_no_constant_variogram": VariantSpec(
        name="leaky_no_constant_variogram",
        description="Leaky equilibrium update, but variogram pairs may cross setpoint jumps.",
        temperature_update="leaky_equilibrium",
        temperature_delta_max_c=2.0,
        variogram_constant_setpoint_only=False,
    ),
    "leaky_no_softopt": VariantSpec(
        name="leaky_no_softopt",
        description="Leaky equilibrium update with the trajectory soft Best-of-K term disabled.",
        temperature_update="leaky_equilibrium",
        temperature_delta_max_c=2.0,
        prob_softopt_weight=0.0,
    ),
    "leaky_clip_0p5": VariantSpec(
        name="leaky_clip_0p5",
        description="Leaky equilibrium update with tighter global gradient clipping.",
        temperature_update="leaky_equilibrium",
        temperature_delta_max_c=2.0,
        gradient_clip_norm=0.5,
    ),
    "leaky_clip_5": VariantSpec(
        name="leaky_clip_5",
        description="Leaky equilibrium update with looser global gradient clipping.",
        temperature_update="leaky_equilibrium",
        temperature_delta_max_c=2.0,
        gradient_clip_norm=5.0,
    ),
}

DEFAULT_VARIANTS = (
    "absolute",
    "legacy_delta_2c",
    "leaky_2c",
    "leaky_no_constant_variogram",
    "leaky_no_softopt",
    "leaky_clip_0p5",
)


def tee_run(command: list[str], *, log_path: Path, dry_run: bool) -> int:
    """Run a command while teeing stdout/stderr to both console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    log_path.write_text(f"$ {command_text}\n\n")
    print(f"$ {command_text}")
    if dry_run:
        return 0

    with log_path.open("a") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return int(process.wait())


def append_flag(command: list[str], flag: str, value: Any | None = None) -> None:
    command.append(flag)
    if value is not None:
        command.append(str(value))


def base_train_command(args: argparse.Namespace, variant: VariantSpec, variant_dir: Path) -> list[str]:
    command: list[str] = [
        sys.executable,
        "-m",
        "neural_building_emulator.train",
        "--dataset",
        str(args.dataset),
        "--max-profiles",
        str(args.max_profiles),
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--input-encoder-dim",
        str(args.input_encoder_dim),
        "--input-encoder-hidden-dim",
        str(args.input_encoder_hidden_dim),
        "--input-encoder-depth",
        str(args.input_encoder_depth),
        "--sequence-length",
        str(args.sequence_length),
        "--state-dim",
        str(args.state_dim),
        "--target-mode",
        "absolute",
        "--num-window-plots",
        str(args.num_window_plots),
        "--num-full-profile-plots",
        str(args.num_full_profile_plots),
        "--hidden-dim",
        str(args.hidden_dim),
        "--model-kind",
        MODEL_KIND,
        "--save-model",
        "--stride",
        str(args.stride),
        "--contracting-gamma",
        str(args.contracting_gamma),
        "--contracting-state-bound",
        str(args.contracting_state_bound),
        "--contracting-temperature-scale",
        str(args.contracting_temperature_scale),
        "--contracting-temperature-delta-max-c",
        str(variant.temperature_delta_max_c),
        "--contracting-temperature-update",
        variant.temperature_update,
        "--prob-particles",
        str(args.prob_particles),
        "--prob-eval-particles",
        str(args.prob_eval_particles),
        "--prob-plot-particles",
        str(args.prob_plot_particles),
        "--prob-latent-dim",
        str(args.prob_latent_dim),
        "--prob-process-noise",
        args.prob_process_noise,
        "--prob-hp-emission-mode",
        args.prob_hp_emission_mode,
        "--prob-softopt-weight",
        str(variant.prob_softopt_weight),
        "--prob-softopt-temperature",
        str(args.prob_softopt_temperature),
        "--prob-variogram-weight",
        str(args.prob_variogram_weight),
        "--prob-variogram-lags",
        *[str(value) for value in args.prob_variogram_lags],
        "--prob-variogram-output-weights",
        *[str(value) for value in args.prob_variogram_output_weights],
        "--prob-variogram-setpoint-threshold-c",
        str(args.prob_variogram_setpoint_threshold_c),
        "--prob-ires-weight",
        str(args.prob_ires_weight),
        "--prob-ires-horizons-hours",
        *[str(value) for value in args.prob_ires_horizons_hours],
        "--prob-ires-setpoint-threshold-c",
        str(args.prob_ires_setpoint_threshold_c),
        "--hp-inactive-leakage-weight",
        str(args.hp_inactive_leakage_weight),
        "--gradient-clip-norm",
        str(variant.gradient_clip_norm),
        "--max-consecutive-nonfinite-updates",
        str(args.max_consecutive_nonfinite_updates),
        "--output-dir",
        str(variant_dir / "training_output"),
        "--model-checkpoint-dir",
        str(variant_dir / "model_checkpoints"),
        "--log-update-diagnostics",
        "--validate-candidate-updates",
    ]
    if variant.variogram_constant_setpoint_only:
        command.append("--prob-variogram-constant-setpoint-only")
    if args.max_train_batches is not None:
        append_flag(command, "--max-train-batches", args.max_train_batches)
    if args.learning_rate is not None:
        append_flag(command, "--learning-rate", args.learning_rate)
    if args.early_stopping_patience is not None:
        append_flag(command, "--early-stopping-patience", args.early_stopping_patience)
    if args.save_model_every_epochs > 0:
        append_flag(command, "--save-model-every-epochs", args.save_model_every_epochs)
    return command


def kpi_command(args: argparse.Namespace, variant_dir: Path) -> list[str]:
    artifact_dir = variant_dir / "model_checkpoints" / MODEL_KIND / "selected"
    return [
        sys.executable,
        "-m",
        "neural_building_emulator.flexibility_event_study_kpis",
        "--artifact-dir",
        str(artifact_dir),
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(variant_dir / "flexibility_event_study"),
        "--profile-source",
        "test",
        "--emulation-kpi-mode",
        args.emulation_kpi_mode,
        "--num-scenarios",
        str(args.num_scenarios),
        "--prob-eval-particles",
        str(args.prob_eval_particles),
        "--prob-hp-scenario-mode",
        args.prob_hp_scenario_mode,
        "--prob-hp-emission-mode",
        args.prob_hp_emission_mode_override,
        "--horizons-hours",
        ",".join(str(value) for value in args.kpi_horizons_hours),
        "--min-setpoint-change",
        str(args.kpi_min_setpoint_change),
        "--min-events",
        str(args.kpi_min_events),
        "--controls",
        args.kpi_controls,
        "--seed",
        str(args.seed),
    ]


def latest_diagnostics_csv(variant_dir: Path) -> Path | None:
    diagnostics = sorted(
        (variant_dir / "model_checkpoints" / MODEL_KIND).glob("epoch_diagnostics_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    return diagnostics[-1] if diagnostics else None


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def parse_full_profile_metrics(log_path: Path) -> dict[str, float | int | str]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace")
    pattern = re.compile(
        r"full_profile_test profiles=(?P<profiles>\d+) "
        r"rmse_c=(?P<rmse>[-+0-9.eE]+) "
        r"mae_c=(?P<mae>[-+0-9.eE]+) "
        r"qroom_rmse_w_m2=(?P<qroom>[-+0-9.eE]+) "
        r"pel_rmse_w_m2=(?P<pel>[-+0-9.eE]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    match = matches[-1]
    return {
        "full_profile_profiles": int(match.group("profiles")),
        "full_profile_rmse_c": float(match.group("rmse")),
        "full_profile_mae_c": float(match.group("mae")),
        "full_profile_qroom_rmse_w_m2": float(match.group("qroom")),
        "full_profile_pel_rmse_w_m2": float(match.group("pel")),
    }


def summarize_training(variant_name: str, variant: VariantSpec, variant_dir: Path, train_returncode: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant_name,
        "description": variant.description,
        "train_returncode": train_returncode,
        "temperature_update": variant.temperature_update,
        "temperature_delta_max_c": variant.temperature_delta_max_c,
        "variogram_constant_setpoint_only": variant.variogram_constant_setpoint_only,
        "prob_softopt_weight": variant.prob_softopt_weight,
        "gradient_clip_norm": variant.gradient_clip_norm,
    }
    diagnostics_path = latest_diagnostics_csv(variant_dir)
    if diagnostics_path is None:
        return row
    diagnostics = pd.read_csv(diagnostics_path)
    row["epoch_diagnostics_csv"] = str(diagnostics_path)
    row["epochs_completed"] = int(len(diagnostics))

    for column in (
        "test_rmse_c",
        "train_rmse_c",
        "test_qroom_rmse_w_m2",
        "test_pel_rmse_w_m2",
        "max_grad_norm",
        "max_update_norm",
        "max_param_norm",
        "skipped_update_batches",
        "skipped_update_reason_grads",
        "nonfinite_train_batches",
        "nonfinite_component_batches",
    ):
        diagnostics[column] = numeric_column(diagnostics, column)

    test_rmse = diagnostics["test_rmse_c"]
    finite_test = test_rmse[np.isfinite(test_rmse)]
    if not finite_test.empty:
        best_index = int(finite_test.idxmin())
        best = diagnostics.loc[best_index]
        row.update(
            {
                "best_epoch": int(best["epoch"]),
                "best_test_rmse_c": float(best["test_rmse_c"]),
                "best_train_rmse_c": float(best["train_rmse_c"]),
                "best_test_qroom_rmse_w_m2": float(best["test_qroom_rmse_w_m2"]),
                "best_test_pel_rmse_w_m2": float(best["test_pel_rmse_w_m2"]),
            }
        )
    if len(diagnostics) > 0:
        last = diagnostics.iloc[-1]
        for source, target in (
            ("test_rmse_c", "last_test_rmse_c"),
            ("train_rmse_c", "last_train_rmse_c"),
            ("test_qroom_rmse_w_m2", "last_test_qroom_rmse_w_m2"),
            ("test_pel_rmse_w_m2", "last_test_pel_rmse_w_m2"),
        ):
            if np.isfinite(last[source]):
                row[target] = float(last[source])

    grad = diagnostics["max_grad_norm"].replace([np.inf, -np.inf], np.nan).dropna()
    if not grad.empty:
        row["max_grad_norm"] = float(grad.max())
        row["median_max_grad_norm"] = float(grad.median())
        row["p95_max_grad_norm"] = float(grad.quantile(0.95))
        row["grad_explosion_epochs_gt_1e3"] = int(np.sum(grad > 1e3))
        row["grad_explosion_epochs_gt_1e6"] = int(np.sum(grad > 1e6))
    row["skipped_update_batches_total"] = int(diagnostics["skipped_update_batches"].fillna(0).sum())
    row["skipped_update_batches_max"] = int(diagnostics["skipped_update_batches"].fillna(0).max())
    row["skipped_grad_batches_total"] = int(diagnostics["skipped_update_reason_grads"].fillna(0).sum())
    row["nonfinite_train_batches_total"] = int(diagnostics["nonfinite_train_batches"].fillna(0).sum())
    row["nonfinite_component_batches_total"] = int(diagnostics["nonfinite_component_batches"].fillna(0).sum())
    row.update(parse_full_profile_metrics(variant_dir / "train_stdout.log"))
    return row


def summarize_kpis(variant_name: str, variant_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir = variant_dir / "flexibility_event_study"
    profile_csv = output_dir / "flexibility_event_study_profile_kpis.csv"
    coverage_csv = output_dir / "flexibility_event_study_kpi_coverage.csv"
    row: dict[str, Any] = {"variant": variant_name}
    by_horizon: list[dict[str, Any]] = []

    if profile_csv.exists():
        profile = pd.read_csv(profile_csv)
        valid = profile[profile["valid"].astype(bool)].copy()
        row["kpi_profile_rows"] = int(len(profile))
        row["kpi_valid_rows"] = int(len(valid))
        row["kpi_profile_count"] = int(profile["profile_id"].nunique()) if "profile_id" in profile else 0
        if not valid.empty:
            row["kpi_up_mae_wh_m2_k"] = float(valid["diff_up_flex_wh_m2_k"].abs().mean())
            row["kpi_down_mae_wh_m2_k"] = float(valid["diff_down_flex_wh_m2_k"].abs().mean())
            row["kpi_up_bias_wh_m2_k"] = float(valid["diff_up_flex_wh_m2_k"].mean())
            row["kpi_down_bias_wh_m2_k"] = float(valid["diff_down_flex_wh_m2_k"].mean())
            for horizon, part in valid.groupby("horizon_hours", sort=True):
                horizon_row = {
                    "variant": variant_name,
                    "horizon_hours": float(horizon),
                    "valid_rows": int(len(part)),
                    "up_mae_wh_m2_k": float(part["diff_up_flex_wh_m2_k"].abs().mean()),
                    "down_mae_wh_m2_k": float(part["diff_down_flex_wh_m2_k"].abs().mean()),
                    "up_bias_wh_m2_k": float(part["diff_up_flex_wh_m2_k"].mean()),
                    "down_bias_wh_m2_k": float(part["diff_down_flex_wh_m2_k"].mean()),
                }
                if abs(float(horizon) - 3.0) < 1e-9:
                    row["kpi_3h_up_mae_wh_m2_k"] = horizon_row["up_mae_wh_m2_k"]
                    row["kpi_3h_down_mae_wh_m2_k"] = horizon_row["down_mae_wh_m2_k"]
                    row["kpi_3h_up_bias_wh_m2_k"] = horizon_row["up_bias_wh_m2_k"]
                    row["kpi_3h_down_bias_wh_m2_k"] = horizon_row["down_bias_wh_m2_k"]
                by_horizon.append(horizon_row)

    if coverage_csv.exists():
        coverage = pd.read_csv(coverage_csv)
        row["coverage_rows"] = int(len(coverage))
        if not coverage.empty:
            row["coverage_mean_abs_error"] = float(coverage["abs_calibration_error"].mean())
            row["coverage_max_abs_error"] = float(coverage["abs_calibration_error"].max())
            for (horizon, direction), part in coverage.groupby(["horizon_hours", "direction"], sort=True):
                for horizon_row in by_horizon:
                    if abs(float(horizon_row["horizon_hours"]) - float(horizon)) < 1e-9:
                        horizon_row[f"{direction}_coverage_mean_abs_error"] = float(
                            part["abs_calibration_error"].mean()
                        )
                        horizon_row[f"{direction}_coverage_max_abs_error"] = float(
                            part["abs_calibration_error"].max()
                        )
                        break
    return row, by_horizon


def selected_variants(args: argparse.Namespace) -> list[VariantSpec]:
    names = list(DEFAULT_VARIANTS if args.variants is None else args.variants)
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Known variants: {sorted(VARIANTS)}")
    return [VARIANTS[name] for name in names]


def write_manifest(args: argparse.Namespace, variants: list[VariantSpec]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "temperature_update": variant.temperature_update,
                "temperature_delta_max_c": variant.temperature_delta_max_c,
                "variogram_constant_setpoint_only": variant.variogram_constant_setpoint_only,
                "prob_softopt_weight": variant.prob_softopt_weight,
                "gradient_clip_norm": variant.gradient_clip_norm,
            }
            for variant in variants
        ],
        "train_epochs": args.epochs,
        "max_profiles": args.max_profiles,
        "test_fraction": args.test_fraction,
        "num_scenarios": args.num_scenarios,
    }
    (args.output_dir / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def run_ablation(args: argparse.Namespace) -> None:
    variants = selected_variants(args)
    if args.list_variants:
        for variant in variants:
            print(f"{variant.name}: {variant.description}")
        return

    write_manifest(args, variants)
    summary_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []

    for variant in variants:
        print(f"\n=== Variant: {variant.name} ===")
        variant_dir = args.output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        train_returncode: int | None = None

        artifact_dir = variant_dir / "model_checkpoints" / MODEL_KIND / "selected"
        if args.reuse_existing and artifact_dir.exists():
            print(f"reusing_existing_artifact={artifact_dir}")
        elif not args.skip_training:
            train_cmd = base_train_command(args, variant, variant_dir)
            (variant_dir / "train_command.txt").write_text(shlex.join(train_cmd) + "\n")
            train_returncode = tee_run(train_cmd, log_path=variant_dir / "train_stdout.log", dry_run=args.dry_run)
            if train_returncode != 0:
                print(f"variant={variant.name} training_failed_returncode={train_returncode}")
                if args.stop_on_failure:
                    raise SystemExit(train_returncode)
        else:
            print("skip_training=enabled")

        if not args.dry_run and not args.skip_kpi:
            if artifact_dir.exists():
                kpi_cmd = kpi_command(args, variant_dir)
                (variant_dir / "kpi_command.txt").write_text(shlex.join(kpi_cmd) + "\n")
                kpi_returncode = tee_run(kpi_cmd, log_path=variant_dir / "kpi_stdout.log", dry_run=False)
                if kpi_returncode != 0:
                    print(f"variant={variant.name} kpi_failed_returncode={kpi_returncode}")
                    if args.stop_on_failure:
                        raise SystemExit(kpi_returncode)
            else:
                print(f"variant={variant.name} missing_artifact={artifact_dir}; skipping KPI evaluation")

        summary = summarize_training(variant.name, variant, variant_dir, train_returncode)
        kpi_summary, variant_horizon_rows = summarize_kpis(variant.name, variant_dir)
        summary.update(kpi_summary)
        summary_rows.append(summary)
        horizon_rows.extend(variant_horizon_rows)

        pd.DataFrame(summary_rows).to_csv(args.output_dir / "ablation_summary.csv", index=False)
        pd.DataFrame(horizon_rows).to_csv(args.output_dir / "ablation_by_horizon.csv", index=False)
        (args.output_dir / "ablation_summary.json").write_text(
            json.dumps(summary_rows, indent=2, sort_keys=True, default=str)
        )

    print(f"saved_ablation_summary={args.output_dir / 'ablation_summary.csv'}")
    print(f"saved_ablation_by_horizon={args.output_dir / 'ablation_by_horizon.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ABLATION_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/neural_building_emulator/closed_loop_prob_ablation"),
    )
    parser.add_argument("--variants", nargs="+", default=None, help=f"Subset of variants. Known: {', '.join(VARIANTS)}")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse an existing selected artifact per variant.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-kpi", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")

    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--stride", type=int, default=2024)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--input-encoder-dim", type=int, default=8)
    parser.add_argument("--input-encoder-hidden-dim", type=int, default=64)
    parser.add_argument("--input-encoder-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--max-consecutive-nonfinite-updates", type=int, default=8)
    parser.add_argument("--save-model-every-epochs", type=int, default=0)
    parser.add_argument("--num-window-plots", type=int, default=0)
    parser.add_argument("--num-full-profile-plots", type=int, default=0)

    parser.add_argument("--contracting-gamma", type=float, default=0.99)
    parser.add_argument("--contracting-state-bound", type=float, default=5.0)
    parser.add_argument("--contracting-temperature-scale", type=float, default=8.0)

    parser.add_argument("--prob-particles", type=int, default=10)
    parser.add_argument("--prob-eval-particles", type=int, default=16)
    parser.add_argument("--prob-plot-particles", type=int, default=32)
    parser.add_argument("--prob-latent-dim", type=int, default=4)
    parser.add_argument("--prob-process-noise", choices=("none", "constant", "heteroscedastic"), default="heteroscedastic")
    parser.add_argument("--prob-hp-emission-mode", choices=("bounded", "legacy_lognormal_mean"), default="bounded")
    parser.add_argument("--prob-softopt-temperature", type=float, default=0.1)
    parser.add_argument("--prob-variogram-weight", type=float, default=1.0)
    parser.add_argument("--prob-variogram-lags", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--prob-variogram-output-weights", nargs="+", type=float, default=[1.0, 0.0, 0.0])
    parser.add_argument("--prob-variogram-setpoint-threshold-c", type=float, default=0.05)
    parser.add_argument("--prob-ires-weight", type=float, default=1.0)
    parser.add_argument("--prob-ires-horizons-hours", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--prob-ires-setpoint-threshold-c", type=float, default=0.05)
    parser.add_argument("--hp-inactive-leakage-weight", type=float, default=0.0)

    parser.add_argument("--emulation-kpi-mode", choices=("mean", "scenario_average"), default="scenario_average")
    parser.add_argument("--num-scenarios", type=int, default=100)
    parser.add_argument("--prob-hp-scenario-mode", choices=("expected", "bernoulli"), default="bernoulli")
    parser.add_argument(
        "--prob-hp-emission-mode-override",
        choices=("artifact", "bounded", "legacy_lognormal_mean"),
        default="artifact",
    )
    parser.add_argument("--kpi-horizons-hours", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--kpi-min-setpoint-change", type=float, default=0.05)
    parser.add_argument("--kpi-min-events", type=int, default=20)
    parser.add_argument("--kpi-controls", choices=("none", "weather", "full"), default="full")
    return parser.parse_args()


def main() -> None:
    run_ablation(parse_args())


if __name__ == "__main__":
    main()
