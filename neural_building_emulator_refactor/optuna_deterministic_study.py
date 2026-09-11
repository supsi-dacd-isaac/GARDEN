"""Conditional Optuna search for the deterministic contracting HP emulator."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from optuna.importance import (
    FanovaImportanceEvaluator,
    PedAnovaImportanceEvaluator,
    get_param_importances,
)
from optuna.trial import FrozenTrial, TrialState
from plotly.subplots import make_subplots

from .deterministic_scaling_study import (
    CLOSED_LOOP_OVERRIDES,
    MODEL_CLOSED_LOOP,
    _artifact_from_log,
    _candidate_ids,
    _flatten_fast_summary,
    _holdout_scale,
    _training_log_metrics,
)
from .fast_kpis import FastKpiConfig, evaluate_fast_kpis
from .artifacts import load_artifact
from .profiles import load_profiles
from .scoring import score_artifact


BASELINE_PARAMS: dict[str, Any] = {
    "thermostat_demand_mode": "monotone",
    "q_to_t_mode": "positive_leaky",
    "controller_calendar_features": True,
    "state_dim": 8,
    "controller_state_dim": 2,
    "hidden_dim": 64,
    "encoder_dim": 8,
    "learning_rate": 3e-4,
    "temperature_delta_max_c": 2.0,
    "qroom_loss_weight": 1.0,
    "pel_loss_weight": 1.0,
    "hp_mode_loss_weight": 0.1,
    "q_to_t_time_constants": "1,24",
    "q_to_t_gain_max": 2.0,
}

CORE_METRICS = (
    "total_nrmse_mean",
    "temperature_nrmse_mean",
    "qroom_nrmse_mean",
    "pel_nrmse_mean",
    "temperature_acf_mae_mean",
    "temperature_increment_spectral_js_mean",
    "flex_event_delta_p_nrmse_mean",
    "flex_event_delta_p_correlation_mean",
    "flex_event_response_sign_agreement_mean",
    "flex_up_energy_gain_abs_error_wh_m2_k_mean",
    "flex_down_energy_gain_abs_error_wh_m2_k_mean",
)

BASELINE_SCALE_KEYS = {
    "trajectory": "total_nrmse_mean",
    "temperature_acf": "temperature_acf_mae_mean",
    "temperature_spectrum": "temperature_increment_spectral_js_mean",
    "flexibility": "flex_event_delta_p_nrmse_mean",
}

ARCHITECTURE_FAMILIES = tuple(
    {
        "thermostat_demand_mode": thermostat,
        "q_to_t_mode": q_to_t,
        "controller_calendar_features": calendar,
    }
    for thermostat, q_to_t, calendar in product(
        ("unconstrained", "monotone"),
        ("unconstrained", "positive_leaky"),
        (False, True),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune deterministic contracting closed-loop structure and loss balance "
            "on a fixed validation cohort while reserving an untouched final test cohort."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("neural_building_emulator/tessin_results.parquet/all_hp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/neural_building_emulator_refactor/optuna_deterministic_study_8ep"),
    )
    parser.add_argument("--study-name", default="contracting_hp_deterministic_8ep")
    parser.add_argument("--max-profiles", type=int, default=1000)
    parser.add_argument("--validation-profiles", type=int, default=30)
    parser.add_argument("--final-test-profiles", type=int, default=30)
    parser.add_argument("--num-trials", type=int, default=60)
    parser.add_argument("--startup-trials", type=int, default=16)
    parser.add_argument(
        "--min-trials-per-architecture",
        type=int,
        default=3,
        help="Minimum queued trials for each thermostat/Q-to-T/calendar family.",
    )
    parser.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--stride", type=int, default=8096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-eval-max-windows", type=int, default=64)
    parser.add_argument(
        "--fixed-learning-rate",
        type=float,
        help="Use one learning rate for the baseline and every trial instead of tuning it.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--timeout-hours", type=float, default=0.0)
    parser.add_argument("--objective-trajectory-weight", type=float, default=0.50)
    parser.add_argument("--objective-temporal-weight", type=float, default=0.20)
    parser.add_argument("--objective-flexibility-weight", type=float, default=0.30)
    parser.add_argument("--report-top-k", type=int, default=10)
    parser.add_argument("--confirmation-top-k", type=int, default=3)
    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        nargs="+",
        default=(13, 29, 47),
        help="Seeds used to confirm the leading configurations before final testing.",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip multi-seed confirmation and untouched final-test evaluation.",
    )
    parser.add_argument(
        "--final-profile-plots",
        type=int,
        default=4,
        help="Full-year traces written for the confirmed winner and baseline.",
    )
    parser.add_argument("--verbose-trials", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_profiles <= args.validation_profiles:
        raise ValueError("max-profiles must exceed validation-profiles")
    if args.validation_profiles < 1 or args.final_test_profiles < 1:
        raise ValueError("validation-profiles and final-test-profiles must be positive")
    if args.num_trials < 1 or args.startup_trials < 1:
        raise ValueError("num-trials and startup-trials must be positive")
    minimum_search_trials = len(ARCHITECTURE_FAMILIES) * args.min_trials_per_architecture
    if args.min_trials_per_architecture < 1:
        raise ValueError("min-trials-per-architecture must be positive")
    if args.num_trials < minimum_search_trials:
        raise ValueError(
            f"num-trials must be at least {minimum_search_trials} to allocate "
            f"{args.min_trials_per_architecture} trials to each architecture family"
        )
    if args.report_top_k < 1 or args.confirmation_top_k < 1:
        raise ValueError("report-top-k and confirmation-top-k must be positive")
    if not args.confirmation_seeds:
        raise ValueError("confirmation-seeds must contain at least one seed")
    if args.final_profile_plots < 0:
        raise ValueError("final-profile-plots must be non-negative")
    if args.epochs < 1 or args.sequence_length < 2 or args.stride < 1:
        raise ValueError("epochs, sequence-length, and stride must be positive")
    if args.fixed_learning_rate is not None and args.fixed_learning_rate <= 0.0:
        raise ValueError("fixed-learning-rate must be positive")
    weights = _objective_weights(args)
    if any(value < 0.0 for value in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError("objective weights must be non-negative with a positive sum")


def _objective_weights(args: argparse.Namespace) -> dict[str, float]:
    raw = {
        "trajectory": float(args.objective_trajectory_weight),
        "temporal": float(args.objective_temporal_weight),
        "flexibility": float(args.objective_flexibility_weight),
    }
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()} if total > 0.0 else raw


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value

    return {
        name: convert(value)
        for name, value in vars(args).items()
    }


def _study_signature(args: argparse.Namespace) -> dict[str, Any]:
    ignored = {
        "timeout_hours",
        "verbose_trials",
        "dry_run",
        "report_top_k",
        "confirmation_top_k",
        "confirmation_seeds",
        "skip_confirmation",
        "final_profile_plots",
    }
    return {
        name: value
        for name, value in _jsonable_args(args).items()
        if name not in ignored
    }


def _prepare_split(args: argparse.Namespace) -> dict[str, list[int]]:
    split_path = args.output_dir / "profile_split.json"
    _, closed_ids = _candidate_ids(args.dataset)
    candidate_set = set(closed_ids)
    if split_path.exists():
        split = json.loads(split_path.read_text())
        used = set(split["validation_ids"]) | set(split["final_test_ids"])
        missing = sorted(used.difference(candidate_set))
        if missing:
            raise ValueError(f"Saved profile split contains unavailable IDs: {missing}")
        return split

    required = args.validation_profiles + args.final_test_profiles
    if required >= len(closed_ids):
        raise ValueError("Validation and final-test cohorts exhaust the eligible profile pool")
    rng = np.random.default_rng(args.seed + 1907)
    shuffled = rng.permutation(np.asarray(closed_ids, dtype=np.int64))
    final_ids = tuple(sorted(int(value) for value in shuffled[: args.final_test_profiles]))
    validation_ids = tuple(
        sorted(
            int(value)
            for value in shuffled[
                args.final_test_profiles : args.final_test_profiles + args.validation_profiles
            ]
        )
    )
    split = {
        "eligible_profile_count": len(closed_ids),
        "validation_ids": list(validation_ids),
        "final_test_ids": list(final_ids),
    }
    split_path.write_text(json.dumps(split, indent=2))
    return split


def _sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.startup_trials,
        multivariate=True,
        group=True,
    )


def _architecture_label(params: dict[str, Any]) -> str:
    return (
        f"{params['thermostat_demand_mode']} | {params['q_to_t_mode']} | "
        f"calendar={params['controller_calendar_features']}"
    )


def _enqueue_architecture_quotas(
    study: optuna.Study,
    minimum_per_family: int,
) -> list[dict[str, Any]]:
    """Queue a balanced architecture block once, before adaptive TPE trials."""
    if study.user_attrs.get("architecture_quotas_enqueued"):
        return []
    baseline_family = {
        name: BASELINE_PARAMS[name]
        for name in (
            "thermostat_demand_mode",
            "q_to_t_mode",
            "controller_calendar_features",
        )
    }
    queued: list[dict[str, Any]] = []
    for repeat in range(minimum_per_family):
        for family in ARCHITECTURE_FAMILIES:
            # The mandatory baseline trial already occupies one slot in its family.
            if family == baseline_family and repeat == minimum_per_family - 1:
                continue
            params = dict(family)
            study.enqueue_trial(
                params,
                user_attrs={
                    "architecture_quota": True,
                    "architecture": _architecture_label(params),
                },
            )
            queued.append(params)
    study.set_user_attr("architecture_quotas_enqueued", True)
    study.set_user_attr("minimum_trials_per_architecture", minimum_per_family)
    return queued


def _storage(path: Path) -> optuna.storages.BaseStorage:
    backend = optuna.storages.journal.JournalFileBackend(str(path))
    return optuna.storages.JournalStorage(backend)


def _suggest_params(
    trial: optuna.Trial,
    fixed_learning_rate: float | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "thermostat_demand_mode": trial.suggest_categorical(
            "thermostat_demand_mode", ["unconstrained", "monotone"]
        ),
        "q_to_t_mode": trial.suggest_categorical(
            "q_to_t_mode", ["unconstrained", "positive_leaky"]
        ),
        "controller_calendar_features": trial.suggest_categorical(
            "controller_calendar_features", [False, True]
        ),
        "state_dim": trial.suggest_categorical("state_dim", [5, 8, 12]),
        "controller_state_dim": trial.suggest_categorical(
            "controller_state_dim", [1, 2, 4]
        ),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96]),
        "encoder_dim": trial.suggest_categorical("encoder_dim", [4, 8, 16]),
        "learning_rate": (
            trial.suggest_categorical("learning_rate", [float(fixed_learning_rate)])
            if fixed_learning_rate is not None
            else trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        ),
        "temperature_delta_max_c": trial.suggest_float(
            "temperature_delta_max_c", 1.0, 3.0
        ),
        # Tin is the unit reference; only the relative Qroom/Pel emphasis matters
        # because the trainer normalizes the three supplied output weights.
        "qroom_loss_weight": trial.suggest_float(
            "qroom_loss_weight", 0.1, 3.0, log=True
        ),
        "pel_loss_weight": trial.suggest_float("pel_loss_weight", 0.05, 2.0, log=True),
        "hp_mode_loss_weight": trial.suggest_float(
            "hp_mode_loss_weight", 0.01, 1.0, log=True
        ),
    }
    if params["q_to_t_mode"] == "positive_leaky":
        params["q_to_t_time_constants"] = trial.suggest_categorical(
            "q_to_t_time_constants", ["0.5,6", "1,24", "1,8,48"]
        )
        params["q_to_t_gain_max"] = trial.suggest_float(
            "q_to_t_gain_max", 0.25, 3.0, log=True
        )
    return params


def _time_constants(value: str) -> list[float]:
    return [float(part) for part in value.split(",")]


def _trial_overrides(
    params: dict[str, Any],
    validation_ids: list[int],
    final_test_ids: list[int],
    train_eval_max_windows: int,
) -> dict[str, Any]:
    values = dict(CLOSED_LOOP_OVERRIDES)
    values.update(
        fixed_test_profile_ids=validation_ids,
        excluded_profile_ids=final_test_ids,
        state_dim=int(params["state_dim"]),
        hp_controller_state_dim=int(params["controller_state_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        input_encoder_dim=int(params["encoder_dim"]),
        hp_controller_calendar_features=bool(params["controller_calendar_features"]),
        hp_thermostat_demand_mode=str(params["thermostat_demand_mode"]),
        contracting_temperature_delta_max_c=float(params["temperature_delta_max_c"]),
        contracting_q_to_t_mode=str(params["q_to_t_mode"]),
        closed_loop_trajectory_output_weights=[
            1.0,
            float(params["qroom_loss_weight"]),
            float(params["pel_loss_weight"]),
        ],
        hp_mode_loss_weight=float(params["hp_mode_loss_weight"]),
        checkpoint_metric="test_total_nrmse",
        full_train_eval_every_epochs=0,
        train_eval_max_windows=train_eval_max_windows,
        num_window_plots=0,
        num_full_profile_plots=0,
        log_update_diagnostics=False,
        save_model_every_epochs=0,
    )
    if params["q_to_t_mode"] == "positive_leaky":
        values.update(
            contracting_q_to_t_time_constants_hours=_time_constants(
                str(params["q_to_t_time_constants"])
            ),
            contracting_q_to_t_gain_min_c_per_w_m2=0.01,
            contracting_q_to_t_gain_max_c_per_w_m2=float(params["q_to_t_gain_max"]),
        )
    return values


def _run_training(command: list[str], log_path: Path, *, verbose: bool) -> tuple[int, float]:
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
            if verbose or line.startswith(("epoch=", "selected_checkpoint", "full_profile_test")):
                print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def _metric_values(summary: dict[str, Any]) -> dict[str, float]:
    flattened = _flatten_fast_summary(summary)
    values = {name: float(flattened.get(name, math.nan)) for name in CORE_METRICS}
    required = tuple(BASELINE_SCALE_KEYS.values())
    if any(not np.isfinite(values[name]) for name in required):
        missing = [name for name in required if not np.isfinite(values[name])]
        raise FloatingPointError(f"Required validation KPIs are non-finite: {missing}")
    return values


def _baseline_scales(study: optuna.Study) -> dict[str, float] | None:
    stored = study.user_attrs.get("baseline_scales")
    if stored is not None:
        return {name: float(value) for name, value in stored.items()}
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if trial.user_attrs.get("is_baseline"):
            scales = {
                name: max(float(trial.user_attrs[f"metric_{metric}"]), 1e-4)
                for name, metric in BASELINE_SCALE_KEYS.items()
            }
            study.set_user_attr("baseline_scales", scales)
            return scales
    return None


def _objective_components(
    metrics: dict[str, float],
    scales: dict[str, float],
    weights: dict[str, float],
) -> dict[str, float]:
    trajectory_ratio = metrics["total_nrmse_mean"] / scales["trajectory"]
    acf_ratio = metrics["temperature_acf_mae_mean"] / scales["temperature_acf"]
    spectrum_ratio = (
        metrics["temperature_increment_spectral_js_mean"] / scales["temperature_spectrum"]
    )
    temporal_ratio = 0.5 * (acf_ratio + spectrum_ratio)
    flexibility_ratio = metrics["flex_event_delta_p_nrmse_mean"] / scales["flexibility"]
    objective = (
        weights["trajectory"] * trajectory_ratio
        + weights["temporal"] * temporal_ratio
        + weights["flexibility"] * flexibility_ratio
    )
    return {
        "objective": float(objective),
        "trajectory_ratio": float(trajectory_ratio),
        "temperature_acf_ratio": float(acf_ratio),
        "temperature_spectrum_ratio": float(spectrum_ratio),
        "temporal_ratio": float(temporal_ratio),
        "flexibility_ratio": float(flexibility_ratio),
    }


def _training_command(
    args: argparse.Namespace,
    params: dict[str, Any],
    split: dict[str, list[int]],
    run_dir: Path,
    *,
    seed: int,
) -> list[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = _trial_overrides(
        params,
        split["validation_ids"],
        split["final_test_ids"],
        args.train_eval_max_windows,
    )
    config_path = run_dir / "legacy_config.json"
    config_path.write_text(json.dumps(overrides, indent=2, sort_keys=True))
    return [
        sys.executable,
        "-m",
        "neural_building_emulator_refactor.train",
        "--model",
        MODEL_CLOSED_LOOP,
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(run_dir),
        "--max-profiles",
        str(args.max_profiles),
        "--test-fraction",
        "0.1",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(params["learning_rate"]),
        "--sequence-length",
        str(args.sequence_length),
        "--stride",
        str(args.stride),
        "--rotate-window-starts",
        "--train-eval-max-windows",
        str(args.train_eval_max_windows),
        "--legacy-config-json",
        str(config_path),
        "--seed",
        str(seed),
    ]


def _objective_factory(
    args: argparse.Namespace,
    study: optuna.Study,
    split: dict[str, list[int]],
):
    weights = _objective_weights(args)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, args.fixed_learning_rate)
        is_baseline = bool(trial.user_attrs.get("is_baseline", False))
        trial_dir = args.output_dir / "trials" / f"trial_{trial.number:04d}"
        command = _training_command(args, params, split, trial_dir, seed=args.seed)
        print(
            f"trial={trial.number} baseline={is_baseline} "
            f"architecture={params['thermostat_demand_mode']}+{params['q_to_t_mode']}"
        )
        return_code, wall_seconds = _run_training(
            command,
            trial_dir / "training.log",
            verbose=args.verbose_trials,
        )
        if return_code != 0:
            trial.set_user_attr("failure", "training")
            raise RuntimeError(f"Training failed; see {trial_dir / 'training.log'}")
        log_text = (trial_dir / "training.log").read_text()
        artifact_dir = _artifact_from_log(log_text)
        scale = _holdout_scale(artifact_dir, args.dataset)
        score_dir = trial_dir / "fast_kpis"
        evaluate_fast_kpis(
            artifact_dir,
            dataset_path=args.dataset,
            output_dir=score_dir,
            profile_source="test",
            max_profiles=args.validation_profiles,
            num_eval_particles=1,
            target_scale_override=scale,
            config=FastKpiConfig(),
            seed=args.seed,
        )
        summary = json.loads((score_dir / "fast_kpi_summary.json").read_text())
        metrics = _metric_values(summary)
        for name, value in metrics.items():
            if np.isfinite(value):
                trial.set_user_attr(f"metric_{name}", value)
        trial.set_user_attr("artifact_dir", str(artifact_dir))
        trial.set_user_attr("wall_minutes", wall_seconds / 60.0)
        trial.set_user_attr("is_baseline", is_baseline)
        for name, value in _training_log_metrics(log_text).items():
            trial.set_user_attr(f"training_{name}", value)

        scales = _baseline_scales(study)
        if scales is None:
            if not is_baseline:
                raise RuntimeError("The baseline trial must complete before adaptive trials")
            scales = {
                name: max(float(metrics[metric]), 1e-4)
                for name, metric in BASELINE_SCALE_KEYS.items()
            }
            study.set_user_attr("baseline_scales", scales)
            (args.output_dir / "baseline_scales.json").write_text(
                json.dumps(scales, indent=2, sort_keys=True)
            )
        components = _objective_components(metrics, scales, weights)
        for name, value in components.items():
            trial.set_user_attr(name, value)
        result = {
            "trial": trial.number,
            "is_baseline": is_baseline,
            "params": params,
            "metrics": metrics,
            "objective_components": components,
            "artifact_dir": str(artifact_dir),
            "wall_minutes": wall_seconds / 60.0,
        }
        (trial_dir / "trial_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        print(
            f"trial_result={trial.number} objective={components['objective']:.4f} "
            f"trajectory={components['trajectory_ratio']:.3f} "
            f"temporal={components['temporal_ratio']:.3f} "
            f"flexibility={components['flexibility_ratio']:.3f}"
        )
        return components["objective"]

    return objective


def _completed_frame(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        row: dict[str, Any] = {
            "trial": trial.number,
            "objective": float(trial.value),
            "duration_minutes": (
                trial.duration.total_seconds() / 60.0 if trial.duration is not None else math.nan
            ),
            "is_baseline": bool(trial.user_attrs.get("is_baseline", False)),
        }
        row.update(trial.params)
        for name in CORE_METRICS:
            row[name] = trial.user_attrs.get(f"metric_{name}", math.nan)
        for name in (
            "trajectory_ratio",
            "temperature_acf_ratio",
            "temperature_spectrum_ratio",
            "temporal_ratio",
            "flexibility_ratio",
        ):
            row[name] = trial.user_attrs.get(name, math.nan)
        row["architecture"] = (
            f"{trial.params.get('thermostat_demand_mode', '?')} | "
            f"{trial.params.get('q_to_t_mode', '?')} | "
            f"calendar={trial.params.get('controller_calendar_features', '?')}"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trial") if rows else pd.DataFrame()


def _common_params(study: optuna.Study) -> list[str]:
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not complete:
        return []
    common = set(complete[0].params)
    for trial in complete[1:]:
        common.intersection_update(trial.params)
    return sorted(
        name
        for name in common
        if len({trial.params[name] for trial in complete}) > 1
    )


def _importance(study: optuna.Study) -> pd.DataFrame:
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    common = _common_params(study)
    if len(complete) < 5 or not common:
        return pd.DataFrame(columns=("parameter", "ped_anova", "fanova"))
    evaluators = {
        "ped_anova": PedAnovaImportanceEvaluator(target_quantile=0.25),
        "fanova": FanovaImportanceEvaluator(seed=17),
    }
    values: dict[str, dict[str, float]] = {}
    for name, evaluator in evaluators.items():
        try:
            values[name] = get_param_importances(
                study,
                evaluator=evaluator,
                params=common,
            )
        except (RuntimeError, ValueError, ZeroDivisionError):
            values[name] = {}
    return pd.DataFrame(
        [
            {
                "parameter": parameter,
                "ped_anova": values["ped_anova"].get(parameter, math.nan),
                "fanova": values["fanova"].get(parameter, math.nan),
            }
            for parameter in common
        ]
    ).sort_values("ped_anova", ascending=False, na_position="last")


def _write_dashboard(
    frame: pd.DataFrame,
    importance: pd.DataFrame,
    path: Path,
    weights: dict[str, float],
) -> None:
    if frame.empty:
        return
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Optimization history",
            "Trajectory-flexibility trade-off",
            "Temperature dynamics trade-off",
            "Architecture-family objective",
            "Parameter importance",
            "Runtime-quality trade-off",
        ),
        vertical_spacing=0.11,
        horizontal_spacing=0.10,
    )
    order = frame.sort_values("trial")
    figure.add_trace(
        go.Scatter(
            x=order["trial"],
            y=order["objective"],
            mode="markers",
            name="Trial objective",
            marker={"color": "#78909c", "size": 8},
            text=order["architecture"],
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=order["trial"],
            y=order["objective"].cummin(),
            mode="lines",
            name="Running best",
            line={"color": "#c62828", "width": 3},
        ),
        row=1,
        col=1,
    )
    palette = ["#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#ad1457"]
    for index, (family, part) in enumerate(frame.groupby("architecture")):
        color = palette[index % len(palette)]
        figure.add_trace(
            go.Scatter(
                x=part["trajectory_ratio"],
                y=part["flexibility_ratio"],
                mode="markers",
                name=family,
                legendgroup=family,
                marker={
                    "color": part["temporal_ratio"],
                    "colorscale": "Viridis",
                    "cmin": frame["temporal_ratio"].min(),
                    "cmax": frame["temporal_ratio"].max(),
                    "size": 10,
                    "line": {"color": color, "width": 1.5},
                    "showscale": index == 0,
                    "colorbar": {"title": "Temporal ratio", "x": 1.02, "y": 0.83, "len": 0.22},
                },
                text=[f"trial {value}" for value in part["trial"]],
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=part["trajectory_ratio"],
                y=part["temporal_ratio"],
                mode="markers",
                name=family,
                legendgroup=family,
                showlegend=False,
                marker={"color": color, "size": 9},
                text=[f"trial {value}" for value in part["trial"]],
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Box(
                x=[family] * len(part),
                y=part["objective"],
                name=family,
                legendgroup=family,
                showlegend=False,
                marker={"color": color},
                boxmean=True,
            ),
            row=2,
            col=2,
        )
    if not importance.empty:
        top = importance.head(12).iloc[::-1]
        figure.add_trace(
            go.Bar(
                x=top["ped_anova"],
                y=top["parameter"],
                orientation="h",
                name="PED-ANOVA",
                marker={"color": "#1565c0"},
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=top["fanova"],
                y=top["parameter"],
                orientation="h",
                name="fANOVA",
                marker={"color": "#ef6c00"},
            ),
            row=3,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=frame["duration_minutes"],
            y=frame["objective"],
            mode="markers",
            name="Trial runtime",
            showlegend=False,
            marker={"color": frame["trial"], "colorscale": "Plasma", "size": 10},
            text=[f"trial {value}" for value in frame["trial"]],
        ),
        row=3,
        col=2,
    )
    figure.update_xaxes(title_text="Trial", row=1, col=1)
    figure.update_yaxes(title_text="Normalized objective", row=1, col=1)
    figure.update_xaxes(title_text="Trajectory ratio", row=1, col=2)
    figure.update_yaxes(title_text="Flexibility ratio", row=1, col=2)
    figure.update_xaxes(title_text="Trajectory ratio", row=2, col=1)
    figure.update_yaxes(title_text="Temperature temporal ratio", row=2, col=1)
    figure.update_yaxes(title_text="Normalized objective", row=2, col=2)
    figure.update_xaxes(title_text="Importance", row=3, col=1)
    figure.update_xaxes(title_text="Trial duration [min]", row=3, col=2)
    figure.update_yaxes(title_text="Normalized objective", row=3, col=2)
    figure.update_layout(
        title=(
            "Deterministic contracting HP model selection"
            f"<br><sup>objective = {weights['trajectory']:.2f} trajectory + "
            f"{weights['temporal']:.2f} temperature dynamics + "
            f"{weights['flexibility']:.2f} intervention response; all terms are "
            "ratios to the baseline</sup>"
        ),
        template="plotly_white",
        height=1450,
        width=1500,
        boxmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 90, "r": 130, "t": 150, "b": 70},
    )
    figure.write_html(path)


def _weight_scenarios(selected: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "accuracy-heavy": {"trajectory": 0.65, "temporal": 0.15, "flexibility": 0.20},
        "selected": selected,
        "flexibility-heavy": {
            "trajectory": 0.35,
            "temporal": 0.20,
            "flexibility": 0.45,
        },
    }


def _write_weight_sensitivity(
    frame: pd.DataFrame,
    path: Path,
    csv_path: Path,
    selected_weights: dict[str, float],
) -> None:
    scenarios = _weight_scenarios(selected_weights)
    sensitivity = frame[
        ["trial", "architecture", "trajectory_ratio", "temporal_ratio", "flexibility_ratio"]
    ].copy()
    for scenario, weights in scenarios.items():
        suffix = scenario.replace("-", "_")
        objective_column = f"objective_{suffix}"
        rank_column = f"rank_{suffix}"
        sensitivity[objective_column] = (
            weights["trajectory"] * sensitivity["trajectory_ratio"]
            + weights["temporal"] * sensitivity["temporal_ratio"]
            + weights["flexibility"] * sensitivity["flexibility_ratio"]
        )
        sensitivity[rank_column] = sensitivity[objective_column].rank(
            method="min", ascending=True
        )
    sensitivity.to_csv(csv_path, index=False)

    selected_order = sensitivity.sort_values("objective_selected")
    objective_columns = [
        f"objective_{scenario.replace('-', '_')}" for scenario in scenarios
    ]
    scenario_labels = list(scenarios)
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Objective under alternative scientific priorities",
            "Rank stability of the 15 best selected-weight trials",
        ),
        vertical_spacing=0.16,
    )
    figure.add_trace(
        go.Heatmap(
            x=scenario_labels,
            y=[f"trial {value}" for value in selected_order["trial"]],
            z=selected_order[objective_columns].to_numpy(),
            colorscale="Viridis",
            colorbar={"title": "Objective", "len": 0.38, "y": 0.81},
            hovertemplate="%{y}<br>%{x}<br>objective=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for _, trial_row in selected_order.head(15).iterrows():
        ranks = [
            trial_row[f"rank_{scenario.replace('-', '_')}"] for scenario in scenarios
        ]
        figure.add_trace(
            go.Scatter(
                x=scenario_labels,
                y=ranks,
                mode="lines+markers",
                name=f"trial {int(trial_row['trial'])}",
                text=[trial_row["architecture"]] * len(scenarios),
                hovertemplate=(
                    "%{fullData.name}<br>%{x}<br>rank=%{y:.0f}<br>%{text}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="Trial (selected-objective order)", row=1, col=1)
    figure.update_yaxes(title_text="Rank (1 is best)", autorange="reversed", row=2, col=1)
    figure.update_layout(
        title=(
            "Objective-weight sensitivity"
            "<br><sup>Each term is a ratio to the baseline; selected weights are "
            f"{selected_weights['trajectory']:.2f} trajectory, "
            f"{selected_weights['temporal']:.2f} temporal, and "
            f"{selected_weights['flexibility']:.2f} flexibility</sup>"
        ),
        template="plotly_white",
        height=1250,
        width=1450,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.16, "x": 0.0},
        margin={"l": 170, "r": 100, "t": 130, "b": 170},
    )
    figure.write_html(path)


def _write_reports(
    study: optuna.Study,
    args: argparse.Namespace,
    *,
    include_importance: bool,
) -> None:
    frame = _completed_frame(study)
    if frame.empty:
        return
    frame.to_csv(args.output_dir / "trials.csv", index=False)
    top = frame.sort_values("objective").head(min(args.report_top_k, len(frame)))
    top.to_csv(args.output_dir / "top_trials.csv", index=False)
    (args.output_dir / "top_trials.json").write_text(
        json.dumps(json.loads(top.to_json(orient="records")), indent=2)
    )
    architecture = (
        frame.groupby("architecture", as_index=False)
        .agg(
            trials=("trial", "count"),
            best_objective=("objective", "min"),
            median_objective=("objective", "median"),
            median_trajectory_ratio=("trajectory_ratio", "median"),
            median_temporal_ratio=("temporal_ratio", "median"),
            median_flexibility_ratio=("flexibility_ratio", "median"),
        )
        .sort_values("best_objective")
    )
    architecture.to_csv(args.output_dir / "architecture_summary.csv", index=False)
    importance = _importance(study) if include_importance else pd.DataFrame()
    if include_importance:
        importance.to_csv(args.output_dir / "parameter_importance.csv", index=False)
    _write_dashboard(
        frame,
        importance,
        args.output_dir / "optuna_study_dashboard.html",
        _objective_weights(args),
    )
    _write_weight_sensitivity(
        frame,
        args.output_dir / "objective_weight_sensitivity.html",
        args.output_dir / "objective_weight_sensitivity.csv",
        _objective_weights(args),
    )
    best = study.best_trial
    best_payload = {
        "trial": best.number,
        "objective": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    (args.output_dir / "best_trial.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True)
    )
    common = _common_params(study)
    if include_importance and len(frame) >= 2 and common:
        try:
            optuna.visualization.plot_slice(study, params=common).write_html(
                args.output_dir / "optuna_parameter_slices.html"
            )
            optuna.visualization.plot_parallel_coordinate(study, params=common).write_html(
                args.output_dir / "optuna_parallel_coordinates.html"
            )
        except (RuntimeError, ValueError):
            pass


def _trial_metrics(trial: FrozenTrial) -> dict[str, float]:
    metrics = {
        name: float(trial.user_attrs.get(f"metric_{name}", math.nan))
        for name in CORE_METRICS
    }
    required = tuple(BASELINE_SCALE_KEYS.values())
    missing = [name for name in required if not np.isfinite(metrics[name])]
    if missing:
        raise FloatingPointError(
            f"Trial {trial.number} has non-finite confirmation metrics: {missing}"
        )
    return metrics


def _baseline_trial(study: optuna.Study) -> FrozenTrial:
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if trial.user_attrs.get("is_baseline"):
            return trial
    raise RuntimeError("No completed baseline trial is available")


def _confirmation_candidates(
    study: optuna.Study,
    top_k: int,
) -> list[tuple[str, FrozenTrial]]:
    baseline = _baseline_trial(study)
    ranked = sorted(
        (
            trial
            for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if not trial.user_attrs.get("is_baseline")
        ),
        key=lambda trial: float(trial.value),
    )[:top_k]
    return [("baseline", baseline)] + [
        (f"trial_{trial.number:04d}", trial) for trial in ranked
    ]


def _confirmation_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys([int(args.seed), *(int(seed) for seed in args.confirmation_seeds)])
    )


def _profile_target_scale(
    artifact_dir: Path,
    dataset: Path,
    profile_ids: list[int],
) -> np.ndarray:
    artifact = load_artifact(artifact_dir)
    profiles = load_profiles(
        artifact,
        dataset_path=dataset,
        profile_ids=profile_ids,
        max_profiles=None,
    )
    target = np.concatenate(
        [profile.targets if hasattr(profile, "targets") else profile.target for profile in profiles],
        axis=0,
    )
    return np.maximum(np.std(target, axis=0, dtype=np.float64), 1e-6)


def _score_fast_metrics(
    artifact_dir: Path,
    *,
    dataset: Path,
    output_dir: Path,
    seed: int,
    max_profiles: int | None,
    profile_ids: list[int] | None = None,
    target_scale: np.ndarray | None = None,
) -> dict[str, float]:
    summary_path = output_dir / "fast_kpi_summary.json"
    if not summary_path.exists():
        evaluate_fast_kpis(
            artifact_dir,
            dataset_path=dataset,
            output_dir=output_dir,
            profile_source="test",
            profile_ids=profile_ids,
            max_profiles=max_profiles,
            num_eval_particles=1,
            target_scale_override=target_scale,
            config=FastKpiConfig(),
            seed=seed,
        )
    return _metric_values(json.loads(summary_path.read_text()))


def _run_confirmation_replica(
    args: argparse.Namespace,
    split: dict[str, list[int]],
    *,
    label: str,
    trial: FrozenTrial,
    seed: int,
) -> tuple[Path, dict[str, float]]:
    if seed == args.seed:
        artifact_dir = Path(str(trial.user_attrs["artifact_dir"]))
        return artifact_dir, _trial_metrics(trial)

    run_dir = args.output_dir / "confirmation" / label / f"seed_{seed}"
    result_path = run_dir / "confirmation_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        artifact_dir = Path(result["artifact_dir"])
        if artifact_dir.exists():
            return artifact_dir, {
                name: float(value) for name, value in result["metrics"].items()
            }

    log_path = run_dir / "training.log"
    artifact_dir: Path | None = None
    if log_path.exists():
        try:
            candidate = _artifact_from_log(log_path.read_text())
            if candidate.exists():
                artifact_dir = candidate
        except ValueError:
            pass
    if artifact_dir is None:
        command = _training_command(
            args,
            dict(trial.params),
            split,
            run_dir,
            seed=seed,
        )
        return_code, wall_seconds = _run_training(
            command,
            log_path,
            verbose=args.verbose_trials,
        )
        if return_code != 0:
            raise RuntimeError(f"Confirmation training failed; see {log_path}")
        artifact_dir = _artifact_from_log(log_path.read_text())
    else:
        wall_seconds = math.nan

    scale = _holdout_scale(artifact_dir, args.dataset)
    metrics = _score_fast_metrics(
        artifact_dir,
        dataset=args.dataset,
        output_dir=run_dir / "validation_fast_kpis",
        seed=seed,
        max_profiles=args.validation_profiles,
        target_scale=scale,
    )
    result_path.write_text(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "metrics": metrics,
                "search_trial": trial.number,
                "seed": seed,
                "wall_minutes": wall_seconds / 60.0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return artifact_dir, metrics


def _aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.columns
        if column not in {"candidate", "artifact_dir", "seed", "search_trial"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for candidate, part in frame.groupby("candidate", sort=False):
        row: dict[str, Any] = {
            "candidate": candidate,
            "search_trial": int(part["search_trial"].iloc[0]),
            "seeds": len(part),
        }
        for column in numeric:
            values = part[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("objective_mean")


def _write_seed_comparison_plot(
    path: Path,
    runs: pd.DataFrame,
    aggregate: pd.DataFrame,
    *,
    title: str,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Paired objective by seed", "Mean normalized trajectory errors"),
        horizontal_spacing=0.14,
    )
    for candidate, part in runs.groupby("candidate", sort=False):
        figure.add_trace(
            go.Scatter(
                x=part["seed"].astype(str),
                y=part["objective"],
                mode="lines+markers",
                name=candidate,
            ),
            row=1,
            col=1,
        )
    colors = {"temperature": "#1769aa", "qroom": "#2e7d32", "pel": "#d32f2f"}
    for channel in ("temperature", "qroom", "pel"):
        figure.add_trace(
            go.Bar(
                x=aggregate["candidate"],
                y=aggregate[f"{channel}_nrmse_mean_mean"],
                error_y={
                    "type": "data",
                    "array": aggregate[f"{channel}_nrmse_mean_std"],
                    "visible": True,
                },
                name=f"{channel} NRMSE",
                marker_color=colors[channel],
            ),
            row=1,
            col=2,
        )
    figure.update_xaxes(title_text="Seed", row=1, col=1)
    figure.update_yaxes(title_text="Objective ratio to same-seed baseline", row=1, col=1)
    figure.update_xaxes(title_text="Configuration", row=1, col=2)
    figure.update_yaxes(title_text="NRMSE", row=1, col=2)
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=720,
        width=1500,
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    figure.write_html(path)


def _run_confirmation(
    args: argparse.Namespace,
    study: optuna.Study,
    split: dict[str, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _confirmation_candidates(study, args.confirmation_top_k)
    rows: list[dict[str, Any]] = []
    weights = _objective_weights(args)
    for seed in _confirmation_seeds(args):
        seed_results: dict[str, tuple[FrozenTrial, Path, dict[str, float]]] = {}
        for label, trial in candidates:
            print(f"confirmation candidate={label} seed={seed}")
            artifact_dir, metrics = _run_confirmation_replica(
                args,
                split,
                label=label,
                trial=trial,
                seed=seed,
            )
            seed_results[label] = (trial, artifact_dir, metrics)
        baseline_metrics = seed_results["baseline"][2]
        scales = {
            name: max(float(baseline_metrics[metric]), 1e-4)
            for name, metric in BASELINE_SCALE_KEYS.items()
        }
        for label, (trial, artifact_dir, metrics) in seed_results.items():
            components = _objective_components(metrics, scales, weights)
            rows.append(
                {
                    "candidate": label,
                    "search_trial": trial.number,
                    "seed": seed,
                    "artifact_dir": str(artifact_dir),
                    **metrics,
                    **components,
                }
            )

    runs = pd.DataFrame(rows)
    aggregate = _aggregate_runs(runs)
    destination = args.output_dir / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    runs.to_csv(destination / "confirmation_runs.csv", index=False)
    aggregate.to_csv(destination / "confirmation_summary.csv", index=False)
    _write_seed_comparison_plot(
        destination / "confirmation_dashboard.html",
        runs,
        aggregate,
        title="Multi-seed validation confirmation",
    )
    winner = aggregate.iloc[0]
    payload = {
        "selection_basis": "lowest mean validation objective across confirmation seeds",
        "winner": str(winner["candidate"]),
        "search_trial": int(winner["search_trial"]),
        "objective_mean": float(winner["objective_mean"]),
        "objective_std": float(winner["objective_std"]),
        "seeds": list(_confirmation_seeds(args)),
    }
    (destination / "confirmed_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return runs, aggregate


def _run_final_test(
    args: argparse.Namespace,
    split: dict[str, list[int]],
    validation_runs: pd.DataFrame,
    validation_aggregate: pd.DataFrame,
) -> None:
    winner = str(validation_aggregate.iloc[0]["candidate"])
    candidates = ["baseline"] if winner == "baseline" else ["baseline", winner]
    selected_runs = validation_runs[validation_runs["candidate"].isin(candidates)].copy()
    final_ids = [int(profile_id) for profile_id in split["final_test_ids"]]
    scale = _profile_target_scale(
        Path(selected_runs.iloc[0]["artifact_dir"]),
        args.dataset,
        final_ids,
    )
    rows: list[dict[str, Any]] = []
    for item in selected_runs.itertuples(index=False):
        output_dir = (
            args.output_dir
            / "final_test"
            / str(item.candidate)
            / f"seed_{int(item.seed)}"
            / "fast_kpis"
        )
        metrics = _score_fast_metrics(
            Path(item.artifact_dir),
            dataset=args.dataset,
            output_dir=output_dir,
            seed=int(item.seed),
            max_profiles=None,
            profile_ids=final_ids,
            target_scale=scale,
        )
        rows.append(
            {
                "candidate": str(item.candidate),
                "search_trial": int(item.search_trial),
                "seed": int(item.seed),
                "artifact_dir": str(item.artifact_dir),
                **metrics,
            }
        )

    runs = pd.DataFrame(rows)
    weights = _objective_weights(args)
    component_rows: list[dict[str, Any]] = []
    for seed, part in runs.groupby("seed"):
        baseline_metrics = part[part["candidate"] == "baseline"].iloc[0]
        scales = {
            name: max(float(baseline_metrics[metric]), 1e-4)
            for name, metric in BASELINE_SCALE_KEYS.items()
        }
        for _, row in part.iterrows():
            metrics = {name: float(row[name]) for name in CORE_METRICS}
            component_rows.append(
                {
                    **row.to_dict(),
                    **_objective_components(metrics, scales, weights),
                }
            )
    runs = pd.DataFrame(component_rows)
    aggregate = _aggregate_runs(runs)
    destination = args.output_dir / "final_test"
    destination.mkdir(parents=True, exist_ok=True)
    runs.to_csv(destination / "final_test_runs.csv", index=False)
    aggregate.to_csv(destination / "final_test_summary.csv", index=False)
    _write_seed_comparison_plot(
        destination / "final_test_comparison.html",
        runs,
        aggregate,
        title="Untouched final-test comparison (not used for model selection)",
    )

    # Use the seed nearest each configuration's mean validation objective for
    # representative annual traces; this avoids cherry-picking its best seed.
    for candidate in candidates:
        part = selected_runs[selected_runs["candidate"] == candidate]
        objective_mean = float(
            validation_aggregate.loc[
                validation_aggregate["candidate"] == candidate,
                "objective_mean",
            ].iloc[0]
        )
        representative = part.iloc[
            np.argmin(np.abs(part["objective"].to_numpy(dtype=float) - objective_mean))
        ]
        trace_dir = destination / "full_year_traces" / candidate
        if not (trace_dir / "summary.json").exists():
            score_artifact(
                Path(representative["artifact_dir"]),
                dataset_path=args.dataset,
                output_dir=trace_dir,
                profile_ids=final_ids,
                max_profiles=None,
                kpi_mode="mean",
                seed=int(representative["seed"]),
                num_profile_plots=args.final_profile_plots,
            )

    payload = {
        "selection_used_final_test": False,
        "confirmed_winner": winner,
        "winner_search_trial": int(validation_aggregate.iloc[0]["search_trial"]),
        "profile_count": len(final_ids),
        "seeds": list(_confirmation_seeds(args)),
    }
    (destination / "final_test_protocol.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def _report_callback(study: optuna.Study, _: FrozenTrial, args: argparse.Namespace) -> None:
    complete_count = len(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    )
    if complete_count > 0 and complete_count % 5 == 0:
        _write_reports(study, args, include_importance=False)


def run_study(args: argparse.Namespace) -> Path:
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = args.output_dir / "study_config.json"
    signature = _study_signature(args)
    if signature_path.exists():
        previous = json.loads(signature_path.read_text())
        if previous != signature:
            raise ValueError(
                "Study-defining arguments differ from the saved study_config.json; "
                "use a new output directory for a different search."
            )
    else:
        signature_path.write_text(json.dumps(signature, indent=2, sort_keys=True))
    split = _prepare_split(args)
    print(
        f"eligible_profiles={split['eligible_profile_count']} "
        f"validation_profiles={len(split['validation_ids'])} "
        f"untouched_final_test_profiles={len(split['final_test_ids'])}"
    )
    print(
        f"objective_weights={_objective_weights(args)} stride={args.stride} "
        f"sequence_length={args.sequence_length} epochs={args.epochs} "
        f"fixed_learning_rate={args.fixed_learning_rate}"
    )
    if args.dry_run:
        return args.output_dir

    study = optuna.create_study(
        study_name=args.study_name,
        storage=_storage(args.output_dir / "optuna_journal.log"),
        sampler=_sampler(args),
        direction="minimize",
        load_if_exists=True,
    )
    if not study.trials:
        baseline_params = dict(BASELINE_PARAMS)
        if args.fixed_learning_rate is not None:
            baseline_params["learning_rate"] = float(args.fixed_learning_rate)
        study.enqueue_trial(
            baseline_params,
            user_attrs={"is_baseline": True},
            skip_if_exists=True,
        )
    queued_architectures = _enqueue_architecture_quotas(
        study,
        args.min_trials_per_architecture,
    )
    if queued_architectures:
        (args.output_dir / "architecture_quota_plan.json").write_text(
            json.dumps(
                {
                    "minimum_trials_per_architecture": args.min_trials_per_architecture,
                    "families": [
                        _architecture_label(dict(family))
                        for family in ARCHITECTURE_FAMILIES
                    ],
                    "queued_trials_excluding_baseline": len(queued_architectures),
                },
                indent=2,
                sort_keys=True,
            )
        )
    _baseline_scales(study)
    finished = sum(
        trial.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
        for trial in study.trials
    )
    remaining = max(0, args.num_trials - finished)
    if remaining:
        timeout = args.timeout_hours * 3600.0 if args.timeout_hours > 0.0 else None
        study.optimize(
            _objective_factory(args, study, split),
            n_trials=remaining,
            timeout=timeout,
            n_jobs=1,
            catch=(RuntimeError, ValueError, FloatingPointError),
            callbacks=[
                lambda current_study, trial: _report_callback(
                    current_study, trial, args
                )
            ],
        )
    _write_reports(study, args, include_importance=True)
    completed = _completed_frame(study)
    print(f"completed_trials={len(completed)}")
    if not completed.empty:
        print(
            f"best_trial={study.best_trial.number} "
            f"best_objective={study.best_value:.6f}"
        )
        top = completed.sort_values("objective").head(
            min(args.report_top_k, len(completed))
        )
        print(
            "top_trials="
            + ",".join(
                f"{int(row.trial)}:{float(row.objective):.6f}"
                for row in top.itertuples(index=False)
            )
        )
    finished = sum(
        trial.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
        for trial in study.trials
    )
    if not args.skip_confirmation and finished >= args.num_trials:
        validation_runs, validation_aggregate = _run_confirmation(args, study, split)
        _run_final_test(args, split, validation_runs, validation_aggregate)
        winner = validation_aggregate.iloc[0]
        print(
            f"confirmed_winner={winner['candidate']} "
            f"validation_objective_mean={winner['objective_mean']:.6f} "
            f"validation_objective_std={winner['objective_std']:.6f}"
        )
    elif not args.skip_confirmation:
        print(
            f"confirmation_deferred finished_trials={finished} "
            f"required_trials={args.num_trials}"
        )
    print(f"saved_optuna_study={args.output_dir}")
    return args.output_dir


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
