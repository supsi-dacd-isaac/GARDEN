"""Resumable Q-to-T response experiment on the saved HP benchmark split.

Run --step 1, then 2, then 3. Steps 4-6 study capped, fixed, and metadata-only alpha.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from neural_building_emulator.data import read_profile_ids
from neural_building_emulator.flexibility_event_study_kpis import (
    EventStudyConfig,
    regression_data_for_trace,
)

from .artifacts import load_artifact
from .config import ExperimentConfig, OptimizerConfig
from .diagnose_rollout_failures import _interval_means
from .prediction import PredictionResult, predict_profile
from .profiles import load_profiles, saved_profile_ids
from .registry import get_model_spec
from .trainer import train

ARM_SETTINGS = {
    "A_leaky_2c": ("leaky_equilibrium", 2.0),
    "B_leaky_3c_diagnostic": ("leaky_equilibrium", 3.0),
    "C_alpha_bounded_2c": ("alpha_bounded_equilibrium", 2.0),
}
COLORS = {
    "A_leaky_2c": "#176c78",
    "B_leaky_3c_diagnostic": "#bf772e",
    "C_alpha_bounded_2c": "#aa3e3e",
    "baseline": "#176c78",
    "observed_Q_to_thermal": "#aa3e3e",
}
HORIZONS_HOURS = (0.5, 1.0)


def _spaced(ids: list[int], count: int) -> list[int]:
    if not ids or count < 1:
        raise ValueError("A nonempty profile pool and positive count are required")
    return [ids[index] for index in np.linspace(
        0, len(ids) - 1, min(count, len(ids)), dtype=int
    )]


def _finite_mean(values) -> float | None:
    numeric = np.asarray(list(values), dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    return float(np.mean(finite)) if finite.size else None


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(error, dtype=np.float64)))))


def _events(profile, steps: int) -> tuple[np.ndarray, np.ndarray]:
    config = EventStudyConfig(
        horizons_hours=(steps * 0.25,),
        horizon_steps=(steps,),
        dt_hours=0.25,
        min_setpoint_change=0.05,
        min_events=0,
        controls="none",
        estimator="direct_ratio",
    )
    data = regression_data_for_trace(
        profile, profile.targets, horizon_steps=steps, config=config
    )
    return data.events, data.deltas


def _responses(values: np.ndarray, events: np.ndarray, steps: int) -> np.ndarray:
    before, after = _interval_means(values, events, steps)
    return after - before


def _temperature_step_metrics(
    temperature_c: np.ndarray,
    initial_temperature_c: float,
    delta_max_c: float,
    *,
    prefix: str,
) -> dict[str, float]:
    values = np.asarray(temperature_c, dtype=np.float64)
    initial = np.full(values.shape[:-1] + (1,), initial_temperature_c, dtype=np.float64)
    absolute_steps = np.abs(
        np.diff(np.concatenate([initial, values], axis=-1), axis=-1)
    )
    finite = absolute_steps[np.isfinite(absolute_steps)]
    if not finite.size or delta_max_c <= 0.0:
        return {
            f"{prefix}_mean_abs_c": float("nan"),
            f"{prefix}_max_abs_c": float("nan"),
            f"{prefix}_fraction_ge_50pct_limit": float("nan"),
            f"{prefix}_fraction_ge_90pct_limit": float("nan"),
            f"{prefix}_fraction_ge_99pct_limit": float("nan"),
        }
    return {
        f"{prefix}_mean_abs_c": float(np.mean(finite)),
        f"{prefix}_max_abs_c": float(np.max(finite)),
        f"{prefix}_fraction_ge_50pct_limit": float(
            np.mean(finite >= 0.50 * delta_max_c)
        ),
        f"{prefix}_fraction_ge_90pct_limit": float(
            np.mean(finite >= 0.90 * delta_max_c)
        ),
        f"{prefix}_fraction_ge_99pct_limit": float(
            np.mean(finite >= 0.99 * delta_max_c)
        ),
    }


def score_prediction(
    profile, result: PredictionResult, *, label: str,
    availability_input_index: int | None = None,
    temperature_delta_max_c: float = 0.0,
) -> list[dict]:
    """Score annual fit and pre/post intervention response without refitting a KPI."""
    truth = np.asarray(result.target)
    mean = np.asarray(result.mean)
    if truth.shape != mean.shape or truth.shape[1] != 3:
        raise ValueError("Closed-loop prediction must contain aligned Tin, Qroom, Pel")
    availability = (
        np.asarray(profile.inputs[:, availability_input_index]) > 0.5
        if availability_input_index is not None else np.zeros(len(truth), dtype=bool)
    )
    annual = {
        "label": label,
        "profile_id": profile.profile_id,
        "horizon_hours": 0.0,
        "event_count": 0,
        "tin_rmse_c": _rmse(mean[:, 0] - truth[:, 0]),
        "qroom_rmse_w_m2": _rmse(mean[:, 1] - truth[:, 1]),
        "pel_rmse_w_m2": _rmse(mean[:, 2] - truth[:, 2]),
        "tin_bias_c": float(np.mean(mean[:, 0] - truth[:, 0])),
        "tin_rmse_sh_available_c": (
            _rmse((mean[:, 0] - truth[:, 0])[availability])
            if np.any(availability) else float("nan")
        ),
        "tin_rmse_sh_unavailable_c": (
            _rmse((mean[:, 0] - truth[:, 0])[~availability])
            if availability_input_index is not None and np.any(~availability)
            else float("nan")
        ),
        "max_abs_tin_step_c": float(np.max(np.abs(np.diff(mean[:, 0])))),
        "tin_response_rmse_c": float("nan"),
        "tin_response_bias_c": float("nan"),
        "true_tin_gain_c_per_k": float("nan"),
        "pred_tin_gain_c_per_k": float("nan"),
        "tin_response_rmse_q_matched_c": float("nan"),
        "q_matched_events": 0,
        "tin_response_rmse_q_matched_heating_c": float("nan"),
        "q_matched_heating_events": 0,
        "tin_response_coverage_90": float("nan"),
    }
    initial_temperature_c = float(np.asarray(profile.initial_temperature).reshape(-1)[0])
    annual.update(_temperature_step_metrics(
        mean[:, 0],
        initial_temperature_c,
        temperature_delta_max_c,
        prefix="tin_step",
    ))
    if result.scenarios is not None:
        annual.update(_temperature_step_metrics(
            np.asarray(result.scenarios)[..., 0],
            initial_temperature_c,
            temperature_delta_max_c,
            prefix="particle_tin_step",
        ))
    rows = [annual]
    for hours in HORIZONS_HOURS:
        steps = int(round(hours / 0.25))
        events, setpoint_change = _events(profile, steps)
        if not events.size:
            continue
        true_t = _responses(truth[:, 0], events, steps)
        pred_t = _responses(mean[:, 0], events, steps)
        true_q = _responses(truth[:, 1], events, steps)
        pred_q = _responses(mean[:, 1], events, steps)
        q_matched = np.abs(pred_q - true_q) <= 5.0
        q_matched_heating = q_matched & (np.abs(true_q) >= 5.0)
        row = dict(annual)
        row.update(
            horizon_hours=hours,
            event_count=int(events.size),
            tin_response_rmse_c=_rmse(pred_t - true_t),
            tin_response_bias_c=float(np.mean(pred_t - true_t)),
            true_tin_gain_c_per_k=float(
                np.sum(true_t * setpoint_change) / np.sum(setpoint_change ** 2)
            ),
            pred_tin_gain_c_per_k=float(
                np.sum(pred_t * setpoint_change) / np.sum(setpoint_change ** 2)
            ),
            tin_response_rmse_q_matched_c=(
                _rmse((pred_t - true_t)[q_matched])
                if np.any(q_matched) else float("nan")
            ),
            q_matched_events=int(np.sum(q_matched)),
            tin_response_rmse_q_matched_heating_c=(
                _rmse((pred_t - true_t)[q_matched_heating])
                if np.any(q_matched_heating) else float("nan")
            ),
            q_matched_heating_events=int(np.sum(q_matched_heating)),
        )
        if result.scenarios is not None:
            sampled = _responses(
                np.asarray(result.scenarios)[:, :, 0], events, steps
            )
            lo, hi = np.quantile(sampled, [0.05, 0.95], axis=0)
            row["tin_response_coverage_90"] = float(
                np.mean((true_t >= lo) & (true_t <= hi))
            )
        rows.append(row)
    return rows


def _summary(rows: pd.DataFrame) -> list[dict]:
    metrics = (
        "tin_rmse_c", "qroom_rmse_w_m2", "pel_rmse_w_m2", "tin_bias_c",
        "tin_rmse_sh_available_c", "tin_rmse_sh_unavailable_c",
        "max_abs_tin_step_c", "tin_response_rmse_c",
        "tin_response_bias_c", "true_tin_gain_c_per_k", "pred_tin_gain_c_per_k",
        "tin_response_rmse_q_matched_c", "tin_response_coverage_90",
        "tin_response_rmse_q_matched_heating_c",
        "tin_step_mean_abs_c", "tin_step_max_abs_c",
        "tin_step_fraction_ge_50pct_limit", "tin_step_fraction_ge_90pct_limit",
        "tin_step_fraction_ge_99pct_limit", "particle_tin_step_mean_abs_c",
        "particle_tin_step_max_abs_c", "particle_tin_step_fraction_ge_50pct_limit",
        "particle_tin_step_fraction_ge_90pct_limit",
        "particle_tin_step_fraction_ge_99pct_limit",
    )
    output = []
    for (label, horizon), group in rows.groupby(["label", "horizon_hours"], sort=False):
        item = {
            "label": label,
            "horizon_hours": float(horizon),
            "profiles": int(group.profile_id.nunique()),
            "events": int(group.event_count.sum()),
            "q_matched_events": int(group.q_matched_events.sum()),
            "q_matched_heating_events": int(group.q_matched_heating_events.sum()),
        }
        item.update({metric: _finite_mean(group[metric]) for metric in metrics})
        output.append(item)
    return output


def _metric(summary: list[dict], label: str, hours: float, metric: str) -> float | None:
    return next(
        (item[metric] for item in summary
         if item["label"] == label and item["horizon_hours"] == hours),
        None,
    )


def _dashboard(path: Path, summary: list[dict], title: str) -> None:
    figure = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Annual Tin RMSE", "0.5 h Tin response RMSE", "1 h Tin response RMSE"),
        horizontal_spacing=0.12,
    )
    labels = list(dict.fromkeys(item["label"] for item in summary))
    for col, (hours, metric) in enumerate((
        (0.0, "tin_rmse_c"), (0.5, "tin_response_rmse_c"),
        (1.0, "tin_response_rmse_c"),
    ), start=1):
        values = [_metric(summary, label, hours, metric) for label in labels]
        figure.add_trace(
            go.Bar(
                x=labels, y=values,
                marker_color=[COLORS.get(label.split(":")[0], "#555") for label in labels],
                hovertemplate="%{x}<br>%{y:.3f} C<extra></extra>",
            ), row=1, col=col,
        )
        figure.update_yaxes(title_text="RMSE [C]", row=1, col=col)
    figure.update_layout(
        title=title, template="plotly_white", height=520, width=1450,
        showlegend=False, margin={"t": 95, "b": 130},
    )
    figure.update_xaxes(tickangle=-22)
    figure.write_html(path)


def _cohort(args: argparse.Namespace, *, create: bool) -> dict:
    cohort_path = args.output_dir / "cohort.json"
    if cohort_path.exists():
        cohort = json.loads(cohort_path.read_text())
        if Path(cohort["benchmark_dir"]).resolve() != args.benchmark_dir.resolve():
            raise ValueError("Existing cohort belongs to a different benchmark directory")
        manifest = json.loads((args.benchmark_dir / "split_manifest.json").read_text())
        if cohort["benchmark_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("The benchmark split changed; use a new experiment output directory")
        return cohort
    if not create:
        raise FileNotFoundError("Run --step 1 first to freeze the pilot cohort")
    manifest = json.loads((args.benchmark_dir / "split_manifest.json").read_text())
    rng = np.random.default_rng(args.seed)
    train_pool = list(manifest["train_ids"])
    validation_pool = list(manifest["validation_ids"])
    train_ids = sorted(int(value) for value in rng.choice(
        train_pool, size=min(args.train_profiles, len(train_pool)), replace=False
    ))
    validation_ids = sorted(int(value) for value in rng.choice(
        validation_pool, size=min(args.validation_profiles, len(validation_pool)), replace=False
    ))
    if set(train_ids) & set(validation_ids) or set(validation_ids) & set(manifest["test_ids"]):
        raise ValueError("The benchmark split is not disjoint")
    cohort = {
        "benchmark_dir": str(args.benchmark_dir.resolve()),
        "dataset_path": str(Path(manifest["dataset_path"]).resolve()),
        "benchmark_fingerprint": manifest["fingerprint"],
        "cohort_seed": args.seed,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "untouched_test_ids": list(manifest["test_ids"]),
        "note": "All three steps use these fixed IDs; test IDs cannot select the model.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(json.dumps(cohort, indent=2))
    return cohort


def step_one(args: argparse.Namespace) -> None:
    cohort = _cohort(args, create=True)
    destination = args.output_dir / "step1_oracle"
    summary_path = destination / "summary.json"
    if summary_path.exists() and not args.rerun:
        print(f"step1_existing={summary_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    paths = json.loads((args.benchmark_dir / "artifact_paths.json").read_text())
    artifact = load_artifact(Path(paths["probabilistic_structured"]))
    input_columns = artifact.legacy_artifact.metadata["input_columns"]
    availability_index = input_columns.index("space_heating_available")
    profile_ids = _spaced(cohort["validation_ids"], args.oracle_profiles)
    profiles = load_profiles(artifact, profile_ids=profile_ids)
    rows = []
    for index, profile in enumerate(profiles):
        key = jax.random.PRNGKey(args.seed + index)
        common = dict(
            key=key, num_scenarios=args.oracle_particles,
            hp_scenario_mode="expected", ventilation_rollout_mode="eplus_rule",
        )
        baseline = predict_profile(artifact, profile, **common)
        oracle = predict_profile(
            artifact, profile,
            thermal_qroom_reference=np.asarray(profile.targets[:, 1]),
            **common,
        )
        rows.extend(score_prediction(
            profile, baseline, label="baseline", availability_input_index=availability_index
        ))
        rows.extend(score_prediction(
            profile, oracle, label="observed_Q_to_thermal",
            availability_input_index=availability_index,
        ))
        print(f"step1 profile={profile.profile_id} ({index + 1}/{len(profiles)})", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "profile_metrics.csv", index=False)
    summary = _summary(frame)
    baseline_event = _metric(summary, "baseline", 0.5, "tin_response_rmse_c")
    oracle_event = _metric(summary, "observed_Q_to_thermal", 0.5, "tin_response_rmse_c")
    result = {
        "scores": summary,
        "profile_ids": profile_ids,
        "num_particles": args.oracle_particles,
        "same_particle_keys": True,
        "oracle_0_5h_response_rmse_ratio": (
            oracle_event / baseline_event
            if baseline_event is not None and oracle_event is not None and baseline_event > 0
            else None
        ),
        "interpretation": (
            "The oracle changes only Q entering thermal forcing and its leaky bank. "
            "Tin feedback can then change predicted HP/Q/P; this is diagnostic, "
            "not an autonomous predictor or a causal estimate."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2))
    _dashboard(destination / "oracle_dashboard.html", summary, "Observed-Q thermal oracle")
    print(f"step1_result={summary_path}", flush=True)


def _recipe(args: argparse.Namespace, kind: str) -> dict:
    source = json.loads((args.benchmark_dir / "resolved_recipe.json").read_text())["recipe"]
    return dict(source[kind])


def _pilot_config(
    args: argparse.Namespace, cohort: dict, arm: str, *, seed: int, probabilistic: bool,
    output_dir: Path, alpha_cap: float = 1.0, fixed_alpha: float = 0.0,
    alpha_conditioning: str = "metadata_xi",
) -> ExperimentConfig:
    kind = "probabilistic_structured" if probabilistic else "deterministic_structured"
    recipe = _recipe(args, kind)
    update, delta_max_c = ARM_SETTINGS[arm]
    recipe.update(
        contracting_temperature_update=update,
        contracting_temperature_delta_max_c=delta_max_c,
        contracting_temperature_alpha_cap=alpha_cap,
        contracting_temperature_fixed_alpha=fixed_alpha,
        contracting_temperature_alpha_conditioning=alpha_conditioning,
        num_full_profile_plots=0,
        num_window_plots=0,
        log_update_diagnostics=True,
        save_model_every_epochs=1,
    )
    if probabilistic:
        recipe.update(
            prob_particles=args.prob_particles,
            prob_eval_particles=args.prob_eval_particles,
            prob_plot_particles=args.prob_eval_particles,
        )
    selected = set(cohort["train_ids"]) | set(cohort["validation_ids"])
    excluded = tuple(sorted(set(read_profile_ids(Path(cohort["dataset_path"]))) - selected))
    base = json.loads((args.benchmark_dir / "resolved_recipe.json").read_text())["recipe"]
    optimizer = base["optimizer"]
    return ExperimentConfig(
        model_name=(
            "closed_loop_hp_contracting_probabilistic"
            if probabilistic else "closed_loop_hp_contracting_deterministic"
        ),
        dataset_path=Path(cohort["dataset_path"]),
        output_dir=output_dir,
        max_profiles=len(selected),
        test_fraction=len(cohort["validation_ids"]) / len(selected),
        fixed_test_profile_ids=tuple(cohort["validation_ids"]),
        excluded_profile_ids=excluded,
        seed=seed,
        sequence_length=int(base["sequence_length"]),
        stride=int(base["stride"]),
        rotate_window_starts=bool(base["rotate_window_starts"]),
        hp_power_area_normalization=str(base["hp_power_area_normalization"]),
        optimizer=OptimizerConfig(
            epochs=args.prob_epochs if probabilistic else args.det_epochs,
            batch_size=int(optimizer["batch_size"]),
            learning_rate=float(optimizer["learning_rate"]),
            gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
            train_eval_max_windows=int(optimizer["train_eval_max_windows"]),
        ),
        legacy_overrides=recipe,
    )


def _train_or_load(config: ExperimentConfig, *, rerun: bool, expected_train: list[int], expected_validation: list[int]):
    kind = get_model_spec(config.model_name).legacy_model_kind
    selected_path = config.output_dir / "artifacts" / kind / "selected"
    if selected_path.exists() and not rerun:
        artifact_path = selected_path
        print(f"reusing_artifact={artifact_path}", flush=True)
    else:
        artifact_path = train(config)
        print(f"trained_artifact={artifact_path}", flush=True)
    artifact = load_artifact(artifact_path)
    if set(saved_profile_ids(artifact, "train")) != set(expected_train):
        raise ValueError(f"{kind}: trained profile IDs differ from the frozen pilot cohort")
    if set(saved_profile_ids(artifact, "test")) != set(expected_validation):
        raise ValueError(f"{kind}: validation IDs differ from the frozen pilot cohort")
    return artifact_path, artifact


def _gradient_diagnostic(run_dir: Path, model_kind: str) -> dict:
    files = sorted((run_dir / "artifacts" / model_kind).glob("epoch_diagnostics_*.csv"))
    if not files:
        return {"max_raw_grad_norm": None, "skipped_update_batches": None, "epoch_minutes": None}
    rows = pd.read_csv(files[-1])
    result = {
        "max_raw_grad_norm": _finite_mean([pd.to_numeric(rows.get("max_grad_norm"), errors="coerce").max()]),
        "skipped_update_batches": int(pd.to_numeric(
            rows.get("skipped_update_batches"), errors="coerce"
        ).fillna(0).sum()),
        "epoch_minutes": _finite_mean(pd.to_numeric(
            rows.get("epoch_elapsed_min"), errors="coerce"
        )),
        "diagnostics_csv": str(files[-1]),
    }
    for column in rows.columns:
        if column.startswith("max_grad_"):
            result[column] = _finite_mean(
                [pd.to_numeric(rows[column], errors="coerce").max()]
            )
    for split in ("train", "test"):
        for name in (
            "temperature_step_mean_abs_c",
            "temperature_step_max_abs_c",
            "temperature_step_max_fraction_of_limit",
            "temperature_step_fraction_ge_50pct_limit",
            "temperature_step_fraction_ge_90pct_limit",
            "temperature_step_fraction_ge_99pct_limit",
            "particle_temperature_step_mean_abs_c",
            "particle_temperature_step_max_abs_c",
            "particle_temperature_step_max_fraction_of_limit",
            "particle_temperature_step_fraction_ge_50pct_limit",
            "particle_temperature_step_fraction_ge_90pct_limit",
            "particle_temperature_step_fraction_ge_99pct_limit",
        ):
            column = f"{split}_{name}"
            if column in rows:
                result[f"final_epoch_{column}"] = float(
                    pd.to_numeric(rows[column], errors="coerce").iloc[-1]
                )
    return result


def _evaluate_arm(args, artifact, ids: list[int], label: str, seed: int, *, probabilistic: bool) -> list[dict]:
    profiles = load_profiles(artifact, profile_ids=ids)
    input_columns = artifact.legacy_artifact.metadata["input_columns"]
    availability_index = input_columns.index("space_heating_available")
    output = []
    for index, profile in enumerate(profiles):
        result = predict_profile(
            artifact, profile,
            key=jax.random.PRNGKey(seed + index),
            num_scenarios=args.prob_score_scenarios if probabilistic else 0,
            hp_scenario_mode="bernoulli", ventilation_rollout_mode="eplus_rule",
        )
        output.extend(score_prediction(
            profile,
            result,
            label=label,
            availability_input_index=availability_index,
            temperature_delta_max_c=float(
                getattr(artifact.model, "temperature_delta_max_c", 0.0)
            ),
        ))
        print(f"scoring {label} profile={profile.profile_id} ({index + 1}/{len(profiles)})", flush=True)
    return output


def _promote(summary: list[dict], diagnostics: list[dict]) -> tuple[bool, dict]:
    a_t = _metric(summary, "A_leaky_2c", 0.0, "tin_rmse_c")
    c_t = _metric(summary, "C_alpha_bounded_2c", 0.0, "tin_rmse_c")
    a_response = _metric(summary, "A_leaky_2c", 0.5, "tin_response_rmse_c")
    c_response = _metric(summary, "C_alpha_bounded_2c", 0.5, "tin_response_rmse_c")
    a_q = _metric(summary, "A_leaky_2c", 0.0, "qroom_rmse_w_m2")
    c_q = _metric(summary, "C_alpha_bounded_2c", 0.0, "qroom_rmse_w_m2")
    a_p = _metric(summary, "A_leaky_2c", 0.0, "pel_rmse_w_m2")
    c_p = _metric(summary, "C_alpha_bounded_2c", 0.0, "pel_rmse_w_m2")
    a_skip = sum(int(item["skipped_update_batches"] or 0) for item in diagnostics if item["arm"] == "A_leaky_2c")
    c_skip = sum(int(item["skipped_update_batches"] or 0) for item in diagnostics if item["arm"] == "C_alpha_bounded_2c")
    tests = {
        "0_5h_Tin_response_improves_at_least_10_percent": (
            a_response is not None and c_response is not None and c_response <= 0.9 * a_response
        ),
        "annual_Tin_RMSE_worsens_at_most_5_percent": (
            a_t is not None and c_t is not None and c_t <= 1.05 * a_t
        ),
        "Qroom_RMSE_worsens_at_most_10_percent": (
            a_q is not None and c_q is not None and c_q <= 1.1 * a_q
        ),
        "Pel_RMSE_worsens_at_most_10_percent": (
            a_p is not None and c_p is not None and c_p <= 1.1 * a_p
        ),
        "no_more_skipped_updates": c_skip <= a_skip,
    }
    return all(tests.values()), tests


def step_two(args: argparse.Namespace) -> None:
    cohort = _cohort(args, create=False)
    if not (args.output_dir / "step1_oracle" / "summary.json").exists():
        raise FileNotFoundError("Run --step 1 before the deterministic ablation")
    destination = args.output_dir / "step2_deterministic"
    summary_path = destination / "summary.json"
    if summary_path.exists() and not args.rerun:
        print(f"step2_existing={summary_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    ids = _spaced(cohort["validation_ids"], args.score_profiles)
    rows, diagnostics = [], []
    for arm in ARM_SETTINGS:
        for seed in args.seeds:
            run_dir = destination / arm / f"seed_{seed}"
            config = _pilot_config(args, cohort, arm, seed=seed, probabilistic=False, output_dir=run_dir)
            artifact_path, artifact = _train_or_load(
                config, rerun=args.rerun,
                expected_train=cohort["train_ids"],
                expected_validation=cohort["validation_ids"],
            )
            rows.extend(_evaluate_arm(args, artifact, ids, arm, seed, probabilistic=False))
            diagnostics.append({
                "arm": arm, "seed": seed, "artifact": str(artifact_path),
                **_gradient_diagnostic(run_dir, get_model_spec(config.model_name).legacy_model_kind),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "validation_profile_metrics.csv", index=False)
    summary = _summary(frame)
    promoted, tests = _promote(summary, diagnostics)
    result = {
        "scores": summary,
        "diagnostics": diagnostics,
        "promotion_tests": tests,
        "promote_C_to_probabilistic": promoted,
        "B_is_diagnostic_only": True,
        "score_profile_ids": ids,
        "test_profile_ids_used": [],
        "note": "C is selected solely on fixed validation buildings. B relaxes the 2 C step cap and cannot be promoted.",
    }
    summary_path.write_text(json.dumps(result, indent=2))
    _dashboard(destination / "deterministic_dashboard.html", summary, "Q-to-T deterministic response ablation")
    print(f"step2_result={summary_path} promote_C={promoted}", flush=True)


def step_three(args: argparse.Namespace) -> None:
    cohort = _cohort(args, create=False)
    selection_path = args.output_dir / "step2_deterministic" / "summary.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run --step 2 before probabilistic replication")
    selection = json.loads(selection_path.read_text())
    if not selection["promote_C_to_probabilistic"]:
        print("step3_skipped=C did not clear the deterministic validation thresholds", flush=True)
        return
    destination = args.output_dir / "step3_probabilistic"
    summary_path = destination / "summary.json"
    if summary_path.exists() and not args.rerun:
        print(f"step3_existing={summary_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    validation_ids = _spaced(cohort["validation_ids"], args.score_profiles)
    test_ids = _spaced(cohort["untouched_test_ids"], args.final_test_profiles)
    rows, diagnostics = [], []
    for arm in ("A_leaky_2c", "C_alpha_bounded_2c"):
        run_dir = destination / arm
        config = _pilot_config(
            args, cohort, arm, seed=args.seeds[0], probabilistic=True, output_dir=run_dir
        )
        artifact_path, artifact = _train_or_load(
            config, rerun=args.rerun,
            expected_train=cohort["train_ids"],
            expected_validation=cohort["validation_ids"],
        )
        for source, ids in (("validation", validation_ids), ("untouched_test", test_ids)):
            scored = _evaluate_arm(
                args, artifact, ids, f"{arm}:{source}", args.seed,
                probabilistic=True,
            )
            for row in scored:
                row["source"] = source
            rows.extend(scored)
        diagnostics.append({
            "arm": arm, "artifact": str(artifact_path),
            **_gradient_diagnostic(run_dir, get_model_spec(config.model_name).legacy_model_kind),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "profile_metrics.csv", index=False)
    summary = _summary(frame)
    result = {
        "scores": summary,
        "diagnostics": diagnostics,
        "validation_ids": validation_ids,
        "untouched_test_ids": test_ids,
        "scenario_particles": args.prob_score_scenarios,
        "note": (
            "Both probabilistic arms are fitted to the same frozen pilot cohort. "
            "These few-particle annual coverage estimates are diagnostics, not a final calibration claim."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2))
    _dashboard(destination / "probabilistic_dashboard.html", summary, "Matched probabilistic replication")
    print(f"step3_result={summary_path}", flush=True)


def step_four(args: argparse.Namespace) -> None:
    """Train capped C and compare it with the existing probabilistic A/C runs."""
    cohort = _cohort(args, create=False)
    previous_dir = args.output_dir / "step3_probabilistic"
    previous_summary = previous_dir / "summary.json"
    previous_metrics = previous_dir / "profile_metrics.csv"
    if not previous_summary.exists() or not previous_metrics.exists():
        raise FileNotFoundError("Run --step 3 before the capped-alpha follow-up")
    destination = args.output_dir / f"step4_alpha_cap_{str(args.alpha_cap).replace('.', 'p')}"
    summary_path = destination / "summary.json"
    if summary_path.exists() and not args.rerun:
        print(f"step4_existing={summary_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    validation_ids = _spaced(cohort["validation_ids"], args.score_profiles)
    test_ids = _spaced(cohort["untouched_test_ids"], args.final_test_profiles)
    label = f"C_alpha_cap_{args.alpha_cap:g}"
    run_dir = destination / label
    config = _pilot_config(
        args,
        cohort,
        "C_alpha_bounded_2c",
        seed=args.seeds[0],
        probabilistic=True,
        output_dir=run_dir,
        alpha_cap=args.alpha_cap,
    )
    artifact_path, artifact = _train_or_load(
        config,
        rerun=args.rerun,
        expected_train=cohort["train_ids"],
        expected_validation=cohort["validation_ids"],
    )
    capped_rows = []
    for source, ids in (("validation", validation_ids), ("untouched_test", test_ids)):
        scored = _evaluate_arm(
            args, artifact, ids, f"{label}:{source}", args.seed,
            probabilistic=True,
        )
        for row in scored:
            row["source"] = source
        capped_rows.extend(scored)
    previous = pd.read_csv(previous_metrics)
    combined = pd.concat([previous, pd.DataFrame(capped_rows)], ignore_index=True)
    combined.to_csv(destination / "profile_metrics.csv", index=False)
    summary = _summary(combined)
    diagnostics = [{
        "arm": label,
        "artifact": str(artifact_path),
        "alpha_cap": args.alpha_cap,
        **_gradient_diagnostic(
            run_dir, get_model_spec(config.model_name).legacy_model_kind
        ),
    }]
    result = {
        "scores": summary,
        "diagnostics": diagnostics,
        "comparison_source": str(previous_summary),
        "alpha_cap": args.alpha_cap,
        "validation_ids": validation_ids,
        "untouched_test_ids": test_ids,
        "scenario_particles": args.prob_score_scenarios,
        "note": (
            "A and uncapped C rows are reused from step 3. Capped C uses the same "
            "cohort, seed, recipe and evaluation keys; only its structural alpha cap changes."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2))
    _dashboard(
        destination / "alpha_cap_dashboard.html",
        summary,
        f"Probabilistic alpha-cap follow-up (cap={args.alpha_cap:g})",
    )
    print(f"step4_result={summary_path}", flush=True)


def step_five(args: argparse.Namespace) -> None:
    """Train C with fixed alpha and compare against all preceding matched arms."""
    cohort = _cohort(args, create=False)
    previous_dir = args.output_dir / f"step4_alpha_cap_{str(args.alpha_cap).replace('.', 'p')}"
    previous_summary = previous_dir / "summary.json"
    previous_metrics = previous_dir / "profile_metrics.csv"
    if not previous_summary.exists() or not previous_metrics.exists():
        raise FileNotFoundError("Run --step 4 with the same --alpha-cap before step 5")
    fixed_tag = str(args.fixed_alpha).replace(".", "p")
    destination = args.output_dir / f"step5_fixed_alpha_{fixed_tag}"
    summary_path = destination / "summary.json"
    if summary_path.exists() and not args.rerun:
        print(f"step5_existing={summary_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    validation_ids = _spaced(cohort["validation_ids"], args.score_profiles)
    test_ids = _spaced(cohort["untouched_test_ids"], args.final_test_profiles)
    label = f"C_fixed_alpha_{args.fixed_alpha:g}"
    run_dir = destination / label
    config = _pilot_config(
        args,
        cohort,
        "C_alpha_bounded_2c",
        seed=args.seeds[0],
        probabilistic=True,
        output_dir=run_dir,
        alpha_cap=args.alpha_cap,
        fixed_alpha=args.fixed_alpha,
    )
    artifact_path, artifact = _train_or_load(
        config,
        rerun=args.rerun,
        expected_train=cohort["train_ids"],
        expected_validation=cohort["validation_ids"],
    )
    fixed_rows = []
    for source, ids in (("validation", validation_ids), ("untouched_test", test_ids)):
        scored = _evaluate_arm(
            args, artifact, ids, f"{label}:{source}", args.seed,
            probabilistic=True,
        )
        for row in scored:
            row["source"] = source
        fixed_rows.extend(scored)
    previous = pd.read_csv(previous_metrics)
    combined = pd.concat([previous, pd.DataFrame(fixed_rows)], ignore_index=True)
    combined.to_csv(destination / "profile_metrics.csv", index=False)
    summary = _summary(combined)
    diagnostics = [{
        "arm": label,
        "artifact": str(artifact_path),
        "alpha_cap": args.alpha_cap,
        "fixed_alpha": args.fixed_alpha,
        **_gradient_diagnostic(
            run_dir, get_model_spec(config.model_name).legacy_model_kind
        ),
    }]
    result = {
        "scores": summary,
        "diagnostics": diagnostics,
        "comparison_source": str(previous_summary),
        "alpha_cap": args.alpha_cap,
        "fixed_alpha": args.fixed_alpha,
        "validation_ids": validation_ids,
        "untouched_test_ids": test_ids,
        "scenario_particles": args.prob_score_scenarios,
        "note": (
            "Earlier rows are reused from step 4. Fixed-alpha C uses the same cohort, "
            "seed, recipe and evaluation keys; its alpha network is structurally absent."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2))
    _dashboard(
        destination / "fixed_alpha_dashboard.html",
        summary,
        f"Probabilistic fixed-alpha follow-up (alpha={args.fixed_alpha:g})",
    )
    print(f"step5_result={summary_path}", flush=True)


def step_six(args: argparse.Namespace) -> None:
    """Train C with metadata-conditioned alpha and no particle-level alpha variation."""
    cohort = _cohort(args, create=False)
    previous_dir = args.output_dir / f"step4_alpha_cap_{str(args.alpha_cap).replace('.', 'p')}"
    previous_summary = previous_dir / "summary.json"
    previous_metrics = previous_dir / "profile_metrics.csv"
    if not previous_summary.exists() or not previous_metrics.exists():
        raise FileNotFoundError("Run --step 4 with the same --alpha-cap before step 6")
    destination = args.output_dir / "step6_metadata_alpha"
    summary_path = destination / "summary.json"
    if summary_path.exists() and not args.rerun:
        print(f"step6_existing={summary_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    validation_ids = _spaced(cohort["validation_ids"], args.score_profiles)
    test_ids = _spaced(cohort["untouched_test_ids"], args.final_test_profiles)
    label = "C_alpha_metadata"
    run_dir = destination / label
    config = _pilot_config(
        args,
        cohort,
        "C_alpha_bounded_2c",
        seed=args.seeds[0],
        probabilistic=True,
        output_dir=run_dir,
        alpha_cap=args.alpha_cap,
        alpha_conditioning="metadata",
    )
    artifact_path, artifact = _train_or_load(
        config,
        rerun=args.rerun,
        expected_train=cohort["train_ids"],
        expected_validation=cohort["validation_ids"],
    )
    metadata_rows = []
    for source, ids in (("validation", validation_ids), ("untouched_test", test_ids)):
        scored = _evaluate_arm(
            args, artifact, ids, f"{label}:{source}", args.seed,
            probabilistic=True,
        )
        for row in scored:
            row["source"] = source
        metadata_rows.extend(scored)
    previous = pd.read_csv(previous_metrics)
    combined = pd.concat([previous, pd.DataFrame(metadata_rows)], ignore_index=True)
    combined.to_csv(destination / "profile_metrics.csv", index=False)
    summary = _summary(combined)
    diagnostics = [{
        "arm": label,
        "artifact": str(artifact_path),
        "alpha_cap": args.alpha_cap,
        "alpha_conditioning": "metadata",
        **_gradient_diagnostic(
            run_dir, get_model_spec(config.model_name).legacy_model_kind
        ),
    }]
    result = {
        "scores": summary,
        "diagnostics": diagnostics,
        "comparison_source": str(previous_summary),
        "alpha_cap": args.alpha_cap,
        "alpha_conditioning": "metadata",
        "validation_ids": validation_ids,
        "untouched_test_ids": test_ids,
        "scenario_particles": args.prob_score_scenarios,
        "note": (
            "Earlier rows are reused from step 4. Metadata-alpha C uses the same "
            "cohort, seed, recipe and evaluation keys; alpha is learned per building "
            "from metadata but is identical across that building's particles."
        ),
    }
    summary_path.write_text(json.dumps(result, indent=2))
    _dashboard(
        destination / "metadata_alpha_dashboard.html",
        summary,
        "Probabilistic metadata-conditioned alpha follow-up",
    )
    print(f"step6_result={summary_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(1, 2, 3, 4, 5, 6), required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(
        "output/neural_building_emulator_refactor/calibrated_four_model_benchmark"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "output/neural_building_emulator_refactor/q_to_t_response_experiments"
    ))
    parser.add_argument("--train-profiles", type=int, default=100)
    parser.add_argument("--validation-profiles", type=int, default=12)
    parser.add_argument("--oracle-profiles", type=int, default=6)
    parser.add_argument("--oracle-particles", type=int, default=8)
    parser.add_argument("--score-profiles", type=int, default=6)
    parser.add_argument("--final-test-profiles", type=int, default=6)
    parser.add_argument("--det-epochs", type=int, default=6)
    parser.add_argument("--prob-epochs", type=int, default=6)
    parser.add_argument("--prob-particles", type=int, default=8)
    parser.add_argument("--prob-eval-particles", type=int, default=8)
    parser.add_argument("--prob-score-scenarios", type=int, default=16)
    parser.add_argument("--alpha-cap", type=float, default=0.5)
    parser.add_argument("--fixed-alpha", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=(13, 17))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "train_profiles", "validation_profiles", "oracle_profiles", "oracle_particles",
        "score_profiles", "final_test_profiles", "det_epochs", "prob_epochs",
        "prob_particles", "prob_eval_particles", "prob_score_scenarios",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.seeds:
        parser.error("--seeds requires at least one seed")
    if not 0.0 < args.alpha_cap <= 1.0:
        parser.error("--alpha-cap must be in (0, 1]")
    if not 0.0 < args.fixed_alpha <= args.alpha_cap:
        parser.error("--fixed-alpha must be in (0, --alpha-cap]")
    return args


def main() -> None:
    args = parse_args()
    {
        1: step_one,
        2: step_two,
        3: step_three,
        4: step_four,
        5: step_five,
        6: step_six,
    }[args.step](args)


if __name__ == "__main__":
    main()
