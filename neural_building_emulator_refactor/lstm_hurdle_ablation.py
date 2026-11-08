"""Matched LSTM HP-head ablation with an optional structured comparator."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .artifacts import json_safe, load_artifact
from .calibrated_model_benchmark import (
    _eligible_hp_ids, _emulator_distribution, _force_off_emulator_distribution,
)
from .fair_probabilistic_lstm_comparison import _experiment_configs
from .hp_flex_loss_ablation import (
    DEFAULT_CONFIG,
    STAGES,
    _annual_rows,
    _artifact_record_path,
    _flexibility_tables,
    _force_off_tables,
    _load_artifact_record,
    _macro_flexibility,
    _rank_components,
    _resolved_base,
    _score_force_off_variant,
    _score_variant,
    _shared_split,
    _trace_rows,
)
from .profiles import saved_profile_ids
from .config import ProbabilisticLSTMConfig
from .trainer import train


DEFAULT_OUTPUT_DIR = Path(
    "output/neural_building_emulator_refactor/lstm_hurdle_ablation_200"
)
DEFAULT_VARIANTS = ("continuous", "hurdle_iid", "hurdle_markov")
VARIANT_ORDER = ("structured", *DEFAULT_VARIANTS)
VARIANT_MODELS = {
    "structured": "closed_loop_hp_contracting_probabilistic",
    "continuous": "closed_loop_hp_vanilla_probabilistic_lstm",
    "hurdle_iid": "closed_loop_hp_hurdle_iid_probabilistic_lstm",
    "hurdle_markov": "closed_loop_hp_hurdle_markov_probabilistic_lstm",
}
VARIANT_LABELS = {
    "structured": "Structured",
    "continuous": "Continuous LSTM",
    "hurdle_iid": "LSTM + IID hurdle HP",
    "hurdle_markov": "LSTM + Markov hurdle HP",
}
VARIANT_COLORS = {
    "structured": "#1769aa",
    "continuous": "#9b4fa6",
    "hurdle_iid": "#d97706",
    "hurdle_markov": "#16846b",
}


def _variant_recipe(
    base: dict[str, Any],
    variant: str,
    *,
    hurdle_mode_loss_weight: float = 0.0,
    hurdle_active_power_nll_weight: float = 0.0,
) -> dict[str, Any]:
    recipe = json.loads(json.dumps(json_safe(base)))
    settings = recipe["probabilistic_lstm"]
    # Keep the default ablation loss-identical. Optional direct supervision is
    # exposed explicitly for a separate experiment rather than silently folded
    # into the architecture comparison.
    if variant not in VARIANT_MODELS:
        raise ValueError(f"Unknown variant: {variant}")
    hurdle = variant in ("hurdle_iid", "hurdle_markov")
    if recipe.get("comparison_protocol") == "current_best_shared_losses":
        if not hurdle:
            for name in ("hp_mode_loss_weight", "hp_active_power_nll_weight", "hp_inactive_leakage_weight"):
                settings[name] = 0.0
        return recipe
    settings["hp_mode_loss_weight"] = (
        float(hurdle_mode_loss_weight) if hurdle else 0.0
    )
    settings["hp_active_power_nll_weight"] = (
        float(hurdle_active_power_nll_weight) if hurdle else 0.0
    )
    settings["hp_inactive_leakage_weight"] = 0.0
    settings["hp_training_mode"] = "expected"
    settings["hp_scenario_mode"] = "bernoulli"
    settings["setpoint_shock_timescales_hours"] = []
    return recipe


def _require_same(path: Path, content: Any) -> None:
    content = json.loads(json.dumps(json_safe(content)))
    if path.exists() and json.loads(path.read_text()) != content:
        raise ValueError(f"Provenance changed: {path}; use a fresh output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def _fingerprints(root: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.iterdir()) if p.suffix in (".json", ".eqx", ".npz")
    }


def _record_implementation(root: Path, hashes: dict[str, str], *, report_only: bool) -> None:
    path = root / "implementation_hashes.json"
    if not report_only:
        _require_same(path, hashes)
        return
    frozen = json.loads(path.read_text())
    changed = {key for key in frozen.keys() | hashes.keys() if frozen.get(key) != hashes.get(key)}
    # A presentation fix may rebuild reports, but never retrain or rescore a
    # frozen experiment under changed model, data, or scoring implementations.
    if changed - {"neural_building_emulator_refactor/lstm_hurdle_ablation.py"}:
        raise ValueError("Scientific implementation changed; cannot rebuild the frozen comparison")
    (root / "report_implementation_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )


def _gradient_diagnostics(artifact_dir: Path) -> tuple[float, float]:
    gradient_files = sorted(artifact_dir.parent.glob("gradient_updates*.csv"))
    if gradient_files:
        gradients = pd.read_csv(gradient_files[-1])
        column = next((name for name in ("grad_norm_pre_clip", "raw_grad_norm", "grad_norm")
                       if name in gradients), None)
        maximum = float(gradients[column].max()) if column is not None else np.nan
        skipped = float((~gradients["accepted"].astype(bool)).sum()) if "accepted" in gradients else np.nan
        return maximum, skipped
    epoch_files = sorted(artifact_dir.parent.glob("epoch_diagnostics*.csv"))
    if epoch_files:
        epochs = pd.read_csv(epoch_files[-1])
        return float(epochs["max_grad_norm"].max()), float(epochs["skipped_update_batches"].sum())
    return np.nan, np.nan


def _validate_shared_objective(recipe: dict[str, Any]) -> None:
    structured = recipe["probabilistic_structured"]
    lstm = recipe["probabilistic_lstm"]
    shared = [
        "particles", "eval_particles", "latent_dim", "process_noise",
        "process_noise_init", "hp_training_mode", "hp_scenario_mode",
        "increment_energy_weight", "increment_energy_lags",
        "increment_energy_lags_per_batch", "increment_energy_output_weights",
        "increment_energy_constant_setpoint_only",
        "increment_energy_setpoint_threshold_c", "event_energy_weight",
        "event_energy_horizons_hours", "event_energy_events_per_profile",
        "event_energy_output_weights", "event_energy_setpoint_threshold_c",
        "softopt_weight", "variogram_weight",
    ]
    for key in shared:
        if structured.get("prob_" + key) != lstm.get(key):
            raise ValueError(f"Unmatched structured/LSTM setting: {key}")
    for key in ("bptt_truncate_steps", "skip_grad_norm_above"):
        if structured.get(key) != lstm.get(key):
            raise ValueError(f"Unmatched structured/LSTM setting: {key}")
    if structured["closed_loop_trajectory_output_weights"] != lstm["output_weights"]:
        raise ValueError("Unmatched trajectory output weights")
    for key in ("hp_mode_loss_weight", "hp_active_power_nll_weight", "hp_inactive_leakage_weight"):
        if structured.get(key, 0.) != 0. or lstm.get(key, 0.) != 0.:
            raise ValueError(f"Auxiliary supervision is not a matched architecture ablation: {key}")
    for key in ("qroom_mode_loss_weight", "prob_physics_weight", "prob_ires_weight",
                "prob_flex_kpi_crps_weight", "closed_loop_stability_weight",
                "prob_block_energy_weight", "prob_targeted_flex_energy_weight"):
        if structured.get(key, 0.) != 0.:
            raise ValueError(f"Structured-only objective is enabled: {key}")
    if structured.get("prob_event_energy_include_force_off", False):
        raise ValueError("Force-off event features are not implemented in the LSTM objective")


def _current_best_base(args) -> dict[str, Any]:
    from .routed_latent_targeted_loss_ablation import _load_context

    recipe, _, artifact_dir = _load_context(args.current_best_root)
    metadata_path = artifact_dir / "metadata_closed_loop_hp_contracting_probabilistic.json"
    settings = json.loads(metadata_path.read_text())["train_config"]
    if settings["prob_hp_activation_model"] != "asymmetric_markov":
        raise ValueError("This benchmark requires the selected asymmetric-Markov model")
    if settings["hp_energy_cap_hours"] != 0.5:
        raise ValueError("The current-best comparator must be the 0.5-hour buffer model")
    if settings["prob_flex_kpi_crps_estimator"] != "direct_ratio":
        raise ValueError("The matched LSTM KPI objective supports direct-F only")
    if settings["prob_flex_kpi_crps_aux_every_steps"]:
        raise ValueError("Auxiliary KPI updates are not matched by this protocol")
    for key in ("prob_increment_energy_weight", "prob_event_energy_weight",
                "prob_block_energy_weight", "prob_targeted_flex_energy_weight", "prob_ires_weight",
                "closed_loop_stability_weight"):
        if settings.get(key, 0.0):
            raise ValueError(f"Unexpected additional objective in current-best artifact: {key}")
    # Dataset, split, budget and destinations belong to ExperimentConfig. Do not
    # carry the source run's excluded IDs or output paths into the new recipe.
    runtime_fields = {
        "dataset_path", "additional_dataset_paths", "max_profiles", "test_fraction",
        "fixed_test_profile_ids", "excluded_profile_ids", "seed", "sequence_length", "stride",
        "rotate_window_starts", "batch_size", "epochs", "learning_rate", "gradient_clip_norm",
        "early_stopping_patience", "early_stopping_min_delta", "train_eval_max_windows",
        "output_dir", "model_checkpoint_dir", "save_model", "model_kind",
    }
    recipe["probabilistic_structured"] = {k: v for k, v in settings.items() if k not in runtime_fields}
    for name in ("dataset_path", "additional_dataset_paths", "sequence_length", "stride",
                 "rotate_window_starts", "seed", "hp_power_area_normalization"):
        recipe[name] = settings[name]
    for name in recipe["optimizer"]:
        recipe["optimizer"][name] = settings[name]
    lstm = recipe["probabilistic_lstm"]
    for field in fields(ProbabilisticLSTMConfig):
        if "prob_" + field.name in settings:
            lstm[field.name] = settings["prob_" + field.name]
    for name in ("bptt_truncate_steps", "skip_grad_norm_above", "hp_mode_loss_weight",
                 "hp_active_power_nll_weight", "hp_inactive_leakage_weight"):
        lstm[name] = settings[name]
    lstm["output_weights"] = settings["closed_loop_trajectory_output_weights"]
    lstm["process_regularization_weight"] = settings["prob_physics_weight"]
    lstm["setpoint_shock_timescales_hours"] = []
    recipe["comparison_protocol"] = "current_best_shared_losses"
    recipe["configuration_source"] = {
        "artifact_dir": str(artifact_dir.resolve()), "hashes": _fingerprints(artifact_dir),
        "scope": "Retrain the selected architecture and recipe, not its checkpoint weights. "
                 "Match shared losses and seasonal latent grouping. The Markov LSTM has matching "
                 "HP likelihood losses and hard expected-gradient switching; vanilla has no HP head. "
                 "LSTM thermal/latent structure and active-power parameterization remain their own.",
    }
    recipe["max_profiles"] = int(args.max_profiles)
    recipe["optimizer"]["epochs"] = int(args.epochs)
    recipe["evaluation"].update(num_scenarios=int(args.num_scenarios),
                                 num_profile_plots=int(args.num_profile_plots),
                                 num_policy_profile_plots=int(args.num_profile_plots))
    recipe.setdefault("force_off_evaluation", {})["raw_only"] = True
    recipe.setdefault("poor_excitation_evaluation", {})["enabled"] = False
    return recipe


def _train_variant(args, recipe, split, variant: str) -> Path:
    record_path = _artifact_record_path(args.output_dir, variant)
    variant_root = args.output_dir / "variants" / variant
    model_root = variant_root / "models"
    eligible = set(_eligible_hp_ids(Path(recipe["dataset_path"])))
    excluded = tuple(
        sorted(eligible.difference({*split.train_ids, *split.validation_ids}))
    )
    configs = _experiment_configs(
        recipe,
        model_root,
        max_profiles=len(split.train_ids) + len(split.validation_ids),
        epochs=int(args.epochs),
        fixed_test_profile_ids=split.validation_ids,
        excluded_profile_ids=excluded,
    )
    config = replace(
        configs["probabilistic_structured" if variant == "structured" else "probabilistic_lstm"],
        model_name=VARIANT_MODELS[variant],
    )
    _require_same(variant_root / "training_request.json", asdict(config))
    if args.resume and record_path.exists():
        artifact_dir = _load_artifact_record(args.output_dir, variant)
        record = json.loads(record_path.read_text())
        if record.get("artifact_hashes") != _fingerprints(artifact_dir):
            raise ValueError(f"Checkpoint provenance missing or changed: {artifact_dir}")
        print(f"lstm_hurdle_variant={variant} stage=train action=reuse artifact={artifact_dir}")
        return artifact_dir
    started = time.perf_counter()
    artifact_dir = Path(train(config))
    elapsed_minutes = (time.perf_counter() - started) / 60.0

    artifact = load_artifact(artifact_dir)
    if saved_profile_ids(artifact, "train") != split.train_ids:
        raise ValueError(f"{variant} training IDs differ from the shared split")
    if saved_profile_ids(artifact, "test") != split.validation_ids:
        raise ValueError(f"{variant} validation IDs differ from the shared split")
    from .calibrated_model_benchmark import _artifact_training_configuration
    effective, source = _artifact_training_configuration(artifact)
    _require_same(variant_root / "effective_training_configuration.json",
                  {"source": source, "configuration": effective,
                   "artifact_dir": str(artifact_dir.resolve()), "artifact_hashes": _fingerprints(artifact_dir)})
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "model_name": VARIANT_MODELS[variant],
                "artifact_dir": str(artifact_dir),
                "training_minutes": elapsed_minutes,
                "artifact_hashes": _fingerprints(artifact_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"lstm_hurdle_variant={variant} stage=train action=complete "
        f"minutes={elapsed_minutes:.2f}"
    )
    return artifact_dir


def _training_rows(root: Path, variants: Sequence[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        artifact_dir = _load_artifact_record(root, variant)
        artifact = load_artifact(artifact_dir)
        metadata = artifact.metadata.get("legacy_metadata", artifact.metadata)
        metrics = metadata.get("test_metrics", {})
        record = json.loads(_artifact_record_path(root, variant).read_text())
        max_grad, skipped = _gradient_diagnostics(artifact_dir)
        rows.append(
            {
                "variant": variant,
                "model_name": artifact.spec.name,
                "checkpoint_epoch": metadata.get("checkpoint_epoch"),
                "test_total_nrmse": metrics.get("total_nrmse"),
                "test_temperature_rmse_c": metrics.get("temperature_rmse_c", metrics.get("rmse_c")),
                "test_qroom_rmse_w_m2": metrics.get("qroom_rmse_w_m2"),
                "test_pel_rmse_w_m2": metrics.get("pel_rmse_w_m2"),
                "max_grad_norm": max_grad,
                "skipped_updates": skipped,
                "training_minutes": record.get("training_minutes"),
            }
        )
    return pd.DataFrame(rows)


def _pel_dynamics_rows(root: Path, variants: Sequence[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        score_dir = root / "variants" / variant / "scoring" / "test"
        annual = pd.read_csv(score_dir / "pel_annual_energy_calibration.csv")
        conditional = pd.read_csv(
            score_dir / "pel_conditional_calibration_summary.csv"
        )
        durations = pd.read_csv(score_dir / "pel_run_duration_distribution.csv")
        duration_pivot = durations.pivot(
            index="bin_label", columns="signal", values="probability"
        ).fillna(0.0)
        duration_tv = 0.5 * float(
            np.abs(duration_pivot["simulation"] - duration_pivot["emulation"]).sum()
        )
        acf = pd.read_csv(score_dir / "pel_mode_autocorrelation.csv")
        acf_rmse = float(
            np.sqrt(np.mean((acf["truth_acf"] - acf["scenario_mean_acf"]) ** 2))
        )
        transitions = pd.read_csv(score_dir / "pel_transition_reliability.csv")
        transition_ece = {}
        for name, group in transitions.groupby("transition"):
            weights = group["observation_count"].to_numpy(dtype=float)
            error = np.abs(
                group["predicted_probability"].to_numpy(dtype=float)
                - group["observed_frequency"].to_numpy(dtype=float)
            )
            transition_ece[name] = float(np.average(error, weights=weights))
        rows.append(
            {
                "variant": variant,
                "annual_energy_calibration_mae": float(
                    annual["abs_calibration_error"].mean()
                ),
                "conditional_calibration_macro_mae": float(
                    conditional["calibration_mae"].mean()
                ),
                "run_duration_total_variation": duration_tv,
                "mode_acf_rmse": acf_rmse,
                "start_probability_ece": transition_ece.get(
                    "start_given_observed_off", np.nan
                ),
                "stay_on_probability_ece": transition_ece.get(
                    "stay_on_given_observed_on", np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_dashboard(
    path: Path,
    training: pd.DataFrame,
    annual: pd.DataFrame,
    traces: pd.DataFrame,
    pel: pd.DataFrame,
    flex: pd.DataFrame,
    force_flex: pd.DataFrame,
) -> None:
    figure = make_subplots(
        rows=3,
        cols=3,
        specs=[[{"type": "table", "colspan": 3}, None, None], [{}, {}, {}], [{}, {}, {}]],
        subplot_titles=(
            "Training and checkpoint",
            "Annual autonomous NRMSE",
            "Trace rank calibration",
            "Trace CRPS",
            "Pel dynamics diagnostics",
            "Raw flexibility CRPS",
            "Raw flexibility 90% coverage",
        ),
        row_heights=(0.20, 0.38, 0.42),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    table = training.copy()
    table["variant"] = table["variant"].map(VARIANT_LABELS)
    columns = [
        "variant",
        "checkpoint_epoch",
        "test_total_nrmse",
        "max_grad_norm",
        "skipped_updates",
        "training_minutes",
    ]
    figure.add_trace(
        go.Table(
            header={"values": ["Model", "Selected epoch", "Validation NRMSE", "Max gradient",
                               "Rejected updates", "Wall minutes (includes pauses)"],
                    "fill_color": "#e8eef5"},
            cells={"values": [table[column] for column in columns]},
        ),
        row=1,
        col=1,
    )
    for variant in training["variant"]:
        color = VARIANT_COLORS[variant]
        label = VARIANT_LABELS[variant]
        annual_row = annual[annual["variant"].eq(variant)].iloc[0]
        figure.add_trace(
            go.Bar(
                x=["Tin", "Qroom", "Pel"],
                y=[
                    annual_row["temperature_nrmse"],
                    annual_row["qroom_nrmse"],
                    annual_row["pel_nrmse"],
                ],
                name=label,
                legendgroup=variant,
                marker_color=color,
            ),
            row=2,
            col=1,
        )
        trace = traces[traces["variant"].eq(variant)]
        figure.add_trace(
            go.Bar(
                x=trace["signal"],
                y=trace["rank_calibration_mae"],
                name=label,
                legendgroup=variant,
                marker_color=color,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Bar(
                x=trace["signal"],
                y=trace["mean_crps"],
                name=label,
                legendgroup=variant,
                marker_color=color,
                showlegend=False,
            ),
            row=2,
            col=3,
        )
        pel_row = pel[pel["variant"].eq(variant)].iloc[0]
        figure.add_trace(
            go.Bar(
                x=["annual cal", "duration TV", "mode ACF", "start ECE", "stay ECE"],
                y=[
                    pel_row["annual_energy_calibration_mae"],
                    pel_row["run_duration_total_variation"],
                    pel_row["mode_acf_rmse"],
                    pel_row["start_probability_ece"],
                    pel_row["stay_on_probability_ece"],
                ],
                name=label,
                legendgroup=variant,
                marker_color=color,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        raw_flex = flex[
            flex["variant"].eq(variant) & flex["calibration"].eq("raw")
        ]
        force = force_flex[force_flex["method"].eq(variant)] if "method" in force_flex else pd.DataFrame()
        if not force.empty:
            force = force.groupby("kpi_kind", as_index=False)[["crps", "coverage_90"]].mean()
            raw_flex = pd.concat([raw_flex, force], ignore_index=True)
        figure.add_trace(
            go.Bar(
                x=raw_flex["kpi_kind"],
                y=raw_flex["crps"],
                name=label,
                legendgroup=variant,
                marker_color=color,
                showlegend=False,
            ),
            row=3,
            col=2,
        )
        labels = list(raw_flex["kpi_kind"])
        values = list(raw_flex["coverage_90"])
        figure.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name=label,
                legendgroup=variant,
                marker_color=color,
                showlegend=False,
            ),
            row=3,
            col=3,
        )
    # ``add_hline`` inspects every subplot and fails when the figure also has a
    # domain/table subplot. The final Cartesian panel is x6/y6.
    figure.add_shape(
        type="line",
        xref="x6 domain",
        yref="y6",
        x0=0.0,
        x1=1.0,
        y0=0.9,
        y1=0.9,
        line={"dash": "dash", "color": "black"},
    )
    figure.update_yaxes(title_text="NRMSE", row=2, col=1)
    figure.update_yaxes(title_text="Mean absolute CDF error", row=2, col=2)
    figure.update_yaxes(title_text="CRPS", row=2, col=3)
    figure.update_yaxes(title_text="Error", row=3, col=1)
    figure.update_yaxes(title_text="CRPS", row=3, col=2)
    figure.update_yaxes(title_text="Coverage", range=[0, 1.05], row=3, col=3)
    figure.update_layout(
        title="Structured / LSTM HP-emission comparison (matched objective)",
        template="plotly_white",
        barmode="group",
        height=1450,
        width=1650,
        legend={"orientation": "h", "y": -0.10},
        hoverlabel={"namelength": -1, "align": "left"},
        margin={"b": 170},
    )
    figure.write_html(path)


def _report(args, recipe, variants: Sequence[str]) -> Path:
    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    training = _training_rows(args.output_dir, variants)
    annual = _annual_rows(args.output_dir, variants)
    traces = _trace_rows(args.output_dir, variants)
    pel = _pel_dynamics_rows(args.output_dir, variants)
    distribution, reliability, points = _flexibility_tables(
        args.output_dir, variants, recipe
    )
    flex = _macro_flexibility(distribution)
    force_signals, force_flex = _force_off_tables(args.output_dir, variants)
    components, assessment = _rank_components(
        annual, traces, distribution, points, force_flex
    )
    tables = {
        "training_summary.csv": training,
        "annual_signal_metrics.csv": annual,
        "temporal_trace_metrics.csv": traces,
        "pel_dynamics_metrics.csv": pel,
        "flexibility_distribution_metrics.csv": distribution,
        "flexibility_distribution_macro.csv": flex,
        "flexibility_reliability.csv": reliability,
        "flexibility_point_metrics.csv": points,
        "force_off_signal_metrics.csv": force_signals,
        "force_off_flexibility_metrics.csv": force_flex,
        "automatic_assessment_components.csv": components,
        "automatic_assessment.csv": assessment,
    }
    for name, frame in tables.items():
        frame.to_csv(report_dir / name, index=False)
    _audit_and_native_report(args, recipe, variants, report_dir)
    dashboard = report_dir / "comparison_dashboard.html"
    _write_dashboard(dashboard, training, annual, traces, pel, flex, force_flex)
    (report_dir / "report_manifest.json").write_text(
        json.dumps(
            json_safe(
                {
                    "variants": list(variants),
                    "labels": VARIANT_LABELS,
                    "tables": list(tables),
                    "dashboard": dashboard.name,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"saved_lstm_hurdle_ablation_dashboard={dashboard}")
    return dashboard


def _audit_and_native_report(args, recipe, variants, report_dir):
    from .generic_response_report import native_scores
    from .plant_accounting_report import paired_crps_bootstrap

    split = json.loads((args.output_dir / "split_manifest.json").read_text())
    frames, traces, audit = [], [], {}
    reference_columns = None
    for variant in variants:
        root = args.output_dir / "variants" / variant
        artifact_dir = _load_artifact_record(args.output_dir, variant)
        record = json.loads((root / "artifact.json").read_text())
        if record.get("artifact_hashes") != _fingerprints(artifact_dir):
            raise ValueError(f"Changed checkpoint: {variant}")
        artifact = load_artifact(artifact_dir)
        if artifact.spec.name != VARIANT_MODELS[variant]:
            raise ValueError(f"Wrong model: {variant}")
        for source, expected in (("train", "train_ids"), ("test", "validation_ids")):
            if list(saved_profile_ids(artifact, source)) != split[expected]:
                raise ValueError(f"Unmatched {source} IDs: {variant}")
        metadata = artifact.metadata.get("legacy_metadata", artifact.metadata)
        columns = {k: metadata[k] for k in ("input_columns", "metadata_columns")}
        if reference_columns is not None and columns != reference_columns:
            raise ValueError(f"Unmatched observed inputs/metadata: {variant}")
        reference_columns = columns
        audit[variant] = {"model": artifact.spec.name, "checkpoint_epoch": metadata.get("checkpoint_epoch"),
                          "training_minutes": record["training_minutes"], **columns}
        for name in ("calibration", "test"):
            request = json.loads((root / "scoring" / name / "request.json").read_text())
            if request["profile_ids"] != split[name + "_ids"] or request["num_scenarios"] != args.num_scenarios:
                raise ValueError(f"Unmatched annual evaluation: {variant}/{name}")
        sp, off = root / "scoring/test", root / "force_off/test"
        for kind in ("regression", "f"):
            frame = _emulator_distribution(sp, method=variant, kpi_kind=kind)
            frame["kpi_kind"] = "setpoint_" + kind
            frames.append(frame)
            frames.append(_force_off_emulator_distribution(
                off, method=variant, source_model="structured_gated",
                estimator="regression" if kind == "regression" else "direct_ratio",
                kpi_kind="force_off_" + kind,
            ))
        traces.extend((variant, p) for p in sorted(sp.glob("profile_*.html")))
        traces.extend((variant, p) for p in sorted((off / "probabilistic_traces").glob("*structured_gated.html")))
    summary, metrics, reliability, buildings, pit = native_scores(pd.concat(frames, ignore_index=True))
    if not pit.particles.eq(args.num_scenarios).all():
        raise ValueError("Unequal annual particle counts")
    paired = paired_crps_bootstrap(buildings, reference=variants[0])
    if len(variants) > 1 and (
        len(paired) != 4 * (len(variants) - 1)
        or not paired.matched_cells.eq(paired.baseline_cells).all()
        or not paired.matched_cells.eq(paired.candidate_cells).all()
        or not paired.matched_buildings.eq(len(split["test_ids"])).all()
    ):
        raise ValueError("Incomplete paired KPI comparison")
    for name, frame in (("native_summary", summary), ("native_cells", metrics),
                        ("native_reliability", reliability), ("native_buildings", buildings),
                        ("native_pit", pit), ("paired_crps", paired)):
        frame.to_csv(report_dir / (name + ".csv"), index=False)
    review_html = ""
    review_path = args.output_dir / "final_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review["native_summary_sha256"] != hashlib.sha256((report_dir / "native_summary.csv").read_bytes()).hexdigest():
            raise ValueError("Final review is stale relative to the native results")
        review_html = "<h2>Findings</h2>" + "".join(
            f"<p>{html.escape(paragraph)}</p>" for paragraph in review["paragraphs"])
    _require_same(report_dir / "fairness_audit.json", {"split": split, "models": audit})
    links = "".join(f'<li>{html.escape(v)}: <a href="{html.escape(str(p.resolve()))}">{html.escape(p.name)}</a></li>'
                    for v, p in traces)
    table = lambda frame: frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")
    calibration_html = ""
    if recipe.get("comparison_protocol") == "current_best_shared_losses":
        joint = _ecc_report(args.output_dir, variants, recipe, report_dir)
        calibration_html = (
            '<h2>Joint KPI calibration</h2><p><a href="joint_distribution_metrics.csv">Joint scores CSV</a> | '
            '<a href="distribution_metrics.csv">Marginal scores CSV</a> | '
            '<a href="rns_ecc_test_distributions.csv">RNS-ECC KPI ensemble</a></p>'
            '<p>RNS and RNS-ECC have identical marginals; ECC changes only joint dependence. '
            'These transformations do not calibrate physical trajectories.</p>' + table(joint)
        )
        scope = ("Current-best 0.5-hour structured model, vanilla LSTM and asymmetric-Markov LSTM. "
                 "Shared trajectory ES, variogram, soft-optimistic and pooled direct-F CRPS losses, "
                 "plus normalized process-noise regularization. Seasonal windows share annual latent draws. "
                 "Structured and Markov LSTM use hard expected-gradient HP training and matching HP "
                 "auxiliary loss weights; vanilla has no HP-specific head or likelihood. "
                 "Architecture-specific parameterizations remain different. All models retrained from scratch. "
                 "RNS and RNS-ECC change KPI ensembles only; annual trace bands remain native. "
                 "The historical holdout has been used in previous experiments and is not a pristine final test.")
    else:
        scope = ("Shared three energy-score objectives. No auxiliary HP likelihood. Structured is the "
                 "matched-recipe comparator, not the separately tuned current-best checkpoint. "
                 "Expected HP emissions during training; Bernoulli at inference.")
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Matched model comparison</title>
    <style>body{{font:15px system-ui;margin:32px;color:#222}}table{{border-collapse:collapse;font-size:13px}}
    td,th{{padding:7px;border-bottom:1px solid #ddd}}th{{text-align:left}}</style></head><body>
    <h1>Structured and LSTM HP-head comparison</h1>
    <p>{len(split['selected_ids'])} eligible buildings (requested {split['requested_max_profiles']});
    shared splits and training budget. {html.escape(scope)}
    No test-set fitting in this run. Single training seed; bootstrap intervals describe held-out
    building variability, not training-seed uncertainty.</p>
    <p>The training table's legacy test_* fields refer to the validation fold. Training minutes are
    elapsed wall time, including pauses and trainer postprocessing, not a compute-time comparison.</p>
    <p><a href="comparison_dashboard.html">Dashboard</a> | <a href="fairness_audit.json">Fairness audit</a> |
    <a href="flexibility_distribution_macro.csv">Raw and RNS setpoint scores</a></p>
    {review_html}
    <h2>Native annual KPIs</h2>{table(summary)}
    <h2>Paired CRPS differences vs {html.escape(variants[0])}</h2>
    <p>Negative means better than the reference. 95% building-bootstrap intervals, 2000 resamples.</p>{table(paired)}
    {calibration_html}
    <h2>Annual traces with uncertainty bands</h2><ul>{links}</ul></body></html>'''
    (report_dir / "index.html").write_text(page)


def _ecc_report(root, variants, recipe, report_dir):
    from .probabilistic_calibration import (
        calibrate_distribution_table, ecc_reorder, summarize_joint_distribution_tables,
    )

    tables = {}
    for fold in ("calibration", "test"):
        tables[fold] = pd.concat([
            _emulator_distribution(root / "variants" / variant / "scoring" / fold,
                                   method=variant, kpi_kind=kind)
            for variant in variants for kind in ("regression_based", "f")
        ], ignore_index=True)
        tables[fold].to_csv(report_dir / f"raw_{fold}_distributions.csv", index=False)
    calibrated, metrics, reliability, buildings = calibrate_distribution_table(
        tables["calibration"], tables["test"],
        minimum_scale=float(recipe["calibration"]["minimum_scale"]),
        minimum_scale_fraction=float(recipe["calibration"]["minimum_scale_fraction"]),
    )
    ecc = ecc_reorder(tables["test"], calibrated)
    calibrated.to_csv(report_dir / "rns_test_distributions.csv", index=False)
    ecc.to_csv(report_dir / "rns_ecc_test_distributions.csv", index=False)
    for name, frame in (("distribution_metrics", metrics), ("distribution_reliability", reliability),
                        ("distribution_by_building", buildings)):
        reordered = frame.loc[frame.calibration.eq("rns")].assign(calibration="rns_ecc")
        pd.concat([frame, reordered], ignore_index=True).to_csv(report_dir / f"{name}.csv", index=False)
    joint = summarize_joint_distribution_tables(
        tables["calibration"], {"raw": tables["test"], "rns": calibrated, "rns_ecc": ecc},
        kpi_kinds=("regression_based", "f"),
    )
    joint.to_csv(report_dir / "joint_distribution_metrics.csv", index=False)
    return joint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--current-best-root", type=Path,
                        help="Use the authoritative 0.5-hour artifact configuration and matched shared losses.")
    parser.add_argument("--campaign-root", type=Path,
                        help="Use a paired four-policy campaign and its supplied building folds.")
    parser.add_argument("--exclude-missing-profile-ids", type=int, nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-profiles", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-scenarios", type=int, default=50)
    parser.add_argument("--num-profile-plots", type=int, default=0)
    parser.add_argument(
        "--hurdle-mode-loss-weight",
        type=float,
        default=0.0,
        help="Optional direct hurdle-mode likelihood weight; zero is the fair default.",
    )
    parser.add_argument(
        "--hurdle-active-power-nll-weight",
        type=float,
        default=0.0,
        help="Optional active-power NLL weight; zero is the fair default.",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANT_ORDER, default=DEFAULT_VARIANTS
    )
    parser.add_argument(
        "--stages", nargs="+", choices=("all", *STAGES), default=("all",)
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--worker-variant", choices=VARIANT_ORDER,
                        help="Execute one arm without changing the frozen comparison roster.")
    parser.add_argument("--isolate-workers", action="store_true",
                        help="Release compiled JAX state between model arms.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.max_profiles < 12:
        raise ValueError("--max-profiles must be at least 12")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.num_scenarios < 2:
        raise ValueError("--num-scenarios must be at least 2")
    if args.hurdle_mode_loss_weight < 0.0:
        raise ValueError("--hurdle-mode-loss-weight must be non-negative")
    if args.hurdle_active_power_nll_weight < 0.0:
        raise ValueError("--hurdle-active-power-nll-weight must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = _current_best_base(args) if args.current_best_root is not None else _resolved_base(args)
    if args.campaign_root is not None:
        from .campaign_benchmark import resolve_campaign
        base, split = resolve_campaign(base, args.campaign_root, args.output_dir,
                                       excluded_ids=args.exclude_missing_profile_ids)
    else:
        split = _shared_split(args, base)
    stages = set(STAGES if "all" in args.stages else args.stages)
    variants = tuple(dict.fromkeys(args.variants))
    if args.worker_variant is not None and args.worker_variant not in variants:
        raise ValueError("The worker variant must belong to the comparison roster")
    if "structured" in variants:
        if args.hurdle_mode_loss_weight or args.hurdle_active_power_nll_weight:
            raise ValueError("Use a separate run for likelihood-supervised hurdle variants")
        if args.current_best_root is None:
            _validate_shared_objective(base)
        elif set(variants) != {"structured", "continuous", "hurdle_markov"}:
            raise ValueError("Current-best comparison requires structured, vanilla and Markov LSTM")
    _require_same(args.output_dir / "comparison_request.json", {
        "recipe": base, "split": asdict(split), "variants": variants,
        "hurdle_mode_loss_weight": args.hurdle_mode_loss_weight,
        "hurdle_active_power_nll_weight": args.hurdle_active_power_nll_weight,
    })
    sources = [p for package in ("neural_building_emulator", "neural_building_emulator_refactor")
               for p in Path(package).rglob("*.py") if "test" not in p.name and "tests" not in p.parts]
    _record_implementation(args.output_dir, {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)
    }, report_only=stages == {"report"})
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(
            json_safe(
                {
                    "source_config": str(args.config),
                    "base_recipe": base,
                    "split": asdict(split),
                    "variants": {
                        name: {
                            "model_name": VARIANT_MODELS[name],
                            "label": VARIANT_LABELS[name],
                        }
                        for name in variants
                    },
                    "stages": list(args.stages),
                    "num_scenarios": int(args.num_scenarios),
                    "hurdle_mode_loss_weight": float(
                        args.hurdle_mode_loss_weight
                    ),
                    "hurdle_active_power_nll_weight": float(
                        args.hurdle_active_power_nll_weight
                    ),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"lstm_hurdle_split train={len(split.train_ids)} "
        f"validation={len(split.validation_ids)} "
        f"calibration={len(split.calibration_ids)} test={len(split.test_ids)} "
        f"fingerprint={split.fingerprint}"
    )
    if stages == {"split"}:
        return args.output_dir / "split_manifest.json"

    for variant in variants:
        if args.worker_variant is not None and variant != args.worker_variant:
            continue
        recipe = _variant_recipe(
            base,
            variant,
            hurdle_mode_loss_weight=args.hurdle_mode_loss_weight,
            hurdle_active_power_nll_weight=(
                args.hurdle_active_power_nll_weight
            ),
        )
        variant_root = args.output_dir / "variants" / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        _require_same(variant_root / "resolved_recipe.json", recipe)
        if "train" in stages:
            artifact_dir = _train_variant(args, recipe, split, variant)
        else:
            artifact_dir = _load_artifact_record(args.output_dir, variant)
        _require_same(variant_root / "scoring_checkpoint.json", _fingerprints(artifact_dir))
        if "score" in stages:
            _score_variant(args, recipe, split, variant, artifact_dir)
        if "force_off" in stages:
            _score_force_off_variant(args, recipe, split, variant, artifact_dir)

    if "report" in stages:
        return _report(args, base, variants)
    return args.output_dir / "experiment_manifest.json"


def main() -> None:
    args = parse_args()
    if not args.isolate_workers:
        run(args)
        return
    if args.worker_variant is not None:
        raise ValueError("Cannot isolate an already isolated worker")
    stages = set(STAGES if "all" in args.stages else args.stages)
    child_args = [value for value in sys.argv[1:] if value != "--isolate-workers"]
    requested = [stage for stage in STAGES if stage in stages and stage not in ("split", "report")]
    if requested:
        for variant in dict.fromkeys(args.variants):
            subprocess.run([
                sys.executable, "-u", "-m", "neural_building_emulator_refactor.lstm_hurdle_ablation",
                *child_args, "--worker-variant", variant, "--stages", *requested,
            ], check=True)
    if "report" in stages or stages == {"split"}:
        subprocess.run([
            sys.executable, "-u", "-m", "neural_building_emulator_refactor.lstm_hurdle_ablation",
            *child_args, "--stages", "report" if "report" in stages else "split",
        ], check=True)


if __name__ == "__main__":
    main()
