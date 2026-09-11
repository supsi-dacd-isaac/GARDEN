"""Command-line entry point for registry-based model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig, LSTMConfig, OptimizerConfig
from .registry import MODEL_REGISTRY
from .trainer import train

LEGACY_KIND_TO_MODEL = {
    spec.legacy_model_kind: spec.name
    for spec in MODEL_REGISTRY.values()
    if spec.legacy_model_kind is not None
}

LEGACY_OPTION_FIELDS = {
    "input_encoder_dim": "input_encoder_dim",
    "input_encoder_hidden_dim": "input_encoder_hidden_dim",
    "input_encoder_depth": "input_encoder_depth",
    "state_dim": "state_dim",
    "hidden_dim": "hidden_dim",
    "depth": "depth",
    "target_mode": "target_mode",
    "contracting_gamma": "contracting_gamma",
    "contracting_state_bound": "contracting_state_bound",
    "contracting_temperature_scale": "contracting_temperature_scale",
    "contracting_temperature_delta_max_c": "contracting_temperature_delta_max_c",
    "contracting_temperature_update": "contracting_temperature_update",
    "contracting_transition_conditioning": "contracting_transition_conditioning",
    "contracting_additive_feedback_gain_bound": "contracting_additive_feedback_gain_bound",
    "contracting_q_to_t_mode": "contracting_q_to_t_mode",
    "contracting_q_to_t_time_constants_hours": "contracting_q_to_t_time_constants_hours",
    "contracting_q_to_t_gain_min_c_per_w_m2": "contracting_q_to_t_gain_min_c_per_w_m2",
    "contracting_q_to_t_gain_max_c_per_w_m2": "contracting_q_to_t_gain_max_c_per_w_m2",
    "bptt_truncate_steps": "bptt_truncate_steps",
    "hp_thermostat_demand_mode": "hp_thermostat_demand_mode",
    "hp_inactive_leakage_weight": "hp_inactive_leakage_weight",
    "hp_cop_cap": "hp_cop_cap",
    "prob_particles": "prob_particles",
    "prob_eval_particles": "prob_eval_particles",
    "prob_plot_particles": "prob_plot_particles",
    "prob_latent_dim": "prob_latent_dim",
    "prob_process_noise": "prob_process_noise",
    "prob_process_noise_init": "prob_process_noise_init",
    "prob_hp_emission_mode": "prob_hp_emission_mode",
    "prob_hp_training_mode": "prob_hp_training_mode",
    "prob_hp_activation_model": "prob_hp_activation_model",
    "prob_hp_scenario_mode": "prob_hp_scenario_mode",
    "prob_hp_history_hours": "prob_hp_history_hours",
    "prob_hp_setpoint_shock_timescales_hours": (
        "prob_hp_setpoint_shock_timescales_hours"
    ),
    "prob_softopt_weight": "prob_softopt_weight",
    "prob_variogram_weight": "prob_variogram_weight",
    "prob_variogram_lags": "prob_variogram_lags",
    "prob_variogram_output_weights": "prob_variogram_output_weights",
    "prob_variogram_constant_setpoint_only": "prob_variogram_constant_setpoint_only",
    "prob_variogram_setpoint_threshold_c": "prob_variogram_setpoint_threshold_c",
    "prob_ires_weight": "prob_ires_weight",
    "prob_flex_kpi_crps_weight": "prob_flex_kpi_crps_weight",
    "prob_flex_kpi_crps_horizons_hours": "prob_flex_kpi_crps_horizons_hours",
    "prob_flex_kpi_crps_setpoint_threshold_c": "prob_flex_kpi_crps_setpoint_threshold_c",
    "prob_flex_kpi_crps_min_events": "prob_flex_kpi_crps_min_events",
    "prob_flex_kpi_crps_estimator": "prob_flex_kpi_crps_estimator",
    "prob_flex_kpi_crps_ridge": "prob_flex_kpi_crps_ridge",
    "prob_flex_kpi_crps_controls": "prob_flex_kpi_crps_controls",
    "skip_grad_norm_above": "skip_grad_norm_above",
    "full_train_eval_every_epochs": "full_train_eval_every_epochs",
    "num_window_plots": "num_window_plots",
    "num_full_profile_plots": "num_full_profile_plots",
    "log_update_diagnostics": "log_update_diagnostics",
    "save_model_every_epochs": "save_model_every_epochs",
}


def _legacy_overrides(path: Path | None, args: argparse.Namespace) -> dict:
    if path is None:
        values = {}
    else:
        values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("--legacy-config-json must contain one JSON object")
    for argument_name, config_name in LEGACY_OPTION_FIELDS.items():
        value = getattr(args, argument_name)
        if value is not None:
            values[config_name] = value
    return values


def _add_legacy_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-encoder-dim", type=int)
    parser.add_argument("--input-encoder-hidden-dim", type=int)
    parser.add_argument("--input-encoder-depth", type=int)
    parser.add_argument("--state-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--target-mode", choices=("absolute", "delta", "residual"))
    parser.add_argument("--contracting-gamma", type=float)
    parser.add_argument("--contracting-state-bound", type=float)
    parser.add_argument("--contracting-temperature-scale", type=float)
    parser.add_argument("--contracting-temperature-delta-max-c", type=float)
    parser.add_argument(
        "--contracting-temperature-update",
        choices=(
            "auto",
            "absolute",
            "delta",
            "leaky_equilibrium",
            "bounded_equilibrium",
            "alpha_bounded_equilibrium",
        ),
    )
    parser.add_argument(
        "--contracting-transition-conditioning",
        choices=(
            "state_feedback",
            "exogenous",
            "exogenous_additive_feedback",
        ),
    )
    parser.add_argument("--contracting-additive-feedback-gain-bound", type=float)
    parser.add_argument(
        "--contracting-q-to-t-mode",
        choices=("unconstrained", "positive_leaky"),
    )
    parser.add_argument("--contracting-q-to-t-time-constants-hours", type=float, nargs="+")
    parser.add_argument("--contracting-q-to-t-gain-min-c-per-w-m2", type=float)
    parser.add_argument("--contracting-q-to-t-gain-max-c-per-w-m2", type=float)
    parser.add_argument("--bptt-truncate-steps", type=int)
    parser.add_argument(
        "--hp-thermostat-demand-mode",
        choices=("unconstrained", "monotone"),
    )
    parser.add_argument("--hp-inactive-leakage-weight", type=float)
    parser.add_argument("--hp-cop-cap", type=float)
    parser.add_argument("--prob-particles", type=int)
    parser.add_argument("--prob-eval-particles", type=int)
    parser.add_argument("--prob-plot-particles", type=int)
    parser.add_argument("--prob-latent-dim", type=int)
    parser.add_argument("--prob-process-noise", choices=("none", "constant", "heteroscedastic"))
    parser.add_argument("--prob-process-noise-init", type=float)
    parser.add_argument(
        "--prob-hp-emission-mode",
        choices=("bounded", "legacy_lognormal_mean"),
    )
    parser.add_argument("--prob-hp-training-mode", choices=("expected", "straight_through"))
    parser.add_argument(
        "--prob-hp-activation-model",
        choices=("independent", "persistent_markov", "power_history", "asymmetric_markov"),
    )
    parser.add_argument("--prob-hp-scenario-mode", choices=("bernoulli", "expected"))
    parser.add_argument("--prob-hp-history-hours", type=float)
    parser.add_argument(
        "--prob-hp-setpoint-shock-timescales-hours",
        type=float,
        nargs="+",
    )
    parser.add_argument("--prob-softopt-weight", type=float)
    parser.add_argument("--prob-variogram-weight", type=float)
    parser.add_argument("--prob-variogram-lags", type=int, nargs="+")
    parser.add_argument("--prob-variogram-output-weights", type=float, nargs="+")
    parser.add_argument(
        "--prob-variogram-constant-setpoint-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prob-variogram-setpoint-threshold-c", type=float)
    parser.add_argument("--prob-ires-weight", type=float)
    parser.add_argument("--prob-flex-kpi-crps-weight", type=float)
    parser.add_argument("--prob-flex-kpi-crps-horizons-hours", type=float, nargs="+")
    parser.add_argument("--prob-flex-kpi-crps-setpoint-threshold-c", type=float)
    parser.add_argument("--prob-flex-kpi-crps-min-events", type=int)
    parser.add_argument(
        "--prob-flex-kpi-crps-estimator",
        choices=("direct_ratio", "regression"),
    )
    parser.add_argument("--prob-flex-kpi-crps-ridge", type=float)
    parser.add_argument(
        "--prob-flex-kpi-crps-controls",
        choices=("none", "weather", "full"),
    )
    parser.add_argument("--skip-grad-norm-above", type=float)
    parser.add_argument("--full-train-eval-every-epochs", type=int)
    parser.add_argument("--num-window-plots", type=int)
    parser.add_argument("--num-full-profile-plots", type=int)
    parser.add_argument("--save-model-every-epochs", type=int)
    parser.add_argument(
        "--log-update-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Accepted for legacy CLI compatibility; selected artifacts are always saved.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a registered building-emulator comparison model."
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", choices=sorted(MODEL_REGISTRY))
    model_group.add_argument(
        "--model-kind",
        choices=sorted(LEGACY_KIND_TO_MODEL),
        help="Compatibility alias mapping an original model kind into the registry.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/neural_building_emulator_refactor"))
    parser.add_argument("--max-profiles", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sequence-length", type=int, default=960)
    parser.add_argument("--stride", type=int, default=2024)
    parser.add_argument("--rotate-window-starts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heating-mode", choices=("zone_thermal", "heating_electric", "A", "B"), default="zone_thermal")
    parser.add_argument("--heat-input-normalization", choices=("raw", "per_floor_area"), default="per_floor_area")
    parser.add_argument(
        "--hp-power-area-normalization",
        choices=("building_heated_area", "zone_floor_area"),
        default="building_heated_area",
        help=(
            "Normalize whole-building HP power by floor_area*totalFloors. "
            "Use zone_floor_area only to reproduce legacy closed-loop training."
        ),
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--train-eval-max-windows", type=int, default=512)
    parser.add_argument(
        "--evaluate-flexibility-after-training",
        action="store_true",
        help=(
            "After saving the selected contracting probabilistic closed-loop HP "
            "artifact, run the standard 100-scenario flexibility calibration on "
            "up to 100 test profiles."
        ),
    )
    parser.add_argument(
        "--post-training-flex-max-profiles",
        type=int,
        default=100,
        help=(
            "Maximum number of saved test profiles used by the optional post-training "
            "flexibility calibration; independent of training --max-profiles."
        ),
    )
    parser.add_argument(
        "--post-training-flex-kpi-estimator",
        choices=("direct_ratio", "regression"),
        default="regression",
        help=(
            "Flexibility estimator used by the optional post-training calibration. "
            "The default matches the final --kpi-estimator value in the established command."
        ),
    )
    parser.add_argument(
        "--post-training-flex-output-dir",
        type=Path,
        help=(
            "Optional output directory for post-training flexibility files. By default "
            "an estimator-specific directory is created under --output-dir."
        ),
    )

    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-metadata-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-metadata-depth", type=int, default=2)
    parser.add_argument("--lstm-bptt-truncate-steps", type=int, default=0)
    parser.add_argument("--lstm-output-weights", type=float, nargs="+", default=(1.0, 1.0, 1.0))
    parser.add_argument(
        "--legacy-config-json",
        type=Path,
        help="JSON object of legacy TrainConfig overrides for a state-space model.",
    )
    _add_legacy_model_arguments(parser)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    model_name = args.model or LEGACY_KIND_TO_MODEL[args.model_kind]
    legacy_overrides = _legacy_overrides(args.legacy_config_json, args)
    if MODEL_REGISTRY[model_name].backend != "legacy" and legacy_overrides:
        names = ", ".join(sorted(legacy_overrides))
        raise ValueError(f"Legacy model options cannot be used with {model_name}: {names}")
    return ExperimentConfig(
        model_name=model_name,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_profiles=args.max_profiles,
        test_fraction=args.test_fraction,
        seed=args.seed,
        sequence_length=args.sequence_length,
        stride=args.stride,
        rotate_window_starts=args.rotate_window_starts,
        heating_mode=args.heating_mode,
        heat_input_normalization=args.heat_input_normalization,
        hp_power_area_normalization=args.hp_power_area_normalization,
        optimizer=OptimizerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_clip_norm=args.gradient_clip_norm,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            train_eval_max_windows=args.train_eval_max_windows,
        ),
        lstm=LSTMConfig(
            hidden_dim=args.lstm_hidden_dim,
            metadata_hidden_dim=args.lstm_metadata_hidden_dim,
            metadata_depth=args.lstm_metadata_depth,
            bptt_truncate_steps=args.lstm_bptt_truncate_steps,
            output_weights=tuple(args.lstm_output_weights),
        ),
        legacy_overrides=legacy_overrides,
    )


def _validate_post_training_flexibility(
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> None:
    if not args.evaluate_flexibility_after_training:
        return
    if config.model_name != "closed_loop_hp_contracting_probabilistic":
        raise ValueError(
            "--evaluate-flexibility-after-training currently requires "
            "--model closed_loop_hp_contracting_probabilistic"
        )
    if args.post_training_flex_max_profiles < 1:
        raise ValueError("--post-training-flex-max-profiles must be positive")


def _run_post_training_flexibility(
    args: argparse.Namespace,
    config: ExperimentConfig,
    artifact_dir: Path,
) -> Path:
    from neural_building_emulator.flexibility_event_study_kpis import run_analysis

    estimator = args.post_training_flex_kpi_estimator
    default_directory_name = (
        "flexibility_event_study_scenarios_regressor"
        if estimator == "regression"
        else "flexibility_event_study_scenarios"
    )
    output_dir = args.post_training_flex_output_dir or (
        Path(config.output_dir) / default_directory_name
    )
    evaluation_args = argparse.Namespace(
        artifact_dir=artifact_dir,
        dataset=Path(config.dataset_path),
        output_dir=output_dir,
        profile_source="test",
        max_profiles=int(args.post_training_flex_max_profiles),
        seed=13,
        emulation_kpi_mode="scenario_average",
        num_scenarios=100,
        prob_eval_particles=None,
        prob_hp_scenario_mode="bernoulli",
        ventilation_rollout_mode="eplus_rule",
        prob_hp_emission_mode="artifact",
        coverage_quantiles="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
        horizons_hours="0.5,1,2,3",
        dt_hours=None,
        min_setpoint_change=0.05,
        min_events=20,
        kpi_estimator=estimator,
        controls="full",
    )
    print(
        "post_training_flexibility_evaluation=started "
        f"artifact={artifact_dir} output_dir={output_dir} "
        "profile_source=test max_profiles="
        f"{evaluation_args.max_profiles} scenarios=100 "
        "hp_scenario_mode=bernoulli hp_emission_mode=artifact "
        "ventilation_rollout_mode=eplus_rule horizons_hours=[0.5,1,2,3] "
        f"kpi_estimator={estimator} controls=full seed=13",
        flush=True,
    )
    run_analysis(evaluation_args)
    return Path(output_dir)


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    _validate_post_training_flexibility(args, config)
    artifact_dir = train(config)
    print(f"selected_artifact={artifact_dir}", flush=True)
    if args.evaluate_flexibility_after_training:
        output_dir = _run_post_training_flexibility(args, config, artifact_dir)
        print(f"post_training_flexibility_output={output_dir}")


if __name__ == "__main__":
    main()
