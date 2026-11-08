"""Typed configuration for the refactored comparison pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neural_building_emulator.data import DEFAULT_DATASET_PATH


@dataclass(frozen=True)
class OptimizerConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    train_eval_max_windows: int = 512


@dataclass(frozen=True)
class LSTMConfig:
    hidden_dim: int = 64
    metadata_hidden_dim: int = 64
    metadata_depth: int = 2
    bptt_truncate_steps: int = 0
    output_weights: tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ProbabilisticLSTMConfig:
    """Matched uncertainty and trajectory-loss settings for the LSTM baseline."""

    hidden_dim: int = 64
    metadata_hidden_dim: int = 64
    metadata_depth: int = 2
    bptt_truncate_steps: int = 32
    latent_dim: int = 4
    particles: int = 24
    eval_particles: int = 16
    plot_particles: int = 100
    process_noise: str = "heteroscedastic"
    process_noise_init: float = -4.0
    hp_scenario_mode: str = "bernoulli"
    hp_training_mode: str = "expected"
    hp_mode_loss_weight: float = 0.1
    hp_active_power_nll_weight: float = 1.0
    hp_inactive_leakage_weight: float = 0.1
    softopt_weight: float = 2.0
    softopt_temperature: float = 0.1
    flex_kpi_crps_weight: float = 0.0
    flex_kpi_crps_horizons_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    flex_kpi_crps_setpoint_threshold_c: float = 0.05
    flex_kpi_crps_min_events: int = 8
    flex_kpi_crps_windows_per_profile: int = 1
    process_regularization_weight: float = 0.0
    increment_energy_weight: float = 0.0
    increment_energy_lags: tuple[int, ...] = (1, 2, 4, 8)
    increment_energy_lags_per_batch: int = 1
    increment_energy_output_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    increment_energy_constant_setpoint_only: bool = True
    increment_energy_setpoint_threshold_c: float = 0.05
    event_energy_weight: float = 0.0
    event_energy_horizons_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    event_energy_events_per_profile: int = 4
    event_energy_output_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    event_energy_setpoint_threshold_c: float = 0.05
    variogram_weight: float = 1.0
    variogram_power: float = 0.5
    variogram_lags: tuple[int, ...] = (1, 2, 4, 8)
    variogram_output_weights: tuple[float, ...] = (1.0, 0.0, 0.0)
    variogram_constant_setpoint_only: bool = True
    variogram_setpoint_threshold_c: float = 0.05
    setpoint_shock_timescales_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    output_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    skip_grad_norm_above: float = 1000.0


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    dataset_path: Path = DEFAULT_DATASET_PATH
    additional_dataset_paths: tuple[Path, ...] = ()
    output_dir: Path = Path("output/neural_building_emulator_refactor")
    max_profiles: int | None = 100
    test_fraction: float = 0.2
    fixed_test_profile_ids: tuple[int, ...] = ()
    excluded_profile_ids: tuple[int, ...] = ()
    seed: int = 13
    sequence_length: int = 960
    stride: int = 2024
    rotate_window_starts: bool = True
    heating_mode: str = "zone_thermal"
    heat_input_normalization: str = "per_floor_area"
    hp_power_area_normalization: str = "building_heated_area"
    hp_dhw_metadata: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    probabilistic_lstm: ProbabilisticLSTMConfig = field(
        default_factory=ProbabilisticLSTMConfig
    )
    legacy_overrides: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.max_profiles is not None and self.max_profiles < 1:
            raise ValueError("max_profiles must be positive or None")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1)")
        if set(self.fixed_test_profile_ids).intersection(self.excluded_profile_ids):
            raise ValueError(
                "fixed_test_profile_ids and excluded_profile_ids must be disjoint"
            )
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.optimizer.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.optimizer.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.optimizer.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer.gradient_clip_norm < 0.0:
            raise ValueError("gradient_clip_norm must be non-negative")
        if self.optimizer.train_eval_max_windows < 0:
            raise ValueError("train_eval_max_windows must be non-negative")
        if self.lstm.hidden_dim < 1 or self.lstm.metadata_hidden_dim < 1:
            raise ValueError("LSTM hidden dimensions must be positive")
        if self.lstm.metadata_depth < 1:
            raise ValueError("metadata_depth must be positive")
        if self.lstm.bptt_truncate_steps < 0:
            raise ValueError("bptt_truncate_steps must be non-negative")
        probabilistic = self.probabilistic_lstm
        if probabilistic.hidden_dim < 1 or probabilistic.metadata_hidden_dim < 1:
            raise ValueError("probabilistic LSTM hidden dimensions must be positive")
        if probabilistic.metadata_depth < 1:
            raise ValueError("probabilistic LSTM metadata_depth must be positive")
        if probabilistic.bptt_truncate_steps < 0:
            raise ValueError("probabilistic LSTM BPTT truncation must be non-negative")
        if probabilistic.latent_dim < 1:
            raise ValueError("probabilistic LSTM latent_dim must be positive")
        if min(
            probabilistic.particles,
            probabilistic.eval_particles,
            probabilistic.plot_particles,
        ) < 1:
            raise ValueError("probabilistic LSTM particle counts must be positive")
        if probabilistic.process_noise not in (
            "none",
            "constant",
            "heteroscedastic",
        ):
            raise ValueError("unknown probabilistic LSTM process-noise mode")
        if probabilistic.hp_training_mode not in ("expected", "hard_expected_grad"):
            raise ValueError(
                "LSTM HP training mode must be expected or hard_expected_grad"
            )
        if (probabilistic.hp_training_mode == "hard_expected_grad"
                and self.model_name == "closed_loop_hp_probabilistic_lstm"):
            raise ValueError("Hard HP training is supported by hurdle LSTMs, not the legacy head")
        if probabilistic.flex_kpi_crps_weight < 0.0:
            raise ValueError("flex_kpi_crps_weight must be non-negative")
        if probabilistic.process_regularization_weight < 0.0:
            raise ValueError("process_regularization_weight must be non-negative")
        if (not probabilistic.flex_kpi_crps_horizons_hours
                or any(h <= 0.0 for h in probabilistic.flex_kpi_crps_horizons_hours)):
            raise ValueError("Flexibility horizons must be nonempty and positive")
        if (probabilistic.flex_kpi_crps_min_events < 1
                or probabilistic.flex_kpi_crps_setpoint_threshold_c < 0.0):
            raise ValueError("Invalid flexibility event settings")
        group_size = probabilistic.flex_kpi_crps_windows_per_profile
        if group_size < 1 or self.optimizer.batch_size % group_size:
            raise ValueError("Batch size must be divisible by flexibility windows_per_profile")
        if probabilistic.hp_scenario_mode not in ("bernoulli", "expected"):
            raise ValueError("unknown probabilistic LSTM HP scenario mode")
        if probabilistic.softopt_temperature <= 0.0:
            raise ValueError("probabilistic LSTM softopt temperature must be positive")
        if probabilistic.increment_energy_weight < 0.0:
            raise ValueError("probabilistic LSTM increment energy weight must be non-negative")
        if any(value <= 0 for value in probabilistic.increment_energy_lags):
            raise ValueError("probabilistic LSTM increment energy lags must be positive")
        if probabilistic.increment_energy_lags_per_batch < 1:
            raise ValueError(
                "probabilistic LSTM increment energy lags_per_batch must be positive"
            )
        if len(probabilistic.increment_energy_output_weights) != 3:
            raise ValueError(
                "probabilistic LSTM increment energy output weights must contain T, Q, P"
            )
        if any(value < 0.0 for value in probabilistic.increment_energy_output_weights):
            raise ValueError(
                "probabilistic LSTM increment energy output weights must be non-negative"
            )
        if sum(probabilistic.increment_energy_output_weights) <= 0.0:
            raise ValueError(
                "probabilistic LSTM increment energy output weights need a positive value"
            )
        if probabilistic.increment_energy_setpoint_threshold_c < 0.0:
            raise ValueError(
                "probabilistic LSTM increment setpoint threshold must be non-negative"
            )
        if probabilistic.event_energy_weight < 0.0:
            raise ValueError("probabilistic LSTM event energy weight must be non-negative")
        if any(value <= 0.0 for value in probabilistic.event_energy_horizons_hours):
            raise ValueError(
                "probabilistic LSTM event energy horizons must be positive"
            )
        if probabilistic.event_energy_events_per_profile < 1:
            raise ValueError(
                "probabilistic LSTM event energy events_per_profile must be positive"
            )
        if len(probabilistic.event_energy_output_weights) != 3:
            raise ValueError(
                "probabilistic LSTM event energy output weights must contain T, Q, P"
            )
        if any(value < 0.0 for value in probabilistic.event_energy_output_weights):
            raise ValueError(
                "probabilistic LSTM event energy output weights must be non-negative"
            )
        if sum(probabilistic.event_energy_output_weights) <= 0.0:
            raise ValueError(
                "probabilistic LSTM event energy output weights need a positive value"
            )
        if probabilistic.event_energy_setpoint_threshold_c < 0.0:
            raise ValueError(
                "probabilistic LSTM event setpoint threshold must be non-negative"
            )
        if len(probabilistic.output_weights) != 3:
            raise ValueError("probabilistic LSTM output_weights must contain T, Q, P")
        if len(probabilistic.variogram_output_weights) != 3:
            raise ValueError(
                "probabilistic LSTM variogram_output_weights must contain T, Q, P"
            )
        if any(value < 0.0 for value in probabilistic.output_weights):
            raise ValueError("probabilistic LSTM output weights must be non-negative")
        if any(value < 0.0 for value in probabilistic.variogram_output_weights):
            raise ValueError("probabilistic LSTM variogram weights must be non-negative")
        if any(value <= 0.0 for value in probabilistic.variogram_lags):
            raise ValueError("probabilistic LSTM variogram lags must be positive")
        if any(
            value <= 0.0
            for value in probabilistic.setpoint_shock_timescales_hours
        ):
            raise ValueError("probabilistic LSTM shock timescales must be positive")
        if probabilistic.skip_grad_norm_above < 0.0:
            raise ValueError("probabilistic LSTM gradient skip threshold must be non-negative")
        if self.hp_power_area_normalization not in (
            "zone_floor_area",
            "building_heated_area",
        ):
            raise ValueError(
                "hp_power_area_normalization must be 'zone_floor_area' or "
                "'building_heated_area'"
            )
