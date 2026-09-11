"""Probabilistic closed-loop HP emulator with bounded transition matrices."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ..ventilation import replace_with_eplus_ventilation
from .emulator import MLP
from .probabilistic_closed_loop_hp import (
    HPActivationModel,
    HPElectricRolloutMode,
    ProbClosedLoopAux,
    ProbHpEmissionMode,
    ProcessNoiseMode,
    bounded_lognormal_expm1_mean,
    straight_through_binary_concrete,
)
from .contracting_closed_loop_hp import (
    TemperatureUpdateMode,
    ThermalQResponseMode,
    ThermostatDemandMode,
    TransitionConditioningMode,
)
from .truncated_bptt import DEFAULT_BPTT_TRUNCATE_STEPS, detach_carry, validate_bptt_truncate_steps


class ProbabilisticContractingClosedLoopHPEmulator(eqx.Module):
    """Probabilistic closed-loop emulator with bounded recurrent state dynamics.

    Each particle samples a persistent latent variable ``xi``. At every step,
    the recurrent transition matrix is generated from metadata, ``xi`` and the
    current thermal forcing. Every generated matrix is Frobenius-normalized so
    its spectral norm is at most ``contraction_gamma``. Local ablations can opt
    into a trajectory-fixed matrix explicitly.
    """

    input_encoder: MLP | None
    transition_net: MLP
    transition_forcing_encoder: MLP | None
    x0_net: MLP
    e0_net: MLP
    hp_param_net: MLP
    output_net: MLP
    temperature_net: MLP
    temperature_alpha_net: MLP | None
    q_response_net: MLP | None
    q_response_init_net: MLP | None
    process_noise_net: MLP | None
    hp_controller_init_net: MLP | None
    hp_controller_transition_net: MLP | None
    hp_transition_net: MLP | None
    hp_mode0_net: MLP | None
    raw_process_scale: jnp.ndarray
    metadata_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    encoded_input_dim: int = eqx.field(static=True)
    thermal_forcing_dim: int = eqx.field(static=True)
    controller_state_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    process_noise_mode: ProcessNoiseMode = eqx.field(static=True)
    process_noise_init: float = eqx.field(static=True)
    process_noise_floor: float = eqx.field(static=True)
    process_noise_cap: float = eqx.field(static=True)
    hp_emission_mode: ProbHpEmissionMode = eqx.field(static=True)
    hp_activation_model: HPActivationModel = eqx.field(static=True)
    hp_persistent_latent_dim: int = eqx.field(static=True)
    hp_controller_leak: float = eqx.field(static=True)
    hp_controller_noise_scale: float = eqx.field(static=True)
    hp_history_hours: float = eqx.field(static=True)
    hp_history_timescales_hours: tuple[float, ...] = eqx.field(static=True)
    hp_setpoint_shock_timescales_hours: tuple[float, ...] = eqx.field(static=True)
    contraction_gamma: float = eqx.field(static=True)
    state_bound: float = eqx.field(static=True)
    temperature_output_scale: float = eqx.field(static=True)
    temperature_delta_max_c: float = eqx.field(static=True)
    temperature_update_mode: TemperatureUpdateMode = eqx.field(static=True)
    hp_dt_hours: float = eqx.field(static=True)
    hp_cop_floor: float = eqx.field(static=True)
    hp_cop_cap: float = eqx.field(static=True)
    hp_pel_cap_w_m2: float = eqx.field(static=True)
    hp_qroom_cap_w_m2: float = eqx.field(static=True)
    hp_energy_cap_wh_m2: float = eqx.field(static=True)
    input_mean: tuple[float, ...] = eqx.field(static=True)
    input_scale: tuple[float, ...] = eqx.field(static=True)
    target_mean: tuple[float, ...] = eqx.field(static=True)
    target_scale: tuple[float, ...] = eqx.field(static=True)
    energy_scale: float = eqx.field(static=True)
    setpoint_input_index: int = eqx.field(static=True)
    outdoor_input_index: int = eqx.field(static=True)
    solar_input_index: int = eqx.field(static=True)
    ventilation_input_index: int = eqx.field(static=True)
    availability_input_index: int = eqx.field(static=True)
    temperature_target_index: int = eqx.field(static=True)
    qroom_target_index: int = eqx.field(static=True)
    pel_target_index: int = eqx.field(static=True)
    bptt_truncate_steps: int = eqx.field(static=True)
    transition_matrix_mode: Literal["fixed", "time_varying"] = eqx.field(static=True)
    transition_conditioning: TransitionConditioningMode = eqx.field(static=True)
    additive_feedback_gain_bound: float = eqx.field(static=True)
    transition_forcing_encoded_dim: int = eqx.field(static=True)
    hp_controller_masked_input_indices: tuple[int, ...] = eqx.field(static=True)
    thermal_dynamics_masked_input_indices: tuple[int, ...] = eqx.field(static=True)
    thermostat_demand_mode: ThermostatDemandMode = eqx.field(static=True)
    thermostat_slope_min: float = eqx.field(static=True)
    thermostat_slope_max: float = eqx.field(static=True)
    thermostat_threshold_min_c: float = eqx.field(static=True)
    thermostat_threshold_max_c: float = eqx.field(static=True)
    q_to_t_mode: ThermalQResponseMode = eqx.field(static=True)
    q_to_t_time_constants_hours: tuple[float, ...] = eqx.field(static=True)
    q_to_t_gain_min_c_per_w_m2: float = eqx.field(static=True)
    q_to_t_gain_max_c_per_w_m2: float = eqx.field(static=True)

    def __init__(
        self,
        metadata_dim: int,
        input_dim: int,
        *,
        state_dim: int = 6,
        controller_state_dim: int = 2,
        latent_dim: int = 4,
        hidden_dim: int = 64,
        depth: int = 3,
        input_encoder_dim: int | None = None,
        input_encoder_hidden_dim: int | None = None,
        input_encoder_depth: int = 2,
        process_noise_mode: ProcessNoiseMode = "constant",
        process_noise_init: float = -6.0,
        process_noise_floor: float = 1e-5,
        process_noise_cap: float = 0.25,
        hp_emission_mode: ProbHpEmissionMode = "bounded",
        hp_activation_model: HPActivationModel = "independent",
        hp_persistent_latent_dim: int = 2,
        hp_controller_leak: float = 0.25,
        hp_controller_noise_scale: float = 0.0,
        hp_history_hours: float = 3.0,
        hp_history_timescales_hours: tuple[float, ...] = (),
        hp_setpoint_shock_timescales_hours: tuple[float, ...] = (),
        contraction_gamma: float = 0.99,
        state_bound: float = 5.0,
        temperature_output_scale: float = 8.0,
        temperature_delta_max_c: float = 0.0,
        temperature_update_mode: TemperatureUpdateMode = "auto",
        hp_dt_hours: float = 0.25,
        hp_cop_floor: float = 1.0,
        hp_cop_cap: float = 0.0,
        hp_pel_cap_w_m2: float = 0.0,
        hp_qroom_cap_w_m2: float = 0.0,
        hp_energy_cap_wh_m2: float = 0.0,
        input_mean: tuple[float, ...] = (),
        input_scale: tuple[float, ...] = (),
        target_mean: tuple[float, ...] = (),
        target_scale: tuple[float, ...] = (),
        availability_input_index: int = -1,
        bptt_truncate_steps: int = DEFAULT_BPTT_TRUNCATE_STEPS,
        transition_matrix_mode: Literal["fixed", "time_varying"] = "time_varying",
        transition_conditioning: TransitionConditioningMode = "state_feedback",
        additive_feedback_gain_bound: float = 1.0,
        hp_controller_masked_input_indices: tuple[int, ...] = (),
        thermal_dynamics_masked_input_indices: tuple[int, ...] = (),
        thermostat_demand_mode: ThermostatDemandMode = "unconstrained",
        thermostat_slope_min: float = 0.1,
        thermostat_slope_max: float = 6.0,
        thermostat_threshold_min_c: float = -1.0,
        thermostat_threshold_max_c: float = 1.0,
        q_to_t_mode: ThermalQResponseMode = "unconstrained",
        q_to_t_time_constants_hours: tuple[float, ...] = (
            0.25,
            1.0,
            4.0,
            16.0,
            24.0,
            48.0,
        ),
        q_to_t_gain_min_c_per_w_m2: float = 0.01,
        q_to_t_gain_max_c_per_w_m2: float = 2.0,
        key: jax.Array,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if controller_state_dim < 1:
            raise ValueError("controller_state_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if input_encoder_dim is not None and input_encoder_dim < 1:
            raise ValueError("input_encoder_dim must be positive or None")
        if input_encoder_depth < 1:
            raise ValueError("input_encoder_depth must be at least 1")
        if process_noise_mode not in ("none", "constant", "heteroscedastic"):
            raise ValueError("process_noise_mode must be 'none', 'constant', or 'heteroscedastic'")
        if process_noise_cap <= 0.0:
            raise ValueError("process_noise_cap must be positive")
        if hp_emission_mode not in ("bounded", "legacy_lognormal_mean"):
            raise ValueError("hp_emission_mode must be 'bounded' or 'legacy_lognormal_mean'")
        if hp_activation_model not in (
            "independent",
            "persistent_markov",
            "power_history",
            "asymmetric_markov",
        ):
            raise ValueError(
                "hp_activation_model must be 'independent', 'persistent_markov', "
                "'power_history', or 'asymmetric_markov'"
            )
        if hp_persistent_latent_dim < 1:
            raise ValueError("hp_persistent_latent_dim must be positive")
        if not 0.0 < hp_controller_leak <= 1.0:
            raise ValueError("hp_controller_leak must be in (0, 1]")
        if hp_controller_noise_scale < 0.0:
            raise ValueError("hp_controller_noise_scale must be non-negative")
        if hp_history_hours <= 0.0:
            raise ValueError("hp_history_hours must be positive")
        hp_history_timescales_hours = tuple(
            float(value) for value in hp_history_timescales_hours
        )
        if any(value <= 0.0 for value in hp_history_timescales_hours):
            raise ValueError("hp_history_timescales_hours values must be positive")
        if len(set(hp_history_timescales_hours)) != len(hp_history_timescales_hours):
            raise ValueError("hp_history_timescales_hours values must be unique")
        hp_setpoint_shock_timescales_hours = tuple(
            float(value) for value in hp_setpoint_shock_timescales_hours
        )
        if any(value <= 0.0 for value in hp_setpoint_shock_timescales_hours):
            raise ValueError(
                "hp_setpoint_shock_timescales_hours values must be positive"
            )
        if len(set(hp_setpoint_shock_timescales_hours)) != len(
            hp_setpoint_shock_timescales_hours
        ):
            raise ValueError(
                "hp_setpoint_shock_timescales_hours values must be unique"
            )
        if not 0.0 < contraction_gamma < 1.0:
            raise ValueError("contraction_gamma must be in (0, 1)")
        if state_bound <= 0.0:
            raise ValueError("state_bound must be positive")
        if temperature_output_scale <= 0.0:
            raise ValueError("temperature_output_scale must be positive")
        if temperature_delta_max_c < 0.0:
            raise ValueError("temperature_delta_max_c must be non-negative")
        if temperature_update_mode not in (
            "auto",
            "absolute",
            "delta",
            "leaky_equilibrium",
            "bounded_equilibrium",
            "alpha_bounded_equilibrium",
        ):
            raise ValueError(
                "temperature_update_mode must be 'auto', 'absolute', 'delta', "
                "'leaky_equilibrium', 'bounded_equilibrium', or "
                "'alpha_bounded_equilibrium'"
            )
        if transition_matrix_mode not in ("fixed", "time_varying"):
            raise ValueError("transition_matrix_mode must be 'fixed' or 'time_varying'")
        if transition_conditioning not in (
            "state_feedback",
            "exogenous",
            "exogenous_additive_feedback",
        ):
            raise ValueError(
                "transition_conditioning must be 'state_feedback', 'exogenous', "
                "or 'exogenous_additive_feedback'"
            )
        if additive_feedback_gain_bound <= 0.0:
            raise ValueError("additive_feedback_gain_bound must be positive")
        if hp_dt_hours <= 0.0:
            raise ValueError("hp_dt_hours must be positive")
        if hp_cop_floor <= 0.0:
            raise ValueError("hp_cop_floor must be positive")
        if hp_cop_cap > 0.0 and hp_cop_cap <= hp_cop_floor:
            raise ValueError("hp_cop_cap must be larger than hp_cop_floor when enabled")
        if hp_pel_cap_w_m2 < 0.0:
            raise ValueError("hp_pel_cap_w_m2 must be non-negative")
        if hp_qroom_cap_w_m2 < 0.0:
            raise ValueError("hp_qroom_cap_w_m2 must be non-negative")
        if hp_energy_cap_wh_m2 < 0.0:
            raise ValueError("hp_energy_cap_wh_m2 must be non-negative")
        if thermostat_demand_mode not in ("unconstrained", "monotone"):
            raise ValueError("thermostat_demand_mode must be 'unconstrained' or 'monotone'")
        if thermostat_slope_min <= 0.0 or thermostat_slope_max <= thermostat_slope_min:
            raise ValueError("thermostat slope bounds must satisfy 0 < min < max")
        if thermostat_threshold_max_c <= thermostat_threshold_min_c:
            raise ValueError("thermostat threshold bounds must satisfy min < max")
        if q_to_t_mode not in ("unconstrained", "positive_leaky"):
            raise ValueError("q_to_t_mode must be 'unconstrained' or 'positive_leaky'")
        q_to_t_time_constants_hours = tuple(
            float(value) for value in q_to_t_time_constants_hours
        )
        if not q_to_t_time_constants_hours:
            raise ValueError("q_to_t_time_constants_hours must not be empty")
        if any(value <= 0.0 for value in q_to_t_time_constants_hours):
            raise ValueError("q_to_t_time_constants_hours values must be positive")
        if len(set(q_to_t_time_constants_hours)) != len(q_to_t_time_constants_hours):
            raise ValueError("q_to_t_time_constants_hours values must be unique")
        if q_to_t_gain_min_c_per_w_m2 < 0.0:
            raise ValueError("q_to_t_gain_min_c_per_w_m2 must be non-negative")
        if q_to_t_gain_max_c_per_w_m2 <= q_to_t_gain_min_c_per_w_m2:
            raise ValueError("q_to_t gain bounds must satisfy min < max")
        resolved_temperature_update = temperature_update_mode
        if resolved_temperature_update == "auto":
            resolved_temperature_update = "delta" if temperature_delta_max_c > 0.0 else "absolute"
        if (
            resolved_temperature_update in (
                "bounded_equilibrium",
                "alpha_bounded_equilibrium",
            )
            and temperature_delta_max_c <= 0.0
        ):
            raise ValueError(
                "bounded equilibrium temperature updates require "
                "temperature_delta_max_c > 0"
            )
        if q_to_t_mode == "positive_leaky" and resolved_temperature_update not in (
            "leaky_equilibrium",
            "bounded_equilibrium",
            "alpha_bounded_equilibrium",
        ):
            raise ValueError(
                "q_to_t_mode='positive_leaky' requires temperature_update_mode="
                "'leaky_equilibrium', 'bounded_equilibrium', or "
                "'alpha_bounded_equilibrium'"
            )
        bptt_truncate_steps = validate_bptt_truncate_steps(bptt_truncate_steps)
        if len(input_mean) != input_dim or len(input_scale) != input_dim:
            raise ValueError("input_mean and input_scale must match input_dim")
        if len(target_mean) != 3 or len(target_scale) != 3:
            raise ValueError("closed-loop probabilistic target scalers must have exactly 3 outputs")
        if availability_input_index >= input_dim:
            raise ValueError("availability_input_index must be smaller than input_dim")
        hp_controller_masked_input_indices = tuple(
            int(index) for index in hp_controller_masked_input_indices
        )
        if any(index < 0 or index >= input_dim for index in hp_controller_masked_input_indices):
            raise ValueError("hp_controller_masked_input_indices must be valid input indices")
        if len(set(hp_controller_masked_input_indices)) != len(
            hp_controller_masked_input_indices
        ):
            raise ValueError("hp_controller_masked_input_indices must be unique")
        thermal_dynamics_masked_input_indices = tuple(
            int(index) for index in thermal_dynamics_masked_input_indices
        )
        if any(index < 0 or index >= input_dim for index in thermal_dynamics_masked_input_indices):
            raise ValueError("thermal_dynamics_masked_input_indices must be valid input indices")
        if len(set(thermal_dynamics_masked_input_indices)) != len(
            thermal_dynamics_masked_input_indices
        ):
            raise ValueError("thermal_dynamics_masked_input_indices must be unique")

        (
            encoder_key,
            transition_key,
            x0_key,
            e0_key,
            hp_param_key,
            output_key,
            temperature_key,
            process_key,
        ) = jax.random.split(key, 8)
        alpha_key = jax.random.fold_in(temperature_key, 1)
        q_response_key = jax.random.fold_in(temperature_key, 2)
        q_response_init_key = jax.random.fold_in(temperature_key, 3)
        transition_forcing_key = jax.random.fold_in(transition_key, 1)
        encoded_input_dim = input_dim if input_encoder_dim is None else input_encoder_dim
        thermal_forcing_dim = input_dim + (1 if q_to_t_mode == "positive_leaky" else 2)
        transition_forcing_encoded_dim = (
            thermal_forcing_dim if input_encoder_dim is None else input_encoder_dim
        )
        uses_persistent_controller = hp_activation_model == "persistent_markov"
        uses_asymmetric_markov = hp_activation_model == "asymmetric_markov"
        uses_markov_transitions = uses_persistent_controller or uses_asymmetric_markov
        uses_power_history = hp_activation_model in ("power_history", "asymmetric_markov")
        hp_history_feature_dim = (
            1 + len(hp_history_timescales_hours)
            if hp_history_timescales_hours
            else 1
        )
        hp_setpoint_shock_feature_dim = 2 * len(
            hp_setpoint_shock_timescales_hours
        )
        hp_latent_dim = hp_persistent_latent_dim if uses_persistent_controller else 0
        generator_input_dim = metadata_dim + latent_dim
        transition_input_dim = generator_input_dim + thermal_forcing_dim
        transition_output_dim = state_dim * state_dim + state_dim
        if transition_conditioning == "exogenous_additive_feedback":
            transition_output_dim += state_dim * transition_forcing_encoded_dim
        output_input_dim = (
            generator_input_dim
            + encoded_input_dim
            + 3
            + hp_latent_dim
            + (controller_state_dim if uses_persistent_controller else 0)
            + (2 * hp_history_feature_dim if uses_power_history else 0)
            + hp_setpoint_shock_feature_dim
        )

        self.input_encoder = None
        if input_encoder_dim is not None:
            self.input_encoder = MLP(
                input_dim,
                input_encoder_dim,
                hidden_dim=input_encoder_hidden_dim or hidden_dim,
                depth=input_encoder_depth,
                key=encoder_key,
            )
        self.transition_net = MLP(
            transition_input_dim,
            transition_output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=transition_key,
        )
        self.transition_forcing_encoder = None
        if transition_conditioning == "exogenous_additive_feedback":
            self.transition_forcing_encoder = MLP(
                thermal_forcing_dim,
                transition_forcing_encoded_dim,
                hidden_dim=input_encoder_hidden_dim or hidden_dim,
                depth=input_encoder_depth,
                key=transition_forcing_key,
            )
        self.x0_net = MLP(metadata_dim + 1 + latent_dim, state_dim, hidden_dim=hidden_dim, depth=depth, key=x0_key)
        self.e0_net = MLP(metadata_dim + 1 + latent_dim, 1, hidden_dim=hidden_dim, depth=depth, key=e0_key)
        self.hp_param_net = MLP(
            generator_input_dim + hp_latent_dim,
            3,
            hidden_dim=hidden_dim,
            depth=depth,
            key=hp_param_key,
        )
        output_dim = 4 if thermostat_demand_mode == "unconstrained" else 6
        self.output_net = MLP(
            output_input_dim,
            output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=output_key,
        )
        self.temperature_net = MLP(
            state_dim + latent_dim,
            1,
            hidden_dim=hidden_dim,
            depth=depth,
            key=temperature_key,
        )
        self.temperature_alpha_net = None
        if temperature_update_mode in (
            "leaky_equilibrium",
            "alpha_bounded_equilibrium",
        ):
            self.temperature_alpha_net = MLP(
                generator_input_dim,
                1,
                hidden_dim=hidden_dim,
                depth=depth,
                key=alpha_key,
            )
        self.q_response_net = None
        self.q_response_init_net = None
        if q_to_t_mode == "positive_leaky":
            q_response_dim = len(q_to_t_time_constants_hours)
            self.q_response_net = MLP(
                generator_input_dim,
                q_response_dim + 1,
                hidden_dim=hidden_dim,
                depth=depth,
                key=q_response_key,
            )
            self.q_response_init_net = MLP(
                generator_input_dim + 1,
                q_response_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                key=q_response_init_key,
            )
        self.process_noise_net = None
        if process_noise_mode == "heteroscedastic":
            self.process_noise_net = MLP(
                generator_input_dim + thermal_forcing_dim + state_dim,
                state_dim,
                hidden_dim=hidden_dim,
                depth=2,
                key=process_key,
            )

        self.hp_controller_init_net = None
        self.hp_controller_transition_net = None
        self.hp_transition_net = None
        self.hp_mode0_net = None
        if uses_persistent_controller:
            controller_init_key = jax.random.fold_in(output_key, 101)
            controller_transition_key = jax.random.fold_in(output_key, 102)
            hp_mode0_key = jax.random.fold_in(output_key, 104)
            initial_controller_input_dim = metadata_dim + hp_latent_dim + 1
            self.hp_controller_init_net = MLP(
                initial_controller_input_dim,
                controller_state_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                key=controller_init_key,
            )
            self.hp_controller_transition_net = MLP(
                output_input_dim + 1,
                controller_state_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                key=controller_transition_key,
            )
            self.hp_mode0_net = MLP(
                initial_controller_input_dim,
                1,
                hidden_dim=hidden_dim,
                depth=depth,
                key=hp_mode0_key,
            )
        if uses_markov_transitions:
            self.hp_transition_net = MLP(
                output_input_dim,
                2,
                hidden_dim=hidden_dim,
                depth=depth,
                key=jax.random.fold_in(output_key, 103),
            )

        self.raw_process_scale = jnp.full((state_dim,), process_noise_init)
        self.metadata_dim = metadata_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.encoded_input_dim = encoded_input_dim
        self.thermal_forcing_dim = thermal_forcing_dim
        self.controller_state_dim = controller_state_dim
        self.latent_dim = latent_dim
        self.process_noise_mode = process_noise_mode
        self.process_noise_init = process_noise_init
        self.process_noise_floor = process_noise_floor
        self.process_noise_cap = process_noise_cap
        self.hp_emission_mode = hp_emission_mode
        self.hp_activation_model = hp_activation_model
        self.hp_persistent_latent_dim = hp_persistent_latent_dim
        self.hp_controller_leak = hp_controller_leak
        self.hp_controller_noise_scale = hp_controller_noise_scale
        self.hp_history_hours = hp_history_hours
        self.hp_history_timescales_hours = hp_history_timescales_hours
        self.hp_setpoint_shock_timescales_hours = (
            hp_setpoint_shock_timescales_hours
        )
        self.contraction_gamma = contraction_gamma
        self.state_bound = state_bound
        self.temperature_output_scale = temperature_output_scale
        self.temperature_delta_max_c = temperature_delta_max_c
        self.temperature_update_mode = temperature_update_mode
        self.hp_dt_hours = hp_dt_hours
        self.hp_cop_floor = hp_cop_floor
        self.hp_cop_cap = hp_cop_cap
        self.hp_pel_cap_w_m2 = hp_pel_cap_w_m2
        self.hp_qroom_cap_w_m2 = hp_qroom_cap_w_m2
        self.hp_energy_cap_wh_m2 = hp_energy_cap_wh_m2
        self.input_mean = tuple(float(value) for value in input_mean)
        self.input_scale = tuple(float(value) for value in input_scale)
        self.target_mean = tuple(float(value) for value in target_mean)
        self.target_scale = tuple(float(value) for value in target_scale)
        self.energy_scale = max(float(target_scale[1]) * 24.0, 1.0)
        self.setpoint_input_index = 0
        self.outdoor_input_index = 1
        self.solar_input_index = 2
        self.ventilation_input_index = 3
        self.availability_input_index = availability_input_index
        self.temperature_target_index = 0
        self.qroom_target_index = 1
        self.pel_target_index = 2
        self.bptt_truncate_steps = bptt_truncate_steps
        self.transition_matrix_mode = transition_matrix_mode
        self.transition_conditioning = transition_conditioning
        self.additive_feedback_gain_bound = float(additive_feedback_gain_bound)
        self.transition_forcing_encoded_dim = transition_forcing_encoded_dim
        self.hp_controller_masked_input_indices = hp_controller_masked_input_indices
        self.thermal_dynamics_masked_input_indices = thermal_dynamics_masked_input_indices
        self.thermostat_demand_mode = thermostat_demand_mode
        self.thermostat_slope_min = float(thermostat_slope_min)
        self.thermostat_slope_max = float(thermostat_slope_max)
        self.thermostat_threshold_min_c = float(thermostat_threshold_min_c)
        self.thermostat_threshold_max_c = float(thermostat_threshold_max_c)
        self.q_to_t_mode = q_to_t_mode
        self.q_to_t_time_constants_hours = q_to_t_time_constants_hours
        self.q_to_t_gain_min_c_per_w_m2 = float(q_to_t_gain_min_c_per_w_m2)
        self.q_to_t_gain_max_c_per_w_m2 = float(q_to_t_gain_max_c_per_w_m2)

    def encode_input(self, input_t: jnp.ndarray) -> jnp.ndarray:
        if self.thermostat_demand_mode == "monotone":
            input_t = input_t.at[self.setpoint_input_index].set(0.0)
        for index in self.hp_controller_masked_input_indices:
            input_t = input_t.at[index].set(0.0)
        if self.input_encoder is None:
            return input_t
        return self.input_encoder(input_t)

    def _physical_input(self, input_t: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            input_t[index] * jnp.asarray(self.input_scale[index], dtype=input_t.dtype)
            + jnp.asarray(self.input_mean[index], dtype=input_t.dtype)
        )

    def _target_scaled(self, value: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            value - jnp.asarray(self.target_mean[index], dtype=value.dtype)
        ) / jnp.asarray(self.target_scale[index], dtype=value.dtype)

    def _space_heating_availability(self, input_t: jnp.ndarray) -> jnp.ndarray:
        if self.availability_input_index < 0:
            return jnp.asarray(1.0, dtype=input_t.dtype)
        return jnp.clip(
            self._physical_input(input_t, self.availability_input_index),
            jnp.asarray(0.0, dtype=input_t.dtype),
            jnp.asarray(1.0, dtype=input_t.dtype),
        )

    def _temperature_output(
        self,
        raw_temperature: jnp.ndarray,
        previous_temperature_scaled: jnp.ndarray,
        alpha_t: jnp.ndarray,
        q_response_temperature_c: jnp.ndarray,
    ) -> jnp.ndarray:
        bound = jnp.asarray(self.temperature_output_scale, dtype=raw_temperature.dtype)
        mode = self._resolved_temperature_update_mode()
        if mode == "absolute":
            return bound * jnp.tanh(raw_temperature)
        if mode in (
            "leaky_equilibrium",
            "bounded_equilibrium",
            "alpha_bounded_equilibrium",
        ):
            target_scale = jnp.maximum(
                jnp.abs(
                    jnp.asarray(
                        self.target_scale[self.temperature_target_index],
                        dtype=raw_temperature.dtype,
                    )
                ),
                jnp.asarray(1e-6, dtype=raw_temperature.dtype),
            )
            equilibrium_scaled = jnp.clip(
                bound * jnp.tanh(raw_temperature)
                + q_response_temperature_c / target_scale,
                -bound,
                bound,
            )
            if mode == "leaky_equilibrium":
                return (
                    (1.0 - alpha_t) * previous_temperature_scaled
                    + alpha_t * equilibrium_scaled
                )
            delta_bound = (
                jnp.asarray(self.temperature_delta_max_c, dtype=raw_temperature.dtype)
                / target_scale
            )
            equilibrium_error = equilibrium_scaled - previous_temperature_scaled
            if mode == "alpha_bounded_equilibrium":
                equilibrium_error = alpha_t * equilibrium_error
            temperature_scaled = previous_temperature_scaled + delta_bound * jnp.tanh(
                equilibrium_error / delta_bound
            )
            return jnp.clip(temperature_scaled, -bound, bound)
        target_scale = jnp.maximum(
            jnp.abs(jnp.asarray(self.target_scale[self.temperature_target_index], dtype=raw_temperature.dtype)),
            jnp.asarray(1e-6, dtype=raw_temperature.dtype),
        )
        delta_bound = jnp.asarray(self.temperature_delta_max_c, dtype=raw_temperature.dtype) / target_scale
        temperature_scaled = previous_temperature_scaled + delta_bound * jnp.tanh(raw_temperature)
        return jnp.clip(temperature_scaled, -bound, bound)

    def _resolved_temperature_update_mode(self) -> TemperatureUpdateMode:
        if self.temperature_update_mode != "auto":
            return self.temperature_update_mode
        if self.temperature_delta_max_c > 0.0:
            return "delta"
        return "absolute"

    def temperature_alpha_max(self, dtype: jnp.dtype) -> jnp.ndarray:
        if self.temperature_delta_max_c <= 0.0:
            return jnp.asarray(1.0, dtype=dtype)
        target_scale = jnp.maximum(
            jnp.abs(jnp.asarray(self.target_scale[self.temperature_target_index], dtype=dtype)),
            jnp.asarray(1e-6, dtype=dtype),
        )
        bound = jnp.asarray(self.temperature_output_scale, dtype=dtype)
        delta_bound = jnp.asarray(self.temperature_delta_max_c, dtype=dtype) / target_scale
        alpha_max = delta_bound / (2.0 * bound)
        return jnp.clip(alpha_max, jnp.asarray(1e-6, dtype=dtype), jnp.asarray(1.0, dtype=dtype))

    def temperature_alpha(self, metadata: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
        if self.temperature_alpha_net is None:
            return jnp.asarray(1.0, dtype=metadata.dtype)
        raw_alpha = self.temperature_alpha_net(jnp.concatenate([metadata, xi], axis=0))[0]
        if self._resolved_temperature_update_mode() == "alpha_bounded_equilibrium":
            return jax.nn.sigmoid(raw_alpha)
        return self.temperature_alpha_max(raw_alpha.dtype) * jax.nn.sigmoid(raw_alpha)

    def _temperature_physical(self, temperature_scaled: jnp.ndarray) -> jnp.ndarray:
        return (
            temperature_scaled
            * jnp.asarray(self.target_scale[self.temperature_target_index], dtype=temperature_scaled.dtype)
            + jnp.asarray(self.target_mean[self.temperature_target_index], dtype=temperature_scaled.dtype)
        )

    def _energy_scaled(self, energy_t: jnp.ndarray) -> jnp.ndarray:
        return energy_t / jnp.asarray(max(self.energy_scale, 1.0), dtype=energy_t.dtype)

    def _setpoint_gap_scaled(
        self,
        input_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> jnp.ndarray:
        setpoint_c = self._physical_input(input_t, self.setpoint_input_index)
        temperature_c = self._temperature_physical(temperature_t)
        target_scale = jnp.maximum(
            jnp.abs(jnp.asarray(self.target_scale[self.temperature_target_index], dtype=input_t.dtype)),
            jnp.asarray(1e-6, dtype=input_t.dtype),
        )
        return (setpoint_c - temperature_c) / target_scale

    def _setpoint_gap_c(
        self,
        input_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._physical_input(input_t, self.setpoint_input_index) - self._temperature_physical(
            temperature_t
        )

    def _thermostat_slope(self, raw: jnp.ndarray) -> jnp.ndarray:
        lower = jnp.asarray(self.thermostat_slope_min, dtype=raw.dtype)
        width = jnp.asarray(
            self.thermostat_slope_max - self.thermostat_slope_min,
            dtype=raw.dtype,
        )
        return lower + width * jax.nn.sigmoid(raw)

    def _thermostat_threshold(self, raw: jnp.ndarray) -> jnp.ndarray:
        lower = jnp.asarray(self.thermostat_threshold_min_c, dtype=raw.dtype)
        width = jnp.asarray(
            self.thermostat_threshold_max_c - self.thermostat_threshold_min_c,
            dtype=raw.dtype,
        )
        return lower + width * jax.nn.sigmoid(raw)

    def _outdoor_gap_scaled(
        self,
        input_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> jnp.ndarray:
        outdoor_c = self._physical_input(input_t, self.outdoor_input_index)
        temperature_c = self._temperature_physical(temperature_t)
        target_scale = jnp.maximum(
            jnp.abs(jnp.asarray(self.target_scale[self.temperature_target_index], dtype=input_t.dtype)),
            jnp.asarray(1e-6, dtype=input_t.dtype),
        )
        return (outdoor_c - temperature_c) / target_scale

    def hp_features(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
        input_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
        controller_state_t: jnp.ndarray | None = None,
        hp_latent: jnp.ndarray | None = None,
        hp_power_history_t: jnp.ndarray | None = None,
        hp_mode_history_t: jnp.ndarray | None = None,
        hp_setpoint_shock_t: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        gap_feature = self._setpoint_gap_scaled(input_t, temperature_t)
        values = [
            metadata,
            xi,
            encoded_input_t,
            jnp.asarray([temperature_t], dtype=encoded_input_t.dtype),
            jnp.asarray([gap_feature], dtype=encoded_input_t.dtype),
            jnp.asarray([self._energy_scaled(energy_t)], dtype=encoded_input_t.dtype),
        ]
        if self.hp_activation_model == "persistent_markov":
            if controller_state_t is None or hp_latent is None:
                raise ValueError("persistent_markov HP activation requires controller state and HP latent")
            values.extend([hp_latent, controller_state_t])
        elif self.hp_activation_model in ("power_history", "asymmetric_markov"):
            if hp_power_history_t is None or hp_mode_history_t is None:
                raise ValueError(
                    f"{self.hp_activation_model} HP activation requires both history states"
                )
            values.extend(
                [
                    jnp.ravel(hp_power_history_t).astype(encoded_input_t.dtype),
                    jnp.ravel(hp_mode_history_t).astype(encoded_input_t.dtype),
                ]
            )
        shock_feature_dim = 2 * len(self.hp_setpoint_shock_timescales_hours)
        if shock_feature_dim:
            if hp_setpoint_shock_t is None:
                hp_setpoint_shock_t = jnp.zeros(
                    (shock_feature_dim,), dtype=encoded_input_t.dtype
                )
            if hp_setpoint_shock_t.shape != (shock_feature_dim,):
                raise ValueError(
                    "hp_setpoint_shock_t must contain one positive and one negative "
                    "trace per configured timescale"
                )
            values.append(hp_setpoint_shock_t.astype(encoded_input_t.dtype))
        return jnp.concatenate(values, axis=0)

    def initial_controller_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        hp_latent: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.hp_activation_model != "persistent_markov":
            return (
                jnp.zeros((self.controller_state_dim,), dtype=metadata.dtype),
                jnp.asarray(0.0, dtype=metadata.dtype),
            )
        assert self.hp_controller_init_net is not None
        assert self.hp_mode0_net is not None
        features = jnp.concatenate([metadata, hp_latent, initial_temperature], axis=0)
        controller_state = jnp.tanh(self.hp_controller_init_net(features))
        previous_mode = jax.nn.sigmoid(self.hp_mode0_net(features)[0])
        return controller_state, previous_mode

    def initial_asymmetric_markov_mode(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        input_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
        hp_power_history_t: jnp.ndarray,
        hp_mode_history_t: jnp.ndarray,
        hp_setpoint_shock_t: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Infer initial occupancy from the local two-state stationary distribution."""
        if self.hp_activation_model != "asymmetric_markov":
            return jnp.asarray(0.0, dtype=input_t.dtype)
        assert self.hp_transition_net is not None
        features = self.hp_features(
            metadata,
            xi,
            self.encode_input(input_t),
            input_t,
            energy_t,
            temperature_t,
            hp_power_history_t=hp_power_history_t,
            hp_mode_history_t=hp_mode_history_t,
            hp_setpoint_shock_t=hp_setpoint_shock_t,
        )
        start_logit, stop_logit = self.hp_transition_net(features)
        p_start = jax.nn.sigmoid(start_logit)
        p_stop = jax.nn.sigmoid(stop_logit)
        eps = jnp.asarray(1e-6, dtype=input_t.dtype)
        stationary_on = p_start / jnp.maximum(p_start + p_stop, eps)
        return self._space_heating_availability(input_t) * stationary_on

    def next_controller_state(
        self,
        hp_features_t: jnp.ndarray,
        hp_on_t: jnp.ndarray,
        controller_state_t: jnp.ndarray,
        controller_noise_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.hp_activation_model != "persistent_markov":
            return controller_state_t
        assert self.hp_controller_transition_net is not None
        raw_candidate = self.hp_controller_transition_net(
            jnp.concatenate(
                [hp_features_t, jnp.asarray([hp_on_t], dtype=hp_features_t.dtype)],
                axis=0,
            )
        )
        candidate = jnp.tanh(
            raw_candidate
            + jnp.asarray(self.hp_controller_noise_scale, dtype=raw_candidate.dtype)
            * controller_noise_t
        )
        leak = jnp.asarray(self.hp_controller_leak, dtype=raw_candidate.dtype)
        return (1.0 - leak) * controller_state_t + leak * candidate

    def hp_history_alpha(self, dtype: jnp.dtype) -> jnp.ndarray:
        dt_hours = jnp.asarray(self.hp_dt_hours, dtype=dtype)
        history_hours = jnp.asarray(self.hp_history_hours, dtype=dtype)
        return -jnp.expm1(-dt_hours / history_hours)

    def setpoint_shock_features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Return causal positive/negative setpoint-shock traces.

        Each trace receives the normalized setpoint jump at the current step
        and subsequently decays with its configured physical time constant.
        The first step has no preceding observation and therefore no shock.
        """
        timescales = self.hp_setpoint_shock_timescales_hours
        if not timescales:
            return jnp.zeros((inputs.shape[0], 0), dtype=inputs.dtype)

        setpoint = inputs[:, self.setpoint_input_index]
        delta = jnp.concatenate(
            [jnp.zeros((1,), dtype=inputs.dtype), jnp.diff(setpoint)], axis=0
        )
        positive = jnp.maximum(delta, jnp.asarray(0.0, dtype=inputs.dtype))
        negative = jnp.maximum(-delta, jnp.asarray(0.0, dtype=inputs.dtype))
        timescale_values = jnp.asarray(timescales, dtype=inputs.dtype)
        retention = jnp.exp(
            -jnp.asarray(self.hp_dt_hours, dtype=inputs.dtype) / timescale_values
        )

        def step(
            traces: tuple[jnp.ndarray, jnp.ndarray],
            shocks: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            positive_trace, negative_trace = traces
            positive_shock, negative_shock = shocks
            positive_next = retention * positive_trace + positive_shock
            negative_next = retention * negative_trace + negative_shock
            next_traces = (positive_next, negative_next)
            return next_traces, jnp.concatenate(next_traces, axis=0)

        initial = (
            jnp.zeros_like(retention),
            jnp.zeros_like(retention),
        )
        _, traces = jax.lax.scan(step, initial, (positive, negative))
        return traces

    def initial_hp_history(self, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not self.hp_history_timescales_hours:
            zero = jnp.asarray(0.0, dtype=dtype)
            return zero, zero
        shape = (1 + len(self.hp_history_timescales_hours),)
        return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)

    def next_hp_history(
        self,
        hp_power_history_t: jnp.ndarray,
        hp_mode_history_t: jnp.ndarray,
        pel_t: jnp.ndarray,
        hp_on_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha = self.hp_history_alpha(pel_t.dtype)
        power_fraction = jnp.clip(
            pel_t / self._max_active_power(pel_t.dtype),
            jnp.asarray(0.0, dtype=pel_t.dtype),
            jnp.asarray(1.0, dtype=pel_t.dtype),
        )
        mode_fraction = jnp.clip(
            hp_on_t,
            jnp.asarray(0.0, dtype=hp_on_t.dtype),
            jnp.asarray(1.0, dtype=hp_on_t.dtype),
        )
        if self.hp_history_timescales_hours:
            timescales = jnp.asarray(self.hp_history_timescales_hours, dtype=pel_t.dtype)
            dt_hours = jnp.asarray(self.hp_dt_hours, dtype=pel_t.dtype)
            alphas = -jnp.expm1(-dt_hours / timescales)
            power_ewma = (
                (1.0 - alphas) * hp_power_history_t[1:]
                + alphas * power_fraction
            )
            mode_ewma = (
                (1.0 - alphas) * hp_mode_history_t[1:]
                + alphas * mode_fraction
            )
            power_next = jnp.concatenate([power_fraction[jnp.newaxis], power_ewma])
            mode_next = jnp.concatenate([mode_fraction[jnp.newaxis], mode_ewma])
            return jnp.clip(power_next, 0.0, 1.0), jnp.clip(mode_next, 0.0, 1.0)
        power_next = (1.0 - alpha) * hp_power_history_t + alpha * power_fraction
        mode_next = (1.0 - alpha) * hp_mode_history_t + alpha * mode_fraction
        return jnp.clip(power_next, 0.0, 1.0), jnp.clip(mode_next, 0.0, 1.0)

    def thermal_forcing(
        self,
        input_t: jnp.ndarray,
        qroom_scaled_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> jnp.ndarray:
        thermal_inputs = input_t[1:]
        if self.availability_input_index >= 1:
            thermal_inputs = thermal_inputs.at[self.availability_input_index - 1].set(0.0)
        for index in self.thermal_dynamics_masked_input_indices:
            if index >= 1:
                thermal_inputs = thermal_inputs.at[index - 1].set(0.0)
        if self.q_to_t_mode == "positive_leaky":
            return jnp.concatenate(
                [
                    thermal_inputs,
                    jnp.asarray([temperature_t], dtype=input_t.dtype),
                    jnp.asarray(
                        [self._outdoor_gap_scaled(input_t, temperature_t)],
                        dtype=input_t.dtype,
                    ),
                ],
                axis=0,
            )
        return jnp.concatenate(
            [
                thermal_inputs,
                jnp.asarray([qroom_scaled_t], dtype=input_t.dtype),
                jnp.asarray([temperature_t], dtype=input_t.dtype),
                jnp.asarray([self._outdoor_gap_scaled(input_t, temperature_t)], dtype=input_t.dtype),
            ],
            axis=0,
        )

    def decode_temperature(
        self,
        state_t: jnp.ndarray,
        xi: jnp.ndarray,
        previous_temperature_scaled: jnp.ndarray,
        alpha_t: jnp.ndarray,
        q_response_state_t: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        features = jnp.concatenate(
            [
                state_t / jnp.asarray(self.state_bound, dtype=state_t.dtype),
                xi,
            ],
            axis=0,
        )
        raw_temperature = self.temperature_net(features)[0]
        q_response_temperature_c = (
            jnp.asarray(0.0, dtype=state_t.dtype)
            if q_response_state_t is None
            else jnp.sum(q_response_state_t)
        )
        return self._temperature_output(
            raw_temperature,
            previous_temperature_scaled,
            alpha_t,
            q_response_temperature_c,
        )

    def q_response_parameters(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.q_response_net is None:
            return (
                jnp.asarray(0.0, dtype=metadata.dtype),
                jnp.zeros((0,), dtype=metadata.dtype),
            )
        raw = self.q_response_net(jnp.concatenate([metadata, xi], axis=0))
        lower = jnp.asarray(self.q_to_t_gain_min_c_per_w_m2, dtype=raw.dtype)
        width = jnp.asarray(
            self.q_to_t_gain_max_c_per_w_m2
            - self.q_to_t_gain_min_c_per_w_m2,
            dtype=raw.dtype,
        )
        total_gain = lower + width * jax.nn.sigmoid(raw[0] - 3.0)
        mode_weights = jax.nn.softmax(raw[1:])
        return total_gain, mode_weights

    def initial_q_response_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        xi: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.q_response_init_net is None:
            return jnp.zeros((0,), dtype=metadata.dtype)
        total_gain, mode_weights = self.q_response_parameters(metadata, xi)
        raw = self.q_response_init_net(
            jnp.concatenate([metadata, initial_temperature, xi], axis=0)
        )
        reference_qroom = jnp.maximum(
            jnp.asarray(self.target_mean[self.qroom_target_index], dtype=raw.dtype),
            jnp.asarray(self.target_scale[self.qroom_target_index], dtype=raw.dtype),
        )
        reference_qroom = jnp.maximum(reference_qroom, jnp.asarray(1.0, dtype=raw.dtype))
        return (
            mode_weights
            * total_gain
            * reference_qroom
            * jax.nn.sigmoid(raw - 4.0)
        )

    def next_q_response_state(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        q_response_state_t: jnp.ndarray,
        qroom_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.q_response_net is None:
            return q_response_state_t
        total_gain, mode_weights = self.q_response_parameters(metadata, xi)
        time_constants = jnp.asarray(
            self.q_to_t_time_constants_hours,
            dtype=qroom_t.dtype,
        )
        retention = jnp.exp(
            -jnp.asarray(self.hp_dt_hours, dtype=qroom_t.dtype) / time_constants
        )
        equilibrium_temperature_c = (
            mode_weights * total_gain * jnp.maximum(qroom_t, 0.0)
        )
        return (
            retention * q_response_state_t
            + (1.0 - retention) * equilibrium_temperature_c
        )

    def _nonnegative_capped(self, value: jnp.ndarray, cap: float) -> jnp.ndarray:
        value = jnp.maximum(value, jnp.asarray(0.0, dtype=value.dtype))
        if cap <= 0.0:
            return value
        return jnp.minimum(value, jnp.asarray(cap, dtype=value.dtype))

    def _positive_capped_from_logit(
        self,
        logit: jnp.ndarray,
        cap: float,
        target_index: int,
    ) -> jnp.ndarray:
        if cap > 0.0:
            return jnp.asarray(cap, dtype=logit.dtype) * jax.nn.sigmoid(logit)
        target_mean = jnp.asarray(self.target_mean[target_index], dtype=logit.dtype)
        target_scale = jnp.asarray(self.target_scale[target_index], dtype=logit.dtype)
        fallback_cap = jnp.maximum(
            target_mean + jnp.asarray(8.0, dtype=logit.dtype) * target_scale,
            jnp.asarray(2.0, dtype=logit.dtype) * target_scale,
        )
        return fallback_cap * jax.nn.sigmoid(logit)

    def hp_parameters(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        hp_latent: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = [metadata, xi]
        if self.hp_activation_model == "persistent_markov":
            if hp_latent is None:
                raise ValueError("persistent_markov HP parameters require HP latent")
            values.append(hp_latent)
        raw = self.hp_param_net(jnp.concatenate(values, axis=0))
        cop_intercept = jnp.asarray(3.0, dtype=raw.dtype) + jnp.asarray(0.1, dtype=raw.dtype) * raw[0]
        cop_slope = jnp.asarray(0.02, dtype=raw.dtype) + jnp.asarray(0.005, dtype=raw.dtype) * raw[1]
        loss_rate = jax.nn.softplus(jnp.asarray(-4.0, dtype=raw.dtype) + 0.1 * raw[2])
        return cop_intercept, cop_slope, loss_rate

    def _cop(
        self,
        input_t: jnp.ndarray,
        cop_intercept: jnp.ndarray,
        cop_slope: jnp.ndarray,
    ) -> jnp.ndarray:
        outdoor_c = self._physical_input(input_t, self.outdoor_input_index)
        cop = jnp.maximum(self.hp_cop_floor, cop_intercept + cop_slope * outdoor_c)
        if self.hp_cop_cap <= 0.0:
            return cop
        return jnp.minimum(cop, jnp.asarray(self.hp_cop_cap, dtype=input_t.dtype))

    def _buffer_draw(
        self,
        energy_t: jnp.ndarray,
        generated_heat_power_t: jnp.ndarray,
        requested_room_heat_t: jnp.ndarray,
        dt_h: jnp.ndarray,
        loss_rate: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        energy_charged_t = self._nonnegative_capped(
            energy_t + dt_h * generated_heat_power_t,
            self.hp_energy_cap_wh_m2,
        )
        loss_fraction_t = 1.0 - jnp.exp(-jnp.maximum(loss_rate, 0.0) * dt_h)
        energy_available_t = self._nonnegative_capped(
            energy_charged_t * (1.0 - loss_fraction_t),
            self.hp_energy_cap_wh_m2,
        )
        available_power_t = jnp.maximum(energy_available_t / dt_h, 0.0)
        qroom_t = self._nonnegative_capped(
            jnp.minimum(requested_room_heat_t, available_power_t),
            self.hp_qroom_cap_w_m2,
        )
        energy_next = self._nonnegative_capped(
            energy_available_t - dt_h * qroom_t,
            self.hp_energy_cap_wh_m2,
        )
        return qroom_t, available_power_t, energy_next

    def initial_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        xi: jnp.ndarray,
    ) -> jnp.ndarray:
        raw = self.x0_net(jnp.concatenate([metadata, initial_temperature, xi], axis=0))
        return jnp.asarray(self.state_bound, dtype=raw.dtype) * jnp.tanh(raw)

    def initial_energy(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        xi: jnp.ndarray,
    ) -> jnp.ndarray:
        raw = self.e0_net(jnp.concatenate([metadata, initial_temperature, xi], axis=0))[0]
        energy = jax.nn.softplus(raw) * jnp.asarray(self.energy_scale, dtype=raw.dtype)
        return self._nonnegative_capped(energy, self.hp_energy_cap_wh_m2)

    def transition_matrix(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        forcing = self._matrix_forcing(thermal_forcing_t, dtype=metadata.dtype)
        matrix, _ = self._transition_from_forcing(metadata, xi, forcing)
        return matrix

    def _matrix_forcing(
        self,
        thermal_forcing_t: jnp.ndarray | None,
        *,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        if self.transition_matrix_mode == "fixed":
            return jnp.zeros((self.thermal_forcing_dim,), dtype=dtype)
        if thermal_forcing_t is None:
            raise ValueError(
                "thermal_forcing_t is required for a time-varying transition matrix"
            )
        return thermal_forcing_t

    def _transition_from_forcing(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        generator_forcing_t = self._transition_forcing(thermal_forcing_t)
        raw = self.transition_net(
            jnp.concatenate([metadata, xi, generator_forcing_t], axis=0)
        )
        matrix_size = self.state_dim * self.state_dim
        raw_matrix = raw[:matrix_size].reshape((self.state_dim, self.state_dim))
        if self.transition_conditioning == "exogenous_additive_feedback":
            input_matrix_size = self.state_dim * self.transition_forcing_encoded_dim
            input_matrix_end = matrix_size + input_matrix_size
            raw_input_matrix = raw[matrix_size:input_matrix_end].reshape(
                (self.state_dim, self.transition_forcing_encoded_dim)
            )
            raw_bias = raw[input_matrix_end:]
        else:
            raw_input_matrix = None
            raw_bias = raw[matrix_size:]
        frobenius = jnp.sqrt(
            jnp.sum(raw_matrix**2) + jnp.asarray(1e-12, dtype=raw_matrix.dtype)
        )
        divisor = jnp.maximum(frobenius, jnp.asarray(1.0, dtype=raw_matrix.dtype))
        matrix = (
            jnp.asarray(self.contraction_gamma, dtype=raw_matrix.dtype)
            * raw_matrix
            / divisor
        )
        bias = jnp.tanh(raw_bias)
        if raw_input_matrix is not None:
            assert self.transition_forcing_encoder is not None
            input_frobenius = jnp.sqrt(
                jnp.sum(raw_input_matrix**2)
                + jnp.asarray(1e-12, dtype=raw_input_matrix.dtype)
            )
            input_divisor = jnp.maximum(
                input_frobenius,
                jnp.asarray(1.0, dtype=raw_input_matrix.dtype),
            )
            input_matrix = (
                jnp.asarray(
                    self.additive_feedback_gain_bound,
                    dtype=raw_input_matrix.dtype,
                )
                * raw_input_matrix
                / input_divisor
            )
            encoded_forcing = jnp.tanh(
                self.transition_forcing_encoder(thermal_forcing_t)
            )
            bias = bias + input_matrix @ encoded_forcing
        return matrix, bias

    def _transition_forcing(self, thermal_forcing_t: jnp.ndarray) -> jnp.ndarray:
        if self.transition_conditioning == "state_feedback":
            return thermal_forcing_t
        endogenous_count = 2 if self.q_to_t_mode == "positive_leaky" else 3
        return thermal_forcing_t.at[-endogenous_count:].set(0.0)

    def transition_bias(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
    ) -> jnp.ndarray:
        _, bias = self._transition_from_forcing(metadata, xi, thermal_forcing_t)
        return bias

    def transition_matrix_and_bias(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.transition_matrix_mode == "time_varying":
            return self._transition_from_forcing(metadata, xi, thermal_forcing_t)
        matrix = self.transition_matrix(metadata, xi)
        _, bias = self._transition_from_forcing(metadata, xi, thermal_forcing_t)
        return matrix, bias

    def process_scale(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
        state_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.process_noise_mode == "none":
            return jnp.zeros((self.state_dim,), dtype=state_t.dtype)
        if self.process_noise_mode == "constant":
            raw = self.raw_process_scale
        else:
            assert self.process_noise_net is not None
            raw = self.process_noise_init + 0.1 * self.process_noise_net(
                jnp.concatenate([metadata, xi, thermal_forcing_t, state_t], axis=0)
            )
        return (
            jnp.asarray(self.process_noise_floor, dtype=state_t.dtype)
            + jnp.asarray(self.process_noise_cap, dtype=state_t.dtype) * jax.nn.sigmoid(raw)
        )

    def one_step_state(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
        state_t: jnp.ndarray,
        eps_t: jnp.ndarray,
        fixed_matrix: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if fixed_matrix is None:
            matrix, bias = self.transition_matrix_and_bias(
                metadata, xi, thermal_forcing_t
            )
        else:
            matrix = fixed_matrix
            bias = self.transition_bias(metadata, xi, thermal_forcing_t)
        scale_t = self.process_scale(metadata, xi, thermal_forcing_t, state_t)
        scaled_state = state_t / jnp.asarray(self.state_bound, dtype=state_t.dtype)
        raw_next = matrix @ scaled_state + bias + scale_t * eps_t
        return jnp.asarray(self.state_bound, dtype=state_t.dtype) * jnp.tanh(raw_next)

    def hp_emission(
        self,
        features: jnp.ndarray,
        pel_mu_logit: jnp.ndarray,
        pel_sigma_logit: jnp.ndarray,
        mode_logit: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del features
        pi_t = jax.nn.sigmoid(mode_logit)
        if self.hp_emission_mode == "legacy_lognormal_mean":
            active_level = self._max_active_power(pel_mu_logit.dtype) * jax.nn.sigmoid(pel_mu_logit)
            log_mu = jnp.log1p(active_level + jnp.asarray(1e-6, dtype=pel_mu_logit.dtype))
            log_sigma = (
                jnp.asarray(0.05, dtype=pel_mu_logit.dtype)
                + jnp.asarray(0.70, dtype=pel_mu_logit.dtype) * jax.nn.sigmoid(pel_sigma_logit)
            )
            expected_active = jnp.maximum(
                jnp.expm1(log_mu + 0.5 * log_sigma**2),
                jnp.asarray(0.0, dtype=pel_mu_logit.dtype),
            )
            expected_active = self._nonnegative_capped(expected_active, self.hp_pel_cap_w_m2)
            expected_total = self._nonnegative_capped(pi_t * expected_active, self.hp_pel_cap_w_m2)
            return pi_t, log_mu, log_sigma, expected_active, expected_total

        max_log_active = self._max_log_active_power(pel_mu_logit.dtype)
        log_mu = max_log_active * jax.nn.sigmoid(pel_mu_logit)
        log_sigma = (
            jnp.asarray(0.05, dtype=pel_mu_logit.dtype)
            + jnp.asarray(0.45, dtype=pel_mu_logit.dtype) * jax.nn.sigmoid(pel_sigma_logit)
        )
        expected_active = self.expected_active_power(log_mu, log_sigma)
        expected_total = pi_t * expected_active
        return pi_t, log_mu, log_sigma, expected_active, expected_total

    def _max_active_power(self, dtype: jnp.dtype) -> jnp.ndarray:
        if self.hp_pel_cap_w_m2 > 0.0:
            return jnp.asarray(self.hp_pel_cap_w_m2, dtype=dtype)
        pel_scale = jnp.asarray(self.target_scale[self.pel_target_index], dtype=dtype)
        pel_mean = jnp.asarray(self.target_mean[self.pel_target_index], dtype=dtype)
        fallback_cap = jnp.maximum(
            pel_mean + jnp.asarray(8.0, dtype=dtype) * pel_scale,
            jnp.asarray(2.0, dtype=dtype) * pel_scale,
        )
        return jnp.maximum(fallback_cap, jnp.asarray(1e-3, dtype=dtype))

    def _max_log_active_power(self, dtype: jnp.dtype) -> jnp.ndarray:
        return jnp.log1p(self._max_active_power(dtype))

    def _active_power_from_log(self, log_active: jnp.ndarray) -> jnp.ndarray:
        capped_log_active = jnp.clip(
            log_active,
            jnp.asarray(0.0, dtype=log_active.dtype),
            self._max_log_active_power(log_active.dtype),
        )
        return jnp.expm1(capped_log_active)

    def expected_active_power(
        self,
        log_mu: jnp.ndarray,
        log_sigma: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.hp_emission_mode == "legacy_lognormal_mean":
            expected_active = jnp.maximum(
                jnp.expm1(log_mu + 0.5 * log_sigma**2),
                jnp.asarray(0.0, dtype=log_mu.dtype),
            )
            return self._nonnegative_capped(expected_active, self.hp_pel_cap_w_m2)
        return bounded_lognormal_expm1_mean(
            log_mu,
            log_sigma,
            self._max_log_active_power(log_mu.dtype),
        )

    def _sample_active_power_from_log(self, log_active: jnp.ndarray) -> jnp.ndarray:
        if self.hp_emission_mode == "legacy_lognormal_mean":
            return self._nonnegative_capped(jnp.expm1(log_active), self.hp_pel_cap_w_m2)
        return self._active_power_from_log(log_active)

    def decode_hp(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
        input_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
        mode_uniform_t: jnp.ndarray,
        power_noise_t: jnp.ndarray,
        controller_state_t: jnp.ndarray,
        previous_mode_t: jnp.ndarray,
        hp_power_history_t: jnp.ndarray,
        hp_mode_history_t: jnp.ndarray,
        hp_latent: jnp.ndarray,
        controller_noise_t: jnp.ndarray,
        hp_setpoint_shock_t: jnp.ndarray | None = None,
        *,
        hp_scenario_mode: HPElectricRolloutMode,
        hp_concrete_temperature: jnp.ndarray | float,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        features = self.hp_features(
            metadata,
            xi,
            encoded_input_t,
            input_t,
            energy_t,
            temperature_t,
            controller_state_t,
            hp_latent,
            hp_power_history_t,
            hp_mode_history_t,
            hp_setpoint_shock_t,
        )
        availability_t = self._space_heating_availability(input_t)
        output = self.output_net(features)
        if self.thermostat_demand_mode == "monotone":
            # Constrain only latent emitter demand. Compressor mode and active
            # power retain their unrestricted dependence on gap, history and E.
            gap_feature_index = self.metadata_dim + self.latent_dim + self.encoded_input_dim + 1
            demand_output = self.output_net(features.at[gap_feature_index].set(0.0))
            (
                _,
                mode_logit,
                pel_mu_logit,
                pel_sigma_logit,
                _,
                _,
            ) = output
            (
                qroom_capacity_logit,
                _,
                _,
                _,
                qroom_slope_raw,
                qroom_threshold_raw,
            ) = demand_output
            gap_c = self._setpoint_gap_c(input_t, temperature_t)
            qroom_gate_t = jax.nn.sigmoid(
                self._thermostat_slope(qroom_slope_raw)
                * (gap_c - self._thermostat_threshold(qroom_threshold_raw))
            )
            requested_qroom_t = (
                availability_t
                * self._positive_capped_from_logit(
                    qroom_capacity_logit,
                    self.hp_qroom_cap_w_m2,
                    self.qroom_target_index,
                )
                * qroom_gate_t
            )
        else:
            qroom_logit, mode_logit, pel_mu_logit, pel_sigma_logit = output
            requested_qroom_t = availability_t * self._positive_capped_from_logit(
                qroom_logit,
                self.hp_qroom_cap_w_m2,
                self.qroom_target_index,
            )
        independent_pi_t, log_mu_t, log_sigma_t, expected_active_t, _ = self.hp_emission(
            features,
            pel_mu_logit,
            pel_sigma_logit,
            mode_logit,
        )
        if self.hp_activation_model in ("persistent_markov", "asymmetric_markov"):
            assert self.hp_transition_net is not None
            start_logit_t, stop_logit_t = self.hp_transition_net(features)
            p_start_t = jax.nn.sigmoid(start_logit_t)
            p_stop_t = jax.nn.sigmoid(stop_logit_t)
            pi_unavailable_t = (
                (1.0 - previous_mode_t) * p_start_t
                + previous_mode_t * (1.0 - p_stop_t)
            )
        else:
            p_start_t = independent_pi_t
            p_stop_t = 1.0 - independent_pi_t
            pi_unavailable_t = independent_pi_t
        pi_t = availability_t * pi_unavailable_t
        expected_pel_t = pi_t * expected_active_t
        if hp_scenario_mode == "expected":
            hp_on_t = pi_t
            pel_active_t = expected_active_t
            pel_t = expected_pel_t
        else:
            sampled_log_active_t = log_mu_t + log_sigma_t * power_noise_t
            pel_active_t = self._sample_active_power_from_log(sampled_log_active_t)
            if hp_scenario_mode == "straight_through":
                if self.hp_activation_model in ("persistent_markov", "asymmetric_markov"):
                    eps = jnp.asarray(1e-5, dtype=input_t.dtype)
                    clipped_pi_t = jnp.clip(pi_unavailable_t, eps, 1.0 - eps)
                    effective_mode_logit = jnp.log(clipped_pi_t) - jnp.log1p(-clipped_pi_t)
                else:
                    effective_mode_logit = mode_logit
                hp_on_t = availability_t * straight_through_binary_concrete(
                    effective_mode_logit,
                    mode_uniform_t,
                    jnp.asarray(hp_concrete_temperature, dtype=input_t.dtype),
                )
            else:
                hp_on_t = (mode_uniform_t < pi_t).astype(input_t.dtype)
            pel_t = self._nonnegative_capped(hp_on_t * pel_active_t, self.hp_pel_cap_w_m2)

        controller_state_next = self.next_controller_state(
            features,
            hp_on_t,
            controller_state_t,
            controller_noise_t,
        )
        cop_intercept, cop_slope, loss_rate = self.hp_parameters(metadata, xi, hp_latent)
        cop_t = self._cop(input_t, cop_intercept, cop_slope)
        qroom_t, available_power_t, energy_next = self._buffer_draw(
            energy_t,
            cop_t * pel_t,
            requested_qroom_t,
            jnp.asarray(self.hp_dt_hours, dtype=input_t.dtype),
            loss_rate,
        )
        return (
            pi_t,
            cop_t,
            qroom_t,
            energy_next,
            available_power_t,
            requested_qroom_t,
            hp_on_t,
            pel_active_t,
            pel_t,
            log_mu_t,
            log_sigma_t,
            controller_state_next,
            p_start_t,
            p_stop_t,
        )

    def rollout_particle(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        xi: jnp.ndarray,
        process_noise: jnp.ndarray,
        hp_mode_uniform: jnp.ndarray,
        hp_power_noise: jnp.ndarray,
        hp_latent: jnp.ndarray,
        controller_noise: jnp.ndarray,
        *,
        hp_scenario_mode: HPElectricRolloutMode = "expected",
        hp_concrete_temperature: jnp.ndarray | float = 0.5,
        hp_feedback_pel: jnp.ndarray | None = None,
        hp_feedback_mode: jnp.ndarray | None = None,
        hp_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
        hp_feedback_temperature: jnp.ndarray | None = None,
        hp_temperature_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
        ventilation_time_available: jnp.ndarray | None = None,
        ventilation_minimum_indoor_temperature_c: jnp.ndarray | None = None,
        ventilation_nominal_flow_m3_s: jnp.ndarray | float = 0.0,
        ventilation_minimum_indoor_outdoor_delta_c: jnp.ndarray | float = 2.0,
    ) -> tuple[jnp.ndarray, ProbClosedLoopAux]:
        if hp_scenario_mode not in ("expected", "bernoulli", "straight_through"):
            raise ValueError(
                "hp_scenario_mode must be 'expected', 'bernoulli', or 'straight_through'"
            )
        if (hp_feedback_pel is None) != (hp_feedback_mode is None):
            raise ValueError("hp_feedback_pel and hp_feedback_mode must be provided together")
        if hp_feedback_pel is not None:
            if hp_feedback_pel.shape != (inputs.shape[0],):
                raise ValueError("hp_feedback_pel must have shape [horizon]")
            if hp_feedback_mode is None or hp_feedback_mode.shape != (inputs.shape[0],):
                raise ValueError("hp_feedback_mode must have shape [horizon]")
        if (
            hp_feedback_temperature is not None
            and hp_feedback_temperature.shape != (inputs.shape[0],)
        ):
            raise ValueError("hp_feedback_temperature must have shape [horizon]")
        generated_ventilation = ventilation_time_available is not None
        if generated_ventilation != (ventilation_minimum_indoor_temperature_c is not None):
            raise ValueError(
                "ventilation_time_available and ventilation_minimum_indoor_temperature_c "
                "must be provided together"
            )
        if generated_ventilation:
            if ventilation_time_available.shape != (inputs.shape[0],):
                raise ValueError("ventilation_time_available must have shape [horizon]")
            if ventilation_minimum_indoor_temperature_c.shape != (inputs.shape[0],):
                raise ValueError(
                    "ventilation_minimum_indoor_temperature_c must have shape [horizon]"
                )
        setpoint_shock_features = self.setpoint_shock_features(inputs)
        state0 = self.initial_state(metadata, initial_temperature, xi)
        energy0 = self.initial_energy(metadata, initial_temperature, xi)
        temperature0 = initial_temperature[0]
        q_response_state0 = self.initial_q_response_state(
            metadata,
            initial_temperature,
            xi,
        )
        alpha = self.temperature_alpha(metadata, xi)
        controller_state0, previous_mode0 = self.initial_controller_state(
            metadata,
            initial_temperature,
            hp_latent,
        )
        hp_power_history0, hp_mode_history0 = self.initial_hp_history(metadata.dtype)
        if self.hp_activation_model == "asymmetric_markov":
            previous_mode0 = self.initial_asymmetric_markov_mode(
                metadata,
                xi,
                inputs[0],
                energy0,
                temperature0,
                hp_power_history0,
                hp_mode_history0,
                setpoint_shock_features[0],
            )
        fixed_matrix = (
            self.transition_matrix(metadata, xi)
            if self.transition_matrix_mode == "fixed"
            else None
        )

        def step(
            carry: tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            step_inputs: tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
        ) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
            (
                state_t,
                energy_t,
                temperature_t,
                controller_state_t,
                previous_mode_t,
                hp_power_history_t,
                hp_mode_history_t,
                q_response_state_t,
            ) = carry
            (
                time_index,
                input_t,
                eps_t,
                mode_uniform_t,
                power_noise_t,
                controller_noise_t,
                setpoint_shock_t,
            ) = step_inputs
            if generated_ventilation:
                input_t = replace_with_eplus_ventilation(
                    input_t,
                    temperature_t,
                    time_available_t=ventilation_time_available[time_index],
                    minimum_indoor_temperature_c_t=(
                        ventilation_minimum_indoor_temperature_c[time_index]
                    ),
                    nominal_flow_m3_s=jnp.asarray(
                        ventilation_nominal_flow_m3_s,
                        dtype=input_t.dtype,
                    ),
                    minimum_indoor_outdoor_delta_c=jnp.asarray(
                        ventilation_minimum_indoor_outdoor_delta_c,
                        dtype=input_t.dtype,
                    ),
                    ventilation_input_index=self.ventilation_input_index,
                    outdoor_input_index=self.outdoor_input_index,
                    input_mean=self.input_mean,
                    input_scale=self.input_scale,
                    temperature_mean_c=self.target_mean[self.temperature_target_index],
                    temperature_scale_c=self.target_scale[self.temperature_target_index],
                )
            encoded_input_t = self.encode_input(input_t)
            hp_temperature_t = (
                temperature_t
                if hp_feedback_temperature is None
                else (
                    (1.0 - jnp.clip(hp_temperature_teacher_forcing_ratio, 0.0, 1.0))
                    * temperature_t
                    + jnp.clip(hp_temperature_teacher_forcing_ratio, 0.0, 1.0)
                    * hp_feedback_temperature[time_index]
                )
            )
            (
                pi_t,
                cop_t,
                qroom_t,
                energy_next,
                available_power_t,
                requested_qroom_t,
                hp_on_t,
                pel_active_t,
                pel_t,
                log_mu_t,
                log_sigma_t,
                controller_state_next,
                p_start_t,
                p_stop_t,
            ) = self.decode_hp(
                metadata,
                xi,
                encoded_input_t,
                input_t,
                energy_t,
                hp_temperature_t,
                mode_uniform_t,
                power_noise_t,
                controller_state_t,
                previous_mode_t,
                hp_power_history_t,
                hp_mode_history_t,
                hp_latent,
                controller_noise_t,
                setpoint_shock_t,
                hp_scenario_mode=hp_scenario_mode,
                hp_concrete_temperature=hp_concrete_temperature,
            )
            hp_power_history_next, hp_mode_history_next = self.next_hp_history(
                hp_power_history_t,
                hp_mode_history_t,
                (
                    pel_t
                    if hp_feedback_pel is None
                    else (
                        (1.0 - jnp.clip(hp_teacher_forcing_ratio, 0.0, 1.0)) * pel_t
                        + jnp.clip(hp_teacher_forcing_ratio, 0.0, 1.0)
                        * hp_feedback_pel[time_index]
                    )
                ),
                (
                    hp_on_t
                    if hp_feedback_mode is None
                    else (
                        (1.0 - jnp.clip(hp_teacher_forcing_ratio, 0.0, 1.0)) * hp_on_t
                        + jnp.clip(hp_teacher_forcing_ratio, 0.0, 1.0)
                        * hp_feedback_mode[time_index]
                    )
                ),
            )
            qroom_scaled_t = self._target_scaled(qroom_t, self.qroom_target_index)
            pel_scaled_t = self._target_scaled(pel_t, self.pel_target_index)
            thermal_forcing_t = self.thermal_forcing(input_t, qroom_scaled_t, temperature_t)
            state_next = self.one_step_state(
                metadata,
                xi,
                thermal_forcing_t,
                state_t,
                eps_t,
                fixed_matrix,
            )
            q_response_state_next = self.next_q_response_state(
                metadata,
                xi,
                q_response_state_t,
                qroom_t,
            )
            temperature_next = self.decode_temperature(
                state_next,
                xi,
                temperature_t,
                alpha,
                q_response_state_next,
            )
            prediction_t = jnp.stack([temperature_next, qroom_scaled_t, pel_scaled_t])
            aux_t = (
                pi_t,
                cop_t,
                energy_t,
                available_power_t,
                requested_qroom_t,
                hp_on_t,
                pel_active_t,
                pel_t,
                log_mu_t,
                log_sigma_t,
                state_t,
                controller_state_t,
                temperature_t,
                p_start_t,
                p_stop_t,
                hp_power_history_t,
                hp_mode_history_t,
                previous_mode_t,
                q_response_state_t,
            )
            return detach_carry(
                (
                    state_next,
                    energy_next,
                    temperature_next,
                    controller_state_next,
                    hp_on_t,
                    hp_power_history_next,
                    hp_mode_history_next,
                    q_response_state_next,
                ),
                time_index,
                self.bptt_truncate_steps,
            ), (prediction_t, *aux_t)

        time_index = jnp.arange(inputs.shape[0])
        _, outputs = jax.lax.scan(
            step,
            (
                state0,
                energy0,
                temperature0,
                controller_state0,
                previous_mode0,
                hp_power_history0,
                hp_mode_history0,
                q_response_state0,
            ),
            (
                time_index,
                inputs,
                process_noise,
                hp_mode_uniform,
                hp_power_noise,
                controller_noise,
                setpoint_shock_features,
            ),
        )
        (
            predictions,
            pi,
            cop,
            energy,
            available_power,
            qroom_raw,
            hp_on,
            pel_active,
            pel,
            log_mu,
            log_sigma,
            x_state,
            w_state,
            temperature_state,
            p_start,
            p_stop,
            hp_power_history,
            hp_mode_history,
            previous_mode_state,
            q_response_state,
        ) = outputs
        aux = (
            pi,
            cop,
            energy,
            available_power,
            qroom_raw,
            hp_on,
            pel_active,
            pel,
            log_mu,
            log_sigma,
            x_state,
            w_state,
            temperature_state,
            p_start,
            p_stop,
            hp_power_history,
            hp_mode_history,
            previous_mode_state,
            q_response_state,
        )
        return predictions, aux

    def sample_with_aux(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool = True,
        hp_scenario_mode: HPElectricRolloutMode = "expected",
        hp_concrete_temperature: jnp.ndarray | float = 0.5,
        hp_feedback_pel: jnp.ndarray | None = None,
        hp_feedback_mode: jnp.ndarray | None = None,
        hp_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
        hp_feedback_temperature: jnp.ndarray | None = None,
        hp_temperature_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
        ventilation_time_available: jnp.ndarray | None = None,
        ventilation_minimum_indoor_temperature_c: jnp.ndarray | None = None,
        ventilation_nominal_flow_m3_s: jnp.ndarray | float = 0.0,
        ventilation_minimum_indoor_outdoor_delta_c: jnp.ndarray | float = 2.0,
    ) -> tuple[jnp.ndarray, ProbClosedLoopAux]:
        if num_particles < 1:
            raise ValueError("num_particles must be positive")
        if hp_scenario_mode not in ("expected", "bernoulli", "straight_through"):
            raise ValueError(
                "hp_scenario_mode must be 'expected', 'bernoulli', or 'straight_through'"
            )
        xi_key, process_key, mode_key, power_key = jax.random.split(key, 4)
        xi = jax.random.normal(xi_key, (num_particles, self.latent_dim))
        if self.hp_activation_model == "persistent_markov":
            hp_latent_key = jax.random.fold_in(xi_key, 17)
            hp_latent = jax.random.normal(
                hp_latent_key,
                (num_particles, self.hp_persistent_latent_dim),
            )
            if self.hp_controller_noise_scale > 0.0:
                controller_noise_key = jax.random.fold_in(process_key, 17)
                controller_noise = jax.random.normal(
                    controller_noise_key,
                    (num_particles, inputs.shape[0], self.controller_state_dim),
                )
            else:
                controller_noise = jnp.zeros(
                    (num_particles, inputs.shape[0], self.controller_state_dim),
                    dtype=inputs.dtype,
                )
        else:
            hp_latent = jnp.zeros((num_particles, 0), dtype=inputs.dtype)
            controller_noise = jnp.zeros(
                (num_particles, inputs.shape[0], self.controller_state_dim),
                dtype=inputs.dtype,
            )
        if sample_process_noise and self.process_noise_mode != "none":
            process_noise = jax.random.normal(process_key, (num_particles, inputs.shape[0], self.state_dim))
        else:
            process_noise = jnp.zeros((num_particles, inputs.shape[0], self.state_dim))
        hp_mode_uniform = jax.random.uniform(mode_key, (num_particles, inputs.shape[0]))
        hp_power_noise = jax.random.normal(power_key, (num_particles, inputs.shape[0]))
        predictions, aux = jax.vmap(
            lambda particle_xi, particle_process_noise, particle_mode_uniform, particle_power_noise, particle_hp_latent, particle_controller_noise: self.rollout_particle(
                metadata,
                inputs,
                initial_temperature,
                particle_xi,
                particle_process_noise,
                particle_mode_uniform,
                particle_power_noise,
                particle_hp_latent,
                particle_controller_noise,
                hp_scenario_mode=hp_scenario_mode,
                hp_concrete_temperature=hp_concrete_temperature,
                hp_feedback_pel=hp_feedback_pel,
                hp_feedback_mode=hp_feedback_mode,
                hp_teacher_forcing_ratio=hp_teacher_forcing_ratio,
                hp_feedback_temperature=hp_feedback_temperature,
                hp_temperature_teacher_forcing_ratio=hp_temperature_teacher_forcing_ratio,
                ventilation_time_available=ventilation_time_available,
                ventilation_minimum_indoor_temperature_c=(
                    ventilation_minimum_indoor_temperature_c
                ),
                ventilation_nominal_flow_m3_s=ventilation_nominal_flow_m3_s,
                ventilation_minimum_indoor_outdoor_delta_c=(
                    ventilation_minimum_indoor_outdoor_delta_c
                ),
            )
        )(xi, process_noise, hp_mode_uniform, hp_power_noise, hp_latent, controller_noise)
        return predictions, (
            *aux[:13],
            xi,
            *aux[13:15],
            hp_latent,
            *aux[15:],
        )

    def sample(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool = True,
        hp_scenario_mode: HPElectricRolloutMode = "expected",
        hp_concrete_temperature: jnp.ndarray | float = 0.5,
        hp_feedback_pel: jnp.ndarray | None = None,
        hp_feedback_mode: jnp.ndarray | None = None,
        hp_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
        hp_feedback_temperature: jnp.ndarray | None = None,
        hp_temperature_teacher_forcing_ratio: jnp.ndarray | float = 0.0,
        ventilation_time_available: jnp.ndarray | None = None,
        ventilation_minimum_indoor_temperature_c: jnp.ndarray | None = None,
        ventilation_nominal_flow_m3_s: jnp.ndarray | float = 0.0,
        ventilation_minimum_indoor_outdoor_delta_c: jnp.ndarray | float = 2.0,
    ) -> jnp.ndarray:
        predictions, _ = self.sample_with_aux(
            metadata,
            inputs,
            initial_temperature,
            key=key,
            num_particles=num_particles,
            sample_process_noise=sample_process_noise,
            hp_scenario_mode=hp_scenario_mode,
            hp_concrete_temperature=hp_concrete_temperature,
            hp_feedback_pel=hp_feedback_pel,
            hp_feedback_mode=hp_feedback_mode,
            hp_teacher_forcing_ratio=hp_teacher_forcing_ratio,
            hp_feedback_temperature=hp_feedback_temperature,
            hp_temperature_teacher_forcing_ratio=hp_temperature_teacher_forcing_ratio,
            ventilation_time_available=ventilation_time_available,
            ventilation_minimum_indoor_temperature_c=(
                ventilation_minimum_indoor_temperature_c
            ),
            ventilation_nominal_flow_m3_s=ventilation_nominal_flow_m3_s,
            ventilation_minimum_indoor_outdoor_delta_c=(
                ventilation_minimum_indoor_outdoor_delta_c
            ),
        )
        return predictions

    def stability_state_dim(self) -> int:
        size = self.state_dim + 2  # thermal state, stored energy, indoor temperature
        if self.q_to_t_mode == "positive_leaky":
            size += len(self.q_to_t_time_constants_hours)
        if self.hp_activation_model == "persistent_markov":
            size += self.controller_state_dim + 1
        elif self.hp_activation_model == "power_history":
            history_dim = 1 + len(self.hp_history_timescales_hours)
            size += 2 * history_dim
        elif self.hp_activation_model == "asymmetric_markov":
            history_dim = 1 + len(self.hp_history_timescales_hours)
            size += 1 + 2 * history_dim
        return size

    def pack_stability_state(
        self,
        state_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
        controller_state_t: jnp.ndarray,
        previous_mode_t: jnp.ndarray,
        hp_power_history_t: jnp.ndarray,
        hp_mode_history_t: jnp.ndarray,
        q_response_state_t: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        parts = [
            state_t,
            jnp.asarray([energy_t, temperature_t], dtype=state_t.dtype),
        ]
        if self.q_to_t_mode == "positive_leaky":
            if q_response_state_t is None:
                q_response_state_t = jnp.zeros(
                    (len(self.q_to_t_time_constants_hours),),
                    dtype=state_t.dtype,
                )
            parts.append(q_response_state_t)
        if self.hp_activation_model == "persistent_markov":
            parts.extend(
                [
                    controller_state_t,
                    jnp.asarray([previous_mode_t], dtype=state_t.dtype),
                ]
            )
        elif self.hp_activation_model == "power_history":
            parts.extend(
                [
                    jnp.ravel(hp_power_history_t),
                    jnp.ravel(hp_mode_history_t),
                ]
            )
        elif self.hp_activation_model == "asymmetric_markov":
            parts.extend(
                [
                    jnp.asarray([previous_mode_t], dtype=state_t.dtype),
                    jnp.ravel(hp_power_history_t),
                    jnp.ravel(hp_mode_history_t),
                ]
            )
        return jnp.concatenate(parts, axis=0)

    def unpack_stability_state(
        self,
        augmented_state: jnp.ndarray,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        cursor = self.state_dim
        state_t = augmented_state[:cursor]
        energy_t = augmented_state[cursor]
        temperature_t = augmented_state[cursor + 1]
        cursor += 2
        if self.q_to_t_mode == "positive_leaky":
            q_response_dim = len(self.q_to_t_time_constants_hours)
            q_response_state_t = augmented_state[cursor : cursor + q_response_dim]
            cursor += q_response_dim
        else:
            q_response_state_t = jnp.zeros((0,), dtype=augmented_state.dtype)
        controller_state_t = jnp.zeros(
            (self.controller_state_dim,), dtype=augmented_state.dtype
        )
        previous_mode_t = jnp.asarray(0.5, dtype=augmented_state.dtype)
        hp_power_history_t, hp_mode_history_t = self.initial_hp_history(
            augmented_state.dtype
        )
        if self.hp_activation_model == "persistent_markov":
            controller_state_t = augmented_state[
                cursor : cursor + self.controller_state_dim
            ]
            cursor += self.controller_state_dim
            previous_mode_t = augmented_state[cursor]
            cursor += 1
        elif self.hp_activation_model == "power_history":
            history_dim = 1 + len(self.hp_history_timescales_hours)
            hp_power_history_t = augmented_state[cursor : cursor + history_dim]
            cursor += history_dim
            hp_mode_history_t = augmented_state[cursor : cursor + history_dim]
            cursor += history_dim
        elif self.hp_activation_model == "asymmetric_markov":
            previous_mode_t = augmented_state[cursor]
            cursor += 1
            history_dim = 1 + len(self.hp_history_timescales_hours)
            hp_power_history_t = augmented_state[cursor : cursor + history_dim]
            cursor += history_dim
            hp_mode_history_t = augmented_state[cursor : cursor + history_dim]
            cursor += history_dim
        if cursor != augmented_state.shape[0]:
            raise ValueError(
                "augmented stability state has incompatible size: "
                f"expected {cursor}, got {augmented_state.shape[0]}"
            )
        return (
            state_t,
            energy_t,
            temperature_t,
            controller_state_t,
            previous_mode_t,
            hp_power_history_t,
            hp_mode_history_t,
            q_response_state_t,
        )

    def one_step_augmented_state(
        self,
        metadata: jnp.ndarray,
        input_t: jnp.ndarray,
        xi: jnp.ndarray,
        hp_latent: jnp.ndarray,
        augmented_state: jnp.ndarray,
        process_noise_t: jnp.ndarray,
        controller_noise_t: jnp.ndarray,
        hp_setpoint_shock_t: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        (
            state_t,
            energy_t,
            temperature_t,
            controller_state_t,
            previous_mode_t,
            hp_power_history_t,
            hp_mode_history_t,
            q_response_state_t,
        ) = self.unpack_stability_state(augmented_state)
        alpha = self.temperature_alpha(metadata, xi)
        encoded_input_t = self.encode_input(input_t)
        (
            _,
            _,
            qroom_t,
            energy_next,
            _,
            _,
            hp_on_t,
            _,
            pel_t,
            _,
            _,
            controller_state_next,
            _,
            _,
        ) = self.decode_hp(
            metadata,
            xi,
            encoded_input_t,
            input_t,
            energy_t,
            temperature_t,
            jnp.asarray(0.5, dtype=state_t.dtype),
            jnp.asarray(0.0, dtype=state_t.dtype),
            controller_state_t,
            previous_mode_t,
            hp_power_history_t,
            hp_mode_history_t,
            hp_latent,
            controller_noise_t,
            hp_setpoint_shock_t,
            hp_scenario_mode="expected",
            hp_concrete_temperature=jnp.asarray(0.5, dtype=state_t.dtype),
        )
        hp_power_history_next, hp_mode_history_next = self.next_hp_history(
            hp_power_history_t,
            hp_mode_history_t,
            pel_t,
            hp_on_t,
        )
        qroom_scaled_t = self._target_scaled(qroom_t, self.qroom_target_index)
        thermal_forcing_t = self.thermal_forcing(input_t, qroom_scaled_t, temperature_t)
        fixed_matrix = (
            self.transition_matrix(metadata, xi)
            if self.transition_matrix_mode == "fixed"
            else None
        )
        state_next = self.one_step_state(
            metadata,
            xi,
            thermal_forcing_t,
            state_t,
            process_noise_t,
            fixed_matrix,
        )
        q_response_state_next = self.next_q_response_state(
            metadata,
            xi,
            q_response_state_t,
            qroom_t,
        )
        temperature_next = self.decode_temperature(
            state_next,
            xi,
            temperature_t,
            alpha,
            q_response_state_next,
        )
        return self.pack_stability_state(
            state_next,
            energy_next,
            temperature_next,
            controller_state_next,
            hp_on_t,
            hp_power_history_next,
            hp_mode_history_next,
            q_response_state_next,
        )

    def closed_loop_jacobian_spectral_norm(
        self,
        metadata: jnp.ndarray,
        input_t: jnp.ndarray,
        xi: jnp.ndarray,
        hp_latent: jnp.ndarray,
        augmented_state: jnp.ndarray,
        process_noise_t: jnp.ndarray,
        controller_noise_t: jnp.ndarray,
    ) -> jnp.ndarray:
        jacobian = jax.jacrev(
            lambda state: self.one_step_augmented_state(
                metadata,
                input_t,
                xi,
                hp_latent,
                state,
                process_noise_t,
                controller_noise_t,
            )
        )(augmented_state)
        return jnp.max(jnp.linalg.svd(jacobian, compute_uv=False))

    def parameter_regularization(self, metadata: jnp.ndarray) -> jnp.ndarray:
        del metadata
        return jnp.asarray(0.0)
