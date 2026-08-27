"""Bounded closed-loop HP emulator with norm-constrained transition matrices.

This model keeps a bounded recurrent state and projects the HP/electrical
outputs through a bounded leaky hydronic buffer. Each time-varying transition
matrix is norm bounded, but this alone is not a global closed-loop contraction
guarantee because the matrix and forcing depend on the recurrent trajectory.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import MLP
from .truncated_bptt import DEFAULT_BPTT_TRUNCATE_STEPS, detach_carry, validate_bptt_truncate_steps

ContractingClosedLoopAux = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
TemperatureUpdateMode = Literal["auto", "absolute", "delta", "leaky_equilibrium"]
ThermostatDemandMode = Literal["unconstrained", "monotone"]
ThermalQResponseMode = Literal["unconstrained", "positive_leaky"]


class ContractingClosedLoopHPEmulator(eqx.Module):
    """Data-driven closed-loop emulator with a bounded recurrent map.

    The recurrent map has the form

        s[t+1] = state_bound * tanh(M_t @ (s[t] / state_bound) + r_t)

    where ``M_t`` and the nonlinear forcing ``r_t`` are generated from metadata
    and the current thermal forcing. Every generated matrix is independently
    normalized so its Frobenius norm is at most ``contraction_gamma``.
    """

    input_encoder: MLP | None
    transition_net: MLP
    x0_net: MLP
    e0_net: MLP
    output_net: MLP
    temperature_net: MLP
    temperature_alpha_net: MLP | None
    q_response_net: MLP | None
    q_response_init_net: MLP | None
    cop_intercept: jax.Array
    cop_slope: jax.Array
    raw_energy_loss_rate: jax.Array
    metadata_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    encoded_input_dim: int = eqx.field(static=True)
    thermal_forcing_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
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
    energy_scale: float = eqx.field(static=True)
    input_mean: tuple[float, ...] = eqx.field(static=True)
    input_scale: tuple[float, ...] = eqx.field(static=True)
    target_mean: tuple[float, ...] = eqx.field(static=True)
    target_scale: tuple[float, ...] = eqx.field(static=True)
    setpoint_input_index: int = eqx.field(static=True)
    outdoor_input_index: int = eqx.field(static=True)
    solar_input_index: int = eqx.field(static=True)
    ventilation_input_index: int = eqx.field(static=True)
    availability_input_index: int = eqx.field(static=True)
    temperature_target_index: int = eqx.field(static=True)
    qroom_target_index: int = eqx.field(static=True)
    pel_target_index: int = eqx.field(static=True)
    bptt_truncate_steps: int = eqx.field(static=True)
    hp_controller_masked_input_indices: tuple[int, ...] = eqx.field(static=True)
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
        hidden_dim: int = 64,
        depth: int = 3,
        input_encoder_dim: int | None = None,
        input_encoder_hidden_dim: int | None = None,
        input_encoder_depth: int = 2,
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
        hp_controller_masked_input_indices: tuple[int, ...] = (),
        thermostat_demand_mode: ThermostatDemandMode = "unconstrained",
        thermostat_slope_min: float = 0.1,
        thermostat_slope_max: float = 6.0,
        thermostat_threshold_min_c: float = -1.0,
        thermostat_threshold_max_c: float = 1.0,
        q_to_t_mode: ThermalQResponseMode = "unconstrained",
        q_to_t_time_constants_hours: tuple[float, ...] = (1.0, 24.0),
        q_to_t_gain_min_c_per_w_m2: float = 0.01,
        q_to_t_gain_max_c_per_w_m2: float = 2.0,
        key: jax.Array,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if input_encoder_dim is not None and input_encoder_dim < 1:
            raise ValueError("input_encoder_dim must be positive or None")
        if input_encoder_depth < 1:
            raise ValueError("input_encoder_depth must be at least 1")
        if not 0.0 < contraction_gamma < 1.0:
            raise ValueError("contraction_gamma must be in (0, 1)")
        if state_bound <= 0.0:
            raise ValueError("state_bound must be positive")
        if temperature_output_scale <= 0.0:
            raise ValueError("temperature_output_scale must be positive")
        if temperature_delta_max_c < 0.0:
            raise ValueError("temperature_delta_max_c must be non-negative")
        if temperature_update_mode not in ("auto", "absolute", "delta", "leaky_equilibrium"):
            raise ValueError(
                "temperature_update_mode must be 'auto', 'absolute', 'delta', or 'leaky_equilibrium'"
            )
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
        if q_to_t_mode == "positive_leaky" and resolved_temperature_update != "leaky_equilibrium":
            raise ValueError(
                "q_to_t_mode='positive_leaky' requires temperature_update_mode="
                "'leaky_equilibrium'"
            )
        bptt_truncate_steps = validate_bptt_truncate_steps(bptt_truncate_steps)
        if len(input_mean) != input_dim or len(input_scale) != input_dim:
            raise ValueError("input_mean and input_scale must match input_dim")
        if len(target_mean) != 3 or len(target_scale) != 3:
            raise ValueError("closed_loop_hp target scalers must have exactly 3 outputs")
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

        (
            encoder_key,
            transition_key,
            x0_key,
            e0_key,
            output_key,
            temperature_key,
        ) = jax.random.split(key, 6)
        alpha_key = jax.random.fold_in(temperature_key, 1)
        q_response_key = jax.random.fold_in(temperature_key, 2)
        q_response_init_key = jax.random.fold_in(temperature_key, 3)
        encoded_input_dim = input_dim if input_encoder_dim is None else input_encoder_dim
        thermal_forcing_dim = input_dim + (1 if q_to_t_mode == "positive_leaky" else 2)
        transition_input_dim = metadata_dim + thermal_forcing_dim
        transition_output_dim = state_dim * state_dim + state_dim

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
        self.x0_net = MLP(
            metadata_dim + 1,
            state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=x0_key,
        )
        self.e0_net = MLP(
            metadata_dim + 1,
            1,
            hidden_dim=hidden_dim,
            depth=depth,
            key=e0_key,
        )
        output_dim = 3 if thermostat_demand_mode == "unconstrained" else 5
        self.output_net = MLP(
            metadata_dim + encoded_input_dim + 3,
            output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=output_key,
        )
        self.temperature_net = MLP(
            state_dim,
            1,
            hidden_dim=hidden_dim,
            depth=depth,
            key=temperature_key,
        )
        self.temperature_alpha_net = None
        if temperature_update_mode == "leaky_equilibrium":
            self.temperature_alpha_net = MLP(
                metadata_dim,
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
                metadata_dim,
                q_response_dim + 1,
                hidden_dim=hidden_dim,
                depth=depth,
                key=q_response_key,
            )
            self.q_response_init_net = MLP(
                metadata_dim + 1,
                q_response_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                key=q_response_init_key,
            )
        self.cop_intercept = jnp.asarray(3.0, dtype=jnp.float32)
        self.cop_slope = jnp.asarray(0.02, dtype=jnp.float32)
        self.raw_energy_loss_rate = jnp.asarray(-4.0, dtype=jnp.float32)

        self.metadata_dim = metadata_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.encoded_input_dim = encoded_input_dim
        self.thermal_forcing_dim = thermal_forcing_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
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
        self.hp_controller_masked_input_indices = hp_controller_masked_input_indices
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

    def initial_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> jnp.ndarray:
        raw = self.x0_net(jnp.concatenate([metadata, initial_temperature], axis=0))
        return jnp.asarray(self.state_bound, dtype=raw.dtype) * jnp.tanh(raw)

    def initial_energy(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> jnp.ndarray:
        raw = self.e0_net(jnp.concatenate([metadata, initial_temperature], axis=0))[0]
        energy = jax.nn.softplus(raw) * jnp.asarray(self.energy_scale, dtype=raw.dtype)
        return self._nonnegative_capped(energy, self.hp_energy_cap_wh_m2)

    def _physical_input(self, input_t: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            input_t[index] * jnp.asarray(self.input_scale[index], dtype=input_t.dtype)
            + jnp.asarray(self.input_mean[index], dtype=input_t.dtype)
        )

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

    def _space_heating_availability(self, input_t: jnp.ndarray) -> jnp.ndarray:
        if self.availability_input_index < 0:
            return jnp.asarray(1.0, dtype=input_t.dtype)
        return jnp.clip(
            self._physical_input(input_t, self.availability_input_index),
            jnp.asarray(0.0, dtype=input_t.dtype),
            jnp.asarray(1.0, dtype=input_t.dtype),
        )

    def _nonnegative_capped(self, value: jnp.ndarray, cap: float) -> jnp.ndarray:
        value = jnp.maximum(value, jnp.asarray(0.0, dtype=value.dtype))
        if cap <= 0.0:
            return value
        return jnp.minimum(value, jnp.asarray(cap, dtype=value.dtype))

    def _target_scaled(self, value: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            value - jnp.asarray(self.target_mean[index], dtype=value.dtype)
        ) / jnp.asarray(self.target_scale[index], dtype=value.dtype)

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
        if mode == "leaky_equilibrium":
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
            return (1.0 - alpha_t) * previous_temperature_scaled + alpha_t * equilibrium_scaled
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

    def temperature_alpha(self, metadata: jnp.ndarray) -> jnp.ndarray:
        if self.temperature_alpha_net is None:
            return jnp.asarray(1.0, dtype=metadata.dtype)
        raw_alpha = self.temperature_alpha_net(metadata)[0]
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
        encoded_input_t: jnp.ndarray,
        input_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> jnp.ndarray:
        gap_feature = self._setpoint_gap_scaled(input_t, temperature_t)
        return jnp.concatenate(
            [
                metadata,
                encoded_input_t,
                jnp.asarray([temperature_t], dtype=encoded_input_t.dtype),
                jnp.asarray([gap_feature], dtype=encoded_input_t.dtype),
                jnp.asarray([self._energy_scaled(energy_t)], dtype=encoded_input_t.dtype),
            ],
            axis=0,
        )

    def thermal_forcing(
        self,
        input_t: jnp.ndarray,
        qroom_scaled_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> jnp.ndarray:
        thermal_inputs = input_t[1:]
        if self.availability_input_index >= 1:
            thermal_inputs = thermal_inputs.at[self.availability_input_index - 1].set(0.0)
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
        previous_temperature_scaled: jnp.ndarray,
        alpha_t: jnp.ndarray,
        q_response_state_t: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        raw_temperature = self.temperature_net(
            state_t / jnp.asarray(self.state_bound, dtype=state_t.dtype)
        )[0]
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
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.q_response_net is None:
            return (
                jnp.asarray(0.0, dtype=metadata.dtype),
                jnp.zeros((0,), dtype=metadata.dtype),
            )
        raw = self.q_response_net(metadata)
        lower = jnp.asarray(self.q_to_t_gain_min_c_per_w_m2, dtype=raw.dtype)
        width = jnp.asarray(
            self.q_to_t_gain_max_c_per_w_m2
            - self.q_to_t_gain_min_c_per_w_m2,
            dtype=raw.dtype,
        )
        # Start near a modest thermal resistance while retaining the full bounds.
        total_gain = lower + width * jax.nn.sigmoid(raw[0] - 3.0)
        mode_weights = jax.nn.softmax(raw[1:])
        return total_gain, mode_weights

    def initial_q_response_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.q_response_init_net is None:
            return jnp.zeros((0,), dtype=metadata.dtype)
        total_gain, mode_weights = self.q_response_parameters(metadata)
        raw = self.q_response_init_net(
            jnp.concatenate([metadata, initial_temperature], axis=0)
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
        q_response_state_t: jnp.ndarray,
        qroom_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.q_response_net is None:
            return q_response_state_t
        total_gain, mode_weights = self.q_response_parameters(metadata)
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

    def _cop(self, input_t: jnp.ndarray) -> jnp.ndarray:
        outdoor_c = self._physical_input(input_t, self.outdoor_input_index)
        cop = jnp.maximum(self.hp_cop_floor, self.cop_intercept + self.cop_slope * outdoor_c)
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

    def transition_matrix(
        self,
        metadata: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
    ) -> jnp.ndarray:
        matrix, _ = self.transition_matrix_and_bias(metadata, thermal_forcing_t)
        return matrix

    def transition_bias(
        self,
        metadata: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
    ) -> jnp.ndarray:
        _, bias = self.transition_matrix_and_bias(metadata, thermal_forcing_t)
        return bias

    def transition_matrix_and_bias(
        self,
        metadata: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        raw = self.transition_net(
            jnp.concatenate([metadata, thermal_forcing_t], axis=0)
        )
        matrix_size = self.state_dim * self.state_dim
        raw_matrix = raw[:matrix_size].reshape((self.state_dim, self.state_dim))
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
        return matrix, jnp.tanh(raw_bias)

    def _apply_state_transition(
        self,
        matrix_t: jnp.ndarray,
        bias_t: jnp.ndarray,
        state_t: jnp.ndarray,
    ) -> jnp.ndarray:
        scaled_state = state_t / jnp.asarray(self.state_bound, dtype=state_t.dtype)
        raw_next = matrix_t @ scaled_state + bias_t
        return jnp.asarray(self.state_bound, dtype=state_t.dtype) * jnp.tanh(raw_next)

    def one_step_state(
        self,
        metadata: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
        state_t: jnp.ndarray,
    ) -> jnp.ndarray:
        matrix_t, bias_t = self.transition_matrix_and_bias(metadata, thermal_forcing_t)
        return self._apply_state_transition(matrix_t, bias_t, state_t)

    def decode_hp(
        self,
        metadata: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
        input_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        features = self.hp_features(metadata, encoded_input_t, input_t, energy_t, temperature_t)
        availability_t = self._space_heating_availability(input_t)
        output = self.output_net(features)
        if self.thermostat_demand_mode == "monotone":
            # The emitter-demand parameters must not depend on the thermostat gap;
            # otherwise their variation could cancel the positive sigmoid slope.
            # HP mode and active power remain unrestricted functions of the gap.
            gap_feature_index = self.metadata_dim + self.encoded_input_dim + 1
            demand_output = self.output_net(features.at[gap_feature_index].set(0.0))
            (
                _,
                pel_logit,
                mode_logit,
                _,
                _,
            ) = output
            (
                qroom_capacity_logit,
                _,
                _,
                qroom_slope_raw,
                qroom_threshold_raw,
            ) = demand_output
            gap_c = self._setpoint_gap_c(input_t, temperature_t)
            qroom_gate = jax.nn.sigmoid(
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
                * qroom_gate
            )
            pi_t = availability_t * jax.nn.sigmoid(mode_logit)
        else:
            qroom_logit, pel_logit, mode_logit = output
            pi_t = availability_t * jax.nn.sigmoid(mode_logit)
            requested_qroom_t = availability_t * self._positive_capped_from_logit(
                qroom_logit,
                self.hp_qroom_cap_w_m2,
                self.qroom_target_index,
            )
        pel_active_t = self._positive_capped_from_logit(
            pel_logit,
            self.hp_pel_cap_w_m2,
            self.pel_target_index,
        )
        pel_t = pi_t * pel_active_t
        cop_t = self._cop(input_t)
        qroom_t, available_power_t, energy_next = self._buffer_draw(
            energy_t,
            cop_t * pel_t,
            requested_qroom_t,
            jnp.asarray(self.hp_dt_hours, dtype=input_t.dtype),
            jax.nn.softplus(self.raw_energy_loss_rate),
        )
        return pi_t, qroom_t, pel_t, energy_next, available_power_t, requested_qroom_t

    def rollout_with_aux(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ContractingClosedLoopAux]:
        state0 = self.initial_state(metadata, initial_temperature)
        energy0 = self.initial_energy(metadata, initial_temperature)
        q_response_state0 = self.initial_q_response_state(metadata, initial_temperature)
        alpha = self.temperature_alpha(metadata)

        def step(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            step_inputs: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
            state_t, energy_t, temperature_t, q_response_state_t = carry
            time_index, input_t = step_inputs
            encoded_input_t = self.encode_input(input_t)
            pi_t, qroom_t, pel_t, energy_next, _, _ = self.decode_hp(
                metadata,
                encoded_input_t,
                input_t,
                energy_t,
                temperature_t,
            )
            qroom_scaled_t = self._target_scaled(qroom_t, self.qroom_target_index)
            pel_scaled_t = self._target_scaled(pel_t, self.pel_target_index)
            thermal_forcing_t = self.thermal_forcing(input_t, qroom_scaled_t, temperature_t)
            matrix_t, bias_t = self.transition_matrix_and_bias(metadata, thermal_forcing_t)
            state_next = self._apply_state_transition(matrix_t, bias_t, state_t)
            q_response_state_next = self.next_q_response_state(
                metadata,
                q_response_state_t,
                qroom_t,
            )
            temperature_next = self.decode_temperature(
                state_next,
                temperature_t,
                alpha,
                q_response_state_next,
            )
            prediction_t = jnp.stack([temperature_next, qroom_scaled_t, pel_scaled_t])
            return detach_carry(
                (state_next, energy_next, temperature_next, q_response_state_next),
                time_index,
                self.bptt_truncate_steps,
            ), (
                prediction_t,
                pi_t,
                qroom_t,
                pel_t,
                state_next,
                jnp.linalg.norm(matrix_t),
            )

        _, outputs = jax.lax.scan(
            step,
            (state0, energy0, initial_temperature[0], q_response_state0),
            (jnp.arange(inputs.shape[0]), inputs),
        )
        predictions, pi, qroom, pel, state, matrix_norm = outputs
        aux = (pi, qroom, pel, state, matrix_norm)
        return predictions, aux

    def __call__(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> jnp.ndarray:
        predictions, _ = self.rollout_with_aux(metadata, inputs, initial_temperature)
        return predictions
