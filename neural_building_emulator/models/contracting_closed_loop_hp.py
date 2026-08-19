"""Contractive closed-loop HP emulator.

This model keeps a globally contractive bounded recurrent state and projects the
HP/electrical outputs through a bounded leaky hydronic buffer.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import MLP

ContractingClosedLoopAux = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


class ContractingClosedLoopHPEmulator(eqx.Module):
    """Data-driven closed-loop emulator with a contractive recurrent map.

    The recurrent map has the form

        s[t+1] = state_bound * tanh(M_t @ (s[t] / state_bound) + r_t)

    where ``M_t`` is generated from metadata and exogenous inputs and normalized
    so that its Frobenius norm is at most ``contraction_gamma``. Since
    ``||M_t||_2 <= ||M_t||_F`` and tanh is 1-Lipschitz, the Jacobian of the
    state transition with respect to ``s[t]`` is bounded by
    ``contraction_gamma`` for every input sequence.
    """

    input_encoder: MLP | None
    transition_net: MLP
    x0_net: MLP
    e0_net: MLP
    output_net: MLP
    temperature_net: MLP
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
    temperature_target_index: int = eqx.field(static=True)
    qroom_target_index: int = eqx.field(static=True)
    pel_target_index: int = eqx.field(static=True)

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
        if len(input_mean) != input_dim or len(input_scale) != input_dim:
            raise ValueError("input_mean and input_scale must match input_dim")
        if len(target_mean) != 3 or len(target_scale) != 3:
            raise ValueError("closed_loop_hp target scalers must have exactly 3 outputs")

        encoder_key, transition_key, x0_key, e0_key, output_key, temperature_key = jax.random.split(key, 6)
        encoded_input_dim = input_dim if input_encoder_dim is None else input_encoder_dim
        thermal_forcing_dim = input_dim + 2
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
        self.output_net = MLP(
            metadata_dim + encoded_input_dim + 3,
            3,
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
        self.temperature_target_index = 0
        self.qroom_target_index = 1
        self.pel_target_index = 2

    def encode_input(self, input_t: jnp.ndarray) -> jnp.ndarray:
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
    ) -> jnp.ndarray:
        bound = jnp.asarray(self.temperature_output_scale, dtype=raw_temperature.dtype)
        if self.temperature_delta_max_c <= 0.0:
            return bound * jnp.tanh(raw_temperature)
        target_scale = jnp.maximum(
            jnp.abs(jnp.asarray(self.target_scale[self.temperature_target_index], dtype=raw_temperature.dtype)),
            jnp.asarray(1e-6, dtype=raw_temperature.dtype),
        )
        delta_bound = jnp.asarray(self.temperature_delta_max_c, dtype=raw_temperature.dtype) / target_scale
        temperature_scaled = previous_temperature_scaled + delta_bound * jnp.tanh(raw_temperature)
        return jnp.clip(temperature_scaled, -bound, bound)

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
        return jnp.concatenate(
            [
                metadata,
                encoded_input_t,
                jnp.asarray([temperature_t], dtype=encoded_input_t.dtype),
                jnp.asarray([self._setpoint_gap_scaled(input_t, temperature_t)], dtype=encoded_input_t.dtype),
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
        return jnp.concatenate(
            [
                input_t[1:],
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
    ) -> jnp.ndarray:
        raw_temperature = self.temperature_net(
            state_t / jnp.asarray(self.state_bound, dtype=state_t.dtype)
        )[0]
        return self._temperature_output(raw_temperature, previous_temperature_scaled)

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

    def transition_matrix_and_bias(
        self,
        metadata: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        raw = self.transition_net(jnp.concatenate([metadata, thermal_forcing_t], axis=0))
        raw_matrix = raw[: self.state_dim * self.state_dim].reshape((self.state_dim, self.state_dim))
        raw_bias = raw[self.state_dim * self.state_dim :]
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
        return matrix, bias

    def one_step_state(
        self,
        metadata: jnp.ndarray,
        thermal_forcing_t: jnp.ndarray,
        state_t: jnp.ndarray,
    ) -> jnp.ndarray:
        matrix, bias = self.transition_matrix_and_bias(metadata, thermal_forcing_t)
        scaled_state = state_t / jnp.asarray(self.state_bound, dtype=state_t.dtype)
        raw_next = matrix @ scaled_state + bias
        return jnp.asarray(self.state_bound, dtype=state_t.dtype) * jnp.tanh(raw_next)

    def decode_hp(
        self,
        metadata: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
        input_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        features = self.hp_features(metadata, encoded_input_t, input_t, energy_t, temperature_t)
        qroom_logit, pel_logit, mode_logit = self.output_net(features)
        pi_t = jax.nn.sigmoid(mode_logit)
        requested_qroom_t = self._positive_capped_from_logit(
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

        def step(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            input_t: jnp.ndarray,
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            state_t, energy_t, temperature_t = carry
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
            state_next = self.one_step_state(metadata, thermal_forcing_t, state_t)
            temperature_next = self.decode_temperature(state_next, temperature_t)
            prediction_t = jnp.stack([temperature_next, qroom_scaled_t, pel_scaled_t])
            matrix, _ = self.transition_matrix_and_bias(metadata, thermal_forcing_t)
            return (state_next, energy_next, temperature_next), (
                prediction_t,
                pi_t,
                qroom_t,
                pel_t,
                state_next,
                jnp.linalg.norm(matrix),
            )

        _, outputs = jax.lax.scan(step, (state0, energy0, initial_temperature[0]), inputs)
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
