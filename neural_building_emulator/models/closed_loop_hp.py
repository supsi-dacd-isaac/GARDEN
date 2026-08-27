"""Closed-loop heat-pump plus thermal state-space emulator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import MLP, ParameterSlices, parameter_slices
from .schur import SchurMode, simba_schur_matrix
from .state_space import StateSpaceMatrices
from .truncated_bptt import DEFAULT_BPTT_TRUNCATE_STEPS, detach_carry, validate_bptt_truncate_steps

ClosedLoopAux = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


class ClosedLoopHPEmulator(eqx.Module):
    """A differentiable space-heating controller, buffer, and thermal emulator."""

    theta_net: MLP
    x0_net: MLP
    w0_net: MLP
    e0_net: MLP
    mode_net: MLP
    pel_net: MLP
    qroom_net: MLP
    w_net: MLP
    thermal_encoder: MLP | None
    cop_intercept: jax.Array
    cop_slope: jax.Array
    raw_energy_loss_rate: jax.Array
    slices: ParameterSlices = eqx.field(static=True)
    metadata_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    encoded_input_dim: int = eqx.field(static=True)
    controller_state_dim: int = eqx.field(static=True)
    hp_dt_hours: float = eqx.field(static=True)
    hp_cop_floor: float = eqx.field(static=True)
    hp_cop_cap: float = eqx.field(static=True)
    hp_pel_cap_w_m2: float = eqx.field(static=True)
    hp_qroom_cap_w_m2: float = eqx.field(static=True)
    hp_energy_cap_wh_m2: float = eqx.field(static=True)
    schur_gamma: float = eqx.field(static=True)
    pf_lambda_min: float = eqx.field(static=True)
    schur_eps: float = eqx.field(static=True)
    schur_mode: SchurMode = eqx.field(static=True)
    theta_scale: float = eqx.field(static=True)
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

    def __init__(
        self,
        metadata_dim: int,
        input_dim: int,
        *,
        state_dim: int = 6,
        controller_state_dim: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        input_encoder_dim: int | None = None,
        input_encoder_hidden_dim: int | None = None,
        input_encoder_depth: int = 2,
        hp_dt_hours: float = 0.25,
        hp_cop_floor: float = 1.0,
        hp_cop_cap: float = 0.0,
        hp_pel_cap_w_m2: float = 0.0,
        hp_qroom_cap_w_m2: float = 0.0,
        hp_energy_cap_wh_m2: float = 0.0,
        schur_gamma: float = 0.995,
        pf_lambda_min: float = 0.0,
        schur_eps: float = 1e-4,
        schur_mode: SchurMode = "near_identity",
        theta_scale: float = 0.05,
        input_mean: tuple[float, ...] = (),
        input_scale: tuple[float, ...] = (),
        target_mean: tuple[float, ...] = (),
        target_scale: tuple[float, ...] = (),
        availability_input_index: int = -1,
        bptt_truncate_steps: int = DEFAULT_BPTT_TRUNCATE_STEPS,
        key: jax.Array,
    ) -> None:
        if controller_state_dim < 1:
            raise ValueError("controller_state_dim must be positive")
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
        bptt_truncate_steps = validate_bptt_truncate_steps(bptt_truncate_steps)
        if input_encoder_dim is not None and input_encoder_dim < 1:
            raise ValueError("input_encoder_dim must be positive or None")
        if input_encoder_depth < 1:
            raise ValueError("input_encoder_depth must be at least 1")
        if len(input_mean) != input_dim or len(input_scale) != input_dim:
            raise ValueError("input_mean and input_scale must match input_dim")
        if len(target_mean) != 3 or len(target_scale) != 3:
            raise ValueError("closed_loop_hp target scalers must have exactly 3 outputs")
        if availability_input_index >= input_dim:
            raise ValueError("availability_input_index must be smaller than input_dim")

        (
            theta_key,
            x0_key,
            w0_key,
            e0_key,
            mode_key,
            pel_key,
            qroom_key,
            w_key,
            encoder_key,
        ) = jax.random.split(key, 9)

        thermal_feature_dim = 7
        encoded_input_dim = thermal_feature_dim if input_encoder_dim is None else input_encoder_dim
        slices = parameter_slices(state_dim, encoded_input_dim, 1)
        controller_feature_dim = input_dim + controller_state_dim + 4

        self.theta_net = MLP(metadata_dim, slices.total, hidden_dim=hidden_dim, depth=depth, key=theta_key)
        self.x0_net = MLP(metadata_dim + 1, state_dim, hidden_dim=hidden_dim, depth=depth, key=x0_key)
        self.w0_net = MLP(
            metadata_dim + 1,
            controller_state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=w0_key,
        )
        self.e0_net = MLP(metadata_dim + 1, 1, hidden_dim=hidden_dim, depth=depth, key=e0_key)
        self.mode_net = MLP(controller_feature_dim, 1, hidden_dim=hidden_dim, depth=depth, key=mode_key)
        self.pel_net = MLP(controller_feature_dim, 1, hidden_dim=hidden_dim, depth=depth, key=pel_key)
        self.qroom_net = MLP(controller_feature_dim, 1, hidden_dim=hidden_dim, depth=depth, key=qroom_key)
        self.w_net = MLP(
            controller_feature_dim,
            controller_state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=w_key,
        )
        self.thermal_encoder = None
        if input_encoder_dim is not None:
            self.thermal_encoder = MLP(
                thermal_feature_dim,
                input_encoder_dim,
                hidden_dim=input_encoder_hidden_dim or hidden_dim,
                depth=input_encoder_depth,
                key=encoder_key,
            )

        self.cop_intercept = jnp.asarray(3.0, dtype=jnp.float32)
        self.cop_slope = jnp.asarray(0.02, dtype=jnp.float32)
        self.raw_energy_loss_rate = jnp.asarray(-4.0, dtype=jnp.float32)
        self.slices = slices
        self.metadata_dim = metadata_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.encoded_input_dim = encoded_input_dim
        self.controller_state_dim = controller_state_dim
        self.hp_dt_hours = hp_dt_hours
        self.hp_cop_floor = hp_cop_floor
        self.hp_cop_cap = hp_cop_cap
        self.hp_pel_cap_w_m2 = hp_pel_cap_w_m2
        self.hp_qroom_cap_w_m2 = hp_qroom_cap_w_m2
        self.hp_energy_cap_wh_m2 = hp_energy_cap_wh_m2
        self.schur_gamma = schur_gamma
        self.pf_lambda_min = pf_lambda_min
        self.schur_eps = schur_eps
        self.schur_mode = schur_mode
        self.theta_scale = theta_scale
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

    def decode_matrices(self, theta: jnp.ndarray) -> StateSpaceMatrices:
        slices = self.slices
        a = simba_schur_matrix(
            theta[slices.raw_w],
            theta[slices.raw_v],
            self.state_dim,
            gamma=self.schur_gamma,
            pf_lambda_min=self.pf_lambda_min,
            eps=self.schur_eps,
            mode=self.schur_mode,
        )
        b = theta[slices.b].reshape((self.state_dim, self.encoded_input_dim))
        c = theta[slices.c].reshape((1, self.state_dim))
        d = jnp.zeros((1, self.encoded_input_dim), dtype=theta.dtype)
        state_bias = theta[slices.state_bias]
        output_bias = theta[slices.output_bias]
        return StateSpaceMatrices(a, b, c, d, state_bias, output_bias)

    def matrices(self, metadata: jnp.ndarray) -> StateSpaceMatrices:
        theta = self.theta_scale * self.theta_net(metadata)
        return self.decode_matrices(theta)

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

    def _nonnegative_capped(self, value: jnp.ndarray, cap: float) -> jnp.ndarray:
        value = jnp.maximum(value, jnp.asarray(0.0, dtype=value.dtype))
        if cap <= 0.0:
            return value
        return jnp.minimum(value, jnp.asarray(cap, dtype=value.dtype))

    def _buffer_draw(
        self,
        energy_t: jnp.ndarray,
        generated_heat_power_t: jnp.ndarray,
        requested_room_heat_t: jnp.ndarray,
        dt_h: jnp.ndarray,
        loss_rate: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Charge the hydronic buffer, leak stored heat, then draw room heat."""
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

    def _weather_as_target_scaled(self, input_t: jnp.ndarray, input_index: int) -> jnp.ndarray:
        return self._target_scaled(self._physical_input(input_t, input_index), self.temperature_target_index)

    def _controller_features(
        self,
        input_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        w_t: jnp.ndarray,
    ) -> jnp.ndarray:
        outdoor_gap = self._weather_as_target_scaled(input_t, self.outdoor_input_index) - temperature_t
        setpoint_gap = self._weather_as_target_scaled(input_t, self.setpoint_input_index) - temperature_t
        energy_scaled = energy_t / jnp.asarray(self.energy_scale, dtype=input_t.dtype)
        return jnp.concatenate(
            [
                input_t,
                temperature_t[jnp.newaxis],
                outdoor_gap[jnp.newaxis],
                setpoint_gap[jnp.newaxis],
                energy_scaled[jnp.newaxis],
                w_t,
            ],
            axis=0,
        )

    def _thermal_features(
        self,
        input_t: jnp.ndarray,
        temperature_t: jnp.ndarray,
        qroom_scaled: jnp.ndarray,
    ) -> jnp.ndarray:
        outdoor_gap = self._weather_as_target_scaled(input_t, self.outdoor_input_index) - temperature_t
        setpoint_gap = self._weather_as_target_scaled(input_t, self.setpoint_input_index) - temperature_t
        features = jnp.stack(
            [
                input_t[self.outdoor_input_index],
                input_t[self.solar_input_index],
                input_t[self.ventilation_input_index],
                qroom_scaled,
                temperature_t,
                outdoor_gap,
                setpoint_gap,
            ],
        )
        if self.thermal_encoder is None:
            return features
        return self.thermal_encoder(features)

    def _cop(self, input_t: jnp.ndarray) -> jnp.ndarray:
        outdoor_c = self._physical_input(input_t, self.outdoor_input_index)
        cop = jnp.maximum(self.hp_cop_floor, self.cop_intercept + self.cop_slope * outdoor_c)
        if self.hp_cop_cap <= 0.0:
            return cop
        return jnp.minimum(cop, jnp.asarray(self.hp_cop_cap, dtype=input_t.dtype))

    def initial_state(self, metadata: jnp.ndarray, initial_temperature: jnp.ndarray) -> jnp.ndarray:
        return self.x0_net(jnp.concatenate([metadata, initial_temperature], axis=0))

    def initial_controller_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        features = jnp.concatenate([metadata, initial_temperature], axis=0)
        w0 = jnp.tanh(self.w0_net(features))
        e0 = (
            jax.nn.softplus(self.e0_net(features)[0])
            * jnp.asarray(self.energy_scale, dtype=metadata.dtype)
        )
        e0 = self._nonnegative_capped(e0, self.hp_energy_cap_wh_m2)
        return w0, e0

    def rollout_with_aux(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ClosedLoopAux]:
        matrices = self.matrices(metadata)
        x0 = self.initial_state(metadata, initial_temperature)
        w0, e0 = self.initial_controller_state(metadata, initial_temperature)
        temperature0 = initial_temperature[0]
        loss_rate = jax.nn.softplus(self.raw_energy_loss_rate)
        dt_h = jnp.asarray(self.hp_dt_hours, dtype=inputs.dtype)

        def step(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            step_inputs: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            x_t, w_t, energy_t, temperature_t = carry
            time_index, input_t = step_inputs
            features = self._controller_features(input_t, temperature_t, energy_t, w_t)
            availability_t = self._space_heating_availability(input_t)

            pi_t = availability_t * jax.nn.sigmoid(self.mode_net(features)[0])
            pel_base = (
                jax.nn.softplus(self.pel_net(features)[0])
                * jnp.asarray(self.target_scale[self.pel_target_index], dtype=input_t.dtype)
            )
            pel_t = self._nonnegative_capped(pi_t * pel_base, self.hp_pel_cap_w_m2)
            cop_t = self._cop(input_t)

            qroom_raw_t = (
                jax.nn.softplus(self.qroom_net(features)[0])
                * jnp.asarray(self.target_scale[self.qroom_target_index], dtype=input_t.dtype)
            )
            qroom_raw_t = availability_t * self._nonnegative_capped(
                qroom_raw_t,
                self.hp_qroom_cap_w_m2,
            )
            qroom_t, available_power_t, energy_next = self._buffer_draw(
                energy_t,
                cop_t * pel_t,
                qroom_raw_t,
                dt_h,
                loss_rate,
            )
            qroom_scaled_t = self._target_scaled(qroom_t, self.qroom_target_index)

            z_t = self._thermal_features(input_t, temperature_t, qroom_scaled_t)
            x_next = matrices.a @ x_t + matrices.b @ z_t + matrices.state_bias
            temperature_next = (matrices.c @ x_next + matrices.output_bias)[0]

            w_next = jnp.tanh(self.w_net(features))
            pel_scaled_t = self._target_scaled(pel_t, self.pel_target_index)
            prediction_t = jnp.stack(
                [temperature_next, qroom_scaled_t, pel_scaled_t],
            )
            aux_t = (pi_t, cop_t, energy_t, available_power_t, qroom_raw_t)
            return detach_carry(
                (x_next, w_next, energy_next, temperature_next),
                time_index,
                self.bptt_truncate_steps,
            ), (prediction_t, *aux_t)

        _, outputs = jax.lax.scan(
            step,
            (x0, w0, e0, temperature0),
            (jnp.arange(inputs.shape[0]), inputs),
        )
        predictions, pi, cop, energy, available_power, qroom_raw = outputs
        aux = (pi, cop, energy, available_power, qroom_raw)
        return predictions, aux

    def __call__(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> jnp.ndarray:
        predictions, _ = self.rollout_with_aux(metadata, inputs, initial_temperature)
        return predictions
