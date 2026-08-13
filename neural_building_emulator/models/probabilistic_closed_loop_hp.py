"""Probabilistic closed-loop heat-pump plus thermal emulator."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import MLP, ParameterSlices, parameter_slices
from .schur import SchurMode, simba_schur_matrix
from .state_space import StateSpaceMatrices

ProcessNoiseMode = Literal["none", "constant", "heteroscedastic"]
HPElectricScenarioMode = Literal["expected", "bernoulli"]
ProbClosedLoopAux = tuple[
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
]


class ProbabilisticClosedLoopHPEmulator(eqx.Module):
    """Closed-loop HP emulator with persistent sampled systems and HP emissions."""

    theta_net: MLP
    x0_net: MLP
    w0_net: MLP
    e0_net: MLP
    hp_param_net: MLP
    mode_net: MLP
    pel_mu_net: MLP
    pel_sigma_net: MLP
    qroom_net: MLP
    w_net: MLP
    thermal_encoder: MLP | None
    process_noise_net: MLP | None
    raw_process_scale: jnp.ndarray
    slices: ParameterSlices = eqx.field(static=True)
    metadata_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    encoded_input_dim: int = eqx.field(static=True)
    controller_state_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    process_noise_mode: ProcessNoiseMode = eqx.field(static=True)
    process_noise_init: float = eqx.field(static=True)
    process_noise_floor: float = eqx.field(static=True)
    process_noise_cap: float = eqx.field(static=True)
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
    temperature_target_index: int = eqx.field(static=True)
    qroom_target_index: int = eqx.field(static=True)
    pel_target_index: int = eqx.field(static=True)

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
        key: jax.Array,
    ) -> None:
        if controller_state_dim < 1:
            raise ValueError("controller_state_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
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
        if input_encoder_dim is not None and input_encoder_dim < 1:
            raise ValueError("input_encoder_dim must be positive or None")
        if input_encoder_depth < 1:
            raise ValueError("input_encoder_depth must be at least 1")
        if process_noise_mode not in ("none", "constant", "heteroscedastic"):
            raise ValueError("process_noise_mode must be 'none', 'constant', or 'heteroscedastic'")
        if process_noise_cap <= 0.0:
            raise ValueError("process_noise_cap must be positive")
        if len(input_mean) != input_dim or len(input_scale) != input_dim:
            raise ValueError("input_mean and input_scale must match input_dim")
        if len(target_mean) != 3 or len(target_scale) != 3:
            raise ValueError("closed-loop probabilistic target scalers must have exactly 3 outputs")

        (
            theta_key,
            x0_key,
            w0_key,
            e0_key,
            hp_param_key,
            mode_key,
            pel_mu_key,
            pel_sigma_key,
            qroom_key,
            w_key,
            encoder_key,
            process_key,
        ) = jax.random.split(key, 12)

        thermal_feature_dim = 7
        encoded_input_dim = thermal_feature_dim if input_encoder_dim is None else input_encoder_dim
        slices = parameter_slices(state_dim, encoded_input_dim, 1)
        controller_feature_dim = input_dim + controller_state_dim + 4
        generator_input_dim = metadata_dim + latent_dim
        initial_input_dim = metadata_dim + 1 + latent_dim

        self.theta_net = MLP(generator_input_dim, slices.total, hidden_dim=hidden_dim, depth=depth, key=theta_key)
        self.x0_net = MLP(initial_input_dim, state_dim, hidden_dim=hidden_dim, depth=depth, key=x0_key)
        self.w0_net = MLP(
            initial_input_dim,
            controller_state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=w0_key,
        )
        self.e0_net = MLP(initial_input_dim, 1, hidden_dim=hidden_dim, depth=depth, key=e0_key)
        self.hp_param_net = MLP(generator_input_dim, 3, hidden_dim=hidden_dim, depth=depth, key=hp_param_key)
        self.mode_net = MLP(controller_feature_dim, 1, hidden_dim=hidden_dim, depth=depth, key=mode_key)
        self.pel_mu_net = MLP(controller_feature_dim, 1, hidden_dim=hidden_dim, depth=depth, key=pel_mu_key)
        self.pel_sigma_net = MLP(
            controller_feature_dim,
            1,
            hidden_dim=hidden_dim,
            depth=depth,
            key=pel_sigma_key,
        )
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
        self.process_noise_net = None
        if process_noise_mode == "heteroscedastic":
            self.process_noise_net = MLP(
                metadata_dim + encoded_input_dim + state_dim,
                state_dim,
                hidden_dim=hidden_dim,
                depth=2,
                key=process_key,
            )

        self.raw_process_scale = jnp.full((state_dim,), process_noise_init)
        self.slices = slices
        self.metadata_dim = metadata_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.encoded_input_dim = encoded_input_dim
        self.controller_state_dim = controller_state_dim
        self.latent_dim = latent_dim
        self.process_noise_mode = process_noise_mode
        self.process_noise_init = process_noise_init
        self.process_noise_floor = process_noise_floor
        self.process_noise_cap = process_noise_cap
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
        self.temperature_target_index = 0
        self.qroom_target_index = 1
        self.pel_target_index = 2

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

    def matrices(self, metadata: jnp.ndarray, xi: jnp.ndarray | None = None) -> StateSpaceMatrices:
        if xi is None:
            xi = jnp.zeros((self.latent_dim,), dtype=metadata.dtype)
        generator_input = jnp.concatenate([metadata, xi], axis=0)
        theta = self.theta_scale * self.theta_net(generator_input)
        return self.decode_matrices(theta)

    def hp_parameters(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raw = self.hp_param_net(jnp.concatenate([metadata, xi], axis=0))
        cop_intercept = jnp.asarray(3.0, dtype=raw.dtype) + jnp.asarray(0.1, dtype=raw.dtype) * raw[0]
        cop_slope = jnp.asarray(0.02, dtype=raw.dtype) + jnp.asarray(0.005, dtype=raw.dtype) * raw[1]
        energy_loss_rate = jax.nn.softplus(jnp.asarray(-4.0, dtype=raw.dtype) + 0.1 * raw[2])
        return cop_intercept, cop_slope, energy_loss_rate

    def _physical_input(self, input_t: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            input_t[index] * jnp.asarray(self.input_scale[index], dtype=input_t.dtype)
            + jnp.asarray(self.input_mean[index], dtype=input_t.dtype)
        )

    def _target_scaled(self, value: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            value - jnp.asarray(self.target_mean[index], dtype=value.dtype)
        ) / jnp.asarray(self.target_scale[index], dtype=value.dtype)

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

    def initial_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        xi: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.x0_net(jnp.concatenate([metadata, initial_temperature, xi], axis=0))

    def initial_controller_state(
        self,
        metadata: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        xi: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        features = jnp.concatenate([metadata, initial_temperature, xi], axis=0)
        w0 = jnp.tanh(self.w0_net(features))
        e0 = (
            jax.nn.softplus(self.e0_net(features)[0])
            * jnp.asarray(self.energy_scale, dtype=metadata.dtype)
        )
        e0 = self._nonnegative_capped(e0, self.hp_energy_cap_wh_m2)
        return w0, e0

    def hp_emission(
        self,
        features: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pi_t = jax.nn.sigmoid(self.mode_net(features)[0])
        pel_scale = jnp.asarray(self.target_scale[self.pel_target_index], dtype=features.dtype)
        pel_mean = jnp.asarray(self.target_mean[self.pel_target_index], dtype=features.dtype)
        max_active = jnp.maximum(
            pel_mean + jnp.asarray(8.0, dtype=features.dtype) * pel_scale,
            jnp.asarray(2.0, dtype=features.dtype) * pel_scale,
        )
        if self.hp_pel_cap_w_m2 > 0.0:
            max_active = jnp.asarray(self.hp_pel_cap_w_m2, dtype=features.dtype)
        active_level = max_active * jax.nn.sigmoid(self.pel_mu_net(features)[0])
        log_mu = jnp.log1p(active_level + jnp.asarray(1e-6, dtype=features.dtype))
        log_sigma = (
            jnp.asarray(0.05, dtype=features.dtype)
            + jnp.asarray(0.70, dtype=features.dtype) * jax.nn.sigmoid(self.pel_sigma_net(features)[0])
        )
        expected_active = jnp.maximum(
            jnp.expm1(log_mu + 0.5 * log_sigma**2),
            jnp.asarray(0.0, dtype=features.dtype),
        )
        expected_active = self._nonnegative_capped(expected_active, self.hp_pel_cap_w_m2)
        expected_total = self._nonnegative_capped(pi_t * expected_active, self.hp_pel_cap_w_m2)
        return pi_t, log_mu, log_sigma, expected_active, expected_total

    def one_step_augmented_state(
        self,
        metadata: jnp.ndarray,
        input_t: jnp.ndarray,
        xi: jnp.ndarray,
        augmented_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """One deterministic expected-HP closed-loop update for s=[x,w,E,T]."""
        matrices = self.matrices(metadata, xi)
        cop_intercept, cop_slope, loss_rate = self.hp_parameters(metadata, xi)
        x_t = augmented_state[: self.state_dim]
        w_start = self.state_dim
        w_end = w_start + self.controller_state_dim
        w_t = augmented_state[w_start:w_end]
        energy_t = augmented_state[w_end]
        temperature_t = augmented_state[w_end + 1]
        dt_h = jnp.asarray(self.hp_dt_hours, dtype=input_t.dtype)

        features = self._controller_features(input_t, temperature_t, energy_t, w_t)
        _, _, _, _, pel_t = self.hp_emission(features)
        cop_t = self._cop(input_t, cop_intercept, cop_slope)
        qroom_raw_t = (
            jax.nn.softplus(self.qroom_net(features)[0])
            * jnp.asarray(self.target_scale[self.qroom_target_index], dtype=input_t.dtype)
        )
        qroom_raw_t = self._nonnegative_capped(qroom_raw_t, self.hp_qroom_cap_w_m2)
        qroom_t, _, energy_next = self._buffer_draw(
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
        return jnp.concatenate(
            [
                x_next,
                w_next,
                energy_next[jnp.newaxis],
                temperature_next[jnp.newaxis],
            ],
            axis=0,
        )

    def closed_loop_jacobian_spectral_norm(
        self,
        metadata: jnp.ndarray,
        input_t: jnp.ndarray,
        xi: jnp.ndarray,
        augmented_state: jnp.ndarray,
    ) -> jnp.ndarray:
        jacobian = jax.jacrev(
            lambda state: self.one_step_augmented_state(metadata, input_t, xi, state)
        )(augmented_state)
        return jnp.max(jnp.linalg.svd(jacobian, compute_uv=False))

    def process_scale(
        self,
        metadata: jnp.ndarray,
        encoded_input: jnp.ndarray,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.process_noise_mode == "none":
            return jnp.zeros((self.state_dim,), dtype=state.dtype)
        if self.process_noise_mode == "constant":
            raw = self.raw_process_scale
        else:
            assert self.process_noise_net is not None
            process_input = jnp.concatenate([metadata, encoded_input, state], axis=0)
            raw = self.process_noise_init + 0.1 * self.process_noise_net(process_input)
        return (
            jnp.asarray(self.process_noise_floor, dtype=state.dtype)
            + jnp.asarray(self.process_noise_cap, dtype=state.dtype) * jax.nn.sigmoid(raw)
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
        *,
        hp_scenario_mode: HPElectricScenarioMode = "expected",
    ) -> tuple[jnp.ndarray, ProbClosedLoopAux]:
        if hp_scenario_mode not in ("expected", "bernoulli"):
            raise ValueError("hp_scenario_mode must be 'expected' or 'bernoulli'")

        matrices = self.matrices(metadata, xi)
        x0 = self.initial_state(metadata, initial_temperature, xi)
        w0, e0 = self.initial_controller_state(metadata, initial_temperature, xi)
        cop_intercept, cop_slope, loss_rate = self.hp_parameters(metadata, xi)
        temperature0 = initial_temperature[0]
        dt_h = jnp.asarray(self.hp_dt_hours, dtype=inputs.dtype)

        def step(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            step_inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            tuple[
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
            ],
        ]:
            x_t, w_t, energy_t, temperature_t = carry
            input_t, eps_t, mode_uniform_t, power_noise_t = step_inputs
            features = self._controller_features(input_t, temperature_t, energy_t, w_t)

            pi_t, log_mu_t, log_sigma_t, expected_active_t, expected_pel_t = self.hp_emission(features)
            if hp_scenario_mode == "expected":
                hp_on_t = pi_t
                pel_active_t = expected_active_t
                pel_t = expected_pel_t
            else:
                hp_on_t = (mode_uniform_t < pi_t).astype(input_t.dtype)
                sampled_log_active_t = log_mu_t + log_sigma_t * power_noise_t
                pel_active_t = self._nonnegative_capped(
                    jnp.expm1(sampled_log_active_t),
                    self.hp_pel_cap_w_m2,
                )
                pel_t = self._nonnegative_capped(hp_on_t * pel_active_t, self.hp_pel_cap_w_m2)

            cop_t = self._cop(input_t, cop_intercept, cop_slope)
            qroom_raw_t = (
                jax.nn.softplus(self.qroom_net(features)[0])
                * jnp.asarray(self.target_scale[self.qroom_target_index], dtype=input_t.dtype)
            )
            qroom_raw_t = self._nonnegative_capped(qroom_raw_t, self.hp_qroom_cap_w_m2)
            qroom_t, available_power_t, energy_next = self._buffer_draw(
                energy_t,
                cop_t * pel_t,
                qroom_raw_t,
                dt_h,
                loss_rate,
            )
            qroom_scaled_t = self._target_scaled(qroom_t, self.qroom_target_index)

            z_t = self._thermal_features(input_t, temperature_t, qroom_scaled_t)
            scale_t = self.process_scale(metadata, z_t, x_t)
            x_next = matrices.a @ x_t + matrices.b @ z_t + matrices.state_bias + scale_t * eps_t
            temperature_next = (matrices.c @ x_next + matrices.output_bias)[0]

            w_next = jnp.tanh(self.w_net(features))
            pel_scaled_t = self._target_scaled(pel_t, self.pel_target_index)
            prediction_t = jnp.stack([temperature_next, qroom_scaled_t, pel_scaled_t])
            aux_t = (
                pi_t,
                cop_t,
                energy_t,
                available_power_t,
                qroom_raw_t,
                hp_on_t,
                pel_active_t,
                pel_t,
                log_mu_t,
                log_sigma_t,
                x_t,
                w_t,
                temperature_t,
            )
            return (x_next, w_next, energy_next, temperature_next), (prediction_t, *aux_t)

        _, outputs = jax.lax.scan(
            step,
            (x0, w0, e0, temperature0),
            (inputs, process_noise, hp_mode_uniform, hp_power_noise),
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
        hp_scenario_mode: HPElectricScenarioMode = "expected",
    ) -> tuple[jnp.ndarray, ProbClosedLoopAux]:
        if num_particles < 1:
            raise ValueError("num_particles must be positive")
        if hp_scenario_mode not in ("expected", "bernoulli"):
            raise ValueError("hp_scenario_mode must be 'expected' or 'bernoulli'")
        xi_key, process_key, mode_key, power_key = jax.random.split(key, 4)
        xi = jax.random.normal(xi_key, (num_particles, self.latent_dim))
        if sample_process_noise and self.process_noise_mode != "none":
            process_noise = jax.random.normal(process_key, (num_particles, inputs.shape[0], self.state_dim))
        else:
            process_noise = jnp.zeros((num_particles, inputs.shape[0], self.state_dim))
        hp_mode_uniform = jax.random.uniform(mode_key, (num_particles, inputs.shape[0]))
        hp_power_noise = jax.random.normal(power_key, (num_particles, inputs.shape[0]))
        predictions, aux = jax.vmap(
            lambda particle_xi, particle_process_noise, particle_mode_uniform, particle_power_noise: self.rollout_particle(
                metadata,
                inputs,
                initial_temperature,
                particle_xi,
                particle_process_noise,
                particle_mode_uniform,
                particle_power_noise,
                hp_scenario_mode=hp_scenario_mode,
            )
        )(xi, process_noise, hp_mode_uniform, hp_power_noise)
        return predictions, (*aux, xi)

    def sample(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool = True,
        hp_scenario_mode: HPElectricScenarioMode = "expected",
    ) -> jnp.ndarray:
        predictions, _ = self.sample_with_aux(
            metadata,
            inputs,
            initial_temperature,
            key=key,
            num_particles=num_particles,
            sample_process_noise=sample_process_noise,
            hp_scenario_mode=hp_scenario_mode,
        )
        return predictions

    def parameter_regularization(self, metadata: jnp.ndarray) -> jnp.ndarray:
        xi = jnp.zeros((self.latent_dim,), dtype=metadata.dtype)
        matrices = self.matrices(metadata, xi)
        bias = jnp.mean(matrices.state_bias**2) + jnp.mean(matrices.output_bias**2)
        process_scale = jax.nn.softplus(self.raw_process_scale) + self.process_noise_floor
        cop_intercept, cop_slope, loss_rate = self.hp_parameters(metadata, xi)
        hp = (cop_intercept - 3.0) ** 2 + 10.0 * (cop_slope - 0.02) ** 2 + loss_rate**2
        return 0.1 * bias + jnp.mean(process_scale**2) + 0.01 * hp

    def __call__(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> jnp.ndarray:
        xi = jnp.zeros((self.latent_dim,), dtype=metadata.dtype)
        process_noise = jnp.zeros((inputs.shape[0], self.state_dim), dtype=inputs.dtype)
        hp_mode_uniform = jnp.zeros((inputs.shape[0],), dtype=inputs.dtype)
        hp_power_noise = jnp.zeros((inputs.shape[0],), dtype=inputs.dtype)
        predictions, _ = self.rollout_particle(
            metadata,
            inputs,
            initial_temperature,
            xi,
            process_noise,
            hp_mode_uniform,
            hp_power_noise,
            hp_scenario_mode="expected",
        )
        return predictions
