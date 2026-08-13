"""Probabilistic contractive closed-loop HP emulator."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import MLP
from .probabilistic_closed_loop_hp import HPElectricScenarioMode, ProcessNoiseMode, ProbClosedLoopAux


class ProbabilisticContractingClosedLoopHPEmulator(eqx.Module):
    """Probabilistic closed-loop emulator with bounded contractive state dynamics.

    Each particle samples a persistent latent variable ``xi``. The recurrent
    transition matrix is generated from metadata, ``xi``, and the current
    exogenous input, then Frobenius-normalized so its spectral norm is at most
    ``contraction_gamma``. Process noise is injected inside the bounded tanh
    transition, so the latent state remains bounded for all scenarios.
    """

    input_encoder: MLP | None
    transition_net: MLP
    x0_net: MLP
    e0_net: MLP
    hp_param_net: MLP
    output_net: MLP
    process_noise_net: MLP | None
    raw_process_scale: jnp.ndarray
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
    contraction_gamma: float = eqx.field(static=True)
    state_bound: float = eqx.field(static=True)
    temperature_output_scale: float = eqx.field(static=True)
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
    outdoor_input_index: int = eqx.field(static=True)
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
        contraction_gamma: float = 0.99,
        state_bound: float = 5.0,
        temperature_output_scale: float = 8.0,
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
        if not 0.0 < contraction_gamma < 1.0:
            raise ValueError("contraction_gamma must be in (0, 1)")
        if state_bound <= 0.0:
            raise ValueError("state_bound must be positive")
        if temperature_output_scale <= 0.0:
            raise ValueError("temperature_output_scale must be positive")
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
            raise ValueError("closed-loop probabilistic target scalers must have exactly 3 outputs")

        (
            encoder_key,
            transition_key,
            x0_key,
            e0_key,
            hp_param_key,
            output_key,
            process_key,
        ) = jax.random.split(key, 7)
        encoded_input_dim = input_dim if input_encoder_dim is None else input_encoder_dim
        generator_input_dim = metadata_dim + latent_dim
        transition_input_dim = generator_input_dim + encoded_input_dim
        transition_output_dim = state_dim * state_dim + state_dim
        output_input_dim = state_dim + encoded_input_dim + latent_dim

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
        self.x0_net = MLP(metadata_dim + 1 + latent_dim, state_dim, hidden_dim=hidden_dim, depth=depth, key=x0_key)
        self.e0_net = MLP(metadata_dim + 1 + latent_dim, 1, hidden_dim=hidden_dim, depth=depth, key=e0_key)
        self.hp_param_net = MLP(generator_input_dim, 3, hidden_dim=hidden_dim, depth=depth, key=hp_param_key)
        self.output_net = MLP(output_input_dim, 5, hidden_dim=hidden_dim, depth=depth, key=output_key)
        self.process_noise_net = None
        if process_noise_mode == "heteroscedastic":
            self.process_noise_net = MLP(
                generator_input_dim + encoded_input_dim + state_dim,
                state_dim,
                hidden_dim=hidden_dim,
                depth=2,
                key=process_key,
            )

        self.raw_process_scale = jnp.full((state_dim,), process_noise_init)
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
        self.contraction_gamma = contraction_gamma
        self.state_bound = state_bound
        self.temperature_output_scale = temperature_output_scale
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
        self.outdoor_input_index = 1
        self.temperature_target_index = 0
        self.qroom_target_index = 1
        self.pel_target_index = 2

    def encode_input(self, input_t: jnp.ndarray) -> jnp.ndarray:
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
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raw = self.hp_param_net(jnp.concatenate([metadata, xi], axis=0))
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

    def transition_matrix_and_bias(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        raw = self.transition_net(jnp.concatenate([metadata, xi, encoded_input_t], axis=0))
        raw_matrix = raw[: self.state_dim * self.state_dim].reshape((self.state_dim, self.state_dim))
        raw_bias = raw[self.state_dim * self.state_dim :]
        frobenius = jnp.linalg.norm(raw_matrix)
        divisor = jnp.maximum(frobenius, jnp.asarray(1.0, dtype=raw_matrix.dtype))
        matrix = jnp.asarray(self.contraction_gamma, dtype=raw_matrix.dtype) * raw_matrix / divisor
        bias = jnp.tanh(raw_bias)
        return matrix, bias

    def process_scale(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
        state_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.process_noise_mode == "none":
            return jnp.zeros((self.state_dim,), dtype=state_t.dtype)
        if self.process_noise_mode == "constant":
            raw = self.raw_process_scale
        else:
            assert self.process_noise_net is not None
            raw = self.process_noise_init + 0.1 * self.process_noise_net(
                jnp.concatenate([metadata, xi, encoded_input_t, state_t], axis=0)
            )
        return (
            jnp.asarray(self.process_noise_floor, dtype=state_t.dtype)
            + jnp.asarray(self.process_noise_cap, dtype=state_t.dtype) * jax.nn.sigmoid(raw)
        )

    def one_step_state(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
        state_t: jnp.ndarray,
        eps_t: jnp.ndarray,
    ) -> jnp.ndarray:
        matrix, bias = self.transition_matrix_and_bias(metadata, xi, encoded_input_t)
        scale_t = self.process_scale(metadata, xi, encoded_input_t, state_t)
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
        active_level = self._positive_capped_from_logit(
            pel_mu_logit,
            self.hp_pel_cap_w_m2,
            self.pel_target_index,
        )
        log_mu = jnp.log1p(active_level + jnp.asarray(1e-6, dtype=active_level.dtype))
        log_sigma = (
            jnp.asarray(0.05, dtype=active_level.dtype)
            + jnp.asarray(0.70, dtype=active_level.dtype) * jax.nn.sigmoid(pel_sigma_logit)
        )
        expected_active = jnp.maximum(
            jnp.expm1(log_mu + 0.5 * log_sigma**2),
            jnp.asarray(0.0, dtype=active_level.dtype),
        )
        expected_active = self._nonnegative_capped(expected_active, self.hp_pel_cap_w_m2)
        expected_total = self._nonnegative_capped(pi_t * expected_active, self.hp_pel_cap_w_m2)
        return pi_t, log_mu, log_sigma, expected_active, expected_total

    def decode_output(
        self,
        state_t: jnp.ndarray,
        encoded_input_t: jnp.ndarray,
        input_t: jnp.ndarray,
        energy_t: jnp.ndarray,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        mode_uniform_t: jnp.ndarray,
        power_noise_t: jnp.ndarray,
        *,
        hp_scenario_mode: HPElectricScenarioMode,
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
    ]:
        features = jnp.concatenate(
            [
                state_t / jnp.asarray(self.state_bound, dtype=state_t.dtype),
                encoded_input_t,
                xi,
            ],
            axis=0,
        )
        raw_temperature, qroom_logit, mode_logit, pel_mu_logit, pel_sigma_logit = self.output_net(features)
        temperature_scaled = (
            jnp.asarray(self.temperature_output_scale, dtype=raw_temperature.dtype)
            * jnp.tanh(raw_temperature)
        )
        requested_qroom_t = self._positive_capped_from_logit(
            qroom_logit,
            self.hp_qroom_cap_w_m2,
            self.qroom_target_index,
        )
        pi_t, log_mu_t, log_sigma_t, expected_active_t, expected_pel_t = self.hp_emission(
            features,
            pel_mu_logit,
            pel_sigma_logit,
            mode_logit,
        )
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

        cop_intercept, cop_slope, loss_rate = self.hp_parameters(metadata, xi)
        cop_t = self._cop(input_t, cop_intercept, cop_slope)
        qroom_t, available_power_t, energy_next = self._buffer_draw(
            energy_t,
            cop_t * pel_t,
            requested_qroom_t,
            jnp.asarray(self.hp_dt_hours, dtype=state_t.dtype),
            loss_rate,
        )
        qroom_scaled_t = self._target_scaled(qroom_t, self.qroom_target_index)
        pel_scaled_t = self._target_scaled(pel_t, self.pel_target_index)
        prediction_t = jnp.stack([temperature_scaled, qroom_scaled_t, pel_scaled_t])
        return (
            prediction_t,
            pi_t,
            cop_t,
            energy_next,
            available_power_t,
            requested_qroom_t,
            hp_on_t,
            pel_active_t,
            pel_t,
            log_mu_t,
            log_sigma_t,
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
        state0 = self.initial_state(metadata, initial_temperature, xi)
        energy0 = self.initial_energy(metadata, initial_temperature, xi)
        temperature0 = initial_temperature[0]
        w_placeholder = jnp.zeros((self.controller_state_dim,), dtype=inputs.dtype)

        def step(
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            step_inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
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
            state_t, energy_t, temperature_t = carry
            input_t, eps_t, mode_uniform_t, power_noise_t = step_inputs
            encoded_input_t = self.encode_input(input_t)
            state_next = self.one_step_state(metadata, xi, encoded_input_t, state_t, eps_t)
            (
                prediction_t,
                pi_t,
                cop_t,
                energy_next,
                available_power_t,
                requested_qroom_t,
                hp_on_t,
                pel_active_t,
                pel_t,
                log_mu_t,
                log_sigma_t,
            ) = self.decode_output(
                state_next,
                encoded_input_t,
                input_t,
                energy_t,
                metadata,
                xi,
                mode_uniform_t,
                power_noise_t,
                hp_scenario_mode=hp_scenario_mode,
            )
            temperature_next = prediction_t[0]
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
                w_placeholder,
                temperature_t,
            )
            return (state_next, energy_next, temperature_next), (prediction_t, *aux_t)

        _, outputs = jax.lax.scan(
            step,
            (state0, energy0, temperature0),
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

    def one_step_augmented_state(
        self,
        metadata: jnp.ndarray,
        input_t: jnp.ndarray,
        xi: jnp.ndarray,
        augmented_state: jnp.ndarray,
    ) -> jnp.ndarray:
        state_t = augmented_state[: self.state_dim]
        energy_index = self.state_dim + self.controller_state_dim
        energy_t = augmented_state[energy_index]
        encoded_input_t = self.encode_input(input_t)
        state_next = self.one_step_state(
            metadata,
            xi,
            encoded_input_t,
            state_t,
            jnp.zeros((self.state_dim,), dtype=state_t.dtype),
        )
        prediction_t, _, _, energy_next, *_ = self.decode_output(
            state_next,
            encoded_input_t,
            input_t,
            energy_t,
            metadata,
            xi,
            jnp.asarray(0.0, dtype=state_t.dtype),
            jnp.asarray(0.0, dtype=state_t.dtype),
            hp_scenario_mode="expected",
        )
        w_next = jnp.zeros((self.controller_state_dim,), dtype=state_t.dtype)
        return jnp.concatenate(
            [
                state_next,
                w_next,
                energy_next[jnp.newaxis],
                prediction_t[0:1],
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

    def parameter_regularization(self, metadata: jnp.ndarray) -> jnp.ndarray:
        del metadata
        return jnp.asarray(0.0)
