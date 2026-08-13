"""Probabilistic metadata-conditioned stable state-space emulator."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import (
    InputEncoderFeedback,
    MLP,
    SwitchingDynamics,
    input_encoder_feedback_extra_dim,
    parameter_slices,
)
from .schur import SchurMode, simba_schur_matrix
from .state_space import OutputTiming, StateSpaceMatrices

ProcessNoiseMode = Literal["none", "constant", "heteroscedastic"]


class ProbabilisticStableStateSpaceEmulator(eqx.Module):
    """Stable SS model with persistent sampled system uncertainty and process noise.

    For each particle, a latent ``xi`` is sampled once and used with metadata to
    generate one stable state-space system that remains fixed over the rollout.
    Inputs are transformed by a memoryless encoder ``z[t] = e(u[t])`` before
    entering the linear latent dynamics.
    """

    input_encoder: MLP
    theta_net: MLP
    x0_net: MLP
    process_noise_net: MLP | None
    raw_process_scale: jnp.ndarray
    state_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    encoded_input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    process_noise_mode: ProcessNoiseMode = eqx.field(static=True)
    process_noise_init: float = eqx.field(static=True)
    schur_gamma: float = eqx.field(static=True)
    pf_lambda_min: float = eqx.field(static=True)
    schur_eps: float = eqx.field(static=True)
    schur_mode: SchurMode = eqx.field(static=True)
    theta_scale: float = eqx.field(static=True)
    process_noise_floor: float = eqx.field(static=True)
    zero_d: bool = eqx.field(static=True)
    output_timing: OutputTiming = eqx.field(static=True)
    input_encoder_feedback: InputEncoderFeedback = eqx.field(static=True)
    switching_dynamics: SwitchingDynamics = eqx.field(static=True)
    switching_alpha_heat_scale: float = eqx.field(static=True)
    switching_alpha_on_weight: float = eqx.field(static=True)
    switching_alpha_recent_weight: float = eqx.field(static=True)
    heat_on_threshold: float = eqx.field(static=True)
    heat_input_index: int = eqx.field(static=True)
    heat_input_mean: float = eqx.field(static=True)
    heat_input_scale: float = eqx.field(static=True)
    heating_on_input_index: int = eqx.field(static=True)
    heating_on_input_mean: float = eqx.field(static=True)
    heating_on_input_scale: float = eqx.field(static=True)
    recently_on_input_index: int = eqx.field(static=True)
    recently_on_input_mean: float = eqx.field(static=True)
    recently_on_input_scale: float = eqx.field(static=True)
    outdoor_temperature_input_index: int = eqx.field(static=True)
    outdoor_temperature_input_mean: float = eqx.field(static=True)
    outdoor_temperature_input_scale: float = eqx.field(static=True)
    setpoint_metadata_index: int = eqx.field(static=True)
    setpoint_metadata_mean: float = eqx.field(static=True)
    setpoint_metadata_scale: float = eqx.field(static=True)
    target_temperature_mean: float = eqx.field(static=True)
    target_temperature_scale: float = eqx.field(static=True)

    def __init__(
        self,
        metadata_dim: int,
        input_dim: int,
        *,
        state_dim: int = 6,
        encoded_input_dim: int = 4,
        output_dim: int = 1,
        latent_dim: int = 4,
        hidden_dim: int = 64,
        depth: int = 3,
        input_encoder_hidden_dim: int | None = None,
        input_encoder_depth: int = 2,
        process_noise_mode: ProcessNoiseMode = "constant",
        process_noise_init: float = -6.0,
        schur_gamma: float = 0.995,
        pf_lambda_min: float = 0.0,
        schur_eps: float = 1e-4,
        schur_mode: SchurMode = "near_identity",
        theta_scale: float = 0.05,
        process_noise_floor: float = 1e-5,
        zero_d: bool = False,
        output_timing: OutputTiming = "pre_update",
        input_encoder_feedback: InputEncoderFeedback = "none",
        switching_dynamics: SwitchingDynamics = "none",
        switching_alpha_heat_scale: float = 1.0,
        switching_alpha_on_weight: float = 2.0,
        switching_alpha_recent_weight: float = 1.0,
        heat_on_threshold: float = 1e-6,
        heat_input_index: int = 0,
        heat_input_mean: float = 0.0,
        heat_input_scale: float = 1.0,
        heating_on_input_index: int = -1,
        heating_on_input_mean: float = 0.0,
        heating_on_input_scale: float = 1.0,
        recently_on_input_index: int = -1,
        recently_on_input_mean: float = 0.0,
        recently_on_input_scale: float = 1.0,
        outdoor_temperature_input_index: int = 1,
        outdoor_temperature_input_mean: float = 0.0,
        outdoor_temperature_input_scale: float = 1.0,
        setpoint_metadata_index: int = 8,
        setpoint_metadata_mean: float = 0.0,
        setpoint_metadata_scale: float = 1.0,
        target_temperature_mean: float = 0.0,
        target_temperature_scale: float = 1.0,
        key: jax.Array,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if encoded_input_dim < 1:
            raise ValueError("encoded_input_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if input_encoder_depth < 1:
            raise ValueError("input_encoder_depth must be at least 1")
        if process_noise_mode not in ("none", "constant", "heteroscedastic"):
            raise ValueError("process_noise_mode must be 'none', 'constant', or 'heteroscedastic'")
        if input_encoder_feedback not in ("none", "predicted_temperature", "thermal_gaps"):
            raise ValueError(
                "input_encoder_feedback must be 'none', 'predicted_temperature', or 'thermal_gaps'"
            )
        if input_encoder_feedback != "none" and not zero_d:
            raise ValueError("input_encoder_feedback requires zero_d=True to avoid direct-feedthrough loops")
        if output_timing not in ("pre_update", "post_update"):
            raise ValueError("output_timing must be 'pre_update' or 'post_update'")
        if switching_dynamics not in ("none", "heating", "heating_bias"):
            raise ValueError("switching_dynamics must be 'none', 'heating', or 'heating_bias'")
        if switching_alpha_heat_scale <= 0.0:
            raise ValueError("switching_alpha_heat_scale must be positive")
        if heat_input_scale == 0.0:
            raise ValueError("heat_input_scale must be non-zero")
        if target_temperature_scale == 0.0:
            raise ValueError("target_temperature_scale must be non-zero")

        encoder_key, theta_key, x0_key, process_key = jax.random.split(key, 4)
        encoder_input_dim = input_dim + input_encoder_feedback_extra_dim(input_encoder_feedback, output_dim)
        slices = parameter_slices(state_dim, encoded_input_dim, output_dim)
        self.input_encoder = MLP(
            encoder_input_dim,
            encoded_input_dim,
            hidden_dim=input_encoder_hidden_dim or hidden_dim,
            depth=input_encoder_depth,
            key=encoder_key,
        )
        if switching_dynamics == "none":
            theta_output_dim = slices.total
        elif switching_dynamics == "heating_bias":
            theta_output_dim = slices.total + state_dim
        else:
            theta_output_dim = 2 * slices.total
        self.theta_net = MLP(
            metadata_dim + latent_dim,
            theta_output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=theta_key,
        )
        self.x0_net = MLP(
            metadata_dim + output_dim + latent_dim,
            state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=x0_key,
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
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.encoded_input_dim = encoded_input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.process_noise_mode = process_noise_mode
        self.process_noise_init = process_noise_init
        self.schur_gamma = schur_gamma
        self.pf_lambda_min = pf_lambda_min
        self.schur_eps = schur_eps
        self.schur_mode = schur_mode
        self.theta_scale = theta_scale
        self.process_noise_floor = process_noise_floor
        self.zero_d = zero_d
        self.output_timing = output_timing
        self.input_encoder_feedback = input_encoder_feedback
        self.switching_dynamics = switching_dynamics
        self.switching_alpha_heat_scale = switching_alpha_heat_scale
        self.switching_alpha_on_weight = switching_alpha_on_weight
        self.switching_alpha_recent_weight = switching_alpha_recent_weight
        self.heat_on_threshold = heat_on_threshold
        self.heat_input_index = heat_input_index
        self.heat_input_mean = heat_input_mean
        self.heat_input_scale = heat_input_scale
        self.heating_on_input_index = heating_on_input_index
        self.heating_on_input_mean = heating_on_input_mean
        self.heating_on_input_scale = heating_on_input_scale
        self.recently_on_input_index = recently_on_input_index
        self.recently_on_input_mean = recently_on_input_mean
        self.recently_on_input_scale = recently_on_input_scale
        self.outdoor_temperature_input_index = outdoor_temperature_input_index
        self.outdoor_temperature_input_mean = outdoor_temperature_input_mean
        self.outdoor_temperature_input_scale = outdoor_temperature_input_scale
        self.setpoint_metadata_index = setpoint_metadata_index
        self.setpoint_metadata_mean = setpoint_metadata_mean
        self.setpoint_metadata_scale = setpoint_metadata_scale
        self.target_temperature_mean = target_temperature_mean
        self.target_temperature_scale = target_temperature_scale

    def encode_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.input_encoder_feedback != "none":
            raise ValueError("Feedback encoders must be evaluated inside the recurrent rollout")
        return jax.vmap(self.input_encoder)(inputs)

    def target_scaled_outdoor_temperature(self, input_t: jnp.ndarray) -> jnp.ndarray:
        outdoor_c = (
            input_t[self.outdoor_temperature_input_index] * self.outdoor_temperature_input_scale
            + self.outdoor_temperature_input_mean
        )
        outdoor_target_scaled = (outdoor_c - self.target_temperature_mean) / self.target_temperature_scale
        return jnp.broadcast_to(outdoor_target_scaled, (self.output_dim,))

    def target_scaled_setpoint(self, metadata: jnp.ndarray) -> jnp.ndarray:
        setpoint_c = (
            metadata[self.setpoint_metadata_index] * self.setpoint_metadata_scale + self.setpoint_metadata_mean
        )
        setpoint_target_scaled = (setpoint_c - self.target_temperature_mean) / self.target_temperature_scale
        return jnp.broadcast_to(setpoint_target_scaled, (self.output_dim,))

    def encode_step_input(
        self,
        metadata: jnp.ndarray,
        input_t: jnp.ndarray,
        predicted_output_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.input_encoder_feedback == "predicted_temperature":
            encoder_input = jnp.concatenate([input_t, predicted_output_t], axis=0)
        elif self.input_encoder_feedback == "thermal_gaps":
            outdoor_gap = self.target_scaled_outdoor_temperature(input_t) - predicted_output_t
            setpoint_gap = self.target_scaled_setpoint(metadata) - predicted_output_t
            encoder_input = jnp.concatenate([input_t, predicted_output_t, outdoor_gap, setpoint_gap], axis=0)
        else:
            encoder_input = input_t
        return self.input_encoder(encoder_input)

    def decode_matrices(self, theta: jnp.ndarray) -> StateSpaceMatrices:
        slices = parameter_slices(self.state_dim, self.encoded_input_dim, self.output_dim)
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
        c = theta[slices.c].reshape((self.output_dim, self.state_dim))
        if self.zero_d:
            d = jnp.zeros((self.output_dim, self.encoded_input_dim), dtype=theta.dtype)
        else:
            d = theta[slices.d].reshape((self.output_dim, self.encoded_input_dim))
        state_bias = theta[slices.state_bias]
        output_bias = theta[slices.output_bias]
        return StateSpaceMatrices(a, b, c, d, state_bias, output_bias)

    def matrices(self, metadata: jnp.ndarray, xi: jnp.ndarray | None = None) -> StateSpaceMatrices:
        if xi is None:
            xi = jnp.zeros((self.latent_dim,))
        generator_input = jnp.concatenate([metadata, xi], axis=0)
        theta = self.theta_scale * self.theta_net(generator_input)
        total = parameter_slices(self.state_dim, self.encoded_input_dim, self.output_dim).total
        return self.decode_matrices(theta[:total])

    def regime_matrices(
        self,
        metadata: jnp.ndarray,
        xi: jnp.ndarray | None = None,
    ) -> tuple[StateSpaceMatrices, StateSpaceMatrices]:
        if xi is None:
            xi = jnp.zeros((self.latent_dim,))
        generator_input = jnp.concatenate([metadata, xi], axis=0)
        theta = self.theta_scale * self.theta_net(generator_input)
        total = parameter_slices(self.state_dim, self.encoded_input_dim, self.output_dim).total
        off = self.decode_matrices(theta[:total])
        if self.switching_dynamics == "none":
            return off, off
        if self.switching_dynamics == "heating_bias":
            on_state_bias = theta[total : total + self.state_dim]
            on = StateSpaceMatrices(off.a, off.b, off.c, off.d, on_state_bias, off.output_bias)
            return off, on
        on = self.decode_matrices(theta[total : 2 * total])
        return off, on

    def switching_alpha(self, input_t: jnp.ndarray) -> jnp.ndarray:
        if self.switching_dynamics == "none":
            return jnp.asarray(0.0, dtype=input_t.dtype)
        heat = input_t[self.heat_input_index] * self.heat_input_scale + self.heat_input_mean
        score = (heat - self.heat_on_threshold) / self.switching_alpha_heat_scale
        if self.heating_on_input_index >= 0:
            heating_on = (
                input_t[self.heating_on_input_index] * self.heating_on_input_scale
                + self.heating_on_input_mean
            )
            score = score + self.switching_alpha_on_weight * (heating_on - 0.5)
        if self.recently_on_input_index >= 0:
            recently_on = (
                input_t[self.recently_on_input_index] * self.recently_on_input_scale
                + self.recently_on_input_mean
            )
            score = score + self.switching_alpha_recent_weight * (recently_on - 0.5)
        return jax.nn.sigmoid(score)

    def initial_state(
        self,
        metadata: jnp.ndarray,
        initial_output: jnp.ndarray,
        xi: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if xi is None:
            xi = jnp.zeros((self.latent_dim,))
        return self.x0_net(jnp.concatenate([metadata, initial_output, xi], axis=0))

    def process_scale(
        self,
        metadata: jnp.ndarray,
        encoded_input: jnp.ndarray,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.process_noise_mode == "none":
            return jnp.zeros((self.state_dim,))
        if self.process_noise_mode == "constant":
            raw = self.raw_process_scale
        else:
            assert self.process_noise_net is not None
            process_input = jnp.concatenate([metadata, encoded_input, state], axis=0)
            raw = self.process_noise_init + 0.1 * self.process_noise_net(process_input)
        return jax.nn.softplus(raw) + self.process_noise_floor

    def rollout_particle(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        xi: jnp.ndarray,
        process_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        x0 = self.initial_state(metadata, initial_output, xi)

        if self.switching_dynamics != "none":
            off_matrices, on_matrices = self.regime_matrices(metadata, xi)

            def mixed_dynamics(input_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                alpha_t = self.switching_alpha(input_t)
                a_t = off_matrices.a + alpha_t * (on_matrices.a - off_matrices.a)
                b_t = off_matrices.b + alpha_t * (on_matrices.b - off_matrices.b)
                state_bias_t = off_matrices.state_bias + alpha_t * (
                    on_matrices.state_bias - off_matrices.state_bias
                )
                return a_t, b_t, state_bias_t

            if self.input_encoder_feedback == "none":
                encoded_inputs = self.encode_inputs(inputs)

                def step(
                    x_t: jnp.ndarray,
                    carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
                    u_t, z_t, eps_t = carry
                    scale_t = self.process_scale(metadata, z_t, x_t)
                    a_t, b_t, state_bias_t = mixed_dynamics(u_t)
                    x_next = a_t @ x_t + b_t @ z_t + state_bias_t + scale_t * eps_t
                    if self.output_timing == "pre_update":
                        y_t = off_matrices.c @ x_t + off_matrices.d @ z_t + off_matrices.output_bias
                    elif self.output_timing == "post_update":
                        y_t = off_matrices.c @ x_next + off_matrices.d @ z_t + off_matrices.output_bias
                    else:
                        raise ValueError(f"Unknown output_timing {self.output_timing!r}")
                    return x_next, y_t

                _, outputs = jax.lax.scan(step, x0, (inputs, encoded_inputs, process_noise))
                return outputs

            def step(x_t: jnp.ndarray, carry: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
                u_t, eps_t = carry
                y_prior = off_matrices.c @ x_t + off_matrices.output_bias
                z_t = self.encode_step_input(metadata, u_t, y_prior)
                scale_t = self.process_scale(metadata, z_t, x_t)
                a_t, b_t, state_bias_t = mixed_dynamics(u_t)
                x_next = a_t @ x_t + b_t @ z_t + state_bias_t + scale_t * eps_t
                if self.output_timing == "pre_update":
                    y_t = y_prior
                elif self.output_timing == "post_update":
                    y_t = off_matrices.c @ x_next + off_matrices.output_bias
                else:
                    raise ValueError(f"Unknown output_timing {self.output_timing!r}")
                return x_next, y_t

            _, outputs = jax.lax.scan(step, x0, (inputs, process_noise))
            return outputs

        matrices = self.matrices(metadata, xi)
        if self.input_encoder_feedback == "none":
            encoded_inputs = self.encode_inputs(inputs)

            def step(
                x_t: jnp.ndarray,
                carry: tuple[jnp.ndarray, jnp.ndarray],
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                z_t, eps_t = carry
                scale_t = self.process_scale(metadata, z_t, x_t)
                x_next = matrices.a @ x_t + matrices.b @ z_t + matrices.state_bias + scale_t * eps_t
                if self.output_timing == "pre_update":
                    y_t = matrices.c @ x_t + matrices.d @ z_t + matrices.output_bias
                elif self.output_timing == "post_update":
                    y_t = matrices.c @ x_next + matrices.d @ z_t + matrices.output_bias
                else:
                    raise ValueError(f"Unknown output_timing {self.output_timing!r}")
                return x_next, y_t

            _, outputs = jax.lax.scan(step, x0, (encoded_inputs, process_noise))
            return outputs

        def step(x_t: jnp.ndarray, carry: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
            u_t, eps_t = carry
            y_prior = matrices.c @ x_t + matrices.output_bias
            z_t = self.encode_step_input(metadata, u_t, y_prior)
            scale_t = self.process_scale(metadata, z_t, x_t)
            x_next = matrices.a @ x_t + matrices.b @ z_t + matrices.state_bias + scale_t * eps_t
            if self.output_timing == "pre_update":
                y_t = y_prior
            elif self.output_timing == "post_update":
                y_t = matrices.c @ x_next + matrices.output_bias
            else:
                raise ValueError(f"Unknown output_timing {self.output_timing!r}")
            return x_next, y_t

        _, outputs = jax.lax.scan(step, x0, (inputs, process_noise))
        return outputs

    def sample(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool = True,
    ) -> jnp.ndarray:
        if num_particles < 1:
            raise ValueError("num_particles must be positive")
        xi_key, process_key = jax.random.split(key)
        xi = jax.random.normal(xi_key, (num_particles, self.latent_dim))
        if sample_process_noise and self.process_noise_mode != "none":
            process_noise = jax.random.normal(process_key, (num_particles, inputs.shape[0], self.state_dim))
        else:
            process_noise = jnp.zeros((num_particles, inputs.shape[0], self.state_dim))
        return jax.vmap(
            lambda particle_xi, particle_noise: self.rollout_particle(
                metadata,
                inputs,
                initial_output,
                particle_xi,
                particle_noise,
            )
        )(xi, process_noise)

    def parameter_regularization(self, metadata: jnp.ndarray) -> jnp.ndarray:
        matrices, on_matrices = self.regime_matrices(metadata)
        direct_gain = jnp.mean(matrices.d**2)
        bias = (
            0.5 * (jnp.mean(matrices.state_bias**2) + jnp.mean(on_matrices.state_bias**2))
            + jnp.mean(matrices.output_bias**2)
        )
        process_scale = jax.nn.softplus(self.raw_process_scale) + self.process_noise_floor
        process = jnp.mean(process_scale**2)
        return direct_gain + 0.1 * bias + process

    def __call__(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
    ) -> jnp.ndarray:
        xi = jnp.zeros((self.latent_dim,))
        process_noise = jnp.zeros((inputs.shape[0], self.state_dim))
        return self.rollout_particle(metadata, inputs, initial_output, xi, process_noise)
