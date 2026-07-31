"""Metadata-conditioned stable state-space emulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from .schur import SchurMode, simba_schur_matrix
from .state_space import OutputTiming, StateSpaceMatrices, rollout_state_space

InputEncoderFeedback = Literal["none", "predicted_temperature", "thermal_gaps"]
SwitchingDynamics = Literal["none", "heating", "heating_bias"]


def input_encoder_feedback_extra_dim(mode: InputEncoderFeedback, output_dim: int) -> int:
    if mode == "none":
        return 0
    if mode == "predicted_temperature":
        return output_dim
    if mode == "thermal_gaps":
        return 3 * output_dim
    raise ValueError(f"Unknown input_encoder_feedback mode: {mode!r}")


class MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        keys = jax.random.split(key, depth)
        dims = [in_dim, *([hidden_dim] * (depth - 1)), out_dim]
        self.layers = tuple(
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(depth)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)


@dataclass(frozen=True)
class ParameterSlices:
    raw_w: slice
    raw_v: slice
    b: slice
    c: slice
    d: slice
    state_bias: slice
    output_bias: slice
    total: int


def parameter_slices(state_dim: int, input_dim: int, output_dim: int) -> ParameterSlices:
    cursor = 0

    def take(size: int) -> slice:
        nonlocal cursor
        result = slice(cursor, cursor + size)
        cursor += size
        return result

    raw_w = take((2 * state_dim) ** 2)
    raw_v = take(state_dim**2)
    b = take(state_dim * input_dim)
    c = take(output_dim * state_dim)
    d = take(output_dim * input_dim)
    state_bias = take(state_dim)
    output_bias = take(output_dim)
    return ParameterSlices(raw_w, raw_v, b, c, d, state_bias, output_bias, cursor)


class MetadataStateSpaceEmulator(eqx.Module):
    theta_net: MLP
    x0_net: MLP
    input_encoder: MLP | None
    slices: ParameterSlices = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    encoded_input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    schur_gamma: float = eqx.field(static=True)
    pf_lambda_min: float = eqx.field(static=True)
    schur_eps: float = eqx.field(static=True)
    schur_mode: SchurMode = eqx.field(static=True)
    theta_scale: float = eqx.field(static=True)
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
        output_dim: int = 1,
        hidden_dim: int = 64,
        depth: int = 3,
        input_encoder_dim: int | None = None,
        input_encoder_hidden_dim: int | None = None,
        input_encoder_depth: int = 2,
        schur_gamma: float = 0.995,
        pf_lambda_min: float = 0.0,
        schur_eps: float = 1e-4,
        schur_mode: SchurMode = "near_identity",
        theta_scale: float = 0.05,
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
        if input_encoder_dim is not None and input_encoder_dim < 1:
            raise ValueError("input_encoder_dim must be positive or None")
        if input_encoder_depth < 1:
            raise ValueError("input_encoder_depth must be at least 1")
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

        theta_key, x0_key, encoder_key = jax.random.split(key, 3)
        encoder_input_dim = input_dim + input_encoder_feedback_extra_dim(input_encoder_feedback, output_dim)
        encoded_input_dim = encoder_input_dim if input_encoder_dim is None else input_encoder_dim
        slices = parameter_slices(state_dim, encoded_input_dim, output_dim)
        if switching_dynamics == "none":
            theta_output_dim = slices.total
        elif switching_dynamics == "heating_bias":
            theta_output_dim = slices.total + state_dim
        else:
            theta_output_dim = 2 * slices.total
        self.theta_net = MLP(
            metadata_dim,
            theta_output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=theta_key,
        )
        self.x0_net = MLP(
            metadata_dim + output_dim,
            state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=x0_key,
        )
        self.input_encoder = None
        if input_encoder_dim is not None:
            self.input_encoder = MLP(
                encoder_input_dim,
                input_encoder_dim,
                hidden_dim=input_encoder_hidden_dim or hidden_dim,
                depth=input_encoder_depth,
                key=encoder_key,
            )
        self.slices = slices
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.encoded_input_dim = encoded_input_dim
        self.output_dim = output_dim
        self.schur_gamma = schur_gamma
        self.pf_lambda_min = pf_lambda_min
        self.schur_eps = schur_eps
        self.schur_mode = schur_mode
        self.theta_scale = theta_scale
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
        c = theta[slices.c].reshape((self.output_dim, self.state_dim))
        if self.zero_d:
            d = jnp.zeros((self.output_dim, self.encoded_input_dim), dtype=theta.dtype)
        else:
            d = theta[slices.d].reshape((self.output_dim, self.encoded_input_dim))
        state_bias = theta[slices.state_bias]
        output_bias = theta[slices.output_bias]
        return StateSpaceMatrices(a, b, c, d, state_bias, output_bias)

    def matrices(self, metadata: jnp.ndarray) -> StateSpaceMatrices:
        theta = self.theta_scale * self.theta_net(metadata)
        return self.decode_matrices(theta[: self.slices.total])

    def regime_matrices(self, metadata: jnp.ndarray) -> tuple[StateSpaceMatrices, StateSpaceMatrices]:
        theta = self.theta_scale * self.theta_net(metadata)
        off = self.decode_matrices(theta[: self.slices.total])
        if self.switching_dynamics == "none":
            return off, off
        if self.switching_dynamics == "heating_bias":
            on_state_bias = theta[self.slices.total : self.slices.total + self.state_dim]
            on = StateSpaceMatrices(off.a, off.b, off.c, off.d, on_state_bias, off.output_bias)
            return off, on
        on = self.decode_matrices(theta[self.slices.total : 2 * self.slices.total])
        return off, on

    def initial_state(self, metadata: jnp.ndarray, initial_output: jnp.ndarray) -> jnp.ndarray:
        return self.x0_net(jnp.concatenate([metadata, initial_output], axis=0))

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

    def encode_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.input_encoder_feedback != "none":
            raise ValueError("Feedback encoders must be evaluated inside the recurrent rollout")
        if self.input_encoder is None:
            return inputs
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
        if self.input_encoder is None:
            return encoder_input
        return self.input_encoder(encoder_input)

    def rollout_with_feedback(
        self,
        matrices: StateSpaceMatrices,
        x0: jnp.ndarray,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        def step(x_t: jnp.ndarray, u_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            y_prior = matrices.c @ x_t + matrices.output_bias
            z_t = self.encode_step_input(metadata, u_t, y_prior)
            x_next = matrices.a @ x_t + matrices.b @ z_t + matrices.state_bias
            if self.output_timing == "pre_update":
                y_t = y_prior
            elif self.output_timing == "post_update":
                y_t = matrices.c @ x_next + matrices.output_bias
            else:
                raise ValueError(f"Unknown output_timing {self.output_timing!r}")
            return x_next, y_t

        _, outputs = jax.lax.scan(step, x0, inputs)
        return outputs

    def rollout_switched(
        self,
        off_matrices: StateSpaceMatrices,
        on_matrices: StateSpaceMatrices,
        x0: jnp.ndarray,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        def mixed_dynamics(input_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            alpha_t = self.switching_alpha(input_t)
            a_t = off_matrices.a + alpha_t * (on_matrices.a - off_matrices.a)
            b_t = off_matrices.b + alpha_t * (on_matrices.b - off_matrices.b)
            state_bias_t = off_matrices.state_bias + alpha_t * (
                on_matrices.state_bias - off_matrices.state_bias
            )
            return a_t, b_t, state_bias_t

        if self.input_encoder_feedback == "none":

            def step(x_t: jnp.ndarray, input_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                z_t = input_t if self.input_encoder is None else self.input_encoder(input_t)
                a_t, b_t, state_bias_t = mixed_dynamics(input_t)
                x_next = a_t @ x_t + b_t @ z_t + state_bias_t
                if self.output_timing == "pre_update":
                    y_t = off_matrices.c @ x_t + off_matrices.d @ z_t + off_matrices.output_bias
                elif self.output_timing == "post_update":
                    y_t = off_matrices.c @ x_next + off_matrices.d @ z_t + off_matrices.output_bias
                else:
                    raise ValueError(f"Unknown output_timing {self.output_timing!r}")
                return x_next, y_t

            _, outputs = jax.lax.scan(step, x0, inputs)
            return outputs

        def step(x_t: jnp.ndarray, input_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            y_prior = off_matrices.c @ x_t + off_matrices.output_bias
            z_t = self.encode_step_input(metadata, input_t, y_prior)
            a_t, b_t, state_bias_t = mixed_dynamics(input_t)
            x_next = a_t @ x_t + b_t @ z_t + state_bias_t
            if self.output_timing == "pre_update":
                y_t = y_prior
            elif self.output_timing == "post_update":
                y_t = off_matrices.c @ x_next + off_matrices.output_bias
            else:
                raise ValueError(f"Unknown output_timing {self.output_timing!r}")
            return x_next, y_t

        _, outputs = jax.lax.scan(step, x0, inputs)
        return outputs

    def __call__(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
    ) -> jnp.ndarray:
        x0 = self.initial_state(metadata, initial_output)
        if self.switching_dynamics != "none":
            off_matrices, on_matrices = self.regime_matrices(metadata)
            return self.rollout_switched(off_matrices, on_matrices, x0, metadata, inputs)
        matrices = self.matrices(metadata)
        if self.input_encoder_feedback != "none":
            return self.rollout_with_feedback(matrices, x0, metadata, inputs)
        encoded_inputs = self.encode_inputs(inputs)
        return rollout_state_space(matrices, x0, encoded_inputs, output_timing=self.output_timing)
