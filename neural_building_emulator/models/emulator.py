"""Metadata-conditioned stable state-space emulator."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .schur import SchurMode, simba_schur_matrix
from .state_space import StateSpaceMatrices, rollout_state_space


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
    schur_eps: float = eqx.field(static=True)
    schur_mode: SchurMode = eqx.field(static=True)
    theta_scale: float = eqx.field(static=True)

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
        schur_eps: float = 1e-4,
        schur_mode: SchurMode = "near_identity",
        theta_scale: float = 0.05,
        key: jax.Array,
    ) -> None:
        if input_encoder_dim is not None and input_encoder_dim < 1:
            raise ValueError("input_encoder_dim must be positive or None")
        if input_encoder_depth < 1:
            raise ValueError("input_encoder_depth must be at least 1")

        theta_key, x0_key, encoder_key = jax.random.split(key, 3)
        encoded_input_dim = input_dim if input_encoder_dim is None else input_encoder_dim
        slices = parameter_slices(state_dim, encoded_input_dim, output_dim)
        self.theta_net = MLP(
            metadata_dim,
            slices.total,
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
                input_dim,
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
        self.schur_eps = schur_eps
        self.schur_mode = schur_mode
        self.theta_scale = theta_scale

    def matrices(self, metadata: jnp.ndarray) -> StateSpaceMatrices:
        theta = self.theta_scale * self.theta_net(metadata)
        slices = self.slices
        a = simba_schur_matrix(
            theta[slices.raw_w],
            theta[slices.raw_v],
            self.state_dim,
            gamma=self.schur_gamma,
            eps=self.schur_eps,
            mode=self.schur_mode,
        )
        b = theta[slices.b].reshape((self.state_dim, self.encoded_input_dim))
        c = theta[slices.c].reshape((self.output_dim, self.state_dim))
        d = theta[slices.d].reshape((self.output_dim, self.encoded_input_dim))
        state_bias = theta[slices.state_bias]
        output_bias = theta[slices.output_bias]
        return StateSpaceMatrices(a, b, c, d, state_bias, output_bias)

    def initial_state(self, metadata: jnp.ndarray, initial_output: jnp.ndarray) -> jnp.ndarray:
        return self.x0_net(jnp.concatenate([metadata, initial_output], axis=0))

    def encode_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.input_encoder is None:
            return inputs
        return jax.vmap(self.input_encoder)(inputs)

    def __call__(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
    ) -> jnp.ndarray:
        matrices = self.matrices(metadata)
        x0 = self.initial_state(metadata, initial_output)
        encoded_inputs = self.encode_inputs(inputs)
        return rollout_state_space(matrices, x0, encoded_inputs)
