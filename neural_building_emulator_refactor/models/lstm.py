"""Metadata-conditioned unstructured autoregressive LSTM baselines."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class AutoregressiveLSTM(eqx.Module):
    """Autonomous recurrent regressor shared by both comparison tasks.

    The previous predicted output is fed back at the next step. The Q-to-T
    model feeds back temperature; the closed-loop model feeds back predicted
    temperature, room heat, and HP electric power. No positivity, energy, or
    stability constraints are imposed.
    """

    initial_hidden: eqx.nn.MLP
    initial_cell: eqx.nn.MLP
    cell: eqx.nn.LSTMCell
    readout: eqx.nn.Linear
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    bptt_truncate_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        metadata_dim: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        metadata_hidden_dim: int,
        metadata_depth: int,
        bptt_truncate_steps: int,
        key: jax.Array,
    ) -> None:
        if metadata_dim < 1 or input_dim < 1 or output_dim < 1 or hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        if bptt_truncate_steps < 0:
            raise ValueError("bptt_truncate_steps must be non-negative")
        keys = jax.random.split(key, 4)
        self.initial_hidden = eqx.nn.MLP(
            in_size=metadata_dim,
            out_size=hidden_dim,
            width_size=metadata_hidden_dim,
            depth=metadata_depth,
            activation=jax.nn.tanh,
            key=keys[0],
        )
        self.initial_cell = eqx.nn.MLP(
            in_size=metadata_dim,
            out_size=hidden_dim,
            width_size=metadata_hidden_dim,
            depth=metadata_depth,
            activation=jax.nn.tanh,
            key=keys[1],
        )
        self.cell = eqx.nn.LSTMCell(
            input_size=input_dim + output_dim,
            hidden_size=hidden_dim,
            key=keys[2],
        )
        self.readout = eqx.nn.Linear(hidden_dim, output_dim, key=keys[3])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate_steps = bptt_truncate_steps

    def __call__(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
    ) -> jnp.ndarray:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {inputs.shape[-1]}")
        if initial_output.shape[-1] != self.output_dim:
            raise ValueError(
                f"expected initial_output dimension {self.output_dim}, got {initial_output.shape[-1]}"
            )
        hidden = jnp.tanh(self.initial_hidden(metadata))
        cell_state = jnp.tanh(self.initial_cell(metadata))
        indices = jnp.arange(inputs.shape[0], dtype=jnp.int32)

        def step(carry, item):
            hidden_t, cell_t, previous_output = carry
            input_t, time_index = item
            recurrent_input = jnp.concatenate([input_t, previous_output], axis=-1)
            hidden_next, cell_next = self.cell(recurrent_input, (hidden_t, cell_t))
            prediction = self.readout(hidden_next)
            next_carry = (hidden_next, cell_next, prediction)
            if self.bptt_truncate_steps > 0:
                detach = (
                    (time_index + 1) % jnp.asarray(self.bptt_truncate_steps, dtype=time_index.dtype)
                ) == 0
                next_carry = jax.tree.map(
                    lambda value: jnp.where(detach, jax.lax.stop_gradient(value), value),
                    next_carry,
                )
            return next_carry, prediction

        _, predictions = jax.lax.scan(
            step,
            (hidden, cell_state, initial_output),
            (inputs, indices),
        )
        return predictions
