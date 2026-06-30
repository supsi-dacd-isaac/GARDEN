"""State-space rollout utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class StateSpaceMatrices:
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    state_bias: jnp.ndarray
    output_bias: jnp.ndarray


def rollout_state_space(
    matrices: StateSpaceMatrices,
    x0: jnp.ndarray,
    inputs: jnp.ndarray,
) -> jnp.ndarray:
    """Roll out y[t] = C x[t] + D u[t] + c, then update x[t+1]."""

    def step(x_t: jnp.ndarray, u_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        y_t = matrices.c @ x_t + matrices.d @ u_t + matrices.output_bias
        x_next = matrices.a @ x_t + matrices.b @ u_t + matrices.state_bias
        return x_next, y_t

    _, outputs = jax.lax.scan(step, x0, inputs)
    return outputs
