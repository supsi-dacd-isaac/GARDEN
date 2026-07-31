"""State-space rollout utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

OutputTiming = Literal["pre_update", "post_update"]


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
    *,
    output_timing: OutputTiming = "pre_update",
) -> jnp.ndarray:
    """Roll out a state-space system with configurable output timing."""

    def step(x_t: jnp.ndarray, u_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_next = matrices.a @ x_t + matrices.b @ u_t + matrices.state_bias
        if output_timing == "pre_update":
            y_t = matrices.c @ x_t + matrices.d @ u_t + matrices.output_bias
        elif output_timing == "post_update":
            y_t = matrices.c @ x_next + matrices.d @ u_t + matrices.output_bias
        else:
            raise ValueError(f"Unknown output_timing {output_timing!r}")
        return x_next, y_t

    _, outputs = jax.lax.scan(step, x0, inputs)
    return outputs
