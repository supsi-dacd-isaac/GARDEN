"""Truncated BPTT helpers for closed-loop rollouts."""

from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp

Carry = TypeVar("Carry")

DEFAULT_BPTT_TRUNCATE_STEPS = 16


def validate_bptt_truncate_steps(truncate_steps: int) -> int:
    if int(truncate_steps) != truncate_steps:
        raise ValueError("bptt_truncate_steps must be an integer")
    truncate_steps = int(truncate_steps)
    if truncate_steps < 0:
        raise ValueError("bptt_truncate_steps must be >= 0; use 0 for full BPTT")
    return truncate_steps


def detach_carry(carry: Carry, time_index: jnp.ndarray, truncate_steps: int) -> Carry:
    """Stop gradient through ``carry`` after every ``truncate_steps`` transitions.

    ``time_index`` is the 0-based scan step. After step ``t``, if
    ``(t + 1) % K == 0``, the carry into the next step is treated as a constant.
    ``truncate_steps <= 0`` disables truncation. Forward values are unchanged.
    """
    if truncate_steps <= 0:
        return carry
    detach = ((time_index + 1) % jnp.asarray(truncate_steps, dtype=time_index.dtype)) == 0
    return jax.tree.map(
        lambda value: jnp.where(detach, jax.lax.stop_gradient(value), value),
        carry,
    )
