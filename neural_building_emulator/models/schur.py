"""SIMBa-style Schur parametrizations for stable state matrices."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

SchurMode = Literal["dense", "near_identity"]


def _right_matmul_inverse(left: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    """Return left @ inv(matrix) using a linear solve."""
    return jnp.linalg.solve(matrix.T, left.T).T


def simba_schur_matrix(
    raw_w: jnp.ndarray,
    raw_v: jnp.ndarray,
    state_dim: int,
    *,
    gamma: float = 0.995,
    eps: float = 1e-4,
    mode: SchurMode = "near_identity",
) -> jnp.ndarray:
    """Map unconstrained parameters to a Schur-stable matrix.

    The dense form follows the SIMBa free parametrization. The near-identity
    variant is useful for 15-minute thermal dynamics, where the physical state
    typically decays slowly from one step to the next.
    """
    n = state_dim
    w = raw_w.reshape((2 * n, 2 * n))
    v = raw_v.reshape((n, n))
    skew_v = v - v.T
    s = w.T @ w + eps * jnp.eye(2 * n)
    s11 = s[:n, :n]
    s12 = s[:n, n:]
    s21 = s[n:, :n]
    s22 = s[n:, n:]

    if mode == "dense":
        middle = 0.5 * (s11 / (gamma**2) + s22) + skew_v
        return _right_matmul_inverse(s12, middle)
    if mode == "near_identity":
        middle = s11 + skew_v
        s22_inv_s21 = jnp.linalg.solve(s22, s21)
        correction = jnp.linalg.solve(middle, s12 @ s22_inv_s21)
        return jnp.eye(n) - 2.0 * correction
    raise ValueError(f"Unknown Schur mode {mode!r}")


def spectral_radius(matrix: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(jnp.linalg.eigvals(matrix)))
