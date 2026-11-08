"""Training-only plant targets; no simulator state enters the rollout."""

from __future__ import annotations

import jax.numpy as jnp


def plant_block_crps(predictions, targets, *, horizon_steps, power_scale):
    """Per-channel CRPS of 4/24-hour mean powers (equivalently scaled energies).

    Inputs are [batch, particle, time, channel] and [batch, time, channel].
    The sqrt(window length) factor matches the existing path score's scale.
    Complete nonoverlapping blocks only; every requested horizon must fit.
    """
    if predictions.ndim != 4 or targets.shape != (predictions.shape[0], *predictions.shape[2:]):
        raise ValueError("Plant predictions and targets have incompatible dimensions")
    if not horizon_steps or any(step < 1 or step > predictions.shape[2] for step in horizon_steps):
        raise ValueError("Every plant block horizon must fit the training window")
    batch, particles, length, channels = predictions.shape
    coefficient = (2 * jnp.arange(particles) - particles + 1)[None, :, None, None]
    values = []
    for step in horizon_steps:
        count = length // step
        forecast = predictions[:, :, :count * step].reshape(batch, particles, count, step, channels).mean(axis=3)
        observed = targets[:, :count * step].reshape(batch, count, step, channels).mean(axis=2)
        first = jnp.mean(jnp.abs(forecast - observed[:, None]), axis=1)
        second = jnp.sum(coefficient * jnp.sort(forecast, axis=1), axis=1) / particles ** 2
        values.append(jnp.mean(first - second, axis=(0, 1)) / power_scale)
    return jnp.mean(jnp.stack(values), axis=0) * jnp.sqrt(jnp.asarray(length, predictions.dtype))
