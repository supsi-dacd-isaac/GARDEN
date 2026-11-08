"""Continuation-value helpers for particle trajectory training.

The critic in this module is training-only. It observes exchangeable particle
state summaries and estimates the remaining trajectory energy score. The
emulator architecture and its inference-time artifact are unchanged.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .models.emulator import MLP


class ParticleContinuationCritic(eqx.Module):
    """Permutation-invariant value model for a particle ensemble."""

    particle_encoder: MLP
    value_net: MLP
    particle_feature_dim: int = eqx.field(static=True)
    context_feature_dim: int = eqx.field(static=True)

    def __init__(
        self,
        particle_feature_dim: int,
        context_feature_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        if particle_feature_dim < 1 or context_feature_dim < 1:
            raise ValueError("critic feature dimensions must be positive")
        if hidden_dim < 1 or depth < 1:
            raise ValueError("critic hidden_dim and depth must be positive")
        particle_key, value_key = jax.random.split(key)
        self.particle_encoder = MLP(
            particle_feature_dim,
            hidden_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            key=particle_key,
        )
        self.value_net = MLP(
            2 * hidden_dim + context_feature_dim,
            1,
            hidden_dim=hidden_dim,
            depth=depth,
            key=value_key,
        )
        self.particle_feature_dim = particle_feature_dim
        self.context_feature_dim = context_feature_dim

    def __call__(
        self,
        particle_features: jnp.ndarray,
        context_features: jnp.ndarray,
    ) -> jnp.ndarray:
        encoded = jax.vmap(self.particle_encoder)(particle_features)
        mean = jnp.mean(encoded, axis=0)
        std = jnp.sqrt(jnp.var(encoded, axis=0) + 1e-6)
        value_features = jnp.concatenate([mean, std, context_features], axis=0)
        return self.value_net(value_features)[0]


def evaluate_critic(
    critic: ParticleContinuationCritic,
    particle_features: jnp.ndarray,
    context_features: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate ``critic`` on arrays shaped [batch, boundary, particle, feature]."""

    return jax.vmap(
        jax.vmap(
            lambda particles, context: critic(particles, context),
            in_axes=(0, 0),
        ),
        in_axes=(0, 0),
    )(particle_features, context_features)


def stop_gradient_module(module: ParticleContinuationCritic) -> ParticleContinuationCritic:
    """Freeze parameter derivatives while retaining derivatives through inputs."""

    return jax.tree.map(jax.lax.stop_gradient, module)


def energy_score_per_profile(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    output_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Return one full-trajectory energy score per profile."""

    root_weights = jnp.sqrt(output_weights)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    centered = (predictions - targets[:, jnp.newaxis, :, :]) * root_weights
    observation_distance = jnp.sqrt(jnp.sum(centered**2, axis=(2, 3)) + 1e-6)
    pairwise = (
        predictions[:, :, jnp.newaxis, :, :]
        - predictions[:, jnp.newaxis, :, :, :]
    ) * root_weights[:, :, jnp.newaxis, :, :]
    pairwise_distance = jnp.sqrt(jnp.sum(pairwise**2, axis=(3, 4)) + 1e-6)
    return jnp.mean(observation_distance, axis=1) - 0.5 * jnp.mean(
        pairwise_distance,
        axis=(1, 2),
    )


def energy_score_prefixes_per_profile(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    output_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Return the trajectory energy score for every inclusive time prefix."""

    root_weights = jnp.sqrt(output_weights)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    centered = (predictions - targets[:, jnp.newaxis, :, :]) * root_weights
    observation_squared = jnp.sum(centered**2, axis=-1)
    observation_distance = jnp.sqrt(jnp.cumsum(observation_squared, axis=2) + 1e-6)

    pairwise = (
        predictions[:, :, jnp.newaxis, :, :]
        - predictions[:, jnp.newaxis, :, :, :]
    ) * root_weights[:, :, jnp.newaxis, :, :]
    pairwise_squared = jnp.sum(pairwise**2, axis=-1)
    pairwise_distance = jnp.sqrt(jnp.cumsum(pairwise_squared, axis=3) + 1e-6)
    return jnp.mean(observation_distance, axis=1) - 0.5 * jnp.mean(
        pairwise_distance,
        axis=(1, 2),
    )


def build_continuation_features(
    *,
    predictions: jnp.ndarray,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    x_state: jnp.ndarray,
    energy: jnp.ndarray,
    temperature_state: jnp.ndarray,
    xi: jnp.ndarray,
    hp_probability: jnp.ndarray,
    hp_mode: jnp.ndarray,
    time_indices: jnp.ndarray,
    energy_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build particle and context features at selected rollout times."""

    selected_state = jnp.take(x_state, time_indices, axis=2)
    selected_energy = jnp.take(energy, time_indices, axis=2)[..., jnp.newaxis]
    selected_temperature = jnp.take(temperature_state, time_indices, axis=2)[
        ..., jnp.newaxis
    ]
    selected_probability = jnp.take(hp_probability, time_indices, axis=2)[
        ..., jnp.newaxis
    ]
    selected_mode = jnp.take(hp_mode, time_indices, axis=2)[..., jnp.newaxis]
    selected_predictions = jnp.take(predictions, time_indices, axis=2)

    boundary_count = time_indices.shape[0]
    xi_at_boundaries = jnp.broadcast_to(
        xi[:, :, jnp.newaxis, :],
        (*xi.shape[:2], boundary_count, xi.shape[-1]),
    )
    safe_energy_scale = jnp.maximum(
        jnp.asarray(energy_scale, dtype=predictions.dtype),
        jnp.asarray(1e-4, dtype=predictions.dtype),
    )
    particle_features = jnp.concatenate(
        [
            selected_state,
            selected_energy / safe_energy_scale,
            selected_temperature,
            xi_at_boundaries,
            selected_probability,
            selected_mode,
            selected_predictions,
        ],
        axis=-1,
    )
    particle_features = jnp.swapaxes(particle_features, 1, 2)

    horizon = targets.shape[1]
    suffix_sum = jnp.flip(jnp.cumsum(jnp.flip(targets, axis=1), axis=1), axis=1)
    suffix_count = jnp.arange(horizon, 0, -1, dtype=targets.dtype)[
        jnp.newaxis, :, jnp.newaxis
    ]
    suffix_mean = suffix_sum / suffix_count
    selected_inputs = jnp.take(inputs, time_indices, axis=1)
    selected_targets = jnp.take(targets, time_indices, axis=1)
    selected_suffix_mean = jnp.take(suffix_mean, time_indices, axis=1)
    metadata_at_boundaries = jnp.broadcast_to(
        metadata[:, jnp.newaxis, :],
        (metadata.shape[0], boundary_count, metadata.shape[-1]),
    )
    progress = jnp.broadcast_to(
        (time_indices.astype(targets.dtype) / jnp.maximum(horizon - 1, 1))[
            jnp.newaxis, :, jnp.newaxis
        ],
        (targets.shape[0], boundary_count, 1),
    )
    context_features = jnp.concatenate(
        [
            metadata_at_boundaries,
            selected_inputs,
            progress,
            selected_targets,
            selected_suffix_mean,
        ],
        axis=-1,
    )
    return particle_features, context_features
