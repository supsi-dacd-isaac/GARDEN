"""Causal, metadata-anchored DHW energy debt and shared-HP priority.

This is a testable surrogate hypothesis, not a reconstruction of the
unavailable EnergyPlus tank-node controller. Its only dynamic forcing is the
known mixed-water draw and generator availability.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import MLP
from .probabilistic_closed_loop_hp import hard_bernoulli_expected_gradient, straight_through_binary_concrete
from .truncated_bptt import detach_carry


class DHWRecovery(eqx.Module):
    parameters: MLP

    def __init__(self, metadata_dim, latent_dim, *, key):
        self.parameters = MLP(metadata_dim + latent_dim, 8, hidden_dim=32, depth=1, key=key)
        self.parameters = eqx.tree_at(
            lambda net: (net.layers[-1].weight, net.layers[-1].bias), self.parameters,
            (jnp.zeros_like(self.parameters.layers[-1].weight), jnp.zeros_like(self.parameters.layers[-1].bias)),
        )

    def rollout(self, metadata, xi, water_l_h_m2, availability, uniforms, *,
                volume_m3_m2, recovery_w_m2, dt_hours, scenario_mode, temperature=.5,
                truncate_steps=0):
        raw = self.parameters(jnp.concatenate([metadata, xi]))
        unit = jax.nn.sigmoid(raw)
        draw_delta_k = 15 + 25 * unit[0]
        capacity = jnp.maximum(1163 * volume_m3_m2 * (20 + 30 * unit[1]), 1e-5)
        power = jnp.maximum(recovery_w_m2 * (.5 + unit[2]), 1e-5)
        on_threshold = capacity * (.05 + .4 * unit[3])
        off_threshold = on_threshold * (.05 + .45 * unit[4])
        softness = capacity * (.01 + .04 * unit[5])
        initial_debt = capacity * unit[6]
        standing_loss = capacity * .01 * unit[7]
        availability = availability * ((volume_m3_m2 > 0) & (recovery_w_m2 > 0))

        def step(carry, forcing):
            debt, previous_mode = carry
            index, draw, available, uniform = forcing
            loaded = jnp.minimum(capacity, debt + dt_hours * (1.163 * jnp.maximum(draw, 0) * draw_delta_k + standing_loss))
            threshold = previous_mode * off_threshold + (1 - previous_mode) * on_threshold
            logit = (loaded - threshold) / softness
            probability = jax.nn.sigmoid(logit)
            if scenario_mode == "expected":
                mode = probability
            elif scenario_mode == "straight_through":
                mode = straight_through_binary_concrete(logit, uniform, jnp.asarray(temperature))
            elif scenario_mode == "hard_expected_grad":
                mode = hard_bernoulli_expected_gradient(probability, uniform)
            else:
                mode = (uniform < probability).astype(loaded.dtype)
            active = available * mode
            recovered = jnp.minimum(loaded, dt_hours * power * active)
            next_debt = loaded - recovered
            carry = detach_carry((next_debt, active), index, truncate_steps)
            return carry, (available * probability, active, debt)

        _, outputs = jax.lax.scan(step, (initial_debt, jnp.asarray(0., metadata.dtype)),
                                 (jnp.arange(len(water_l_h_m2)), water_l_h_m2, availability, uniforms))
        return outputs
