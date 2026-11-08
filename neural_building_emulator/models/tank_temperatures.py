"""Two effective thermal stores, supervised by interval-mean tank temperatures.

These are lumped surrogate states, not EnergyPlus tank nodes. Volume anchors
heat capacity, with bounded learned corrections for unresolved stratification.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .emulator import MLP


class TankTemperatures(eqx.Module):
    parameter_net: MLP

    def __init__(self, metadata_dim, latent_dim, *, key):
        self.parameter_net = MLP(metadata_dim + latent_dim, 15, hidden_dim=32, depth=1, key=key)
        self.parameter_net = eqx.tree_at(
            lambda n: (n.layers[-1].weight, n.layers[-1].bias), self.parameter_net,
            (self.parameter_net.layers[-1].weight * .05,
             jnp.zeros_like(self.parameter_net.layers[-1].bias)),
        )

    def parameters(self, metadata, xi):
        return self.parameter_net(jnp.concatenate([metadata, xi]))

    def initial(self, raw, heating_available=1.):
        temperatures = jnp.asarray([40., 55.]) + 20 * jnp.tanh(raw[:2])
        ambient = 10 + 15 * jax.nn.sigmoid(raw[7])
        return temperatures.at[0].set(ambient + heating_available * (temperatures[0] - ambient))

    def capacities(self, raw, volumes):
        return jnp.maximum(1163 * volumes * jnp.exp(jnp.log(4.) * jnp.tanh(raw[2:4])), 1e-3)

    def zone_ratio(self, raw):
        return jnp.exp(jnp.log(4.) * jnp.tanh(raw[4]))

    def energy(self, temperatures, capacities, room_c):
        return capacities[0] * jnp.maximum(temperatures[0] - room_c, 0.)

    def deliver(self, raw, temperatures, capacities, room_c, requested, generation, dt):
        # Warm water permits delivery, but demand and available energy still limit it.
        gate = jax.nn.sigmoid((temperatures[0] - room_c - (2 + 8 * jax.nn.sigmoid(raw[5])))
                             / (1 + 4 * jax.nn.sigmoid(raw[6])))
        available = self.zone_ratio(raw) * (self.energy(temperatures, capacities, room_c) / dt + generation)
        return jnp.minimum(jnp.maximum(requested * gate, 0.), available), available

    def advance(self, raw, temperatures, capacities, *, room_c, generation_sh, qroom,
                hp_on, water_l_h_m2, recovery_w_m2, generator_available, dt, noise):
        ambient = 10 + 15 * jax.nn.sigmoid(raw[7])
        loss_rates = .1 * jax.nn.sigmoid(raw[8:10] - 2.)
        dhw_target = 45 + 20 * jax.nn.sigmoid(raw[10])
        dhw_gate = jax.nn.sigmoid((dhw_target - temperatures[1]) / (1 + 4 * jax.nn.sigmoid(raw[11])))
        recovery = (recovery_w_m2 * jnp.exp(jnp.log(2.) * jnp.tanh(raw[12]))
                    * generator_available * (1 - hp_on) * dhw_gate)
        withdrawal = 1.163 * jnp.maximum(water_l_h_m2, 0.) * (15 + 25 * jax.nn.sigmoid(raw[13]))
        net_power = jnp.stack([generation_sh - qroom / self.zone_ratio(raw), recovery - withdrawal])
        decay = jnp.exp(-loss_rates * dt)
        # Exact constant-forcing leaky-store transition avoids explicit-Euler instability.
        next_temperature = ambient + (temperatures - ambient) * decay + net_power / capacities * (-jnp.expm1(-loss_rates * dt)) / loss_rates
        scale = .5 * jax.nn.sigmoid(raw[14] - 4.)
        next_temperature = jnp.clip(next_temperature + scale * noise, 1., 90.)
        # Export labels are reporting-interval means; midpoint is the coarse observation model.
        interval_mean = .5 * (temperatures + next_temperature)
        return next_temperature, interval_mean
