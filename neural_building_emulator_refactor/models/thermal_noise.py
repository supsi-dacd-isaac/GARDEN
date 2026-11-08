"""Thermal-only stochastic models for an isolated noise-conditioning comparison.

The constrained diagonal noise pair shares every deterministic equation.
An optional matched positive-diagonal transition removes the upper pole bound.
Inputs, including observed Qroom and ventilation, are prescribed trajectories.
No predicted temperature, generated ventilation, or controller feeds the
structured forcing encoder. Contraction is conditional on fixed metadata/xi,
inputs and common noise, and is guaranteed only for the exogenous-noise arm.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.models.truncated_bptt import detach_carry


class ThermalNoiseModel(eqx.Module):
    parameter_net: eqx.nn.MLP | None
    encoder: eqx.nn.MLP | None
    initial_hidden: eqx.nn.MLP | None
    initial_cell: eqx.nn.MLP | None
    cell: eqx.nn.LSTMCell | None
    readout: eqx.nn.Linear | None
    noise_net: eqx.nn.MLP
    base_parameters: jax.Array
    raw_process_scale: jax.Array
    architecture: str = eqx.field(static=True)
    noise_conditioning: str = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    encoded_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    truncate_steps: int = eqx.field(static=True)
    tau_min: float = eqx.field(static=True)
    tau_max: float = eqx.field(static=True)
    dt_hours: float = eqx.field(static=True)
    transition_constraint: str = eqx.field(static=True)
    matched_pole_initialization: bool = eqx.field(static=True)

    def __init__(self, metadata_dim, input_dim, *, architecture="diagonal",
                 noise_conditioning="exogenous", state_dim=8, encoded_dim=8,
                 latent_dim=4, width=64, truncate_steps=32, tau_min=0.1,
                 tau_max=504., dt_hours=.25, transition_constraint="contractive",
                 matched_pole_initialization=False, key):
        if architecture not in ("diagonal", "lstm"):
            raise ValueError(architecture)
        if noise_conditioning not in ("exogenous", "state"):
            raise ValueError(noise_conditioning)
        if not 0 < tau_min < tau_max or truncate_steps < 0:
            raise ValueError("Invalid time constants or truncation")
        if transition_constraint not in ("contractive", "none"):
            raise ValueError(transition_constraint)
        self.architecture, self.noise_conditioning = architecture, noise_conditioning
        self.state_dim, self.encoded_dim, self.latent_dim = state_dim, encoded_dim, latent_dim
        self.truncate_steps = truncate_steps
        self.tau_min, self.tau_max, self.dt_hours = tau_min, tau_max, dt_hours
        self.transition_constraint = transition_constraint
        self.matched_pole_initialization = matched_pole_initialization
        keys = jax.random.split(key, 7)
        context_dim = metadata_dim + latent_dim
        self.parameter_net = self.encoder = None
        self.initial_hidden = self.initial_cell = self.cell = self.readout = None
        self.base_parameters = jnp.zeros(0)
        if architecture == "diagonal":
            count = state_dim * (encoded_dim + 3) + 1
            self.parameter_net = eqx.nn.MLP(context_dim, count, width, 2,
                                           activation=jax.nn.tanh, key=keys[0])
            self.encoder = eqx.nn.MLP(input_dim, encoded_dim, width, 2,
                                     activation=jax.nn.tanh, key=keys[1])
            # Initialise a fast-to-slow bank. Small forcing for the slow poles
            # avoids arbitrary large initial equilibria before learning.
            tau = np.geomspace(max(.25, tau_min * 1.01), tau_max * .8, state_dim)
            frac = (np.log(tau) - np.log(tau_min)) / np.log(tau_max / tau_min)
            poles = np.log(frac / (1-frac))
            rho = np.exp(-dt_hours / tau)
            b = jax.random.normal(keys[5], (state_dim, encoded_dim)) * jnp.asarray(1-rho)[:, None]
            c = np.full(state_dim, np.arctanh(1 / np.sqrt(state_dim + 1)))
            self.base_parameters = jnp.concatenate((jnp.asarray(poles), b.ravel(),
                                                    jnp.zeros(state_dim), jnp.asarray(c), jnp.zeros(1)))
            # Keep the initial system close to the declared pole bank.
            last = self.parameter_net.layers[-1]
            self.parameter_net = eqx.tree_at(
                lambda n: (n.layers[-1].weight, n.layers[-1].bias), self.parameter_net,
                (last.weight * .01, jnp.zeros_like(last.bias)))
            if matched_pole_initialization:
                # Exact initial pole equality for every metadata/latent draw.
                # This option is shared by all diagonal arms in the new run;
                # default False preserves older checkpoint initialization.
                self.parameter_net = eqx.tree_at(
                    lambda n: n.layers[-1].weight, self.parameter_net,
                    self.parameter_net.layers[-1].weight.at[:state_dim].set(0.))
        else:
            self.initial_hidden = eqx.nn.MLP(context_dim + 1, state_dim, width, 2,
                                            activation=jax.nn.tanh, key=keys[0])
            self.initial_cell = eqx.nn.MLP(context_dim + 1, state_dim, width, 2,
                                          activation=jax.nn.tanh, key=keys[1])
            self.cell = eqx.nn.LSTMCell(input_dim + 1 + context_dim, state_dim, key=keys[2])
            self.readout = eqx.nn.Linear(state_dim, 1, key=keys[3])
        # Same noise network shape and initial parameters in the two diagonal
        # arms; inactive state columns are identically zero in the exogenous arm.
        self.noise_net = eqx.nn.MLP(context_dim + input_dim + state_dim, state_dim,
                                   width, 1, activation=jax.nn.tanh, key=keys[4])
        self.raw_process_scale = jnp.full((state_dim,), -3.)

    def pole_values(self, raw_poles):
        """Positive poles; optional C1 continuation permits values above one.

        The relaxed map equals the bounded map below each initial raw pole,
        and its tangent above that point. Values AND first derivatives agree
        at initialization. No extra learned parameter, clipping, or instability
        penalty is introduced. Positivity is preserved, contractivity is not.
        """
        tau = jnp.exp(jnp.log(self.tau_min) + jnp.log(self.tau_max/self.tau_min)
                      * jax.nn.sigmoid(raw_poles))
        rho = jnp.exp(-self.dt_hours/tau)
        if self.transition_constraint == "none":
            tau0 = np.geomspace(max(.25, self.tau_min*1.01), self.tau_max*.8, self.state_dim)
            fraction0 = (np.log(tau0)-np.log(self.tau_min))/np.log(self.tau_max/self.tau_min)
            z0 = jnp.asarray(np.log(fraction0/(1-fraction0)), dtype=raw_poles.dtype)
            fraction = jax.nn.sigmoid(z0)
            tau_reference = jnp.exp(jnp.log(self.tau_min)+jnp.log(self.tau_max/self.tau_min)*fraction)
            rho0 = jnp.exp(-self.dt_hours/tau_reference)
            slope = rho0*(self.dt_hours/tau_reference)*jnp.log(self.tau_max/self.tau_min)*fraction*(1-fraction)
            rho = jnp.where(raw_poles > z0, rho0+slope*(raw_poles-z0), rho)
        return rho

    def coefficients(self, metadata, xi):
        raw = self.base_parameters + self.parameter_net(jnp.concatenate((metadata, xi)))
        n, k = self.state_dim, self.encoded_dim
        rho = self.pole_values(raw[:n])
        b = .30 * jnp.tanh(raw[n:n+n*k].reshape(n, k))
        bias = .15 * jnp.tanh(raw[n+n*k:n+n*k+n])
        c = jnp.tanh(raw[n+n*k+n:n+n*k+2*n])
        d = 2 * jnp.tanh(raw[-1])
        return rho, b, bias, c, d

    def noise_scale(self, metadata, xi, inputs, state):
        state_features = state if self.noise_conditioning == "state" else jnp.zeros_like(state)
        raw = self.raw_process_scale + .1 * self.noise_net(
            jnp.concatenate((metadata, xi, inputs, state_features)))
        return 1e-5 + .25 * jax.nn.sigmoid(raw)

    def initial_carry(self, metadata, xi, temperature):
        if self.architecture == "diagonal":
            _, _, _, c, d = self.coefficients(metadata, xi)
            return c * ((temperature-d) / (jnp.dot(c,c) + 1e-5))
        context = jnp.concatenate((metadata, xi, jnp.atleast_1d(temperature)))
        return (jnp.tanh(self.initial_hidden(context)),
                jnp.tanh(self.initial_cell(context)), temperature)

    def rollout(self, metadata, inputs, initial_temperature, xi, innovations,
                *, carry=None):
        if carry is None:
            carry = self.initial_carry(metadata, xi, initial_temperature)
        indices = jnp.arange(len(inputs))
        if self.architecture == "diagonal":
            rho, b, bias, c, d = self.coefficients(metadata, xi)
            # Compute exogenous features outside the recurrent scan. This is
            # also a structural safeguard against inadvertent state feedback.
            forcing = jax.vmap(lambda u: b @ jnp.tanh(self.encoder(u)) + bias)(inputs)
            def step(state, row):
                u, drive, eps, index = row
                scale = self.noise_scale(metadata, xi, u, state)
                next_state = rho * state + drive + scale * eps
                y = c @ next_state + d
                return detach_carry(next_state, index, self.truncate_steps), y
            return jax.lax.scan(step, carry, (inputs, forcing, innovations, indices))
        context = jnp.concatenate((metadata, xi))
        def step(state, row):
            hidden, cell, previous = state
            u, eps, index = row
            hidden, cell = self.cell(jnp.concatenate((u, jnp.atleast_1d(previous), context)),
                                     (hidden, cell))
            scale = self.noise_scale(metadata, xi, u, hidden)
            hidden = jnp.tanh(hidden + scale * eps)
            y = self.readout(hidden)[0]
            return detach_carry((hidden, cell, y), index, self.truncate_steps), y
        return jax.lax.scan(step, carry, (inputs, innovations, indices))

    def sample(self, metadata, inputs, initial_temperature, key, particles):
        latent_key, process_key = jax.random.split(key)
        xi = jax.random.normal(latent_key, (particles, self.latent_dim))
        eps = jax.random.normal(process_key, (particles, len(inputs), self.state_dim))
        return jax.vmap(lambda z, e: self.rollout(metadata, inputs, initial_temperature, z, e)[1])(xi, eps)


def distribution_loss(predictions, targets):
    """Fair RMS trajectory energy score + lagged variogram score.

    Shapes [batch, particle, time], [batch, time]. Off-diagonal ensemble
    distances give the unbiased (fair) Monte Carlo energy-score estimator.
    The objective is identical for all arms. No scale shrinkage penalty,
    best-particle loss, temperature teacher forcing or ex-post calibration.
    """
    count = predictions.shape[1]
    if count < 2:
        raise ValueError("At least two training particles required")
    obs = jnp.sqrt(jnp.mean((predictions-targets[:, None])**2, axis=-1) + 1e-8)
    pairs = jnp.sqrt(jnp.mean((predictions[:, :, None]-predictions[:, None, :])**2,
                             axis=-1) + 1e-8)
    mask = 1-jnp.eye(count)
    energy = jnp.mean(obs) - .5*jnp.sum(pairs*mask)/(len(targets)*count*(count-1))
    terms = []
    for lag in (1, 2, 4, 8):
        if lag < targets.shape[-1]:
            # Variogram order 0.5, matching the existing thermal objective.
            dy = ((targets[:, lag:]-targets[:, :-lag])**2 + 1e-6)**.25
            dp = ((predictions[:, :, lag:]-predictions[:, :, :-lag])**2 + 1e-6)**.25
            terms.append(jnp.mean((dy-jnp.mean(dp, axis=1))**2))
    variogram = jnp.mean(jnp.stack(terms)) if terms else jnp.asarray(0.)
    return energy + variogram, jnp.stack((energy, variogram))
