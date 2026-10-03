"""Metadata-conditioned unstructured autoregressive LSTM baselines."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from neural_building_emulator.models.probabilistic_closed_loop_hp import (
    bounded_lognormal_expm1_mean,
)
from neural_building_emulator.ventilation import EPlusVentilationRollout


def _detach_carry(carry, time_index: jnp.ndarray, truncate_steps: int):
    if truncate_steps <= 0:
        return carry
    detach = (
        (time_index + 1)
        % jnp.asarray(truncate_steps, dtype=time_index.dtype)
    ) == 0
    return jax.tree.map(
        lambda value: jnp.where(detach, jax.lax.stop_gradient(value), value),
        carry,
    )


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
        *,
        ventilation_rollout: EPlusVentilationRollout | None = None,
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
            if ventilation_rollout is not None:
                input_t = ventilation_rollout.apply(input_t, previous_output[0], time_index)
            recurrent_input = jnp.concatenate([input_t, previous_output], axis=-1)
            hidden_next, cell_next = self.cell(recurrent_input, (hidden_t, cell_t))
            prediction = self.readout(hidden_next)
            next_carry = (hidden_next, cell_next, prediction)
            next_carry = _detach_carry(
                next_carry,
                time_index,
                self.bptt_truncate_steps,
            )
            return next_carry, prediction

        _, predictions = jax.lax.scan(
            step,
            (hidden, cell_state, initial_output),
            (inputs, indices),
        )
        return predictions


class VanillaProbabilisticAutoregressiveLSTM(eqx.Module):
    """Unstructured stochastic autoregressive LSTM with a linear readout.

    The model treats indoor temperature, room heat, and HP electric power as
    three generic continuous outputs. It has no availability gate, positivity
    transform, thermostat-shock features, switching process, or physical
    equations. Uncertainty enters through a trajectory-persistent latent and
    recurrent process noise.
    """

    initial_hidden: eqx.nn.MLP
    initial_cell: eqx.nn.MLP
    cell: eqx.nn.LSTMCell
    readout: eqx.nn.Linear
    process_noise_net: eqx.nn.MLP | None
    raw_process_scale: jnp.ndarray
    metadata_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    process_noise_mode: str = eqx.field(static=True)
    process_noise_init: float = eqx.field(static=True)
    process_noise_floor: float = eqx.field(static=True)
    process_noise_cap: float = eqx.field(static=True)
    bptt_truncate_steps: int = eqx.field(static=True)
    setpoint_input_index: int = eqx.field(static=True)
    input_scale: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        metadata_dim: int,
        input_dim: int,
        hidden_dim: int,
        metadata_hidden_dim: int,
        metadata_depth: int,
        latent_dim: int,
        process_noise_mode: str,
        process_noise_init: float,
        bptt_truncate_steps: int,
        setpoint_input_index: int,
        input_scale: tuple[float, ...],
        key: jax.Array,
        process_noise_floor: float = 1e-5,
        process_noise_cap: float = 0.25,
    ) -> None:
        if metadata_dim < 1 or input_dim < 1 or hidden_dim < 1 or latent_dim < 1:
            raise ValueError("model dimensions must be positive")
        if process_noise_mode not in ("none", "constant", "heteroscedastic"):
            raise ValueError(
                "process_noise_mode must be 'none', 'constant', or 'heteroscedastic'"
            )
        if bptt_truncate_steps < 0:
            raise ValueError("bptt_truncate_steps must be non-negative")
        if len(input_scale) != input_dim:
            raise ValueError("input_scale must match input_dim")

        keys = jax.random.split(key, 5)
        initial_input_dim = metadata_dim + latent_dim
        recurrent_input_dim = input_dim + 3 + metadata_dim + latent_dim
        self.initial_hidden = eqx.nn.MLP(
            in_size=initial_input_dim,
            out_size=hidden_dim,
            width_size=metadata_hidden_dim,
            depth=metadata_depth,
            activation=jax.nn.tanh,
            key=keys[0],
        )
        self.initial_cell = eqx.nn.MLP(
            in_size=initial_input_dim,
            out_size=hidden_dim,
            width_size=metadata_hidden_dim,
            depth=metadata_depth,
            activation=jax.nn.tanh,
            key=keys[1],
        )
        self.cell = eqx.nn.LSTMCell(
            input_size=recurrent_input_dim,
            hidden_size=hidden_dim,
            key=keys[2],
        )
        self.readout = eqx.nn.Linear(hidden_dim, 3, key=keys[3])
        if process_noise_mode == "heteroscedastic":
            self.process_noise_net = eqx.nn.MLP(
                in_size=hidden_dim + metadata_dim + latent_dim + input_dim,
                out_size=hidden_dim,
                width_size=hidden_dim,
                depth=1,
                activation=jax.nn.tanh,
                key=keys[4],
            )
        else:
            self.process_noise_net = None
        self.raw_process_scale = jnp.full((hidden_dim,), process_noise_init)
        self.metadata_dim = metadata_dim
        self.input_dim = input_dim
        self.output_dim = 3
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.process_noise_mode = process_noise_mode
        self.process_noise_init = process_noise_init
        self.process_noise_floor = process_noise_floor
        self.process_noise_cap = process_noise_cap
        self.bptt_truncate_steps = bptt_truncate_steps
        self.setpoint_input_index = setpoint_input_index
        self.input_scale = tuple(float(value) for value in input_scale)

    def process_scale(
        self,
        hidden_t: jnp.ndarray,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        input_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.process_noise_mode == "none":
            return jnp.zeros_like(hidden_t)
        raw = self.raw_process_scale
        if self.process_noise_mode == "heteroscedastic":
            assert self.process_noise_net is not None
            features = jnp.concatenate([hidden_t, metadata, xi, input_t], axis=0)
            raw = raw + 0.1 * self.process_noise_net(features)
        return self.process_noise_floor + self.process_noise_cap * jax.nn.sigmoid(raw)

    def rollout_particle(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        xi: jnp.ndarray,
        process_noise: jnp.ndarray,
        *,
        ventilation_rollout: EPlusVentilationRollout | None = None,
    ) -> jnp.ndarray:
        initial_features = jnp.concatenate([metadata, xi])
        hidden = jnp.tanh(self.initial_hidden(initial_features))
        cell_state = jnp.tanh(self.initial_cell(initial_features))
        indices = jnp.arange(inputs.shape[0], dtype=jnp.int32)

        def step(carry, item):
            hidden_t, cell_t, previous_output = carry
            input_t, eps_t, time_index = item
            if ventilation_rollout is not None:
                input_t = ventilation_rollout.apply(input_t, previous_output[0], time_index)
            recurrent_input = jnp.concatenate(
                [input_t, previous_output, metadata, xi], axis=0
            )
            hidden_next, cell_next = self.cell(
                recurrent_input, (hidden_t, cell_t)
            )
            scale_t = self.process_scale(hidden_next, metadata, xi, input_t)
            hidden_next = jnp.tanh(hidden_next + scale_t * eps_t)
            prediction_t = self.readout(hidden_next)
            next_carry = _detach_carry(
                (hidden_next, cell_next, prediction_t),
                time_index,
                self.bptt_truncate_steps,
            )
            return next_carry, prediction_t

        _, predictions = jax.lax.scan(
            step,
            (hidden, cell_state, initial_output),
            (inputs, process_noise, indices),
        )
        return predictions

    def sample_with_aux(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool,
        hp_scenario_mode: str,
        ventilation_rollout: EPlusVentilationRollout | None = None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        del hp_scenario_mode
        if num_particles < 1:
            raise ValueError("num_particles must be positive")
        xi_key, process_key = jax.random.split(key)
        xi = jax.random.normal(xi_key, (num_particles, self.latent_dim))
        process_noise = jax.random.normal(
            process_key,
            (num_particles, inputs.shape[0], self.hidden_dim),
        )
        if not sample_process_noise:
            process_noise = jnp.zeros_like(process_noise)
        predictions = jax.vmap(
            lambda row_xi, row_process: self.rollout_particle(
                metadata,
                inputs,
                initial_output,
                row_xi,
                row_process,
                ventilation_rollout=ventilation_rollout,
            )
        )(xi, process_noise)
        return predictions, ()

    def sample(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool = True,
        hp_scenario_mode: str = "bernoulli",
        ventilation_rollout: EPlusVentilationRollout | None = None,
    ) -> jnp.ndarray:
        predictions, _ = self.sample_with_aux(
            metadata,
            inputs,
            initial_output,
            key=key,
            num_particles=num_particles,
            sample_process_noise=sample_process_noise,
            hp_scenario_mode=hp_scenario_mode,
            ventilation_rollout=ventilation_rollout,
        )
        return predictions


class ProbabilisticAutoregressiveLSTM(eqx.Module):
    """Semi-structured stochastic LSTM retained for artifact compatibility.

    A persistent latent ``xi`` is sampled once per trajectory and recurrent
    process noise is sampled at every step. Qroom has a continuous positive
    readout. Pel uses an asymmetric start/stop emission with bounded active
    power, matching the probabilistic ingredients of the structured model
    without imposing its buffer, COP, thermal, or contraction equations. Use
    :class:`VanillaProbabilisticAutoregressiveLSTM` as the unstructured control.
    """

    initial_hidden: eqx.nn.MLP
    initial_cell: eqx.nn.MLP
    cell: eqx.nn.LSTMCell
    readout: eqx.nn.Linear
    process_noise_net: eqx.nn.MLP | None
    raw_process_scale: jnp.ndarray
    metadata_dim: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    process_noise_mode: str = eqx.field(static=True)
    process_noise_init: float = eqx.field(static=True)
    process_noise_floor: float = eqx.field(static=True)
    process_noise_cap: float = eqx.field(static=True)
    bptt_truncate_steps: int = eqx.field(static=True)
    setpoint_input_index: int = eqx.field(static=True)
    availability_input_index: int = eqx.field(static=True)
    dt_hours: float = eqx.field(static=True)
    setpoint_shock_timescales_hours: tuple[float, ...] = eqx.field(static=True)
    input_mean: tuple[float, ...] = eqx.field(static=True)
    input_scale: tuple[float, ...] = eqx.field(static=True)
    target_mean: tuple[float, ...] = eqx.field(static=True)
    target_scale: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        metadata_dim: int,
        input_dim: int,
        hidden_dim: int,
        metadata_hidden_dim: int,
        metadata_depth: int,
        latent_dim: int,
        process_noise_mode: str,
        process_noise_init: float,
        bptt_truncate_steps: int,
        setpoint_input_index: int,
        availability_input_index: int,
        dt_hours: float,
        setpoint_shock_timescales_hours: tuple[float, ...],
        input_mean: tuple[float, ...],
        input_scale: tuple[float, ...],
        target_mean: tuple[float, ...],
        target_scale: tuple[float, ...],
        key: jax.Array,
        process_noise_floor: float = 1e-5,
        process_noise_cap: float = 0.25,
    ) -> None:
        if metadata_dim < 1 or input_dim < 1 or hidden_dim < 1 or latent_dim < 1:
            raise ValueError("model dimensions must be positive")
        if process_noise_mode not in ("none", "constant", "heteroscedastic"):
            raise ValueError(
                "process_noise_mode must be 'none', 'constant', or 'heteroscedastic'"
            )
        if bptt_truncate_steps < 0:
            raise ValueError("bptt_truncate_steps must be non-negative")
        if len(input_mean) != input_dim or len(input_scale) != input_dim:
            raise ValueError("input statistics must match input_dim")
        if len(target_mean) != 3 or len(target_scale) != 3:
            raise ValueError("probabilistic closed-loop LSTM requires three targets")
        if any(value <= 0.0 for value in setpoint_shock_timescales_hours):
            raise ValueError("setpoint shock timescales must be positive")
        shock_dim = 2 * len(setpoint_shock_timescales_hours)
        recurrent_input_dim = input_dim + 3 + metadata_dim + latent_dim + shock_dim
        keys = jax.random.split(key, 6)
        initial_input_dim = metadata_dim + latent_dim
        self.initial_hidden = eqx.nn.MLP(
            in_size=initial_input_dim,
            out_size=hidden_dim,
            width_size=metadata_hidden_dim,
            depth=metadata_depth,
            activation=jax.nn.tanh,
            key=keys[0],
        )
        self.initial_cell = eqx.nn.MLP(
            in_size=initial_input_dim,
            out_size=hidden_dim,
            width_size=metadata_hidden_dim,
            depth=metadata_depth,
            activation=jax.nn.tanh,
            key=keys[1],
        )
        self.cell = eqx.nn.LSTMCell(
            input_size=recurrent_input_dim,
            hidden_size=hidden_dim,
            key=keys[2],
        )
        # Tin, Qroom, Pel-start, Pel-stop, active-power location and scale.
        self.readout = eqx.nn.Linear(hidden_dim, 6, key=keys[3])
        if process_noise_mode == "heteroscedastic":
            self.process_noise_net = eqx.nn.MLP(
                in_size=hidden_dim + metadata_dim + latent_dim + input_dim + shock_dim,
                out_size=hidden_dim,
                width_size=hidden_dim,
                depth=1,
                activation=jax.nn.tanh,
                key=keys[4],
            )
        else:
            self.process_noise_net = None
        self.raw_process_scale = jnp.full((hidden_dim,), process_noise_init)
        self.metadata_dim = metadata_dim
        self.input_dim = input_dim
        self.output_dim = 3
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.process_noise_mode = process_noise_mode
        self.process_noise_init = process_noise_init
        self.process_noise_floor = process_noise_floor
        self.process_noise_cap = process_noise_cap
        self.bptt_truncate_steps = bptt_truncate_steps
        self.setpoint_input_index = setpoint_input_index
        self.availability_input_index = availability_input_index
        self.dt_hours = dt_hours
        self.setpoint_shock_timescales_hours = tuple(
            float(value) for value in setpoint_shock_timescales_hours
        )
        self.input_mean = tuple(float(value) for value in input_mean)
        self.input_scale = tuple(float(value) for value in input_scale)
        self.target_mean = tuple(float(value) for value in target_mean)
        self.target_scale = tuple(float(value) for value in target_scale)

    def _physical_input(self, input_t: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            input_t[index] * jnp.asarray(self.input_scale[index], dtype=input_t.dtype)
            + jnp.asarray(self.input_mean[index], dtype=input_t.dtype)
        )

    def _availability(self, input_t: jnp.ndarray) -> jnp.ndarray:
        if self.availability_input_index < 0:
            return jnp.asarray(1.0, dtype=input_t.dtype)
        return (
            self._physical_input(input_t, self.availability_input_index) > 0.5
        ).astype(input_t.dtype)

    def _positive_cap(self, index: int, dtype: jnp.dtype) -> jnp.ndarray:
        mean = jnp.asarray(self.target_mean[index], dtype=dtype)
        scale = jnp.asarray(self.target_scale[index], dtype=dtype)
        return jnp.maximum(
            jnp.maximum(mean + 8.0 * scale, 2.0 * scale),
            jnp.asarray(1e-3, dtype=dtype),
        )

    def _normalize_target(self, value: jnp.ndarray, index: int) -> jnp.ndarray:
        return (
            value - jnp.asarray(self.target_mean[index], dtype=value.dtype)
        ) / jnp.asarray(self.target_scale[index], dtype=value.dtype)

    def setpoint_shock_features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        timescales = self.setpoint_shock_timescales_hours
        if not timescales:
            return jnp.zeros((inputs.shape[0], 0), dtype=inputs.dtype)
        setpoint = inputs[:, self.setpoint_input_index]
        delta = jnp.concatenate(
            [jnp.zeros((1,), dtype=inputs.dtype), jnp.diff(setpoint)], axis=0
        )
        positive = jnp.maximum(delta, 0.0)
        negative = jnp.maximum(-delta, 0.0)
        retention = jnp.exp(
            -jnp.asarray(self.dt_hours, dtype=inputs.dtype)
            / jnp.asarray(timescales, dtype=inputs.dtype)
        )

        def step(carry, shocks):
            positive_trace, negative_trace = carry
            positive_next = retention * positive_trace + shocks[0]
            negative_next = retention * negative_trace + shocks[1]
            next_carry = (positive_next, negative_next)
            return next_carry, jnp.concatenate(next_carry)

        initial = (jnp.zeros_like(retention), jnp.zeros_like(retention))
        _, traces = jax.lax.scan(step, initial, (positive, negative))
        return traces

    def process_scale(
        self,
        hidden_t: jnp.ndarray,
        metadata: jnp.ndarray,
        xi: jnp.ndarray,
        input_t: jnp.ndarray,
        shock_t: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.process_noise_mode == "none":
            return jnp.zeros_like(hidden_t)
        raw = self.raw_process_scale
        if self.process_noise_mode == "heteroscedastic":
            assert self.process_noise_net is not None
            features = jnp.concatenate(
                [hidden_t, metadata, xi, input_t, shock_t], axis=0
            )
            raw = raw + 0.1 * self.process_noise_net(features)
        return self.process_noise_floor + self.process_noise_cap * jax.nn.sigmoid(raw)

    def rollout_particle(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        xi: jnp.ndarray,
        process_noise: jnp.ndarray,
        mode_uniform: jnp.ndarray,
        active_power_noise: jnp.ndarray,
        *,
        hp_scenario_mode: str,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        initial_features = jnp.concatenate([metadata, xi])
        hidden = jnp.tanh(self.initial_hidden(initial_features))
        cell_state = jnp.tanh(self.initial_cell(initial_features))
        previous_mode = jnp.asarray(0.0, dtype=inputs.dtype)
        shock_features = self.setpoint_shock_features(inputs)
        indices = jnp.arange(inputs.shape[0], dtype=jnp.int32)

        def step(carry, item):
            hidden_t, cell_t, previous_output, previous_mode_t = carry
            input_t, shock_t, eps_t, uniform_t, power_eps_t, time_index = item
            recurrent_input = jnp.concatenate(
                [input_t, previous_output, metadata, xi, shock_t], axis=0
            )
            hidden_next, cell_next = self.cell(
                recurrent_input, (hidden_t, cell_t)
            )
            scale_t = self.process_scale(
                hidden_next, metadata, xi, input_t, shock_t
            )
            hidden_next = jnp.tanh(hidden_next + scale_t * eps_t)
            raw = self.readout(hidden_next)

            temperature_t = raw[0]
            availability_t = self._availability(input_t)
            qroom_cap = self._positive_cap(1, raw.dtype)
            qroom_physical = availability_t * qroom_cap * jax.nn.sigmoid(raw[1])
            qroom_t = self._normalize_target(qroom_physical, 1)

            p_start_t = jax.nn.sigmoid(raw[2])
            p_stop_t = jax.nn.sigmoid(raw[3])
            pi_unavailable_t = (
                (1.0 - previous_mode_t) * p_start_t
                + previous_mode_t * (1.0 - p_stop_t)
            )
            pi_t = availability_t * pi_unavailable_t
            pel_cap = self._positive_cap(2, raw.dtype)
            max_log_active = jnp.log1p(pel_cap)
            log_mu_t = max_log_active * jax.nn.sigmoid(raw[4])
            log_sigma_t = 0.05 + 0.45 * jax.nn.sigmoid(raw[5])
            expected_active_t = bounded_lognormal_expm1_mean(
                log_mu_t, log_sigma_t, max_log_active
            )
            if hp_scenario_mode == "expected":
                hp_mode_t = pi_t
                pel_physical = pi_t * expected_active_t
            elif hp_scenario_mode == "bernoulli":
                hp_mode_t = (uniform_t < pi_t).astype(raw.dtype)
                sampled_log = jnp.clip(
                    log_mu_t + log_sigma_t * power_eps_t,
                    0.0,
                    max_log_active,
                )
                pel_physical = hp_mode_t * jnp.expm1(sampled_log)
            else:
                raise ValueError("hp_scenario_mode must be 'expected' or 'bernoulli'")
            pel_t = self._normalize_target(pel_physical, 2)
            prediction_t = jnp.stack([temperature_t, qroom_t, pel_t])
            next_carry = _detach_carry(
                (hidden_next, cell_next, prediction_t, hp_mode_t),
                time_index,
                self.bptt_truncate_steps,
            )
            aux_t = (
                pi_t,
                log_mu_t,
                log_sigma_t,
                p_start_t,
                p_stop_t,
                expected_active_t,
            )
            return next_carry, (prediction_t, aux_t)

        _, outputs = jax.lax.scan(
            step,
            (hidden, cell_state, initial_output, previous_mode),
            (
                inputs,
                shock_features,
                process_noise,
                mode_uniform,
                active_power_noise,
                indices,
            ),
        )
        predictions, aux = outputs
        return predictions, aux

    def sample_with_aux(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool,
        hp_scenario_mode: str,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
        if num_particles < 1:
            raise ValueError("num_particles must be positive")
        xi_key, process_key, mode_key, power_key = jax.random.split(key, 4)
        xi = jax.random.normal(xi_key, (num_particles, self.latent_dim))
        process_noise = jax.random.normal(
            process_key,
            (num_particles, inputs.shape[0], self.hidden_dim),
        )
        if not sample_process_noise:
            process_noise = jnp.zeros_like(process_noise)
        mode_uniform = jax.random.uniform(
            mode_key, (num_particles, inputs.shape[0])
        )
        active_power_noise = jax.random.normal(
            power_key, (num_particles, inputs.shape[0])
        )
        return jax.vmap(
            lambda row_xi, row_process, row_uniform, row_power: self.rollout_particle(
                metadata,
                inputs,
                initial_output,
                row_xi,
                row_process,
                row_uniform,
                row_power,
                hp_scenario_mode=hp_scenario_mode,
            )
        )(xi, process_noise, mode_uniform, active_power_noise)

    def sample(
        self,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_output: jnp.ndarray,
        *,
        key: jax.Array,
        num_particles: int,
        sample_process_noise: bool = True,
        hp_scenario_mode: str = "bernoulli",
    ) -> jnp.ndarray:
        predictions, _ = self.sample_with_aux(
            metadata,
            inputs,
            initial_output,
            key=key,
            num_particles=num_particles,
            sample_process_noise=sample_process_noise,
            hp_scenario_mode=hp_scenario_mode,
        )
        return predictions
