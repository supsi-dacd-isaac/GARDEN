"""Building-local causal HP, hydronic, and passive-RC emulator.

The model is intentionally independent of the metadata-conditioned global
emulators.  It is used to test whether a more structured model class can fit a
single building before paying the cost of learning a global parameter
generator.

The important structural invariant is that the thermostat setpoint is visible
to the HP controller but not to either the hydronic or RC transition.  It can
therefore affect indoor temperature only through predicted electric power and
delivered room heat.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp


HydronicMode = Literal["convolution", "instant"]


class CausalHybridRollout(eqx.Module):
    """Outputs and likelihood parameters from one autonomous rollout."""

    prediction: jnp.ndarray
    mode_probability: jnp.ndarray
    transition_probability_observed: jnp.ndarray
    duration_logits: jnp.ndarray
    active_energy_log_mean: jnp.ndarray
    active_energy_log_scale: jnp.ndarray
    event_power_mean: jnp.ndarray
    event_power_scale: jnp.ndarray
    tau_power_mean: jnp.ndarray
    tau_power_scale: jnp.ndarray
    tau_temperature_mean: jnp.ndarray
    tau_temperature_scale: jnp.ndarray
    controller_state: jnp.ndarray
    hydronic_energy: jnp.ndarray
    rc_temperature: jnp.ndarray


def _positive(raw: jnp.ndarray, floor: float = 0.0) -> jnp.ndarray:
    return jax.nn.softplus(raw) + jnp.asarray(floor, dtype=raw.dtype)


class CausalHybridHPEmulator(eqx.Module):
    """Semi-Markov HP + positive hydronics + passive RC network.

    The HP mode is represented by a distribution over ``(mode, dwell age)``.
    Transition hazards depend on dwell age, which makes the exact observed-path
    likelihood semi-Markov rather than IID Bernoulli.  Electrical energy is an
    interval-level log-normal emission; its phase inside the interval is not a
    pointwise training target.
    """

    history_net: eqx.nn.MLP
    hazard_net: eqx.nn.MLP
    interval_net: eqx.nn.MLP
    controller_net: eqx.nn.MLP
    event_net: eqx.nn.MLP

    raw_cop_intercept: jax.Array
    raw_cop_slope: jax.Array
    raw_hydronic_retention: jax.Array
    raw_hydronic_allocation: jax.Array
    raw_hydronic_delivery: jax.Array
    raw_rc_capacity: jax.Array
    raw_rc_conductance: jax.Array
    raw_rc_outdoor_conductance: jax.Array
    raw_ventilation_conductance: jax.Array
    raw_solar_gain: jax.Array

    input_mean: jax.Array
    input_scale: jax.Array
    temperature_mean: jax.Array
    temperature_scale: jax.Array

    input_dim: int = eqx.field(static=True)
    history_dim: int = eqx.field(static=True)
    controller_dim: int = eqx.field(static=True)
    rc_nodes: int = eqx.field(static=True)
    hydronic_modes: int = eqx.field(static=True)
    max_dwell_steps: int = eqx.field(static=True)
    control_interval_steps: int = eqx.field(static=True)
    horizon_steps: tuple[int, ...] = eqx.field(static=True)
    dt_hours: float = eqx.field(static=True)
    energy_cap_wh_m2: float = eqx.field(static=True)
    event_power_cap_w_m2: float = eqx.field(static=True)
    hydronic_mode: HydronicMode = eqx.field(static=True)
    use_history_encoder: bool = eqx.field(static=True)
    use_dwell_age: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_mean: tuple[float, ...],
        input_scale: tuple[float, ...],
        temperature_mean: float,
        temperature_scale: float,
        history_dim: int,
        controller_dim: int = 8,
        rc_nodes: int = 3,
        hydronic_modes: int = 3,
        max_dwell_steps: int = 96,
        control_interval_steps: int = 12,
        horizon_steps: tuple[int, ...] = (2, 4, 8, 12),
        dt_hours: float = 0.25,
        hidden_dim: int = 48,
        depth: int = 2,
        energy_cap_wh_m2: float = 200.0,
        event_power_cap_w_m2: float = 100.0,
        hydronic_mode: HydronicMode = "convolution",
        use_history_encoder: bool = True,
        use_dwell_age: bool = True,
        key: jax.Array,
    ):
        if rc_nodes < 1 or hydronic_modes < 1:
            raise ValueError("rc_nodes and hydronic_modes must be positive")
        if max_dwell_steps < 1 or control_interval_steps < 1:
            raise ValueError("duration limits must be positive")
        if any(step < 1 or step > control_interval_steps for step in horizon_steps):
            raise ValueError("event horizons must lie inside the control interval")
        keys = jax.random.split(key, 10)
        self.input_dim = len(input_mean)
        self.history_dim = history_dim
        self.controller_dim = controller_dim
        self.rc_nodes = rc_nodes
        self.hydronic_modes = hydronic_modes
        self.max_dwell_steps = max_dwell_steps
        self.control_interval_steps = control_interval_steps
        self.horizon_steps = tuple(int(value) for value in horizon_steps)
        self.dt_hours = float(dt_hours)
        self.energy_cap_wh_m2 = float(energy_cap_wh_m2)
        self.event_power_cap_w_m2 = float(event_power_cap_w_m2)
        self.hydronic_mode = hydronic_mode
        self.use_history_encoder = use_history_encoder
        self.use_dwell_age = use_dwell_age

        history_outputs = controller_dim + hydronic_modes + max(rc_nodes - 1, 0)
        self.history_net = eqx.nn.MLP(
            in_size=history_dim,
            out_size=history_outputs,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=keys[0],
        )
        dynamic_dim = self.input_dim + controller_dim + 6
        self.hazard_net = eqx.nn.MLP(
            in_size=dynamic_dim,
            out_size=6,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=keys[1],
        )
        # Duration logits (including zero duration), active-energy log mean/scale.
        self.interval_net = eqx.nn.MLP(
            in_size=dynamic_dim,
            out_size=control_interval_steps + 3,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=keys[2],
        )
        self.controller_net = eqx.nn.MLP(
            in_size=dynamic_dim + 3,
            out_size=controller_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=keys[3],
        )
        # For every H: post-event P, tau_P, tau_T; each has mean and scale.
        self.event_net = eqx.nn.MLP(
            in_size=dynamic_dim,
            out_size=6 * len(horizon_steps),
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=keys[4],
        )

        # Initial COP is close to 3 at mean outdoor temperature.
        self.raw_cop_intercept = jnp.asarray(-0.9163)
        self.raw_cop_slope = jnp.asarray(0.0)
        retention_initial = jnp.linspace(1.0, 4.0, hydronic_modes)
        self.raw_hydronic_retention = retention_initial
        self.raw_hydronic_allocation = jnp.zeros((hydronic_modes + 1,))
        self.raw_hydronic_delivery = jnp.zeros((hydronic_modes,))
        # C is in Wh m^-2 K^-1; G is in W m^-2 K^-1.
        self.raw_rc_capacity = jnp.linspace(20.0, 200.0, rc_nodes)
        self.raw_rc_conductance = jnp.full((max(rc_nodes - 1, 0),), 0.0)
        self.raw_rc_outdoor_conductance = jnp.full((rc_nodes,), -1.0)
        self.raw_ventilation_conductance = jnp.asarray(-2.0)
        self.raw_solar_gain = jnp.asarray(-4.5)
        self.input_mean = jnp.asarray(input_mean, dtype=jnp.float32)
        self.input_scale = jnp.asarray(input_scale, dtype=jnp.float32)
        self.temperature_mean = jnp.asarray(temperature_mean, dtype=jnp.float32)
        self.temperature_scale = jnp.asarray(temperature_scale, dtype=jnp.float32)

    def _normalise_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return (inputs - self.input_mean) / self.input_scale

    def _initial_states(
        self,
        history_summary: jnp.ndarray,
        initial_temperature: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if self.use_history_encoder:
            encoded = self.history_net(history_summary)
        else:
            encoded = jnp.zeros(
                (self.controller_dim + self.hydronic_modes + max(self.rc_nodes - 1, 0),),
                dtype=initial_temperature.dtype,
            )
        controller = jnp.tanh(encoded[: self.controller_dim])
        energy_start = self.controller_dim
        energy_end = energy_start + self.hydronic_modes
        energy = self.energy_cap_wh_m2 * jax.nn.sigmoid(
            encoded[energy_start:energy_end] - 5.0
        )
        if not self.use_history_encoder:
            energy = jnp.zeros_like(energy)
        offsets = 8.0 * jnp.tanh(encoded[energy_end:])
        rc_temperature = jnp.concatenate(
            [jnp.atleast_1d(initial_temperature), initial_temperature + offsets]
        )
        return controller, energy, rc_temperature

    def cop(self, outdoor_temperature: jnp.ndarray) -> jnp.ndarray:
        outdoor_scaled = (outdoor_temperature - self.input_mean[1]) / self.input_scale[1]
        raw = self.raw_cop_intercept + self.raw_cop_slope * outdoor_scaled
        return 1.0 + 7.0 * jax.nn.sigmoid(raw)

    def _dynamic_features(
        self,
        normalised_input: jnp.ndarray,
        controller: jnp.ndarray,
        indoor_temperature: jnp.ndarray,
        mode_probability: jnp.ndarray,
        expected_age: jnp.ndarray,
        previous_power: jnp.ndarray,
        setpoint_change: jnp.ndarray,
    ) -> jnp.ndarray:
        temperature_scaled = (
            indoor_temperature - self.temperature_mean
        ) / self.temperature_scale
        setpoint_gap = (
            normalised_input[0] * self.input_scale[0]
            + self.input_mean[0]
            - indoor_temperature
        ) / self.temperature_scale
        return jnp.concatenate(
            [
                normalised_input,
                controller,
                jnp.asarray(
                    [
                        temperature_scaled,
                        setpoint_gap,
                        mode_probability,
                        expected_age,
                        previous_power / self.event_power_cap_w_m2,
                        setpoint_change / self.temperature_scale,
                    ],
                    dtype=normalised_input.dtype,
                ),
            ]
        )

    def _hazards(
        self,
        dynamic_features: jnp.ndarray,
        availability: jnp.ndarray,
    ) -> jnp.ndarray:
        ages = jnp.arange(self.max_dwell_steps + 1, dtype=dynamic_features.dtype)
        age_feature = ages / float(self.max_dwell_steps)
        if not self.use_dwell_age:
            age_feature = jnp.zeros_like(age_feature)
        coefficients = self.hazard_net(dynamic_features).reshape(2, 3)
        logits = (
            coefficients[:, 0, None]
            + coefficients[:, 1, None] * age_feature[None, :]
            + coefficients[:, 2, None] * age_feature[None, :] ** 2
        )
        hazard = jax.nn.sigmoid(logits)
        # During the seasonal lockout, off->on is impossible and on->off is immediate.
        hazard = hazard.at[0].set(jnp.where(availability > 0.5, hazard[0], 0.0))
        hazard = hazard.at[1].set(jnp.where(availability > 0.5, hazard[1], 1.0))
        return jnp.clip(hazard, 1e-6, 1.0 - 1e-6)

    def _advance_mode_distribution(
        self,
        distribution: jnp.ndarray,
        hazard: jnp.ndarray,
    ) -> jnp.ndarray:
        next_distribution = jnp.zeros_like(distribution)
        ages = jnp.arange(self.max_dwell_steps + 1)
        next_ages = jnp.minimum(ages + 1, self.max_dwell_steps)
        for mode in (0, 1):
            stay = distribution[mode] * (1.0 - hazard[mode])
            switch = jnp.sum(distribution[mode] * hazard[mode])
            next_distribution = next_distribution.at[mode, next_ages].add(stay)
            next_distribution = next_distribution.at[1 - mode, 0].add(switch)
        return next_distribution / jnp.maximum(jnp.sum(next_distribution), 1e-8)

    def _interval_parameters(
        self,
        dynamic_features: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raw = self.interval_net(dynamic_features)
        duration_logits = raw[: self.control_interval_steps + 1]
        maximum_log_energy = jnp.log1p(self.energy_cap_wh_m2)
        log_mean = maximum_log_energy * jax.nn.sigmoid(raw[-2])
        log_scale = 0.05 + 1.45 * jax.nn.sigmoid(raw[-1])
        probabilities = jax.nn.softmax(duration_logits)
        durations = jnp.arange(self.control_interval_steps + 1, dtype=raw.dtype)
        expected_duration = jnp.sum(probabilities * durations)
        active_probability = 1.0 - probabilities[0]
        active_energy = jnp.minimum(
            jnp.expm1(log_mean + 0.5 * log_scale**2),
            self.energy_cap_wh_m2,
        )
        expected_energy = active_probability * active_energy
        active_power = expected_energy / (
            self.dt_hours * jnp.maximum(expected_duration, 0.25)
        )
        return duration_logits, log_mean, log_scale, active_power

    def _hydronic_step(
        self,
        stored_energy: jnp.ndarray,
        source_power: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.hydronic_mode == "instant":
            delivery = jax.nn.sigmoid(self.raw_hydronic_delivery[0])
            return stored_energy, delivery * source_power
        allocation = jax.nn.softmax(self.raw_hydronic_allocation)[:-1]
        retention = 0.5 + 0.499 * jax.nn.sigmoid(self.raw_hydronic_retention)
        delivery = jax.nn.sigmoid(self.raw_hydronic_delivery)
        available = stored_energy + self.dt_hours * allocation * source_power
        released = (1.0 - retention) * available
        next_energy = retention * available
        q_room = jnp.sum(delivery * released) / self.dt_hours
        return next_energy, q_room

    def _rc_step(
        self,
        temperatures: jnp.ndarray,
        *,
        outdoor_temperature: jnp.ndarray,
        solar: jnp.ndarray,
        ventilation: jnp.ndarray,
        q_room: jnp.ndarray,
    ) -> jnp.ndarray:
        capacity = _positive(self.raw_rc_capacity, 1.0)
        edge_g = _positive(self.raw_rc_conductance, 1e-4)
        outdoor_g = _positive(self.raw_rc_outdoor_conductance, 1e-4)
        ventilation_g = _positive(self.raw_ventilation_conductance, 1e-5) * jnp.maximum(
            ventilation, 0.0
        )
        solar_gain = jax.nn.sigmoid(self.raw_solar_gain)

        matrix = jnp.diag(capacity / self.dt_hours + outdoor_g)
        rhs = capacity / self.dt_hours * temperatures + outdoor_g * outdoor_temperature
        for edge in range(self.rc_nodes - 1):
            conductance = edge_g[edge]
            matrix = matrix.at[edge, edge].add(conductance)
            matrix = matrix.at[edge + 1, edge + 1].add(conductance)
            matrix = matrix.at[edge, edge + 1].add(-conductance)
            matrix = matrix.at[edge + 1, edge].add(-conductance)
        matrix = matrix.at[0, 0].add(ventilation_g)
        rhs = rhs.at[0].add(
            ventilation_g * outdoor_temperature + q_room + solar_gain * jnp.maximum(solar, 0.0)
        )
        return jnp.linalg.solve(matrix, rhs)

    def _event_parameters(
        self,
        dynamic_features: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        count = len(self.horizon_steps)
        raw = self.event_net(dynamic_features).reshape(3, 2, count)
        power_mean = self.event_power_cap_w_m2 * jax.nn.sigmoid(raw[0, 0])
        power_scale = 0.1 + self.event_power_cap_w_m2 * jax.nn.sigmoid(raw[0, 1])
        tau_power_mean = self.event_power_cap_w_m2 * jnp.tanh(raw[1, 0])
        tau_power_scale = 0.1 + self.event_power_cap_w_m2 * jax.nn.sigmoid(raw[1, 1])
        tau_temperature_mean = 4.0 * jnp.tanh(raw[2, 0])
        tau_temperature_scale = 0.01 + 4.0 * jax.nn.sigmoid(raw[2, 1])
        return (
            power_mean,
            power_scale,
            tau_power_mean,
            tau_power_scale,
            tau_temperature_mean,
            tau_temperature_scale,
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        history_summary: jnp.ndarray,
        initial_mode_distribution: jnp.ndarray,
        observed_mode: jnp.ndarray,
        observed_age: jnp.ndarray,
        interval_start: jnp.ndarray,
    ) -> CausalHybridRollout:
        normalised_inputs = jax.vmap(self._normalise_inputs)(inputs)
        controller, hydronic_energy, rc_temperature = self._initial_states(
            history_summary,
            initial_temperature,
        )
        initial_duration = jnp.zeros((self.control_interval_steps + 1,), dtype=inputs.dtype)
        carry = (
            controller,
            hydronic_energy,
            rc_temperature,
            initial_mode_distribution,
            initial_duration,
            jnp.asarray(0.0, dtype=inputs.dtype),
            jnp.asarray(0.2, dtype=inputs.dtype),
            jnp.asarray(0.0, dtype=inputs.dtype),
            jnp.asarray(0.0, dtype=inputs.dtype),
            inputs[0, 0],
        )

        ages = jnp.arange(self.max_dwell_steps + 1, dtype=inputs.dtype)

        def step(carry, row):
            (
                controller,
                stored_energy,
                temperatures,
                mode_distribution,
                held_duration_logits,
                held_log_mean,
                held_log_scale,
                held_active_power,
                previous_power,
                previous_setpoint,
            ) = carry
            normalised_input, physical_input, observed_mode_t, observed_age_t, is_start = row
            mode_probability = jnp.sum(mode_distribution[1])
            expected_age = jnp.sum(mode_distribution * ages[jnp.newaxis, :]) / float(
                self.max_dwell_steps
            )
            dynamic = self._dynamic_features(
                normalised_input,
                controller,
                temperatures[0],
                mode_probability,
                expected_age,
                previous_power,
                physical_input[0] - previous_setpoint,
            )
            new_interval = self._interval_parameters(dynamic)
            duration_logits = jnp.where(is_start, new_interval[0], held_duration_logits)
            log_mean = jnp.where(is_start, new_interval[1], held_log_mean)
            log_scale = jnp.where(is_start, new_interval[2], held_log_scale)
            active_power = jnp.where(is_start, new_interval[3], held_active_power)

            availability = physical_input[4]
            hazard = self._hazards(dynamic, availability)
            selected_hazard = hazard[
                jnp.asarray(observed_mode_t, dtype=jnp.int32),
                jnp.minimum(jnp.asarray(observed_age_t, dtype=jnp.int32), self.max_dwell_steps),
            ]
            next_mode_distribution = self._advance_mode_distribution(mode_distribution, hazard)
            next_mode_probability = jnp.sum(next_mode_distribution[1])
            pel = availability * next_mode_probability * active_power
            source = self.cop(physical_input[1]) * pel
            next_stored_energy, q_room = self._hydronic_step(stored_energy, source)
            next_temperatures = self._rc_step(
                temperatures,
                outdoor_temperature=physical_input[1],
                solar=physical_input[2],
                ventilation=physical_input[3],
                q_room=q_room,
            )
            event_parameters = self._event_parameters(dynamic)
            controller_input = jnp.concatenate(
                [dynamic, jnp.asarray([next_mode_probability, pel, q_room], dtype=inputs.dtype)]
            )
            next_controller = 0.9 * controller + 0.1 * jnp.tanh(
                self.controller_net(controller_input)
            )
            prediction = jnp.asarray([next_temperatures[0], q_room, pel])
            output = (
                prediction,
                next_mode_probability,
                selected_hazard,
                duration_logits,
                log_mean,
                log_scale,
                *event_parameters,
                next_controller,
                next_stored_energy,
                next_temperatures,
            )
            next_carry = (
                next_controller,
                next_stored_energy,
                next_temperatures,
                next_mode_distribution,
                duration_logits,
                log_mean,
                log_scale,
                active_power,
                pel,
                physical_input[0],
            )
            return next_carry, output

        _, outputs = jax.lax.scan(
            step,
            carry,
            (
                normalised_inputs,
                inputs,
                observed_mode,
                observed_age,
                interval_start,
            ),
        )
        return CausalHybridRollout(*outputs)
