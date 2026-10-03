from __future__ import annotations

import unittest

import jax.numpy as jnp

from neural_building_emulator_refactor.rc_order_sufficiency_diagnostic import (
    INTERNAL_INDEX,
    OCCUPANCY_INDEX,
    SOLAR_INDEX,
    TOUT_INDEX,
    VENT_INDEX,
    _initial_parameters,
    _parameter_values,
    _rc_step,
)


class RCOrderSufficiencyTests(unittest.TestCase):
    def test_monotone_qroom_gain_preserves_increasing_heat_response(self) -> None:
        values = _parameter_values(_initial_parameters(order=2))
        input_dim = max(
            TOUT_INDEX,
            VENT_INDEX,
            SOLAR_INDEX,
            INTERNAL_INDEX,
            OCCUPANCY_INDEX,
        ) + 1
        row = jnp.zeros((input_dim,), dtype=jnp.float32).at[TOUT_INDEX].set(20.0)
        scales = jnp.ones((5,), dtype=jnp.float32)
        air = jnp.asarray(20.0)
        mass = jnp.asarray([20.0])

        next_temperatures = []
        for qroom in (0.0, 0.5, 1.0, 2.0):
            next_air, _ = _rc_step(
                values,
                air,
                mass,
                row,
                jnp.asarray(qroom),
                scales,
                delta_cap_c=2.0,
                qroom_routing="air_only",
                qroom_gain_mode="monotone",
            )
            next_temperatures.append(float(next_air))

        self.assertEqual(next_temperatures, sorted(next_temperatures))
        self.assertGreater(next_temperatures[-1], next_temperatures[0])


if __name__ == "__main__":
    unittest.main()
