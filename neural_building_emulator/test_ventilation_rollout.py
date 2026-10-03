from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from .ventilation import (
    build_eplus_ventilation_schedule,
    replace_with_eplus_ventilation,
)


class EPlusVentilationRolloutTests(unittest.TestCase):
    def test_schedule_uses_night_hours_and_seasonal_offsets(self) -> None:
        datetimes = np.asarray(
            [
                "2020-05-15T07:45:00",
                "2020-05-15T08:00:00",
                "2020-05-16T20:00:00",
                "2020-09-30T20:15:00",
            ],
            dtype="datetime64[s]",
        )
        schedule = build_eplus_ventilation_schedule(
            datetimes,
            np.asarray([21.0, 180.0], dtype=np.float32),
            ("shSetpoint", "volume"),
        )

        np.testing.assert_array_equal(schedule.time_available, [1.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(
            schedule.minimum_indoor_temperature_c,
            [24.0, 24.0, 22.0, 24.0],
        )
        self.assertAlmostEqual(schedule.nominal_flow_m3_s, 0.125)

    def test_rule_uses_the_particle_temperature(self) -> None:
        common = dict(
            time_available_t=jnp.asarray(1.0),
            minimum_indoor_temperature_c_t=jnp.asarray(22.0),
            nominal_flow_m3_s=jnp.asarray(0.125),
            minimum_indoor_outdoor_delta_c=jnp.asarray(2.0),
            ventilation_input_index=3,
            outdoor_input_index=1,
            input_mean=(0.0, 0.0, 0.0, 0.05),
            input_scale=(1.0, 1.0, 1.0, 0.05),
            temperature_mean_c=20.0,
            temperature_scale_c=2.0,
        )
        scaled_input = jnp.asarray([0.0, 15.0, 0.0, 99.0])

        closed = replace_with_eplus_ventilation(
            scaled_input,
            jnp.asarray(0.5),  # 21 C
            **common,
        )
        opened = replace_with_eplus_ventilation(
            scaled_input,
            jnp.asarray(1.5),  # 23 C
            **common,
        )

        self.assertAlmostEqual(float(closed[3]), -1.0)
        self.assertAlmostEqual(float(opened[3]), 1.5)


if __name__ == "__main__":
    unittest.main()
