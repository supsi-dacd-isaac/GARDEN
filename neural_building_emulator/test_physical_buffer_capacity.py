"""Tests for the metadata-conditioned physical HP buffer capacity."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from neural_building_emulator.models import (
    ProbabilisticContractingClosedLoopHPEmulator,
)


def _physical_buffer_model() -> ProbabilisticContractingClosedLoopHPEmulator:
    return ProbabilisticContractingClosedLoopHPEmulator(
        metadata_dim=1,
        input_dim=9,
        state_dim=4,
        controller_state_dim=2,
        latent_dim=2,
        hidden_dim=8,
        depth=2,
        process_noise_mode="none",
        hp_pel_cap_w_m2=20.0,
        hp_qroom_cap_w_m2=100.0,
        hp_energy_cap_wh_m2=0.0,
        hp_buffer_capacity_mode="physical_volume",
        hp_buffer_delta_t_usable_init_k=5.0,
        hp_buffer_delta_t_min_k=1.0,
        hp_buffer_delta_t_max_k=15.0,
        hp_buffer_water_heat_capacity_wh_m3_k=1163.0,
        hp_buffer_volume_metadata_index=0,
        input_mean=(0.0,) * 9,
        input_scale=(1.0,) * 9,
        target_mean=(20.0, 5.0, 2.0),
        target_scale=(2.0, 10.0, 4.0),
        metadata_mean=(0.003,),
        metadata_scale=(0.001,),
        availability_input_index=8,
        key=jax.random.PRNGKey(0),
    )


class PhysicalBufferCapacityTest(unittest.TestCase):
    def test_capacity_uses_denormalized_volume_and_learned_delta_t(self) -> None:
        model = _physical_buffer_model()
        metadata = jnp.asarray([1.0], dtype=jnp.float32)

        self.assertAlmostEqual(float(model.buffer_delta_t_usable_k(jnp.float32)), 5.0)
        self.assertAlmostEqual(
            float(model.buffer_capacity_wh_m2(metadata)),
            0.004 * 1163.0 * 5.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(
                model._energy_scaled(
                    model.buffer_capacity_wh_m2(metadata) / 2.0,
                    metadata,
                )
            ),
            0.5,
            places=6,
        )

    def test_generated_heat_serves_room_before_storage_is_clipped(self) -> None:
        model = _physical_buffer_model()
        qroom, available_power, energy_next = model._buffer_draw(
            jnp.asarray(10.0),
            jnp.asarray(20.0),
            jnp.asarray(20.0),
            jnp.asarray(0.25),
            jnp.asarray(0.0),
            jnp.asarray(10.0),
        )

        self.assertAlmostEqual(float(qroom), 20.0)
        self.assertAlmostEqual(float(available_power), 60.0)
        self.assertAlmostEqual(float(energy_next), 10.0)

    def test_initial_energy_is_a_fraction_of_building_capacity(self) -> None:
        model = _physical_buffer_model()
        metadata = jnp.asarray([0.0], dtype=jnp.float32)
        capacity = model.buffer_capacity_wh_m2(metadata)
        energy = model.initial_energy(
            metadata,
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
        )

        self.assertGreaterEqual(float(energy), 0.0)
        self.assertLessEqual(float(energy), float(capacity))


if __name__ == "__main__":
    unittest.main()
