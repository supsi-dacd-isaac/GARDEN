"""Tests for seasonal HP gating and epoch-wise window-grid rotation."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from neural_building_emulator.columns import (
    CLOSED_LOOP_INPUT_COLUMNS,
    CLOSED_LOOP_METADATA_COLUMNS,
    DATETIME_COLUMN,
    DISTURBANCE_COLUMNS,
    HEAT_PUMP_ELECTRIC_POWER_COLUMN,
    HP_MODEL_NAME_COLUMN,
    HP_MODE_IS_DHW_COLUMN,
    HP_REF_CAPACITY_COLUMN,
    HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN,
    HP_REF_COP_COLUMN,
    METADATA_COLUMNS,
    PROFILE_ID_COLUMN,
    SETPOINT_TIMESERIES_COLUMN,
    SH_DESIGN_CAPACITY_COLUMN,
    SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN,
    SH_VOLUME_COLUMN,
    SH_VOLUME_PER_HEATED_AREA_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    TARGET_COLUMN,
    ZONE_THERMAL_HEATING_POWER_COLUMN,
)
from neural_building_emulator.data import (
    _space_heating_availability,
    _window_starts,
    to_closed_loop_profiles,
    to_profiles,
)
from neural_building_emulator.models import (
    ClosedLoopHPEmulator,
    ContractingClosedLoopHPEmulator,
    ProbabilisticClosedLoopHPEmulator,
    ProbabilisticContractingClosedLoopHPEmulator,
)
from neural_building_emulator.models.probabilistic_closed_loop_hp import (
    bounded_lognormal_expm1_mean,
    hard_bernoulli_expected_gradient,
    straight_through_binary_concrete,
)
from neural_building_emulator.train import rotating_window_start_offset


class SeasonalAvailabilityTests(unittest.TestCase):
    def test_energyplus_summer_lockout_boundaries(self) -> None:
        timestamps = np.asarray(
            [
                "2021-05-14T23:45:00",
                "2021-05-15T00:00:00",
                "2021-09-30T23:45:00",
                "2021-10-01T00:00:00",
            ],
            dtype="datetime64[s]",
        )
        np.testing.assert_array_equal(
            _space_heating_availability(timestamps),
            np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_all_closed_loop_models_hard_gate_heat_and_power(self) -> None:
        input_dim = 9
        common = dict(
            metadata_dim=3,
            input_dim=input_dim,
            state_dim=3,
            hidden_dim=8,
            depth=2,
            hp_cop_cap=8.0,
            hp_pel_cap_w_m2=80.0,
            hp_qroom_cap_w_m2=120.0,
            hp_energy_cap_wh_m2=400.0,
            input_mean=(0.0,) * input_dim,
            input_scale=(1.0,) * input_dim,
            target_mean=(20.0, 10.0, 5.0),
            target_scale=(2.0, 4.0, 3.0),
            availability_input_index=4,
            bptt_truncate_steps=0,
        )
        keys = jax.random.split(jax.random.PRNGKey(17), 4)
        models = (
            ClosedLoopHPEmulator(**common, controller_state_dim=2, key=keys[0]),
            ContractingClosedLoopHPEmulator(**common, key=keys[1]),
            ProbabilisticClosedLoopHPEmulator(
                **common,
                controller_state_dim=2,
                latent_dim=2,
                process_noise_mode="none",
                key=keys[2],
            ),
            ProbabilisticContractingClosedLoopHPEmulator(
                **common,
                controller_state_dim=2,
                latent_dim=2,
                process_noise_mode="none",
                key=keys[3],
            ),
        )
        metadata = jnp.zeros((3,), dtype=jnp.float32)
        inputs = jnp.zeros((3, input_dim), dtype=jnp.float32)
        initial_temperature = jnp.zeros((1,), dtype=jnp.float32)

        for model in models:
            if isinstance(
                model,
                (ProbabilisticClosedLoopHPEmulator, ProbabilisticContractingClosedLoopHPEmulator),
            ):
                prediction = model.sample(
                    metadata,
                    inputs,
                    initial_temperature,
                    key=jax.random.PRNGKey(23),
                    num_particles=2,
                    sample_process_noise=False,
                    hp_scenario_mode="bernoulli",
                )
            else:
                prediction = model(metadata, inputs, initial_temperature)[jnp.newaxis, ...]
            qroom = prediction[..., 1] * 4.0 + 10.0
            pel = prediction[..., 2] * 3.0 + 5.0
            np.testing.assert_allclose(np.asarray(qroom), 0.0, atol=1e-6)
            np.testing.assert_allclose(np.asarray(pel), 0.0, atol=1e-6)


class ClosedLoopPowerAreaNormalizationTests(unittest.TestCase):
    @staticmethod
    def _profile_frame() -> pd.DataFrame:
        row_count = 3
        data: dict[str, object] = {
            DATETIME_COLUMN: pd.date_range(
                "2021-01-01",
                periods=row_count,
                freq="15min",
            ),
            PROFILE_ID_COLUMN: np.full(row_count, 42),
            TARGET_COLUMN: np.asarray([20.0, 21.0, 22.0]),
            SETPOINT_TIMESERIES_COLUMN: np.full(row_count, 21.0),
            ZONE_THERMAL_HEATING_POWER_COLUMN: np.asarray([400.0, 800.0, 1200.0]),
            HEAT_PUMP_ELECTRIC_POWER_COLUMN: np.asarray([800.0, 1600.0, 2400.0]),
            HP_MODE_IS_DHW_COLUMN: np.zeros(row_count),
        }
        for column in DISTURBANCE_COLUMNS:
            data[column] = np.zeros(row_count)
        for index, column in enumerate(METADATA_COLUMNS):
            data[column] = np.full(row_count, float(index + 1))
        data["floor_area"] = np.full(row_count, 100.0)
        data["totalFloors"] = np.full(row_count, 4.0)
        data[HP_REF_CAPACITY_COLUMN] = np.full(row_count, 40_000.0)
        data[SH_DESIGN_CAPACITY_COLUMN] = np.full(row_count, 20_000.0)
        data[SH_VOLUME_COLUMN] = np.full(row_count, 2.0)
        data[HP_REF_COP_COLUMN] = np.full(row_count, 4.1)
        data[HP_MODEL_NAME_COLUMN] = np.full(row_count, "aerotop_t35r")
        return pd.DataFrame(data)

    def test_closed_loop_hp_equipment_metadata_is_scaled_and_one_hot(self) -> None:
        profile = to_closed_loop_profiles(
            self._profile_frame(),
            hp_power_area_normalization="building_heated_area",
            metadata_columns=CLOSED_LOOP_METADATA_COLUMNS,
        )[0]
        values = dict(zip(CLOSED_LOOP_METADATA_COLUMNS, profile.metadata))

        self.assertEqual(profile.metadata.shape, (len(CLOSED_LOOP_METADATA_COLUMNS),))
        self.assertEqual(values["totalFloors"], 4.0)
        self.assertAlmostEqual(values[HP_REF_CAPACITY_PER_HEATED_AREA_COLUMN], 100.0)
        self.assertAlmostEqual(values[SH_DESIGN_CAPACITY_PER_HEATED_AREA_COLUMN], 50.0)
        self.assertAlmostEqual(values[SH_VOLUME_PER_HEATED_AREA_COLUMN], 0.005)
        self.assertAlmostEqual(values[HP_REF_COP_COLUMN], 4.1)
        self.assertEqual(values["hp_model_name__aerotop_g07_14m"], 0.0)
        self.assertEqual(values["hp_model_name__aerotop_t35r"], 1.0)

    def test_corrected_mode_uses_zone_area_for_q_and_building_area_for_pel(self) -> None:
        profile = to_closed_loop_profiles(
            self._profile_frame(),
            hp_power_area_normalization="building_heated_area",
        )[0]

        np.testing.assert_allclose(profile.targets[:, 0], [21.0, 22.0])
        np.testing.assert_allclose(profile.targets[:, 1], [8.0, 12.0])
        np.testing.assert_allclose(profile.targets[:, 2], [4.0, 6.0])

    def test_energyplus_interval_rows_are_aligned_without_a_one_step_shift(self) -> None:
        frame = self._profile_frame()
        frame[SETPOINT_TIMESERIES_COLUMN] = [19.0, 20.0, 21.0]
        profile = to_closed_loop_profiles(frame)[0]

        np.testing.assert_allclose(profile.initial_temperature[:, 0], [20.0, 21.0])
        np.testing.assert_allclose(profile.inputs[:, 0], [20.0, 21.0])
        np.testing.assert_allclose(profile.targets[:, 0], [21.0, 22.0])

    def test_availability_does_not_erase_simulator_power_targets(self) -> None:
        frame = self._profile_frame()
        frame[SPACE_HEATING_AVAILABILITY_COLUMN] = 0.0
        profile = to_closed_loop_profiles(
            frame,
            hp_power_area_normalization="building_heated_area",
        )[0]

        availability_index = list(CLOSED_LOOP_INPUT_COLUMNS).index(
            SPACE_HEATING_AVAILABILITY_COLUMN
        )
        np.testing.assert_array_equal(profile.inputs[:, availability_index], 0.0)
        np.testing.assert_allclose(profile.targets[:, 1], [8.0, 12.0])
        np.testing.assert_allclose(profile.targets[:, 2], [4.0, 6.0])

    def test_legacy_mode_reproduces_zone_area_hp_normalization(self) -> None:
        profile = to_closed_loop_profiles(
            self._profile_frame(),
            hp_power_area_normalization="zone_floor_area",
        )[0]

        np.testing.assert_allclose(profile.targets[:, 1], [8.0, 12.0])
        np.testing.assert_allclose(profile.targets[:, 2], [16.0, 24.0])

    def test_q_to_t_electric_mode_uses_the_same_hp_area_semantics(self) -> None:
        corrected = to_profiles(
            self._profile_frame(),
            "heating_electric",
            hp_power_area_normalization="building_heated_area",
        )[0]
        legacy = to_profiles(
            self._profile_frame(),
            "heating_electric",
            hp_power_area_normalization="zone_floor_area",
        )[0]

        np.testing.assert_allclose(corrected.inputs[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(legacy.inputs[:, 0], [8.0, 16.0, 24.0])


class RotatingWindowTests(unittest.TestCase):
    def test_rotation_preserves_window_count_and_reaches_full_stride(self) -> None:
        canonical = _window_starts(last_start=100, stride=30, start_offset=0)
        shifted = _window_starts(last_start=100, stride=30, start_offset=29)
        self.assertEqual(len(canonical), len(shifted))
        self.assertEqual(canonical, [0, 30, 60, 90])
        self.assertEqual(shifted, [29, 59, 89, 100])

        offsets = [rotating_window_start_offset(epoch, 20, 2024) for epoch in range(1, 21)]
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], 2023)
        self.assertEqual(offsets, sorted(set(offsets)))


class ProbabilisticHPEmissionTests(unittest.TestCase):
    def test_bounded_expected_power_matches_clipped_monte_carlo_samples(self) -> None:
        key = jax.random.PRNGKey(29)
        log_mu = jnp.asarray([0.2, 2.0, 4.0], dtype=jnp.float32)
        log_sigma = jnp.asarray([0.3, 0.5, 0.2], dtype=jnp.float32)
        max_log = jnp.log1p(jnp.asarray(80.0, dtype=jnp.float32))
        expected = bounded_lognormal_expm1_mean(log_mu, log_sigma, max_log)
        noise = jax.random.normal(key, (3, 200_000), dtype=jnp.float32)
        samples = jnp.expm1(
            jnp.clip(log_mu[:, None] + log_sigma[:, None] * noise, 0.0, max_log)
        )
        np.testing.assert_allclose(
            np.asarray(expected),
            np.asarray(jnp.mean(samples, axis=1)),
            rtol=1e-2,
            atol=2e-2,
        )

    def test_straight_through_binary_concrete_is_hard_with_finite_gradient(self) -> None:
        uniform = jnp.asarray(0.8, dtype=jnp.float32)
        temperature = jnp.asarray(0.5, dtype=jnp.float32)

        def sampled_mode(logit: jnp.ndarray) -> jnp.ndarray:
            return straight_through_binary_concrete(logit, uniform, temperature)

        value = sampled_mode(jnp.asarray(0.0, dtype=jnp.float32))
        gradient = jax.grad(sampled_mode)(jnp.asarray(0.0, dtype=jnp.float32))
        self.assertIn(float(value), (0.0, 1.0))
        self.assertTrue(np.isfinite(float(gradient)))
        self.assertGreater(float(gradient), 0.0)

    def test_hard_bernoulli_uses_hard_forward_and_probability_gradient(self) -> None:
        uniform = jnp.asarray(0.3, dtype=jnp.float32)

        def sampled_mode(logit: jnp.ndarray) -> jnp.ndarray:
            probability = jax.nn.sigmoid(logit)
            return hard_bernoulli_expected_gradient(probability, uniform)

        logit = jnp.asarray(0.0, dtype=jnp.float32)
        value = sampled_mode(logit)
        gradient = jax.grad(sampled_mode)(logit)
        self.assertEqual(float(value), 1.0)
        self.assertAlmostEqual(float(gradient), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
