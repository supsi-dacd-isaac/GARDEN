"""Plant target alignment, normalized energy boundaries and optional losses."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from neural_building_emulator.columns import (
    BUILDING_THERMAL_HEATING_POWER_COLUMN, CLOSED_LOOP_METADATA_COLUMNS,
    HEAT_PUMP_ELECTRIC_POWER_COLUMN, HP_MODE_IS_DHW_COLUMN,
    PLANT_TARGET_SOURCE_COLUMNS, SH_STORAGE_CHARGE_COLUMN, closed_loop_required_columns,
)
from neural_building_emulator.data import (
    WindowConfig, make_closed_loop_windows, read_dataset_frame, to_closed_loop_profiles,
)
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator
from neural_building_emulator.plant_supervision import plant_block_crps
from neural_building_emulator.scaling import fit_window_scalers, transform_closed_loop_profiles, transform_closed_loop_windows
from neural_building_emulator.test_dataset_adapter import _write_entity_profile
from neural_building_emulator.train import fixed_evaluation_subset, grouped_profile_minibatches, minibatches


def model_kwargs():
    return dict(metadata_dim=1, input_dim=9, state_dim=4, controller_state_dim=2,
                latent_dim=2, hidden_dim=8, depth=2, process_noise_mode="none",
                hp_pel_cap_w_m2=20., hp_qroom_cap_w_m2=100., hp_energy_cap_wh_m2=10.,
                input_mean=(0.,) * 9, input_scale=(1.,) * 9,
                target_mean=(20., 5., 2.), target_scale=(2., 10., 4.),
                availability_input_index=8, key=jax.random.PRNGKey(0))


class PlantSupervisionTests(unittest.TestCase):
    def test_targets_are_separate_aligned_and_not_scaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = _write_entity_profile(root)
            path = folder / "timeseries.parquet"
            raw = pd.read_parquet(path)
            raw[SH_STORAGE_CHARGE_COLUMN] = np.where(raw.zone == "technical_room", 3., np.nan)
            raw.to_parquet(path, index=False)
            columns = closed_loop_required_columns(CLOSED_LOOP_METADATA_COLUMNS)
            frame = read_dataset_frame(root, columns=columns + PLANT_TARGET_SOURCE_COLUMNS, profile_ids=[123])
            np.testing.assert_allclose(frame[BUILDING_THERMAL_HEATING_POWER_COLUMN], [2000, 2002, 2004, 2006])
            profile = to_closed_loop_profiles(frame, metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)[0]
            legacy = to_closed_loop_profiles(frame.drop(columns=PLANT_TARGET_SOURCE_COLUMNS),
                                             metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)[0]
            np.testing.assert_array_equal(profile.inputs, legacy.inputs)
            np.testing.assert_array_equal(profile.targets, legacy.targets)
            np.testing.assert_allclose(profile.plant_targets, [[30, 20.02, 0], [30, 20.04, 0], [30, 20.06, 0]], rtol=1e-6)
            windows = make_closed_loop_windows([profile], WindowConfig(sequence_length=2, stride=1))
            scalers = fit_window_scalers(windows)
            scaled = transform_closed_loop_windows(windows, scalers)
            np.testing.assert_array_equal(scaled.plant_targets, windows.plant_targets)
            scaled_profiles = transform_closed_loop_profiles([profile], scalers)
            np.testing.assert_array_equal(scaled_profiles[0].plant_targets, profile.plant_targets)
            subset = fixed_evaluation_subset(scaled, max_windows=1, seed=1)
            np.testing.assert_array_equal(subset.plant_targets[0], windows.plant_targets[subset.start_indices[0]])
            batch = next(minibatches(scaled, batch_size=2, rng=np.random.default_rng(1), shuffle=False, include_plant_targets=True))
            np.testing.assert_array_equal(batch[4], windows.plant_targets)
            grouped = next(grouped_profile_minibatches(scaled, batch_size=2, windows_per_profile=2,
                           rng=np.random.default_rng(1), shuffle=False, include_plant_targets=True))
            np.testing.assert_array_equal(grouped[4], windows.plant_targets)
            self.assertEqual(len(next(minibatches(scaled, batch_size=2, rng=np.random.default_rng(1), shuffle=False))), 4)
            with self.assertRaises(ValueError):
                make_closed_loop_windows([profile, replace(profile, plant_targets=None)], WindowConfig(2, 1))

    def test_whole_building_aggregation_rejects_missing_zone_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = _write_entity_profile(root)
            path = folder / "timeseries.parquet"
            raw = pd.read_parquet(path)
            index = raw.index[raw.zone == "fl1_thz0"][0]
            raw = raw.drop(index)
            raw.to_parquet(path, index=False)
            with self.assertRaises(ValueError):
                read_dataset_frame(root, columns=[BUILDING_THERMAL_HEATING_POWER_COLUMN], profile_ids=[123])

    def test_dhw_supervision_ignores_latched_mode_without_electricity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_entity_profile(root)
            columns = closed_loop_required_columns(CLOSED_LOOP_METADATA_COLUMNS)
            frame = read_dataset_frame(root, columns=columns, profile_ids=[123])
            frame[SH_STORAGE_CHARGE_COLUMN] = 0.
            frame[BUILDING_THERMAL_HEATING_POWER_COLUMN] = 2000.
            frame[HP_MODE_IS_DHW_COLUMN] = [1., 1., 1., 0.]
            frame[HEAT_PUMP_ELECTRIC_POWER_COLUMN] = [0., 0., 200., 300.]
            profile = to_closed_loop_profiles(frame, metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)[0]
            np.testing.assert_array_equal(profile.plant_targets[:, 2], [0., 1., 0.])

    def test_buffer_withdraws_building_heat_not_zone_specific_heat(self):
        model = ProbabilisticContractingClosedLoopHPEmulator(**model_kwargs())
        for cap in (None, jnp.asarray(10.)):
            q, available, energy = model._buffer_draw(jnp.asarray(10.), jnp.asarray(0.),
                jnp.asarray(20.), jnp.asarray(.25), jnp.asarray(0.), cap, zone_heat_ratio=.5)
            self.assertAlmostEqual(float(q), 20.)
            self.assertAlmostEqual(float(available), 20.)
            self.assertAlmostEqual(float(energy), 0.)
            self.assertAlmostEqual(float(10 - .25 * q / .5 - energy), 0.)

    def test_optional_head_starts_at_identity_and_serializes(self):
        model = ProbabilisticContractingClosedLoopHPEmulator(**model_kwargs(), hp_whole_building_accounting=True)
        self.assertEqual(float(model.zone_heat_ratio(jnp.ones(1), jnp.ones(2))), 1.)
        legacy = ProbabilisticContractingClosedLoopHPEmulator(**model_kwargs())
        self.assertIsNone(legacy.zone_heat_ratio_net)
        inputs = jnp.zeros((4, 9)).at[:, 8].set(1.)
        kwargs = dict(key=jax.random.PRNGKey(2), num_particles=2)
        before = model.sample(jnp.zeros(1), inputs, jnp.zeros(1), **kwargs)
        np.testing.assert_allclose(before, legacy.sample(jnp.zeros(1), inputs, jnp.zeros(1), **kwargs), atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.eqx"
            eqx.tree_serialise_leaves(path, model)
            recovered = eqx.tree_deserialise_leaves(path, model)
            np.testing.assert_array_equal(before, recovered.sample(jnp.zeros(1), inputs, jnp.zeros(1), **kwargs))

    def test_block_crps_matches_brute_force_and_has_finite_gradients(self):
        prediction = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 8, 2))
        truth = jax.random.normal(jax.random.PRNGKey(3), (2, 8, 2))
        score = plant_block_crps(prediction, truth, horizon_steps=(2, 4), power_scale=2.)
        expected = []
        for step in (2, 4):
            x = np.asarray(prediction).reshape(2, 3, 8 // step, step, 2).mean(axis=3)
            y = np.asarray(truth).reshape(2, 8 // step, step, 2).mean(axis=2)
            first = np.abs(x - y[:, None]).mean(axis=(0, 1, 2))
            second = .5 * np.abs(x[:, :, None] - x[:, None, :]).mean(axis=(0, 1, 2, 3))
            expected.append((first - second) / 2 * np.sqrt(8))
        np.testing.assert_allclose(score, np.mean(expected, axis=0), rtol=1e-6)
        gradient = jax.grad(lambda x: plant_block_crps(x, truth, horizon_steps=(2, 4), power_scale=2.).sum())(prediction)
        self.assertTrue(np.isfinite(gradient).all())
        with self.assertRaises(ValueError):
            plant_block_crps(prediction, truth, horizon_steps=(16,), power_scale=2.)


if __name__ == "__main__":
    unittest.main()
