"""Two-store causality, supervision and disabled-path compatibility."""

from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from neural_building_emulator.columns import CLOSED_LOOP_METADATA_COLUMNS, PLANT_TARGET_SOURCE_COLUMNS, TANK_TEMPERATURE_COLUMNS, closed_loop_required_columns
from neural_building_emulator.data import WindowConfig, make_closed_loop_windows, read_dataset_frame, to_closed_loop_profiles
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator
from neural_building_emulator.scaling import fit_window_scalers, transform_closed_loop_windows
from neural_building_emulator.test_dataset_adapter import _write_entity_profile
from neural_building_emulator.test_plant_supervision import model_kwargs


def make_model(enabled=True):
    options = model_kwargs()
    options.update(metadata_dim=3, metadata_mean=(.004, .0015, 20.), metadata_scale=(1., 1., 1.),
        tank_temperatures_enabled=enabled, hp_buffer_volume_metadata_index=0,
        dhw_volume_metadata_index=1, dhw_recovery_metadata_index=2,
        dhw_request_input_index=6, hp_force_off_input_index=7,
        hp_activation_model="asymmetric_markov", process_noise_mode="heteroscedastic")
    return ProbabilisticContractingClosedLoopHPEmulator(**options)


class TankTemperatureTests(unittest.TestCase):
    def test_label_alignment_and_no_input_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = _write_entity_profile(root)
            path = folder / "timeseries.parquet"
            raw = pd.read_parquet(path)
            raw["thermal_storage_charge_sh_kw"] = np.where(raw.zone == "technical_room", 3., np.nan)
            for col in ["sh_tank_measured_temperature_c", "dhw_tank_mean_temperature_c"]:
                raw[col] = np.nan
                raw.loc[raw.zone == "technical_room", col] = [40., 41., 42., 43.]
            raw.to_parquet(path, index=False)
            cols = closed_loop_required_columns(CLOSED_LOOP_METADATA_COLUMNS)
            df = read_dataset_frame(root, columns=cols + PLANT_TARGET_SOURCE_COLUMNS + TANK_TEMPERATURE_COLUMNS, profile_ids=[123])
            p = to_closed_loop_profiles(df, metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)[0]
            base = to_closed_loop_profiles(df.drop(columns=TANK_TEMPERATURE_COLUMNS), metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)[0]
            np.testing.assert_array_equal(p.plant_targets[:, 3:], [[41., 41.], [42., 42.], [43., 43.]])
            np.testing.assert_array_equal(p.inputs, base.inputs)
            np.testing.assert_array_equal(p.targets, base.targets)
            windows = make_closed_loop_windows([p], WindowConfig(2, 1))
            scaled = transform_closed_loop_windows(windows, fit_window_scalers(windows))
            np.testing.assert_array_equal(scaled.plant_targets, windows.plant_targets)
            df.loc[0, TANK_TEMPERATURE_COLUMNS[0]] = np.nan
            # A missing first interval is unused, but a missing aligned label must fail.
            df.loc[1, TANK_TEMPERATURE_COLUMNS[0]] = np.nan
            with self.assertRaises(ValueError):
                to_closed_loop_profiles(df, metadata_columns=CLOSED_LOOP_METADATA_COLUMNS)

    def test_cold_sh_limits_heat_and_lockout_stops_recharge(self):
        module = make_model().tank_temperatures
        raw = jnp.zeros(15)
        capacities = module.capacities(raw, jnp.array([.004, .0015]))
        q_cold, _ = module.deliver(raw, jnp.array([20., 55.]), capacities, 20., 30., 0., .25)
        q_warm, _ = module.deliver(raw, jnp.array([45., 55.]), capacities, 20., 30., 0., .25)
        self.assertEqual(float(q_cold), 0.)
        self.assertGreater(float(q_warm), 25.)
        common = dict(room_c=20., generation_sh=0., qroom=0., hp_on=0., water_l_h_m2=.1,
                      recovery_w_m2=20., dt=.25, noise=jnp.zeros(2))
        start = jnp.array([40., 40.])
        off, observed = module.advance(raw, start, capacities, generator_available=0., **common)
        on, _ = module.advance(raw, start, capacities, generator_available=1., **common)
        self.assertLess(float(off[1]), 40.)
        self.assertGreater(float(on[1]), float(off[1]))
        np.testing.assert_allclose(observed, (start + off) / 2)

    def test_autonomous_rollout_causality_and_roundtrip(self):
        model = make_model()
        u = jnp.zeros((16, 9)).at[:, 0].set(22.).at[:, 8].set(1.)
        args = dict(key=jax.random.PRNGKey(5), num_particles=3, hp_scenario_mode="bernoulli")
        prediction, aux = model.sample_with_aux(jnp.zeros(3), u, jnp.zeros(1), **args)
        self.assertEqual(prediction.shape, (3, 16, 3))
        self.assertEqual(aux[-1].shape, (3, 16, 2))
        self.assertTrue(np.isfinite(aux[-1]).all())
        changed, other = model.sample_with_aux(jnp.zeros(3), u.at[8:, 6].set(2.), jnp.zeros(1), **args)
        np.testing.assert_array_equal(prediction[:, :8], changed[:, :8])
        np.testing.assert_array_equal(aux[-1][:, :8], other[-1][:, :8])
        self.assertGreater(float(jnp.max(jnp.abs(aux[-1][:, 8:] - other[-1][:, 8:]))), .01)
        off, _ = model.sample_with_aux(jnp.zeros(3), u.at[:, 7].set(1.), jnp.zeros(1), **args)
        np.testing.assert_allclose(off[..., 2] * 4 + 2, 0., atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.eqx"
            eqx.tree_serialise_leaves(path, model)
            recovered = eqx.tree_deserialise_leaves(path, model)
            np.testing.assert_array_equal(prediction, recovered.sample(jnp.zeros(3), u, jnp.zeros(1), **args))

    def test_hard_switch_gradients_and_disabled_shape(self):
        model = make_model()
        u = jnp.zeros((8, 9)).at[:, 0].set(22.).at[:, 8].set(1.)
        def loss(m):
            y, aux = m.sample_with_aux(jnp.zeros(3), u, jnp.zeros(1), key=jax.random.PRNGKey(4),
                                      num_particles=2, hp_scenario_mode="hard_expected_grad")
            return jnp.mean(y ** 2) + jnp.mean((aux[-1] / 10.) ** 2)
        value, grad = eqx.filter_value_and_grad(loss)(model)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad)))
        self.assertGreater(float(jnp.linalg.norm(grad.tank_temperatures.parameter_net.layers[-1].weight)), 0.)
        legacy = make_model(False)
        self.assertIsNone(legacy.tank_temperatures)
        _, aux = legacy.sample_with_aux(jnp.zeros(3), u, jnp.zeros(1), key=jax.random.PRNGKey(4), num_particles=2)
        self.assertEqual(len(aux), 21)


if __name__ == "__main__":
    unittest.main()
