"""Tests for the building-local architecture prototype."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.columns import closed_loop_input_columns
from neural_building_emulator_refactor.single_building_architecture_search import (
    AVAILABILITY_INDEX,
    FORCE_OFF_INDEX,
    VENT_INDEX,
    Architecture,
    PolicyProfile,
    _architectures,
    _annual_ensemble_metrics,
    _blocked_segments,
    _bounded_mlp_alpha,
    _candidate_parameters,
    _event_aligned_temperature_response_loss,
    _observed_q_thermal_rollout,
    _passive_memory_positive_q_step,
    _positive_thermal_q_forcing,
    _rollout,
)
from neural_building_emulator_refactor.single_building_architecture_search import Scale
from neural_building_emulator.data import ClosedLoopProfile


class SingleBuildingArchitectureSearchTests(unittest.TestCase):
    def test_default_grid_contains_27_unique_architectures(self) -> None:
        class Args:
            hp_architectures = ("continuous", "hurdle", "hysteretic")
            hydronic_architectures = ("direct", "buffer", "causal_filter")
            thermal_architectures = ("bounded_mlp", "passive_rc", "gru")
            max_candidates = 0
            seed = 13

        rows = _architectures(Args())
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({row.name for row in rows}), 27)

    def test_blocked_split_distributes_holdouts(self) -> None:
        steps = 96 * 84
        profile = ClosedLoopProfile(
            profile_id=1,
            datetime=np.arange(steps),
            metadata=np.empty((0,), dtype=np.float32),
            inputs=np.zeros((steps, 13), dtype=np.float32),
            targets=np.zeros((steps, 3), dtype=np.float32),
            initial_temperature=np.full((steps, 1), 20.0, dtype=np.float32),
        )
        rows = _blocked_segments(
            PolicyProfile("setpoint", profile),
            train_days=28,
            test_days=14,
            steps_per_day=96,
        )
        held_out = [row for row in rows if row.held_out]
        self.assertEqual(len(held_out), 2)
        self.assertGreater(held_out[1].start, held_out[0].end)

    def test_representative_rollouts_are_finite_bounded_and_force_gated(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        inputs = np.zeros((8, input_dim), dtype=np.float32)
        inputs[:, 0] = 22.0
        inputs[:, AVAILABILITY_INDEX] = 1.0
        inputs[:, FORCE_OFF_INDEX] = np.asarray([0, 0, 1, 1, 0, 0, 0, 0])
        inputs[:, 1] = 5.0
        variants = (
            Architecture("continuous", "direct", "bounded_mlp"),
            Architecture("hurdle", "buffer", "bounded_mlp_feedback"),
            Architecture("hurdle", "buffer", "bounded_mlp_dual_alpha"),
            Architecture("hurdle", "buffer", "bounded_mlp_physical_alpha"),
            Architecture("hurdle", "buffer", "passive_memory_positive_q"),
            Architecture("hurdle", "buffer", "state_aware_mlp"),
            Architecture("hurdle", "buffer", "setpoint_error_ssm"),
            Architecture("hurdle", "buffer", "passive_rc"),
            Architecture("hurdle", "leaky_emitter", "bounded_mlp"),
            Architecture("hysteretic", "causal_filter", "gru"),
            Architecture(
                "asymmetric_hurdle",
                "positive_convolution",
                "stable_rc",
            ),
            Architecture(
                "asymmetric_hurdle",
                "positive_convolution",
                "stable_rc_narx",
            ),
            Architecture(
                "recovery_hurdle",
                "positive_convolution",
                "stable_rc",
            ),
            Architecture(
                "recovery_hurdle",
                "positive_convolution",
                "stable_rc_narx",
            ),
            Architecture(
                "recovery_hurdle",
                "buffer",
                "stable_rc_narx_positive_q",
            ),
        )
        for index, architecture in enumerate(variants):
            params = _candidate_parameters(
                architecture,
                input_dim=input_dim,
                hidden_dim=8,
                depth=1,
                key=jax.random.PRNGKey(index),
            )
            prediction, _ = _rollout(
                params,
                architecture,
                jnp.asarray(inputs),
                jnp.asarray([20.0]),
                input_mean=jnp.zeros((input_dim,)),
                input_std=jnp.ones((input_dim,)),
                target_mean=jnp.asarray([20.0, 0.0, 0.0]),
                target_std=jnp.ones((3,)),
                pel_cap=20.0,
                q_cap=80.0,
                energy_cap=1000.0,
                hidden_dim=8,
                dt_hours=0.25,
                temperature_delta_cap=2.0,
            )
            values = np.asarray(prediction)
            self.assertTrue(np.all(np.isfinite(values)), architecture.name)
            self.assertTrue(np.all((values[:, 0] >= 5.0) & (values[:, 0] <= 40.0)))
            np.testing.assert_array_equal(values[2:4, 2], np.zeros((2,)))

    def test_disabled_teacher_feedback_is_exactly_a_noop(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        inputs = jnp.zeros((8, input_dim)).at[:, 0].set(21.0)
        architecture = Architecture("hurdle", "buffer", "bounded_mlp")
        params = _candidate_parameters(
            architecture,
            input_dim=input_dim,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(106),
        )
        kwargs = {
            "input_mean": jnp.zeros((input_dim,)),
            "input_std": jnp.ones((input_dim,)),
            "target_mean": jnp.asarray([20.0, 0.0, 0.0]),
            "target_std": jnp.ones((3,)),
            "pel_cap": 20.0,
            "q_cap": 80.0,
            "energy_cap": 1000.0,
            "hidden_dim": 8,
            "dt_hours": 0.25,
            "temperature_delta_cap": 2.0,
        }
        baseline, baseline_modes = _rollout(
            params, architecture, inputs, jnp.asarray([20.0]), **kwargs
        )
        explicit, explicit_modes = _rollout(
            params,
            architecture,
            inputs,
            jnp.asarray([20.0]),
            teacher_feedback=jnp.ones((8, 4)),
            **kwargs,
        )
        np.testing.assert_array_equal(
            np.asarray(baseline), np.asarray(explicit)
        )
        np.testing.assert_array_equal(
            np.asarray(baseline_modes), np.asarray(explicit_modes)
        )

    def test_bounded_mlp_alpha_variants_have_expected_physical_ordering(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        mean = jnp.zeros((input_dim,))
        std = jnp.ones((input_dim,))
        row = jnp.zeros((input_dim,)).at[1].set(10.0)

        dual = Architecture("hurdle", "buffer", "bounded_mlp_dual_alpha")
        dual_params = _candidate_parameters(
            dual,
            input_dim=input_dim,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(101),
        )
        dual_params["thermal_alpha_free_raw"] = jnp.asarray(-3.0)
        dual_params["thermal_alpha_heating_raw"] = jnp.asarray(-1.0)
        alpha_free = _bounded_mlp_alpha(
            dual_params,
            dual,
            row.at[AVAILABILITY_INDEX].set(0.0),
            jnp.asarray(20.0),
            input_mean=mean,
            input_std=std,
            dt_hours=0.25,
        )
        alpha_heating = _bounded_mlp_alpha(
            dual_params,
            dual,
            row.at[AVAILABILITY_INDEX].set(1.0),
            jnp.asarray(20.0),
            input_mean=mean,
            input_std=std,
            dt_hours=0.25,
        )
        self.assertGreater(float(alpha_heating), float(alpha_free))

        physical = Architecture(
            "hurdle", "buffer", "bounded_mlp_physical_alpha"
        )
        physical_params = _candidate_parameters(
            physical,
            input_dim=input_dim,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(102),
        )
        calm = _bounded_mlp_alpha(
            physical_params,
            physical,
            row.at[VENT_INDEX].set(0.0),
            jnp.asarray(12.0),
            input_mean=mean,
            input_std=std,
            dt_hours=0.25,
        )
        ventilated = _bounded_mlp_alpha(
            physical_params,
            physical,
            row.at[VENT_INDEX].set(2.0),
            jnp.asarray(24.0),
            input_mean=mean,
            input_std=std,
            dt_hours=0.25,
        )
        self.assertGreater(float(ventilated), float(calm))
        self.assertGreater(float(calm), 0.0)
        self.assertLess(float(ventilated), 1.0)

    def test_positive_q_thermal_branch_is_monotone(self) -> None:
        modes = jnp.asarray([5.0, 7.0, 9.0])
        kwargs = {
            "gain_raw": jnp.asarray(-1.0),
            "weight_raw": jnp.asarray([0.2, -0.4, 0.1, 0.7]),
            "dt_hours": 0.25,
        }
        low, low_modes = _positive_thermal_q_forcing(
            jnp.asarray(2.0), modes, **kwargs
        )
        high, high_modes = _positive_thermal_q_forcing(
            jnp.asarray(12.0), modes, **kwargs
        )
        self.assertGreater(float(high), float(low))
        self.assertTrue(np.all(np.asarray(high_modes) > np.asarray(low_modes)))
        self.assertGreaterEqual(float(low), 0.0)

    def test_passive_memory_temperature_step_is_strictly_monotone_in_q(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        architecture = Architecture(
            "hurdle", "buffer", "passive_memory_positive_q"
        )
        params = _candidate_parameters(
            architecture,
            input_dim=input_dim,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(103),
        )
        row = jnp.zeros((input_dim,)).at[1].set(5.0)
        memory = jnp.zeros((4, 7))
        kwargs = {
            "input_mean": jnp.zeros((input_dim,)),
            "input_std": jnp.ones((input_dim,)),
            "target_t_mean": jnp.asarray(20.0),
            "target_t_std": jnp.asarray(2.0),
            "dt_hours": 0.25,
            "temperature_delta_cap": 2.0,
        }
        low, low_memory, low_gain = _passive_memory_positive_q_step(
            params,
            row,
            jnp.asarray(20.0),
            jnp.asarray(0.0),
            memory,
            **kwargs,
        )
        high, high_memory, high_gain = _passive_memory_positive_q_step(
            params,
            row,
            jnp.asarray(20.0),
            jnp.asarray(20.0),
            memory,
            **kwargs,
        )
        self.assertGreater(float(high), float(low))
        self.assertGreater(float(low_gain), 0.0)
        self.assertAlmostEqual(float(low_gain), float(high_gain), places=7)
        np.testing.assert_allclose(
            np.asarray(low_memory), np.asarray(high_memory), atol=1e-7
        )

    def test_observed_q_thermal_rollout_is_autonomous_and_finite(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        architecture = Architecture(
            "hurdle", "buffer", "passive_memory_positive_q"
        )
        params = _candidate_parameters(
            architecture,
            input_dim=input_dim,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(104),
        )
        inputs = jnp.zeros((12, input_dim)).at[:, 1].set(5.0)
        temperatures = _observed_q_thermal_rollout(
            params,
            architecture,
            inputs,
            jnp.linspace(0.0, 20.0, 12),
            jnp.asarray([20.0]),
            input_mean=jnp.zeros((input_dim,)),
            input_std=jnp.ones((input_dim,)),
            target_t_mean=jnp.asarray(20.0),
            target_t_std=jnp.asarray(2.0),
            q_cap=80.0,
            dt_hours=0.25,
            temperature_delta_cap=2.0,
        )
        values = np.asarray(temperatures)
        self.assertEqual(values.shape, (12,))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all((values >= 5.0) & (values <= 40.0)))

    def test_observed_q_thermal_rollout_supports_legacy_bounded_mlp(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        architecture = Architecture("hurdle", "buffer", "bounded_mlp")
        params = _candidate_parameters(
            architecture,
            input_dim=input_dim,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(105),
        )
        inputs = jnp.zeros((12, input_dim)).at[:, 1].set(5.0)
        temperatures = _observed_q_thermal_rollout(
            params,
            architecture,
            inputs,
            jnp.linspace(0.0, 20.0, 12),
            jnp.asarray([20.0]),
            input_mean=jnp.zeros((input_dim,)),
            input_std=jnp.ones((input_dim,)),
            target_t_mean=jnp.asarray(20.0),
            target_t_std=jnp.asarray(2.0),
            q_cap=80.0,
            dt_hours=0.25,
            temperature_delta_cap=2.0,
        )
        values = np.asarray(temperatures)
        self.assertEqual(values.shape, (12,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_event_aligned_tin_loss_detects_missed_force_off_drop(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        inputs = np.zeros((1, 9, input_dim), dtype=np.float32)
        inputs[0, 4:, FORCE_OFF_INDEX] = 1.0
        targets = np.zeros((1, 9, 3), dtype=np.float32)
        targets[..., 0] = np.asarray(
            [20.0, 20.0, 20.0, 20.0, 19.5, 19.0, 18.5, 18.0, 18.0]
        )
        missed = targets.copy()
        missed[..., 0] = 20.0
        kwargs = {
            "temperature_scale": jnp.asarray(1.0),
            "horizon_steps": 2,
            "setpoint_threshold_c": 0.05,
        }
        perfect_loss = _event_aligned_temperature_response_loss(
            jnp.asarray(targets),
            jnp.asarray(targets),
            jnp.asarray(inputs),
            **kwargs,
        )
        missed_loss = _event_aligned_temperature_response_loss(
            jnp.asarray(missed),
            jnp.asarray(targets),
            jnp.asarray(inputs),
            **kwargs,
        )
        self.assertAlmostEqual(float(perfect_loss), 0.0, places=7)
        self.assertGreater(float(missed_loss), 0.1)

    def test_recovery_hurdle_cannot_ignore_a_large_temperature_deficit(self) -> None:
        input_dim = len(closed_loop_input_columns(include_hp_force_off=True))
        inputs = np.zeros((4, input_dim), dtype=np.float32)
        inputs[:, 0] = 35.0
        inputs[:, 1] = 5.0
        inputs[:, AVAILABILITY_INDEX] = 1.0
        architecture = Architecture(
            "recovery_hurdle",
            "positive_convolution",
            "stable_rc_narx",
        )
        params = _candidate_parameters(
            architecture,
            input_dim=input_dim,
            hidden_dim=8,
            depth=1,
            key=jax.random.PRNGKey(91),
        )
        prediction, modes = _rollout(
            params,
            architecture,
            jnp.asarray(inputs),
            jnp.asarray([20.0]),
            input_mean=jnp.zeros((input_dim,)),
            input_std=jnp.ones((input_dim,)),
            target_mean=jnp.asarray([20.0, 0.0, 0.0]),
            target_std=jnp.ones((3,)),
            pel_cap=20.0,
            q_cap=80.0,
            energy_cap=1000.0,
            hidden_dim=8,
            dt_hours=0.25,
            temperature_delta_cap=2.0,
        )
        self.assertGreater(float(modes[0]), 0.99)
        self.assertGreater(float(prediction[0, 2]), 0.01 * 20.0)

    def test_annual_ensemble_metrics_expose_restart_spread(self) -> None:
        target = np.zeros((8, 3), dtype=np.float32)
        target[:, 0] = 20.0
        predictions = np.stack((target - 1.0, target + 1.0))
        datetime = np.asarray(
            [f"2023-01-01T0{index}:00:00" for index in range(8)],
            dtype="datetime64[s]",
        )
        metrics = _annual_ensemble_metrics(
            predictions,
            target,
            datetime,
            Scale(mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32)),
        )
        self.assertAlmostEqual(metrics["annual_force_off_ensemble_temperature_rmse_c"], 0.0)
        self.assertGreater(
            metrics["annual_force_off_winter_temperature_restart_band_mean_width_c"],
            1.0,
        )
        self.assertAlmostEqual(
            metrics["annual_force_off_temperature_restart_coverage95"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
