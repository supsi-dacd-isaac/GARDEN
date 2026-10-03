"""Pinpoint non-finite / exploding gradients in probabilistic closed-loop training.

The current 1000-profile leaky-equilibrium run keeps a finite loss while the
candidate-update validator starts skipping batches with ``reason=grads`` from
epoch 3 onward. This module isolates each scoring term, then (optionally)
replays the saved epoch-2/3 checkpoints on synthetic and real windows.

Run:

    .venv/bin/python -m neural_building_emulator.test_prob_closed_loop_grad_stability
    .venv/bin/python -m neural_building_emulator.test_prob_closed_loop_grad_stability --skip-checkpoint
    .venv/bin/python -m unittest neural_building_emulator.test_prob_closed_loop_grad_stability
"""

from __future__ import annotations

import argparse
import math
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.columns import (
    PROFILE_ID_COLUMN,
    SPACE_HEATING_AVAILABILITY_COLUMN,
    closed_loop_required_columns,
)
from neural_building_emulator.data import (
    ClosedLoopWindowedArrays,
    WindowConfig,
    _read_parquet,
    make_closed_loop_windows,
    to_closed_loop_profiles,
)
from neural_building_emulator.model_io import load_training_artifact
from neural_building_emulator.models import ProbabilisticContractingClosedLoopHPEmulator
from neural_building_emulator.models.truncated_bptt import detach_carry, scale_gradient
from neural_building_emulator.scaling import transform_closed_loop_windows
from neural_building_emulator.train import (
    PROB_CLOSED_LOOP_AUX_ENERGY,
    PROB_CLOSED_LOOP_AUX_HP_LATENT,
    PROB_CLOSED_LOOP_AUX_HP_MODE_HISTORY,
    PROB_CLOSED_LOOP_AUX_HP_POWER_HISTORY,
    PROB_CLOSED_LOOP_AUX_HP_PREVIOUS_MODE,
    PROB_CLOSED_LOOP_AUX_LOG_MU,
    PROB_CLOSED_LOOP_AUX_LOG_SIGMA,
    PROB_CLOSED_LOOP_AUX_QROOM_ON,
    PROB_CLOSED_LOOP_AUX_QROOM_PI,
    PROB_CLOSED_LOOP_AUX_QROOM_PREVIOUS_MODE,
    PROB_CLOSED_LOOP_AUX_QROOM_P_START,
    PROB_CLOSED_LOOP_AUX_QROOM_P_STOP,
    PROB_CLOSED_LOOP_AUX_TEMPERATURE_STATE,
    PROB_CLOSED_LOOP_AUX_W_STATE,
    PROB_CLOSED_LOOP_AUX_X_STATE,
    PROB_CLOSED_LOOP_AUX_XI,
    _array_tree_all_finite,
    _array_tree_global_norm,
    _array_tree_max_abs,
    _physical_target_component,
    energy_score,
    intervention_response_energy_score,
    predict_probabilistic_closed_loop_batch_with_aux,
    smooth_abs_power,
    soft_optimistic_loss,
    variogram_score,
)

jax.config.update("jax_default_matmul_precision", "highest")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = (
    REPO_ROOT
    / "output"
    / "neural_building_emulator"
    / "model_checkpoints"
    / "closed_loop_hp_contracting_probabilistic"
)
DEFAULT_DATASET = REPO_ROOT / "neural_building_emulator" / "tessin_results.parquet" / "all_hp"
MODULE_NAMES = (
    "input_encoder",
    "transition_net",
    "x0_net",
    "e0_net",
    "hp_param_net",
    "output_net",
    "qroom_transition_net",
    "temperature_net",
    "temperature_alpha_net",
    "process_noise_net",
)
COMPONENT_ORDER = ("energy", "variogram", "ires", "hp_bce", "hp_active_nll")
WORST_REAL_WINDOWS = (
    (698682, 22264),
    (11105295, 12144),
    (11105295, 28336),
    (706008, 10120),
    (744162, 12144),
)


@dataclass(frozen=True)
class ComponentWeights:
    energy: float = 0.0
    variogram: float = 0.0
    ires: float = 0.0
    hp_bce: float = 0.0
    hp_active_nll: float = 0.0
    variogram_lags: tuple[int, ...] = (1, 2, 4, 8)
    variogram_power: float = 0.5
    variogram_output_weights: tuple[float, ...] = (1.0, 0.0, 0.0)
    ires_horizon_steps: tuple[int, ...] = (2, 4, 8, 12)
    ires_horizon_hours: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    heat_on_threshold: float = 1e-6
    hp_mode_loss_weight: float = 0.1
    hp_active_power_nll_weight: float = 1.0


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _scalar(value: jnp.ndarray | float) -> float:
    return float(np.asarray(value))


def hp_active_nll(log_target: jnp.ndarray, log_mu: jnp.ndarray, log_sigma: jnp.ndarray) -> jnp.ndarray:
    log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype=log_target.dtype)
    return 0.5 * ((log_target - log_mu) / log_sigma) ** 2 + jnp.log(log_sigma) + 0.5 * log_two_pi


def scoring_grad_norm(fn: Callable[[jnp.ndarray], jnp.ndarray], values: jnp.ndarray) -> float:
    grad = jax.grad(lambda x: jnp.mean(fn(x)))(values)
    return _scalar(jnp.sqrt(jnp.sum(grad.astype(jnp.float32) ** 2) + 1e-12))


def make_tiny_model(
    key: jax.Array,
    *,
    horizon_safe: bool = True,
    bptt_truncate_steps: int = 0,
    hp_activation_model: str = "independent",
    qroom_activation_model: str = "continuous",
    hp_persistent_latent_dim: int = 2,
    hp_controller_noise_scale: float = 0.0,
    hp_history_hours: float = 3.0,
    hp_history_timescales_hours: tuple[float, ...] = (),
    hp_setpoint_shock_timescales_hours: tuple[float, ...] = (),
    temperature_feedback_gradient_scale: float = 1.0,
    hydronic_innovation_enabled: bool = False,
    hydronic_innovation_cap_w_m2: float = 20.0,
) -> ProbabilisticContractingClosedLoopHPEmulator:
    del horizon_safe
    return ProbabilisticContractingClosedLoopHPEmulator(
        metadata_dim=4,
        input_dim=8,
        state_dim=4,
        controller_state_dim=2,
        latent_dim=2,
        hidden_dim=16,
        depth=2,
        input_encoder_dim=4,
        input_encoder_hidden_dim=16,
        input_encoder_depth=1,
        process_noise_mode="heteroscedastic",
        process_noise_init=-6.0,
        hp_emission_mode="bounded",
        hp_activation_model=hp_activation_model,
        qroom_activation_model=qroom_activation_model,
        hp_persistent_latent_dim=hp_persistent_latent_dim,
        hp_controller_noise_scale=hp_controller_noise_scale,
        hp_history_hours=hp_history_hours,
        hp_history_timescales_hours=hp_history_timescales_hours,
        hp_setpoint_shock_timescales_hours=hp_setpoint_shock_timescales_hours,
        contraction_gamma=0.99,
        state_bound=5.0,
        temperature_output_scale=8.0,
        temperature_delta_max_c=2.0,
        temperature_update_mode="leaky_equilibrium",
        temperature_feedback_gradient_scale=temperature_feedback_gradient_scale,
        hp_dt_hours=0.25,
        hp_cop_floor=1.0,
        hp_cop_cap=8.0,
        hp_pel_cap_w_m2=80.0,
        hp_qroom_cap_w_m2=120.0,
        hp_energy_cap_wh_m2=400.0,
        input_mean=(20.0, 5.0, 50.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        input_scale=(2.0, 8.0, 80.0, 0.2, 1.0, 1.0, 1.0, 1.0),
        target_mean=(20.0, 8.0, 6.0),
        target_scale=(2.0, 10.0, 8.0),
        thermostat_demand_mode=(
            "monotone"
            if qroom_activation_model == "asymmetric_markov"
            else "unconstrained"
        ),
        hydronic_innovation_enabled=hydronic_innovation_enabled,
        hydronic_innovation_cap_w_m2=hydronic_innovation_cap_w_m2,
        bptt_truncate_steps=bptt_truncate_steps,
        key=key,
    )


class PersistentHPControllerTests(unittest.TestCase):
    def test_hydronic_innovation_is_bounded_and_cannot_bypass_empty_buffer(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(126),
            hydronic_innovation_enabled=True,
            hydronic_innovation_cap_w_m2=12.0,
        )
        metadata = jnp.zeros((model.metadata_dim,), dtype=jnp.float32)
        input_t = jnp.zeros((model.input_dim,), dtype=jnp.float32)
        encoded_input_t = model.encode_input(input_t)
        outputs = model.decode_hp(
            metadata,
            jnp.zeros((model.latent_dim,), dtype=jnp.float32),
            encoded_input_t,
            input_t,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros((model.controller_state_dim,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((model.controller_state_dim,), dtype=jnp.float32),
            hp_scenario_mode="expected",
            hp_concrete_temperature=jnp.asarray(0.5, dtype=jnp.float32),
            hp_generation_available_t=jnp.asarray(0.0, dtype=jnp.float32),
            hydronic_innovation_t=jnp.asarray(1_000.0, dtype=jnp.float32),
        )
        qroom_t, energy_next = outputs[2], outputs[3]
        self.assertGreater(float(model.hydronic_innovation_scale(jnp.float32)), 0.0)
        self.assertLessEqual(
            float(model.hydronic_innovation_scale(jnp.float32)),
            model.hydronic_innovation_cap_w_m2,
        )
        np.testing.assert_allclose(np.asarray(qroom_t), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(energy_next), 0.0, atol=1e-6)

        predictions, _ = model.sample_with_aux(
            metadata,
            jnp.zeros((6, model.input_dim), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            key=jax.random.PRNGKey(127),
            num_particles=3,
            sample_process_noise=False,
            hp_scenario_mode="expected",
        )
        qroom_physical = predictions[..., 1] * model.target_scale[1] + model.target_mean[1]
        self.assertTrue(np.isfinite(np.asarray(predictions)).all())
        self.assertTrue(np.all(np.asarray(qroom_physical) >= 0.0))
        self.assertTrue(np.all(np.asarray(qroom_physical) <= model.hp_qroom_cap_w_m2))

        stability_state = model.pack_stability_state(
            jnp.zeros((model.state_dim,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros((model.controller_state_dim,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            hydronic_delivery_state_t=jnp.asarray(3.0, dtype=jnp.float32),
        )
        self.assertEqual(stability_state.shape, (model.stability_state_dim(),))
        stability_next = model.one_step_augmented_state(
            metadata,
            input_t,
            jnp.zeros((model.latent_dim,), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            stability_state,
            jnp.zeros((model.state_dim,), dtype=jnp.float32),
            jnp.zeros((model.controller_state_dim,), dtype=jnp.float32),
        )
        self.assertTrue(np.isfinite(np.asarray(stability_next)).all())

    def test_setpoint_shock_features_preserve_direction_and_decay(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(127),
            hp_setpoint_shock_timescales_hours=(0.25, 1.0),
        )
        inputs = jnp.zeros((6, model.input_dim))
        inputs = inputs.at[:, 0].set(jnp.asarray([0.0, 0.0, 2.0, 2.0, 1.0, 1.0]))
        traces = np.asarray(model.setpoint_shock_features(inputs))

        self.assertEqual(traces.shape, (6, 4))
        np.testing.assert_allclose(traces[:2], 0.0, atol=1e-7)
        np.testing.assert_allclose(traces[2, :2], [2.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(traces[2, 2:], 0.0, atol=1e-7)
        np.testing.assert_allclose(
            traces[3, :2],
            2.0 * np.exp(-0.25 / np.asarray([0.25, 1.0])),
            rtol=1e-6,
        )
        np.testing.assert_allclose(traces[4, 2:], [1.0, 1.0], atol=1e-7)

    def test_asymmetric_markov_accepts_setpoint_shock_features(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(128),
            hp_activation_model="asymmetric_markov",
            hp_setpoint_shock_timescales_hours=(0.5, 1.0, 2.0, 3.0),
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=24,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(129),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        predictions, _ = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(130),
            num_particles=3,
            hp_scenario_mode="expected",
        )
        self.assertEqual(predictions.shape, (3, 24, 3))
        self.assertTrue(np.isfinite(np.asarray(predictions)).all())

    def test_asymmetric_markov_uses_expected_two_state_recursion(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(130),
            hp_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=24,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(131),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        predictions, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(132),
            num_particles=4,
            hp_scenario_mode="expected",
        )
        pi = aux[0]
        p_start = aux[14]
        p_stop = aux[15]
        previous_mode = aux[PROB_CLOSED_LOOP_AUX_HP_PREVIOUS_MODE]
        availability = jax.vmap(model._space_heating_availability)(inputs[0])
        expected_pi = availability[jnp.newaxis, :] * (
            (1.0 - previous_mode) * p_start
            + previous_mode * (1.0 - p_stop)
        )
        np.testing.assert_allclose(
            np.asarray(pi),
            np.asarray(expected_pi),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(predictions.shape, (4, 24, 3))
        self.assertEqual(aux[PROB_CLOSED_LOOP_AUX_HP_LATENT].shape, (4, 0))
        self.assertTrue(np.isfinite(np.asarray(predictions)).all())

    def test_asymmetric_markov_bernoulli_rollout_has_hard_modes(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(133),
            hp_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=32,
            metadata_dim=4,
            input_dim=8,
            kind="smooth",
            key=jax.random.PRNGKey(134),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        _, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(135),
            num_particles=8,
            hp_scenario_mode="bernoulli",
        )
        modes = np.asarray(aux[5])
        self.assertTrue(np.all((modes == 0.0) | (modes == 1.0)))
        self.assertTrue(
            np.isfinite(np.asarray(aux[PROB_CLOSED_LOOP_AUX_HP_POWER_HISTORY])).all()
        )
        self.assertTrue(
            np.isfinite(np.asarray(aux[PROB_CLOSED_LOOP_AUX_HP_MODE_HISTORY])).all()
        )

    def test_expected_rollout_obeys_start_stop_recursion(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(30),
            hp_activation_model="persistent_markov",
            hp_persistent_latent_dim=3,
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=24,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(31),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        predictions, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(32),
            num_particles=4,
            hp_scenario_mode="expected",
        )
        pi = aux[0]
        hp_on = aux[5]
        p_start = aux[14]
        p_stop = aux[15]
        expected_pi = (
            (1.0 - hp_on[:, :-1]) * p_start[:, 1:]
            + hp_on[:, :-1] * (1.0 - p_stop[:, 1:])
        )
        np.testing.assert_allclose(np.asarray(pi[:, 1:]), np.asarray(expected_pi), rtol=1e-5, atol=1e-5)
        self.assertEqual(predictions.shape, (4, 24, 3))
        self.assertEqual(aux[16].shape, (4, 3))
        self.assertLessEqual(float(jnp.max(jnp.abs(aux[11]))), 1.0 + 1e-6)

    def test_bernoulli_rollout_has_hard_persistent_modes(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(33),
            hp_activation_model="persistent_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=32,
            metadata_dim=4,
            input_dim=8,
            kind="smooth",
            key=jax.random.PRNGKey(34),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        _, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(35),
            num_particles=8,
            hp_scenario_mode="bernoulli",
        )
        modes = np.asarray(aux[5])
        self.assertTrue(np.all((modes == 0.0) | (modes == 1.0)))
        self.assertGreater(float(np.std(np.asarray(aux[14]))), 0.0)

    def test_straight_through_controller_gradients_are_finite(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(36),
            hp_activation_model="persistent_markov",
            hp_controller_noise_scale=0.05,
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=16,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(37),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )

        def objective(candidate):
            predictions, aux = candidate.sample_with_aux(
                metadata[0],
                inputs[0],
                initial_temperature[0],
                key=jax.random.PRNGKey(38),
                num_particles=3,
                hp_scenario_mode="straight_through",
                hp_concrete_temperature=jnp.asarray(0.5),
            )
            return jnp.mean(predictions**2) + 0.01 * (
                jnp.mean(aux[14]) + jnp.mean(aux[15])
            )

        loss, grads = eqx.filter_value_and_grad(objective)(model)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(_array_tree_all_finite(grads)))

    def test_hard_expected_gradient_controller_is_binary_with_finite_gradients(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(360),
            hp_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=16,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(370),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )

        def objective(candidate):
            predictions, aux = candidate.sample_with_aux(
                metadata[0],
                inputs[0],
                initial_temperature[0],
                key=jax.random.PRNGKey(380),
                num_particles=3,
                hp_scenario_mode="hard_expected_grad",
            )
            return jnp.mean(predictions**2), aux[5]

        (loss, modes), grads = eqx.filter_value_and_grad(
            objective,
            has_aux=True,
        )(model)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(_array_tree_all_finite(grads)))
        self.assertTrue(bool(jnp.all((modes == 0.0) | (modes == 1.0))))


class HydronicDemandMarkovTests(unittest.TestCase):
    def test_expected_rollout_obeys_hydronic_start_stop_recursion(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(160),
            qroom_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=24,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(161),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        predictions, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(162),
            num_particles=4,
            hp_scenario_mode="expected",
        )
        availability = jax.vmap(model._space_heating_availability)(inputs[0])
        expected_pi = availability[jnp.newaxis, :] * (
            (1.0 - aux[PROB_CLOSED_LOOP_AUX_QROOM_PREVIOUS_MODE])
            * aux[PROB_CLOSED_LOOP_AUX_QROOM_P_START]
            + aux[PROB_CLOSED_LOOP_AUX_QROOM_PREVIOUS_MODE]
            * (1.0 - aux[PROB_CLOSED_LOOP_AUX_QROOM_P_STOP])
        )
        np.testing.assert_allclose(
            np.asarray(aux[PROB_CLOSED_LOOP_AUX_QROOM_PI]),
            np.asarray(expected_pi),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(aux[PROB_CLOSED_LOOP_AUX_QROOM_ON]),
            np.asarray(expected_pi),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(predictions.shape, (4, 24, 3))
        self.assertEqual(len(aux), 26)

    def test_bernoulli_rollout_samples_hard_hydronic_modes(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(163),
            qroom_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=32,
            metadata_dim=4,
            input_dim=8,
            kind="smooth",
            key=jax.random.PRNGKey(164),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        _, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(165),
            num_particles=8,
            hp_scenario_mode="bernoulli",
        )
        modes = np.asarray(aux[PROB_CLOSED_LOOP_AUX_QROOM_ON])
        self.assertTrue(np.all((modes == 0.0) | (modes == 1.0)))

    def test_transition_hazards_are_monotone_in_thermostat_gap(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(166),
            qroom_activation_model="asymmetric_markov",
        )
        assert model.qroom_transition_net is not None
        demand_features = jnp.zeros(
            (model.qroom_transition_net.layers[0].weight.shape[1],),
            dtype=jnp.float32,
        )
        low_input = jnp.zeros((model.input_dim,), dtype=jnp.float32).at[0].set(-1.0)
        high_input = low_input.at[0].set(1.0)
        temperature = jnp.asarray(0.0, dtype=jnp.float32)
        low_start, low_stop = model.qroom_transition_probabilities(
            demand_features, low_input, temperature
        )
        high_start, high_stop = model.qroom_transition_probabilities(
            demand_features, high_input, temperature
        )
        self.assertGreater(float(high_start), float(low_start))
        self.assertLess(float(high_stop), float(low_stop))

    def test_expected_hydronic_state_has_finite_nonzero_transition_gradients(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(167),
            qroom_activation_model="asymmetric_markov",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=16,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(168),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )

        def objective(candidate):
            _, aux = candidate.sample_with_aux(
                metadata[0],
                inputs[0],
                initial_temperature[0],
                key=jax.random.PRNGKey(169),
                num_particles=3,
                hp_scenario_mode="expected",
            )
            return jnp.mean(aux[PROB_CLOSED_LOOP_AUX_QROOM_PI])

        loss, grads = eqx.filter_value_and_grad(objective)(model)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(_array_tree_all_finite(grads.qroom_transition_net)))
        self.assertGreater(
            float(_array_tree_global_norm(grads.qroom_transition_net)),
            0.0,
        )


class PowerHistoryHPTests(unittest.TestCase):
    def test_history_is_causal_ewma_of_generated_hp_emissions(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(40),
            hp_activation_model="power_history",
            hp_history_hours=3.0,
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=32,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(41),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        predictions, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(42),
            num_particles=4,
            hp_scenario_mode="expected",
        )

        hp_on = aux[5]
        pel = aux[7]
        power_history = aux[17]
        mode_history = aux[18]
        alpha = model.hp_history_alpha(pel.dtype)
        power_fraction = jnp.clip(pel / model._max_active_power(pel.dtype), 0.0, 1.0)
        expected_power_history = (
            (1.0 - alpha) * power_history[:, :-1]
            + alpha * power_fraction[:, :-1]
        )
        expected_mode_history = (
            (1.0 - alpha) * mode_history[:, :-1]
            + alpha * hp_on[:, :-1]
        )

        self.assertEqual(predictions.shape, (4, 32, 3))
        np.testing.assert_allclose(np.asarray(power_history[:, 0]), 0.0, atol=0.0)
        np.testing.assert_allclose(np.asarray(mode_history[:, 0]), 0.0, atol=0.0)
        np.testing.assert_allclose(
            np.asarray(power_history[:, 1:]),
            np.asarray(expected_power_history),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(mode_history[:, 1:]),
            np.asarray(expected_mode_history),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(bool(jnp.all((power_history >= 0.0) & (power_history <= 1.0))))
        self.assertTrue(bool(jnp.all((mode_history >= 0.0) & (mode_history <= 1.0))))

    def test_teacher_forcing_history_uses_observed_hp_emissions(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(46),
            hp_activation_model="power_history",
            hp_history_hours=3.0,
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=20,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(47),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        observed_pel = jnp.linspace(0.0, 40.0, inputs.shape[1], dtype=inputs.dtype)
        observed_mode = (observed_pel > 10.0).astype(inputs.dtype)
        _, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(48),
            num_particles=3,
            hp_scenario_mode="expected",
            hp_feedback_pel=observed_pel,
            hp_feedback_mode=observed_mode,
            hp_teacher_forcing_ratio=1.0,
        )

        power_history = aux[17]
        mode_history = aux[18]
        alpha = model.hp_history_alpha(inputs.dtype)
        observed_fraction = jnp.clip(
            observed_pel / model._max_active_power(inputs.dtype),
            0.0,
            1.0,
        )
        expected_power = np.zeros((inputs.shape[1],), dtype=np.float32)
        expected_mode = np.zeros((inputs.shape[1],), dtype=np.float32)
        for step in range(1, inputs.shape[1]):
            expected_power[step] = (
                (1.0 - float(alpha)) * expected_power[step - 1]
                + float(alpha) * float(observed_fraction[step - 1])
            )
            expected_mode[step] = (
                (1.0 - float(alpha)) * expected_mode[step - 1]
                + float(alpha) * float(observed_mode[step - 1])
            )

        np.testing.assert_allclose(
            np.asarray(power_history),
            np.broadcast_to(expected_power, power_history.shape),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(mode_history),
            np.broadcast_to(expected_mode, mode_history.shape),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_multiscale_history_contains_immediate_lag_and_all_ewmas(self) -> None:
        timescales = (0.5, 1.0, 3.0, 6.0)
        model = make_tiny_model(
            jax.random.PRNGKey(49),
            hp_activation_model="power_history",
            hp_history_timescales_hours=timescales,
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=20,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(50),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        observed_pel = jnp.linspace(0.0, 40.0, inputs.shape[1], dtype=inputs.dtype)
        observed_mode = (observed_pel > 10.0).astype(inputs.dtype)
        _, aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(51),
            num_particles=2,
            hp_scenario_mode="expected",
            hp_feedback_pel=observed_pel,
            hp_feedback_mode=observed_mode,
            hp_teacher_forcing_ratio=1.0,
        )

        observed_fraction = np.asarray(
            jnp.clip(
                observed_pel / model._max_active_power(inputs.dtype),
                0.0,
                1.0,
            )
        )
        mode_fraction = np.asarray(observed_mode)
        alphas = -np.expm1(-model.hp_dt_hours / np.asarray(timescales))
        expected_power = np.zeros((inputs.shape[1], 1 + len(timescales)), dtype=np.float32)
        expected_mode = np.zeros_like(expected_power)
        for step in range(1, inputs.shape[1]):
            expected_power[step, 0] = observed_fraction[step - 1]
            expected_mode[step, 0] = mode_fraction[step - 1]
            expected_power[step, 1:] = (
                (1.0 - alphas) * expected_power[step - 1, 1:]
                + alphas * observed_fraction[step - 1]
            )
            expected_mode[step, 1:] = (
                (1.0 - alphas) * expected_mode[step - 1, 1:]
                + alphas * mode_fraction[step - 1]
            )

        power_history = np.asarray(aux[17])
        mode_history = np.asarray(aux[18])
        self.assertEqual(power_history.shape, (2, inputs.shape[1], 5))
        np.testing.assert_allclose(
            power_history,
            np.broadcast_to(expected_power, power_history.shape),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            mode_history,
            np.broadcast_to(expected_mode, mode_history.shape),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_temperature_feedback_changes_only_the_opt_in_hp_rollout_path(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(52),
            hp_activation_model="power_history",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=20,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(53),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        observed_temperature = jnp.linspace(-2.0, 2.0, inputs.shape[1], dtype=inputs.dtype)
        base_predictions, base_aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(54),
            num_particles=2,
            hp_scenario_mode="expected",
        )
        zero_predictions, zero_aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(54),
            num_particles=2,
            hp_scenario_mode="expected",
            hp_feedback_temperature=observed_temperature,
            hp_temperature_teacher_forcing_ratio=0.0,
        )
        forced_predictions, forced_aux = model.sample_with_aux(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=jax.random.PRNGKey(54),
            num_particles=2,
            hp_scenario_mode="expected",
            hp_feedback_temperature=observed_temperature,
            hp_temperature_teacher_forcing_ratio=1.0,
        )

        np.testing.assert_allclose(base_predictions, zero_predictions, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(base_aux[0], zero_aux[0], rtol=0.0, atol=0.0)
        self.assertGreater(
            float(jnp.max(jnp.abs(forced_aux[0] - base_aux[0]))),
            1e-6,
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(forced_predictions - base_predictions))),
            1e-6,
        )

    def test_history_enters_hp_head_and_straight_through_gradients_are_finite(self) -> None:
        model = make_tiny_model(
            jax.random.PRNGKey(43),
            hp_activation_model="power_history",
        )
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=16,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(44),
            target_mean=model.target_mean,
            target_scale=model.target_scale,
        )
        encoded_input = model.encode_input(inputs[0, 0])

        def mode_logit(history: jnp.ndarray) -> jnp.ndarray:
            features = model.hp_features(
                metadata[0],
                jnp.zeros((model.latent_dim,), dtype=inputs.dtype),
                encoded_input,
                inputs[0, 0],
                jnp.asarray(0.0, dtype=inputs.dtype),
                initial_temperature[0, 0],
                hp_power_history_t=history[0],
                hp_mode_history_t=history[1],
            )
            return model.output_net(features)[1]

        history_grad = jax.grad(mode_logit)(jnp.asarray([0.2, 0.3], dtype=inputs.dtype))
        self.assertTrue(bool(jnp.all(jnp.isfinite(history_grad))))
        self.assertGreater(float(jnp.linalg.norm(history_grad)), 1e-8)

        def objective(candidate):
            predictions, aux = candidate.sample_with_aux(
                metadata[0],
                inputs[0],
                initial_temperature[0],
                key=jax.random.PRNGKey(45),
                num_particles=3,
                hp_scenario_mode="straight_through",
                hp_concrete_temperature=jnp.asarray(0.5),
            )
            return jnp.mean(predictions**2) + 0.01 * (
                jnp.mean(aux[17]) + jnp.mean(aux[18])
            )

        loss, grads = eqx.filter_value_and_grad(objective)(model)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(_array_tree_all_finite(grads)))


class TruncatedBPTTTests(unittest.TestCase):
    def test_scale_gradient_preserves_value_and_scales_derivative(self) -> None:
        value = jnp.asarray(2.5)
        for scale in (0.0, 0.05, 0.1, 1.0):
            actual = scale_gradient(value, scale)
            derivative = jax.grad(lambda x: scale_gradient(x, scale))(value)
            self.assertAlmostEqual(float(actual), float(value), places=7)
            self.assertAlmostEqual(float(derivative), scale, places=7)

    def test_feedback_gradient_scale_preserves_probabilistic_forward_rollout(self) -> None:
        key = jax.random.PRNGKey(90)
        full = make_tiny_model(key, temperature_feedback_gradient_scale=1.0)
        damped = make_tiny_model(key, temperature_feedback_gradient_scale=0.05)
        metadata, inputs, initial_temperature, _ = synthetic_batch(
            batch_size=1,
            horizon=16,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(91),
            target_mean=full.target_mean,
            target_scale=full.target_scale,
        )
        sample_key = jax.random.PRNGKey(92)
        full_prediction = full.sample(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=sample_key,
            num_particles=3,
            sample_process_noise=True,
            hp_scenario_mode="expected",
        )
        damped_prediction = damped.sample(
            metadata[0],
            inputs[0],
            initial_temperature[0],
            key=sample_key,
            num_particles=3,
            sample_process_noise=True,
            hp_scenario_mode="expected",
        )
        np.testing.assert_array_equal(
            np.asarray(full_prediction),
            np.asarray(damped_prediction),
        )

    def test_detach_carry_zeros_gradient_across_chunk(self) -> None:
        def rollout(x0: jnp.ndarray, truncate_steps: int) -> jnp.ndarray:
            def step(carry, time_index):
                nxt = 1.5 * carry
                nxt = detach_carry(nxt, time_index, truncate_steps)
                return nxt, nxt

            _, ys = jax.lax.scan(step, x0, jnp.arange(6))
            return ys[-1]

        x0 = jnp.asarray(1.0)
        full = float(jax.grad(lambda x: rollout(x, 0))(x0))
        truncated = float(jax.grad(lambda x: rollout(x, 2))(x0))
        self.assertAlmostEqual(full, 1.5 ** 6, places=4)
        self.assertAlmostEqual(truncated, 0.0, places=5)
        self.assertAlmostEqual(float(rollout(x0, 0)), float(rollout(x0, 2)), places=5)

    def test_tiny_model_forward_matches_full_bptt(self) -> None:
        key = jax.random.PRNGKey(0)
        full = make_tiny_model(key, bptt_truncate_steps=0)
        truncated = make_tiny_model(key, bptt_truncate_steps=32)
        metadata, inputs, initial_temperature, _targets = synthetic_batch(
            batch_size=2,
            horizon=64,
            metadata_dim=4,
            input_dim=8,
            kind="setpoint_jump",
            key=jax.random.PRNGKey(1),
            target_mean=full.target_mean,
            target_scale=full.target_scale,
        )
        predict_key = jax.random.PRNGKey(2)
        y_full, _ = predict_probabilistic_closed_loop_batch_with_aux(
            full,
            metadata,
            inputs,
            initial_temperature,
            key=predict_key,
            num_particles=4,
            sample_process_noise=True,
            hp_scenario_mode="expected",
        )
        y_truncated, _ = predict_probabilistic_closed_loop_batch_with_aux(
            truncated,
            metadata,
            inputs,
            initial_temperature,
            key=predict_key,
            num_particles=4,
            sample_process_noise=True,
            hp_scenario_mode="expected",
        )
        np.testing.assert_allclose(np.asarray(y_full), np.asarray(y_truncated), rtol=1e-5, atol=1e-5)


def synthetic_batch(
    *,
    batch_size: int,
    horizon: int,
    metadata_dim: int,
    input_dim: int,
    kind: str,
    key: jax.Array,
    target_mean: tuple[float, ...] | np.ndarray,
    target_scale: tuple[float, ...] | np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = np.asarray(target_mean, dtype=np.float32).reshape(-1)
    scale = np.asarray(target_scale, dtype=np.float32).reshape(-1)
    meta_key, input_key, temp_key = jax.random.split(key, 3)
    metadata = jax.random.normal(meta_key, (batch_size, metadata_dim), dtype=jnp.float32) * 0.25
    inputs = jax.random.normal(input_key, (batch_size, horizon, input_dim), dtype=jnp.float32) * 0.25
    initial_temperature = jax.random.normal(temp_key, (batch_size, 1), dtype=jnp.float32) * 0.1
    time = jnp.arange(horizon, dtype=jnp.float32)
    temperature = jnp.broadcast_to(0.1 * jnp.sin(time / 48.0), (batch_size, horizon))
    qroom = 0.2 * jnp.ones((batch_size, horizon), dtype=jnp.float32)
    pel = 0.15 * jnp.ones((batch_size, horizon), dtype=jnp.float32)
    if kind == "sharp_hp":
        on = ((time.astype(jnp.int32) // 24) % 2 == 0).astype(jnp.float32)
        pel = jnp.broadcast_to(jnp.where(on, 1.5, -mean[2] / jnp.maximum(scale[2], 1e-6)), (batch_size, horizon))
        qroom = 0.8 * jnp.maximum(pel, 0.0)
    elif kind == "setpoint_jump":
        jump = (time >= horizon // 2).astype(jnp.float32)
        inputs = inputs.at[:, :, 0].set(jump)
        pel = jnp.broadcast_to(jump * 1.2, (batch_size, horizon))
        qroom = jnp.broadcast_to(jump * 1.0, (batch_size, horizon))
        temperature = temperature + 0.05 * jnp.cumsum(jnp.broadcast_to(jump, (batch_size, horizon)), axis=1) / horizon
    elif kind != "smooth":
        raise ValueError(f"Unknown synthetic batch kind {kind!r}")
    targets = jnp.stack([temperature, qroom, pel], axis=-1)
    return metadata, inputs, initial_temperature, targets


def loss_and_components(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    metadata: jnp.ndarray,
    inputs: jnp.ndarray,
    initial_temperature: jnp.ndarray,
    targets: jnp.ndarray,
    key: jax.Array,
    *,
    num_particles: int,
    weights: ComponentWeights,
) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, ...]]:
    predictions, aux = predict_probabilistic_closed_loop_batch_with_aux(
        model,
        metadata,
        inputs,
        initial_temperature,
        key=key,
        num_particles=num_particles,
        sample_process_noise=True,
        hp_scenario_mode="expected",
    )
    zero = jnp.asarray(0.0, dtype=predictions.dtype)
    energy_component = energy_score(predictions, targets) if weights.energy else zero
    variogram_component = zero
    if weights.variogram:
        variogram_component = variogram_score(
            predictions,
            targets,
            lags=weights.variogram_lags,
            power=weights.variogram_power,
            output_weights=weights.variogram_output_weights,
            constant_setpoint_inputs=inputs,
            setpoint_delta_threshold=0.05,
        )
    ires_component = zero
    if weights.ires:
        ires_component = intervention_response_energy_score(
            predictions,
            targets,
            inputs,
            horizon_steps=weights.ires_horizon_steps,
            horizon_hours=weights.ires_horizon_hours,
            setpoint_delta_threshold=0.05,
            target_channel=2,
        )
    pi = aux[0]
    log_mu = aux[PROB_CLOSED_LOOP_AUX_LOG_MU]
    log_sigma = aux[PROB_CLOSED_LOOP_AUX_LOG_SIGMA]
    target_mean = jnp.asarray(model.target_mean, dtype=predictions.dtype)
    target_scale = jnp.asarray(model.target_scale, dtype=predictions.dtype)
    pel_target = _physical_target_component(targets, target_mean, target_scale, 2)
    mode_target = (pel_target > weights.heat_on_threshold).astype(predictions.dtype)
    mode_target_particles = mode_target[:, jnp.newaxis, :]
    hp_bce_component = zero
    if weights.hp_bce:
        clipped_pi = jnp.clip(pi, 1e-5, 1.0 - 1e-5)
        bce = -jnp.mean(
            mode_target_particles * jnp.log(clipped_pi)
            + (1.0 - mode_target_particles) * jnp.log(1.0 - clipped_pi)
        )
        hp_bce_component = weights.hp_mode_loss_weight * bce
    hp_active_nll_component = zero
    if weights.hp_active_nll:
        active_mask = jnp.broadcast_to(mode_target_particles, log_mu.shape)
        log_target = jnp.log1p(jnp.maximum(pel_target, 0.0))[:, jnp.newaxis, :]
        nll = hp_active_nll(log_target, log_mu, log_sigma)
        active_count = jnp.maximum(jnp.sum(active_mask), 1.0)
        hp_active_nll_component = weights.hp_active_power_nll_weight * (
            jnp.sum(active_mask * nll) / active_count
        )
    components = jnp.stack(
        [
            energy_component,
            variogram_component,
            ires_component,
            hp_bce_component,
            hp_active_nll_component,
        ]
    )
    return jnp.sum(components), components, aux, predictions


def value_and_grad_for_weights(
    weights: ComponentWeights,
):
    @eqx.filter_value_and_grad(has_aux=True)
    def _fn(
        model: ProbabilisticContractingClosedLoopHPEmulator,
        metadata: jnp.ndarray,
        inputs: jnp.ndarray,
        initial_temperature: jnp.ndarray,
        targets: jnp.ndarray,
        key: jax.Array,
        num_particles: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        loss, components, aux, predictions = loss_and_components(
            model,
            metadata,
            inputs,
            initial_temperature,
            targets,
            key,
            num_particles=num_particles,
            weights=weights,
        )
        return loss, (components, aux, predictions)

    return eqx.filter_jit(_fn)


def module_grad_norms(grads: object) -> dict[str, float]:
    report: dict[str, float] = {}
    for name in MODULE_NAMES:
        subtree = getattr(grads, name, None)
        if subtree is None:
            continue
        report[name] = _scalar(_array_tree_global_norm(subtree))
    return report


def aux_stats(aux: tuple[jnp.ndarray, ...], predictions: jnp.ndarray | None = None) -> dict[str, float]:
    pi = np.asarray(aux[0])
    log_mu = np.asarray(aux[PROB_CLOSED_LOOP_AUX_LOG_MU])
    log_sigma = np.asarray(aux[PROB_CLOSED_LOOP_AUX_LOG_SIGMA])
    stats = {
        "pi_min": float(np.min(pi)),
        "pi_max": float(np.max(pi)),
        "log_mu_min": float(np.min(log_mu)),
        "log_mu_max": float(np.max(log_mu)),
        "log_sigma_min": float(np.min(log_sigma)),
        "log_sigma_max": float(np.max(log_sigma)),
    }
    if predictions is not None:
        pred = np.asarray(predictions)
        pairwise = pred[:, :, jnp.newaxis] - pred[:, jnp.newaxis]
        pairwise_norm = np.sqrt(np.sum(pairwise**2, axis=(3, 4)) + 1e-6)
        eye = np.eye(pred.shape[1], dtype=bool)
        offdiag = pairwise_norm[:, ~eye]
        stats["particle_pairwise_min"] = float(np.min(offdiag))
        stats["particle_pairwise_mean"] = float(np.mean(offdiag))
        stats["pred_abs_max"] = float(np.max(np.abs(pred)))
    return stats


def evaluate_component(
    model: ProbabilisticContractingClosedLoopHPEmulator,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    key: jax.Array,
    *,
    num_particles: int,
    weights: ComponentWeights,
) -> dict[str, object]:
    loss_fn = value_and_grad_for_weights(weights)
    metadata, inputs, initial_temperature, targets = batch
    (loss, (components, aux, predictions)), grads = loss_fn(
        model,
        metadata,
        inputs,
        initial_temperature,
        targets,
        key,
        num_particles,
    )
    return {
        "loss": _scalar(loss),
        "components": {name: _scalar(value) for name, value in zip(COMPONENT_ORDER, components)},
        "grad_finite": bool(np.asarray(_array_tree_all_finite(grads))),
        "grad_norm": _scalar(_array_tree_global_norm(grads)),
        "grad_max_abs": _scalar(_array_tree_max_abs(grads)),
        "module_grad_norms": module_grad_norms(grads),
        "aux": aux_stats(aux, predictions),
        "pred_finite": bool(np.asarray(_array_tree_all_finite(predictions))),
        "aux_finite": bool(np.asarray(_array_tree_all_finite(aux))),
    }


def training_like_weights() -> ComponentWeights:
    return ComponentWeights(
        energy=1.0,
        variogram=1.0,
        ires=1.0,
        hp_bce=1.0,
        hp_active_nll=1.0,
    )


def isolated_weights(name: str) -> ComponentWeights:
    kwargs = {component: 0.0 for component in COMPONENT_ORDER}
    kwargs[name] = 1.0
    return ComponentWeights(**kwargs)


def format_report(title: str, report: dict[str, object]) -> str:
    modules = report["module_grad_norms"]
    top_module = max(modules, key=modules.get) if modules else "none"
    aux = report["aux"]
    return (
        f"{title}: loss={report['loss']:.6g} finite_loss={_finite(report['loss'])} "
        f"grad_finite={report['grad_finite']} grad_norm={report['grad_norm']:.6g} "
        f"grad_max_abs={report['grad_max_abs']:.6g} top_module={top_module} "
        f"log_sigma=[{aux['log_sigma_min']:.4g},{aux['log_sigma_max']:.4g}] "
        f"pi=[{aux['pi_min']:.4g},{aux['pi_max']:.4g}]"
    )


class IsolatedScoringTests(unittest.TestCase):
    def test_zero_pel_trajectory_weight_removes_pointwise_pel_error(self) -> None:
        targets = jnp.zeros((1, 8, 3), dtype=jnp.float32)
        predictions = jnp.zeros((1, 3, 8, 3), dtype=jnp.float32)
        predictions = predictions.at[..., 2].set(100.0)
        perfect = jnp.zeros_like(predictions)
        weights = (1.0, 1.0, 0.0)

        np.testing.assert_allclose(
            energy_score(predictions, targets, output_weights=weights),
            energy_score(perfect, targets, output_weights=weights),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            soft_optimistic_loss(
                predictions,
                targets,
                temperature=0.1,
                horizon_weight_power=0.0,
                output_weights=weights,
            ),
            0.0,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_energy_score_stays_finite_with_collapsed_particles(self) -> None:
        targets = jnp.zeros((2, 32, 3), dtype=jnp.float32)
        predictions = jnp.broadcast_to(targets[:, jnp.newaxis, :, :], (2, 8, 32, 3))
        loss = energy_score(predictions, targets)
        grad = jax.grad(lambda x: energy_score(x, targets))(predictions)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(bool(np.all(np.isfinite(grad))))
        self.assertLess(float(jnp.max(jnp.abs(grad))), 1e3)

    def test_variogram_power_half_is_smooth_at_zero(self) -> None:
        values = jnp.linspace(-1e-3, 1e-3, 21, dtype=jnp.float32)
        grad_norm = scoring_grad_norm(lambda x: smooth_abs_power(x, 0.5), values)
        self.assertTrue(math.isfinite(grad_norm))
        self.assertLess(grad_norm, 50.0)

    def test_ires_score_finite_on_sharp_setpoint_jumps(self) -> None:
        horizon = 64
        targets = jnp.zeros((2, horizon, 3), dtype=jnp.float32)
        predictions = jnp.zeros((2, 4, horizon, 3), dtype=jnp.float32)
        inputs = jnp.zeros((2, horizon, 8), dtype=jnp.float32)
        inputs = inputs.at[:, horizon // 2 :, 0].set(2.0)
        targets = targets.at[:, horizon // 2 :, 2].set(1.5)
        predictions = predictions.at[:, :, horizon // 2 :, 2].set(1.4)
        loss = intervention_response_energy_score(
            predictions,
            targets,
            inputs,
            horizon_steps=(2, 4, 8),
            horizon_hours=(0.5, 1.0, 2.0),
            setpoint_delta_threshold=0.05,
            target_channel=2,
        )
        grad = jax.grad(
            lambda x: intervention_response_energy_score(
                x,
                targets,
                inputs,
                horizon_steps=(2, 4, 8),
                horizon_hours=(0.5, 1.0, 2.0),
                setpoint_delta_threshold=0.05,
                target_channel=2,
            )
        )(predictions)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(bool(np.all(np.isfinite(grad))))

    def test_hp_active_nll_at_sigma_floor_is_large_but_finite(self) -> None:
        residual = jnp.asarray(4.6, dtype=jnp.float32)
        sigma = jnp.asarray(0.05, dtype=jnp.float32)

        def nll(log_sigma: jnp.ndarray) -> jnp.ndarray:
            return hp_active_nll(residual, jnp.asarray(0.0, dtype=residual.dtype), log_sigma)

        value = nll(sigma)
        grad = jax.grad(nll)(sigma)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(grad))
        self.assertLess(abs(float(grad)), 1e7)

    def test_hp_active_nll_is_singular_if_sigma_is_unbounded(self) -> None:
        residual = jnp.asarray(2.0, dtype=jnp.float32)

        def nll(log_sigma: jnp.ndarray) -> jnp.ndarray:
            return hp_active_nll(residual, jnp.asarray(0.0, dtype=residual.dtype), log_sigma)

        tiny = jnp.asarray(1e-6, dtype=jnp.float32)
        grad = jax.grad(nll)(tiny)
        self.assertTrue(np.isfinite(nll(tiny)))
        self.assertGreater(abs(float(grad)), 1e16)


class TinyModelAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = make_tiny_model(jax.random.PRNGKey(0))
        cls.num_particles = 4

    def _batch(self, kind: str, horizon: int, key: jax.Array):
        return synthetic_batch(
            batch_size=2,
            horizon=horizon,
            metadata_dim=4,
            input_dim=8,
            kind=kind,
            key=key,
            target_mean=self.model.target_mean,
            target_scale=self.model.target_scale,
        )

    def test_untrained_tiny_model_components_stay_finite(self) -> None:
        batch = self._batch("sharp_hp", 48, jax.random.PRNGKey(1))
        for name in COMPONENT_ORDER:
            report = evaluate_component(
                self.model,
                batch,
                jax.random.PRNGKey(2),
                num_particles=self.num_particles,
                weights=isolated_weights(name),
            )
            self.assertTrue(report["pred_finite"], msg=name)
            self.assertTrue(report["grad_finite"], msg=format_report(name, report))
            self.assertLess(report["grad_norm"], 1e6, msg=format_report(name, report))

    def test_untrained_combined_loss_stays_finite_on_short_horizon(self) -> None:
        batch = self._batch("setpoint_jump", 64, jax.random.PRNGKey(3))
        report = evaluate_component(
            self.model,
            batch,
            jax.random.PRNGKey(4),
            num_particles=self.num_particles,
            weights=training_like_weights(),
        )
        self.assertTrue(report["grad_finite"], msg=format_report("combined", report))
        self.assertLess(report["grad_norm"], 1e6, msg=format_report("combined", report))


def load_train_windows(artifact, *, max_profiles: int, sequence_length: int, stride: int):
    train_ids = [int(value) for value in artifact.metadata.get("train_ids", [])]
    if not train_ids:
        return None
    profile_ids = train_ids[:max_profiles]
    df = _read_parquet(
        DEFAULT_DATASET,
        columns=closed_loop_required_columns(),
        filters=[(PROFILE_ID_COLUMN, "in", profile_ids)],
    )
    if df.empty:
        return None
    profiles = to_closed_loop_profiles(
        df,
        include_space_heating_availability=(
            SPACE_HEATING_AVAILABILITY_COLUMN in artifact.metadata.get("input_columns", [])
        ),
        hp_power_area_normalization=artifact.metadata.get(
            "hp_power_area_normalization",
            artifact.metadata.get("train_config", {}).get(
                "hp_power_area_normalization",
                "zone_floor_area",
            ),
        ),
    )
    windows = make_closed_loop_windows(
        profiles,
        WindowConfig(sequence_length=sequence_length, stride=stride),
    )
    windows = transform_closed_loop_windows(windows, artifact.scalers)
    return windows


def iterate_window_batches(windows, batch_size: int, *, shuffle_seed: int | None = None):
    n = int(windows.metadata.shape[0])
    order = np.arange(n)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        indices = order[start : min(start + batch_size, n)]
        yield (
            start,
            (
                jnp.asarray(windows.metadata[indices]),
                jnp.asarray(windows.inputs[indices]),
                jnp.asarray(windows.initial_temperature[indices]),
                jnp.asarray(windows.targets[indices]),
            ),
            windows.profile_ids[indices],
            windows.start_indices[indices],
        )


def load_epoch_model(epoch: int, *, bptt_truncate_steps: int | None = None):
    artifact_dir = DEFAULT_CHECKPOINT_DIR / f"epoch_{epoch:03d}"
    if not artifact_dir.exists():
        return None
    overrides = None
    if bptt_truncate_steps is not None:
        overrides = {"bptt_truncate_steps": int(bptt_truncate_steps)}
    return load_training_artifact(artifact_dir, train_config_overrides=overrides)


def slice_batch(batch, horizon: int):
    metadata, inputs, initial_temperature, targets = batch
    return metadata, inputs[:, :horizon], initial_temperature, targets[:, :horizon]


def real_worst_case_batch(artifact, horizon: int):
    if not DEFAULT_DATASET.exists():
        return None
    profile_ids = sorted({profile_id for profile_id, _start in WORST_REAL_WINDOWS})
    df = _read_parquet(
        DEFAULT_DATASET,
        columns=closed_loop_required_columns(),
        filters=[(PROFILE_ID_COLUMN, "in", profile_ids)],
    )
    if df.empty:
        return None
    profiles = {
        profile.profile_id: profile
        for profile in to_closed_loop_profiles(
            df,
            include_space_heating_availability=(
                SPACE_HEATING_AVAILABILITY_COLUMN in artifact.metadata.get("input_columns", [])
            ),
            hp_power_area_normalization=artifact.metadata.get(
                "hp_power_area_normalization",
                artifact.metadata.get("train_config", {}).get(
                    "hp_power_area_normalization",
                    "zone_floor_area",
                ),
            ),
        )
    }
    profile_ids_out = []
    starts = []
    metadata = []
    inputs = []
    targets = []
    initial_temperature = []
    for profile_id, start in WORST_REAL_WINDOWS:
        profile = profiles.get(profile_id)
        if profile is None:
            continue
        end = start + horizon
        if end > profile.targets.shape[0]:
            continue
        profile_ids_out.append(profile_id)
        starts.append(start)
        metadata.append(profile.metadata)
        inputs.append(profile.inputs[start:end])
        targets.append(profile.targets[start:end])
        initial_temperature.append(profile.initial_temperature[start])
    if not inputs:
        return None
    windows = ClosedLoopWindowedArrays(
        profile_ids=np.asarray(profile_ids_out, dtype=np.int64),
        start_indices=np.asarray(starts, dtype=np.int64),
        metadata=np.stack(metadata).astype(np.float32),
        inputs=np.stack(inputs).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        initial_temperature=np.stack(initial_temperature).astype(np.float32),
    )
    windows = transform_closed_loop_windows(windows, artifact.scalers)
    return (
        jnp.asarray(windows.metadata),
        jnp.asarray(windows.inputs),
        jnp.asarray(windows.initial_temperature),
        jnp.asarray(windows.targets),
    )


def jacobian_gain_report(model, batch, key: jax.Array, num_particles: int) -> dict[str, float]:
    metadata, inputs, initial_temperature, _targets = batch
    _predictions, aux = predict_probabilistic_closed_loop_batch_with_aux(
        model,
        metadata,
        inputs,
        initial_temperature,
        key=key,
        num_particles=num_particles,
        sample_process_noise=True,
        hp_scenario_mode="expected",
    )
    horizon = int(inputs.shape[1])
    sample_times = np.unique(np.linspace(0, horizon - 1, num=min(8, horizon), dtype=int))
    sigma_values = []
    for time_index in sample_times:
        xi = aux[PROB_CLOSED_LOOP_AUX_XI][0, 0]
        energy = aux[PROB_CLOSED_LOOP_AUX_ENERGY][0, 0, time_index]
        temperature = aux[PROB_CLOSED_LOOP_AUX_TEMPERATURE_STATE][0, 0, time_index]
        state = aux[PROB_CLOSED_LOOP_AUX_X_STATE][0, 0, time_index]
        controller_state = aux[PROB_CLOSED_LOOP_AUX_W_STATE][0, 0, time_index]
        hp_latent = aux[PROB_CLOSED_LOOP_AUX_HP_LATENT][0, 0]
        augmented = model.pack_stability_state(
            state,
            energy,
            temperature,
            controller_state,
            aux[PROB_CLOSED_LOOP_AUX_HP_PREVIOUS_MODE][0, 0, time_index],
            aux[PROB_CLOSED_LOOP_AUX_HP_POWER_HISTORY][0, 0, time_index],
            aux[PROB_CLOSED_LOOP_AUX_HP_MODE_HISTORY][0, 0, time_index],
        )
        sigma_max = model.closed_loop_jacobian_spectral_norm(
            metadata[0],
            inputs[0, time_index],
            xi,
            hp_latent,
            augmented,
            jnp.zeros((model.state_dim,), dtype=state.dtype),
            jnp.zeros((model.controller_state_dim,), dtype=state.dtype),
        )
        sigma_values.append(_scalar(sigma_max))
    return {
        "sigma_max": float(np.max(sigma_values)),
        "sigma_median": float(np.median(sigma_values)),
        "sigma_min": float(np.min(sigma_values)),
    }


def is_exploding(report: dict[str, object], grad_norm_limit: float = 1e8) -> bool:
    return (not report["grad_finite"]) or report["grad_norm"] > grad_norm_limit


def probe_batch(
    model,
    batch,
    key: jax.Array,
    jac_key: jax.Array,
    *,
    num_particles: int,
    label: str,
    with_jacobian: bool = True,
) -> int:
    print(f"\n-- {label} --")
    if with_jacobian:
        gain = jacobian_gain_report(model, batch, jac_key, num_particles=num_particles)
        print(
            "closed_loop_jacobian_sigma="
            f"[min={gain['sigma_min']:.4g}, median={gain['sigma_median']:.4g}, max={gain['sigma_max']:.4g}]"
        )
    combined = evaluate_component(
        model,
        batch,
        key,
        num_particles=num_particles,
        weights=training_like_weights(),
    )
    exploding = is_exploding(combined)
    flag = "  <-- EXPLODING" if exploding else ""
    print(format_report("combined", combined) + flag)
    print(f"  modules={combined['module_grad_norms']}")
    print(f"  aux={combined['aux']}")
    failures = int(exploding)
    if not exploding:
        return failures
    for name in COMPONENT_ORDER:
        report = evaluate_component(
            model,
            batch,
            key,
            num_particles=num_particles,
            weights=isolated_weights(name),
        )
        flag = ""
        if not report["grad_finite"]:
            flag = "  <-- NONFINITE GRADS"
            failures += 1
        elif report["grad_norm"] > 1e8:
            flag = "  <-- HUGE GRADS"
            failures += 1
        print("  " + format_report(name, report) + flag)
        if flag:
            print(f"    modules={report['module_grad_norms']}")
    return failures


def run_checkpoint_probe(
    epochs: tuple[int, ...],
    horizons: tuple[int, ...],
    include_real: bool,
    include_synthetic: bool = True,
    bptt_truncate_steps: int | None = None,
) -> int:
    failures = 0
    for epoch in epochs:
        artifact = load_epoch_model(epoch, bptt_truncate_steps=bptt_truncate_steps)
        if artifact is None:
            print(f"skip epoch_{epoch:03d}: checkpoint not found")
            continue
        model = artifact.model
        assert isinstance(model, ProbabilisticContractingClosedLoopHPEmulator)
        print(
            f"\n=== checkpoint epoch {epoch:03d} "
            f"bptt_truncate_steps={int(model.bptt_truncate_steps)} ==="
        )
        key = jax.random.PRNGKey(13 + epoch)
        batch_key, loss_key, jac_key = jax.random.split(key, 3)
        if include_synthetic:
            for kind in ("smooth", "sharp_hp", "setpoint_jump"):
                full = synthetic_batch(
                    batch_size=2,
                    horizon=max(horizons),
                    metadata_dim=int(model.metadata_dim),
                    input_dim=int(model.input_dim),
                    kind=kind,
                    key=batch_key,
                    target_mean=model.target_mean,
                    target_scale=model.target_scale,
                )
                for horizon in horizons:
                    failures += probe_batch(
                        model,
                        slice_batch(full, horizon),
                        loss_key,
                        jac_key,
                        num_particles=4,
                        label=f"synthetic {kind} T={horizon}",
                    )
        if include_real:
            real_batch = real_worst_case_batch(artifact, max(horizons))
            if real_batch is None:
                print("skip real windows: dataset missing or empty")
                continue
            print("\n-- real worst-case windows --")
            for horizon in horizons:
                batch = slice_batch(real_batch, horizon)
                failures += probe_batch(
                    model,
                    batch,
                    loss_key,
                    jac_key,
                    num_particles=4,
                    label=f"real worst-case T={horizon} n={batch[0].shape[0]}",
                )
    return failures


def scan_train_minibatches(
    epochs: tuple[int, ...],
    *,
    max_profiles: int,
    batch_size: int,
    num_particles: int,
    max_batches: int,
    sequence_length: int = 960,
    stride: int = 2024,
    bptt_truncate_steps: int | None = None,
) -> int:
    failures = 0
    for epoch in epochs:
        artifact = load_epoch_model(epoch, bptt_truncate_steps=bptt_truncate_steps)
        if artifact is None:
            print(f"skip epoch_{epoch:03d}: checkpoint not found")
            continue
        model = artifact.model
        assert isinstance(model, ProbabilisticContractingClosedLoopHPEmulator)
        windows = load_train_windows(
            artifact,
            max_profiles=max_profiles,
            sequence_length=sequence_length,
            stride=stride,
        )
        if windows is None:
            print("skip train scan: no windows")
            continue
        print(
            f"\n=== train-window scan epoch {epoch:03d} "
            f"bptt_truncate_steps={int(model.bptt_truncate_steps)} "
            f"windows={windows.metadata.shape[0]} profiles={max_profiles} "
            f"batch_size={batch_size} particles={num_particles} ==="
        )
        key = jax.random.PRNGKey(1000 + epoch)
        scanned = 0
        for _offset, batch, profile_ids, starts in iterate_window_batches(
            windows,
            batch_size,
            shuffle_seed=13 + epoch,
        ):
            scanned += 1
            if scanned > max_batches:
                break
            key, step_key, jac_key = jax.random.split(key, 3)
            print(
                f"\nbatch={scanned} profiles={list(map(int, profile_ids[:8]))} "
                f"starts={list(map(int, starts[:8]))}"
            )
            before = failures
            failures += probe_batch(
                model,
                batch,
                step_key,
                jac_key,
                num_particles=num_particles,
                label=f"train batch {scanned} K={int(model.bptt_truncate_steps)}",
                with_jacobian=False,
            )
            if scanned == 1 or failures > before:
                print(
                    "\n-- horizon ablation "
                    f"{'on exploding batch' if failures > before else 'on first batch'} "
                    f"K={int(model.bptt_truncate_steps)} --"
                )
                ablate_horizon(
                    model,
                    batch,
                    step_key,
                    jac_key,
                    num_particles=num_particles,
                )
            if failures > before:
                return failures
    return failures


def ablate_horizon(
    model,
    batch,
    key: jax.Array,
    jac_key: jax.Array,
    *,
    num_particles: int,
    horizons: tuple[int, ...] = (16, 32, 64, 128, 256, 480, 960),
) -> None:
    full_horizon = int(batch[1].shape[1])
    for horizon in horizons:
        if horizon > full_horizon:
            continue
        sliced = slice_batch(batch, horizon)
        energy = evaluate_component(
            model,
            sliced,
            key,
            num_particles=num_particles,
            weights=isolated_weights("energy"),
        )
        bce = evaluate_component(
            model,
            sliced,
            key,
            num_particles=num_particles,
            weights=isolated_weights("hp_bce"),
        )
        gain = jacobian_gain_report(model, sliced, jac_key, num_particles=num_particles)
        print(
            f"T={horizon:4d} energy_grad={energy['grad_norm']:.6g} "
            f"energy_max_abs={energy['grad_max_abs']:.6g} "
            f"bce_grad={bce['grad_norm']:.6g} "
            f"bce_max_abs={bce['grad_max_abs']:.6g} "
            f"sigma_max={gain['sigma_max']:.4g} "
            f"pred_abs_max={energy['aux']['pred_abs_max']:.4g}"
        )


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--scan-train", action="store_true")
    parser.add_argument("--scan-profiles", type=int, default=24)
    parser.add_argument("--scan-batches", type=int, default=16)
    parser.add_argument("--scan-batch-size", type=int, default=8)
    parser.add_argument("--scan-particles", type=int, default=10)
    parser.add_argument("--epochs", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--horizons", nargs="+", type=int, default=[64, 256, 960])
    parser.add_argument(
        "--bptt-truncate-steps",
        nargs="+",
        type=int,
        default=[16],
        help=(
            "Override checkpoint BPTT truncation when loading models. "
            "Pass multiple values to compare, e.g. `--bptt-truncate-steps 16 0`. "
            "Default 16 matches the new training default. Old checkpoints load as 0 "
            "unless this flag is set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    if not args.skip_unit:
        suite = unittest.defaultTestLoader.loadTestsFromModule(
            __import__(__name__, fromlist=["*"])
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    if args.skip_checkpoint and not args.scan_train:
        return
    failures = 0
    truncate_values = tuple(int(value) for value in args.bptt_truncate_steps)
    for truncate_steps in truncate_values:
        print(f"\n##### bptt_truncate_steps={truncate_steps} #####")
        if not args.skip_checkpoint:
            failures += run_checkpoint_probe(
                tuple(args.epochs),
                tuple(args.horizons),
                include_real=not args.skip_real,
                include_synthetic=not args.skip_synthetic,
                bptt_truncate_steps=truncate_steps,
            )
        if args.scan_train:
            failures += scan_train_minibatches(
                tuple(args.epochs),
                max_profiles=args.scan_profiles,
                batch_size=args.scan_batch_size,
                num_particles=args.scan_particles,
                max_batches=args.scan_batches,
                bptt_truncate_steps=truncate_steps,
            )
    if failures:
        print(f"\nprobe_failures={failures}")
        raise SystemExit(2)
    print("\nprobe_failures=0")


if __name__ == "__main__":
    main()
