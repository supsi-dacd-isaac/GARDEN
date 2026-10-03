from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from neural_building_emulator.columns import CLOSED_LOOP_METADATA_COLUMNS
from neural_building_emulator_refactor.conditional_state_space_sufficiency import (
    INPUT_NAMES,
    Scales,
    VARIANTS,
    _initial_parameters,
    _rollout,
    _selected_profile_split,
    _state_space_parameters,
)


class ConditionalStateSpaceSufficiencyTests(unittest.TestCase):
    @patch(
        "neural_building_emulator_refactor.conditional_state_space_sufficiency."
        "read_profile_ids"
    )
    def test_holdout_split_uses_disjoint_complete_buildings(
        self, read_profile_ids
    ) -> None:
        read_profile_ids.side_effect = (
            tuple(range(1, 21)),
            tuple(range(5, 25)),
        )
        args = Namespace(
            setpoint_dataset=Path("setpoint"),
            force_off_dataset=Path("force_off"),
            profile_ids=None,
            max_profiles=10,
            profile_selection="random",
            profile_filter="common",
            test_fraction=0.2,
            seed=13,
        )

        selected, train, test = _selected_profile_split(args)

        self.assertEqual(len(selected), 10)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)
        self.assertFalse(set(train).intersection(test))
        self.assertEqual(set(selected), set(train).union(test))

    def test_all_variants_keep_latent_modes_below_gamma(self) -> None:
        gamma = 0.995
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                params = _initial_parameters(
                    variant,
                    key=jax.random.PRNGKey(13),
                    num_buildings=2,
                    metadata_dim=5,
                    exogenous_dim=4,
                    state_dim=3,
                    encoded_dim=2,
                    hidden_dim=6,
                    depth=2,
                    gamma=gamma,
                    dt_hours=0.25,
                )
                rho, *_ = _state_space_parameters(
                    params,
                    variant,
                    jnp.asarray(0),
                    jnp.zeros((5,)),
                    jnp.ones((4,)),
                    state_dim=3,
                    encoded_dim=2,
                    gamma=gamma,
                )
                self.assertTrue(np.all(np.asarray(rho) > 0.0))
                self.assertTrue(np.all(np.asarray(rho) < gamma))

    def test_fixed_variants_ignore_exogenous_inputs_in_matrices(self) -> None:
        for variant in (
            "local_fixed",
            "metadata_fixed",
            "metadata_fixed_positive_leaky",
            "metadata_fixed_pel",
            "metadata_fixed_pel_cop_positive_leaky",
        ):
            with self.subTest(variant=variant):
                params = _initial_parameters(
                    variant,
                    key=jax.random.PRNGKey(7),
                    num_buildings=2,
                    metadata_dim=5,
                    exogenous_dim=4,
                    state_dim=3,
                    encoded_dim=2,
                    hidden_dim=6,
                    depth=2,
                    gamma=0.995,
                    dt_hours=0.25,
                )
                common = dict(
                    params=params,
                    variant=variant,
                    building_index=jnp.asarray(1),
                    metadata=jnp.arange(5, dtype=jnp.float32),
                    state_dim=3,
                    encoded_dim=2,
                    gamma=0.995,
                )
                first = _state_space_parameters(
                    exogenous=jnp.zeros((4,)), **common
                )
                second = _state_space_parameters(
                    exogenous=jnp.ones((4,)), **common
                )
                for first_value, second_value in zip(first, second):
                    np.testing.assert_array_equal(first_value, second_value)

    def test_positive_leaky_branch_is_monotone_in_qroom(self) -> None:
        metadata_dim = 5
        params = _initial_parameters(
            "metadata_fixed_positive_leaky",
            key=jax.random.PRNGKey(19),
            num_buildings=1,
            metadata_dim=metadata_dim,
            exogenous_dim=10,
            state_dim=3,
            encoded_dim=2,
            hidden_dim=6,
            depth=2,
            gamma=0.995,
            dt_hours=0.25,
            q_response_modes=2,
        )
        scales = Scales(
            metadata_mean=np.zeros((metadata_dim,), dtype=np.float32),
            metadata_std=np.ones((metadata_dim,), dtype=np.float32),
            exogenous_mean=np.zeros((10,), dtype=np.float32),
            exogenous_std=np.ones((10,), dtype=np.float32),
            temperature_mean=20.0,
            temperature_std=2.0,
            qroom_scale=10.0,
            pel_scale=5.0,
        )
        common = dict(
            params=params,
            variant="metadata_fixed_positive_leaky",
            building_index=jnp.asarray(0),
            metadata_raw=jnp.zeros((metadata_dim,), dtype=jnp.float32),
            inputs=jnp.zeros((1, len(INPUT_NAMES)), dtype=jnp.float32),
            initial_temperature=jnp.asarray(20.0),
            scales=scales,
            state_dim=3,
            encoded_dim=2,
            gamma=0.995,
            bptt_truncate_steps=32,
            dt_hours=0.25,
            q_to_t_time_constants_hours=(0.25, 1.0),
            q_to_t_gain_min=0.01,
            q_to_t_gain_max=2.0,
            pel_cop_slope_per_c=0.05,
            pel_cop_reference_outdoor_c=7.0,
            pel_cop_floor=1.0,
            pel_cop_cap=8.0,
            pel=jnp.zeros((1,)),
        )

        no_heat = _rollout(qroom=jnp.zeros((1,)), **common)
        more_heat = _rollout(qroom=jnp.asarray([10.0]), **common)

        self.assertGreater(float(more_heat[0]), float(no_heat[0]))

    def test_direct_pel_variant_does_not_use_qroom(self) -> None:
        metadata_dim = len(CLOSED_LOOP_METADATA_COLUMNS)
        params = _initial_parameters(
            "metadata_fixed_pel",
            key=jax.random.PRNGKey(23),
            num_buildings=1,
            metadata_dim=metadata_dim,
            exogenous_dim=10,
            state_dim=3,
            encoded_dim=2,
            hidden_dim=6,
            depth=2,
            gamma=0.995,
            dt_hours=0.25,
        )
        scales = Scales(
            metadata_mean=np.zeros((metadata_dim,), dtype=np.float32),
            metadata_std=np.ones((metadata_dim,), dtype=np.float32),
            exogenous_mean=np.zeros((10,), dtype=np.float32),
            exogenous_std=np.ones((10,), dtype=np.float32),
            temperature_mean=20.0,
            temperature_std=2.0,
            qroom_scale=10.0,
            pel_scale=5.0,
        )
        common = dict(
            params=params,
            variant="metadata_fixed_pel",
            building_index=jnp.asarray(0),
            metadata_raw=jnp.zeros((metadata_dim,), dtype=jnp.float32),
            inputs=jnp.zeros((2, len(INPUT_NAMES)), dtype=jnp.float32),
            pel=jnp.ones((2,), dtype=jnp.float32),
            initial_temperature=jnp.asarray(20.0),
            scales=scales,
            state_dim=3,
            encoded_dim=2,
            gamma=0.995,
            bptt_truncate_steps=32,
            dt_hours=0.25,
            q_to_t_time_constants_hours=(0.25, 1.0),
            q_to_t_gain_min=0.01,
            q_to_t_gain_max=2.0,
            pel_cop_slope_per_c=0.05,
            pel_cop_reference_outdoor_c=7.0,
            pel_cop_floor=1.0,
            pel_cop_cap=8.0,
        )

        zero_q = _rollout(qroom=jnp.zeros((2,)), **common)
        arbitrary_q = _rollout(qroom=jnp.asarray([20.0, 100.0]), **common)

        np.testing.assert_array_equal(zero_q, arbitrary_q)

    def test_positive_leaky_pel_branch_is_monotone(self) -> None:
        metadata_dim = len(CLOSED_LOOP_METADATA_COLUMNS)
        params = _initial_parameters(
            "metadata_fixed_pel_cop_positive_leaky",
            key=jax.random.PRNGKey(29),
            num_buildings=1,
            metadata_dim=metadata_dim,
            exogenous_dim=10,
            state_dim=3,
            encoded_dim=2,
            hidden_dim=6,
            depth=2,
            gamma=0.995,
            dt_hours=0.25,
            q_response_modes=2,
        )
        scales = Scales(
            metadata_mean=np.zeros((metadata_dim,), dtype=np.float32),
            metadata_std=np.ones((metadata_dim,), dtype=np.float32),
            exogenous_mean=np.zeros((10,), dtype=np.float32),
            exogenous_std=np.ones((10,), dtype=np.float32),
            temperature_mean=20.0,
            temperature_std=2.0,
            qroom_scale=10.0,
            pel_scale=5.0,
        )
        metadata = jnp.zeros((metadata_dim,), dtype=jnp.float32)
        metadata = metadata.at[
            tuple(CLOSED_LOOP_METADATA_COLUMNS).index("hp_ref_cop")
        ].set(4.0)
        common = dict(
            params=params,
            variant="metadata_fixed_pel_cop_positive_leaky",
            building_index=jnp.asarray(0),
            metadata_raw=metadata,
            inputs=jnp.zeros((1, len(INPUT_NAMES)), dtype=jnp.float32),
            qroom=jnp.asarray([100.0]),
            initial_temperature=jnp.asarray(20.0),
            scales=scales,
            state_dim=3,
            encoded_dim=2,
            gamma=0.995,
            bptt_truncate_steps=32,
            dt_hours=0.25,
            q_to_t_time_constants_hours=(0.25, 1.0),
            q_to_t_gain_min=0.01,
            q_to_t_gain_max=2.0,
            pel_cop_slope_per_c=0.05,
            pel_cop_reference_outdoor_c=7.0,
            pel_cop_floor=1.0,
            pel_cop_cap=8.0,
        )

        off = _rollout(pel=jnp.zeros((1,)), **common)
        on = _rollout(pel=jnp.asarray([10.0]), **common)

        self.assertGreater(float(on[0]), float(off[0]))


if __name__ == "__main__":
    unittest.main()
