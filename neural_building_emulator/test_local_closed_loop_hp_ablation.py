"""Tests for the building-local closed-loop HP ablation design."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from neural_building_emulator.data import ClosedLoopProfile
from neural_building_emulator.local_closed_loop_hp_ablation import (
    LocalVariant,
    balanced_random_design,
    factor_effects,
    hp_history_teacher_forcing_ratio,
    hp_temperature_teacher_forcing_ratio,
    observed_current_temperature_sequence,
    prepare_profiles,
)


class LocalClosedLoopHPAblationTests(unittest.TestCase):
    def test_prepare_profiles_uses_chronological_split_and_removes_metadata(self) -> None:
        steps = 20
        profile = ClosedLoopProfile(
            profile_id=7,
            datetime=np.arange(steps),
            metadata=np.asarray([1.0, 2.0], dtype=np.float32),
            inputs=np.arange(steps * 3, dtype=np.float32).reshape(steps, 3),
            targets=np.arange(steps * 3, dtype=np.float32).reshape(steps, 3),
            initial_temperature=np.arange(steps, dtype=np.float32)[:, np.newaxis],
        )
        prepared = prepare_profiles([profile], train_fraction=0.5)[0]

        self.assertEqual(prepared.split_index, 10)
        self.assertEqual(prepared.local.metadata.shape, (0,))
        self.assertEqual(prepared.train.targets.shape[0], 10)
        self.assertEqual(prepared.test.targets.shape[0], 10)
        np.testing.assert_array_equal(prepared.train.targets, profile.targets[:10])
        np.testing.assert_array_equal(prepared.test.targets, profile.targets[10:])

    def test_balanced_design_covers_each_level_equally(self) -> None:
        space = {
            "hp_activation_model": ["independent", "persistent_markov", "power_history"],
            "hp_training_mode": ["expected", "straight_through"],
        }
        variants = balanced_random_design(
            space,
            num_configurations=6,
            seed=13,
            base=LocalVariant(),
        )
        activation_counts = {
            level: sum(variant.hp_activation_model == level for variant in variants)
            for level in space["hp_activation_model"]
        }
        training_counts = {
            level: sum(variant.hp_training_mode == level for variant in variants)
            for level in space["hp_training_mode"]
        }

        self.assertEqual(variants[0].hp_activation_model, "independent")
        self.assertEqual(variants[0].hp_training_mode, "expected")
        self.assertEqual(activation_counts, {"independent": 2, "persistent_markov": 2, "power_history": 2})
        self.assertEqual(training_counts, {"expected": 3, "straight_through": 3})
        self.assertEqual(len(set(variants)), len(variants))

    def test_default_randomized_design_has_no_duplicate_configurations(self) -> None:
        space = {
            "hp_activation_model": ["independent", "persistent_markov", "power_history"],
            "hp_training_mode": ["expected", "straight_through"],
            "process_noise_mode": ["none", "heteroscedastic"],
            "temperature_update_mode": ["delta", "leaky_equilibrium"],
        }
        variants = balanced_random_design(
            space,
            num_configurations=12,
            seed=13,
            base=LocalVariant(),
        )

        self.assertEqual(len(set(variants)), 12)

    def test_pointwise_pel_weight_is_an_ablation_factor(self) -> None:
        variants = balanced_random_design(
            {"pointwise_pel_weight": [1.0, 0.1, 0.0]},
            num_configurations=3,
            seed=13,
            base=LocalVariant(),
        )

        self.assertEqual(
            {variant.pointwise_pel_weight for variant in variants},
            {0.0, 0.1, 1.0},
        )

    def test_factor_effects_first_average_over_buildings(self) -> None:
        runs = pd.DataFrame(
            {
                "config_id": ["a", "a", "b", "b"],
                "profile_id": [1, 2, 1, 2],
                "hp_training_mode": ["expected", "expected", "straight_through", "straight_through"],
                "continuous_test_rmse_c": [1.0, 3.0, 5.0, 7.0],
            }
        )
        effects = factor_effects(
            runs,
            {"hp_training_mode": ["expected", "straight_through"]},
        )
        expected = effects[
            (effects["metric"] == "continuous_test_rmse_c")
            & (effects["level"] == "expected")
        ].iloc[0]
        straight_through = effects[
            (effects["metric"] == "continuous_test_rmse_c")
            & (effects["level"] == "straight_through")
        ].iloc[0]

        self.assertAlmostEqual(float(expected["mean"]), 2.0)
        self.assertAlmostEqual(float(expected["mean_effect_vs_other_levels"]), -4.0)
        self.assertAlmostEqual(float(straight_through["mean"]), 6.0)
        self.assertAlmostEqual(float(straight_through["mean_effect_vs_other_levels"]), 4.0)

    def test_scheduled_history_feedback_anneals_to_free_running(self) -> None:
        variant = LocalVariant(
            hp_activation_model="power_history",
            hp_history_feedback="scheduled",
        )
        args = SimpleNamespace(
            hp_history_teacher_forcing_anneal_epochs=5,
            hp_history_teacher_forcing_start=1.0,
            hp_history_teacher_forcing_end=0.0,
            epochs=8,
        )

        self.assertAlmostEqual(hp_history_teacher_forcing_ratio(variant, args, 0), 1.0)
        self.assertAlmostEqual(hp_history_teacher_forcing_ratio(variant, args, 2), 0.5)
        self.assertAlmostEqual(hp_history_teacher_forcing_ratio(variant, args, 4), 0.0)
        self.assertAlmostEqual(hp_history_teacher_forcing_ratio(variant, args, 7), 0.0)

    def test_observed_history_feedback_requires_power_history_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "power_history"):
            balanced_random_design(
                {
                    "hp_activation_model": ["independent"],
                    "hp_history_feedback": ["teacher_forced"],
                },
                num_configurations=1,
                seed=13,
                base=LocalVariant(),
            )

    def test_multiscale_history_requires_power_history_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "power_history"):
            balanced_random_design(
                {
                    "hp_activation_model": ["independent"],
                    "hp_history_representation": ["multiscale"],
                },
                num_configurations=1,
                seed=13,
                base=LocalVariant(),
            )

    def test_observed_current_temperature_alignment(self) -> None:
        initial = np.asarray([[10.0], [20.0]], dtype=np.float32)
        targets = np.zeros((2, 4, 3), dtype=np.float32)
        targets[0, :, 0] = [11.0, 12.0, 13.0, 14.0]
        targets[1, :, 0] = [21.0, 22.0, 23.0, 24.0]

        aligned = np.asarray(observed_current_temperature_sequence(initial, targets))

        np.testing.assert_array_equal(aligned[0], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_array_equal(aligned[1], [20.0, 21.0, 22.0, 23.0])

    def test_scheduled_temperature_feedback_anneals_to_predicted_temperature(self) -> None:
        variant = LocalVariant(hp_temperature_feedback="scheduled")
        args = SimpleNamespace(
            hp_temperature_teacher_forcing_anneal_epochs=3,
            hp_temperature_teacher_forcing_start=1.0,
            hp_temperature_teacher_forcing_end=0.0,
            epochs=6,
        )

        self.assertAlmostEqual(hp_temperature_teacher_forcing_ratio(variant, args, 0), 1.0)
        self.assertAlmostEqual(hp_temperature_teacher_forcing_ratio(variant, args, 1), 0.5)
        self.assertAlmostEqual(hp_temperature_teacher_forcing_ratio(variant, args, 2), 0.0)
        self.assertAlmostEqual(hp_temperature_teacher_forcing_ratio(variant, args, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
