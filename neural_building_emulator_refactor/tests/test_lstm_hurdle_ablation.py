from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from neural_building_emulator_refactor.lstm_hurdle_ablation import (
    _variant_recipe,
    _require_same,
    _validate_shared_objective,
    _gradient_diagnostics,
    _record_implementation,
    parse_args,
)


class LSTMHurdleAblationTests(unittest.TestCase):
    def test_structured_epoch_logs_report_rejected_updates(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "epoch_diagnostics.csv").write_text(
                "max_grad_norm,skipped_update_batches\n100,0\n7000,5\n300,0\n")
            self.assertEqual(_gradient_diagnostics(path / "selected"), (7000., 5.))

    def test_lstm_gradient_logs_report_rejected_updates(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "gradient_updates.csv").write_text(
                "grad_norm_pre_clip,accepted\n10,1\n2000,0\n20,1\n")
            self.assertEqual(_gradient_diagnostics(path / "selected"), (2000., 1.))

    def test_report_revision_preserves_frozen_training_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            harness = "neural_building_emulator_refactor/lstm_hurdle_ablation.py"
            frozen = {harness: "before", "model.py": "unchanged"}
            revised = {**frozen, harness: "after"}
            _record_implementation(path, frozen, report_only=False)
            _record_implementation(path, revised, report_only=True)
            self.assertEqual(json.loads((path / "implementation_hashes.json").read_text()), frozen)
            with self.assertRaises(ValueError):
                _record_implementation(path, revised, report_only=False)
            with self.assertRaises(ValueError):
                _record_implementation(path, {**revised, "model.py": "changed"}, report_only=True)

    def test_shared_recipe_accepts_only_matched_objectives(self) -> None:
        path = Path(__file__).parents[1] / "configs/calibrated_four_model_benchmark_metadata_fixed_three_energy_scores_both.json"
        recipe = json.loads(path.read_text())
        _validate_shared_objective(recipe)
        for key in ("prob_event_energy_weight", "hp_mode_loss_weight", "prob_flex_kpi_crps_weight"):
            changed = json.loads(json.dumps(recipe))
            changed["probabilistic_structured"][key] = 3.0
            with self.assertRaises(ValueError):
                _validate_shared_objective(changed)

    def test_provenance_cannot_be_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "request.json"
            _require_same(path, {"epochs": 5, "ids": (1, 2)})
            _require_same(path, {"epochs": 5, "ids": [1, 2]})
            with self.assertRaises(ValueError):
                _require_same(path, {"epochs": 10, "ids": [1, 2]})
            self.assertEqual(json.loads(path.read_text())["epochs"], 5)

    def test_structured_comparator_is_explicit_and_does_not_gain_hurdle_losses(self) -> None:
        args = parse_args(["--variants", "structured", "continuous", "hurdle_iid", "hurdle_markov"])
        self.assertEqual(args.variants[0], "structured")
        recipe = _variant_recipe({"probabilistic_lstm": {}}, "structured",
                                 hurdle_mode_loss_weight=2., hurdle_active_power_nll_weight=3.)
        self.assertEqual(recipe["probabilistic_lstm"]["hp_mode_loss_weight"], 0.)
        self.assertEqual(recipe["probabilistic_lstm"]["hp_active_power_nll_weight"], 0.)

    def test_default_comparison_keeps_all_hp_auxiliary_losses_zero(self) -> None:
        base = {
            "probabilistic_lstm": {
                "hp_mode_loss_weight": 0.1,
                "hp_active_power_nll_weight": 1.0,
                "hp_inactive_leakage_weight": 0.1,
            }
        }
        for variant in ("continuous", "hurdle_iid", "hurdle_markov"):
            settings = _variant_recipe(base, variant)["probabilistic_lstm"]
            self.assertEqual(settings["hp_mode_loss_weight"], 0.0)
            self.assertEqual(settings["hp_active_power_nll_weight"], 0.0)
            self.assertEqual(settings["hp_inactive_leakage_weight"], 0.0)

    def test_direct_hurdle_supervision_is_explicit_and_hurdle_only(self) -> None:
        base = {"probabilistic_lstm": {}}
        hurdle = _variant_recipe(
            base,
            "hurdle_markov",
            hurdle_mode_loss_weight=0.2,
            hurdle_active_power_nll_weight=0.7,
        )["probabilistic_lstm"]
        continuous = _variant_recipe(
            base,
            "continuous",
            hurdle_mode_loss_weight=0.2,
            hurdle_active_power_nll_weight=0.7,
        )["probabilistic_lstm"]
        self.assertEqual(hurdle["hp_mode_loss_weight"], 0.2)
        self.assertEqual(hurdle["hp_active_power_nll_weight"], 0.7)
        self.assertEqual(continuous["hp_mode_loss_weight"], 0.0)
        self.assertEqual(continuous["hp_active_power_nll_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
