import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neural_building_emulator_refactor.fixed_epoch_campaign import (
    MODEL_KIND, assert_same_recipe, job_plan, verify_replay_checkpoint,
)


class FixedEpochTests(unittest.TestCase):
    def test_replay_refuses_recipe_changes_beyond_output_paths(self):
        before = dict(epochs=15, early_stopping_patience=None, learning_rate=.0003,
                      output_dir='/old', model_checkpoint_dir='/old/models')
        assert_same_recipe(before, dict(before, output_dir='/new', model_checkpoint_dir='/new/models'))
        for changed in (dict(before, learning_rate=.001), dict(before, epochs=10),
                        dict(before, early_stopping_patience=2)):
            with self.assertRaises(ValueError):
                assert_same_recipe(before, changed)

    def test_plan_rescores_every_structured_fold_and_transfer_without_refitting_other_models(self):
        args = SimpleNamespace(output_dir=Path('/tmp/new'), campaign_root=Path('/tmp/data'))
        jobs = job_plan(args)
        self.assertEqual([name for name, _ in jobs if name.startswith('train_')],
                         ['train_baseline', 'train_two_tanks'])
        scores = [(name, command) for name, command in jobs if name.startswith('score_')]
        self.assertEqual(len(scores), 12)
        for variant in ('baseline', 'two_tanks'):
            self.assertEqual(sum(command[-1] == variant for _, command in scores), 6)
        self.assertFalse(any('tariff' in name and 'calibration' in name for name, _ in jobs))
        self.assertFalse(any('lightgbm' in name for name, _ in jobs))
        self.assertEqual(sum(name.startswith('report_') for name, _ in jobs), 3)
        self.assertEqual(sum(name.startswith('tank_diagnostics_') for name, _ in jobs), 4)

    def checkpoints(self, root):
        metadata = dict(train_config=dict(epochs=15, early_stopping_patience=None), checkpoint_epoch=10,
                        train_ids=[1], test_ids=[2], selected_ids=[1, 2], input_columns=['tset'], metadata_columns=['area'])
        paths = [root/'old', root/'new']
        for p in paths:
            p.mkdir()
            (p/f'metadata_{MODEL_KIND}.json').write_text(json.dumps(metadata))
            (p/f'model_{MODEL_KIND}.eqx').write_bytes(b'unchanged weights')
            (p/f'scalers_{MODEL_KIND}.npz').write_bytes(b'unchanged scalers')
        return paths, metadata

    def test_checkpoint_verification_requires_same_epoch_and_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            (old, new), metadata = self.checkpoints(Path(tmp))
            self.assertTrue(verify_replay_checkpoint(old, new)['byte_identical'])
            for change in (dict(checkpoint_epoch=9), dict(test_ids=[3])):
                (new/f'metadata_{MODEL_KIND}.json').write_text(json.dumps(dict(metadata, **change)))
                with self.assertRaisesRegex(ValueError, 'Replay checkpoint mismatch'):
                    verify_replay_checkpoint(old, new)

    def test_replay_refuses_material_parameter_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            (old, new), _ = self.checkpoints(Path(tmp))
            (new/f'model_{MODEL_KIND}.eqx').write_bytes(b'different weights')
            a = SimpleNamespace(model={'weight': np.array([1.], dtype=np.float32)})
            b = SimpleNamespace(model={'weight': np.array([1.1], dtype=np.float32)})
            with patch('neural_building_emulator_refactor.artifacts.load_artifact', side_effect=[a, b]):
                with self.assertRaisesRegex(AssertionError, 'Replay weights diverged'):
                    verify_replay_checkpoint(old, new)


if __name__ == '__main__':
    unittest.main()
