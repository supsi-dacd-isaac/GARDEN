from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from neural_building_emulator.scaling import StandardScaler, WindowScalers
from neural_building_emulator_refactor.artifacts import load_artifact
from neural_building_emulator_refactor.campaign_benchmark import POLICIES, validate_distributions
from neural_building_emulator_refactor.complete_campaign_benchmark import VARIANTS, job_plan
from neural_building_emulator_refactor.config import ExperimentConfig, OptimizerConfig, ProbabilisticLSTMConfig
from neural_building_emulator_refactor.probabilistic_lstm_trainer import train_probabilistic_lstm
from neural_building_emulator_refactor.registry import get_model_spec
from neural_building_emulator_refactor.trainer import _prepare_data


class CompletionTests(unittest.TestCase):
    def test_job_plan_only_trains_missing_models_and_never_calibrates_tariff(self):
        args = SimpleNamespace(output_dir=Path('/tmp/new'), campaign_root=Path('/tmp/data'),
                               completed_structured_run=Path('/tmp/old'))
        jobs = job_plan(args)
        self.assertEqual([n for n, _ in jobs if n.startswith('train_')],
                         ['train_continuous', 'train_hurdle_markov'])
        self.assertEqual(sum(n.startswith('score_') for n, _ in jobs), 24)
        self.assertEqual(sum(n.startswith('lightgbm') for n, _ in jobs), 3)
        for policy in POLICIES[2:]:
            self.assertFalse(any(policy in n and 'calibration' in n for n, _ in jobs))

    def test_five_model_validation_rejects_a_missing_comparator(self):
        from neural_building_emulator_refactor.tests.test_campaign_benchmark import CampaignTests
        frame = CampaignTests.samples([5, 6]).replace({'method': {'structured': 'baseline'}})
        frame = pd.concat([frame, frame[frame.method.eq('baseline')].assign(method='two_tanks')])
        validate_distributions(frame, [5, 6], 20, (.5, 1., 2., 3.), (*VARIANTS, 'lightgbm'))
        with self.assertRaisesRegex(ValueError, 'Incomplete model roster'):
            validate_distributions(frame[frame.method.ne('two_tanks')], [5, 6], 20,
                                   (.5, 1., 2., 3.), (*VARIANTS, 'lightgbm'))

    def test_dhw_metadata_reaches_lstm_dataset_loader(self):
        config = ExperimentConfig(model_name='closed_loop_hp_vanilla_probabilistic_lstm', hp_dhw_metadata=True)
        with patch('neural_building_emulator_refactor.trainer.load_closed_loop_training_profiles',
                   side_effect=RuntimeError('loader boundary')) as loader:
            with self.assertRaisesRegex(RuntimeError, 'loader boundary'):
                _prepare_data(config, get_model_spec(config.model_name))
            self.assertTrue(loader.call_args.args[0].include_dhw_metadata)

    def test_observer_stops_both_lstm_trainers_without_replacing_best_checkpoint(self):
        windows = SimpleNamespace(metadata=np.zeros((2, 3), np.float32),
            inputs=np.zeros((2, 16, 5), np.float32), targets=np.ones((2, 16, 3), np.float32),
            initial_temperature=np.zeros((2, 1), np.float32), profile_ids=np.array([1, 1]))
        windows.inputs[:, :, 3] = 1.
        splits = SimpleNamespace(input_columns=['tset', 'weather', 'dhw', 'space_heating_available', 'hp_force_off_requested'],
            metadata_columns=['a', 'b', 'c'], selected_ids=[1, 2], train_ids=[1], test_ids=[2])
        scale = lambda n: StandardScaler(np.zeros(n, np.float32), np.ones(n, np.float32))
        scalers = WindowScalers(scale(3), scale(5), scale(3))
        def metrics(value):
            return dict(total_nrmse=value, temperature_rmse_c=value,
                        qroom_rmse_w_m2=value, pel_rmse_w_m2=value)
        for name in ('closed_loop_hp_vanilla_probabilistic_lstm', 'closed_loop_hp_hurdle_markov_probabilistic_lstm'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                config = ExperimentConfig(model_name=name, output_dir=Path(tmp), sequence_length=16,
                    optimizer=OptimizerConfig(epochs=3, batch_size=2),
                    probabilistic_lstm=ProbabilisticLSTMConfig(hidden_dim=8, metadata_hidden_dim=8,
                        metadata_depth=1, latent_dim=2, bptt_truncate_steps=4, particles=2, eval_particles=2,
                        hp_training_mode='hard_expected_grad', setpoint_shock_timescales_hours=()))
                seen = []
                def observer(epoch, path):
                    artifact = load_artifact(path)
                    self.assertEqual(artifact.metadata['checkpoint_epoch'], epoch)
                    seen.append(epoch)
                    return epoch == 2
                module = 'neural_building_emulator_refactor.probabilistic_lstm_trainer.'
                with patch(module+'_prepare_data', return_value=(splits, [], [], windows, windows, scalers)), \
                     patch(module+'_epoch_windows', return_value=windows), \
                     patch(module+'_evaluate', side_effect=[metrics(1), metrics(1), metrics(2), metrics(2)]):
                    path = train_probabilistic_lstm(config, get_model_spec(name), epoch_observer=observer)
                self.assertEqual(seen, [1, 2])
                self.assertEqual(load_artifact(path).metadata['checkpoint_epoch'], 1)
                diagnostics = pd.read_csv(Path(tmp) / f'epoch_diagnostics_{name}.csv')
                self.assertEqual(list(diagnostics.epoch), [1, 2])
                self.assertTrue(np.isfinite(diagnostics.train_loss).all())


if __name__ == '__main__':
    unittest.main()
