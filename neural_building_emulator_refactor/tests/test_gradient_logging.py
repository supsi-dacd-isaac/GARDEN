from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neural_building_emulator_refactor.gradient_logging import GradientUpdateLogger


class GradientLoggingTests(unittest.TestCase):
    def test_rows_are_persisted_before_close(self) -> None:
        with TemporaryDirectory() as directory:
            first = GradientUpdateLogger(Path(directory), "closed_loop_hp_lstm")
            first.log(
                epoch=1,
                batch_index=0,
                loss=0.5,
                grad_norm=1234.0,
                gradient_clip_norm=1.0,
                skip_grad_norm_above=1000.0,
                accepted=False,
                reason="grad_threshold",
            )
            with first.path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["grad_norm_pre_clip"], "1234.0")
            self.assertEqual(rows[0]["accepted"], "0")
            first.close()

            second = GradientUpdateLogger(Path(directory), "closed_loop_hp_lstm")
            self.assertNotEqual(first.path, second.path)
            second.close()


if __name__ == "__main__":
    unittest.main()
