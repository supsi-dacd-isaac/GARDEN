from __future__ import annotations

import unittest

from neural_building_emulator_refactor.scoring import _output_name


class ScoringOutputSuffixTests(unittest.TestCase):
    def test_suffix_is_inserted_before_extension(self) -> None:
        self.assertEqual(
            _output_name("flexibility_kpi_coverage.html", "_f"),
            "flexibility_kpi_coverage_f.html",
        )

    def test_path_like_suffix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _output_name("summary.json", "results/f")


if __name__ == "__main__":
    unittest.main()
