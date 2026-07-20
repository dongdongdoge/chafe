from __future__ import annotations

import math
import unittest
from unittest import mock

import padding_correction_utility as utility


class PaddingCorrectionUtilityTest(unittest.TestCase):
    def test_small_sweep_exposes_naive_bias_and_corrects_it(self) -> None:
        with (
            mock.patch.object(utility, "SAMPLE_M", 2_000),
            mock.patch.object(utility, "N_SLOTS", 2_000),
            mock.patch.object(utility, "TAU_GRID", (1.0, 2.0)),
            mock.patch.object(utility, "SEEDS", tuple(range(80))),
        ):
            rows = utility.simulate()

        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row["full_release"], "true")
            self.assertEqual(row["dummy_labels_visible"], "false")
            self.assertEqual(row["estimator_observes_real_output_count"], "false")
            self.assertTrue(math.isfinite(float(row["relative_bias"])))
            self.assertTrue(math.isfinite(float(row["relative_rmse"])))

        by_key = {
            (row["task"], float(row["tau"]), row["estimator"]): row
            for row in rows
        }
        for task in (utility.TASK_DROP, utility.TASK_POOL):
            naive = abs(float(by_key[(task, 2.0, utility.ESTIMATOR_NAIVE)]["relative_bias"]))
            corrected = abs(
                float(by_key[(task, 2.0, utility.ESTIMATOR_CORRECTED)]["relative_bias"])
            )
            self.assertGreater(naive, 0.5)
            self.assertLess(corrected, 0.08)


if __name__ == "__main__":
    unittest.main()
