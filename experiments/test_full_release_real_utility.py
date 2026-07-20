from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

import full_release_real_utility as utility


class FullReleaseWorkloadTest(unittest.TestCase):
    def test_categorical_releases_have_public_length_and_hidden_labels(self) -> None:
        reports = np.arange(256, dtype=np.int64) % 32
        with (
            mock.patch.object(utility, "SAMPLE_M", 256),
            mock.patch.object(utility, "N_SLOTS", 256),
            mock.patch.object(utility, "RELEASE_M", 512),
        ):
            ideal, ideal_survival, ideal_real = utility.complete_categorical_release(
                reports, 32, "Ideal", np.random.default_rng(7)
            )
            drop, drop_survival, drop_real = utility.complete_categorical_release(
                reports, 32, "Drop", np.random.default_rng(8)
            )

        self.assertEqual(len(ideal), 512)
        self.assertEqual(len(drop), 512)
        self.assertEqual(ideal_real, 256)
        self.assertLess(drop_real, 256)
        self.assertEqual(ideal_survival, 1.0)
        self.assertAlmostEqual(drop_survival, (1.0 - 1.0 / 256.0) ** 255)
        self.assertTrue(bool(((ideal >= 0) & (ideal < 32)).all()))
        self.assertTrue(bool(((drop >= 0) & (drop < 32)).all()))
        self.assertFalse(np.array_equal(ideal[: len(reports)], reports))

    def test_synthetic_experiment_covers_all_tasks_with_truthful_scope(self) -> None:
        with (
            mock.patch.object(utility, "SAMPLE_M", 512),
            mock.patch.object(utility, "N_SLOTS", 512),
            mock.patch.object(utility, "RELEASE_M", 1024),
            mock.patch.object(utility, "SEEDS", [0, 1, 2]),
            mock.patch.object(utility, "RANGE_QUERIES", 40),
        ):
            rows = utility.simulate_dataset(utility.synthetic_workloads())

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["dataset"] for row in rows}, {"Synthetic"})
        self.assertEqual(
            {(row["task"], row["method"]) for row in rows},
            {
                ("Range", "Ideal"),
                ("Range", "Drop"),
                ("Heavy hitter", "Ideal"),
                ("Heavy hitter", "Drop"),
                ("Sum", "Ideal"),
                ("Sum", "Pool"),
            },
        )
        for row in rows:
            self.assertTrue(math.isfinite(float(row["value"])))
            self.assertTrue(math.isfinite(float(row["standard_error"])))
            self.assertEqual(row["full_release"], "true")
            self.assertEqual(row["output_length_observed"], 1024)
            self.assertEqual(row["dummy_labels_visible"], "false")
            self.assertEqual(row["estimator_observes_real_output_count"], "false")
            self.assertEqual(row["final_release_permuted"], "true")
            self.assertEqual(row["central_epsilon_claimed"], "false")
            self.assertEqual(row["proxy_used_in_calibration"], "false")
            self.assertEqual(row["shared_local_epsilon0_across_methods"], "true")

        hh_rows = [row for row in rows if row["task"] == "Heavy hitter"]
        self.assertEqual({row["metric"] for row in hh_rows}, {"top_50_f1"})
        self.assertEqual({row["top_k"] for row in hh_rows}, {50})
        self.assertEqual(
            {row["dummy_law"] for row in hh_rows}, {"uniform_categorical"}
        )

        sum_rows = [row for row in rows if row["task"] == "Sum"]
        self.assertEqual(len(sum_rows), 2)
        for row in sum_rows:
            self.assertEqual(
                row["sum_randomizer"],
                "one_bit_randomized_response_with_affine_expectation",
            )
            self.assertEqual(row["estimator_uses_expected_dummy_contribution"], "true")
            self.assertNotEqual(float(row["dummy_mean"]), 0.0)
            self.assertTrue(math.isfinite(float(row["sum_bias"])))
            self.assertTrue(math.isfinite(float(row["sum_rmse"])))


if __name__ == "__main__":
    unittest.main()
