from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

import full_release_proxy_utility as utility


class FullReleaseProxyUtilityTest(unittest.TestCase):
    def test_proxy_calibration_reaches_common_target(self) -> None:
        self.assertEqual(utility.TARGETS, (0.08, 0.16, 0.32))
        self.assertEqual(len(utility.SEEDS), 100)
        previous_pure = -1.0
        previous_ideal = -1.0
        previous_drop = -1.0
        for target in utility.TARGETS:
            pure_eps0 = utility.calibrated_eps0("Pure IdealShuffle", target)
            ideal_eps0 = utility.calibrated_eps0("Padded Ideal", target)
            drop_eps0 = utility.calibrated_eps0("Drop", target)
            self.assertGreater(ideal_eps0, pure_eps0)
            self.assertGreater(drop_eps0, ideal_eps0)
            self.assertGreater(pure_eps0, previous_pure)
            self.assertGreater(ideal_eps0, previous_ideal)
            self.assertGreater(drop_eps0, previous_drop)
            self.assertAlmostEqual(
                utility.proxy_epsilon("Pure IdealShuffle", pure_eps0),
                target,
                places=7,
            )
            self.assertAlmostEqual(
                utility.proxy_epsilon("Padded Ideal", ideal_eps0),
                target,
                places=7,
            )
            self.assertAlmostEqual(
                utility.proxy_epsilon("Drop", drop_eps0),
                target,
                places=7,
            )
            previous_pure = pure_eps0
            previous_ideal = ideal_eps0
            previous_drop = drop_eps0

    def test_simplex_projection_is_nonnegative_and_has_public_mass(self) -> None:
        projected = utility.project_nonnegative_simplex(
            np.asarray((-3.0, 2.0, 9.0, -1.0)),
            17.0,
        )
        self.assertTrue(bool((projected >= 0.0).all()))
        self.assertAlmostEqual(float(projected.sum()), 17.0, places=10)

    def test_complete_release_has_public_length(self) -> None:
        reports = np.arange(256, dtype=np.int64) % 32
        prior = np.full(32, 1.0 / 32.0)
        with (
            mock.patch.object(utility, "SAMPLE_M", 256),
            mock.patch.object(utility, "N_SLOTS", 256),
            mock.patch.object(utility, "RELEASE_M", 512),
        ):
            pure, pure_survival, pure_real = utility.complete_release(
                reports,
                "Pure IdealShuffle",
                prior,
                np.random.default_rng(0),
                np.random.default_rng(1),
                np.random.default_rng(2),
            )
            ideal, ideal_survival, ideal_real = utility.complete_release(
                reports,
                "Padded Ideal",
                prior,
                np.random.default_rng(1),
                np.random.default_rng(2),
                np.random.default_rng(3),
            )
            drop, drop_survival, drop_real = utility.complete_release(
                reports,
                "Drop",
                prior,
                np.random.default_rng(4),
                np.random.default_rng(5),
                np.random.default_rng(6),
            )
        self.assertEqual(len(pure), 256)
        self.assertEqual(len(ideal), 512)
        self.assertEqual(len(drop), 512)
        self.assertEqual(pure_real, 256)
        self.assertEqual(ideal_real, 256)
        self.assertLess(drop_real, 256)
        self.assertEqual(pure_survival, 1.0)
        self.assertEqual(ideal_survival, 1.0)
        self.assertAlmostEqual(drop_survival, (1.0 - 1.0 / 256.0) ** 255)
        self.assertFalse(np.array_equal(ideal[: len(reports)], reports))
        self.assertTrue(np.array_equal(np.sort(pure), np.sort(reports)))
        self.assertFalse(np.array_equal(pure, reports))

    def test_synthetic_smoke_has_complete_provenance(self) -> None:
        with (
            mock.patch.object(utility, "SAMPLE_M", 512),
            mock.patch.object(utility, "N_SLOTS", 512),
            mock.patch.object(utility, "RELEASE_M", 1024),
            mock.patch.object(utility, "SEEDS", (0, 1, 2)),
            mock.patch.object(utility, "RANGE_QUERIES", 40),
        ):
            rows = utility.simulate_dataset(utility.synthetic_workloads())

        self.assertEqual(len(rows), 18)
        self.assertEqual(
            {
                (
                    float(row["target_tilde_epsilon_prime"]),
                    row["task"],
                    row["method"],
                )
                for row in rows
            },
            {
                (target, task, method)
                for target in utility.TARGETS
                for task in ("Range", "Heavy hitter")
                for method in utility.METHODS
            },
        )
        for row in rows:
            self.assertTrue(math.isfinite(float(row["value"])))
            self.assertTrue(math.isfinite(float(row["standard_error"])))
            self.assertTrue(
                math.isfinite(float(row["paired_drop_minus_padded_ideal"]))
            )
            self.assertTrue(
                math.isfinite(float(row["paired_difference_standard_error"]))
            )
            self.assertTrue(
                math.isfinite(
                    float(row["paired_drop_minus_padded_ideal_standard_error"])
                )
            )
            self.assertTrue(
                math.isfinite(float(row["paired_drop_minus_pure_ideal_shuffle"]))
            )
            self.assertTrue(
                math.isfinite(
                    float(
                        row[
                            "paired_drop_minus_pure_ideal_shuffle_standard_error"
                        ]
                    )
                )
            )
            self.assertGreater(float(row["drop_over_padded_ideal_ratio"]), 0.0)
            self.assertGreater(
                float(row["drop_over_pure_ideal_shuffle_ratio"]), 0.0
            )
            self.assertEqual(row["full_release"], "true")
            self.assertEqual(row["central_epsilon_claimed"], "false")
            self.assertEqual(row["proxy_used_in_calibration"], "true")
            self.assertEqual(row["dummy_labels_visible"], "false")
            self.assertEqual(row["final_release_permuted"], "true")
            self.assertEqual(row["paired_workload_seeds"], "true")
            self.assertEqual(row["paired_query_seeds"], "true")
            self.assertEqual(row["paired_across_proxy_targets"], "true")
            self.assertEqual(row["seeds"], 3)
            self.assertAlmostEqual(
                float(row["achieved_tilde_epsilon_prime"]),
                float(row["target_tilde_epsilon_prime"]),
                places=7,
            )
            if row["method"] == "Pure IdealShuffle":
                self.assertEqual(row["output_length_observed"], 512)
                self.assertEqual(row["M"], 512)
                self.assertEqual(row["tau"], 1.0)
                self.assertEqual(row["N"], "")
                self.assertEqual(row["rho"], "")
                self.assertEqual(row["uses_dummies"], "false")
                self.assertEqual(row["dummy_law"], "none")
                self.assertEqual(
                    row["dummy_law_matches_proxy_reference"], "not_applicable"
                )
                self.assertEqual(row["expected_dummy_count"], 0.0)
                self.assertEqual(row["observed_mean_dummy_count"], 0.0)
                self.assertEqual(row["real_output_count_visible"], "true")
                self.assertEqual(
                    row["estimator_observes_real_output_count"], "true"
                )
                self.assertEqual(
                    row["privacy_target_type"],
                    "pure_ideal_shuffle_quadratic_proxy",
                )
            else:
                self.assertEqual(row["output_length_observed"], 1024)
                self.assertEqual(row["M"], 1024)
                self.assertEqual(row["tau"], 2.0)
                self.assertEqual(row["uses_dummies"], "true")
                self.assertEqual(
                    row["dummy_law"],
                    "expected_GRR_report_law_under_fixed_public_prior",
                )
                self.assertEqual(
                    row["privacy_target_type"],
                    "matched_public_prior_quadratic_proxy",
                )
                if row["method"] == "Drop":
                    self.assertEqual(row["N"], 512)
                    self.assertEqual(row["rho"], 1.0)
                    self.assertEqual(row["real_output_count_visible"], "false")
                    self.assertEqual(
                        row["estimator_observes_real_output_count"], "false"
                    )
                else:
                    self.assertEqual(row["N"], "")
                    self.assertEqual(row["rho"], "")
                    self.assertEqual(row["real_output_count_visible"], "true")
                    self.assertEqual(
                        row["estimator_observes_real_output_count"], "true"
                    )
            if row["task"] == "Range":
                self.assertGreaterEqual(float(row["value"]), 0.0)
                self.assertLessEqual(float(row["value"]), 1.0)


if __name__ == "__main__":
    unittest.main()
