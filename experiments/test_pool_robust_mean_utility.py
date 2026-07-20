from __future__ import annotations

import unittest

import numpy as np

import pool_robust_mean_utility as workload


class PoolRobustMeanUtilityTest(unittest.TestCase):
    def test_bounded_heavy_tail_stays_in_declared_domain(self) -> None:
        rng = np.random.default_rng(11)
        for center in workload.CLEAN_CENTERS:
            values = workload.bounded_heavy_tail(rng, center, 10_000)
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)
            self.assertLess(abs(float(values.mean()) - center), 0.01)

    def test_huber_location_limits_endpoint_contamination(self) -> None:
        clean = np.linspace(0.28, 0.32, 900)
        contaminated = np.concatenate((clean, np.ones(100)))
        robust = workload.huber_location(contaminated)
        ordinary = float(contaminated.mean())
        self.assertLess(abs(robust - 0.30), abs(ordinary - 0.30))

    def test_pool_release_has_fixed_length_and_no_zero_count_label(self) -> None:
        rng = np.random.default_rng(12)
        reports = np.linspace(-0.5, 1.5, workload.SAMPLE_M)
        counts, sums, metadata = workload.complete_pool_release(
            rng, reports, rho=4.0, eps0=8.0
        )
        self.assertEqual(len(counts), int(metadata["N"]))
        self.assertEqual(len(sums), int(metadata["M"]))
        self.assertTrue(bool((counts > 0).all()))
        self.assertEqual(
            int(metadata["real_output_count"] + metadata["dummy_count"]),
            len(counts),
        )

    def test_padded_ideal_release_hides_positions_by_permutation(self) -> None:
        rng = np.random.default_rng(13)
        reports = np.arange(workload.SAMPLE_M, dtype=float)
        release, dummy_count = workload.complete_padded_ideal_release(
            rng, reports, eps0=8.0
        )
        self.assertEqual(
            len(release), int(np.ceil(workload.IDEAL_TAU * workload.SAMPLE_M))
        )
        self.assertGreater(dummy_count, 0)
        self.assertFalse(np.array_equal(release[: workload.SAMPLE_M], reports))


if __name__ == "__main__":
    unittest.main()
