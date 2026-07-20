from __future__ import annotations

import unittest

import numpy as np

from padded_estimators import (
    corrected_affine_total,
    corrected_grr_histogram,
    drop_survival,
    expected_pool_outputs,
    public_epoch_mean,
)


class PaddedEstimatorTest(unittest.TestCase):
    def test_drop_count_removes_nonzero_dummy_mass(self) -> None:
        m, N, M = 100, 200, 300
        target = 37.0
        retention = drop_survival(m, N)
        offset, slope, dummy_mean = 0.2, 0.6, 0.45
        expected_real = m * retention
        released_expectation = (
            retention * (offset * m + slope * target)
            + (M - expected_real) * dummy_mean
        )
        estimate = corrected_affine_total(
            released_expectation,
            epoch_size=m,
            release_length=M,
            expected_real_outputs=expected_real,
            real_report_retention=retention,
            dummy_mean=dummy_mean,
            affine_offset=offset,
            affine_slope=slope,
        )
        self.assertAlmostEqual(estimate, target)

    def test_pool_total_uses_expected_occupied_slots_not_m(self) -> None:
        m, N, M = 100, 100, 250
        target = 28.5
        offset, slope, dummy_mean = 0.25, 0.5, 1.2
        expected_real = expected_pool_outputs(m, N)
        released_expectation = (
            offset * m + slope * target + (M - expected_real) * dummy_mean
        )
        estimate = corrected_affine_total(
            released_expectation,
            epoch_size=m,
            release_length=M,
            expected_real_outputs=expected_real,
            real_report_retention=1.0,
            dummy_mean=dummy_mean,
            affine_offset=offset,
            affine_slope=slope,
        )
        self.assertAlmostEqual(estimate, target)

    def test_grr_histogram_correction_recovers_expectation(self) -> None:
        histogram = np.asarray([20.0, 30.0, 50.0])
        m, M = int(histogram.sum()), 240
        retention = 0.4
        expected_real = m * retention
        p, q = 0.7, 0.15
        dummy_law = np.asarray([0.2, 0.3, 0.5])
        released_expectation = (
            retention * (q * m + (p - q) * histogram)
            + (M - expected_real) * dummy_law
        )
        estimate = corrected_grr_histogram(
            released_expectation,
            epoch_size=m,
            release_length=M,
            expected_real_outputs=expected_real,
            real_report_retention=retention,
            dummy_law=dummy_law,
            truthful_probability=p,
            alternative_probability=q,
        )
        np.testing.assert_allclose(estimate, histogram, atol=1e-12)

    def test_mean_divides_by_public_epoch_size_not_release_length(self) -> None:
        self.assertEqual(public_epoch_mean(40.0, 100), 0.4)


if __name__ == "__main__":
    unittest.main()
