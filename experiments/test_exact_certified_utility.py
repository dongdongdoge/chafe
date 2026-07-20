from __future__ import annotations

import math
import unittest
from fractions import Fraction

import exact_certified_utility as ecu


F = Fraction


class ExactCertifiedUtilityTests(unittest.TestCase):
    def test_exact_grid_calibration_is_certified_and_maximal(self) -> None:
        for method in ecu.METHODS:
            calibration, trace = ecu.calibrate(method)
            self.assertLessEqual(calibration.exact_alpha, ecu.TARGET_ALPHA)
            self.assertIsNotNone(calibration.next_exact_alpha)
            self.assertGreater(calibration.next_exact_alpha, ecu.TARGET_ALPHA)
            self.assertLessEqual(
                ecu.TARGET_EPSILON
                - ecu.epsilon_from_alpha_fraction(calibration.exact_alpha),
                1e-3,
            )
            feasible = [
                F(row["truthful_probability_fraction"])
                for row in trace
                if row["certified_at_common_target"] == "true"
            ]
            self.assertEqual(max(feasible), calibration.truthful_probability)

    def test_exact_utility_laws_are_complete_releases(self) -> None:
        for method in ecu.METHODS:
            probability = ecu.calibrate(method)[0].truthful_probability
            distribution = ecu.exact_utility_distribution(method, probability)
            self.assertEqual(sum(distribution.values(), F(0)), F(1))
            self.assertTrue(
                all(sum(histogram) == ecu.RELEASE_M for histogram in distribution)
            )
            self.assertTrue(
                all(len(histogram) == ecu.PAYLOAD_K for histogram in distribution)
            )
            mae, mse = ecu.exact_utility_metrics(distribution, method, probability)
            self.assertGreaterEqual(mae, 0)
            self.assertGreaterEqual(mse, 0)
            self.assertLessEqual(mae, ecu.PUBLIC_M)

    def test_independent_complete_release_simulation_matches_exact_mae(self) -> None:
        for method in ecu.METHODS:
            probability = ecu.calibrate(method)[0].truthful_probability
            distribution = ecu.exact_utility_distribution(method, probability)
            exact_mae, _ = ecu.exact_utility_metrics(
                distribution, method, probability
            )
            mc_mae, standard_error, _, _ = ecu.simulate_complete_release(
                method, probability, replicates=10_000
            )
            self.assertTrue(math.isfinite(mc_mae))
            self.assertGreater(standard_error, 0)
            self.assertLessEqual(
                abs(mc_mae - float(exact_mae)),
                max(6.0 * standard_error, 0.025),
            )


if __name__ == "__main__":
    unittest.main()
