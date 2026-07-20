from __future__ import annotations

import math
import unittest
from fractions import Fraction

import numpy as np

import exhaustive_hidden_slot_oracle as oracle
import large_exact_real_utility as leru


class LargeExactRealUtilityTests(unittest.TestCase):
    def test_pair_law_matches_independent_rational_hidden_slot_oracle(self) -> None:
        f0 = (Fraction(3, 4), Fraction(1, 4))
        f1 = (Fraction(1, 4), Fraction(3, 4))
        pair = oracle.RationalNeighborPair.homogeneous(
            name="all_zero_to_one_positive",
            n=3,
            background_law=f0,
            target_law_p=f1,
            target_law_q=f0,
        )
        P, Q = oracle.exhaustive_neighbor_distributions(
            pair=pair,
            decoder="drop",
            N=4,
            M=8,
            nu=f0,
        )
        alpha = Fraction(3, 2)
        rational_profile = max(
            oracle.hockey_stick_alpha_fraction(P, Q, alpha),
            oracle.hockey_stick_alpha_fraction(Q, P, alpha),
        )
        reduced_profile = leru.exact_profile(
            M=8,
            eps0=math.log(3.0),
            survival=(3.0 / 4.0) ** 3,
            central_epsilon=math.log(1.5),
        )
        self.assertAlmostEqual(float(rational_profile), reduced_profile, places=15)

    def test_tail_identity_matches_direct_pmf(self) -> None:
        cases = [
            (31, 0.8, 1.0, 0.15),
            (71, 2.3, 0.37, 0.4),
            (200, 4.0, math.exp(-2.0), 0.7),
        ]
        for M, eps0, survival, central_epsilon in cases:
            tail = leru.exact_profile(
                M=M,
                eps0=eps0,
                survival=survival,
                central_epsilon=central_epsilon,
            )
            direct = leru.direct_pmf_profile(
                M=M,
                eps0=eps0,
                survival=survival,
                central_epsilon=central_epsilon,
            )
            self.assertAlmostEqual(tail, direct, places=13)

    def test_exact_calibration_hits_declared_target_without_proxy(self) -> None:
        calibration = leru.calibrate_local_epsilon(
            method="Drop",
            target_epsilon=0.3,
            rho=1.0,
            m=2_000,
            M=4_000,
        )
        self.assertLessEqual(calibration.profile_at_target, leru.DELTA)
        self.assertGreater(
            leru.exact_profile(
                M=4_000,
                eps0=calibration.infeasible_eps0,
                survival=calibration.survival,
                central_epsilon=0.3,
            ),
            calibration.profile_limit,
        )
        self.assertLess(abs(calibration.achieved_epsilon - 0.3), 2e-8)
        self.assertLess(
            abs(calibration.profile_at_target - calibration.direct_pmf_profile),
            2e-11,
        )

    def test_simulation_constructs_complete_hidden_release(self) -> None:
        bits = np.asarray(([0, 1, 0, 1] * 100), dtype=np.uint8)
        calibration = leru.calibrate_local_epsilon(
            method="Drop",
            target_epsilon=0.3,
            rho=1.0,
            m=len(bits),
            M=800,
        )
        observation = leru.simulate_complete_release(
            bits=bits,
            calibration=calibration,
            M=800,
            seed=19,
        )
        self.assertEqual(len(observation.release), 800)
        self.assertTrue(np.all((observation.release == 0) | (observation.release == 1)))
        self.assertGreaterEqual(observation.real_output_count, 0)
        self.assertLessEqual(observation.real_output_count, len(bits))
        self.assertGreaterEqual(observation.estimate, 0.0)
        self.assertLessEqual(observation.estimate, len(bits))

    def test_padded_ideal_and_drop_use_same_public_length(self) -> None:
        bits = np.asarray(([0, 1] * 200), dtype=np.uint8)
        ideal = leru.calibrate_local_epsilon(
            method="Padded Ideal",
            target_epsilon=0.1,
            rho=None,
            m=len(bits),
            M=800,
        )
        drop = leru.calibrate_local_epsilon(
            method="Drop",
            target_epsilon=0.1,
            rho=1.0,
            m=len(bits),
            M=800,
        )
        ideal_release = leru.simulate_complete_release(
            bits=bits, calibration=ideal, M=800, seed=31
        )
        drop_release = leru.simulate_complete_release(
            bits=bits, calibration=drop, M=800, seed=32
        )
        self.assertEqual(len(ideal_release.release), 800)
        self.assertEqual(len(drop_release.release), 800)
        self.assertGreater(drop.local_epsilon0, ideal.local_epsilon0)

    def test_exact_family_utility_law_matches_independent_simulation(self) -> None:
        calibration = leru.calibrate_local_epsilon(
            method="Drop",
            target_epsilon=0.3,
            rho=1.0,
            m=400,
            M=800,
        )
        exact = leru.exact_family_utility_metrics(
            calibration=calibration,
            true_count=1,
            m=400,
            M=800,
        )
        bits = np.zeros(400, dtype=np.uint8)
        bits[0] = 1
        errors = []
        for seed in range(2_000):
            observation = leru.simulate_complete_release(
                bits=bits,
                calibration=calibration,
                M=800,
                seed=50_000 + seed,
            )
            errors.append(abs(observation.estimate - 1.0))
        mc_mae = float(np.mean(errors))
        mc_se = float(np.std(errors, ddof=1) / math.sqrt(len(errors)))
        self.assertLess(exact["mass_error"], 1e-13)
        self.assertLess(
            abs(mc_mae - exact["mae"]),
            max(6.0 * mc_se, 0.02),
        )


if __name__ == "__main__":
    unittest.main()
