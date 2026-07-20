from __future__ import annotations

import math
import unittest

from chafe_eval_lib import brr_probs
from privacy_large_m_accounting import (
    account_chernoff_pair,
    account_pair,
    binary_slot_envelope,
    chernoff_directed_profile,
    clone_shuffle_numerical_upper,
    exact_directed_profile,
    poisson_case_laws,
)


class ChernoffPrivacyAccountTest(unittest.TestCase):
    def test_directed_bound_dominates_exact_profile(self) -> None:
        f0 = brr_probs(1.0, 0)
        f1 = brr_probs(1.0, 1)
        for decoder in ("drop", "pool"):
            for adjacency in ("replacement", "add_remove"):
                laws = poisson_case_laws(decoder, adjacency, 1.0, f0, f1)
                for numerator, denominator in (
                    (laws.left, laws.right),
                    (laws.right, laws.left),
                ):
                    exact = exact_directed_profile(
                        laws.reference, numerator, denominator, 257
                    )
                    bound = chernoff_directed_profile(
                        laws.reference, numerator, denominator, 257
                    )
                    for epsilon in (0.0, 0.01, 0.05, 0.1, 0.25):
                        with self.subTest(
                            decoder=decoder,
                            adjacency=adjacency,
                            epsilon=epsilon,
                        ):
                            alpha = math.exp(epsilon)
                            self.assertGreaterEqual(
                                bound(alpha) + 1e-12, exact(alpha)
                            )

    def test_two_sided_inversion_is_conservative(self) -> None:
        f0 = brr_probs(1.0, 0)
        f1 = brr_probs(1.0, 1)
        laws = poisson_case_laws("pool", "add_remove", 1.0, f0, f1)
        exact = account_pair(laws, 20_000, 1e-6, "exact")
        bound = account_chernoff_pair(
            laws.reference, laws.left, laws.right, 20_000, 1e-6
        )
        self.assertTrue(bound["feasible"])
        self.assertGreaterEqual(
            float(bound["two_sided"]), exact["two_sided"]
        )
        self.assertLessEqual(float(bound["profile_bound"]), 1.000000001e-6)


class MechanismWideSlotEnvelopeTest(unittest.TestCase):
    def test_binary_family_extrema_cover_all_background_mixtures(self) -> None:
        f0 = brr_probs(1.0, 0)
        f1 = brr_probs(1.0, 1)
        drop = binary_slot_envelope("drop", 1.0, f0, f1)
        pool = binary_slot_envelope("pool", 1.0, f0, f1)
        self.assertAlmostEqual(drop.epsilon_slot, 0.4898801256, places=9)
        self.assertAlmostEqual(pool.epsilon_slot, 0.6572683215, places=9)
        self.assertLess(drop.p_min, drop.p_max)
        self.assertLess(pool.p_min, pool.p_max)

    def test_mechanism_wide_upper_dominates_matched_pair(self) -> None:
        f0 = brr_probs(1.0, 0)
        f1 = brr_probs(1.0, 1)
        release_length = 20_000
        for decoder in ("drop", "pool"):
            envelope = binary_slot_envelope(decoder, 1.0, f0, f1)
            upper = clone_shuffle_numerical_upper(
                release_length, round(envelope.epsilon_slot, 12), 1e-6
            )
            laws = poisson_case_laws(
                decoder, "replacement", 1.0, f0, f1
            )
            exact = account_pair(laws, release_length, 1e-6, "exact")[
                "two_sided"
            ]
            self.assertGreaterEqual(upper + 1e-12, exact)


if __name__ == "__main__":
    unittest.main()
