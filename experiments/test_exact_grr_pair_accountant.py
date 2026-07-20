from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

import exact_grr_pair_accountant as grr_account
import large_exact_real_utility as binary_account


def brute_force_profile(
    *,
    release_length: int,
    domain_size: int,
    eps0: float,
    survival: float,
    central_epsilon: float,
) -> float:
    p, q = grr_account.grr_probabilities(eps0, domain_size)
    f0 = np.full(domain_size, q, dtype=float)
    f1 = np.full(domain_size, q, dtype=float)
    f0[0] = p
    f1[1] = p
    alpha = math.exp(central_epsilon)
    forward = 0.0
    reverse = 0.0
    for output in itertools.product(range(domain_size), repeat=release_length):
        reference = float(np.prod(f0[list(output)]))
        changed = 0.0
        for position in range(release_length):
            probability = 1.0
            for index, value in enumerate(output):
                probability *= f1[value] if index == position else f0[value]
            changed += probability / float(release_length)
        neighbor = (1.0 - survival) * reference + survival * changed
        forward += max(0.0, neighbor - alpha * reference)
        reverse += max(0.0, reference - alpha * neighbor)
    return max(forward, reverse)


class ExactGrrPairAccountantTests(unittest.TestCase):
    def test_conditional_tail_identity_matches_brute_force(self) -> None:
        expected = brute_force_profile(
            release_length=4,
            domain_size=3,
            eps0=1.3,
            survival=0.47,
            central_epsilon=0.2,
        )
        actual = grr_account.exact_pair_profile(
            release_length=4,
            domain_size=3,
            eps0=1.3,
            survival=0.47,
            central_epsilon=0.2,
        )
        self.assertAlmostEqual(actual, expected, places=14)

    def test_binary_specialization_matches_existing_exact_account(self) -> None:
        for M, eps0, survival, central_epsilon in (
            (31, 0.8, 1.0, 0.15),
            (71, 2.3, 0.37, 0.4),
            (200, 4.0, math.exp(-2.0), 0.7),
        ):
            expected = binary_account.exact_profile(
                M=M,
                eps0=eps0,
                survival=survival,
                central_epsilon=central_epsilon,
            )
            actual = grr_account.exact_pair_profile(
                release_length=M,
                domain_size=2,
                eps0=eps0,
                survival=survival,
                central_epsilon=central_epsilon,
            )
            self.assertAlmostEqual(actual, expected, places=12)

    def test_drop_calibration_reinvests_singleton_attenuation(self) -> None:
        ideal = grr_account.calibrate_local_epsilon(
            method="Padded Ideal",
            domain_size=32,
            epoch_size=2_000,
            release_length=4_000,
            target_epsilon=0.3,
            delta=1e-6,
        )
        drop = grr_account.calibrate_local_epsilon(
            method="Drop",
            domain_size=32,
            epoch_size=2_000,
            slot_count=2_000,
            release_length=4_000,
            target_epsilon=0.3,
            delta=1e-6,
        )
        self.assertGreater(drop.local_epsilon0, ideal.local_epsilon0)
        self.assertLessEqual(drop.profile_at_target, drop.delta)
        self.assertGreater(
            grr_account.exact_pair_profile(
                release_length=drop.release_length,
                domain_size=drop.domain_size,
                eps0=drop.infeasible_eps0,
                survival=drop.survival,
                central_epsilon=drop.target_epsilon,
            ),
            drop.delta * (1.0 - grr_account.PROFILE_RELATIVE_SLACK),
        )


if __name__ == "__main__":
    unittest.main()
