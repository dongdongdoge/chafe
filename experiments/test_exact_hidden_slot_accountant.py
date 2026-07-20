from __future__ import annotations

import math
import unittest
from fractions import Fraction

try:
    from .exact_hidden_slot_accountant import (
        NeighborPair,
        _FiniteRecurrence,
        _one_sided_min_alpha,
        all_background_sizes_add_remove_pair_family,
        all_sizes_replacement_pair_family,
        account_pair,
        account_pair_family,
        fixed_background_add_remove_pair_family,
        fixed_size_replacement_pair_family,
        hockey_stick_alpha,
        two_sided_min_alpha,
        two_sided_profile_alpha,
    )
    from .exhaustive_hidden_slot_oracle import (
        RationalNeighborPair,
        exhaustive_neighbor_distributions,
        finite_matched_dummy_law_fraction,
        log_fraction_safe,
        two_sided_min_alpha_fraction,
    )
except ImportError:
    from exact_hidden_slot_accountant import (
        NeighborPair,
        _FiniteRecurrence,
        _one_sided_min_alpha,
        all_background_sizes_add_remove_pair_family,
        all_sizes_replacement_pair_family,
        account_pair,
        account_pair_family,
        fixed_background_add_remove_pair_family,
        fixed_size_replacement_pair_family,
        hockey_stick_alpha,
        two_sided_min_alpha,
        two_sided_profile_alpha,
    )
    from exhaustive_hidden_slot_oracle import (
        RationalNeighborPair,
        exhaustive_neighbor_distributions,
        finite_matched_dummy_law_fraction,
        log_fraction_safe,
        two_sided_min_alpha_fraction,
    )


F = Fraction


def float_law(law):
    return tuple(float(value) for value in law)


def float_pair(pair: RationalNeighborPair) -> NeighborPair:
    return NeighborPair(
        name=pair.name,
        background_counts=pair.background_counts,
        background_laws=tuple(float_law(law) for law in pair.background_laws),
        target_law_p=float_law(pair.target_law_p),
        target_law_q=(float_law(pair.target_law_q) if pair.target_law_q else None),
        family_scope=pair.family_scope,
    )


class ExactHiddenSlotAccountantTest(unittest.TestCase):
    def assert_matches_oracle(
        self,
        pair: RationalNeighborPair,
        decoder: str,
        N: int,
        M: int,
        nu,
        delta: Fraction = F(1, 10_000),
    ) -> None:
        oracle_P, oracle_Q = exhaustive_neighbor_distributions(
            pair, decoder, N, M, nu
        )
        result = account_pair(
            float_pair(pair), decoder, N, M, float_law(nu), float(delta)
        )
        support = set(oracle_P) | set(oracle_Q) | set(result.P) | set(result.Q)
        self.assertTrue(support)
        for output in support:
            self.assertAlmostEqual(
                result.P.get(output, 0.0),
                float(oracle_P.get(output, F(0))),
                places=13,
                msg=f"P mismatch for {decoder}, output={output}",
            )
            self.assertAlmostEqual(
                result.Q.get(output, 0.0),
                float(oracle_Q.get(output, F(0))),
                places=13,
                msg=f"Q mismatch for {decoder}, output={output}",
            )
            self.assertEqual(sum(output), M)
        exact_alpha = two_sided_min_alpha_fraction(oracle_P, oracle_Q, delta)
        self.assertIsNotNone(exact_alpha)
        self.assertAlmostEqual(result.alpha_nominal, float(exact_alpha), places=12)
        self.assertLessEqual(result.profile_at_nominal, float(delta))
        self.assertEqual(result.diagnostics.underflowed_positive_terms, 0)
        observed_l1 = sum(
            abs(result.P.get(output, 0.0) - float(oracle_P.get(output, F(0))))
            + abs(result.Q.get(output, 0.0) - float(oracle_Q.get(output, F(0))))
            for output in support
        )
        self.assertLessEqual(
            observed_l1,
            2.0 * result.float_l1_error_diagnostic_each_distribution + 1e-15,
        )

    def test_homogeneous_matched_and_mismatched_slotwise_padding(self) -> None:
        background = (F(1, 2), F(1, 3), F(1, 6))
        target = (F(1, 5), F(1, 5), F(3, 5))
        pair = RationalNeighborPair.homogeneous(
            "homogeneous", 3, background, target
        )
        mismatch = (F(1, 7), F(2, 7), F(4, 7))
        for decoder in ("drop", "pool"):
            matched = finite_matched_dummy_law_fraction(
                decoder, 3, 2, background
            )
            for nu in (matched, mismatch):
                with self.subTest(decoder=decoder, nu=nu):
                    self.assert_matches_oracle(pair, decoder, 2, 3, nu)

    def test_heterogeneous_dataset_matches_every_probability(self) -> None:
        laws = (
            (F(3, 5), F(3, 10), F(1, 10)),
            (F(1, 10), F(1, 5), F(7, 10)),
        )
        pair = RationalNeighborPair(
            name="heterogeneous",
            background_counts=(2, 1),
            background_laws=laws,
            target_law_p=laws[1],
        )
        mismatch = (F(1, 7), F(2, 7), F(4, 7))
        for decoder in ("drop", "pool"):
            with self.subTest(decoder=decoder):
                self.assert_matches_oracle(pair, decoder, 2, 3, mismatch)

    def test_general_padding_path_when_M_is_less_than_N(self) -> None:
        background = (F(1, 2), F(1, 3), F(1, 6))
        target = (F(1, 5), F(1, 5), F(3, 5))
        pair = RationalNeighborPair.homogeneous(
            "general-padding", 1, background, target
        )
        nu = (F(1, 7), F(2, 7), F(4, 7))
        for decoder in ("drop", "pool"):
            with self.subTest(decoder=decoder):
                self.assert_matches_oracle(pair, decoder, 3, 2, nu)

    def test_replacement_adjacency_matches_fraction_oracle(self) -> None:
        background = (F(1, 2), F(1, 3), F(1, 6))
        target_q = (F(3, 5), F(3, 10), F(1, 10))
        target_p = (F(1, 10), F(1, 5), F(7, 10))
        pair = RationalNeighborPair.homogeneous(
            "replacement", 3, background, target_p, target_q
        )
        mismatch = (F(1, 7), F(2, 7), F(4, 7))
        for decoder in ("drop", "pool"):
            with self.subTest(decoder=decoder):
                self.assert_matches_oracle(pair, decoder, 2, 3, mismatch)

    def test_mechanism_family_worst_case_is_explicit_maximum(self) -> None:
        laws = [
            (0.6, 0.3, 0.1),
            (0.1, 0.2, 0.7),
        ]
        pairs = fixed_background_add_remove_pair_family(3, laws, "mechanism")
        self.assertEqual(len(pairs), 8)
        family = account_pair_family(
            pairs,
            "drop",
            N=2,
            M=3,
            nu=(1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0),
            delta=1e-4,
        )
        self.assertAlmostEqual(
            family.alpha_nominal,
            max(result.alpha_nominal for result in family.pair_results),
            places=14,
        )
        self.assertIn(
            family.worst_pair_nominal,
            {result.pair_name for result in family.pair_results},
        )
        self.assertTrue(math.isfinite(family.epsilon_nominal))
        self.assertEqual(family.family_scope, "fixed_background_size_add_remove")

    def test_arbitrary_public_nu_with_support_mismatch_returns_infinity(self) -> None:
        pair = RationalNeighborPair.homogeneous(
            "singular",
            0,
            (F(1), F(0)),
            (F(0), F(1)),
        )
        for decoder in ("drop", "pool"):
            oracle_P, oracle_Q = exhaustive_neighbor_distributions(
                pair, decoder, N=1, M=1, nu=(F(1), F(0))
            )
            exact_alpha = two_sided_min_alpha_fraction(
                oracle_P, oracle_Q, F(1, 10)
            )
            self.assertIsNone(exact_alpha)
            result = account_pair(
                float_pair(pair),
                decoder,
                N=1,
                M=1,
                nu=(1.0, 0.0),
                delta=0.1,
            )
            self.assertTrue(math.isinf(result.alpha_nominal))
            self.assertEqual(result.singular_delta, 1.0)

    def test_family_builders_have_unambiguous_size_scope(self) -> None:
        laws = [(0.75, 0.25), (0.25, 0.75)]
        fixed_add_remove = fixed_background_add_remove_pair_family(3, laws)
        all_add_remove = all_background_sizes_add_remove_pair_family(2, laws)
        fixed_replacement = fixed_size_replacement_pair_family(3, laws)
        all_replacement = all_sizes_replacement_pair_family(3, laws)
        self.assertEqual(len(fixed_add_remove), 8)
        self.assertEqual(len(all_add_remove), 12)
        self.assertEqual(len(fixed_replacement), 6)
        self.assertEqual(len(all_replacement), 12)
        self.assertEqual(
            {pair.family_scope for pair in fixed_add_remove},
            {"fixed_background_size_add_remove"},
        )
        self.assertEqual(
            {pair.family_scope for pair in all_add_remove},
            {"all_background_sizes_add_remove"},
        )
        self.assertEqual(
            {pair.family_scope for pair in fixed_replacement},
            {"fixed_public_size_replacement"},
        )
        self.assertEqual(
            {pair.family_scope for pair in all_replacement},
            {"all_public_sizes_replacement"},
        )

    def test_distinct_nearby_likelihood_ratios_are_not_grouped(self) -> None:
        d = F(1, 2**50)
        Q_fraction = {("a",): F(1, 3), ("b",): F(1, 3), ("c",): F(1, 3)}
        P_fraction = {
            ("a",): (1 + d) / 3,
            ("b",): (1 + 2 * d) / 3,
            ("c",): (1 - 3 * d) / 3,
        }
        P = {key: float(value) for key, value in P_fraction.items()}
        Q = {key: float(value) for key, value in Q_fraction.items()}
        delta = float(d / 6)
        ratios = sorted(P[key] / Q[key] for key in (("a",), ("b",)))
        self.assertLess(ratios[1] - ratios[0], 1e-15)
        self.assertNotEqual(ratios[0], ratios[1])
        alpha = _one_sided_min_alpha(P, Q, delta)
        self.assertGreater(alpha, ratios[0])
        self.assertLess(alpha, ratios[1])
        self.assertLessEqual(hockey_stick_alpha(P, Q, alpha), delta + 1e-30)

    def test_float_alpha_is_repaired_to_feasible_endpoint(self) -> None:
        laws = ((0.75, 0.25), (0.25, 0.75))
        pair = NeighborPair(
            name="repair-regression",
            background_counts=(2, 0),
            background_laws=laws,
            target_law_p=laws[1],
            target_law_q=laws[0],
        )
        result = account_pair(
            pair,
            "pool",
            N=2,
            M=3,
            nu=(0.5, 0.5),
            delta=1e-3,
        )
        self.assertLessEqual(result.profile_at_nominal, 1e-3)

    def test_close_breakpoint_audit_counterexample_is_feasible(self) -> None:
        q = 0.1
        r1 = 1.0 + 1e-12
        r2 = math.nextafter(math.nextafter(r1, math.inf), math.inf)
        Q = {("a",): q, ("b",): q, ("c",): 1.0 - 2.0 * q}
        P = {("a",): q * r1, ("b",): q * r2}
        P[("c",)] = 1.0 - P[("a",)] - P[("b",)]
        delta = q * (r2 - r1) / 2.0
        alpha = two_sided_min_alpha(P, Q, delta)
        self.assertLessEqual(two_sided_profile_alpha(P, Q, alpha), delta)
        pair = NeighborPair(
            name="close-breakpoint-mechanism-image",
            background_counts=(0,),
            background_laws=((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),),
            target_law_p=(P[("a",)], P[("b",)], P[("c",)]),
            target_law_q=(Q[("a",)], Q[("b",)], Q[("c",)]),
        )
        for decoder in ("drop", "pool"):
            result = account_pair(
                pair,
                decoder,
                N=1,
                M=1,
                nu=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                delta=delta,
            )
            self.assertLessEqual(result.profile_at_nominal, delta)

    def test_dummy_histograms_are_lazy_and_final_count_only(self) -> None:
        pair = NeighborPair.homogeneous(
            "lazy-dummy",
            1,
            (0.75, 0.25),
            (0.25, 0.75),
        )
        recurrence = _FiniteRecurrence(pair, "drop", N=2, M=100, nu=(0.5, 0.5))
        self.assertEqual(set(recurrence._dummy_histogram_cache), {0})
        recurrence.output_distributions()
        self.assertEqual(set(recurrence._dummy_histogram_cache), {0, 98})

    def test_nonintegral_and_negative_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NeighborPair.homogeneous("bad-n", 1.5, (1.0,), (1.0,))
        pair = NeighborPair.homogeneous("ok", 0, (1.0,), (1.0,))
        with self.assertRaises(ValueError):
            account_pair(pair, "drop", N=1.5, M=1, nu=(1.0,), delta=0.1)
        with self.assertRaises(ValueError):
            account_pair(pair, "drop", N=1, M=-1, nu=(1.0,), delta=0.1)
        bad_count = NeighborPair(
            name="bad-count",
            background_counts=(-1,),
            background_laws=((1.0,),),
            target_law_p=(1.0,),
        )
        with self.assertRaises(ValueError):
            account_pair(bad_count, "drop", N=1, M=1, nu=(1.0,), delta=0.1)
        bad_oracle = RationalNeighborPair(
            name="bad-oracle-count",
            background_counts=(-1,),
            background_laws=((F(1),),),
            target_law_p=(F(1),),
        )
        with self.assertRaises(ValueError):
            exhaustive_neighbor_distributions(
                bad_oracle, "drop", N=1, M=1, nu=(F(1),)
            )
        with self.assertRaises(ValueError):
            exhaustive_neighbor_distributions(
                RationalNeighborPair.homogeneous(
                    "oracle", 0, (F(1),), (F(1),)
                ),
                "drop",
                N=1.25,
                M=1,
                nu=(F(1),),
            )

    def test_large_fraction_log_does_not_convert_to_float_first(self) -> None:
        value = F(10**10_000, 3)
        observed = log_fraction_safe(value)
        expected = 10_000 * math.log(10.0) - math.log(3.0)
        self.assertTrue(math.isfinite(observed))
        self.assertAlmostEqual(observed, expected, places=10)


if __name__ == "__main__":
    unittest.main()
