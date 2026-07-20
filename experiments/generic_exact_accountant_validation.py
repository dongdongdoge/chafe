from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from chafe_eval_lib import DATA_DIR, write_csv
from exact_hidden_slot_accountant import (
    NeighborPair,
    account_pair_family,
    enumerate_type_count_vectors,
)
from exhaustive_hidden_slot_oracle import (
    RationalNeighborPair,
    epsilon_from_alpha_fraction,
    exhaustive_neighbor_distributions,
    hockey_stick_alpha_fraction,
    two_sided_min_alpha_fraction,
)


F = Fraction
DELTA = F(1, 1_000)
BRR_F0 = (F(3, 4), F(1, 4))
BRR_F1 = (F(1, 4), F(3, 4))
UNIFORM_NU = (F(1, 2), F(1, 2))
SKEWED_NU = (F(7, 10), F(3, 10))


def float_law(law: Sequence[Fraction]) -> tuple[float, ...]:
    return tuple(float(value) for value in law)


def float_pair(pair: RationalNeighborPair) -> NeighborPair:
    return NeighborPair(
        name=pair.name,
        background_counts=pair.background_counts,
        background_laws=tuple(float_law(law) for law in pair.background_laws),
        target_law_p=float_law(pair.target_law_p),
        target_law_q=(
            float_law(pair.target_law_q)
            if pair.target_law_q is not None
            else None
        ),
        family_scope=pair.family_scope,
    )


def fraction_text(value: Fraction | None) -> str:
    if value is None:
        return "inf"
    return f"{value.numerator}/{value.denominator}"


def law_text(law: Sequence[Fraction]) -> str:
    return ";".join(fraction_text(value) for value in law)


def rational_mechanism_family(
    background_size: int,
    input_type_laws: Sequence[Sequence[Fraction]],
    prefix: str,
    family_scope: str = "fixed_background_size_add_remove",
) -> list[RationalNeighborPair]:
    laws = tuple(tuple(F(value) for value in law) for law in input_type_laws)
    pairs = []
    for counts in enumerate_type_count_vectors(background_size, len(laws)):
        counts_text = "_".join(str(value) for value in counts)
        for target_type, target_law in enumerate(laws):
            pairs.append(
                RationalNeighborPair(
                    name=f"{prefix}_counts_{counts_text}_target_{target_type}",
                    background_counts=counts,
                    background_laws=laws,
                    target_law_p=target_law,
                    family_scope=family_scope,
                )
            )
    return pairs


def exact_two_sided_profile(P, Q, alpha: Fraction) -> Fraction:
    return max(
        hockey_stick_alpha_fraction(P, Q, alpha),
        hockey_stick_alpha_fraction(Q, P, alpha),
    )


def evaluate_family(
    *,
    case: str,
    accounting_scope: str,
    adjacency_scope: str,
    pairs: Sequence[RationalNeighborPair],
    decoder: str,
    N: int,
    M: int,
    nu: Sequence[Fraction],
    dummy_law_mode: str,
    dummy_law_source: str,
) -> list[dict]:
    float_pairs = [float_pair(pair) for pair in pairs]
    family = account_pair_family(
        float_pairs,
        decoder,
        N,
        M,
        float_law(nu),
        float(DELTA),
    )
    result_by_name = {result.pair_name: result for result in family.pair_results}
    oracle_by_name = {}
    exact_alpha_by_name: dict[str, Fraction | None] = {}
    for pair in pairs:
        oracle_P, oracle_Q = exhaustive_neighbor_distributions(
            pair, decoder, N, M, nu
        )
        oracle_by_name[pair.name] = (oracle_P, oracle_Q)
        exact_alpha_by_name[pair.name] = two_sided_min_alpha_fraction(
            oracle_P, oracle_Q, DELTA
        )
    finite_alphas = [alpha for alpha in exact_alpha_by_name.values() if alpha is not None]
    family_exact_alpha = max(finite_alphas) if len(finite_alphas) == len(pairs) else None
    family_exact_epsilon = epsilon_from_alpha_fraction(family_exact_alpha)

    rows = []
    for pair in pairs:
        result = result_by_name[pair.name]
        oracle_P, oracle_Q = oracle_by_name[pair.name]
        support = set(oracle_P) | set(oracle_Q) | set(result.P) | set(result.Q)
        p_errors = [
            abs(result.P.get(output, 0.0) - float(oracle_P.get(output, F(0))))
            for output in support
        ]
        q_errors = [
            abs(result.Q.get(output, 0.0) - float(oracle_Q.get(output, F(0))))
            for output in support
        ]
        exact_alpha = exact_alpha_by_name[pair.name]
        exact_epsilon = epsilon_from_alpha_fraction(exact_alpha)
        exact_profile = (
            exact_two_sided_profile(oracle_P, oracle_Q, exact_alpha)
            if exact_alpha is not None
            else F(1)
        )
        background_size = sum(pair.background_counts)
        rows.append(
            {
                "case": case,
                "accounting_scope": accounting_scope,
                "adjacency_scope": adjacency_scope,
                "adjacency": (
                    "add_remove" if pair.target_law_q is None else "replacement"
                ),
                "decoder": decoder,
                "dummy_law_mode": dummy_law_mode,
                "dummy_law_source": dummy_law_source,
                "nu": law_text(nu),
                "pair_name": pair.name,
                "family_size": len(pairs),
                "family_scope_metadata": family.family_scope,
                "is_worst_case_exact": str(exact_alpha == family_exact_alpha).lower(),
                "worst_pair_float_nominal": family.worst_pair_nominal,
                "alpha_exact_fraction": fraction_text(exact_alpha),
                "epsilon_exact_reported": exact_epsilon,
                "family_alpha_exact_fraction": fraction_text(family_exact_alpha),
                "family_epsilon_exact_reported": family_exact_epsilon,
                "alpha_float_nominal": result.alpha_nominal,
                "epsilon_float_nominal": result.epsilon_nominal,
                "alpha_float_guarded_diagnostic": result.alpha_float_guarded,
                "epsilon_float_guarded_diagnostic": result.epsilon_float_guarded,
                "alpha_float_absolute_error": (
                    abs(result.alpha_nominal - float(exact_alpha))
                    if exact_alpha is not None
                    else ""
                ),
                "exact_profile_at_alpha_fraction": fraction_text(exact_profile),
                "float_profile_at_nominal_alpha": result.profile_at_nominal,
                "delta": float(DELTA),
                "delta_fraction": fraction_text(DELTA),
                "oracle_max_abs_probability_error": max(p_errors + q_errors),
                "oracle_l1_error_P": sum(p_errors),
                "oracle_l1_error_Q": sum(q_errors),
                "rational_distribution_tv_error_certificate": 0,
                "rational_hockey_stick_error_certificate": 0,
                "float_l1_error_diagnostic_each_distribution": (
                    result.float_l1_error_diagnostic_each_distribution
                ),
                "float_hockey_stick_error_diagnostic": (
                    result.float_hockey_stick_error_diagnostic
                ),
                "max_float_normalization_mass_error": (
                    result.diagnostics.max_normalization_mass_error
                ),
                "float_normalization_residual_l1_sum": (
                    result.diagnostics.normalization_residual_l1_sum
                ),
                "float_underflowed_positive_terms": (
                    result.diagnostics.underflowed_positive_terms
                ),
                "P_support": len(result.P),
                "Q_support": len(result.Q),
                "support_union": len(support),
                "P_mass_exact": fraction_text(sum(oracle_P.values(), F(0))),
                "Q_mass_exact": fraction_text(sum(oracle_Q.values(), F(0))),
                "source_type": "exact_accountant",
                "mathematical_recurrence": "finite_type_count_coefficient_dp",
                "float_evaluation": "ieee754_with_diagnostics_not_interval_certified",
                "exact_oracle": "fraction_exhaustive_labeled_states",
                "oracle_rational_exact": "true",
                "full_release": "true",
                "output_representation": "exchangeable_count_vector",
                "payload_group": f"Z_{len(nu)}",
                "poissonized": "false",
                "arbitrary_public_nu": "true",
                "slotwise_padding_optimization": str(M >= N).lower(),
                "background_independence": "independent_iid_within_type",
                "background_size": background_size,
                "background_type_counts": ";".join(
                    str(value) for value in pair.background_counts
                ),
                "background_types": len(pair.background_counts),
                "N": N,
                "M": M,
                "K": len(nu),
                "rho_background": background_size / N,
                "n_total_Q": background_size
                + (1 if pair.target_law_q is not None else 0),
                "n_total_P": background_size + 1,
                "provenance": "coefficient_dp_checked_against_independent_fraction_oracle_every_output",
            }
        )
    return rows


def replacement_rows() -> list[dict]:
    laws = (BRR_F0, BRR_F1)
    pairs = []
    # Public replacement size is three: two common background records plus one
    # replaced target record on both sides.
    for counts in enumerate_type_count_vectors(2, 2):
        counts_text = "_".join(str(value) for value in counts)
        for q_type, p_type in ((0, 1), (1, 0)):
            pairs.append(
                RationalNeighborPair(
                    name=(
                        f"replacement_counts_{counts_text}_target_{q_type}_to_{p_type}"
                    ),
                    background_counts=counts,
                    background_laws=laws,
                    target_law_p=laws[p_type],
                    target_law_q=laws[q_type],
                    family_scope="fixed_public_size_replacement",
                )
            )
    rows = []
    for decoder in ("drop", "pool"):
        for mode, nu, source in [
            (
                "uniform_public_reference",
                UNIFORM_NU,
                "equal_mixture_BRR_public_reference",
            ),
            ("mismatched_public", SKEWED_NU, "fixed_skewed_public_prior"),
        ]:
            rows.extend(
                evaluate_family(
                    case="fixed_n_replacement",
                    accounting_scope="mechanism_level_exhaustive",
                    adjacency_scope="fixed_n_replacement_exhaustive_histograms_and_ordered_flips",
                    pairs=pairs,
                    decoder=decoder,
                    N=2,
                    M=3,
                    nu=nu,
                    dummy_law_mode=mode,
                    dummy_law_source=source,
                )
            )
    return rows


def mechanism_add_remove_rows() -> list[dict]:
    type_laws = (BRR_F0, BRR_F1)
    pairs = []
    for background_size in range(3):
        pairs.extend(
            rational_mechanism_family(
                background_size,
                type_laws,
                f"add_remove_n_{background_size}",
                "all_background_sizes_add_remove",
            )
        )
    rows = []
    for decoder in ("drop", "pool"):
        for mode, nu, source in [
            (
                "uniform_public_reference",
                UNIFORM_NU,
                "equal_mixture_BRR_public_reference",
            ),
            ("mismatched_public", SKEWED_NU, "fixed_skewed_public_prior"),
        ]:
            rows.extend(
                evaluate_family(
                    case="small_mechanism_add_remove",
                    accounting_scope="mechanism_level_exhaustive",
                    adjacency_scope="add_remove_exhaustive_background_histograms_and_target_types",
                    pairs=pairs,
                    decoder=decoder,
                    N=2,
                    M=3,
                    nu=nu,
                    dummy_law_mode=mode,
                    dummy_law_source=source,
                )
            )
    return rows


def zero_background_rows() -> list[dict]:
    pair = RationalNeighborPair(
        name="add_remove_n0",
        background_counts=(0, 0),
        background_laws=(BRR_F0, BRR_F1),
        target_law_p=BRR_F1,
    )
    rows = []
    for N in (2, 5):
        for decoder in ("drop", "pool"):
            rows.extend(
                evaluate_family(
                    case="add_remove_n0_control",
                    accounting_scope="pair_specific",
                    adjacency_scope="add_remove_zero_background_no_collision",
                    pairs=[pair],
                    decoder=decoder,
                    N=N,
                    M=5,
                    nu=SKEWED_NU,
                    dummy_law_mode="mismatched_public",
                    dummy_law_source="fixed_public_prior",
                )
            )
    return rows


def heterogeneous_k3_general_padding_rows() -> list[dict]:
    laws = (
        (F(3, 5), F(3, 10), F(1, 10)),
        (F(1, 10), F(1, 5), F(7, 10)),
    )
    pair = RationalNeighborPair(
        name="heterogeneous_k3_counts_1_1_target_1",
        background_counts=(1, 1),
        background_laws=laws,
        target_law_p=laws[1],
    )
    nu = (F(1, 7), F(2, 7), F(4, 7))
    rows = []
    for decoder in ("drop", "pool"):
        rows.extend(
            evaluate_family(
                case="heterogeneous_k3_general_padding",
                accounting_scope="pair_specific",
                adjacency_scope="add_remove_heterogeneous_two_type_general_padding",
                pairs=[pair],
                decoder=decoder,
                N=4,
                M=3,
                nu=nu,
                dummy_law_mode="mismatched_public",
                dummy_law_source="fixed_three_symbol_public_prior",
            )
        )
    return rows


def generate_all() -> dict[str, int]:
    rows = (
        replacement_rows()
        + mechanism_add_remove_rows()
        + zero_background_rows()
        + heterogeneous_k3_general_padding_rows()
    )
    filename = "fig_generic_exact_accountant_validation.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
