"""Finite-versus-Poisson privacy validation for all four analysis cases.

The tiny grid uses a homogeneous binary pair on Z_2 and exact rational
enumeration of the complete finite padded release.  Its Poisson comparison is
also a complete count law, obtained by multiplying independent one-slot
factors and adding M-N iid dummies.  The same finite matched public dummy law
is used on both sides of that comparison.

The moderate grid uses the coefficient accountant at n in {8,16,32}, plus
n=64 at rho=1 for a convergence check.  It uses a Poisson-matched dummy law
and complete finite and Poisson count laws.  It
reports exact-law numerical hockey-stick inversions, full-release chi2
certificates, and released-position marginal scores.

The quadratic epsilon values are display predictions
sqrt(2 eta log(1/delta)); they are not DP upper bounds.  The certified value in
the CSV is the two-sided hockey-stick inversion of the complete finite release.
"""

from __future__ import annotations

import math
from fractions import Fraction
from collections import defaultdict
from typing import Mapping, Sequence

from chafe_eval_lib import DATA_DIR, write_csv
from exact_hidden_slot_accountant import (
    NeighborPair,
    account_pair,
    two_sided_min_alpha,
    two_sided_profile_alpha,
)
from exhaustive_hidden_slot_oracle import (
    RationalNeighborPair,
    epsilon_from_alpha_fraction,
    exhaustive_neighbor_distributions,
    finite_matched_dummy_law_fraction,
    two_sided_min_alpha_fraction,
)


F = Fraction
F0 = (F(3, 4), F(1, 4))
F1 = (F(1, 4), F(3, 4))
BACKGROUND_SIZE = 3
REPLACEMENT_SIZE = BACKGROUND_SIZE + 1
M = 6
DELTA = F(1, 1_000)
N_VALUES = (2, 3, 4, 5, 6)
MODERATE_BACKGROUND_SIZES = (8, 16, 32, 64)
MODERATE_RHOS = (0.5, 1.0, 2.0)
MISMATCH_GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TAU_MULTIPLIERS = (1, 2, 4, 8)


def fraction_text(value: Fraction | None) -> str:
    if value is None:
        return "inf"
    return f"{value.numerator}/{value.denominator}"


def law_text(law: Sequence[Fraction]) -> str:
    return ";".join(fraction_text(F(value)) for value in law)


def float_law_text(law: Sequence[float]) -> str:
    return ";".join(f"{float(value):.17g}" for value in law)


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


def rational_marginal(
    distribution: Mapping[tuple[int, ...], Fraction], M_value: int
) -> tuple[Fraction, ...]:
    K = len(next(iter(distribution)))
    masses = [F(0) for _ in range(K)]
    for histogram, probability in distribution.items():
        for value, count in enumerate(histogram):
            masses[value] += probability * count
    return tuple(mass / M_value for mass in masses)


def float_marginal(
    distribution: Mapping[tuple[int, ...], float], M_value: int
) -> tuple[float, ...]:
    K = len(next(iter(distribution)))
    masses = [0.0 for _ in range(K)]
    for histogram, probability in distribution.items():
        for value, count in enumerate(histogram):
            masses[value] += probability * count
    return tuple(mass / M_value for mass in masses)


def chi2_fraction(
    numerator: Sequence[Fraction], reference: Sequence[Fraction]
) -> Fraction | None:
    total = F(0)
    for p, q in zip(numerator, reference):
        if q == 0:
            if p != 0:
                return None
            continue
        total += (p - q) ** 2 / q
    return total


def chi2_float(numerator: Sequence[float], reference: Sequence[float]) -> float:
    total = 0.0
    for p, q in zip(numerator, reference):
        if q == 0.0:
            if p != 0.0:
                return math.inf
            continue
        total += (p - q) ** 2 / q
    return total


def distribution_chi2(
    numerator: Mapping[tuple[int, ...], float],
    reference: Mapping[tuple[int, ...], float],
) -> float:
    total = 0.0
    for outcome in set(numerator) | set(reference):
        p = numerator.get(outcome, 0.0)
        q = reference.get(outcome, 0.0)
        if q == 0.0:
            if p != 0.0:
                return math.inf
            continue
        total += (p - q) ** 2 / q
    return total


def chi2_hockey_certificate(
    P: Mapping[tuple[int, ...], float],
    Q: Mapping[tuple[int, ...], float],
    delta: float,
) -> tuple[float, float, float, float]:
    """Return two directed chi2 values, their max, and a certified epsilon.

    If X=dP/dQ-1 has variance chi2, then

      H_{1+t}(P||Q) <= (sqrt(t^2+chi2)-t)/2.

    Thus t=max(0,chi2/(4 delta)-delta) certifies a directed hockey-stick
    value at most delta.  Taking the larger directed chi2 certifies both
    directions at the same epsilon.
    """
    forward = distribution_chi2(P, Q)
    reverse = distribution_chi2(Q, P)
    two_sided = max(forward, reverse)
    if not math.isfinite(two_sided):
        return forward, reverse, two_sided, math.inf
    t = max(0.0, two_sided / (4.0 * delta) - delta)
    epsilon = math.log1p(t)
    alpha = math.exp(epsilon)
    if two_sided_profile_alpha(P, Q, alpha) > delta + 1e-10:
        raise AssertionError("full-chi2 hockey certificate is infeasible")
    return forward, reverse, two_sided, epsilon


def cyclic_convolution_float(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float]:
    if len(left) != 2 or len(right) != 2:
        raise ValueError("this validation intentionally fixes the group to Z_2")
    return (
        left[0] * right[0] + left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def poisson_z2_slot_sublaws(
    decoder: str,
    rho: float,
    background_law: Sequence[float],
    target_law: Sequence[float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ordinary-slot and target-hit emitted subprobability laws.

    The Pool calculation is closed form.  Poisson thinning makes the number
    of value-one reports Pois(rho * background_law[1]), so the parity law is
    available without truncating the infinite occupancy mixture.
    """
    if decoder == "drop":
        empty = math.exp(-rho)
        singleton = rho * empty
        background = tuple(singleton * value for value in background_law)
        target = tuple(empty * value for value in target_law)
        return background, target
    if decoder != "pool":
        raise ValueError("decoder must be drop or pool")
    parity_bias = math.exp(-2.0 * rho * background_law[1])
    compound = ((1.0 + parity_bias) / 2.0, (1.0 - parity_bias) / 2.0)
    empty = math.exp(-rho)
    background = (compound[0] - empty, compound[1])
    target = cyclic_convolution_float(target_law, compound)
    return background, target


def padded_slot_law(
    emitted_sublaw: Sequence[float], nu: Sequence[float]
) -> tuple[float, ...]:
    emitted_mass = math.fsum(emitted_sublaw)
    if emitted_mass < -1e-14 or emitted_mass > 1.0 + 1e-14:
        raise ValueError("invalid emitted subprobability law")
    law = tuple(
        real + (1.0 - emitted_mass) * dummy
        for real, dummy in zip(emitted_sublaw, nu)
    )
    if abs(math.fsum(law) - 1.0) > 1e-12:
        raise AssertionError("padded slot law lost mass")
    return law


def convolve_histogram_laws(
    left: Mapping[tuple[int, ...], float],
    right: Mapping[tuple[int, ...], float],
) -> dict[tuple[int, ...], float]:
    output: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for left_histogram, left_probability in left.items():
        for right_histogram, right_probability in right.items():
            histogram = tuple(
                a + b for a, b in zip(left_histogram, right_histogram)
            )
            output[histogram] += left_probability * right_probability
    return dict(output)


def categorical_histogram_law(
    law: Sequence[float], count: int
) -> dict[tuple[int, ...], float]:
    K = len(law)
    distribution: dict[tuple[int, ...], float] = {(0,) * K: 1.0}
    for _ in range(count):
        output: defaultdict[tuple[int, ...], float] = defaultdict(float)
        for histogram, probability in distribution.items():
            for value, value_probability in enumerate(law):
                if value_probability == 0.0:
                    continue
                updated = list(histogram)
                updated[value] += 1
                output[tuple(updated)] += probability * value_probability
        distribution = dict(output)
    return distribution


def poisson_full_release_laws(
    *,
    decoder: str,
    adjacency: str,
    background_size: int,
    N: int,
    M_value: int,
    nu: Sequence[float],
) -> tuple[dict[tuple[int, ...], float], dict[tuple[int, ...], float]]:
    """Complete Poisson one-slot-factorized count laws (P,Q).

    This construction requires N<=M.  Every one-slot factor emits its real
    decoder output or, on an empty/suppressed outcome, one iid nu dummy.
    Convolution with M-N further dummies therefore produces exactly M values.
    """
    if N > M_value:
        raise ValueError("Poisson slotwise factorization requires N <= M")
    rho = background_size / N
    background_sublaw, target_q_sublaw = poisson_z2_slot_sublaws(
        decoder, rho, float_law(F0), float_law(F0)
    )
    _, target_p_sublaw = poisson_z2_slot_sublaws(
        decoder, rho, float_law(F0), float_law(F1)
    )
    background_slot = padded_slot_law(background_sublaw, nu)
    target_q_slot = padded_slot_law(target_q_sublaw, nu)
    target_p_slot = padded_slot_law(target_p_sublaw, nu)
    background_n_minus_one = categorical_histogram_law(background_slot, N - 1)
    extra_dummies = categorical_histogram_law(nu, M_value - N)

    target_p = categorical_histogram_law(target_p_slot, 1)
    if adjacency == "replacement":
        target_q = categorical_histogram_law(target_q_slot, 1)
        P = convolve_histogram_laws(background_n_minus_one, target_p)
        Q = convolve_histogram_laws(background_n_minus_one, target_q)
    elif adjacency == "add_remove":
        background_one = categorical_histogram_law(background_slot, 1)
        P = convolve_histogram_laws(background_n_minus_one, target_p)
        Q = convolve_histogram_laws(background_n_minus_one, background_one)
    else:
        raise ValueError("invalid adjacency")
    P = convolve_histogram_laws(P, extra_dummies)
    Q = convolve_histogram_laws(Q, extra_dummies)
    for label, distribution in (("P", P), ("Q", Q)):
        if abs(math.fsum(distribution.values()) - 1.0) > 1e-10:
            raise AssertionError(f"Poisson full {label} law lost mass")
        if any(sum(histogram) != M_value for histogram in distribution):
            raise AssertionError("Poisson full law has the wrong release length")
    return P, Q


def poisson_matched_dummy_law(
    decoder: str, rho: float
) -> tuple[float, float]:
    if decoder == "drop":
        return float_law(F0)
    background_sublaw, _ = poisson_z2_slot_sublaws(
        decoder, rho, float_law(F0), float_law(F0)
    )
    nonempty = math.fsum(background_sublaw)
    if nonempty <= 0.0:
        raise ValueError("Pool matched law requires positive load")
    return tuple(value / nonempty for value in background_sublaw)


def poisson_reference_eta(
    *,
    decoder: str,
    adjacency: str,
    rho: float,
    M_value: int,
    nu: Sequence[float],
) -> float:
    """Return the paper's Poisson matched-reference quadratic score.

    This is (1/M)||K_P-K_Q||^2_{nu^{-1}} for the distinguished one-slot
    factors.  It is deliberately distinct from M*chi2(q_P||q_Q), the actual
    uniformly sampled released-position marginal score.
    """
    background, target_q = poisson_z2_slot_sublaws(
        decoder, rho, float_law(F0), float_law(F0)
    )
    _, target_p = poisson_z2_slot_sublaws(
        decoder, rho, float_law(F0), float_law(F1)
    )
    if adjacency == "replacement":
        kernel_q = padded_slot_law(target_q, nu)
    elif adjacency == "add_remove":
        kernel_q = padded_slot_law(background, nu)
    else:
        raise ValueError("invalid adjacency")
    kernel_p = padded_slot_law(target_p, nu)
    score = 0.0
    for p, q, reference in zip(kernel_p, kernel_q, nu):
        if reference == 0.0:
            if p != q:
                return math.inf
            continue
        score += (p - q) ** 2 / reference
    return score / M_value


def padded_position_marginal(
    expected_real_counts: Sequence[float],
    nu: Sequence[float],
    M_value: int,
) -> tuple[float, ...]:
    real_count = math.fsum(expected_real_counts)
    if real_count > M_value + 1e-12:
        raise ValueError("Poisson expected real count exceeds release length")
    marginal = tuple(
        (real + (M_value - real_count) * dummy) / M_value
        for real, dummy in zip(expected_real_counts, nu)
    )
    if abs(math.fsum(marginal) - 1.0) > 1e-12:
        raise AssertionError("Poisson padded marginal lost mass")
    return marginal


def poisson_release_marginals(
    *,
    decoder: str,
    adjacency: str,
    background_size: int,
    N: int,
    M_value: int,
    nu: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (P,Q) position marginals under the Poisson occupancy surrogate."""
    rho = background_size / N
    background_q, target_q = poisson_z2_slot_sublaws(
        decoder, rho, float_law(F0), float_law(F0)
    )
    _, target_p = poisson_z2_slot_sublaws(
        decoder, rho, float_law(F0), float_law(F1)
    )
    if adjacency == "replacement":
        real_q = tuple((N - 1) * bg + target for bg, target in zip(background_q, target_q))
        real_p = tuple((N - 1) * bg + target for bg, target in zip(background_q, target_p))
    elif adjacency == "add_remove":
        real_q = tuple(N * bg for bg in background_q)
        real_p = tuple((N - 1) * bg + target for bg, target in zip(background_q, target_p))
    else:
        raise ValueError("invalid adjacency")
    return (
        padded_position_marginal(real_p, nu, M_value),
        padded_position_marginal(real_q, nu, M_value),
    )


def make_pair(adjacency: str, decoder: str, N: int) -> tuple[RationalNeighborPair, tuple[Fraction, ...]]:
    if adjacency == "replacement":
        pair = RationalNeighborPair.homogeneous(
            name=f"{decoder}_replacement_n{BACKGROUND_SIZE}_N{N}",
            n=BACKGROUND_SIZE,
            background_law=F0,
            target_law_p=F1,
            target_law_q=F0,
            family_scope="homogeneous_pair_specific",
        )
        matched_population = REPLACEMENT_SIZE
    elif adjacency == "add_remove":
        pair = RationalNeighborPair.homogeneous(
            name=f"{decoder}_add_remove_n{BACKGROUND_SIZE}_N{N}",
            n=BACKGROUND_SIZE,
            background_law=F0,
            target_law_p=F1,
            family_scope="homogeneous_pair_specific",
        )
        matched_population = BACKGROUND_SIZE
    else:
        raise ValueError("invalid adjacency")
    nu = finite_matched_dummy_law_fraction(
        decoder, matched_population, N, F0
    )
    return pair, nu


def evaluate_case(decoder: str, adjacency: str, N: int) -> dict:
    pair, nu = make_pair(adjacency, decoder, N)
    oracle_p, oracle_q = exhaustive_neighbor_distributions(
        pair, decoder, N, M, nu
    )
    alpha_exact = two_sided_min_alpha_fraction(oracle_p, oracle_q, DELTA)
    epsilon_exact = epsilon_from_alpha_fraction(alpha_exact)

    marginal_p = rational_marginal(oracle_p, M)
    marginal_q = rational_marginal(oracle_q, M)
    finite_forward_chi2 = chi2_fraction(marginal_p, marginal_q)
    finite_reverse_chi2 = chi2_fraction(marginal_q, marginal_p)
    if finite_forward_chi2 is None or finite_reverse_chi2 is None:
        raise AssertionError("chosen matched laws should have common marginal support")
    eta_finite_forward = F(M) * finite_forward_chi2
    eta_finite_reverse = F(M) * finite_reverse_chi2
    eta_finite_two_sided = max(eta_finite_forward, eta_finite_reverse)

    poisson_p, poisson_q = poisson_release_marginals(
        decoder=decoder,
        adjacency=adjacency,
        background_size=BACKGROUND_SIZE,
        N=N,
        M_value=M,
        nu=float_law(nu),
    )
    poisson_full_p, poisson_full_q = poisson_full_release_laws(
        decoder=decoder,
        adjacency=adjacency,
        background_size=BACKGROUND_SIZE,
        N=N,
        M_value=M,
        nu=float_law(nu),
    )
    poisson_full_marginal_p = float_marginal(poisson_full_p, M)
    poisson_full_marginal_q = float_marginal(poisson_full_q, M)
    marginal_factorization_error = max(
        abs(left - right)
        for left, right in zip(
            poisson_p + poisson_q,
            poisson_full_marginal_p + poisson_full_marginal_q,
        )
    )
    if marginal_factorization_error > 1e-12:
        raise AssertionError("Poisson full law and direct marginal disagree")
    eta_poisson_forward = M * chi2_float(poisson_p, poisson_q)
    eta_poisson_reverse = M * chi2_float(poisson_q, poisson_p)
    eta_poisson_two_sided = max(eta_poisson_forward, eta_poisson_reverse)

    delta_float = float(DELTA)
    epsilon_quad_finite = math.sqrt(
        2.0 * float(eta_finite_two_sided) * math.log(1.0 / delta_float)
    )
    epsilon_quad_poisson = math.sqrt(
        2.0 * eta_poisson_two_sided * math.log(1.0 / delta_float)
    )
    eta_poisson_reference = poisson_reference_eta(
        decoder=decoder,
        adjacency=adjacency,
        rho=BACKGROUND_SIZE / N,
        M_value=M,
        nu=float_law(nu),
    )
    epsilon_reference_quadratic = math.sqrt(
        2.0 * eta_poisson_reference * math.log(1.0 / delta_float)
    )
    alpha_poisson_full = two_sided_min_alpha(
        poisson_full_p, poisson_full_q, delta_float
    )
    epsilon_poisson_full = (
        math.log(alpha_poisson_full)
        if math.isfinite(alpha_poisson_full)
        else math.inf
    )

    recurrence = account_pair(
        float_pair(pair), decoder, N, M, float_law(nu), delta_float
    )
    support = set(oracle_p) | set(oracle_q) | set(recurrence.P) | set(recurrence.Q)
    oracle_error = max(
        [
            abs(recurrence.P.get(state, 0.0) - float(oracle_p.get(state, F(0))))
            for state in support
        ]
        + [
            abs(recurrence.Q.get(state, 0.0) - float(oracle_q.get(state, F(0))))
            for state in support
        ]
    )
    if oracle_error > 1e-11:
        raise AssertionError(f"coefficient recurrence disagrees with oracle: {oracle_error}")
    finite_chi2 = chi2_hockey_certificate(
        recurrence.P, recurrence.Q, delta_float
    )
    poisson_chi2 = chi2_hockey_certificate(
        poisson_full_p, poisson_full_q, delta_float
    )

    return {
        "panel": f"{decoder}_{adjacency}",
        "decoder": decoder,
        "adjacency": adjacency,
        "pair_scope": "homogeneous_binary_pair_specific",
        "forward_direction": "a_to_b" if adjacency == "replacement" else "remove_to_add",
        "reverse_direction": "b_to_a" if adjacency == "replacement" else "add_to_remove",
        "background_law_f0": law_text(F0),
        "target_law_f1": law_text(F1),
        "local_epsilon0": math.log(3.0),
        "dummy_law_mode": "finite_matched_baseline",
        "nu": law_text(nu),
        "background_size": BACKGROUND_SIZE,
        "replacement_total_size": REPLACEMENT_SIZE if adjacency == "replacement" else "",
        "N": N,
        "M": M,
        "rho_common_background": BACKGROUND_SIZE / N,
        "rho_Q_total": (
            REPLACEMENT_SIZE / N
            if adjacency == "replacement"
            else BACKGROUND_SIZE / N
        ),
        "rho_P_total": REPLACEMENT_SIZE / N,
        "delta": delta_float,
        "delta_fraction": fraction_text(DELTA),
        "eta_definition": "M_times_chi2_of_uniform_released_position_marginals",
        "finite_occupancy_model": "Binomial(common_background_size,1_over_N)",
        "eta_finite_forward_fraction": fraction_text(eta_finite_forward),
        "eta_finite_forward": float(eta_finite_forward),
        "eta_finite_reverse_fraction": fraction_text(eta_finite_reverse),
        "eta_finite_reverse": float(eta_finite_reverse),
        "eta_finite_two_sided": float(eta_finite_two_sided),
        "eta_poisson_forward": eta_poisson_forward,
        "eta_poisson_reverse": eta_poisson_reverse,
        "eta_poisson_two_sided": eta_poisson_two_sided,
        "eta_poisson_over_finite": eta_poisson_two_sided / float(eta_finite_two_sided),
        "eta_poisson_reference": eta_poisson_reference,
        "bar_eta_poisson_reference": eta_poisson_reference,
        "M_eta_finite_two_sided": M * float(eta_finite_two_sided),
        "M_eta_poisson_two_sided": M * eta_poisson_two_sided,
        "M_eta_poisson_reference": M * eta_poisson_reference,
        "epsilon_quad_finite_marginal": epsilon_quad_finite,
        "epsilon_quad_poisson_marginal": epsilon_quad_poisson,
        "epsilon_reference_quadratic": epsilon_reference_quadratic,
        "alpha_exact_fraction": fraction_text(alpha_exact),
        "epsilon_finite_exact_two_sided": epsilon_exact,
        "alpha_poisson_full_two_sided": alpha_poisson_full,
        "epsilon_poisson_full_exact_two_sided": epsilon_poisson_full,
        "poisson_full_profile_at_exact_alpha": two_sided_profile_alpha(
            poisson_full_p, poisson_full_q, alpha_poisson_full
        ),
        "full_chi2_finite_forward": finite_chi2[0],
        "full_chi2_finite_reverse": finite_chi2[1],
        "full_chi2_finite_two_sided": finite_chi2[2],
        "epsilon_full_chi2_finite_certificate": finite_chi2[3],
        "full_chi2_poisson_forward": poisson_chi2[0],
        "full_chi2_poisson_reverse": poisson_chi2[1],
        "full_chi2_poisson_two_sided": poisson_chi2[2],
        "epsilon_full_chi2_poisson_certificate": poisson_chi2[3],
        "poisson_full_support_P": len(poisson_full_p),
        "poisson_full_support_Q": len(poisson_full_q),
        "poisson_full_marginal_factorization_error": marginal_factorization_error,
        "epsilon_quad_finite_over_exact": epsilon_quad_finite / epsilon_exact,
        "epsilon_quad_poisson_over_exact": epsilon_quad_poisson / epsilon_exact,
        "poisson_model": "replace_Binomial_common_background_occupancy_by_Pois(n_over_N);retain_distinguished_target;reuse_same_finite_nu",
        "poisson_eta_scope": "exact_single_position_marginal_of_full_one_slot_factorization",
        "poisson_full_release_scope": "independent_Poisson_one_slot_factors_with_slotwise_padding",
        "poisson_pool_evaluation": "closed_form_Z2_compound_Poisson_no_tail_truncation",
        "full_release_accounting": "exact_rational_exhaustive_labeled_states",
        "accountant_cross_check": "float_coefficient_recurrence_every_count_state",
        "accountant_max_abs_probability_error": oracle_error,
        "quadratic_epsilon_status": "display_prediction_not_DP_bound",
        "exact_epsilon_status": "pair_specific_two_sided_hockey_stick_certificate",
        "source_type": "exact_plus_poisson_validation",
    }


def moderate_pair(
    decoder: str, adjacency: str, background_size: int, N: int
) -> NeighborPair:
    return NeighborPair.homogeneous(
        name=f"moderate_{decoder}_{adjacency}_n{background_size}_N{N}",
        n=background_size,
        background_law=float_law(F0),
        target_law_p=float_law(F1),
        target_law_q=(float_law(F0) if adjacency == "replacement" else None),
        family_scope="homogeneous_pair_specific",
    )


def evaluate_moderate_case(
    decoder: str, adjacency: str, background_size: int, rho: float
) -> dict:
    N = int(round(background_size / rho))
    if not math.isclose(background_size / N, rho, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("moderate grid requires an integral slot count")
    M_value = 2 * (background_size + 1)
    if N > M_value:
        raise ValueError("moderate grid must satisfy N <= M")
    delta = float(DELTA)
    nu = poisson_matched_dummy_law(decoder, rho)
    pair = moderate_pair(decoder, adjacency, background_size, N)

    finite = account_pair(pair, decoder, N, M_value, nu, delta)
    finite_p = finite.P
    finite_q = finite.Q
    finite_marginal_p = float_marginal(finite_p, M_value)
    finite_marginal_q = float_marginal(finite_q, M_value)
    eta_finite_forward = M_value * chi2_float(
        finite_marginal_p, finite_marginal_q
    )
    eta_finite_reverse = M_value * chi2_float(
        finite_marginal_q, finite_marginal_p
    )
    eta_finite_two_sided = max(eta_finite_forward, eta_finite_reverse)

    poisson_p, poisson_q = poisson_full_release_laws(
        decoder=decoder,
        adjacency=adjacency,
        background_size=background_size,
        N=N,
        M_value=M_value,
        nu=nu,
    )
    poisson_marginal_p = float_marginal(poisson_p, M_value)
    poisson_marginal_q = float_marginal(poisson_q, M_value)
    eta_poisson_forward = M_value * chi2_float(
        poisson_marginal_p, poisson_marginal_q
    )
    eta_poisson_reverse = M_value * chi2_float(
        poisson_marginal_q, poisson_marginal_p
    )
    eta_poisson_two_sided = max(eta_poisson_forward, eta_poisson_reverse)

    alpha_poisson = two_sided_min_alpha(poisson_p, poisson_q, delta)
    epsilon_poisson = (
        math.log(alpha_poisson) if math.isfinite(alpha_poisson) else math.inf
    )
    eta_poisson_reference = poisson_reference_eta(
        decoder=decoder,
        adjacency=adjacency,
        rho=rho,
        M_value=M_value,
        nu=nu,
    )
    epsilon_reference_quadratic = math.sqrt(
        2.0 * eta_poisson_reference * math.log(1.0 / delta)
    )
    finite_chi2 = chi2_hockey_certificate(finite_p, finite_q, delta)
    poisson_chi2 = chi2_hockey_certificate(poisson_p, poisson_q, delta)

    return {
        "panel": f"{decoder}_{adjacency}",
        "decoder": decoder,
        "adjacency": adjacency,
        "pair_scope": "homogeneous_binary_pair_specific",
        "forward_direction": (
            "a_to_b" if adjacency == "replacement" else "remove_to_add"
        ),
        "reverse_direction": (
            "b_to_a" if adjacency == "replacement" else "add_to_remove"
        ),
        "background_law_f0": law_text(F0),
        "target_law_f1": law_text(F1),
        "local_epsilon0": math.log(3.0),
        "background_size_common": background_size,
        "replacement_total_size": (
            background_size + 1 if adjacency == "replacement" else ""
        ),
        "N": N,
        "M": M_value,
        "rho_common_background": rho,
        "rho_Q_total": (
            (background_size + 1) / N
            if adjacency == "replacement"
            else background_size / N
        ),
        "rho_P_total": (background_size + 1) / N,
        "tau_P": M_value / (background_size + 1),
        "delta": delta,
        "dummy_law_mode": "poisson_matched_background_nonempty_aggregate",
        "nu": float_law_text(nu),
        "eta_definition": "M_times_chi2_of_uniform_released_position_marginals",
        "eta_finite_forward": eta_finite_forward,
        "eta_finite_reverse": eta_finite_reverse,
        "eta_finite_two_sided": eta_finite_two_sided,
        "eta_poisson_forward": eta_poisson_forward,
        "eta_poisson_reverse": eta_poisson_reverse,
        "eta_poisson_two_sided": eta_poisson_two_sided,
        "eta_poisson_over_finite": eta_poisson_two_sided / eta_finite_two_sided,
        "eta_poisson_reference": eta_poisson_reference,
        "bar_eta_poisson_reference": eta_poisson_reference,
        "M_eta_finite_two_sided": M_value * eta_finite_two_sided,
        "M_eta_poisson_two_sided": M_value * eta_poisson_two_sided,
        "M_eta_poisson_reference": M_value * eta_poisson_reference,
        "epsilon_finite_exact_two_sided": finite.epsilon_nominal,
        "finite_profile_at_exact_epsilon": finite.profile_at_nominal,
        "epsilon_poisson_exact_two_sided": epsilon_poisson,
        "poisson_profile_at_exact_epsilon": two_sided_profile_alpha(
            poisson_p, poisson_q, alpha_poisson
        ),
        "epsilon_reference_quadratic": epsilon_reference_quadratic,
        "quadratic_reference_source": "poisson_one_slot_reference_eta_Kp_minus_Kq_over_nu",
        "epsilon_quadratic_over_finite_exact": (
            epsilon_reference_quadratic / finite.epsilon_nominal
        ),
        "epsilon_quadratic_over_poisson_exact": (
            epsilon_reference_quadratic / epsilon_poisson
        ),
        "epsilon_poisson_exact_over_finite_exact": (
            epsilon_poisson / finite.epsilon_nominal
        ),
        "full_chi2_finite_forward": finite_chi2[0],
        "full_chi2_finite_reverse": finite_chi2[1],
        "full_chi2_finite_two_sided": finite_chi2[2],
        "epsilon_full_chi2_finite_certificate": finite_chi2[3],
        "full_chi2_poisson_forward": poisson_chi2[0],
        "full_chi2_poisson_reverse": poisson_chi2[1],
        "full_chi2_poisson_two_sided": poisson_chi2[2],
        "epsilon_full_chi2_poisson_certificate": poisson_chi2[3],
        "finite_support_P": len(finite_p),
        "finite_support_Q": len(finite_q),
        "poisson_support_P": len(poisson_p),
        "poisson_support_Q": len(poisson_q),
        "finite_arithmetic": "coefficient_accountant_ieee754_exact_law_numerical",
        "poisson_arithmetic": "closed_Z2_one_slot_factors_then_count_convolution",
        "poisson_full_release_scope": "independent_Poisson_one_slot_factors_with_slotwise_padding",
        "full_chi2_certificate": "centered_second_moment_hockey_bound",
        "quadratic_epsilon_status": "reference_prediction_not_DP_bound",
        "exact_epsilon_status": "pair_specific_two_sided_hockey_stick",
        "source_type": "finite_vs_poisson_exact_law_scaling",
    }


def evaluate_mismatch_case(
    decoder: str, adjacency: str, gamma: float
) -> dict:
    """Evaluate an arbitrary public dummy law between matched and uniform."""
    background_size = 32
    rho = 1.0
    N = 32
    M_value = 2 * (background_size + 1)
    delta = float(DELTA)
    matched = poisson_matched_dummy_law(decoder, rho)
    neutral = (0.5, 0.5)
    nu = tuple(
        (1.0 - gamma) * left + gamma * right
        for left, right in zip(matched, neutral)
    )
    pair = moderate_pair(decoder, adjacency, background_size, N)
    finite = account_pair(pair, decoder, N, M_value, nu, delta)
    poisson_p, poisson_q = poisson_full_release_laws(
        decoder=decoder,
        adjacency=adjacency,
        background_size=background_size,
        N=N,
        M_value=M_value,
        nu=nu,
    )
    alpha_poisson = two_sided_min_alpha(poisson_p, poisson_q, delta)
    epsilon_poisson = (
        math.log(alpha_poisson) if math.isfinite(alpha_poisson) else math.inf
    )
    return {
        "panel": f"{decoder}_{adjacency}",
        "decoder": decoder,
        "adjacency": adjacency,
        "gamma": gamma,
        "dummy_law_path": "one_minus_gamma_times_poisson_matched_plus_gamma_times_uniform_Z2",
        "dummy_tv_from_matched": 0.5
        * math.fsum(abs(left - right) for left, right in zip(nu, matched)),
        "nu_matched": float_law_text(matched),
        "nu_public_endpoint": "0.5;0.5",
        "nu": float_law_text(nu),
        "background_law_f0": law_text(F0),
        "target_law_f1": law_text(F1),
        "background_size_common": background_size,
        "N": N,
        "M": M_value,
        "rho_common_background": rho,
        "delta": delta,
        "epsilon_finite_exact_two_sided": finite.epsilon_nominal,
        "epsilon_poisson_exact_two_sided": epsilon_poisson,
        "epsilon_poisson_exact_over_finite_exact": (
            epsilon_poisson / finite.epsilon_nominal
        ),
        "finite_profile_at_exact_epsilon": finite.profile_at_nominal,
        "poisson_profile_at_exact_epsilon": two_sided_profile_alpha(
            poisson_p, poisson_q, alpha_poisson
        ),
        "finite_arithmetic": "coefficient_accountant_ieee754_exact_law_numerical",
        "poisson_arithmetic": "closed_Z2_one_slot_factors_then_count_convolution",
        "accounting_scope": "homogeneous_binary_pair_specific",
        "source_type": "finite_and_poisson_dummy_mismatch_sensitivity",
    }


def evaluate_tau_case(
    decoder: str, adjacency: str, tau_multiplier: int
) -> dict:
    """Evaluate finite and Poisson budgets as public release length changes."""
    background_size = 32
    rho = 1.0
    N = 32
    M_value = tau_multiplier * (background_size + 1)
    delta = float(DELTA)
    nu = poisson_matched_dummy_law(decoder, rho)
    pair = moderate_pair(decoder, adjacency, background_size, N)
    finite = account_pair(pair, decoder, N, M_value, nu, delta)
    poisson_p, poisson_q = poisson_full_release_laws(
        decoder=decoder,
        adjacency=adjacency,
        background_size=background_size,
        N=N,
        M_value=M_value,
        nu=nu,
    )
    alpha_poisson = two_sided_min_alpha(poisson_p, poisson_q, delta)
    epsilon_poisson = (
        math.log(alpha_poisson) if math.isfinite(alpha_poisson) else math.inf
    )
    eta_reference = poisson_reference_eta(
        decoder=decoder,
        adjacency=adjacency,
        rho=rho,
        M_value=M_value,
        nu=nu,
    )
    epsilon_quadratic = math.sqrt(
        2.0 * eta_reference * math.log(1.0 / delta)
    )
    return {
        "panel": f"{decoder}_{adjacency}",
        "decoder": decoder,
        "adjacency": adjacency,
        "background_size_common": background_size,
        "N": N,
        "M": M_value,
        "rho_common_background": rho,
        "tau_P": M_value / (background_size + 1),
        "delta": delta,
        "dummy_law_mode": "poisson_matched_background_nonempty_aggregate",
        "nu": float_law_text(nu),
        "epsilon_finite_exact_two_sided": finite.epsilon_nominal,
        "epsilon_poisson_exact_two_sided": epsilon_poisson,
        "epsilon_reference_quadratic": epsilon_quadratic,
        "epsilon_poisson_exact_over_finite_exact": (
            epsilon_poisson / finite.epsilon_nominal
        ),
        "epsilon_quadratic_over_poisson_exact": (
            epsilon_quadratic / epsilon_poisson
        ),
        "finite_profile_at_exact_epsilon": finite.profile_at_nominal,
        "poisson_profile_at_exact_epsilon": two_sided_profile_alpha(
            poisson_p, poisson_q, alpha_poisson
        ),
        "finite_arithmetic": "coefficient_accountant_ieee754_exact_law_numerical",
        "poisson_arithmetic": "closed_Z2_one_slot_factors_then_count_convolution",
        "accounting_scope": "homogeneous_binary_pair_specific",
        "quadratic_epsilon_status": "reference_prediction_not_DP_bound",
        "source_type": "finite_poisson_quadratic_tau_sensitivity",
    }


def generate_all() -> dict[str, int]:
    tiny_rows = [
        evaluate_case(decoder, adjacency, N)
        for decoder in ("drop", "pool")
        for adjacency in ("replacement", "add_remove")
        for N in N_VALUES
    ]
    moderate_rows = [
        evaluate_moderate_case(decoder, adjacency, background_size, rho)
        for decoder in ("drop", "pool")
        for adjacency in ("replacement", "add_remove")
        for background_size in MODERATE_BACKGROUND_SIZES
        for rho in MODERATE_RHOS
        if background_size < 64 or rho == 1.0
    ]
    mismatch_rows = [
        evaluate_mismatch_case(decoder, adjacency, gamma)
        for decoder in ("drop", "pool")
        for adjacency in ("replacement", "add_remove")
        for gamma in MISMATCH_GAMMAS
    ]
    tau_rows = [
        evaluate_tau_case(decoder, adjacency, tau_multiplier)
        for decoder in ("drop", "pool")
        for adjacency in ("replacement", "add_remove")
        for tau_multiplier in TAU_MULTIPLIERS
    ]
    for panel in {row["panel"] for row in mismatch_rows}:
        panel_rows = [row for row in mismatch_rows if row["panel"] == panel]
        matched = next(row for row in panel_rows if row["gamma"] == 0.0)
        baseline = matched["epsilon_finite_exact_two_sided"]
        for row in panel_rows:
            row["epsilon_finite_over_matched"] = (
                row["epsilon_finite_exact_two_sided"] / baseline
            )
    for panel in {row["panel"] for row in tau_rows}:
        panel_rows = [row for row in tau_rows if row["panel"] == panel]
        tau_one = next(row for row in panel_rows if row["tau_P"] == 1.0)
        baseline = tau_one["epsilon_finite_exact_two_sided"]
        for row in panel_rows:
            row["epsilon_finite_over_tau1"] = (
                row["epsilon_finite_exact_two_sided"] / baseline
            )
    tiny_filename = "fig_privacy_2x2_validation.csv"
    moderate_filename = "fig_privacy_2x2_moderate_scaling.csv"
    mismatch_filename = "fig_privacy_dummy_mismatch_exact.csv"
    tau_filename = "fig_privacy_2x2_tau_exact.csv"
    write_csv(DATA_DIR / tiny_filename, tiny_rows)
    write_csv(DATA_DIR / moderate_filename, moderate_rows)
    write_csv(DATA_DIR / mismatch_filename, mismatch_rows)
    write_csv(DATA_DIR / tau_filename, tau_rows)
    return {
        tiny_filename: len(tiny_rows),
        moderate_filename: len(moderate_rows),
        mismatch_filename: len(mismatch_rows),
        tau_filename: len(tau_rows),
    }


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
