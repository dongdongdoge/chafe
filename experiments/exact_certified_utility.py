"""Exact-certified utility for the complete finite Chafe release.

This experiment deliberately uses a small, fully enumerable workload.  The
goal is not to model a large deployment; it is to connect the paper's exact
privacy accountant to an end-to-end utility result without using a quadratic
proxy anywhere in the calibration path.

We consider a fixed-size binary count with m=3, N=3 hidden slots, a length-4
release, and replacement adjacency.  Binary randomized response is embedded
in Z_4, so Pool sums cannot wrap.  Padding uses a fixed public Bernoulli(1/2)
law on payloads 0 and 1.  Dummies are indistinguishable from real payloads and
the estimator sees only the shuffled length-M release.

For each release (same-M padded Ideal, Drop, and Pool), calibration exhausts a
declared rational grid of truthful-report probabilities.  Every neighboring
dataset composition and both replacement directions are enumerated with
``fractions.Fraction``.  We choose the largest grid point whose exact
two-sided hockey-stick profile is at most delta at alpha=2, giving a common
certified target (ln 2, 10^-3)-DP.  Utility is then evaluated both from the
exact output law and by an independent end-to-end Monte Carlo simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

from chafe_eval_lib import DATA_DIR, write_csv
from exact_hidden_slot_accountant import NeighborPair, account_pair_family
from exhaustive_hidden_slot_oracle import (
    RationalNeighborPair,
    epsilon_from_alpha_fraction,
    exhaustive_neighbor_distributions,
    exhaustive_output_distribution,
    two_sided_min_alpha_fraction,
)


F = Fraction
PUBLIC_M = 3
SLOT_N = 3
RELEASE_M = 4
PAYLOAD_K = PUBLIC_M + 1
TRUE_ONES = 2
DELTA = F(1, 1_000)
TARGET_ALPHA = F(2)
TARGET_EPSILON = math.log(2.0)
# The final rational grid is fine enough that the certified epsilon is within
# 1e-3 of the common target.  We locate its last feasible point by exact
# bisection and separately scan a coarse global grid to detect nonmonotone
# re-entry before trusting the bracket.
GRID_DENOMINATOR = 1_000_000
GRID_MIN_NUMERATOR = GRID_DENOMINATOR // 2 + 1
GRID_MAX_NUMERATOR = GRID_DENOMINATOR - 1
COARSE_DENOMINATOR = 100
COARSE_NUMERATORS = tuple(range(COARSE_DENOMINATOR // 2 + 1, COARSE_DENOMINATOR))
PUBLIC_NU = (F(1, 2), F(1, 2)) + (F(0),) * (PAYLOAD_K - 2)
MC_REPLICATES = 50_000
MC_SEED = 7_412_021
METHODS = ("Padded Ideal", "Drop", "Pool")


RationalDistribution = Mapping[tuple[int, ...], Fraction]


def fraction_text(value: Fraction) -> str:
    value = F(value)
    return f"{value.numerator}/{value.denominator}"


def rr_laws(truthful_probability: Fraction) -> tuple[tuple[Fraction, ...], ...]:
    """Return symmetric binary-RR laws embedded in Z_PAYLOAD_K."""
    p = F(truthful_probability)
    if not F(1, 2) < p < 1:
        raise ValueError("truthful_probability must lie strictly between 1/2 and 1")
    q = 1 - p
    law_zero = (p, q) + (F(0),) * (PAYLOAD_K - 2)
    law_one = (q, p) + (F(0),) * (PAYLOAD_K - 2)
    return law_zero, law_one


def convolve_bernoulli_count(
    law: Sequence[Fraction],
    probability_one: Fraction,
) -> list[Fraction]:
    out = [F(0)] * (len(law) + 1)
    p_one = F(probability_one)
    for count, probability in enumerate(law):
        out[count] += probability * (1 - p_one)
        out[count + 1] += probability * p_one
    return out


def padded_ideal_distribution(
    true_ones: int,
    truthful_probability: Fraction,
) -> dict[tuple[int, ...], Fraction]:
    """Exact count-vector law of the shuffled same-M padded ideal release."""
    if not 0 <= true_ones <= PUBLIC_M:
        raise ValueError("true_ones is outside the public dataset size")
    p = F(truthful_probability)
    q = 1 - p
    reported_one_count = [F(1)]
    for index in range(PUBLIC_M):
        raw_bit = 1 if index < true_ones else 0
        reported_one_count = convolve_bernoulli_count(
            reported_one_count,
            p if raw_bit else q,
        )
    output: dict[tuple[int, ...], Fraction] = {}
    dummy_count = RELEASE_M - PUBLIC_M
    for reported_ones, report_probability in enumerate(reported_one_count):
        for dummy_ones in range(dummy_count + 1):
            dummy_probability = (
                F(math.comb(dummy_count, dummy_ones), 2**dummy_count)
            )
            ones = reported_ones + dummy_ones
            histogram = (RELEASE_M - ones, ones) + (0,) * (PAYLOAD_K - 2)
            output[histogram] = (
                output.get(histogram, F(0))
                + report_probability * dummy_probability
            )
    if sum(output.values(), F(0)) != 1:
        raise AssertionError("padded ideal output law lost mass")
    return output


def rational_replacement_pairs(
    truthful_probability: Fraction,
) -> list[RationalNeighborPair]:
    laws = rr_laws(truthful_probability)
    pairs = []
    for background_ones in range(PUBLIC_M):
        counts = (PUBLIC_M - 1 - background_ones, background_ones)
        for q_type, p_type in ((0, 1), (1, 0)):
            pairs.append(
                RationalNeighborPair(
                    name=(
                        f"background_ones_{background_ones}_"
                        f"target_{q_type}_to_{p_type}"
                    ),
                    background_counts=counts,
                    background_laws=laws,
                    target_law_p=laws[p_type],
                    target_law_q=laws[q_type],
                    family_scope="fixed_public_size_replacement",
                )
            )
    return pairs


def _family_alpha_from_distributions(
    distributions: Sequence[
        tuple[dict[tuple[int, ...], Fraction], dict[tuple[int, ...], Fraction]]
    ],
) -> Fraction:
    pair_alphas = []
    for P, Q in distributions:
        alpha = two_sided_min_alpha_fraction(P, Q, DELTA)
        if alpha is None:
            raise AssertionError("finite replacement family has infinite budget")
        pair_alphas.append(alpha)
    return max(pair_alphas)


@lru_cache(maxsize=None)
def exact_family_alpha(method: str, truthful_probability: Fraction) -> Fraction:
    """Exact mechanism-level replacement budget for one rational RR law."""
    p = F(truthful_probability)
    if method == "Padded Ideal":
        distributions = [
            (
                padded_ideal_distribution(ones, p),
                padded_ideal_distribution(ones + 1, p),
            )
            for ones in range(PUBLIC_M)
        ]
        return _family_alpha_from_distributions(distributions)
    if method not in ("Drop", "Pool"):
        raise ValueError(f"unknown method {method}")
    decoder = method.lower()
    distributions = [
        exhaustive_neighbor_distributions(
            pair,
            decoder,
            SLOT_N,
            RELEASE_M,
            PUBLIC_NU,
        )
        for pair in rational_replacement_pairs(p)
    ]
    return _family_alpha_from_distributions(distributions)


@dataclass(frozen=True)
class Calibration:
    method: str
    truthful_probability: Fraction
    exact_alpha: Fraction
    next_probability: Fraction | None
    next_exact_alpha: Fraction | None


def calibrate(method: str) -> tuple[Calibration, list[dict]]:
    trace_by_probability: dict[Fraction, dict] = {}

    def evaluate(probability: Fraction, phase: str) -> tuple[Fraction, bool]:
        alpha = exact_family_alpha(method, probability)
        certified = alpha <= TARGET_ALPHA
        trace_by_probability[probability] = {
            "method": method,
            "truthful_probability_fraction": fraction_text(probability),
            "truthful_probability": float(probability),
            "local_epsilon0": math.log(float(probability / (1 - probability))),
            "exact_alpha_fraction": fraction_text(alpha),
            "exact_epsilon_prime": epsilon_from_alpha_fraction(alpha),
            "target_alpha_fraction": fraction_text(TARGET_ALPHA),
            "target_epsilon_prime": TARGET_EPSILON,
            "delta_fraction": fraction_text(DELTA),
            "delta": float(DELTA),
            "certified_at_common_target": str(certified).lower(),
            "calibration_grid": (
                f"exact rational bisection on p=a/{GRID_DENOMINATOR}, "
                f"a={GRID_MIN_NUMERATOR}..{GRID_MAX_NUMERATOR}; "
                f"global coarse scan p=a/{COARSE_DENOMINATOR}"
            ),
            "calibration_phase": phase,
            "adjacency": "fixed_size_replacement",
            "accounting_scope": "all_binary_dataset_compositions_and_both_replacement_directions",
            "source_type": "exact_rational_accountant",
            "arithmetic": "fractions.Fraction",
        }
        return alpha, certified

    # Exhaust the coarse global grid and require a single certified prefix.
    coarse_flags = []
    for numerator in COARSE_NUMERATORS:
        _, certified = evaluate(F(numerator, COARSE_DENOMINATOR), "coarse_global_scan")
        coarse_flags.append(certified)
    first_false = next((i for i, flag in enumerate(coarse_flags) if not flag), None)
    if first_false is None or not any(coarse_flags):
        raise AssertionError(f"coarse calibration grid does not bracket {method}")
    if any(coarse_flags[first_false:]):
        raise AssertionError(f"coarse exact profile re-enters the feasible region for {method}")

    low = GRID_MIN_NUMERATOR
    high = GRID_MAX_NUMERATOR
    selected_numerator: int | None = None
    selected_alpha: Fraction | None = None
    while low <= high:
        middle = (low + high) // 2
        alpha, certified = evaluate(F(middle, GRID_DENOMINATOR), "fine_exact_bisection")
        if certified:
            selected_numerator = middle
            selected_alpha = alpha
            low = middle + 1
        else:
            high = middle - 1
    if selected_numerator is None or selected_alpha is None:
        raise AssertionError(f"fine calibration grid has no feasible point for {method}")

    selected_probability = F(selected_numerator, GRID_DENOMINATOR)
    # Re-evaluate the boundary explicitly so both rows appear in the trace.
    selected_alpha, selected_ok = evaluate(selected_probability, "selected_boundary")
    next_probability = F(selected_numerator + 1, GRID_DENOMINATOR)
    next_alpha, next_ok = evaluate(next_probability, "first_rejected_boundary")
    if not selected_ok or next_ok:
        raise AssertionError(f"exact bisection did not isolate the target for {method}")
    if TARGET_EPSILON - epsilon_from_alpha_fraction(selected_alpha) > 1e-3:
        raise AssertionError(f"exact calibration remains too far below target for {method}")

    return (
        Calibration(
            method=method,
            truthful_probability=selected_probability,
            exact_alpha=selected_alpha,
            next_probability=next_probability,
            next_exact_alpha=next_alpha,
        ),
        [trace_by_probability[p] for p in sorted(trace_by_probability)],
    )


def exact_utility_distribution(
    method: str,
    truthful_probability: Fraction,
) -> dict[tuple[int, ...], Fraction]:
    if method == "Padded Ideal":
        return padded_ideal_distribution(TRUE_ONES, truthful_probability)
    laws = rr_laws(truthful_probability)
    dataset = RationalNeighborPair(
        name="utility_dataset_two_ones_of_three",
        background_counts=(PUBLIC_M - TRUE_ONES, TRUE_ONES),
        background_laws=laws,
        target_law_p=laws[0],
        target_law_q=None,
        family_scope="fixed_utility_dataset_not_a_neighbor_pair",
    )
    return exhaustive_output_distribution(
        dataset,
        method.lower(),
        SLOT_N,
        RELEASE_M,
        PUBLIC_NU,
        side="Q",
    )


def clipped_count_estimate(
    histogram: Sequence[int],
    method: str,
    truthful_probability: Fraction,
) -> Fraction:
    p = F(truthful_probability)
    q = 1 - p
    contrast = p - q
    released_sum = sum(F(value * count) for value, count in enumerate(histogram))
    if method == "Drop":
        survival = F(SLOT_N - 1, SLOT_N) ** (PUBLIC_M - 1)
        expected_real_count = PUBLIC_M * survival
        expected_dummy_sum = F(1, 2) * (RELEASE_M - expected_real_count)
        estimate = (
            released_sum - expected_dummy_sum - survival * PUBLIC_M * q
        ) / (
            survival * contrast
        )
    elif method == "Padded Ideal":
        expected_dummy_sum = F(1, 2) * (RELEASE_M - PUBLIC_M)
        estimate = (released_sum - expected_dummy_sum - PUBLIC_M * q) / contrast
    elif method == "Pool":
        expected_occupied = SLOT_N * (1 - F(SLOT_N - 1, SLOT_N) ** PUBLIC_M)
        expected_dummy_sum = F(1, 2) * (RELEASE_M - expected_occupied)
        estimate = (released_sum - expected_dummy_sum - PUBLIC_M * q) / contrast
    else:
        raise ValueError(f"unknown method {method}")
    return min(F(PUBLIC_M), max(F(0), estimate))


def exact_utility_metrics(
    distribution: RationalDistribution,
    method: str,
    truthful_probability: Fraction,
) -> tuple[Fraction, Fraction]:
    mae = F(0)
    mse = F(0)
    for histogram, probability in distribution.items():
        estimate = clipped_count_estimate(histogram, method, truthful_probability)
        error = estimate - TRUE_ONES
        mae += probability * abs(error)
        mse += probability * error * error
    return mae, mse


def simulate_complete_release(
    method: str,
    truthful_probability: Fraction,
    replicates: int = MC_REPLICATES,
) -> tuple[float, float, float, float]:
    """Independently simulate reports, slots, padding, and final permutation."""
    method_index = METHODS.index(method)
    rng = np.random.default_rng(MC_SEED + 100_000 * method_index)
    p = float(truthful_probability)
    raw = np.asarray(
        (0,) * (PUBLIC_M - TRUE_ONES) + (1,) * TRUE_ONES,
        dtype=np.int64,
    )
    errors = np.empty(replicates, dtype=float)
    real_counts = np.empty(replicates, dtype=float)
    for repeat in range(replicates):
        truthful = rng.random(PUBLIC_M) < p
        reports = np.where(truthful, raw, 1 - raw).astype(np.int64)
        if method == "Padded Ideal":
            real = reports.tolist()
        else:
            slots = rng.integers(0, SLOT_N, size=PUBLIC_M)
            occupancy = np.bincount(slots, minlength=SLOT_N)
            if method == "Drop":
                real = [
                    int(report)
                    for report, slot in zip(reports, slots)
                    if occupancy[slot] == 1
                ]
            elif method == "Pool":
                sums = np.bincount(slots, weights=reports, minlength=SLOT_N)
                real = [int(sums[slot]) for slot in range(SLOT_N) if occupancy[slot] > 0]
            else:
                raise ValueError(f"unknown method {method}")
        dummies = rng.integers(0, 2, size=RELEASE_M - len(real)).tolist()
        release = np.asarray(real + dummies, dtype=np.int64)
        if len(release) != RELEASE_M:
            raise AssertionError("complete release does not have public length M")
        if bool(np.any(release >= PAYLOAD_K)):
            raise AssertionError("Pool payload wrapped or exceeded Z_K")
        rng.shuffle(release)
        histogram = np.bincount(release, minlength=PAYLOAD_K)
        estimate = float(clipped_count_estimate(histogram, method, truthful_probability))
        errors[repeat] = abs(estimate - TRUE_ONES)
        real_counts[repeat] = len(real)
    mae = float(np.mean(errors))
    standard_error = float(np.std(errors, ddof=1) / math.sqrt(replicates))
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    return mae, standard_error, rmse, float(np.mean(real_counts))


def float_recurrence_crosscheck(
    calibration: Calibration,
) -> tuple[float | None, float | None]:
    if calibration.method == "Padded Ideal":
        return None, None
    rational_pairs = rational_replacement_pairs(calibration.truthful_probability)
    float_pairs = [
        NeighborPair(
            name=pair.name,
            background_counts=pair.background_counts,
            background_laws=tuple(
                tuple(float(value) for value in law)
                for law in pair.background_laws
            ),
            target_law_p=tuple(float(value) for value in pair.target_law_p),
            target_law_q=tuple(float(value) for value in pair.target_law_q or ()),
            family_scope=pair.family_scope,
        )
        for pair in rational_pairs
    ]
    result = account_pair_family(
        float_pairs,
        calibration.method.lower(),
        SLOT_N,
        RELEASE_M,
        tuple(float(value) for value in PUBLIC_NU),
        float(DELTA),
    )
    return result.alpha_nominal, abs(result.alpha_nominal - float(calibration.exact_alpha))


def generate_rows() -> tuple[list[dict], list[dict]]:
    utility_rows = []
    calibration_rows = []
    for method in METHODS:
        calibration, trace = calibrate(method)
        calibration_rows.extend(trace)
        distribution = exact_utility_distribution(
            method, calibration.truthful_probability
        )
        if sum(distribution.values(), F(0)) != 1:
            raise AssertionError("exact utility distribution lost mass")
        if any(sum(histogram) != RELEASE_M for histogram in distribution):
            raise AssertionError("exact utility distribution is not length M")
        exact_mae, exact_mse = exact_utility_metrics(
            distribution,
            method,
            calibration.truthful_probability,
        )
        mc_mae, mc_se, mc_rmse, mean_real_count = simulate_complete_release(
            method,
            calibration.truthful_probability,
        )
        float_alpha, float_error = float_recurrence_crosscheck(calibration)
        p = calibration.truthful_probability
        next_p = calibration.next_probability
        next_alpha = calibration.next_exact_alpha
        utility_rows.append(
            {
                "method": method,
                "decoder": {
                    "Padded Ideal": "all_reports_same_M_padding",
                    "Drop": "drop",
                    "Pool": "pool",
                }[method],
                "task": "binary_count",
                "dataset": "fixed_bits_0_1_1",
                "true_count": TRUE_ONES,
                "truthful_probability_fraction": fraction_text(p),
                "truthful_probability": float(p),
                "local_epsilon0": math.log(float(p / (1 - p))),
                "target_alpha_fraction": fraction_text(TARGET_ALPHA),
                "target_epsilon_prime": TARGET_EPSILON,
                "exact_alpha_fraction": fraction_text(calibration.exact_alpha),
                "exact_epsilon_prime": epsilon_from_alpha_fraction(
                    calibration.exact_alpha
                ),
                "exact_delta_fraction": fraction_text(DELTA),
                "delta": float(DELTA),
                "certified_at_common_target": "true",
                "next_grid_truthful_probability_fraction": (
                    fraction_text(next_p) if next_p is not None else ""
                ),
                "next_grid_exact_alpha_fraction": (
                    fraction_text(next_alpha) if next_alpha is not None else ""
                ),
                "next_grid_exceeds_target": str(
                    next_alpha is not None and next_alpha > TARGET_ALPHA
                ).lower(),
                "exact_mean_absolute_count_error_fraction": fraction_text(exact_mae),
                "exact_mean_absolute_count_error": float(exact_mae),
                "exact_rmse": math.sqrt(float(exact_mse)),
                "mc_mean_absolute_count_error": mc_mae,
                "mc_mean_absolute_count_error_standard_error": mc_se,
                "mc_rmse": mc_rmse,
                "mc_abs_error_vs_exact_mae": abs(mc_mae - float(exact_mae)),
                "mc_replicates": MC_REPLICATES,
                "observed_mean_real_output_count": mean_real_count,
                "m": PUBLIC_M,
                "N": SLOT_N,
                "M": RELEASE_M,
                "rho": PUBLIC_M / SLOT_N,
                "tau": RELEASE_M / PUBLIC_M,
                "K": PAYLOAD_K,
                "payload_group": f"Z_{PAYLOAD_K}",
                "pool_wraparound_possible": "false",
                "dummy_law": "public_bernoulli_half_on_zero_one",
                "dummy_law_fraction": ";".join(
                    fraction_text(value) for value in PUBLIC_NU
                ),
                "full_release": "true",
                "output_length_observed": RELEASE_M,
                "dummy_labels_visible": "false",
                "estimator_observes_real_output_count": "false",
                "final_release_permuted": "true",
                "adjacency": "fixed_size_replacement",
                "accounting_scope": "all_binary_dataset_compositions_and_both_replacement_directions",
                "calibration_grid": (
                    f"exact rational bisection on p=a/{GRID_DENOMINATOR}, "
                    f"a={GRID_MIN_NUMERATOR}..{GRID_MAX_NUMERATOR}; "
                    f"global coarse scan p=a/{COARSE_DENOMINATOR}"
                ),
                "privacy_source_type": "exact_rational_accountant",
                "utility_source_type": "exact_output_distribution_plus_independent_monte_carlo",
                "arithmetic": "fractions.Fraction",
                "exact_output_support": len(distribution),
                "float_recurrence_alpha_crosscheck": (
                    float_alpha if float_alpha is not None else ""
                ),
                "float_recurrence_abs_error": (
                    float_error if float_error is not None else ""
                ),
                "provenance": (
                    "complete_length_M_release; exact rational family calibration; "
                    "exact utility law; independently seeded report-slot-padding-permutation simulation"
                ),
            }
        )
    return utility_rows, calibration_rows


def generate_all() -> dict[str, int]:
    utility_rows, calibration_rows = generate_rows()
    utility_filename = "fig_exact_certified_utility.csv"
    calibration_filename = "fig_exact_certified_utility_calibration.csv"
    write_csv(DATA_DIR / utility_filename, utility_rows)
    write_csv(DATA_DIR / calibration_filename, calibration_rows)
    return {
        utility_filename: len(utility_rows),
        calibration_filename: len(calibration_rows),
    }


if __name__ == "__main__":
    for filename, count in generate_all().items():
        print(f"{filename}: {count} rows")
