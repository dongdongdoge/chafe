"""Large-workload utility under exact finite-law privacy calibration.

This experiment places the scalable homogeneous matched-Drop accountant next
to complete length-M releases on two real binary-count workloads.  Privacy
and utility have deliberately separate provenance and are not conflated:

* Privacy calibration is deterministic.  It evaluates the exact finite output
  law for one declared homogeneous replacement pair (all-zero versus one
  positive) under binary randomized response and padding matched to the
  all-zero report law.  The implementation uses binomial tail
  identities; it does not invoke the paper's quadratic proxy or Monte Carlo.
* Adult and Covertype are independently seeded transferred-reference
  sensitivity checks.  Each method uses the local epsilon selected for the
  separate homogeneous reference, so these rows are neither shared-local-
  budget nor shared-central-budget comparisons.  Every replicate constructs
  the complete length-M release, and the estimator sees neither dummy labels
  nor the real-output count.

The exact profile is two-sided for one declared homogeneous-background pair. It is
not advertised as a mechanism-wide worst-case accountant over
arbitrary heterogeneous datasets.  That qualification is recorded in every
output row.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

import fetch_adult_dataset
import fetch_covertype_dataset
from chafe_eval_lib import DATA_DIR, binom_pmf, mean_and_se, two_sided_profile, write_csv


SAMPLE_M = 100_000
TAU = 2.0
RELEASE_M = int(TAU * SAMPLE_M)
DELTA = 1e-6
TARGET_EPSILONS = (0.1, 0.3, 0.7)
RHOS = (0.5, 1.0, 2.0)
MC_REPLICATES = 100
DATASET_SEED = 26_071_100
MC_SEED = 26_071_200
CALIBRATION_EPS0_MAX = 25.0
CALIBRATION_SCAN_POINTS = 101
CALIBRATION_TOL = 1e-11
CALIBRATION_PROFILE_RELATIVE_SLACK = 1e-8

PRIVACY_FAMILY = "single_homogeneous_background_endpoint_replacement_pair"
PRIVACY_SCOPE = "pairwise_two_sided_profile_only_not_all_binary_compositions"
PRIVACY_SOURCE = "exact_finite_law_numerical"
PRIVACY_ARITHMETIC = "affine_likelihood_ratio_with_scipy_binomial_tail_cdf_float64"
UTILITY_SOURCE = "monte_carlo_complete_length_M_release"


@dataclass(frozen=True)
class Calibration:
    method: str
    rho: float | None
    N: int | None
    survival: float
    target_epsilon: float
    local_epsilon0: float
    achieved_epsilon: float
    profile_at_target: float
    feasible_eps0: float
    infeasible_eps0: float
    direct_pmf_profile: float
    profile_limit: float


@dataclass(frozen=True)
class Workload:
    dataset: str
    predicate: str
    bits: np.ndarray
    sampling_model: str
    source_rows: int
    evidence_scope: str


@dataclass(frozen=True)
class ReleaseObservation:
    release: np.ndarray
    output_sum: int
    estimate: float
    real_output_count: int


def brr_probabilities(eps0: float) -> tuple[float, float]:
    """Return (Pr[1|1], Pr[1|0]) for symmetric binary RR."""
    if eps0 < 0.0 or not math.isfinite(eps0):
        raise ValueError("eps0 must be finite and nonnegative")
    q = 1.0 / (1.0 + math.exp(eps0))
    return 1.0 - q, q


def singleton_survival(m: int, N: int) -> float:
    """Exact probability that one tagged report is a singleton."""
    if m <= 0 or N <= 0:
        raise ValueError("m and N must be positive")
    return (1.0 - 1.0 / float(N)) ** (m - 1)


def affine_likelihood_parameters(
    M: int,
    eps0: float,
    survival: float,
) -> tuple[float, float, float]:
    """Return q,c,d for P(k)/Q(k)=c+d*k and Q=Bin(M,q).

    Q is the complete matched-padded release for the all-zero dataset.  The
    neighboring dataset changes one zero to one.  The changed report reaches
    the Drop release with probability ``survival``; padded Ideal uses one.
    """
    if M <= 0:
        raise ValueError("M must be positive")
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival must lie in [0,1]")
    p, q = brr_probabilities(eps0)
    if eps0 == 0.0 or survival == 0.0:
        return q, 1.0, 0.0
    c = 1.0 - survival + survival * q / p
    d = survival * (p / q - q / p) / float(M)
    return q, c, d


def _upper_probability_and_first_moment(
    M: int,
    q: float,
    threshold: int,
) -> tuple[float, float]:
    """Pr[K>=threshold] and E[K 1{K>=threshold}], K~Bin(M,q)."""
    if threshold <= 0:
        return 1.0, float(M) * q
    if threshold > M:
        return 0.0, 0.0
    probability = float(binom.sf(threshold - 1, M, q))
    first_moment = float(M) * q * float(binom.sf(threshold - 2, M - 1, q))
    return probability, first_moment


def _lower_probability_and_first_moment(
    M: int,
    q: float,
    threshold: int,
) -> tuple[float, float]:
    """Pr[K<=threshold] and E[K 1{K<=threshold}], K~Bin(M,q)."""
    if threshold < 0:
        return 0.0, 0.0
    if threshold >= M:
        return 1.0, float(M) * q
    probability = float(binom.cdf(threshold, M, q))
    first_moment = (
        float(M) * q * float(binom.cdf(threshold - 1, M - 1, q))
        if threshold >= 1
        else 0.0
    )
    return probability, first_moment


def exact_profile(
    *,
    M: int,
    eps0: float,
    survival: float,
    central_epsilon: float,
) -> float:
    """Two-sided hockey-stick profile of the declared exact finite law.

    The calculation sums the binomial law analytically through tail
    probabilities and truncated first moments.  No samples or asymptotic
    occupancy approximation enter this function.
    """
    if central_epsilon < 0.0 or not math.isfinite(central_epsilon):
        raise ValueError("central_epsilon must be finite and nonnegative")
    q, c, d = affine_likelihood_parameters(M, eps0, survival)
    if d == 0.0:
        return 0.0
    alpha = math.exp(central_epsilon)

    # Forward direction: E_Q[(c+dK-alpha)_+].
    forward_threshold = math.floor((alpha - c) / d) + 1
    while forward_threshold <= M and c + d * forward_threshold <= alpha:
        forward_threshold += 1
    while forward_threshold > 0 and c + d * (forward_threshold - 1) > alpha:
        forward_threshold -= 1
    upper_probability, upper_first_moment = _upper_probability_and_first_moment(
        M, q, forward_threshold
    )
    forward = (
        (c - alpha) * upper_probability + d * upper_first_moment
    )

    # Reverse direction: E_Q[(1-alpha(c+dK))_+].
    reverse_threshold = math.ceil((1.0 / alpha - c) / d) - 1
    while reverse_threshold >= 0 and alpha * (c + d * reverse_threshold) >= 1.0:
        reverse_threshold -= 1
    while reverse_threshold < M and alpha * (c + d * (reverse_threshold + 1)) < 1.0:
        reverse_threshold += 1
    lower_probability, lower_first_moment = _lower_probability_and_first_moment(
        M, q, reverse_threshold
    )
    reverse = (
        (1.0 - alpha * c) * lower_probability
        - alpha * d * lower_first_moment
    )
    return max(0.0, forward, reverse)


def direct_pmf_profile(
    *,
    M: int,
    eps0: float,
    survival: float,
    central_epsilon: float,
) -> float:
    """Independent O(M) PMF cross-check of :func:`exact_profile`."""
    q, c, d = affine_likelihood_parameters(M, eps0, survival)
    Q = binom_pmf(M, q).copy()
    k = np.arange(M + 1, dtype=float)
    P = Q * (c + d * k)
    P /= P.sum()
    return two_sided_profile(P, Q, central_epsilon)


def exact_central_epsilon(
    *,
    M: int,
    eps0: float,
    survival: float,
    delta: float,
    tolerance: float = 1e-12,
) -> float:
    """Invert the exact two-sided hockey-stick profile in epsilon."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if exact_profile(M=M, eps0=eps0, survival=survival, central_epsilon=0.0) <= delta:
        return 0.0
    low = 0.0
    high = max(1.0, eps0)
    while exact_profile(
        M=M,
        eps0=eps0,
        survival=survival,
        central_epsilon=high,
    ) > delta:
        high *= 2.0
        if high > 100.0:
            raise RuntimeError("failed to bracket exact central epsilon")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if exact_profile(
            M=M,
            eps0=eps0,
            survival=survival,
            central_epsilon=middle,
        ) <= delta:
            high = middle
        else:
            low = middle
    return high


def calibrate_local_epsilon(
    *,
    method: str,
    target_epsilon: float,
    rho: float | None,
    m: int = SAMPLE_M,
    M: int = RELEASE_M,
    delta: float = DELTA,
) -> Calibration:
    """Find the largest eps0 meeting one exact central privacy target."""
    if method == "Padded Ideal":
        if rho is not None:
            raise ValueError("Padded Ideal does not use rho")
        N = None
        survival = 1.0
    elif method == "Drop":
        if rho is None or rho <= 0.0:
            raise ValueError("Drop requires positive rho")
        N = int(math.ceil(float(m) / rho))
        survival = singleton_survival(m, N)
    else:
        raise ValueError("method must be Padded Ideal or Drop")

    scan = np.linspace(0.0, CALIBRATION_EPS0_MAX, CALIBRATION_SCAN_POINTS)
    profile_limit = delta * (1.0 - CALIBRATION_PROFILE_RELATIVE_SLACK)
    profiles = [
        exact_profile(
            M=M,
            eps0=float(eps0),
            survival=survival,
            central_epsilon=target_epsilon,
        )
        for eps0 in scan
    ]
    feasible = [value <= profile_limit for value in profiles]
    first_infeasible = next(
        (index for index, is_feasible in enumerate(feasible) if not is_feasible),
        None,
    )
    if first_infeasible is None or first_infeasible == 0:
        raise AssertionError("coarse exact-law scan did not bracket eps0")
    if any(feasible[first_infeasible:]):
        raise AssertionError("exact profile re-entered the feasible region")

    low = float(scan[first_infeasible - 1])
    high = float(scan[first_infeasible])
    while high - low > CALIBRATION_TOL:
        middle = 0.5 * (low + high)
        profile = exact_profile(
            M=M,
            eps0=middle,
            survival=survival,
            central_epsilon=target_epsilon,
        )
        if profile <= profile_limit:
            low = middle
        else:
            high = middle
    eps0 = low
    profile = exact_profile(
        M=M,
        eps0=eps0,
        survival=survival,
        central_epsilon=target_epsilon,
    )
    achieved = exact_central_epsilon(
        M=M,
        eps0=eps0,
        survival=survival,
        delta=delta,
    )
    pmf_profile = direct_pmf_profile(
        M=M,
        eps0=eps0,
        survival=survival,
        central_epsilon=target_epsilon,
    )
    if abs(profile - pmf_profile) > 2e-11:
        raise AssertionError("binomial-tail and direct-PMF profiles disagree")
    return Calibration(
        method=method,
        rho=rho,
        N=N,
        survival=survival,
        target_epsilon=target_epsilon,
        local_epsilon0=eps0,
        achieved_epsilon=achieved,
        profile_at_target=profile,
        feasible_eps0=low,
        infeasible_eps0=high,
        direct_pmf_profile=pmf_profile,
        profile_limit=profile_limit,
    )


def load_workloads(m: int = SAMPLE_M) -> list[Workload]:
    """Create fixed real-data binary-count workloads of exactly ``m`` rows."""
    adult_values: list[int] = []
    with fetch_adult_dataset.ensure_adult_dataset().open(newline="") as handle:
        for row in csv.DictReader(handle):
            adult_values.append(int(row["income"] == ">50K"))
    adult_source = np.asarray(adult_values, dtype=np.uint8)
    adult_rng = np.random.default_rng(DATASET_SEED)
    adult_indices = adult_rng.choice(len(adult_source), size=m, replace=True)

    cover_values: list[int] = []
    with fetch_covertype_dataset.ensure_covertype_dataset().open(newline="") as handle:
        for row in csv.DictReader(handle):
            cover_values.append(int(row["cover_type"] == "2"))
    cover_source = np.asarray(cover_values, dtype=np.uint8)
    if m > len(cover_source):
        raise ValueError("Covertype sample exceeds the source table")
    cover_rng = np.random.default_rng(DATASET_SEED + 1)
    cover_indices = cover_rng.choice(len(cover_source), size=m, replace=False)

    return [
        Workload(
            dataset="Adult",
            predicate="income_>50K",
            bits=adult_source[adult_indices],
            sampling_model="fixed_seed_empirical_bootstrap_with_replacement",
            source_rows=len(adult_source),
            evidence_scope="heterogeneous_real_data_transferred_reference_sensitivity",
        ),
        Workload(
            dataset="Covertype",
            predicate="cover_type_2",
            bits=cover_source[cover_indices],
            sampling_model="fixed_seed_without_replacement_sample",
            source_rows=len(cover_source),
            evidence_scope="heterogeneous_real_data_transferred_reference_sensitivity",
        ),
    ]


def exact_family_workloads(m: int = SAMPLE_M) -> list[Workload]:
    """Return the two data vectors that define the exact neighboring family."""
    all_zero = np.zeros(m, dtype=np.uint8)
    one_positive = all_zero.copy()
    one_positive[0] = 1
    return [
        Workload(
            dataset="Synthetic-Homogeneous",
            predicate="all_zero_count",
            bits=all_zero,
            sampling_model="deterministic_declared_neighbor_endpoint",
            source_rows=m,
            evidence_scope="declared_exact_single_pair",
        ),
        Workload(
            dataset="Synthetic-Homogeneous",
            predicate="one_positive_count",
            bits=one_positive,
            sampling_model="deterministic_declared_neighbor_endpoint",
            source_rows=m,
            evidence_scope="declared_exact_single_pair",
        ),
    ]


def simulate_complete_release(
    *,
    bits: np.ndarray,
    calibration: Calibration,
    M: int,
    seed: int,
) -> ReleaseObservation:
    """Generate reports, hidden slots, dummies, and the final permutation."""
    raw = np.asarray(bits, dtype=np.uint8)
    m = len(raw)
    if M < m:
        raise ValueError("M must be at least m for the padded Ideal comparison")
    rng = np.random.default_rng(seed)
    p, q = brr_probabilities(calibration.local_epsilon0)
    report_one_probability = np.where(raw == 1, p, q)
    reports = (rng.random(m) < report_one_probability).astype(np.uint8)

    if calibration.method == "Padded Ideal":
        real = reports
        estimator_survival = 1.0
    elif calibration.method == "Drop":
        if calibration.N is None:
            raise AssertionError("Drop calibration has no slot count")
        slots = rng.integers(0, calibration.N, size=m)
        occupancies = np.bincount(slots, minlength=calibration.N)
        real = reports[occupancies[slots] == 1]
        estimator_survival = calibration.survival
    else:
        raise ValueError("unsupported method")

    dummy_count = M - len(real)
    if dummy_count < 0:
        raise AssertionError("real decoder output exceeds M")
    dummies = (rng.random(dummy_count) < q).astype(np.uint8)
    release = np.concatenate((real, dummies))
    rng.shuffle(release)
    if len(release) != M:
        raise AssertionError("release does not have public length M")

    output_sum = int(release.sum(dtype=np.int64))
    denominator = estimator_survival * (p - q)
    estimate = (float(output_sum) - float(M) * q) / denominator
    estimate = min(float(m), max(0.0, estimate))
    return ReleaseObservation(
        release=release,
        output_sum=output_sum,
        estimate=estimate,
        real_output_count=len(real),
    )


def exact_family_utility_metrics(
    *,
    calibration: Calibration,
    true_count: int,
    m: int = SAMPLE_M,
    M: int = RELEASE_M,
) -> dict[str, float]:
    """Exact-law numerical utility for an endpoint of the declared pair.

    The final number of ones is sufficient for the count estimator.  The
    all-zero endpoint has law Q=Bin(M,q); the one-positive endpoint has law
    P(k)=Q(k)(c+dk).  This calculation sums the complete finite output law and
    is independent of the end-to-end Monte Carlo validation.
    """
    if true_count not in (0, 1):
        raise ValueError("the declared exact family has true count zero or one")
    p, q = brr_probabilities(calibration.local_epsilon0)
    _, c, d = affine_likelihood_parameters(
        M, calibration.local_epsilon0, calibration.survival
    )
    counts = np.arange(M + 1, dtype=float)
    distribution = binom_pmf(M, q).copy()
    if true_count == 1:
        distribution *= c + d * counts
        distribution /= distribution.sum()
    estimates = (
        counts - float(M) * q
    ) / (
        calibration.survival * (p - q)
    )
    estimates = np.clip(estimates, 0.0, float(m))
    errors = estimates - float(true_count)
    return {
        "mae": float(np.dot(distribution, np.abs(errors))),
        "mse": float(np.dot(distribution, np.square(errors))),
        "bias": float(np.dot(distribution, errors)),
        "mass_error": abs(float(distribution.sum()) - 1.0),
    }


def _calibration_rows(calibrations: list[Calibration]) -> list[dict]:
    ideal_eps0 = {
        calibration.target_epsilon: calibration.local_epsilon0
        for calibration in calibrations
        if calibration.method == "Padded Ideal"
    }
    rows = []
    for calibration in calibrations:
        rows.append(
            {
                "method": calibration.method,
                "decoder": "all_reports_same_M_padding" if calibration.method == "Padded Ideal" else "drop",
                "target_epsilon_prime": calibration.target_epsilon,
                "achieved_exact_epsilon_prime": calibration.achieved_epsilon,
                "delta": DELTA,
                "local_epsilon0": calibration.local_epsilon0,
                "local_epsilon0_over_padded_ideal": (
                    calibration.local_epsilon0 / ideal_eps0[calibration.target_epsilon]
                ),
                "profile_delta_at_target_epsilon": calibration.profile_at_target,
                "profile_limit_with_relative_slack": calibration.profile_limit,
                "accounting_delta": DELTA,
                "calibration_profile_relative_slack": CALIBRATION_PROFILE_RELATIVE_SLACK,
                "direct_pmf_profile_delta_at_target_epsilon": calibration.direct_pmf_profile,
                "tail_vs_direct_pmf_abs_error": abs(
                    calibration.profile_at_target - calibration.direct_pmf_profile
                ),
                "feasible_eps0_bracket_endpoint": calibration.feasible_eps0,
                "infeasible_eps0_bracket_endpoint": calibration.infeasible_eps0,
                "eps0_bracket_width": calibration.infeasible_eps0 - calibration.feasible_eps0,
                "m": SAMPLE_M,
                "N": calibration.N if calibration.N is not None else "",
                "M": RELEASE_M,
                "rho": calibration.rho if calibration.rho is not None else "",
                "tau": TAU,
                "singleton_survival": calibration.survival,
                "adjacency": "fixed_size_replacement",
                "privacy_family": PRIVACY_FAMILY,
                "accounting_scope": PRIVACY_SCOPE,
                "declared_neighbor_pairs": 1,
                "pairwise_exact_profile_only": "true",
                "mechanism_wide_dp_claimed": "false",
                "background_law": "BRR(true_bit=0)",
                "target_q_law": "BRR(true_bit=0)",
                "target_p_law": "BRR(true_bit=1)",
                "dummy_law": "matched_BRR(true_bit=0)",
                "privacy_source_type": PRIVACY_SOURCE,
                "privacy_arithmetic": PRIVACY_ARITHMETIC,
                "proxy_used_in_calibration": "false",
                "monte_carlo_used_in_calibration": "false",
                "numerical_status": "exact_finite_law_float64_with_conservative_profile_slack_not_interval_certified",
            }
        )
    return rows


def generate_rows() -> tuple[list[dict], list[dict], list[dict]]:
    calibrations = [
        calibrate_local_epsilon(
            method="Padded Ideal",
            target_epsilon=target,
            rho=None,
        )
        for target in TARGET_EPSILONS
    ]
    calibrations.extend(
        calibrate_local_epsilon(
            method="Drop",
            target_epsilon=target,
            rho=rho,
        )
        for target in TARGET_EPSILONS
        for rho in RHOS
    )
    calibration_rows = _calibration_rows(calibrations)
    ideal_eps0 = {
        calibration.target_epsilon: calibration.local_epsilon0
        for calibration in calibrations
        if calibration.method == "Padded Ideal"
    }

    replicate_rows: list[dict] = []
    summary_rows: list[dict] = []
    workloads = exact_family_workloads() + load_workloads()
    for dataset_index, workload in enumerate(workloads):
        exact_family_utility = workload.evidence_scope == "declared_exact_single_pair"
        true_count = int(workload.bits.sum(dtype=np.int64))
        for calibration_index, calibration in enumerate(calibrations):
            absolute_errors = []
            squared_errors = []
            signed_errors = []
            real_counts = []
            for replicate in range(MC_REPLICATES):
                seed = MC_SEED + 1_000_000 * dataset_index + 10_000 * calibration_index + replicate
                observation = simulate_complete_release(
                    bits=workload.bits,
                    calibration=calibration,
                    M=RELEASE_M,
                    seed=seed,
                )
                signed_error = observation.estimate - true_count
                absolute_error = abs(signed_error)
                squared_error = signed_error * signed_error
                signed_errors.append(signed_error)
                absolute_errors.append(absolute_error)
                squared_errors.append(squared_error)
                real_counts.append(float(observation.real_output_count))
                replicate_rows.append(
                    {
                        "dataset": workload.dataset,
                        "predicate": workload.predicate,
                        "evidence_scope": workload.evidence_scope,
                        "method": calibration.method,
                        "exact_target_epsilon_prime": (
                            calibration.target_epsilon if exact_family_utility else ""
                        ),
                        "homogeneous_reference_target_epsilon_prime": calibration.target_epsilon,
                        "rho": calibration.rho if calibration.rho is not None else "",
                        "replicate": replicate,
                        "seed": seed,
                        "true_count": true_count,
                        "estimate": observation.estimate,
                        "signed_count_error": signed_error,
                        "absolute_count_error": absolute_error,
                        "squared_count_error": squared_error,
                        "output_sum": observation.output_sum,
                        "internal_real_output_count_diagnostic": observation.real_output_count,
                        "output_length": len(observation.release),
                        "utility_source_type": UTILITY_SOURCE,
                        "reference_privacy_source_type": PRIVACY_SOURCE,
                        "central_epsilon_claimed_for_this_workload": str(
                            exact_family_utility
                        ).lower(),
                        "comparison_basis": (
                            "same_declared_pairwise_exact_central_profile"
                            if exact_family_utility
                            else "transferred_reference_sensitivity_not_shared_eps0_or_central_epsilon"
                        ),
                        "shared_local_epsilon0_across_methods": "false",
                        "pairwise_exact_profile_only": str(exact_family_utility).lower(),
                        "mechanism_wide_dp_claimed": "false",
                    }
                )

            mae, mae_se = mean_and_se(absolute_errors)
            bias, bias_se = mean_and_se(signed_errors)
            mean_real, real_se = mean_and_se(real_counts)
            rmse = math.sqrt(float(np.mean(squared_errors)))
            exact_utility = (
                exact_family_utility_metrics(
                    calibration=calibration,
                    true_count=true_count,
                )
                if exact_family_utility
                else None
            )
            summary_rows.append(
                {
                    "dataset": workload.dataset,
                    "predicate": workload.predicate,
                    "evidence_scope": workload.evidence_scope,
                    "task": "binary_count",
                    "method": calibration.method,
                    "decoder": "all_reports_same_M_padding" if calibration.method == "Padded Ideal" else "drop",
                    "true_count": true_count,
                    "mean_absolute_count_error": mae,
                    "mean_absolute_count_error_standard_error": mae_se,
                    "rmse_count_error": rmse,
                    "mean_signed_count_error": bias,
                    "mean_signed_count_error_standard_error": bias_se,
                    "exact_law_mean_absolute_count_error": (
                        exact_utility["mae"] if exact_utility is not None else ""
                    ),
                    "exact_law_rmse_count_error": (
                        math.sqrt(exact_utility["mse"])
                        if exact_utility is not None
                        else ""
                    ),
                    "exact_law_mean_signed_count_error": (
                        exact_utility["bias"] if exact_utility is not None else ""
                    ),
                    "mc_abs_error_vs_exact_law_mae": (
                        abs(mae - exact_utility["mae"])
                        if exact_utility is not None
                        else ""
                    ),
                    "exact_utility_distribution_mass_error": (
                        exact_utility["mass_error"]
                        if exact_utility is not None
                        else ""
                    ),
                    "mae_over_m": mae / SAMPLE_M,
                    "mae_over_true_count": (
                        mae / true_count if true_count > 0 else ""
                    ),
                    "exact_target_epsilon_prime": (
                        calibration.target_epsilon if exact_family_utility else ""
                    ),
                    "homogeneous_reference_target_epsilon_prime": calibration.target_epsilon,
                    "homogeneous_reference_achieved_exact_epsilon_prime": calibration.achieved_epsilon,
                    "delta": DELTA,
                    "local_epsilon0": calibration.local_epsilon0,
                    "local_epsilon0_over_reference_padded_ideal": (
                        calibration.local_epsilon0 / ideal_eps0[calibration.target_epsilon]
                    ),
                    "homogeneous_reference_profile_delta_at_target_epsilon": calibration.profile_at_target,
                    "homogeneous_reference_profile_limit_with_slack": calibration.profile_limit,
                    "calibration_profile_relative_slack": CALIBRATION_PROFILE_RELATIVE_SLACK,
                    "m": SAMPLE_M,
                    "N": calibration.N if calibration.N is not None else "",
                    "M": RELEASE_M,
                    "rho": calibration.rho if calibration.rho is not None else "",
                    "tau": TAU,
                    "singleton_survival": calibration.survival,
                    "observed_mean_real_output_count": mean_real,
                    "observed_real_output_count_standard_error": real_se,
                    "full_release": "true",
                    "output_length_observed": RELEASE_M,
                    "dummy_labels_visible": "false",
                    "estimator_observes_real_output_count": "false",
                    "final_release_permuted": "true",
                    "dummy_law": "matched_BRR(true_bit=0)",
                    "randomizer": "binary_randomized_response",
                    "adjacency": "fixed_size_replacement",
                    "reference_privacy_family": PRIVACY_FAMILY,
                    "reference_accounting_scope": PRIVACY_SCOPE,
                    "declared_neighbor_pairs": 1,
                    "pairwise_exact_profile_only": str(exact_family_utility).lower(),
                    "mechanism_wide_dp_claimed": "false",
                    "central_epsilon_claimed_for_this_workload": str(
                        exact_family_utility
                    ).lower(),
                    "reference_privacy_source_type": PRIVACY_SOURCE,
                    "reference_privacy_arithmetic": PRIVACY_ARITHMETIC,
                    "proxy_used_in_calibration": "false",
                    "monte_carlo_used_in_calibration": "false",
                    "utility_source_type": (
                        "exact_finite_output_law_plus_independent_complete_release_monte_carlo"
                        if exact_family_utility
                        else UTILITY_SOURCE
                    ),
                    "comparison_basis": (
                        "same_declared_pairwise_exact_central_profile"
                        if exact_family_utility
                        else "transferred_reference_sensitivity_not_shared_eps0_or_central_epsilon"
                    ),
                    "shared_local_epsilon0_across_methods": "false",
                    "mc_replicates": MC_REPLICATES,
                    "dataset_sampling_model": workload.sampling_model,
                    "dataset_sampling_seed": (
                        DATASET_SEED + int(workload.dataset == "Covertype")
                        if not exact_family_utility
                        else ""
                    ),
                    "source_dataset_rows": workload.source_rows,
                    "numerical_status": "exact_finite_privacy_law_float64_with_conservative_profile_slack_not_interval_certified; utility_is_monte_carlo",
                    "provenance": (
                        "privacy=deterministic exact finite law for one declared homogeneous-background pair; "
                        "utility=independent complete-release Monte Carlo within that family"
                        if exact_family_utility
                        else
                        "reference_privacy=deterministic pairwise exact finite law for one homogeneous background; "
                        "real_data_utility=transferred-reference complete-release Monte Carlo; "
                        "methods do not share local epsilon and no real-data central-epsilon claim"
                    ),
                }
            )
    return summary_rows, calibration_rows, replicate_rows


def generate_all() -> dict[str, int]:
    summary_rows, calibration_rows, replicate_rows = generate_rows()
    exact_summary = [
        row for row in summary_rows
        if row["evidence_scope"] == "declared_exact_single_pair"
    ]
    real_summary = [
        row for row in summary_rows
        if row["evidence_scope"] == "heterogeneous_real_data_transferred_reference_sensitivity"
    ]
    exact_replicates = [
        row for row in replicate_rows
        if row["evidence_scope"] == "declared_exact_single_pair"
    ]
    real_replicates = [
        row for row in replicate_rows
        if row["evidence_scope"] == "heterogeneous_real_data_transferred_reference_sensitivity"
    ]
    filenames = {
        "fig_large_exact_family_utility.csv": exact_summary,
        "fig_large_exact_family_calibration.csv": calibration_rows,
        "fig_large_exact_family_utility_replicates.csv": exact_replicates,
        "fig_large_real_transferred_reference_sensitivity.csv": real_summary,
        "fig_large_real_transferred_reference_sensitivity_replicates.csv": real_replicates,
    }
    for filename, rows in filenames.items():
        write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows) for filename, rows in filenames.items()}


if __name__ == "__main__":
    for filename, count in generate_all().items():
        print(f"{filename}: {count} rows")
