"""Exact finite homogeneous-pair accounting for shuffled k-ary GRR.

The reference dataset contains ``m`` copies of category zero and the neighbor
replaces one record by category one.  Padding reports follow the GRR law of
category zero.  After the output order is hidden, padded Ideal has one
uniformly located changed report; Drop has the same changed report with its
exact singleton-survival probability.  The likelihood ratio is therefore an
affine function of two multinomial counts.  Conditioning on their sum reduces
the two-sided hockey-stick profile to one-dimensional binomial tails.

This is an exact numerical evaluation of the declared finite pair, not a
quadratic proxy and not a mechanism-wide maximization over heterogeneous
datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


PROFILE_RELATIVE_SLACK = 1e-8
CALIBRATION_TOLERANCE = 1e-10


@dataclass(frozen=True)
class PairCalibration:
    method: str
    domain_size: int
    epoch_size: int
    slot_count: int | None
    release_length: int
    survival: float
    target_epsilon: float
    delta: float
    local_epsilon0: float
    achieved_epsilon: float
    profile_at_target: float
    feasible_eps0: float
    infeasible_eps0: float
    privacy_family: str = "homogeneous_category_0_to_1_replacement_pair"
    privacy_scope: str = "two_sided_pair_profile_not_heterogeneous_worst_case"
    privacy_source: str = "exact_finite_multinomial_tail_account"


def grr_probabilities(eps0: float, domain_size: int) -> tuple[float, float]:
    """Return truthful and per-alternative probabilities for k-ary GRR."""
    if eps0 < 0.0 or not math.isfinite(eps0):
        raise ValueError("eps0 must be finite and nonnegative")
    if domain_size < 2:
        raise ValueError("GRR requires a domain of size at least two")
    exponential = math.exp(eps0)
    denominator = exponential + float(domain_size - 1)
    return exponential / denominator, 1.0 / denominator


def singleton_survival(epoch_size: int, slot_count: int) -> float:
    """Exact probability that a tagged report occupies a singleton slot."""
    if epoch_size <= 0 or slot_count <= 0:
        raise ValueError("epoch_size and slot_count must be positive")
    return (1.0 - 1.0 / float(slot_count)) ** (epoch_size - 1)


def _conditional_binomial_terms(
    sizes: np.ndarray,
    probability: float,
    thresholds: np.ndarray,
    *,
    upper: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional tail probabilities and truncated first moments."""
    sizes = np.asarray(sizes, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.int64)
    probabilities = np.empty(len(sizes), dtype=float)
    first_moments = np.empty(len(sizes), dtype=float)

    if upper:
        all_mask = thresholds <= 0
        none_mask = thresholds > sizes
        interior = ~(all_mask | none_mask)
        probabilities[all_mask] = 1.0
        first_moments[all_mask] = sizes[all_mask] * probability
        probabilities[none_mask] = 0.0
        first_moments[none_mask] = 0.0
        if bool(interior.any()):
            n = sizes[interior]
            t = thresholds[interior]
            probabilities[interior] = binom.sf(t - 1, n, probability)
            first_moments[interior] = (
                n
                * probability
                * binom.sf(t - 2, n - 1, probability)
            )
    else:
        none_mask = thresholds < 0
        all_mask = thresholds >= sizes
        interior = ~(none_mask | all_mask)
        probabilities[none_mask] = 0.0
        first_moments[none_mask] = 0.0
        probabilities[all_mask] = 1.0
        first_moments[all_mask] = sizes[all_mask] * probability
        if bool(interior.any()):
            n = sizes[interior]
            t = thresholds[interior]
            probabilities[interior] = binom.cdf(t, n, probability)
            first_moments[interior] = (
                n
                * probability
                * binom.cdf(t - 1, n - 1, probability)
            )
    return probabilities, first_moments


def exact_pair_profile(
    *,
    release_length: int,
    domain_size: int,
    eps0: float,
    survival: float,
    central_epsilon: float,
) -> float:
    """Return the exact two-sided hockey-stick profile of the declared pair.

    Under the reference law, let ``A`` and ``B`` count outputs zero and one.
    The release likelihood ratio is

        L = 1 + survival / M * (A(e^-eps0 - 1) + B(e^eps0 - 1)).

    Given ``S=A+B``, ``B`` is binomial, so both positive parts are evaluated
    from binomial tail probabilities and truncated first moments.
    """
    if release_length <= 0:
        raise ValueError("release_length must be positive")
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival must lie in [0,1]")
    if central_epsilon < 0.0 or not math.isfinite(central_epsilon):
        raise ValueError("central_epsilon must be finite and nonnegative")
    p, q = grr_probabilities(eps0, domain_size)
    if eps0 == 0.0 or survival == 0.0:
        return 0.0

    alpha = math.exp(central_epsilon)
    r_zero = math.exp(-eps0)
    r_one = math.exp(eps0)
    pair_mass = p + q
    conditional_one = q / pair_mass
    sizes = np.arange(release_length + 1, dtype=np.int64)

    # For fixed S=A+B and B=b, L=c_S+d*b.
    c_s = 1.0 + survival * sizes * (r_zero - 1.0) / float(release_length)
    d = survival * (r_one - r_zero) / float(release_length)
    if d <= 0.0:
        raise AssertionError("positive eps0 must produce an increasing likelihood ratio")

    forward_threshold = np.floor((alpha - c_s) / d).astype(np.int64) + 1
    upper_probability, upper_first_moment = _conditional_binomial_terms(
        sizes, conditional_one, forward_threshold, upper=True
    )
    forward_given_s = (
        (c_s - alpha) * upper_probability + d * upper_first_moment
    )

    reverse_threshold = np.ceil((1.0 / alpha - c_s) / d).astype(np.int64) - 1
    lower_probability, lower_first_moment = _conditional_binomial_terms(
        sizes, conditional_one, reverse_threshold, upper=False
    )
    reverse_given_s = (
        (1.0 - alpha * c_s) * lower_probability
        - alpha * d * lower_first_moment
    )

    outer = binom.pmf(sizes, release_length, pair_mass)
    mass = float(outer.sum())
    if abs(mass - 1.0) > 5e-11:
        raise ArithmeticError(f"outer binomial mass drifted by {mass - 1.0}")
    forward = float(np.dot(outer, np.maximum(forward_given_s, 0.0)))
    reverse = float(np.dot(outer, np.maximum(reverse_given_s, 0.0)))
    return max(0.0, forward, reverse)


def exact_central_epsilon(
    *,
    release_length: int,
    domain_size: int,
    eps0: float,
    survival: float,
    delta: float,
    tolerance: float = 1e-11,
) -> float:
    """Invert the exact two-sided pair profile in central epsilon."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if exact_pair_profile(
        release_length=release_length,
        domain_size=domain_size,
        eps0=eps0,
        survival=survival,
        central_epsilon=0.0,
    ) <= delta:
        return 0.0
    low, high = 0.0, max(1.0, eps0)
    while exact_pair_profile(
        release_length=release_length,
        domain_size=domain_size,
        eps0=eps0,
        survival=survival,
        central_epsilon=high,
    ) > delta:
        high *= 2.0
        if high > 100.0:
            raise RuntimeError("failed to bracket exact central epsilon")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if exact_pair_profile(
            release_length=release_length,
            domain_size=domain_size,
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
    domain_size: int,
    epoch_size: int,
    release_length: int,
    target_epsilon: float,
    delta: float,
    slot_count: int | None = None,
    eps0_max: float = 30.0,
) -> PairCalibration:
    """Find the largest local epsilon satisfying one exact pair budget."""
    if method == "Padded Ideal":
        if slot_count is not None:
            raise ValueError("Padded Ideal does not use slot_count")
        survival = 1.0
    elif method == "Drop":
        if slot_count is None:
            raise ValueError("Drop requires slot_count")
        survival = singleton_survival(epoch_size, slot_count)
    else:
        raise ValueError("method must be Padded Ideal or Drop")
    if release_length < epoch_size:
        raise ValueError("release_length must be at least epoch_size")

    profile_limit = delta * (1.0 - PROFILE_RELATIVE_SLACK)

    def profile(eps0: float) -> float:
        return exact_pair_profile(
            release_length=release_length,
            domain_size=domain_size,
            eps0=eps0,
            survival=survival,
            central_epsilon=target_epsilon,
        )

    low, high = 0.0, 1.0
    while high < eps0_max and profile(high) <= profile_limit:
        low, high = high, min(2.0 * high, eps0_max)
    if profile(high) <= profile_limit:
        raise RuntimeError("eps0_max does not bracket the privacy target")
    while high - low > CALIBRATION_TOLERANCE:
        middle = 0.5 * (low + high)
        if profile(middle) <= profile_limit:
            low = middle
        else:
            high = middle

    achieved = exact_central_epsilon(
        release_length=release_length,
        domain_size=domain_size,
        eps0=low,
        survival=survival,
        delta=delta,
    )
    return PairCalibration(
        method=method,
        domain_size=domain_size,
        epoch_size=epoch_size,
        slot_count=slot_count,
        release_length=release_length,
        survival=survival,
        target_epsilon=target_epsilon,
        delta=delta,
        local_epsilon0=low,
        achieved_epsilon=achieved,
        profile_at_target=profile(low),
        feasible_eps0=low,
        infeasible_eps0=high,
    )
