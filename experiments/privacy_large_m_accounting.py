"""Large-epoch privacy accounting for the four Chafe analysis cases.

The paper's main figure uses three evidence layers from this experiment:

1. ``epsilon_poisson_mechanism_wide_upper`` is a decoder-wide certificate for
   the binary Poisson proxy.  It first maximizes the completed-slot family to
   obtain a pure-LDP slot diameter, then applies the FMT clone accountant.
2. ``epsilon_poisson_exact_two_sided`` exactly inverts the tighter declared
   matched-pair Poisson profile using binomial tails.  The rho and tau sweeps
   use it to expose collision and padding trends, not as a heterogeneous
   fixed-size guarantee.
3. ``finite_drop_exact`` replaces exp(-lambda) by the exact fixed-size
   singleton probability in the one scalable case where that specialization
   is exact.

The script retains Gaussian and Chernoff diagnostics for artifact regression,
but they are no longer separate claims or curves in the paper's main analysis.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr
from scipy.stats import binom

from chafe_eval_lib import DATA_DIR, brr_probs, write_csv


EPS0 = 1.0
DELTA = 1e-6
EPOCH_SIZES = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
RHO_VALUES = (0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0)
MAIN_TAU = 2.0
TAU_VALUES = (1.0, 2.0, 4.0, 8.0)
TAU_SWEEP_M = 100_000
TAU_SWEEP_RHO = 1.0
GENERAL_K_VALUES = (2, 4, 8, 16)
GENERAL_M_VALUES = (1_000, 10_000, 100_000, 1_000_000)


Law = tuple[float, float]
Profile = Callable[[float], float]
CategoricalLaw = tuple[float, ...]


@dataclass(frozen=True)
class CaseLaws:
    """One-position laws used by the hidden-position release identity.

    ``reference`` is the matched dummy law nu and therefore the iid base law
    of every released position before one hidden position is changed.
    ``left`` and ``right`` are the completed laws of that hidden position in
    the two neighboring executions.
    """

    reference: Law
    left: Law
    right: Law
    forward_label: str
    reverse_label: str


@dataclass(frozen=True)
class BinarySlotEnvelope:
    """Exact Bernoulli parameter envelope for all completed Poisson slots."""

    p_min: float
    p_max: float
    epsilon_slot: float


def normalize(law: Sequence[float]) -> Law:
    total = math.fsum(law)
    if total <= 0.0:
        raise ValueError("cannot normalize a zero-mass law")
    result = (float(law[0]) / total, float(law[1]) / total)
    if min(result) <= 0.0:
        raise ValueError("the large-m accountant expects full binary support")
    return result


def z2_convolution(left: Sequence[float], right: Sequence[float]) -> Law:
    """Distribution of X+Y mod 2 for independent binary laws."""

    return (
        left[0] * right[0] + left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def compound_poisson_z2(background: Law, load: float) -> Law:
    """Law of the Z_2 sum of Pois(load) iid background payloads."""

    parity_bias = math.exp(-2.0 * load * background[1])
    return ((1.0 + parity_bias) / 2.0, (1.0 - parity_bias) / 2.0)


def completed_target_law_drop(
    target: Law, reference: Law, singleton_survival: float
) -> Law:
    return (
        singleton_survival * target[0]
        + (1.0 - singleton_survival) * reference[0],
        singleton_survival * target[1]
        + (1.0 - singleton_survival) * reference[1],
    )


def poisson_case_laws(
    decoder: str,
    adjacency: str,
    common_load: float,
    f0: Law,
    f1: Law,
) -> CaseLaws:
    """Construct matched-padding one-position laws for one analysis case.

    Replacement compares a type-0 target with a type-1 target among the same
    homogeneous type-0 background.  Add/remove compares absence with a type-1
    target.  Both use m-1 common reports and the same public N and M.
    """

    if decoder == "drop":
        reference = f0
        survival = math.exp(-common_load)
        # A type-0 target has exactly the matched reference law, independent
        # of whether it survives; assign it directly to avoid roundoff drift.
        target0 = reference
        target1 = completed_target_law_drop(f1, reference, survival)
    elif decoder == "pool":
        compound = compound_poisson_z2(f0, common_load)
        empty = math.exp(-common_load)
        # An ordinary Pool slot emits iff it is nonempty.  Matched padding is
        # its conditional emitted-value law.
        reference = normalize((compound[0] - empty, compound[1]))
        target0 = normalize(z2_convolution(f0, compound))
        target1 = normalize(z2_convolution(f1, compound))
    else:
        raise ValueError("decoder must be 'drop' or 'pool'")

    if adjacency == "replacement":
        return CaseLaws(
            reference=reference,
            left=target0,
            right=target1,
            forward_label="target_0_to_target_1",
            reverse_label="target_1_to_target_0",
        )
    if adjacency == "add_remove":
        return CaseLaws(
            reference=reference,
            left=reference,
            right=target1,
            forward_label="absent_to_present",
            reverse_label="present_to_absent",
        )
    raise ValueError("adjacency must be 'replacement' or 'add_remove'")


def likelihood(law: Law, reference: Law) -> Law:
    return (law[0] / reference[0], law[1] / reference[1])


def _positive_region(M: int, intercept: float, slope: float) -> tuple[str, int]:
    """Return the integer tail on which intercept+slope*k is strictly positive."""

    if slope > 0.0:
        boundary = -intercept / slope
        first = math.floor(boundary) + 1
        first = max(0, min(M + 1, first))
        while first > 0 and intercept + slope * (first - 1) > 0.0:
            first -= 1
        while first <= M and intercept + slope * first <= 0.0:
            first += 1
        return "upper", first
    if slope < 0.0:
        boundary = -intercept / slope
        last = math.ceil(boundary) - 1
        last = max(-1, min(M, last))
        while last < M and intercept + slope * (last + 1) > 0.0:
            last += 1
        while last >= 0 and intercept + slope * last <= 0.0:
            last -= 1
        return "lower", last
    return ("all", M) if intercept > 0.0 else ("none", -1)


def binomial_positive_part(M: int, p: float, value0: float, value1: float) -> float:
    """Evaluate E[(M*value0 + K*(value1-value0))_+]/M exactly.

    Here K~Bin(M,p).  The expression uses a binomial tail probability and the
    identity E[K 1{K>=k}]=M p Pr[Bin(M-1,p)>=k-1] (and its lower-tail
    counterpart), so its running time does not grow with M.
    """

    intercept = M * value0
    slope = value1 - value0
    region, cutoff = _positive_region(M, intercept, slope)
    if region == "none":
        return 0.0
    if region == "all":
        return max(0.0, (intercept + slope * M * p) / M)
    if region == "upper":
        if cutoff > M:
            return 0.0
        probability = float(binom.sf(cutoff - 1, M, p))
        first_moment = (
            M * p
            if cutoff <= 0
            else M * p * float(binom.sf(cutoff - 2, M - 1, p))
        )
    else:
        if cutoff < 0:
            return 0.0
        probability = float(binom.cdf(cutoff, M, p))
        first_moment = (
            0.0
            if cutoff <= 0
            else M * p * float(binom.cdf(cutoff - 1, M - 1, p))
        )
    # The two terms have opposite signs near the threshold.  fsum and the
    # final numerical guard avoid reporting a negative roundoff artifact.
    value = math.fsum((intercept * probability, slope * first_moment)) / M
    return max(0.0, value)


def exact_directed_profile(
    reference: Law, numerator: Law, denominator: Law, M: int
) -> Profile:
    """Exact Poisson-surrogate H_alpha(numerator || denominator)."""

    h_num = likelihood(numerator, reference)
    h_den = likelihood(denominator, reference)
    p = reference[1]

    def profile(alpha: float) -> float:
        return binomial_positive_part(
            M,
            p,
            h_num[0] - alpha * h_den[0],
            h_num[1] - alpha * h_den[1],
        )

    return profile


def gaussian_directed_profile(
    reference: Law, numerator: Law, denominator: Law, M: int
) -> Profile:
    """Alpha-dependent normal approximation to the directed profile."""

    h_num = likelihood(numerator, reference)
    h_den = likelihood(denominator, reference)

    def profile(alpha: float) -> float:
        w0 = h_num[0] - alpha * h_den[0]
        w1 = h_num[1] - alpha * h_den[1]
        mean = reference[0] * w0 + reference[1] * w1
        variance = (
            reference[0] * (w0 - mean) ** 2
            + reference[1] * (w1 - mean) ** 2
        )
        return gaussian_positive_part(mean, variance, M)

    return profile


def gaussian_directed_profile_categorical(
    reference: CategoricalLaw,
    numerator: CategoricalLaw,
    denominator: CategoricalLaw,
    M: int,
) -> Profile:
    """Alpha-dependent Gaussian profile for an arbitrary finite output law."""

    if not (len(reference) == len(numerator) == len(denominator)):
        raise ValueError("categorical laws must share one output domain")
    if min(reference) <= 0.0:
        raise ValueError("the reference law must have full support")
    h_num = tuple(p / q for p, q in zip(numerator, reference))
    h_den = tuple(p / q for p, q in zip(denominator, reference))

    def profile(alpha: float) -> float:
        evidence = tuple(a - alpha * b for a, b in zip(h_num, h_den))
        mean = math.fsum(q * w for q, w in zip(reference, evidence))
        variance = math.fsum(
            q * (w - mean) ** 2 for q, w in zip(reference, evidence)
        )
        return gaussian_positive_part(mean, variance, M)

    return profile


def gaussian_positive_part(mean: float, variance: float, M: int) -> float:
    """Positive-part expectation of N(mean, variance/M)."""

    if variance <= 1e-30:
        return max(mean, 0.0)
    scale = math.sqrt(variance / M)
    z = mean / scale
    value = math.fsum(
        (
            scale * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi),
            mean * float(ndtr(z)),
        )
    )
    return max(0.0, value)


def chernoff_directed_profile(
    reference: Sequence[float],
    numerator: Sequence[float],
    denominator: Sequence[float],
    M: int,
) -> Profile:
    r"""Rigorous Chernoff upper bound on one directed hockey-stick profile.

    For ``W_alpha=h_num(Z)-alpha*h_den(Z)``, ``Z~reference``, the hidden-slot
    identity and ``x_+ <= exp(s*x-1)/s`` give

      H_alpha <= min{1, inf_{s>0}
          exp(M log E exp(s W_alpha) - 1 - log M - log s)}.

    We optimize in ``t=log(s)``.  Every log-MGF is evaluated with log-sum-exp,
    and the stationary equation is solved in t, avoiding probability-domain
    underflow even when the requested profile is 1e-6.
    """

    if not (len(reference) == len(numerator) == len(denominator)):
        raise ValueError("laws must share one output domain")
    if min(reference) <= 0.0:
        raise ValueError("the Chernoff reference law must have full support")
    log_probabilities = tuple(math.log(q) for q in reference)
    h_num = tuple(p / q for p, q in zip(numerator, reference))
    h_den = tuple(p / q for p, q in zip(denominator, reference))
    log_M = math.log(M)

    def profile(alpha: float) -> float:
        evidence = tuple(a - alpha * b for a, b in zip(h_num, h_den))
        if max(evidence) <= 0.0:
            return 0.0

        def log_mgf_and_tilted_mean(log_s: float) -> tuple[float, float]:
            s = math.exp(log_s)
            terms = tuple(
                log_q + s * w for log_q, w in zip(log_probabilities, evidence)
            )
            log_mgf = float(logsumexp(terms))
            tilted_mean = math.fsum(
                math.exp(term - log_mgf) * w for term, w in zip(terms, evidence)
            )
            return log_mgf, tilted_mean

        def stationary(log_s: float) -> float:
            s = math.exp(log_s)
            _, tilted_mean = log_mgf_and_tilted_mean(log_s)
            return M * s * tilted_mean - 1.0

        lo = -40.0
        while stationary(lo) >= 0.0:
            lo -= 10.0
            if lo < -700.0:
                raise RuntimeError("failed to bracket the Chernoff optimum below")
        hi = 0.0
        while stationary(hi) <= 0.0:
            hi += math.log(2.0)
            if hi > 700.0:
                raise RuntimeError("failed to bracket the Chernoff optimum above")
        optimum = brentq(stationary, lo, hi, xtol=1e-12, rtol=1e-12)
        log_mgf, _ = log_mgf_and_tilted_mean(optimum)
        log_bound = M * log_mgf - 1.0 - log_M - optimum
        if log_bound >= 0.0:
            return 1.0
        if log_bound <= -745.0:
            return 0.0
        return math.exp(log_bound)

    return profile


def account_chernoff_pair(
    reference: Sequence[float],
    left: Sequence[float],
    right: Sequence[float],
    M: int,
    delta: float,
) -> dict[str, float | bool]:
    """Invert the rigorous bound in both neighboring directions."""

    forward_profile = chernoff_directed_profile(reference, left, right, M)
    reverse_profile = chernoff_directed_profile(reference, right, left, M)
    epsilon_forward = invert_directed_profile(forward_profile, delta)
    epsilon_reverse = invert_directed_profile(reverse_profile, delta)
    epsilon_two_sided = max(epsilon_forward, epsilon_reverse)
    alpha = math.exp(epsilon_two_sided)
    forward_bound = forward_profile(alpha)
    reverse_bound = reverse_profile(alpha)
    profile_bound = max(forward_bound, reverse_bound)
    return {
        "forward": epsilon_forward,
        "reverse": epsilon_reverse,
        "two_sided": epsilon_two_sided,
        "forward_bound": forward_bound,
        "reverse_bound": reverse_bound,
        "profile_bound": profile_bound,
        "feasible": profile_bound <= delta * (1.0 + 1e-9),
    }


def invert_directed_profile(profile: Profile, delta: float) -> float:
    """Smallest epsilon whose directed profile at alpha=exp(epsilon) is <= delta."""

    if profile(1.0) <= delta:
        return 0.0
    lo = 0.0
    hi = 1.0
    while profile(math.exp(hi)) > delta:
        hi *= 2.0
        if hi > 64.0:
            raise RuntimeError("failed to bracket a directed privacy budget")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if profile(math.exp(mid)) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def account_pair(laws: CaseLaws, M: int, delta: float, method: str) -> dict[str, float]:
    if method == "exact":
        constructor = exact_directed_profile
    elif method == "gaussian":
        constructor = gaussian_directed_profile
    else:
        raise ValueError("method must be exact or gaussian")
    forward_profile = constructor(laws.reference, laws.left, laws.right, M)
    reverse_profile = constructor(laws.reference, laws.right, laws.left, M)
    epsilon_forward = invert_directed_profile(forward_profile, delta)
    epsilon_reverse = invert_directed_profile(reverse_profile, delta)
    epsilon_two_sided = max(epsilon_forward, epsilon_reverse)
    return {
        "forward": epsilon_forward,
        "reverse": epsilon_reverse,
        "two_sided": epsilon_two_sided,
        "profile_at_two_sided": max(
            forward_profile(math.exp(epsilon_two_sided)),
            reverse_profile(math.exp(epsilon_two_sided)),
        ),
    }


def _clone_conditional_divergence(
    clone_count: int,
    central_epsilon: float,
    local_epsilon: float,
    forward: bool,
) -> float:
    """Conditional hockey-stick divergence in the FMT clone reduction.

    This is the two-binomial reduction in Theorem 3.2 of Feldman, McMillan,
    and Talwar (FOCS 2021).  It is used only as a mechanism-wide upper bound,
    not as the matched-pair accountant.
    """

    if central_epsilon >= local_epsilon:
        return 0.0
    local_alpha = math.exp(local_epsilon) / (math.exp(local_epsilon) + 1.0)
    central_alpha = math.exp(central_epsilon)
    numerator = (central_alpha + 1.0) * local_alpha - 1.0
    denominator = (central_alpha + 1.0) * local_alpha - central_alpha
    if numerator <= 0.0 or denominator <= 0.0:
        return 0.0
    effective = math.log(numerator / denominator)
    beta = 1.0 / (
        math.exp(effective if forward else -effective) + 1.0
    )
    cutoff = beta * (clone_count + 1)
    p_cdf = (
        local_alpha * float(binom.cdf(cutoff, clone_count, 0.5))
        + (1.0 - local_alpha)
        * float(binom.cdf(cutoff - 1.0, clone_count, 0.5))
    )
    q_cdf = (
        (1.0 - local_alpha) * float(binom.cdf(cutoff, clone_count, 0.5))
        + local_alpha
        * float(binom.cdf(cutoff - 1.0, clone_count, 0.5))
    )
    if forward:
        value = p_cdf - central_alpha * q_cdf
    else:
        value = (1.0 - q_cdf) - central_alpha * (1.0 - p_cdf)
    return max(0.0, value)


def _clone_profile_upper(
    report_count: int,
    local_epsilon: float,
    central_epsilon: float,
    delta_cap: float,
    step: int,
) -> float:
    """Rigorous segmented upper bound for the FMT clone profile."""

    clone_probability = math.exp(-local_epsilon)
    mean = (report_count - 1) * clone_probability
    forward = 0.0
    reverse = 0.0
    probability_used = 0.0

    for block in range(1, int(math.ceil(report_count / step)) + 2):
        any_interval = False
        for upper_side in (True, False):
            if upper_side:
                upper = int(math.ceil(mean + block * step))
                lower = upper - step if block == 1 else upper - step + 1
                if lower > report_count - 1:
                    continue
                upper = min(upper, report_count - 1)
            else:
                lower = int(math.ceil(mean - block * step))
                upper = lower + step - 1
                if upper < 0:
                    continue
                lower = max(0, lower)
            any_interval = True
            mass = float(
                binom.cdf(upper, report_count - 1, clone_probability)
                - binom.cdf(lower - 1, report_count - 1, clone_probability)
            )
            if mass <= 0.0:
                continue
            forward += mass * max(
                _clone_conditional_divergence(
                    upper, central_epsilon, local_epsilon, True
                ),
                _clone_conditional_divergence(
                    lower, central_epsilon, local_epsilon, True
                ),
            )
            reverse += mass * max(
                _clone_conditional_divergence(
                    upper, central_epsilon, local_epsilon, False
                ),
                _clone_conditional_divergence(
                    lower, central_epsilon, local_epsilon, False
                ),
            )
            probability_used += mass
            if max(forward, reverse) > delta_cap:
                return delta_cap

        tail = max(0.0, 1.0 - probability_used)
        # An unprocessed conditional divergence is at most one.  Adding the
        # full residual mass therefore preserves an upper bound.
        if tail < min(forward, reverse) or not any_interval:
            return min(1.0, max(forward, reverse) + tail)

    return min(1.0, max(forward, reverse) + max(0.0, 1.0 - probability_used))


@lru_cache(maxsize=None)
def clone_shuffle_numerical_upper(
    report_count: int,
    local_epsilon_rounded: float,
    delta: float,
) -> float:
    """Mechanism-wide numerical shuffle upper bound from the clone reduction."""

    local_epsilon = float(local_epsilon_rounded)
    if local_epsilon <= 0.0:
        return 0.0
    amplification_term = (
        8.0
        * math.sqrt(
            math.exp(local_epsilon)
            * math.log(4.0 / delta)
            / report_count
        )
        + 8.0 * math.exp(local_epsilon) / report_count
    )
    preliminary = math.log1p(amplification_term)
    coefficient = (1.0 - math.exp(-local_epsilon)) / (
        1.0 + math.exp(-local_epsilon - preliminary)
    )
    closed_form = math.log1p(coefficient * amplification_term)
    validity_threshold = math.log(
        report_count / (16.0 * math.log(4.0 / delta))
    )
    hi = (
        min(local_epsilon, closed_form)
        if local_epsilon < validity_threshold
        else local_epsilon
    )
    lo = 0.0
    step = max(1, min(100, int(math.sqrt(report_count) / 4.0)))
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        profile = _clone_profile_upper(
            report_count, local_epsilon, mid, delta, step
        )
        # ``_clone_profile_upper`` deliberately returns ``delta`` as soon as
        # its partial sum proves the candidate infeasible.  Equality must
        # therefore remain on the infeasible side of the binary search, as
        # in the reference implementation's strict comparison.
        if profile < delta:
            hi = mid
        else:
            lo = mid
    return hi


def binary_slot_envelope(
    decoder: str,
    common_load: float,
    f0: Law,
    f1: Law,
) -> BinarySlotEnvelope:
    """Completed-slot family over every binary background mixture.

    The public dummy remains fixed to the reference chosen by the evaluated
    mechanism.  For Z_2 BRR, all candidate probabilities are monotone in the
    background mixture, so the two mixture endpoints exhaust the range.
    """

    empty = math.exp(-common_load)
    probabilities: list[float] = []
    if decoder == "drop":
        reference = f0
        singleton = common_load * empty
        probabilities.append(reference[1])
        for background in (f0, f1):
            probabilities.append(
                (1.0 - singleton) * reference[1]
                + singleton * background[1]
            )
        for target in (f0, f1):
            probabilities.append(
                (1.0 - empty) * reference[1] + empty * target[1]
            )
    elif decoder == "pool":
        reference_compound = compound_poisson_z2(f0, common_load)
        reference = normalize(
            (reference_compound[0] - empty, reference_compound[1])
        )
        probabilities.append(reference[1])
        for background in (f0, f1):
            compound = compound_poisson_z2(background, common_load)
            probabilities.append(empty * reference[1] + compound[1])
            for target in (f0, f1):
                probabilities.append(
                    z2_convolution(target, compound)[1]
                )
    else:
        raise ValueError("decoder must be 'drop' or 'pool'")

    p_min = min(probabilities)
    p_max = max(probabilities)
    if p_min <= 0.0 or p_max >= 1.0:
        raise ValueError("the binary slot envelope requires full support")
    epsilon_slot = math.log(
        max(
            p_max / p_min,
            (1.0 - p_min) / (1.0 - p_max),
        )
    )
    return BinarySlotEnvelope(p_min, p_max, epsilon_slot)


@lru_cache(maxsize=None)
def ideal_hidden_position_account(release_length: int) -> float:
    """Exact BRR profile for one changed report hidden among ``release_length`` positions.

    With matched BRR padding this is the comparable ideal-shuffle reference.
    Calling it with ``m`` gives the usual length-m shuffle; calling it with
    ``M`` isolates the additional collision effect at Chafe's release length.
    """

    f0 = brr_probs(EPS0, 0)
    f1 = brr_probs(EPS0, 1)
    laws = CaseLaws(
        reference=f0,
        left=f0,
        right=f1,
        forward_label="target_0_to_target_1",
        reverse_label="target_1_to_target_0",
    )
    return account_pair(laws, release_length, DELTA, "exact")["two_sided"]


def finite_drop_case_laws(
    adjacency: str, f0: Law, f1: Law, survival: float
) -> CaseLaws:
    target1 = completed_target_law_drop(f1, f0, survival)
    if adjacency == "replacement":
        labels = ("target_0_to_target_1", "target_1_to_target_0")
    elif adjacency == "add_remove":
        labels = ("absent_to_present", "present_to_absent")
    else:
        raise ValueError("unknown adjacency")
    return CaseLaws(f0, f0, target1, labels[0], labels[1])


def evaluate_case(
    sweep: str,
    decoder: str,
    adjacency: str,
    m: int,
    rho: float,
    tau: float,
) -> dict[str, object]:
    if m < 2:
        raise ValueError("m must leave at least one common report")
    N = int(round(m / rho))
    M = int(round(tau * m))
    if N <= 0 or M < N:
        raise ValueError("this matched-padding construction requires M >= N")
    common_reports = m - 1
    common_load = common_reports / N
    f0 = brr_probs(EPS0, 0)
    f1 = brr_probs(EPS0, 1)
    laws = poisson_case_laws(decoder, adjacency, common_load, f0, f1)
    slot_envelope = binary_slot_envelope(decoder, common_load, f0, f1)
    mechanism_wide_upper: float | str = ""
    if (
        sweep == "m_scaling"
        and abs(rho - 1.0) < 1e-12
        and abs(tau - MAIN_TAU) < 1e-12
    ):
        mechanism_wide_upper = clone_shuffle_numerical_upper(
            M,
            math.ceil(slot_envelope.epsilon_slot * 1e12) / 1e12,
            DELTA,
        )
    exact = account_pair(laws, M, DELTA, "exact")
    gaussian = account_pair(laws, M, DELTA, "gaussian")

    h_left = likelihood(laws.left, laws.reference)
    h_right = likelihood(laws.right, laws.reference)
    one_position_evidence_variance = math.fsum(
        q * (left - right) ** 2
        for q, left, right in zip(laws.reference, h_left, h_right)
    )
    quadratic_mu_squared = one_position_evidence_variance / M
    quadratic_epsilon_scale = math.sqrt(
        2.0 * quadratic_mu_squared * math.log(1.0 / DELTA)
    )

    epsilon_ideal_release_M = ideal_hidden_position_account(M)
    epsilon_ideal_release_m = ideal_hidden_position_account(m)
    load_contraction = (
        (1.0 - math.exp(-common_load)) / common_load
        if common_load > 0.0
        else 1.0
    )
    kappa = math.exp(EPS0) + math.exp(-EPS0) - 2.0
    padding_factor = math.sqrt(m / M)
    drop_collision_coefficient = math.exp(-common_load)
    pool_replacement_collision_coefficient = math.sqrt(load_contraction)
    pool_add_remove_collision_coefficient = math.sqrt(
        ((load_contraction - math.exp(-common_load)) + load_contraction * kappa)
        / kappa
    )
    drop_total_coefficient = drop_collision_coefficient * padding_factor
    pool_replacement_total_coefficient = (
        pool_replacement_collision_coefficient * padding_factor
    )
    pool_add_remove_total_coefficient = (
        pool_add_remove_collision_coefficient * padding_factor
    )
    chernoff_start = time.perf_counter()
    chernoff = account_chernoff_pair(
        laws.reference, laws.left, laws.right, M, DELTA
    )
    chernoff_runtime_ms = 1_000.0 * (time.perf_counter() - chernoff_start)
    if not chernoff["feasible"]:
        raise AssertionError("Chernoff inversion did not reach the target delta")
    if chernoff["two_sided"] + 1e-11 < exact["two_sided"]:
        raise AssertionError("Chernoff certificate improved on the exact account")

    exact_forward = exact_directed_profile(
        laws.reference, laws.left, laws.right, M
    )
    exact_reverse = exact_directed_profile(
        laws.reference, laws.right, laws.left, M
    )
    gaussian_forward = gaussian_directed_profile(
        laws.reference, laws.left, laws.right, M
    )
    gaussian_reverse = gaussian_directed_profile(
        laws.reference, laws.right, laws.left, M
    )
    chernoff_forward = chernoff_directed_profile(
        laws.reference, laws.left, laws.right, M
    )
    chernoff_reverse = chernoff_directed_profile(
        laws.reference, laws.right, laws.left, M
    )
    gaussian_alpha = math.exp(gaussian["two_sided"])
    exact_alpha = math.exp(exact["two_sided"])
    exact_profile_at_gaussian = max(
        exact_forward(gaussian_alpha), exact_reverse(gaussian_alpha)
    )
    gaussian_profile_at_exact = max(
        gaussian_forward(exact_alpha), gaussian_reverse(exact_alpha)
    )
    chernoff_profile_at_exact = max(
        chernoff_forward(exact_alpha), chernoff_reverse(exact_alpha)
    )

    finite_two_sided: float | str = ""
    finite_forward: float | str = ""
    finite_reverse: float | str = ""
    poisson_over_finite: float | str = ""
    finite_status = "not_applicable_pool"
    finite_survival: float | str = ""
    if decoder == "drop":
        finite_survival = math.exp(common_reports * math.log1p(-1.0 / N))
        finite_laws = finite_drop_case_laws(adjacency, f0, f1, finite_survival)
        finite = account_pair(finite_laws, M, DELTA, "exact")
        finite_forward = finite["forward"]
        finite_reverse = finite["reverse"]
        finite_two_sided = finite["two_sided"]
        poisson_over_finite = exact["two_sided"] / finite["two_sided"]
        finite_status = "exact_complete_release_pair_matched_homogeneous_drop"

    return {
        "sweep": sweep,
        "panel": f"{decoder}_{adjacency}",
        "decoder": decoder,
        "adjacency": adjacency,
        "forward_direction": laws.forward_label,
        "reverse_direction": laws.reverse_label,
        "randomizer": "binary_randomized_response_Z2",
        "epsilon0": EPS0,
        "delta": DELTA,
        "m_public_larger_epoch": m,
        "common_reports": common_reports,
        "N_slots": N,
        "M_release": M,
        "rho_requested_m_over_N": rho,
        "rho_common": common_load,
        "tau_M_over_m": M / m,
        "matched_dummy_nu0": laws.reference[0],
        "matched_dummy_nu1": laws.reference[1],
        "slot_envelope_p_min": slot_envelope.p_min,
        "slot_envelope_p_max": slot_envelope.p_max,
        "epsilon_slot_envelope": slot_envelope.epsilon_slot,
        "epsilon_poisson_mechanism_wide_upper": mechanism_wide_upper,
        "mechanism_wide_account_method": (
            "fmt_clone_numerical_upper_over_binary_completed_slot_family"
            if mechanism_wide_upper != ""
            else "not_evaluated_for_this_sweep"
        ),
        "epsilon_poisson_exact_forward": exact["forward"],
        "epsilon_poisson_exact_reverse": exact["reverse"],
        "epsilon_poisson_exact_two_sided": exact["two_sided"],
        "epsilon_ideal_shuffle_release_M": epsilon_ideal_release_M,
        "epsilon_ideal_shuffle_release_m": epsilon_ideal_release_m,
        "epsilon_poisson_over_ideal_release_M": (
            exact["two_sided"] / epsilon_ideal_release_M
        ),
        "epsilon_poisson_over_ideal_release_m": (
            exact["two_sided"] / epsilon_ideal_release_m
        ),
        "poisson_profile_at_exact_epsilon": exact["profile_at_two_sided"],
        "epsilon_gaussian_forward": gaussian["forward"],
        "epsilon_gaussian_reverse": gaussian["reverse"],
        "epsilon_gaussian_two_sided": gaussian["two_sided"],
        "gaussian_profile_at_gaussian_epsilon": gaussian[
            "profile_at_two_sided"
        ],
        "gaussian_relative_epsilon_error": (
            gaussian["two_sided"] / exact["two_sided"] - 1.0
        ),
        "quadratic_mu_squared": quadratic_mu_squared,
        "quadratic_epsilon_scale": quadratic_epsilon_scale,
        "quadratic_over_poisson_exact": (
            quadratic_epsilon_scale / exact["two_sided"]
        ),
        "closed_form_drop_over_ideal_m": drop_total_coefficient,
        "closed_form_drop_over_ideal_M": drop_collision_coefficient,
        "closed_form_pool_replacement_over_ideal_m": (
            pool_replacement_total_coefficient
        ),
        "closed_form_pool_replacement_over_ideal_M": (
            pool_replacement_collision_coefficient
        ),
        "closed_form_pool_add_remove_over_ideal_m": (
            pool_add_remove_total_coefficient
        ),
        "closed_form_pool_add_remove_over_ideal_M": (
            pool_add_remove_collision_coefficient
        ),
        "poisson_profile_at_gaussian_epsilon": exact_profile_at_gaussian,
        "gaussian_profile_at_poisson_epsilon": gaussian_profile_at_exact,
        "epsilon_chernoff_forward": chernoff["forward"],
        "epsilon_chernoff_reverse": chernoff["reverse"],
        "epsilon_chernoff_two_sided": chernoff["two_sided"],
        "chernoff_forward_bound_at_certified_epsilon": chernoff[
            "forward_bound"
        ],
        "chernoff_reverse_bound_at_certified_epsilon": chernoff[
            "reverse_bound"
        ],
        "chernoff_profile_bound_at_certified_epsilon": chernoff[
            "profile_bound"
        ],
        "chernoff_profile_bound_at_poisson_exact_epsilon": (
            chernoff_profile_at_exact
        ),
        "chernoff_feasible_at_delta": chernoff["feasible"],
        "epsilon_chernoff_over_poisson_exact": (
            chernoff["two_sided"] / exact["two_sided"]
        ),
        "chernoff_account_runtime_ms": chernoff_runtime_ms,
        "finite_drop_survival": finite_survival,
        "epsilon_finite_drop_exact_forward": finite_forward,
        "epsilon_finite_drop_exact_reverse": finite_reverse,
        "epsilon_finite_drop_exact_two_sided": finite_two_sided,
        "epsilon_poisson_over_finite_drop": poisson_over_finite,
        "padding_mode": "matched_to_ordinary_decoded_slot_law",
        "poisson_account_status": "exact_for_poisson_surrogate",
        "poisson_account_method": "binary_binomial_positive_part_identity",
        "gaussian_account_status": "profile_approximation_not_DP_certificate",
        "gaussian_account_method": "alpha_dependent_mean_variance_normal",
        "chernoff_account_status": "rigorous_for_poisson_surrogate",
        "chernoff_account_method": "optimized_log_mgf_positive_part_bound",
        "finite_account_status": finite_status,
        "scope": "homogeneous_binary_pair_specific_validation",
        "source_type": "formula_exact_and_profile_approximation",
    }


def _small_enumeration_check() -> None:
    """Check the binomial-tail identity against direct enumeration."""

    M = 17
    p = 0.37
    for value0, value1 in ((-0.4, 0.8), (0.7, -0.2), (-0.1, -0.3), (0.2, 0.4)):
        direct = math.fsum(
            float(binom.pmf(k, M, p))
            * max(M * value0 + k * (value1 - value0), 0.0)
            / M
            for k in range(M + 1)
        )
        closed = binomial_positive_part(M, p, value0, value1)
        if abs(direct - closed) > 1e-12:
            raise AssertionError((value0, value1, direct, closed))


def krr_probs(epsilon0: float, domain_size: int, true_value: int) -> CategoricalLaw:
    """Pure-epsilon0 k-ary randomized response on {0,...,K-1}."""

    if domain_size < 2 or not 0 <= true_value < domain_size:
        raise ValueError("invalid k-ary randomized-response parameters")
    denominator = math.exp(epsilon0) + domain_size - 1
    low = 1.0 / denominator
    high = math.exp(epsilon0) / denominator
    return tuple(high if value == true_value else low for value in range(domain_size))


def gaussian_variance_profile(
    base_variance: float, M: int, alpha_multiplier: bool
) -> Profile:
    """Drop profile when Var(W_alpha) is base_variance or alpha^2 times it."""

    def profile(alpha: float) -> float:
        variance = base_variance * (alpha * alpha if alpha_multiplier else 1.0)
        return gaussian_positive_part(1.0 - alpha, variance, M)

    return profile


def evaluate_general_randomizer_case(domain_size: int, m: int) -> dict[str, object]:
    """Gaussian Drop-replacement account for k-ary randomized response."""

    rho = 1.0
    tau = 2.0
    N = m
    M = int(tau * m)
    common_reports = m - 1
    common_load = common_reports / N
    survival = math.exp(-common_load)
    f0 = krr_probs(EPS0, domain_size, 0)
    f1 = krr_probs(EPS0, domain_size, 1)
    reference = f0
    target0 = reference
    target1 = tuple(
        survival * p1 + (1.0 - survival) * p0 for p0, p1 in zip(f0, f1)
    )

    forward_profile = gaussian_directed_profile_categorical(
        reference, target0, target1, M
    )
    reverse_profile = gaussian_directed_profile_categorical(
        reference, target1, target0, M
    )
    epsilon_forward = invert_directed_profile(forward_profile, DELTA)
    epsilon_reverse = invert_directed_profile(reverse_profile, DELTA)
    epsilon_two_sided = max(epsilon_forward, epsilon_reverse)
    chernoff_start = time.perf_counter()
    chernoff = account_chernoff_pair(reference, target0, target1, M, DELTA)
    chernoff_runtime_ms = 1_000.0 * (time.perf_counter() - chernoff_start)
    if not chernoff["feasible"]:
        raise AssertionError("Chernoff inversion did not reach the target delta")
    chernoff_forward_profile = chernoff_directed_profile(
        reference, target0, target1, M
    )
    chernoff_reverse_profile = chernoff_directed_profile(
        reference, target1, target0, M
    )

    payload_chi2 = math.fsum(
        (p1 - p0) ** 2 / p0 for p0, p1 in zip(f0, f1)
    )
    kappa_envelope = math.exp(EPS0) + math.exp(-EPS0) - 2.0
    evidence_variance = survival * survival * payload_chi2
    evidence_variance_envelope = survival * survival * kappa_envelope

    # For H_alpha(P_0 || P_1), W_alpha=1-alpha*h_1 and its
    # variance is alpha^2 times the alpha=1 evidence variance.  In the reverse
    # direction, W_alpha=h_1-alpha and the variance is unchanged.
    envelope_forward_profile = gaussian_variance_profile(
        evidence_variance_envelope, M, alpha_multiplier=True
    )
    envelope_reverse_profile = gaussian_variance_profile(
        evidence_variance_envelope, M, alpha_multiplier=False
    )
    epsilon_envelope_forward = invert_directed_profile(
        envelope_forward_profile, DELTA
    )
    epsilon_envelope_reverse = invert_directed_profile(
        envelope_reverse_profile, DELTA
    )
    epsilon_envelope_two_sided = max(
        epsilon_envelope_forward, epsilon_envelope_reverse
    )

    alpha = math.exp(epsilon_two_sided)
    profile_at_epsilon = max(forward_profile(alpha), reverse_profile(alpha))
    chernoff_bound_at_gaussian = max(
        chernoff_forward_profile(alpha), chernoff_reverse_profile(alpha)
    )
    return {
        "sweep": "general_randomizer_K_m",
        "decoder": "drop",
        "adjacency": "replacement",
        "randomizer": "k_ary_randomized_response",
        "domain_size_K": domain_size,
        "epsilon0": EPS0,
        "delta": DELTA,
        "m_public_epoch": m,
        "common_reports": common_reports,
        "N_slots": N,
        "M_release": M,
        "rho_m_over_N": rho,
        "rho_common": common_load,
        "tau_M_over_m": tau,
        "poisson_singleton_survival": survival,
        "payload_chi2_actual": payload_chi2,
        "payload_chi2_kappa_envelope": kappa_envelope,
        "payload_chi2_over_kappa": payload_chi2 / kappa_envelope,
        "one_position_evidence_variance_actual": evidence_variance,
        "one_position_evidence_variance_kappa_envelope": (
            evidence_variance_envelope
        ),
        "release_mean_variance_alpha1_actual": evidence_variance / M,
        "epsilon_gaussian_actual_forward": epsilon_forward,
        "epsilon_gaussian_actual_reverse": epsilon_reverse,
        "epsilon_gaussian_actual_two_sided": epsilon_two_sided,
        "gaussian_profile_at_actual_epsilon": profile_at_epsilon,
        "epsilon_chernoff_forward": chernoff["forward"],
        "epsilon_chernoff_reverse": chernoff["reverse"],
        "epsilon_chernoff_two_sided": chernoff["two_sided"],
        "chernoff_forward_bound_at_certified_epsilon": chernoff[
            "forward_bound"
        ],
        "chernoff_reverse_bound_at_certified_epsilon": chernoff[
            "reverse_bound"
        ],
        "chernoff_profile_bound_at_certified_epsilon": chernoff[
            "profile_bound"
        ],
        "chernoff_profile_bound_at_gaussian_epsilon": (
            chernoff_bound_at_gaussian
        ),
        "chernoff_feasible_at_delta": chernoff["feasible"],
        "epsilon_chernoff_over_gaussian_actual": (
            chernoff["two_sided"] / epsilon_two_sided
        ),
        "chernoff_account_runtime_ms": chernoff_runtime_ms,
        "epsilon_gaussian_kappa_forward": epsilon_envelope_forward,
        "epsilon_gaussian_kappa_reverse": epsilon_envelope_reverse,
        "epsilon_gaussian_kappa_two_sided": epsilon_envelope_two_sided,
        "epsilon_actual_over_kappa_curve": (
            epsilon_two_sided / epsilon_envelope_two_sided
        ),
        "evidence_support_groups": 2 if domain_size == 2 else 3,
        "padding_mode": "matched_to_homogeneous_background_payload_law",
        "account_status": "alpha_dependent_gaussian_profile_approximation",
        "chernoff_account_status": "rigorous_for_poisson_surrogate",
        "chernoff_account_method": "optimized_log_mgf_positive_part_bound",
        "exact_profile_status": "not_computed_for_general_K",
        "kappa_status": "pure_epsilon0_LDP_randomizer_independent_envelope",
        "source_type": "formula_profile_approximation",
    }


def generate_general_randomizer_sensitivity() -> int:
    rows = [
        evaluate_general_randomizer_case(domain_size, m)
        for domain_size in GENERAL_K_VALUES
        for m in GENERAL_M_VALUES
    ]
    filename = DATA_DIR / "fig_privacy_general_randomizer_sensitivity.csv"
    write_csv(filename, rows)
    return len(rows)


def generate() -> int:
    _small_enumeration_check()
    rows = [
        evaluate_case("m_scaling", decoder, adjacency, m, rho, MAIN_TAU)
        for decoder in ("drop", "pool")
        for adjacency in ("replacement", "add_remove")
        for rho in RHO_VALUES
        for m in EPOCH_SIZES
    ]
    rows.extend(
        evaluate_case(
            "tau_sensitivity",
            decoder,
            adjacency,
            TAU_SWEEP_M,
            TAU_SWEEP_RHO,
            tau,
        )
        for decoder in ("drop", "pool")
        for adjacency in ("replacement", "add_remove")
        for tau in TAU_VALUES
    )
    filename = DATA_DIR / "fig_privacy_large_m_accounting.csv"
    write_csv(filename, rows)
    return len(rows)


def generate_all() -> dict[str, int]:
    """Regenerate both the main large-m grid and randomizer sensitivity."""

    return {
        "data/fig_privacy_large_m_accounting.csv": generate(),
        "data/fig_privacy_general_randomizer_sensitivity.csv": (
            generate_general_randomizer_sensitivity()
        ),
    }


if __name__ == "__main__":
    print(generate_all())
