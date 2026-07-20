"""Matched-Poisson accounting for an additive discrete-Laplace randomizer.

Values in ``[0,1]`` use scale-8 fixed point (three fractional bits) and a
wrapped discrete-Laplace noise law on ``Z_256``.  The cyclic distance between
any two encoded endpoints is at most eight, so the mechanism is
pure-epsilon0 LDP.  Pool pads empty slots with the additive identity; its
completed ordinary-slot law is therefore the compound-Poisson sum itself.
The paper's two-sided Chernoff accountant gives an operational upper bound for
the declared Poisson replacement pair.  No quadratic proxy is used for
calibration.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from privacy_large_m_accounting import account_chernoff_pair


GROUP_SIZE = 256
QUANTIZATION_SCALE = 8
ACCOUNT_SUPPORT_FLOOR = 1e-14


@dataclass(frozen=True)
class LaplaceCalibration:
    method: str
    target_epsilon: float
    local_epsilon0: float
    achieved_chernoff_epsilon: float
    forward_epsilon: float
    reverse_epsilon: float
    profile_at_target: float
    dummy_mean: float
    dummy_variance: float
    local_law_mean_zero: float
    local_law_variance_zero: float


def centered_values() -> np.ndarray:
    indices = np.arange(GROUP_SIZE, dtype=np.int64)
    signed = np.where(indices < GROUP_SIZE // 2, indices, indices - GROUP_SIZE)
    return signed.astype(float) / QUANTIZATION_SCALE


def quantized_laplace_law(raw_value: float, epsilon0: float) -> np.ndarray:
    """Wrapped discrete-Laplace law centered at a scale-8 encoded input."""

    if not 0.0 <= raw_value <= 1.0:
        raise ValueError("raw values must lie in [0,1]")
    if epsilon0 <= 0.0:
        raise ValueError("epsilon0 must be positive")
    center = int(round(raw_value * QUANTIZATION_SCALE)) % GROUP_SIZE
    indices = np.arange(GROUP_SIZE, dtype=np.int64)
    clockwise = (indices - center) % GROUP_SIZE
    distance = np.minimum(clockwise, GROUP_SIZE - clockwise).astype(float)
    law = np.exp(-epsilon0 * distance / QUANTIZATION_SCALE)
    law /= law.sum()
    return law


def verify_endpoint_ldp(epsilon0: float, tolerance: float = 1e-12) -> None:
    """Check the endpoint likelihood-ratio bound used by the accountant."""

    f0 = quantized_laplace_law(0.0, epsilon0)
    f1 = quantized_laplace_law(1.0, epsilon0)
    bound = math.exp(epsilon0) * (1.0 + tolerance)
    if float(np.max(f0 / f1)) > bound or float(np.max(f1 / f0)) > bound:
        raise AssertionError("wrapped discrete-Laplace law violates epsilon0-LDP")


def circular_convolution(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.fft.ifft(np.fft.fft(left) * np.fft.fft(right)).real
    result[result < 0.0] = 0.0
    result /= result.sum()
    return result


def compound_poisson_law(payload: np.ndarray, load: float) -> np.ndarray:
    if load <= 0.0:
        raise ValueError("load must be positive")
    transform = np.fft.fft(payload)
    result = np.fft.ifft(np.exp(load * (transform - 1.0))).real
    # FFT roundoff can create tiny negative entries.  The true modular
    # compound-Poisson law has full support because the payload support has
    # gcd one.  Flooring below machine relevance preserves numerical
    # stability while remaining far below every reported delta.
    result = np.maximum(result, 1e-300)
    result /= result.sum()
    return result


def pool_case_laws(epsilon0: float, common_load: float) -> tuple[np.ndarray, ...]:
    f0 = quantized_laplace_law(0.0, epsilon0)
    f1 = quantized_laplace_law(1.0, epsilon0)
    compound = compound_poisson_law(f0, common_load)
    # Pool emits the aggregate of a nonempty slot and pads an empty slot with
    # the additive identity.  Thus a completed ordinary slot has exactly the
    # (unconditioned) compound-Poisson law.
    reference = compound
    target0 = circular_convolution(f0, compound)
    target1 = circular_convolution(f1, compound)
    return reference, target0, target1, f0, f1


def ideal_case_laws(epsilon0: float) -> tuple[np.ndarray, ...]:
    f0 = quantized_laplace_law(0.0, epsilon0)
    f1 = quantized_laplace_law(1.0, epsilon0)
    return f0, f0, f1


def _moments(law: np.ndarray) -> tuple[float, float]:
    values = centered_values()
    mean = float(np.dot(law, values))
    variance = float(np.dot(law, np.square(values - mean)))
    return mean, variance


def _common_support(
    reference: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    release_length: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    """Condition away a numerically negligible tail and bound its mass."""

    support = reference >= ACCOUNT_SUPPORT_FLOOR
    reference_tail = float(reference[~support].sum())
    target_tail = float(max(left[~support].sum(), right[~support].sum()))
    # A release contains M-1 ordinary positions and one target position.  The
    # union bound controls the probability that conditioning changes either
    # neighboring release.  Five times this mass safely covers both hockey-
    # stick directions throughout the epsilon range used in the experiments.
    tail_slack = 5.0 * (release_length * reference_tail + target_tail)
    ref = reference[support]
    lhs = left[support]
    rhs = right[support]
    ref /= ref.sum()
    lhs /= lhs.sum()
    rhs /= rhs.sum()
    return tuple(ref), tuple(lhs), tuple(rhs), tail_slack


@functools.lru_cache(maxsize=None)
def chernoff_budget(
    method: str,
    epsilon0_rounded: float,
    common_load: float,
    release_length: int,
    delta: float,
) -> tuple[float, float, float, float, float, float]:
    epsilon0 = float(epsilon0_rounded)
    verify_endpoint_ldp(epsilon0)
    if method == "Padded Ideal":
        reference, left, right = ideal_case_laws(epsilon0)
        dummy = reference
    elif method == "Pool":
        reference, left, right, _, _ = pool_case_laws(epsilon0, common_load)
        dummy = np.zeros_like(reference)
        dummy[0] = 1.0
    else:
        raise ValueError("method must be Padded Ideal or Pool")
    ref, lhs, rhs, tail_slack = _common_support(
        reference, left, right, release_length
    )
    if tail_slack >= delta:
        raise ValueError(
            f"accounting tail slack {tail_slack:.3e} exceeds delta {delta:.3e}"
        )
    account = account_chernoff_pair(
        ref, lhs, rhs, release_length, delta - tail_slack
    )
    dummy_mean, dummy_variance = _moments(dummy)
    return (
        float(account["two_sided"]),
        float(account["forward"]),
        float(account["reverse"]),
        float(account["profile_bound"]) + tail_slack,
        dummy_mean,
        dummy_variance,
    )


def calibrate_local_epsilon(
    method: str,
    target_epsilon: float,
    epoch_size: int,
    slot_count: int,
    release_length: int,
    delta: float,
) -> LaplaceCalibration:
    if method == "Pool" and release_length < slot_count:
        raise ValueError("matched one-slot accounting requires M >= N")
    common_load = (epoch_size - 1) / slot_count
    account_length = epoch_size if method == "Padded Ideal" else release_length

    def budget(epsilon0: float) -> tuple[float, float, float, float, float, float]:
        return chernoff_budget(
            method,
            round(float(epsilon0), 12),
            common_load,
            account_length,
            delta,
        )

    lo, hi = 0.05, 2.0
    while budget(hi)[0] < target_epsilon:
        hi *= 1.5
        if hi > 24.0:
            raise RuntimeError("failed to bracket the local epsilon")
    for _ in range(38):
        mid = 0.5 * (lo + hi)
        if budget(mid)[0] <= target_epsilon:
            lo = mid
        else:
            hi = mid
    achieved, forward, reverse, profile, dummy_mean, dummy_variance = budget(lo)
    f0 = quantized_laplace_law(0.0, lo)
    local_mean, local_variance = _moments(f0)
    return LaplaceCalibration(
        method=method,
        target_epsilon=target_epsilon,
        local_epsilon0=lo,
        achieved_chernoff_epsilon=achieved,
        forward_epsilon=forward,
        reverse_epsilon=reverse,
        profile_at_target=profile,
        dummy_mean=dummy_mean,
        dummy_variance=dummy_variance,
        local_law_mean_zero=local_mean,
        local_law_variance_zero=local_variance,
    )


def sample_law(rng: np.random.Generator, law: np.ndarray, size: int) -> np.ndarray:
    indices = rng.choice(GROUP_SIZE, size=size, p=law)
    signed = np.where(indices < GROUP_SIZE // 2, indices, indices - GROUP_SIZE)
    return signed.astype(np.int64)


if __name__ == "__main__":
    for target in (0.1, 0.3, 0.7):
        for method in ("Padded Ideal", "Pool"):
            result = calibrate_local_epsilon(
                method=method,
                target_epsilon=target,
                epoch_size=100_000,
                slot_count=200_000,
                release_length=200_000,
                delta=1e-6,
            )
            print(
                target,
                method,
                f"eps0={result.local_epsilon0:.6f}",
                f"eps'={result.achieved_chernoff_epsilon:.6f}",
                f"dummy_var={result.dummy_variance:.6f}",
            )
