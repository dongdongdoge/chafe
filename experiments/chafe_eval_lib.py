from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
DATASET_DIR = ROOT / "datasets"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    ensure_dirs()
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(x: float, digits: int = 8) -> str:
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 1000 or abs(x) < 1e-3:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def kappa(eps0: float) -> float:
    return math.exp(eps0) + math.exp(-eps0) - 2.0


def epsilon_proxy_from_eta(eta: float, delta: float) -> float:
    """Return the labeled high-privacy proxy sqrt(2 eta log(1/delta)).

    This is a second-order design score, not a finite-mechanism DP accountant.
    Keeping the conversion in one function makes every formula CSV implement
    Equation (epsilon-proxy-conversion) in the paper literally.
    """
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    return math.sqrt(2.0 * eta * math.log(1.0 / delta))


def eps_ideal_shuffle(eps0: float, m: int, delta: float) -> float:
    if m <= 0:
        raise ValueError("m must be positive")
    return epsilon_proxy_from_eta(kappa(eps0) / float(m), delta)


def eps_ideal_padded(eps0: float, M: int, delta: float) -> float:
    """Quadratic ideal-shuffle reference after matched padding to length M."""
    if M <= 0:
        raise ValueError("M must be positive")
    return epsilon_proxy_from_eta(kappa(eps0) / float(M), delta)


def eps_drop_proxy(eps0: float, rho: float, M: int, delta: float) -> float:
    if rho <= 0.0 or M <= 0:
        raise ValueError("rho and M must be positive")
    eta = math.exp(-2.0 * rho) * kappa(eps0) / float(M)
    return epsilon_proxy_from_eta(eta, delta)


def eps_pool_replacement(eps0: float, rho: float, M: int, delta: float) -> float:
    """Homogeneous-background Pool proxy using the proven 1/(c+1) contraction."""
    if rho <= 0.0 or M <= 0:
        raise ValueError("rho and M must be positive")
    eta = ((1.0 - math.exp(-rho)) / rho) * kappa(eps0) / float(M)
    return epsilon_proxy_from_eta(eta, delta)


def eps_pool_replacement_generic(eps0: float, M: int, delta: float) -> float:
    """Generic Pool replacement proxy using only data processing (beta_c <= 1)."""
    return eps_ideal_padded(eps0, M, delta)


def eps_pool_add_remove(eps0: float, rho: float, M: int, delta: float) -> float:
    """I.i.d.-background Pool proxy using exchangeable convolution contraction."""
    if rho <= 0.0 or M <= 0:
        raise ValueError("rho and M must be positive")
    a0 = (1.0 - math.exp(-rho)) / rho - math.exp(-rho)
    ap = (1.0 - math.exp(-rho)) / rho
    eta = (a0 + ap * kappa(eps0)) / float(M)
    return epsilon_proxy_from_eta(eta, delta)


def invert_eps0(
    target_eps: float,
    fn: Callable[[float], float],
    lo: float = 0.0,
    hi: float = 10.0,
    tol: float = 1e-8,
) -> float:
    if target_eps < 0.0:
        raise ValueError("target_eps must be nonnegative")
    flo = fn(lo)
    fhi = fn(hi)
    if target_eps < flo - tol:
        raise ValueError(
            f"target epsilon {target_eps} is below the proxy floor {flo} at eps0={lo}"
        )
    if target_eps <= flo + tol:
        return lo
    if target_eps > fhi:
        raise ValueError(
            f"target epsilon {target_eps} exceeds value {fhi} at eps0={hi}; "
            "increase the search range only if the experiment explicitly permits it"
        )
    left, right = lo, hi
    for _ in range(100):
        mid = 0.5 * (left + right)
        if fn(mid) >= target_eps:
            right = mid
        else:
            left = mid
        if right - left <= tol:
            break
    return right


def brr_probs(eps0: float, true_bit: int) -> tuple[float, float]:
    p = math.exp(eps0) / (math.exp(eps0) + 1.0)
    if true_bit == 1:
        return 1.0 - p, p
    if true_bit == 0:
        return p, 1.0 - p
    raise ValueError("true_bit must be 0 or 1")


_BINOM_CACHE: dict[tuple[int, float], np.ndarray] = {}


def binom_pmf(n: int, p: float) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be nonnegative")
    key = (int(n), round(float(p), 15))
    cached = _BINOM_CACHE.get(key)
    if cached is not None:
        return cached
    if n == 0:
        out = np.array([1.0], dtype=float)
    elif p <= 0.0:
        out = np.zeros(n + 1, dtype=float)
        out[0] = 1.0
    elif p >= 1.0:
        out = np.zeros(n + 1, dtype=float)
        out[n] = 1.0
    else:
        ks = np.arange(n + 1, dtype=float)
        log_coeff = np.array(
            [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in range(n + 1)],
            dtype=float,
        )
        logs = log_coeff + ks * math.log(p) + (n - ks) * math.log1p(-p)
        max_log = float(logs.max())
        out = np.exp(logs - max_log)
        out /= out.sum()
    _BINOM_CACHE[key] = out
    return out


def final_count_distribution_iid_slots(
    num_slots: int,
    M: int,
    singleton_prob: float,
    real_one_prob: float,
    dummy_one_prob: float,
) -> np.ndarray:
    if num_slots > M:
        raise ValueError("exact fixed-length padding DP requires num_slots <= M in this Poissonized validation")
    final = np.zeros(M + 1, dtype=float)
    real_count_pmf = binom_pmf(num_slots, singleton_prob)
    for r, pr in enumerate(real_count_pmf):
        if pr == 0.0:
            continue
        ones_real = binom_pmf(r, real_one_prob)
        ones_dummy = binom_pmf(M - r, dummy_one_prob)
        final += float(pr) * np.convolve(ones_real, ones_dummy)
    final /= final.sum()
    return final


def final_count_distribution_target_add(
    num_background_slots: int,
    M: int,
    singleton_prob: float,
    background_one_prob: float,
    dummy_one_prob: float,
    target_states: list[tuple[int, int, float]],
) -> np.ndarray:
    if num_background_slots + 1 > M:
        raise ValueError("exact fixed-length padding DP requires N <= M in this Poissonized validation")
    final = np.zeros(M + 1, dtype=float)
    real_count_pmf = binom_pmf(num_background_slots, singleton_prob)
    for rb, pr in enumerate(real_count_pmf):
        if pr == 0.0:
            continue
        ones_background = binom_pmf(rb, background_one_prob)
        for rt, yt, pt in target_states:
            if pt == 0.0:
                continue
            r_total = rb + rt
            if r_total > M:
                continue
            ones_dummy = binom_pmf(M - r_total, dummy_one_prob)
            conv = np.convolve(ones_background, ones_dummy)
            padded = np.zeros(M + 1, dtype=float)
            if yt:
                padded[1 : 1 + len(conv)] = conv
            else:
                padded[: len(conv)] = conv
            final += float(pr) * float(pt) * padded
    final /= final.sum()
    return final


def hockey_stick(P: np.ndarray, Q: np.ndarray, eps: float) -> float:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.maximum(P - math.exp(eps) * Q, 0.0).sum())


def two_sided_profile(P: np.ndarray, Q: np.ndarray, eps: float) -> float:
    return max(hockey_stick(P, Q, eps), hockey_stick(Q, P, eps))


def invert_hockey_stick(P: np.ndarray, Q: np.ndarray, delta: float, tol: float = 1e-7) -> float:
    if two_sided_profile(P, Q, 0.0) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while two_sided_profile(P, Q, hi) > delta:
        hi *= 2.0
        if hi > 100:
            raise RuntimeError("hockey-stick inversion did not bracket epsilon")
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if two_sided_profile(P, Q, mid) <= delta:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol:
            break
    return hi


def mean_and_se(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("empty value list")
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return mean, 0.0
    return mean, statistics.stdev(vals) / math.sqrt(len(vals))
