from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from chafe_eval_lib import (
    DATA_DIR,
    eps_drop_proxy,
    eps_ideal_padded,
    eps_pool_replacement_generic,
    mean_and_se,
    write_csv,
)


EPS0_GRID = [0.5, 1.0, 2.0, 4.0]
SEEDS = list(range(30))
SAMPLE_M = 100_000
K = 1024
RHO = 1.0
TAU = 2.0
DELTA = 1e-6
RANGE_QUERIES = 1_000
MIN_RANGE_COUNT = 100.0


def release_length() -> int:
    return int(math.ceil(TAU * SAMPLE_M))


def slot_count() -> int:
    return int(math.ceil(SAMPLE_M / RHO))


def grr_params(eps0: float, k: int) -> tuple[float, float]:
    denom = math.exp(eps0) + k - 1.0
    return math.exp(eps0) / denom, 1.0 / denom


def sample_grr(values: np.ndarray, k: int, eps0: float, rng: np.random.Generator) -> np.ndarray:
    p, _ = grr_params(eps0, k)
    keep_true = rng.random(len(values)) < p
    off = rng.integers(0, k - 1, size=len(values))
    off = off + (off >= values)
    return np.where(keep_true, values, off)


def zipf_workload(k: int) -> np.ndarray:
    ranks = np.arange(1, k + 1, dtype=float)
    weights = 1.0 / np.power(ranks, 1.08)
    weights[:8] *= 1.75
    weights[40:56] *= 1.25
    return weights / weights.sum()


def unbiased_histogram(reports: np.ndarray, eps0: float, k: int, m: int, report_scale: float = 1.0) -> np.ndarray:
    p, q = grr_params(eps0, k)
    counts = np.bincount(reports, minlength=k).astype(float)
    return (counts / report_scale - m * q) / (p - q)


def interval_error(true_hist: np.ndarray, est_hist: np.ndarray, rng: np.random.Generator) -> float:
    starts = rng.integers(0, K, size=RANGE_QUERIES)
    ends = rng.integers(0, K, size=RANGE_QUERIES)
    lo = np.minimum(starts, ends)
    hi = np.maximum(starts, ends)
    true_prefix = np.concatenate([[0.0], np.cumsum(true_hist)])
    est_prefix = np.concatenate([[0.0], np.cumsum(est_hist)])
    true_counts = true_prefix[hi + 1] - true_prefix[lo]
    est_counts = est_prefix[hi + 1] - est_prefix[lo]
    mask = true_counts >= MIN_RANGE_COUNT
    if not bool(mask.any()):
        mask = true_counts >= 1.0
    return float(np.mean(np.abs(est_counts[mask] - true_counts[mask]) / true_counts[mask]))


def privacy_reference(method: str, eps0: float) -> tuple[float, str]:
    rel = release_length()
    if method == "All reports":
        return eps_ideal_padded(eps0, rel, DELTA), "second_order_shuffle_same_public_padding"
    if method == "Drop":
        return eps_drop_proxy(eps0, RHO, rel, DELTA), "second_order_matched_drop"
    if method == "Pool":
        return eps_pool_replacement_generic(eps0, rel, DELTA), "second_order_replacement_generic"
    raise ValueError(method)


def simulate_range_fixed_eps0() -> list[dict]:
    metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    methods = ["All reports", "Drop"]
    probs = zipf_workload(K)
    N = slot_count()
    s = math.exp(-RHO)

    for seed in SEEDS:
        rng = np.random.default_rng(1_500_000 + seed)
        values = rng.choice(K, size=SAMPLE_M, p=probs)
        true_hist = np.bincount(values, minlength=K).astype(float)
        interval_rng = np.random.default_rng(1_600_000 + seed)
        for eps0 in EPS0_GRID:
            reports = sample_grr(values, K, eps0, rng)
            for method in methods:
                if method == "Drop":
                    slots = rng.integers(0, N, size=SAMPLE_M)
                    slot_loads = np.bincount(slots, minlength=N)
                    kept = slot_loads[slots] == 1
                    est_hist = unbiased_histogram(reports[kept], eps0, K, SAMPLE_M, report_scale=s)
                else:
                    est_hist = unbiased_histogram(reports, eps0, K, SAMPLE_M)
                metrics[(eps0, method)].append(interval_error(true_hist, est_hist, interval_rng))

    rows: list[dict] = []
    for eps0 in EPS0_GRID:
        for method in methods:
            mean_error, se = mean_and_se(metrics[(eps0, method)])
            eps_prime, bound = privacy_reference(method, eps0)
            rows.append(
                {
                    "eps0": eps0,
                    "epsilon_prime": eps_prime,
                    "method": method,
                    "mean_relative_error": mean_error,
                    "standard_error": se,
                    "source_type": "monte_carlo",
                    "m": SAMPLE_M,
                    "N": N,
                    "rho": RHO,
                    "M": release_length(),
                    "tau": TAU,
                    "delta": DELTA,
                    "decoder": "drop" if method == "Drop" else "all_reports",
                    "bound": bound,
                    "randomizer": "k_ary_randomized_response",
                    "domain_size": K,
                    "seeds": len(SEEDS),
                    "workload": "zipf_histogram_range_fixed_eps0",
                    "metric": f"mean_relative_error_over_intervals_with_true_count_ge_{int(MIN_RANGE_COUNT)}",
                }
            )
    write_csv(DATA_DIR / "fig_fixed_eps0_range_error.csv", rows)
    return rows


def simulate_sum_fixed_eps0() -> list[dict]:
    metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    methods = ["All reports", "Drop", "Pool"]
    N = slot_count()
    s = math.exp(-RHO)

    for seed in SEEDS:
        rng = np.random.default_rng(1_700_000 + seed)
        values = rng.beta(2.0, 5.0, size=SAMPLE_M)
        true_sum = float(values.sum())
        for eps0 in EPS0_GRID:
            reports = values + rng.laplace(0.0, 1.0 / eps0, size=SAMPLE_M)
            for method in methods:
                if method == "Drop":
                    slots = rng.integers(0, N, size=SAMPLE_M)
                    slot_loads = np.bincount(slots, minlength=N)
                    kept = slot_loads[slots] == 1
                    estimate = float(reports[kept].sum() / s)
                else:
                    estimate = float(reports.sum())
                metrics[(eps0, method)].append(abs(estimate - true_sum) / max(true_sum, 1e-12))

    rows: list[dict] = []
    for eps0 in EPS0_GRID:
        for method in methods:
            mean_error, se = mean_and_se(metrics[(eps0, method)])
            eps_prime, bound = privacy_reference(method, eps0)
            rows.append(
                {
                    "eps0": eps0,
                    "epsilon_prime": eps_prime,
                    "method": method,
                    "mean_relative_error": mean_error,
                    "standard_error": se,
                    "source_type": "monte_carlo",
                    "m": SAMPLE_M,
                    "N": N,
                    "rho": RHO,
                    "M": release_length(),
                    "tau": TAU,
                    "delta": DELTA,
                    "decoder": "pool" if method == "Pool" else ("drop" if method == "Drop" else "all_reports"),
                    "bound": bound,
                    "randomizer": "laplace_additive_utility_model",
                    "domain": "[0,1]_beta_2_5",
                    "seeds": len(SEEDS),
                    "workload": "bounded_sum_fixed_eps0",
                    "metric": "relative_sum_error",
                }
            )
    write_csv(DATA_DIR / "fig_fixed_eps0_sum_error.csv", rows)
    return rows


def generate_all() -> dict[str, int]:
    range_rows = simulate_range_fixed_eps0()
    sum_rows = simulate_sum_fixed_eps0()
    return {
        "fig_fixed_eps0_range_error.csv": len(range_rows),
        "fig_fixed_eps0_sum_error.csv": len(sum_rows),
    }


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
