from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from chafe_eval_lib import (
    DATA_DIR,
    eps_drop_proxy,
    eps_ideal_padded,
    eps_pool_replacement_generic,
    invert_eps0,
    mean_and_se,
    write_csv,
)


EPS_PRIME_GRID = [0.02, 0.04, 0.08, 0.16, 0.32]
SEEDS = list(range(30))
M = 100_000
K = 1024
RHO = 1.0
TAU = 2.0
DELTA = 1e-6
RANGE_QUERIES = 1_000
TOP_K = 50
MIN_RANGE_COUNT = 100.0


def release_length() -> int:
    return int(math.ceil(TAU * M))


def slot_count() -> int:
    return int(math.ceil(M / RHO))


def eps0_for_method(method: str, eps_prime: float) -> float:
    rel = release_length()
    if method == "Local":
        return eps_prime
    if method == "IdealShuffle":
        return invert_eps0(eps_prime, lambda eps0: eps_ideal_padded(eps0, rel, DELTA))
    if method == "Drop":
        return invert_eps0(eps_prime, lambda eps0: eps_drop_proxy(eps0, RHO, rel, DELTA))
    if method == "Pool":
        return invert_eps0(eps_prime, lambda eps0: eps_pool_replacement_generic(eps0, rel, DELTA))
    raise ValueError(f"unknown method: {method}")


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


def unbiased_histogram_from_reports(reports: np.ndarray, eps0: float, k: int, report_scale: float = 1.0) -> np.ndarray:
    p, q = grr_params(eps0, k)
    counts = np.bincount(reports, minlength=k).astype(float)
    return (counts / report_scale - M * q) / (p - q)


def interval_query_errors(true_hist: np.ndarray, est_hist: np.ndarray, rng: np.random.Generator) -> float:
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


def heavy_hitter_f1(true_hist: np.ndarray, est_hist: np.ndarray) -> float:
    true_top = set(np.argsort(-true_hist)[:TOP_K].tolist())
    est_top = set(np.argsort(-est_hist)[:TOP_K].tolist())
    overlap = len(true_top & est_top)
    precision = overlap / TOP_K
    recall = overlap / TOP_K
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def simulate_histogram_workload() -> tuple[list[dict], list[dict]]:
    range_metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    f1_metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    eps0_used: dict[tuple[float, str], float] = {}
    methods = ["Local", "IdealShuffle", "Drop"]
    probs = zipf_workload(K)
    s = math.exp(-RHO)
    N = slot_count()

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        values = rng.choice(K, size=M, p=probs)
        true_hist = np.bincount(values, minlength=K).astype(float)
        intervals_rng = np.random.default_rng(100_000 + seed)
        for eps_prime in EPS_PRIME_GRID:
            for method in methods:
                eps0 = eps0_for_method(method, eps_prime)
                eps0_used[(eps_prime, method)] = eps0
                reports = sample_grr(values, K, eps0, rng)
                if method == "Drop":
                    slots = rng.integers(0, N, size=M)
                    slot_loads = np.bincount(slots, minlength=N)
                    kept = slot_loads[slots] == 1
                    est_hist = unbiased_histogram_from_reports(reports[kept], eps0, K, report_scale=s)
                else:
                    est_hist = unbiased_histogram_from_reports(reports, eps0, K)
                range_metrics[(eps_prime, method)].append(interval_query_errors(true_hist, est_hist, intervals_rng))
                f1_metrics[(eps_prime, method)].append(heavy_hitter_f1(true_hist, est_hist))

    range_rows: list[dict] = []
    f1_rows: list[dict] = []
    for eps_prime in EPS_PRIME_GRID:
        for method in methods:
            range_mean, range_se = mean_and_se(range_metrics[(eps_prime, method)])
            f1_mean, f1_se = mean_and_se(f1_metrics[(eps_prime, method)])
            decoder = "drop" if method == "Drop" else "all_reports"
            common = {
                "epsilon_prime": eps_prime,
                "method": method,
                "source_type": "monte_carlo",
                "m": M,
                "N": N,
                "rho": RHO,
                "M": release_length(),
                "tau": TAU,
                "eps0": eps0_used[(eps_prime, method)],
                "delta": DELTA,
                "decoder": decoder,
                "randomizer": "k_ary_randomized_response",
                "domain_size": K,
                "seeds": len(SEEDS),
                "workload": "zipf_histogram_range_and_heavy_hitter",
            }
            range_rows.append(
                {
                    **common,
                    "mean_relative_error": range_mean,
                    "standard_error": range_se,
                    "metric": f"mean_relative_error_over_intervals_with_true_count_ge_{int(MIN_RANGE_COUNT)}",
                }
            )
            f1_rows.append(
                {
                    **common,
                    "f1": f1_mean,
                    "standard_error": f1_se,
                    "metric": "top_50_f1",
                }
            )
    write_csv(DATA_DIR / "fig_range_error_vs_epsilon.csv", range_rows)
    write_csv(DATA_DIR / "fig_heavyhitter_f1_vs_epsilon.csv", f1_rows)
    return range_rows, f1_rows


def simulate_sum_workload() -> list[dict]:
    metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    eps0_used: dict[tuple[float, str], float] = {}
    methods = ["Local", "IdealShuffle", "Drop", "Pool"]
    N = slot_count()
    s = math.exp(-RHO)

    for seed in SEEDS:
        rng = np.random.default_rng(200_000 + seed)
        values = rng.beta(2.0, 5.0, size=M)
        true_sum = float(values.sum())
        for eps_prime in EPS_PRIME_GRID:
            for method in methods:
                eps0 = eps0_for_method(method, eps_prime)
                eps0_used[(eps_prime, method)] = eps0
                reports = values + rng.laplace(0.0, 1.0 / eps0, size=M)
                if method == "Drop":
                    slots = rng.integers(0, N, size=M)
                    slot_loads = np.bincount(slots, minlength=N)
                    kept = slot_loads[slots] == 1
                    estimate = float(reports[kept].sum() / s)
                else:
                    estimate = float(reports.sum())
                metrics[(eps_prime, method)].append(abs(estimate - true_sum) / max(true_sum, 1e-12))

    rows: list[dict] = []
    for eps_prime in EPS_PRIME_GRID:
        for method in methods:
            mean_error, se = mean_and_se(metrics[(eps_prime, method)])
            decoder = "pool" if method == "Pool" else ("drop" if method == "Drop" else "all_reports")
            rows.append(
                {
                    "epsilon_prime": eps_prime,
                    "method": method,
                    "mean_relative_error": mean_error,
                    "standard_error": se,
                    "source_type": "monte_carlo",
                    "m": M,
                    "N": N,
                    "rho": RHO,
                    "M": release_length(),
                    "tau": TAU,
                    "eps0": eps0_used[(eps_prime, method)],
                    "delta": DELTA,
                    "decoder": decoder,
                    "randomizer": "laplace_additive_utility_model",
                    "domain": "[0,1]_beta_2_5",
                    "seeds": len(SEEDS),
                    "workload": "bounded_sum",
                    "metric": "relative_sum_error",
                }
            )
    write_csv(DATA_DIR / "fig_sum_error_vs_epsilon.csv", rows)
    return rows


def generate_all() -> dict[str, int]:
    range_rows, f1_rows = simulate_histogram_workload()
    sum_rows = simulate_sum_workload()
    return {
        "fig_range_error_vs_epsilon.csv": len(range_rows),
        "fig_heavyhitter_f1_vs_epsilon.csv": len(f1_rows),
        "fig_sum_error_vs_epsilon.csv": len(sum_rows),
    }


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
