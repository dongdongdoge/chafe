from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import fetch_adult_dataset
from chafe_eval_lib import (
    DATA_DIR,
    eps_drop_proxy,
    eps_ideal_shuffle,
    eps_pool_replacement,
    invert_eps0,
    mean_and_se,
    write_csv,
)


EPS_PRIME_GRID = [0.02, 0.04, 0.08, 0.16, 0.32]
SEEDS = list(range(30))
SAMPLE_M = 5_000
RHO = 1.0
TAU = 2.0
DELTA = 1e-6
RANGE_QUERIES = 1_000
MIN_RANGE_COUNT = 25.0
TOP_K = 50
DATASET_NAME = "UCI Adult / Census Income"
DATASET_URL = "https://archive.ics.uci.edu/dataset/2/adult"


def release_length() -> int:
    return int(math.ceil(TAU * SAMPLE_M))


def slot_count() -> int:
    return int(math.ceil(SAMPLE_M / RHO))


def eps0_for_method(method: str, eps_prime: float) -> float:
    rel = release_length()
    if method == "Local":
        return eps_prime
    if method == "IdealShuffle":
        return invert_eps0(eps_prime, lambda eps0: eps_ideal_shuffle(eps0, SAMPLE_M, DELTA))
    if method == "Drop":
        return invert_eps0(eps_prime, lambda eps0: eps_drop_proxy(eps0, RHO, rel, DELTA))
    if method == "Pool":
        return invert_eps0(eps_prime, lambda eps0: eps_pool_replacement(eps0, RHO, rel, DELTA))
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


def unbiased_histogram(reports: np.ndarray, eps0: float, k: int, m: int, report_scale: float = 1.0) -> np.ndarray:
    p, q = grr_params(eps0, k)
    counts = np.bincount(reports, minlength=k).astype(float)
    return (counts / report_scale - m * q) / (p - q)


def thresholded_interval_error(
    true_hist: np.ndarray,
    est_hist: np.ndarray,
    rng: np.random.Generator,
    min_count: float = MIN_RANGE_COUNT,
) -> float:
    k = len(true_hist)
    starts = rng.integers(0, k, size=RANGE_QUERIES)
    ends = rng.integers(0, k, size=RANGE_QUERIES)
    lo = np.minimum(starts, ends)
    hi = np.maximum(starts, ends)
    true_prefix = np.concatenate([[0.0], np.cumsum(true_hist)])
    est_prefix = np.concatenate([[0.0], np.cumsum(est_hist)])
    true_counts = true_prefix[hi + 1] - true_prefix[lo]
    est_counts = est_prefix[hi + 1] - est_prefix[lo]
    mask = true_counts >= min_count
    if not bool(mask.any()):
        mask = true_counts >= 1.0
    return float(np.mean(np.abs(est_counts[mask] - true_counts[mask]) / true_counts[mask]))


def heavy_hitter_f1(true_hist: np.ndarray, est_hist: np.ndarray, top_k: int = TOP_K) -> float:
    top_k = min(top_k, len(true_hist))
    true_top = set(np.argsort(-true_hist)[:top_k].tolist())
    est_top = set(np.argsort(-est_hist)[:top_k].tolist())
    overlap = len(true_top & est_top)
    return float(overlap / top_k)


def load_adult_columns() -> dict[str, list[str]]:
    clean_path = fetch_adult_dataset.ensure_adult_dataset()
    columns: dict[str, list[str]] = defaultdict(list)
    with clean_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                columns[key].append(value)
    return columns


def encode_categories(values: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    labels = sorted(set(values))
    index = {label: idx for idx, label in enumerate(labels)}
    return np.array([index[value] for value in values], dtype=np.int64), index


def simulate_age_range(age_values: np.ndarray, dataset_rows: int) -> list[dict]:
    age_min = int(age_values.min())
    age_codes = age_values - age_min
    k = int(age_codes.max()) + 1
    metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    eps0_used: dict[tuple[float, str], float] = {}
    methods = ["Local", "IdealShuffle", "Drop"]
    s = math.exp(-RHO)
    N = slot_count()

    for seed in SEEDS:
        rng = np.random.default_rng(1_000_000 + seed)
        sample_idx = rng.choice(len(age_codes), size=SAMPLE_M, replace=False)
        values = age_codes[sample_idx]
        true_hist = np.bincount(values, minlength=k).astype(float)
        interval_rng = np.random.default_rng(1_100_000 + seed)
        for eps_prime in EPS_PRIME_GRID:
            for method in methods:
                eps0 = eps0_for_method(method, eps_prime)
                eps0_used[(eps_prime, method)] = eps0
                reports = sample_grr(values, k, eps0, rng)
                if method == "Drop":
                    slots = rng.integers(0, N, size=SAMPLE_M)
                    slot_loads = np.bincount(slots, minlength=N)
                    kept = slot_loads[slots] == 1
                    est_hist = unbiased_histogram(reports[kept], eps0, k, SAMPLE_M, report_scale=s)
                else:
                    est_hist = unbiased_histogram(reports, eps0, k, SAMPLE_M)
                metrics[(eps_prime, method)].append(thresholded_interval_error(true_hist, est_hist, interval_rng))

    rows: list[dict] = []
    for eps_prime in EPS_PRIME_GRID:
        for method in methods:
            mean_error, se = mean_and_se(metrics[(eps_prime, method)])
            rows.append(
                {
                    "epsilon_prime": eps_prime,
                    "method": method,
                    "mean_relative_error": mean_error,
                    "standard_error": se,
                    "source_type": "monte_carlo",
                    "dataset": DATASET_NAME,
                    "dataset_source_url": DATASET_URL,
                    "dataset_rows_total": dataset_rows,
                    "m": SAMPLE_M,
                    "N": N,
                    "rho": RHO,
                    "M": release_length(),
                    "tau": TAU,
                    "eps0": eps0_used[(eps_prime, method)],
                    "delta": DELTA,
                    "decoder": "drop" if method == "Drop" else "all_reports",
                    "randomizer": "k_ary_randomized_response",
                    "domain": f"age_{age_min}_to_{age_min + k - 1}",
                    "domain_size": k,
                    "seeds": len(SEEDS),
                    "workload": "adult_age_range_count",
                    "metric": f"mean_relative_error_over_intervals_with_true_count_ge_{int(MIN_RANGE_COUNT)}",
                }
            )
    write_csv(DATA_DIR / "fig_real_age_range_error_vs_epsilon.csv", rows)
    return rows


def simulate_heavy_hitters(joint_codes: np.ndarray, k: int, dataset_rows: int) -> list[dict]:
    metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    eps0_used: dict[tuple[float, str], float] = {}
    methods = ["Local", "IdealShuffle", "Drop"]
    s = math.exp(-RHO)
    N = slot_count()

    for seed in SEEDS:
        rng = np.random.default_rng(1_200_000 + seed)
        sample_idx = rng.choice(len(joint_codes), size=SAMPLE_M, replace=False)
        values = joint_codes[sample_idx]
        true_hist = np.bincount(values, minlength=k).astype(float)
        for eps_prime in EPS_PRIME_GRID:
            for method in methods:
                eps0 = eps0_for_method(method, eps_prime)
                eps0_used[(eps_prime, method)] = eps0
                reports = sample_grr(values, k, eps0, rng)
                if method == "Drop":
                    slots = rng.integers(0, N, size=SAMPLE_M)
                    slot_loads = np.bincount(slots, minlength=N)
                    kept = slot_loads[slots] == 1
                    est_hist = unbiased_histogram(reports[kept], eps0, k, SAMPLE_M, report_scale=s)
                else:
                    est_hist = unbiased_histogram(reports, eps0, k, SAMPLE_M)
                metrics[(eps_prime, method)].append(heavy_hitter_f1(true_hist, est_hist))

    rows: list[dict] = []
    for eps_prime in EPS_PRIME_GRID:
        for method in methods:
            f1, se = mean_and_se(metrics[(eps_prime, method)])
            rows.append(
                {
                    "epsilon_prime": eps_prime,
                    "method": method,
                    "f1": f1,
                    "standard_error": se,
                    "source_type": "monte_carlo",
                    "dataset": DATASET_NAME,
                    "dataset_source_url": DATASET_URL,
                    "dataset_rows_total": dataset_rows,
                    "m": SAMPLE_M,
                    "N": N,
                    "rho": RHO,
                    "M": release_length(),
                    "tau": TAU,
                    "eps0": eps0_used[(eps_prime, method)],
                    "delta": DELTA,
                    "decoder": "drop" if method == "Drop" else "all_reports",
                    "randomizer": "k_ary_randomized_response",
                    "domain": "occupation_x_education",
                    "domain_size": k,
                    "seeds": len(SEEDS),
                    "workload": "adult_occupation_education_heavy_hitter",
                    "metric": "top_50_f1",
                }
            )
    write_csv(DATA_DIR / "fig_real_heavyhitter_f1_vs_epsilon.csv", rows)
    return rows


def simulate_hours_sum(hours: np.ndarray, dataset_rows: int) -> list[dict]:
    metrics: dict[tuple[float, str], list[float]] = defaultdict(list)
    eps0_used: dict[tuple[float, str], float] = {}
    methods = ["Local", "IdealShuffle", "Drop", "Pool"]
    N = slot_count()
    s = math.exp(-RHO)
    bounded = np.clip(hours.astype(float) / 99.0, 0.0, 1.0)

    for seed in SEEDS:
        rng = np.random.default_rng(1_300_000 + seed)
        sample_idx = rng.choice(len(bounded), size=SAMPLE_M, replace=False)
        values = bounded[sample_idx]
        true_sum = float(values.sum())
        for eps_prime in EPS_PRIME_GRID:
            for method in methods:
                eps0 = eps0_for_method(method, eps_prime)
                eps0_used[(eps_prime, method)] = eps0
                reports = values + rng.laplace(0.0, 1.0 / eps0, size=SAMPLE_M)
                if method == "Drop":
                    slots = rng.integers(0, N, size=SAMPLE_M)
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
            rows.append(
                {
                    "epsilon_prime": eps_prime,
                    "method": method,
                    "mean_relative_error": mean_error,
                    "standard_error": se,
                    "source_type": "monte_carlo",
                    "dataset": DATASET_NAME,
                    "dataset_source_url": DATASET_URL,
                    "dataset_rows_total": dataset_rows,
                    "m": SAMPLE_M,
                    "N": N,
                    "rho": RHO,
                    "M": release_length(),
                    "tau": TAU,
                    "eps0": eps0_used[(eps_prime, method)],
                    "delta": DELTA,
                    "decoder": "pool" if method == "Pool" else ("drop" if method == "Drop" else "all_reports"),
                    "randomizer": "laplace_additive_utility_model",
                    "domain": "hours_per_week_div_99",
                    "seeds": len(SEEDS),
                    "workload": "adult_hours_per_week_sum",
                    "metric": "relative_sum_error",
                }
            )
    write_csv(DATA_DIR / "fig_real_hours_sum_error_vs_epsilon.csv", rows)
    return rows


def generate_all() -> dict[str, int]:
    columns = load_adult_columns()
    dataset_rows = len(columns["age"])
    ages = np.array([int(x) for x in columns["age"]], dtype=np.int64)
    hours = np.array([int(x) for x in columns["hours_per_week"]], dtype=np.int64)
    joint = [f"{occupation}|{education}" for occupation, education in zip(columns["occupation"], columns["education"])]
    joint_codes, joint_index = encode_categories(joint)

    age_rows = simulate_age_range(ages, dataset_rows)
    hh_rows = simulate_heavy_hitters(joint_codes, len(joint_index), dataset_rows)
    sum_rows = simulate_hours_sum(hours, dataset_rows)
    return {
        "fig_real_age_range_error_vs_epsilon.csv": len(age_rows),
        "fig_real_heavyhitter_f1_vs_epsilon.csv": len(hh_rows),
        "fig_real_hours_sum_error_vs_epsilon.csv": len(sum_rows),
    }


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
