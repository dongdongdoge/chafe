from __future__ import annotations

import csv
import math
from collections import defaultdict

import numpy as np

import fetch_covertype_dataset
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
SEEDS = list(range(20))
SAMPLE_M = 300_000
RHO = 1.0
TAU = 2.0
DELTA = 1e-6
ELEVATION_BINS = 512
HH_DOMAIN = 4 * 40 * 7
RANGE_QUERIES = 1_000
MIN_RANGE_COUNT = 100.0
TOP_K = 50
DATASET_NAME = "UCI Covertype"
DATASET_URL = "https://archive.ics.uci.edu/dataset/31/covertype"
DATASET_DOI = "10.24432/C50K5N"


def release_length() -> int:
    return int(math.ceil(TAU * SAMPLE_M))


def slot_count() -> int:
    return int(math.ceil(SAMPLE_M / RHO))


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


def unbiased_histogram(reports: np.ndarray, eps0: float, k: int, report_scale: float = 1.0) -> np.ndarray:
    p, q = grr_params(eps0, k)
    counts = np.bincount(reports, minlength=k).astype(float)
    return (counts / report_scale - SAMPLE_M * q) / (p - q)


def interval_error(true_hist: np.ndarray, est_hist: np.ndarray, rng: np.random.Generator) -> float:
    k = len(true_hist)
    starts = rng.integers(0, k, size=RANGE_QUERIES)
    ends = rng.integers(0, k, size=RANGE_QUERIES)
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
    return float(len(true_top & est_top) / TOP_K)


def load_selected_columns() -> dict[str, np.ndarray]:
    path = fetch_covertype_dataset.ensure_covertype_dataset()
    columns = {
        "elevation": [],
        "horizontal_distance_to_roadways": [],
        "wilderness_area": [],
        "soil_type": [],
        "cover_type": [],
    }
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in columns:
                columns[name].append(int(row[name]))
    return {name: np.asarray(values, dtype=np.int64) for name, values in columns.items()}


def base_row(task: str, method: str, eps_prime: float, eps0: float, decoder: str, dataset_rows: int) -> dict:
    return {
        "epsilon_prime": eps_prime,
        "method": method,
        "source_type": "monte_carlo",
        "dataset": DATASET_NAME,
        "dataset_source_url": DATASET_URL,
        "dataset_doi": DATASET_DOI,
        "dataset_rows_total": dataset_rows,
        "sampling_model": "without_replacement_sample",
        "m": SAMPLE_M,
        "N": slot_count(),
        "rho": RHO,
        "M": release_length(),
        "tau": TAU,
        "eps0": eps0,
        "delta": DELTA,
        "decoder": decoder,
        "seeds": len(SEEDS),
        "task": task,
    }


def generate_all() -> dict[str, int]:
    data = load_selected_columns()
    dataset_rows = len(data["elevation"])
    if dataset_rows < SAMPLE_M:
        raise ValueError(f"Covertype has {dataset_rows} rows, below SAMPLE_M={SAMPLE_M}")

    elevation = data["elevation"]
    elev_min = int(elevation.min())
    elev_span = int(elevation.max() - elev_min + 1)
    elevation_bins = np.minimum(
        ELEVATION_BINS - 1,
        ((elevation - elev_min) * ELEVATION_BINS // elev_span).astype(np.int64),
    )
    hh_codes = (
        (data["wilderness_area"] - 1) * (40 * 7)
        + (data["soil_type"] - 1) * 7
        + (data["cover_type"] - 1)
    ).astype(np.int64)
    road = data["horizontal_distance_to_roadways"].astype(float)
    road_norm = np.clip(road / max(float(road.max()), 1.0), 0.0, 1.0)

    metrics: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    eps0_used: dict[tuple[str, float, str], float] = {}
    categorical_methods = ["Local", "IdealShuffle", "Drop"]
    sum_methods = ["Local", "IdealShuffle", "Drop", "Pool"]
    s = math.exp(-RHO)
    n_slots = slot_count()

    for seed in SEEDS:
        rng = np.random.default_rng(3_000_000 + seed)
        sample_idx = rng.choice(dataset_rows, size=SAMPLE_M, replace=False)

        range_values = elevation_bins[sample_idx]
        true_range_hist = np.bincount(range_values, minlength=ELEVATION_BINS).astype(float)
        interval_rng = np.random.default_rng(3_100_000 + seed)
        hh_values = hh_codes[sample_idx]
        true_hh_hist = np.bincount(hh_values, minlength=HH_DOMAIN).astype(float)
        sum_values = road_norm[sample_idx]
        true_sum = float(sum_values.sum())

        for eps_prime in EPS_PRIME_GRID:
            for method in categorical_methods:
                eps0 = eps0_for_method(method, eps_prime)
                eps0_used[("elevation_range", eps_prime, method)] = eps0
                reports = sample_grr(range_values, ELEVATION_BINS, eps0, rng)
                if method == "Drop":
                    slots = rng.integers(0, n_slots, size=SAMPLE_M)
                    kept = np.bincount(slots, minlength=n_slots)[slots] == 1
                    est_hist = unbiased_histogram(reports[kept], eps0, ELEVATION_BINS, report_scale=s)
                else:
                    est_hist = unbiased_histogram(reports, eps0, ELEVATION_BINS)
                metrics[("elevation_range", eps_prime, method)].append(
                    interval_error(true_range_hist, est_hist, interval_rng)
                )

                eps0_used[("soil_cover_hh", eps_prime, method)] = eps0
                reports = sample_grr(hh_values, HH_DOMAIN, eps0, rng)
                if method == "Drop":
                    slots = rng.integers(0, n_slots, size=SAMPLE_M)
                    kept = np.bincount(slots, minlength=n_slots)[slots] == 1
                    est_hist = unbiased_histogram(reports[kept], eps0, HH_DOMAIN, report_scale=s)
                else:
                    est_hist = unbiased_histogram(reports, eps0, HH_DOMAIN)
                metrics[("soil_cover_hh", eps_prime, method)].append(heavy_hitter_f1(true_hh_hist, est_hist))

            for method in sum_methods:
                eps0 = eps0_for_method(method, eps_prime)
                eps0_used[("road_distance_sum", eps_prime, method)] = eps0
                reports = sum_values + rng.laplace(0.0, 1.0 / eps0, size=SAMPLE_M)
                if method == "Drop":
                    slots = rng.integers(0, n_slots, size=SAMPLE_M)
                    kept = np.bincount(slots, minlength=n_slots)[slots] == 1
                    estimate = float(reports[kept].sum() / s)
                else:
                    estimate = float(reports.sum())
                metrics[("road_distance_sum", eps_prime, method)].append(
                    abs(estimate - true_sum) / max(true_sum, 1e-12)
                )

    rows: list[dict] = []
    task_meta = {
        "elevation_range": (
            categorical_methods,
            "k_ary_randomized_response",
            ELEVATION_BINS,
            "mean_relative_error_over_elevation_intervals",
            "mean_relative_error",
            "elevation_binned_512",
        ),
        "soil_cover_hh": (
            categorical_methods,
            "k_ary_randomized_response",
            HH_DOMAIN,
            "top_50_f1",
            "f1",
            "wilderness_x_soil_x_cover_type",
        ),
        "road_distance_sum": (
            sum_methods,
            "laplace_additive_utility_model",
            "",
            "relative_sum_error",
            "mean_relative_error",
            "horizontal_distance_to_roadways_normalized",
        ),
    }
    for task, (methods, randomizer, domain_size, metric, value_key, domain) in task_meta.items():
        for eps_prime in EPS_PRIME_GRID:
            for method in methods:
                mean, se = mean_and_se(metrics[(task, eps_prime, method)])
                decoder = "pool" if method == "Pool" else ("drop" if method == "Drop" else "all_reports")
                row = {
                    **base_row(task, method, eps_prime, eps0_used[(task, eps_prime, method)], decoder, dataset_rows),
                    "randomizer": randomizer,
                    "domain": domain,
                    "domain_size": domain_size,
                    "metric": metric,
                    value_key: mean,
                    "standard_error": se,
                }
                rows.append(row)

    write_csv(DATA_DIR / "fig_covertype_workloads.csv", rows)
    return {"fig_covertype_workloads.csv": len(rows)}


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
