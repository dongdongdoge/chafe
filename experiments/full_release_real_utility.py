from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import fetch_adult_dataset
import fetch_covertype_dataset
from chafe_eval_lib import DATA_DIR, mean_and_se, write_csv
from padded_estimators import (
    corrected_affine_total,
    corrected_grr_histogram,
    expected_pool_outputs,
)


SAMPLE_M = 40_000
N_SLOTS = 40_000
RELEASE_M = 80_000
SEEDS = list(range(100))
CATEGORICAL_EPS0 = 5.0
SUM_EPS0 = 1.0
# Match the 56-bit payload field of the packed64 backend.
SUM_MODULUS = 1 << 56
SUM_DUMMY_INPUT_MEAN = 0.5
ELEVATION_BINS = 512
SYNTHETIC_CATEGORIES = 1024
RANGE_QUERIES = 500
TOP_K = 50


@dataclass(frozen=True)
class DatasetWorkloads:
    name: str
    range_values: np.ndarray | None
    bounded_sum_values: np.ndarray | None
    heavy_hitter_values: np.ndarray | None
    range_domain_size: int
    heavy_hitter_domain_size: int
    seed_offset: int
    dataset_source: str
    dataset_rows_total: int | None
    synthetic: bool
    sampling_models: dict[str, str]
    workload_names: dict[str, str]
    domain_names: dict[str, str]


def encode_categories(values: list[str]) -> tuple[np.ndarray, int]:
    labels = sorted(set(values))
    index = {label: position for position, label in enumerate(labels)}
    return np.fromiter((index[value] for value in values), dtype=np.int64), len(labels)


def load_adult() -> DatasetWorkloads:
    ages, hours, joint_categories = [], [], []
    with fetch_adult_dataset.ensure_adult_dataset().open(newline="") as f:
        for row in csv.DictReader(f):
            ages.append(int(row["age"]))
            hours.append(int(row["hours_per_week"]))
            joint_categories.append(f"{row['occupation']}|{row['education']}")
    age = np.asarray(ages, dtype=np.int64)
    age -= int(age.min())
    bounded_hours = np.clip(np.asarray(hours, dtype=float) / 99.0, 0.0, 1.0)
    hh_codes, hh_domain_size = encode_categories(joint_categories)
    return DatasetWorkloads(
        name="Adult",
        range_values=age,
        bounded_sum_values=bounded_hours,
        heavy_hitter_values=hh_codes,
        range_domain_size=int(age.max()) + 1,
        heavy_hitter_domain_size=hh_domain_size,
        seed_offset=10_000,
        dataset_source="UCI Adult / Census Income",
        dataset_rows_total=len(age),
        synthetic=False,
        sampling_models={
            "Range": "empirical_bootstrap_with_replacement",
            "Sum": "empirical_bootstrap_with_replacement",
            "Heavy hitter": "empirical_bootstrap_with_replacement",
        },
        workload_names={
            "Range": "adult_age_range",
            "Sum": "adult_hours_per_week_bounded_sum",
            "Heavy hitter": "adult_occupation_x_education_top50",
        },
        domain_names={
            "Range": "adult_age_shifted_to_zero",
            "Sum": "adult_hours_per_week_div_99",
            "Heavy hitter": "adult_occupation_x_education",
        },
    )


def load_covertype() -> DatasetWorkloads:
    elevation, road, wilderness, soil, cover = [], [], [], [], []
    with fetch_covertype_dataset.ensure_covertype_dataset().open(newline="") as f:
        for row in csv.DictReader(f):
            elevation.append(int(row["elevation"]))
            road.append(int(row["horizontal_distance_to_roadways"]))
            wilderness.append(int(row["wilderness_area"]))
            soil.append(int(row["soil_type"]))
            cover.append(int(row["cover_type"]))
    elevation_array = np.asarray(elevation, dtype=np.int64)
    lo = int(elevation_array.min())
    span = int(elevation_array.max()) - lo + 1
    bins = np.minimum(
        ELEVATION_BINS - 1,
        ((elevation_array - lo) * ELEVATION_BINS // span).astype(np.int64),
    )
    road_array = np.asarray(road, dtype=float)
    bounded_road = np.clip(road_array / max(float(road_array.max()), 1.0), 0.0, 1.0)
    hh_codes = (
        (np.asarray(wilderness, dtype=np.int64) - 1) * (40 * 7)
        + (np.asarray(soil, dtype=np.int64) - 1) * 7
        + (np.asarray(cover, dtype=np.int64) - 1)
    )
    return DatasetWorkloads(
        name="Covertype",
        range_values=bins,
        bounded_sum_values=bounded_road,
        heavy_hitter_values=hh_codes,
        range_domain_size=ELEVATION_BINS,
        heavy_hitter_domain_size=4 * 40 * 7,
        seed_offset=20_000,
        dataset_source="UCI Covertype",
        dataset_rows_total=len(bins),
        synthetic=False,
        sampling_models={
            "Range": "without_replacement_sample",
            "Sum": "without_replacement_sample",
            "Heavy hitter": "without_replacement_sample",
        },
        workload_names={
            "Range": "covertype_elevation_range",
            "Sum": "covertype_road_distance_bounded_sum",
            "Heavy hitter": "covertype_wilderness_x_soil_x_cover_top50",
        },
        domain_names={
            "Range": "covertype_elevation_binned_512",
            "Sum": "covertype_road_distance_normalized",
            "Heavy hitter": "covertype_wilderness_x_soil_x_cover",
        },
    )


def synthetic_workloads() -> DatasetWorkloads:
    return DatasetWorkloads(
        name="Synthetic",
        range_values=None,
        bounded_sum_values=None,
        heavy_hitter_values=None,
        range_domain_size=SYNTHETIC_CATEGORIES,
        heavy_hitter_domain_size=SYNTHETIC_CATEGORIES,
        seed_offset=0,
        dataset_source="synthetic_distribution",
        dataset_rows_total=None,
        synthetic=True,
        sampling_models={
            "Range": "iid_modified_zipf_1.08",
            "Sum": "iid_beta_2_5",
            "Heavy hitter": "iid_modified_zipf_1.08",
        },
        workload_names={
            "Range": "synthetic_zipf_range",
            "Sum": "synthetic_beta_bounded_sum",
            "Heavy hitter": "synthetic_zipf_top50",
        },
        domain_names={
            "Range": "modified_zipf_categories_1024",
            "Sum": "beta_2_5_on_unit_interval",
            "Heavy hitter": "modified_zipf_categories_1024",
        },
    )


def zipf_probabilities(k: int) -> np.ndarray:
    ranks = np.arange(1, k + 1, dtype=float)
    weights = 1.0 / np.power(ranks, 1.08)
    weights[:8] *= 1.75
    weights[40:56] *= 1.25
    return weights / weights.sum()


def sample_epoch(
    workload: DatasetWorkloads,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if workload.synthetic:
        categories = rng.choice(
            workload.range_domain_size,
            size=SAMPLE_M,
            p=zipf_probabilities(workload.range_domain_size),
        ).astype(np.int64)
        return categories, rng.beta(2.0, 5.0, size=SAMPLE_M), categories.copy()

    if (
        workload.range_values is None
        or workload.bounded_sum_values is None
        or workload.heavy_hitter_values is None
    ):
        raise AssertionError("real workload is missing aligned source columns")
    source_rows = len(workload.range_values)
    replace = workload.sampling_models["Range"] == "empirical_bootstrap_with_replacement"
    indices = rng.choice(source_rows, size=SAMPLE_M, replace=replace)
    return (
        workload.range_values[indices],
        workload.bounded_sum_values[indices],
        workload.heavy_hitter_values[indices],
    )


def grr_params(eps0: float, k: int) -> tuple[float, float]:
    denom = math.exp(eps0) + k - 1.0
    return math.exp(eps0) / denom, 1.0 / denom


def sample_grr(values: np.ndarray, k: int, eps0: float, rng: np.random.Generator) -> np.ndarray:
    p, _ = grr_params(eps0, k)
    keep = rng.random(len(values)) < p
    off = rng.integers(0, k - 1, size=len(values))
    off += off >= values
    return np.where(keep, values, off)


def bounded_sum_reports(
    values: np.ndarray,
    eps0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One-bit pure-LDP reports with an exact affine expectation on [0,1]."""
    values = np.asarray(values, dtype=float)
    if bool((values < 0.0).any()) or bool((values > 1.0).any()):
        raise ValueError("bounded-sum inputs must lie in [0,1]")
    truth, alternative = grr_params(eps0, 2)
    probabilities = alternative + (truth - alternative) * values
    return (rng.random(len(values)) < probabilities).astype(np.int64)


def additive_dummies(
    method: str,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, str]:
    """Sample the same nonzero public law for ideal and Pool releases."""
    if method not in {"Ideal", "Pool"}:
        raise ValueError(f"unsupported additive dummy method: {method}")
    # A public input mean of 1/2 induces Bernoulli(1/2) reports for every
    # epsilon0.  This is an arbitrary public nu rather than matched pooled-slot
    # padding; the fixed-eps0 experiment makes no central-DP claim.
    return (
        rng.binomial(1, 0.5, size=count).astype(np.int64),
        0.5,
        "public_bernoulli_half",
    )


def corrected_histogram(
    release: np.ndarray,
    k: int,
    survival: float,
) -> np.ndarray:
    p, q = grr_params(CATEGORICAL_EPS0, k)
    counts = np.bincount(release, minlength=k).astype(float)
    expected_real = SAMPLE_M * survival
    uniform = np.full(k, 1.0 / k)
    return corrected_grr_histogram(
        counts,
        epoch_size=SAMPLE_M,
        release_length=len(release),
        expected_real_outputs=expected_real,
        real_report_retention=survival,
        dummy_law=uniform,
        truthful_probability=p,
        alternative_probability=q,
    )


def interval_error(
    true_hist: np.ndarray,
    estimate: np.ndarray,
    rng: np.random.Generator,
) -> float:
    k = len(true_hist)
    left = rng.integers(0, k, size=RANGE_QUERIES)
    right = rng.integers(0, k, size=RANGE_QUERIES)
    lo = np.minimum(left, right)
    hi = np.maximum(left, right)
    true_prefix = np.concatenate(([0.0], np.cumsum(true_hist)))
    est_prefix = np.concatenate(([0.0], np.cumsum(estimate)))
    truth = true_prefix[hi + 1] - true_prefix[lo]
    pred = est_prefix[hi + 1] - est_prefix[lo]
    mask = truth >= 50.0
    if not bool(mask.any()):
        mask = truth > 0
    return float(np.mean(np.abs(pred[mask] - truth[mask]) / truth[mask]))


def heavy_hitter_f1(
    true_hist: np.ndarray,
    estimate: np.ndarray,
) -> float:
    top_k = min(TOP_K, len(true_hist))
    true_top = set(np.argsort(-true_hist)[:top_k].tolist())
    estimated_top = set(np.argsort(-estimate)[:top_k].tolist())
    return float(len(true_top & estimated_top) / top_k)


def complete_categorical_release(
    reports: np.ndarray,
    domain_size: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    if method == "Ideal":
        real = reports
        survival = 1.0
    elif method == "Drop":
        slots = rng.integers(0, N_SLOTS, size=SAMPLE_M)
        loads = np.bincount(slots, minlength=N_SLOTS)
        real = reports[loads[slots] == 1]
        survival = (1.0 - 1.0 / N_SLOTS) ** (SAMPLE_M - 1)
    else:
        raise ValueError(f"unsupported categorical release: {method}")

    dummy_count = RELEASE_M - len(real)
    if dummy_count < 0:
        raise AssertionError("categorical output exceeds the public release length")
    dummies = rng.integers(0, domain_size, size=dummy_count)
    release = np.concatenate((real, dummies)).astype(np.int64, copy=False)
    rng.shuffle(release)
    if len(release) != RELEASE_M:
        raise AssertionError("categorical release does not have length M")
    return release, survival, len(real)


def simulate_dataset(
    workload: DatasetWorkloads,
) -> list[dict]:
    metrics: dict[tuple[str, str], list[float]] = defaultdict(list)
    real_counts: dict[tuple[str, str], list[float]] = defaultdict(list)
    sum_signed_errors: dict[str, list[float]] = defaultdict(list)

    for seed in SEEDS:
        namespace = workload.seed_offset
        data_rng = np.random.default_rng(4_000_000 + namespace + seed)
        range_values, values, heavy_hitter_values = sample_epoch(workload, data_rng)
        true_range_hist = np.bincount(
            range_values, minlength=workload.range_domain_size
        ).astype(float)
        true_heavy_hitter_hist = np.bincount(
            heavy_hitter_values, minlength=workload.heavy_hitter_domain_size
        ).astype(float)
        true_sum = float(values.sum())

        range_report_rng = np.random.default_rng(4_100_000 + namespace + seed)
        range_reports = sample_grr(
            range_values,
            workload.range_domain_size,
            CATEGORICAL_EPS0,
            range_report_rng,
        )
        for method_index, method in enumerate(["Ideal", "Drop"]):
            release_rng = np.random.default_rng(
                4_200_000 + 100_000 * method_index + namespace + seed
            )
            release, survival, real_count = complete_categorical_release(
                range_reports, workload.range_domain_size, method, release_rng
            )
            estimate = corrected_histogram(
                release, workload.range_domain_size, survival
            )
            query_rng = np.random.default_rng(
                4_300_000 + namespace + seed
            )
            metrics[("Range", method)].append(
                interval_error(true_range_hist, estimate, query_rng)
            )
            real_counts[("Range", method)].append(float(real_count))

        hh_report_rng = np.random.default_rng(4_600_000 + namespace + seed)
        hh_reports = sample_grr(
            heavy_hitter_values,
            workload.heavy_hitter_domain_size,
            CATEGORICAL_EPS0,
            hh_report_rng,
        )
        for method_index, method in enumerate(["Ideal", "Drop"]):
            release_rng = np.random.default_rng(
                4_700_000 + 100_000 * method_index + namespace + seed
            )
            release, survival, real_count = complete_categorical_release(
                hh_reports,
                workload.heavy_hitter_domain_size,
                method,
                release_rng,
            )
            estimate = corrected_histogram(
                release, workload.heavy_hitter_domain_size, survival
            )
            metrics[("Heavy hitter", method)].append(
                heavy_hitter_f1(true_heavy_hitter_hist, estimate)
            )
            real_counts[("Heavy hitter", method)].append(float(real_count))

        sum_rng = np.random.default_rng(4_400_000 + namespace + seed)
        sum_reports = bounded_sum_reports(values, SUM_EPS0, sum_rng)
        pool_slots = sum_rng.integers(0, N_SLOTS, size=SAMPLE_M)
        slot_sums = np.bincount(pool_slots, weights=sum_reports, minlength=N_SLOTS)
        pool_real = slot_sums[
            np.bincount(pool_slots, minlength=N_SLOTS) > 0
        ].astype(np.int64)

        for method_index, (method, real) in enumerate(
            [("Ideal", sum_reports), ("Pool", pool_real)]
        ):
            dummy_count = RELEASE_M - len(real)
            dummy_rng = np.random.default_rng(
                4_500_000 + 100_000 * method_index + namespace + seed
            )
            dummies, dummy_mean, _ = additive_dummies(method, dummy_count, dummy_rng)
            release = np.concatenate((real, dummies))
            dummy_rng.shuffle(release)
            if len(release) != RELEASE_M:
                raise AssertionError("additive release does not have length M")
            if SAMPLE_M * RELEASE_M >= SUM_MODULUS // 2:
                raise AssertionError("declared centered modular envelope can wrap")
            truth, alternative = grr_params(SUM_EPS0, 2)
            expected_real = (
                float(SAMPLE_M)
                if method == "Ideal"
                else expected_pool_outputs(SAMPLE_M, N_SLOTS)
            )
            estimate = corrected_affine_total(
                float(release.sum(dtype=np.int64)),
                epoch_size=SAMPLE_M,
                release_length=len(release),
                expected_real_outputs=expected_real,
                real_report_retention=1.0,
                dummy_mean=dummy_mean,
                affine_offset=alternative,
                affine_slope=truth - alternative,
            )
            signed_error = estimate - true_sum
            sum_signed_errors[method].append(signed_error)
            metrics[("Sum", method)].append(abs(signed_error) / max(true_sum, 1e-12))
            real_counts[("Sum", method)].append(float(len(real)))

    rows = []
    for (task, method), metric_values in sorted(metrics.items()):
        mean, se = mean_and_se(np.asarray(metric_values, dtype=float))
        count_mean, count_se = mean_and_se(np.asarray(real_counts[(task, method)], dtype=float))
        decoder = {"Ideal": "all_reports", "Drop": "drop", "Pool": "pool"}[method]
        categorical = task in {"Range", "Heavy hitter"}
        sum_errors = np.asarray(sum_signed_errors[method], dtype=float) if task == "Sum" else None
        sum_bias = float(sum_errors.mean()) if sum_errors is not None else ""
        sum_rmse = (
            float(np.sqrt(np.square(sum_errors).mean()))
            if sum_errors is not None
            else ""
        )
        if task == "Sum":
            truth, alternative = grr_params(SUM_EPS0, 2)
            dummy_mean = (
                0.5
            )
            expected_real = (
                float(SAMPLE_M)
                if method == "Ideal"
                else expected_pool_outputs(SAMPLE_M, N_SLOTS)
            )
            additive_dummy_law = (
                "public_bernoulli_half"
            )
        else:
            truth = alternative = dummy_mean = expected_real = ""
            additive_dummy_law = ""
        rows.append(
            {
                "dataset": workload.name,
                "task": task,
                "method": method,
                "value": mean,
                "standard_error": se,
                "metric": {
                    "Range": "mean_relative_range_error",
                    "Sum": "relative_sum_error",
                    "Heavy hitter": "top_50_f1",
                }[task],
                "source_type": "monte_carlo",
                "dataset_source": workload.dataset_source,
                "dataset_rows_total": workload.dataset_rows_total
                if workload.dataset_rows_total is not None
                else "",
                "synthetic": str(workload.synthetic).lower(),
                "workload": workload.workload_names[task],
                "domain": workload.domain_names[task],
                "domain_size": (
                    workload.range_domain_size
                    if task == "Range"
                    else workload.heavy_hitter_domain_size
                    if task == "Heavy hitter"
                    else ""
                ),
                "m": SAMPLE_M,
                "N": N_SLOTS,
                "rho": SAMPLE_M / N_SLOTS,
                "M": RELEASE_M,
                "tau": RELEASE_M / SAMPLE_M,
                "eps0": CATEGORICAL_EPS0 if categorical else SUM_EPS0,
                "decoder": decoder,
                "dummy_law": (
                    "uniform_categorical"
                    if categorical
                    else additive_dummy_law
                ),
                "sum_randomizer": (
                    "one_bit_randomized_response_with_affine_expectation"
                    if task == "Sum"
                    else ""
                ),
                "sum_affine_offset": alternative,
                "sum_affine_slope": (
                    truth - alternative if task == "Sum" else ""
                ),
                "dummy_mean": dummy_mean,
                "expected_real_output_count_exact": expected_real,
                "expected_dummy_count": (
                    RELEASE_M - expected_real if task == "Sum" else ""
                ),
                "estimator": (
                    "[W-(M-E[R])*mu_nu-m*q]/(p-q)"
                    if task == "Sum"
                    else "public_dummy_corrected_GRR_histogram"
                ),
                "sum_bias": sum_bias,
                "sum_rmse": sum_rmse,
                "payload_group": f"Z_{SUM_MODULUS}" if task == "Sum" else "categorical",
                "no_wrap_envelope_checked": "true" if task == "Sum" else "not_applicable",
                "full_release": "true",
                "output_length_observed": RELEASE_M,
                "dummy_labels_visible": "false",
                "estimator_observes_real_output_count": "false",
                "final_release_permuted": "true",
                "dummy_law_public": "true",
                "estimator_uses_expected_dummy_contribution": "true",
                "shared_local_epsilon0_across_methods": "true",
                "central_epsilon_claimed": "false",
                "proxy_used_in_calibration": "false",
                "observed_mean_real_output_count": count_mean,
                "observed_real_output_count_standard_error": count_se,
                "seeds": len(SEEDS),
                "sampling_model": workload.sampling_models[task],
                "top_k": TOP_K if task == "Heavy hitter" else "",
                "range_queries": RANGE_QUERIES if task == "Range" else "",
            }
        )
    return rows


def generate_all() -> dict[str, int]:
    workloads = [synthetic_workloads(), load_adult(), load_covertype()]
    rows = []
    for workload in workloads:
        rows.extend(simulate_dataset(workload))
    filename = "fig_full_release_real_workloads.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
