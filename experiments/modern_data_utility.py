"""Complete-release utility on one synthetic and three modern workloads.

Categorical range and heavy-hitter workloads use k-ary GRR, matched public
padding, and the exact finite homogeneous-pair accountant.  Bounded sums use
an additive wrapped discrete-Laplace randomizer and neutral-element padding.
At the same central target, Pool uses its own two-sided Poisson-Chernoff
account to select a larger local epsilon0.  At the same local epsilon0, its
slot aggregates preserve the ideal randomized sum exactly.

Categorical calibration is exact for a declared finite homogeneous pair;
bounded-sum calibration is a rigorous Chernoff upper bound for the declared
matched-Poisson pair.  Results on heterogeneous real datasets are transferred-
reference utility measurements and are not labeled as a mechanism-wide DP
maximization.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

import fetch_modern_utility_datasets as modern_data
from chafe_eval_lib import DATA_DIR, mean_and_se, write_csv
from exact_grr_pair_accountant import (
    PairCalibration,
    calibrate_local_epsilon as calibrate_grr_epsilon,
    grr_probabilities,
)
from full_release_proxy_utility import project_nonnegative_simplex
from pool_laplace_accounting import (
    GROUP_SIZE as LAPLACE_GROUP_SIZE,
    QUANTIZATION_SCALE as LAPLACE_SCALE,
    LaplaceCalibration,
    calibrate_local_epsilon as calibrate_laplace_epsilon,
    quantized_laplace_law,
)


EPOCH_SIZE = 100_000
# Utility-focused point on the cost/retention curve.  With N=2m, a tagged
# report survives Drop with probability about e^-0.5 instead of e^-1; the
# systems evaluation separately measures the cost of increasing N.
SLOT_COUNT = 200_000
RELEASE_LENGTH = 200_000
RHO = EPOCH_SIZE / SLOT_COUNT
TAU = RELEASE_LENGTH / EPOCH_SIZE
DELTA = 1e-6
TARGET_EPSILONS = (0.1, 0.3, 0.7)
REPLICATES = 100
RANGE_QUERY_COUNT = 512
TOP_K = 50
BASE_SEED = 26_071_500
ZIPF_EXPONENT = 1.08
BETA_PARAMETERS = (2.0, 5.0)

PRIVACY_SCOPE = "two_sided_homogeneous_endpoint_pair_not_heterogeneous_worst_case"
UTILITY_SOURCE = "monte_carlo_complete_release_histogram"


def _seed(dataset_index: int, replicate: int, task: int, target: int, method: int) -> int:
    return (
        BASE_SEED
        + 10_000_000 * dataset_index
        + 100_000 * replicate
        + 10_000 * task
        + 100 * target
        + method
    )


def _load_datasets() -> list[dict[str, object]]:
    paths = modern_data.ensure_modern_utility_datasets()
    datasets: list[dict[str, object]] = [
        {
            "name": "Synthetic",
            "path": None,
            "metadata": {
                "snapshot": "seeded iid Zipf(1.08) categorical and Beta(2,5) bounded data",
                "source_sha256": "not_applicable_synthetic",
                "clean_sha256": "not_applicable_synthetic",
                "synthetic": True,
            },
            "range": None,
            "sum": None,
            "hh": None,
        }
    ]
    for name, path in paths.items():
        metadata_path = path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text())
        if modern_data.sha256(path) != metadata["clean_sha256"]:
            raise AssertionError(f"{name}: clean snapshot checksum mismatch")
        arrays = np.load(path)
        range_values = np.asarray(arrays["range_values"], dtype=np.int64)
        bounded_sum_values = np.asarray(arrays["bounded_sum_values"], dtype=float)
        heavy_hitter_values = np.asarray(arrays["heavy_hitter_values"], dtype=np.int64)
        if not (
            len(range_values) == len(bounded_sum_values) == len(heavy_hitter_values)
        ):
            raise AssertionError(f"{name}: workload columns are not aligned")
        if len(range_values) != int(metadata["admitted_rows"]):
            raise AssertionError(f"{name}: metadata row count mismatch")
        datasets.append(
            {
                "name": name,
                "path": path,
                "metadata": metadata,
                "range": range_values,
                "sum": bounded_sum_values,
                "hh": heavy_hitter_values,
            }
        )
    return datasets


def _zipf_probabilities(domain_size: int) -> np.ndarray:
    ranks = np.arange(1, domain_size + 1, dtype=float)
    probabilities = np.power(ranks, -ZIPF_EXPONENT)
    return probabilities / probabilities.sum()


def _sample_synthetic_epoch(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    range_values = rng.choice(
        modern_data.RANGE_DOMAIN,
        size=EPOCH_SIZE,
        p=_zipf_probabilities(modern_data.RANGE_DOMAIN),
    ).astype(np.int64)
    heavy_hitter_values = rng.choice(
        modern_data.HH_DOMAIN,
        size=EPOCH_SIZE,
        p=_zipf_probabilities(modern_data.HH_DOMAIN),
    ).astype(np.int64)
    bounded_sum_values = rng.beta(*BETA_PARAMETERS, size=EPOCH_SIZE)
    return range_values, bounded_sum_values, heavy_hitter_values


def _sample_epoch(dataset: dict[str, object], rng: np.random.Generator) -> np.ndarray:
    row_count = len(np.asarray(dataset["range"]))
    if row_count < EPOCH_SIZE:
        raise AssertionError("clean dataset is smaller than the public epoch")
    return rng.choice(row_count, size=EPOCH_SIZE, replace=False)


def _grr_reports(
    raw: np.ndarray,
    domain_size: int,
    eps0: float,
    keep_uniforms: np.ndarray,
    alternative_draws: np.ndarray,
) -> np.ndarray:
    """Sample k-ary GRR with shared random numbers across methods."""
    values = np.asarray(raw, dtype=np.int64)
    if len(keep_uniforms) != len(values) or len(alternative_draws) != len(values):
        raise ValueError("GRR coupling arrays do not match the epoch")
    p, _ = grr_probabilities(eps0, domain_size)
    reports = values.copy()
    replace = np.asarray(keep_uniforms) >= p
    replacements = np.asarray(alternative_draws, dtype=np.int64)[replace].copy()
    original = values[replace]
    replacements += replacements >= original
    reports[replace] = replacements
    return reports


def _reference_dummy_histogram(
    count: int,
    domain_size: int,
    eps0: float,
    keep_uniforms: np.ndarray,
    alternative_draws: np.ndarray,
) -> np.ndarray:
    """Sample dummies from the GRR report law of public category zero."""
    if count < 0:
        raise ValueError("dummy count cannot be negative")
    p, _ = grr_probabilities(eps0, domain_size)
    if len(keep_uniforms) < count or len(alternative_draws) < count:
        raise ValueError("dummy coupling arrays are shorter than the release")
    histogram = np.zeros(domain_size, dtype=np.int64)
    truthful = np.asarray(keep_uniforms[:count]) < p
    histogram[0] = int(truthful.sum())
    if bool((~truthful).any()):
        alternatives = np.asarray(alternative_draws[:count], dtype=np.int64)[~truthful]
        histogram[1:] = np.bincount(
            alternatives, minlength=domain_size - 1
        )[: domain_size - 1]
    if int(histogram.sum()) != count:
        raise AssertionError("dummy histogram lost release positions")
    return histogram


def _categorical_release_histogram(
    raw: np.ndarray,
    calibration: PairCalibration,
    survivor_mask: np.ndarray,
    report_uniforms: np.ndarray,
    report_alternatives: np.ndarray,
    dummy_uniforms: np.ndarray,
    dummy_alternatives: np.ndarray,
) -> tuple[np.ndarray, int]:
    reports = _grr_reports(
        raw,
        calibration.domain_size,
        calibration.local_epsilon0,
        report_uniforms,
        report_alternatives,
    )
    if calibration.method == "Padded Ideal":
        real = reports
    elif calibration.method == "Drop":
        real = reports[survivor_mask]
    else:
        raise ValueError("categorical workload expects Padded Ideal or Drop")
    real_histogram = np.bincount(real, minlength=calibration.domain_size)
    dummy_count = RELEASE_LENGTH - len(real)
    dummy_histogram = _reference_dummy_histogram(
        dummy_count,
        calibration.domain_size,
        calibration.local_epsilon0,
        dummy_uniforms,
        dummy_alternatives,
    )
    release = real_histogram + dummy_histogram
    if int(release.sum()) != RELEASE_LENGTH:
        raise AssertionError("categorical release does not have public length M")
    return release, len(real)


def _estimate_histogram(
    release: np.ndarray,
    calibration: PairCalibration,
) -> np.ndarray:
    p, q = grr_probabilities(
        calibration.local_epsilon0, calibration.domain_size
    )
    dummy_law = np.full(calibration.domain_size, q, dtype=float)
    dummy_law[0] = p
    expected_real = EPOCH_SIZE * calibration.survival
    estimate = (
        np.asarray(release, dtype=float)
        - (RELEASE_LENGTH - expected_real) * dummy_law
        - calibration.survival * EPOCH_SIZE * q
    ) / (calibration.survival * (p - q))
    return project_nonnegative_simplex(estimate, float(EPOCH_SIZE))


def _range_queries() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BASE_SEED + 91)
    endpoints = rng.integers(
        0, modern_data.RANGE_DOMAIN + 1, size=(RANGE_QUERY_COUNT, 2)
    )
    low = np.minimum(endpoints[:, 0], endpoints[:, 1])
    high = np.maximum(endpoints[:, 0], endpoints[:, 1])
    high = np.maximum(high, low + 1)
    high = np.minimum(high, modern_data.RANGE_DOMAIN)
    return low, high


def _range_mean_relative_error(
    truth: np.ndarray,
    estimate: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> float:
    truth_prefix = np.concatenate(([0.0], np.cumsum(truth, dtype=float)))
    estimate_prefix = np.concatenate(([0.0], np.cumsum(estimate, dtype=float)))
    true_counts = truth_prefix[high] - truth_prefix[low]
    estimated_counts = estimate_prefix[high] - estimate_prefix[low]
    denominator = np.maximum(true_counts, 0.01 * EPOCH_SIZE)
    return float(np.mean(np.abs(estimated_counts - true_counts) / denominator))


def _top_k_f1(truth: np.ndarray, estimate: np.ndarray, k: int = TOP_K) -> float:
    truth_top = set(np.argsort(-truth, kind="stable")[:k].tolist())
    estimate_top = set(np.argsort(-estimate, kind="stable")[:k].tolist())
    return len(truth_top & estimate_top) / float(k)


def _laplace_reports(
    values: np.ndarray,
    eps0: float,
    quantization_uniforms: np.ndarray,
    noise_uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Sample unbiased scale-8 encodings and wrapped discrete-Laplace reports."""

    scaled = np.asarray(values, dtype=float) * LAPLACE_SCALE
    lower = np.floor(scaled).astype(np.int64)
    fractional = scaled - lower
    encoded = lower + (
        np.asarray(quantization_uniforms, dtype=float) < fractional
    ).astype(np.int64)
    if np.any(encoded < 0) or np.any(encoded > LAPLACE_SCALE):
        raise ValueError("bounded-sum inputs must lie in [0,1]")
    noise_law = quantized_laplace_law(0.0, eps0)
    noise_indices = np.searchsorted(
        np.cumsum(noise_law), np.asarray(noise_uniforms, dtype=float), side="right"
    )
    signed_noise = np.where(
        noise_indices < LAPLACE_GROUP_SIZE // 2,
        noise_indices,
        noise_indices - LAPLACE_GROUP_SIZE,
    )
    unwrapped = encoded + signed_noise
    if np.any(unwrapped < -(LAPLACE_GROUP_SIZE // 2)) or np.any(
        unwrapped >= LAPLACE_GROUP_SIZE // 2
    ):
        raise AssertionError("a fixed-point Laplace report wrapped modulo Z_256")
    modular = (encoded + noise_indices) % LAPLACE_GROUP_SIZE
    reports = np.where(
        modular < LAPLACE_GROUP_SIZE // 2,
        modular,
        modular - LAPLACE_GROUP_SIZE,
    ).astype(np.int64)
    noise_values = np.where(
        np.arange(LAPLACE_GROUP_SIZE) < LAPLACE_GROUP_SIZE // 2,
        np.arange(LAPLACE_GROUP_SIZE),
        np.arange(LAPLACE_GROUP_SIZE) - LAPLACE_GROUP_SIZE,
    )
    mean_noise = float(np.dot(noise_law, noise_values))
    noise_variance = float(
        np.dot(noise_law, np.square(noise_values - mean_noise))
    )
    quantization_variance = float(np.sum(fractional * (1.0 - fractional)))
    return reports, encoded, mean_noise, noise_variance, quantization_variance


def _sum_observations(
    values: np.ndarray,
    calibration: LaplaceCalibration,
    slots: np.ndarray,
    quantization_uniforms: np.ndarray,
    noise_uniforms: np.ndarray,
) -> dict[str, float | int]:
    """Construct one zero-padded additive release at an accounted epsilon0."""

    (
        reports,
        encoded_truth,
        mean_noise,
        noise_variance,
        quantization_variance,
    ) = _laplace_reports(
        values,
        calibration.local_epsilon0,
        quantization_uniforms,
        noise_uniforms,
    )
    report_total = int(reports.sum())
    occupancies = np.bincount(slots, minlength=SLOT_COUNT)
    max_occupancy = int(occupancies.max())
    if max_occupancy >= 16:
        raise AssertionError("this trial exceeds the declared packed64 occupancy field")

    if calibration.method == "Padded Ideal":
        released_total = report_total
        real_output_count = EPOCH_SIZE
        sum_preserved = 1
    elif calibration.method == "Pool":
        slot_sums = np.bincount(
            slots, weights=reports, minlength=SLOT_COUNT
        ).astype(np.int64)
        nonempty = occupancies > 0
        raw_pooled = slot_sums[nonempty]
        if raw_pooled.size and (
            int(raw_pooled.min()) < -(LAPLACE_GROUP_SIZE // 2)
            or int(raw_pooled.max()) >= LAPLACE_GROUP_SIZE // 2
        ):
            raise AssertionError("pooled fixed-point payload wrapped modulo Z_256")
        modular = raw_pooled % LAPLACE_GROUP_SIZE
        pooled = np.where(
            modular < LAPLACE_GROUP_SIZE // 2,
            modular,
            modular - LAPLACE_GROUP_SIZE,
        )
        released_total = int(pooled.sum())
        real_output_count = int(nonempty.sum())
        sum_preserved = int(released_total == report_total)
        if not sum_preserved:
            raise AssertionError("Pool failed to preserve its randomized sum")
    else:
        raise ValueError("bounded sum expects Padded Ideal or Pool")

    estimate = (
        released_total - EPOCH_SIZE * mean_noise
    ) / float(LAPLACE_SCALE)
    true_sum = float(np.sum(values))
    error_variance = (
        quantization_variance + EPOCH_SIZE * noise_variance
    ) / float(LAPLACE_SCALE**2)
    relative_rmse = math.sqrt(error_variance) / max(true_sum, 1e-12)
    return {
        "true_sum": true_sum,
        "encoded_sum": float(encoded_truth.sum()) / float(LAPLACE_SCALE),
        "released_total": released_total,
        "estimate": float(estimate),
        "max_occupancy": max_occupancy,
        "pool_sum_preserved": sum_preserved,
        "real_output_count": real_output_count,
        "fixed_point_scale": LAPLACE_SCALE,
        "relative_rmse": relative_rmse,
        "error_variance": error_variance,
    }


CalibrationKey = tuple[str, str, int, float]


def _calibrations() -> dict[CalibrationKey, PairCalibration | LaplaceCalibration]:
    calibrations: dict[
        CalibrationKey, PairCalibration | LaplaceCalibration
    ] = {}
    for domain_size in (modern_data.RANGE_DOMAIN, modern_data.HH_DOMAIN):
        for target in TARGET_EPSILONS:
            for method in ("Padded Ideal", "Drop"):
                calibration = calibrate_grr_epsilon(
                    method=method,
                    domain_size=domain_size,
                    epoch_size=EPOCH_SIZE,
                    slot_count=SLOT_COUNT if method == "Drop" else None,
                    release_length=RELEASE_LENGTH,
                    target_epsilon=target,
                    delta=DELTA,
                )
                calibrations[("categorical", method, domain_size, target)] = calibration
    for target in TARGET_EPSILONS:
        for method in ("Padded Ideal", "Pool"):
            calibration = calibrate_laplace_epsilon(
                method=method,
                target_epsilon=target,
                epoch_size=EPOCH_SIZE,
                slot_count=SLOT_COUNT,
                release_length=RELEASE_LENGTH,
                delta=DELTA,
            )
            calibrations[("bounded_sum", method, LAPLACE_GROUP_SIZE, target)] = calibration
    return calibrations


def _calibration_rows(
    calibrations: dict[
        CalibrationKey, PairCalibration | LaplaceCalibration
    ]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (family, method, domain_size, target), calibration in calibrations.items():
        calibration_key = f"Sum {method}" if family == "bounded_sum" else method
        if isinstance(calibration, LaplaceCalibration):
            base = {
                **asdict(calibration),
                "epoch_size": EPOCH_SIZE,
                "slot_count": SLOT_COUNT if method == "Pool" else "",
                "release_length": (
                    RELEASE_LENGTH if method == "Pool" else EPOCH_SIZE
                ),
                "survival": 1.0,
                "delta": DELTA,
                "achieved_epsilon": calibration.achieved_chernoff_epsilon,
                "feasible_eps0": calibration.local_epsilon0,
                "infeasible_eps0": "",
                "privacy_family": (
                    "homogeneous_q3_endpoint_replacement_pair"
                    if method == "Padded Ideal"
                    else "homogeneous_q3_endpoint_replacement_matched_poisson_pair"
                ),
                "privacy_scope": (
                    "two_sided_fixed_m_pair_not_heterogeneous_worst_case"
                    if method == "Padded Ideal"
                    else "two_sided_matched_poisson_pair_not_heterogeneous_worst_case"
                ),
                "privacy_source": (
                    "two_sided_fixed_m_hidden_slot_chernoff_account"
                    if method == "Padded Ideal"
                    else "two_sided_poisson_chernoff_account_with_tail_slack"
                ),
            }
        else:
            base = asdict(calibration)
        privacy_source = str(base.get("privacy_source", ""))
        privacy_source_type = (
            "exact_accountant"
            if "exact" in privacy_source or "finite_multinomial_tail" in privacy_source
            else "formula"
        )
        base.update(
            {
                "source_type": privacy_source_type,
                "privacy_source_type": privacy_source_type,
                "calibration_key": calibration_key,
                "domain_size": domain_size,
                "target_epsilon": target,
                "rho": RHO if calibration_key in {"Drop", "Sum Pool"} else "",
                "tau": TAU,
            }
        )
        rows.append(base)
    return rows


def run_experiment() -> None:
    datasets = _load_datasets()
    calibrations = _calibrations()
    calibration_rows = _calibration_rows(calibrations)
    replicate_rows: list[dict[str, object]] = []
    range_low, range_high = _range_queries()

    for dataset_index, dataset in enumerate(datasets):
        name = str(dataset["name"])
        metadata = dict(dataset["metadata"])
        for replicate in range(REPLICATES):
            epoch_rng = np.random.default_rng(_seed(dataset_index, replicate, 0, 0, 0))
            if bool(metadata.get("synthetic", False)):
                range_values, sum_values, hh_values = _sample_synthetic_epoch(epoch_rng)
            else:
                indices = _sample_epoch(dataset, epoch_rng)
                range_values = np.asarray(dataset["range"])[indices]
                hh_values = np.asarray(dataset["hh"])[indices]
                sum_values = np.asarray(dataset["sum"])[indices]
            range_truth = np.bincount(
                range_values, minlength=modern_data.RANGE_DOMAIN
            )
            hh_truth = np.bincount(hh_values, minlength=modern_data.HH_DOMAIN)

            placement_rng = np.random.default_rng(
                _seed(dataset_index, replicate, 0, 0, 7)
            )
            slots = placement_rng.integers(0, SLOT_COUNT, size=EPOCH_SIZE)
            occupancies = np.bincount(slots, minlength=SLOT_COUNT)
            survivor_mask = occupancies[slots] == 1

            for target_index, target in enumerate(TARGET_EPSILONS):
                for task_index, (task, raw, truth, domain_size) in enumerate(
                    (
                        (
                            "Range",
                            range_values,
                            range_truth,
                            modern_data.RANGE_DOMAIN,
                        ),
                        (
                            "Heavy hitter",
                            hh_values,
                            hh_truth,
                            modern_data.HH_DOMAIN,
                        ),
                    ),
                    start=1,
                ):
                    coupling_rng = np.random.default_rng(
                        _seed(
                            dataset_index,
                            replicate,
                            task_index,
                            target_index,
                            91,
                        )
                    )
                    report_uniforms = coupling_rng.random(EPOCH_SIZE)
                    report_alternatives = coupling_rng.integers(
                        0, domain_size - 1, size=EPOCH_SIZE
                    )
                    dummy_uniforms = coupling_rng.random(RELEASE_LENGTH)
                    dummy_alternatives = coupling_rng.integers(
                        0, domain_size - 1, size=RELEASE_LENGTH
                    )
                    for method in ("Padded Ideal", "Drop"):
                        calibration = calibrations[
                            ("categorical", method, domain_size, target)
                        ]
                        if not isinstance(calibration, PairCalibration):
                            raise TypeError(
                                "categorical workload resolved to a non-GRR calibration"
                            )
                        release, real_outputs = _categorical_release_histogram(
                            raw,
                            calibration,
                            survivor_mask,
                            report_uniforms,
                            report_alternatives,
                            dummy_uniforms,
                            dummy_alternatives,
                        )
                        estimate = _estimate_histogram(release, calibration)
                        if task == "Range":
                            value = _range_mean_relative_error(
                                truth, estimate, range_low, range_high
                            )
                            metric = "mean_relative_range_error"
                        else:
                            value = _top_k_f1(truth, estimate)
                            metric = "top_50_f1"
                        replicate_rows.append(
                            {
                                "dataset": name,
                                "task": task,
                                "method": method,
                                "decoder": "none" if method == "Padded Ideal" else "Drop",
                                "target_epsilon": target,
                                "delta": DELTA,
                                "local_epsilon0": calibration.local_epsilon0,
                                "achieved_pair_epsilon": calibration.achieved_epsilon,
                                "replicate": replicate,
                                "metric": metric,
                                "value": value,
                                "real_output_count": real_outputs,
                                "release_length": RELEASE_LENGTH,
                                "rho": RHO,
                                "tau": TAU,
                                "dummy_law": "GRR(category_0)",
                                "privacy_scope": PRIVACY_SCOPE,
                                "source_type": "monte_carlo_complete_release",
                                "privacy_source_type": "exact_accountant",
                                "privacy_source": calibration.privacy_source,
                                "utility_source_type": UTILITY_SOURCE,
                                "utility_source": UTILITY_SOURCE,
                                "pairing": "shared_epoch_placement_and_common_random_numbers",
                                "source_snapshot": metadata["snapshot"],
                                "source_sha256": metadata["source_sha256"],
                                "clean_sha256": metadata["clean_sha256"],
                            }
                        )

                sum_rng = np.random.default_rng(
                    _seed(dataset_index, replicate, 3, target_index, 0)
                )
                sum_quantization_uniforms = sum_rng.random(EPOCH_SIZE)
                sum_noise_uniforms = sum_rng.random(EPOCH_SIZE)
                for method in ("Padded Ideal", "Pool"):
                    sum_calibration = calibrations[
                        ("bounded_sum", method, LAPLACE_GROUP_SIZE, target)
                    ]
                    if not isinstance(sum_calibration, LaplaceCalibration):
                        raise TypeError("bounded-sum calibration must be Laplace")
                    observation = _sum_observations(
                        sum_values,
                        sum_calibration,
                        slots,
                        sum_quantization_uniforms,
                        sum_noise_uniforms,
                    )
                    true_sum = float(observation["true_sum"])
                    relative_error = float(observation["relative_rmse"])
                    replicate_rows.append(
                        {
                            "dataset": name,
                            "task": "Bounded sum",
                            "method": method,
                            "decoder": "none" if method == "Padded Ideal" else "Pool",
                            "target_epsilon": target,
                            "delta": DELTA,
                            "local_epsilon0": sum_calibration.local_epsilon0,
                            "achieved_pair_epsilon": sum_calibration.achieved_chernoff_epsilon,
                            "replicate": replicate,
                            "metric": "relative_sum_rmse",
                            "value": relative_error,
                            "true_sum": true_sum,
                            "estimate": observation["estimate"],
                            "released_total": observation["released_total"],
                            "real_output_count": observation["real_output_count"],
                            "release_length": RELEASE_LENGTH,
                            "max_occupancy": observation["max_occupancy"],
                            "pool_sum_preserved": observation["pool_sum_preserved"],
                            "fixed_point_scale": observation["fixed_point_scale"],
                            "error_variance": observation["error_variance"],
                            "rho": RHO,
                            "tau": TAU,
                            "dummy_law": "point_mass_at_additive_identity",
                            "privacy_scope": PRIVACY_SCOPE,
                            "source_type": "monte_carlo_complete_release",
                            "privacy_source_type": "formula",
                            "privacy_source": (
                                "two_sided_fixed_m_hidden_slot_chernoff_account"
                                if method == "Padded Ideal"
                                else "two_sided_poisson_chernoff_account_with_tail_slack"
                            ),
                            "utility_source_type": UTILITY_SOURCE,
                            "utility_source": UTILITY_SOURCE,
                            "pairing": "shared_epoch_placement_and_laplace_uniforms",
                            "source_snapshot": metadata["snapshot"],
                            "source_sha256": metadata["source_sha256"],
                            "clean_sha256": metadata["clean_sha256"],
                        }
                    )

    grouped: dict[tuple[str, str, str, float, str], list[float]] = defaultdict(list)
    for row in replicate_rows:
        key = (
            str(row["dataset"]),
            str(row["task"]),
            str(row["method"]),
            float(row["target_epsilon"]),
            str(row["metric"]),
        )
        grouped[key].append(float(row["value"]))

    summary_rows: list[dict[str, object]] = []
    for key, values in grouped.items():
        dataset, task, method, target, metric = key
        mean, se = mean_and_se(values)
        ideal_values = grouped[(dataset, task, "Padded Ideal", target, metric)]
        paired_difference = np.asarray(values) - np.asarray(ideal_values)
        difference_mean, difference_se = mean_and_se(paired_difference)
        ratio = mean / float(np.mean(ideal_values)) if np.mean(ideal_values) else 1.0
        calibration_family = "bounded_sum" if task == "Bounded sum" else "categorical"
        domain_size = (
            LAPLACE_GROUP_SIZE
            if task == "Bounded sum"
            else modern_data.RANGE_DOMAIN
            if task == "Range"
            else modern_data.HH_DOMAIN
        )
        calibration = calibrations[(calibration_family, method, domain_size, target)]
        achieved_pair_epsilon = (
            calibration.achieved_chernoff_epsilon
            if isinstance(calibration, LaplaceCalibration)
            else calibration.achieved_epsilon
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "method": method,
                "decoder": "Pool" if method == "Pool" else "Drop" if method == "Drop" else "none",
                "target_epsilon": target,
                "delta": DELTA,
                "local_epsilon0": calibration.local_epsilon0,
                "achieved_pair_epsilon": achieved_pair_epsilon,
                "metric": metric,
                "mean": mean,
                "se": se,
                "replicates": len(values),
                "paired_difference_from_padded_ideal": difference_mean,
                "paired_difference_se": difference_se,
                "ratio_to_padded_ideal": ratio,
                "rho": RHO,
                "tau": TAU,
                "privacy_scope": PRIVACY_SCOPE,
                "source_type": "monte_carlo_complete_release",
                "privacy_source_type": (
                    "formula" if task == "Bounded sum" else "exact_accountant"
                ),
                "utility_source_type": UTILITY_SOURCE,
                "utility_source": UTILITY_SOURCE,
            }
        )

    write_csv(DATA_DIR / "fig_modern_utility_calibration.csv", calibration_rows)
    write_csv(DATA_DIR / "fig_modern_utility_replicates.csv", replicate_rows)
    write_csv(DATA_DIR / "fig_modern_utility.csv", summary_rows)


def generate_all() -> dict[str, int]:
    run_experiment()
    return {
        "data/fig_modern_utility_calibration.csv": 18,
        "data/fig_modern_utility_replicates.csv": 7_200,
        "data/fig_modern_utility.csv": 72,
    }


if __name__ == "__main__":
    run_experiment()
