"""Full-release utility after quadratic-proxy budget selection.

This experiment asks a deliberately narrower question than the exact finite
accountant: after the design proxy selects a local budget for each release,
does reinvesting Drop's stronger proxy attenuation recover the utility lost to
singleton filtering?  It does *not* claim a finite central-DP guarantee.

The experiment includes two ideal-shuffle references.  ``Pure IdealShuffle''
releases the shuffled reports at their natural length m, without padding;
``Padded Ideal'' adds matched public-prior padding to the same length M used by
Drop.  Synthetic uses its generating distribution as the public prior; Adult
and Covertype use the empirical distribution of the full public benchmark
table.  For a padded method's calibrated GRR budget, dummies are sampled from
the expected report law under that prior.  Every analyzer-facing release is
shuffled and unlabeled.  The estimator uses only public expected dummy mass
and, for Drop, the exact finite singleton-survival probability.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from chafe_eval_lib import (
    DATA_DIR,
    eps_drop_proxy,
    eps_ideal_padded,
    eps_ideal_shuffle,
    invert_eps0,
    mean_and_se,
    write_csv,
)
from full_release_real_utility import (
    DatasetWorkloads,
    load_adult,
    load_covertype,
    synthetic_workloads,
    zipf_probabilities,
)
from padded_estimators import corrected_grr_histogram


SAMPLE_M = 40_000
N_SLOTS = 40_000
RELEASE_M = 80_000
RHO = 1.0
TAU = 2.0
DELTA = 1e-6
TARGETS = (0.08, 0.16, 0.32)
SEEDS = tuple(range(100))
RANGE_QUERIES = 500
TOP_K = 50
METHODS = ("Pure IdealShuffle", "Padded Ideal", "Drop")


def grr_params(eps0: float, domain_size: int) -> tuple[float, float]:
    if eps0 < 0.0 or not math.isfinite(eps0):
        raise ValueError("eps0 must be finite and nonnegative")
    if domain_size < 2:
        raise ValueError("GRR requires at least two categories")
    denominator = math.exp(eps0) + domain_size - 1.0
    return math.exp(eps0) / denominator, 1.0 / denominator


def calibrated_eps0(method: str, target_proxy_epsilon: float) -> float:
    """Invert the labeled quadratic proxy for one release method."""
    if method == "Pure IdealShuffle":
        return invert_eps0(
            target_proxy_epsilon,
            lambda eps0: eps_ideal_shuffle(eps0, SAMPLE_M, DELTA),
        )
    if method == "Padded Ideal":
        return invert_eps0(
            target_proxy_epsilon,
            lambda eps0: eps_ideal_padded(eps0, RELEASE_M, DELTA),
        )
    if method == "Drop":
        return invert_eps0(
            target_proxy_epsilon,
            lambda eps0: eps_drop_proxy(eps0, RHO, RELEASE_M, DELTA),
        )
    raise ValueError(f"unsupported method: {method}")


def proxy_epsilon(method: str, eps0: float) -> float:
    if method == "Pure IdealShuffle":
        return eps_ideal_shuffle(eps0, SAMPLE_M, DELTA)
    if method == "Padded Ideal":
        return eps_ideal_padded(eps0, RELEASE_M, DELTA)
    if method == "Drop":
        return eps_drop_proxy(eps0, RHO, RELEASE_M, DELTA)
    raise ValueError(f"unsupported method: {method}")


def singleton_survival() -> float:
    if SAMPLE_M <= 0 or N_SLOTS <= 0:
        raise ValueError("m and N must be positive")
    return (1.0 - 1.0 / float(N_SLOTS)) ** (SAMPLE_M - 1)


def project_nonnegative_simplex(values: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = total}."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError("simplex projection expects a nonempty vector")
    if total < 0.0 or not math.isfinite(total):
        raise ValueError("simplex total must be finite and nonnegative")
    if total == 0.0:
        return np.zeros_like(vector)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(vector) + 1, dtype=float)
    active = ordered - (cumulative - total) / indices > 0.0
    if not bool(active.any()):
        raise ArithmeticError("simplex projection found no active coordinate")
    rho = int(np.nonzero(active)[0][-1])
    threshold = (cumulative[rho] - total) / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    # Remove the last few ulps of summation drift without changing support.
    mass_error = total - float(projected.sum())
    if abs(mass_error) > 1e-8 * max(1.0, total):
        raise ArithmeticError("simplex projection lost mass")
    if mass_error != 0.0:
        projected[int(np.argmax(projected))] += mass_error
    return projected


def public_prior(
    workload: DatasetWorkloads,
    task: str,
) -> tuple[np.ndarray, str]:
    """Return a fixed public raw-data prior for one benchmark/task."""
    if task not in {"Range", "Heavy hitter"}:
        raise ValueError("unsupported task")
    domain_size = (
        workload.range_domain_size
        if task == "Range"
        else workload.heavy_hitter_domain_size
    )
    if workload.synthetic:
        prior = zipf_probabilities(domain_size)
        source = "known_synthetic_modified_zipf_1.08_distribution"
    else:
        values = (
            workload.range_values
            if task == "Range"
            else workload.heavy_hitter_values
        )
        if values is None:
            raise AssertionError("real benchmark is missing its public prior column")
        counts = np.bincount(values, minlength=domain_size).astype(float)
        prior = counts / counts.sum()
        source = (
            "full_public_UCI_Adult_empirical_distribution"
            if workload.name == "Adult"
            else "full_public_UCI_Covertype_empirical_distribution"
        )
    if len(prior) != domain_size or not np.isclose(prior.sum(), 1.0):
        raise AssertionError("public prior has invalid dimension or mass")
    return np.asarray(prior, dtype=float), source


def expected_grr_report_law(prior: np.ndarray, eps0: float) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    p, q = grr_params(eps0, len(prior))
    law = q + (p - q) * prior
    law /= law.sum()
    return law


def sample_epoch(
    workload: DatasetWorkloads,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample aligned range and heavy-hitter columns for one public epoch."""
    if workload.synthetic:
        categories = rng.choice(
            workload.range_domain_size,
            size=SAMPLE_M,
            p=zipf_probabilities(workload.range_domain_size),
        ).astype(np.int64)
        return categories, categories.copy()
    if workload.range_values is None or workload.heavy_hitter_values is None:
        raise AssertionError("real benchmark is missing aligned source columns")
    source_rows = len(workload.range_values)
    replace = workload.sampling_models["Range"] == "empirical_bootstrap_with_replacement"
    indices = rng.choice(source_rows, size=SAMPLE_M, replace=replace)
    return workload.range_values[indices], workload.heavy_hitter_values[indices]


def coupled_grr(
    values: np.ndarray,
    domain_size: int,
    eps0: float,
    keep_uniforms: np.ndarray,
    alternative_draws: np.ndarray,
) -> np.ndarray:
    """Sample GRR with common random numbers across release methods."""
    raw = np.asarray(values, dtype=np.int64)
    if len(raw) != len(keep_uniforms) or len(raw) != len(alternative_draws):
        raise ValueError("coupled GRR arrays have inconsistent lengths")
    p, _ = grr_params(eps0, domain_size)
    off = np.asarray(alternative_draws, dtype=np.int64).copy()
    off += off >= raw
    return np.where(keep_uniforms < p, raw, off)


def sample_from_uniforms(law: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(np.asarray(law, dtype=float))
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, uniforms, side="right").astype(np.int64)


def complete_release(
    reports: np.ndarray,
    method: str,
    dummy_law: np.ndarray,
    placement_rng: np.random.Generator,
    dummy_rng: np.random.Generator,
    permutation_rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    """Construct one analyzer-facing shuffled, unlabeled release."""
    if len(reports) != SAMPLE_M:
        raise ValueError("report vector does not match public epoch size")
    if method == "Pure IdealShuffle":
        real = reports
        survival = 1.0
        release_length = SAMPLE_M
    elif method == "Padded Ideal":
        real = reports
        survival = 1.0
        release_length = RELEASE_M
    elif method == "Drop":
        slots = placement_rng.integers(0, N_SLOTS, size=SAMPLE_M)
        loads = np.bincount(slots, minlength=N_SLOTS)
        real = reports[loads[slots] == 1]
        survival = singleton_survival()
        release_length = RELEASE_M
    else:
        raise ValueError(f"unsupported method: {method}")
    dummy_count = release_length - len(real)
    if dummy_count < 0:
        raise AssertionError("decoded output exceeds the public release length")
    dummies = sample_from_uniforms(dummy_law, dummy_rng.random(dummy_count))
    release = np.concatenate((real, dummies)).astype(np.int64, copy=False)
    permutation_rng.shuffle(release)
    if len(release) != release_length:
        raise AssertionError("release does not have its public method-specific length")
    return release, survival, len(real)


def estimate_histogram(
    release: np.ndarray,
    domain_size: int,
    eps0: float,
    survival: float,
    dummy_law: np.ndarray,
) -> np.ndarray:
    """Estimate and project the raw histogram using public expectations only."""
    p, q = grr_params(eps0, domain_size)
    expected_real = float(SAMPLE_M) * survival
    counts = np.bincount(release, minlength=domain_size).astype(float)
    unprojected = corrected_grr_histogram(
        counts,
        epoch_size=SAMPLE_M,
        release_length=len(release),
        expected_real_outputs=expected_real,
        real_report_retention=survival,
        dummy_law=dummy_law,
        truthful_probability=p,
        alternative_probability=q,
    )
    return project_nonnegative_simplex(unprojected, float(SAMPLE_M))


def random_intervals(
    domain_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    left = rng.integers(0, domain_size, size=RANGE_QUERIES)
    right = rng.integers(0, domain_size, size=RANGE_QUERIES)
    return np.minimum(left, right), np.maximum(left, right)


def normalized_range_error(
    truth: np.ndarray,
    estimate: np.ndarray,
    intervals: tuple[np.ndarray, np.ndarray],
) -> float:
    lo, hi = intervals
    true_prefix = np.concatenate(([0.0], np.cumsum(truth)))
    estimate_prefix = np.concatenate(([0.0], np.cumsum(estimate)))
    true_answers = true_prefix[hi + 1] - true_prefix[lo]
    estimates = estimate_prefix[hi + 1] - estimate_prefix[lo]
    return float(np.mean(np.abs(estimates - true_answers)) / float(SAMPLE_M))


def top_k_f1(truth: np.ndarray, estimate: np.ndarray) -> float:
    k = min(TOP_K, len(truth))
    true_top = set(np.argsort(-truth, kind="stable")[:k].tolist())
    estimated_top = set(np.argsort(-estimate, kind="stable")[:k].tolist())
    return float(len(true_top & estimated_top) / k)


def simulate_dataset(workload: DatasetWorkloads) -> list[dict]:
    eps0_by_target_method = {
        (target, method): calibrated_eps0(method, target)
        for target in TARGETS
        for method in METHODS
    }
    metrics: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    real_counts: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    prior_metadata: dict[str, tuple[np.ndarray, str]] = {
        task: public_prior(workload, task)
        for task in ("Range", "Heavy hitter")
    }

    for seed in SEEDS:
        namespace = workload.seed_offset
        epoch_rng = np.random.default_rng(31_000_000 + namespace + seed)
        range_values, hh_values = sample_epoch(workload, epoch_rng)
        values_by_task = {"Range": range_values, "Heavy hitter": hh_values}

        for task_index, task in enumerate(("Range", "Heavy hitter")):
            values = values_by_task[task]
            domain_size = (
                workload.range_domain_size
                if task == "Range"
                else workload.heavy_hitter_domain_size
            )
            truth = np.bincount(values, minlength=domain_size).astype(float)
            prior, _ = prior_metadata[task]

            report_rng = np.random.default_rng(
                31_100_000 + 100_000 * task_index + namespace + seed
            )
            keep_uniforms = report_rng.random(SAMPLE_M)
            alternative_draws = report_rng.integers(
                0, domain_size - 1, size=SAMPLE_M
            )
            intervals = (
                random_intervals(
                    domain_size,
                    np.random.default_rng(
                        31_200_000 + 100_000 * task_index + namespace + seed
                    ),
                )
                if task == "Range"
                else None
            )

            for target in TARGETS:
                for method in METHODS:
                    eps0 = eps0_by_target_method[(target, method)]
                    reports = coupled_grr(
                        values,
                        domain_size,
                        eps0,
                        keep_uniforms,
                        alternative_draws,
                    )
                    dummy_law = expected_grr_report_law(prior, eps0)
                    # Identical seeds implement common random numbers across
                    # methods and proxy targets; Drop alone consumes placement.
                    common_seed = (
                        31_300_000 + 100_000 * task_index + namespace + seed
                    )
                    release, survival, real_count = complete_release(
                        reports,
                        method,
                        dummy_law,
                        np.random.default_rng(common_seed + 10_000_000),
                        np.random.default_rng(common_seed + 20_000_000),
                        np.random.default_rng(common_seed + 30_000_000),
                    )
                    estimate = estimate_histogram(
                        release,
                        domain_size,
                        eps0,
                        survival,
                        dummy_law,
                    )
                    if task == "Range":
                        if intervals is None:
                            raise AssertionError("range intervals were not generated")
                        value = normalized_range_error(truth, estimate, intervals)
                    else:
                        value = top_k_f1(truth, estimate)
                    metrics[(target, task, method)].append(value)
                    real_counts[(target, task, method)].append(float(real_count))

    rows: list[dict] = []
    for target in TARGETS:
        for task in ("Range", "Heavy hitter"):
            prior, prior_source = prior_metadata[task]
            paired_drop_minus_padded = np.asarray(
                metrics[(target, task, "Drop")]
            ) - np.asarray(metrics[(target, task, "Padded Ideal")])
            paired_drop_minus_padded_mean, paired_drop_minus_padded_se = mean_and_se(
                paired_drop_minus_padded
            )
            paired_drop_over_padded = np.asarray(
                metrics[(target, task, "Drop")], dtype=float
            ) / np.asarray(metrics[(target, task, "Padded Ideal")], dtype=float)
            paired_drop_over_padded_mean, paired_drop_over_padded_se = mean_and_se(
                paired_drop_over_padded
            )
            paired_drop_minus_pure = np.asarray(
                metrics[(target, task, "Drop")]
            ) - np.asarray(metrics[(target, task, "Pure IdealShuffle")])
            paired_drop_minus_pure_mean, paired_drop_minus_pure_se = mean_and_se(
                paired_drop_minus_pure
            )
            padded_ideal_mean = float(
                np.mean(metrics[(target, task, "Padded Ideal")])
            )
            pure_ideal_mean = float(
                np.mean(metrics[(target, task, "Pure IdealShuffle")])
            )
            drop_mean = float(np.mean(metrics[(target, task, "Drop")]))
            drop_samples = np.asarray(metrics[(target, task, "Drop")], dtype=float)
            padded_samples = np.asarray(
                metrics[(target, task, "Padded Ideal")], dtype=float
            )
            ratio_of_means = drop_mean / padded_ideal_mean
            ratio_influence = drop_samples - ratio_of_means * padded_samples
            ratio_of_means_se = float(
                ratio_influence.std(ddof=1)
                / math.sqrt(len(ratio_influence))
                / padded_ideal_mean
            )
            domain_size = (
                workload.range_domain_size
                if task == "Range"
                else workload.heavy_hitter_domain_size
            )
            for method in METHODS:
                eps0 = eps0_by_target_method[(target, method)]
                is_pure = method == "Pure IdealShuffle"
                is_padded = method == "Padded Ideal"
                uses_dummies = not is_pure
                output_length = SAMPLE_M if is_pure else RELEASE_M
                method_tau = 1.0 if is_pure else TAU
                if not math.isclose(
                    float(output_length), method_tau * float(SAMPLE_M)
                ):
                    raise AssertionError("release length and tau are inconsistent")
                survival = (
                    singleton_survival() if method == "Drop" else 1.0
                )
                mean, standard_error = mean_and_se(
                    metrics[(target, task, method)]
                )
                real_mean, real_standard_error = mean_and_se(
                    real_counts[(target, task, method)]
                )
                rows.append(
                    {
                    "dataset": workload.name,
                    "task": task,
                    "method": method,
                    "decoder": (
                        "all_reports_no_padding"
                        if is_pure
                        else (
                            "all_reports_same_M_padding"
                            if is_padded
                            else "drop"
                        )
                    ),
                    "value": mean,
                    "standard_error": standard_error,
                    "paired_drop_minus_padded_ideal": paired_drop_minus_padded_mean,
                    "paired_drop_minus_padded_ideal_standard_error": paired_drop_minus_padded_se,
                    # Backward-compatible alias for the former padded-only field.
                    "paired_difference_standard_error": paired_drop_minus_padded_se,
                    "drop_over_padded_ideal_ratio": ratio_of_means,
                    "drop_over_padded_ideal_ratio_standard_error": ratio_of_means_se,
                    "paired_drop_over_padded_ideal_ratio": paired_drop_over_padded_mean,
                    "paired_drop_over_padded_ideal_ratio_standard_error": paired_drop_over_padded_se,
                    "paired_drop_minus_pure_ideal_shuffle": paired_drop_minus_pure_mean,
                    "paired_drop_minus_pure_ideal_shuffle_standard_error": paired_drop_minus_pure_se,
                    "drop_over_pure_ideal_shuffle_ratio": drop_mean / pure_ideal_mean,
                    "higher_metric_is_better": str(task == "Heavy hitter").lower(),
                    "metric": (
                        "mean_normalized_absolute_range_error_over_m"
                        if task == "Range"
                        else "top_50_f1"
                    ),
                    "source_type": "monte_carlo_complete_release",
                    "privacy_target_type": (
                        "pure_ideal_shuffle_quadratic_proxy"
                        if is_pure
                        else "matched_public_prior_quadratic_proxy"
                    ),
                    "target_tilde_epsilon_prime": target,
                    "achieved_tilde_epsilon_prime": proxy_epsilon(method, eps0),
                    "local_epsilon0": eps0,
                    "delta": DELTA,
                    "central_epsilon_claimed": "false",
                    "mechanism_wide_dp_claimed": "false",
                    "exact_accountant_used": "false",
                    "proxy_used_in_calibration": "true",
                    "privacy_scope": "quadratic_design_proxy_not_finite_DP_guarantee",
                    "public_prior_source": prior_source,
                    "public_prior_fixed_before_epoch": "true",
                    "public_prior_used_for_padding": str(uses_dummies).lower(),
                    "public_prior_domain_size": len(prior),
                    "public_prior_min_probability": float(prior.min()),
                    "public_prior_max_probability": float(prior.max()),
                    "dummy_law": (
                        "none"
                        if is_pure
                        else "expected_GRR_report_law_under_fixed_public_prior"
                    ),
                    "dummy_law_matches_proxy_reference": (
                        "not_applicable" if is_pure else "true"
                    ),
                    "uses_dummies": str(uses_dummies).lower(),
                    "dummy_count_fixed": str(method != "Drop").lower(),
                    "expected_dummy_count": output_length - SAMPLE_M * survival,
                    "observed_mean_dummy_count": output_length - real_mean,
                    "observed_dummy_count_standard_error": real_standard_error,
                    "randomizer": "generalized_randomized_response",
                    "histogram_postprocessing": "euclidean_nonnegative_simplex_sum_m",
                    "m": SAMPLE_M,
                    "N": N_SLOTS if method == "Drop" else "",
                    "M": output_length,
                    "rho": RHO if method == "Drop" else "",
                    "tau": method_tau,
                    "singleton_survival_exact_finite": survival,
                    "observed_mean_real_output_count": real_mean,
                    "observed_real_output_count_standard_error": real_standard_error,
                    "full_release": "true",
                    "output_length_observed": output_length,
                    "release_length_rule": (
                        "m_no_padding" if is_pure else "M_equals_tau_times_m"
                    ),
                    "dummy_labels_visible": "false",
                    "real_output_count_visible": str(method != "Drop").lower(),
                    "estimator_observes_real_output_count": str(method != "Drop").lower(),
                    "estimator_uses_expected_singleton_survival": str(method == "Drop").lower(),
                    "estimator_uses_expected_dummy_contribution": str(uses_dummies).lower(),
                    "final_release_permuted": "true",
                    "paired_workload_seeds": "true",
                    "paired_query_seeds": "true",
                    "paired_GRR_randomness": "true",
                    "paired_across_proxy_targets": "true",
                    "proxy_targets_evaluated": "|".join(
                        str(value) for value in TARGETS
                    ),
                    "seeds": len(SEEDS),
                    "range_queries": RANGE_QUERIES if task == "Range" else "",
                    "top_k": TOP_K if task == "Heavy hitter" else "",
                    "dataset_source": workload.dataset_source,
                    "dataset_rows_total": (
                        workload.dataset_rows_total
                        if workload.dataset_rows_total is not None
                        else ""
                    ),
                    "sampling_model": workload.sampling_models[task],
                    "provenance": (
                        (
                            "no padding; complete pure ideal-shuffle length-m release; "
                            if is_pure
                            else "fixed public prior; method-specific matched GRR padding; "
                            "complete shuffled unlabeled length-M release; "
                        )
                        + "public-expectation correction; projected histogram; "
                        "proxy-selected local budget; no finite central-DP claim"
                    ),
                    }
                )
    return rows


def generate_all() -> dict[str, int]:
    rows: list[dict] = []
    for workload in (synthetic_workloads(), load_adult(), load_covertype()):
        rows.extend(simulate_dataset(workload))
    filename = "fig_full_release_proxy_utility.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    for filename, count in generate_all().items():
        print(f"{filename}: {count} rows")
