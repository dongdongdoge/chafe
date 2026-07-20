from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from chafe_eval_lib import (
    DATA_DIR,
    eps_drop_proxy,
    eps_pool_replacement_generic,
    invert_eps0,
    mean_and_se,
    write_csv,
)
from padded_estimators import corrected_affine_total


# This experiment is intentionally smaller than the workload sweeps.  Its role
# is to exercise the complete analyzer-facing release: exactly M values, no
# real/dummy labels, and an estimator that uses only public mechanism metadata.
SAMPLE_M = 20_000
RHO = 1.0
TAU = 2.0
DELTA = 1e-6
SEEDS = list(range(100))
FIXED_EPS0 = 1.0
MATCHED_PROXY_TARGET = 0.02

DROP_THETA = 0.30
DROP_PRIOR_THETA = 0.80
POOL_MATCHED_BETA = (2.0, 5.0)
POOL_PRIOR_BETA = (5.0, 2.0)

PRIVACY_MODES = ["fixed_eps0", "matched_proxy_selected_eps0"]
DUMMY_LAWS = ["matched_reference", "neutral_public", "mismatched_prior"]


def release_length() -> int:
    return int(math.ceil(TAU * SAMPLE_M))


def slot_count() -> int:
    return int(math.ceil(SAMPLE_M / RHO))


def singleton_occupancy_stats(n: int, N: int) -> tuple[float, float, float]:
    """Return exact item survival, E[number of singletons], and its variance."""
    s = (1.0 - 1.0 / N) ** (n - 1)
    if n < 2:
        joint = 0.0
    else:
        joint = (1.0 - 1.0 / N) * (1.0 - 2.0 / N) ** (n - 2)
    variance = n * s * (1.0 - s) + n * (n - 1) * (joint - s * s)
    return s, n * s, max(0.0, variance)


def occupied_slot_stats(n: int, N: int) -> tuple[float, float, float]:
    """Return exact bin nonempty probability, E[occupied bins], and variance."""
    empty = (1.0 - 1.0 / N) ** n
    p_nonempty = 1.0 - empty
    both_empty = (1.0 - 2.0 / N) ** n
    variance = (
        N * p_nonempty * (1.0 - p_nonempty)
        + N * (N - 1) * (both_empty - empty * empty)
    )
    return p_nonempty, N * p_nonempty, max(0.0, variance)


def brr_affine(eps0: float) -> tuple[float, float]:
    """Return a,b such that E[Y|x]=a+b*x for binary randomized response."""
    p_truth = math.exp(eps0) / (math.exp(eps0) + 1.0)
    a = 1.0 - p_truth
    b = 2.0 * p_truth - 1.0
    return a, b


def eps0_for(decoder: str, privacy_mode: str) -> float:
    if privacy_mode == "fixed_eps0":
        return FIXED_EPS0
    M = release_length()
    if decoder == "drop":
        return invert_eps0(
            MATCHED_PROXY_TARGET,
            lambda eps0: eps_drop_proxy(eps0, RHO, M, DELTA),
        )
    if decoder == "pool":
        return invert_eps0(
            MATCHED_PROXY_TARGET,
            lambda eps0: eps_pool_replacement_generic(eps0, M, DELTA),
        )
    raise ValueError(decoder)


def proxy_epsilon(decoder: str, eps0: float) -> float:
    if decoder == "drop":
        return eps_drop_proxy(eps0, RHO, release_length(), DELTA)
    if decoder == "pool":
        return eps_pool_replacement_generic(eps0, release_length(), DELTA)
    raise ValueError(decoder)


def conditional_binomial_positive(
    rng: np.random.Generator,
    n: int,
    p: float,
    size: int,
) -> np.ndarray:
    """Sample Binomial(n,p) conditioned on being positive without approximation."""
    counts = rng.binomial(n, p, size=size)
    zero = counts == 0
    while bool(zero.any()):
        counts[zero] = rng.binomial(n, p, size=int(zero.sum()))
        zero = counts == 0
    return counts


def beta_moments(alpha: float, beta: float) -> tuple[float, float]:
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    return mean, variance


def pooled_dummy_values(
    rng: np.random.Generator,
    law: str,
    count: int,
    eps0: float,
    p_nonempty: float,
) -> tuple[np.ndarray, float, float, str]:
    """Sample a public Pool dummy law and return samples plus exact moments."""
    if law == "neutral_public":
        variance = 2.0 / (eps0 * eps0)
        return (
            rng.laplace(0.0, 1.0 / eps0, size=count),
            0.0,
            variance,
            "zero_mean_laplace_public_law",
        )

    if law == "matched_reference":
        alpha, beta = POOL_MATCHED_BETA
        source = "oracle_population_G_plus_reference"
    elif law == "mismatched_prior":
        alpha, beta = POOL_PRIOR_BETA
        source = "public_beta_5_2_prior_G_plus"
    else:
        raise ValueError(law)

    N = slot_count()
    occupancy = conditional_binomial_positive(rng, SAMPLE_M, 1.0 / N, count)
    total_reports = int(occupancy.sum())
    report_values = rng.beta(alpha, beta, size=total_reports)
    report_values += rng.laplace(0.0, 1.0 / eps0, size=total_reports)
    starts = np.concatenate(([0], np.cumsum(occupancy[:-1])))
    samples = np.add.reduceat(report_values, starts)

    value_mean, value_variance = beta_moments(alpha, beta)
    binomial_mean = SAMPLE_M / N
    binomial_variance = SAMPLE_M * (1.0 / N) * (1.0 - 1.0 / N)
    conditional_count_mean = binomial_mean / p_nonempty
    conditional_count_second_moment = (
        binomial_variance + binomial_mean * binomial_mean
    ) / p_nonempty
    conditional_count_variance = (
        conditional_count_second_moment - conditional_count_mean**2
    )
    report_variance = value_variance + 2.0 / (eps0 * eps0)
    dummy_mean = conditional_count_mean * value_mean
    dummy_variance = (
        conditional_count_mean * report_variance
        + conditional_count_variance * value_mean * value_mean
    )
    return samples, dummy_mean, dummy_variance, source


def rmse_and_se(errors: np.ndarray) -> tuple[float, float]:
    squares = np.square(errors)
    rmse = float(math.sqrt(float(squares.mean())))
    if len(errors) < 2 or rmse == 0.0:
        return rmse, 0.0
    square_se = float(squares.std(ddof=1) / math.sqrt(len(squares)))
    return rmse, square_se / (2.0 * rmse)


def add_trial(
    trials: dict[tuple[str, str, str], list[dict[str, float]]],
    decoder: str,
    privacy_mode: str,
    dummy_law: str,
    estimate: float,
    target: float,
    real_output_count: int,
    dummy_count: int,
) -> None:
    error = estimate - target
    trials[(decoder, privacy_mode, dummy_law)].append(
        {
            "estimate": estimate,
            "target": target,
            "error": error,
            "relative_error": error / max(abs(target), 1e-12),
            "real_output_count": float(real_output_count),
            "dummy_count": float(dummy_count),
        }
    )


def simulate_trials() -> tuple[
    dict[tuple[str, str, str], list[dict[str, float]]],
    dict[tuple[str, str, str], dict[str, float | str]],
]:
    trials: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    law_metadata: dict[tuple[str, str, str], dict[str, float | str]] = {}
    n = SAMPLE_M
    N = slot_count()
    M = release_length()
    singleton_s, _, _ = singleton_occupancy_stats(n, N)
    p_nonempty, expected_pool_outputs, _ = occupied_slot_stats(n, N)

    for seed in SEEDS:
        data_rng = np.random.default_rng(2_100_000 + seed)
        bits = data_rng.binomial(1, DROP_THETA, size=n)
        true_count = float(bits.sum())
        drop_slots = data_rng.integers(0, N, size=n)
        drop_loads = np.bincount(drop_slots, minlength=N)
        drop_kept = drop_loads[drop_slots] == 1

        values = data_rng.beta(*POOL_MATCHED_BETA, size=n)
        true_sum = float(values.sum())
        pool_slots = data_rng.integers(0, N, size=n)
        pool_loads = np.bincount(pool_slots, minlength=N)
        pool_occupied = pool_loads > 0

        for mode_index, privacy_mode in enumerate(PRIVACY_MODES):
            drop_eps0 = eps0_for("drop", privacy_mode)
            a, b = brr_affine(drop_eps0)
            report_rng = np.random.default_rng(2_200_000 + 10_000 * mode_index + seed)
            report_one_prob = a + b * bits
            drop_reports = (report_rng.random(n) < report_one_prob).astype(float)
            drop_real = drop_reports[drop_kept]
            drop_real_count = len(drop_real)
            drop_dummy_count = M - drop_real_count

            for law_index, dummy_law in enumerate(DUMMY_LAWS):
                if dummy_law == "matched_reference":
                    q = a + b * DROP_THETA
                    law_source = "oracle_population_BRR_marginal_reference"
                elif dummy_law == "neutral_public":
                    q = 0.5
                    law_source = "public_uniform_bernoulli"
                elif dummy_law == "mismatched_prior":
                    q = a + b * DROP_PRIOR_THETA
                    law_source = "public_theta_0_8_BRR_prior"
                else:
                    raise ValueError(dummy_law)
                dummy_rng = np.random.default_rng(
                    2_300_000 + 100_000 * mode_index + 10_000 * law_index + seed
                )
                dummies = dummy_rng.binomial(1, q, size=drop_dummy_count).astype(float)
                release = np.concatenate((drop_real, dummies))
                dummy_rng.shuffle(release)
                if len(release) != M:
                    raise AssertionError("Drop release length is not M")
                released_ones = float(release.sum())
                estimate = corrected_affine_total(
                    released_ones,
                    epoch_size=n,
                    release_length=M,
                    expected_real_outputs=singleton_s * n,
                    real_report_retention=singleton_s,
                    dummy_mean=q,
                    affine_offset=a,
                    affine_slope=b,
                )
                add_trial(
                    trials,
                    "drop",
                    privacy_mode,
                    dummy_law,
                    estimate,
                    true_count,
                    drop_real_count,
                    drop_dummy_count,
                )
                law_metadata[("drop", privacy_mode, dummy_law)] = {
                    "dummy_mean": q,
                    "dummy_variance": q * (1.0 - q),
                    "dummy_law_source": law_source,
                    "dummy_law_parameter": q,
                }

            pool_eps0 = eps0_for("pool", privacy_mode)
            pool_report_rng = np.random.default_rng(
                2_600_000 + 10_000 * mode_index + seed
            )
            pool_reports = values + pool_report_rng.laplace(
                0.0, 1.0 / pool_eps0, size=n
            )
            pool_slot_sums = np.bincount(
                pool_slots, weights=pool_reports, minlength=N
            )
            pool_real = pool_slot_sums[pool_occupied]
            pool_real_count = len(pool_real)
            pool_dummy_count = M - pool_real_count

            for law_index, dummy_law in enumerate(DUMMY_LAWS):
                dummy_rng = np.random.default_rng(
                    2_700_000 + 100_000 * mode_index + 10_000 * law_index + seed
                )
                dummies, dummy_mean, dummy_variance, law_source = pooled_dummy_values(
                    dummy_rng,
                    dummy_law,
                    pool_dummy_count,
                    pool_eps0,
                    p_nonempty,
                )
                release = np.concatenate((pool_real, dummies))
                dummy_rng.shuffle(release)
                if len(release) != M:
                    raise AssertionError("Pool release length is not M")
                released_sum = float(release.sum())
                estimate = corrected_affine_total(
                    released_sum,
                    epoch_size=n,
                    release_length=M,
                    expected_real_outputs=expected_pool_outputs,
                    real_report_retention=1.0,
                    dummy_mean=dummy_mean,
                    affine_offset=0.0,
                    affine_slope=1.0,
                )
                add_trial(
                    trials,
                    "pool",
                    privacy_mode,
                    dummy_law,
                    estimate,
                    true_sum,
                    pool_real_count,
                    pool_dummy_count,
                )
                law_metadata[("pool", privacy_mode, dummy_law)] = {
                    "dummy_mean": dummy_mean,
                    "dummy_variance": dummy_variance,
                    "dummy_law_source": law_source,
                    "dummy_law_parameter": dummy_mean,
                }

    return trials, law_metadata


def aggregate_rows() -> list[dict]:
    trials, law_metadata = simulate_trials()
    n = SAMPLE_M
    N = slot_count()
    M = release_length()
    singleton_s, expected_singletons, singleton_variance = singleton_occupancy_stats(n, N)
    p_nonempty, expected_occupied, occupied_variance = occupied_slot_stats(n, N)
    dummy_labels = {
        "matched_reference": "Matched reference",
        "neutral_public": "Neutral public",
        "mismatched_prior": "Mismatched prior",
    }
    method_labels = {
        "fixed_eps0": "Fixed local eps0",
        "matched_proxy_selected_eps0": "Matched-proxy-selected eps0",
    }
    rows: list[dict] = []

    for decoder in ["drop", "pool"]:
        for privacy_mode in PRIVACY_MODES:
            eps0 = eps0_for(decoder, privacy_mode)
            for dummy_law in DUMMY_LAWS:
                sample = trials[(decoder, privacy_mode, dummy_law)]
                errors = np.asarray([row["error"] for row in sample], dtype=float)
                relative_errors = np.asarray(
                    [row["relative_error"] for row in sample], dtype=float
                )
                estimates = np.asarray([row["estimate"] for row in sample], dtype=float)
                targets = np.asarray([row["target"] for row in sample], dtype=float)
                real_counts = np.asarray(
                    [row["real_output_count"] for row in sample], dtype=float
                )
                dummy_counts = np.asarray(
                    [row["dummy_count"] for row in sample], dtype=float
                )
                bias, bias_se = mean_and_se(errors)
                relative_bias, relative_bias_se = mean_and_se(relative_errors)
                rmse, rmse_se = rmse_and_se(errors)
                relative_rmse, relative_rmse_se = rmse_and_se(relative_errors)
                mean_real_count, real_count_se = mean_and_se(real_counts)
                mean_dummy_count, dummy_count_se = mean_and_se(dummy_counts)
                if decoder == "drop":
                    exact_probability = singleton_s
                    exact_real_count = expected_singletons
                    exact_real_count_variance = singleton_variance
                    estimator = "[W-qM-s*n*(a-q)]/(s*b)"
                    workload = "binary_count_full_release"
                    randomizer = "binary_randomized_response"
                else:
                    exact_probability = p_nonempty
                    exact_real_count = expected_occupied
                    exact_real_count_variance = occupied_variance
                    estimator = "W-mu_nu*(M-E[K_occupied])"
                    workload = "additive_sum_full_release"
                    randomizer = "bounded_value_plus_laplace"
                metadata = law_metadata[(decoder, privacy_mode, dummy_law)]
                rows.append(
                    {
                        "decoder": decoder,
                        "panel": decoder.capitalize(),
                        "method": method_labels[privacy_mode],
                        "privacy_mode": privacy_mode,
                        "dummy_law": dummy_law,
                        "dummy_law_label": dummy_labels[dummy_law],
                        "dummy_law_source": metadata["dummy_law_source"],
                        "dummy_mean": metadata["dummy_mean"],
                        "dummy_variance": metadata["dummy_variance"],
                        "dummy_law_parameter": metadata["dummy_law_parameter"],
                        "matched_reference_is_oracle": str(
                            dummy_law == "matched_reference"
                        ).lower(),
                        "bias": bias,
                        "bias_standard_error": bias_se,
                        "relative_bias": relative_bias,
                        "relative_bias_standard_error": relative_bias_se,
                        "rmse": rmse,
                        "rmse_standard_error": rmse_se,
                        "relative_rmse": relative_rmse,
                        "relative_rmse_standard_error": relative_rmse_se,
                        "mean_absolute_error": float(np.abs(errors).mean()),
                        "mean_true_target": float(targets.mean()),
                        "mean_estimate": float(estimates.mean()),
                        "observed_mean_real_output_count": mean_real_count,
                        "observed_real_output_count_standard_error": real_count_se,
                        "observed_mean_dummy_count": mean_dummy_count,
                        "observed_dummy_count_standard_error": dummy_count_se,
                        "real_output_probability_exact": exact_probability,
                        "expected_real_output_count_exact": exact_real_count,
                        "real_output_count_variance_exact": exact_real_count_variance,
                        "source_type": "monte_carlo",
                        "m": n,
                        "N": N,
                        "rho": n / N,
                        "M": M,
                        "tau": M / n,
                        "eps0": eps0,
                        "epsilon_prime_proxy": proxy_epsilon(decoder, eps0),
                        "proxy_target": (
                            MATCHED_PROXY_TARGET
                            if privacy_mode == "matched_proxy_selected_eps0"
                            else ""
                        ),
                        "proxy_scope": "matched_law_second_order_selector_only",
                        "proxy_applies_to_row_law": str(
                            dummy_law == "matched_reference"
                        ).lower(),
                        "delta": DELTA,
                        "seeds": len(SEEDS),
                        "workload": workload,
                        "randomizer": randomizer,
                        "estimator": estimator,
                        "full_release": "true",
                        "output_length_observed": M,
                        "dummy_labels_visible": "false",
                        "estimator_observes_real_output_count": "false",
                        "estimator_uses_expected_dummy_mass": "true",
                        "public_epoch_size_n_required": str(
                            decoder == "drop" or abs(float(metadata["dummy_mean"])) > 1e-15
                        ).lower(),
                        "occupancy_model": "finite_multinomial_exact_m_balls_N_bins",
                    }
                )
    return rows


def generate_all() -> dict[str, int]:
    rows = aggregate_rows()
    filename = "fig_full_release_dummy_mismatch.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
