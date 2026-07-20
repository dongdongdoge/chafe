"""Validate task-aware correction on complete, unlabeled padded releases."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from chafe_eval_lib import DATA_DIR, mean_and_se, write_csv
from padded_estimators import (
    corrected_affine_total,
    drop_survival,
    expected_pool_outputs,
)


SAMPLE_M = 20_000
N_SLOTS = 20_000
TAU_GRID = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
EPS0 = 1.0
SEEDS = tuple(range(100))
DROP_THETA = 0.30
POOL_BETA = (2.0, 5.0)
PUBLIC_DUMMY_INPUT_MEAN = 0.50

TASK_DROP = "Drop count"
TASK_POOL = "Pool sum"
ESTIMATOR_NAIVE = "Naive"
ESTIMATOR_CORRECTED = "Corrected"


def brr_affine(eps0: float) -> tuple[float, float]:
    truthful = math.exp(eps0) / (math.exp(eps0) + 1.0)
    alternative = 1.0 - truthful
    return alternative, truthful - alternative


def conditional_positive_binomial(
    rng: np.random.Generator,
    trials: int,
    probability: float,
    size: int,
) -> np.ndarray:
    if size == 0:
        return np.empty(0, dtype=np.int64)
    values = rng.binomial(trials, probability, size=size)
    zero = values == 0
    while bool(zero.any()):
        values[zero] = rng.binomial(trials, probability, size=int(zero.sum()))
        zero = values == 0
    return values


def rmse_and_se(errors: np.ndarray) -> tuple[float, float]:
    squares = np.square(errors)
    rmse = math.sqrt(float(squares.mean()))
    if len(errors) < 2 or rmse == 0.0:
        return rmse, 0.0
    square_se = float(squares.std(ddof=1) / math.sqrt(len(squares)))
    return rmse, square_se / (2.0 * rmse)


def simulate() -> list[dict]:
    if N_SLOTS != SAMPLE_M:
        raise ValueError("the published sweep fixes rho=1")
    offset, slope = brr_affine(EPS0)
    survival = drop_survival(SAMPLE_M, N_SLOTS)
    expected_drop = SAMPLE_M * survival
    expected_pool = expected_pool_outputs(SAMPLE_M, N_SLOTS)
    nonempty_probability = expected_pool / N_SLOTS
    conditional_pool_occupancy = (SAMPLE_M / N_SLOTS) / nonempty_probability
    drop_dummy_mean = offset + slope * PUBLIC_DUMMY_INPUT_MEAN
    pool_dummy_mean = conditional_pool_occupancy * drop_dummy_mean

    relative_errors: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    real_counts: dict[tuple[str, float], list[float]] = defaultdict(list)

    for seed in SEEDS:
        data_rng = np.random.default_rng(51_000_000 + seed)
        bits = data_rng.binomial(1, DROP_THETA, size=SAMPLE_M)
        values = data_rng.beta(*POOL_BETA, size=SAMPLE_M)
        true_count = float(bits.sum())
        true_sum = float(values.sum())

        report_rng = np.random.default_rng(51_100_000 + seed)
        drop_reports = (
            report_rng.random(SAMPLE_M) < (offset + slope * bits)
        ).astype(np.int64)
        pool_reports = (
            report_rng.random(SAMPLE_M) < (offset + slope * values)
        ).astype(np.int64)

        placement_rng = np.random.default_rng(51_200_000 + seed)
        drop_slots = placement_rng.integers(0, N_SLOTS, size=SAMPLE_M)
        drop_loads = np.bincount(drop_slots, minlength=N_SLOTS)
        drop_real = drop_reports[drop_loads[drop_slots] == 1]

        pool_slots = placement_rng.integers(0, N_SLOTS, size=SAMPLE_M)
        pool_loads = np.bincount(pool_slots, minlength=N_SLOTS)
        pool_real = np.bincount(
            pool_slots,
            weights=pool_reports,
            minlength=N_SLOTS,
        )[pool_loads > 0].astype(np.int64)

        for tau_index, tau in enumerate(TAU_GRID):
            M = int(math.ceil(tau * SAMPLE_M))
            if M < max(len(drop_real), len(pool_real)):
                raise AssertionError("release length is smaller than decoded output")

            drop_dummy_count = M - len(drop_real)
            drop_dummy_rng = np.random.default_rng(
                51_300_000 + 100_000 * tau_index + seed
            )
            drop_dummies = drop_dummy_rng.binomial(
                1,
                drop_dummy_mean,
                size=drop_dummy_count,
            )
            drop_total = float(drop_real.sum() + drop_dummies.sum())
            drop_estimates = {
                ESTIMATOR_NAIVE: (
                    drop_total - survival * offset * SAMPLE_M
                ) / (survival * slope),
                ESTIMATOR_CORRECTED: corrected_affine_total(
                    drop_total,
                    epoch_size=SAMPLE_M,
                    release_length=M,
                    expected_real_outputs=expected_drop,
                    real_report_retention=survival,
                    dummy_mean=drop_dummy_mean,
                    affine_offset=offset,
                    affine_slope=slope,
                ),
            }

            pool_dummy_count = M - len(pool_real)
            pool_dummy_rng = np.random.default_rng(
                51_400_000 + 100_000 * tau_index + seed
            )
            pool_dummy_occupancies = conditional_positive_binomial(
                pool_dummy_rng,
                SAMPLE_M,
                1.0 / N_SLOTS,
                pool_dummy_count,
            )
            pool_dummies = pool_dummy_rng.binomial(
                pool_dummy_occupancies,
                drop_dummy_mean,
            )
            pool_total = float(pool_real.sum() + pool_dummies.sum())
            pool_estimates = {
                ESTIMATOR_NAIVE: (pool_total - offset * SAMPLE_M) / slope,
                ESTIMATOR_CORRECTED: corrected_affine_total(
                    pool_total,
                    epoch_size=SAMPLE_M,
                    release_length=M,
                    expected_real_outputs=expected_pool,
                    real_report_retention=1.0,
                    dummy_mean=pool_dummy_mean,
                    affine_offset=offset,
                    affine_slope=slope,
                ),
            }

            for estimator, estimate in drop_estimates.items():
                relative_errors[(TASK_DROP, tau, estimator)].append(
                    (estimate - true_count) / true_count
                )
            for estimator, estimate in pool_estimates.items():
                relative_errors[(TASK_POOL, tau, estimator)].append(
                    (estimate - true_sum) / true_sum
                )
            real_counts[(TASK_DROP, tau)].append(float(len(drop_real)))
            real_counts[(TASK_POOL, tau)].append(float(len(pool_real)))

    rows: list[dict] = []
    expected_targets = {
        TASK_DROP: SAMPLE_M * DROP_THETA,
        TASK_POOL: SAMPLE_M * POOL_BETA[0] / sum(POOL_BETA),
    }
    expected_outputs = {TASK_DROP: expected_drop, TASK_POOL: expected_pool}
    dummy_means = {TASK_DROP: drop_dummy_mean, TASK_POOL: pool_dummy_mean}
    retentions = {TASK_DROP: survival, TASK_POOL: 1.0}

    for task in (TASK_DROP, TASK_POOL):
        for tau in TAU_GRID:
            M = int(math.ceil(tau * SAMPLE_M))
            observed_real_mean, observed_real_se = mean_and_se(real_counts[(task, tau)])
            for estimator in (ESTIMATOR_NAIVE, ESTIMATOR_CORRECTED):
                errors = np.asarray(relative_errors[(task, tau, estimator)], dtype=float)
                bias, bias_se = mean_and_se(errors)
                rmse, rmse_se = rmse_and_se(errors)
                naive_bias = (
                    (M - expected_outputs[task]) * dummy_means[task]
                    / (retentions[task] * slope * expected_targets[task])
                )
                rows.append(
                    {
                        "task": task,
                        "estimator": estimator,
                        "tau": tau,
                        "relative_bias": bias,
                        "relative_bias_standard_error": bias_se,
                        "relative_rmse": rmse,
                        "relative_rmse_standard_error": rmse_se,
                        "relative_mae": float(np.abs(errors).mean()),
                        "theory_relative_bias": (
                            0.0 if estimator == ESTIMATOR_CORRECTED else naive_bias
                        ),
                        "source_type": "monte_carlo_complete_release",
                        "m": SAMPLE_M,
                        "N": N_SLOTS,
                        "rho": SAMPLE_M / N_SLOTS,
                        "M": M,
                        "eps0": EPS0,
                        "seeds": len(SEEDS),
                        "dummy_input_mean": PUBLIC_DUMMY_INPUT_MEAN,
                        "dummy_payload_mean": dummy_means[task],
                        "dummy_law": (
                            "public_bernoulli_half"
                            if task == TASK_DROP
                            else "public_nonempty_pool_of_bernoulli_half_reports"
                        ),
                        "expected_real_output_count_exact": expected_outputs[task],
                        "observed_mean_real_output_count": observed_real_mean,
                        "observed_real_output_count_standard_error": observed_real_se,
                        "full_release": "true",
                        "output_length_observed": M,
                        "dummy_labels_visible": "false",
                        "estimator_observes_real_output_count": "false",
                        "estimator_uses_expected_dummy_contribution": str(
                            estimator == ESTIMATOR_CORRECTED
                        ).lower(),
                        "randomizer": "one_bit_randomized_response_with_affine_expectation",
                        "affine_offset": offset,
                        "affine_slope": slope,
                        "provenance": (
                            "finite multinomial placement; nonzero public dummy law; "
                            "complete permuted length-M release; analyzer uses no labels"
                        ),
                    }
                )
    return rows


def generate_all() -> dict[str, int]:
    rows = simulate()
    filename = "fig_padding_correction_utility.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
