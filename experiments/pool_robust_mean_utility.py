from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from chafe_eval_lib import DATA_DIR, mean_and_se, write_csv
from padded_estimators import corrected_affine_total, expected_pool_outputs, public_epoch_mean


# This experiment asks whether keeping anonymous slot-level partial aggregates
# enables an analysis that a single secure sum cannot perform.  Every method
# receives the same locally randomized reports; privacy is therefore held at a
# fixed local eps0.  The comparison is a utility experiment, not a claim that
# the releases have equal central-DP budgets.
SAMPLE_M = 20_000
EPS0_GRID = [4.0, 8.0]
RHO_GRID = [2.0, 4.0, 8.0]
CONTAMINATION_GRID = [0.0, 0.02, 0.05, 0.10]
CLEAN_CENTERS = [0.30, 0.50, 0.70]
SEEDS = list(range(60))

# The padded ideal release contains a small, explicit amount of public
# padding.  Pool uses M=N because at most N slots are nonempty.
IDEAL_TAU = 1.01
PUBLIC_DUMMY_CENTER = 0.50
HUBER_TUNING = 0.75
REPORT_CLIP = 8.0
REPORT_SCALE = 4096
# Centered fixed-point payloads use the packed64 backend's 56-bit field.
REPORT_MODULUS = 1 << 56

METHOD_SECURE = "Single secure aggregate (mean)"
METHOD_POOL_LINEAR = "Pool complete release (linear)"
METHOD_POOL_ROBUST = "Pool partial aggregates (Huber)"
METHOD_IDEAL_ROBUST = "Padded ideal reports (Huber)"
METHOD_DUMMY_ONLY = "Public-dummy-only control"
METHODS = [
    METHOD_SECURE,
    METHOD_POOL_LINEAR,
    METHOD_POOL_ROBUST,
    METHOD_IDEAL_ROBUST,
    METHOD_DUMMY_ONLY,
]


def huber_location(values: np.ndarray, tuning: float = HUBER_TUNING) -> float:
    """Robust location computed only from the released, unlabeled values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Huber location requires a nonempty vector")
    if tuning <= 0.0:
        raise ValueError("Huber tuning must be positive")

    location = float(np.median(values))
    mad = float(np.median(np.abs(values - location)))
    scale = max(1.4826 * mad, 1e-9)
    cutoff = tuning * scale
    for _ in range(50):
        residual = values - location
        absolute = np.abs(residual)
        weights = np.ones_like(values)
        mask = absolute > cutoff
        weights[mask] = cutoff / absolute[mask]
        updated = float(np.dot(weights, values) / weights.sum())
        if abs(updated - location) <= 1e-10 * max(1.0, abs(location)):
            return updated
        location = updated
    return location


def bounded_heavy_tail(
    rng: np.random.Generator,
    center: float,
    size: int,
) -> np.ndarray:
    """A symmetric, bounded t_2 workload whose mean equals its center."""
    if not 0.20 <= center <= 0.80:
        raise ValueError("center must leave room for the symmetric support")
    offsets = np.clip(0.035 * rng.standard_t(df=2.0, size=size), -0.20, 0.20)
    return center + offsets


def contaminated_epoch(
    rng: np.random.Generator,
    center: float,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, float, str]:
    """Return contaminated records and the clean-sample mean target."""
    clean = bounded_heavy_tail(rng, center, SAMPLE_M)
    clean_target = float(clean.mean())
    count = int(round(fraction * SAMPLE_M))
    if center < 0.5 or (center == 0.5 and seed % 2 == 0):
        endpoint = 1.0
        direction = "upper_endpoint"
    else:
        endpoint = 0.0
        direction = "lower_endpoint"
    observed = clean.copy()
    if count:
        indices = rng.choice(SAMPLE_M, size=count, replace=False)
        observed[indices] = endpoint
    return observed, clean_target, direction


def local_randomize(
    rng: np.random.Generator,
    values: np.ndarray,
    eps0: float,
) -> np.ndarray:
    """Pure-eps0 LDP identity release for inputs in [0,1]."""
    if eps0 <= 0.0:
        raise ValueError("eps0 must be positive")
    noisy = values + rng.laplace(0.0, 1.0 / eps0, size=len(values))
    encoded = np.rint(np.clip(noisy, -REPORT_CLIP, REPORT_CLIP) * REPORT_SCALE)
    return encoded.astype(np.int64).astype(float) / REPORT_SCALE


def conditional_positive_binomial(
    rng: np.random.Generator,
    trials: int,
    probability: float,
    size: int,
) -> np.ndarray:
    """Sample Binomial(trials, probability) conditioned on being positive."""
    if size == 0:
        return np.empty(0, dtype=np.int64)
    result = rng.binomial(trials, probability, size=size)
    zero = result == 0
    while bool(zero.any()):
        result[zero] = rng.binomial(trials, probability, size=int(zero.sum()))
        zero = result == 0
    return result


def public_pool_dummies(
    rng: np.random.Generator,
    count: int,
    N: int,
    eps0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate public (count,sum) dummies with support overlapping real pools."""
    occupancies = conditional_positive_binomial(rng, SAMPLE_M, 1.0 / N, count)
    if count == 0:
        return occupancies, np.empty(0, dtype=float)
    report_count = int(occupancies.sum())
    reports = local_randomize(
        rng,
        np.full(report_count, PUBLIC_DUMMY_CENTER, dtype=float),
        eps0,
    )
    starts = np.concatenate(([0], np.cumsum(occupancies[:-1])))
    sums = np.add.reduceat(reports, starts)
    return occupancies, sums


def complete_pool_release(
    rng: np.random.Generator,
    reports: np.ndarray,
    rho: float,
    eps0: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return a permuted length-N release without real/dummy labels."""
    N = int(round(SAMPLE_M / rho))
    if N <= 0 or abs(SAMPLE_M / N - rho) > 1e-12:
        raise ValueError("rho must divide SAMPLE_M on this experiment grid")
    slots = rng.integers(0, N, size=SAMPLE_M)
    real_counts_all = np.bincount(slots, minlength=N)
    real_sums_all = np.bincount(slots, weights=reports, minlength=N)
    occupied = real_counts_all > 0
    real_counts = real_counts_all[occupied].astype(np.int64)
    real_sums = real_sums_all[occupied]
    dummy_count = N - len(real_counts)
    dummy_counts, dummy_sums = public_pool_dummies(
        rng, dummy_count, N, eps0
    )
    counts = np.concatenate((real_counts, dummy_counts))
    sums = np.concatenate((real_sums, dummy_sums))
    order = rng.permutation(N)
    counts = counts[order]
    sums = sums[order]
    if len(counts) != N or bool((counts <= 0).any()):
        raise AssertionError("Pool must release N positive-count pairs")
    return counts, sums, {
        "N": float(N),
        "M": float(N),
        "real_output_count": float(len(real_counts)),
        "dummy_count": float(dummy_count),
    }


def complete_padded_ideal_release(
    rng: np.random.Generator,
    reports: np.ndarray,
    eps0: float,
) -> tuple[np.ndarray, int]:
    """Return permuted reports plus public dummies, with no dummy labels."""
    M = int(math.ceil(IDEAL_TAU * SAMPLE_M))
    dummy_count = M - SAMPLE_M
    dummies = PUBLIC_DUMMY_CENTER + rng.laplace(
        0.0, 1.0 / eps0, size=dummy_count
    )
    release = np.concatenate((reports, dummies))
    rng.shuffle(release)
    if len(release) != M:
        raise AssertionError("ideal release length is not M")
    return release, dummy_count


def pool_linear_estimate(counts: np.ndarray, sums: np.ndarray, rho: float) -> float:
    """Collapse the complete Pool release using only public expected dummy mass."""
    N = len(counts)
    empty_probability = (1.0 - 1.0 / N) ** SAMPLE_M
    nonempty_probability = 1.0 - empty_probability
    expected_conditional_occupancy = rho / nonempty_probability
    corrected_sum = corrected_affine_total(
        float(sums.sum()),
        epoch_size=SAMPLE_M,
        release_length=N,
        expected_real_outputs=expected_pool_outputs(SAMPLE_M, N),
        real_report_retention=1.0,
        dummy_mean=expected_conditional_occupancy * PUBLIC_DUMMY_CENTER,
        affine_offset=0.0,
        affine_slope=1.0,
    )
    return public_epoch_mean(corrected_sum, SAMPLE_M)


def dummy_only_control(
    rng: np.random.Generator,
    N: int,
    eps0: float,
) -> float:
    counts, sums = public_pool_dummies(rng, N, N, eps0)
    order = rng.permutation(N)
    return huber_location(sums[order] / counts[order])


def _rmse_and_se(errors: list[float]) -> tuple[float, float]:
    squares = np.square(np.asarray(errors, dtype=float))
    rmse = math.sqrt(float(squares.mean()))
    if len(errors) < 2 or rmse == 0.0:
        return rmse, 0.0
    square_se = float(squares.std(ddof=1) / math.sqrt(len(squares)))
    return rmse, square_se / (2.0 * rmse)


def simulate() -> list[dict]:
    if SAMPLE_M * REPORT_CLIP * REPORT_SCALE >= REPORT_MODULUS // 2:
        raise AssertionError("declared centered modular envelope can wrap")
    errors: dict[tuple[float, float, float, str], list[float]] = defaultdict(list)
    absolute_errors: dict[
        tuple[float, float, float, str], list[float]
    ] = defaultdict(list)
    real_counts: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    dummy_counts: dict[tuple[float, float, float], list[float]] = defaultdict(list)

    for seed in SEEDS:
        for center_index, center in enumerate(CLEAN_CENTERS):
            for contamination_index, contamination in enumerate(CONTAMINATION_GRID):
                data_rng = np.random.default_rng(
                    8_100_000
                    + 100_000 * center_index
                    + 10_000 * contamination_index
                    + seed
                )
                values, target, _ = contaminated_epoch(
                    data_rng, center, contamination, seed
                )
                for eps_index, eps0 in enumerate(EPS0_GRID):
                    report_rng = np.random.default_rng(
                        8_300_000
                        + 1_000_000 * eps_index
                        + 100_000 * center_index
                        + 10_000 * contamination_index
                        + seed
                    )
                    reports = local_randomize(report_rng, values, eps0)
                    secure_estimate = float(reports.mean())
                    ideal_rng = np.random.default_rng(
                        8_500_000
                        + 1_000_000 * eps_index
                        + 100_000 * center_index
                        + 10_000 * contamination_index
                        + seed
                    )
                    ideal_release, _ = complete_padded_ideal_release(
                        ideal_rng, reports, eps0
                    )
                    ideal_estimate = huber_location(ideal_release)

                    for rho_index, rho in enumerate(RHO_GRID):
                        key_base = (eps0, rho, contamination)
                        pool_rng = np.random.default_rng(
                            8_700_000
                            + 10_000_000 * eps_index
                            + 1_000_000 * rho_index
                            + 100_000 * center_index
                            + 10_000 * contamination_index
                            + seed
                        )
                        counts, sums, metadata = complete_pool_release(
                            pool_rng, reports, rho, eps0
                        )
                        pool_linear = pool_linear_estimate(counts, sums, rho)
                        pool_robust = huber_location(sums / counts)
                        control_rng = np.random.default_rng(
                            8_900_000
                            + 10_000_000 * eps_index
                            + 1_000_000 * rho_index
                            + 100_000 * center_index
                            + 10_000 * contamination_index
                            + seed
                        )
                        control = dummy_only_control(
                            control_rng, int(metadata["N"]), eps0
                        )
                        estimates = {
                            METHOD_SECURE: secure_estimate,
                            METHOD_POOL_LINEAR: pool_linear,
                            METHOD_POOL_ROBUST: pool_robust,
                            METHOD_IDEAL_ROBUST: ideal_estimate,
                            METHOD_DUMMY_ONLY: control,
                        }
                        for method, estimate in estimates.items():
                            error = estimate - target
                            errors[key_base + (method,)].append(error)
                            absolute_errors[key_base + (method,)].append(abs(error))
                        real_counts[key_base].append(metadata["real_output_count"])
                        dummy_counts[key_base].append(metadata["dummy_count"])

    rows: list[dict] = []
    trials_per_cell = len(SEEDS) * len(CLEAN_CENTERS)
    for eps0 in EPS0_GRID:
        for rho in RHO_GRID:
            N = int(round(SAMPLE_M / rho))
            M_pool = N
            M_ideal = int(math.ceil(IDEAL_TAU * SAMPLE_M))
            expected_empty_fraction = (1.0 - 1.0 / N) ** SAMPLE_M
            for contamination in CONTAMINATION_GRID:
                key_base = (eps0, rho, contamination)
                mean_real, se_real = mean_and_se(real_counts[key_base])
                mean_dummy, se_dummy = mean_and_se(dummy_counts[key_base])
                for method in METHODS:
                    sample_errors = errors[key_base + (method,)]
                    sample_absolute = absolute_errors[key_base + (method,)]
                    bias, bias_se = mean_and_se(sample_errors)
                    mae, mae_se = mean_and_se(sample_absolute)
                    rmse, rmse_se = _rmse_and_se(sample_errors)
                    p95 = float(np.quantile(sample_absolute, 0.95))
                    rows.append(
                        {
                            "method": method,
                            "eps0": eps0,
                            "rho": rho,
                            "contamination_fraction": contamination,
                            "mean_absolute_error": mae,
                            "mean_absolute_error_standard_error": mae_se,
                            "rmse": rmse,
                            "rmse_standard_error": rmse_se,
                            "bias": bias,
                            "bias_standard_error": bias_se,
                            "p95_absolute_error": p95,
                            "source_type": "monte_carlo_fixed_eps0",
                            "privacy_comparison_scope": "same_pure_eps0_local_randomizer_not_equal_central_budgets",
                            "randomizer": "bounded_identity_plus_laplace_then_clip_and_quantize",
                            "report_clip": REPORT_CLIP,
                            "report_scale": REPORT_SCALE,
                            "payload_group": f"Z_{REPORT_MODULUS}",
                            "no_wrap_envelope_checked": "true",
                            "workload": "bounded_heavy_tail_clean_mean_under_endpoint_contamination",
                            "target": "uncontaminated_epoch_sample_mean",
                            "estimator": (
                                "single_global_mean"
                                if method == METHOD_SECURE
                                else (
                                    "public_dummy_mass_corrected_global_sum"
                                    if method == METHOD_POOL_LINEAR
                                    else "huber_location_tuning_0_75"
                                )
                            ),
                            "m": SAMPLE_M,
                            "N": N,
                            "M_pool": M_pool,
                            "M_ideal": M_ideal,
                            "ideal_tau": IDEAL_TAU,
                            "centers": "0.30|0.50|0.70",
                            "seeds": len(SEEDS),
                            "trials_per_cell": trials_per_cell,
                            "public_dummy_law": "neutral_x_0_5_plus_laplace_with_positive_binomial_count_for_pool",
                            "public_dummy_center": PUBLIC_DUMMY_CENTER,
                            "pool_release_payload": "count_and_sum",
                            "full_release": "true",
                            "final_release_permuted": "true",
                            "dummy_labels_visible": "false",
                            "estimator_observes_real_output_count": "false",
                            "observed_mean_real_output_count": mean_real,
                            "observed_real_output_count_standard_error": se_real,
                            "observed_mean_dummy_count": mean_dummy,
                            "observed_dummy_count_standard_error": se_dummy,
                            "expected_pool_dummy_fraction": expected_empty_fraction,
                            "ideal_dummy_count": M_ideal - SAMPLE_M,
                            "dummy_only_control": str(
                                method == METHOD_DUMMY_ONLY
                            ).lower(),
                        }
                    )
    return rows


def generate_all() -> dict[str, int]:
    rows = simulate()
    filename = "fig_pool_robust_mean.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
