"""Task-aware estimators for fixed-length unlabeled releases.

The analyzer observes exactly ``release_length`` decoded payloads but does not
observe which payloads are real or how many real payloads were decoded.  These
helpers remove the *expected* contribution of a fixed public dummy law before
inverting an affine local randomizer.  They intentionally never consume the
realized real-output count.
"""

from __future__ import annotations

import math

import numpy as np


def drop_survival(epoch_size: int, slot_count: int) -> float:
    """Marginal probability that one report occupies a singleton slot."""
    if epoch_size < 0 or slot_count <= 0:
        raise ValueError("epoch_size must be nonnegative and slot_count positive")
    if epoch_size == 0:
        return 0.0
    return (1.0 - 1.0 / slot_count) ** (epoch_size - 1)


def expected_drop_outputs(epoch_size: int, slot_count: int) -> float:
    return float(epoch_size) * drop_survival(epoch_size, slot_count)


def expected_pool_outputs(epoch_size: int, slot_count: int) -> float:
    """Expected number of nonempty slots under finite multinomial placement."""
    if epoch_size < 0 or slot_count <= 0:
        raise ValueError("epoch_size must be nonnegative and slot_count positive")
    return float(slot_count) * (1.0 - (1.0 - 1.0 / slot_count) ** epoch_size)


def corrected_affine_total(
    released_total: float,
    *,
    epoch_size: int,
    release_length: int,
    expected_real_outputs: float,
    real_report_retention: float,
    dummy_mean: float,
    affine_offset: float,
    affine_slope: float,
) -> float:
    """Recover a linear total from an unlabeled padded release.

    For a target statistic ``T`` assume one local report satisfies

        E[Y_i | x_i] = affine_offset + affine_slope * g(x_i),

    and the decoder preserves each report's contribution in expectation by the
    factor ``real_report_retention``.  ``expected_real_outputs`` is the expected
    number of decoded payloads, which differs from the retained report mass for
    Pool.  If public dummies have mean ``dummy_mean``, then

        E[released_total]
          = retention * (offset * epoch_size + slope * T)
            + (release_length - E[real outputs]) * dummy_mean.

    The returned estimator is therefore unbiased whenever these declared
    moments are correct.  It uses no realized dummy labels or output count.
    """
    if epoch_size < 0 or release_length < 0:
        raise ValueError("epoch_size and release_length must be nonnegative")
    if not 0.0 < real_report_retention <= 1.0:
        raise ValueError("real_report_retention must lie in (0,1]")
    if affine_slope == 0.0 or not math.isfinite(affine_slope):
        raise ValueError("affine_slope must be finite and nonzero")
    if expected_real_outputs < -1e-12 or expected_real_outputs > release_length + 1e-12:
        raise ValueError("expected real-output count must lie within release length")

    expected_dummy_total = (release_length - expected_real_outputs) * dummy_mean
    corrected_real_total = released_total - expected_dummy_total
    return (
        corrected_real_total
        - real_report_retention * affine_offset * epoch_size
    ) / (real_report_retention * affine_slope)


def corrected_grr_histogram(
    released_counts: np.ndarray,
    *,
    epoch_size: int,
    release_length: int,
    expected_real_outputs: float,
    real_report_retention: float,
    dummy_law: np.ndarray,
    truthful_probability: float,
    alternative_probability: float,
) -> np.ndarray:
    """Recover a histogram after GRR, collision decoding, and public padding."""
    counts = np.asarray(released_counts, dtype=float)
    law = np.asarray(dummy_law, dtype=float)
    if counts.ndim != 1 or law.shape != counts.shape:
        raise ValueError("released_counts and dummy_law must be aligned vectors")
    if bool((law < 0.0).any()) or not np.isclose(float(law.sum()), 1.0):
        raise ValueError("dummy_law must be a probability vector")
    slope = truthful_probability - alternative_probability
    if slope <= 0.0:
        raise ValueError("GRR truthful probability must exceed its alternative probability")

    expected_dummy_histogram = (release_length - expected_real_outputs) * law
    expected_grr_offset = (
        real_report_retention * epoch_size * alternative_probability
    )
    return (
        counts - expected_dummy_histogram - expected_grr_offset
    ) / (real_report_retention * slope)


def public_epoch_mean(corrected_sum: float, epoch_size: int) -> float:
    """Convert a corrected sum to a mean when the replacement epoch size is public."""
    if epoch_size <= 0:
        raise ValueError("a public mean requires a positive epoch size")
    return corrected_sum / float(epoch_size)
