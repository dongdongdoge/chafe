from __future__ import annotations

import math

from chafe_eval_lib import DATA_DIR, write_csv


COUNT_BITS = 8
PAYLOAD_BITS = 56
MAX_REPORT_PAYLOAD = (1 << 31) - 1


def unsafe_occupancy_threshold(max_report_payload: int) -> int:
    """First slot occupancy for which either packed field can overflow."""
    if max_report_payload <= 0:
        raise ValueError("max_report_payload must be positive")
    count_threshold = 1 << COUNT_BITS
    payload_threshold = ((1 << PAYLOAD_BITS) - 1) // max_report_payload + 1
    return min(count_threshold, payload_threshold)


def chernoff_log10_epoch_bound(m: int, N: int, threshold: int) -> float:
    """Return log10 of the epoch-wide Chernoff/union upper bound."""
    if m <= 0 or N <= 0 or threshold <= 0:
        raise ValueError("m, N, and threshold must be positive")
    rho = m / N
    if threshold <= rho:
        raise ValueError("Chernoff upper-tail form requires threshold > rho")
    return math.log10(N) + threshold * math.log10(math.e * rho / threshold)


def generate_all() -> dict[str, int]:
    threshold = unsafe_occupancy_threshold(MAX_REPORT_PAYLOAD)
    configurations = [
        ("ingestion_projection", 100_000, 1 << 17),
        ("largest_real_workload", 300_000, 300_000),
    ]
    rows: list[dict] = []
    for name, m, N in configurations:
        rows.append(
            {
                "configuration": name,
                "m": m,
                "N": N,
                "rho": m / N,
                "count_bits": COUNT_BITS,
                "payload_bits": PAYLOAD_BITS,
                "max_report_payload": MAX_REPORT_PAYLOAD,
                "unsafe_occupancy_threshold": threshold,
                "log10_epoch_overflow_upper_bound": chernoff_log10_epoch_bound(
                    m, N, threshold
                ),
                "bound": "N*(e*rho/L)^L",
                "source_type": "formula",
            }
        )
    path = DATA_DIR / "packed64_safety.csv"
    write_csv(path, rows)
    return {str(path.relative_to(DATA_DIR.parent)): len(rows)}


if __name__ == "__main__":
    generate_all()
