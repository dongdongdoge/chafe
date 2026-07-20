"""Scoped cost scenarios for the CGP secret-shared shuffle.

The full row instantiates the analytical Permute+Share cost in the revised
ePrint and doubles it because a complete secret-shared shuffle invokes
Permute+Share twice.  The payload-only row is an author-derived idealized
scenario, not a measured result or a protocol-independent lower bound.
"""

from __future__ import annotations

import math

from chafe_eval_lib import DATA_DIR, write_csv


REPORT_COUNTS = (
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
)
SECURITY_BITS = 128
AES_SECONDS_PER_BLOCK = 3.5e-9
PAYLOAD_BITS = 64
BANDWIDTH_BITS_PER_SECOND = 1_000_000_000

SOURCE_TYPE = "literature_derived_analytic_cost"
CITATION_KEY = "DBLP:conf/asiacrypt/ChaseGP20"
EPRINT_URL = "https://eprint.iacr.org/2019/1340"
FORMULA_SCOPE = (
    "revised ePrint Sec. 7.2 Permute+Share AES-plus-bandwidth model; "
    "complete secret-shared shuffle is two sequential Permute+Share calls; "
    "paper model excludes base OT, XOR, RTT, collection, and implementation overhead"
)
PAYLOAD_ONLY_SCOPE = (
    "author-derived payload-only ideal 2*N*ell/BW; one masked N*ell-bit "
    "vector per sequential Permute+Share call; all preprocessing, other "
    "communication, computation, RTT, validation, collection, and analysis "
    "are free; not a protocol-independent lower bound"
)


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two at least ``value``."""
    if value < 2:
        raise ValueError("CGP batch size must be at least two")
    return 1 << (value - 1).bit_length()


def permute_share_cost(
    N: int,
    T: int,
    *,
    payload_bits: int = PAYLOAD_BITS,
    security_bits: int = SECURITY_BITS,
    aes_seconds_per_block: float = AES_SECONDS_PER_BLOCK,
    bandwidth_bits_per_second: float = BANDWIDTH_BITS_PER_SECOND,
) -> dict[str, float | int]:
    """Instantiate the revised-ePrint cost of one Permute+Share call.

    The proven Benes decomposition uses power-of-two ``N`` and ``T``.  The
    returned AES count is the paper's per-party count; parties execute in
    parallel, as assumed by the paper's latency model.
    """
    if N < 2 or N & (N - 1):
        raise ValueError("N must be a power of two and at least two")
    if T < 2 or T > N or T & (T - 1):
        raise ValueError("T must be a power of two in [2,N]")
    if payload_bits <= 0 or security_bits <= 0:
        raise ValueError("payload and security widths must be positive")
    if aes_seconds_per_block <= 0.0 or bandwidth_bits_per_second <= 0.0:
        raise ValueError("AES time and bandwidth must be positive")

    log_n = N.bit_length() - 1
    log_t = T.bit_length() - 1
    d = 2 * math.ceil(log_n / log_t) - 1
    aes_calls = (
        3 * d * N * log_t
        + d * N * T * (2.0 + payload_bits / security_bits)
    )
    communication_bits = (
        3 * d * N * security_bits * log_t
        + (d + 1) * N * payload_bits
    )
    computation_seconds = aes_calls * aes_seconds_per_block
    communication_seconds = communication_bits / bandwidth_bits_per_second
    return {
        "T": T,
        "d": d,
        "aes_calls_one_permute_share": aes_calls,
        "communication_bits_one_permute_share": communication_bits,
        "computation_seconds_one_permute_share": computation_seconds,
        "communication_seconds_one_permute_share": communication_seconds,
        "one_permute_share_seconds": computation_seconds + communication_seconds,
    }


def best_permute_share_cost(
    N: int,
    *,
    payload_bits: int = PAYLOAD_BITS,
    security_bits: int = SECURITY_BITS,
    aes_seconds_per_block: float = AES_SECONDS_PER_BLOCK,
    bandwidth_bits_per_second: float = BANDWIDTH_BITS_PER_SECOND,
) -> dict[str, float | int]:
    """Minimize the paper-derived cost over proven power-of-two ``T``."""
    if N < 2 or N & (N - 1):
        raise ValueError("N must be a power of two and at least two")
    candidates = [
        permute_share_cost(
            N,
            1 << log_t,
            payload_bits=payload_bits,
            security_bits=security_bits,
            aes_seconds_per_block=aes_seconds_per_block,
            bandwidth_bits_per_second=bandwidth_bits_per_second,
        )
        for log_t in range(1, N.bit_length())
    ]
    return min(candidates, key=lambda row: (float(row["one_permute_share_seconds"]), int(row["T"])))


def model_rows(report_counts: tuple[int, ...] = REPORT_COUNTS) -> list[dict]:
    """Return final-arrival CGP work rows for each batch size."""
    rows: list[dict] = []
    for m in report_counts:
        N = next_power_of_two(m)
        best = best_permute_share_cost(N)
        one_permute_share_seconds = float(best["one_permute_share_seconds"])
        full_shuffle_seconds = 2.0 * one_permute_share_seconds
        payload_only_ideal_seconds = (
            2.0 * N * PAYLOAD_BITS / BANDWIDTH_BITS_PER_SECOND
        )

        common = {
            "m": m,
            "N": N,
            "log_n": N.bit_length() - 1,
            "T": best["T"],
            "d": best["d"],
            "security_bits": SECURITY_BITS,
            "payload_bits": PAYLOAD_BITS,
            "bandwidth_bits_per_second": BANDWIDTH_BITS_PER_SECOND,
            "aes_seconds_per_block": AES_SECONDS_PER_BLOCK,
            "aes_calls_one_permute_share": best[
                "aes_calls_one_permute_share"
            ],
            "communication_bits_one_permute_share": best[
                "communication_bits_one_permute_share"
            ],
            "computation_seconds_one_permute_share": best[
                "computation_seconds_one_permute_share"
            ],
            "communication_seconds_one_permute_share": best[
                "communication_seconds_one_permute_share"
            ],
            "one_permute_share_seconds": one_permute_share_seconds,
            "full_shuffle_seconds": full_shuffle_seconds,
            "payload_only_ideal_seconds": payload_only_ideal_seconds,
            "measured": "false",
            "citation_key": CITATION_KEY,
            "eprint_url": EPRINT_URL,
        }
        rows.append(
            {
                **common,
                "method": "CGP full secret-shared shuffle (paper-derived)",
                "source_type": SOURCE_TYPE,
                "cost_seconds": full_shuffle_seconds,
                "formula_scope": FORMULA_SCOPE,
            }
        )
        rows.append(
            {
                **common,
                "method": "CGP payload-only ideal (author-derived)",
                "source_type": "author_derived_analytic_scenario",
                "cost_seconds": payload_only_ideal_seconds,
                "formula_scope": PAYLOAD_ONLY_SCOPE,
            }
        )
    return rows


def generate_all() -> dict[str, int]:
    rows = model_rows()
    path = DATA_DIR / "fig_cgp_shuffle_model.csv"
    write_csv(path, rows)
    return {str(path.relative_to(DATA_DIR.parent)): len(rows)}


if __name__ == "__main__":
    generate_all()
