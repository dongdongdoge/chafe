"""Share-level extraction for Chafe's packed64 DPF value.

The DPF shares live in Z_(2^64).  The low eight bits are therefore not
independent additive count shares: adding their residues can carry into the
high payload field.  Epoch close opens the two low residues, uses their carry
to correct one high-field share, and keeps the payload itself secret-shared.
"""

from __future__ import annotations

import numpy as np


COUNT_BITS = 8
COUNT_MASK = np.uint64((1 << COUNT_BITS) - 1)
PAYLOAD_MASK = np.uint64((1 << (64 - COUNT_BITS)) - 1)


def extract_packed_fields(
    share0: np.ndarray,
    share1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open occupancies and return corrected 56-bit payload shares."""
    h0 = np.asarray(share0, dtype=np.uint64)
    h1 = np.asarray(share1, dtype=np.uint64)
    if h0.shape != h1.shape:
        raise ValueError("packed share arrays must have the same shape")

    low0 = h0 & COUNT_MASK
    low1 = h1 & COUNT_MASK
    low_sum = low0.astype(np.uint16) + low1.astype(np.uint16)
    occupancy = (low_sum & int(COUNT_MASK)).astype(np.uint8)
    carry = (low_sum >> COUNT_BITS).astype(np.uint64)

    payload0 = ((h0 >> COUNT_BITS) + carry) & PAYLOAD_MASK
    payload1 = (h1 >> COUNT_BITS) & PAYLOAD_MASK
    return occupancy, payload0, payload1


def reconstruct_payload(
    payload0: np.ndarray,
    payload1: np.ndarray,
) -> np.ndarray:
    """Reconstruct the corrected payload in Z_(2^56)."""
    return (
        np.asarray(payload0, dtype=np.uint64)
        + np.asarray(payload1, dtype=np.uint64)
    ) & PAYLOAD_MASK
