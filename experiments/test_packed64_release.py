from __future__ import annotations

import unittest

import numpy as np

from packed64_release import (
    COUNT_BITS,
    COUNT_MASK,
    PAYLOAD_MASK,
    extract_packed_fields,
    reconstruct_payload,
)


class Packed64ReleaseTests(unittest.TestCase):
    def test_random_scalar_shares_survive_field_extraction(self) -> None:
        rng = np.random.default_rng(7_711_043)
        occupancy = rng.integers(0, 64, size=20_000, dtype=np.uint64)
        payload = rng.integers(0, 1 << 48, size=20_000, dtype=np.uint64)
        logical = (payload << COUNT_BITS) | occupancy
        share0 = rng.integers(
            0, np.iinfo(np.uint64).max, size=len(logical), dtype=np.uint64
        )
        share1 = logical - share0  # uint64 subtraction intentionally wraps.

        opened, payload0, payload1 = extract_packed_fields(share0, share1)
        np.testing.assert_array_equal(opened.astype(np.uint64), occupancy)
        np.testing.assert_array_equal(
            reconstruct_payload(payload0, payload1), payload & PAYLOAD_MASK
        )

    def test_low_field_carry_is_required(self) -> None:
        logical = np.asarray([(17 << COUNT_BITS) | 3], dtype=np.uint64)
        # Force scalar wrap and a low-residue carry: 250 + 9 = 256 + 3.
        share0 = np.asarray([(29 << COUNT_BITS) | 250], dtype=np.uint64)
        share1 = logical - share0
        self.assertEqual(int(share1[0] & COUNT_MASK), 9)

        opened, payload0, payload1 = extract_packed_fields(share0, share1)
        self.assertEqual(int(opened[0]), 3)
        self.assertEqual(int(reconstruct_payload(payload0, payload1)[0]), 17)
        naive = int(((share0 >> COUNT_BITS) + (share1 >> COUNT_BITS))[0] & PAYLOAD_MASK)
        self.assertNotEqual(naive, 17)


if __name__ == "__main__":
    unittest.main()
