from __future__ import annotations

import unittest
from fractions import Fraction


F = Fraction


def convolution(left: tuple[Fraction, ...], right: tuple[Fraction, ...]):
    modulus = len(left)
    out = [F(0)] * modulus
    for i, p in enumerate(left):
        for j, q in enumerate(right):
            out[(i + j) % modulus] += p * q
    return tuple(out)


def chisq(P: tuple[Fraction, ...], Q: tuple[Fraction, ...]) -> Fraction:
    total = F(0)
    for p, q in zip(P, Q):
        if q == 0:
            if p:
                raise ValueError("P is not absolutely continuous with Q")
            continue
        total += (p - q) ** 2 / q
    return total


class ConvolutionContractionTest(unittest.TestCase):
    def test_exchangeable_convolution_contracts_by_one_over_t(self) -> None:
        P = (F(1, 2), F(1, 3), F(1, 6))
        Q = (F(1, 5), F(3, 10), F(1, 2))
        denominator = chisq(P, Q)
        p_convolved = P
        q_convolved = Q
        for t in range(1, 7):
            with self.subTest(t=t):
                self.assertLessEqual(
                    chisq(p_convolved, q_convolved),
                    denominator / t,
                )
            p_convolved = convolution(p_convolved, Q)
            q_convolved = convolution(q_convolved, Q)


if __name__ == "__main__":
    unittest.main()
