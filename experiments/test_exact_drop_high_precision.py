from __future__ import annotations

import unittest

try:
    from .exact_drop_accountant import (
        exact_drop_for_case,
        high_precision_exact_drop_epsilon,
    )
except ImportError:
    from exact_drop_accountant import (
        exact_drop_for_case,
        high_precision_exact_drop_epsilon,
    )


class ExactDropHighPrecisionTest(unittest.TestCase):
    def test_decimal_precision_converges_and_matches_reported_float(self) -> None:
        common = dict(
            m=1_000,
            N=1_000,
            M=2_000,
            eps0=1.0,
            delta=1e-6,
            background_law="BRR(true_bit=0)",
        )
        eps50 = high_precision_exact_drop_epsilon(**common, precision=50)
        eps70 = high_precision_exact_drop_epsilon(**common, precision=70)
        self.assertLess(abs(eps70 - eps50), eps70.__class__("1e-40"))

        result = exact_drop_for_case(
            m=1_000,
            rho=1.0,
            tau=2.0,
            eps0=1.0,
            delta=1e-6,
            background_law="BRR(true_bit=0)",
            sweep="test",
        )
        self.assertLess(
            abs(result["epsilon_exact"] - float(eps70)),
            1e-10,
        )
        self.assertIn("not_interval_certified", result["numerical_status"])


if __name__ == "__main__":
    unittest.main()
