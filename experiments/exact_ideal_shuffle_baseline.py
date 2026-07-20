from __future__ import annotations

from fractions import Fraction

from chafe_eval_lib import DATA_DIR, write_csv
from exhaustive_hidden_slot_oracle import (
    epsilon_from_alpha_fraction,
    two_sided_min_alpha_fraction,
)


F = Fraction
M = 3
DELTA = F(1, 1000)
F0 = (F(3, 4), F(1, 4))
F1 = (F(1, 4), F(3, 4))


def convolve(a: list[Fraction], b: tuple[Fraction, Fraction]) -> list[Fraction]:
    out = [F(0)] * (len(a) + 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            out[i + j] += left * right
    return out


def shuffled_count_law(ones: int) -> dict[tuple[int, ...], Fraction]:
    law = [F(1)]
    for index in range(M):
        law = convolve(law, F1 if index < ones else F0)
    return {(count,): probability for count, probability in enumerate(law)}


def generate_all() -> dict[str, int]:
    rows = []
    pair_alphas = []
    for ones in range(M):
        P = shuffled_count_law(ones)
        Q = shuffled_count_law(ones + 1)
        alpha = two_sided_min_alpha_fraction(P, Q, DELTA)
        if alpha is None:
            raise AssertionError("finite ideal-shuffle baseline has infinite budget")
        pair_alphas.append(alpha)
        rows.append(
            {
                "pair": f"ones_{ones}_to_{ones + 1}",
                "m": M,
                "eps0": "ln(3)",
                "delta": "1/1000",
                "alpha_exact_fraction": f"{alpha.numerator}/{alpha.denominator}",
                "epsilon_exact": epsilon_from_alpha_fraction(alpha),
                "source_type": "exact_accountant",
                "accounting_scope": "fixed_size_replacement_exhaustive",
                "output_representation": "binary_report_count",
            }
        )
    family_alpha = max(pair_alphas)
    for row in rows:
        row["family_alpha_exact_fraction"] = (
            f"{family_alpha.numerator}/{family_alpha.denominator}"
        )
        row["family_epsilon_exact"] = epsilon_from_alpha_fraction(family_alpha)
        row["is_worst_case"] = str(
            row["alpha_exact_fraction"]
            == row["family_alpha_exact_fraction"]
        ).lower()
    filename = "fig_exact_ideal_shuffle_baseline.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
