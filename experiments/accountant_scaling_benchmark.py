from __future__ import annotations

import math
import platform
import statistics
import time

from chafe_eval_lib import DATA_DIR, invert_hockey_stick, write_csv
from exact_drop_accountant import exact_count_distributions
from exact_hidden_slot_accountant import NeighborPair, account_pair


def _median_run(fn, repetitions: int = 3):
    timings = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return result, statistics.median(timings), min(timings), max(timings)


def _coefficient_rows() -> list[dict]:
    configurations = [
        (
            "Coefficient DP (K=2)",
            (0.75, 0.25),
            (0.25, 0.75),
            (0.5, 0.5),
            [4, 8, 12, 16, 24, 32, 40],
        ),
        (
            "Coefficient DP (K=3)",
            (0.6, 0.3, 0.1),
            (0.1, 0.2, 0.7),
            (1 / 3, 1 / 3, 1 / 3),
            [4, 8, 12, 16, 20, 24],
        ),
    ]
    rows: list[dict] = []
    for method, background, target, nu, sizes in configurations:
        K = len(nu)
        for m in sizes:
            N = m
            M = 2 * m
            pair = NeighborPair.homogeneous(
                f"runtime_{K}_{m}",
                m,
                background,
                target,
                family_scope="one_homogeneous_add_remove_pair",
            )
            result, median_s, min_s, max_s = _median_run(
                lambda: account_pair(
                    pair,
                    "pool",
                    N=N,
                    M=M,
                    nu=nu,
                    delta=1e-3,
                )
            )
            support = len(set(result.P) | set(result.Q))
            real_limit = min(m + 1, N)
            rows.append(
                {
                    "method": method,
                    "m": m,
                    "N": N,
                    "M": M,
                    "K": K,
                    "decoder": "pool",
                    "runtime_median_s": median_s,
                    "runtime_min_s": min_s,
                    "runtime_max_s": max_s,
                    "repetitions": 3,
                    "final_support_states": support,
                    "real_count_state_upper": math.comb(real_limit + K, K),
                    "final_count_state_upper": math.comb(M + K - 1, K - 1),
                    "arithmetic_operations_diagnostic": (
                        result.diagnostics.arithmetic_operations
                    ),
                    "epsilon_nominal": result.epsilon_nominal,
                    "source_type": "measured_accountant_runtime",
                    "accountant_scope": "one_homogeneous_add_remove_pair",
                    "numerical_status": "float64_coefficient_dp_not_interval_certified",
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                }
            )
    return rows


def _specialized_drop_rows() -> list[dict]:
    rows: list[dict] = []
    for m in [100, 1_000, 10_000, 100_000]:
        N = m
        M = 2 * m

        def run():
            P, Q, _ = exact_count_distributions(
                m=m,
                N=N,
                tau=2.0,
                eps0=1.0,
                background_law="BRR(true_bit=0)",
            )
            epsilon = invert_hockey_stick(P, Q, 1e-6, tol=1e-13)
            return P, Q, epsilon

        result, median_s, min_s, max_s = _median_run(run)
        P, Q, epsilon = result
        rows.append(
            {
                "method": "Matched-Drop sufficient count law",
                "m": m,
                "N": N,
                "M": M,
                "K": 2,
                "decoder": "drop",
                "runtime_median_s": median_s,
                "runtime_min_s": min_s,
                "runtime_max_s": max_s,
                "repetitions": 3,
                "final_support_states": len(P),
                "real_count_state_upper": "",
                "final_count_state_upper": M + 1,
                "arithmetic_operations_diagnostic": "",
                "epsilon_nominal": epsilon,
                "source_type": "measured_accountant_runtime",
                "accountant_scope": "one_homogeneous_matched_drop_pair",
                "numerical_status": "exact_law_float64_not_interval_certified",
                "machine": platform.machine(),
                "python": platform.python_version(),
            }
        )
    return rows


def generate_all() -> dict[str, int]:
    rows = _coefficient_rows() + _specialized_drop_rows()
    filename = "fig_accountant_scaling.csv"
    write_csv(DATA_DIR / filename, rows)
    return {filename: len(rows)}


if __name__ == "__main__":
    print(generate_all())
