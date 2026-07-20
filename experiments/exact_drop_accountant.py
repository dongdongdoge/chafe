from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np

from chafe_eval_lib import (
    DATA_DIR,
    binom_pmf,
    brr_probs,
    eps_drop_proxy,
    invert_hockey_stick,
    two_sided_profile,
    write_csv,
)


def _decimal_one_sided_min_alpha(
    P: list[Decimal],
    Q: list[Decimal],
    delta: Decimal,
) -> Decimal:
    """Likelihood-ratio breakpoint inversion in Decimal arithmetic."""
    singular = sum(
        (p for p, q in zip(P, Q) if q == 0),
        Decimal(0),
    )
    terms = [
        (p / q, p, q)
        for p, q in zip(P, Q)
        if q > 0 and p / q > 1
    ]
    terms.sort(key=lambda item: item[0])
    active_p = sum((item[1] for item in terms), Decimal(0))
    active_q = sum((item[2] for item in terms), Decimal(0))
    lower = Decimal(1)
    index = 0
    while index < len(terms):
        breakpoint = terms[index][0]
        if active_q > 0:
            candidate = (singular + active_p - delta) / active_q
            if candidate <= breakpoint:
                return max(Decimal(1), lower, candidate)
        while index < len(terms) and terms[index][0] == breakpoint:
            active_p -= terms[index][1]
            active_q -= terms[index][2]
            index += 1
        lower = breakpoint
    return lower


def high_precision_exact_drop_epsilon(
    *,
    m: int,
    N: int,
    M: int,
    eps0: float,
    delta: float,
    background_law: str,
    precision: int = 70,
) -> Decimal:
    """Evaluate the homogeneous matched-Drop exact law with Decimal arithmetic.

    Under Q=Bin(M,p), the add-one law has the affine likelihood ratio

      P(k)/Q(k) = 1-s + s[(k/M)(p*/p)
                         +((M-k)/M)((1-p*)/(1-p))].

    This avoids reconstructing the hidden-slot state.  Decimal recurrence and
    likelihood-ratio breakpoint inversion provide an independent high-
    precision check of the reported float64 value.  They are not outward-
    rounded interval arithmetic.
    """
    if m < 0 or N <= 0 or M <= 0:
        raise ValueError("invalid dimensions")
    with localcontext() as context:
        context.prec = precision
        eps = Decimal(str(eps0))
        exp_eps = eps.exp()
        target_one = exp_eps / (exp_eps + 1)
        if background_law == "Bernoulli(1/2)":
            background_one = Decimal(1) / 2
        elif background_law == "BRR(true_bit=0)":
            background_one = Decimal(1) / (exp_eps + 1)
        else:
            raise ValueError(f"unknown background law: {background_law}")

        one = Decimal(1)
        M_dec = Decimal(M)
        singleton = (one - one / Decimal(N)) ** m
        q0 = (one - background_one) ** M
        odds = background_one / (one - background_one)
        Q = [q0]
        for k in range(M):
            Q.append(
                Q[-1]
                * Decimal(M - k)
                / Decimal(k + 1)
                * odds
            )
        q_mass = sum(Q, Decimal(0))
        Q = [value / q_mass for value in Q]

        intercept = (
            one
            - singleton
            + singleton
            * (one - target_one)
            / (one - background_one)
        )
        slope = (
            singleton
            * (
                target_one / background_one
                - (one - target_one) / (one - background_one)
            )
            / M_dec
        )
        P = [
            probability * (intercept + slope * Decimal(k))
            for k, probability in enumerate(Q)
        ]
        p_mass = sum(P, Decimal(0))
        P = [value / p_mass for value in P]
        delta_dec = Decimal(str(delta))
        alpha = max(
            _decimal_one_sided_min_alpha(P, Q, delta_dec),
            _decimal_one_sided_min_alpha(Q, P, delta_dec),
        )
        return +alpha.ln()


def background_one_probability(eps0: float, background_law: str) -> float:
    if background_law == "Bernoulli(1/2)":
        return 0.5
    if background_law == "BRR(true_bit=0)":
        _, one_prob = brr_probs(eps0, true_bit=0)
        return one_prob
    raise ValueError(f"unknown background law: {background_law}")


def exact_count_distributions(
    *,
    m: int,
    N: int,
    tau: float,
    eps0: float,
    background_law: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact finite-(m,N) final-count laws for matched Bernoulli padding.

    With matched dummy and background laws, all non-target released values have
    the same Bernoulli one probability. The sufficient statistic is therefore
    the final number of ones in the fixed-length release. Under add/remove
    adjacency, the only distribution-changing event is that the added target is
    a singleton. Its exact finite occupancy probability is (1 - 1/N)^m.

    This reduction is exact for the homogeneous binary-background,
    matched-padding case used by this experiment. It is not a generic
    finite-domain accountant for arbitrary report or dummy laws.
    """
    M = int(math.ceil(float(m) * float(tau)))
    if M < min(m + 1, N):
        raise ValueError(
            f"release length M={M} can overflow for add-one Drop with m={m}, N={N}"
        )
    background_one_prob = background_one_probability(eps0, background_law)
    f_star_zero, f_star_one = brr_probs(eps0, true_bit=1)

    background = binom_pmf(M, background_one_prob)
    non_target = binom_pmf(M - 1, background_one_prob)
    target_when_singleton = np.zeros(M + 1, dtype=float)
    target_when_singleton[:M] += f_star_zero * non_target
    target_when_singleton[1:] += f_star_one * non_target

    singleton_target_prob = (1.0 - 1.0 / float(N)) ** m
    target_add = (1.0 - singleton_target_prob) * background + singleton_target_prob * target_when_singleton
    target_add /= target_add.sum()
    return target_add, background, background_one_prob


def exact_drop_for_case(
    m: int,
    rho: float,
    tau: float,
    eps0: float,
    delta: float,
    background_law: str,
    sweep: str,
) -> dict:
    N = int(math.ceil(float(m) / float(rho)))
    M = int(math.ceil(float(m) * float(tau)))
    target_add, background, background_one_prob = exact_count_distributions(
        m=m,
        N=N,
        tau=tau,
        eps0=eps0,
        background_law=background_law,
    )

    exact_eps = invert_hockey_stick(
        target_add, background, delta, tol=1e-13
    )
    M = int(math.ceil(float(m) * float(tau)))
    high_precision_eps = high_precision_exact_drop_epsilon(
        m=m,
        N=N,
        M=M,
        eps0=eps0,
        delta=delta,
        background_law=background_law,
        precision=70,
    )
    proxy_eps = eps_drop_proxy(eps0, rho, M, delta)
    profile_at_proxy = two_sided_profile(target_add, background, proxy_eps)
    ratio = exact_eps / proxy_eps if proxy_eps > 0 else float("inf")

    return {
        "rho": rho,
        "epsilon_exact": exact_eps,
        "epsilon_proxy": proxy_eps,
        "exact_to_proxy_ratio": ratio,
        "profile_delta_at_proxy": profile_at_proxy,
        "m": m,
        "N": N,
        "M": M,
        "tau": tau,
        "eps0": eps0,
        "delta": delta,
        "decoder": "drop",
        "domain": "binary",
        "randomizer": "brr",
        "background_law": background_law,
        "target_law": "BRR(true_bit=1)",
        "dummy_law": f"matched_background_Bernoulli({background_one_prob:.8g})",
        "sweep": sweep,
        "accountant_model": "exact_finite_occupancy_matched_binary_count_distribution",
        "epsilon_decimal70": str(high_precision_eps),
        "float64_abs_error_vs_decimal70": abs(
            exact_eps - float(high_precision_eps)
        ),
        "numerical_status": "exact_law_float64_cross_checked_decimal70_not_interval_certified",
    }


def append_exact_and_proxy_rows(rows: list[dict], result: dict) -> None:
    common = {
        "rho": result["rho"],
        "m": result["m"],
        "N": result["N"],
        "M": result["M"],
        "tau": result["tau"],
        "eps0": result["eps0"],
        "delta": result["delta"],
        "decoder": result["decoder"],
        "domain": result["domain"],
        "randomizer": result["randomizer"],
        "background_law": result["background_law"],
        "target_law": result["target_law"],
        "dummy_law": result["dummy_law"],
        "sweep": result["sweep"],
        "accountant_model": result["accountant_model"],
        "exact_to_proxy_ratio": result["exact_to_proxy_ratio"],
        "profile_delta_at_proxy": result["profile_delta_at_proxy"],
    }
    rows.append(
        {
            **common,
            "method": "Exact-law numerical",
            "epsilon": result["epsilon_exact"],
            "source_type": "exact_law_numerical",
            "bound": "exact_finite_law_decimal70_cross_checked",
            "epsilon_decimal70": result["epsilon_decimal70"],
            "float64_abs_error_vs_decimal70": result[
                "float64_abs_error_vs_decimal70"
            ],
            "numerical_status": result["numerical_status"],
        }
    )
    rows.append(
        {
            **common,
            "method": "Second-order proxy",
            "epsilon": result["epsilon_proxy"],
            "source_type": "formula",
            "bound": "second_order_matched_drop",
        }
    )


def generate_exact_vs_proxy() -> list[dict]:
    tau = 2.0
    eps0 = 1.0
    delta = 1e-6
    background_law = "BRR(true_bit=0)"
    rhos = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    m_grid = [200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]

    rows: list[dict] = []
    for rho in rhos:
        result = exact_drop_for_case(
            m=100_000,
            rho=rho,
            tau=tau,
            eps0=eps0,
            delta=delta,
            background_law=background_law,
            sweep="rho",
        )
        append_exact_and_proxy_rows(rows, result)
    for m in m_grid:
        result = exact_drop_for_case(
            m=m,
            rho=1.0,
            tau=tau,
            eps0=eps0,
            delta=delta,
            background_law=background_law,
            sweep="m",
        )
        append_exact_and_proxy_rows(rows, result)
    write_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv", rows)
    return rows


def generate_all() -> dict[str, int]:
    rows = generate_exact_vs_proxy()
    return {"fig_exact_vs_proxy_drop.csv": len(rows)}


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
