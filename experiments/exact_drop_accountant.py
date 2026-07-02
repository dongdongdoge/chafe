from __future__ import annotations

import math

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
    rho: float,
    tau: float,
    eps0: float,
    background_law: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact final-count laws for matched Bernoulli padding.

    With matched dummy and background laws, all non-target released values have
    the same Bernoulli one probability. The sufficient statistic is therefore
    the final number of ones in the fixed-length release. Under the target-add
    execution, the only perturbation is the target singleton event, which occurs
    with probability exp(-rho).
    """
    M = int(math.ceil(float(m) * float(tau)))
    background_one_prob = background_one_probability(eps0, background_law)
    f_star_zero, f_star_one = brr_probs(eps0, true_bit=1)

    background = binom_pmf(M, background_one_prob)
    non_target = binom_pmf(M - 1, background_one_prob)
    target_when_singleton = np.zeros(M + 1, dtype=float)
    target_when_singleton[:M] += f_star_zero * non_target
    target_when_singleton[1:] += f_star_one * non_target

    singleton_target_prob = math.exp(-rho)
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
        rho=rho,
        tau=tau,
        eps0=eps0,
        background_law=background_law,
    )

    exact_eps = invert_hockey_stick(target_add, background, delta)
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
        "accountant_model": "exact_final_count_distribution_matched_padding",
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
            "method": "Exact hockey-stick",
            "epsilon": result["epsilon_exact"],
            "source_type": "exact_accountant",
            "bound": "finite_domain_hockey_stick_count_distribution",
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
