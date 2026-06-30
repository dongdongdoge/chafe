from __future__ import annotations

import math

from chafe_eval_lib import (
    DATA_DIR,
    brr_probs,
    eps_drop_proxy,
    final_count_distribution_iid_slots,
    final_count_distribution_target_add,
    invert_hockey_stick,
    two_sided_profile,
    write_csv,
)


def exact_drop_for_rho(m: int, rho: float, tau: float, eps0: float, delta: float) -> dict:
    N = int(math.ceil(float(m) / float(rho)))
    M = int(math.ceil(float(m) * float(tau)))
    if N > M:
        raise ValueError(
            f"rho={rho} gives N={N} > M={M}; this exact Poissonized fixed-length validation "
            "uses rho values with N <= M"
        )

    background_one_prob = 0.5
    dummy_one_prob = 0.5
    f_star_zero, f_star_one = brr_probs(eps0, true_bit=1)
    singleton_prob = rho * math.exp(-rho)

    background = final_count_distribution_iid_slots(
        num_slots=N,
        M=M,
        singleton_prob=singleton_prob,
        real_one_prob=background_one_prob,
        dummy_one_prob=dummy_one_prob,
    )

    target_states = [
        (0, 0, 1.0 - math.exp(-rho)),
        (1, 0, math.exp(-rho) * f_star_zero),
        (1, 1, math.exp(-rho) * f_star_one),
    ]
    target_add = final_count_distribution_target_add(
        num_background_slots=N - 1,
        M=M,
        singleton_prob=singleton_prob,
        background_one_prob=background_one_prob,
        dummy_one_prob=dummy_one_prob,
        target_states=target_states,
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
        "background_law": "Bernoulli(1/2)",
        "target_law": "BRR(true_bit=1)",
        "dummy_law": "Bernoulli(1/2)",
    }


def generate_exact_vs_proxy() -> list[dict]:
    m = 200
    tau = 2.0
    eps0 = 1.0
    delta = 1e-6
    rhos = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    rows: list[dict] = []
    for rho in rhos:
        result = exact_drop_for_rho(m=m, rho=rho, tau=tau, eps0=eps0, delta=delta)
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
            "exact_to_proxy_ratio": result["exact_to_proxy_ratio"],
            "profile_delta_at_proxy": result["profile_delta_at_proxy"],
        }
        rows.append(
            {
                **common,
                "method": "Drop exact hockey-stick",
                "epsilon": result["epsilon_exact"],
                "source_type": "exact_accountant",
                "bound": "finite_domain_hockey_stick_count_distribution",
            }
        )
        rows.append(
            {
                **common,
                "method": "Drop second-order proxy",
                "epsilon": result["epsilon_proxy"],
                "source_type": "formula",
                "bound": "second_order_matched_drop",
            }
        )
    write_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv", rows)
    return rows


def generate_all() -> dict[str, int]:
    rows = generate_exact_vs_proxy()
    return {"fig_exact_vs_proxy_drop.csv": len(rows)}


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")

