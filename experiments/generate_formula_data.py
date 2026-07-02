from __future__ import annotations

import math

import numpy as np

from chafe_eval_lib import (
    DATA_DIR,
    eps_drop_proxy,
    eps_ideal_shuffle,
    eps_pool_add_remove,
    eps_pool_replacement,
    write_csv,
)


DEFAULT_M = 1_000_000
DEFAULT_EPS0 = 1.0
DEFAULT_DELTA = 1e-6
DEFAULT_TAU = 2.0
DEFAULT_RHO = 1.0
RHO_GRID = np.linspace(0.05, 3.0, 60)
TAU_GRID = np.linspace(1.0, 8.0, 57)
EPS0_GRID = np.linspace(0.1, 5.0, 50)


def ceil_slots(m: int, rho: float) -> int:
    return max(1, int(math.ceil(float(m) / float(rho))))


def ceil_release(m: int, tau: float) -> int:
    return max(1, int(math.ceil(float(m) * float(tau))))


def base_row(
    *,
    method: str,
    epsilon: float,
    source_type: str,
    m: int,
    rho: float,
    tau: float,
    eps0: float,
    delta: float,
    decoder: str,
    bound: str,
) -> dict:
    M = ceil_release(m, tau)
    return {
        "method": method,
        "epsilon": epsilon,
        "source_type": source_type,
        "m": m,
        "N": ceil_slots(m, rho),
        "rho": rho,
        "M": M,
        "tau": tau,
        "eps0": eps0,
        "delta": delta,
        "decoder": decoder,
        "bound": bound,
    }


def generate_privacy_vs_rho() -> list[dict]:
    rows: list[dict] = []
    m = DEFAULT_M
    eps0 = DEFAULT_EPS0
    delta = DEFAULT_DELTA
    tau = DEFAULT_TAU
    M = ceil_release(m, tau)
    for rho in RHO_GRID:
        rho = float(rho)
        rows.append(
            base_row(
                method="IdealShuffle",
                epsilon=eps_ideal_shuffle(eps0, m, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="ideal_shuffle",
                bound="second_order_shuffle",
            )
        )
        rows.append(
            base_row(
                method="Drop (matched padding)",
                epsilon=eps_drop_proxy(eps0, rho, M, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="drop",
                bound="second_order_matched_drop",
            )
        )
        rows.append(
            base_row(
                method="Pool (replacement)",
                epsilon=eps_pool_replacement(eps0, rho, M, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="pool",
                bound="second_order_replacement",
            )
        )
        rows.append(
            base_row(
                method="Pool (add/remove)",
                epsilon=eps_pool_add_remove(eps0, rho, M, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="pool",
                bound="second_order_add_remove",
            )
        )
    write_csv(DATA_DIR / "fig_privacy_vs_rho.csv", rows)
    return rows


def generate_privacy_vs_eps0() -> list[dict]:
    rows: list[dict] = []
    m = DEFAULT_M
    rho = DEFAULT_RHO
    tau = DEFAULT_TAU
    delta = DEFAULT_DELTA
    M = ceil_release(m, tau)
    for eps0 in EPS0_GRID:
        eps0 = float(eps0)
        rows.append(
            base_row(
                method="IdealShuffle",
                epsilon=eps_ideal_shuffle(eps0, m, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="ideal_shuffle",
                bound="second_order_shuffle",
            )
        )
        rows.append(
            base_row(
                method="Drop (matched padding)",
                epsilon=eps_drop_proxy(eps0, rho, M, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="drop",
                bound="second_order_matched_drop",
            )
        )
        rows.append(
            base_row(
                method="Pool (replacement)",
                epsilon=eps_pool_replacement(eps0, rho, M, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="pool",
                bound="second_order_replacement",
            )
        )
    write_csv(DATA_DIR / "fig_privacy_vs_eps0.csv", rows)
    return rows


def generate_privacy_vs_tau() -> list[dict]:
    rows: list[dict] = []
    m = DEFAULT_M
    rho = DEFAULT_RHO
    eps0 = DEFAULT_EPS0
    delta = DEFAULT_DELTA
    for tau in TAU_GRID:
        tau = float(tau)
        M = ceil_release(m, tau)
        rows.append(
            base_row(
                method="IdealShuffle",
                epsilon=eps_ideal_shuffle(eps0, m, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="ideal_shuffle",
                bound="second_order_shuffle",
            )
        )
        rows.append(
            base_row(
                method="Drop (matched padding)",
                epsilon=eps_drop_proxy(eps0, rho, M, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="drop",
                bound="second_order_matched_drop",
            )
        )
        rows.append(
            base_row(
                method="Pool (replacement)",
                epsilon=eps_pool_replacement(eps0, rho, M, delta),
                source_type="formula",
                m=m,
                rho=rho,
                tau=tau,
                eps0=eps0,
                delta=delta,
                decoder="pool",
                bound="second_order_replacement",
            )
        )
    write_csv(DATA_DIR / "fig_privacy_vs_tau.csv", rows)
    return rows


def generate_collision_regimes() -> list[dict]:
    rows: list[dict] = []
    for rho in RHO_GRID:
        rho = float(rho)
        singleton = math.exp(-rho)
        pair = rho * math.exp(-rho)
        higher = 1.0 - (1.0 + rho) * math.exp(-rho)
        for method, fraction in [
            ("Singleton", singleton),
            ("Two-collision", pair),
            ("Three-or-more collision", higher),
        ]:
            rows.append(
                {
                    "rho": rho,
                    "fraction": fraction,
                    "method": method,
                    "source_type": "formula",
                    "m": DEFAULT_M,
                    "N": ceil_slots(DEFAULT_M, rho),
                    "M": ceil_release(DEFAULT_M, DEFAULT_TAU),
                    "tau": DEFAULT_TAU,
                    "eps0": DEFAULT_EPS0,
                    "delta": DEFAULT_DELTA,
                    "decoder": "report_side_collision_regime",
                    "bound": "poissonization",
                }
            )
    write_csv(DATA_DIR / "fig_collision_regimes.csv", rows)
    return rows


def generate_utility_factors() -> list[dict]:
    rows: list[dict] = []
    for rho in RHO_GRID:
        rho = float(rho)
        for method, factor, decoder in [
            ("Drop retained mass", math.exp(-rho), "drop"),
            ("Drop SE factor", math.exp(rho / 2.0), "drop"),
            ("Pool additive mass", 1.0, "pool"),
        ]:
            rows.append(
                {
                    "rho": rho,
                    "factor": factor,
                    "method": method,
                    "source_type": "formula",
                    "m": DEFAULT_M,
                    "N": ceil_slots(DEFAULT_M, rho),
                    "M": ceil_release(DEFAULT_M, DEFAULT_TAU),
                    "tau": DEFAULT_TAU,
                    "eps0": DEFAULT_EPS0,
                    "delta": DEFAULT_DELTA,
                    "decoder": decoder,
                    "bound": "poissonization_utility_factor",
                }
            )
    write_csv(DATA_DIR / "fig_utility_factors.csv", rows)
    return rows


def generate_frontier() -> list[dict]:
    rows: list[dict] = []
    m = DEFAULT_M
    eps0 = DEFAULT_EPS0
    delta = DEFAULT_DELTA
    tau = DEFAULT_TAU
    M = ceil_release(m, tau)
    for rho in RHO_GRID:
        rho = float(rho)
        rows.append(
            {
                "rho": rho,
                "utility_cost": math.exp(rho / 2.0),
                "epsilon": eps_drop_proxy(eps0, rho, M, delta),
                "method": "Drop (matched padding)",
                "source_type": "formula",
                "m": m,
                "N": ceil_slots(m, rho),
                "M": M,
                "tau": tau,
                "eps0": eps0,
                "delta": delta,
                "decoder": "drop",
                "bound": "second_order_matched_drop",
            }
        )
        rows.append(
            {
                "rho": rho,
                "utility_cost": 1.0,
                "epsilon": eps_pool_replacement(eps0, rho, M, delta),
                "method": "Pool (replacement)",
                "source_type": "formula",
                "m": m,
                "N": ceil_slots(m, rho),
                "M": M,
                "tau": tau,
                "eps0": eps0,
                "delta": delta,
                "decoder": "pool",
                "bound": "second_order_replacement",
            }
        )
    write_csv(DATA_DIR / "fig_privacy_utility_frontier.csv", rows)
    return rows


def generate_batch_gating_delay() -> list[dict]:
    rows: list[dict] = []
    arrival_rate = 20_000.0
    for m in [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]:
        collect_delay = (m - 1.0) / (2.0 * arrival_rate)
        rows.append(
            {
                "m": m,
                "delay_seconds": collect_delay,
                "method": "Collect-then-shuffle",
                "source_type": "analytic_cost",
                "arrival_rate_reports_per_s": arrival_rate,
                "N": "",
                "rho": "",
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "batch_gate",
                "bound": "average_waiting_time",
            }
        )
        rows.append(
            {
                "m": m,
                "delay_seconds": 0.0,
                "method": "Chafe online anonymization",
                "source_type": "analytic_cost",
                "arrival_rate_reports_per_s": arrival_rate,
                "N": "",
                "rho": "",
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "hidden_slot_online",
                "bound": "ingestion_waiting_time",
            }
        )
    write_csv(DATA_DIR / "fig_batch_gating_delay.csv", rows)
    return rows


def generate_privacy_surface_rho_tau() -> list[dict]:
    rows: list[dict] = []
    m = DEFAULT_M
    eps0 = DEFAULT_EPS0
    delta = DEFAULT_DELTA
    for tau in TAU_GRID:
        tau = float(tau)
        M = ceil_release(m, tau)
        for rho in RHO_GRID:
            rho = float(rho)
            rows.append(
                base_row(
                    method="Drop (matched padding)",
                    epsilon=eps_drop_proxy(eps0, rho, M, delta),
                    source_type="formula",
                    m=m,
                    rho=rho,
                    tau=tau,
                    eps0=eps0,
                    delta=delta,
                    decoder="drop",
                    bound="second_order_matched_drop_surface",
                )
            )
            rows.append(
                base_row(
                    method="Pool (replacement)",
                    epsilon=eps_pool_replacement(eps0, rho, M, delta),
                    source_type="formula",
                    m=m,
                    rho=rho,
                    tau=tau,
                    eps0=eps0,
                    delta=delta,
                    decoder="pool",
                    bound="second_order_replacement_surface",
                )
            )
            rows.append(
                base_row(
                    method="Pool (add/remove)",
                    epsilon=eps_pool_add_remove(eps0, rho, M, delta),
                    source_type="formula",
                    m=m,
                    rho=rho,
                    tau=tau,
                    eps0=eps0,
                    delta=delta,
                    decoder="pool",
                    bound="second_order_add_remove_surface",
                )
            )
    write_csv(DATA_DIR / "fig_privacy_surface_rho_tau.csv", rows)
    return rows


def generate_all() -> dict[str, int]:
    generated = {
        "fig_privacy_vs_rho.csv": len(generate_privacy_vs_rho()),
        "fig_privacy_vs_eps0.csv": len(generate_privacy_vs_eps0()),
        "fig_privacy_vs_tau.csv": len(generate_privacy_vs_tau()),
        "fig_collision_regimes.csv": len(generate_collision_regimes()),
        "fig_utility_factors.csv": len(generate_utility_factors()),
        "fig_privacy_utility_frontier.csv": len(generate_frontier()),
        "fig_batch_gating_delay.csv": len(generate_batch_gating_delay()),
        "fig_privacy_surface_rho_tau.csv": len(generate_privacy_surface_rho_tau()),
    }
    return generated


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
