"""Finite-burst latency model for the ingestion comparison.

The model deliberately avoids steady-state queue formulas when the offered
load exceeds service capacity. A finite epoch of ``m`` reports arrives at a
constant rate. Chafe processes reports online and drains any remaining queue
after the last arrival; CGP starts its modeled batch shuffle only after the
epoch has been collected.
"""

from __future__ import annotations

import math

from chafe_eval_lib import DATA_DIR, read_csv, write_csv


REPORT_COUNT = 100_000
N = 1 << 17
RELEASE_LENGTH = 2 * REPORT_COUNT
ARRIVAL_RATE = 10_000.0
CPU_THREADS = 8
CPU_NODE_EQUIVALENTS = 31
NODE_SCALING_EXPONENT = 1.0
TARGET_UTILIZATION = 0.70


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return slope and intercept for ordinary least squares ``y=a*x+b``."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("a fit requires at least two paired observations")
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    denominator = sum((x - x_bar) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("fit inputs have zero x variance")
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denominator
    return slope, y_bar - slope * x_bar


def _cpu_rate_at_target_domain() -> tuple[float, float, float]:
    domain_rows = [
        row
        for row in read_csv(DATA_DIR / "fig_dpf_scaling.csv")
        if row["method"] == "Google DPF packed64"
        and row["source_type"] == "microbenchmark"
        and int(row["threads"]) == 1
    ]
    if len(domain_rows) < 4:
        raise ValueError("CPU domain fit requires the measured packed64 grid")
    slope, _ = _linear_fit(
        [math.log2(float(row["N"])) for row in domain_rows],
        [math.log2(float(row["reports_per_second"])) for row in domain_rows],
    )
    thread_rows = [
        row
        for row in read_csv(DATA_DIR / "fig_dpf_thread_scaling.csv")
        if row["method"] == "Google DPF packed64"
        and row["source_type"] == "microbenchmark"
        and int(row["threads"]) == CPU_THREADS
        and int(row["N"]) == N
    ]
    if len(thread_rows) != 1:
        raise ValueError("CPU model requires one measured eight-thread speedup")
    speedup = float(thread_rows[0]["thread_speedup_vs_1"])
    return float(thread_rows[0]["reports_per_second"]), slope, speedup


def _gpu_rate_at_target_domain() -> tuple[float, int, int]:
    rows = sorted(
        [
            row
            for row in read_csv(DATA_DIR / "fig_gpu_dpf_external.csv")
            if row["source_type"] == "external_artifact_measurement"
        ],
        key=lambda row: int(row["N"]),
    )
    exact = [row for row in rows if int(row["N"]) == N]
    if exact:
        return float(exact[0]["dpf_per_second"]), N, N
    lower = max(
        (row for row in rows if int(row["N"]) < N), key=lambda row: int(row["N"])
    )
    upper = min(
        (row for row in rows if int(row["N"]) > N), key=lambda row: int(row["N"])
    )
    log_n = math.log2(N)
    lower_log_n = math.log2(int(lower["N"]))
    upper_log_n = math.log2(int(upper["N"]))
    weight = (log_n - lower_log_n) / (upper_log_n - lower_log_n)
    log_rate = (
        (1.0 - weight) * math.log(float(lower["dpf_per_second"]))
        + weight * math.log(float(upper["dpf_per_second"]))
    )
    return math.exp(log_rate), int(lower["N"]), int(upper["N"])


def _cgp_shuffle_seconds() -> float:
    rows = [
        row
        for row in read_csv(DATA_DIR / "fig_cgp_shuffle_model.csv")
        if int(row["m"]) == REPORT_COUNT
        and row["method"] == "CGP full secret-shared shuffle (paper-derived)"
        and row["source_type"] == "literature_derived_analytic_cost"
    ]
    if len(rows) != 1:
        raise ValueError("CGP latency requires the paper-derived m=10^6 row")
    return float(rows[0]["cost_seconds"])


def _chafe_close_seconds() -> float:
    rows = [
        row
        for row in read_csv(DATA_DIR / "fig_epoch_close_scaling.csv")
        if int(row["m"]) == REPORT_COUNT
        and row["source_type"] == "microbenchmark"
        and row["correctness"] == "true"
    ]
    if {row["decoder"] for row in rows} != {"drop", "pool"}:
        raise ValueError("finite-burst model requires both measured close paths")
    return max(float(row["time_ms"]) for row in rows) / 1000.0


def _chafe_row(
    method: str,
    service_rate: float,
    close_seconds: float,
    source_type: str,
    service_scope: str,
    **metadata: float | str,
) -> dict:
    arrival_seconds = REPORT_COUNT / ARRIVAL_RATE
    processed_by_last_arrival = min(float(REPORT_COUNT), service_rate * arrival_seconds)
    backlog = float(REPORT_COUNT) - processed_by_last_arrival
    drain_seconds = backlog / service_rate
    total_seconds = arrival_seconds + drain_seconds + close_seconds
    return {
        "method": method,
        "m": REPORT_COUNT,
        "N": N,
        "M": RELEASE_LENGTH,
        "rho": REPORT_COUNT / N,
        "tau": RELEASE_LENGTH / REPORT_COUNT,
        "arrival_rate_reports_per_s": ARRIVAL_RATE,
        "arrival_seconds": arrival_seconds,
        "service_rate_reports_per_s": service_rate,
        "utilization_offered_load": ARRIVAL_RATE / service_rate,
        "processed_by_last_arrival": processed_by_last_arrival,
        "backlog_at_last_arrival": backlog,
        "backlog_fraction": backlog / REPORT_COUNT,
        "drain_seconds": drain_seconds,
        "epoch_close_seconds": close_seconds,
        "post_arrival_seconds": drain_seconds + close_seconds,
        "total_seconds": total_seconds,
        "source_type": source_type,
        "simulation": "deterministic_finite_fluid_queue",
        "service_scope": service_scope,
        **metadata,
    }


def model_rows() -> list[dict]:
    cpu_rate, cpu_domain_exponent, cpu_thread_speedup = _cpu_rate_at_target_domain()
    gpu_rate, gpu_lower_n, gpu_upper_n = _gpu_rate_at_target_domain()
    cpu_cluster_rate = cpu_rate * CPU_NODE_EQUIVALENTS
    cgp_shuffle_seconds = _cgp_shuffle_seconds()
    close_seconds = _chafe_close_seconds()
    arrival_seconds = REPORT_COUNT / ARRIVAL_RATE

    rows = [
        _chafe_row(
            "Chafe CPU 8T (measured-rate model)",
            cpu_rate,
            close_seconds,
            "analytic_cost_from_measured_cpu_inputs",
            "measured N=2^17 eight-thread combined-work rate used in the finite fluid queue",
            cpu_domain_loglog_exponent=cpu_domain_exponent,
            measured_thread_speedup=cpu_thread_speedup,
        ),
        _chafe_row(
            "Chafe GPU-kernel proxy (interpolated)",
            gpu_rate,
            close_seconds,
            "analytic_cost_from_external_gpu_kernel",
            "log-rate interpolation between published one-server V100 DPF-PIR kernel points; client, second-server, and accumulation overhead omitted",
            gpu_interpolation_lower_N=gpu_lower_n,
            gpu_interpolation_upper_N=gpu_upper_n,
            citation_key="lam2024gpudpf",
        ),
        _chafe_row(
            f"Chafe CPU {CPU_NODE_EQUIVALENTS}-equivalent (ideal capacity)",
            cpu_cluster_rate,
            close_seconds,
            "analytic_cost_from_measured_cpu_inputs",
            "ideal independent-capacity replication from the measured N=2^17 eight-thread combined-work anchor; no cluster was measured",
            node_equivalents=CPU_NODE_EQUIVALENTS,
            node_scaling_exponent=NODE_SCALING_EXPONENT,
            target_utilization=TARGET_UTILIZATION,
            sustainable_rate_reports_per_s=TARGET_UTILIZATION * cpu_cluster_rate,
        ),
        {
            "method": "CGP full model",
            "m": REPORT_COUNT,
            "N": N,
            "M": "",
            "rho": REPORT_COUNT / N,
            "tau": "",
            "arrival_rate_reports_per_s": ARRIVAL_RATE,
            "arrival_seconds": arrival_seconds,
            "service_rate_reports_per_s": REPORT_COUNT / cgp_shuffle_seconds,
            "utilization_offered_load": "",
            "processed_by_last_arrival": 0.0,
            "backlog_at_last_arrival": float(REPORT_COUNT),
            "backlog_fraction": 1.0,
            "drain_seconds": cgp_shuffle_seconds,
            "epoch_close_seconds": cgp_shuffle_seconds,
            "post_arrival_seconds": cgp_shuffle_seconds,
            "total_seconds": arrival_seconds + cgp_shuffle_seconds,
            "source_type": "arrival_plus_literature_derived_analytic_cost",
            "simulation": "deterministic_collect_then_shuffle",
            "service_scope": "10 s collection followed by the revised-ePrint two-Permute+Share cost; base OT, XOR, RTT, and implementation overhead excluded",
            "citation_key": "DBLP:conf/asiacrypt/ChaseGP20",
        },
    ]
    if not all(float(row["total_seconds"]) >= arrival_seconds for row in rows):
        raise AssertionError("end-to-end completion cannot precede the last arrival")
    return rows


def generate_all() -> dict[str, int]:
    rows = model_rows()
    path = DATA_DIR / "fig_finite_burst_latency.csv"
    write_csv(path, rows)
    return {str(path.relative_to(DATA_DIR.parent)): len(rows)}


if __name__ == "__main__":
    print(generate_all())
