from __future__ import annotations

import csv
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from chafe_eval_lib import DATA_DIR, ROOT, mean_and_se, read_csv, write_csv


DEFAULT_DPF_LOG_N = 17
DEFAULT_DPF_N = 1 << DEFAULT_DPF_LOG_N
LATENCY_EPOCH_M = 100_000
SCALING_LOG_NS = [12, 14, 16, 17, 18]
PACKED_METHOD = "Google DPF packed64"
TUPLE_METHOD = "Google DPF tuple product"
DUAL_METHOD = "Google DPF dual-track"
CLARION_SHUFFLE_ENV = "CLARION_SHUFFLE_SECONDS_AT_100K"
DEFAULT_CLARION_SHUFFLE_SECONDS_AT_100K = 5.0


def platform_metadata() -> dict[str, str]:
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count() or ""),
    }


def product_valued_kernel_proxy(N: int, reports: np.ndarray, seed: int) -> int:
    rng = np.random.default_rng(seed)
    occupancy = np.zeros(N, dtype=np.uint64)
    payload = np.zeros(N, dtype=np.uint64)
    mask = np.uint64((1 << 16) - 1)
    for report in reports:
        expanded = rng.integers(0, 1 << 32, size=N, dtype=np.uint64)
        occupancy += expanded & np.uint64(1)
        payload += ((expanded >> np.uint64(1)) & mask) * np.uint64(report + 1)
    return int((occupancy.sum() + payload.sum()) & np.uint64((1 << 63) - 1))


def dual_track_kernel_proxy(N: int, reports: np.ndarray, seed: int) -> int:
    rng = np.random.default_rng(seed)
    occupancy = np.zeros(N, dtype=np.uint64)
    payload = np.zeros(N, dtype=np.uint64)
    mask = np.uint64((1 << 16) - 1)
    for _report in reports:
        occ_expanded = rng.integers(0, 1 << 32, size=N, dtype=np.uint64)
        occupancy += occ_expanded & np.uint64(1)
    for report in reports:
        payload_expanded = rng.integers(0, 1 << 32, size=N, dtype=np.uint64)
        payload += (payload_expanded & mask) * np.uint64(report + 1)
    return int((occupancy.sum() + payload.sum()) & np.uint64((1 << 63) - 1))


def time_kernel(fn, N: int, reports: np.ndarray, repeats: int) -> list[float]:
    times: list[float] = []
    fn(N, reports[:2], 9_999)
    for repeat in range(repeats):
        start = time.perf_counter()
        checksum = fn(N, reports, 10_000 + repeat)
        elapsed = time.perf_counter() - start
        if checksum == -1:
            raise RuntimeError("unreachable checksum guard")
        times.append(elapsed / len(reports))
    return times


def generate_microbench() -> list[dict]:
    google_rows = generate_google_dpf_microbench()
    if google_rows is not None:
        return google_rows
    if os.environ.get("CHAFE_ALLOW_DPF_PROXY") != "1":
        raise RuntimeError(
            "Google DPF benchmark binary not found. Build it with:\n"
            "cd third_party/distributed_point_functions && "
            "env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH "
            "BAZELISK_HOME=/private/tmp/chafe-bazelisk-7 USE_BAZEL_VERSION=7.4.1 "
            "bazel --output_user_root=/private/tmp/chafe-bazel-output-7 "
            "build //chafe:google_dpf_write_benchmark "
            "--macos_minimum_os=10.14 --host_copt=-mmacosx-version-min=10.14 "
            "--host_linkopt=-mmacosx-version-min=10.14 --copt=-nostdinc "
            "--copt=-isystem/Library/Developer/CommandLineTools/usr/include/c++/v1 "
            "--copt=-isystem/Library/Developer/CommandLineTools/usr/lib/clang/12.0.0/include "
            "--copt=-isystem/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include "
            "--copt=-iframework/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks"
        )
    return generate_numpy_proxy_microbench()


def google_dpf_binary() -> Path:
    override = os.environ.get("CHAFE_GOOGLE_DPF_BENCHMARK")
    if override:
        return Path(override)
    return ROOT / "third_party" / "distributed_point_functions" / "bazel-bin" / "chafe" / "google_dpf_write_benchmark"


def run_google_dpf(args: list[str]) -> list[dict[str, str]]:
    binary = google_dpf_binary()
    if not binary.exists():
        raise FileNotFoundError(str(binary))
    cmd = [str(binary), *args]
    result = subprocess.run(
        cmd,
        cwd=str(binary.parents[2]),
        check=True,
        text=True,
        capture_output=True,
    )
    csv_text = "\n".join(
        line for line in result.stdout.splitlines() if line.startswith("method,") or line.startswith("Google DPF")
    )
    if not csv_text.strip():
        raise RuntimeError(f"Google DPF benchmark emitted no CSV rows:\n{result.stdout}\n{result.stderr}")
    return list(csv.DictReader(csv_text.splitlines()))


def enrich_google_summary_row(row: dict[str, str]) -> dict:
    time_ratio = row.get("time_ratio_vs_packed", row.get("speedup_vs_product", ""))
    return {
        "method": row["method"],
        "time_ms_per_report": float(row["time_ms_per_report"]),
        "standard_error_ms": float(row["standard_error_ms"]),
        "p50_ms_per_report": float(row.get("p50_ms_per_report", row["time_ms_per_report"])),
        "p95_ms_per_report": float(row.get("p95_ms_per_report", row["time_ms_per_report"])),
        "normalized_expansion_work": float(row["normalized_expansion_work"]),
        "time_ratio_vs_packed": float(time_ratio),
        "speedup_vs_product": float(time_ratio),
        "source_type": row["source_type"],
        "benchmark": row["benchmark"],
        "N": int(row["N"]),
        "log_n": int(row["log_n"]),
        "reports_per_repeat": int(row["reports_per_repeat"]),
        "repeats": int(row["repeats"]),
        "threads": int(row.get("threads", "1")),
        "cpu_acceleration": row.get("cpu_acceleration", ""),
        "checksum": row["checksum"],
        "packed_share_field_extraction_verified": str(
            row["method"] == PACKED_METHOD
        ).lower(),
        "reports_per_second": 1000.0 / float(row["time_ms_per_report"]),
        "m": "",
        "rho": "",
        "M": "",
        "tau": "",
        "eps0": "",
        "delta": "",
        "decoder": "google_dpf_ingestion_kernel",
        "build_target": "//chafe:google_dpf_write_benchmark",
        "timed_scope": "client key generation, both server evaluations and accumulation in one process, worker-array reduction, and reconstruction verification",
        "deployment_scope": "combined-work single-process kernel; not a two-server network deployment",
        **platform_metadata(),
    }


def enrich_google_breakdown_row(row: dict[str, str]) -> dict:
    return {
        "method": row["method"],
        "phase": row["phase"],
        "time_ms_per_report": float(row["time_ms_per_report"]),
        "standard_error_ms": float(row["standard_error_ms"]),
        "p50_ms_per_report": float(row.get("p50_ms_per_report", row["time_ms_per_report"])),
        "p95_ms_per_report": float(row.get("p95_ms_per_report", row["time_ms_per_report"])),
        "key_bytes_per_report": float(row["key_bytes_per_report"]),
        "normalized_expansion_work": float(row["normalized_expansion_work"]),
        "source_type": row["source_type"],
        "benchmark": row["benchmark"],
        "N": int(row["N"]),
        "log_n": int(row["log_n"]),
        "reports_per_repeat": int(row["reports_per_repeat"]),
        "repeats": int(row["repeats"]),
        "threads": int(row.get("threads", "1")),
        "cpu_acceleration": row.get("cpu_acceleration", ""),
        "checksum": row["checksum"],
        "packed_share_field_extraction_verified": str(
            row["method"] == PACKED_METHOD
        ).lower(),
        "m": "",
        "rho": "",
        "M": "",
        "tau": "",
        "eps0": "",
        "delta": "",
        "decoder": "google_dpf_ingestion_kernel",
        "build_target": "//chafe:google_dpf_write_benchmark",
        "timed_scope": "phase timer inside the combined-work single-process kernel; total additionally includes worker-array reduction and reconstruction verification",
        **platform_metadata(),
    }


def generate_google_dpf_microbench() -> list[dict] | None:
    binary = google_dpf_binary()
    if not binary.exists():
        return None
    rows = run_google_dpf(
        [
            f"--log_n={DEFAULT_DPF_LOG_N}",
            "--reports=64",
            "--repeats=7",
            "--threads=1",
            f"--seed={40_000 + DEFAULT_DPF_LOG_N}",
        ]
    )
    if len(rows) != 3:
        raise RuntimeError(f"unexpected Google DPF benchmark row count: {len(rows)}")
    enriched = [enrich_google_summary_row(row) for row in rows]
    write_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv", enriched)
    return enriched


def generate_google_dpf_scaling() -> list[dict]:
    rows: list[dict] = []
    for log_n in SCALING_LOG_NS:
        reports = 32 if log_n <= 14 else 16 if log_n == 16 else 12 if log_n == 17 else 8
        repeats = 7 if log_n <= 14 else 5
        raw_rows = run_google_dpf(
            [
                f"--log_n={log_n}",
                f"--reports={reports}",
                f"--repeats={repeats}",
                "--threads=1",
                f"--seed={50_000 + log_n}",
            ]
        )
        if len(raw_rows) != 3:
            raise RuntimeError(f"unexpected Google DPF scaling row count for log_n={log_n}: {len(raw_rows)}")
        rows.extend(enrich_google_summary_row(row) for row in raw_rows)
    write_csv(DATA_DIR / "fig_dpf_scaling.csv", rows)
    return rows


def generate_dpf_throughput_from_scaling(scaling_rows: list[dict] | None = None) -> list[dict]:
    if scaling_rows is None:
        scaling_rows = read_csv(DATA_DIR / "fig_dpf_scaling.csv")
    rows: list[dict] = []
    for row in scaling_rows:
        time_ms = float(row["time_ms_per_report"])
        rows.append(
            {
                "method": row["method"],
                "reports_per_second": 1000.0 / time_ms,
                "time_ms_per_report": time_ms,
                "standard_error_ms": row["standard_error_ms"],
                "p50_ms_per_report": row.get("p50_ms_per_report", row["time_ms_per_report"]),
                "p95_ms_per_report": row.get("p95_ms_per_report", row["time_ms_per_report"]),
                "time_ratio_vs_packed": row.get("time_ratio_vs_packed", row.get("speedup_vs_product", "")),
                "speedup_vs_product": row.get("speedup_vs_product", ""),
                "source_type": "microbenchmark",
                "benchmark": row["benchmark"],
                "N": row["N"],
                "log_n": row["log_n"],
                "reports_per_repeat": row["reports_per_repeat"],
                "repeats": row["repeats"],
                "threads": row.get("threads", 1),
                "cpu_acceleration": row.get("cpu_acceleration", ""),
                "m": "",
                "rho": "",
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "google_dpf_ingestion_kernel",
                "derived_from": "fig_dpf_scaling.csv",
                "model": "reciprocal_of_measured_time",
                "build_target": row["build_target"],
                **platform_metadata(),
            }
        )
    write_csv(DATA_DIR / "fig_dpf_throughput_vs_n.csv", rows)
    return rows


def clarion_shuffle_seconds(epoch_m: int) -> float | None:
    value = os.environ.get(CLARION_SHUFFLE_ENV, "").strip()
    seconds_at_100k = DEFAULT_CLARION_SHUFFLE_SECONDS_AT_100K if not value else float(value)
    if seconds_at_100k < 0:
        raise ValueError(f"{CLARION_SHUFFLE_ENV} must be non-negative")
    return seconds_at_100k * (epoch_m / 100_000.0)


def select_latency_service_row(
    scaling_rows: list[dict],
    thread_rows: list[dict] | None,
) -> tuple[dict, str]:
    candidates: list[dict] = []
    if thread_rows is not None:
        candidates = [
            row
            for row in thread_rows
            if row["method"] == PACKED_METHOD and int(float(row["N"])) == DEFAULT_DPF_N
        ]
    if candidates:
        return max(candidates, key=lambda row: float(row["reports_per_second"])), "fig_dpf_thread_scaling.csv"
    product_rows = [
        row
        for row in scaling_rows
        if row["method"] == PACKED_METHOD and int(float(row["N"])) == DEFAULT_DPF_N
    ]
    if not product_rows:
        product_rows = [row for row in scaling_rows if row["method"] == PACKED_METHOD]
    return min(product_rows, key=lambda row: abs(int(float(row["N"])) - DEFAULT_DPF_N)), "fig_dpf_scaling.csv"


def generate_ingestion_latency_model(
    scaling_rows: list[dict] | None = None,
    thread_rows: list[dict] | None = None,
) -> list[dict]:
    if scaling_rows is None:
        scaling_rows = read_csv(DATA_DIR / "fig_dpf_scaling.csv")
    if thread_rows is None and (DATA_DIR / "fig_dpf_thread_scaling.csv").exists():
        thread_rows = read_csv(DATA_DIR / "fig_dpf_thread_scaling.csv")
    service_row, service_source = select_latency_service_row(scaling_rows, thread_rows)
    service_ms = float(service_row["time_ms_per_report"])
    service_rate = 1000.0 / service_ms
    service_seconds = service_ms / 1000.0
    epoch_m = LATENCY_EPOCH_M
    service_n = int(float(service_row["N"]))
    service_threads = int(float(service_row.get("threads", "1") or "1"))
    service_rho = epoch_m / service_n
    clarion_seconds = clarion_shuffle_seconds(epoch_m)
    stable_rates = [
        1,
        2,
        5,
        10,
        20,
        40,
        60,
        80,
        90,
        95,
        98,
        100,
        101,
        102,
        103,
        200,
        500,
        1_000,
        2_000,
        2_500,
        0.25 * service_rate,
        0.50 * service_rate,
        0.75 * service_rate,
        0.90 * service_rate,
        0.95 * service_rate,
        0.98 * service_rate,
        0.995 * service_rate,
        0.999 * service_rate,
        0.9999 * service_rate,
    ]
    overload_rates = [
        1.0001 * service_rate,
        1.001 * service_rate,
        1.01 * service_rate,
        1.05 * service_rate,
        1.1 * service_rate,
        200,
        500,
        1_000,
        5_000,
        20_000,
    ]
    batch_rates = sorted(
        {
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
            40.0,
            60.0,
            80.0,
            90.0,
            100.0,
            service_rate,
            200.0,
            500.0,
            1_000.0,
            2_000.0,
            5_000.0,
            20_000.0,
        }
    )

    def md1_latency(arrival_rate: float) -> float:
        utilization = arrival_rate / service_rate
        return service_seconds + arrival_rate * service_seconds * service_seconds / (2.0 * (1.0 - utilization))

    def batch_delay(arrival_rate: float) -> float:
        return (epoch_m - 1.0) / (2.0 * arrival_rate)

    def collect_shuffle_delay(arrival_rate: float) -> float:
        if clarion_seconds is None:
            return batch_delay(arrival_rate)
        return batch_delay(arrival_rate) + clarion_seconds

    left, right = 0.99 * service_rate, (1.0 - 1e-12) * service_rate
    for _ in range(120):
        mid = 0.5 * (left + right)
        if md1_latency(mid) < collect_shuffle_delay(mid):
            left = mid
        else:
            right = mid
    crossover_rate = right
    crossover_delay = collect_shuffle_delay(crossover_rate)
    rows: list[dict] = []
    for arrival_rate in sorted({float(rate) for rate in stable_rates if rate < service_rate}):
        utilization = arrival_rate / service_rate
        chafe_latency = md1_latency(arrival_rate)
        rows.append(
            {
                "arrival_rate_reports_per_s": arrival_rate,
                "service_rate_reports_per_s": service_rate,
                "utilization": utilization,
                "avg_latency_seconds": chafe_latency,
                "batch_gating_seconds": 0.0,
                "backlog_growth_reports_per_s": 0.0,
                "stable": "true",
                "method": "Chafe online DPF service",
                "source_type": "analytic_cost",
                "benchmark": "md1_queue_from_measured_google_dpf",
                "N": service_n,
                "log_n": service_row["log_n"],
                "m": epoch_m,
                "rho": service_rho,
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "hidden_slot_online",
                "threads": service_threads,
                "shuffle_seconds": "",
                "derived_from": service_source,
                "model": "M/D/1 mean sojourn time from measured multi-worker service rate; unstable rows record backlog growth",
            }
        )
    for arrival_rate in sorted({float(rate) for rate in overload_rates if rate > service_rate}):
        backlog = arrival_rate - service_rate
        rows.append(
            {
                "arrival_rate_reports_per_s": arrival_rate,
                "service_rate_reports_per_s": service_rate,
                "utilization": arrival_rate / service_rate,
                "avg_latency_seconds": epoch_m / backlog,
                "batch_gating_seconds": 0.0,
                "backlog_growth_reports_per_s": backlog,
                "stable": "false",
                "method": "Chafe overload horizon",
                "source_type": "analytic_cost",
                "benchmark": "overload_backlog_model_from_measured_google_dpf",
                "N": service_n,
                "log_n": service_row["log_n"],
                "m": epoch_m,
                "rho": service_rho,
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "hidden_slot_online",
                "threads": service_threads,
                "shuffle_seconds": "",
                "derived_from": service_source,
                "model": "time to accumulate one epoch of backlog after lambda exceeds service rate",
            }
        )
    for arrival_rate in batch_rates:
        rows.append(
            {
                "arrival_rate_reports_per_s": arrival_rate,
                "service_rate_reports_per_s": "",
                "utilization": "",
                "avg_latency_seconds": batch_delay(arrival_rate),
                "batch_gating_seconds": batch_delay(arrival_rate),
                "backlog_growth_reports_per_s": 0.0,
                "stable": "true",
                "method": "Collect-then-shuffle gate",
                "source_type": "analytic_cost",
                "benchmark": "batch_gating_wait",
                "N": "",
                "log_n": "",
                "m": epoch_m,
                "rho": "",
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "batch_gate",
                "threads": "",
                "shuffle_seconds": 0.0,
                "derived_from": "closed_form_waiting_time",
                "model": "(m-1)/(2 lambda)",
            }
        )
        if clarion_seconds is not None:
            rows.append(
                {
                    "arrival_rate_reports_per_s": arrival_rate,
                    "service_rate_reports_per_s": "",
                    "utilization": "",
                    "avg_latency_seconds": batch_delay(arrival_rate) + clarion_seconds,
                    "batch_gating_seconds": batch_delay(arrival_rate),
                    "backlog_growth_reports_per_s": 0.0,
                    "stable": "true",
                    "method": "Collect gate + Clarion shuffle",
                    "source_type": "analytic_cost",
                    "benchmark": "batch_gating_plus_literature_shuffle_time",
                    "N": "",
                    "log_n": "",
                    "m": epoch_m,
                    "rho": "",
                    "M": "",
                    "tau": "",
                    "eps0": "",
                    "delta": "",
                    "decoder": "batch_secret_shared_shuffle",
                    "threads": "",
                    "shuffle_seconds": clarion_seconds,
                    "derived_from": f"conservative Clarion-style 4-5s estimate; {CLARION_SHUFFLE_ENV} override; cite eskandarian2021clarion",
                    "model": "(m-1)/(2 lambda) + conservative Clarion-style online shuffle time at m=1e5, scaled linearly in m",
                }
            )
    rows.append(
        {
            "arrival_rate_reports_per_s": crossover_rate,
            "service_rate_reports_per_s": service_rate,
            "utilization": crossover_rate / service_rate,
            "avg_latency_seconds": crossover_delay,
            "batch_gating_seconds": batch_delay(crossover_rate),
            "backlog_growth_reports_per_s": 0.0,
            "stable": "true",
            "method": "Queue/gate equality",
            "source_type": "analytic_cost",
            "benchmark": "md1_queue_and_batch_gate_intersection",
            "N": service_n,
            "log_n": service_row["log_n"],
            "m": epoch_m,
            "rho": service_rho,
            "M": "",
            "tau": "",
            "eps0": "",
            "delta": "",
            "decoder": "hidden_slot_online_vs_batch_gate",
            "threads": service_threads,
            "shuffle_seconds": "" if clarion_seconds is None else clarion_seconds,
            "derived_from": service_source
            if clarion_seconds is None
            else f"{service_source}; conservative Clarion-style 4-5s estimate; {CLARION_SHUFFLE_ENV} override",
            "model": "binary search for M/D/1 latency equal to collect gate plus optional Clarion shuffle time",
        }
    )
    write_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv", rows)
    return rows


def generate_ingestion_joint_scaling(
    scaling_rows: list[dict] | None = None,
    thread_rows: list[dict] | None = None,
) -> list[dict]:
    """Project node-by-thread capacity from measured packed64 host anchors.

    The Google-DPF benchmark is a combined-work, single-process kernel: it
    includes client key generation, both server evaluations, accumulation,
    reduction, and reconstruction.  Consequently, ``node_equivalents`` below
    are units of that measured work, not measurements from a physical
    two-server cluster.  The scenario deliberately uses ideal independent
    replication, ``R(k,t)=k r_t``, as a transparent capacity calculation and
    separately records a 70% utilization envelope.  Only the ``k=1`` input is
    measured; every multi-equivalent value is an optimistic analytic output.
    """
    if scaling_rows is None:
        scaling_rows = read_csv(DATA_DIR / "fig_dpf_scaling.csv")
    if thread_rows is None and (DATA_DIR / "fig_dpf_thread_scaling.csv").exists():
        thread_rows = read_csv(DATA_DIR / "fig_dpf_thread_scaling.csv")
    if not thread_rows:
        raise RuntimeError("joint scaling requires fig_dpf_thread_scaling.csv")

    packed_rows = sorted(
        (row for row in thread_rows if row["method"] == PACKED_METHOD),
        key=lambda row: int(float(row["threads"])),
    )
    expected_threads = [1, 2, 4, 8]
    if [int(float(row["threads"])) for row in packed_rows] != expected_threads:
        raise RuntimeError("packed64 joint scaling requires measured 1/2/4/8-thread anchors")
    if {int(float(row["N"])) for row in packed_rows} != {DEFAULT_DPF_N}:
        raise RuntimeError("packed64 joint scaling anchors must use N=2^17")

    node_equivalents = [1, 2, 3, 4, 8, 16, 32, 64, 128, 160]
    target_rate = 20_000.0
    target_utilization = 0.70
    node_scaling_exponent = 1.0
    rows: list[dict] = []
    one_thread_rate = float(packed_rows[0]["reports_per_second"])
    for anchor in packed_rows:
        threads = int(float(anchor["threads"]))
        measured_rate = float(anchor["reports_per_second"])
        speedup = measured_rate / one_thread_rate
        parallel_efficiency = speedup / threads
        required_raw = math.ceil(target_rate / measured_rate)
        required_planning = math.ceil(
            target_rate / (target_utilization * measured_rate)
        )
        for node_count in node_equivalents:
            node_scaling_efficiency = 1.0
            projected_raw = measured_rate * node_count
            rows.append(
                {
                    "method": "Chafe packed64 CPU projection",
                    "backend": PACKED_METHOD,
                    "N": int(float(anchor["N"])),
                    "log_n": int(float(anchor["log_n"])),
                    "m": LATENCY_EPOCH_M,
                    "rho": LATENCY_EPOCH_M / int(float(anchor["N"])),
                    "threads_per_node": threads,
                    "node_equivalents": node_count,
                    "measured_single_node_rate_reports_per_s": measured_rate,
                    "thread_speedup_vs_1": speedup,
                    "thread_parallel_efficiency": parallel_efficiency,
                    "projected_raw_rate_reports_per_s": projected_raw,
                    "target_utilization": target_utilization,
                    "projected_sustainable_rate_reports_per_s": target_utilization
                    * projected_raw,
                    "node_scaling_efficiency_assumed": node_scaling_efficiency,
                    "node_scaling_exponent": node_scaling_exponent,
                    "plot_role": "measured_input"
                    if node_count == 1
                    else "ideal_capacity_projection",
                    "required_node_equivalents_for_20k_raw": required_raw,
                    "required_node_equivalents_for_20k_at_target_utilization": required_planning,
                    "source_type": "analytic_cost",
                    "benchmark": "joint_node_thread_capacity_projection_from_measured_google_dpf",
                    "derived_from": "fig_dpf_thread_scaling.csv",
                    "single_node_anchor_measured": "true",
                    "multi_node_measured": "false",
                    "model": "R(k,t)=k*r_t under ideal independent replication; k=1 is measured, every k>1 value is unmeasured, and the sustainable envelope is 0.70*R",
                    "timed_scope": anchor["timed_scope"],
                    "deployment_scope": "combined-work node equivalents projected from a single-process benchmark; not a physical two-server cluster measurement",
                    "platform_system": anchor["platform_system"],
                    "platform_release": anchor["platform_release"],
                    "platform_machine": anchor["platform_machine"],
                    "cpu_count": anchor["cpu_count"],
                }
            )
    write_csv(DATA_DIR / "fig_ingestion_joint_scaling.csv", rows)
    return rows


def generate_google_dpf_phase_breakdown() -> list[dict]:
    raw_rows = run_google_dpf(
        [
            f"--log_n={DEFAULT_DPF_LOG_N}",
            "--reports=32",
            "--repeats=7",
            "--threads=1",
            "--breakdown",
            f"--seed={60_000 + DEFAULT_DPF_LOG_N}",
        ]
    )
    if len(raw_rows) != 15:
        raise RuntimeError(f"unexpected Google DPF breakdown row count: {len(raw_rows)}")
    rows = [enrich_google_breakdown_row(row) for row in raw_rows]
    write_csv(DATA_DIR / "fig_dpf_phase_breakdown.csv", rows)
    return rows


def thread_grid() -> list[int]:
    cpu_count = max(1, os.cpu_count() or 1)
    max_thread = min(8, cpu_count)
    grid = {1, max_thread}
    for value in [2, 4, 8]:
        if value <= max_thread:
            grid.add(value)
    return sorted(grid)


def generate_google_dpf_thread_scaling() -> list[dict]:
    rows: list[dict] = []
    for threads in thread_grid():
        raw_rows = run_google_dpf(
            [
                f"--log_n={DEFAULT_DPF_LOG_N}",
                "--reports=128",
                "--repeats=7",
                f"--threads={threads}",
                f"--seed={70_000 + threads}",
            ]
        )
        if len(raw_rows) != 3:
            raise RuntimeError(f"unexpected Google DPF thread-scaling row count for threads={threads}: {len(raw_rows)}")
        rows.extend(enrich_google_summary_row(row) for row in raw_rows)

    single_thread = {
        row["method"]: float(row["time_ms_per_report"])
        for row in rows
        if int(float(row["threads"])) == 1
    }
    for row in rows:
        baseline_ms = single_thread[row["method"]]
        row["thread_speedup_vs_1"] = baseline_ms / float(row["time_ms_per_report"])
        row["parallelism_model"] = "independent per-worker accumulators; outer wall time includes the final array reduction and reconstruction check"
    write_csv(DATA_DIR / "fig_dpf_thread_scaling.csv", rows)
    return rows


def generate_numpy_proxy_microbench() -> list[dict]:
    N = DEFAULT_DPF_N
    reports_per_repeat = 64
    repeats = 9
    reports = np.arange(reports_per_repeat, dtype=np.uint64) % np.uint64(251)

    product_times = time_kernel(product_valued_kernel_proxy, N, reports, repeats)
    dual_times = time_kernel(dual_track_kernel_proxy, N, reports, repeats)
    product_mean, product_se = mean_and_se(product_times)
    dual_mean, dual_se = mean_and_se(dual_times)
    speedup = dual_mean / product_mean

    rows = [
        {
            "method": "Product-valued kernel",
            "time_ms_per_report": product_mean * 1_000.0,
            "standard_error_ms": product_se * 1_000.0,
            "normalized_expansion_work": 1.0,
            "speedup_vs_product": 1.0,
            "source_type": "microbenchmark",
            "benchmark": "numpy_dense_dpf_expansion_proxy",
            "N": N,
            "reports_per_repeat": reports_per_repeat,
            "repeats": repeats,
            "threads": 1,
            "cpu_acceleration": "numpy_vectorized",
            "reports_per_second": 1_000.0 / (product_mean * 1_000.0),
            "m": "",
            "rho": "",
            "M": "",
            "tau": "",
            "eps0": "",
            "delta": "",
            "decoder": "ingestion_kernel",
            "log_n": 16,
            "checksum": "",
            "build_target": "",
            **platform_metadata(),
        },
        {
            "method": "Dual-track kernel",
            "time_ms_per_report": dual_mean * 1_000.0,
            "standard_error_ms": dual_se * 1_000.0,
            "normalized_expansion_work": 2.0,
            "speedup_vs_product": speedup,
            "source_type": "microbenchmark",
            "benchmark": "numpy_dense_dpf_expansion_proxy",
            "N": N,
            "reports_per_repeat": reports_per_repeat,
            "repeats": repeats,
            "threads": 1,
            "cpu_acceleration": "numpy_vectorized",
            "reports_per_second": 1_000.0 / (dual_mean * 1_000.0),
            "m": "",
            "rho": "",
            "M": "",
            "tau": "",
            "eps0": "",
            "delta": "",
            "decoder": "ingestion_kernel",
            "log_n": 16,
            "checksum": "",
            "build_target": "",
            **platform_metadata(),
        },
    ]
    write_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv", rows)
    return rows


def generate_all() -> dict[str, int]:
    rows = generate_microbench()
    outputs = {"fig_product_vs_dual_microbench.csv": len(rows)}
    if google_dpf_binary().exists():
        scaling_rows = generate_google_dpf_scaling()
        breakdown_rows = generate_google_dpf_phase_breakdown()
        thread_rows = generate_google_dpf_thread_scaling()
        throughput_rows = generate_dpf_throughput_from_scaling(scaling_rows)
        latency_rows = generate_ingestion_latency_model(scaling_rows, thread_rows)
        node_rows = generate_ingestion_joint_scaling(scaling_rows, thread_rows)
        outputs["fig_dpf_scaling.csv"] = len(scaling_rows)
        outputs["fig_dpf_phase_breakdown.csv"] = len(breakdown_rows)
        outputs["fig_dpf_thread_scaling.csv"] = len(thread_rows)
        outputs["fig_dpf_throughput_vs_n.csv"] = len(throughput_rows)
        outputs["fig_ingestion_latency_vs_rate.csv"] = len(latency_rows)
        outputs["fig_ingestion_joint_scaling.csv"] = len(node_rows)
    return outputs


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
