from __future__ import annotations

import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from chafe_eval_lib import DATA_DIR, ROOT, mean_and_se, read_csv, write_csv


DEFAULT_DPF_LOG_N = 17
DEFAULT_DPF_N = 1 << DEFAULT_DPF_LOG_N
SCALING_LOG_NS = [12, 14, 16, 17, 18]
PACKED_METHOD = "Google DPF packed64"
TUPLE_METHOD = "Google DPF tuple product"
DUAL_METHOD = "Google DPF dual-track"


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
        "reports_per_second": 1000.0 / float(row["time_ms_per_report"]),
        "m": "",
        "rho": "",
        "M": "",
        "tau": "",
        "eps0": "",
        "delta": "",
        "decoder": "google_dpf_ingestion_kernel",
        "build_target": "//chafe:google_dpf_write_benchmark",
        **platform_metadata(),
    }


def enrich_google_breakdown_row(row: dict[str, str]) -> dict:
    return {
        "method": row["method"],
        "phase": row["phase"],
        "time_ms_per_report": float(row["time_ms_per_report"]),
        "standard_error_ms": float(row["standard_error_ms"]),
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
        "m": "",
        "rho": "",
        "M": "",
        "tau": "",
        "eps0": "",
        "delta": "",
        "decoder": "google_dpf_ingestion_kernel",
        "build_target": "//chafe:google_dpf_write_benchmark",
        **platform_metadata(),
    }


def generate_google_dpf_microbench() -> list[dict] | None:
    binary = google_dpf_binary()
    if not binary.exists():
        return None
    rows = run_google_dpf(
        [
            f"--log_n={DEFAULT_DPF_LOG_N}",
            "--reports=4",
            "--repeats=3",
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
        reports = 8 if log_n <= 14 else 4 if log_n == 16 else 3 if log_n == 17 else 2
        repeats = 5 if log_n <= 14 else 3 if log_n <= 17 else 2
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


def generate_ingestion_latency_model(scaling_rows: list[dict] | None = None) -> list[dict]:
    if scaling_rows is None:
        scaling_rows = read_csv(DATA_DIR / "fig_dpf_scaling.csv")
    product_rows = [
        row
        for row in scaling_rows
        if row["method"] == PACKED_METHOD and int(float(row["N"])) == DEFAULT_DPF_N
    ]
    if not product_rows:
        product_rows = [row for row in scaling_rows if row["method"] == PACKED_METHOD]
    service_row = min(product_rows, key=lambda row: abs(int(float(row["N"])) - DEFAULT_DPF_N))
    service_ms = float(service_row["time_ms_per_report"])
    service_rate = 1000.0 / service_ms
    service_seconds = service_ms / 1000.0
    epoch_m = 1_000_000
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

    left, right = 0.99 * service_rate, (1.0 - 1e-12) * service_rate
    for _ in range(120):
        mid = 0.5 * (left + right)
        if md1_latency(mid) < batch_delay(mid):
            left = mid
        else:
            right = mid
    crossover_rate = right
    crossover_delay = batch_delay(crossover_rate)
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
                "N": int(float(service_row["N"])),
                "log_n": service_row["log_n"],
                "m": epoch_m,
                "rho": "",
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "hidden_slot_online",
                "derived_from": "fig_dpf_scaling.csv",
                "model": "M/D/1 mean sojourn time; unstable rows record backlog growth",
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
                "N": int(float(service_row["N"])),
                "log_n": service_row["log_n"],
                "m": epoch_m,
                "rho": "",
                "M": "",
                "tau": "",
                "eps0": "",
                "delta": "",
                "decoder": "hidden_slot_online",
                "derived_from": "fig_dpf_scaling.csv",
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
                "derived_from": "closed_form_waiting_time",
                "model": "(m-1)/(2 lambda)",
            }
        )
    rows.append(
        {
            "arrival_rate_reports_per_s": crossover_rate,
            "service_rate_reports_per_s": service_rate,
            "utilization": crossover_rate / service_rate,
            "avg_latency_seconds": crossover_delay,
            "batch_gating_seconds": crossover_delay,
            "backlog_growth_reports_per_s": 0.0,
            "stable": "true",
            "method": "Queue/gate equality",
            "source_type": "analytic_cost",
            "benchmark": "md1_queue_and_batch_gate_intersection",
            "N": int(float(service_row["N"])),
            "log_n": service_row["log_n"],
            "m": epoch_m,
            "rho": "",
            "M": "",
            "tau": "",
            "eps0": "",
            "delta": "",
            "decoder": "hidden_slot_online_vs_batch_gate",
            "derived_from": "fig_dpf_scaling.csv",
            "model": "binary search for M/D/1 latency equal to batch-gating lower bound",
        }
    )
    write_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv", rows)
    return rows


def generate_google_dpf_phase_breakdown() -> list[dict]:
    raw_rows = run_google_dpf(
        [
            f"--log_n={DEFAULT_DPF_LOG_N}",
            "--reports=3",
            "--repeats=3",
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
                "--reports=8",
                "--repeats=3",
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
        row["parallelism_model"] = "independent per-worker accumulators reduced after measurement"
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
        latency_rows = generate_ingestion_latency_model(scaling_rows)
        outputs["fig_dpf_scaling.csv"] = len(scaling_rows)
        outputs["fig_dpf_phase_breakdown.csv"] = len(breakdown_rows)
        outputs["fig_dpf_thread_scaling.csv"] = len(thread_rows)
        outputs["fig_dpf_throughput_vs_n.csv"] = len(throughput_rows)
        outputs["fig_ingestion_latency_vs_rate.csv"] = len(latency_rows)
    return outputs


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
