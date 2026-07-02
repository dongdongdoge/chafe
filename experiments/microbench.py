from __future__ import annotations

import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from chafe_eval_lib import DATA_DIR, ROOT, mean_and_se, read_csv, write_csv


def platform_metadata() -> dict[str, str]:
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
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
    return {
        "method": row["method"],
        "time_ms_per_report": float(row["time_ms_per_report"]),
        "standard_error_ms": float(row["standard_error_ms"]),
        "normalized_expansion_work": float(row["normalized_expansion_work"]),
        "speedup_vs_product": float(row.get("speedup_vs_product", "")),
        "source_type": row["source_type"],
        "benchmark": row["benchmark"],
        "N": int(row["N"]),
        "log_n": int(row["log_n"]),
        "reports_per_repeat": int(row["reports_per_repeat"]),
        "repeats": int(row["repeats"]),
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
    rows = run_google_dpf(["--log_n=12", "--reports=8", "--repeats=5"])
    if len(rows) != 2:
        raise RuntimeError(f"unexpected Google DPF benchmark row count: {len(rows)}")
    enriched = [enrich_google_summary_row(row) for row in rows]
    write_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv", enriched)
    return enriched


def generate_google_dpf_scaling() -> list[dict]:
    rows: list[dict] = []
    for log_n in [8, 10, 12, 14, 16]:
        reports = 8 if log_n <= 14 else 4
        repeats = 5 if log_n <= 14 else 3
        raw_rows = run_google_dpf(
            [
                f"--log_n={log_n}",
                f"--reports={reports}",
                f"--repeats={repeats}",
                f"--seed={50_000 + log_n}",
            ]
        )
        if len(raw_rows) != 2:
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
                "source_type": "microbenchmark",
                "benchmark": row["benchmark"],
                "N": row["N"],
                "log_n": row["log_n"],
                "reports_per_repeat": row["reports_per_repeat"],
                "repeats": row["repeats"],
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
        row for row in scaling_rows if row["method"] == "Google DPF product-valued" and int(float(row["N"])) == 4096
    ]
    if not product_rows:
        product_rows = [row for row in scaling_rows if row["method"] == "Google DPF product-valued"]
    service_row = min(product_rows, key=lambda row: abs(int(float(row["N"])) - 4096))
    service_ms = float(service_row["time_ms_per_report"])
    service_rate = 1000.0 / service_ms
    epoch_m = 1_000_000
    arrival_rates = [1, 2, 5, 10, 20, 40, 60, 80, 90, 100, 200, 500, 1_000, 5_000, 20_000]
    rows: list[dict] = []
    for arrival_rate in arrival_rates:
        utilization = arrival_rate / service_rate
        stable = utilization < 1.0
        if stable:
            service_seconds = service_ms / 1000.0
            queue_seconds = arrival_rate * service_seconds * service_seconds / (2.0 * (1.0 - utilization))
            chafe_latency = service_seconds + queue_seconds
            backlog = 0.0
        else:
            chafe_latency = ""
            backlog = arrival_rate - service_rate
        rows.append(
            {
                "arrival_rate_reports_per_s": arrival_rate,
                "service_rate_reports_per_s": service_rate,
                "utilization": utilization,
                "avg_latency_seconds": chafe_latency,
                "batch_gating_seconds": 0.0,
                "backlog_growth_reports_per_s": backlog,
                "stable": str(stable).lower(),
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
        batch_delay = (epoch_m - 1.0) / (2.0 * arrival_rate)
        rows.append(
            {
                "arrival_rate_reports_per_s": arrival_rate,
                "service_rate_reports_per_s": "",
                "utilization": "",
                "avg_latency_seconds": batch_delay,
                "batch_gating_seconds": batch_delay,
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
    write_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv", rows)
    return rows


def generate_google_dpf_phase_breakdown() -> list[dict]:
    raw_rows = run_google_dpf(["--log_n=12", "--reports=8", "--repeats=5", "--breakdown"])
    if len(raw_rows) != 10:
        raise RuntimeError(f"unexpected Google DPF breakdown row count: {len(raw_rows)}")
    rows = [enrich_google_breakdown_row(row) for row in raw_rows]
    write_csv(DATA_DIR / "fig_dpf_phase_breakdown.csv", rows)
    return rows


def generate_numpy_proxy_microbench() -> list[dict]:
    N = 65_536
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
        throughput_rows = generate_dpf_throughput_from_scaling(scaling_rows)
        latency_rows = generate_ingestion_latency_model(scaling_rows)
        outputs["fig_dpf_scaling.csv"] = len(scaling_rows)
        outputs["fig_dpf_phase_breakdown.csv"] = len(breakdown_rows)
        outputs["fig_dpf_throughput_vs_n.csv"] = len(throughput_rows)
        outputs["fig_ingestion_latency_vs_rate.csv"] = len(latency_rows)
    return outputs


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
