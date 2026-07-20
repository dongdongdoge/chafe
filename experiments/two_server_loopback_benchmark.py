from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import selectors
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from chafe_eval_lib import DATA_DIR, ROOT, write_csv


BINARY = (
    ROOT
    / "third_party"
    / "distributed_point_functions"
    / "bazel-bin"
    / "chafe"
    / "google_dpf_loopback_e2e_benchmark"
)
RAW_OUTPUT = DATA_DIR / "fig_two_server_loopback_e2e.csv"
SUMMARY_OUTPUT = DATA_DIR / "fig_two_server_loopback_e2e_summary.csv"
DOMAIN_SWEEP_CASES = [(12, 128), (14, 128), (16, 128), (17, 128)]
COLLISION_LOAD_CASES = [(12, 1 << 12), (14, 1 << 14)]


def _ready_ports(
    process: subprocess.Popen[str], timeout_s: float = 15.0
) -> tuple[int, int]:
    if process.stdout is None:
        raise RuntimeError("server stdout pipe is unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        events = selector.select(timeout_s)
        if not events:
            raise TimeoutError("server did not announce its loopback port")
        line = process.stdout.readline().strip()
    finally:
        selector.close()
    if not line.startswith("READY,"):
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"unexpected server readiness line {line!r}; stderr={stderr}")
    fields = line.split(",")
    if len(fields) != 3:
        raise RuntimeError(f"unexpected server readiness fields: {line!r}")
    return int(fields[1]), int(fields[2])


def _parse_client_csv(stdout: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("deployment_scope,")),
        None,
    )
    if header_index is None or header_index + 1 >= len(lines):
        raise RuntimeError(f"client emitted no CSV row:\n{stdout}")
    rows = list(csv.DictReader(lines[header_index : header_index + 2]))
    if len(rows) != 1:
        raise RuntimeError(f"client emitted an unexpected CSV shape:\n{stdout}")
    row = rows[0]
    if None in row or any(value is None for value in row.values()):
        raise RuntimeError(f"client emitted misaligned CSV fields:\n{stdout}")
    return row


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_once(
    *,
    binary: Path,
    log_n: int,
    reports: int,
    release_m: int,
    decoder: str,
    seed: int,
    timeout_s: float = 180.0,
) -> dict[str, str]:
    if not binary.exists():
        raise FileNotFoundError(
            f"missing {binary}; build //chafe:google_dpf_loopback_e2e_benchmark first"
        )
    server_processes: list[subprocess.Popen[str]] = []
    try:
        common_server_args = [
            str(binary),
            "--mode=server",
            "--port=0",
            f"--log_n={log_n}",
            f"--release_m={release_m}",
            f"--decoder={decoder}",
            f"--seed={seed}",
        ]
        server1 = subprocess.Popen(
            [*common_server_args, "--server_id=1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        server_processes.append(server1)
        server1_port, peer_port = _ready_ports(server1)
        server0 = subprocess.Popen(
            [*common_server_args, "--server_id=0", f"--peer_port={peer_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        server_processes.append(server0)
        server0_port, unused_peer_port = _ready_ports(server0)
        if unused_peer_port != 0:
            raise RuntimeError("server 0 unexpectedly opened a peer listener")

        result = subprocess.run(
            [
                str(binary),
                "--mode=client",
                f"--server0_port={server0_port}",
                f"--server1_port={server1_port}",
                f"--log_n={log_n}",
                f"--reports={reports}",
                f"--release_m={release_m}",
                f"--decoder={decoder}",
                f"--seed={seed}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"client failed with exit code {result.returncode}:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        row = _parse_client_csv(result.stdout)
        for index, process in enumerate(server_processes):
            try:
                _stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"server {index} did not exit after epoch close") from error
            if process.returncode != 0:
                raise RuntimeError(
                    f"server {index} failed with exit code {process.returncode}:\n{stderr}"
                )
        if row["deployment_scope"] != "loopback_local_processes":
            raise AssertionError("loopback benchmark has an unsafe deployment label")
        if row["correctness"] != "true":
            raise AssertionError("loopback reconstruction was not verified")
        return row
    finally:
        for process in server_processes:
            _terminate(process)


def standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, int, int, int, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["scenario"]),
                int(row["log_n"]),
                int(row["m"]),
                int(row["M"]),
                str(row["decoder"]),
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    metrics = [
        "reports_per_second",
        "mean_write_latency_ms",
        "p50_write_latency_ms",
        "p95_write_latency_ms",
        "keygen_ms_per_report",
        "server0_eval_ms_per_report",
        "server1_eval_ms_per_report",
        "epoch_close_ms",
        "close_to_release_shares_received_ms",
        "analyzer_reconstruct_release_ms",
        "verification_ms",
    ]
    byte_metrics = [
        "serialized_dpf_key_bytes",
        "client_to_server_application_bytes",
        "server_to_driver_application_bytes",
        "epoch_close_inter_server_bytes",
    ]
    invariant_fields = [
        "deployment_scope",
        "transport",
        "roles",
        "server_processes",
        "online_inter_server_bytes",
        "driver_socket_payload",
        "benchmark_metadata_excluded_from_driver_view",
        "instrumentation_metadata_view",
        "online_protocol",
        "dummy_law",
        "release_share_rerandomization",
        "permutation_seed_recipients",
        "occupancy_opening_recipients",
        "verification_scope",
        "source_type",
        "benchmark",
    ]
    for (scenario, log_n, m, release_m, decoder), group in sorted(grouped.items()):
        first = group[0]
        for field in invariant_fields:
            values = {str(row[field]) for row in group}
            if len(values) != 1:
                raise ValueError(
                    f"{scenario}/{log_n}/{decoder}: inconsistent {field}: {sorted(values)}"
                )
        summary: dict[str, object] = {
            "scenario": scenario,
            "deployment_scope": first["deployment_scope"],
            "transport": first["transport"],
            "roles": first["roles"],
            "N": int(first["N"]),
            "log_n": log_n,
            "m": m,
            "M": release_m,
            "rho": float(first["rho"]),
            "tau": float(first["tau"]),
            "decoder": decoder,
            "repeats": len(group),
            "server_processes": int(first["server_processes"]),
            "online_inter_server_bytes": int(first["online_inter_server_bytes"]),
            "driver_socket_payload": first["driver_socket_payload"],
            "benchmark_metadata_excluded_from_driver_view": first[
                "benchmark_metadata_excluded_from_driver_view"
            ],
            "instrumentation_metadata_view": first["instrumentation_metadata_view"],
            "online_protocol": first["online_protocol"],
            "dummy_law": first["dummy_law"],
            "release_share_rerandomization": first["release_share_rerandomization"],
            "permutation_seed_recipients": first["permutation_seed_recipients"],
            "occupancy_opening_recipients": first["occupancy_opening_recipients"],
            "verification_scope": first["verification_scope"],
            "harness_limitation": (
                "workload_generator_and_analyzer_colocated_in_driver_process"
            ),
            "source_type": first["source_type"],
            "benchmark": first["benchmark"],
            "measurement_scope": (
                "localhost_only_not_multi_node; combined workload_generator_analyzer_driver "
                "server0 server1 processes; plaintext TCP; full keygen-to-fixed-release path; "
                "epoch-close occupancy opening between servers; application bytes exclude "
                "TCP_IP framing"
            ),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_se"] = standard_error(values)
        for metric in byte_metrics:
            values = [int(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
        summaries.append(summary)
    return summaries


def platform_metadata() -> dict[str, object]:
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or "",
    }


def generate_all(
    *,
    binary: Path = BINARY,
    domain_cases: list[tuple[int, int]] = DOMAIN_SWEEP_CASES,
    domain_repeats: int = 5,
    collision_cases: list[tuple[int, int]] = COLLISION_LOAD_CASES,
    collision_repeats: int = 3,
    decoders: tuple[str, ...] = ("drop", "pool"),
) -> dict[str, int]:
    raw_rows: list[dict[str, object]] = []
    metadata = platform_metadata()
    schedules = [
        ("fixed_128_write_domain_sweep", domain_cases, domain_repeats),
        ("collision_load_rho_1", collision_cases, collision_repeats),
    ]
    for scenario, cases, repeats in schedules:
        for log_n, reports in cases:
            release_m = 2 * reports
            for decoder in decoders:
                for repeat in range(repeats):
                    row: dict[str, object] = dict(
                        run_once(
                            binary=binary,
                            log_n=log_n,
                            reports=reports,
                            release_m=release_m,
                            decoder=decoder,
                            seed=(
                                810_000
                                + 10_000 * log_n
                                + 1_000 * (decoder == "pool")
                                + 2_000_000 * (scenario == "collision_load_rho_1")
                                + repeat
                            ),
                        )
                    )
                    row["scenario"] = scenario
                    row["repeat"] = repeat
                    row["harness_limitation"] = (
                        "workload_generator_and_analyzer_colocated_in_driver_process"
                    )
                    row.update(metadata)
                    raw_rows.append(row)
    summaries = summarize_rows(raw_rows)
    write_csv(RAW_OUTPUT, raw_rows)
    write_csv(SUMMARY_OUTPUT, summaries)
    return {RAW_OUTPUT.name: len(raw_rows), SUMMARY_OUTPUT.name: len(summaries)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the process-separated Google-DPF Chafe path over localhost TCP."
    )
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--collision-repeats", type=int, default=3)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one small correctness/performance case instead of the paper grid",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_domain_cases = [(10, 12)] if args.quick else DOMAIN_SWEEP_CASES
    selected_collision_cases = [] if args.quick else COLLISION_LOAD_CASES
    selected_domain_repeats = 1 if args.quick else args.repeats
    for name, count in generate_all(
        binary=args.binary,
        domain_cases=selected_domain_cases,
        domain_repeats=selected_domain_repeats,
        collision_cases=selected_collision_cases,
        collision_repeats=args.collision_repeats,
    ).items():
        print(f"{name}: {count} rows")
