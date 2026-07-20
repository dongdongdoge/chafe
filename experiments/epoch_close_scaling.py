from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import platform
import socket
import statistics
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from chafe_eval_lib import DATA_DIR, write_csv


DEFAULT_M_VALUES = (
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
)
DEFAULT_REPEATS = 10
DEFAULT_WARMUPS = 1
BASE_SEED = 31_415_926
OUTPUT = DATA_DIR / "fig_epoch_close_scaling.csv"

TIMED_SCOPE = (
    "two server processes over a localhost socketpair with the analyzer colocated "
    "in the server-0 driver; decoder command, "
    "counter-share exchange in both directions, occupancy reconstruction, "
    "decoder compaction, public-uniform dummy-share generation, shared-seed "
    "release permutation, one release-share transfer, and analyzer "
    "reconstruction; excludes epoch-state generation, additive-share setup, "
    "process startup, warmup runs, and post-timing correctness validation"
)
VALIDATION_SCOPE = (
    "outside the timed region, reconstruct occupancies exactly and compare the "
    "entire fixed-length release against payload shares, independently regenerated "
    "dummy shares, and the shared permutation; fold every validated release into "
    "a checksum"
)


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def platform_metadata() -> dict[str, object]:
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or "",
    }


def summarize(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("cannot summarize an empty timing sample")
    values = np.asarray(samples_ms, dtype=float)
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return {
        "time_ms": float(values.mean()),
        "standard_error_ms": standard_error,
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("loopback peer closed before sending the full payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _seed(
    *,
    m: int,
    run_index: int,
    decoder: str,
    server_id: int,
    purpose: int,
) -> int:
    decoder_tag = 1 if decoder == "pool" else 0
    return (
        BASE_SEED
        + 10_000_019 * m
        + 100_003 * run_index
        + 10_007 * decoder_tag
        + 101 * server_id
        + purpose
    )


def _uniform_uint64(seed: int, size: int) -> np.ndarray:
    if size < 0:
        raise ValueError("array size must be nonnegative")
    return np.random.default_rng(seed).bit_generator.random_raw(size)


def epoch_state(m: int, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    if m <= 0 or n_slots < m:
        raise ValueError("epoch state requires 0 < m <= N")
    rng = np.random.default_rng(BASE_SEED + m)
    slots = rng.integers(0, n_slots, size=m, dtype=np.int64)
    reports = rng.integers(0, 1 << 20, size=m, dtype=np.uint64)
    occupancy = np.bincount(slots, minlength=n_slots).astype(np.uint32)
    payload = np.zeros(n_slots, dtype=np.uint64)
    np.add.at(payload, slots, reports)
    if int(occupancy.sum(dtype=np.uint64)) != m:
        raise AssertionError("generated occupancy does not contain exactly m writes")
    return occupancy, payload


def loopback_worker(
    sock: socket.socket,
    *,
    m: int,
    release_m: int,
    counter_share: np.ndarray,
    payload_share: np.ndarray,
) -> None:
    try:
        while True:
            command = recv_exact(sock, 1)
            if command == b"Q":
                return
            if command not in (b"D", b"P"):
                raise RuntimeError(f"unknown decoder command: {command!r}")
            decoder = "drop" if command == b"D" else "pool"
            run_index = struct.unpack("<Q", recv_exact(sock, 8))[0]

            peer_counter = np.frombuffer(
                recv_exact(sock, counter_share.nbytes), dtype=np.uint32
            ).copy()
            sock.sendall(counter_share.tobytes(order="C"))
            occupancy = peer_counter + counter_share
            predicate = occupancy == 1 if decoder == "drop" else occupancy > 0
            real = payload_share[predicate]
            dummy_count = release_m - len(real)
            if dummy_count < 0:
                raise AssertionError("decoder emitted more than the fixed release length")

            release = np.empty(release_m, dtype=np.uint64)
            release[: len(real)] = real
            release[len(real) :] = _uniform_uint64(
                _seed(
                    m=m,
                    run_index=run_index,
                    decoder=decoder,
                    server_id=1,
                    purpose=1,
                ),
                dummy_count,
            )
            permutation = np.random.default_rng(
                _seed(
                    m=m,
                    run_index=run_index,
                    decoder=decoder,
                    server_id=0,
                    purpose=2,
                )
            ).permutation(release_m)
            release = release[permutation]
            sock.sendall(release.tobytes(order="C"))
    finally:
        sock.close()


def _validate_release(
    *,
    reconstructed: np.ndarray,
    expected_real: np.ndarray,
    dummy_share0: np.ndarray,
    m: int,
    run_index: int,
    decoder: str,
    permutation: np.ndarray,
) -> int:
    release_m = len(reconstructed)
    dummy_count = release_m - len(expected_real)
    dummy_share1 = _uniform_uint64(
        _seed(
            m=m,
            run_index=run_index,
            decoder=decoder,
            server_id=1,
            purpose=1,
        ),
        dummy_count,
    )
    expected = np.empty(release_m, dtype=np.uint64)
    expected[: len(expected_real)] = expected_real
    expected[len(expected_real) :] = dummy_share0 + dummy_share1
    expected = expected[permutation]
    if not np.array_equal(reconstructed, expected):
        mismatch = int(np.flatnonzero(reconstructed != expected)[0])
        raise AssertionError(
            f"release reconstruction mismatch for m={m}, decoder={decoder}, "
            f"run={run_index}, position={mismatch}"
        )
    return int(np.bitwise_xor.reduce(reconstructed))


def benchmark_epoch(
    m: int,
    *,
    repeats: int = DEFAULT_REPEATS,
    warmups: int = DEFAULT_WARMUPS,
    decoders: tuple[str, ...] = ("drop", "pool"),
) -> list[dict[str, object]]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    if warmups < 1:
        raise ValueError("warmups must be at least one")
    if not decoders or any(decoder not in ("drop", "pool") for decoder in decoders):
        raise ValueError("decoders must contain only 'drop' and/or 'pool'")

    n_slots = next_power_of_two(m)
    release_m = 2 * m
    occupancy, payload = epoch_state(m, n_slots)
    share_rng = np.random.default_rng(BASE_SEED + 7 * m)
    counter0 = share_rng.integers(0, 1 << 32, size=n_slots, dtype=np.uint32)
    counter1 = occupancy - counter0
    payload0 = share_rng.bit_generator.random_raw(n_slots)
    payload1 = payload - payload0

    expected_real = {
        "drop": payload[occupancy == 1],
        "pool": payload[occupancy > 0],
    }
    real_counts = {decoder: len(expected_real[decoder]) for decoder in decoders}
    if any(count > release_m for count in real_counts.values()):
        raise AssertionError("fixed release length is shorter than a decoder output")

    parent_sock, child_sock = socket.socketpair()
    context = mp.get_context("fork")
    worker = context.Process(
        target=loopback_worker,
        kwargs={
            "sock": child_sock,
            "m": m,
            "release_m": release_m,
            "counter_share": counter1,
            "payload_share": payload1,
        },
    )
    worker.start()
    child_sock.close()

    timings: dict[str, list[float]] = defaultdict(list)
    checksums = {decoder: 0 for decoder in decoders}
    validated_total = {decoder: 0 for decoder in decoders}
    validated_measured = {decoder: 0 for decoder in decoders}
    try:
        total_runs = warmups + repeats
        for run_index in range(total_runs):
            for decoder in decoders:
                start = time.perf_counter_ns()
                parent_sock.sendall(b"D" if decoder == "drop" else b"P")
                parent_sock.sendall(struct.pack("<Q", run_index))
                parent_sock.sendall(counter0.tobytes(order="C"))
                peer_counter = np.frombuffer(
                    recv_exact(parent_sock, counter1.nbytes), dtype=np.uint32
                ).copy()
                reconstructed_occupancy = counter0 + peer_counter
                predicate = (
                    reconstructed_occupancy == 1
                    if decoder == "drop"
                    else reconstructed_occupancy > 0
                )
                real0 = payload0[predicate]
                dummy_count = release_m - len(real0)
                dummy0 = _uniform_uint64(
                    _seed(
                        m=m,
                        run_index=run_index,
                        decoder=decoder,
                        server_id=0,
                        purpose=1,
                    ),
                    dummy_count,
                )
                release0 = np.empty(release_m, dtype=np.uint64)
                release0[: len(real0)] = real0
                release0[len(real0) :] = dummy0
                permutation = np.random.default_rng(
                    _seed(
                        m=m,
                        run_index=run_index,
                        decoder=decoder,
                        server_id=0,
                        purpose=2,
                    )
                ).permutation(release_m)
                release0 = release0[permutation]
                release1 = np.frombuffer(
                    recv_exact(parent_sock, release_m * np.dtype(np.uint64).itemsize),
                    dtype=np.uint64,
                ).copy()
                reconstructed = release0 + release1
                end = time.perf_counter_ns()

                if not np.array_equal(reconstructed_occupancy, occupancy):
                    raise AssertionError("counter shares did not reconstruct occupancy")
                release_checksum = _validate_release(
                    reconstructed=reconstructed,
                    expected_real=expected_real[decoder],
                    dummy_share0=dummy0,
                    m=m,
                    run_index=run_index,
                    decoder=decoder,
                    permutation=permutation,
                )
                checksums[decoder] ^= release_checksum ^ run_index
                validated_total[decoder] += 1
                if run_index >= warmups:
                    timings[decoder].append((end - start) / 1e6)
                    validated_measured[decoder] += 1
        parent_sock.sendall(b"Q")
    finally:
        parent_sock.close()
        worker.join(timeout=30)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            raise RuntimeError(f"epoch-close worker did not terminate for m={m}")
        if worker.exitcode not in (0, None):
            raise RuntimeError(
                f"epoch-close worker failed for m={m} with exit code {worker.exitcode}"
            )

    metadata = platform_metadata()
    rows: list[dict[str, object]] = []
    for decoder in decoders:
        if (
            len(timings[decoder]) != repeats
            or validated_measured[decoder] != repeats
            or validated_total[decoder] != warmups + repeats
        ):
            raise AssertionError("measured and validated repeat counts diverged")
        rows.append(
            {
                "method": f"Chafe {decoder.capitalize()} two-process close",
                "decoder": decoder,
                "m": m,
                "N": n_slots,
                "M": release_m,
                "rho": m / n_slots,
                "tau": release_m / m,
                "real_output_count": real_counts[decoder],
                **summarize(timings[decoder]),
                "repeats": repeats,
                "warmups": warmups,
                "validated_releases": validated_total[decoder],
                "validated_measured_releases": validated_measured[decoder],
                "counter_bytes_each_direction": counter0.nbytes,
                "release_share_bytes_transferred": release_m
                * np.dtype(np.uint64).itemsize,
                "checksum": checksums[decoder],
                "correctness": "true",
                "source_type": "microbenchmark",
                "benchmark": "two_process_localhost_epoch_close_scaling",
                "deployment_scope": "same_host_two_processes_not_network_deployment",
                "transport": "local_socketpair",
                "roles": "driver_process_server0_plus_analyzer; child_process_server1",
                "harness_limitation": (
                    "localhost_only; no cross-machine network, TLS, or independent "
                    "analyzer process"
                ),
                "timed_scope": TIMED_SCOPE,
                "validation_scope": VALIDATION_SCOPE,
                **metadata,
            }
        )
    return rows


def generate_all(
    *,
    m_values: Iterable[int] = DEFAULT_M_VALUES,
    repeats: int = DEFAULT_REPEATS,
    warmups: int = DEFAULT_WARMUPS,
    output_path: Path = OUTPUT,
) -> list[dict[str, object]]:
    selected = tuple(int(value) for value in m_values)
    if not selected or any(value <= 0 for value in selected):
        raise ValueError("m_values must contain positive epoch sizes")
    rows: list[dict[str, object]] = []
    for m in selected:
        rows.extend(benchmark_epoch(m, repeats=repeats, warmups=warmups))
    write_csv(output_path, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Chafe Drop/Pool epoch-close scaling using two processes over "
            "a localhost socketpair."
        )
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_M_VALUES),
        help="epoch sizes to benchmark",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generated = generate_all(
        m_values=args.m_values,
        repeats=args.repeats,
        warmups=args.warmups,
        output_path=args.output,
    )
    print(f"{args.output}: {len(generated)} rows")
