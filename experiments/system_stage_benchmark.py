from __future__ import annotations

import math
import multiprocessing as mp
import os
import platform
import socket
import struct
import time
from collections import defaultdict

import numpy as np

from chafe_eval_lib import DATA_DIR, write_csv


EPOCH_M = 100_000
N_SLOTS = 1 << 17
RELEASE_M = 200_000
LOCAL_REPEATS = 31
LOOPBACK_REPEATS = 15
SEED = 9_100_117


def platform_metadata() -> dict[str, str]:
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count() or ""),
    }


def summarize(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=float)
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "time_ms": float(values.mean()),
        "standard_error_ms": se,
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def epoch_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    slots = rng.integers(0, N_SLOTS, size=EPOCH_M, dtype=np.int64)
    reports = rng.integers(0, 1 << 20, size=EPOCH_M, dtype=np.uint64)
    occupancy = np.bincount(slots, minlength=N_SLOTS).astype(np.uint32)
    payload = np.zeros(N_SLOTS, dtype=np.uint64)
    np.add.at(payload, slots, reports)
    return occupancy, payload, reports


def local_release_rows(
    occupancy: np.ndarray,
    payload: np.ndarray,
    reports: np.ndarray,
) -> list[dict]:
    timings: dict[tuple[str, str], list[float]] = defaultdict(list)
    real_counts: dict[str, int] = {}
    checksum = np.uint64(0)

    for repeat in range(LOCAL_REPEATS + 1):
        for decoder, predicate in [
            ("Chafe Drop", occupancy == 1),
            ("Chafe Pool", occupancy > 0),
        ]:
            rng = np.random.default_rng(SEED + 10_000 * (decoder == "Chafe Pool") + repeat)
            total_start = time.perf_counter_ns()
            stage_start = total_start
            real = payload[predicate].copy()
            compact_end = time.perf_counter_ns()
            dummies = rng.integers(0, 1 << 56, size=RELEASE_M - len(real), dtype=np.uint64)
            release = np.concatenate((real, dummies))
            pad_end = time.perf_counter_ns()
            rng.shuffle(release)
            permute_end = time.perf_counter_ns()
            checksum ^= release[0] ^ release[-1]
            if repeat > 0:
                timings[(decoder, "compact")].append((compact_end - stage_start) / 1e6)
                timings[(decoder, "pad")].append((pad_end - compact_end) / 1e6)
                timings[(decoder, "permute")].append((permute_end - pad_end) / 1e6)
                timings[(decoder, "total")].append((permute_end - total_start) / 1e6)
            real_counts[decoder] = len(real)

        for method, padded in [
            ("Trusted shuffle", False),
            ("Trusted padded shuffle", True),
        ]:
            rng = np.random.default_rng(SEED + 30_000 * padded + repeat)
            total_start = time.perf_counter_ns()
            stage_start = total_start
            release = reports.copy()
            copy_end = time.perf_counter_ns()
            if padded:
                dummies = rng.integers(0, 1 << 56, size=RELEASE_M - EPOCH_M, dtype=np.uint64)
                release = np.concatenate((release, dummies))
            pad_end = time.perf_counter_ns()
            rng.shuffle(release)
            permute_end = time.perf_counter_ns()
            checksum ^= release[0] ^ release[-1]
            if repeat > 0:
                timings[(method, "copy")].append((copy_end - stage_start) / 1e6)
                timings[(method, "pad")].append((pad_end - copy_end) / 1e6)
                timings[(method, "permute")].append((permute_end - pad_end) / 1e6)
                timings[(method, "total")].append((permute_end - total_start) / 1e6)
            real_counts[method] = EPOCH_M

    rows: list[dict] = []
    for (method, phase), samples in sorted(timings.items()):
        rows.append(
            {
                "method": method,
                "phase": phase,
                **summarize(samples),
                "source_type": "microbenchmark",
                "benchmark": "numpy_epoch_release_and_trusted_shuffle",
                "N": N_SLOTS if method.startswith("Chafe") else "",
                "m": EPOCH_M,
                "M": RELEASE_M if method != "Trusted shuffle" else EPOCH_M,
                "rho": EPOCH_M / N_SLOTS if method.startswith("Chafe") else "",
                "tau": (RELEASE_M if method != "Trusted shuffle" else EPOCH_M) / EPOCH_M,
                "decoder": (
                    "drop" if method == "Chafe Drop" else
                    "pool" if method == "Chafe Pool" else
                    "ideal_trusted"
                ),
                "real_output_count": real_counts[method],
                "repeats": LOCAL_REPEATS,
                "warmups": 1,
                "checksum": int(checksum),
                "timed_scope": "local cleartext compaction, dummy generation, and permutation; no network or MPC",
                **platform_metadata(),
            }
        )
    return rows


def recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("loopback peer closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def loopback_worker(
    sock: socket.socket,
    counter_share: np.ndarray,
    payload_share: np.ndarray,
) -> None:
    try:
        while True:
            command = recv_exact(sock, 1)
            if command == b"Q":
                return
            decoder = "drop" if command == b"D" else "pool"
            repeat = struct.unpack("<Q", recv_exact(sock, 8))[0]
            peer_counter = np.frombuffer(
                recv_exact(sock, counter_share.nbytes), dtype=np.uint32
            ).copy()
            sock.sendall(counter_share.tobytes(order="C"))
            occupancy = peer_counter + counter_share
            predicate = occupancy == 1 if decoder == "drop" else occupancy > 0
            real = payload_share[predicate].copy()
            rng = np.random.default_rng(SEED + 200_000 + 10_000 * (decoder == "pool") + repeat)
            dummies = rng.integers(0, np.iinfo(np.uint64).max, size=RELEASE_M - len(real), dtype=np.uint64)
            release = np.concatenate((real, dummies))
            permutation = np.random.default_rng(SEED + 300_000 + repeat).permutation(RELEASE_M)
            release = release[permutation]
            sock.sendall(release.tobytes(order="C"))
    finally:
        sock.close()


def loopback_release_rows(occupancy: np.ndarray, payload: np.ndarray) -> list[dict]:
    share_rng = np.random.default_rng(SEED + 100_000)
    counter0 = share_rng.integers(0, 1 << 32, size=N_SLOTS, dtype=np.uint32)
    counter1 = occupancy - counter0
    payload0 = share_rng.integers(0, np.iinfo(np.uint64).max, size=N_SLOTS, dtype=np.uint64)
    payload1 = payload - payload0
    parent_sock, child_sock = socket.socketpair()
    context = mp.get_context("fork")
    worker = context.Process(target=loopback_worker, args=(child_sock, counter1, payload1))
    worker.start()
    child_sock.close()
    timings: dict[str, list[float]] = defaultdict(list)
    checksum = np.uint64(0)
    try:
        for repeat in range(LOOPBACK_REPEATS + 1):
            for decoder in ["drop", "pool"]:
                start = time.perf_counter_ns()
                parent_sock.sendall(b"D" if decoder == "drop" else b"P")
                parent_sock.sendall(struct.pack("<Q", repeat))
                parent_sock.sendall(counter0.tobytes(order="C"))
                peer_counter = np.frombuffer(
                    recv_exact(parent_sock, counter1.nbytes), dtype=np.uint32
                ).copy()
                reconstructed_occupancy = counter0 + peer_counter
                predicate = reconstructed_occupancy == 1 if decoder == "drop" else reconstructed_occupancy > 0
                real0 = payload0[predicate].copy()
                rng = np.random.default_rng(SEED + 400_000 + 10_000 * (decoder == "pool") + repeat)
                dummy0 = rng.integers(0, np.iinfo(np.uint64).max, size=RELEASE_M - len(real0), dtype=np.uint64)
                release0 = np.concatenate((real0, dummy0))
                permutation = np.random.default_rng(SEED + 300_000 + repeat).permutation(RELEASE_M)
                release0 = release0[permutation]
                release1 = np.frombuffer(
                    recv_exact(parent_sock, RELEASE_M * np.dtype(np.uint64).itemsize),
                    dtype=np.uint64,
                ).copy()
                reconstructed = release0 + release1
                end = time.perf_counter_ns()
                checksum ^= reconstructed[0] ^ reconstructed[-1]
                if repeat > 0:
                    timings[decoder].append((end - start) / 1e6)
        parent_sock.sendall(b"Q")
    finally:
        parent_sock.close()
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join()
        if worker.exitcode not in (0, None):
            raise RuntimeError(f"loopback worker failed with exit code {worker.exitcode}")

    rows = []
    for decoder, samples in timings.items():
        rows.append(
            {
                "method": f"Two-process loopback {decoder.capitalize()}",
                "phase": "total",
                **summarize(samples),
                "source_type": "microbenchmark",
                "benchmark": "two_process_loopback_epoch_release",
                "N": N_SLOTS,
                "m": EPOCH_M,
                "M": RELEASE_M,
                "rho": EPOCH_M / N_SLOTS,
                "tau": RELEASE_M / EPOCH_M,
                "decoder": decoder,
                "real_output_count": int(np.count_nonzero(occupancy == 1 if decoder == "drop" else occupancy > 0)),
                "repeats": LOOPBACK_REPEATS,
                "warmups": 1,
                "checksum": int(checksum),
                "timed_scope": "two server processes over local socketpair; counter-share exchange, local compaction/padding/permutation, one release-share transfer, analyzer colocated with S0",
                "counter_bytes_each_direction": counter0.nbytes,
                "release_share_bytes_transferred": RELEASE_M * np.dtype(np.uint64).itemsize,
                **platform_metadata(),
            }
        )
    return rows


def generate_all() -> dict[str, int]:
    occupancy, payload, reports = epoch_state()
    local_rows = local_release_rows(occupancy, payload, reports)
    loopback_rows = loopback_release_rows(occupancy, payload)
    local_name = "fig_epoch_end_stage_cost.csv"
    loopback_name = "fig_epoch_end_loopback.csv"
    write_csv(DATA_DIR / local_name, local_rows)
    write_csv(DATA_DIR / loopback_name, loopback_rows)
    return {local_name: len(local_rows), loopback_name: len(loopback_rows)}


if __name__ == "__main__":
    for name, count in generate_all().items():
        print(f"{name}: {count} rows")
