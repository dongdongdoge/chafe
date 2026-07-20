"""Interface for external GPU-DPF benchmark rows.

This module is an adapter, not a GPU implementation.  It defines the normalized
CSV schema used by the Chafe artifact when importing cited GPU-DPF throughput
context, validates that the rows are scoped as external one-server kernel
measurements, and provides the log-rate interpolation used by the finite-burst
model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from chafe_eval_lib import DATA_DIR, read_csv, write_csv


NORMALIZED_COLUMNS = (
    "method",
    "N",
    "log_n",
    "dpf_per_second",
    "hardware",
    "prf",
    "security_parameter_bits",
    "entry_words",
    "entry_word_bits",
    "entry_bits",
    "batch_size",
    "client_keygen_included",
    "second_server_included",
    "chafe_packed_write_accumulation_included",
    "directly_comparable_to_chafe",
    "source_type",
    "benchmark",
    "timed_scope",
    "deployment_scope",
    "citation_key",
    "paper_doi",
    "artifact_url",
)

DEFAULT_METADATA = {
    "method": "GPU-DPF AES-128 (external V100)",
    "hardware": "NVIDIA V100",
    "prf": "AES-128",
    "security_parameter_bits": 128,
    "entry_words": 16,
    "entry_word_bits": 32,
    "entry_bits": 512,
    "batch_size": 512,
    "client_keygen_included": "false",
    "second_server_included": "false",
    "chafe_packed_write_accumulation_included": "false",
    "directly_comparable_to_chafe": "false",
    "source_type": "external_artifact_measurement",
    "benchmark": "facebookresearch_gpu_dpf_eval_gpu_aes128",
    "timed_scope": (
        "one server's batched DPF-PIR table evaluation, including key H2D "
        "and result D2H; table initialization and client key generation occur "
        "before timing"
    ),
    "deployment_scope": (
        "external GPU-DPF PIR kernel; not a Chafe hidden-write backend or "
        "end-to-end two-server measurement"
    ),
    "citation_key": "lam2024gpudpf",
    "paper_doi": "10.1145/3617232.3624855",
    "artifact_url": "https://github.com/facebookresearch/GPU-DPF",
}

# Cited V100/AES-128 points used as deterministic external context.
REFERENCE_GPU_DPF_AES128_V100 = {
    1 << 14: 52_536,
    1 << 16: 15_392,
    1 << 18: 3_967,
    1 << 20: 923,
}


@dataclass(frozen=True)
class InterpolatedRate:
    dpf_per_second: float
    lower_n: int
    upper_n: int
    interpolation: str


def interface_contract() -> dict[str, object]:
    return {
        "schema": "chafe.external_gpu_dpf.v1",
        "required_input_columns": ["N", "dpf_per_second"],
        "normalized_columns": list(NORMALIZED_COLUMNS),
        "rate_unit": "DPF evaluations per second",
        "scope": DEFAULT_METADATA["deployment_scope"],
    }


def _value(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    raise KeyError(f"row is missing one of {names}")


def _as_int(value: object, name: str) -> int:
    out = int(float(str(value)))
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _as_float(value: object, name: str) -> float:
    out = float(str(value))
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def normalize_row(row: Mapping[str, object]) -> dict[str, object]:
    n = _as_int(_value(row, "N", "n", "domain_size"), "N")
    rate = _as_float(
        _value(
            row,
            "dpf_per_second",
            "dpf_per_sec",
            "throughput_dpf_per_second",
            "throughput",
        ),
        "dpf_per_second",
    )
    normalized = dict(DEFAULT_METADATA)
    normalized.update({key: row[key] for key in row if key in NORMALIZED_COLUMNS})
    normalized["N"] = n
    normalized["log_n"] = n.bit_length() - 1
    normalized["dpf_per_second"] = int(rate) if rate.is_integer() else rate
    return normalized


def normalize_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized = [normalize_row(row) for row in rows]
    validate_rows(normalized)
    return sorted(normalized, key=lambda row: int(row["N"]))


def reference_rows() -> list[dict[str, object]]:
    return normalize_rows(
        {"N": n, "dpf_per_second": rate}
        for n, rate in REFERENCE_GPU_DPF_AES128_V100.items()
    )


def validate_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("GPU-DPF interface requires at least one row")
    seen: set[int] = set()
    for row in normalized:
        missing = [name for name in NORMALIZED_COLUMNS if name not in row]
        if missing:
            raise ValueError(f"GPU-DPF row is missing columns: {missing}")
        n = _as_int(row["N"], "N")
        if n & (n - 1):
            raise ValueError("GPU-DPF domain size N must be a power of two")
        if int(row["log_n"]) != n.bit_length() - 1:
            raise ValueError("GPU-DPF log_n is inconsistent with N")
        _as_float(row["dpf_per_second"], "dpf_per_second")
        if n in seen:
            raise ValueError(f"duplicate GPU-DPF domain size: {n}")
        seen.add(n)
        if row["source_type"] != "external_artifact_measurement":
            raise ValueError("GPU-DPF rows must remain external measurements")
        if row["directly_comparable_to_chafe"] != "false":
            raise ValueError("external GPU-DPF rows must not be marked comparable")
        for key in (
            "client_keygen_included",
            "second_server_included",
            "chafe_packed_write_accumulation_included",
        ):
            if row[key] != "false":
                raise ValueError(f"{key} must be false for this interface")
    return sorted(normalized, key=lambda row: int(row["N"]))


def load_rows(path: Path | None = None) -> list[dict[str, object]]:
    if path is None:
        path = DATA_DIR / "fig_gpu_dpf_external.csv"
    return validate_rows(read_csv(path))


def interpolate_rate(rows: Iterable[Mapping[str, object]], target_n: int) -> InterpolatedRate:
    ordered = validate_rows(rows)
    target_n = _as_int(target_n, "target_n")
    exact = [row for row in ordered if int(row["N"]) == target_n]
    if exact:
        return InterpolatedRate(
            dpf_per_second=float(exact[0]["dpf_per_second"]),
            lower_n=target_n,
            upper_n=target_n,
            interpolation="exact",
        )
    lower_candidates = [row for row in ordered if int(row["N"]) < target_n]
    upper_candidates = [row for row in ordered if int(row["N"]) > target_n]
    if not lower_candidates or not upper_candidates:
        raise ValueError("target_n lies outside the GPU-DPF reference grid")
    lower = max(lower_candidates, key=lambda row: int(row["N"]))
    upper = min(upper_candidates, key=lambda row: int(row["N"]))
    log_n = math.log2(target_n)
    lower_log_n = math.log2(int(lower["N"]))
    upper_log_n = math.log2(int(upper["N"]))
    weight = (log_n - lower_log_n) / (upper_log_n - lower_log_n)
    log_rate = (
        (1.0 - weight) * math.log(float(lower["dpf_per_second"]))
        + weight * math.log(float(upper["dpf_per_second"]))
    )
    return InterpolatedRate(
        dpf_per_second=math.exp(log_rate),
        lower_n=int(lower["N"]),
        upper_n=int(upper["N"]),
        interpolation="log_rate",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="raw or normalized GPU-DPF CSV")
    parser.add_argument("--output", type=Path, help="write normalized CSV")
    parser.add_argument("--target-n", type=int, default=1 << 17)
    parser.add_argument("--print-contract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_contract:
        print(json.dumps(interface_contract(), indent=2, sort_keys=True))
        return 0
    if args.input:
        rows = normalize_rows(read_csv(args.input))
    else:
        rows = load_rows()
    if args.output:
        write_csv(args.output, rows, fieldnames=list(NORMALIZED_COLUMNS))
    rate = interpolate_rate(rows, args.target_n)
    print(
        "GPU-DPF external interface: "
        f"N={args.target_n}, rate={rate.dpf_per_second:.6g} DPF/s, "
        f"bounds=({rate.lower_n},{rate.upper_n}), mode={rate.interpolation}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
