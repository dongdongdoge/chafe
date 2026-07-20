from __future__ import annotations

import math

import pytest

import gpu_dpf_interface as gpu
from chafe_eval_lib import DATA_DIR


def test_reference_rows_match_checked_in_external_context() -> None:
    rows = gpu.reference_rows()
    assert {int(row["N"]): int(row["dpf_per_second"]) for row in rows} == {
        16_384: 52_536,
        65_536: 15_392,
        262_144: 3_967,
        1_048_576: 923,
    }
    assert {row["source_type"] for row in rows} == {
        "external_artifact_measurement"
    }
    assert {row["directly_comparable_to_chafe"] for row in rows} == {"false"}


def test_gpu_dpf_interface_accepts_minimal_raw_csv_rows() -> None:
    rows = gpu.normalize_rows([{"domain_size": 1 << 16, "throughput": 1000}])
    assert rows[0]["N"] == 1 << 16
    assert rows[0]["log_n"] == 16
    assert rows[0]["dpf_per_second"] == 1000.0
    assert rows[0]["benchmark"] == "facebookresearch_gpu_dpf_eval_gpu_aes128"


def test_gpu_dpf_interpolation_uses_log_rate_between_powers_of_two() -> None:
    rows = gpu.reference_rows()
    rate = gpu.interpolate_rate(rows, 1 << 17)
    assert rate.lower_n == 1 << 16
    assert rate.upper_n == 1 << 18
    assert rate.interpolation == "log_rate"
    assert math.isclose(rate.dpf_per_second, math.sqrt(15_392 * 3_967))


def test_gpu_dpf_interface_rejects_comparable_rows() -> None:
    row = gpu.reference_rows()[0].copy()
    row["directly_comparable_to_chafe"] = "true"
    with pytest.raises(ValueError):
        gpu.validate_rows([row])


def test_checked_in_gpu_dpf_csv_satisfies_interface() -> None:
    rows = gpu.load_rows(DATA_DIR / "fig_gpu_dpf_external.csv")
    assert len(rows) == 4
    assert gpu.interpolate_rate(rows, 1 << 18).interpolation == "exact"
