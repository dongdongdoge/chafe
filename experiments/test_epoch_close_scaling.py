from __future__ import annotations

import csv

from epoch_close_scaling import generate_all, next_power_of_two


def test_next_power_of_two() -> None:
    assert next_power_of_two(1) == 1
    assert next_power_of_two(64) == 64
    assert next_power_of_two(65) == 128


def test_two_process_epoch_close_scaling_smoke(tmp_path) -> None:
    output = tmp_path / "epoch_close.csv"
    rows = generate_all(
        m_values=(64, 100),
        repeats=2,
        warmups=1,
        output_path=output,
    )
    assert len(rows) == 4
    assert output.exists()
    assert {(int(row["m"]), row["decoder"]) for row in rows} == {
        (64, "drop"),
        (64, "pool"),
        (100, "drop"),
        (100, "pool"),
    }

    for row in rows:
        m = int(row["m"])
        n_slots = next_power_of_two(m)
        assert int(row["N"]) == n_slots
        assert int(row["M"]) == 2 * m
        assert float(row["rho"]) == m / n_slots
        assert float(row["tau"]) == 2.0
        assert 0 < int(row["real_output_count"]) <= m
        assert float(row["time_ms"]) > 0.0
        assert float(row["p50_ms"]) > 0.0
        assert float(row["p95_ms"]) > 0.0
        assert int(row["repeats"]) == 2
        assert int(row["warmups"]) == 1
        assert int(row["validated_releases"]) == 3
        assert int(row["validated_measured_releases"]) == 2
        assert int(row["counter_bytes_each_direction"]) == 4 * n_slots
        assert int(row["release_share_bytes_transferred"]) == 16 * m
        assert row["correctness"] == "true"
        assert row["source_type"] == "microbenchmark"
        assert row["deployment_scope"] == (
            "same_host_two_processes_not_network_deployment"
        )
        assert "post-timing correctness validation" in row["timed_scope"]
        assert "entire fixed-length release" in row["validation_scope"]

    with output.open(newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == len(rows)
    assert {row["correctness"] for row in persisted} == {"true"}
