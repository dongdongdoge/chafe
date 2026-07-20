from __future__ import annotations

import os
from pathlib import Path

import pytest

import two_server_loopback_benchmark as benchmark


RAW_INVARIANTS = {
    "deployment_scope": "loopback_local_processes",
    "transport": "plaintext_tcp_ipv4_loopback",
    "roles": "workload_generator_analyzer_driver_plus_two_server_processes",
    "server_processes": "2",
    "online_inter_server_bytes": "0",
    "driver_socket_payload": "two_fixed_length_M_release_shares_plus_benchmark_metadata",
    "benchmark_metadata_excluded_from_driver_view": "false",
    "instrumentation_metadata_view": "combined_workload_generator_analyzer_driver",
    "online_protocol": "sequential_ack_no_pipelining",
    "dummy_law": "uniform_mod_2^56_aes128ctr_independent_server_shares",
    "release_share_rerandomization": "joint_aes128ctr_zero_share_mask",
    "permutation_seed_recipients": "server0_and_server1_only",
    "occupancy_opening_recipients": "server0_and_server1_only",
    "verification_scope": "excluded_from_protocol_and_epoch_close_timing",
    "source_type": "microbenchmark",
    "benchmark": "google_dpf_two_server_tcp_loopback",
}


def test_summary_preserves_loopback_scope_and_aggregates() -> None:
    base = {
        **RAW_INVARIANTS,
        "scenario": "unit_test",
        "N": "64",
        "log_n": "6",
        "m": "4",
        "M": "8",
        "rho": "0.0625",
        "tau": "2",
        "decoder": "drop",
        "client_to_server_application_bytes": "100",
        "server_to_driver_application_bytes": "200",
        "serialized_dpf_key_bytes": "50",
        "reports_per_second": "1000",
        "mean_write_latency_ms": "1",
        "p50_write_latency_ms": "0.9",
        "p95_write_latency_ms": "1.2",
        "keygen_ms_per_report": "0.1",
        "server0_eval_ms_per_report": "0.4",
        "server1_eval_ms_per_report": "0.5",
        "epoch_close_ms": "0.7",
        "close_to_release_shares_received_ms": "0.5",
        "epoch_close_inter_server_bytes": "160",
        "analyzer_reconstruct_release_ms": "0.2",
        "verification_ms": "0.3",
    }
    second = dict(base)
    second["reports_per_second"] = "1200"
    second["serialized_dpf_key_bytes"] = "70"
    second["client_to_server_application_bytes"] = "120"
    second["server_to_driver_application_bytes"] = "220"
    rows = benchmark.summarize_rows([base, second])
    assert len(rows) == 1
    assert rows[0]["deployment_scope"] == "loopback_local_processes"
    assert rows[0]["reports_per_second_mean"] == 1100.0
    assert float(rows[0]["reports_per_second_se"]) > 0
    assert rows[0]["serialized_dpf_key_bytes_mean"] == 60
    assert rows[0]["serialized_dpf_key_bytes_min"] == 50
    assert rows[0]["serialized_dpf_key_bytes_max"] == 70
    assert rows[0]["client_to_server_application_bytes_mean"] == 110
    assert rows[0]["server_to_driver_application_bytes_max"] == 220
    assert "not_multi_node" in str(rows[0]["measurement_scope"])


def test_summary_keeps_domain_and_collision_scenarios_separate() -> None:
    shared = {
        **RAW_INVARIANTS,
        "N": "64",
        "log_n": "6",
        "M": "128",
        "tau": "2",
        "decoder": "pool",
        "serialized_dpf_key_bytes": "50",
        "client_to_server_application_bytes": "100",
        "server_to_driver_application_bytes": "200",
        "epoch_close_inter_server_bytes": "160",
        "reports_per_second": "1000",
        "mean_write_latency_ms": "1",
        "p50_write_latency_ms": "0.9",
        "p95_write_latency_ms": "1.2",
        "keygen_ms_per_report": "0.1",
        "server0_eval_ms_per_report": "0.4",
        "server1_eval_ms_per_report": "0.5",
        "epoch_close_ms": "0.7",
        "close_to_release_shares_received_ms": "0.5",
        "analyzer_reconstruct_release_ms": "0.2",
        "verification_ms": "0.3",
    }
    domain = {
        **shared,
        "scenario": "fixed_128_write_domain_sweep",
        "m": "64",
        "rho": "1",
    }
    collision = {
        **shared,
        "scenario": "collision_load_rho_1",
        "m": "64",
        "rho": "1",
    }
    rows = benchmark.summarize_rows([domain, collision])
    assert len(rows) == 2
    assert {row["scenario"] for row in rows} == {
        "fixed_128_write_domain_sweep",
        "collision_load_rho_1",
    }


def test_client_csv_parser_rejects_misaligned_rows() -> None:
    with pytest.raises(RuntimeError, match="misaligned CSV fields"):
        benchmark._parse_client_csv("deployment_scope,transport\nlocal,tcp,extra\n")
    with pytest.raises(RuntimeError, match="misaligned CSV fields"):
        benchmark._parse_client_csv("deployment_scope,transport\nlocal\n")


@pytest.mark.skipif(
    not benchmark.BINARY.exists() or os.environ.get("CHAFE_RUN_LOOPBACK_INTEGRATION") != "1",
    reason="set CHAFE_RUN_LOOPBACK_INTEGRATION=1 after building the loopback binary",
)
def test_process_separated_google_dpf_smoke() -> None:
    row = benchmark.run_once(
        binary=Path(benchmark.BINARY),
        log_n=6,
        reports=4,
        release_m=8,
        decoder="pool",
        seed=91,
    )
    assert row["correctness"] == "true"
    assert row["deployment_scope"] == "loopback_local_processes"
    assert int(row["server_processes"]) == 2
    assert int(row["online_inter_server_bytes"]) == 0
    assert int(row["epoch_close_inter_server_bytes"]) == 160
    assert row["driver_socket_payload"] == (
        "two_fixed_length_M_release_shares_plus_benchmark_metadata"
    )
    assert row["benchmark_metadata_excluded_from_driver_view"] == "false"
    assert row["instrumentation_metadata_view"] == (
        "combined_workload_generator_analyzer_driver"
    )
    assert row["dummy_law"] == "uniform_mod_2^56_aes128ctr_independent_server_shares"
    assert row["release_share_rerandomization"] == "joint_aes128ctr_zero_share_mask"
    assert row["permutation_seed_recipients"] == "server0_and_server1_only"
    assert row["occupancy_opening_recipients"] == "server0_and_server1_only"
    assert row["verification_scope"] == "excluded_from_protocol_and_epoch_close_timing"
    assert float(row["verification_ms"]) > 0
    assert float(row["reports_per_second"]) > 0
