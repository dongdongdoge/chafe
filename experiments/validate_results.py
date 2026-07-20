from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

import gpu_dpf_interface
from chafe_eval_lib import DATA_DIR, ROOT, read_csv


def floats(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def by_method(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row["method"]].append(row)
    return out


def assert_nonincreasing(values: list[float], name: str, tol: float = 1e-12) -> None:
    for a, b in zip(values, values[1:]):
        if b > a + tol:
            raise AssertionError(f"{name} is not nonincreasing: {a} -> {b}")


def assert_constant(values: list[float], name: str, tol: float = 1e-12) -> None:
    first = values[0]
    for value in values[1:]:
        if abs(value - first) > tol:
            raise AssertionError(f"{name} is not constant: {first} vs {value}")


def validate_privacy_monotonicity(notes: list[str]) -> None:
    rows = by_method(read_csv(DATA_DIR / "fig_privacy_vs_rho.csv"))
    drop = sorted(rows["Drop (matched padding)"], key=lambda row: float(row["rho"]))
    assert_nonincreasing(floats(drop, "epsilon"), "Drop epsilon vs rho")
    ideal = sorted(rows["IdealShuffle"], key=lambda row: float(row["rho"]))
    assert_constant(floats(ideal, "epsilon"), "IdealShuffle epsilon vs rho")
    ideal_padded = sorted(rows["IdealShuffle (same M padding)"], key=lambda row: float(row["rho"]))
    assert_constant(floats(ideal_padded, "epsilon"), "Padded IdealShuffle epsilon vs rho")
    if any(float(p["epsilon"]) > float(u["epsilon"]) + 1e-12 for p, u in zip(ideal_padded, ideal)):
        raise AssertionError("same-M padded ideal reference exceeds the standard ideal reference")

    rows_tau = by_method(read_csv(DATA_DIR / "fig_privacy_vs_tau.csv"))
    drop_tau = sorted(rows_tau["Drop (matched padding)"], key=lambda row: float(row["tau"]))
    assert_nonincreasing(floats(drop_tau, "epsilon"), "Drop epsilon vs tau")
    ideal_tau = sorted(rows_tau["IdealShuffle"], key=lambda row: float(row["tau"]))
    assert_constant(floats(ideal_tau, "epsilon"), "IdealShuffle epsilon vs tau")
    padded_tau = sorted(rows_tau["IdealShuffle (same M padding)"], key=lambda row: float(row["tau"]))
    assert_nonincreasing(floats(padded_tau, "epsilon"), "Padded IdealShuffle epsilon vs tau")
    notes.append("Privacy monotonicity checks passed.")


def validate_collision_sum(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_collision_regimes.csv")
    sums: dict[float, float] = defaultdict(float)
    for row in rows:
        sums[round(float(row["rho"]), 12)] += float(row["fraction"])
    for rho, total in sums.items():
        if abs(total - 1.0) > 1e-10:
            raise AssertionError(f"collision fractions do not sum to one at rho={rho}: {total}")
    notes.append("Collision regime normalization checks passed.")


def validate_exact_proxy(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv")
    exact_rows = [
        row for row in rows
        if row["source_type"] == "exact_law_numerical"
    ]
    if not exact_rows:
        raise AssertionError("missing exact-law numerical Drop rows")
    ratios = [float(row["exact_to_proxy_ratio"]) for row in exact_rows]
    decimal_errors = [
        float(row["float64_abs_error_vs_decimal70"])
        for row in exact_rows
    ]
    if max(decimal_errors) > 1e-10:
        raise AssertionError(
            "float64 exact-law inversion disagrees with decimal70: "
            f"max error {max(decimal_errors)}"
        )
    if any(
        row["numerical_status"]
        != "exact_law_float64_cross_checked_decimal70_not_interval_certified"
        for row in exact_rows
    ):
        raise AssertionError("exact-law rows have ambiguous numerical provenance")
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    if min_ratio <= 0 or max_ratio >= 100:
        raise AssertionError(f"exact/proxy ratios are outside a broad same-order range: {min_ratio}..{max_ratio}")
    if min_ratio < 0.1 or max_ratio > 10:
        notes.append(
            "Exact/proxy ratios are outside a tight 10x band for some rho; evaluation text treats this as "
            "finite-domain validation of trend/order, not as equality."
        )
    else:
        notes.append("Exact/proxy ratios remain within a 10x band on the finite-domain validation grid.")
    notes.append(
        "Homogeneous Drop float64 inversions agree with the independent "
        f"decimal70 recurrence within {max(decimal_errors):.3g}."
    )


def validate_accountant_scaling(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_accountant_scaling.csv")
    methods = {
        "Coefficient DP (K=2)",
        "Coefficient DP (K=3)",
        "Matched-Drop sufficient count law",
    }
    if {row["method"] for row in rows} != methods:
        raise AssertionError("accountant-scaling methods are incomplete")
    if any(row["source_type"] != "measured_accountant_runtime" for row in rows):
        raise AssertionError("accountant scaling must be labeled measured runtime")
    if any(float(row["runtime_median_s"]) <= 0 for row in rows):
        raise AssertionError("accountant runtime must be positive")
    if any(int(float(row["final_support_states"])) <= 0 for row in rows):
        raise AssertionError("accountant support state count must be positive")
    for method in methods:
        subset = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(float(row["m"])),
        )
        states = [int(float(row["final_support_states"])) for row in subset]
        if any(right < left for left, right in zip(states, states[1:])):
            raise AssertionError(f"support states decrease for {method}")
    special = [
        row for row in rows
        if row["method"] == "Matched-Drop sufficient count law"
    ]
    if max(int(float(row["m"])) for row in special) < 100_000:
        raise AssertionError("matched-Drop scaling does not reach m=100000")
    notes.append(
        "Measured accountant-runtime artifact distinguishes generic coefficient "
        "DP from the scalable matched-Drop sufficient law."
    )


def validate_exact_ideal_shuffle(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_exact_ideal_shuffle_baseline.csv")
    if len(rows) != 3:
        raise AssertionError("ideal-shuffle baseline must exhaust the three binary replacement pairs")
    family = {row["family_alpha_exact_fraction"] for row in rows}
    if family != {"3367/1125"}:
        raise AssertionError(f"unexpected exact ideal-shuffle family alpha: {family}")
    if sum(row["is_worst_case"] == "true" for row in rows) != 2:
        raise AssertionError("the two symmetric edge pairs must attain the ideal-shuffle maximum")
    if any(row["source_type"] != "exact_accountant" for row in rows):
        raise AssertionError("ideal-shuffle exact rows have the wrong provenance")
    notes.append("Exact rational ideal-shuffle replacement baseline passed exhaustive validation.")


def validate_microbench(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv")
    work = {row["method"]: float(row["normalized_expansion_work"]) for row in rows}
    product_name = "Google DPF packed64" if "Google DPF packed64" in work else "Product-valued kernel"
    dual_name = "Google DPF dual-track" if "Google DPF dual-track" in work else "Dual-track kernel"
    if work[product_name] > work[dual_name]:
        raise AssertionError("product-valued normalized work exceeds dual-track work")
    for row in rows:
        if row["source_type"] != "microbenchmark":
            raise AssertionError("microbenchmark CSV row has wrong source_type")
        if float(row["time_ms_per_report"]) <= 0:
            raise AssertionError("microbenchmark time must be positive")
    if rows[0].get("benchmark") == "google_distributed_point_functions":
        methods = {row["method"] for row in rows}
        expected_methods = {"Google DPF packed64", "Google DPF tuple product", "Google DPF dual-track"}
        if methods != expected_methods:
            raise AssertionError(f"unexpected Google DPF methods: {methods}")
        main_ns = {int(float(row["N"])) for row in rows}
        if min(main_ns) < 100_000:
            raise AssertionError(f"main Google DPF microbenchmark must use N>=100000, saw {sorted(main_ns)}")
        checksums = {row.get("checksum") for row in rows}
        if len(checksums) != 1:
            raise AssertionError("Google DPF product and dual-track reconstructions disagree")
        packed_rows = [row for row in rows if row["method"] == "Google DPF packed64"]
        if len(packed_rows) != 1 or packed_rows[0].get(
            "packed_share_field_extraction_verified"
        ) != "true":
            raise AssertionError("packed64 DPF shares were not field-extraction checked")
        scaling_rows = read_csv(DATA_DIR / "fig_dpf_scaling.csv")
        max_scaling_n = max(int(float(row["N"])) for row in scaling_rows)
        if max_scaling_n < 100_000:
            raise AssertionError(f"Google DPF scaling must include N>=100000, max N={max_scaling_n}")
        scaling_by_n: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in scaling_rows:
            if row["source_type"] != "microbenchmark":
                raise AssertionError("DPF scaling row has wrong source_type")
            if float(row["time_ms_per_report"]) <= 0:
                raise AssertionError("DPF scaling time must be positive")
            scaling_by_n[row["N"]].append(row)
        for n, n_rows in scaling_by_n.items():
            if len(n_rows) != 3:
                raise AssertionError(f"DPF scaling expected three rows at N={n}")
            if len({row["checksum"] for row in n_rows}) != 1:
                raise AssertionError(f"Google DPF scaling reconstructions disagree at N={n}")
        breakdown_rows = read_csv(DATA_DIR / "fig_dpf_phase_breakdown.csv")
        phases = {row["phase"] for row in breakdown_rows}
        if phases != {"total", "keygen", "context", "eval", "accumulate"}:
            raise AssertionError(f"unexpected DPF phase set: {phases}")
        for row in breakdown_rows:
            if row["source_type"] != "microbenchmark":
                raise AssertionError("DPF breakdown row has wrong source_type")
            if float(row["time_ms_per_report"]) < 0:
                raise AssertionError("DPF breakdown time must be nonnegative")
            if float(row["key_bytes_per_report"]) <= 0:
                raise AssertionError("DPF key bytes must be positive")
        throughput_rows = read_csv(DATA_DIR / "fig_dpf_throughput_vs_n.csv")
        for row in throughput_rows:
            if row["source_type"] != "microbenchmark":
                raise AssertionError("DPF throughput row has wrong source_type")
            if float(row["reports_per_second"]) <= 0:
                raise AssertionError("DPF throughput must be positive")
        thread_rows = read_csv(DATA_DIR / "fig_dpf_thread_scaling.csv")
        thread_by_count: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in thread_rows:
            if row["source_type"] != "microbenchmark":
                raise AssertionError("DPF thread-scaling row has wrong source_type")
            if int(float(row["threads"])) < 1:
                raise AssertionError("DPF thread count must be positive")
            if float(row["reports_per_second"]) <= 0:
                raise AssertionError("DPF thread-scaling throughput must be positive")
            if float(row["thread_speedup_vs_1"]) <= 0:
                raise AssertionError("DPF thread-scaling speedup must be positive")
            thread_by_count[row["threads"]].append(row)
        for threads, t_rows in thread_by_count.items():
            if len(t_rows) != 3:
                raise AssertionError(f"DPF thread scaling expected three rows at threads={threads}")
            if len({row["checksum"] for row in t_rows}) != 1:
                raise AssertionError(f"Google DPF thread-scaling reconstructions disagree at threads={threads}")
        packed_thread_rows = {
            int(float(row["threads"])): row
            for row in thread_rows
            if row["method"] == "Google DPF packed64"
        }
        if set(packed_thread_rows) != {1, 2, 4, 8}:
            raise AssertionError("packed64 thread anchors must be exactly 1/2/4/8")
        one_thread_rate = float(packed_thread_rows[1]["reports_per_second"])
        for threads, row in packed_thread_rows.items():
            if int(float(row["N"])) != 131_072:
                raise AssertionError("packed64 thread anchors must use N=2^17")
            rate = float(row["reports_per_second"])
            if not math.isclose(rate, 1_000.0 / float(row["time_ms_per_report"]), rel_tol=1e-10):
                raise AssertionError("thread-anchor rate disagrees with measured time")
            expected_speedup = rate / one_thread_rate
            if not math.isclose(
                float(row["thread_speedup_vs_1"]), expected_speedup, rel_tol=1e-9
            ):
                raise AssertionError("thread-anchor speedup is inconsistent")
        latency_rows = read_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv")
        for row in latency_rows:
            if row["source_type"] != "analytic_cost":
                raise AssertionError("ingestion latency row has wrong source_type")
            if row["method"] == "Chafe online DPF service" and row["stable"] == "false":
                if float(row["backlog_growth_reports_per_s"]) <= 0:
                    raise AssertionError("unstable Chafe rows must report positive backlog growth")
            if row["method"] == "Chafe online DPF service" and row["stable"] == "true":
                if int(float(row["N"])) < 100_000:
                    raise AssertionError("ingestion latency model must use a measured service point with N>=100000")
                if float(row["avg_latency_seconds"]) <= 0:
                    raise AssertionError("stable Chafe latency must be positive")
            if row["method"] == "Collect-then-shuffle gate":
                if float(row["avg_latency_seconds"]) <= 0:
                    raise AssertionError("batch-gating latency must be positive")
            if row["method"] == "Chafe overload horizon":
                if row["stable"] != "false":
                    raise AssertionError("overload horizon rows must be marked unstable")
                if float(row["backlog_growth_reports_per_s"]) <= 0:
                    raise AssertionError("overload horizon rows must report positive backlog growth")
                if float(row["avg_latency_seconds"]) <= 0:
                    raise AssertionError("overload horizon seconds must be positive")
            if row["method"] == "Estimated crossover":
                if float(row["avg_latency_seconds"]) <= 0:
                    raise AssertionError("estimated crossover seconds must be positive")
                if not (0.99 <= float(row["utilization"]) <= 1.0):
                    raise AssertionError("estimated crossover should occur near service saturation")
        joint_rows = read_csv(DATA_DIR / "fig_ingestion_joint_scaling.csv")
        expected_nodes = {1, 2, 3, 4, 8, 16, 32, 64, 128, 160}
        expected_pairs = {(threads, node) for threads in {1, 2, 4, 8} for node in expected_nodes}
        actual_pairs = {
            (int(float(row["threads_per_node"])), int(float(row["node_equivalents"])))
            for row in joint_rows
        }
        if actual_pairs != expected_pairs or len(joint_rows) != len(expected_pairs):
            raise AssertionError("joint node-thread projection is not the exact Cartesian grid")
        node_scaling_exponent = 1.0
        for row in joint_rows:
            if row["source_type"] != "analytic_cost":
                raise AssertionError("joint scaling projection has wrong source_type")
            if row["derived_from"] != "fig_dpf_thread_scaling.csv":
                raise AssertionError("joint scaling projection lost its measured anchor provenance")
            if row["single_node_anchor_measured"] != "true" or row["multi_node_measured"] != "false":
                raise AssertionError("joint scaling measured/projection flags are incorrect")
            threads = int(float(row["threads_per_node"]))
            nodes = int(float(row["node_equivalents"]))
            anchor_rate = float(packed_thread_rows[threads]["reports_per_second"])
            speedup = anchor_rate / one_thread_rate
            if not math.isclose(
                float(row["measured_single_node_rate_reports_per_s"]),
                anchor_rate,
                rel_tol=1e-12,
            ):
                raise AssertionError("joint scaling anchor differs from measured packed64 rate")
            if not math.isclose(float(row["thread_speedup_vs_1"]), speedup, rel_tol=1e-12):
                raise AssertionError("joint scaling thread speedup is inconsistent")
            if not math.isclose(
                float(row["thread_parallel_efficiency"]), speedup / threads, rel_tol=1e-12
            ):
                raise AssertionError("joint scaling thread efficiency is inconsistent")
            if not math.isclose(
                float(row["node_scaling_exponent"]),
                node_scaling_exponent,
                rel_tol=1e-12,
            ):
                raise AssertionError("joint scaling exponent is inconsistent")
            expected_efficiency = 1.0
            if not math.isclose(
                float(row["node_scaling_efficiency_assumed"]),
                expected_efficiency,
                rel_tol=1e-12,
            ):
                raise AssertionError("joint scaling efficiency is inconsistent")
            expected_role = (
                "measured_input" if nodes == 1 else "ideal_capacity_projection"
            )
            if row["plot_role"] != expected_role:
                raise AssertionError("joint scaling plot-role provenance is inconsistent")
            raw = anchor_rate * nodes
            if not math.isclose(
                float(row["projected_raw_rate_reports_per_s"]), raw, rel_tol=1e-12
            ):
                raise AssertionError("joint scaling raw capacity formula is inconsistent")
            if float(row["target_utilization"]) != 0.7:
                raise AssertionError("joint scaling target utilization must be 0.70")
            if not math.isclose(
                float(row["projected_sustainable_rate_reports_per_s"]),
                0.7 * raw,
                rel_tol=1e-12,
            ):
                raise AssertionError("joint scaling 70% capacity formula is inconsistent")
            required_raw = math.ceil(20_000.0 / anchor_rate)
            if int(float(row["required_node_equivalents_for_20k_raw"])) != required_raw:
                raise AssertionError("joint scaling raw node requirement is inconsistent")
            required_planning = math.ceil(20_000.0 / (0.7 * anchor_rate))
            if int(float(
                row["required_node_equivalents_for_20k_at_target_utilization"]
            )) != required_planning:
                raise AssertionError("joint scaling planning node requirement is inconsistent")

        gpu_rows = gpu_dpf_interface.load_rows(DATA_DIR / "fig_gpu_dpf_external.csv")
        expected_gpu = {16_384: 52_536, 65_536: 15_392, 262_144: 3_967, 1_048_576: 923}
        if {int(row["N"]): int(float(row["dpf_per_second"])) for row in gpu_rows} != expected_gpu:
            raise AssertionError("external GPU-DPF rows differ from the published V100 table")
        interpolated = gpu_dpf_interface.interpolate_rate(gpu_rows, 1 << 17)
        if (
            interpolated.lower_n != 65_536
            or interpolated.upper_n != 262_144
            or not math.isclose(
                interpolated.dpf_per_second,
                math.sqrt(15_392.0 * 3_967.0),
                rel_tol=1e-12,
            )
        ):
            raise AssertionError("GPU-DPF interface interpolation changed")
        for row in gpu_rows:
            if row["source_type"] != "external_artifact_measurement":
                raise AssertionError("GPU-DPF context has wrong source_type")
            if row["hardware"] != "NVIDIA V100" or row["prf"] != "AES-128":
                raise AssertionError("GPU-DPF context hardware or PRF is inconsistent")
            if int(row["batch_size"]) != 512 or int(row["entry_bits"]) != 512:
                raise AssertionError("GPU-DPF context batch or entry width is inconsistent")
            if any(
                row[key] != "false"
                for key in (
                    "client_keygen_included",
                    "second_server_included",
                    "chafe_packed_write_accumulation_included",
                    "directly_comparable_to_chafe",
                )
            ):
                raise AssertionError("GPU-DPF context must remain explicitly non-comparable")
        notes.append(
            "Google DPF thread anchors, joint node-thread projection, and external GPU scope checks passed."
        )
        return
    notes.append("Microbenchmark normalized-work and timing checks passed.")


def validate_packed64_safety(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "packed64_safety.csv")
    if not rows:
        raise AssertionError("packed64 safety table is empty")
    for row in rows:
        if row["source_type"] != "formula":
            raise AssertionError("packed64 safety rows must be formula bounds")
        if int(row["count_bits"]) != 8 or int(row["payload_bits"]) != 56:
            raise AssertionError("packed64 field widths disagree with the implementation")
        if int(row["unsafe_occupancy_threshold"]) != 256:
            raise AssertionError("31-bit payload configuration should be count-limited at 256")
        if float(row["log10_epoch_overflow_upper_bound"]) >= -100:
            raise AssertionError("packed64 overflow bound is too large")
    notes.append("Packed64 epoch-wide occupancy-overflow bounds passed.")


def validate_epoch_stage_benchmarks(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_epoch_end_stage_cost.csv")
    expected_phases = {
        "Chafe Drop": {"compact", "pad", "permute", "total"},
        "Chafe Pool": {"compact", "pad", "permute", "total"},
        "Trusted shuffle": {"copy", "pad", "permute", "total"},
        "Trusted padded shuffle": {"copy", "pad", "permute", "total"},
    }
    grouped = by_method(rows)
    if set(grouped) != set(expected_phases):
        raise AssertionError(f"unexpected epoch-stage methods: {sorted(grouped)}")

    local_real_counts: dict[str, int] = {}
    local_totals: dict[str, float] = {}
    for method, method_rows in grouped.items():
        if {row["phase"] for row in method_rows} != expected_phases[method]:
            raise AssertionError(f"incomplete epoch-stage phases for {method}")
        if any(row["source_type"] != "microbenchmark" for row in method_rows):
            raise AssertionError("epoch-stage rows must be measured microbenchmarks")
        if any(
            row["benchmark"] != "numpy_epoch_release_and_trusted_shuffle"
            for row in method_rows
        ):
            raise AssertionError("epoch-stage benchmark provenance is inconsistent")
        if any(int(float(row["repeats"])) < 20 for row in method_rows):
            raise AssertionError("epoch-stage benchmark needs at least 20 measured repeats")
        if any(int(float(row["warmups"])) < 1 for row in method_rows):
            raise AssertionError("epoch-stage benchmark must record a warmup")
        for row in method_rows:
            mean = float(row["time_ms"])
            p50 = float(row["p50_ms"])
            p95 = float(row["p95_ms"])
            if min(mean, p50, p95) <= 0 or p95 < p50:
                raise AssertionError(f"invalid epoch-stage timing summary for {method}")
            m = int(float(row["m"]))
            M = int(float(row["M"]))
            if m != 100_000 or abs(float(row["tau"]) - M / m) > 1e-12:
                raise AssertionError("epoch-stage m/M/tau metadata is inconsistent")
            if "no network or MPC" not in row["timed_scope"]:
                raise AssertionError("local stage scope must explicitly exclude network and MPC")

        phase_means = {
            row["phase"]: float(row["time_ms"])
            for row in method_rows
        }
        component_sum = sum(
            value for phase, value in phase_means.items() if phase != "total"
        )
        if abs(phase_means["total"] - component_sum) > max(0.05, 0.03 * phase_means["total"]):
            raise AssertionError(f"epoch-stage phase times do not sum to total for {method}")
        local_totals[method] = phase_means["total"]
        local_real_counts[method] = int(float(method_rows[0]["real_output_count"]))

        if method.startswith("Chafe"):
            if int(float(method_rows[0]["N"])) != 131_072:
                raise AssertionError("Chafe epoch-stage benchmark uses an unexpected N")
            if int(float(method_rows[0]["M"])) != 200_000:
                raise AssertionError("Chafe epoch-stage benchmark uses an unexpected M")

    if not (
        0 < local_real_counts["Chafe Drop"]
        <= local_real_counts["Chafe Pool"]
        <= 100_000
    ):
        raise AssertionError("Drop/Pool real-output counts violate occupancy semantics")

    loopback = read_csv(DATA_DIR / "fig_epoch_end_loopback.csv")
    if {row["method"] for row in loopback} != {
        "Two-process loopback Drop",
        "Two-process loopback Pool",
    } or len(loopback) != 2:
        raise AssertionError("two-process loopback rows are incomplete")
    for row in loopback:
        if row["source_type"] != "microbenchmark":
            raise AssertionError("loopback rows must be measured microbenchmarks")
        if row["benchmark"] != "two_process_loopback_epoch_release":
            raise AssertionError("loopback benchmark provenance is inconsistent")
        if int(float(row["repeats"])) < 10 or int(float(row["warmups"])) < 1:
            raise AssertionError("loopback benchmark has insufficient repeats/warmup")
        mean = float(row["time_ms"])
        p50 = float(row["p50_ms"])
        p95 = float(row["p95_ms"])
        if min(mean, p50, p95) <= 0 or p95 < p50:
            raise AssertionError("loopback timing summary is invalid")
        N = int(float(row["N"]))
        M = int(float(row["M"]))
        m = int(float(row["m"]))
        if (N, m, M) != (131_072, 100_000, 200_000):
            raise AssertionError("loopback configuration metadata is inconsistent")
        if int(float(row["counter_bytes_each_direction"])) != N * 4:
            raise AssertionError("loopback counter byte accounting is incorrect")
        if int(float(row["release_share_bytes_transferred"])) != M * 8:
            raise AssertionError("loopback release-share byte accounting is incorrect")
        if "analyzer colocated with S0" not in row["timed_scope"]:
            raise AssertionError("loopback scope must disclose analyzer colocation")
        decoder = row["decoder"].capitalize()
        local_method = f"Chafe {decoder}"
        if int(float(row["real_output_count"])) != local_real_counts[local_method]:
            raise AssertionError("loopback and local stage output counts disagree")
        if mean <= local_totals[local_method]:
            raise AssertionError("loopback total unexpectedly excludes its IPC work")

    notes.append(
        "Local epoch-stage and scoped two-process loopback measurements passed provenance, "
        "configuration, repeat-count, timing-summary, byte-accounting, and output-count checks."
    )


def validate_cgp_shuffle_model(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_cgp_shuffle_model.csv")
    report_counts = {
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
    }
    methods = {
        "CGP full secret-shared shuffle (paper-derived)",
        "CGP payload-only ideal (author-derived)",
    }
    expected = {(m, method) for m in report_counts for method in methods}
    observed = {(int(row["m"]), row["method"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise AssertionError("CGP model grid is incomplete or duplicated")

    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        m = int(row["m"])
        N = int(row["N"])
        log_n = int(row["log_n"])
        T = int(row["T"])
        d = int(row["d"])
        security_bits = int(row["security_bits"])
        payload_bits = int(row["payload_bits"])
        bandwidth = float(row["bandwidth_bits_per_second"])
        aes_seconds = float(row["aes_seconds_per_block"])

        expected_N = 1 << (m - 1).bit_length()
        if N != expected_N or N & (N - 1):
            raise AssertionError(f"CGP N is not the next power of two at m={m}")
        if log_n != N.bit_length() - 1:
            raise AssertionError(f"CGP log_n is inconsistent at m={m}")
        if T < 2 or T > N or T & (T - 1):
            raise AssertionError(f"CGP Benes parameter T is invalid at m={m}")
        log_t = T.bit_length() - 1
        if d != 2 * math.ceil(log_n / log_t) - 1:
            raise AssertionError(f"CGP decomposition depth d is inconsistent at m={m}")
        if (security_bits, payload_bits) != (128, 64):
            raise AssertionError("CGP security or payload width changed")
        if not math.isclose(bandwidth, 1e9, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError("CGP model must disclose the 1 Gbps assumption")
        if not math.isclose(aes_seconds, 3.5e-9, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError("CGP model must disclose the 3.5 ns AES assumption")
        expected_source = (
            "literature_derived_analytic_cost"
            if row["method"] == "CGP full secret-shared shuffle (paper-derived)"
            else "author_derived_analytic_scenario"
        )
        if (
            row["source_type"] != expected_source
            or row["measured"] != "false"
            or row["citation_key"] != "DBLP:conf/asiacrypt/ChaseGP20"
            or row["eprint_url"] != "https://eprint.iacr.org/2019/1340"
        ):
            raise AssertionError("CGP row has ambiguous literature-derived provenance")

        aes_calls = float(row["aes_calls_one_permute_share"])
        communication_bits = float(row["communication_bits_one_permute_share"])
        computation_seconds = float(row["computation_seconds_one_permute_share"])
        communication_seconds = float(row["communication_seconds_one_permute_share"])
        one_call_seconds = float(row["one_permute_share_seconds"])
        full_seconds = float(row["full_shuffle_seconds"])
        payload_only_seconds = float(
            row["payload_only_ideal_seconds"]
        )
        cost_seconds = float(row["cost_seconds"])
        numeric = [
            aes_calls,
            communication_bits,
            computation_seconds,
            communication_seconds,
            one_call_seconds,
            full_seconds,
            payload_only_seconds,
            cost_seconds,
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise AssertionError("CGP model contains a non-finite cost")
        if min(numeric[:7]) <= 0 or cost_seconds <= 0:
            raise AssertionError("CGP model contains a non-positive primitive cost")

        expected_aes_calls = (
            3 * d * N * log_t
            + d * N * T * (2.0 + payload_bits / security_bits)
        )
        expected_communication_bits = (
            3 * d * N * security_bits * log_t
            + (d + 1) * N * payload_bits
        )
        expected_payload_only = 2.0 * N * payload_bits / bandwidth
        equations = [
            (aes_calls, expected_aes_calls, "AES-call count"),
            (communication_bits, expected_communication_bits, "communication bits"),
            (computation_seconds, aes_calls * aes_seconds, "computation time"),
            (communication_seconds, communication_bits / bandwidth, "communication time"),
            (one_call_seconds, computation_seconds + communication_seconds, "Permute+Share time"),
            (full_seconds, 2.0 * one_call_seconds, "full-shuffle time"),
            (payload_only_seconds, expected_payload_only, "payload-only ideal"),
        ]
        for actual, expected_value, name in equations:
            if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-15):
                raise AssertionError(f"CGP {name} formula is inconsistent at m={m}")

        # The model claims to minimize the stated AES-plus-bandwidth expression
        # over all admissible power-of-two T values, so independently recheck it.
        candidate_costs = []
        for candidate_log_t in range(1, log_n + 1):
            candidate_T = 1 << candidate_log_t
            candidate_d = 2 * math.ceil(log_n / candidate_log_t) - 1
            candidate_aes = (
                3 * candidate_d * N * candidate_log_t
                + candidate_d
                * N
                * candidate_T
                * (2.0 + payload_bits / security_bits)
            )
            candidate_bits = (
                3 * candidate_d * N * security_bits * candidate_log_t
                + (candidate_d + 1) * N * payload_bits
            )
            candidate_costs.append(
                candidate_aes * aes_seconds + candidate_bits / bandwidth
            )
        if not math.isclose(one_call_seconds, min(candidate_costs), rel_tol=1e-12):
            raise AssertionError(f"CGP model did not choose a minimum-cost T at m={m}")

        if row["method"] == "CGP full secret-shared shuffle (paper-derived)":
            if not math.isclose(cost_seconds, full_seconds, rel_tol=1e-12):
                raise AssertionError("CGP full row does not charge two Permute+Share calls")
            if "two sequential Permute+Share calls" not in row["formula_scope"]:
                raise AssertionError("CGP full-row scope omits the two-call construction")
        else:
            if not math.isclose(cost_seconds, payload_only_seconds, rel_tol=1e-12):
                raise AssertionError("CGP payload-only row does not match its stated scenario")
            if (
                "author-derived payload-only ideal" not in row["formula_scope"]
                or "not a protocol-independent lower bound" not in row["formula_scope"]
            ):
                raise AssertionError("CGP payload-only row does not disclose its idealized scope")
        grouped[m][row["method"]] = row

    for m, group in grouped.items():
        full = float(
            group["CGP full secret-shared shuffle (paper-derived)"]["cost_seconds"]
        )
        payload_only = float(
            group["CGP payload-only ideal (author-derived)"]["cost_seconds"]
        )
        if payload_only >= full:
            raise AssertionError(f"CGP payload-only ideal is not below full cost at m={m}")

    representative = grouped[100_000][
        "CGP full secret-shared shuffle (paper-derived)"
    ]
    if (
        int(representative["N"]) != 131_072
        or int(representative["T"]) != 8
        or int(representative["d"]) != 11
        or not math.isclose(
            float(representative["full_shuffle_seconds"]),
            3.815899136,
            rel_tol=1e-12,
        )
    ):
        raise AssertionError("CGP m=100000 representative cost no longer matches the stated model")
    notes.append(
        "CGP paper-derived full-shuffle and author-derived payload-only grids passed "
        "formula, minimization, citation, and non-measurement provenance checks."
    )


def validate_epoch_close_scaling(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_epoch_close_scaling.csv")
    report_counts = {
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
    }
    method_by_decoder = {
        "drop": "Chafe Drop two-process close",
        "pool": "Chafe Pool two-process close",
    }
    expected = {
        (m, decoder)
        for m in report_counts
        for decoder in method_by_decoder
    }
    observed = {(int(row["m"]), row["decoder"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise AssertionError("epoch-close scaling grid is incomplete or duplicated")

    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    platform_tuples: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        decoder = row["decoder"]
        m = int(row["m"])
        N = int(row["N"])
        M = int(row["M"])
        repeats = int(row["repeats"])
        warmups = int(row["warmups"])
        real_count = int(row["real_output_count"])
        expected_N = 1 << (m - 1).bit_length()
        if row["method"] != method_by_decoder.get(decoder):
            raise AssertionError("epoch-close method and decoder labels disagree")
        if N != expected_N or M != 2 * m:
            raise AssertionError(f"epoch-close dimensions are inconsistent at m={m}")
        if not math.isclose(float(row["rho"]), m / N, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"epoch-close rho is inconsistent at m={m}")
        if not math.isclose(float(row["tau"]), M / m, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"epoch-close tau is inconsistent at m={m}")
        if not 0 < real_count <= m:
            raise AssertionError(f"epoch-close real-output count is invalid at m={m}")
        if repeats < 10 or warmups < 1:
            raise AssertionError("epoch-close scaling has insufficient repeats or warmups")
        if int(row["validated_releases"]) != repeats + warmups:
            raise AssertionError("epoch-close validated-release count excludes a run")
        if int(row["validated_measured_releases"]) != repeats:
            raise AssertionError("epoch-close did not validate every measured release")
        if int(row["counter_bytes_each_direction"]) != 4 * N:
            raise AssertionError("epoch-close counter byte accounting is inconsistent")
        if int(row["release_share_bytes_transferred"]) != 8 * M:
            raise AssertionError("epoch-close release-share byte accounting is inconsistent")

        timing = [
            float(row["time_ms"]),
            float(row["standard_error_ms"]),
            float(row["p50_ms"]),
            float(row["p95_ms"]),
        ]
        if not all(math.isfinite(value) for value in timing):
            raise AssertionError("epoch-close timing summary contains a non-finite value")
        if timing[0] <= 0 or timing[1] < 0 or timing[2] <= 0 or timing[3] < timing[2]:
            raise AssertionError("epoch-close timing summary is internally invalid")
        if (
            row["correctness"] != "true"
            or row["source_type"] != "microbenchmark"
            or row["benchmark"] != "two_process_localhost_epoch_close_scaling"
            or row["deployment_scope"] != "same_host_two_processes_not_network_deployment"
            or row["transport"] != "local_socketpair"
        ):
            raise AssertionError("epoch-close measurement provenance is ambiguous")
        if "analyzer colocated" not in row["timed_scope"]:
            raise AssertionError("epoch-close timed scope omits analyzer colocation")
        if "excludes epoch-state generation" not in row["timed_scope"]:
            raise AssertionError("epoch-close timed scope omits an important exclusion")
        if "entire fixed-length release" not in row["validation_scope"]:
            raise AssertionError("epoch-close validation scope is not release-complete")
        if "no cross-machine network" not in row["harness_limitation"]:
            raise AssertionError("epoch-close row does not disclose localhost scope")
        try:
            if int(row["checksum"]) < 0:
                raise ValueError
        except ValueError as exc:
            raise AssertionError("epoch-close correctness checksum is invalid") from exc
        platform_tuple = (
            row["platform_system"],
            row["platform_release"],
            row["platform_machine"],
            row["python_version"],
            row["cpu_count"],
        )
        if not all(platform_tuple) or int(row["cpu_count"]) < 1:
            raise AssertionError("epoch-close platform metadata is incomplete")
        platform_tuples.add(platform_tuple)
        grouped[m][decoder] = row

    if len(platform_tuples) != 1:
        raise AssertionError("epoch-close CSV mixes measurements from different platforms")
    for m, group in grouped.items():
        if int(group["drop"]["real_output_count"]) > int(
            group["pool"]["real_output_count"]
        ):
            raise AssertionError(f"Drop emits more real slots than Pool at m={m}")

    representative = grouped[100_000]
    if any(
        int(representative[decoder]["N"]) != 131_072
        or int(representative[decoder]["M"]) != 200_000
        or representative[decoder]["correctness"] != "true"
        for decoder in method_by_decoder
    ):
        raise AssertionError("epoch-close m=100000 representative rows lost their declared scope")
    notes.append(
        "Measured two-process epoch-close scaling passed complete-grid, full-release "
        "correctness, byte-accounting, platform, and localhost-scope checks."
    )


def validate_finite_burst_latency(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_finite_burst_latency.csv")
    expected_methods = {
        "Chafe CPU 8T (measured-rate model)",
        "Chafe GPU-kernel proxy (interpolated)",
        "Chafe CPU 31-equivalent (ideal capacity)",
        "CGP full model",
    }
    if {row["method"] for row in rows} != expected_methods or len(rows) != 4:
        raise AssertionError("finite-burst latency grid is incomplete or duplicated")
    for row in rows:
        if (int(row["m"]), int(row["N"])) != (100_000, 1 << 17):
            raise AssertionError("finite-burst dimensions changed")
        if not math.isclose(float(row["arrival_rate_reports_per_s"]), 10_000.0):
            raise AssertionError("finite-burst arrival rate changed")
        if not math.isclose(float(row["arrival_seconds"]), 10.0):
            raise AssertionError("finite-burst collection horizon must be 10 seconds")
        if float(row["total_seconds"]) < 10.0:
            raise AssertionError("finite-burst completion precedes the last arrival")
    by_method = {row["method"]: row for row in rows}
    cpu = by_method["Chafe CPU 8T (measured-rate model)"]
    gpu = by_method["Chafe GPU-kernel proxy (interpolated)"]
    cluster = by_method["Chafe CPU 31-equivalent (ideal capacity)"]
    cgp = by_method["CGP full model"]
    if cpu["source_type"] != "analytic_cost_from_measured_cpu_inputs":
        raise AssertionError("CPU burst point lost its measured-input provenance")
    if gpu["source_type"] != "analytic_cost_from_external_gpu_kernel":
        raise AssertionError("GPU burst point lost its external-kernel provenance")
    if cluster["source_type"] != "analytic_cost_from_measured_cpu_inputs":
        raise AssertionError("CPU projection lost its measured-input provenance")
    if not (
        int(float(cluster["node_equivalents"])) == 31
        and math.isclose(float(cluster["node_scaling_exponent"]), 1.0)
        and "ideal independent-capacity replication" in cluster["service_scope"]
    ):
        raise AssertionError("finite-burst CPU scenario is not an explicit ideal projection")
    if cgp["source_type"] != "arrival_plus_literature_derived_analytic_cost":
        raise AssertionError("CGP burst point lost its literature-derived provenance")
    for row in (cpu, gpu, cluster):
        arrival = float(row["arrival_seconds"])
        rate = float(row["service_rate_reports_per_s"])
        processed = min(float(row["m"]), arrival * rate)
        backlog = float(row["m"]) - processed
        drain = backlog / rate
        if not (
            math.isclose(float(row["processed_by_last_arrival"]), processed)
            and math.isclose(float(row["backlog_at_last_arrival"]), backlog)
            and math.isclose(float(row["drain_seconds"]), drain)
            and math.isclose(
                float(row["total_seconds"]),
                arrival + drain + float(row["epoch_close_seconds"]),
            )
        ):
            raise AssertionError("Chafe finite-burst queue identity changed")
    if not (
        math.isclose(float(cgp["processed_by_last_arrival"]), 0.0)
        and math.isclose(float(cgp["backlog_at_last_arrival"]), float(cgp["m"]))
        and math.isclose(
            float(cgp["total_seconds"]),
            float(cgp["arrival_seconds"]) + float(cgp["post_arrival_seconds"]),
        )
    ):
        raise AssertionError("CGP total must include collection plus shuffle")
    if not (
        float(cluster["total_seconds"])
        < float(gpu["total_seconds"])
        < float(cgp["total_seconds"])
        < float(cpu["total_seconds"])
    ):
        raise AssertionError("finite-burst ordering changed unexpectedly")
    if (
        float(cpu["backlog_fraction"]) < 0.95
        or not 0.20 < float(gpu["backlog_fraction"]) < 0.25
        or not math.isclose(float(cluster["backlog_fraction"]), 0.0)
    ):
        raise AssertionError("finite-burst capacity regimes changed")
    crossover = float(cpu["m"]) / (
        float(cgp["total_seconds"]) - float(cpu["epoch_close_seconds"])
    )
    if not 7_200.0 < crossover < float(gpu["service_rate_reports_per_s"]) < 8_000.0:
        raise AssertionError("finite-burst CGP crossover changed")
    if not (
        float(cluster["sustainable_rate_reports_per_s"])
        >= float(cluster["arrival_rate_reports_per_s"])
        and 0.65 < float(cluster["utilization_offered_load"]) < 0.70
    ):
        raise AssertionError("finite-burst planned CPU capacity changed")
    notes.append(
        "Finite-burst latency passed dimensions, queue identities, capacity regimes, "
        "CGP crossover, and provenance checks for CPU, external-GPU, and CGP inputs."
    )


def validate_full_release_proxy_utility(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_full_release_proxy_utility.csv")
    datasets = {"Synthetic", "Adult", "Covertype"}
    tasks = {"Range", "Heavy hitter"}
    methods = {"Pure IdealShuffle", "Padded Ideal", "Drop"}
    targets = {0.08, 0.16, 0.32}
    expected = {
        (dataset, task, target, method)
        for dataset in datasets
        for task in tasks
        for target in targets
        for method in methods
    }
    observed = {
        (
            row["dataset"],
            row["task"],
            float(row["target_tilde_epsilon_prime"]),
            row["method"],
        )
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise AssertionError("proxy-calibrated full-release utility grid is incomplete or duplicated")

    grouped: dict[
        tuple[str, str, float], dict[str, dict[str, str]]
    ] = defaultdict(dict)
    real_count_summaries: dict[
        tuple[str, str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    budgets: dict[tuple[float, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        task = row["task"]
        method = row["method"]
        target = float(row["target_tilde_epsilon_prime"])
        m = int(row["m"])
        M = int(row["M"])
        eps0 = float(row["local_epsilon0"])
        achieved = float(row["achieved_tilde_epsilon_prime"])
        value = float(row["value"])
        standard_error = float(row["standard_error"])
        padded_difference = float(row["paired_drop_minus_padded_ideal"])
        padded_se = float(row["paired_drop_minus_padded_ideal_standard_error"])
        padded_se_alias = float(row["paired_difference_standard_error"])
        padded_ratio = float(row["drop_over_padded_ideal_ratio"])
        pure_difference = float(row["paired_drop_minus_pure_ideal_shuffle"])
        pure_se = float(row["paired_drop_minus_pure_ideal_shuffle_standard_error"])
        pure_ratio = float(row["drop_over_pure_ideal_shuffle_ratio"])
        real_mean = float(row["observed_mean_real_output_count"])
        real_se = float(row["observed_real_output_count_standard_error"])
        expected_dummy = float(row["expected_dummy_count"])
        dummy_mean = float(row["observed_mean_dummy_count"])
        dummy_se = float(row["observed_dummy_count_standard_error"])
        numeric = [
            eps0,
            achieved,
            value,
            standard_error,
            padded_difference,
            padded_se,
            padded_se_alias,
            padded_ratio,
            pure_difference,
            pure_se,
            pure_ratio,
            real_mean,
            real_se,
            expected_dummy,
            dummy_mean,
            dummy_se,
        ]
        if not all(math.isfinite(number) for number in numeric):
            raise AssertionError("proxy-calibrated utility contains a non-finite value")
        if (
            eps0 <= 0
            or standard_error < 0
            or min(padded_se, padded_se_alias, pure_se) < 0
            or min(padded_ratio, pure_ratio) <= 0
            or min(real_se, dummy_se) < 0
        ):
            raise AssertionError("proxy-calibrated utility contains an invalid estimate")
        if not 0.0 <= value <= 1.0:
            raise AssertionError("complete-release utility metric escaped its [0,1] range")
        if not math.isclose(achieved, target, rel_tol=0.0, abs_tol=1e-7):
            raise AssertionError("proxy inversion did not reach its declared target")
        if m != 40_000:
            raise AssertionError("proxy-calibrated utility epoch size changed")
        expected_M = m if method == "Pure IdealShuffle" else 2 * m
        if M != expected_M:
            raise AssertionError(f"{method} has the wrong public release length")
        if not math.isclose(float(row["tau"]), M / m, abs_tol=1e-15):
            raise AssertionError("proxy-calibrated utility tau is inconsistent")
        if int(row["output_length_observed"]) != M:
            raise AssertionError("observed release length differs from its public method length")
        if int(row["seeds"]) != 100:
            raise AssertionError("proxy-calibrated utility must retain 100 paired replicates")
        if row["proxy_targets_evaluated"] != "0.08|0.16|0.32":
            raise AssertionError("proxy-calibrated utility target metadata is incomplete")

        required_true = {
            "proxy_used_in_calibration",
            "public_prior_fixed_before_epoch",
            "full_release",
            "final_release_permuted",
            "paired_workload_seeds",
            "paired_query_seeds",
            "paired_GRR_randomness",
            "paired_across_proxy_targets",
        }
        required_false = {
            "central_epsilon_claimed",
            "mechanism_wide_dp_claimed",
            "exact_accountant_used",
            "dummy_labels_visible",
        }
        if any(row[field] != "true" for field in required_true):
            raise AssertionError("proxy-calibrated utility lost a required release/proxy flag")
        if any(row[field] != "false" for field in required_false):
            raise AssertionError("proxy-calibrated utility overstates privacy or analyzer visibility")
        if (
            row["source_type"] != "monte_carlo_complete_release"
            or row["privacy_scope"] != "quadratic_design_proxy_not_finite_DP_guarantee"
            or row["randomizer"] != "generalized_randomized_response"
            or row["histogram_postprocessing"] != "euclidean_nonnegative_simplex_sum_m"
        ):
            raise AssertionError("proxy-calibrated utility has ambiguous evidence provenance")
        prior_size = int(row["public_prior_domain_size"])
        prior_min = float(row["public_prior_min_probability"])
        prior_max = float(row["public_prior_max_probability"])
        if (
            prior_size < 2
            or not all(math.isfinite(value) for value in (prior_min, prior_max))
            or not 0.0 <= prior_min <= prior_max <= 1.0
            or not row["public_prior_source"]
        ):
            raise AssertionError("proxy-calibrated utility public-prior metadata is invalid")

        if task == "Range":
            if (
                row["metric"] != "mean_normalized_absolute_range_error_over_m"
                or row["higher_metric_is_better"] != "false"
                or int(row["range_queries"]) != 500
                or row["top_k"]
            ):
                raise AssertionError("Range utility metric metadata is inconsistent")
        else:
            if (
                row["metric"] != "top_50_f1"
                or row["higher_metric_is_better"] != "true"
                or int(row["top_k"]) != 50
                or row["range_queries"]
            ):
                raise AssertionError("heavy-hitter utility metric metadata is inconsistent")

        survival = float(row["singleton_survival_exact_finite"])
        if method == "Pure IdealShuffle":
            if (
                row["decoder"] != "all_reports_no_padding"
                or row["privacy_target_type"] != "pure_ideal_shuffle_quadratic_proxy"
                or row["N"]
                or row["rho"]
                or row["release_length_rule"] != "m_no_padding"
                or row["public_prior_used_for_padding"] != "false"
                or row["uses_dummies"] != "false"
                or row["dummy_count_fixed"] != "true"
                or row["dummy_law"] != "none"
                or row["dummy_law_matches_proxy_reference"] != "not_applicable"
                or row["real_output_count_visible"] != "true"
                or row["estimator_observes_real_output_count"] != "true"
                or row["estimator_uses_expected_singleton_survival"] != "false"
                or row["estimator_uses_expected_dummy_contribution"] != "false"
                or survival != 1.0
                or real_mean != m
                or real_se != 0.0
                or expected_dummy != 0.0
                or dummy_mean != 0.0
                or dummy_se != 0.0
            ):
                raise AssertionError("Pure IdealShuffle release/dummy semantics are inconsistent")
        elif method == "Padded Ideal":
            if (
                row["decoder"] != "all_reports_same_M_padding"
                or row["privacy_target_type"] != "matched_public_prior_quadratic_proxy"
                or row["N"]
                or row["rho"]
                or row["release_length_rule"] != "M_equals_tau_times_m"
                or row["public_prior_used_for_padding"] != "true"
                or row["uses_dummies"] != "true"
                or row["dummy_count_fixed"] != "true"
                or row["dummy_law"] != "expected_GRR_report_law_under_fixed_public_prior"
                or row["dummy_law_matches_proxy_reference"] != "true"
                or row["real_output_count_visible"] != "true"
                or row["estimator_observes_real_output_count"] != "true"
                or row["estimator_uses_expected_singleton_survival"] != "false"
                or row["estimator_uses_expected_dummy_contribution"] != "true"
                or survival != 1.0
                or real_mean != m
                or real_se != 0.0
                or expected_dummy != M - m
                or dummy_mean != M - m
                or dummy_se != 0.0
            ):
                raise AssertionError("Padded Ideal complete-release metadata is inconsistent")
        else:
            if not row["N"] or not row["rho"]:
                raise AssertionError("Drop row omits N or rho")
            N = int(row["N"])
            if N != m or not math.isclose(float(row["rho"]), m / N, abs_tol=1e-15):
                raise AssertionError("Drop slot count or load factor is inconsistent")
            expected_survival = (1.0 - 1.0 / N) ** (m - 1)
            if (
                row["decoder"] != "drop"
                or row["privacy_target_type"] != "matched_public_prior_quadratic_proxy"
                or row["release_length_rule"] != "M_equals_tau_times_m"
                or row["public_prior_used_for_padding"] != "true"
                or row["uses_dummies"] != "true"
                or row["dummy_count_fixed"] != "false"
                or row["dummy_law"] != "expected_GRR_report_law_under_fixed_public_prior"
                or row["dummy_law_matches_proxy_reference"] != "true"
                or row["real_output_count_visible"] != "false"
                or row["estimator_observes_real_output_count"] != "false"
                or row["estimator_uses_expected_singleton_survival"] != "true"
                or row["estimator_uses_expected_dummy_contribution"] != "true"
                or not math.isclose(
                    survival, expected_survival, rel_tol=0.0, abs_tol=1e-15
                )
            ):
                raise AssertionError("Drop singleton-survival metadata is inconsistent")
            expected_real = m * expected_survival
            if abs(real_mean - expected_real) > 6.0 * real_se + 1e-8:
                raise AssertionError("Drop observed singleton count misses its finite expectation")
            if not math.isclose(expected_dummy, M - expected_real, abs_tol=1e-8):
                raise AssertionError("Drop expected dummy count is inconsistent")
            if not math.isclose(dummy_mean, M - real_mean, abs_tol=1e-8):
                raise AssertionError("Drop observed real and dummy means do not sum to M")
            if not math.isclose(dummy_se, real_se, abs_tol=1e-15):
                raise AssertionError("Drop real/dummy count standard errors disagree")

        grouped[(dataset, task, target)][method] = row
        real_count_summaries[(dataset, task, method)].append((real_mean, real_se))
        budgets[(target, method)].append((eps0, achieved))

    for key, group in grouped.items():
        pure = group["Pure IdealShuffle"]
        padded = group["Padded Ideal"]
        drop = group["Drop"]
        pure_value = float(pure["value"])
        padded_value = float(padded["value"])
        drop_value = float(drop["value"])
        expected_padded_difference = drop_value - padded_value
        expected_padded_ratio = drop_value / padded_value
        expected_pure_difference = drop_value - pure_value
        expected_pure_ratio = drop_value / pure_value
        padded_ses = []
        pure_ses = []
        for row in (pure, padded, drop):
            if not math.isclose(
                float(row["paired_drop_minus_padded_ideal"]),
                expected_padded_difference,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise AssertionError(f"Drop/Padded Ideal paired difference is inconsistent for {key}")
            if not math.isclose(
                float(row["drop_over_padded_ideal_ratio"]),
                expected_padded_ratio,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise AssertionError(f"Drop/Padded Ideal ratio is inconsistent for {key}")
            if not math.isclose(
                float(row["paired_drop_minus_pure_ideal_shuffle"]),
                expected_pure_difference,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise AssertionError(f"Drop/Pure Ideal paired difference is inconsistent for {key}")
            if not math.isclose(
                float(row["drop_over_pure_ideal_shuffle_ratio"]),
                expected_pure_ratio,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise AssertionError(f"Drop/Pure Ideal ratio is inconsistent for {key}")
            padded_se = float(row["paired_drop_minus_padded_ideal_standard_error"])
            padded_ses.append(padded_se)
            pure_ses.append(
                float(row["paired_drop_minus_pure_ideal_shuffle_standard_error"])
            )
            if not math.isclose(
                float(row["paired_difference_standard_error"]),
                padded_se,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise AssertionError("legacy paired-SE alias no longer denotes Drop/Padded Ideal")
        assert_constant(padded_ses, f"Drop/Padded Ideal paired SE for {key}")
        assert_constant(pure_ses, f"Drop/Pure Ideal paired SE for {key}")
        if float(drop["local_epsilon0"]) <= float(padded["local_epsilon0"]):
            raise AssertionError(f"Drop did not reinvest its proxy attenuation for {key}")
        if float(padded["local_epsilon0"]) <= float(pure["local_epsilon0"]):
            raise AssertionError(f"same-M padding did not receive its expected proxy dilution for {key}")

    for key, summaries in real_count_summaries.items():
        if len(set(summaries)) != 1:
            raise AssertionError(f"placement-dependent real counts changed across targets for {key}")
    for key, values in budgets.items():
        eps0_values = [value[0] for value in values]
        achieved_values = [value[1] for value in values]
        assert_constant(eps0_values, f"proxy-selected eps0 across workloads for {key}")
        assert_constant(achieved_values, f"achieved proxy target across workloads for {key}")
    for method in methods:
        selected = [budgets[(target, method)][0][0] for target in sorted(targets)]
        if any(right <= left for left, right in zip(selected, selected[1:])):
            raise AssertionError(f"proxy-selected local budget is not increasing for {method}")

    # Utility should improve as the design target is relaxed.  Allow a small
    # Monte Carlo reversal, but reject changes larger than three pooled SEs.
    for dataset in datasets:
        for task in tasks:
            for method in methods:
                selected = [
                    grouped[(dataset, task, target)][method]
                    for target in sorted(targets)
                ]
                for left, right in zip(selected, selected[1:]):
                    left_value = float(left["value"])
                    right_value = float(right["value"])
                    pooled_se = math.hypot(
                        float(left["standard_error"]),
                        float(right["standard_error"]),
                    )
                    if task == "Range":
                        reversal = right_value - left_value
                    else:
                        reversal = left_value - right_value
                    if reversal > 3.0 * pooled_se:
                        raise AssertionError(
                            f"{dataset} {task} {method} utility significantly worsens "
                            "as the proxy target is relaxed"
                        )

    def drop_beats_pure(group: dict[str, dict[str, str]]) -> bool:
        drop_value = float(group["Drop"]["value"])
        pure_value = float(group["Pure IdealShuffle"]["value"])
        return drop_value < pure_value if group["Drop"]["task"] == "Range" else drop_value > pure_value

    tight_rows = [group for (_, _, target), group in grouped.items() if target == 0.08]
    if len(tight_rows) != 6 or not all(drop_beats_pure(group) for group in tight_rows):
        raise AssertionError("Drop no longer beats Pure IdealShuffle on all six tight-target workloads")

    adult_expected = {
        ("Range", 0.08): True,
        ("Range", 0.16): False,
        ("Range", 0.32): False,
        ("Heavy hitter", 0.08): True,
        ("Heavy hitter", 0.16): True,
        ("Heavy hitter", 0.32): False,
    }
    for (task, target), expected_gain in adult_expected.items():
        observed_gain = drop_beats_pure(grouped[("Adult", task, target)])
        if observed_gain != expected_gain:
            raise AssertionError(
                f"Adult {task} target={target} no longer exhibits the declared "
                f"Drop/Pure IdealShuffle crossover"
            )
    notes.append(
        "Proxy-calibrated Synthetic/Adult/Covertype releases passed the 54-row Pure "
        "IdealShuffle/Padded Ideal/Drop grid, method-specific release and dummy semantics, "
        "paired metrics, absolute-utility trends, tight-target gains, Adult crossovers, "
        "and non-central-DP scope checks."
    )


def validate_source_types(notes: list[str]) -> None:
    expected = {
        "fig_privacy_vs_rho.csv": {"formula"},
        "fig_privacy_vs_eps0.csv": {"formula"},
        "fig_privacy_vs_tau.csv": {"formula"},
        "fig_privacy_vs_m_large.csv": {"formula"},
        "fig_privacy_surface_rho_tau.csv": {"formula"},
        "fig_exact_vs_proxy_drop.csv": {"formula", "exact_law_numerical"},
        "fig_accountant_scaling.csv": {"measured_accountant_runtime"},
        "fig_collision_regimes.csv": {"formula"},
        "fig_utility_factors.csv": {"formula"},
        "fig_privacy_utility_frontier.csv": {"formula"},
        "fig_batch_gating_delay.csv": {"analytic_cost"},
        "fig_product_vs_dual_microbench.csv": {"microbenchmark"},
        "fig_dpf_scaling.csv": {"microbenchmark"},
        "fig_dpf_phase_breakdown.csv": {"microbenchmark"},
        "fig_dpf_throughput_vs_n.csv": {"microbenchmark"},
        "fig_dpf_thread_scaling.csv": {"microbenchmark"},
        "fig_ingestion_joint_scaling.csv": {"analytic_cost"},
        "fig_gpu_dpf_external.csv": {"external_artifact_measurement"},
        "fig_ingestion_latency_vs_rate.csv": {"analytic_cost"},
        "fig_epoch_end_stage_cost.csv": {"microbenchmark"},
        "fig_epoch_end_loopback.csv": {"microbenchmark"},
        "fig_cgp_shuffle_model.csv": {
            "literature_derived_analytic_cost",
            "author_derived_analytic_scenario",
        },
        "fig_epoch_close_scaling.csv": {"microbenchmark"},
        "fig_finite_burst_latency.csv": {
            "analytic_cost_from_measured_cpu_inputs",
            "analytic_cost_from_external_gpu_kernel",
            "arrival_plus_literature_derived_analytic_cost",
        },
        "fig_full_release_proxy_utility.csv": {"monte_carlo_complete_release"},
        "fig_range_error_vs_epsilon.csv": {"monte_carlo"},
        "fig_sum_error_vs_epsilon.csv": {"monte_carlo"},
        "fig_heavyhitter_f1_vs_epsilon.csv": {"monte_carlo"},
        "fig_fixed_eps0_range_error.csv": {"monte_carlo"},
        "fig_fixed_eps0_sum_error.csv": {"monte_carlo"},
        "fig_real_age_range_error_vs_epsilon.csv": {"monte_carlo"},
        "fig_real_hours_sum_error_vs_epsilon.csv": {"monte_carlo"},
        "fig_real_heavyhitter_f1_vs_epsilon.csv": {"monte_carlo"},
        "fig_range_error_zipf_adult.csv": {"monte_carlo"},
        "fig_sum_error_zipf_adult.csv": {"monte_carlo"},
        "fig_heavyhitter_f1_zipf_adult.csv": {"monte_carlo"},
        "fig_covertype_workloads.csv": {"monte_carlo"},
        "fig_utility_cross_dataset_summary.csv": {"monte_carlo"},
        "fig_full_release_dummy_mismatch.csv": {"monte_carlo"},
        "fig_padding_correction_utility.csv": {"monte_carlo_complete_release"},
        "fig_pool_robust_mean.csv": {"monte_carlo_fixed_eps0"},
        "fig_modern_utility.csv": {"monte_carlo_complete_release"},
        "fig_modern_utility_replicates.csv": {"monte_carlo_complete_release"},
        "fig_modern_utility_calibration.csv": {"exact_accountant", "formula"},
        "fig_modern_fixed_eps0_utility.csv": {"monte_carlo_fixed_eps0"},
        "fig_modern_fixed_eps0_utility_replicates.csv": {"monte_carlo_fixed_eps0"},
        "fig_generic_exact_accountant_validation.csv": {"exact_accountant"},
        "fig_privacy_2x2_validation.csv": {"exact_plus_poisson_validation"},
        "fig_privacy_2x2_moderate_scaling.csv": {
            "finite_vs_poisson_exact_law_scaling"
        },
        "fig_privacy_dummy_mismatch_exact.csv": {
            "finite_and_poisson_dummy_mismatch_sensitivity"
        },
        "fig_privacy_2x2_tau_exact.csv": {
            "finite_poisson_quadratic_tau_sensitivity"
        },
        "fig_privacy_large_m_accounting.csv": {
            "formula_exact_and_profile_approximation"
        },
        "fig_privacy_general_randomizer_sensitivity.csv": {
            "formula_profile_approximation"
        },
        "packed64_safety.csv": {"formula"},
    }
    for filename, allowed in expected.items():
        rows = read_csv(DATA_DIR / filename)
        seen = {row["source_type"] for row in rows}
        if not seen <= allowed:
            raise AssertionError(f"{filename} has unexpected source types {seen}; expected subset of {allowed}")
    notes.append("Source-type checks passed for all checked CSVs.")


def validate_monte_carlo_replicates(notes: list[str]) -> None:
    filenames = [
        "fig_range_error_vs_epsilon.csv",
        "fig_sum_error_vs_epsilon.csv",
        "fig_heavyhitter_f1_vs_epsilon.csv",
        "fig_fixed_eps0_range_error.csv",
        "fig_fixed_eps0_sum_error.csv",
        "fig_real_age_range_error_vs_epsilon.csv",
        "fig_real_hours_sum_error_vs_epsilon.csv",
        "fig_real_heavyhitter_f1_vs_epsilon.csv",
        "fig_range_error_zipf_adult.csv",
        "fig_sum_error_zipf_adult.csv",
        "fig_heavyhitter_f1_zipf_adult.csv",
        "fig_covertype_workloads.csv",
        "fig_full_release_dummy_mismatch.csv",
        "fig_padding_correction_utility.csv",
        "fig_pool_robust_mean.csv",
        "fig_full_release_proxy_utility.csv",
    ]
    for filename in filenames:
        for row in read_csv(DATA_DIR / filename):
            if int(float(row["seeds"])) < 20:
                raise AssertionError(f"{filename} has fewer than 20 Monte Carlo seeds")
    notes.append("Monte Carlo replicate-count checks passed.")


def validate_full_release_dummy_mismatch(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_full_release_dummy_mismatch.csv")
    decoders = {"drop", "pool"}
    privacy_modes = {"fixed_eps0", "matched_proxy_selected_eps0"}
    dummy_laws = {"matched_reference", "neutral_public", "mismatched_prior"}
    expected_keys = {
        (decoder, mode, law)
        for decoder in decoders
        for mode in privacy_modes
        for law in dummy_laws
    }
    seen_keys = {
        (row["decoder"], row["privacy_mode"], row["dummy_law"])
        for row in rows
    }
    if seen_keys != expected_keys or len(rows) != len(expected_keys):
        raise AssertionError(
            f"full-release dummy-law grid is incomplete: {sorted(seen_keys)}"
        )

    for row in rows:
        decoder = row["decoder"]
        m = int(float(row["m"]))
        N = int(float(row["N"]))
        M = int(float(row["M"]))
        tau = float(row["tau"])
        seeds = int(float(row["seeds"]))
        if seeds < 20:
            raise AssertionError("full-release experiment requires at least 20 seeds")
        if row["full_release"] != "true" or row["dummy_labels_visible"] != "false":
            raise AssertionError("full-release rows must hide real/dummy labels")
        if row["estimator_observes_real_output_count"] != "false":
            raise AssertionError("full-release estimator may not observe realized real-output count")
        if row["estimator_uses_expected_dummy_mass"] != "true":
            raise AssertionError("full-release estimator must correct expected dummy mass")
        public_n_required = decoder == "drop" or abs(float(row["dummy_mean"])) > 1e-15
        if row["public_epoch_size_n_required"] != str(public_n_required).lower():
            raise AssertionError("public-n estimator assumption is not law-specific")
        if int(float(row["output_length_observed"])) != M:
            raise AssertionError("observed full-release length does not equal M")
        if M != int(math.ceil(tau * m)):
            raise AssertionError("M/tau/m metadata is inconsistent")
        if abs(float(row["rho"]) - m / N) > 1e-12:
            raise AssertionError("rho metadata is inconsistent")

        if decoder == "drop":
            probability = (1.0 - 1.0 / N) ** (m - 1)
            expected_count = m * probability
        else:
            probability = 1.0 - (1.0 - 1.0 / N) ** m
            expected_count = N * probability
        if abs(float(row["real_output_probability_exact"]) - probability) > 1e-12:
            raise AssertionError(f"{decoder} finite occupancy probability is incorrect")
        if abs(float(row["expected_real_output_count_exact"]) - expected_count) > 1e-8:
            raise AssertionError(f"{decoder} exact expected real-output count is incorrect")
        variance = float(row["real_output_count_variance_exact"])
        if variance < 0 or not math.isfinite(variance):
            raise AssertionError(f"{decoder} exact occupancy variance is invalid")
        observed_count = float(row["observed_mean_real_output_count"])
        observed_count_se = float(row["observed_real_output_count_standard_error"])
        if abs(observed_count - expected_count) > 6.0 * observed_count_se + 1e-8:
            raise AssertionError(f"{decoder} simulated occupancy mean misses exact expectation")
        if abs(
            float(row["observed_mean_dummy_count"]) - (M - observed_count)
        ) > 1e-8:
            raise AssertionError("mean real and dummy counts do not sum to M")

        bias = float(row["bias"])
        bias_se = float(row["bias_standard_error"])
        rmse = float(row["rmse"])
        relative_bias = float(row["relative_bias"])
        relative_bias_se = float(row["relative_bias_standard_error"])
        relative_rmse = float(row["relative_rmse"])
        values = [bias, bias_se, rmse, relative_bias, relative_bias_se, relative_rmse]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError("full-release error metric is non-finite")
        if rmse + 1e-12 < abs(bias):
            raise AssertionError("RMSE must be at least absolute bias")
        if relative_rmse + 1e-12 < abs(relative_bias):
            raise AssertionError("relative RMSE must be at least absolute relative bias")
        if abs(relative_bias) > 4.0 * relative_bias_se + 1e-12:
            raise AssertionError(
                f"{decoder}/{row['dummy_law']} empirical bias is inconsistent with the known-law estimator"
            )
        if row["dummy_law"] == "matched_reference":
            if row["proxy_applies_to_row_law"] != "true":
                raise AssertionError("matched reference should be marked as proxy-applicable")
        elif row["proxy_applies_to_row_law"] != "false":
            raise AssertionError("mismatched/neutral law must not inherit matched proxy scope")

    notes.append(
        "Full length-M Drop/Pool releases, hidden dummy labels, exact finite occupancy corrections, "
        "and three public dummy laws passed validation."
    )


def validate_padding_correction_utility(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_padding_correction_utility.csv")
    tasks = {"Drop count", "Pool sum"}
    estimators = {"Naive", "Corrected"}
    taus = {1.0, 1.25, 1.5, 2.0, 3.0, 4.0}
    expected = {
        (task, tau, estimator)
        for task in tasks
        for tau in taus
        for estimator in estimators
    }
    observed = {
        (row["task"], float(row["tau"]), row["estimator"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise AssertionError("padding-correction sweep is incomplete")
    for row in rows:
        if (
            row["full_release"] != "true"
            or row["dummy_labels_visible"] != "false"
            or row["estimator_observes_real_output_count"] != "false"
            or float(row["dummy_payload_mean"]) <= 0.0
        ):
            raise AssertionError("padding-correction experiment lost its unlabeled nonzero-dummy scope")
        values = [
            float(row["relative_bias"]),
            float(row["relative_bias_standard_error"]),
            float(row["relative_rmse"]),
            float(row["relative_rmse_standard_error"]),
            float(row["theory_relative_bias"]),
        ]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError("padding-correction metric is non-finite")
        if row["estimator"] == "Corrected":
            if row["estimator_uses_expected_dummy_contribution"] != "true":
                raise AssertionError("corrected estimator does not declare dummy correction")
            if abs(values[0]) > 5.0 * values[1] + 2e-3:
                raise AssertionError("corrected estimator retains statistically significant bias")
        else:
            if row["estimator_uses_expected_dummy_contribution"] != "false":
                raise AssertionError("naive estimator unexpectedly declares dummy correction")
            if abs(values[0] - values[4]) > 5.0 * values[1] + 2e-2:
                raise AssertionError("naive padding bias disagrees with its expectation")
    for task in tasks:
        naive_bias = [
            abs(float(row["relative_bias"]))
            for row in rows
            if row["task"] == task and row["estimator"] == "Naive"
        ]
        if any(right <= left for left, right in zip(naive_bias, naive_bias[1:])):
            raise AssertionError(f"{task} naive bias does not grow with padding")
    notes.append(
        "Nonzero-dummy padding sweep passed complete-release, expected-mass correction, "
        "finite-occupancy, and naive-bias calibration checks for Drop count and Pool sum."
    )


def validate_pool_robust_mean(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_pool_robust_mean.csv")
    methods = {
        "Single secure aggregate (mean)",
        "Pool complete release (linear)",
        "Pool partial aggregates (Huber)",
        "Padded ideal reports (Huber)",
        "Public-dummy-only control",
    }
    eps0_grid = {4.0, 8.0}
    rho_grid = {2.0, 4.0, 8.0}
    contamination_grid = {0.0, 0.02, 0.05, 0.10}
    expected_keys = {
        (eps0, rho, contamination, method)
        for eps0 in eps0_grid
        for rho in rho_grid
        for contamination in contamination_grid
        for method in methods
    }
    seen_keys = {
        (
            float(row["eps0"]),
            float(row["rho"]),
            float(row["contamination_fraction"]),
            row["method"],
        )
        for row in rows
    }
    if seen_keys != expected_keys or len(rows) != len(expected_keys):
        raise AssertionError("Pool robust-mean grid is incomplete or duplicated")

    for row in rows:
        if (
            row["full_release"] != "true"
            or row["final_release_permuted"] != "true"
            or row["dummy_labels_visible"] != "false"
            or row["estimator_observes_real_output_count"] != "false"
        ):
            raise AssertionError("robust-mean estimator does not use the full unlabeled release")
        if row["pool_release_payload"] != "count_and_sum":
            raise AssertionError("Pool robust mean must use additive (count,sum) payloads")
        if row["payload_group"] != f"Z_{1 << 56}":
            raise AssertionError("robust-mean payload group no longer matches packed64")
        if row["privacy_comparison_scope"] != (
            "same_pure_eps0_local_randomizer_not_equal_central_budgets"
        ):
            raise AssertionError("fixed-eps0 comparison scope is ambiguous")
        if row["source_type"] != "monte_carlo_fixed_eps0":
            raise AssertionError("robust-mean rows have the wrong evidence type")
        m = int(row["m"])
        N = int(row["N"])
        M_pool = int(row["M_pool"])
        rho = float(row["rho"])
        if M_pool != N or abs(m / N - rho) > 1e-12:
            raise AssertionError("Pool robust-mean m/N/M metadata is inconsistent")
        expected_dummy_fraction = (1.0 - 1.0 / N) ** m
        if abs(
            float(row["expected_pool_dummy_fraction"])
            - expected_dummy_fraction
        ) > 1e-12:
            raise AssertionError("Pool expected empty-slot fraction is incorrect")
        observed_dummy = float(row["observed_mean_dummy_count"])
        observed_dummy_se = float(row["observed_dummy_count_standard_error"])
        if abs(observed_dummy / N - expected_dummy_fraction) > (
            6.0 * observed_dummy_se / N + 1e-4
        ):
            raise AssertionError("simulated Pool dummy count misses finite occupancy expectation")
        if abs(
            float(row["observed_mean_real_output_count"])
            + observed_dummy
            - N
        ) > 1e-8:
            raise AssertionError("real and dummy Pool outputs do not sum to M=N")
        metrics = [
            float(row["mean_absolute_error"]),
            float(row["rmse"]),
            float(row["p95_absolute_error"]),
            float(row["bias"]),
        ]
        if not all(math.isfinite(value) for value in metrics):
            raise AssertionError("robust-mean metric is non-finite")
        if metrics[1] + 1e-12 < abs(metrics[3]):
            raise AssertionError("robust-mean RMSE is below absolute bias")

    def row_for(method: str, eps0: float, rho: float, contamination: float) -> dict[str, str]:
        return next(
            row
            for row in rows
            if row["method"] == method
            and float(row["eps0"]) == eps0
            and float(row["rho"]) == rho
            and float(row["contamination_fraction"]) == contamination
        )

    secure = row_for("Single secure aggregate (mean)", 8.0, 4.0, 0.05)
    linear = row_for("Pool complete release (linear)", 8.0, 4.0, 0.05)
    robust = row_for("Pool partial aggregates (Huber)", 8.0, 4.0, 0.05)
    ideal = row_for("Padded ideal reports (Huber)", 8.0, 4.0, 0.05)
    control = row_for("Public-dummy-only control", 8.0, 4.0, 0.05)
    secure_mae = float(secure["mean_absolute_error"])
    linear_mae = float(linear["mean_absolute_error"])
    robust_mae = float(robust["mean_absolute_error"])
    ideal_mae = float(ideal["mean_absolute_error"])
    control_mae = float(control["mean_absolute_error"])
    if abs(linear_mae / secure_mae - 1.0) > 0.05:
        raise AssertionError("collapsing Pool does not reproduce the secure-sum utility baseline")
    if robust_mae >= 0.85 * min(secure_mae, linear_mae):
        raise AssertionError("Pool partial aggregates do not show the declared robust-analysis gain")
    if ideal_mae > robust_mae:
        raise AssertionError("Pool unexpectedly outperforms the more informative ideal report view")
    if control_mae < 3.0 * robust_mae:
        raise AssertionError("the public-dummy-only control explains too much of Pool utility")

    for eps0 in eps0_grid:
        for contamination in contamination_grid:
            for method in {
                "Single secure aggregate (mean)",
                "Padded ideal reports (Huber)",
            }:
                values = [
                    float(row_for(method, eps0, rho, contamination)["mean_absolute_error"])
                    for rho in sorted(rho_grid)
                ]
                assert_constant(values, f"{method} utility vs irrelevant rho", tol=1e-14)

    clean_secure = float(
        row_for("Single secure aggregate (mean)", 8.0, 4.0, 0.0)[
            "mean_absolute_error"
        ]
    )
    clean_robust = float(
        row_for("Pool partial aggregates (Huber)", 8.0, 4.0, 0.0)[
            "mean_absolute_error"
        ]
    )
    gain = 1.0 - robust_mae / secure_mae
    notes.append(
        "Fixed-eps0 robust-mean workload passed complete unlabeled (count,sum) release, "
        "shared-report, occupancy, dummy-only-control, and comparator checks. At eps0=8, "
        f"rho=4, and 5% contamination, Pool-Huber reduces MAE by {gain:.1%} relative "
        f"to the single secure sum, while its uncontaminated MAE changes from "
        f"{clean_secure:.4g} to {clean_robust:.4g}; this is a robustness tradeoff, not "
        "an equal-central-privacy comparison."
    )


def validate_full_release_real_workloads(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_full_release_real_workloads.csv")
    if len(rows) != 18:
        raise AssertionError("full-release workload control must contain 18 method rows")
    datasets = {"Synthetic", "Adult", "Covertype"}
    if {row["dataset"] for row in rows} != datasets:
        raise AssertionError("full-release control is missing a declared dataset")
    if {row["task"] for row in rows} != {"Range", "Heavy hitter", "Sum"}:
        raise AssertionError("full-release control is missing a declared task")
    expected = {
        (dataset, task, method)
        for dataset in datasets
        for task, methods in [
            ("Range", ["Ideal", "Drop"]),
            ("Heavy hitter", ["Ideal", "Drop"]),
            ("Sum", ["Ideal", "Pool"]),
        ]
        for method in methods
    }
    observed = {(row["dataset"], row["task"], row["method"]) for row in rows}
    if observed != expected:
        raise AssertionError(
            f"full-release workload grid mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if any(
        row["full_release"] != "true"
        or row["dummy_labels_visible"] != "false"
        or row["estimator_observes_real_output_count"] != "false"
        or row["final_release_permuted"] != "true"
        or row["dummy_law_public"] != "true"
        or row["shared_local_epsilon0_across_methods"] != "true"
        or row["central_epsilon_claimed"] != "false"
        or row["proxy_used_in_calibration"] != "false"
        for row in rows
    ):
        raise AssertionError("full-release workload view or evidence scope changed")
    if {
        (int(float(row["m"])), int(float(row["N"])), int(float(row["M"])))
        for row in rows
    } != {(40_000, 40_000, 80_000)}:
        raise AssertionError("full-release workload dimensions changed")
    if {int(float(row["seeds"])) for row in rows} != {100}:
        raise AssertionError("full-release workload replicate count changed")
    for row in rows:
        expected_eps0 = 1.0 if row["task"] == "Sum" else 5.0
        if float(row["eps0"]) != expected_eps0:
            raise AssertionError("full-release workload local budget changed")
        if row["task"] == "Heavy hitter" and int(float(row["top_k"])) != 50:
            raise AssertionError("heavy-hitter top-k changed")
        if row["task"] == "Sum" and row["no_wrap_envelope_checked"] != "true":
            raise AssertionError("full-release sum lost its no-wrap check")
        if row["task"] == "Sum":
            if (
                row["payload_group"] != f"Z_{1 << 56}"
                or row["estimator_uses_expected_dummy_contribution"] != "true"
                or float(row["dummy_mean"]) <= 0.0
                or row["sum_randomizer"]
                != "one_bit_randomized_response_with_affine_expectation"
            ):
                raise AssertionError("full-release sum is not using the declared nonzero-dummy correction")
            if not math.isfinite(float(row["sum_bias"])) or not math.isfinite(float(row["sum_rmse"])):
                raise AssertionError("full-release sum calibration metric is non-finite")
    notes.append(
        "Synthetic, Adult, and Covertype full-release controls passed the 3-task "
        "view, method-grid, fixed-local-budget, no-wrap, and 100-replicate checks."
    )


def validate_generic_exact_accountant(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_generic_exact_accountant_validation.csv")
    if not rows:
        raise AssertionError("generic exact-accountant validation CSV is empty")
    scopes = {row["accounting_scope"] for row in rows}
    if scopes != {"pair_specific", "mechanism_level_exhaustive"}:
        raise AssertionError(f"unexpected accountant scopes: {scopes}")
    cases = {row["case"] for row in rows}
    required_cases = {
        "fixed_n_replacement",
        "small_mechanism_add_remove",
        "add_remove_n0_control",
        "heterogeneous_k3_general_padding",
    }
    if cases != required_cases:
        raise AssertionError(f"missing exact-accountant cases: {cases}")

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["oracle_rational_exact"] != "true":
            raise AssertionError("every published exact-accountant row must have a Fraction oracle")
        if row["full_release"] != "true" or row["poissonized"] != "false":
            raise AssertionError("generic accountant must use the finite full release")
        if row["arbitrary_public_nu"] != "true":
            raise AssertionError("generic accountant row does not support arbitrary public nu")
        if row["P_mass_exact"] != "1/1" or row["Q_mass_exact"] != "1/1":
            raise AssertionError("Fraction oracle output distribution lost mass")
        if row["rational_distribution_tv_error_certificate"] != "0":
            raise AssertionError("exact rational distribution must have zero TV evaluation error")
        if row["rational_hockey_stick_error_certificate"] != "0":
            raise AssertionError("exact rational hockey-stick calculation must have zero error")
        if int(float(row["float_underflowed_positive_terms"])) != 0:
            raise AssertionError("float recurrence underflowed a positive coefficient")
        if float(row["oracle_max_abs_probability_error"]) > 1e-12:
            raise AssertionError("coefficient recurrence disagrees with an exhaustive output probability")
        if float(row["oracle_l1_error_P"]) > 1e-11 or float(row["oracle_l1_error_Q"]) > 1e-11:
            raise AssertionError("coefficient recurrence has excessive L1 error against oracle")
        if float(row["alpha_float_absolute_error"]) > 1e-11:
            raise AssertionError("piecewise float alpha disagrees with exact rational alpha")
        if float(row["float_profile_at_nominal_alpha"]) > float(row["delta"]):
            raise AssertionError("float alpha endpoint was not repaired upward to feasibility")
        profile = Fraction(row["exact_profile_at_alpha_fraction"])
        delta = Fraction(row["delta_fraction"])
        if profile > delta:
            raise AssertionError("exact rational alpha does not satisfy the hockey-stick target")
        if int(float(row["n_total_P"])) != int(float(row["background_size"])) + 1:
            raise AssertionError("P participation metadata is inconsistent")
        if row["adjacency"] == "replacement":
            if int(float(row["n_total_Q"])) != int(float(row["n_total_P"])):
                raise AssertionError("replacement row does not keep participation fixed")
        elif int(float(row["n_total_Q"])) != int(float(row["background_size"])):
            raise AssertionError("add/remove row has inconsistent Q participation")
        key = (
            row["case"],
            row["decoder"],
            row["dummy_law_mode"],
            row["N"],
            row["M"],
        )
        grouped[key].append(row)

    for key, group in grouped.items():
        declared_size = {int(float(row["family_size"])) for row in group}
        if declared_size != {len(group)}:
            raise AssertionError(f"family size metadata mismatch for {key}")
        pair_alphas = [Fraction(row["alpha_exact_fraction"]) for row in group]
        family_alphas = {Fraction(row["family_alpha_exact_fraction"]) for row in group}
        if family_alphas != {max(pair_alphas)}:
            raise AssertionError(f"family exact alpha is not the explicit maximum for {key}")
        marked = {
            row["pair_name"]
            for row in group
            if row["is_worst_case_exact"] == "true"
        }
        exact_worst = {
            row["pair_name"]
            for row in group
            if Fraction(row["alpha_exact_fraction"]) == max(pair_alphas)
        }
        if marked != exact_worst:
            raise AssertionError(f"wrong exact worst-case marker for {key}")
        if {row["worst_pair_float_nominal"] for row in group} - exact_worst:
            raise AssertionError(f"float family maximum disagrees with exact maximum for {key}")

    replacement = [row for row in rows if row["case"] == "fixed_n_replacement"]
    if {row["decoder"] for row in replacement} != {"drop", "pool"}:
        raise AssertionError("replacement validation is missing a decoder")
    if {row["dummy_law_mode"] for row in replacement} != {
        "uniform_public_reference",
        "mismatched_public",
    }:
        raise AssertionError("replacement validation is missing matched/mismatched nu")
    if any(
        row["accounting_scope"] != "mechanism_level_exhaustive"
        for row in replacement
    ):
        raise AssertionError("replacement rows must exhaust histograms and ordered target flips")
    if {row["family_scope_metadata"] for row in replacement} != {
        "fixed_public_size_replacement"
    }:
        raise AssertionError("replacement builder scope metadata is ambiguous")
    if {int(float(row["family_size"])) for row in replacement} != {6}:
        raise AssertionError("replacement family does not exhaust 3 histograms x 2 flips")
    mechanism = [row for row in rows if row["case"] == "small_mechanism_add_remove"]
    if any(row["accounting_scope"] != "mechanism_level_exhaustive" for row in mechanism):
        raise AssertionError("mechanism-level add/remove rows are mislabeled")
    if {row["family_scope_metadata"] for row in mechanism} != {
        "all_background_sizes_add_remove"
    }:
        raise AssertionError("all-size add/remove builder scope metadata is ambiguous")
    if {int(float(row["family_size"])) for row in mechanism} != {12}:
        raise AssertionError(
            "add/remove family does not exhaust n=0..2 histograms and both targets"
        )

    n0_rows = [row for row in rows if row["case"] == "add_remove_n0_control"]
    if {row["N"] for row in n0_rows} != {"2", "5"}:
        raise AssertionError("n=0 control must compare two slot counts")
    if {row["decoder"] for row in n0_rows} != {"drop", "pool"}:
        raise AssertionError("n=0 control must compare both decoders")
    if len({row["alpha_exact_fraction"] for row in n0_rows}) != 1:
        raise AssertionError("n=0 add/remove profile should be decoder- and N-independent")
    k3_rows = [
        row for row in rows if row["case"] == "heterogeneous_k3_general_padding"
    ]
    if {row["decoder"] for row in k3_rows} != {"drop", "pool"}:
        raise AssertionError("K=3 validation must include both decoders")
    if {row["K"] for row in k3_rows} != {"3"}:
        raise AssertionError("heterogeneous validation is not on a three-symbol domain")
    if any(row["background_types"] != "2" for row in k3_rows):
        raise AssertionError("K=3 validation must use two heterogeneous background types")
    if any(row["slotwise_padding_optimization"] != "false" for row in k3_rows):
        raise AssertionError("K=3 rows must exercise the M<N general padding path")
    if any(int(float(row["M"])) >= int(float(row["N"])) for row in k3_rows):
        raise AssertionError("K=3 general-padding metadata does not have M<N")
    notes.append(
        "Finite Drop/Pool count distributions, rational hockey-stick inversion, heterogeneous "
        "mechanism-level maxima, replacement adjacency, and n=0 controls passed exhaustive validation."
    )


def validate_privacy_2x2(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_privacy_2x2_validation.csv")
    if len(rows) != 20:
        raise AssertionError("2x2 privacy validation must contain 20 rows")
    expected_panels = {
        "drop_replacement",
        "drop_add_remove",
        "pool_replacement",
        "pool_add_remove",
    }
    if {row["panel"] for row in rows} != expected_panels:
        raise AssertionError("2x2 privacy validation is missing a decoder/adjacency case")
    if {int(row["N"]) for row in rows} != {2, 3, 4, 5, 6}:
        raise AssertionError("2x2 privacy validation has the wrong slot-count grid")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["panel"]].append(row)
        if row["quadratic_epsilon_status"] != "display_prediction_not_DP_bound":
            raise AssertionError("quadratic display is mislabeled as a DP certificate")
        if (
            row["exact_epsilon_status"]
            != "pair_specific_two_sided_hockey_stick_certificate"
        ):
            raise AssertionError("exact epsilon scope is ambiguous")
        if float(row["accountant_max_abs_probability_error"]) > 1e-12:
            raise AssertionError("2x2 coefficient accountant disagrees with Fraction oracle")
        if int(row["N"]) > int(row["M"]):
            raise AssertionError("tiny Poisson one-slot factorization requires N <= M")
        if float(row["poisson_full_marginal_factorization_error"]) > 1e-12:
            raise AssertionError("Poisson full law disagrees with its direct marginal")
        if float(row["eta_finite_two_sided"]) <= 0.0:
            raise AssertionError("finite marginal eta must be positive")
        if not math.isfinite(float(row["epsilon_finite_exact_two_sided"])):
            raise AssertionError("2x2 exact epsilon must be finite")
        if not math.isfinite(float(row["epsilon_poisson_full_exact_two_sided"])):
            raise AssertionError("2x2 full Poisson epsilon must be finite")
        if float(row["poisson_full_profile_at_exact_alpha"]) > float(row["delta"]) + 1e-10:
            raise AssertionError("full Poisson alpha is not hockey-stick feasible")
        if row["poisson_full_release_scope"] != (
            "independent_Poisson_one_slot_factors_with_slotwise_padding"
        ):
            raise AssertionError("tiny Poisson full-law provenance is ambiguous")
    for panel, group in grouped.items():
        finite_quad_order = [
            row["N"]
            for row in sorted(
                group, key=lambda row: float(row["epsilon_quad_finite_marginal"])
            )
        ]
        exact_order = [
            row["N"]
            for row in sorted(
                group, key=lambda row: float(row["epsilon_finite_exact_two_sided"])
            )
        ]
        if finite_quad_order != exact_order:
            raise AssertionError(f"quadratic and exact load ordering differ for {panel}")
    notes.append(
        "All four Drop/Pool and replacement/add-remove cases passed the exact-rational "
        "marginal-Poisson and quadratic-versus-hockey-stick validation grid."
    )


def validate_privacy_2x2_moderate(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_privacy_2x2_moderate_scaling.csv")
    if len(rows) != 40:
        raise AssertionError("moderate 2x2 privacy validation must contain 40 rows")
    expected_panels = {
        "drop_replacement",
        "drop_add_remove",
        "pool_replacement",
        "pool_add_remove",
    }
    if {row["panel"] for row in rows} != expected_panels:
        raise AssertionError("moderate validation is missing a decoder/adjacency case")
    if {int(row["background_size_common"]) for row in rows} != {8, 16, 32, 64}:
        raise AssertionError("moderate validation has the wrong background-size grid")
    if {float(row["rho_common_background"]) for row in rows} != {0.5, 1.0, 2.0}:
        raise AssertionError("moderate validation has the wrong load grid")
    keys = {
        (
            row["panel"],
            int(row["background_size_common"]),
            float(row["rho_common_background"]),
        )
        for row in rows
    }
    if len(keys) != 40:
        raise AssertionError("moderate validation contains duplicate or missing grid points")
    if any(
        int(row["background_size_common"]) == 64
        and float(row["rho_common_background"]) != 1.0
        for row in rows
    ):
        raise AssertionError("n=64 is reserved for the rho=1 convergence check")

    for row in rows:
        n = int(row["background_size_common"])
        rho = float(row["rho_common_background"])
        N = int(row["N"])
        M = int(row["M"])
        delta = float(row["delta"])
        if N != round(n / rho) or M != 2 * (n + 1) or N > M:
            raise AssertionError("moderate validation dimensions are inconsistent")
        if row["source_type"] != "finite_vs_poisson_exact_law_scaling":
            raise AssertionError("moderate validation has the wrong source type")
        if row["dummy_law_mode"] != (
            "poisson_matched_background_nonempty_aggregate"
        ):
            raise AssertionError("moderate validation does not use Poisson-matched nu")
        if row["quadratic_reference_source"] != (
            "poisson_one_slot_reference_eta_Kp_minus_Kq_over_nu"
        ):
            raise AssertionError("quadratic epsilon is not tied to reference eta")
        if abs(
            float(row["bar_eta_poisson_reference"])
            - float(row["eta_poisson_reference"])
        ) > 1e-14:
            raise AssertionError("reference eta aliases disagree")
        if row["poisson_full_release_scope"] != (
            "independent_Poisson_one_slot_factors_with_slotwise_padding"
        ):
            raise AssertionError("moderate Poisson full-law provenance is ambiguous")
        finite_epsilon = float(row["epsilon_finite_exact_two_sided"])
        poisson_epsilon = float(row["epsilon_poisson_exact_two_sided"])
        quadratic_epsilon = float(row["epsilon_reference_quadratic"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (finite_epsilon, poisson_epsilon, quadratic_epsilon)
        ):
            raise AssertionError("moderate privacy comparison contains a non-finite epsilon")
        if float(row["finite_profile_at_exact_epsilon"]) > delta + 1e-10:
            raise AssertionError("finite exact-law alpha is not profile-feasible")
        if float(row["poisson_profile_at_exact_epsilon"]) > delta + 1e-10:
            raise AssertionError("Poisson exact-law alpha is not profile-feasible")
        if float(row["epsilon_full_chi2_finite_certificate"]) + 1e-10 < finite_epsilon:
            raise AssertionError("finite full-chi2 certificate beats the exact optimum")
        if float(row["epsilon_full_chi2_poisson_certificate"]) + 1e-10 < poisson_epsilon:
            raise AssertionError("Poisson full-chi2 certificate beats the exact optimum")
        if min(
            float(row["eta_finite_two_sided"]),
            float(row["eta_poisson_two_sided"]),
            float(row["eta_poisson_reference"]),
        ) <= 0.0:
            raise AssertionError("moderate validation contains a non-positive eta")
    notes.append(
        "The 40-point moderate Drop/Pool by adjacency grid passed dimension, "
        "Poisson-reference provenance, exact-profile feasibility, full-chi2, and "
        "marginal-eta checks."
    )


def validate_privacy_dummy_mismatch(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_privacy_dummy_mismatch_exact.csv")
    expected_panels = {
        "drop_replacement",
        "drop_add_remove",
        "pool_replacement",
        "pool_add_remove",
    }
    if len(rows) != 20 or {row["panel"] for row in rows} != expected_panels:
        raise AssertionError("dummy-mismatch sweep must cover 4 cases x 5 anchors")
    if {float(row["gamma"]) for row in rows} != {0.0, 0.25, 0.5, 0.75, 1.0}:
        raise AssertionError("dummy-mismatch sweep has the wrong interpolation grid")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["panel"]].append(row)
        if (int(row["background_size_common"]), int(row["N"]), int(row["M"])) != (
            32,
            32,
            66,
        ):
            raise AssertionError("dummy-mismatch dimensions changed")
        if row["accounting_scope"] != "homogeneous_binary_pair_specific":
            raise AssertionError("dummy-mismatch accounting scope is ambiguous")
        if not all(
            math.isfinite(float(row[key])) and float(row[key]) > 0.0
            for key in (
                "epsilon_finite_exact_two_sided",
                "epsilon_poisson_exact_two_sided",
            )
        ):
            raise AssertionError("dummy-mismatch sweep contains non-finite epsilon")
        if float(row["finite_profile_at_exact_epsilon"]) > float(row["delta"]) + 1e-10:
            raise AssertionError("dummy-mismatch finite alpha is not profile-feasible")
        if float(row["poisson_profile_at_exact_epsilon"]) > float(row["delta"]) + 1e-10:
            raise AssertionError("dummy-mismatch Poisson alpha is not profile-feasible")
    endpoint_ratios = []
    for panel, group in grouped.items():
        baseline = next(row for row in group if float(row["gamma"]) == 0.0)
        if abs(float(baseline["epsilon_finite_over_matched"]) - 1.0) > 1e-12:
            raise AssertionError(f"{panel}: matched endpoint is not normalized")
        endpoint = next(row for row in group if float(row["gamma"]) == 1.0)
        endpoint_ratios.append(float(endpoint["epsilon_finite_over_matched"]))
    if not (0.67 < min(endpoint_ratios) < 0.90 and max(endpoint_ratios) < 0.90):
        raise AssertionError("dummy-mismatch endpoint no longer matches the reported sensitivity")
    notes.append(
        "Arbitrary-public-dummy sensitivity passed the four-case finite/Poisson "
        "profile sweep from matched to uniform padding."
    )


def validate_privacy_tau_exact(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_privacy_2x2_tau_exact.csv")
    expected_panels = {
        "drop_replacement",
        "drop_add_remove",
        "pool_replacement",
        "pool_add_remove",
    }
    if len(rows) != 16 or {row["panel"] for row in rows} != expected_panels:
        raise AssertionError("exact tau sweep must cover 4 cases x 4 anchors")
    if {float(row["tau_P"]) for row in rows} != {1.0, 2.0, 4.0, 8.0}:
        raise AssertionError("exact tau sweep has the wrong release ratios")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["panel"]].append(row)
        if (int(row["background_size_common"]), int(row["N"])) != (32, 32):
            raise AssertionError("exact tau sweep changed its background dimensions")
        if row["quadratic_epsilon_status"] != "reference_prediction_not_DP_bound":
            raise AssertionError("exact tau sweep overstates the quadratic curve")
        if not all(
            math.isfinite(float(row[key])) and float(row[key]) > 0.0
            for key in (
                "epsilon_finite_exact_two_sided",
                "epsilon_poisson_exact_two_sided",
                "epsilon_reference_quadratic",
            )
        ):
            raise AssertionError("exact tau sweep contains non-finite epsilon")
        if float(row["finite_profile_at_exact_epsilon"]) > float(row["delta"]) + 1e-10:
            raise AssertionError("exact tau finite alpha is not profile-feasible")
        if float(row["poisson_profile_at_exact_epsilon"]) > float(row["delta"]) + 1e-10:
            raise AssertionError("exact tau Poisson alpha is not profile-feasible")
    endpoint_ratios = []
    for panel, group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["tau_P"]))
        epsilons = [float(row["epsilon_finite_exact_two_sided"]) for row in ordered]
        if epsilons != sorted(epsilons, reverse=True):
            raise AssertionError(f"{panel}: finite privacy did not improve with padding")
        if abs(float(ordered[0]["epsilon_finite_over_tau1"]) - 1.0) > 1e-12:
            raise AssertionError(f"{panel}: tau sweep is not normalized at one")
        endpoint_ratios.append(float(ordered[-1]["epsilon_finite_over_tau1"]))
    if min(endpoint_ratios) < 0.258 or max(endpoint_ratios) > 0.273:
        raise AssertionError("exact tau endpoint no longer matches the reported range")
    notes.append(
        "Finite/Poisson/quadratic padding sensitivity passed all four decoder-adjacency cases."
    )


def validate_exact_certified_utility(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_exact_certified_utility.csv")
    trace = read_csv(DATA_DIR / "fig_exact_certified_utility_calibration.csv")
    methods = {"Padded Ideal", "Drop", "Pool"}
    if {row["method"] for row in rows} != methods or len(rows) != 3:
        raise AssertionError("exact-certified utility must contain one row per release")
    if {row["method"] for row in trace} != methods:
        raise AssertionError("exact-certified calibration trace is missing a release")
    if any(row["privacy_source_type"] != "exact_rational_accountant" for row in rows):
        raise AssertionError("certified utility has a non-exact privacy source")
    if any(
        row["utility_source_type"]
        != "exact_output_distribution_plus_independent_monte_carlo"
        for row in rows
    ):
        raise AssertionError("certified utility has ambiguous utility provenance")
    if any(
        row["full_release"] != "true"
        or row["dummy_labels_visible"] != "false"
        or row["estimator_observes_real_output_count"] != "false"
        or row["final_release_permuted"] != "true"
        for row in rows
    ):
        raise AssertionError("certified utility does not use the complete analyzer view")
    if any(row["adjacency"] != "fixed_size_replacement" for row in rows):
        raise AssertionError("certified utility adjacency is not fixed-size replacement")
    if any(row["pool_wraparound_possible"] != "false" for row in rows):
        raise AssertionError("certified utility does not rule out Pool wraparound")
    if any(int(row["K"]) <= int(row["m"]) for row in rows):
        raise AssertionError("certified utility payload group is too small for Pool sums")

    for row in rows:
        target = Fraction(row["target_alpha_fraction"])
        exact = Fraction(row["exact_alpha_fraction"])
        next_alpha = Fraction(row["next_grid_exact_alpha_fraction"])
        if target != Fraction(2):
            raise AssertionError("certified utility target alpha changed unexpectedly")
        if exact > target or row["certified_at_common_target"] != "true":
            raise AssertionError(f"{row['method']} exceeds the common exact target")
        if float(row["target_epsilon_prime"]) - float(row["exact_epsilon_prime"]) > 1e-3:
            raise AssertionError(f"{row['method']} leaves excessive privacy slack")
        if next_alpha <= target or row["next_grid_exceeds_target"] != "true":
            raise AssertionError(f"{row['method']} calibration is not grid-maximal")
        if int(row["output_length_observed"]) != int(row["M"]):
            raise AssertionError("certified utility release length metadata is inconsistent")
        exact_mae = float(row["exact_mean_absolute_count_error"])
        mc_mae = float(row["mc_mean_absolute_count_error"])
        mc_se = float(row["mc_mean_absolute_count_error_standard_error"])
        if mc_se <= 0:
            raise AssertionError("certified utility Monte Carlo SE must be positive")
        if abs(mc_mae - exact_mae) > max(6.0 * mc_se, 0.01):
            raise AssertionError(
                f"{row['method']} Monte Carlo utility disagrees with the exact law"
            )
        if int(row["exact_output_support"]) <= 0:
            raise AssertionError("certified utility exact output support is empty")
        if row["method"] in {"Drop", "Pool"}:
            if float(row["float_recurrence_abs_error"]) > 1e-12:
                raise AssertionError("float recurrence disagrees with rational calibration")
        elif row["float_recurrence_abs_error"]:
            raise AssertionError("padded Ideal incorrectly claims a hidden-slot recurrence")

    if len(trace) < len(methods) * 49:
        raise AssertionError("certified calibration trace omits the global coarse scan")
    by_trace_method = by_method(trace)
    selected = {row["method"]: Fraction(row["truthful_probability_fraction"]) for row in rows}
    for method, group in by_trace_method.items():
        feasible = [
            Fraction(row["truthful_probability_fraction"])
            for row in group
            if row["certified_at_common_target"] == "true"
        ]
        if not feasible or max(feasible) != selected[method]:
            raise AssertionError(f"{method} selected probability is not the exact grid maximum")
        if any(row["source_type"] != "exact_rational_accountant" for row in group):
            raise AssertionError("certified calibration trace has non-exact provenance")
    notes.append(
        "Exact-certified binary utility passed full length-M view, rational family-budget, "
        "grid-maximal calibration, no-wrap, and independent Monte Carlo checks."
    )


def validate_large_pairwise_exact_utility(notes: list[str]) -> None:
    calibration = read_csv(DATA_DIR / "fig_large_exact_family_calibration.csv")
    exact_rows = read_csv(DATA_DIR / "fig_large_exact_family_utility.csv")
    transferred = read_csv(
        DATA_DIR / "fig_large_real_transferred_reference_sensitivity.csv"
    )
    if len(calibration) != 12 or len(exact_rows) != 24 or len(transferred) != 24:
        raise AssertionError("large pairwise exact experiment has incomplete output")
    if any(
        row["pairwise_exact_profile_only"] != "true"
        or row["mechanism_wide_dp_claimed"] != "false"
        or row["proxy_used_in_calibration"] != "false"
        or row["monte_carlo_used_in_calibration"] != "false"
        for row in calibration
    ):
        raise AssertionError("large exact calibration overstates its privacy scope")
    for row in calibration:
        profile = float(row["profile_delta_at_target_epsilon"])
        limit = float(row["profile_limit_with_relative_slack"])
        delta = float(row["accounting_delta"])
        if profile > limit or limit >= delta:
            raise AssertionError("large exact calibration lacks conservative slack")
        if abs(
            float(row["achieved_exact_epsilon_prime"])
            - float(row["target_epsilon_prime"])
        ) > 2e-8:
            raise AssertionError("large exact calibration misses its target")
        if "not_interval_certified" not in row["numerical_status"]:
            raise AssertionError("large exact calibration hides numerical scope")
    if any(
        row["central_epsilon_claimed_for_this_workload"] != "true"
        or row["pairwise_exact_profile_only"] != "true"
        or row["mechanism_wide_dp_claimed"] != "false"
        or row["full_release"] != "true"
        or row["dummy_labels_visible"] != "false"
        or row["estimator_observes_real_output_count"] != "false"
        for row in exact_rows
    ):
        raise AssertionError("large exact-family utility has inconsistent scope")
    if any(
        row["central_epsilon_claimed_for_this_workload"] != "false"
        or row["shared_local_epsilon0_across_methods"] != "false"
        or row["mechanism_wide_dp_claimed"] != "false"
        or "transferred_reference_sensitivity" not in row["comparison_basis"]
        for row in transferred
    ):
        raise AssertionError("real-data transferred-reference rows overclaim privacy")
    notes.append(
        "Large pairwise exact-law calibration reached m=100000 with conservative "
        "profile slack and complete-release utility; heterogeneous real-data rows "
        "remain explicitly non-central-DP transferred-reference sensitivities."
    )


def validate_privacy_large_m_accounting(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_privacy_large_m_accounting.csv")
    main = [row for row in rows if row["sweep"] == "m_scaling"]
    tau = [row for row in rows if row["sweep"] == "tau_sensitivity"]
    expected_panels = {
        "drop_replacement",
        "drop_add_remove",
        "pool_replacement",
        "pool_add_remove",
    }
    if len(main) != 196 or len(tau) != 16:
        raise AssertionError("large-m privacy grid has the wrong size")
    if {row["panel"] for row in rows} != expected_panels:
        raise AssertionError("large-m privacy grid is missing a case")
    if any(
        row["poisson_account_status"] != "exact_for_poisson_surrogate"
        or row["gaussian_account_status"]
        != "profile_approximation_not_DP_certificate"
        or row["chernoff_account_status"] != "rigorous_for_poisson_surrogate"
        or row["chernoff_feasible_at_delta"] != "True"
        or float(row["epsilon_chernoff_two_sided"])
        + 1e-12
        < float(row["epsilon_poisson_exact_two_sided"])
        for row in rows
    ):
        raise AssertionError("large-m privacy rows blur exact and approximate evidence")
    for row in rows:
        exact = float(row["epsilon_poisson_exact_two_sided"])
        ideal_m = float(row["epsilon_ideal_shuffle_release_m"])
        ideal_M = float(row["epsilon_ideal_shuffle_release_M"])
        ratio_m = float(row["epsilon_poisson_over_ideal_release_m"])
        ratio_M = float(row["epsilon_poisson_over_ideal_release_M"])
        quadratic = float(row["quadratic_epsilon_scale"])
        load = float(row["rho_common"])
        padding_factor = math.sqrt(
            int(row["m_public_larger_epoch"]) / int(row["M_release"])
        )
        contraction = (1.0 - math.exp(-load)) / load
        kappa = math.exp(float(row["epsilon0"])) + math.exp(
            -float(row["epsilon0"])
        ) - 2.0
        expected_collision = {
            "closed_form_drop_over_ideal_M": math.exp(-load),
            "closed_form_pool_replacement_over_ideal_M": math.sqrt(contraction),
            "closed_form_pool_add_remove_over_ideal_M": math.sqrt(
                ((contraction - math.exp(-load)) + contraction * kappa) / kappa
            ),
        }
        if not (
            exact > 0.0
            and ideal_m > 0.0
            and ideal_M > 0.0
            and quadratic > 0.0
            and abs(ratio_m - exact / ideal_m) <= 1e-12
            and abs(ratio_M - exact / ideal_M) <= 1e-12
            and abs(
                float(row["quadratic_over_poisson_exact"])
                - quadratic / exact
            )
            <= 1e-12
        ):
            raise AssertionError("ideal-shuffle or quadratic normalization is inconsistent")
        for collision_column, expected in expected_collision.items():
            total_column = collision_column.replace("_M", "_m")
            if (
                abs(float(row[collision_column]) - expected) > 1e-12
                or abs(float(row[total_column]) - expected * padding_factor) > 1e-12
            ):
                raise AssertionError("closed-form collision or padding law is inconsistent")
        if float(row["tau_M_over_m"]) >= 1.0 and ideal_M > ideal_m + 1e-12:
            raise AssertionError("matched padding unexpectedly worsens the ideal reference")
    maximum_error = max(
        abs(float(row["gaussian_relative_epsilon_error"])) for row in main
    )
    large_error = max(
        abs(float(row["gaussian_relative_epsilon_error"]))
        for row in main
        if int(row["m_public_larger_epoch"]) >= 100_000
    )
    if maximum_error > 0.02 or large_error > 0.002:
        raise AssertionError("normal-profile approximation exceeds its error gate")
    chernoff_ratios = [
        float(row["epsilon_chernoff_over_poisson_exact"]) for row in main
    ]
    if min(chernoff_ratios) < 1.0 - 1e-10 or max(chernoff_ratios) > 1.25:
        raise AssertionError("Chernoff account is invalid or unexpectedly loose")
    for row in main:
        if row["decoder"] != "drop":
            continue
        finite = float(row["epsilon_finite_drop_exact_two_sided"])
        poisson = float(row["epsilon_poisson_exact_two_sided"])
        if abs(finite - poisson) / finite > 0.003:
            raise AssertionError("matched Drop finite and Poisson laws diverge unexpectedly")

    mechanism_wide_rows = [
        row
        for row in main
        if abs(float(row["rho_requested_m_over_N"]) - 1.0) < 1e-12
        and abs(float(row["tau_M_over_m"]) - 2.0) < 1e-12
    ]
    if len(mechanism_wide_rows) != 28:
        raise AssertionError("mechanism-wide Poisson grid is incomplete")
    for row in mechanism_wide_rows:
        p_min = float(row["slot_envelope_p_min"])
        p_max = float(row["slot_envelope_p_max"])
        epsilon_slot = float(row["epsilon_slot_envelope"])
        expected_slot = math.log(
            max(p_max / p_min, (1.0 - p_min) / (1.0 - p_max))
        )
        upper = float(row["epsilon_poisson_mechanism_wide_upper"])
        matched = float(row["epsilon_poisson_exact_two_sided"])
        if not (
            0.0 < p_min < p_max < 1.0
            and abs(epsilon_slot - expected_slot) <= 1e-12
            and upper + 1e-12 >= matched
            and upper / matched < 2.0
            and row["mechanism_wide_account_method"]
            == "fmt_clone_numerical_upper_over_binary_completed_slot_family"
        ):
            raise AssertionError("mechanism-wide Poisson slot envelope failed")
    by_decoder_m: dict[tuple[str, str], set[float]] = {}
    for row in mechanism_wide_rows:
        key = (row["decoder"], row["m_public_larger_epoch"])
        by_decoder_m.setdefault(key, set()).add(
            float(row["epsilon_poisson_mechanism_wide_upper"])
        )
    if len(by_decoder_m) != 14 or any(
        len(values) != 1 for values in by_decoder_m.values()
    ):
        raise AssertionError("decoder-wide Poisson bounds disagree by adjacency")

    general = read_csv(DATA_DIR / "fig_privacy_general_randomizer_sensitivity.csv")
    if len(general) != 16 or {int(row["domain_size_K"]) for row in general} != {
        2,
        4,
        8,
        16,
    }:
        raise AssertionError("general-randomizer sensitivity grid is incomplete")
    if any(
        float(row["payload_chi2_actual"])
        > float(row["payload_chi2_kappa_envelope"]) + 1e-12
        or float(row["epsilon_gaussian_actual_two_sided"])
        > float(row["epsilon_gaussian_kappa_two_sided"]) + 1e-12
        or row["account_status"]
        != "alpha_dependent_gaussian_profile_approximation"
        or row["chernoff_account_status"] != "rigorous_for_poisson_surrogate"
        or row["chernoff_feasible_at_delta"] != "True"
        or float(row["epsilon_chernoff_two_sided"])
        + 1e-12
        < float(row["epsilon_gaussian_actual_two_sided"])
        for row in general
    ):
        raise AssertionError("general-randomizer envelope or evidence label failed")
    notes.append(
        "Large-m one-slot accounting covers all four cases through m=10^6; "
        "the decoder-wide Poisson slot envelope dominates every matched pair "
        "at rho=1 and remains within 2x on the evaluated grid.  "
        f"Maximum normal-profile epsilon error is {maximum_error:.3%} overall "
        f"and {large_error:.3%} from m=10^5 onward, while the rigorous "
        f"Chernoff/exact ratio is {min(chernoff_ratios):.3f}--"
        f"{max(chernoff_ratios):.3f}.  K-RR sensitivity keeps "
        "actual laws separate from the pure-LDP kappa envelope."
    )


def validate_modern_utility(notes: list[str]) -> None:
    calibrations = read_csv(DATA_DIR / "fig_modern_utility_calibration.csv")
    replicates = read_csv(DATA_DIR / "fig_modern_utility_replicates.csv")
    summary = read_csv(DATA_DIR / "fig_modern_utility.csv")
    if len(calibrations) != 18 or len(replicates) != 7_200 or len(summary) != 72:
        raise AssertionError("modern utility grid is incomplete")
    if {row["dataset"] for row in summary} != {
        "Synthetic",
        "ACS 2024",
        "NYC TLC 2024",
        "Amazon Reviews'23",
    }:
        raise AssertionError("modern utility datasets are incomplete")
    if any(
        float(row["profile_at_target"]) > float(row["delta"])
        or abs(float(row["achieved_epsilon"]) - float(row["target_epsilon"]))
        > (
            5e-6
            if "poisson_chernoff" in row["privacy_source"]
            else 2e-8
        )
        for row in calibrations
    ):
        raise AssertionError("modern utility exact pair calibration missed its target")
    if any(
        int(float(row["release_length"])) != 200_000
        or row["privacy_scope"]
        != "two_sided_homogeneous_endpoint_pair_not_heterogeneous_worst_case"
        or row["source_type"] != "monte_carlo_complete_release"
        or row["utility_source_type"] != "monte_carlo_complete_release_histogram"
        or row["utility_source"] != "monte_carlo_complete_release_histogram"
        for row in replicates
    ):
        raise AssertionError("modern utility release or provenance label is inconsistent")
    if any(
        row["pairing"]
        != (
            "shared_epoch_placement_and_laplace_uniforms"
            if row["task"] == "Bounded sum"
            else "shared_epoch_placement_and_common_random_numbers"
        )
        for row in replicates
    ):
        raise AssertionError("modern utility pairing provenance is inconsistent")

    ideal_sums = {
        (row["dataset"], row["target_epsilon"], row["replicate"]): row
        for row in replicates
        if row["task"] == "Bounded sum" and row["method"] == "Padded Ideal"
    }
    pool_sums = [
        row
        for row in replicates
        if row["task"] == "Bounded sum" and row["method"] == "Pool"
    ]
    if len(ideal_sums) != 1_200 or len(pool_sums) != 1_200:
        raise AssertionError("modern bounded-sum paired grid is incomplete")
    maximum_occupancy = 0
    for row in pool_sums:
        key = (row["dataset"], row["target_epsilon"], row["replicate"])
        maximum_occupancy = max(maximum_occupancy, int(float(row["max_occupancy"])))
        if row["pool_sum_preserved"] != "1":
            raise AssertionError("Pool did not preserve its randomized bounded sum")
    if maximum_occupancy >= 16:
        raise AssertionError("modern utility trial exceeded packed64 occupancy capacity")

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in replicates:
        grouped[
            (row["dataset"], row["task"], row["method"], row["target_epsilon"])
        ].append(float(row["value"]))
    for row in summary:
        key = (row["dataset"], row["task"], row["method"], row["target_epsilon"])
        values = grouped[key]
        if len(values) != 100 or not math.isclose(
            float(row["mean"]), sum(values) / len(values), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise AssertionError("modern utility summary does not match its replicates")
    sum_calibrations = {
        (row["method"], row["target_epsilon"]): float(row["local_epsilon0"])
        for row in calibrations
        if row["privacy_family"].startswith(
            "homogeneous_q3_endpoint_replacement"
        )
    }
    if len(sum_calibrations) != 6 or any(
        sum_calibrations[("Pool", target)]
        <= sum_calibrations[("Padded Ideal", target)]
        for target in {"0.1", "0.3", "0.7"}
    ):
        raise AssertionError("Pool calibration did not increase the local Laplace budget")
    notes.append(
        "Modern utility covers one seeded synthetic workload, three frozen public "
        "snapshots, and 100 common-random-number trials per "
        "configuration; all 18 pair calibrations hit their hockey-stick targets, and "
        f"Pool preserves all 1,200 randomized sums while using a larger accounted "
        f"Laplace epsilon0 (max occupancy {maximum_occupancy})."
    )


def validate_modern_fixed_eps0_utility(notes: list[str]) -> None:
    replicates = read_csv(DATA_DIR / "fig_modern_fixed_eps0_utility_replicates.csv")
    summary = read_csv(DATA_DIR / "fig_modern_fixed_eps0_utility.csv")
    if len(replicates) != 2_400 or len(summary) != 24:
        raise AssertionError("modern fixed-eps0 utility grid is incomplete")
    if {row["dataset"] for row in summary} != {
        "Synthetic",
        "ACS 2024",
        "NYC TLC 2024",
        "Amazon Reviews'23",
    }:
        raise AssertionError("modern fixed-eps0 datasets are incomplete")
    if any(
        float(row["fixed_epsilon0"]) != 8.0
        or row["privacy_scope"]
        != "fixed_local_budget_utility_control_no_central_dp_claim"
        or row["source_type"] != "monte_carlo_fixed_eps0"
        or row["utility_source_type"] != "monte_carlo_complete_release_histogram"
        or row["utility_source"] != "monte_carlo_complete_release_histogram"
        for row in replicates
    ):
        raise AssertionError("modern fixed-eps0 scope or provenance is inconsistent")

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in replicates:
        grouped[(row["dataset"], row["task"], row["method"])].append(
            float(row["value"])
        )
    for row in summary:
        key = (row["dataset"], row["task"], row["method"])
        values = grouped[key]
        ideal = grouped[(row["dataset"], row["task"], "Padded Ideal")]
        if len(values) != 100 or not math.isclose(
            float(row["mean"]), sum(values) / len(values), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise AssertionError("modern fixed-eps0 summary does not match replicates")
        paired_gap = sum(v - i for v, i in zip(values, ideal)) / len(values)
        if not math.isclose(
            float(row["paired_difference_from_padded_ideal"]),
            paired_gap,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise AssertionError("modern fixed-eps0 paired gap is inconsistent")

    ideal_sums = {
        (row["dataset"], row["replicate"]): row
        for row in replicates
        if row["task"] == "Bounded sum" and row["method"] == "Padded Ideal"
    }
    pool_sums = [
        row
        for row in replicates
        if row["task"] == "Bounded sum" and row["method"] == "Pool"
    ]
    if len(ideal_sums) != 400 or len(pool_sums) != 400:
        raise AssertionError("modern fixed-eps0 sum control is incomplete")
    maximum_occupancy = 0
    for row in pool_sums:
        ideal = ideal_sums[(row["dataset"], row["replicate"])]
        maximum_occupancy = max(maximum_occupancy, int(row["max_occupancy"]))
        if (
            row["pool_ideal_total_equal"] != "1"
            or row["pool_ideal_estimate_equal"] != "1"
            or row["value"] != ideal["value"]
        ):
            raise AssertionError("fixed-eps0 Pool did not preserve the bounded sum")
    if maximum_occupancy >= 16:
        raise AssertionError("fixed-eps0 utility exceeded packed64 occupancy capacity")

    drop_ranges = [
        row for row in summary if row["task"] == "Range" and row["method"] == "Drop"
    ]
    drop_hh = [
        row
        for row in summary
        if row["task"] == "Heavy hitter" and row["method"] == "Drop"
    ]
    if any(float(row["ratio_to_padded_ideal"]) <= 1.0 for row in drop_ranges):
        raise AssertionError("fixed-eps0 Drop range control no longer isolates sampling loss")
    if any(float(row["paired_difference_from_padded_ideal"]) >= 0.0 for row in drop_hh):
        raise AssertionError("fixed-eps0 Drop HH control no longer isolates sampling loss")
    notes.append(
        "At fixed local epsilon0=8, Drop has 1.74--2.28x ideal range error "
        "and 0.014--0.114 lower top-50 F1; Pool equals padded ideal in all "
        f"400 bounded-sum pairs (max occupancy {maximum_occupancy})."
    )


def validate_artifact_documentation(notes: list[str]) -> None:
    readme = (ROOT / "README.md").read_text()
    artifact = (ROOT / "ARTIFACT.md").read_text()
    forbidden = {
        "k^{0.97}",
        "h^0.97",
        "35-equivalent",
        "49 equivalents",
        "70 at 70%",
        "identical randomized totals and estimates",
    }
    stale = sorted(token for token in forbidden if token in readme or token in artifact)
    if stale:
        raise AssertionError(f"artifact documentation contains stale claims: {stale}")
    required_readme = {
        "experiments/artifact_check.py",
        "experiments/build_gpu_dpf.py",
        "experiments/gpu_dpf_interface.py",
        "experiments/fetch_adult_dataset.py",
        "experiments/fetch_covertype_dataset.py",
        "experiments/fetch_modern_utility_datasets.py",
        "third_party/distributed_point_functions",
        "third_party/GPU-DPF",
    }
    missing = sorted(token for token in required_readme if token not in readme)
    if missing:
        raise AssertionError(f"artifact README is missing core artifact markers: {missing}")
    required_artifact = {
        "python experiments/artifact_check.py",
        "third_party/distributed_point_functions/",
        "third_party/GPU-DPF/",
        "experiments/build_gpu_dpf.py",
        "experiments/gpu_dpf_interface.py",
    }
    missing = sorted(token for token in required_artifact if token not in artifact)
    if missing:
        raise AssertionError(f"ARTIFACT.md is missing core artifact guidance: {missing}")
    for relative in (
        "experiments/artifact_check.py",
        "experiments/build_artifact_bundle.py",
    ):
        if not (ROOT / relative).is_file():
            raise AssertionError(f"artifact release helper is missing: {relative}")
    notes.append(
        "Artifact documentation covers the minimal run path, dataset rebuild "
        "scripts, and vendored DPF backends without stale claims."
    )


def validate_all() -> list[str]:
    notes: list[str] = []
    validate_privacy_monotonicity(notes)
    validate_collision_sum(notes)
    validate_exact_proxy(notes)
    validate_accountant_scaling(notes)
    validate_exact_ideal_shuffle(notes)
    validate_microbench(notes)
    validate_packed64_safety(notes)
    validate_epoch_stage_benchmarks(notes)
    validate_cgp_shuffle_model(notes)
    validate_epoch_close_scaling(notes)
    validate_finite_burst_latency(notes)
    validate_full_release_proxy_utility(notes)
    validate_source_types(notes)
    validate_monte_carlo_replicates(notes)
    validate_full_release_dummy_mismatch(notes)
    validate_padding_correction_utility(notes)
    validate_pool_robust_mean(notes)
    validate_full_release_real_workloads(notes)
    validate_privacy_2x2(notes)
    validate_privacy_2x2_moderate(notes)
    validate_privacy_dummy_mismatch(notes)
    validate_privacy_tau_exact(notes)
    validate_privacy_large_m_accounting(notes)
    validate_generic_exact_accountant(notes)
    validate_exact_certified_utility(notes)
    validate_large_pairwise_exact_utility(notes)
    validate_modern_utility(notes)
    validate_modern_fixed_eps0_utility(notes)
    validate_artifact_documentation(notes)
    (DATA_DIR / "validation_summary.txt").write_text("\n".join(notes) + "\n")
    return notes


if __name__ == "__main__":
    for note in validate_all():
        print(note)
