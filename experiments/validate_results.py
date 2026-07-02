from __future__ import annotations

from collections import defaultdict

from chafe_eval_lib import DATA_DIR, read_csv


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

    rows_tau = by_method(read_csv(DATA_DIR / "fig_privacy_vs_tau.csv"))
    drop_tau = sorted(rows_tau["Drop (matched padding)"], key=lambda row: float(row["tau"]))
    assert_nonincreasing(floats(drop_tau, "epsilon"), "Drop epsilon vs tau")
    ideal_tau = sorted(rows_tau["IdealShuffle"], key=lambda row: float(row["tau"]))
    assert_constant(floats(ideal_tau, "epsilon"), "IdealShuffle epsilon vs tau")
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
    exact_rows = [row for row in rows if row["source_type"] == "exact_accountant"]
    ratios = [float(row["exact_to_proxy_ratio"]) for row in exact_rows]
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


def validate_microbench(notes: list[str]) -> None:
    rows = read_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv")
    work = {row["method"]: float(row["normalized_expansion_work"]) for row in rows}
    product_name = "Google DPF product-valued" if "Google DPF product-valued" in work else "Product-valued kernel"
    dual_name = "Google DPF dual-track" if "Google DPF dual-track" in work else "Dual-track kernel"
    if work[product_name] > work[dual_name]:
        raise AssertionError("product-valued normalized work exceeds dual-track work")
    for row in rows:
        if row["source_type"] != "microbenchmark":
            raise AssertionError("microbenchmark CSV row has wrong source_type")
        if float(row["time_ms_per_report"]) <= 0:
            raise AssertionError("microbenchmark time must be positive")
    if rows[0].get("benchmark") == "google_distributed_point_functions":
        checksums = {row.get("checksum") for row in rows}
        if len(checksums) != 1:
            raise AssertionError("Google DPF product and dual-track reconstructions disagree")
        scaling_rows = read_csv(DATA_DIR / "fig_dpf_scaling.csv")
        scaling_by_n: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in scaling_rows:
            if row["source_type"] != "microbenchmark":
                raise AssertionError("DPF scaling row has wrong source_type")
            if float(row["time_ms_per_report"]) <= 0:
                raise AssertionError("DPF scaling time must be positive")
            scaling_by_n[row["N"]].append(row)
        for n, n_rows in scaling_by_n.items():
            if len(n_rows) != 2:
                raise AssertionError(f"DPF scaling expected two rows at N={n}")
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
        latency_rows = read_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv")
        for row in latency_rows:
            if row["source_type"] != "analytic_cost":
                raise AssertionError("ingestion latency row has wrong source_type")
            if row["method"] == "Chafe online DPF service" and row["stable"] == "false":
                if float(row["backlog_growth_reports_per_s"]) <= 0:
                    raise AssertionError("unstable Chafe rows must report positive backlog growth")
            if row["method"] == "Chafe online DPF service" and row["stable"] == "true":
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
        notes.append("Google DPF backend timing and reconstruction checks passed.")
        return
    notes.append("Microbenchmark normalized-work and timing checks passed.")


def validate_source_types(notes: list[str]) -> None:
    expected = {
        "fig_privacy_vs_rho.csv": {"formula"},
        "fig_privacy_vs_eps0.csv": {"formula"},
        "fig_privacy_vs_tau.csv": {"formula"},
        "fig_privacy_surface_rho_tau.csv": {"formula"},
        "fig_exact_vs_proxy_drop.csv": {"formula", "exact_accountant"},
        "fig_collision_regimes.csv": {"formula"},
        "fig_utility_factors.csv": {"formula"},
        "fig_privacy_utility_frontier.csv": {"formula"},
        "fig_batch_gating_delay.csv": {"analytic_cost"},
        "fig_product_vs_dual_microbench.csv": {"microbenchmark"},
        "fig_dpf_scaling.csv": {"microbenchmark"},
        "fig_dpf_phase_breakdown.csv": {"microbenchmark"},
        "fig_dpf_throughput_vs_n.csv": {"microbenchmark"},
        "fig_ingestion_latency_vs_rate.csv": {"analytic_cost"},
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
    }
    for filename, allowed in expected.items():
        rows = read_csv(DATA_DIR / filename)
        seen = {row["source_type"] for row in rows}
        if not seen <= allowed:
            raise AssertionError(f"{filename} has unexpected source types {seen}; expected subset of {allowed}")
    notes.append("Source-type checks passed for all figure CSVs.")


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
    ]
    for filename in filenames:
        for row in read_csv(DATA_DIR / filename):
            if int(float(row["seeds"])) < 20:
                raise AssertionError(f"{filename} has fewer than 20 Monte Carlo seeds")
    notes.append("Monte Carlo replicate-count checks passed.")


def validate_all() -> list[str]:
    notes: list[str] = []
    validate_privacy_monotonicity(notes)
    validate_collision_sum(notes)
    validate_exact_proxy(notes)
    validate_microbench(notes)
    validate_source_types(notes)
    validate_monte_carlo_replicates(notes)
    (DATA_DIR / "validation_summary.txt").write_text("\n".join(notes) + "\n")
    return notes


if __name__ == "__main__":
    for note in validate_all():
        print(note)
