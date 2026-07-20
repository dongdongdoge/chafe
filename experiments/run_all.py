from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chafe_eval_lib import DATA_DIR, ensure_dirs, read_csv

import accountant_scaling_benchmark
import exact_drop_accountant
import exact_certified_utility
import exact_ideal_shuffle_baseline
import fetch_adult_dataset
import fixed_eps0_utility
import full_release_dummy_utility
import full_release_real_utility
import generate_formula_data
import generic_exact_accountant_validation
import large_exact_real_utility
import microbench
import packed64_safety
import pool_robust_mean_utility
import privacy_2x2_validation
import privacy_large_m_accounting
import padding_correction_utility
import real_data_utility
import system_stage_benchmark
import utility_simulations
import validate_results
import compose_workload_comparisons
import covertype_utility
import cgp_shuffle_model
import epoch_close_scaling
import full_release_proxy_utility
import external_gpu_dpf
import finite_burst_latency
import modern_data_utility
import modern_fixed_eps0_utility


def closest(rows: list[dict[str, str]], key: str, value: float, method: str | None = None) -> dict[str, str]:
    candidates = rows
    if method is not None:
        candidates = [row for row in rows if row["method"] == method]
    if not candidates:
        raise ValueError(f"no rows for method {method}")
    return min(candidates, key=lambda row: abs(float(row[key]) - value))


def summarize_outputs() -> str:
    privacy_rho = read_csv(DATA_DIR / "fig_privacy_vs_rho.csv")
    exact_rows = [
        row
        for row in read_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv")
        if row["source_type"] == "exact_law_numerical"
    ]
    generic_exact_rows = read_csv(DATA_DIR / "fig_generic_exact_accountant_validation.csv")
    micro_rows = read_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv")
    delay_rows = read_csv(DATA_DIR / "fig_batch_gating_delay.csv")
    range_rows = read_csv(DATA_DIR / "fig_range_error_vs_epsilon.csv")
    sum_rows = read_csv(DATA_DIR / "fig_sum_error_vs_epsilon.csv")
    f1_rows = read_csv(DATA_DIR / "fig_heavyhitter_f1_vs_epsilon.csv")
    scaling_rows = read_csv(DATA_DIR / "fig_dpf_scaling.csv")
    throughput_rows = read_csv(DATA_DIR / "fig_dpf_throughput_vs_n.csv")
    latency_rows = read_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv")
    phase_rows = read_csv(DATA_DIR / "fig_dpf_phase_breakdown.csv")
    fixed_range_rows = read_csv(DATA_DIR / "fig_fixed_eps0_range_error.csv")
    fixed_sum_rows = read_csv(DATA_DIR / "fig_fixed_eps0_sum_error.csv")
    real_age_rows = read_csv(DATA_DIR / "fig_real_age_range_error_vs_epsilon.csv")
    real_sum_rows = read_csv(DATA_DIR / "fig_real_hours_sum_error_vs_epsilon.csv")
    real_f1_rows = read_csv(DATA_DIR / "fig_real_heavyhitter_f1_vs_epsilon.csv")
    covertype_rows = read_csv(DATA_DIR / "fig_covertype_workloads.csv")
    certified_utility_rows = read_csv(DATA_DIR / "fig_exact_certified_utility.csv")

    ideal_rho1 = closest(privacy_rho, "rho", 1.0, "IdealShuffle")
    ideal_padded_rho1 = closest(
        privacy_rho, "rho", 1.0, "IdealShuffle (same M padding)"
    )
    drop_rho1 = closest(privacy_rho, "rho", 1.0, "Drop (matched padding)")
    pool_rho1 = closest(privacy_rho, "rho", 1.0, "Pool (replacement)")
    pool_add_rho1 = closest(privacy_rho, "rho", 1.0, "Pool (add/remove)")
    drop_rho2 = closest(privacy_rho, "rho", 2.0, "Drop (matched padding)")

    exact_ratios = [float(row["exact_to_proxy_ratio"]) for row in exact_rows]

    def family_exact(case: str, decoder: str, dummy_mode: str) -> dict[str, str]:
        return next(
            row
            for row in generic_exact_rows
            if row["case"] == case
            and row["decoder"] == decoder
            and row["dummy_law_mode"] == dummy_mode
        )

    repl_drop_exact = family_exact(
        "fixed_n_replacement", "drop", "uniform_public_reference"
    )
    repl_pool_exact = family_exact(
        "fixed_n_replacement", "pool", "uniform_public_reference"
    )
    ar_drop_exact = family_exact(
        "small_mechanism_add_remove", "drop", "uniform_public_reference"
    )
    ar_pool_exact = family_exact(
        "small_mechanism_add_remove", "pool", "uniform_public_reference"
    )
    product = next(
        row for row in micro_rows if row["method"] in ("Google DPF packed64", "Product-valued kernel")
    )
    tuple_product = next((row for row in micro_rows if row["method"] == "Google DPF tuple product"), None)
    dual = next(row for row in micro_rows if row["method"] in ("Google DPF dual-track", "Dual-track kernel"))
    range_drop_016 = closest(range_rows, "epsilon_prime", 0.16, "Drop")
    range_ideal_016 = closest(range_rows, "epsilon_prime", 0.16, "IdealShuffle")
    sum_pool_016 = closest(sum_rows, "epsilon_prime", 0.16, "Pool")
    sum_drop_016 = closest(sum_rows, "epsilon_prime", 0.16, "Drop")
    f1_drop_032 = closest(f1_rows, "epsilon_prime", 0.32, "Drop")
    f1_ideal_032 = closest(f1_rows, "epsilon_prime", 0.32, "IdealShuffle")
    product_scaling_rows = [row for row in scaling_rows if row["method"] == "Google DPF packed64"]
    product_small = min(product_scaling_rows, key=lambda row: float(row["N"]))
    product_large = max(product_scaling_rows, key=lambda row: float(row["N"]))
    online_latency_rows = [row for row in latency_rows if row["method"] == "Chafe online DPF service"]
    service_n = int(float(online_latency_rows[0]["N"])) if online_latency_rows else 2**17
    service_threads = int(float(online_latency_rows[0].get("threads", "1") or "1"))
    service_rate = float(online_latency_rows[0]["service_rate_reports_per_s"])
    chafe_latency_40 = closest(latency_rows, "arrival_rate_reports_per_s", 40.0, "Chafe online DPF service")
    batch_latency_40 = closest(latency_rows, "arrival_rate_reports_per_s", 40.0, "Collect-then-shuffle gate")
    batch_latency_20k = closest(latency_rows, "arrival_rate_reports_per_s", 20_000.0, "Collect-then-shuffle gate")
    clarion_rows = [row for row in latency_rows if row["method"] == "Collect gate + Clarion shuffle"]
    clarion_latency_40 = closest(latency_rows, "arrival_rate_reports_per_s", 40.0, "Collect gate + Clarion shuffle") if clarion_rows else None
    clarion_latency_20k = closest(latency_rows, "arrival_rate_reports_per_s", 20_000.0, "Collect gate + Clarion shuffle") if clarion_rows else None
    latency_m = int(float(batch_latency_40["m"]))
    phase_eval_product = next(
        row for row in phase_rows if row["method"] == "Google DPF packed64" and row["phase"] == "eval"
    )
    phase_total_product = next(
        row for row in phase_rows if row["method"] == "Google DPF packed64" and row["phase"] == "total"
    )
    fixed_range_all_1 = closest(fixed_range_rows, "eps0", 1.0, "All reports")
    fixed_range_drop_1 = closest(fixed_range_rows, "eps0", 1.0, "Drop")
    fixed_sum_all_1 = closest(fixed_sum_rows, "eps0", 1.0, "All reports")
    fixed_sum_drop_1 = closest(fixed_sum_rows, "eps0", 1.0, "Drop")
    fixed_sum_pool_1 = closest(fixed_sum_rows, "eps0", 1.0, "Pool")
    real_age_ideal_016 = closest(real_age_rows, "epsilon_prime", 0.16, "IdealShuffle")
    real_age_drop_016 = closest(real_age_rows, "epsilon_prime", 0.16, "Drop")
    real_sum_pool_016 = closest(real_sum_rows, "epsilon_prime", 0.16, "Pool")
    real_sum_drop_016 = closest(real_sum_rows, "epsilon_prime", 0.16, "Drop")
    real_f1_ideal_032 = closest(real_f1_rows, "epsilon_prime", 0.32, "IdealShuffle")
    real_f1_drop_032 = closest(real_f1_rows, "epsilon_prime", 0.32, "Drop")
    covertype_range_drop_016 = next(
        row
        for row in covertype_rows
        if row["task"] == "elevation_range" and row["method"] == "Drop" and abs(float(row["epsilon_prime"]) - 0.16) < 1e-12
    )
    covertype_hh_drop_032 = next(
        row
        for row in covertype_rows
        if row["task"] == "soil_cover_hh" and row["method"] == "Drop" and abs(float(row["epsilon_prime"]) - 0.32) < 1e-12
    )
    covertype_sum_pool_016 = next(
        row
        for row in covertype_rows
        if row["task"] == "road_distance_sum" and row["method"] == "Pool" and abs(float(row["epsilon_prime"]) - 0.16) < 1e-12
    )

    return "\n".join(
        [
            "Main numerical takeaways:",
            (
                f"- At rho=1, eps0=1, m=1000000, tau=2, delta=1e-6: "
                f"replacement proxies are standard IdealShuffle epsilon'={float(ideal_rho1['epsilon']):.4g}, "
                f"same-M padded IdealShuffle={float(ideal_padded_rho1['epsilon']):.4g}, "
                f"Drop={float(drop_rho1['epsilon']):.4g}, and directed homogeneous "
                f"Pool score={float(pool_rho1['epsilon']):.4g}; "
                f"the separately scoped Pool add/remove proxy is {float(pool_add_rho1['epsilon']):.4g}."
            ),
            f"- The Drop formula proxy falls to epsilon'={float(drop_rho2['epsilon']):.4g} at rho=2 on the same grid.",
            (
                f"- Exact finite-domain Drop hockey-stick validation has exact/proxy ratios "
                f"{min(exact_ratios):.3g}..{max(exact_ratios):.3g} over the rho grid."
            ),
            (
                "- Exact rational finite-family budgets at N=2, M=3, delta=1e-3 and uniform public nu: "
                f"replacement Drop epsilon={float(repl_drop_exact['family_epsilon_exact_reported']):.4g} "
                f"(alpha={repl_drop_exact['family_alpha_exact_fraction']}), replacement Pool "
                f"epsilon={float(repl_pool_exact['family_epsilon_exact_reported']):.4g} "
                f"(alpha={repl_pool_exact['family_alpha_exact_fraction']}), cap-wide add/remove Drop "
                f"epsilon={float(ar_drop_exact['family_epsilon_exact_reported']):.4g}, and Pool "
                f"epsilon={float(ar_pool_exact['family_epsilon_exact_reported']):.4g}."
            ),
            (
                f"- {product['benchmark']} packed64 backend measured "
                f"{float(product['time_ms_per_report']):.3g} ms/report; "
                + (
                    f"tuple-product measured {float(tuple_product['time_ms_per_report']):.3g} ms/report; "
                    if tuple_product is not None
                    else ""
                )
                + f"dual-track measured {float(dual['time_ms_per_report']):.3g} ms/report."
            ),
            (
                f"- Google DPF packed64 scaling: {float(product_small['time_ms_per_report']):.3g} "
                f"ms/report at N={int(float(product_small['N']))} and {float(product_large['time_ms_per_report']):.3g} "
                f"ms/report at N={int(float(product_large['N']))}."
            ),
            (
                f"- At N={int(float(phase_total_product['N']))}, packed64 DPF evaluation accounts for "
                f"{float(phase_eval_product['time_ms_per_report']) / float(phase_total_product['time_ms_per_report']):.1%} "
                f"of measured per-report backend time."
            ),
            (
                f"- The measured packed64 DPF service rate at N={service_n} with {service_threads} worker(s) is "
                f"{service_rate:.3g} reports/s. At arrival rate 40 reports/s, "
                f"the analytic online service latency is {float(chafe_latency_40['avg_latency_seconds']):.3g} s, "
                f"while collect-then-shuffle batch gating for m={latency_m} is "
                f"{float(batch_latency_40['avg_latency_seconds']):.3g} s"
                + (
                    f" and gate+Clarion is {float(clarion_latency_40['avg_latency_seconds']):.3g} s."
                    if clarion_latency_40 is not None
                    else "."
                )
            ),
            (
                f"- Analytic collect-then-shuffle average wait at m={latency_m} and lambda=20000/s is "
                f"{float(batch_latency_20k['avg_latency_seconds']):.3g} s"
                + (
                    f" with gate+Clarion at {float(clarion_latency_20k['avg_latency_seconds']):.3g} s;"
                    if clarion_latency_20k is not None
                    else ";"
                )
                + " Chafe online batch-gating wait is 0."
            ),
            (
                f"- Monte Carlo range error at epsilon'=0.16: IdealShuffle "
                f"{float(range_ideal_016['mean_relative_error']):.3g}, Drop "
                f"{float(range_drop_016['mean_relative_error']):.3g}."
            ),
            (
                f"- Monte Carlo sum relative error at epsilon'=0.16: Pool "
                f"{float(sum_pool_016['mean_relative_error']):.3g}, Drop "
                f"{float(sum_drop_016['mean_relative_error']):.3g}."
            ),
            (
                f"- Monte Carlo heavy-hitter F1 at epsilon'=0.32: IdealShuffle "
                f"{float(f1_ideal_032['f1']):.3g}, Drop {float(f1_drop_032['f1']):.3g}."
            ),
            (
                f"- Fixed-eps0 synthetic range error at eps0=1: all reports "
                f"{float(fixed_range_all_1['mean_relative_error']):.3g}, Drop "
                f"{float(fixed_range_drop_1['mean_relative_error']):.3g}; fixed-eps0 sum error: "
                f"all reports {float(fixed_sum_all_1['mean_relative_error']):.3g}, Drop "
                f"{float(fixed_sum_drop_1['mean_relative_error']):.3g}, Pool "
                f"{float(fixed_sum_pool_1['mean_relative_error']):.3g}."
            ),
            (
                f"- UCI Adult age-range error at epsilon'=0.16: IdealShuffle "
                f"{float(real_age_ideal_016['mean_relative_error']):.3g}, Drop "
                f"{float(real_age_drop_016['mean_relative_error']):.3g}; hours sum error: Pool "
                f"{float(real_sum_pool_016['mean_relative_error']):.3g}, Drop "
                f"{float(real_sum_drop_016['mean_relative_error']):.3g}."
            ),
            (
                f"- UCI Adult occupation-education top-50 F1 at epsilon'=0.32: IdealShuffle "
                f"{float(real_f1_ideal_032['f1']):.3g}, Drop {float(real_f1_drop_032['f1']):.3g}."
            ),
            (
                f"- UCI Covertype m=300000 workloads: at epsilon'=0.16, Drop elevation-range error is "
                f"{float(covertype_range_drop_016['mean_relative_error']):.3g}; at epsilon'=0.32, "
                f"Drop soil/cover top-50 F1 is {float(covertype_hh_drop_032['f1']):.3g}; at epsilon'=0.16, "
                f"Pool road-distance sum error is {float(covertype_sum_pool_016['mean_relative_error']):.3g}."
            ),
            (
                "- Exact-certified complete-release binary count at target "
                "(epsilon'=ln 2, delta=1e-3): "
                + "; ".join(
                    f"{row['method']} uses p={row['truthful_probability_fraction']} "
                    f"(exact epsilon'={float(row['exact_epsilon_prime']):.3g}) and has "
                    f"exact MAE={float(row['exact_mean_absolute_count_error']):.3g}"
                    for row in certified_utility_rows
                )
                + "."
            ),
        ]
    )


def print_file_list(title: str, folder: Path, suffix: str) -> None:
    files = sorted(path.name for path in folder.glob(f"*{suffix}"))
    print(title)
    for name in files:
        print(f"- {folder.name}/{name}")


def adult_row_count() -> int:
    path = fetch_adult_dataset.ensure_adult_dataset()
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def run(skip_utility: bool = False, skip_microbench: bool = False) -> None:
    ensure_dirs()
    outputs: dict[str, int] = {}
    modules = [
        generate_formula_data,
        exact_drop_accountant,
        accountant_scaling_benchmark,
        exact_ideal_shuffle_baseline,
        generic_exact_accountant_validation,
        privacy_2x2_validation,
        privacy_large_m_accounting,
        packed64_safety,
        system_stage_benchmark,
        cgp_shuffle_model,
    ]
    if not skip_microbench:
        modules.append(microbench)
    else:
        print(
            "Skipping Google DPF microbenchmark regeneration; existing DPF CSVs "
            "are retained and the fast epoch-stage benchmarks are still rerun."
        )
    for module in modules:
        outputs.update(module.generate_all())
    if not skip_microbench:
        close_rows = epoch_close_scaling.generate_all()
        outputs["data/fig_epoch_close_scaling.csv"] = len(close_rows)
    if not skip_utility:
        outputs["datasets/adult/adult_clean.csv"] = adult_row_count()
        outputs.update(utility_simulations.generate_all())
        outputs.update(fixed_eps0_utility.generate_all())
        outputs.update(full_release_dummy_utility.generate_all())
        outputs.update(padding_correction_utility.generate_all())
        outputs.update(pool_robust_mean_utility.generate_all())
        outputs.update(full_release_real_utility.generate_all())
        outputs.update(real_data_utility.generate_all())
        outputs.update(covertype_utility.generate_all())
        outputs.update(compose_workload_comparisons.generate_all())
        outputs.update(exact_certified_utility.generate_all())
        outputs.update(large_exact_real_utility.generate_all())
        outputs.update(full_release_proxy_utility.generate_all())
        outputs.update(modern_data_utility.generate_all())
        outputs.update(modern_fixed_eps0_utility.generate_all())
    else:
        print("Skipping utility simulations by request; existing workload CSVs are retained.")
    gpu_rows = external_gpu_dpf.generate()
    outputs["fig_gpu_dpf_external.csv"] = len(gpu_rows)
    outputs.update(finite_burst_latency.generate_all())
    notes = validate_results.validate_all()

    print("Generated CSV row counts:")
    for name, rows in sorted(outputs.items()):
        print(f"- {name}: {rows}")
    print_file_list("Generated CSVs:", DATA_DIR, ".csv")
    print("Validation:")
    for note in notes:
        print(f"- {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-utility", action="store_true", help="retain existing workload CSVs")
    parser.add_argument(
        "--skip-microbench",
        action="store_true",
        help="retain existing measured backend CSVs instead of requiring the benchmark binary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        skip_utility=args.skip_utility,
        skip_microbench=args.skip_microbench,
    )
