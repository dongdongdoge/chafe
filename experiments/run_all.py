from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chafe_eval_lib import DATA_DIR, FIGURE_DIR, ensure_dirs, read_csv

import exact_drop_accountant
import fetch_adult_dataset
import fixed_eps0_utility
import generate_formula_data
import make_figures
import microbench
import real_data_utility
import utility_simulations
import validate_results
import compose_workload_comparisons


def closest(rows: list[dict[str, str]], key: str, value: float, method: str | None = None) -> dict[str, str]:
    candidates = rows
    if method is not None:
        candidates = [row for row in rows if row["method"] == method]
    if not candidates:
        raise ValueError(f"no rows for method {method}")
    return min(candidates, key=lambda row: abs(float(row[key]) - value))


def summarize_outputs() -> str:
    privacy_rho = read_csv(DATA_DIR / "fig_privacy_vs_rho.csv")
    exact_rows = [row for row in read_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv") if row["source_type"] == "exact_accountant"]
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

    ideal_rho1 = closest(privacy_rho, "rho", 1.0, "IdealShuffle")
    drop_rho1 = closest(privacy_rho, "rho", 1.0, "Drop (matched padding)")
    pool_rho1 = closest(privacy_rho, "rho", 1.0, "Pool (replacement)")
    pool_add_rho1 = closest(privacy_rho, "rho", 1.0, "Pool (add/remove)")
    drop_rho2 = closest(privacy_rho, "rho", 2.0, "Drop (matched padding)")

    exact_ratios = [float(row["exact_to_proxy_ratio"]) for row in exact_rows]
    product = next(
        row for row in micro_rows if row["method"] in ("Google DPF product-valued", "Product-valued kernel")
    )
    dual = next(row for row in micro_rows if row["method"] in ("Google DPF dual-track", "Dual-track kernel"))
    delay_1m = closest(delay_rows, "m", 1_000_000, "Collect-then-shuffle")
    range_drop_016 = closest(range_rows, "epsilon_prime", 0.16, "Drop")
    range_ideal_016 = closest(range_rows, "epsilon_prime", 0.16, "IdealShuffle")
    sum_pool_016 = closest(sum_rows, "epsilon_prime", 0.16, "Pool")
    sum_drop_016 = closest(sum_rows, "epsilon_prime", 0.16, "Drop")
    f1_drop_032 = closest(f1_rows, "epsilon_prime", 0.32, "Drop")
    f1_ideal_032 = closest(f1_rows, "epsilon_prime", 0.32, "IdealShuffle")
    product_small = closest(scaling_rows, "N", 256, "Google DPF product-valued")
    product_large = closest(scaling_rows, "N", 65_536, "Google DPF product-valued")
    throughput_4096 = closest(throughput_rows, "N", 4_096, "Google DPF product-valued")
    chafe_latency_40 = closest(latency_rows, "arrival_rate_reports_per_s", 40.0, "Chafe online DPF service")
    batch_latency_40 = closest(latency_rows, "arrival_rate_reports_per_s", 40.0, "Collect-then-shuffle gate")
    phase_eval_product = next(
        row for row in phase_rows if row["method"] == "Google DPF product-valued" and row["phase"] == "eval"
    )
    phase_total_product = next(
        row for row in phase_rows if row["method"] == "Google DPF product-valued" and row["phase"] == "total"
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

    return "\n".join(
        [
            "Main numerical takeaways:",
            (
                f"- At rho=1, eps0=1, m=1000000, tau=2, delta=1e-6: "
                f"IdealShuffle epsilon'={float(ideal_rho1['epsilon']):.4g}, "
                f"Drop proxy={float(drop_rho1['epsilon']):.4g}, "
                f"Pool replacement={float(pool_rho1['epsilon']):.4g}, "
                f"Pool add/remove={float(pool_add_rho1['epsilon']):.4g}."
            ),
            f"- The Drop formula proxy falls to epsilon'={float(drop_rho2['epsilon']):.4g} at rho=2 on the same grid.",
            (
                f"- Exact small-scale Drop hockey-stick validation has exact/proxy ratios "
                f"{min(exact_ratios):.3g}..{max(exact_ratios):.3g} over the rho grid."
            ),
            (
                f"- {product['benchmark']} product-valued backend measured "
                f"{float(product['time_ms_per_report']):.3g} ms/report; "
                f"dual-track measured {float(dual['time_ms_per_report']):.3g} ms/report "
                f"({float(dual['speedup_vs_product']):.2f}x dual/product time ratio)."
            ),
            (
                f"- Google DPF product-valued scaling: {float(product_small['time_ms_per_report']):.3g} "
                f"ms/report at N=256 and {float(product_large['time_ms_per_report']):.3g} "
                f"ms/report at N={int(float(product_large['N']))}."
            ),
            (
                f"- At N=4096, product-valued DPF evaluation accounts for "
                f"{float(phase_eval_product['time_ms_per_report']) / float(phase_total_product['time_ms_per_report']):.1%} "
                f"of measured per-report backend time."
            ),
            (
                f"- The measured product-valued DPF service rate at N=4096 is "
                f"{float(throughput_4096['reports_per_second']):.3g} reports/s. At arrival rate 40 reports/s, "
                f"the analytic online service latency is {float(chafe_latency_40['avg_latency_seconds']):.3g} s, "
                f"while collect-then-shuffle batch gating for m=1e6 is "
                f"{float(batch_latency_40['avg_latency_seconds']):.3g} s."
            ),
            (
                f"- Analytic collect-then-shuffle average wait at m=1000000 and lambda=20000/s is "
                f"{float(delay_1m['delay_seconds']):.3g} s; Chafe online batch-gating wait is 0."
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


def run(skip_utility: bool = False) -> None:
    ensure_dirs()
    outputs: dict[str, int] = {}
    for module in [generate_formula_data, exact_drop_accountant, microbench]:
        outputs.update(module.generate_all())
    if not skip_utility:
        outputs["datasets/adult/adult_clean.csv"] = adult_row_count()
        outputs.update(utility_simulations.generate_all())
        outputs.update(fixed_eps0_utility.generate_all())
        outputs.update(real_data_utility.generate_all())
        outputs.update(compose_workload_comparisons.generate_all())
    else:
        print("Skipping utility simulations by request; workload figures will not be regenerated.")
    outputs.update(make_figures.generate_all())
    notes = validate_results.validate_all()

    print("Generated CSV row counts:")
    for name, rows in sorted(outputs.items()):
        print(f"- {name}: {rows}")
    print_file_list("Generated figures:", FIGURE_DIR, ".tex")
    print_file_list("Generated CSVs:", DATA_DIR, ".csv")
    print("Validation:")
    for note in notes:
        print(f"- {note}")
    print(summarize_outputs())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-utility", action="store_true", help="regenerate formula/exact/microbench figures only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(skip_utility=args.skip_utility)
