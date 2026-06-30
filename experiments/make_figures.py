from __future__ import annotations

from pathlib import Path

from chafe_eval_lib import DATA_DIR, FIGURE_DIR, ensure_dirs, fnum, read_csv


LINE_STYLES = [
    "black, solid, mark=*, mark size=1.2pt",
    "black!70, dashed, mark=square*, mark size=1.2pt",
    "black!55, dotted, mark=triangle*, mark size=1.4pt",
    "black!85, dashdotted, mark=diamond*, mark size=1.3pt",
]


def axis_options(width: str, xlabel: str, ylabel: str, extra: str = "") -> str:
    opts = [
        f"width={width}",
        "height=3.6cm",
        "grid=both",
        "grid style={gray!18}",
        "major grid style={gray!25}",
        "tick label style={font=\\footnotesize}",
        "label style={font=\\small}",
        "legend style={font=\\scriptsize, fill=white, draw=black!20, at={(0.98,0.98)}, anchor=north east}",
        "axis line style={black!70}",
        "tick style={black!70}",
        f"xlabel={{{xlabel}}}",
        f"ylabel={{{ylabel}}}",
    ]
    if extra:
        opts.append(extra)
    return ",\n".join(opts)


def coordinates(rows: list[dict[str, str]], x_col: str, y_col: str) -> str:
    pairs = sorted((float(row[x_col]), float(row[y_col])) for row in rows)
    return " ".join(f"({fnum(x)},{fnum(y)})" for x, y in pairs)


def method_rows(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    return [row for row in rows if row["method"] == method]


def line_figure(
    *,
    csv_name: str,
    tex_name: str,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    methods: list[str],
    caption: str,
    label: str,
    width: str = "0.48\\textwidth",
    extra_axis: str = "",
) -> None:
    rows = read_csv(DATA_DIR / csv_name)
    plots = []
    for idx, method in enumerate(methods):
        subset = method_rows(rows, method)
        if not subset:
            raise ValueError(f"{csv_name}: no rows for method {method}")
        plots.append(
            "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {"
            + coordinates(subset, x_col, y_col)
            + "};\n"
            + f"\\addlegendentry{{{method}}}"
        )
    body = f"""% Auto-generated from data/{csv_name}.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{axis_options(width, xlabel, ylabel, extra_axis)}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    (FIGURE_DIR / tex_name).write_text(body)


def bar_figure_microbench() -> None:
    rows = read_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv")
    vals = {row["method"]: float(row["time_ms_per_report"]) for row in rows}
    product_name = "Google DPF product-valued" if "Google DPF product-valued" in vals else "Product-valued kernel"
    dual_name = "Google DPF dual-track" if "Google DPF dual-track" in vals else "Dual-track kernel"
    benchmark = rows[0].get("benchmark", "")
    if benchmark == "google_distributed_point_functions":
        caption = "Measured Google distributed-point-functions backend for product-valued versus dual-track hidden-slot ingestion. The benchmark generates DPF keys, evaluates both server keys over the full slot domain, accumulates shares, and verifies reconstruction."
    else:
        caption = "Measured NumPy dense DPF-expansion kernel proxy for product-valued versus dual-track ingestion. The benchmark measures wall-clock traversal and pseudorandom expansion cost per report; it is not a full DPF implementation throughput claim."
    opts = axis_options(
        "0.48\\textwidth",
        "",
        "ms per report",
        "ybar, bar width=13pt, symbolic x coords={Product,Dual}, xtick=data, ymin=0, enlarge x limits=0.35",
    )
    body = f"""% Auto-generated from data/fig_product_vs_dual_microbench.csv.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{opts}
]
\\addplot+[black!70, fill=black!20] coordinates {{
(Product,{fnum(vals[product_name])}) (Dual,{fnum(vals[dual_name])})
}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{caption}}}
\\label{{fig:product-vs-dual-microbench}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_product_vs_dual_microbench.tex").write_text(body)


def bar_figure_dpf_phase_breakdown() -> None:
    rows = read_csv(DATA_DIR / "fig_dpf_phase_breakdown.csv")
    phase_names = {
        "keygen": "Keygen",
        "context": "Ctx",
        "eval": "Eval",
        "accumulate": "Acc",
    }
    phases = ["keygen", "context", "eval", "accumulate"]
    product = {row["phase"]: float(row["time_ms_per_report"]) for row in rows if row["method"] == "Google DPF product-valued"}
    dual = {row["phase"]: float(row["time_ms_per_report"]) for row in rows if row["method"] == "Google DPF dual-track"}
    product_coords = " ".join(f"({phase_names[phase]},{fnum(product[phase])})" for phase in phases)
    dual_coords = " ".join(f"({phase_names[phase]},{fnum(dual[phase])})" for phase in phases)
    opts = axis_options(
        "0.48\\textwidth",
        "",
        "ms per report",
        (
            "ybar, bar width=6pt, symbolic x coords={Keygen,Ctx,Eval,Acc}, "
            "xtick=data, ymin=0, enlarge x limits=0.22, "
            "legend style={font=\\scriptsize, fill=white, draw=black!20, at={(0.98,0.98)}, anchor=north east}"
        ),
    )
    body = f"""% Auto-generated from data/fig_dpf_phase_breakdown.csv.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{opts}
]
\\addplot+[black!70, fill=black!20] coordinates {{{product_coords}}};
\\addlegendentry{{Product}}
\\addplot+[black!35, fill=black!5] coordinates {{{dual_coords}}};
\\addlegendentry{{Dual}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Measured phase breakdown for the Google DPF write backend at $N=2^{{12}}$. Each bar is wall-clock time per report for key generation, evaluation-context construction, full-domain DPF evaluation, and share accumulation.}}
\\label{{fig:dpf-phase-breakdown}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_dpf_phase_breakdown.tex").write_text(body)


def generate_all() -> dict[str, int]:
    ensure_dirs()
    figure_specs = [
        {
            "csv_name": "fig_privacy_vs_rho.csv",
            "tex_name": "fig_privacy_vs_rho.tex",
            "x_col": "rho",
            "y_col": "epsilon",
            "xlabel": "$\\rho=m/N$",
            "ylabel": "$\\epsilon'$",
            "methods": ["IdealShuffle", "Drop (matched padding)", "Pool (replacement)", "Pool (add/remove)"],
            "caption": "Formula-derived privacy amplification versus load factor for $m=10^5$, $\\tau=2$, $\\epsilon_0=1$, and $\\delta=10^{-6}$. Pool replacement and add/remove curves use their respective second-order bounds.",
            "label": "fig:privacy-vs-rho",
        },
        {
            "csv_name": "fig_privacy_vs_eps0.csv",
            "tex_name": "fig_privacy_vs_eps0.tex",
            "x_col": "eps0",
            "y_col": "epsilon",
            "xlabel": "$\\epsilon_0$",
            "ylabel": "$\\epsilon'$",
            "methods": ["IdealShuffle", "Drop (matched padding)", "Pool (replacement)"],
            "caption": "Formula-derived privacy amplification as the local privacy parameter varies for $m=10^5$, $\\rho=1$, $\\tau=2$, and $\\delta=10^{-6}$.",
            "label": "fig:privacy-vs-eps0",
        },
        {
            "csv_name": "fig_privacy_vs_tau.csv",
            "tex_name": "fig_privacy_vs_tau.tex",
            "x_col": "tau",
            "y_col": "epsilon",
            "xlabel": "$\\tau=M/m$",
            "ylabel": "$\\epsilon'$",
            "methods": ["IdealShuffle", "Drop (matched padding)", "Pool (replacement)"],
            "caption": "Formula-derived privacy amplification versus matched release expansion $\\tau=M/m$ for $m=10^5$, $\\rho=1$, $\\epsilon_0=1$, and $\\delta=10^{-6}$.",
            "label": "fig:privacy-vs-tau",
        },
        {
            "csv_name": "fig_exact_vs_proxy_drop.csv",
            "tex_name": "fig_exact_vs_proxy_drop.tex",
            "x_col": "rho",
            "y_col": "epsilon",
            "xlabel": "$\\rho=m/N$",
            "ylabel": "$\\epsilon'$",
            "methods": ["Drop second-order proxy", "Drop exact hockey-stick"],
            "caption": "Drop second-order proxy versus exact small-scale finite-domain hockey-stick accounting for BRR with matched Bernoulli padding ($m=200$, $\\tau=2$, $\\epsilon_0=1$, $\\delta=10^{-6}$). The exact curve is a validation experiment, not the large-scale accountant.",
            "label": "fig:exact-vs-proxy-drop",
        },
        {
            "csv_name": "fig_collision_regimes.csv",
            "tex_name": "fig_collision_regimes.tex",
            "x_col": "rho",
            "y_col": "fraction",
            "xlabel": "$\\rho=m/N$",
            "ylabel": "fraction",
            "methods": ["Singleton", "Two-collision", "Three-or-more collision"],
            "caption": "Poissonization formula for report-side collision regimes: singleton reports, reports in a pair collision, and reports in slots of size at least three.",
            "label": "fig:collision-regimes",
        },
        {
            "csv_name": "fig_utility_factors.csv",
            "tex_name": "fig_utility_factors.tex",
            "x_col": "rho",
            "y_col": "factor",
            "xlabel": "$\\rho=m/N$",
            "ylabel": "normalized factor",
            "methods": ["Drop retained mass", "Drop SE factor", "Pool additive mass"],
            "caption": "Formula-derived utility factors under Poissonization. Drop retains an $e^{-\\rho}$ mass and has unbiased-correction standard-error factor $e^{\\rho/2}$; Pool preserves additive mass.",
            "label": "fig:utility-factors",
        },
        {
            "csv_name": "fig_privacy_utility_frontier.csv",
            "tex_name": "fig_privacy_utility_frontier.tex",
            "x_col": "utility_cost",
            "y_col": "epsilon",
            "xlabel": "normalized utility cost",
            "ylabel": "$\\epsilon'$",
            "methods": ["Drop (matched padding)", "Pool (replacement)"],
            "caption": "Formula-derived privacy--utility frontier from scanning $\\rho$ for $m=10^5$, $\\tau=2$, $\\epsilon_0=1$, and $\\delta=10^{-6}$. Drop uses utility cost $e^{\\rho/2}$; Pool replacement keeps additive cost at one.",
            "label": "fig:privacy-utility-frontier",
            "width": "0.48\\textwidth",
            "extra_axis": "xmin=0.95",
        },
        {
            "csv_name": "fig_batch_gating_delay.csv",
            "tex_name": "fig_batch_gating_delay.tex",
            "x_col": "m",
            "y_col": "delay_seconds",
            "xlabel": "epoch size $m$",
            "ylabel": "average wait (s)",
            "methods": ["Collect-then-shuffle", "Chafe online anonymization"],
            "caption": "Analytic batch-gating delay at arrival rate $\\lambda=2\\times10^4$ reports/s. Collect-then-shuffle waits $(m-1)/(2\\lambda)$ on average; Chafe's online hidden-slot ingestion has no batch-gating wait.",
            "label": "fig:batch-gating-delay",
            "width": "0.48\\textwidth",
        },
        {
            "csv_name": "fig_range_error_vs_epsilon.csv",
            "tex_name": "fig_range_error_vs_epsilon.tex",
            "x_col": "epsilon_prime",
            "y_col": "mean_relative_error",
            "xlabel": "target $\\epsilon'$",
            "ylabel": "mean relative error",
            "methods": ["Local", "IdealShuffle", "Drop"],
            "caption": "Monte Carlo utility simulation for 1,000 random range-count queries over a Zipf histogram ($m=5000$, $k=256$, 30 seeds). Points plot means; standard errors are saved in the CSV.",
            "label": "fig:range-error-vs-epsilon",
            "extra_axis": "ymode=log",
        },
        {
            "csv_name": "fig_sum_error_vs_epsilon.csv",
            "tex_name": "fig_sum_error_vs_epsilon.tex",
            "x_col": "epsilon_prime",
            "y_col": "mean_relative_error",
            "xlabel": "target $\\epsilon'$",
            "ylabel": "relative sum error",
            "methods": ["Local", "IdealShuffle", "Drop", "Pool"],
            "caption": "Monte Carlo utility simulation for bounded sums with Laplace additive reports ($m=5000$, 30 seeds). Pool uses the replacement-style formula to choose $\\epsilon_0$ and preserves additive sums in the simulator.",
            "label": "fig:sum-error-vs-epsilon",
            "extra_axis": "ymode=log",
        },
        {
            "csv_name": "fig_heavyhitter_f1_vs_epsilon.csv",
            "tex_name": "fig_heavyhitter_f1_vs_epsilon.tex",
            "x_col": "epsilon_prime",
            "y_col": "f1",
            "xlabel": "target $\\epsilon'$",
            "ylabel": "top-50 F1",
            "methods": ["Local", "IdealShuffle", "Drop"],
            "caption": "Monte Carlo utility simulation for top-50 heavy-hitter recovery over a Zipf histogram ($m=5000$, $k=256$, 30 seeds). Points plot means; standard errors are saved in the CSV.",
            "label": "fig:heavyhitter-f1-vs-epsilon",
            "extra_axis": "ymin=0, ymax=1.02",
        },
        {
            "csv_name": "fig_dpf_scaling.csv",
            "tex_name": "fig_dpf_scaling.tex",
            "x_col": "N",
            "y_col": "time_ms_per_report",
            "xlabel": "slot domain $N$",
            "ylabel": "ms per report",
            "methods": ["Google DPF product-valued", "Google DPF dual-track"],
            "caption": "Measured Google DPF backend scaling with the hidden-slot domain size. Each point includes DPF key generation, evaluation of both server keys over all slots, accumulation, and reconstruction checks.",
            "label": "fig:dpf-scaling",
            "extra_axis": "xmode=log, ymode=log, log basis x=2",
        },
        {
            "csv_name": "fig_fixed_eps0_range_error.csv",
            "tex_name": "fig_fixed_eps0_range_error.tex",
            "x_col": "eps0",
            "y_col": "mean_relative_error",
            "xlabel": "fixed local $\\epsilon_0$",
            "ylabel": "mean relative error",
            "methods": ["All reports", "Drop"],
            "caption": "Monte Carlo utility simulation at fixed local $\\epsilon_0$ for range-count queries over the synthetic Zipf histogram. This separates Drop's retention-correction cost from the larger $\\epsilon_0$ permitted when fixing a target central $\\epsilon'$.",
            "label": "fig:fixed-eps0-range-error",
            "extra_axis": "ymode=log",
        },
        {
            "csv_name": "fig_fixed_eps0_sum_error.csv",
            "tex_name": "fig_fixed_eps0_sum_error.tex",
            "x_col": "eps0",
            "y_col": "mean_relative_error",
            "xlabel": "fixed local $\\epsilon_0$",
            "ylabel": "relative sum error",
            "methods": ["All reports", "Drop", "Pool"],
            "caption": "Monte Carlo utility simulation at fixed local $\\epsilon_0$ for bounded sums. Drop pays singleton-retention variance, while Pool and the all-report baseline preserve additive mass in this utility model.",
            "label": "fig:fixed-eps0-sum-error",
            "extra_axis": "ymode=log",
        },
        {
            "csv_name": "fig_real_age_range_error_vs_epsilon.csv",
            "tex_name": "fig_real_age_range_error_vs_epsilon.tex",
            "x_col": "epsilon_prime",
            "y_col": "mean_relative_error",
            "xlabel": "target $\\epsilon'$",
            "ylabel": "mean relative error",
            "methods": ["Local", "IdealShuffle", "Drop"],
            "caption": "Monte Carlo utility simulation on the UCI Adult dataset for age range-count queries ($m=5000$, 30 seeds). Points plot thresholded mean relative error; standard errors and dataset provenance are saved in the CSV.",
            "label": "fig:real-age-range-error-vs-epsilon",
            "extra_axis": "ymode=log",
        },
        {
            "csv_name": "fig_real_hours_sum_error_vs_epsilon.csv",
            "tex_name": "fig_real_hours_sum_error_vs_epsilon.tex",
            "x_col": "epsilon_prime",
            "y_col": "mean_relative_error",
            "xlabel": "target $\\epsilon'$",
            "ylabel": "relative sum error",
            "methods": ["Local", "IdealShuffle", "Drop", "Pool"],
            "caption": "Monte Carlo utility simulation on the UCI Adult dataset for the sum of bounded hours-per-week values ($m=5000$, 30 seeds). The additive Laplace report is a utility model, not a pure-LDP claim.",
            "label": "fig:real-hours-sum-error-vs-epsilon",
            "extra_axis": "ymode=log",
        },
        {
            "csv_name": "fig_real_heavyhitter_f1_vs_epsilon.csv",
            "tex_name": "fig_real_heavyhitter_f1_vs_epsilon.tex",
            "x_col": "epsilon_prime",
            "y_col": "f1",
            "xlabel": "target $\\epsilon'$",
            "ylabel": "top-50 F1",
            "methods": ["Local", "IdealShuffle", "Drop"],
            "caption": "Monte Carlo utility simulation on the UCI Adult dataset for top-50 heavy hitters over occupation--education categories ($m=5000$, 30 seeds). Points plot means; standard errors are saved in the CSV.",
            "label": "fig:real-heavyhitter-f1-vs-epsilon",
            "extra_axis": "ymin=0, ymax=1.02",
        },
    ]
    count = 0
    for spec in figure_specs:
        line_figure(**spec)
        count += 1
    bar_figure_microbench()
    count += 1
    bar_figure_dpf_phase_breakdown()
    count += 1
    return {"figures": count}


if __name__ == "__main__":
    print(generate_all())
