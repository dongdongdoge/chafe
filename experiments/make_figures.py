from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from chafe_eval_lib import DATA_DIR, FIGURE_DIR, ensure_dirs, fnum, read_csv


T = TypeVar("T")

BLUE = "{rgb,255:red,37;green,99;blue,235}"
TEAL = "{rgb,255:red,0;green,150;blue,136}"
ORANGE = "{rgb,255:red,213;green,94;blue,0}"
PURPLE = "{rgb,255:red,126;green,87;blue,194}"
GOLD = "{rgb,255:red,230;green,159;blue,0}"
SKY = "{rgb,255:red,86;green,180;blue,233}"
INK = "{rgb,255:red,31;green,41;blue,55}"
SLATE = "{rgb,255:red,100;green,116;blue,139}"
BLUE_FILL = "{rgb,255:red,219;green,234;blue,254}"
ORANGE_FILL = "{rgb,255:red,255;green,237;blue,213}"
TEAL_FILL = "{rgb,255:red,204;green,251;blue,241}"

LINE_STYLES = [
    f"color={BLUE}, solid, line width=0.82pt, mark=*, mark size=1.35pt, mark options={{solid, fill=white, line width=0.55pt}}",
    f"color={ORANGE}, dashed, line width=0.82pt, mark=square*, mark size=1.35pt, mark options={{solid, fill=white, line width=0.55pt}}",
    f"color={TEAL}, dotted, line width=0.95pt, mark=triangle*, mark size=1.5pt, mark options={{solid, fill=white, line width=0.55pt}}",
    f"color={PURPLE}, dashdotted, line width=0.82pt, mark=diamond*, mark size=1.45pt, mark options={{solid, fill=white, line width=0.55pt}}",
    f"color={GOLD}, densely dashed, line width=0.82pt, mark=pentagon*, mark size=1.35pt, mark options={{solid, fill=white, line width=0.55pt}}",
]

FORMULA_LINE_STYLES = [
    f"color={INK}, solid, line width=1.08pt, no markers",
    f"color={BLUE}, dashed, line width=1.05pt, no markers",
    f"color={TEAL}, dotted, line width=1.15pt, no markers",
    f"color={ORANGE}, dashdotted, line width=1.05pt, no markers",
    f"color={PURPLE}, densely dashed, line width=1.05pt, no markers",
]

BAR_STYLES = [
    f"draw={BLUE}, fill={BLUE_FILL}, line width=0.55pt",
    f"draw={ORANGE}, fill={ORANGE_FILL}, line width=0.55pt",
    f"draw={TEAL}, fill={TEAL_FILL}, line width=0.55pt",
]

MAX_LINE_COORDS = 34
MAX_SURFACE_RHO = 23
MAX_SURFACE_TAU = 21
PACKED_METHOD = "Google DPF packed64"
TUPLE_METHOD = "Google DPF tuple product"
DUAL_METHOD = "Google DPF dual-track"


def axis_options(width: str, xlabel: str, ylabel: str, extra: str = "") -> str:
    opts = [
        f"width={width}",
        "height=3.7cm",
        "scale only axis",
        "axis background/.style={fill=white}",
        "grid=major",
        "major grid style={draw=gray!22, line width=0.22pt}",
        "minor tick num=1",
        "minor grid style={draw=gray!10, line width=0.15pt}",
        "tick label style={font=\\footnotesize}",
        "label style={font=\\small}",
        "legend style={font=\\scriptsize, fill=white, fill opacity=0.94, text opacity=1, draw=gray!18, line width=0.25pt, at={(0.98,0.98)}, anchor=north east, cells={anchor=west}}",
        "legend image post style={line width=0.9pt}",
        "axis line style={black!55, line width=0.35pt}",
        "tick style={black!55, line width=0.35pt}",
        "tick align=outside",
        "clip marker paths=true",
        "filter discard warning=false",
        f"xlabel={{{xlabel}}}",
        f"ylabel={{{ylabel}}}",
    ]
    if extra:
        opts.append(extra)
    return ",\n".join(opts)


def thin_sequence(values: list[T], max_values: int | None) -> list[T]:
    if max_values is None or len(values) <= max_values:
        return values
    if max_values < 2:
        return values[:1]
    indices = {
        round(i * (len(values) - 1) / (max_values - 1))
        for i in range(max_values)
    }
    return [values[i] for i in sorted(indices)]


def coordinates(
    rows: list[dict[str, str]],
    x_col: str,
    y_col: str,
    max_points: int | None = MAX_LINE_COORDS,
) -> str:
    pairs = []
    for row in rows:
        if row.get(x_col, "") == "" or row.get(y_col, "") == "":
            continue
        pairs.append((float(row[x_col]), float(row[y_col])))
    pairs = sorted(pairs)
    if not pairs:
        raise ValueError(f"no numeric coordinates for {x_col}/{y_col}")
    pairs = thin_sequence(pairs, max_points)
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
    line_styles: list[str] | None = None,
) -> None:
    rows = read_csv(DATA_DIR / csv_name)
    plots = []
    styles = line_styles or LINE_STYLES
    axis_extra = extra_axis
    if len(methods) >= 4 and "legend columns" not in axis_extra:
        axis_extra = (axis_extra + ", " if axis_extra else "") + "legend columns=2"
    for idx, method in enumerate(methods):
        subset = method_rows(rows, method)
        if not subset:
            raise ValueError(f"{csv_name}: no rows for method {method}")
        plots.append(
            "\\addplot+[" + styles[idx % len(styles)] + "] coordinates {"
            + coordinates(subset, x_col, y_col)
            + "};\n"
            + f"\\addlegendentry{{{method}}}"
        )
    body = f"""% Auto-generated from data/{csv_name}.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{axis_options(width, xlabel, ylabel, axis_extra)}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    (FIGURE_DIR / tex_name).write_text(body)


def surface_coordinates(rows: list[dict[str, str]], method: str) -> tuple[str, int]:
    subset = [row for row in rows if row["method"] == method]
    if not subset:
        raise ValueError(f"surface has no rows for method {method}")
    rhos = thin_sequence(sorted({float(row["rho"]) for row in subset}), MAX_SURFACE_RHO)
    taus = thin_sequence(sorted({float(row["tau"]) for row in subset}), MAX_SURFACE_TAU)
    by_key = {(float(row["rho"]), float(row["tau"])): float(row["epsilon"]) for row in subset}
    lines = []
    for tau in taus:
        for rho in rhos:
            lines.append(f"({fnum(rho)},{fnum(tau)},{fnum(by_key[(rho, tau)])})")
        lines.append("")
    return "\n".join(lines), len(taus)


def privacy_surface_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_privacy_surface_rho_tau.csv")
    panels = []
    for method, title in [
        ("Drop (matched padding)", "Drop"),
        ("Pool (replacement)", "Pool repl."),
        ("Pool (add/remove)", "Pool add/rem."),
    ]:
        coords, mesh_rows = surface_coordinates(rows, method)
        panels.append(
            f"""\\begin{{minipage}}{{0.323\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
width=\\linewidth,
height=4.15cm,
scale only axis,
axis background/.style={{fill=white}},
grid=major,
grid style={{draw=gray!18, line width=0.18pt}},
tick label style={{font=\\scriptsize}},
label style={{font=\\scriptsize}},
title={{{title}}},
title style={{font=\\small, yshift=-0.5ex}},
xlabel={{$\\rho=m/N$}},
ylabel={{$\\tau=M/m$}},
zlabel={{$\\epsilon'$}},
view={{44}}{{27}},
colormap={{chafesurface}}{{rgb255(0cm)=(30,58,138); rgb255(1cm)=(14,116,144); rgb255(2cm)=(13,148,136); rgb255(3cm)=(132,204,22); rgb255(4cm)=(250,204,21)}},
colorbar=false,
axis line style={{black!50, line width=0.3pt}},
tick style={{black!55}},
z buffer=sort
]
\\addplot3[
surf,
shader=faceted interp,
faceted color=white!38,
draw=white!35,
mesh/rows={mesh_rows}
] coordinates {{
{coords}
}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )
    body = f"""% Auto-generated from data/fig_privacy_surface_rho_tau.csv.
\\begin{{figure*}}[t]
\\centering
{chr(10).join(panels)}
\\caption{{Formula-derived large-scale privacy surface over load factor and release expansion for $m=10^6$, $\\epsilon_0=1$, and $\\delta=10^{{-6}}$. Color encodes $\\epsilon'$ within each surface. The panels show matched Drop, Pool under the replacement-style bound, and Pool under the add/remove participation bound.}}
\\label{{fig:privacy-surface-rho-tau}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_privacy_surface_rho_tau.tex").write_text(body)


def paired_coordinates(rows: list[dict[str, str]], panel: str, method: str, x_col: str, y_col: str) -> str:
    return coordinates(
        [row for row in rows if row["panel"] == panel and row["method"] == method],
        x_col,
        y_col,
    )


def paired_line_figure(
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
    extra_axis: str = "",
) -> None:
    rows = read_csv(DATA_DIR / csv_name)
    legend_name = "legend:" + tex_name.replace(".tex", "").replace("_", "-")
    legend_plots = []
    for idx, method in enumerate(methods):
        legend_plots.append(
            "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {(0,0)};\n"
            + f"\\addlegendentry{{{method}}}"
        )
    panels = []
    for panel, title in [("Synthetic Zipf", "Zipf"), ("Adult empirical bootstrap", "Adult")]:
        plots = []
        for idx, method in enumerate(methods):
            subset = [row for row in rows if row["panel"] == panel and row["method"] == method]
            if not subset:
                continue
            plot = (
                "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {"
                + paired_coordinates(rows, panel, method, x_col, y_col)
                + "};"
            )
            plots.append(plot)
        axis_opts = [
            "scale only axis",
            "width=\\linewidth",
            "height=2.72cm",
            "axis background/.style={fill=white}",
            "grid=major",
            "major grid style={draw=gray!22, line width=0.2pt}",
            "minor tick num=1",
            "minor grid style={draw=gray!10, line width=0.15pt}",
            "axis line style={black!55, line width=0.32pt}",
            "tick style={black!55, line width=0.32pt}",
            "tick align=outside",
            "tick label style={font=\\tiny}",
            "label style={font=\\scriptsize}",
            "title={" + title + "}",
            "title style={font=\\scriptsize, yshift=-0.4ex}",
            "xlabel={" + xlabel + "}",
            "ylabel={" + (ylabel if panel == "Synthetic Zipf" else "") + "}",
        ]
        if panel != "Synthetic Zipf":
            axis_opts.append("yticklabel=\\empty")
        if extra_axis:
            axis_opts.append(extra_axis)
        panel_axis_options = ",\n".join(axis_opts)
        panels.append(
            f"""\\begin{{minipage}}{{0.49\\columnwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{panel_axis_options}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )
    body = f"""% Auto-generated from data/{csv_name}.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
hide axis,
scale only axis,
height=0pt,
width=0pt,
legend columns={min(4, len(methods))},
legend style={{font=\\tiny, draw=none, fill=none, /tikz/every even column/.append style={{column sep=0.5em}}}},
legend image post style={{line width=0.9pt}},
legend to name={legend_name}
]
{chr(10).join(legend_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\ref{{{legend_name}}}\\par\\vspace{{0.4mm}}
{chr(10).join(panels)}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    (FIGURE_DIR / tex_name).write_text(body)


def exact_vs_proxy_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv")
    panels = [
        ("rho", "rho", "$\\rho=m/N$", "$m=10^5$"),
        ("m", "m", "$m$", "$\\rho=1$"),
    ]
    methods = ["Exact hockey-stick", "Second-order proxy"]
    legend_name = "legend:exact-vs-proxy-drop"
    legend_plots = []
    for idx, method in enumerate(methods):
        legend_plots.append(
            "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {(0,0)};\n"
            + f"\\addlegendentry{{{method}}}"
        )
    rendered = []
    for sweep, x_col, xlabel, title in panels:
        subset = [row for row in rows if row["sweep"] == sweep]
        plots = []
        for idx, method in enumerate(methods):
            method_subset = [row for row in subset if row["method"] == method]
            plots.append(
                "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {"
                + coordinates(method_subset, x_col, "epsilon")
                + "};"
            )
        extra = "ymode=log"
        if x_col == "m":
            extra += ", xmode=log"
        panel_axis_options = axis_options(
            "\\linewidth",
            xlabel,
            "$\\epsilon'$",
            "height=3.4cm, title={" + title + "}, title style={font=\\small}, " + extra,
        )
        rendered.append(
            f"""\\begin{{minipage}}{{0.485\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{panel_axis_options}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )
    body = f"""% Auto-generated from data/fig_exact_vs_proxy_drop.csv.
\\begin{{figure*}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
hide axis,
scale only axis,
height=0pt,
width=0pt,
legend columns=2,
legend style={{font=\\scriptsize, draw=none, fill=none, /tikz/every even column/.append style={{column sep=0.7em}}}},
legend image post style={{line width=0.9pt}},
legend to name={legend_name}
]
{chr(10).join(legend_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\ref{{{legend_name}}}\\par\\vspace{{0.7mm}}
{chr(10).join(rendered)}
\\caption{{Exact finite-domain hockey-stick validation for binary randomized response with matched BRR-neighbor padding ($\\epsilon_0=1$, $\\tau=2$, $\\delta=10^{{-6}}$). The left panel varies $\\rho$ at $m=10^5$; the right panel varies $m$ at $\\rho=1$. The second-order proxy remains conservative while tracking the same large-scale trend; the exact curve is still a count-domain validation experiment rather than a full large-domain implementation accountant.}}
\\label{{fig:exact-vs-proxy-drop}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_exact_vs_proxy_drop.tex").write_text(body)


def ingestion_latency_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv")
    online = method_rows(rows, "Chafe online DPF service")
    batch = method_rows(rows, "Collect-then-shuffle gate")
    overload = method_rows(rows, "Chafe overload horizon")
    crossover = method_rows(rows, "Queue/gate equality")
    if not crossover:
        crossover = method_rows(rows, "Estimated crossover")
    if not online or not batch or not overload or not crossover:
        raise ValueError("fig_ingestion_latency_vs_rate.csv is missing latency-model rows")
    service_rate = float(online[0]["service_rate_reports_per_s"])
    service_n = int(float(online[0]["N"]))
    service_log_n = int(float(online[0]["log_n"]))
    cross_x = float(crossover[0]["arrival_rate_reports_per_s"])
    cross_y = float(crossover[0]["avg_latency_seconds"])
    opts = axis_options(
        "0.48\\textwidth",
        "arrival rate $\\lambda$ (reports/s)",
        "seconds",
        (
            "xmode=log, ymode=log, xmin=1, xmax=22000, ymin=1e-4, ymax=1e6, "
            "legend style={font=\\tiny, fill=white, draw=black!20, at={(0.03,0.97)}, anchor=north west, cells={anchor=west}}"
        ),
    )
    body = f"""% Auto-generated from data/fig_ingestion_latency_vs_rate.csv.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{opts}
]
\\addplot+[color={BLUE}, solid, line width=1.12pt, no markers] coordinates {{{coordinates(online, "arrival_rate_reports_per_s", "avg_latency_seconds")}}};
\\addlegendentry{{Chafe packed64 M/D/1 latency}}
\\addplot+[color={GOLD}, dashed, line width=1.02pt, no markers] coordinates {{{coordinates(batch, "arrival_rate_reports_per_s", "avg_latency_seconds")}}};
\\addlegendentry{{collect gate}}
\\addplot+[color={ORANGE}, dotted, line width=1.18pt, no markers] coordinates {{{coordinates(overload, "arrival_rate_reports_per_s", "avg_latency_seconds")}}};
\\addlegendentry{{overload horizon}}
\\addplot+[black!35, densely dashed, no markers, forget plot] coordinates {{({fnum(service_rate)},1e-4) ({fnum(service_rate)},1e6)}};
\\addplot+[color={PURPLE}, only marks, mark=diamond*, mark size=1.8pt, mark options={{solid, fill=white, line width=0.65pt}}] coordinates {{({fnum(cross_x)},{fnum(cross_y)})}};
\\addlegendentry{{queue/gate equality}}
\\node[font=\\tiny, rotate=90, anchor=south] at (axis cs:{fnum(service_rate)},2e-3) {{service rate}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Analytic latency model combining the measured packed64 Google DPF eval-all service rate at $N=2^{{{service_log_n}}}={service_n}$ with an M/D/1 queue, compared with a collect-then-shuffle batch-gating lower bound for $m=10^6$. The vertical line marks the measured single-thread service rate. The dotted overload curve is not steady-state latency; it is the time to accumulate one epoch of backlog after arrivals exceed service capacity.}}
\\label{{fig:ingestion-latency-vs-rate}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_ingestion_latency_vs_rate.tex").write_text(body)


def bar_figure_microbench() -> None:
    rows = read_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv")
    vals = {row["method"]: float(row["time_ms_per_report"]) for row in rows}
    benchmark_n = int(float(rows[0]["N"]))
    benchmark_log_n = int(float(rows[0]["log_n"]))
    product_name = PACKED_METHOD if PACKED_METHOD in vals else "Product-valued kernel"
    tuple_name = TUPLE_METHOD if TUPLE_METHOD in vals else ""
    dual_name = DUAL_METHOD if DUAL_METHOD in vals else "Dual-track kernel"
    benchmark = rows[0].get("benchmark", "")
    if benchmark == "google_distributed_point_functions":
        caption = f"Measured Google distributed-point-functions write backends at $N=2^{{{benchmark_log_n}}}={benchmark_n}$. Packed64 encodes the occupancy counter and payload in one 64-bit value; tuple product and dual-track are unoptimized baselines. The benchmark generates DPF keys, evaluates both server keys over the full slot domain, accumulates shares, and verifies reconstruction."
    else:
        caption = "Measured NumPy dense DPF-expansion kernel proxy for product-valued versus dual-track ingestion. The benchmark measures wall-clock traversal and pseudorandom expansion cost per report; it is not a full DPF implementation throughput claim."
    opts = axis_options(
        "0.48\\textwidth",
        "",
        "ms per report",
        "ybar, bar width=11pt, symbolic x coords={Packed,Tuple,Dual}, xtick=data, ymin=0, enlarge x limits=0.25, ymajorgrids=true",
    )
    body = f"""% Auto-generated from data/fig_product_vs_dual_microbench.csv.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{opts}
]
\\addplot+[{BAR_STYLES[0]}, bar shift=0pt] coordinates {{(Packed,{fnum(vals[product_name])})}};
{"%" if not tuple_name else "\\addplot+[" + BAR_STYLES[2] + ", bar shift=0pt] coordinates {(Tuple," + fnum(vals[tuple_name]) + ")};"}
\\addplot+[{BAR_STYLES[1]}, bar shift=0pt] coordinates {{(Dual,{fnum(vals[dual_name])})}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{caption}}}
\\label{{fig:product-vs-dual-microbench}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_product_vs_dual_microbench.tex").write_text(body)


def bar_figure_dpf_phase_breakdown() -> None:
    rows = read_csv(DATA_DIR / "fig_dpf_phase_breakdown.csv")
    benchmark_n = int(float(rows[0]["N"]))
    benchmark_log_n = int(float(rows[0]["log_n"]))
    phase_names = {
        "keygen": "Keygen",
        "context": "Ctx",
        "eval": "Eval",
        "accumulate": "Acc",
    }
    phases = ["keygen", "context", "eval", "accumulate"]
    product = {row["phase"]: float(row["time_ms_per_report"]) for row in rows if row["method"] == PACKED_METHOD}
    tuple_product = {row["phase"]: float(row["time_ms_per_report"]) for row in rows if row["method"] == TUPLE_METHOD}
    dual = {row["phase"]: float(row["time_ms_per_report"]) for row in rows if row["method"] == DUAL_METHOD}
    product_coords = " ".join(f"({phase_names[phase]},{fnum(product[phase])})" for phase in phases)
    tuple_coords = " ".join(f"({phase_names[phase]},{fnum(tuple_product[phase])})" for phase in phases)
    dual_coords = " ".join(f"({phase_names[phase]},{fnum(dual[phase])})" for phase in phases)
    opts = axis_options(
        "0.48\\textwidth",
        "",
        "ms per report",
        (
            "ybar, bar width=4.5pt, symbolic x coords={Keygen,Ctx,Eval,Acc}, "
            "xtick=data, ymin=0, enlarge x limits=0.22, "
            "legend style={font=\\scriptsize, fill=white, fill opacity=0.94, text opacity=1, draw=gray!18, line width=0.25pt, at={(0.98,0.98)}, anchor=north east}"
        ),
    )
    body = f"""% Auto-generated from data/fig_dpf_phase_breakdown.csv.
\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{opts}
]
\\addplot+[{BAR_STYLES[0]}] coordinates {{{product_coords}}};
\\addlegendentry{{Packed64}}
\\addplot+[{BAR_STYLES[2]}] coordinates {{{tuple_coords}}};
\\addlegendentry{{Tuple}}
\\addplot+[{BAR_STYLES[1]}] coordinates {{{dual_coords}}};
\\addlegendentry{{Dual}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Measured phase breakdown for Google DPF write backends at $N=2^{{{benchmark_log_n}}}={benchmark_n}$. Packed64 is the optimized single-word encoding; tuple product and dual-track are unoptimized baselines. Each bar is wall-clock time per report for key generation, evaluation-context construction, full-domain DPF evaluation, and share accumulation.}}
\\label{{fig:dpf-phase-breakdown}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_dpf_phase_breakdown.tex").write_text(body)


def dpf_backend_summary_figure() -> None:
    methods = [PACKED_METHOD, TUPLE_METHOD, DUAL_METHOD]
    legend_name = "legend:dpf-backend-summary"
    legend_plots = []
    for idx, method in enumerate(methods):
        legend_plots.append(
            "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {(0,0)};\n"
            + f"\\addlegendentry{{{method.replace('Google DPF ', '')}}}"
        )

    panels = []
    for csv_name, x_col, y_col, xlabel, ylabel, title, extra_axis in [
        (
            "fig_dpf_scaling.csv",
            "N",
            "time_ms_per_report",
            "slot domain $N$",
            "ms/report",
            "Domain Scaling",
            "xmode=log, ymode=log, log basis x=2",
        ),
        (
            "fig_dpf_thread_scaling.csv",
            "threads",
            "reports_per_second",
            "worker threads",
            "reports/s",
            "$N=2^{17}$ Thread Scaling",
            "xmin=0.8, ymin=0",
        ),
    ]:
        rows = read_csv(DATA_DIR / csv_name)
        plots = []
        for idx, method in enumerate(methods):
            subset = method_rows(rows, method)
            if not subset:
                raise ValueError(f"{csv_name}: no rows for method {method}")
            plots.append(
                "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {"
                + coordinates(subset, x_col, y_col)
                + "};"
            )
        panel_axis_options = axis_options(
            "\\linewidth",
            xlabel,
            ylabel,
            "height=3.35cm, title={" + title + "}, title style={font=\\small}, " + extra_axis,
        )
        panels.append(
            f"""\\begin{{minipage}}{{0.485\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{panel_axis_options}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )

    body = f"""% Auto-generated from data/fig_dpf_scaling.csv and data/fig_dpf_thread_scaling.csv.
\\begin{{figure*}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
hide axis,
scale only axis,
height=0pt,
width=0pt,
legend columns=3,
legend style={{font=\\scriptsize, draw=none, fill=none, /tikz/every even column/.append style={{column sep=0.7em}}}},
legend image post style={{line width=0.9pt}},
legend to name={legend_name}
]
{chr(10).join(legend_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\ref{{{legend_name}}}\\par\\vspace{{0.7mm}}
{chr(10).join(panels)}
\\caption{{Measured Google DPF write-backend scaling. Packed64 is the optimized single-word Chafe backend; tuple product and dual-track are unoptimized baselines. Left: single-thread per-report cost as the hidden-slot domain grows. Right: report-batch parallelism at $N=2^{{17}}$.}}
\\label{{fig:dpf-backend-summary}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_dpf_backend_summary.tex").write_text(body)


def dpf_backend_breakdown_summary_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv")
    phase_rows = read_csv(DATA_DIR / "fig_dpf_phase_breakdown.csv")
    methods = [PACKED_METHOD, TUPLE_METHOD, DUAL_METHOD]
    short_names = {
        PACKED_METHOD: "Packed",
        TUPLE_METHOD: "Tuple",
        DUAL_METHOD: "Dual",
    }
    vals = {row["method"]: float(row["time_ms_per_report"]) for row in rows}
    benchmark_n = int(float(rows[0]["N"]))
    benchmark_log_n = int(float(rows[0]["log_n"]))
    legend_name = "legend:dpf-backend-breakdown"
    legend_plots = []
    for idx, method in enumerate(methods):
        legend_plots.append(
            "\\addplot+[ybar, " + BAR_STYLES[idx % len(BAR_STYLES)] + "] coordinates {(0,0)};\n"
            + f"\\addlegendentry{{{short_names[method]}}}"
        )

    total_plots = []
    for idx, method in enumerate(methods):
        total_plots.append(
            "\\addplot+[" + BAR_STYLES[idx % len(BAR_STYLES)] + ", bar shift=0pt] coordinates {"
            + f"({short_names[method]},{fnum(vals[method])})"
            + "};"
        )
    total_axis_options = axis_options(
        "\\linewidth",
        "",
        "ms/report",
        (
            "height=3.35cm, title={End-to-End Backend}, title style={font=\\small}, "
            "ybar, bar width=10pt, symbolic x coords={Packed,Tuple,Dual}, xtick=data, ymin=0, enlarge x limits=0.25"
        ),
    )

    phase_names = {
        "keygen": "Keygen",
        "context": "Ctx",
        "eval": "Eval",
        "accumulate": "Acc",
    }
    phases = ["keygen", "context", "eval", "accumulate"]
    phase_plots = []
    for idx, method in enumerate(methods):
        by_phase = {
            row["phase"]: float(row["time_ms_per_report"])
            for row in phase_rows
            if row["method"] == method
        }
        coords = " ".join(f"({phase_names[phase]},{fnum(by_phase[phase])})" for phase in phases)
        phase_plots.append("\\addplot+[" + BAR_STYLES[idx % len(BAR_STYLES)] + "] coordinates {" + coords + "};")
    phase_axis_options = axis_options(
        "\\linewidth",
        "",
        "ms/report",
        (
            "height=3.35cm, title={Phase Breakdown}, title style={font=\\small}, "
            "ybar, bar width=4.3pt, symbolic x coords={Keygen,Ctx,Eval,Acc}, xtick=data, ymin=0, enlarge x limits=0.20"
        ),
    )

    body = f"""% Auto-generated from data/fig_product_vs_dual_microbench.csv and data/fig_dpf_phase_breakdown.csv.
\\begin{{figure*}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
hide axis,
scale only axis,
height=0pt,
width=0pt,
legend columns=3,
legend style={{font=\\scriptsize, draw=none, fill=none, /tikz/every even column/.append style={{column sep=0.7em}}}},
legend to name={legend_name}
]
{chr(10).join(legend_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\ref{{{legend_name}}}\\par\\vspace{{0.7mm}}
\\begin{{minipage}}{{0.485\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{total_axis_options}
]
{chr(10).join(total_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}
\\begin{{minipage}}{{0.485\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{phase_axis_options}
]
{chr(10).join(phase_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}
\\caption{{Measured Google DPF write-backend decomposition at $N=2^{{{benchmark_log_n}}}={benchmark_n}$. Packed64 encodes occupancy and payload in one 64-bit value; tuple product and dual-track show the cost of less specialized encodings.}}
\\label{{fig:dpf-backend-breakdown}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_dpf_backend_breakdown.tex").write_text(body)


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
            "caption": "Formula-derived privacy amplification versus load factor for $m=10^6$, $\\tau=2$, $\\epsilon_0=1$, and $\\delta=10^{-6}$. Pool replacement and add/remove curves use their respective second-order bounds.",
            "label": "fig:privacy-vs-rho",
            "line_styles": FORMULA_LINE_STYLES,
        },
        {
            "csv_name": "fig_privacy_vs_eps0.csv",
            "tex_name": "fig_privacy_vs_eps0.tex",
            "x_col": "eps0",
            "y_col": "epsilon",
            "xlabel": "$\\epsilon_0$",
            "ylabel": "$\\epsilon'$",
            "methods": ["IdealShuffle", "Drop (matched padding)", "Pool (replacement)"],
            "caption": "Formula-derived privacy amplification as the local privacy parameter varies for $m=10^6$, $\\rho=1$, $\\tau=2$, and $\\delta=10^{-6}$.",
            "label": "fig:privacy-vs-eps0",
            "line_styles": FORMULA_LINE_STYLES,
        },
        {
            "csv_name": "fig_privacy_vs_tau.csv",
            "tex_name": "fig_privacy_vs_tau.tex",
            "x_col": "tau",
            "y_col": "epsilon",
            "xlabel": "$\\tau=M/m$",
            "ylabel": "$\\epsilon'$",
            "methods": ["IdealShuffle", "Drop (matched padding)", "Pool (replacement)"],
            "caption": "Formula-derived privacy amplification versus matched release expansion $\\tau=M/m$ for $m=10^6$, $\\rho=1$, $\\epsilon_0=1$, and $\\delta=10^{-6}$.",
            "label": "fig:privacy-vs-tau",
            "line_styles": FORMULA_LINE_STYLES,
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
            "caption": "Formula-derived privacy--utility frontier from scanning $\\rho$ for $m=10^6$, $\\tau=2$, $\\epsilon_0=1$, and $\\delta=10^{-6}$. Drop uses utility cost $e^{\\rho/2}$; Pool replacement keeps additive cost at one.",
            "label": "fig:privacy-utility-frontier",
            "width": "0.48\\textwidth",
            "extra_axis": "xmin=0.95",
            "line_styles": FORMULA_LINE_STYLES,
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
            "caption": "Monte Carlo utility simulation for 1,000 random range-count queries over a Zipf histogram ($m=10^5$, $k=1024$, 30 seeds). Points plot means; standard errors are saved in the CSV.",
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
            "caption": "Monte Carlo utility simulation for bounded sums with Laplace additive reports ($m=10^5$, 30 seeds). Pool uses the replacement-style formula to choose $\\epsilon_0$ and preserves additive sums in the simulator.",
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
            "caption": "Monte Carlo utility simulation for top-50 heavy-hitter recovery over a Zipf histogram ($m=10^5$, $k=1024$, 30 seeds). Points plot means; standard errors are saved in the CSV.",
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
            "methods": [PACKED_METHOD, TUPLE_METHOD, DUAL_METHOD],
            "caption": "Measured Google DPF backend scaling with the hidden-slot domain size. Packed64 is the optimized single-word write backend; tuple product and dual-track are unoptimized baselines. Each point includes DPF key generation, evaluation of both server keys over all slots, accumulation, and reconstruction checks.",
            "label": "fig:dpf-scaling",
            "extra_axis": "xmode=log, ymode=log, log basis x=2",
        },
        {
            "csv_name": "fig_dpf_throughput_vs_n.csv",
            "tex_name": "fig_dpf_throughput_vs_n.tex",
            "x_col": "N",
            "y_col": "reports_per_second",
            "xlabel": "slot domain $N$",
            "ylabel": "reports/s",
            "methods": [PACKED_METHOD, TUPLE_METHOD, DUAL_METHOD],
            "caption": "Measured single-thread Google DPF backend service rate obtained as the reciprocal of measured wall-clock time per report. The curve isolates the eval-all write backend; it is not a full deployment throughput claim.",
            "label": "fig:dpf-throughput-vs-n",
            "extra_axis": "xmode=log, ymode=log, log basis x=2",
        },
        {
            "csv_name": "fig_dpf_thread_scaling.csv",
            "tex_name": "fig_dpf_thread_scaling.tex",
            "x_col": "threads",
            "y_col": "reports_per_second",
            "xlabel": "worker threads",
            "ylabel": "reports/s",
            "methods": [PACKED_METHOD, TUPLE_METHOD, DUAL_METHOD],
            "caption": "Measured thread scaling of Google DPF write backends at $N=2^{17}=131072$. Packed64 is the optimized single-word encoding; tuple product and dual-track are unoptimized baselines. Each worker evaluates independent report batches into private accumulators, followed by deterministic share reduction; the CSV records whether the binary was compiled with AES-NI, ARM crypto, or portable code.",
            "label": "fig:dpf-thread-scaling",
            "extra_axis": "xmin=0.8, ymin=0",
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
            "caption": "Monte Carlo utility simulation on an empirical bootstrap from the UCI Adult dataset for age range-count queries ($m=10^5$, 30 seeds). Points plot thresholded mean relative error; standard errors and dataset provenance are saved in the CSV.",
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
            "caption": "Monte Carlo utility simulation on an empirical bootstrap from the UCI Adult dataset for the sum of bounded hours-per-week values ($m=10^5$, 30 seeds). The additive Laplace report is a utility model, not a pure-LDP claim.",
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
            "caption": "Monte Carlo utility simulation on an empirical bootstrap from the UCI Adult dataset for top-50 heavy hitters over occupation--education categories ($m=10^5$, 30 seeds). Points plot means; standard errors are saved in the CSV.",
            "label": "fig:real-heavyhitter-f1-vs-epsilon",
            "extra_axis": "ymin=0, ymax=1.02",
        },
    ]
    count = 0
    for spec in figure_specs:
        line_figure(**spec)
        count += 1
    exact_vs_proxy_figure()
    count += 1
    ingestion_latency_figure()
    count += 1
    dpf_backend_summary_figure()
    count += 1
    dpf_backend_breakdown_summary_figure()
    count += 1
    privacy_surface_figure()
    count += 1
    paired_line_figure(
        csv_name="fig_range_error_zipf_adult.csv",
        tex_name="fig_range_error_zipf_adult.tex",
        x_col="epsilon_prime",
        y_col="mean_relative_error",
        xlabel="$\\epsilon'$",
        ylabel="rel. error",
        methods=["Local", "IdealShuffle", "Drop"],
        caption="Side-by-side Monte Carlo range-count utility at $m=10^5$: synthetic Zipf histogram (left) and Adult empirical bootstrap age ranges (right). Each curve plots means over 30 seeds; standard errors remain in the CSV.",
        label="fig:range-error-zipf-adult",
        extra_axis="ymode=log",
    )
    count += 1
    paired_line_figure(
        csv_name="fig_sum_error_zipf_adult.csv",
        tex_name="fig_sum_error_zipf_adult.tex",
        x_col="epsilon_prime",
        y_col="mean_relative_error",
        xlabel="$\\epsilon'$",
        ylabel="rel. error",
        methods=["Local", "IdealShuffle", "Drop", "Pool"],
        caption="Side-by-side Monte Carlo bounded-sum utility at $m=10^5$: synthetic beta-distributed values (left) and Adult hours-per-week empirical bootstrap (right). The Laplace report is a utility model.",
        label="fig:sum-error-zipf-adult",
        extra_axis="ymode=log",
    )
    count += 1
    paired_line_figure(
        csv_name="fig_heavyhitter_f1_zipf_adult.csv",
        tex_name="fig_heavyhitter_f1_zipf_adult.tex",
        x_col="epsilon_prime",
        y_col="f1",
        xlabel="$\\epsilon'$",
        ylabel="top-50 F1",
        methods=["Local", "IdealShuffle", "Drop"],
        caption="Side-by-side Monte Carlo heavy-hitter utility at $m=10^5$: synthetic Zipf categories (left) and Adult occupation--education categories (right).",
        label="fig:heavyhitter-f1-zipf-adult",
        extra_axis="ymin=0, ymax=1.02",
    )
    count += 1
    bar_figure_microbench()
    count += 1
    bar_figure_dpf_phase_breakdown()
    count += 1
    return {"figures": count}


if __name__ == "__main__":
    print(generate_all())
