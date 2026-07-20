from __future__ import annotations

import math
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
PURPLE_FILL = "{rgb,255:red,237;green,233;blue,246}"
GOLD_FILL = "{rgb,255:red,255;green,247;blue,204}"

LINE_STYLES = [
    f"color={BLUE}, solid, line width=0.98pt, mark=*, mark size=1.75pt, mark options={{solid, fill={BLUE}, line width=0.55pt}}",
    f"color={ORANGE}, dashed, line width=0.94pt, mark=square*, mark size=1.65pt, mark options={{solid, fill=white, line width=0.65pt}}",
    f"color={TEAL}, dotted, line width=1.02pt, mark=triangle*, mark size=1.85pt, mark options={{solid, fill=white, line width=0.65pt}}",
    f"color={PURPLE}, dashdotted, line width=0.94pt, mark=diamond*, mark size=1.75pt, mark options={{solid, fill=white, line width=0.65pt}}",
    f"color={GOLD}, densely dashed, line width=0.94pt, mark=pentagon*, mark size=1.65pt, mark options={{solid, fill=white, line width=0.65pt}}",
]

FORMULA_LINE_STYLES = [
    f"color={INK}, solid, line width=1.08pt, no markers",
    f"color={BLUE}, dashed, line width=1.05pt, no markers",
    f"color={TEAL}, dotted, line width=1.15pt, no markers",
    f"color={ORANGE}, dashdotted, line width=1.05pt, no markers",
    f"color={PURPLE}, densely dashed, line width=1.05pt, no markers",
]

BAR_STYLES = [
    "draw=black!70, fill=black!10, line width=0.62pt",
    f"draw={ORANGE}, fill={ORANGE_FILL}, pattern=north east lines, pattern color={ORANGE}, line width=0.62pt",
    f"draw={TEAL}, fill={TEAL_FILL}, pattern=crosshatch, pattern color={TEAL}, line width=0.62pt",
]

# Semantic styles used by the three manuscript summary figures.  Evidence type
# is encoded by line and marker before color: measured/exact series are solid
# and marked; projections/proxies are neutral and dashed.
MEASURED_STYLE = LINE_STYLES[0]
PROJECTED_STYLE = f"color={SLATE}, dashed, line width=0.95pt, no markers"
IDEAL_STYLE = f"color={INK}, solid, line width=1.02pt, no markers"
SAME_M_STYLE = f"color={BLUE}, dashed, line width=0.98pt, no markers"
DROP_STYLE = f"color={ORANGE}, dashdotted, line width=1.02pt, no markers"
POOL_STYLE = f"color={TEAL}, dotted, line width=1.10pt, no markers"
EXACT_STYLE = f"color={ORANGE}, solid, line width=1.02pt, mark=square*, mark size=1.7pt, mark options={{solid, fill={ORANGE}, line width=0.55pt}}"
PROXY_STYLE = f"color={SLATE}, dashed, line width=0.95pt, mark=o, mark size=1.65pt, mark options={{solid, fill=white, line width=0.65pt}}"
ETA_FINITE_STYLE = f"color={BLUE}, solid, line width=1.00pt, mark=*, mark size=1.55pt, mark options={{solid, fill={BLUE}, line width=0.50pt}}"
ETA_POISSON_STYLE = f"color={ORANGE}, dashed, line width=0.95pt, mark=square*, mark size=1.45pt, mark options={{solid, fill=white, line width=0.60pt}}"
EPS_EXACT_STYLE = f"color={INK}, solid, line width=1.05pt, mark=diamond*, mark size=1.55pt, mark options={{solid, fill={INK}, line width=0.50pt}}"
EPS_QUAD_STYLE = f"color={TEAL}, dashed, line width=1.00pt, mark=triangle*, mark size=1.65pt, mark options={{solid, fill=white, line width=0.60pt}}"

MAX_LINE_COORDS = 34
MAX_SURFACE_RHO = 23
MAX_SURFACE_TAU = 21
PACKED_METHOD = "Google DPF packed64"
TUPLE_METHOD = "Google DPF tuple product"
DUAL_METHOD = "Google DPF dual-track"


def split_tikz_options(style: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(style):
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(style[start:idx].strip())
            start = idx + 1
    parts.append(style[start:].strip())
    return [part for part in parts if part]


def legend_line_style(style: str) -> str:
    drop_prefixes = ("mark=", "mark size=", "mark options=")
    parts = []
    for part in split_tikz_options(style):
        if part == "no markers" or part.startswith(drop_prefixes):
            continue
        parts.append(part)
    return ", ".join(parts)


def legend_marker_style(style: str) -> tuple[str, str]:
    draw_parts = []
    plot_parts = ["only marks"]
    for part in split_tikz_options(style):
        if part.startswith("color="):
            draw_parts.append(part)
        elif part.startswith(("mark=", "mark size=", "mark options=")):
            plot_parts.append(part)
    if len(plot_parts) == 1:
        return "", ""
    return ", ".join(draw_parts), ", ".join(plot_parts)


def legend_tabular_preamble(columns: int) -> str:
    cells = ["c@{\\hspace{0.18em}}l" for _ in range(columns)]
    return "@{}" + "@{\\hspace{0.62em}}".join(cells) + "@{}"


def inline_line_legend(entries: list[tuple[str, str]], columns: int) -> str:
    cells = []
    for label, style in entries:
        marker_draw, marker_plot = legend_marker_style(style)
        marker = ""
        if marker_plot:
            marker = (
                " \\draw["
                + marker_draw
                + "] plot["
                + marker_plot
                + "] coordinates {(0.21,0)};"
            )
        cells.append(
            "\\tikz[baseline=-0.55ex]{\\draw["
            + legend_line_style(style)
            + "] (0,0) -- (0.42,0);"
            + marker
            + "}"
        )
        cells.append("{\\scriptsize " + label + "}")
    row = " & ".join(cells) + r" \\"
    return (
        "\\begingroup\n"
        "\\setlength{\\tabcolsep}{2.2pt}\n"
        "\\renewcommand{\\arraystretch}{0.90}\n"
        "\\makebox[\\linewidth][c]{%\n"
        "\\begin{tabular}{" + legend_tabular_preamble(columns) + "}\n"
        + row
        + "\n\\end{tabular}%\n"
        "}\n"
        "\\endgroup"
    )


def inline_bar_legend(entries: list[tuple[str, str]], columns: int) -> str:
    cells = []
    for label, style in entries:
        cells.append(
            "\\tikz[baseline=-0.55ex]{\\draw["
            + style
            + "] (0,0) rectangle (0.34,0.11);}"
        )
        cells.append("{\\scriptsize " + label + "}")
    row = " & ".join(cells) + r" \\"
    return (
        "\\begingroup\n"
        "\\setlength{\\tabcolsep}{2.2pt}\n"
        "\\renewcommand{\\arraystretch}{0.90}\n"
        "\\makebox[\\linewidth][c]{%\n"
        "\\begin{tabular}{" + legend_tabular_preamble(columns) + "}\n"
        + row
        + "\n\\end{tabular}%\n"
        "}\n"
        "\\endgroup"
    )


def axis_options(width: str, xlabel: str, ylabel: str, extra: str = "") -> str:
    opts = [
        f"width={width}",
        "height=3.7cm",
        "scale only axis",
        "axis background/.style={fill=white}",
        "ymajorgrids=true",
        "xmajorgrids=false",
        "major grid style={draw=gray!20, line width=0.20pt}",
        "minor tick num=1",
        "minor grid style={draw=none}",
        "tick label style={font=\\scriptsize}",
        "label style={font=\\scriptsize}",
        "legend style={font=\\scriptsize, fill=white, fill opacity=0.94, text opacity=1, draw=gray!18, line width=0.25pt, at={(0.98,0.98)}, anchor=north east, cells={anchor=west}}",
        "legend image post style={line width=1.0pt}",
        "axis line style={black!55, line width=0.35pt}",
        "tick style={black!55, line width=0.35pt}",
        "tick align=outside",
        "axis on top",
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


def latex_int(value: int) -> str:
    text = f"{value:,}".replace(",", "{,}")
    exponent = 0
    tmp = value
    while tmp > 1 and tmp % 10 == 0:
        tmp //= 10
        exponent += 1
    if tmp == 1 and exponent >= 4:
        return f"10^{exponent}"
    return text


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
\\begin{{figure*}}[!t]
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


def paired_panel_specs(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    title_by_panel = {
        "Synthetic Zipf": "Zipf",
        "Synthetic Beta": "Beta",
        "Adult empirical bootstrap": "Adult",
    }
    seen = {row["panel"] for row in rows if row.get("panel")}
    ordered = ["Synthetic Zipf", "Synthetic Beta", "Adult empirical bootstrap"]
    specs = [(panel, title_by_panel.get(panel, panel)) for panel in ordered if panel in seen]
    if len(specs) != 2:
        raise ValueError(f"expected two panels in paired figure, found {sorted(seen)}")
    return specs


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
    legend = inline_line_legend(
        [(method, LINE_STYLES[idx % len(LINE_STYLES)]) for idx, method in enumerate(methods)],
        min(4, len(methods)),
    )
    panels = []
    for panel_index, (panel, title) in enumerate(paired_panel_specs(rows)):
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
            "ylabel={" + (ylabel if panel_index == 0 else "") + "}",
        ]
        if panel_index != 0:
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
{legend}\\par\\vspace{{0.4mm}}
{chr(10).join(panels)}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    (FIGURE_DIR / tex_name).write_text(body)


def full_release_dummy_mismatch_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_full_release_dummy_mismatch.csv")
    law_order = [
        ("matched_reference", 0),
        ("neutral_public", 1),
        ("mismatched_prior", 2),
    ]
    methods = [
        ("fixed_eps0", "fixed $\\epsilon_0=1$", BAR_STYLES[0], "-3.0pt"),
        (
            "matched_proxy_selected_eps0",
            "repl.-proxy-selected $\\epsilon_0$",
            BAR_STYLES[1],
            "3.0pt",
        ),
    ]
    panels = []
    for decoder, title in [("drop", "Drop binary count"), ("pool", "Pool additive sum")]:
        plots = []
        for privacy_mode, _, style, shift in methods:
            subset = {
                row["dummy_law"]: row
                for row in rows
                if row["decoder"] == decoder and row["privacy_mode"] == privacy_mode
            }
            if set(subset) != {law for law, _ in law_order}:
                raise ValueError(
                    f"full-release figure has incomplete {decoder}/{privacy_mode} dummy laws"
                )
            coords = " ".join(
                f"({x},{fnum(float(subset[law]['relative_rmse']))})"
                for law, x in law_order
            )
            plots.append(
                "\\addplot+[ybar, bar width=5.2pt, bar shift="
                + shift
                + ", "
                + style
                + "] coordinates {"
                + coords
                + "};"
            )
        panels.append(
            f"""\\begin{{minipage}}{{0.485\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
width=\\linewidth,
height=3.25cm,
scale only axis,
ybar,
ymin=0,
bar width=5.2pt,
axis background/.style={{fill=white}},
grid=major,
major grid style={{draw=gray!22, line width=0.2pt}},
axis line style={{black!55, line width=0.32pt}},
tick style={{black!55, line width=0.32pt}},
tick label style={{font=\\scriptsize}},
label style={{font=\\small}},
title={{{title}}},
title style={{font=\\small, yshift=-0.6ex}},
xtick={{0,1,2}},
xticklabels={{Matched ref.,Neutral,Prior mismatch}},
x tick label style={{rotate=18, anchor=north east}},
xmin=-0.55,
xmax=2.55,
ylabel={{relative RMSE}},
scaled y ticks=false,
clip=false
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )
    legend = inline_bar_legend(
        [(label, style) for _, label, style, _ in methods],
        2,
    )
    body = f"""% Auto-generated from data/fig_full_release_dummy_mismatch.csv.
\\begin{{figure*}}[t]
\\centering
{chr(10).join(panels)}
\\par\\vspace{{0.25ex}}
{legend}
\\caption{{Full-release Monte Carlo at $m=N=20{{,}}000$, $M=40{{,}}000$, and 100 seeds.  The analyzer receives only the shuffled length-$M$ values, without real/dummy labels or the realized real-output count.  Estimators subtract the known expected dummy contribution and use exact finite-bin occupancy probabilities.  ``Matched'' is an oracle population-law reference; neutral and prior-mismatch rows use their own known public law.  The matched replacement proxy only selects $\\epsilon_0$ and does not certify equal privacy for the other laws.  Bars show relative RMSE; the CSV also reports signed bias and standard errors.}}
\\label{{fig:full-release-dummy-mismatch}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_full_release_dummy_mismatch.tex").write_text(body)


def covertype_workloads_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_covertype_workloads.csv")
    methods = ["Local", "IdealShuffle", "Drop", "Pool"]
    legend = inline_line_legend(
        [(method, LINE_STYLES[idx % len(LINE_STYLES)]) for idx, method in enumerate(methods)],
        4,
    )
    panel_specs = [
        ("elevation_range", "Elevation Range", "mean_relative_error", "rel. error", "ymode=log"),
        ("soil_cover_hh", "Soil/Cover HH", "f1", "top-50 F1", "ymin=0, ymax=1.03"),
        ("road_distance_sum", "Road Sum", "mean_relative_error", "rel. error", "ymode=log"),
    ]
    panels = []
    for panel_index, (task, title, y_col, ylabel, extra_axis) in enumerate(panel_specs):
        plots = []
        for idx, method in enumerate(methods):
            subset = [row for row in rows if row["task"] == task and row["method"] == method]
            if not subset:
                continue
            plots.append(
                "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {"
                + coordinates(subset, "epsilon_prime", y_col)
                + "};"
            )
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
            "xlabel={target $\\epsilon'$}",
            "ylabel={" + (ylabel if panel_index == 0 else "") + "}",
            extra_axis,
        ]
        panel_axis_options = ",\n".join(opt for opt in axis_opts if opt)
        panels.append(
            f"""\\begin{{minipage}}{{0.323\\textwidth}}
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
    body = f"""% Auto-generated from data/fig_covertype_workloads.csv.
\\begin{{figure*}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
{chr(10).join(panels)}
\\caption{{Monte Carlo utility on the UCI Covertype dataset ($581{{,}}012$ rows; $m=300{{,}}000$ without replacement; 20 seeds). The panels evaluate elevation range-count queries, top-50 heavy hitters over wilderness--soil--cover categories, and bounded sums of normalized road distance. GRR is used for categorical tasks; the additive Laplace report is a utility model for sums.}}
\\label{{fig:covertype-workloads}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_covertype_workloads.tex").write_text(body)


def exact_vs_proxy_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv")
    panels = [
        ("rho", "rho", "$\\rho=m/N$", "$m=10^5$"),
        ("m", "m", "$m$", "$\\rho=1$"),
    ]
    methods = ["Exact-law numerical", "Second-order proxy"]
    legend = inline_line_legend(
        [(method, LINE_STYLES[idx % len(LINE_STYLES)]) for idx, method in enumerate(methods)],
        2,
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
        ylabel = "$\\epsilon'$" if sweep == "rho" else ""
        right_axis = ", yticklabel=\\empty" if sweep != "rho" else ""
        panel_axis_options = axis_options(
            "\\linewidth",
            xlabel,
            ylabel,
            (
                "height=2.72cm, tick label style={font=\\tiny}, label style={font=\\scriptsize}, "
                "title={" + title + "}, title style={font=\\scriptsize, yshift=-0.4ex}, "
                + extra
                + right_axis
            ),
        )
        rendered.append(
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
    body = f"""% Auto-generated from data/fig_exact_vs_proxy_drop.csv.
\\begin{{figure}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
{chr(10).join(rendered)}
\\caption{{Numerical evaluation of the exact finite-$(m,N)$ count law for one add/remove pair in the restricted homogeneous binary-RR case with matched background and dummy laws ($\\epsilon_0=1$, $\\tau=2$, $\\delta=10^{{-6}}$). Reported float64 values are cross-checked by a 70-digit decimal likelihood-ratio recurrence; this is not an interval certificate. The left panel varies $\\rho$ at $m=10^5$; the right varies $m$ at $\\rho=1$.}}
\\label{{fig:exact-vs-proxy-drop}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_exact_vs_proxy_drop.tex").write_text(body)


def accountant_scaling_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_accountant_scaling.csv")
    methods = [
        "Coefficient DP (K=2)",
        "Coefficient DP (K=3)",
        "Matched-Drop sufficient count law",
    ]
    legend = inline_line_legend(
        [
            (method, LINE_STYLES[index % len(LINE_STYLES)])
            for index, method in enumerate(methods)
        ],
        3,
    )
    panels = [
        ("runtime_median_s", "measured runtime (s)"),
        ("final_support_states", "final support states"),
    ]
    rendered = []
    for panel_index, (y_col, ylabel) in enumerate(panels):
        plots = []
        for method_index, method in enumerate(methods):
            subset = [
                row for row in rows
                if row["method"] == method
            ]
            plots.append(
                "\\addplot+["
                + LINE_STYLES[method_index % len(LINE_STYLES)]
                + "] coordinates {"
                + coordinates(subset, "m", y_col)
                + "};"
            )
        panel_axis_options = axis_options(
            "\\linewidth",
            "$m$",
            ylabel,
            (
                "height=2.72cm, xmode=log, ymode=log, "
                "tick label style={font=\\tiny}, "
                "label style={font=\\scriptsize}"
            ),
        )
        rendered.append(
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
    body = f"""% Auto-generated from data/fig_accountant_scaling.csv.
\\begin{{figure}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
{chr(10).join(rendered)}
\\caption{{Measured accountant scaling (median of three runs on the evaluation host).  The generic coefficient DP is shown only for one homogeneous \\textsc{{Pool}} add/remove pair and becomes count-state limited as the output alphabet grows.  The binary matched-\\textsc{{Drop}} line uses its sufficient count law and reaches $m=10^5$.  These timings do not claim general-accountant scalability or interval-certified arithmetic.}}
\\label{{fig:accountant-scaling}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_accountant_scaling.tex").write_text(body)


def ingestion_latency_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_ingestion_latency_vs_rate.csv")
    online = method_rows(rows, "Chafe online DPF service")
    batch = method_rows(rows, "Collect-then-shuffle gate")
    clarion = method_rows(rows, "Collect gate + Clarion shuffle")
    overload = method_rows(rows, "Chafe overload horizon")
    crossover = method_rows(rows, "Queue/gate equality")
    if not crossover:
        crossover = method_rows(rows, "Estimated crossover")
    if not online or not batch or not overload or not crossover:
        raise ValueError("fig_ingestion_latency_vs_rate.csv is missing latency-model rows")
    service_rate = float(online[0]["service_rate_reports_per_s"])
    service_n = int(float(online[0]["N"]))
    service_log_n = int(float(online[0]["log_n"]))
    service_threads = int(float(online[0].get("threads", "1") or "1"))
    epoch_m = int(float(online[0]["m"]))
    rho_value = online[0].get("rho", "")
    rho_text = f", $\\rho={float(rho_value):.2f}$" if rho_value else ""
    shuffle_seconds = crossover[0].get("shuffle_seconds", "")
    if shuffle_seconds:
        baseline_text = (
            f"The batch baseline waits to collect $m={latex_int(epoch_m)}$ reports "
            f"and then adds a conservative {float(shuffle_seconds):.2f} s "
            "Clarion-style secret-shared shuffle term~\\cite{eskandarian2021clarion}"
        )
    else:
        baseline_text = (
            f"The batch baseline is the collect-then-shuffle gate-only lower bound "
            f"for $m={latex_int(epoch_m)}$"
        )
    equality_label = "queue/batch equality" if clarion else "queue/gate equality"
    if clarion:
        clarion_plot = (
            f"\\addplot+[color={TEAL}, dashdotted, line width=1.05pt, no markers] "
            "coordinates {"
            + coordinates(clarion, "arrival_rate_reports_per_s", "avg_latency_seconds")
            + "};\n\\addlegendentry{collect + Clarion shuffle}"
        )
    else:
        clarion_plot = ""
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
\\addlegendentry{{collect gate only}}
{clarion_plot}
\\addplot+[color={ORANGE}, dotted, line width=1.18pt, no markers] coordinates {{{coordinates(overload, "arrival_rate_reports_per_s", "avg_latency_seconds")}}};
\\addlegendentry{{queue backlog horizon}}
\\addplot+[black!35, densely dashed, no markers, forget plot] coordinates {{({fnum(service_rate)},1e-4) ({fnum(service_rate)},1e6)}};
\\addplot+[color={PURPLE}, only marks, mark=diamond*, mark size=1.8pt, mark options={{solid, fill=white, line width=0.65pt}}] coordinates {{({fnum(cross_x)},{fnum(cross_y)})}};
\\addlegendentry{{{equality_label}}}
\\node[font=\\tiny, rotate=90, anchor=south] at (axis cs:{fnum(service_rate)},2e-3) {{service rate}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Analytic ingestion-latency model for \\sys{{}} and collect-then-shuffle. The \\sys{{}} curve models the measured {service_threads}-worker packed64 Google DPF eval-all backend at $N=2^{{{service_log_n}}}={service_n}$ as an M/D/1 queue. {baseline_text} ({rho_text.lstrip(", ")}). The vertical line marks the measured service rate; the dotted curve reports the time to accumulate one epoch of backlog after overload, not steady-state latency.}}
\\label{{fig:ingestion-latency-vs-rate}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_ingestion_latency_vs_rate.tex").write_text(body)


def _legacy_ingestion_cost_summary_figure() -> None:
    methods = [PACKED_METHOD, TUPLE_METHOD, DUAL_METHOD]
    short = {
        PACKED_METHOD: "packed64",
        TUPLE_METHOD: "tuple",
        DUAL_METHOD: "dual-track",
    }

    domain_rows = read_csv(DATA_DIR / "fig_dpf_throughput_vs_n.csv")
    if {row["source_type"] for row in domain_rows} != {"microbenchmark"}:
        raise ValueError("domain-cost panel must contain measured microbenchmarks only")
    domain_plots = []
    for idx, method in enumerate(methods):
        domain_plots.append(
            "\\addplot+[" + LINE_STYLES[idx % len(LINE_STYLES)] + "] coordinates {"
            + coordinates(method_rows(domain_rows, method), "N", "reports_per_second")
            + "};\n\\addlegendentry{" + short[method] + "}"
        )
    domain_axis = axis_options(
        "0.73\\linewidth",
        "slot domain $N$",
        "hidden writes/s",
        (
            "height=2.78cm, title={(a) Domain cost}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "xmode=log, ymode=log, log basis x=2, "
            "legend style={font=\\scriptsize, draw=none, fill=none, at={(0.98,0.98)}, anchor=north east}"
        ),
    )

    node_rows = sorted(
        read_csv(DATA_DIR / "fig_ingestion_node_scaling.csv"),
        key=lambda row: float(row["node_count"]),
    )
    if {row["source_type"] for row in node_rows} != {"analytic_cost"}:
        raise ValueError("node-capacity panel must remain an analytic projection")
    parallel_axis = axis_options(
        "0.73\\linewidth",
        "nodes",
        "capacity (writes/s)",
        (
            "height=2.78cm, title={(b) Projected capacity}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "xmode=log, ymode=log, log basis x=2, xmin=1, xmax=64, "
            "xtick={1,2,4,8,16,32,64}, "
            "legend style={font=\\scriptsize, draw=none, fill=none, at={(0.50,0.98)}, "
            "anchor=north, legend columns=2}"
        ),
    )

    stage_rows = {
        row["method"]: row
        for row in read_csv(DATA_DIR / "fig_epoch_end_stage_cost.csv")
        if row["phase"] == "total"
    }
    loopback_rows = {
        row["method"]: row
        for row in read_csv(DATA_DIR / "fig_epoch_end_loopback.csv")
        if row["phase"] == "total"
    }
    if {
        row["benchmark"] for row in loopback_rows.values()
    } != {"two_process_loopback_epoch_release"}:
        raise ValueError("epoch-close panel must use the audited release-stage loopback data")
    release_values = [
        float(stage_rows["Trusted padded shuffle"]["time_ms"]),
        float(stage_rows["Chafe Drop"]["time_ms"]),
        float(stage_rows["Chafe Pool"]["time_ms"]),
        float(loopback_rows["Two-process loopback Drop"]["time_ms"]),
        float(loopback_rows["Two-process loopback Pool"]["time_ms"]),
    ]
    release_ymax = 1.18 * max(release_values)
    release_axis = axis_options(
        "0.73\\linewidth",
        "",
        "time (ms)",
        (
            "height=2.78cm, title={(c) Epoch close}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            f"ybar, bar width=5.2pt, ymin=0, ymax={release_ymax:.4g}, xmin=0.3, xmax=7.7, clip=false, "
            "xtick={1,3,4,6,7}, xticklabels={Ideal,Drop,Pool,Drop,Pool}, "
            "x tick label style={font=\\scriptsize}, "
            "extra x ticks={1,3.5,6.5}, extra x tick labels={trusted,stage-only,2-process}, "
            "extra x tick style={tick label style={font=\\scriptsize, yshift=-11pt}, tick style={draw=none}}"
        ),
    )
    capacity_band_coordinates = " ".join(
        [
            f"({fnum(float(row['node_count']))},{fnum(float(row['aggregate_service_rate_reports_per_s']))})"
            for row in node_rows
        ]
        + [
            f"({fnum(float(row['node_count']))},{fnum(float(row['capacity_at_70pct_reports_per_s']))})"
            for row in reversed(node_rows)
        ]
    )

    body = f"""% Auto-generated from measured ingestion/release and node-projection CSVs.
\\begin{{figure*}}[t]
\\centering
\\begin{{minipage}}{{0.32\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{domain_axis}
]
{chr(10).join(domain_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}{{0.32\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{parallel_axis}
]
\\addplot+[draw=none, fill=gray!18, forget plot] coordinates {{{capacity_band_coordinates}}} \\closedcycle;
\\addplot+[{PROJECTED_STYLE}] coordinates {{{coordinates(node_rows, "node_count", "aggregate_service_rate_reports_per_s", max_points=None)}}};
\\addlegendentry{{analytic 70--100\\% band}}
\\addplot+[color={BLUE}, only marks, mark=*, mark size=2.0pt, mark options={{fill={BLUE}}}] coordinates {{(1,{fnum(float(node_rows[0]["per_node_service_rate_reports_per_s"]))})}};
\\addlegendentry{{measured anchor}}
\\node[font=\\scriptsize, anchor=north east, text=black!70] at (axis cs:64,{fnum(float(node_rows[-1]["capacity_at_70pct_reports_per_s"]))}) {{20.8k/s at 70\\%}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}{{0.32\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{release_axis}
]
\\addplot+[{BAR_STYLES[0]}, bar shift=0pt] coordinates {{(1,{fnum(release_values[0])})}};
\\addplot+[{BAR_STYLES[1]}, bar shift=0pt] coordinates {{(3,{fnum(release_values[1])}) (6,{fnum(release_values[3])})}};
\\addplot+[{BAR_STYLES[2]}, bar shift=0pt] coordinates {{(4,{fnum(release_values[2])}) (7,{fnum(release_values[4])})}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}
\\caption{{Online hidden-write throughput and epoch-close cost. (a) Measured single-worker Google-DPF throughput as the slot domain grows. (b) Node capacity anchored at the measured eight-worker, single-host rate for $N=2^{{17}}$; the dashed line and shaded 70--100\\% band are analytic projections, not cluster measurements. (c) Same-host release microbenchmarks at $m=10^5$ and $M=2m$; ``stage-only'' is local cleartext work, whereas ``2-process'' adds local socket transfer and reconstruction.}}
\\Description{{Three panels show measured DPF write throughput versus slot domain, analytically projected multi-node capacity anchored at one host, and measured same-host epoch-close latency.}}
\\label{{fig:ingestion-cost-summary}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_ingestion_cost_summary.tex").write_text(body)


def ingestion_cost_summary_figure() -> None:
    """Build the main systems figure from measured kernels and scoped models."""
    methods = [PACKED_METHOD, TUPLE_METHOD, DUAL_METHOD]
    short = {
        PACKED_METHOD: "packed64",
        TUPLE_METHOD: "tuple",
        DUAL_METHOD: "dual-track",
    }
    backend_styles = [LINE_STYLES[0], LINE_STYLES[1], LINE_STYLES[2]]

    domain_rows = read_csv(DATA_DIR / "fig_dpf_throughput_vs_n.csv")
    if {row["source_type"] for row in domain_rows} != {"microbenchmark"}:
        raise ValueError("domain panel must contain measured microbenchmark rows")
    def throughput_error_coordinates(rows: list[dict]) -> str:
        selected = sorted(rows, key=lambda row: float(row["N"]))
        values = []
        for row in selected:
            rate = float(row["reports_per_second"])
            mean_ms = float(row["time_ms_per_report"])
            se_ms = float(row["standard_error_ms"])
            se_rate = rate * se_ms / mean_ms
            values.append(
                f"({fnum(float(row['N']))},{fnum(rate)}) +- (0,{fnum(se_rate)})"
            )
        return " ".join(values)

    domain_plots = []
    for index, method in enumerate(methods):
        subset = sorted(method_rows(domain_rows, method), key=lambda row: float(row["N"]))
        domain_plots.append(
            "\\addplot+["
            + backend_styles[index]
            + ", error bars/.cd, y dir=both, y explicit] coordinates {"
            + throughput_error_coordinates(subset)
            + "};\n\\addlegendentry{CPU "
            + short[method]
            + "}"
        )
    gpu_rows = sorted(
        read_csv(DATA_DIR / "fig_gpu_dpf_external.csv"),
        key=lambda row: float(row["N"]),
    )
    if not gpu_rows or any(
        row["source_type"] != "external_artifact_measurement"
        or row["directly_comparable_to_chafe"] != "false"
        or row["client_keygen_included"] != "false"
        or row["second_server_included"] != "false"
        or row["chafe_packed_write_accumulation_included"] != "false"
        for row in gpu_rows
    ):
        raise ValueError("GPU-DPF context must remain external and non-comparable")
    domain_plots.append(
        f"\\addplot+[color={SLATE}, dashdotted, line width=0.90pt, "
        "mark=diamond*, mark size=1.7pt, mark options={solid, fill=white, line width=0.65pt}] "
        f"coordinates {{{coordinates(gpu_rows, 'N', 'dpf_per_second', max_points=None)}}};\n"
        "\\addlegendentry{ext. GPU-DPF}"
    )
    domain_axis = axis_options(
        "0.73\\linewidth",
        "slot domain $N$",
        "throughput (/s)",
        (
            "height=2.48cm, title={(a) Backend Efficiency}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "xmode=log, ymode=log, log basis x=2, xmin=3000, xmax=1.3e6, ymax=4e5, "
            "legend style={font=\\tiny, draw=none, fill=white, fill opacity=0.88, "
            "text opacity=1, at={(0.02,0.02)}, anchor=south west, legend columns=2}"
        ),
    )

    thread_measurements = {
        int(float(row["threads"])): row
        for row in read_csv(DATA_DIR / "fig_dpf_thread_scaling.csv")
        if row["method"] == PACKED_METHOD and row["source_type"] == "microbenchmark"
    }
    thread_levels = [1, 2, 4, 8]
    if set(thread_measurements) != set(thread_levels):
        raise ValueError("joint scaling requires measured packed64 1/2/4/8-thread inputs")
    thread_colors = [SLATE, ORANGE, TEAL, BLUE]
    thread_marks = ["*", "square*", "triangle*", "diamond*"]
    projection_styles = [
        f"color={SLATE}, densely dotted, line width=0.86pt, no markers",
        f"color={ORANGE}, dashdotted, line width=0.86pt, no markers",
        f"color={TEAL}, dashed, line width=0.90pt, no markers",
        f"color={BLUE}, densely dashed, line width=0.96pt, no markers",
    ]
    joint_plots: list[str] = []
    measured_anchors: list[str] = []
    required_raw: dict[int, int] = {}
    node_equivalents = [1, 2, 4, 8, 16, 32, 64]
    for index, threads in enumerate(thread_levels):
        measured_rate = float(thread_measurements[threads]["reports_per_second"])
        measured_mean_ms = float(thread_measurements[threads]["time_ms_per_report"])
        measured_se_ms = float(thread_measurements[threads]["standard_error_ms"])
        measured_se_rate = measured_rate * measured_se_ms / measured_mean_ms
        required_raw[threads] = math.ceil(20_000.0 / measured_rate)
        projected_coordinates = " ".join(
            f"({equivalent},{fnum(equivalent * measured_rate)})"
            for equivalent in node_equivalents
        )
        joint_plots.append(
            "\\addplot+["
            + projection_styles[index]
            + "] coordinates {"
            + projected_coordinates
            + "};\n\\addlegendentry{"
            + str(threads)
            + " t proj.}"
        )
        measured_anchors.append(
            f"\\addplot+[only marks, color={thread_colors[index]}, "
            f"mark={thread_marks[index]}, mark size=1.65pt, mark options={{solid, fill="
            f"{thread_colors[index]}"
            ", line width=0.5pt}, forget plot, error bars/y dir=both, error bars/y explicit] coordinates {"
            f"(1,{fnum(measured_rate)}) +- (0,{fnum(measured_se_rate)})"
            "};"
        )
    joint_axis = axis_options(
        "0.73\\linewidth",
        "combined-work CPU equivalents",
        "capacity (/s)",
        (
            "height=2.48cm, title={(b) Threads + Ideal Capacity}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "xmode=log, ymode=log, log basis x=2, xmin=0.9, xmax=70, ymin=100, ymax=4e4, "
            "xtick={1,4,16,64}, xticklabels={1,4,16,64}, "
            "legend style={font=\\tiny, draw=none, fill=white, fill opacity=0.88, "
            "text opacity=1, at={(0.02,0.98)}, anchor=north west, legend columns=2}"
        ),
    )

    cgp_rows = read_csv(DATA_DIR / "fig_cgp_shuffle_model.csv")
    expected_cgp_sources = {
        "CGP full secret-shared shuffle (paper-derived)":
            "literature_derived_analytic_cost",
        "CGP payload-only ideal (author-derived)":
            "author_derived_analytic_scenario",
    }
    if not cgp_rows or any(
        expected_cgp_sources.get(row["method"]) != row["source_type"]
        or row["measured"] != "false"
        for row in cgp_rows
    ):
        raise ValueError("CGP model and ideal-scenario curves have invalid provenance")

    close_rows = read_csv(DATA_DIR / "fig_epoch_close_scaling.csv")
    if not close_rows or any(
        row["source_type"] != "microbenchmark"
        or row["correctness"] != "true"
        or row["deployment_scope"] != "same_host_two_processes_not_network_deployment"
        for row in close_rows
    ):
        raise ValueError("epoch-close panel needs validated Chafe close measurements")

    def seconds_coords(
        rows: list[dict], x_col: str, y_col: str, scale: float = 1.0
    ) -> str:
        selected = sorted(rows, key=lambda row: float(row[x_col]))
        return " ".join(
            f"({fnum(float(row[x_col]))},{fnum(scale * float(row[y_col]))})"
            for row in selected
        )

    cgp_full = [
        row for row in cgp_rows
        if row["method"] == "CGP full secret-shared shuffle (paper-derived)"
    ]
    cgp_payload_only = [
        row for row in cgp_rows
        if row["method"] == "CGP payload-only ideal (author-derived)"
    ]

    def chafe_close_coords(decoder: str) -> str:
        selected = sorted(
            (row for row in close_rows if row["decoder"] == decoder),
            key=lambda row: float(row["m"]),
        )
        return " ".join(
            f"({fnum(float(row['m']))},{fnum(1e-3 * float(row['time_ms']))}) "
            f"+- (0,{fnum(1e-3 * float(row['standard_error_ms']))})"
            for row in selected
        )

    barrier_axis = axis_options(
        "0.73\\linewidth",
        "epoch size $m$",
        "post-collection stage (s)",
        (
            "height=2.48cm, title={(c) Post-Collection Latency}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "xmode=log, ymode=log, xmin=800, xmax=1.25e6, ymin=2e-4, ymax=50, "
            "legend style={font=\\tiny, draw=none, fill=none, at={(0.02,0.98)}, anchor=north west}"
        ),
    )

    body = f"""% Auto-generated from measured Chafe, external GPU-DPF, and scoped model CSVs.
\\begin{{figure*}}[!t]
\\centering
\\begin{{minipage}}{{0.32\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{domain_axis}
]
{chr(10).join(domain_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}{{0.32\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{joint_axis}
]
{chr(10).join(joint_plots)}
{chr(10).join(measured_anchors)}
\\addplot+[black!50, solid, line width=0.55pt, no markers, forget plot] coordinates {{(1,20000) (70,20000)}};
\\node[font=\\tiny, anchor=north east, text=black!70] at (axis cs:68,20000) {{20k}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}{{0.315\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{barrier_axis}
]
\\addplot+[color={ORANGE}, solid, line width=1.0pt, mark=*, mark size=1.55pt, error bars/.cd, y dir=both, y explicit] coordinates {{{chafe_close_coords("drop")}}};
\\addlegendentry{{\\textsc{{Chafe-Drop}}}}
\\addplot+[color={TEAL}, solid, line width=1.0pt, mark=triangle*, mark size=1.75pt, error bars/.cd, y dir=both, y explicit] coordinates {{{chafe_close_coords("pool")}}};
\\addlegendentry{{\\textsc{{Chafe-Pool}}}}
\\addplot+[color=black!70, dashed, line width=0.95pt, mark=square*, mark size=1.45pt] coordinates {{{seconds_coords(cgp_full, "m", "cost_seconds")}}};
\\addlegendentry{{CGP model (1 Gb/s)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}
\\caption{{System-cost evidence. (a) Measured combined-work CPU kernels (error
bars: standard error); GPU-DPF is an external one-server kernel~\\cite{{lam2024gpudpf}}.
(b) Filled points at one equivalent are measured at $N=2^{{17}}$; all other
values are ideal linear projections, requiring {required_raw[8]} eight-thread
equivalents for 20k reports/s. (c) Measured same-host close versus the
literature-derived CGP AES-plus-1-Gb/s model~\\cite{{DBLP:conf/asiacrypt/ChaseGP20}}.}}
\\Description{{Three panels show measured DPF throughput, ideal combined-work
CPU-equivalent capacity projections from measured thread anchors, and measured
same-host epoch-close time against a literature-derived shuffle model.}}
\\label{{fig:ingestion-cost-summary}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_ingestion_cost_summary.tex").write_text(body)
    submission_figure_dir = (
        FIGURE_DIR.parent / "chafe-vldb-submission-source" / "figures"
    )
    submission_figure_dir.mkdir(parents=True, exist_ok=True)
    (submission_figure_dir / "fig_ingestion_cost_summary.tex").write_text(body)


def finite_burst_latency_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_finite_burst_latency.csv")
    expected_sources = {
        "Chafe CPU 8T (measured-rate model)": "analytic_cost_from_measured_cpu_inputs",
        "Chafe GPU-kernel proxy (interpolated)": "analytic_cost_from_external_gpu_kernel",
        "Chafe CPU 31-equivalent (ideal capacity)": "analytic_cost_from_measured_cpu_inputs",
        "CGP full model": "arrival_plus_literature_derived_analytic_cost",
    }
    if (
        len(rows) != 4
        or any(
            expected_sources.get(row["method"]) != row["source_type"]
            for row in rows
        )
        or {float(row["arrival_seconds"]) for row in rows} != {10.0}
    ):
        raise ValueError("finite-burst figure has invalid scope or provenance")

    by_method = {row["method"]: row for row in rows}
    cpu = by_method["Chafe CPU 8T (measured-rate model)"]
    gpu = by_method["Chafe GPU-kernel proxy (interpolated)"]
    cluster = by_method["Chafe CPU 31-equivalent (ideal capacity)"]
    cgp = by_method["CGP full model"]
    report_count = int(cpu["m"])
    arrival_end = float(cpu["arrival_seconds"])
    close_seconds = float(cpu["epoch_close_seconds"])
    arrival_rate = float(cpu["arrival_rate_reports_per_s"])
    cpu_rate = float(cpu["service_rate_reports_per_s"])
    gpu_rate = float(gpu["service_rate_reports_per_s"])
    cluster_rate = float(cluster["service_rate_reports_per_s"])
    cluster_equivalents = int(float(cluster["node_equivalents"]))
    cpu_total = float(cpu["total_seconds"])
    gpu_total = float(gpu["total_seconds"])
    cluster_total = float(cluster["total_seconds"])
    cgp_total = float(cgp["total_seconds"])
    crossover_rate = report_count / (cgp_total - close_seconds)

    capacities = [
        400,
        500,
        700,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        crossover_rate,
        8000,
        9000,
        arrival_rate,
        12000,
        15000,
        20000,
    ]
    capacity_curve = " ".join(
        f"({fnum(rate)},{fnum(max(arrival_end, report_count / rate) + close_seconds)})"
        for rate in capacities
    )

    burst_axis = axis_options(
        "0.62\\linewidth",
        "service rate $R_{\\rm svc}$",
        "completion (s)",
        (
            "height=2.55cm, xmode=log, ymode=log, "
            "xmin=400, xmax=2e4, ymin=8.5, ymax=300, "
            "xtick={500,2000,10000,20000}, "
            "xticklabels={0.5k,2k,10k,20k}, "
            "ytick={10,15,100,300}, yticklabels={10,15,100,300}, "
            "tick label style={font=\\tiny}, label style={font=\\tiny}, "
            "title={(a) Capacity boundary}, "
            "title style={font=\\scriptsize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "legend style={draw=none, fill=none}"
        ),
    )
    progress_axis = axis_options(
        "0.66\\linewidth",
        "reports received (\\%)",
        "work left (\\%)",
        (
            "height=2.55cm, xmin=0, xmax=100, ymin=0, ymax=105, "
            "xtick={0,50,95}, ytick={0,50,100}, "
            "tick label style={font=\\tiny}, label style={font=\\tiny}, "
            "title={(b) Work before the tail}, "
            "title style={font=\\scriptsize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "legend style={draw=none, fill=none}"
        ),
    )

    body = f"""% Auto-generated finite-burst capacity model; projected markers are labeled.
\\begin{{figure}}[t]
\\centering
\\begin{{minipage}}{{0.49\\linewidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{burst_axis}
]
% Chafe processes online; below the arrival rate, the finite queue drains later.
\\addplot+[color={BLUE}, solid, line width=1.15pt, no markers]
  coordinates {{{capacity_curve}}};
\\addplot+[color={ORANGE}, dashed, line width=1.05pt, no markers]
  coordinates {{(400,{fnum(cgp_total)}) (20000,{fnum(cgp_total)})}};

\\addplot+[black!45, densely dotted, line width=0.70pt, no markers, forget plot]
  coordinates {{({fnum(crossover_rate)},9) ({fnum(crossover_rate)},300)}};
\\node[font=\\tiny, rotate=90, anchor=south, text=black!60]
  at (axis cs:{fnum(crossover_rate)},55) {{7.25k}};
\\addplot+[black!45, densely dashed, line width=0.70pt, no markers, forget plot]
  coordinates {{({fnum(arrival_rate)},9) ({fnum(arrival_rate)},300)}};
\\node[font=\\tiny, rotate=90, anchor=south, text=black!60]
  at (axis cs:{fnum(arrival_rate)},55) {{10k arrival}};

\\addplot+[color={BLUE}, only marks, mark=*, mark size=1.8pt, forget plot]
  coordinates {{({fnum(cpu_rate)},{fnum(cpu_total)})}};
\\node[font=\\tiny, anchor=west]
  at (axis cs:{fnum(cpu_rate * 1.08)},{fnum(cpu_total)}) {{CPU}};
\\addplot+[color={TEAL}, only marks, mark=triangle*, mark size=2.1pt, forget plot]
  coordinates {{({fnum(gpu_rate)},{fnum(gpu_total)})}};
\\node[font=\\tiny, anchor=north east]
  at (axis cs:6900,12.0) {{GPU}};
\\addplot+[color={PURPLE}, only marks, mark=diamond*, mark size=2.0pt, forget plot]
  coordinates {{({fnum(cluster_rate)},{fnum(cluster_total)})}};
\\node[font=\\tiny, anchor=south east]
  at (axis cs:19500,10.25) {{{cluster_equivalents}$\\times$CPU}};
\\node[font=\\tiny, anchor=south east, text={ORANGE}]
  at (axis cs:19500,{fnum(cgp_total * 1.01)}) {{CGP}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}{{0.49\\linewidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{progress_axis}
]
\\addplot+[draw=none, fill=gray!10, forget plot]
  coordinates {{(95,0) (100,0) (100,105) (95,105)}} \\closedcycle;
\\addplot+[black!45, densely dotted, line width=0.70pt, no markers, forget plot]
  coordinates {{(95,0) (95,105)}};
\\node[font=\\tiny, rotate=90, anchor=south, text=black!60]
  at (axis cs:95,45) {{q95}};
\\addplot+[color={BLUE}, solid, line width=1.15pt, mark=*, mark size=1.45pt]
  coordinates {{(0,100) (25,75) (50,50) (75,25) (95,5) (100,0)}};
\\addplot+[color={ORANGE}, dashed, line width=1.05pt, mark=square*, mark size=1.35pt]
  coordinates {{(0,100) (50,100) (95,100) (100,100)}};
\\node[font=\\tiny, anchor=south west, text={BLUE}]
  at (axis cs:54,44) {{Chafe}};
\\node[font=\\tiny, anchor=north, text={ORANGE}]
  at (axis cs:55,99) {{batch shuffle}};
\\node[font=\\tiny, anchor=south east, text={BLUE}]
  at (axis cs:93,5) {{5\\%}};
\\node[font=\\tiny, anchor=north east, text={ORANGE}]
  at (axis cs:93,100) {{100\\%}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}
\\caption{{Finite-burst scenarios for $m=10^5$, $N=2^{{17}}$, $M=2m$, and
$\\lambda=10$k reports/s. (a) Completion versus service rate: measured CPU-rate
model, external GPU proxy, ideal {cluster_equivalents}$\\times$CPU projection,
and literature CGP model. No cluster was measured. (b) Work remaining.}}
\\Description{{The left panel shows the finite-burst capacity crossover. The
right panel shows remaining shuffle-layer work as reports arrive: Chafe falls
linearly to five percent at the ninety-fifth percentile, while collect-then-
shuffle retains all of its shuffle work until epoch close.}}
\\label{{fig:finite-burst-latency}}
\\end{{figure}}
"""
    (FIGURE_DIR / "fig_finite_burst_latency.tex").write_text(body)
    submission_figure_dir = (
        FIGURE_DIR.parent / "chafe-vldb-submission-source" / "figures"
    )
    submission_figure_dir.mkdir(parents=True, exist_ok=True)
    (submission_figure_dir / "fig_finite_burst_latency.tex").write_text(body)


def privacy_compact_summary_figure() -> None:
    methods = [
        "IdealShuffle",
        "IdealShuffle (same M padding)",
        "Drop (matched padding)",
        "Pool (replacement)",
    ]
    privacy_styles = [IDEAL_STYLE, SAME_M_STYLE, DROP_STYLE, POOL_STYLE]
    labels = [
        "Ideal shuffle ($m$)",
        "Padded ideal ($M$)",
        "Drop (matched)",
        "Pool (directed)",
    ]
    legend = inline_line_legend(
        [(label, privacy_styles[idx]) for idx, label in enumerate(labels)],
        4,
    )
    panels = []
    specs = [
        ("fig_privacy_vs_rho.csv", "rho", "load $\\rho=m/N$", "(a) Proxy vs. load", ""),
        ("fig_privacy_vs_tau.csv", "tau", "release ratio $\\tau=M/m$", "(b) Proxy vs. padding", ""),
    ]
    for panel_index, (csv_name, x_col, xlabel, title, extra) in enumerate(specs):
        rows = read_csv(DATA_DIR / csv_name)
        if {row["source_type"] for row in rows} != {"formula"}:
            raise ValueError(f"{csv_name}: proxy panels must be formula-derived")
        plots = []
        for idx, method in enumerate(methods):
            subset = method_rows(rows, method)
            if not subset:
                continue
            plots.append(
                "\\addplot+[" + privacy_styles[idx] + "] coordinates {"
                + coordinates(subset, x_col, "epsilon")
                + "};"
            )
        y_label = "$\\widetilde{\\epsilon}'$" if panel_index == 0 else ""
        y_tick = "" if panel_index == 0 else ", yticklabel=\\empty"
        axis = axis_options(
                "0.66\\linewidth",
            xlabel,
            y_label,
            (
                "height=2.78cm, title={" + title + "}, "
                "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
                "ymode=log, ymin=1e-4, ymax=1e-2" + y_tick + (", " + extra if extra else "")
            ),
        )
        panels.append(
            f"""\\begin{{minipage}}{{0.32\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{axis}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )
    exact_rows = [
        row
        for row in read_csv(DATA_DIR / "fig_exact_vs_proxy_drop.csv")
        if row["sweep"] == "rho"
    ]
    exact_law_rows = [
        row for row in exact_rows if row["method"] == "Exact-law numerical"
    ]
    proxy_rows = [
        row for row in exact_rows if row["method"] == "Second-order proxy"
    ]
    if {row["source_type"] for row in exact_law_rows} != {"exact_law_numerical"}:
        raise ValueError("exact/proxy panel must use the finite-law numerical rows")
    if {row["source_type"] for row in proxy_rows} != {"formula"}:
        raise ValueError("exact/proxy panel must use formula-derived proxy rows")
    exact_plots = [
        "\\addplot+["
        + EXACT_STYLE
        + "] coordinates {"
        + coordinates(exact_law_rows, "rho", "epsilon")
        + "};\n\\addlegendentry{finite-law numerical}",
        "\\addplot+["
        + PROXY_STYLE
        + "] coordinates {"
        + coordinates(proxy_rows, "rho", "epsilon")
        + "};\n\\addlegendentry{proxy}",
    ]
    exact_axis = axis_options(
        "0.72\\linewidth",
        "$\\rho=m/N$",
        "pairwise $\\epsilon'$",
        (
            "height=2.78cm, title={(c) Exact vs. proxy}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "ymode=log, ymin=1e-4, ymax=1e-2, "
            "legend style={font=\\scriptsize, draw=none, fill=none, at={(0.97,0.97)}, anchor=north east}"
        ),
    )
    panels.append(
        f"""\\begin{{minipage}}{{0.32\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{exact_axis}
]
{chr(10).join(exact_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
    )
    joined_panels = "\n\\hfill\n".join(panels)
    body = f"""% Auto-generated from formula-derived privacy CSVs.
\\begin{{figure*}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
{joined_panels}
\\caption{{Privacy design curves. (a,b) Quadratic proxy $\\widetilde{{\\epsilon}}'$ at $m=10^6$, $\\epsilon_0=1$, and $\\delta=10^{{-6}}$: (a) varies $\\rho$ at $\\tau=2$; (b) varies $\\tau$ at $\\rho=1$. (c) Absolute pairwise $\\epsilon'$ from finite-law numerical inversion and the proxy for one homogeneous binary add/remove pair under matched Drop at $m=10^5$ and $\\tau=2$. Pool is a directed diagnostic, not a two-sided certificate.}}
\\Description{{Three logarithmic plots show privacy proxy trends with slot load and release padding, followed by finite-law numerical and proxy pairwise epsilon values across slot load.}}
\\label{{fig:privacy-compact-summary}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_privacy_compact_summary.tex").write_text(body)


def privacy_2x2_validation_figure() -> None:
    """Validate all four privacy cases and retain both design knobs compactly."""
    rows = read_csv(DATA_DIR / "fig_privacy_2x2_moderate_scaling.csv")
    tau_rows = read_csv(DATA_DIR / "fig_privacy_2x2_tau_exact.csv")
    panel_specs = [
        ("drop_replacement", "(a) \\textsc{Drop}: replacement"),
        ("drop_add_remove", "(b) \\textsc{Drop}: add/remove"),
        ("pool_replacement", "(c) \\textsc{Pool}: replacement"),
        ("pool_add_remove", "(d) \\textsc{Pool}: add/remove"),
    ]
    if len(rows) != 40 or {row["panel"] for row in rows} != {
        panel for panel, _ in panel_specs
    }:
        raise ValueError("moderate 2x2 validation grid is incomplete")
    if any(
        row["source_type"] != "finite_vs_poisson_exact_law_scaling"
        or row["quadratic_epsilon_status"] != "reference_prediction_not_DP_bound"
        or int(row["N"]) > int(row["M"])
        for row in rows
    ):
        raise ValueError("moderate 2x2 validation has invalid evidence scope")

    top_legend = inline_line_legend(
        [
            ("finite-law numerical $\\epsilon'$", EPS_EXACT_STYLE),
            ("Poisson-surrogate $\\epsilon'$", ETA_POISSON_STYLE),
            ("quadratic $\\widehat{\\epsilon}'$", EPS_QUAD_STYLE),
        ],
        3,
    )

    columns: list[str] = []
    for panel_index, (panel_name, title) in enumerate(panel_specs):
        panel_rows = [row for row in rows if row["panel"] == panel_name]
        load_rows = [
            row for row in panel_rows if int(row["background_size_common"]) == 32
        ]
        if len(load_rows) != 3:
            raise ValueError(f"{panel_name}: expected three load and four scale anchors")

        axis_width = "0.66\\linewidth" if panel_index == 0 else "0.78\\linewidth"
        top_ylabel = "$\\epsilon'$" if panel_index == 0 else ""
        hide_y = "" if panel_index == 0 else ", yticklabel=\\empty"
        top_axis = axis_options(
            axis_width,
            "load $\\rho$",
            top_ylabel,
            (
                "height=1.95cm, title={" + title + "}, "
                "title style={font=\\scriptsize\\bfseries, at={(0,1.03)}, anchor=south west}, "
                "ymode=log, ymin=0.015, ymax=0.50, "
                "ytick={0.02,0.1,0.4}, yticklabels={.02,.1,.4}, "
                "xmin=0.45, xmax=2.05, xtick={0.5,1,2}, "
                "legend style={draw=none}" + hide_y
            ),
        )
        top_plots = [
            "\\addplot+["
            + EPS_EXACT_STYLE
            + "] coordinates {"
            + coordinates(
                load_rows,
                "rho_common_background",
                "epsilon_finite_exact_two_sided",
                None,
            )
            + "};",
            "\\addplot+["
            + ETA_POISSON_STYLE
            + "] coordinates {"
            + coordinates(
                load_rows,
                "rho_common_background",
                "epsilon_poisson_exact_two_sided",
                None,
            )
            + "};",
            "\\addplot+["
            + EPS_QUAD_STYLE
            + "] coordinates {"
            + coordinates(
                load_rows,
                "rho_common_background",
                "epsilon_reference_quadratic",
                None,
            )
            + "};",
        ]
        columns.append(
            f"""\\begin{{minipage}}[t]{{0.238\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{top_axis}
]
{chr(10).join(top_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )

    joined_columns = "\n\\hfill\n".join(columns)
    case_styles = [
        ("drop_replacement", "\\textsc{Drop} rep.", LINE_STYLES[0]),
        ("drop_add_remove", "\\textsc{Drop} add/rem.", LINE_STYLES[1]),
        ("pool_replacement", "\\textsc{Pool} rep.", LINE_STYLES[2]),
        ("pool_add_remove", "\\textsc{Pool} add/rem.", LINE_STYLES[3]),
    ]
    convergence_plots = [
        "\\addplot+[gray!55, densely dotted, line width=0.7pt, mark=none, forget plot] "
        "coordinates {(8,1) (64,1)};"
    ]
    for panel_name, label, style in case_styles:
        scale_rows = [
            row
            for row in rows
            if row["panel"] == panel_name
            and abs(float(row["rho_common_background"]) - 1.0) < 1e-12
        ]
        if len(scale_rows) != 4:
            raise ValueError(f"{panel_name}: expected four scale anchors")
        convergence_plots.append(
            "\\addplot+["
            + style
            + "] coordinates {"
            + coordinates(
                scale_rows,
                "background_size_common",
                "epsilon_poisson_exact_over_finite_exact",
                None,
            )
            + "};"
        )

    convergence_axis = axis_options(
        "0.86\\linewidth",
        "common reports $m_B$",
        "$\\epsilon'_{\\rm poi}/\\epsilon'_{\\rm fin}$",
        (
            "height=1.85cm, title={(e) Poissonization error at $\\rho=1$}, "
            "title style={font=\\scriptsize\\bfseries, at={(0,1.03)}, anchor=south west}, "
            "ymin=0.985, ymax=1.155, ytick={1,1.05,1.10,1.15}, "
            "xmode=log, log basis x=2, xmin=7.2, xmax=69, "
            "xtick={8,16,32,64}, xticklabels={8,16,32,64}, "
            "legend style={draw=none}"
        ),
    )

    tau_plots: list[str] = [
        "\\addplot+[gray!65, densely dotted, line width=0.9pt, mark=none] "
        "coordinates {(1,1) (2,0.70710678) (4,0.5) (8,0.35355339)};\n"
        "\\addlegendentry{$1/\\sqrt\\tau$ design law}"
    ]
    for panel_name, label, style in case_styles:
        method_rows = [row for row in tau_rows if row["panel"] == panel_name]
        if len(method_rows) != 4:
            raise ValueError(f"missing exact tau sweep for {panel_name}")
        tau_plots.append(
            "\\addplot+["
            + style
            + "] coordinates {"
            + coordinates(method_rows, "tau_P", "epsilon_finite_over_tau1", None)
            + "};\n\\addlegendentry{" + label + "}"
        )
    tau_axis = axis_options(
        "0.86\\linewidth",
        "release ratio $\\tau=M/(m_B+1)$",
        "$\\epsilon'_{\\rm fin}(\\tau)/\\epsilon'_{\\rm fin}(1)$",
        (
            "height=1.85cm, title={(f) Finite-law padding at $\\rho=1$}, "
            "title style={font=\\scriptsize\\bfseries, at={(0,1.03)}, anchor=south west}, "
            "ymin=0.23, ymax=1.05, ytick={0.25,0.5,0.75,1}, "
            "xmin=0.9, xmax=8.1, xtick={1,2,4,8}, "
            "legend columns=3, legend style={font=\\tiny, draw=none, fill=white, "
            "fill opacity=0.88, text opacity=1, at={(0.98,0.98)}, "
            "anchor=north east, column sep=2pt, row sep=-1pt}"
        ),
    )
    body = f"""% Auto-generated from finite and Poisson exact-law scaling rows.
\\begin{{figure*}}[t]
\\centering
{top_legend}\\par\\vspace{{0.3mm}}
{joined_columns}
\\par\\vspace{{0.4mm}}
\\begin{{minipage}}[t]{{0.49\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{convergence_axis}
]
{chr(10).join(convergence_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}[t]{{0.49\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{tau_axis}
]
{chr(10).join(tau_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}
\\caption{{Accounting validation and design knobs.  Panels (a--e) use
homogeneous BRR ($f_0=(3/4,1/4)$, $f_1=(1/4,3/4)$,
$\\epsilon_0=\\log3$, $\\delta=10^{{-3}}$); \\textsc{{Drop}} uses
$\\nu=f_0$ and \\textsc{{Pool}} uses a fixed public Poisson-matched law.
(a--d) Compare finite-law, one-slot Poisson, and quadratic values at $m_B=32$
and $M=2(m_B+1)\\ge N$.  (e) Shows Poisson-to-finite convergence for all four
cases.  (f) Compares finite-law dilution with the $1/\\sqrt\\tau$ design law.
Matched \\textsc{{Drop}} makes its two adjacency curves
coincide; all profiles are pair-specific, and quadratic curves are not DP
certificates.}}
\\Description{{Six privacy plots.  Four compare finite, Poisson, and quadratic
epsilon across load for Drop and Pool under replacement and add/remove.  One
shows Poisson-to-finite convergence, and one shows finite-law padding dilution.}}
\\label{{fig:privacy-2x2-validation}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_privacy_2x2_validation.tex").write_text(body)


def privacy_large_m_accounting_figure() -> None:
    """Compact exact/profile/scaling summary for the main evaluation."""

    rows = read_csv(DATA_DIR / "fig_privacy_large_m_accounting.csv")
    # The two Drop relations coincide for this matched homogeneous BRR pair,
    # so plotting both would hide one curve without adding information.
    case_specs = [
        ("drop_replacement", "\\textsc{Drop} (both)", BLUE),
        ("pool_replacement", "\\textsc{Pool} rep.", TEAL),
        ("pool_add_remove", "\\textsc{Pool} add/rem.", PURPLE),
    ]
    main_rows = [row for row in rows if row["sweep"] == "m_scaling"]
    tau_rows = [row for row in rows if row["sweep"] == "tau_sensitivity"]
    if len(main_rows) != 196 or len(tau_rows) != 16:
        raise ValueError("large-m privacy accounting grid is incomplete")

    def exact_style(color: str, *, marks: bool = False) -> str:
        marker = ", mark=*, mark size=1.35pt" if marks else ", no markers"
        return f"color={color}, solid, line width=0.98pt{marker}"

    def mechanism_wide_style(color: str) -> str:
        return (
            f"color={color}, densely dashdotted, line width=0.98pt, "
            "mark=square*, mark size=1.15pt, mark options={fill=white}"
        )

    def law_style(color: str) -> str:
        return f"color={color}, densely dashed, line width=0.88pt, no markers"

    def transformed_coordinates(selected, x_col: str, transform) -> str:
        values = sorted((float(row[x_col]), float(transform(row))) for row in selected)
        return " ".join(f"({fnum(x)},{fnum(y)})" for x, y in values)

    case_legend = inline_line_legend(
        [(label, exact_style(color)) for _, label, color in case_specs], 3
    )
    evidence_legend = inline_line_legend(
        [
            ("matched-pair exact", f"color={INK}, solid, line width=0.98pt, mark=*, mark size=1.35pt"),
            ("mechanism-wide slot bound", f"color={INK}, densely dashdotted, line width=0.98pt, mark=square*, mark size=1.15pt, mark options={{fill=white}}"),
            ("$L_2$ separation", f"color={INK}, densely dashed, line width=0.88pt"),
        ],
        3,
    )

    rho_one = [
        row for row in main_rows
        if abs(float(row["rho_requested_m_over_N"]) - 1.0) < 1e-12
    ]
    scaling_plots: list[str] = []
    fidelity_plots: list[str] = []
    load_plots: list[str] = []
    padding_plots: list[str] = []
    load_law_columns = {
        "drop_replacement": "closed_form_drop_over_ideal_M",
        "pool_replacement": "closed_form_pool_replacement_over_ideal_M",
        "pool_add_remove": "closed_form_pool_add_remove_over_ideal_M",
    }
    total_law_columns = {
        "drop_replacement": "closed_form_drop_over_ideal_m",
        "pool_replacement": "closed_form_pool_replacement_over_ideal_m",
        "pool_add_remove": "closed_form_pool_add_remove_over_ideal_m",
    }
    for panel_name, _label, color in case_specs:
        selected = [row for row in rho_one if row["panel"] == panel_name]
        scaling_plots.append(
            "\\addplot+[" + exact_style(color) + "] coordinates {"
            + coordinates(selected, "m_public_larger_epoch", "epsilon_poisson_exact_two_sided", None)
            + "};"
        )
        # The slot envelope is decoder-wide and therefore identical for Pool
        # replacement and Pool add/remove.  Draw it once per decoder instead
        # of hiding one duplicate curve behind another.
        if panel_name != "pool_add_remove":
            scaling_plots.append(
                "\\addplot+[" + mechanism_wide_style(color) + "] coordinates {"
                + coordinates(selected, "m_public_larger_epoch", "epsilon_poisson_mechanism_wide_upper", None)
                + "};"
            )
        fidelity_plots.extend(
            [
                "\\addplot+[" + mechanism_wide_style(color) + "] coordinates {"
                + transformed_coordinates(
                    selected,
                    "m_public_larger_epoch",
                    lambda row: float(row["epsilon_poisson_mechanism_wide_upper"])
                    / float(row["epsilon_poisson_exact_two_sided"]),
                )
                + "};",
            ]
        )

        load_selected = [
            row for row in main_rows
            if row["panel"] == panel_name
            and int(row["m_public_larger_epoch"]) == 100_000
        ]
        load_plots.extend(
            [
                "\\addplot+[" + exact_style(color, marks=True) + "] coordinates {"
                + coordinates(load_selected, "rho_requested_m_over_N", "epsilon_poisson_over_ideal_release_M", None)
                + "};",
                "\\addplot+[" + law_style(color) + "] coordinates {"
                + coordinates(load_selected, "rho_requested_m_over_N", load_law_columns[panel_name], None)
                + "};",
            ]
        )

        tau_selected = [row for row in tau_rows if row["panel"] == panel_name]
        padding_plots.extend(
            [
                "\\addplot+[" + exact_style(color, marks=True) + "] coordinates {"
                + coordinates(tau_selected, "tau_M_over_m", "epsilon_poisson_over_ideal_release_m", None)
                + "};",
                "\\addplot+[" + law_style(color) + "] coordinates {"
                + coordinates(tau_selected, "tau_M_over_m", total_law_columns[panel_name], None)
                + "};",
            ]
        )

    fidelity_plots.insert(
        0,
        "\\addplot+[black!35, densely dotted, no markers, forget plot] "
        "coordinates {(1e3,1) (1e6,1)};",
    )

    panel_specs = [
        (
            "(a) Absolute budget",
            "epoch size $m$",
            "$\\epsilon'$",
            "xmode=log, ymode=log, xmin=850, xmax=1.2e6, "
            "xtick={1e3,1e4,1e5,1e6}, xticklabels={$10^3$,$10^4$,$10^5$,$10^6$}, "
            "ymin=4e-4, ymax=9e-2",
            scaling_plots,
        ),
        (
            "(b) Uniform / matched",
            "epoch size $m$",
            "ratio",
            "xmode=log, xmin=850, xmax=1.2e6, "
            "xtick={1e3,1e4,1e5,1e6}, xticklabels={$10^3$,$10^4$,$10^5$,$10^6$}, "
            "ymin=0.95, ymax=3.2, ytick={1,1.5,2,3}",
            fidelity_plots,
        ),
        (
            "(c) Varying load $\\rho$",
            "load $\\rho$",
            "$\\varepsilon'/\\varepsilon'_{\\rm ideal}(M)$",
            "xmin=0.45, xmax=2.05, xtick={0.5,1,2}, "
            "ymode=log, ymin=0.075, ymax=1.08, ytick={0.1,0.2,0.5,1}, "
            "yticklabels={.1,.2,.5,1}",
            load_plots,
        ),
        (
            "(d) Varying release $\\tau$",
            "release ratio $\\tau$",
            "$\\varepsilon'/\\varepsilon'_{\\rm ideal}(m)$",
            "xmode=log, log basis x=2, xmin=0.9, xmax=8.5, xtick={1,2,4,8}, "
            "ymode=log, ymin=0.07, ymax=1.08, ytick={0.1,0.2,0.5,1}, "
            "yticklabels={.1,.2,.5,1}",
            padding_plots,
        ),
    ]

    panels: list[str] = []
    for title, xlabel, ylabel, extra, plots in panel_specs:
        options = axis_options(
            "0.78\\linewidth",
            xlabel,
            ylabel,
            "height=2.05cm, tick label style={font=\\scriptsize}, "
            "label style={font=\\scriptsize}, title={" + title + "}, "
            "title style={font=\\scriptsize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "legend style={draw=none}, " + extra,
        )
        panels.append(
            "\\begin{minipage}{0.242\\textwidth}\n\\centering\n"
            "\\makebox[\\linewidth][c]{%\n"
            "\\begin{tikzpicture}\n\\begin{axis}[\n" + options + "\n]\n"
            + "\n".join(plots)
            + "\n\\end{axis}\n\\end{tikzpicture}%\n}\n\\end{minipage}"
        )

    body = f"""% Auto-generated from the scalable binary one-slot accountant.
\\begin{{figure*}}[!t]
\\centering
{case_legend}\\par\\vspace{{0.05mm}}
{evidence_legend}\\par\\vspace{{0.35mm}}
{panels[0]}\\hfill{panels[1]}\\hfill{panels[2]}\\hfill{panels[3]}
\\caption{{Privacy accounting for BRR
($\\varepsilon_0=1$, $\\delta=10^{{-6}}$). (a) Exact matched-pair profiles
and two decoder-wide Poisson slot-envelope bounds at $\\rho=1$, $\\tau=2$;
the two \\textsc{{Drop}} relations coincide. (b) Tightness cost of uniformity.
(c) Matched-pair budget
relative to the same-$M$ padded ideal as $\\rho$ varies ($m=10^5$, $\\tau=2$).
(d) Budget relative to the length-$m$ ideal as $\\tau$ varies ($m=10^5$,
$\\rho=1$). Dashed curves in (c--d) normalize the matched $L_2$-separation
laws and are not DP budgets.}}
\\Description{{Four compact panels compare mechanism-wide Poisson upper
bounds with exact matched-pair profiles, then show how collision load and
release expansion change matched release-law separation.}}
\\label{{fig:privacy-large-m-accounting}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_privacy_large_m_accounting.tex").write_text(body)

def symbolic_coordinates(rows: list[dict[str, str]], method: str, task: str, datasets: list[str]) -> str:
    by_dataset = {
        row["dataset_panel"]: float(row["value"])
        for row in rows
        if row["task"] == task and row["method"] == method
    }
    return " ".join(
        f"({dataset},{fnum(by_dataset[dataset])})"
        for dataset in datasets
        if dataset in by_dataset
    )


def utility_cross_dataset_summary_figure() -> None:
    rows = read_csv(DATA_DIR / "fig_utility_cross_dataset_summary.csv")
    datasets = ["Synthetic", "Adult", "Covertype"]
    methods = ["IdealShuffle", "Drop", "Pool"]
    method_labels = {"IdealShuffle": "Ideal", "Drop": "Drop", "Pool": "Pool"}
    legend = inline_bar_legend(
        [(method_labels[method], BAR_STYLES[idx % len(BAR_STYLES)]) for idx, method in enumerate(methods)],
        3,
    )
    panels = []
    for task, title, ylabel, extra in [
        ("Range", "Range ($\\widetilde{\\epsilon}'=0.16$)", "rel. error", "ymode=log"),
        ("Sum", "Sum ($\\widetilde{\\epsilon}'=0.16$)", "rel. error", "ymode=log"),
        ("Heavy hitter", "HH ($\\widetilde{\\epsilon}'=0.32$)", "top-50 F1", "ymin=0, ymax=1.04"),
    ]:
        plots = []
        for idx, method in enumerate(methods):
            coords = symbolic_coordinates(rows, method, task, datasets)
            if not coords:
                continue
            plots.append(
                "\\addplot+[" + BAR_STYLES[idx % len(BAR_STYLES)] + "] coordinates {"
                + coords
                + "};"
            )
        axis = axis_options(
            "0.75\\linewidth",
            "",
            ylabel,
            (
                "height=2.72cm, tick label style={font=\\tiny}, label style={font=\\scriptsize}, "
                "title={" + title + "}, title style={font=\\scriptsize, yshift=-0.4ex}, "
                "ybar, bar width=4.5pt, symbolic x coords={Synthetic,Adult,Covertype}, "
                "xtick=data, enlarge x limits=0.18, " + extra
            ),
        )
        panels.append(
            f"""\\begin{{minipage}}{{0.323\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{axis}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )
    body = f"""% Auto-generated from data/fig_utility_cross_dataset_summary.csv.
\\begin{{figure*}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
{chr(10).join(panels)}
\\caption{{Supplementary collision-only Monte Carlo on decoded outputs before padding. Each panel fixes a quadratic proxy target $\\widetilde{{\\epsilon}}'$ and compares synthetic data, an Adult bootstrap, and Covertype sampled without replacement. Range and heavy-hitter tasks use GRR; sums use an additive Laplace model. These are parameter-sensitivity comparisons, not full-release or certified equal-privacy results.}}
\\label{{fig:utility-cross-dataset-summary}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_utility_cross_dataset_summary.tex").write_text(body)


def _legacy_utility_full_release_summary_figure() -> None:
    workload_rows = read_csv(DATA_DIR / "fig_full_release_real_workloads.csv")
    if any(
        row["source_type"] != "monte_carlo"
        or row["full_release"] != "true"
        or row["dummy_labels_visible"] != "false"
        or row["estimator_observes_real_output_count"] != "false"
        or row["central_epsilon_claimed"] != "false"
        for row in workload_rows
    ):
        raise ValueError(
            "workload panels require unlabeled full-release Monte Carlo rows "
            "without a central-DP claim"
        )
    datasets = ["Synthetic", "Adult", "Covertype"]
    expected_workload_rows = {
        (dataset, task, method)
        for dataset in datasets
        for task, methods in [
            ("Range", ["Ideal", "Drop"]),
            ("Heavy hitter", ["Ideal", "Drop"]),
            ("Sum", ["Ideal", "Pool"]),
        ]
        for method in methods
    }
    observed_workload_rows = {
        (row["dataset"], row["task"], row["method"])
        for row in workload_rows
    }
    if observed_workload_rows != expected_workload_rows:
        raise ValueError(
            "full-release utility grid is incomplete: "
            f"missing={sorted(expected_workload_rows - observed_workload_rows)}, "
            f"extra={sorted(observed_workload_rows - expected_workload_rows)}"
        )
    exact_family_rows = [
        row
        for row in read_csv(DATA_DIR / "fig_large_exact_family_utility.csv")
        if row["predicate"] == "one_positive_count"
    ]
    if not exact_family_rows or any(
        row["pairwise_exact_profile_only"] != "true"
        or row["mechanism_wide_dp_claimed"] != "false"
        or row["full_release"] != "true"
        for row in exact_family_rows
    ):
        raise ValueError("exact utility panel must remain scoped to pairwise full-release laws")
    legend = inline_bar_legend(
        [
            ("Padded ideal", BAR_STYLES[0]),
            ("Drop", BAR_STYLES[1]),
            ("Pool", BAR_STYLES[2]),
        ],
        3,
    )

    def workload_coords(task: str, method: str, scale: float) -> str:
        selected = {
            row["dataset"]: row
            for row in workload_rows
            if row["task"] == task and row["method"] == method
        }
        return " ".join(
            f"({dataset},{fnum(scale * float(selected[dataset]['value']))}) "
            f"+- (0,{fnum(scale * float(selected[dataset]['standard_error']))})"
            for dataset in datasets
        )

    panels = []
    for task, title, ylabel, scale, methods, extra in [
        (
            "Range",
            "(a) Range ($\\epsilon_0=5$)",
            "relative error (\\%)",
            100.0,
            ["Ideal", "Drop"],
            "ymode=log",
        ),
        (
            "Heavy hitter",
            "(b) Top-50 heavy hitter ($\\epsilon_0=5$)",
            "F1",
            1.0,
            ["Ideal", "Drop"],
            "ymin=0, ymax=1.04",
        ),
        (
            "Sum",
            "(c) Bounded sum ($\\epsilon_0=1$)",
            "relative error (\\%)",
            100.0,
            ["Ideal", "Pool"],
            "ymin=0",
        ),
    ]:
        plots = []
        style_index = {"Ideal": 0, "Drop": 1, "Pool": 2}
        for method in methods:
            plots.append(
                "\\addplot+["
                + BAR_STYLES[style_index[method]]
                + ", error bars/.cd, y dir=both, y explicit, "
                + "error bar style={line width=0.45pt}] coordinates {"
                + workload_coords(task, method, scale)
                + "};"
            )
        axis = axis_options(
            "0.78\\linewidth",
            "",
            ylabel,
            (
                "height=2.52cm, title={" + title + "}, "
                "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, ybar, "
                "bar width=5.0pt, symbolic x coords={Synthetic,Adult,Covertype}, "
                "xtick=data, xticklabels={Synthetic,Adult,Covertype}, "
                "enlarge x limits=0.18, " + extra
            ),
        )
        panels.append(
            f"""\\begin{{minipage}}{{0.485\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{axis}
]
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
        )

    ideal_exact_mae = {
        float(row["exact_target_epsilon_prime"]): float(
            row["exact_law_mean_absolute_count_error"]
        )
        for row in exact_family_rows
        if row["method"] == "Padded Ideal"
    }
    rho_styles = [
        f"color={ORANGE}, solid, line width=1.0pt, mark=*, mark size=1.7pt",
        f"color={ORANGE}, dashed, line width=1.0pt, mark=square*, mark size=1.7pt, mark options={{fill=white}}",
        f"color={ORANGE}, dotted, line width=1.08pt, mark=triangle*, mark size=1.85pt, mark options={{fill=white}}",
    ]
    exact_utility_plots = []
    for style, rho in zip(rho_styles, [0.5, 1.0, 2.0]):
        selected = sorted(
            (
                float(row["exact_target_epsilon_prime"]),
                float(row["exact_law_mean_absolute_count_error"]),
            )
            for row in exact_family_rows
            if row["method"] == "Drop" and float(row["rho"]) == rho
        )
        coords = " ".join(
            f"({fnum(epsilon)},{fnum(mae / ideal_exact_mae[epsilon])})"
            for epsilon, mae in selected
        )
        exact_utility_plots.append(
            "\\addplot+["
            + style
            + "] coordinates {"
            + coords
            + "};\n\\addlegendentry{$\\rho="
            + fnum(rho)
            + "$}"
        )
    exact_utility_axis = axis_options(
        "0.78\\linewidth",
        "pairwise $\\epsilon'$",
        "Drop / ideal MAE",
        (
            "height=2.52cm, title={(d) Pairwise exact crossover}, "
            "title style={font=\\footnotesize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "xmin=0.08, xmax=0.72, xtick={0.1,0.3,0.7}, ymin=0.75, ymax=1.04, "
            "ytick={0.8,0.9,1.0}, "
            "legend style={font=\\scriptsize, draw=none, fill=none, at={(0.02,0.05)}, anchor=south west}"
        ),
    )
    exact_panel = (
        f"""\\begin{{minipage}}{{0.485\\textwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{exact_utility_axis}
]
{chr(10).join(exact_utility_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}"""
    )

    body = f"""% Auto-generated from full-release utility CSVs.
\\begin{{figure*}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
{panels[0]}\\hfill
{panels[1]}\\par\\vspace{{1.0mm}}
{panels[2]}\\hfill
{exact_panel}
\\caption{{Complete-release utility. (a--c) Synthetic, Adult, and Covertype at a shared local budget per task, with $m=N=40{{,}}000$ and $M=80{{,}}000$; the padded ideal and Chafe estimators receive the same public length, and bars show means and standard errors over 100 runs. (d) Exact-law Drop/ideal MAE at $D_1=(1,0^{{m-1}})$ after two-sided calibration over $(D_0=0^m,D_1)$ to the same pairwise $\\epsilon'$ ($m=10^5$, $M=2m$, $\\delta=10^{{-6}}$). Panel (d)'s pairwise guarantee does not apply to (a--c).}}
\\Description{{Four panels compare padded ideal with Drop on range and heavy-hitter tasks, padded ideal with Pool on bounded sums, and the exact-calibrated Drop-to-ideal error ratio for one declared neighboring pair.}}
\\label{{fig:utility-full-release-summary}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_utility_full_release_summary.tex").write_text(body)


def _legacy_proxy_utility_full_release_summary_figure() -> None:
    """Build absolute, relative, exact-pair, and additive utility panels."""
    proxy_rows = read_csv(DATA_DIR / "fig_full_release_proxy_utility.csv")
    if not proxy_rows or any(
        row["source_type"] != "monte_carlo_complete_release"
        or row["full_release"] != "true"
        or row["central_epsilon_claimed"] != "false"
        or row["mechanism_wide_dp_claimed"] != "false"
        for row in proxy_rows
    ):
        raise ValueError("proxy panels require complete releases without central-DP claims")
    for row in proxy_rows:
        if row["method"] == "Pure IdealShuffle":
            if (
                row["privacy_target_type"] != "pure_ideal_shuffle_quadratic_proxy"
                or row["uses_dummies"] != "false"
                or row["dummy_law"] != "none"
                or row["dummy_law_matches_proxy_reference"] != "not_applicable"
            ):
                raise ValueError("pure IdealShuffle rows must be unpadded proxy references")
        elif row["method"] in {"Padded Ideal", "Drop"}:
            if (
                row["privacy_target_type"] != "matched_public_prior_quadratic_proxy"
                or row["uses_dummies"] != "true"
                or row["dummy_law_matches_proxy_reference"] != "true"
                or row["dummy_labels_visible"] != "false"
                or (
                    row["method"] == "Drop"
                    and row["estimator_observes_real_output_count"] != "false"
                )
            ):
                raise ValueError("padded proxy rows require matched, unlabeled releases")
        else:
            raise ValueError(f"unexpected proxy utility method: {row['method']}")

    datasets = ["Synthetic", "Adult", "Covertype"]
    targets = [0.08, 0.16, 0.32]
    expected_proxy = {
        (dataset, task, method, target)
        for dataset in datasets
        for task in ["Range", "Heavy hitter"]
        for method in ["Pure IdealShuffle", "Padded Ideal", "Drop"]
        for target in targets
    }
    observed_proxy = {
        (
            row["dataset"],
            row["task"],
            row["method"],
            float(row["target_tilde_epsilon_prime"]),
        )
        for row in proxy_rows
    }
    if observed_proxy != expected_proxy:
        raise ValueError("proxy utility grid is incomplete")

    dataset_specs = [
        ("Synthetic", BLUE, "*", "1.70pt"),
        ("Adult", ORANGE, "square*", "1.65pt"),
        ("Covertype", TEAL, "triangle*", "1.80pt"),
    ]

    def dataset_style(color: str, marker: str, size: str) -> str:
        return (
            f"color={color}, solid, line width=1.00pt, mark={marker}, "
            f"mark size={size}, mark options={{solid, fill={color}, line width=0.55pt}}"
        )

    def method_style(color: str, marker: str, size: str, method: str) -> str:
        if method == "Drop":
            return dataset_style(color, marker, size)
        if method == "Padded Ideal":
            return (
                f"color={color}, dashed, line width=0.96pt, mark={marker}, "
                f"mark size={size}, mark options={{solid, fill=white, line width=0.65pt}}"
            )
        raise ValueError(f"unsupported absolute-utility method: {method}")

    shared_legend = inline_line_legend(
        [
            (dataset, dataset_style(color, marker, size))
            for dataset, color, marker, size in dataset_specs
        ]
        + [
            ("Drop", "color=black!70, solid, line width=1.00pt, no markers"),
            ("Padded ideal", "color=black!70, dashed, line width=0.96pt, no markers"),
        ],
        5,
    )

    absolute_range_plots: list[str] = []
    absolute_hh_plots: list[str] = []
    range_gain_plots: list[str] = []
    hh_gain_plots: list[str] = []
    for dataset, color, marker, size in dataset_specs:
        for task, output in [
            ("Range", absolute_range_plots),
            ("Heavy hitter", absolute_hh_plots),
        ]:
            for method in ["Padded Ideal", "Drop"]:
                selected = sorted(
                    (
                        float(row["target_tilde_epsilon_prime"]),
                        float(row["value"]),
                        float(row["standard_error"]),
                    )
                    for row in proxy_rows
                    if row["dataset"] == dataset
                    and row["task"] == task
                    and row["method"] == method
                )
                output.append(
                    "\\addplot+["
                    + method_style(color, marker, size, method)
                    + ", error bars/.cd, y dir=both, y explicit] coordinates {"
                    + " ".join(
                        f"({fnum(x)},{fnum(y)}) +- (0,{fnum(se)})"
                        for x, y, se in selected
                    )
                    + "};"
                )

        relative_style = dataset_style(color, marker, size)
        range_selected = sorted(
            (
                float(row["target_tilde_epsilon_prime"]),
                100.0 * (1.0 - float(row["drop_over_padded_ideal_ratio"])),
                100.0 * float(row["drop_over_padded_ideal_ratio_standard_error"]),
            )
            for row in proxy_rows
            if row["dataset"] == dataset
            and row["task"] == "Range"
            and row["method"] == "Drop"
        )
        hh_selected = sorted(
            (
                float(row["target_tilde_epsilon_prime"]),
                100.0 * float(row["paired_drop_minus_padded_ideal"]),
                100.0
                * float(row["paired_drop_minus_padded_ideal_standard_error"]),
            )
            for row in proxy_rows
            if row["dataset"] == dataset
            and row["task"] == "Heavy hitter"
            and row["method"] == "Drop"
        )
        for output, selected in [
            (range_gain_plots, range_selected),
            (hh_gain_plots, hh_selected),
        ]:
            output.append(
                "\\addplot+["
                + relative_style
                + ", error bars/.cd, y dir=both, y explicit] coordinates {"
                + " ".join(
                    f"({fnum(x)},{fnum(y)}) +- (0,{fnum(se)})"
                    for x, y, se in selected
                )
                + "};"
            )

    common_target_axis = (
        "height=2.05cm, tick label style={font=\\scriptsize}, "
        "label style={font=\\scriptsize}, title style={font=\\scriptsize\\bfseries, "
        "at={(0,1.02)}, anchor=south west}, xmin=0.065, xmax=0.335, "
        "xtick={0.08,0.16,0.32}, xticklabels={0.08,0.16,0.32}"
    )
    absolute_range_axis = axis_options(
        "0.78\\linewidth",
        "design scale $\\widetilde{\\varepsilon}'$",
        "range NAE",
        common_target_axis
        + ", title={(a) Range queries}, ymode=log, "
        "ymin=0.0015, ymax=0.15, ytick={0.002,0.005,0.01,0.02,0.05,0.1}, "
        "yticklabels={.002,.005,.01,.02,.05,.1}",
    )
    absolute_hh_axis = axis_options(
        "0.78\\linewidth",
        "design scale $\\widetilde{\\varepsilon}'$",
        "top-50 F1",
        common_target_axis
        + ", title={(b) Heavy hitters}, "
        "ymin=0.15, ymax=1.01, ytick={0.2,0.4,0.6,0.8,1.0}",
    )
    range_gain_axis = axis_options(
        "0.78\\linewidth",
        "design target $\\widetilde{\\epsilon}'$",
        "NAE reduction (\\%)",
        common_target_axis
        + ", title={(c) Range gain over padded ideal}, "
        "ymin=-170, ymax=60, ytick={-150,-100,-50,0,50}",
    )
    hh_gain_axis = axis_options(
        "0.78\\linewidth",
        "design target $\\widetilde{\\epsilon}'$",
        "F1 gain (points)",
        common_target_axis
        + ", title={(d) Heavy-hitter gain over padded ideal}, "
        "ymin=-6, ymax=26, ytick={-5,0,10,20}",
    )

    exact_rows = read_csv(DATA_DIR / "fig_large_exact_family_utility.csv")
    if len(exact_rows) != 24 or any(
        row["pairwise_exact_profile_only"] != "true"
        or row["mechanism_wide_dp_claimed"] != "false"
        or row["proxy_used_in_calibration"] != "false"
        or row["central_epsilon_claimed_for_this_workload"] != "true"
        for row in exact_rows
    ):
        raise ValueError("exact utility panel requires the complete pairwise exact-law grid")
    exact_targets = [0.1, 0.3, 0.7]

    def worst_endpoint_exact_mae(
        method: str,
        epsilon: float,
        rho: float | None = None,
    ) -> float:
        selected = [
            float(row["exact_law_mean_absolute_count_error"])
            for row in exact_rows
            if row["method"] == method
            and float(row["exact_target_epsilon_prime"]) == epsilon
            and (rho is None or float(row["rho"]) == rho)
        ]
        if len(selected) != 2:
            raise ValueError("exact panel must contain both endpoints of its declared pair")
        return max(selected)

    ideal_exact = {
        epsilon: worst_endpoint_exact_mae("Padded Ideal", epsilon)
        for epsilon in exact_targets
    }
    exact_styles = [
        f"color={PURPLE}, dashed, line width=0.98pt, mark=square*, mark size=1.65pt, mark options={{fill=white}}",
        f"color={ORANGE}, solid, line width=1.02pt, mark=*, mark size=1.70pt",
        f"color={TEAL}, dotted, line width=1.08pt, mark=triangle*, mark size=1.80pt, mark options={{fill=white}}",
    ]
    exact_plots: list[str] = []
    for rho, style in zip([0.5, 1.0, 2.0], exact_styles):
        coords = " ".join(
            f"({fnum(epsilon)},{fnum(100.0 * (1.0 - worst_endpoint_exact_mae('Drop', epsilon, rho) / ideal_exact[epsilon]))})"
            for epsilon in exact_targets
        )
        exact_plots.append(
            "\\addplot+["
            + style
            + "] coordinates {"
            + coords
            + "};\\addlegendentry{$\\rho="
            + (".5" if rho == 0.5 else fnum(rho))
            + "$}"
        )
    exact_axis = axis_options(
        "0.78\\linewidth",
        "pairwise $\\varepsilon'$",
        "MAE gain (\\%)",
        (
            "height=2.05cm, tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, "
            "title={(d) Exact pairwise gain}, "
            "title style={font=\\scriptsize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "xmin=0.07, xmax=0.73, xtick={0.1,0.3,0.7}, ymin=-4, ymax=24, "
            "ytick={0,10,20}, legend style={font=\\scriptsize, draw=none, "
            "fill=none, at={(0.02,0.98)}, anchor=north west, legend columns=3}"
        ),
    )

    sum_rows = [
        row
        for row in read_csv(DATA_DIR / "fig_full_release_real_workloads.csv")
        if row["task"] == "Sum"
    ]
    if {(row["dataset"], row["method"]) for row in sum_rows} != {
        (dataset, method)
        for dataset in datasets
        for method in ["Ideal", "Pool"]
    }:
        raise ValueError("sum preservation panel is incomplete")

    def sum_coords(method: str) -> str:
        selected = {
            row["dataset"]: row for row in sum_rows if row["method"] == method
        }
        return " ".join(
            f"({dataset},{fnum(100.0 * float(selected[dataset]['value']))}) "
            f"+- (0,{fnum(100.0 * float(selected[dataset]['standard_error']))})"
            for dataset in datasets
        )

    sum_axis = axis_options(
        "0.78\\linewidth",
        "",
        "sum error (\\%)",
        (
            "height=2.05cm, tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, "
            "title={(c) Additive sums}, "
            "title style={font=\\scriptsize\\bfseries, at={(0,1.02)}, anchor=south west}, "
            "ybar, bar width=5.0pt, ymin=0, symbolic x coords={Synthetic,Adult,Covertype}, "
            "xtick=data, xticklabels={Synthetic,Adult,Covertype}, enlarge x limits=0.18, "
            "legend style={font=\\scriptsize, draw=none, fill=none, at={(0.50,0.98)}, "
            "anchor=north, legend columns=2}"
        ),
    )

    def panel(axis: str, plots: list[str], reference: str = "") -> str:
        return f"""\\begin{{minipage}}{{0.242\\textwidth}}
\\centering
\\makebox[\\linewidth][c]{{%
\\begin{{tikzpicture}}
\\begin{{axis}}[
{axis}
]
{reference}
{chr(10).join(plots)}
\\end{{axis}}
\\end{{tikzpicture}}%
}}
\\end{{minipage}}"""

    sum_plots = [
        f"\\addplot+[{BAR_STYLES[0]}, error bars/.cd, y dir=both, y explicit] "
        f"coordinates {{{sum_coords('Ideal')}}};\\addlegendentry{{Padded ideal}}",
        f"\\addplot+[{BAR_STYLES[2]}, error bars/.cd, y dir=both, y explicit] "
        f"coordinates {{{sum_coords('Pool')}}};\\addlegendentry{{Pool}}",
    ]
    zero_gain_reference = (
        "\\addplot+[black!35, densely dashed, no markers, forget plot] "
        "coordinates {(0.08,0) (0.32,0)};"
    )
    exact_parity_reference = (
        "\\addplot+[black!35, densely dashed, no markers, forget plot] "
        "coordinates {(0.1,0) (0.7,0)};"
    )
    body = f"""% Auto-generated from complete-release proxy, pairwise finite-law numerical, and corrected-sum CSVs.
\\begin{{figure*}}[!t]
\\centering
{shared_legend}\\par\\vspace{{0.45mm}}
{panel(absolute_range_axis, absolute_range_plots)}\\hfill
{panel(absolute_hh_axis, absolute_hh_plots)}\\hfill
{panel(sum_axis, sum_plots)}\\hfill
{panel(exact_axis, exact_plots, exact_parity_reference)}
\\caption{{Complete-release utility. (a--b) Absolute utility over 100 paired
trials at $M=2m$; padded ideal and \\textsc{{Drop}} use budgets selected by the
non-certifying quadratic scale. (c) Padded ideal versus \\textsc{{Pool}} at
shared $\\varepsilon_0=1$; error bars in (a--c) are standard errors. (d)
Finite-law numerical MAE gain at equal pair-specific $\\varepsilon'$
($m=10^5$, $M=2m$, $\\delta=10^{{-6}}$).}}
\\Description{{Four panels show absolute range and heavy-hitter utility for
padded ideal shuffling and Drop, corrected additive utility for Pool, and a
finite-law pairwise Drop MAE reduction across load.}}
\\label{{fig:utility-full-release-summary}}
\\end{{figure*}}
"""
    (FIGURE_DIR / "fig_utility_full_release_summary.tex").write_text(body)


def utility_full_release_summary_figure() -> None:
    """Build the modern exact-pair-calibrated figure used by the paper.

    Keep the older Synthetic/Adult/Covertype proxy composition available as
    ``_legacy_proxy_utility_full_release_summary_figure`` for diagnostics, but
    do not let the all-figures entry point overwrite the paper's four-workload
    figure with that differently scoped experiment.
    """
    from make_modern_utility_figure import generate

    generate()


def bar_figure_microbench() -> None:
    rows = read_csv(DATA_DIR / "fig_product_vs_dual_microbench.csv")
    vals = {row["method"]: float(row["time_ms_per_report"]) for row in rows}
    benchmark_n = int(float(rows[0]["N"]))
    benchmark_log_n = int(float(rows[0]["log_n"]))
    product_name = PACKED_METHOD if PACKED_METHOD in vals else "Product-valued kernel"
    tuple_name = TUPLE_METHOD if TUPLE_METHOD in vals else ""
    dual_name = DUAL_METHOD if DUAL_METHOD in vals else "Dual-track kernel"
    if tuple_name:
        tuple_plot = (
            f"\\addplot+[{BAR_STYLES[2]}, bar shift=0pt] "
            f"coordinates {{(Tuple,{fnum(vals[tuple_name])})}};"
        )
    else:
        tuple_plot = "%"
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
{tuple_plot}
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
    legend = inline_line_legend(
        [
            (method.replace("Google DPF ", ""), LINE_STYLES[idx % len(LINE_STYLES)])
            for idx, method in enumerate(methods)
        ],
        3,
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
        compact_ylabel = ylabel
        if y_col == "reports_per_second":
            compact_ylabel = "r/s"
        panel_axis_options = axis_options(
            "\\linewidth",
            xlabel,
            compact_ylabel,
            (
                "height=2.72cm, tick label style={font=\\tiny}, label style={font=\\scriptsize}, "
                "title={" + title + "}, title style={font=\\scriptsize, yshift=-0.4ex}, "
                + extra_axis
            ),
        )
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

    body = f"""% Auto-generated from data/fig_dpf_scaling.csv and data/fig_dpf_thread_scaling.csv.
\\begin{{figure}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
{chr(10).join(panels)}
\\caption{{Measured Google DPF write-backend scaling. Packed64 is the optimized single-word Chafe backend; tuple product and dual-track are unoptimized baselines. Left: single-thread per-report cost as the hidden-slot domain grows. Right: report-batch parallelism at $N=2^{{17}}$.}}
\\label{{fig:dpf-backend-summary}}
\\end{{figure}}
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
    legend = inline_bar_legend(
        [
            (short_names[method], BAR_STYLES[idx % len(BAR_STYLES)])
            for idx, method in enumerate(methods)
        ],
        3,
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
            "height=2.72cm, tick label style={font=\\tiny}, label style={font=\\scriptsize}, "
            "title={End-to-End}, title style={font=\\scriptsize, yshift=-0.4ex}, "
            "ybar, bar width=6pt, symbolic x coords={Packed,Tuple,Dual}, xtick=data, ymin=0, enlarge x limits=0.18"
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
        "",
        (
            "height=2.72cm, tick label style={font=\\tiny}, label style={font=\\scriptsize}, "
            "title={Phases}, title style={font=\\scriptsize, yshift=-0.4ex}, "
            "ybar, bar width=2.5pt, symbolic x coords={Keygen,Ctx,Eval,Acc}, xtick=data, ymin=0, enlarge x limits=0.12, yticklabel=\\empty"
        ),
    )

    body = f"""% Auto-generated from data/fig_product_vs_dual_microbench.csv and data/fig_dpf_phase_breakdown.csv.
\\begin{{figure}}[t]
\\centering
{legend}\\par\\vspace{{0.7mm}}
\\begin{{minipage}}{{0.49\\columnwidth}}
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
{total_axis_options}
]
{chr(10).join(total_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{minipage}}
\\begin{{minipage}}{{0.49\\columnwidth}}
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
\\end{{figure}}
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
            "csv_name": "fig_privacy_vs_m_large.csv",
            "tex_name": "fig_privacy_vs_m_large.tex",
            "x_col": "m",
            "y_col": "epsilon",
            "xlabel": "epoch size $m$",
            "ylabel": "$\\epsilon'$",
            "methods": ["IdealShuffle", "Drop (matched padding)", "Pool (replacement)", "Pool (add/remove)"],
            "caption": "Formula-derived large-scale amplification as the epoch size grows, with $\\rho=1$, $\\tau=2$, $\\epsilon_0=1$, and $\\delta=10^{-6}$. All curves are second-order proxies and use $N=m$.",
            "label": "fig:privacy-vs-m-large",
            "extra_axis": "xmode=log, ymode=log, xmin=10000, xmax=100000000, legend columns=2",
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
    accountant_scaling_figure()
    count += 1
    ingestion_latency_figure()
    count += 1
    ingestion_cost_summary_figure()
    count += 1
    finite_burst_latency_figure()
    count += 1
    privacy_compact_summary_figure()
    count += 1
    privacy_2x2_validation_figure()
    count += 1
    privacy_large_m_accounting_figure()
    count += 1
    dpf_backend_summary_figure()
    count += 1
    dpf_backend_breakdown_summary_figure()
    count += 1
    privacy_surface_figure()
    count += 1
    covertype_workloads_figure()
    count += 1
    utility_cross_dataset_summary_figure()
    count += 1
    utility_full_release_summary_figure()
    count += 1
    full_release_dummy_mismatch_figure()
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
