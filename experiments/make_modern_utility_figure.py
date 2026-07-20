"""Generate the exact-calibrated synthetic and modern utility figure."""

from __future__ import annotations

import json
from collections import defaultdict

from chafe_eval_lib import DATA_DIR, DATASET_DIR, FIGURE_DIR, read_csv


SUBMISSION_FIGURE_DIR = (
    FIGURE_DIR.parent / "chafe-vldb-submission-source" / "figures"
)


DATASETS = (
    ("Synthetic", "{rgb,255:red,126;green,87;blue,194}", "diamond*", "1.75pt"),
    ("ACS 2024", "{rgb,255:red,37;green,99;blue,235}", "*", "1.70pt"),
    ("NYC TLC 2024", "{rgb,255:red,213;green,94;blue,0}", "square*", "1.65pt"),
    ("Amazon Reviews'23", "{rgb,255:red,0;green,150;blue,136}", "triangle*", "1.80pt"),
)
TARGETS = (0.1, 0.3, 0.7)


def _workload_table() -> str:
    acs = json.loads((DATASET_DIR / "acs2024" / "metadata.json").read_text())
    tlc = json.loads((DATASET_DIR / "nyc_tlc2024" / "metadata.json").read_text())
    reviews = json.loads(
        (DATASET_DIR / "amazon_reviews_2023" / "metadata.json").read_text()
    )
    return rf"""
% Auto-generated from frozen dataset metadata.
\begin{{table*}}[t]
\caption{{Datasets and query workloads.  Each trial uses an epoch of
$m=10^5$ records.  Synthetic epochs are fresh seeded i.i.d. draws; real-data
epochs are sampled without replacement from the retained snapshot.}}
\label{{tab:utility-workloads}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.10}}
\begin{{tabular}}{{@{{}}L{{0.15\textwidth}}L{{0.14\textwidth}}L{{0.21\textwidth}}L{{0.24\textwidth}}L{{0.19\textwidth}}@{{}}}}
\toprule
Dataset & Retained records & Range-count workload & Top-50 heavy hitters & Bounded-sum workload \\
\midrule
Synthetic & $10^5$ per trial & 512 intervals over 256 $\operatorname{{Zipf}}(1.08)$ labels & $\operatorname{{Zipf}}(1.08)$ labels over a 4,096-symbol domain & Sum of $\operatorname{{Beta}}(2,5)$ values \\
ACS 2024 & {int(acs['admitted_rows']):,} & Age in $[16,90]$ over 256 bins & Occupation codes & Weekly hours$/99$ \\
NYC TLC 2024 & {int(tlc['reservoir_rows']):,} & Trip distance in $[0,50]$ miles over 256 bins & Pickup--dropoff zone pairs & Total amount$/200$ \\
Amazon Reviews'23 & {int(reviews['admitted_rows']):,} & Review time over 256 bins (2018--2023) & Parent products & Rating$/5$ \\
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def _fixed_eps0_table(rows: list[dict[str, str]]) -> str:
    indexed = {
        (row["dataset"], row["task"], row["method"]): row for row in rows
    }
    labels = (
        ("Synthetic", "Synthetic"),
        ("ACS 2024", "ACS"),
        ("NYC TLC 2024", "TLC"),
        ("Amazon Reviews'23", "Reviews'23"),
    )
    body = []
    for dataset, label in labels:
        range_ideal = float(
            indexed[(dataset, "Range", "Padded Ideal")]["mean"]
        )
        range_drop = float(
            indexed[(dataset, "Range", "Drop")]["mean"]
        )
        range_ratio = float(
            indexed[(dataset, "Range", "Drop")]["ratio_to_padded_ideal"]
        )
        hh_gap = float(
            indexed[(dataset, "Heavy hitter", "Drop")][
                "paired_difference_from_padded_ideal"
            ]
        )
        sum_gap = float(
            indexed[(dataset, "Bounded sum", "Pool")][
                "paired_difference_from_padded_ideal"
            ]
        )
        sum_text = "0.000" if abs(sum_gap) < 5e-13 else f"{sum_gap:+.3f}"
        body.append(
            f"{label} & {range_ideal:.4f}$\\to${range_drop:.4f} ({range_ratio:.2f}$\\times$) & {hh_gap:+.3f} & {sum_text} \\\\"
        )
    return rf"""
% Auto-generated from fixed-local-budget complete-release controls.
\begin{{table}}[t]
\caption{{Utility at the same local budget ($\varepsilon_0=8$; 100 paired
trials).  Ratios above one and negative F1 gaps favor padded ideal;
\textsc{{Pool}}'s sum-error gap is exactly zero.}}
\label{{tab:fixed-eps0-utility}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.2pt}}
\renewcommand{{\arraystretch}}{{1.04}}
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
Dataset & Range error: ideal$\to$Drop & HH $\Delta$F1 & Sum $\Delta$error \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def _coordinates(rows: dict[tuple[str, str, str, float], dict[str, str]], dataset: str, task: str, method: str, *, percent: bool = False) -> str:
    points = []
    for target in TARGETS:
        row = rows[(dataset, task, method, target)]
        scale = 100.0 if percent else 1.0
        mean = scale * float(row["mean"])
        se = scale * float(row["se"])
        points.append(f"({target:g},{mean:.9g}) +- (0,{se:.8g})")
    return " ".join(points)


def _method_plot(color: str, mark: str, mark_size: str, coordinates: str, *, chafe: bool) -> str:
    if chafe:
        style = (
            f"color={color}, solid, line width=1.08pt, mark={mark}, "
            f"mark size={mark_size}, mark options={{solid, fill={color}, line width=0.55pt}}"
        )
    else:
        style = (
            f"color={color}, dashed, opacity=.62, line width=0.82pt, mark={mark}, "
            f"mark size={mark_size}, mark options={{solid, fill=white, opacity=.78, line width=0.58pt}}"
        )
    return (
        f"\\addplot+[{style}, error bars/.cd, y dir=both, y explicit] "
        f"coordinates {{{coordinates}}};\n"
    )


def _axis(task: str, rows: dict[tuple[str, str, str, float], dict[str, str]]) -> str:
    if task == "Range":
        title = "(a) Range counts"
        ylabel = "mean relative error"
        extra = (
            "ymode=log, ymin=0.002, ymax=0.30, "
            "ytick={0.002,0.005,0.01,0.02,0.05,0.1,0.2}, "
            "yticklabels={.002,.005,.01,.02,.05,.1,.2}"
        )
        chafe_method = "Drop"
        percent = False
    elif task == "Heavy hitter":
        title = "(b) Top-50 heavy hitters"
        ylabel = "F1"
        extra = "ymin=0.40, ymax=1.01, ytick={0.4,0.6,0.8,1.0}"
        chafe_method = "Drop"
        percent = False
    elif task == "Bounded sum":
        title = "(c) Bounded sums (Laplace)"
        ylabel = "relative RMSE (\\%)"
        extra = (
            "ymode=log, ymin=0.04, ymax=0.9, "
            "ytick={0.05,0.1,0.2,0.5,0.8}, yticklabels={.05,.1,.2,.5,.8}"
        )
        chafe_method = "Pool"
        percent = True
    else:
        raise ValueError(f"unknown task: {task}")

    plots = []
    for dataset, color, mark, mark_size in DATASETS:
        ideal = _coordinates(rows, dataset, task, "Padded Ideal", percent=percent)
        chafe = _coordinates(rows, dataset, task, chafe_method, percent=percent)
        plots.append(_method_plot(color, mark, mark_size, ideal, chafe=False))
        plots.append(_method_plot(color, mark, mark_size, chafe, chafe=True))
    annotation = ""
    if task == "Bounded sum":
        annotation = (
            "\\node[font=\\scriptsize, anchor=north east, fill=white, "
            "fill opacity=.88, text opacity=1] at (rel axis cs:0.98,0.96) "
            "{Pool: larger $\\varepsilon_0$};\n"
        )
    return rf"""
\begin{{minipage}}{{0.322\textwidth}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
width=0.78\linewidth,
height=2.40cm,
scale only axis,
axis background/.style={{fill=white}},
ymajorgrids=true,
xmajorgrids=false,
major grid style={{draw=gray!20, line width=0.20pt}},
tick label style={{font=\scriptsize}},
label style={{font=\scriptsize}},
title style={{font=\scriptsize\bfseries, at={{(0,1.025)}}, anchor=south west}},
axis line style={{black!55, line width=0.35pt}},
tick style={{black!55, line width=0.35pt}},
tick align=outside,
axis on top,
line cap=round,
line join=round,
clip marker paths=true,
xlabel={{declared-pair $\varepsilon'$}},
ylabel={{{ylabel}}},
xmin=0.07, xmax=0.73,
xtick={{0.1,0.3,0.7}},
title={{{title}}},
{extra}
]
{''.join(plots)}{annotation}\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}
"""


def generate() -> None:
    summary = read_csv(DATA_DIR / "fig_modern_utility.csv")
    replicates = read_csv(DATA_DIR / "fig_modern_utility_replicates.csv")
    calibrations = read_csv(DATA_DIR / "fig_modern_utility_calibration.csv")
    fixed_eps0 = read_csv(DATA_DIR / "fig_modern_fixed_eps0_utility.csv")
    if len(summary) != 72 or len(replicates) != 7_200:
        raise ValueError("modern utility figure requires the complete 100-trial grid")
    if any(int(row["replicates"]) != 100 for row in summary):
        raise ValueError("modern utility summary contains an incomplete configuration")
    if any(float(row["profile_at_target"]) > float(row["delta"]) for row in calibrations):
        raise ValueError("utility calibration exceeds its declared pair target")
    if len(fixed_eps0) != 24 or any(int(row["replicates"]) != 100 for row in fixed_eps0):
        raise ValueError("fixed-local-budget utility control is incomplete")
    sum_rows = [
        row
        for row in replicates
        if row["task"] == "Bounded sum" and row["method"] == "Pool"
    ]
    if len(sum_rows) != 1_200 or any(
        row["pool_sum_preserved"] != "1" for row in sum_rows
    ):
        raise ValueError("bounded-sum Pool preservation check failed")

    indexed = {
        (row["dataset"], row["task"], row["method"], float(row["target_epsilon"])): row
        for row in summary
    }
    legend = r"""
\begingroup
\setlength{\tabcolsep}{2.0pt}
\renewcommand{\arraystretch}{0.90}
\makebox[\linewidth][c]{%
\begin{tabular}{@{}c@{\hspace{0.14em}}l@{\hspace{0.44em}}c@{\hspace{0.14em}}l@{\hspace{0.44em}}c@{\hspace{0.14em}}l@{\hspace{0.44em}}c@{\hspace{0.14em}}l@{\hspace{0.62em}}c@{\hspace{0.14em}}l@{\hspace{0.44em}}c@{\hspace{0.14em}}l@{}}
\tikz[baseline=-0.55ex]{\draw[color={rgb,255:red,126;green,87;blue,194}, line width=1pt] (0,0)--(.34,0); \fill[color={rgb,255:red,126;green,87;blue,194}] (.17,.055)--(.225,0)--(.17,-.055)--(.115,0)--cycle;} & {\scriptsize Synthetic} &
\tikz[baseline=-0.55ex]{\draw[color={rgb,255:red,37;green,99;blue,235}, line width=1pt] (0,0)--(.38,0); \fill[color={rgb,255:red,37;green,99;blue,235}] (.19,0) circle (1.25pt);} & {\scriptsize ACS 2024} &
\tikz[baseline=-0.55ex]{\draw[color={rgb,255:red,213;green,94;blue,0}, line width=1pt] (0,0)--(.38,0); \fill[color={rgb,255:red,213;green,94;blue,0}] (.15,-.04) rectangle (.23,.04);} & {\scriptsize TLC 2024} &
\tikz[baseline=-0.55ex]{\draw[color={rgb,255:red,0;green,150;blue,136}, line width=1pt] (0,0)--(.38,0); \fill[color={rgb,255:red,0;green,150;blue,136}] (.19,.055)--(.13,-.045)--(.25,-.045)--cycle;} & {\scriptsize Reviews'23} &
\tikz[baseline=-0.55ex]{\draw[color=black!70, solid, line width=1.08pt] (0,0)--(.34,0); \fill[color=black!70] (.17,0) circle (1.20pt);} & {\scriptsize Chafe} &
\tikz[baseline=-0.55ex]{\draw[color=black!55, dashed, line width=.82pt] (0,0)--(.34,0); \draw[color=black!55, fill=white] (.17,0) circle (1.20pt);} & {\scriptsize padded ideal}
\end{tabular}%
}
\endgroup\par\vspace{0.35mm}
"""
    body = (
        "% Auto-generated from exact-pair-calibrated complete-release utility CSVs.\n"
        "\\begin{figure*}[!t]\n\\centering\n"
        + legend
        + _axis("Range", indexed).strip()
        + "%\n\\hfill\n"
        + _axis("Heavy hitter", indexed).strip()
        + "%\n\\hfill\n"
        + _axis("Bounded sum", indexed).strip()
        + r"""
\caption{Complete length-$M$ release utility on one synthetic and three modern
workloads (100 paired trials; $m=10^5$, $N=M=2\times10^5$,
$\delta=10^{-6}$). Categorical budgets use exact finite inversion for a
declared endpoint pair; sums use unbiased scale-8 fixed-point discrete Laplace
and two-sided one-slot
Chernoff inversion. \textsc{Pool} preserves each randomized sum and uses a
larger accounted $\varepsilon_0$. Modern data are transferred-reference
checks; pairing only reduces Monte Carlo variance. Error bars show standard
errors.}
\Description{Three panels compare padded ideal shuffling with Drop for range
counts and top-50 heavy hitters, and with Pool for bounded sums, on a seeded
Zipf/Beta workload, ACS 2024, NYC TLC 2024, and Amazon Reviews'23.}
\label{fig:utility-full-release-summary}
\end{figure*}
"""
    )
    (FIGURE_DIR / "fig_utility_full_release_summary.tex").write_text(body)
    table = _workload_table()
    fixed_table = _fixed_eps0_table(fixed_eps0)
    (FIGURE_DIR / "tab_utility_workloads.tex").write_text(table)
    (FIGURE_DIR / "tab_fixed_eps0_utility.tex").write_text(fixed_table)
    SUBMISSION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (SUBMISSION_FIGURE_DIR / "fig_utility_full_release_summary.tex").write_text(body)
    (SUBMISSION_FIGURE_DIR / "tab_utility_workloads.tex").write_text(table)
    (SUBMISSION_FIGURE_DIR / "tab_fixed_eps0_utility.tex").write_text(fixed_table)


def generate_all() -> dict[str, int]:
    generate()
    return {
        "figures/fig_utility_full_release_summary.tex": 1,
        "figures/tab_utility_workloads.tex": 1,
        "figures/tab_fixed_eps0_utility.tex": 1,
    }


if __name__ == "__main__":
    generate()
