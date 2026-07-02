from __future__ import annotations

from chafe_eval_lib import DATA_DIR, read_csv, write_csv


def panel_rows(csv_name: str, panel: str, workload_family: str) -> list[dict]:
    rows: list[dict] = []
    for row in read_csv(DATA_DIR / csv_name):
        rows.append(
            {
                **row,
                "panel": panel,
                "workload_family": workload_family,
                "derived_from": csv_name,
            }
        )
    return rows


def compose_pair(
    *,
    out_name: str,
    synthetic_csv: str,
    real_csv: str,
    workload_family: str,
) -> list[dict]:
    rows = [
        *panel_rows(synthetic_csv, "Synthetic Zipf", workload_family),
        *panel_rows(real_csv, "Adult empirical bootstrap", workload_family),
    ]
    write_csv(DATA_DIR / out_name, rows)
    return rows


def generate_all() -> dict[str, int]:
    range_rows = compose_pair(
        out_name="fig_range_error_zipf_adult.csv",
        synthetic_csv="fig_range_error_vs_epsilon.csv",
        real_csv="fig_real_age_range_error_vs_epsilon.csv",
        workload_family="range_count",
    )
    sum_rows = compose_pair(
        out_name="fig_sum_error_zipf_adult.csv",
        synthetic_csv="fig_sum_error_vs_epsilon.csv",
        real_csv="fig_real_hours_sum_error_vs_epsilon.csv",
        workload_family="bounded_sum",
    )
    hh_rows = compose_pair(
        out_name="fig_heavyhitter_f1_zipf_adult.csv",
        synthetic_csv="fig_heavyhitter_f1_vs_epsilon.csv",
        real_csv="fig_real_heavyhitter_f1_vs_epsilon.csv",
        workload_family="heavy_hitter",
    )
    return {
        "fig_range_error_zipf_adult.csv": len(range_rows),
        "fig_sum_error_zipf_adult.csv": len(sum_rows),
        "fig_heavyhitter_f1_zipf_adult.csv": len(hh_rows),
    }


if __name__ == "__main__":
    for name, rows in generate_all().items():
        print(f"{name}: {rows} rows")
