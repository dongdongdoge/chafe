from __future__ import annotations

from chafe_eval_lib import DATA_DIR, write_csv
from gpu_dpf_interface import NORMALIZED_COLUMNS, reference_rows


def generate() -> list[dict]:
    rows = reference_rows()
    write_csv(DATA_DIR / "fig_gpu_dpf_external.csv", rows, fieldnames=list(NORMALIZED_COLUMNS))
    return rows


if __name__ == "__main__":
    print(f"fig_gpu_dpf_external.csv: {len(generate())} rows")
