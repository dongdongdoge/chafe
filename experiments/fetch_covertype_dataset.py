from __future__ import annotations

import csv
import gzip
import json
import urllib.request
from pathlib import Path

from chafe_eval_lib import DATASET_DIR, ensure_dirs


COVERTYPE_DIR = DATASET_DIR / "covertype"
RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/31/covertype"
DOI = "10.24432/C50K5N"
RAW_NAME = "covtype.data.gz"
CLEAN_NAME = "covertype_selected.csv"
FIELDNAMES = [
    "elevation",
    "horizontal_distance_to_roadways",
    "wilderness_area",
    "soil_type",
    "cover_type",
]


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def one_hot_index(values: list[int], offset: int, width: int) -> int:
    active = [idx + 1 for idx, value in enumerate(values[offset : offset + width]) if value == 1]
    if len(active) != 1:
        raise ValueError(f"expected exactly one active value in one-hot block, saw {active}")
    return active[0]


def fetch_and_clean() -> dict[str, int]:
    ensure_dirs()
    COVERTYPE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = COVERTYPE_DIR / RAW_NAME
    clean_path = COVERTYPE_DIR / CLEAN_NAME
    download(RAW_URL, raw_path)

    rows = 0
    with gzip.open(raw_path, "rt", newline="") as src, clean_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDNAMES)
        writer.writeheader()
        for raw in csv.reader(src):
            if len(raw) != 55:
                raise ValueError(f"expected 55 columns in Covertype row, saw {len(raw)}")
            values = [int(value) for value in raw]
            writer.writerow(
                {
                    "elevation": values[0],
                    "horizontal_distance_to_roadways": values[5],
                    "wilderness_area": one_hot_index(values, 10, 4),
                    "soil_type": one_hot_index(values, 14, 40),
                    "cover_type": values[54],
                }
            )
            rows += 1

    metadata = {
        "dataset": "UCI Covertype",
        "uci_id": 31,
        "uci_page": UCI_PAGE,
        "doi": DOI,
        "raw_url": RAW_URL,
        "raw_file": str(raw_path.relative_to(DATASET_DIR.parents[0])),
        "clean_csv": str(clean_path.relative_to(DATASET_DIR.parents[0])),
        "rows_total": rows,
        "selected_fields": FIELDNAMES,
        "notes": (
            "The selected table keeps one numeric field for range queries, one bounded numeric field "
            "for sum queries, and wilderness/soil/cover categorical fields for heavy hitters."
        ),
    }
    (COVERTYPE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"total": rows}


def ensure_covertype_dataset() -> Path:
    clean_path = COVERTYPE_DIR / CLEAN_NAME
    if not clean_path.exists():
        fetch_and_clean()
    return clean_path


if __name__ == "__main__":
    counts = fetch_and_clean()
    print(f"{CLEAN_NAME}: {counts['total']} rows")
