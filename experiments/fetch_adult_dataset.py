from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

from chafe_eval_lib import DATASET_DIR, ensure_dirs


ADULT_DIR = DATASET_DIR / "adult"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
FILES = {
    "adult.data": f"{BASE_URL}/adult.data",
    "adult.test": f"{BASE_URL}/adult.test",
    "adult.names": f"{BASE_URL}/adult.names",
}

FIELDNAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_FIELDS = {
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
}


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def clean_value(name: str, value: str) -> str | int:
    value = value.strip()
    if name == "income":
        value = value.rstrip(".")
    if name in NUMERIC_FIELDS:
        return int(value)
    return value


def iter_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for raw in reader:
            if not raw or raw[0].startswith("|"):
                continue
            if len(raw) != len(FIELDNAMES):
                raise ValueError(f"{path}: expected {len(FIELDNAMES)} columns, got {len(raw)} in {raw}")
            rows.append({name: clean_value(name, value) for name, value in zip(FIELDNAMES, raw)})
    return rows


def fetch_and_clean() -> dict[str, int]:
    ensure_dirs()
    ADULT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(url, ADULT_DIR / name)

    train_rows = iter_rows(ADULT_DIR / "adult.data")
    test_rows = iter_rows(ADULT_DIR / "adult.test")
    all_rows = train_rows + test_rows

    clean_path = ADULT_DIR / "adult_clean.csv"
    with clean_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "dataset": "UCI Adult / Census Income",
        "uci_id": 2,
        "uci_page": "https://archive.ics.uci.edu/dataset/2/adult",
        "raw_urls": FILES,
        "clean_csv": str(clean_path.relative_to(DATASET_DIR.parents[0])),
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
        "rows_total": len(all_rows),
        "fields": FIELDNAMES,
        "missing_token": "?",
        "notes": (
            "The script strips whitespace, removes the trailing period in the test-set income label, "
            "and keeps '?' as an explicit categorical value."
        ),
    }
    (ADULT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"train": len(train_rows), "test": len(test_rows), "total": len(all_rows)}


def ensure_adult_dataset() -> Path:
    clean_path = ADULT_DIR / "adult_clean.csv"
    if not clean_path.exists():
        fetch_and_clean()
    return clean_path


if __name__ == "__main__":
    counts = fetch_and_clean()
    print(f"adult_clean.csv: {counts['total']} rows ({counts['train']} train, {counts['test']} test)")
