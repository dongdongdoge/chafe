"""Fetch and freeze modern utility workloads used by the paper.

The checked-in experiment consumes compact ``utility.npz`` snapshots.  This
script reconstructs them from three public raw releases and records every
filter, clipping bound, source checksum, and resulting row count in metadata.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from chafe_eval_lib import DATASET_DIR, ensure_dirs


RANGE_DOMAIN = 256
HH_DOMAIN = 4096
TLC_RESERVOIR_ROWS = 500_000

ACS_DIR = DATASET_DIR / "acs2024"
TLC_DIR = DATASET_DIR / "nyc_tlc2024"
AMAZON_DIR = DATASET_DIR / "amazon_reviews_2023"

SOURCES = {
    "ACS 2024": {
        "dir": ACS_DIR,
        "raw": "csv_pny.zip",
        "url": "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pny.zip",
        "sha256": "89d73997653d2a1d1b535a4a6618987819390b57a6022a167ec76fb8c7bf6699",
        "snapshot": "2024 ACS 1-year New York person PUMS",
    },
    "NYC TLC 2024": {
        "dir": TLC_DIR,
        "raw": "yellow_tripdata_2024-01.parquet",
        "url": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
        "sha256": "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510",
        "snapshot": "January 2024 Yellow Taxi trip records",
    },
    "Amazon Reviews'23": {
        "dir": AMAZON_DIR,
        "raw": "All_Beauty.jsonl.gz",
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz",
        "sha256": "ee00e66835567c3f12fde6a482f8d7055c22cac2d0924f677263affdd8a0e349",
        "snapshot": "Amazon Reviews'23 All Beauty, interactions from 2018 onward",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_raw(name: str) -> Path:
    source = SOURCES[name]
    directory = Path(source["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / str(source["raw"])
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(str(source["url"]), timeout=300) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    actual = sha256(path)
    if actual != source["sha256"]:
        raise ValueError(f"{name} checksum mismatch: {actual}")
    return path


def encode_top_categories(labels: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    counts = Counter(labels)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    retained = ranked[: HH_DOMAIN - 1]
    code = {label: index for index, (label, _) in enumerate(retained)}
    other = HH_DOMAIN - 1
    encoded = np.fromiter((code.get(label, other) for label in labels), dtype=np.int32)
    return encoded, {
        "raw_category_count": len(ranked),
        "retained_category_count": len(retained),
        "other_code": other,
        "reference_category_code": 0,
        "reference_category_label": retained[0][0],
        "reference_category_count": retained[0][1],
    }


def write_snapshot(
    name: str,
    range_values: np.ndarray,
    sum_values: np.ndarray,
    hh_labels: list[str],
    metadata: dict[str, object],
) -> dict[str, object]:
    if not (len(range_values) == len(sum_values) == len(hh_labels)):
        raise AssertionError("modern workload columns are not aligned")
    if len(range_values) < 100_000:
        raise AssertionError(f"{name} has fewer than 100,000 admitted records")
    hh_values, hh_metadata = encode_top_categories(hh_labels)
    if bool((range_values < 0).any()) or bool((range_values >= RANGE_DOMAIN).any()):
        raise AssertionError("range encoding left its public domain")
    if bool((sum_values < 0.0).any()) or bool((sum_values > 1.0).any()):
        raise AssertionError("bounded-sum encoding left [0,1]")

    source = SOURCES[name]
    directory = Path(source["dir"])
    clean_path = directory / "utility.npz"
    np.savez_compressed(
        clean_path,
        range_values=np.asarray(range_values, dtype=np.int16),
        bounded_sum_values=np.asarray(sum_values, dtype=np.float64),
        heavy_hitter_values=hh_values,
    )
    record = {
        "dataset": name,
        "snapshot": source["snapshot"],
        "source_url": source["url"],
        "source_sha256": source["sha256"],
        "clean_file": str(clean_path.relative_to(DATASET_DIR.parent)),
        "clean_sha256": sha256(clean_path),
        "admitted_rows": len(range_values),
        "range_domain_size": RANGE_DOMAIN,
        "heavy_hitter_domain_size": HH_DOMAIN,
        **hh_metadata,
        **metadata,
    }
    (directory / "metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def preprocess_acs() -> dict[str, object]:
    raw_path = ensure_raw("ACS 2024")
    ages: list[int] = []
    hours: list[float] = []
    occupations: list[str] = []
    raw_rows = 0
    with zipfile.ZipFile(raw_path) as archive, archive.open("psam_p36.csv") as binary:
        reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8", newline=""))
        for row in reader:
            raw_rows += 1
            age_text, hours_text, occupation = row["AGEP"], row["WKHP"], row["OCCP"]
            if not age_text or not hours_text or not occupation:
                continue
            age, weekly_hours = int(age_text), int(hours_text)
            if not (16 <= age <= 90 and 1 <= weekly_hours <= 99):
                continue
            ages.append(min(RANGE_DOMAIN - 1, (age - 16) * RANGE_DOMAIN // 75))
            hours.append(weekly_hours / 99.0)
            occupations.append(occupation)
    return write_snapshot(
        "ACS 2024",
        np.asarray(ages, dtype=np.int16),
        np.asarray(hours, dtype=float),
        occupations,
        {
            "raw_rows": raw_rows,
            "filter": "AGEP in [16,90], WKHP in [1,99], and nonmissing OCCP",
            "range_workload": "age over 256 public bins spanning [16,90]",
            "heavy_hitter_workload": "top-50 occupation codes",
            "bounded_sum_workload": "weekly hours divided by 99",
            "sampling": "without replacement within each epoch",
        },
    )


def preprocess_tlc() -> dict[str, object]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised in artifact setup
        raise RuntimeError("TLC preprocessing requires duckdb>=1.3") from exc

    raw_path = ensure_raw("NYC TLC 2024")
    connection = duckdb.connect()
    raw_rows = int(
        connection.execute("SELECT count(*) FROM read_parquet(?)", [str(raw_path)]).fetchone()[0]
    )
    predicate = (
        "passenger_count BETWEEN 1 AND 8 AND trip_distance BETWEEN 0 AND 50 "
        "AND total_amount BETWEEN 0 AND 200 "
        "AND PULocationID BETWEEN 1 AND 265 AND DOLocationID BETWEEN 1 AND 265"
    )
    filtered_rows = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet(?) WHERE {predicate}", [str(raw_path)]
        ).fetchone()[0]
    )
    rows = connection.execute(
        f"""
        SELECT trip_distance, total_amount, PULocationID, DOLocationID
        FROM read_parquet(?)
        WHERE {predicate}
        ORDER BY hash(VendorID, tpep_pickup_datetime, tpep_dropoff_datetime,
                      PULocationID, DOLocationID, trip_distance, total_amount)
        LIMIT {TLC_RESERVOIR_ROWS}
        """,
        [str(raw_path)],
    ).fetchall()
    connection.close()
    distance = np.fromiter((float(row[0]) for row in rows), dtype=float)
    total = np.fromiter((float(row[1]) for row in rows), dtype=float)
    routes = [f"{int(row[2]):03d}-{int(row[3]):03d}" for row in rows]
    bins = np.minimum(RANGE_DOMAIN - 1, (distance * RANGE_DOMAIN / 50.0).astype(np.int16))
    return write_snapshot(
        "NYC TLC 2024",
        bins,
        total / 200.0,
        routes,
        {
            "raw_rows": raw_rows,
            "filtered_rows": filtered_rows,
            "reservoir_rows": len(rows),
            "filter": predicate,
            "range_workload": "trip distance over 256 public bins spanning [0,50] miles",
            "heavy_hitter_workload": "top-50 pickup-dropoff zone pairs",
            "bounded_sum_workload": "total amount divided by 200",
            "sampling": "deterministic hash reservoir, then without replacement within each epoch",
            "duckdb_version": duckdb.__version__,
        },
    )


def preprocess_amazon() -> dict[str, object]:
    raw_path = ensure_raw("Amazon Reviews'23")
    cutoff = int(dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    end = int(dt.datetime(2023, 10, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    timestamps: list[int] = []
    ratings: list[float] = []
    products: list[str] = []
    raw_rows = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw_rows += 1
            row = json.loads(line)
            timestamp = int(row.get("timestamp", 0))
            rating = float(row.get("rating", 0.0))
            product = str(row.get("parent_asin", ""))
            if not (cutoff <= timestamp < end and 1.0 <= rating <= 5.0 and product):
                continue
            timestamps.append(timestamp)
            ratings.append(rating / 5.0)
            products.append(product)
    span = end - cutoff
    bins = np.minimum(
        RANGE_DOMAIN - 1,
        ((np.asarray(timestamps, dtype=np.int64) - cutoff) * RANGE_DOMAIN // span).astype(np.int16),
    )
    return write_snapshot(
        "Amazon Reviews'23",
        bins,
        np.asarray(ratings, dtype=float),
        products,
        {
            "raw_rows": raw_rows,
            "filter": "All Beauty reviews from 2018-01-01 through 2023-09-30 with valid rating and parent_asin",
            "range_workload": "review timestamp over 256 bins spanning the admitted interval",
            "heavy_hitter_workload": "top-50 parent products",
            "bounded_sum_workload": "rating divided by 5",
            "sampling": "without replacement within each epoch",
        },
    )


def generate_all() -> dict[str, int]:
    ensure_dirs()
    records = [preprocess_acs(), preprocess_tlc(), preprocess_amazon()]
    return {str(record["clean_file"]): int(record["admitted_rows"]) for record in records}


def ensure_modern_utility_datasets() -> dict[str, Path]:
    expected = {
        name: Path(source["dir"]) / "utility.npz" for name, source in SOURCES.items()
    }
    if not all(path.exists() for path in expected.values()):
        generate_all()
    return expected


if __name__ == "__main__":
    for filename, rows in generate_all().items():
        print(f"{filename}: {rows} rows")
