"""Build a deterministic public-release ZIP with an internal SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "chafe-shuffle-artifact.zip"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)

TOP_LEVEL_FILES = (
    ".gitignore",
    "README.md",
    "ARTIFACT.md",
    "requirements.txt",
)
OPTIONAL_TOP_LEVEL_FILES = ("LICENSE", "CITATION.cff")
PROCESSED_DATASET_FILES = (
    "datasets/adult/adult_clean.csv",
    "datasets/adult/metadata.json",
    "datasets/covertype/covertype_selected.csv",
    "datasets/covertype/metadata.json",
    "datasets/acs2024/utility.npz",
    "datasets/acs2024/metadata.json",
    "datasets/nyc_tlc2024/utility.npz",
    "datasets/nyc_tlc2024/metadata.json",
    "datasets/amazon_reviews_2023/utility.npz",
    "datasets/amazon_reviews_2023/metadata.json",
)
RAW_DATASET_NAMES = {
    "adult.data",
    "adult.names",
    "adult.test",
    "covtype.data.gz",
    "csv_pny.zip",
    "yellow_tripdata_2024-01.parquet",
    "All_Beauty.jsonl.gz",
    "Digital_Music.jsonl.gz",
}


def is_release_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in path.parts):
        return False
    if path.name == ".DS_Store" or path.name in RAW_DATASET_NAMES:
        return False
    return True


def collect_files(root: Path = ROOT) -> list[Path]:
    files: set[Path] = set()
    for relative in (*TOP_LEVEL_FILES, *OPTIONAL_TOP_LEVEL_FILES, *PROCESSED_DATASET_FILES):
        path = root / relative
        if is_release_file(path):
            files.add(path)
    for folder, suffixes in (
        ("experiments", {".py", ".md"}),
        ("data", {".csv", ".txt"}),
        ("figures", {".tex"}),
    ):
        for path in (root / folder).rglob("*"):
            if path.suffix in suffixes and is_release_file(path):
                files.add(path)
    for path in (root / "third_party" / "distributed_point_functions").rglob("*"):
        if is_release_file(path):
            files.add(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build_bundle(output: Path, publication_ready: bool = False) -> tuple[int, str]:
    if publication_ready and not (ROOT / "LICENSE").is_file():
        raise RuntimeError(
            "publication-ready bundles require an author-selected top-level LICENSE"
        )
    files = collect_files()
    if not files:
        raise RuntimeError("release file set is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        entries.append((relative, path.read_bytes()))
    manifest = "".join(f"{sha256(data)}  {name}\n" for name, data in entries)
    build_metadata = {
        "schema": 1,
        "file_count_excluding_manifests": len(entries),
        "license_present": (ROOT / "LICENSE").is_file(),
        "raw_upstream_datasets_included": False,
        "raw_upstream_dataset_names_excluded": sorted(RAW_DATASET_NAMES),
    }
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in entries:
            write_member(archive, name, data)
        write_member(archive, "ARTIFACT_MANIFEST.sha256", manifest.encode())
        write_member(
            archive,
            "ARTIFACT_BUILD.json",
            (json.dumps(build_metadata, indent=2, sort_keys=True) + "\n").encode(),
        )
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
        names = set(archive.namelist())
        if {Path(name).name for name in names} & RAW_DATASET_NAMES:
            raise RuntimeError("raw upstream dataset leaked into release bundle")
    return len(entries), sha256(output.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--publication-ready",
        action="store_true",
        help="fail unless an author-selected top-level LICENSE is present",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count, digest = build_bundle(args.output.resolve(), args.publication_ready)
    print(f"Wrote {args.output.resolve()} with {count} files")
    print(f"SHA-256 {digest}")
