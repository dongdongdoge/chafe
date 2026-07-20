"""Fail-fast integrity check for the frozen Chafe evaluation artifact."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "README.md",
    "ARTIFACT.md",
    "requirements.txt",
    "experiments/run_all.py",
    "experiments/validate_results.py",
    "experiments/gpu_dpf_interface.py",
    "experiments/make_modern_utility_figure.py",
    "datasets/adult/adult_clean.csv",
    "datasets/adult/metadata.json",
    "datasets/covertype/covertype_selected.csv",
    "datasets/covertype/metadata.json",
    "data/fig_dpf_throughput_vs_n.csv",
    "data/fig_dpf_thread_scaling.csv",
    "data/fig_epoch_close_scaling.csv",
    "data/fig_finite_burst_latency.csv",
    "data/fig_privacy_large_m_accounting.csv",
    "data/fig_modern_utility.csv",
    "data/fig_modern_utility_calibration.csv",
    "data/fig_modern_utility_replicates.csv",
    "data/fig_modern_fixed_eps0_utility.csv",
    "data/fig_modern_fixed_eps0_utility_replicates.csv",
    "datasets/acs2024/utility.npz",
    "datasets/acs2024/metadata.json",
    "datasets/nyc_tlc2024/utility.npz",
    "datasets/nyc_tlc2024/metadata.json",
    "datasets/amazon_reviews_2023/utility.npz",
    "datasets/amazon_reviews_2023/metadata.json",
    "third_party/distributed_point_functions/LICENSE",
)


def run_checked(label: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate-software",
        action="store_true",
        help=(
            "regenerate formulas, exact-accountant rows, analytic models, and "
            "figures while retaining measured and Monte Carlo CSVs"
        ),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="run provenance/numerical validation without the pytest suite",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [relative for relative in REQUIRED_PATHS if not (ROOT / relative).is_file()]
    if missing:
        print("Artifact check failed; required files are missing:", file=sys.stderr)
        for relative in missing:
            print(f"- {relative}", file=sys.stderr)
        return 2

    if args.regenerate_software:
        run_checked(
            "software-only regeneration",
            [
                sys.executable,
                "experiments/run_all.py",
                "--skip-microbench",
                "--skip-utility",
            ],
        )
    else:
        run_checked(
            "frozen provenance and numerical validation",
            [sys.executable, "experiments/validate_results.py"],
        )

    if not args.skip_tests:
        test_files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "experiments").glob("test_*.py"))
        test_env = os.environ.copy()
        test_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        run_checked(
            "Python regression suite",
            [sys.executable, "-m", "pytest", "-q", *test_files],
            env=test_env,
        )

    print("\nArtifact integrity check passed.")
    if not (ROOT / "LICENSE").is_file():
        print(
            "Publication reminder: add a top-level author-selected LICENSE "
            "before publishing the artifact."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
