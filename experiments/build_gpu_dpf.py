"""Check or build the vendored GPU-DPF backend."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_DPF_ROOT = ROOT / "third_party" / "GPU-DPF"

REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "README.md",
    "VENDORED_COMMIT",
    "setup.py",
    "dpf.py",
    "dpf_base/dpf.cc",
    "dpf_base/dpf.h",
    "dpf_gpu/dpf_benchmark.cu",
    "dpf_gpu/dpf/dpf_hybrid.cu",
    "dpf_gpu/prf/prf.cu",
    "dpf_wrapper.cu",
)


def check_source_tree() -> None:
    missing = [
        relative for relative in REQUIRED_SOURCE_FILES
        if not (GPU_DPF_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "vendored GPU-DPF source tree is incomplete: " + ", ".join(missing)
        )


def can_build_cuda_extension() -> tuple[bool, str]:
    if shutil.which("nvcc") is None:
        return False, "CUDA nvcc is not on PATH"
    if importlib.util.find_spec("torch") is None:
        return False, "PyTorch is not installed in this Python environment"
    return True, "CUDA and PyTorch are available"


def build_gpu_dpf(require_build: bool = False, check_only: bool = False) -> None:
    check_source_tree()
    print(f"GPU-DPF source tree: {GPU_DPF_ROOT}")
    if check_only:
        print("GPU-DPF source check passed.")
        return

    ready, reason = can_build_cuda_extension()
    if not ready:
        if require_build:
            raise RuntimeError(reason)
        print(f"GPU-DPF build skipped: {reason}.")
        return

    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=GPU_DPF_ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--require-build",
        action="store_true",
        help="fail instead of skipping when CUDA/PyTorch is unavailable",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_gpu_dpf(require_build=args.require_build, check_only=args.check_only)
