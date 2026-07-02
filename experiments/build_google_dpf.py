from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DPF_ROOT = ROOT / "third_party" / "distributed_point_functions"


def bazel_binary() -> str:
    for candidate in [
        os.environ.get("CHAFE_BAZEL"),
        shutil.which("bazel"),
        shutil.which("bazelisk"),
        "/usr/local/bin/bazel",
        "/usr/local/bin/bazelisk",
        "/opt/homebrew/bin/bazel",
        "/opt/homebrew/bin/bazelisk",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Could not find bazel or bazelisk; set CHAFE_BAZEL=/path/to/bazel.")


def build_google_dpf() -> None:
    env = os.environ.copy()
    for key in ["CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"]:
        env.pop(key, None)
    env.setdefault("BAZELISK_HOME", "/private/tmp/chafe-bazelisk-7")
    env.setdefault("USE_BAZEL_VERSION", "7.4.1")

    machine = platform.machine().lower()
    optimization_copts = [
        "--copt=-O3",
        "--host_copt=-O3",
    ]
    if machine in {"x86_64", "amd64"} and os.environ.get("CHAFE_DISABLE_AES_NI") != "1":
        optimization_copts.extend(
            [
                "--copt=-maes",
                "--copt=-msse4.1",
                "--host_copt=-maes",
                "--host_copt=-msse4.1",
            ]
        )

    cmd = [
        bazel_binary(),
        "--output_user_root=/private/tmp/chafe-bazel-output-7",
        "build",
        "//chafe:google_dpf_write_benchmark",
        "--macos_minimum_os=10.14",
        "--host_copt=-mmacosx-version-min=10.14",
        "--host_linkopt=-mmacosx-version-min=10.14",
        "--copt=-nostdinc",
        "--copt=-isystem/Library/Developer/CommandLineTools/usr/include/c++/v1",
        "--copt=-isystem/Library/Developer/CommandLineTools/usr/lib/clang/12.0.0/include",
        "--copt=-isystem/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include",
        "--copt=-iframework/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks",
        *optimization_copts,
    ]
    subprocess.run(cmd, cwd=DPF_ROOT, env=env, check=True)


if __name__ == "__main__":
    build_google_dpf()
