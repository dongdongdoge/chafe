from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DPF_ROOT = ROOT / "third_party" / "distributed_point_functions"


def build_google_dpf() -> None:
    env = os.environ.copy()
    for key in ["CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"]:
        env.pop(key, None)
    env.setdefault("BAZELISK_HOME", "/private/tmp/chafe-bazelisk-7")
    env.setdefault("USE_BAZEL_VERSION", "7.4.1")

    cmd = [
        "bazel",
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
    ]
    subprocess.run(cmd, cwd=DPF_ROOT, env=env, check=True)


if __name__ == "__main__":
    build_google_dpf()

