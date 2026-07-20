from __future__ import annotations

from build_artifact_bundle import RAW_DATASET_NAMES, ROOT, collect_files


def test_release_file_set_contains_reproduction_inputs_without_raw_downloads() -> None:
    relatives = {path.relative_to(ROOT).as_posix() for path in collect_files()}
    required = {
        "README.md",
        "ARTIFACT.md",
        "requirements.txt",
        "experiments/artifact_check.py",
        "experiments/build_gpu_dpf.py",
        "experiments/gpu_dpf_interface.py",
        "experiments/run_all.py",
        "experiments/validate_results.py",
        "data/fig_modern_utility.csv",
        "data/fig_modern_utility_replicates.csv",
        "datasets/acs2024/utility.npz",
        "datasets/nyc_tlc2024/utility.npz",
        "datasets/amazon_reviews_2023/utility.npz",
        "third_party/distributed_point_functions/LICENSE",
        "third_party/GPU-DPF/LICENSE",
        "third_party/GPU-DPF/VENDORED_COMMIT",
        "third_party/GPU-DPF/dpf_base/dpf.cc",
    }
    assert required <= relatives
    assert not {path.split("/")[-1] for path in relatives} & RAW_DATASET_NAMES
    assert not any(path.startswith("figures/") for path in relatives)
    assert "experiments/make_figures.py" not in relatives
    assert "experiments/make_modern_utility_figure.py" not in relatives
    assert all("__pycache__" not in path and not path.endswith(".DS_Store") for path in relatives)
