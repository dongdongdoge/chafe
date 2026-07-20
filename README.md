# Chafe Shuffle Artifact

This repository contains the runnable Chafe Shuffle evaluation artifact:
experiment scripts, checked-in CSV outputs, processed dataset snapshots, and
vendored DPF backends.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python experiments/artifact_check.py
```

This checks that the artifact files are present and that the checked-in outputs
are internally consistent.

## Datasets

Processed inputs are included under `datasets/`. To rebuild them from public
sources, run:

```bash
python experiments/fetch_adult_dataset.py
python experiments/fetch_covertype_dataset.py
python experiments/fetch_modern_utility_datasets.py
```

## DPF Backends

Google DPF is vendored under `third_party/distributed_point_functions/`.
GPU-DPF is vendored under `third_party/GPU-DPF/`.

```bash
python experiments/build_google_dpf.py
python experiments/build_gpu_dpf.py --check-only
python experiments/gpu_dpf_interface.py --print-contract
```
