# Chafe Shuffle Artifact

This repository contains the evaluation artifact for Chafe Shuffle: experiment
scripts, checked-in CSV evidence, processed workload inputs, and the vendored
Google DPF backend used by the ingestion benchmark.

## Contents

- `experiments/`: formula generators, exact accountants, workload simulators,
  system benchmarks, figure builders, and validation checks.
- `data/`: CSVs used by the evaluation. Rows carry provenance fields such as
  `source_type`, `privacy_source_type`, or `utility_source_type`.
- `datasets/`: compact processed inputs and metadata. Raw upstream downloads
  are excluded from git.
- `third_party/distributed_point_functions/`: Google DPF source plus the Chafe
  packed64 write-backend benchmark.

## Setup

Use Python 3.9 or newer:

```bash
python -m pip install -r requirements.txt
```

The Python checks do not require network access. Re-running the Google DPF
microbenchmark additionally requires Bazel and a C++ toolchain.

## Data

The repository includes processed snapshots needed for offline reruns:

- `datasets/adult/adult_clean.csv`
- `datasets/covertype/covertype_selected.csv`
- `datasets/acs2024/utility.npz`
- `datasets/nyc_tlc2024/utility.npz`
- `datasets/amazon_reviews_2023/utility.npz`

To rebuild them from public sources, use:

```bash
python experiments/fetch_adult_dataset.py
python experiments/fetch_covertype_dataset.py
python experiments/fetch_modern_utility_datasets.py
```

Metadata in each dataset directory records the upstream source and checksum
where applicable.

## Reproduce

Run the non-destructive artifact check:

```bash
python experiments/artifact_check.py
```

Regenerate formulas, exact-accountant rows, analytic models, validation, and
TikZ figures while retaining checked-in measured and Monte Carlo CSVs:

```bash
python experiments/run_all.py --skip-microbench --skip-utility
```

Rerun all local experiments, including machine-specific measurements:

```bash
python experiments/build_google_dpf.py
python experiments/run_all.py
```

Generated TikZ files under `figures/` and manuscript build outputs are not
tracked by git. The checked-in `data/*.csv` files are the durable evidence.

## Provenance

CSV provenance fields distinguish formula-derived values, exact finite-domain
accounting, Monte Carlo utility simulations, local microbenchmarks, analytic
capacity or latency models, and external literature/artifact context. Analytic
and external rows are not local measurements.
