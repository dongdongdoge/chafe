# Chafe Shuffle Artifact Guide

This artifact separates four evidence classes: formula and exact-accountant
privacy results, Monte Carlo utility simulations, local DPF-backend
microbenchmarks, and analytic or external performance context. CSV provenance
fields record the class for each row.

## Main Commands

| Purpose | Command |
|---|---|
| Check the frozen artifact | `python experiments/artifact_check.py` |
| Regenerate non-measured outputs | `python experiments/run_all.py --skip-microbench --skip-utility` |
| Rerun all local experiments | `python experiments/build_google_dpf.py` then `python experiments/run_all.py` |
| Build a release bundle | `python experiments/build_artifact_bundle.py` |

The frozen check validates required files, CSV provenance, numerical
invariants, and the Python regression suite. It does not relabel checked-in
measurements or Monte Carlo rows as newly generated values.

## Evidence Map

| Result family | Files | Scope |
|---|---|---|
| Privacy formulas and exact accounting | `data/fig_privacy*.csv`, `data/fig_*account*.csv` | Closed-form curves and finite-domain accountants. |
| Workload utility | `data/fig_*utility*.csv`, `data/fig_*workload*.csv` | Seeded Monte Carlo simulations over processed snapshots. |
| DPF write backend | `data/fig_dpf_*.csv`, `data/fig_product_vs_dual_microbench.csv` | Local Google DPF microbenchmarks when regenerated; checked-in rows preserve their original provenance. |
| Capacity and latency models | `data/fig_ingestion*.csv`, `data/fig_finite_burst_latency.csv` | Analytic models from measured or literature-sourced inputs. |

Generated TikZ figures are reproducible from the CSVs and scripts, but they are
not tracked in this repository.

## Dataset Sources

Processed inputs are checked in under `datasets/`. Raw downloads are excluded.

- Adult/Census Income: UCI Machine Learning Repository,
  rebuilt by `experiments/fetch_adult_dataset.py`.
- Covertype: UCI Machine Learning Repository,
  rebuilt by `experiments/fetch_covertype_dataset.py`.
- ACS 2024 PUMS, NYC TLC 2024 Yellow Taxi, and Amazon Reviews'23:
  rebuilt by `experiments/fetch_modern_utility_datasets.py`.

Each dataset directory contains metadata with the upstream location and checksum
where applicable.

## Reading the CSVs

- `formula` denotes direct evaluation of closed-form expressions.
- `exact_accountant` denotes finite-domain hockey-stick accounting.
- `monte_carlo` denotes seeded workload simulation.
- `microbenchmark` denotes local wall-clock measurement.
- `analytic_cost` denotes a queueing, capacity, or latency calculation.
- `external_*` denotes cited third-party context, not a local Chafe
  measurement.
