"""Fixed-local-budget control for the modern utility workloads.

This experiment reruns the complete-release utility pipeline at one common
local privacy budget.  It isolates the decoder's information loss from the
larger local budget that Drop can obtain after central-budget calibration.
No central-DP claim is attached to these rows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import fetch_modern_utility_datasets as modern_data
import modern_data_utility as utility
from chafe_eval_lib import DATA_DIR, mean_and_se, write_csv
from exact_grr_pair_accountant import singleton_survival


FIXED_EPS0 = 8.0
PRIVACY_SCOPE = "fixed_local_budget_utility_control_no_central_dp_claim"
UTILITY_SOURCE = "monte_carlo_complete_release_histogram"


@dataclass(frozen=True)
class FixedBudgetConfig:
    method: str
    domain_size: int
    local_epsilon0: float
    survival: float


def _config(method: str, domain_size: int) -> FixedBudgetConfig:
    survival = (
        singleton_survival(utility.EPOCH_SIZE, utility.SLOT_COUNT)
        if method == "Drop"
        else 1.0
    )
    return FixedBudgetConfig(method, domain_size, FIXED_EPS0, survival)


def run_experiment() -> None:
    datasets = utility._load_datasets()
    replicate_rows: list[dict[str, object]] = []
    range_low, range_high = utility._range_queries()

    for dataset_index, dataset in enumerate(datasets):
        name = str(dataset["name"])
        metadata = dict(dataset["metadata"])
        for replicate in range(utility.REPLICATES):
            epoch_rng = np.random.default_rng(
                utility._seed(dataset_index, replicate, 0, 73, 0)
            )
            if bool(metadata.get("synthetic", False)):
                range_values, sum_values, hh_values = utility._sample_synthetic_epoch(
                    epoch_rng
                )
            else:
                indices = utility._sample_epoch(dataset, epoch_rng)
                range_values = np.asarray(dataset["range"])[indices]
                sum_values = np.asarray(dataset["sum"])[indices]
                hh_values = np.asarray(dataset["hh"])[indices]

            range_truth = np.bincount(
                range_values, minlength=modern_data.RANGE_DOMAIN
            )
            hh_truth = np.bincount(hh_values, minlength=modern_data.HH_DOMAIN)

            placement_rng = np.random.default_rng(
                utility._seed(dataset_index, replicate, 0, 73, 7)
            )
            slots = placement_rng.integers(
                0, utility.SLOT_COUNT, size=utility.EPOCH_SIZE
            )
            occupancies = np.bincount(slots, minlength=utility.SLOT_COUNT)
            survivor_mask = occupancies[slots] == 1

            for task_index, (task, raw, truth, domain_size) in enumerate(
                (
                    (
                        "Range",
                        range_values,
                        range_truth,
                        modern_data.RANGE_DOMAIN,
                    ),
                    (
                        "Heavy hitter",
                        hh_values,
                        hh_truth,
                        modern_data.HH_DOMAIN,
                    ),
                ),
                start=1,
            ):
                coupling_rng = np.random.default_rng(
                    utility._seed(dataset_index, replicate, task_index, 73, 91)
                )
                report_uniforms = coupling_rng.random(utility.EPOCH_SIZE)
                report_alternatives = coupling_rng.integers(
                    0, domain_size - 1, size=utility.EPOCH_SIZE
                )
                dummy_uniforms = coupling_rng.random(utility.RELEASE_LENGTH)
                dummy_alternatives = coupling_rng.integers(
                    0, domain_size - 1, size=utility.RELEASE_LENGTH
                )

                for method in ("Padded Ideal", "Drop"):
                    config = _config(method, domain_size)
                    release, real_outputs = utility._categorical_release_histogram(
                        raw,
                        config,
                        survivor_mask,
                        report_uniforms,
                        report_alternatives,
                        dummy_uniforms,
                        dummy_alternatives,
                    )
                    estimate = utility._estimate_histogram(release, config)
                    if task == "Range":
                        value = utility._range_mean_relative_error(
                            truth, estimate, range_low, range_high
                        )
                        metric = "mean_relative_range_error"
                    else:
                        value = utility._top_k_f1(truth, estimate)
                        metric = "top_50_f1"
                    replicate_rows.append(
                        {
                            "dataset": name,
                            "task": task,
                            "method": method,
                            "fixed_epsilon0": FIXED_EPS0,
                            "replicate": replicate,
                            "metric": metric,
                            "value": value,
                            "real_output_count": real_outputs,
                            "release_length": utility.RELEASE_LENGTH,
                            "rho": utility.RHO,
                            "tau": utility.TAU,
                            "dummy_law": "GRR(category_0)",
                            "privacy_scope": PRIVACY_SCOPE,
                            "source_type": "monte_carlo_fixed_eps0",
                            "privacy_source_type": "not_applicable_fixed_local_budget_control",
                            "utility_source_type": UTILITY_SOURCE,
                            "utility_source": UTILITY_SOURCE,
                            "pairing": "shared_epoch_placement_and_common_random_numbers",
                            "source_snapshot": metadata["snapshot"],
                            "source_sha256": metadata["source_sha256"],
                            "clean_sha256": metadata["clean_sha256"],
                        }
                    )

            sum_rng = np.random.default_rng(
                utility._seed(dataset_index, replicate, 3, 73, 0)
            )
            sum_quantization_uniforms = sum_rng.random(utility.EPOCH_SIZE)
            sum_noise_uniforms = sum_rng.random(utility.EPOCH_SIZE)
            observations = {
                method: utility._sum_observations(
                    sum_values,
                    _config(method, utility.LAPLACE_GROUP_SIZE),
                    slots,
                    sum_quantization_uniforms,
                    sum_noise_uniforms,
                )
                for method in ("Padded Ideal", "Pool")
            }
            ideal = observations["Padded Ideal"]
            pool = observations["Pool"]
            if (
                ideal["released_total"] != pool["released_total"]
                or ideal["estimate"] != pool["estimate"]
            ):
                raise AssertionError("fixed-epsilon Pool did not preserve ideal sum")
            for method, observation in observations.items():
                true_sum = float(observation["true_sum"])
                relative_error = float(observation["relative_rmse"])
                replicate_rows.append(
                    {
                        "dataset": name,
                        "task": "Bounded sum",
                        "method": method,
                        "fixed_epsilon0": FIXED_EPS0,
                        "replicate": replicate,
                        "metric": "relative_sum_rmse",
                        "value": relative_error,
                        "true_sum": true_sum,
                        "estimate": observation["estimate"],
                        "released_total": observation["released_total"],
                        "real_output_count": observation["real_output_count"],
                        "release_length": utility.RELEASE_LENGTH,
                        "max_occupancy": observation["max_occupancy"],
                        "pool_ideal_total_equal": int(
                            ideal["released_total"] == pool["released_total"]
                        ),
                        "pool_ideal_estimate_equal": int(
                            ideal["estimate"] == pool["estimate"]
                        ),
                        "pool_sum_preserved": observation["pool_sum_preserved"],
                        "fixed_point_scale": observation["fixed_point_scale"],
                        "error_variance": observation["error_variance"],
                        "rho": utility.RHO,
                        "tau": utility.TAU,
                        "dummy_law": "point_mass_at_additive_identity",
                        "privacy_scope": PRIVACY_SCOPE,
                        "source_type": "monte_carlo_fixed_eps0",
                        "privacy_source_type": "not_applicable_fixed_local_budget_control",
                        "utility_source_type": UTILITY_SOURCE,
                        "utility_source": UTILITY_SOURCE,
                        "pairing": "shared_epoch_placement_and_laplace_uniforms",
                        "source_snapshot": metadata["snapshot"],
                        "source_sha256": metadata["source_sha256"],
                        "clean_sha256": metadata["clean_sha256"],
                    }
                )

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in replicate_rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["task"]),
                str(row["method"]),
                str(row["metric"]),
            )
        ].append(float(row["value"]))

    summary_rows: list[dict[str, object]] = []
    for (dataset, task, method, metric), values in grouped.items():
        ideal_values = grouped[(dataset, task, "Padded Ideal", metric)]
        mean, se = mean_and_se(values)
        paired_difference = np.asarray(values) - np.asarray(ideal_values)
        difference_mean, difference_se = mean_and_se(paired_difference)
        ideal_mean = float(np.mean(ideal_values))
        summary_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "method": method,
                "fixed_epsilon0": FIXED_EPS0,
                "metric": metric,
                "mean": mean,
                "se": se,
                "replicates": len(values),
                "paired_difference_from_padded_ideal": difference_mean,
                "paired_difference_se": difference_se,
                "ratio_to_padded_ideal": mean / ideal_mean if ideal_mean else 1.0,
                "rho": utility.RHO,
                "tau": utility.TAU,
                "privacy_scope": PRIVACY_SCOPE,
                "source_type": "monte_carlo_fixed_eps0",
                "privacy_source_type": "not_applicable_fixed_local_budget_control",
                "utility_source_type": UTILITY_SOURCE,
                "utility_source": UTILITY_SOURCE,
            }
        )

    write_csv(DATA_DIR / "fig_modern_fixed_eps0_utility_replicates.csv", replicate_rows)
    write_csv(DATA_DIR / "fig_modern_fixed_eps0_utility.csv", summary_rows)


def generate_all() -> dict[str, int]:
    run_experiment()
    return {
        "data/fig_modern_fixed_eps0_utility_replicates.csv": 2_400,
        "data/fig_modern_fixed_eps0_utility.csv": 24,
    }


if __name__ == "__main__":
    run_experiment()
