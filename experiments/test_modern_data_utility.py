from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import exact_grr_pair_accountant as accountant
import modern_data_utility as utility


class ModernDataUtilityTests(unittest.TestCase):
    def test_calibration_cache_separates_mechanism_families(self) -> None:
        def fake_grr(**kwargs: object) -> tuple[str, object, object]:
            return ("grr", kwargs["method"], kwargs["domain_size"])

        def fake_laplace(**kwargs: object) -> tuple[str, object]:
            return ("laplace", kwargs["method"])

        with patch.object(utility, "calibrate_grr_epsilon", side_effect=fake_grr), patch.object(
            utility, "calibrate_laplace_epsilon", side_effect=fake_laplace
        ):
            calibrations = utility._calibrations()

        self.assertEqual(len(calibrations), 18)
        self.assertEqual(
            calibrations[("categorical", "Padded Ideal", 256, 0.1)],
            ("grr", "Padded Ideal", 256),
        )
        self.assertEqual(
            calibrations[("bounded_sum", "Padded Ideal", 256, 0.1)],
            ("laplace", "Padded Ideal"),
        )

    def test_clean_snapshots_are_aligned_and_bounded(self) -> None:
        datasets = utility._load_datasets()
        self.assertEqual(
            {str(dataset["name"]) for dataset in datasets},
            {"Synthetic", "ACS 2024", "NYC TLC 2024", "Amazon Reviews'23"},
        )
        for dataset in datasets:
            if bool(dict(dataset["metadata"]).get("synthetic", False)):
                continue
            range_values = np.asarray(dataset["range"])
            sum_values = np.asarray(dataset["sum"])
            hh_values = np.asarray(dataset["hh"])
            self.assertEqual(len(range_values), len(sum_values))
            self.assertEqual(len(range_values), len(hh_values))
            self.assertGreaterEqual(len(range_values), utility.EPOCH_SIZE)
            self.assertTrue(np.all((range_values >= 0) & (range_values < 256)))
            self.assertTrue(np.all((sum_values >= 0.0) & (sum_values <= 1.0)))
            self.assertTrue(np.all((hh_values >= 0) & (hh_values < 4096)))

    def test_synthetic_epoch_uses_declared_zipf_and_beta_laws(self) -> None:
        range_values, sum_values, hh_values = utility._sample_synthetic_epoch(
            np.random.default_rng(260715)
        )
        self.assertEqual(len(range_values), utility.EPOCH_SIZE)
        self.assertEqual(len(sum_values), utility.EPOCH_SIZE)
        self.assertEqual(len(hh_values), utility.EPOCH_SIZE)
        self.assertTrue(np.all((range_values >= 0) & (range_values < 256)))
        self.assertTrue(np.all((sum_values >= 0.0) & (sum_values <= 1.0)))
        self.assertTrue(np.all((hh_values >= 0) & (hh_values < 4096)))
        self.assertAlmostEqual(float(np.mean(sum_values)), 2.0 / 7.0, delta=0.01)
        range_hist = np.bincount(range_values, minlength=256)
        hh_hist = np.bincount(hh_values, minlength=4096)
        self.assertGreater(int(range_hist[0]), int(range_hist[-1]))
        self.assertGreater(int(hh_hist[0]), int(hh_hist[-1]))

    def test_pool_preserves_discrete_laplace_sum_at_fixed_epsilon(self) -> None:
        values = np.linspace(0.0, 1.0, utility.EPOCH_SIZE)
        class Config:
            def __init__(self, method: str) -> None:
                self.method = method
                self.local_epsilon0 = 6.0

        rng = np.random.default_rng(19)
        slots = rng.integers(0, utility.SLOT_COUNT, size=utility.EPOCH_SIZE)
        quantization_uniforms = rng.random(utility.EPOCH_SIZE)
        noise_uniforms = rng.random(utility.EPOCH_SIZE)
        ideal = utility._sum_observations(
            values,
            Config("Padded Ideal"),
            slots,
            quantization_uniforms,
            noise_uniforms,
        )
        pool = utility._sum_observations(
            values,
            Config("Pool"),
            slots,
            quantization_uniforms,
            noise_uniforms,
        )
        self.assertEqual(ideal["released_total"], pool["released_total"])
        self.assertEqual(ideal["estimate"], pool["estimate"])
        self.assertEqual(pool["pool_sum_preserved"], 1)
        self.assertLess(pool["max_occupancy"], 16)

    def test_categorical_release_has_fixed_public_length(self) -> None:
        domain_size = 32
        raw = np.arange(utility.EPOCH_SIZE, dtype=np.int64) % domain_size
        calibration = accountant.calibrate_local_epsilon(
            method="Drop",
            domain_size=domain_size,
            epoch_size=utility.EPOCH_SIZE,
            slot_count=utility.SLOT_COUNT,
            release_length=utility.RELEASE_LENGTH,
            target_epsilon=0.3,
            delta=utility.DELTA,
        )
        rng = np.random.default_rng(31)
        slots = rng.integers(0, utility.SLOT_COUNT, size=utility.EPOCH_SIZE)
        occupancies = np.bincount(slots, minlength=utility.SLOT_COUNT)
        survivor_mask = occupancies[slots] == 1
        report_uniforms = rng.random(utility.EPOCH_SIZE)
        report_alternatives = rng.integers(
            0, domain_size - 1, size=utility.EPOCH_SIZE
        )
        dummy_uniforms = rng.random(utility.RELEASE_LENGTH)
        dummy_alternatives = rng.integers(
            0, domain_size - 1, size=utility.RELEASE_LENGTH
        )
        release, real_outputs = utility._categorical_release_histogram(
            raw,
            calibration,
            survivor_mask,
            report_uniforms,
            report_alternatives,
            dummy_uniforms,
            dummy_alternatives,
        )
        self.assertEqual(int(release.sum()), utility.RELEASE_LENGTH)
        self.assertEqual(real_outputs, int(survivor_mask.sum()))
        estimate = utility._estimate_histogram(release, calibration)
        self.assertAlmostEqual(float(estimate.sum()), utility.EPOCH_SIZE, places=7)
        self.assertTrue(np.all(estimate >= 0.0))


if __name__ == "__main__":
    unittest.main()
