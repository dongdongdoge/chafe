from __future__ import annotations

import math
import unittest

import cgp_shuffle_model as cgp


class CgpShuffleModelTest(unittest.TestCase):
    def test_published_formula_at_one_hundred_thousand_reports(self) -> None:
        rows = cgp.model_rows((100_000,))
        full = next(
            row
            for row in rows
            if row["method"]
            == "CGP full secret-shared shuffle (paper-derived)"
        )
        payload_only = next(
            row
            for row in rows
            if row["method"]
            == "CGP payload-only ideal (author-derived)"
        )

        self.assertEqual(full["N"], 131_072)
        self.assertEqual(full["T"], 8)
        self.assertEqual(full["d"], 11)
        self.assertAlmostEqual(full["full_shuffle_seconds"], 3.815899136)
        self.assertAlmostEqual(full["cost_seconds"], 3.815899136)
        self.assertAlmostEqual(
            payload_only["cost_seconds"], 2 * 131_072 * 64 / 1_000_000_000
        )

    def test_grid_and_provenance_are_explicit(self) -> None:
        rows = cgp.model_rows()
        self.assertEqual(len(rows), 2 * len(cgp.REPORT_COUNTS))
        self.assertEqual({row["m"] for row in rows}, set(cgp.REPORT_COUNTS))
        self.assertEqual(
            {row["source_type"] for row in rows},
            {"literature_derived_analytic_cost", "author_derived_analytic_scenario"},
        )
        self.assertEqual({row["measured"] for row in rows}, {"false"})
        self.assertEqual(
            {row["citation_key"] for row in rows},
            {"DBLP:conf/asiacrypt/ChaseGP20"},
        )
        self.assertEqual(
            {row["eprint_url"] for row in rows},
            {"https://eprint.iacr.org/2019/1340"},
        )
        for row in rows:
            self.assertTrue(row["formula_scope"])
            self.assertTrue(math.isfinite(float(row["cost_seconds"])))
            self.assertGreater(float(row["cost_seconds"]), 0.0)

    def test_cost_function_rejects_non_benes_domains(self) -> None:
        with self.assertRaises(ValueError):
            cgp.permute_share_cost(100_000, 8)
        with self.assertRaises(ValueError):
            cgp.permute_share_cost(131_072, 20)


if __name__ == "__main__":
    unittest.main()
