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
        self.assertEqual(
            len(rows),
            2 * len(cgp.REPORT_COUNTS) + len(cgp.DOPPIO_TABLE5_ROWS) + 1,
        )
        self.assertEqual(
            {row["m"] for row in rows if str(row["method"]).startswith("CGP")},
            set(cgp.REPORT_COUNTS),
        )
        self.assertEqual(
            {row["source_type"] for row in rows},
            {
                "literature_derived_analytic_cost",
                "author_derived_analytic_scenario",
                "literature_reported_measurement",
                "author_extrapolated_literature_model",
            },
        )
        self.assertEqual({row["measured"] for row in rows}, {"false"})
        self.assertEqual(
            {row["citation_key"] for row in rows},
            {"DBLP:conf/asiacrypt/ChaseGP20", "dong2025doppio"},
        )
        for row in rows:
            self.assertTrue(row["formula_scope"])
            self.assertTrue(math.isfinite(float(row["cost_seconds"])))
            self.assertGreater(float(row["cost_seconds"]), 0.0)
        cgp_rows = [row for row in rows if str(row["method"]).startswith("CGP")]
        self.assertEqual(
            {row["eprint_url"] for row in cgp_rows},
            {"https://eprint.iacr.org/2019/1340"},
        )

    def test_doppio_table5_rows_are_scoped(self) -> None:
        rows = [
            row
            for row in cgp.model_rows()
            if row["method"] == "Doppio 2+1 online shuffle (paper Table 5)"
        ]
        self.assertEqual(len(rows), 4)
        row = next(
            row
            for row in rows
            if row["m"] == 100_000 and row["payload_bits"] == 64
        )
        self.assertEqual(row["source_type"], "literature_reported_measurement")
        self.assertEqual(row["citation_key"], "dong2025doppio")
        self.assertAlmostEqual(row["cost_seconds"], 13.17)
        self.assertAlmostEqual(row["online_execution_comm_mb"], 0.0)
        self.assertAlmostEqual(row["reconstruction_comm_mb"], 1.60)
        self.assertAlmostEqual(row["communication_mb"], 1.60)

    def test_doppio_extrapolated_row_is_scoped(self) -> None:
        row = cgp.doppio_extrapolated_64bit_row()
        exponent = math.log(13.17 / 0.42) / math.log(10)
        self.assertEqual(row["m"], 1_000_000)
        self.assertEqual(row["payload_bits"], 64)
        self.assertEqual(row["source_type"], "author_extrapolated_literature_model")
        self.assertEqual(row["measured"], "false")
        self.assertAlmostEqual(row["scaling_exponent"], exponent)
        self.assertAlmostEqual(row["cost_seconds"], 13.17 * 10**exponent)
        self.assertAlmostEqual(row["communication_mb"], 16.0)
        self.assertIn("not a reported measurement", row["formula_scope"])

    def test_cost_function_rejects_non_benes_domains(self) -> None:
        with self.assertRaises(ValueError):
            cgp.permute_share_cost(100_000, 8)
        with self.assertRaises(ValueError):
            cgp.permute_share_cost(131_072, 20)


if __name__ == "__main__":
    unittest.main()
