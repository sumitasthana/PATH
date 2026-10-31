"""Checks for the indicator reporting.

The long table is the only thing that reads the boundary records. Everything
else reads the long table. So most of these check that the flattening keeps
every measurement, drops everything that is not one, and puts the image set in
its own column instead of leaving it stuck to the measure's name.
"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.indicator_report import (curves, endpoints, long_table, read_rows,
                                    split_key, write_long_table)


def boundary(t, seed, order="order_01", **measures):
    """One boundary record, shaped the way BoundaryLog writes it."""
    row = {"study": "test", "order": order, "seed": seed, "t": t,
           "task": None if t == 0 else str(t),
           "delta_w_target": 0.0 if t == 0 else 0.1 * t,
           "effective_rank_in_stream": 10.0 + t,
           "effective_rank_eurosat": 8.0,
           "loss_progress_mean": None if t == 0 else 0.05,
           "loss_progress_curve": [[10, 0.05]]}
    row.update(measures)
    return row


def written(folder, rows, name="seed_0.jsonl"):
    """A boundary record file, written the way the log writes one."""
    path = Path(folder) / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class ReadRows(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_several_runs_read_into_one_list(self):
        """Three seeds are three files, and the table spans all of them."""
        first = written(self.folder.name, [boundary(0, 0), boundary(1, 0)])
        second = written(self.folder.name, [boundary(0, 1)], "seed_1.jsonl")
        self.assertEqual(len(read_rows(first, second)), 3)

    def test_a_blank_line_is_not_a_row(self):
        """A run killed mid-write can leave one, and it must not become a row."""
        path = Path(self.folder.name) / "seed_0.jsonl"
        path.write_text(json.dumps(boundary(0, 0)) + "\n\n", encoding="utf-8")
        self.assertEqual(len(read_rows(path)), 1)


class SplitKey(unittest.TestCase):

    def test_an_image_set_comes_off_the_end(self):
        self.assertEqual(split_key("effective_rank_in_stream"),
                         ("effective_rank", "in_stream"))

    def test_a_threshold_stays_with_its_measure(self):
        """The cut-off is part of the measure's name, not part of the set's."""
        self.assertEqual(split_key("dormant_0.025_eurosat"),
                         ("dormant_0.025", "eurosat"))

    def test_a_measure_with_no_image_set_keeps_its_whole_name(self):
        """Weight change depends on no images, so nothing may be cut off it."""
        self.assertEqual(split_key("delta_w_target"), ("delta_w_target", None))


class LongTable(unittest.TestCase):

    def flat(self, rows=None):
        return long_table(rows or [boundary(0, 0), boundary(1, 0)])

    def test_identity_fields_are_not_measurements(self):
        """Seed and t name the row; they are not things that were measured."""
        measures = {row["measure"] for row in self.flat()}
        self.assertNotIn("seed", measures)
        self.assertNotIn("t", measures)

    def test_the_curve_is_left_out(self):
        """It is a list, and the table holds one number per row."""
        self.assertNotIn("loss_progress_curve",
                         {row["measure"] for row in self.flat()})

    def test_a_missing_value_is_left_out_rather_than_blank(self):
        """Boundary zero has no task behind it, so it has no loss progress."""
        rows = [row for row in self.flat() if row["measure"] == "loss_progress_mean"]
        self.assertEqual([row["t"] for row in rows], [1])

    def test_the_image_set_gets_its_own_column(self):
        rows = [row for row in self.flat() if row["measure"] == "effective_rank"]
        self.assertEqual(sorted({row["image_set"] for row in rows}),
                         ["eurosat", "in_stream"])

    def test_a_measure_with_no_image_set_leaves_the_column_empty(self):
        rows = [row for row in self.flat() if row["measure"] == "delta_w_target"]
        self.assertEqual({row["image_set"] for row in rows}, {""})

    def test_the_run_names_the_order_and_the_seed(self):
        """Two orders at the same seed are two runs, not one line with a kink."""
        flat = self.flat([boundary(0, 0, "order_01"), boundary(0, 0, "order_02")])
        self.assertEqual(sorted({row["run"] for row in flat}),
                         ["order_01/seed_0", "order_02/seed_0"])


class WriteLongTable(unittest.TestCase):

    def test_the_columns_match_the_adapter_study(self):
        """Side by side without reshaping is the whole point of this shape."""
        with tempfile.TemporaryDirectory() as folder:
            path = write_long_table(long_table([boundary(1, 0)]),
                                    Path(folder) / "deeper" / "long.csv")
            with path.open(encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames,
                                 ["run", "order", "seed", "t", "task",
                                  "image_set", "measure", "value"])
                self.assertTrue(list(reader))


class Endpoints(unittest.TestCase):

    def summary(self):
        flat = long_table([boundary(0, seed) for seed in (0, 1)] +
                          [boundary(1, seed) for seed in (0, 1)])
        return {(row["measure"], row["image_set"]): row for row in endpoints(flat)}

    def test_the_change_is_the_last_boundary_minus_the_first(self):
        row = self.summary()[("delta_w_target", "")]
        self.assertAlmostEqual(row["start_mean"], 0.0)
        self.assertAlmostEqual(row["end_mean"], 0.1)
        self.assertAlmostEqual(row["change"], 0.1)

    def test_the_spread_is_across_runs(self):
        """Two seeds that agree exactly have no spread, which is the test."""
        self.assertAlmostEqual(self.summary()[("effective_rank", "eurosat")]["end_sd"], 0.0)

    def test_the_run_count_is_reported(self):
        """A change over two seeds is weaker evidence than one over three."""
        self.assertEqual(self.summary()[("delta_w_target", "")]["runs"], 2)

    def test_one_boundary_gives_no_change(self):
        """A run stopped after boundary zero has nothing to compare against."""
        rows = {(r["measure"], r["image_set"]): r
                for r in endpoints(long_table([boundary(0, 0)]))}
        self.assertIsNone(rows[("delta_w_target", "")]["change"])

    def test_image_sets_are_kept_apart(self):
        """Moving on the stream's images only is a different claim from moving."""
        summary = self.summary()
        self.assertAlmostEqual(summary[("effective_rank", "in_stream")]["change"], 1.0)
        self.assertAlmostEqual(summary[("effective_rank", "eurosat")]["change"], 0.0)


class Curves(unittest.TestCase):

    def test_one_figure_per_measure(self):
        flat = long_table([boundary(0, 0), boundary(1, 0)])
        with tempfile.TemporaryDirectory() as folder:
            paths = curves(flat, folder)
            names = sorted(path.name for path in paths)
            self.assertIn("fig_effective_rank.png", names)
            self.assertIn("fig_delta_w_target.png", names)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
