"""Checks for the continual-learning stream metrics.

The four summary numbers are checked against a three-task matrix small enough
to work out by hand, so a wrong index or a wrong denominator shows up.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.stream_metrics import (accuracy_matrix, across_seeds, backward_transfer,
                                  bootstrap_interval, diagonal, final_accuracy,
                                  forgetting, learning_accuracy, slope, summary)


# After task 0: task 0 scores 60.
# After task 1: task 0 scores 50, task 1 scores 70.
# After task 2: task 0 scores 40, task 1 scores 55, task 2 scores 30.
HAND = [[60.0, None, None],
        [50.0, 70.0, None],
        [40.0, 55.0, 30.0]]


class HandWorkedTests(unittest.TestCase):
    def test_diagonal(self):
        self.assertEqual(diagonal(HAND), [60.0, 70.0, 30.0])

    def test_final_accuracy(self):
        # (40 + 55 + 30) / 3
        self.assertAlmostEqual(final_accuracy(HAND), 125.0 / 3)

    def test_learning_accuracy(self):
        # (60 + 70 + 30) / 3
        self.assertAlmostEqual(learning_accuracy(HAND), 160.0 / 3)

    def test_forgetting_uses_the_peak_not_the_first_score(self):
        # Task 0's best before the last row is max(60, 50) = 60, ending at 40,
        # so it drops 20. Task 1's best is 70, ending at 55, so it drops 15.
        self.assertAlmostEqual(forgetting(HAND), (20.0 + 15.0) / 2)

    def test_forgetting_excludes_the_last_task(self):
        # Task 2 is learned last and never revisited, so it cannot forget.
        # Averaging over 2 tasks rather than 3 is what makes this 17.5.
        self.assertAlmostEqual(forgetting(HAND), 17.5)

    def test_backward_transfer(self):
        # Task 0: 40 - 60 = -20. Task 1: 55 - 70 = -15.
        self.assertAlmostEqual(backward_transfer(HAND), -17.5)

    def test_backward_transfer_is_positive_when_tasks_improve(self):
        rising = [[60.0, None], [70.0, 50.0]]
        self.assertAlmostEqual(backward_transfer(rising), 10.0)
        # A task that only improved has not forgotten anything.
        self.assertAlmostEqual(forgetting(rising), -10.0)


class SlopeTests(unittest.TestCase):
    def test_exact_straight_line(self):
        self.assertAlmostEqual(slope([10.0, 20.0, 30.0, 40.0]), 10.0)

    def test_falling_curve_is_negative(self):
        self.assertLess(slope([50.0, 45.0, 30.0]), 0)

    def test_flat_curve_is_zero(self):
        self.assertAlmostEqual(slope([7.0, 7.0, 7.0]), 0.0)

    def test_per_ten_tasks_is_ten_times_per_task(self):
        values = summary(HAND)
        self.assertAlmostEqual(values["learning_curve_slope_per_10_tasks"],
                               10 * values["learning_curve_slope_per_task"])


class MatrixBuildingTests(unittest.TestCase):
    def history(self):
        return [
            {"action": "learn", "task": "3", "after": {"3": 60.0}},
            {"action": "learn", "task": "0", "after": {"3": 50.0, "0": 70.0}},
            {"action": "learn", "task": "9", "after": {"3": 40.0, "0": 55.0, "9": 30.0}},
        ]

    def test_builds_the_hand_matrix(self):
        self.assertEqual(accuracy_matrix(self.history(), ["3", "0", "9"]), HAND)

    def test_rejects_a_forget_request(self):
        history = self.history()
        history[1]["action"] = "forget"
        with self.assertRaises(ValueError):
            accuracy_matrix(history, ["3", "0", "9"])

    def test_rejects_a_different_order(self):
        with self.assertRaises(ValueError):
            accuracy_matrix(self.history(), ["0", "3", "9"])

    def test_rejects_a_missing_accuracy(self):
        history = self.history()
        del history[2]["after"]["3"]
        with self.assertRaises(ValueError):
            accuracy_matrix(history, ["3", "0", "9"])


class AcrossSeedsTests(unittest.TestCase):
    def test_reports_every_seed_and_their_mean(self):
        other = [[62.0, None, None], [52.0, 72.0, None], [42.0, 57.0, 32.0]]
        result = across_seeds({"0": HAND, "1": other})
        self.assertEqual(result["n_seeds"], 2)
        final = result["across_seeds"]["final_accuracy"]
        self.assertAlmostEqual(final["values"][0], 125.0 / 3)
        self.assertAlmostEqual(final["values"][1], 131.0 / 3)
        self.assertAlmostEqual(final["mean"], (125.0 / 3 + 131.0 / 3) / 2)
        self.assertLessEqual(final["ci_low"], final["mean"])
        self.assertLessEqual(final["mean"], final["ci_high"])

    def test_bootstrap_is_reproducible_and_brackets_the_mean(self):
        values = [10.0, 12.0, 20.0]
        first = bootstrap_interval(values, resamples=2000, seed=7)
        self.assertEqual(first, bootstrap_interval(values, resamples=2000, seed=7))
        self.assertLessEqual(first[0], 14.0)
        self.assertLessEqual(14.0, first[1])

    def test_identical_seeds_give_a_zero_width_interval(self):
        # Every resample is the same number, so the interval collapses. This is
        # what makes a three-seed interval worth reporting with its seed count.
        low, high = bootstrap_interval([5.0, 5.0, 5.0], resamples=500, seed=1)
        self.assertEqual((low, high), (5.0, 5.0))


if __name__ == "__main__":
    unittest.main()
