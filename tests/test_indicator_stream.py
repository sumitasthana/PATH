"""Checks for riding along with a real stream run.

The indicator set is only worth having if it is a passenger: it must come back
with one line per task and leave the run it measured bit for bit unchanged.
Both claims are checked here on a two-task stream small enough to run on a CPU.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_paired_diagnostic import target
from test_paired_study import Objects
from uncle.config import Config
from uncle.paired_study import StudyStore, jobs, make_plan, run_job
from uncle.plasticity_indicators import StreamObserver, fixed_image_sets
from uncle.storage import R2Store


class FixedImageSets(unittest.TestCase):

    def dataset(self, size=20):
        torch.manual_seed(1)
        return TensorDataset(torch.rand(size, 1, 8, 8), torch.arange(size) % 2)

    def test_the_same_pictures_every_time(self):
        """Two runs must differ in the model, never in what was pushed through."""
        data = self.dataset()
        first = fixed_image_sets({"in_stream": data}, count=5)
        second = fixed_image_sets({"in_stream": data}, count=5)
        torch.testing.assert_close(first["in_stream"], second["in_stream"],
                                   rtol=0, atol=0)

    def test_the_count_is_honoured(self):
        chosen = fixed_image_sets({"in_stream": self.dataset()}, count=5)
        self.assertEqual(len(chosen["in_stream"]), 5)

    def test_no_picture_is_drawn_twice(self):
        """Drawing with replacement would quietly shrink the set it measures."""
        chosen = fixed_image_sets({"in_stream": self.dataset()}, count=20)
        flat = chosen["in_stream"].reshape(20, -1)
        self.assertEqual(len({tuple(row.tolist()) for row in flat}), 20)

    def test_the_seed_is_what_picks_them(self):
        """A generator left unseeded still draws the same pictures every time.

        So sameness alone does not show the seed is used. Changing it must
        change the draw, or the set is fixed by accident rather than on purpose.
        """
        data = self.dataset()
        first = fixed_image_sets({"in_stream": data}, count=5)
        second = fixed_image_sets({"in_stream": data}, count=5, seed=7)
        self.assertFalse(torch.equal(first["in_stream"], second["in_stream"]))

    def test_each_set_is_drawn_from_its_own_dataset(self):
        data = self.dataset()
        other = TensorDataset(torch.zeros(20, 1, 8, 8), torch.arange(20) % 2)
        chosen = fixed_image_sets({"in_stream": data, "eurosat": other}, count=5)
        self.assertEqual(float(chosen["eurosat"].abs().sum()), 0.0)


class RidingAlong(unittest.TestCase):
    """One real two-task stream, run with and without the observer."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.data = TensorDataset(torch.rand(8, 1, 8, 8), torch.arange(8) % 2)
        self.tasks = {task: {"train": self.data, "test": self.data}
                      for task in ("3", "0", "15")}
        self.plan = make_plan("test-revision",
                              Path(__file__).resolve().parents[1] / "uncle/task_partition.json")
        self.plan.update(orders={"order_01": ["3", "0"]}, seeds=[0], forget_tasks=["3"],
                         forget_checkpoint_interval=2, forget_evaluation_interval=2,
                         keep_branch_checkpoints=True, keep_post_unlearning_checkpoint=True)
        self.plan["config"].update(tasks=["3", "0", "15"], device="cpu",
                                   classes_per_task=2, hidden=[16], code_dim=4,
                                   chunks=8, epochs=2, batch_size=4,
                                   eval_batch_size=4, noise_samples=2, burn_in=4)
        self.source = jobs(self.plan)[0]
        self.patch = patch("uncle.paired_study.build_target", side_effect=target)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def run_stream(self, observer=None):
        root = Path(self.folder.name) / f"run{id(observer)}"
        root.mkdir()
        storage = StudyStore(R2Store(Objects(), "test"), self.plan, root)
        storage.prepare()
        return run_job(self.plan, self.source, self.tasks, storage,
                       device="cpu", progress=False, observer=observer)

    def observe(self):
        path = Path(self.folder.name) / "seed_0.jsonl"
        observer = StreamObserver(path, fixed_image_sets({"in_stream": self.data}, count=4),
                                  study="test", order="order_01", seed=0, every=1)
        report = self.run_stream(observer)
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines()]
        return report, rows

    def test_one_line_per_task_boundary(self):
        """Two tasks give three boundaries, counting the fresh generator."""
        _, rows = self.observe()
        self.assertEqual([row["t"] for row in rows], [0, 1, 2])
        self.assertEqual([row["task"] for row in rows], [None, "3", "0"])

    def test_the_fresh_generator_has_not_moved(self):
        """Boundary zero is written before the first update, not during it."""
        _, rows = self.observe()
        self.assertEqual(rows[0]["delta_w_generator"], 0.0)

    def test_the_generator_moves_over_the_stream(self):
        _, rows = self.observe()
        self.assertGreater(rows[2]["delta_w_generator"], 0)

    def test_loss_progress_arrives_with_the_task_that_produced_it(self):
        """Boundary zero has no task behind it; the later ones each have one."""
        _, rows = self.observe()
        self.assertIsNone(rows[0]["loss_progress_mean"])
        self.assertIsNotNone(rows[1]["loss_progress_mean"])
        self.assertTrue(rows[1]["loss_progress_curve"])

    def test_the_curve_is_not_carried_into_the_next_task(self):
        """A curve that accumulated would make every later task look slower."""
        _, rows = self.observe()
        steps = [[step for step, _ in row["loss_progress_curve"]] for row in rows[1:]]
        self.assertEqual(steps[0], steps[1])

    def test_watching_leaves_the_run_unchanged(self):
        """The whole point of a passenger is that it does not steer."""
        quiet = self.run_stream()
        watched, _ = self.observe()
        self.assertEqual(quiet["final_accuracies"], watched["final_accuracies"])
        for first, second in zip(quiet["requests"], watched["requests"]):
            self.assertEqual([row["loss"] for row in first["trace"][1:]],
                             [row["loss"] for row in second["trace"][1:]])
            self.assertEqual([row["input_sha256"] for row in first["trace"][1:]],
                             [row["input_sha256"] for row in second["trace"][1:]])


if __name__ == "__main__":
    unittest.main()
