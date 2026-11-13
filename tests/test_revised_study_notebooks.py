"""Notebooks 44 and 45 can lose their machine and pick the run back up.

Both jobs run for hours on a rented GPU. A RunPod pod that goes away takes its
disk with it, so everything needed to continue has to be in R2: the indicator
record, its boundary-zero baseline, and the training checkpoint. If any one of
those is missing, a restarted run retrains tasks it has already paid for, and
nothing reports an error while it happens.

This takes the `run_one` function out of each notebook and runs it for real on
a CPU with a tiny network. The run is cut off part way, every local file is
deleted, and `run_one` is called again against the same fake store.

Only the restart is checked here. The indicators themselves are covered in
test_plasticity_indicators.py.
"""
import json
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch

import uncle.trainer as trainer_module
from uncle import Config
from uncle.experiment import run as real_run
from uncle.plasticity_indicators import indicator_key

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
NAMES = ("44_order_A_five_seeds", "45_orders_B_and_C")

REQUESTS = (("learn", "0"), ("learn", "1"), ("learn", "2"))


def run_one_from(name):
    """The notebook's own `run_one`, as source, so a drift is caught here."""
    book = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text(encoding="utf-8"))
    for cell in book["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "def run_one(" in source:
            return source
    raise AssertionError(f"{name} has no run_one cell")


class Stop(Exception):
    """Ends the first attempt early, the way a dead machine would."""


class FakeStore:
    """Stands in for R2Store: a folder, with the three methods run_one uses."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads = []

    def _at(self, key):
        return self.root / key.replace("/", "__")

    def info(self, key):
        path = self._at(key)
        return {"ContentLength": path.stat().st_size} if path.exists() else None

    def upload(self, path, key, *, overwrite=False):
        shutil.copyfile(path, self._at(key))
        self.uploads.append(key)

    def download(self, key, cache):
        path = Path(cache) / Path(key).name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._at(key), path)
        return path


def tiny_task():
    """Two classes, four images. Enough to take an optimizer step."""
    images = torch.rand(4, 1, 28, 28)
    return torch.utils.data.TensorDataset(images, torch.tensor([0, 1, 0, 1]))


class RestartTests(unittest.TestCase):

    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.learn = trainer_module.UnCLe.learn
        self.addCleanup(setattr, trainer_module.UnCLe, "learn", self.learn)

    def environment(self, source, store, output_root, stop_after):
        """Build the globals the notebook cell expects, then define run_one.

        The cell reads its settings from the notebook's earlier cells, so they
        are supplied here instead. `run` is wrapped so the first attempt can be
        cut off after a chosen number of finished requests.
        """
        trained = []

        def watched(self, task, protected, **kwargs):
            trained.append(task)
            return RestartTests.learn_original(self, task, protected, **kwargs)

        RestartTests.learn_original = self.learn
        trainer_module.UnCLe.learn = watched

        seen = [0]

        def run(config, **kwargs):
            inner = kwargs.pop("on_request")

            def counted(record):
                inner(record)
                seen[0] += 1
                if stop_after is not None and seen[0] >= stop_after:
                    raise Stop()

            return real_run(config, on_request=counted, **kwargs)

        cache = output_root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        namespace = {
            "Path": Path, "json": json, "shutil": shutil, "torch": torch,
            "time": __import__("time"), "mean": lambda xs: sum(xs) / len(xs),
            "datetime": datetime,
            "clock": lambda: "00:00:00", "spell": lambda s: str(int(s)),
            "StreamObserver": __import__(
                "uncle.plasticity_indicators", fromlist=["StreamObserver"]
            ).StreamObserver,
            "indicator_key": indicator_key,
            "run": run, "store": store, "CACHE": cache,
            "OUTPUT_ROOT": output_root,
            "STUDY_ID": "restart_check", "LEARNING_RATE": 1e-3, "HEAD": "test",
            "KEEP_BOUNDARIES": (2,),
            "image_sets": {"in_stream": torch.rand(8, 1, 28, 28)},
            "tasks": {name: {"train": tiny_task(), "test": tiny_task()}
                      for name in "012"},
        }
        exec(compile(source, "run_one-cell", "exec"), namespace)
        return namespace["run_one"], trained

    def config(self):
        return Config(
            dataset="permuted_mnist", backbone="cnn", tasks=("0", "1", "2"),
            requests=REQUESTS, classes_per_task=2, chunks=4, epochs=1,
            batch_size=2, seed=0, device="cpu", learning_rate=1e-3,
        )

    def rows(self, store, cache):
        key = indicator_key("restart_check", "order_A", 0)
        text = store.download(key, cache).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def check_one(self, name):
        source = run_one_from(name)
        store = FakeStore(self.home / "r2")
        output_root = self.home / "out"
        output_root.mkdir()

        # First attempt, cut off after two of the three requests.
        run_one, _ = self.environment(source, store, output_root, stop_after=2)
        with self.assertRaises(Stop):
            run_one("try1", self.config(), "order_A", 0, "0", REQUESTS)

        resume_key = "uncle/checkpoints/restart_check/order_A/seed_0/resume.pt"
        self.assertIsNotNone(store.info(resume_key),
                             "the training checkpoint never reached R2")
        self.assertEqual([row["t"] for row in
                          self.rows(store, output_root / "cache")], [0, 1, 2])

        # The machine goes away. Nothing local survives.
        shutil.rmtree(output_root)
        output_root.mkdir()

        # Second attempt. It must finish the stream without starting over.
        run_one, trained = self.environment(source, store, output_root,
                                            stop_after=None)
        result = run_one("try2", self.config(), "order_A", 0, "0", REQUESTS)

        # Task 0 is finished and saved, so it must not be trained again. Task 1
        # is expected: the checkpoint is written after the indicator row, so a
        # stop between the two leaves the newest checkpoint at task 0. Replaying
        # one request is the right cost of a stop in that window, and the replay
        # is exact because the checkpoint carries the random number state.
        self.assertNotIn("0", trained, f"retrained a finished task: {trained}")
        self.assertEqual(trained, ["1", "2"])

        # No duplicate rows: the observer refuses a boundary already recorded.
        self.assertEqual([row["t"] for row in
                          self.rows(store, output_root / "cache")],
                         [0, 1, 2, 3])
        self.assertEqual(result["boundaries_measured"], 4)

    def test_notebook_44_resumes_from_r2_after_the_machine_is_lost(self):
        self.check_one(NAMES[0])

    def test_notebook_45_resumes_from_r2_after_the_machine_is_lost(self):
        self.check_one(NAMES[1])


if __name__ == "__main__":
    unittest.main()
