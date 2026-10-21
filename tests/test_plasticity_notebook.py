"""Notebook 29 can be stopped and restarted. These check that it resumes right.

A bug here is expensive and silent: a run identity that is not unique makes the
notebook skip work it never did, and one that is unstable makes it repeat work
it already paid for. Both cost GPU hours without any error appearing.

Only the resume logic is read out of the notebook. The probe itself is tested
in test_probe.py.
"""
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks/29_plasticity_probe.ipynb"

# Mirrors the notebook's own settings cell. If these drift apart the count
# assertion below is the thing that notices.
SEEDS = [0, 1, 2]
ARMS = ["full", "code_only", "generator_only"]
PROBE_TASKS = ["2", "4", "6"]
RANDOM_LABEL_TASK = "8"
RATES = [1e-4, 3e-4, 1e-3]
REPETITIONS = {0: 5, 8: 2}


def notebook_identity():
    """The identity function as the notebook actually defines it."""
    source = "".join(
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code" and "def identity(run)" in "".join(cell["source"])
    )
    body = source.split("def identity(run)")[1].split("PARTIAL =")[0]
    namespace = {}
    exec("def identity(run)" + body, namespace)
    return namespace["identity"]


def every_run(seeds):
    for seed in seeds:
        for point, repetitions in REPETITIONS.items():
            pairs = [(task, "E2") for task in PROBE_TASKS] + [(RANDOM_LABEL_TASK, "E1")]
            for task, endpoint in pairs:
                for rate in RATES:
                    for arm in ARMS:
                        for repetition in range(repetitions):
                            yield {"seed_label": seed, "point": point,
                                   "endpoint": endpoint, "task": task,
                                   "learning_rate": rate, "arm": arm,
                                   "repetition": repetition}


class ResumeTests(unittest.TestCase):

    def setUp(self):
        self.identity = notebook_identity()

    def test_every_run_has_its_own_identity(self):
        runs = list(every_run(SEEDS))
        self.assertEqual(len(runs), len({self.identity(r) for r in runs}))

    def test_the_same_run_always_gives_the_same_identity(self):
        run = next(every_run([1]))
        self.assertEqual(self.identity(run), self.identity(dict(run)))

    def test_finished_seeds_are_skipped_and_the_rest_are_not(self):
        done = {self.identity(r) for r in every_run([0])}
        left = [r for r in every_run(SEEDS) if self.identity(r) not in done]
        self.assertEqual(len(left), len(list(every_run([1, 2]))))
        self.assertEqual({r["seed_label"] for r in left}, {1, 2})

    def test_a_stopped_probe_point_resumes_part_way(self):
        # Half of seed 0's first probe point finished before the session died.
        partial = [r for r in every_run([0]) if r["point"] == 0][:40]
        done = {self.identity(r) for r in partial}
        left = [r for r in every_run([0]) if self.identity(r) not in done]
        self.assertEqual(len(left), len(list(every_run([0]))) - 40)

    def test_arm_names_match_what_the_probe_records(self):
        # The resume check compares the arm name in ARMS against the one the
        # probe writes into each result. When those were spelled differently,
        # two of the three arms were never skipped and were run again every
        # session: silent, and hours of wasted GPU time.
        from uncle.probe import probe_once           # noqa: F401, import check only
        settings = "".join(
            "".join(cell["source"])
            for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code" and "ARMS = [" in "".join(cell["source"])
        )
        namespace = {}
        exec(settings[settings.index("ARMS = ["):].split("]")[0] + "]", namespace)
        for name, code_only, train_code in namespace["ARMS"]:
            recorded = ("code_only" if code_only
                        else "full" if train_code else "generator_only")
            self.assertEqual(name, recorded,
                             f"arm {name!r} is recorded as {recorded!r}, so resume "
                             f"will never skip it")

    def test_the_notebook_still_lists_the_seeds_this_file_assumes(self):
        settings = "".join(
            "".join(cell["source"])
            for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code" and "SEEDS = " in "".join(cell["source"])
        )
        self.assertIn(f"SEEDS = {SEEDS}", settings)


if __name__ == "__main__":
    unittest.main()
