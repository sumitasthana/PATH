"""Run notebook 56's gate, its rate choice and its decision rule on made-up runs.

Why this exists. The notebook needs a GPU and three and a half hours, but the
parts that decide what the result means are ordinary arithmetic over a list of
finished runs: picking the best rate, the gate that stops a session whose probe
never memorised, and the rule that turns two means into a verdict. A mistake in
any of them would only show up at the end of the session.

So those cells are pulled out of the notebook as text and run here against runs
invented for the purpose. Nothing is checked for correctness of science. What
is checked is that the best rate is the lowest training loss, that the gate
passes on a fit below chance and raises on one at chance, and that the rule
reads a fall as more capacity and a rise as plasticity loss, which is the one
thing about this measurement that is easy to get backwards.

Run it with: python scripts/upgrade_suite/check_notebook_56_logic.py
"""
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

REPO = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
            if (p / "notebooks").is_dir())
NOTEBOOK = REPO / "notebooks/56_random_labels_on_the_revised_streams.ipynb"

CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
CODE = ["".join(c["source"]) for c in CELLS if c["cell_type"] == "code"]


def cell(starts_with):
    """The one code cell that begins with this text."""
    found = [c for c in CODE if c.lstrip().startswith(starts_with)]
    assert len(found) == 1, (starts_with, len(found))
    return found[0]


# Settings are read out of the notebook rather than retyped, so this check
# cannot drift away from what the notebook will actually run.
SETTINGS = cell("STUDY_ID")
space = {"Path": Path, "IN_COLAB": False, "REPO": REPO, "PROBE_TASK": "9"}
exec(compile(SETTINGS[:SETTINGS.index("DATA_ROOT")], "settings", "exec"), space)
RATES = space["RATES"]
REPS = space["REPETITIONS_AT_ZERO"]
TASK = space["RANDOM_LABEL_TASK"]
POINTS = space["POINTS"]
MARGIN = space["CHANCE_MARGIN"]
CHANCE = math.log(20)

from uncle.probe import best_of_grid, noise_floor  # noqa: E402  (needs sys.path below)


STREAMS = [("order_A", seed) for seed in range(5)]


def runs_for(mean_at_zero, mean_at_end, streams=STREAMS):
    """Finished runs whose selected loss is the mean given, per point.

    The three rates are given losses in a fixed order with the middle rate
    lowest, so the selection has something to choose and the answer is known.
    The repetitions are spread around the mean, because the floor is built from
    that spread and a flat set would make every floor zero.
    """
    spread = [-0.06, -0.03, 0.0, 0.03, 0.06]
    out = []
    for order, seed in streams:
        for point, mean in ((POINTS[0], mean_at_zero), (POINTS[-1], mean_at_end)):
            for rep in range(REPS):
                best = mean + spread[rep % len(spread)]
                for which, rate in enumerate(RATES):
                    # The middle rate is the best one. The others are worse by
                    # a clear margin, so a selection that picked the wrong one
                    # would change the mean and fail the checks below.
                    penalty = 0.0 if which == 1 else 0.5
                    out.append({"order": order, "seed_label": seed, "point": point,
                                "task": TASK, "arm": "full", "repetition": rep,
                                "learning_rate": rate,
                                "final_train_loss": best + penalty})
    return out


def harness(results):
    """The notebook's losses_at and check_memorised, over the given runs."""
    space = {"results": results, "RANDOM_LABEL_TASK": TASK, "POINTS": POINTS,
             "REPETITIONS_AT_ZERO": REPS, "REPETITIONS_LATER": REPS,
             "CHANCE_LOSS": CHANCE, "CHANCE_MARGIN": MARGIN,
             "best_of_grid": best_of_grid}
    grid = cell("def identity(run):")
    start = grid.index("def losses_at(")
    exec(compile(grid[start:grid.index("def grid_at(")], "gate", "exec"), space)
    return space


def check_rate_choice():
    """The selected run is the lowest training loss, not the lowest rate."""
    space = harness(runs_for(1.40, 0.70))
    values = space["losses_at"]("order_A", 0, POINTS[0])
    assert len(values) == REPS, values
    # The middle rate was the best by 0.5, so the selected losses carry no
    # penalty and average to the mean asked for.
    assert abs(sum(values) / len(values) - 1.40) < 1e-9, values
    print("rate choice: picks the lowest training loss  ok")


def check_gate_passes():
    """A fit well below chance lets the session continue."""
    space = harness(runs_for(1.40, 0.70))
    with redirect_stdout(io.StringIO()):
        space["check_memorised"]("order_A", 0)
    print(f"gate: passes at a fit of 1.40 and 0.70 against chance {CHANCE:.3f}  ok")


def check_gate_raises():
    """A fit sitting at chance stops the session instead of spending hours."""
    space = harness(runs_for(CHANCE - 0.05, CHANCE - 0.02))
    try:
        with redirect_stdout(io.StringIO()):
            space["check_memorised"]("order_A", 0)
    except RuntimeError as error:
        assert "near chance" in str(error), error
        print("gate: raises when the probe ends at chance  ok")
        return
    raise AssertionError("The gate let a probe through that never memorised.")


def verdicts(mean_at_zero, mean_at_end):
    """Run the notebook's result cell and collect the rows it decided."""
    results = runs_for(mean_at_zero, mean_at_end)
    space = harness(results)
    space.update({"results": results, "RUNS": [("order_A", s) for s in range(5)],
                  "ORDERS": ["order_A"], "STUDY_ID": "check", "CLASSES": 20,
                  "RANDOM_LABEL_SEED": space.get("RANDOM_LABEL_SEED", 1020),
                  "RANDOM_LABEL_IMAGES": 2000, "RANDOM_LABEL_STEPS": 600,
                  "RATES": RATES, "HEAD": "check", "noise_floor": noise_floor,
                  "store": None, "CACHE": None, "json": json})
    body = cell("import statistics")
    # Everything up to the upload, which needs R2.
    body = body[:body.index('summary = {"study"')]
    with redirect_stdout(io.StringIO()) as printed:
        exec(compile(body, "result", "exec"), space)
    return space["rows"], printed.getvalue()


def check_the_direction():
    """Lower loss at the end means more capacity, not less.

    This is the one thing about a random-label result that is easy to read
    backwards, because every other measurement in the study is one where higher
    is better.
    """
    rows, printed = verdicts(1.40, 0.70)
    assert all(r["verdict"] == "no loss, more capacity to change" for r in rows), rows
    assert "capacity to change rose" in printed.lower(), printed

    rows, printed = verdicts(0.70, 1.40)
    assert all(r["verdict"] == "PLASTICITY LOSS" for r in rows), rows
    assert "memorised less at the end" in printed, printed

    rows, _ = verdicts(1.00, 1.001)
    assert all(r["verdict"] == "NO FINDING, inside the floor" for r in rows), rows
    print("direction: a fall reads as more capacity, a rise as plasticity loss  ok")


def check_void():
    """A fit at chance is voided even if the two points differ."""
    rows, _ = verdicts(CHANCE - 0.05, CHANCE - 0.40)
    assert all(r["verdict"] == "VOID, the probe never memorised" for r in rows), rows
    print("void: a fit at chance is refused a verdict  ok")


if __name__ == "__main__":
    check_rate_choice()
    check_gate_passes()
    check_gate_raises()
    check_the_direction()
    check_void()
    print("\nnotebook 56 logic ok")
