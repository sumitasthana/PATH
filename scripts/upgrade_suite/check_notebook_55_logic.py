"""Run notebook 55's gate and its two tables against made-up readings.

Why this exists. The notebook itself cannot be run without a GPU and 16 GB of
downloads, but three of its cells are ordinary arithmetic over a list of rows:
the sanity gate, the part 1 table and the part 2 table. A mistake in any of
them would only show up at the end of an hour-long run, which is an expensive
place to find a typo.

So those three cells are pulled out of the notebook as text and run here
against rows invented for the purpose. Nothing is checked for correctness of
science; what is checked is that the gate passes when the numbers match,
raises when they do not, and that both tables print without falling over.

Run it with: python scripts/upgrade_suite/check_notebook_55_logic.py
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

REPO = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
            if (p / "notebooks").is_dir())
NOTEBOOK = REPO / "notebooks/55_matched_statistics_on_every_stream.ipynb"

CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
CODE = ["".join(c["source"]) for c in CELLS if c["cell_type"] == "code"]


def cell(starts_with):
    """The one code cell that begins with this text."""
    found = [c for c in CODE if c.lstrip().startswith(starts_with)]
    assert len(found) == 1, (starts_with, len(found))
    return found[0]


# The published means the gate compares against, read out of the settings cell
# rather than retyped, so this check cannot drift away from the notebook.
SETTINGS = cell("STUDY_ID")
space = {}
exec(compile(SETTINGS[SETTINGS.index("GATE = {"):SETTINGS.index("DATA_ROOT")],
             "settings", "exec"), space)
GATE = space["GATE"]
TOLERANCE = space["GATE_TOLERANCE"]
SEEDS = [0, 1, 2, 3, 4]

MEASURES = ["effective_rank", "srank", "mean_share",
            "dormant_0.01", "dormant_0.025", "dormant_0.1"]


def indicator_rows(off=0.0):
    """Five ordering A seeds whose mean is each published value times 1 + off.

    The seeds are spread around that mean on purpose, because the real readings
    spread very widely and the gate is supposed to look at the mean alone.
    """
    spread = [-0.4, -0.2, 0.0, 0.2, 0.4]
    rows = []
    for order in ("order_A", "order_B", "order_C"):
        for seed in SEEDS:
            for point in (0, 9, 10):
                for mode in ("start", "own"):
                    base = GATE.get((point, mode), GATE[(9, mode)])
                    for images in ("in_stream", "probe"):
                        value = base * (1 + off) + (spread[seed] if images == "in_stream" else 1.0)
                        rows.append({"order": order, "seed": seed, "point": point,
                                     "buffers": mode, "images": images,
                                     **{m: value for m in MEASURES}})
    return rows


def rescaling_rows():
    rows = []
    for order in ("order_A", "order_B", "order_C"):
        for seed in SEEDS:
            for arm in ("b9", "b10", "b10_at_b9_norm", "b9_at_b10_norm"):
                for mode in ("start", "own"):
                    for images in ("in_stream", "probe"):
                        rows.append({"order": order, "seed": seed, "arm": arm,
                                     "buffers": mode, "images": images,
                                     "effective_rank": 3.0 if arm == "b9" else 4.0,
                                     "srank": 20, "mean_share": 0.9,
                                     "norm_ratio": 0.8})
    return rows


def run_gate(rows):
    """Execute the notebook's gate cell over `rows`, returning what it printed."""
    space = {"GATE": GATE, "GATE_TOLERANCE": TOLERANCE, "READ_SEEDS": SEEDS}
    exec(compile(cell("def check_gate"), "gate", "exec"), space)
    out = io.StringIO()
    with redirect_stdout(out):
        space["check_gate"](rows)
    return out.getvalue()


# 1. Readings that match the published means pass.
printed = run_gate(indicator_rows(off=0.0))
assert "passed" in printed, printed
assert "boundary 0, matched statistics" in printed, printed

# 2. Readings half a percent out still pass, because the gate allows one.
assert "passed" in run_gate(indicator_rows(off=0.005))

# 3. Readings five percent out stop the run.
try:
    run_gate(indicator_rows(off=0.05))
    raise AssertionError("the gate let a five percent error through")
except RuntimeError as error:
    assert "Stop and find out why" in str(error), error

# 4. Both tables print over a full set of rows without failing.
space = {"part1": indicator_rows(), "part2": rescaling_rows(),
         "READ_RUNS": [(o, s) for o in ("order_A", "order_B", "order_C")
                       for s in SEEDS]}
out = io.StringIO()
with redirect_stdout(out):
    exec(compile(cell("from statistics import mean, stdev"), "table1", "exec"), space)
    exec(compile(cell('print("THE RESCALING CONTROL'), "table2", "exec"), space)
printed = out.getvalue()
assert "DIRECTION OF TRAVEL" in printed, printed
assert "THE RESCALING CONTROL" in printed, printed
assert "of 15" in printed, printed

print("notebook 55: the gate passes, catches a five percent error, "
      "and both tables print.")
