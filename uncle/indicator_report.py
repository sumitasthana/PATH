"""Turn the boundary records into one table, one set of curves, and one summary.

`BoundaryLog` writes one JSON line per task boundary, one file per run. This
module reads those files and produces three things, in the order they are
useful:

    long_table      one row per (run, t, image set, measure, value)
    curves          each measure against task position, one line per run
    endpoints       mean and spread across runs at the first and last boundary

The long table comes first on purpose. It has the same columns as
`study_long.csv` in `reference/adapter-plasticity-study`, so the two sets of
numbers can be put side by side without reshaping either one, and everything
else here is computed from it rather than from the model. A figure that
disagrees with the table is then a bug in one line of code, not a second
measurement that has to be chased back to the run.

The scatter of probe accuracy against each indicator, which is the adapter
study's headline figure, is deliberately absent. That study probes at four
points and we probe at two, so three seeds give six points, which is too few
to say whether an indicator tracks the probe. The curve and the endpoint table
are what two probe points can honestly support.
"""

import csv
import json
from pathlib import Path

from .plasticity_indicators import THRESHOLDS
from .stream_metrics import mean_and_sd

# Measures that depend on which images were pushed through. The boundary record
# writes these with the image set's name appended, so splitting a key means
# checking it against this list rather than guessing where the name starts.
PER_IMAGE_SET = ("effective_rank", "srank", "images", "units",
                 *(f"dormant_{threshold}" for threshold in THRESHOLDS))

# Written by the record to say which run and which boundary this is. These are
# numbers too, so they have to be named; everything that is not a single number,
# such as the loss progress curve, is left out by its type instead.
IDENTITY = ("study", "order", "seed", "t", "task")


def read_rows(*paths):
    """Every boundary line from one or more run files, in the order written."""
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_key(key):
    """A record key as (measure, image set), with None where no set applies."""
    for measure in PER_IMAGE_SET:
        if key.startswith(measure + "_"):
            return measure, key[len(measure) + 1:]
    return key, None


def long_table(rows):
    """One flat row per measurement, which every table and figure reads from.

    Keys that name the run or the boundary are dropped, and so is any value
    that is not a single number: the loss progress curve is a list, and
    `loss_progress_mean` is None at boundary zero, where no task was learned.
    A missing value is left out rather than carried as a blank row.
    """
    flat = []
    for row in rows:
        run = f"{row['order']}/seed_{row['seed']}"
        for key, value in row.items():
            if key in IDENTITY or not isinstance(value, (int, float)):
                continue
            measure, image_set = split_key(key)
            flat.append({"run": run, "order": row["order"], "seed": row["seed"],
                         "t": row["t"], "task": row["task"],
                         "image_set": image_set or "", "measure": measure,
                         "value": float(value)})
    return flat


def write_long_table(flat, path):
    """The long table as a CSV, which is the one file everything else reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["run", "order", "seed", "t", "task", "image_set", "measure", "value"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    return path


def endpoints(flat):
    """Mean and spread across runs at the first and last boundary, and the change.

    The spread is the sample standard deviation across runs, the same figure
    the probe tables already report, so an indicator's change can be read next
    to the probe's change in the same units of evidence. A measure present at
    only one boundary gets that boundary and no change.

    A change is worth attention when it is larger than the spread it sits in.
    This function does not make that call; it lays the two numbers side by side
    so a reader can.
    """
    grouped = {}
    for row in flat:
        grouped.setdefault((row["measure"], row["image_set"]), {}) \
               .setdefault(row["t"], []).append(row["value"])

    summary = []
    for (measure, image_set), by_boundary in sorted(grouped.items()):
        first, last = min(by_boundary), max(by_boundary)
        start_mean, start_sd = mean_and_sd(by_boundary[first])
        end_mean, end_sd = mean_and_sd(by_boundary[last])
        summary.append({"measure": measure, "image_set": image_set,
                        "first_t": first, "last_t": last,
                        "runs": len(by_boundary[last]),
                        "start_mean": start_mean, "start_sd": start_sd,
                        "end_mean": end_mean, "end_sd": end_sd,
                        "change": None if first == last else end_mean - start_mean})
    return summary


def curves(flat, folder, *, prefix="fig_"):
    """One figure per measure: value against task position, one line per run.

    Panels split the image sets, because a measure that moves on the stream's
    own images and stays flat on a different dataset is saying something
    different from one that moves on both. That split is the correction the
    adapter study had to make after the fact, so it is built in here.

    The first question these answer is whether anything moves at all over eight
    tasks. A flat line is a result, not a failure, because the adapter study's
    indicators move a lot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for measure in sorted({row["measure"] for row in flat}):
        picked = [row for row in flat if row["measure"] == measure]
        sets = sorted({row["image_set"] for row in picked})
        figure, axes = plt.subplots(1, len(sets), figsize=(4 * len(sets), 3.4),
                                    sharey=True, squeeze=False)
        for axis, image_set in zip(axes[0], sets):
            for run in sorted({row["run"] for row in picked}):
                points = sorted((row["t"], row["value"]) for row in picked
                                if row["run"] == run and row["image_set"] == image_set)
                axis.plot([t for t, _ in points], [v for _, v in points],
                          marker="o", markersize=3, label=run)
            axis.set_title(image_set or measure, fontsize=9)
            axis.set_xlabel("tasks learned")
        axes[0][0].set_ylabel(measure)
        axes[0][-1].legend(fontsize=7)
        figure.tight_layout()
        path = folder / f"{prefix}{measure}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        written.append(path)
    return written
