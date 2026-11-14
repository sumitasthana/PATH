# Run map

Back to the [index](Experiment-trajectory).

One picture per run, so a whole stream can be read without opening a single
JSON file.

![Run map template](figures/run-map-template.svg)

## How to read it

Three panels sit above each other on one shared axis. Every column is one
moment in the run, so you can read straight down a column.

**The top row is the run itself.** B0 is the untrained model, before anything
is learned. L0 through L8 are the nine tasks, in the order that run learned
them. F0 is the task broken on purpose at the end. Each box carries the
accuracy that task reached at the moment it was learned.

**The middle panel is every task, at every later moment.** One row per learned
task, one column per boundary. A cell is that task's accuracy then. Cells are
empty to the left of where the task was learned, because it did not exist yet.

Read a row left to right and you are watching one task age. Read a column top
to bottom and you are seeing the whole model at one instant. Green means the
task is holding up, red means it has fallen towards the 5 percent a random
guess would score.

**The bottom panel is what the indicators said.** Effective rank on each of the
three fixed image sets, each divided by its own value at B0, so three
measurements on different scales can share one axis. The dashed line is "same
as the untrained model". Below it, rank has fallen.

## Why the panels are stacked

The dashed vertical line marked "broken here" is the point of the figure.

At that column the top row of the grid falls to 5 percent, because that task
was destroyed deliberately. The damage is certain, and we know exactly where
it is.

So whatever the indicator lines do in that same column is what those
indicators say about a model that is definitely broken. If the lines rise, the
indicators are calling a damaged model healthier than the intact one that came
before it. That is the claim this study exists to test, and this is the figure
that shows it in one glance instead of in a table of eleven rows.

## Building one

```bash
python scripts/build_run_map.py --label order_A-seed0 --indicators <run>/indicators.jsonl --history <run>/history_seq1_resnet50_seed0.json
```

The indicator record is the JSONL file the run uploads to R2, and the history
is the JSON that `run_experiment` writes beside the checkpoint. Either can be
missing. Values that are not there are drawn as a dash, so a run that is still
going, or one whose indicator record has not been fetched yet, still maps.

With neither file it draws the empty template above, which is what the layout
was reviewed against before any run finished.

```bash
python scripts/build_run_map.py --label template
```

Pass `--dark` for the dark palette. The figure uses the same two palettes as
the trajectory figures, so the wiki reads as one document either way.

## What it does not show

Accuracy comes from the run's own history, and the indicators come from the
boundary record, so a map is only as complete as those two files.

The map is one run. It says nothing about whether a difference between two
runs is real. That question belongs to the seed spread, which is measured
across runs and reported in the dated pages.
