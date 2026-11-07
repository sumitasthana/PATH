# Experiment trajectory: 7 November 2026

Back to the [index](Experiment-trajectory).

## Learning a new task after unlearning is not slower

### What this is, and what it is not

Nothing was trained on 7 November. This is a reading of runs that already
finished in October, taken from their reports in R2 under
`uncle/paired_generalization/paired_generalization_v2/`. The runs are the paired
study's own branch and control jobs. No new GPU time was spent.

The question behind the reading: the plasticity indicators report damage where
no direct measurement finds any, so can they detect damage when it is really
there? Unlearning is the operation that really damages. Before planning a new
run, the existing records were checked for anything that already answers it.
Two earlier claims that something was unmeasured turned out to be wrong, so the
check came first.

### The design, which was already in place

Each branch job starts from a finished eight-task model, unlearns one task, then
learns task 15. Each control job starts from the same model and learns task 15
with no unlearning. Same source checkpoint, same new task, same settings. The
only difference is the unlearning. That is exactly the comparison needed, and it
has been sitting in R2 since October.

### The unlearning worked

In all six branch jobs the forgotten task falls to 10.0 percent, which is chance
for a ten-class task, while the other seven tasks hold. So these are models that
have really been damaged, not models that were nudged.

### Task 15 is learned just as fast either way

Training loss on task 15, one value per epoch, order 01:

| Seed | Control, no unlearning | After forgetting task 3 | After forgetting task 14 |
| ---: | --- | --- | --- |
| 0 | 1.9373, 1.6779, 1.5259, 1.4061, 1.3373 | 1.9056, 1.6084, 1.4721, 1.3625, 1.2769 | 1.9641, 1.6758, 1.5418, 1.4180, 1.3356 |
| 1 | 1.8877, 1.6309, 1.4771, 1.3586, 1.2618 | 1.8824, 1.6239, 1.4604, 1.3589, 1.2488 | 1.9026, 1.6305, 1.4928, 1.3757, 1.2863 |
| 2 | 2.0146, 1.7457, 1.6071, 1.4931, 1.3750 | 2.0466, 1.7609, 1.6115, 1.4777, 1.3439 | 2.1743, 1.8325, 1.6310, 1.4964, 1.3766 |

Mean final loss: control 1.3247, after forgetting task 3 1.2899, after
forgetting task 14 1.3328. The three curves sit on top of each other. The spread
between the three seeds is larger than the gap between any arm and its control.

So a model that has just had a task destroyed learns the next task at the same
speed as a model that has not. This is the sixth direct measurement on the "no
plasticity loss" side, and it is the first one taken after genuine damage. The
other five were taken after ordinary learning, where the question of whether
anything was damaged was open.

### A hint in the other direction, which this design cannot resolve

Accuracy on task 15 does not sit as quietly as the loss does:

| Seed | Control | After forgetting task 3 | After forgetting task 14 |
| ---: | ---: | ---: | ---: |
| 0 | 50.8 | 50.8 | 48.2 |
| 1 | 44.0 | 47.2 | 47.8 |
| 2 | 34.4 | 47.4 | 32.6 |

Forgetting task 3 beats its control in two seeds of three, by 3.2 and 13.0
points, and ties in the third. The mean is 48.47 against the control's 43.07.
But the control's own spread across three seeds is 16.4 points, which is wider
than the effect, and the loss curves show nothing of the sort. Read it as
unresolved. It is recorded because it points the same way as the collaborator's
claim that returning toward the starting point helps, and that claim is about to
be tested directly.

### Limitations

Three seeds, one order, one new task. Task 15 is a task the model never learned,
so this says nothing about relearning the task that was forgotten; the 26 to 28
October records cover that and find a 6 point penalty. Epoch loss is a coarser
measure than the loss progress indicator, which samples one update in ten.

## Why the indicator stream cannot measure an unlearn step as the code stands

The other half of the question needs the weight-reading indicators, effective
rank, srank and dormant units, read on a model that has just been unlearned.
That reading does not exist and cannot be recovered.

### The observer attaches, but it skips the moment that matters

`run_job` in `uncle/paired_study.py` passes the observer to both of its hooks
without checking the job kind, so a branch job would run them. But the boundary
number is `len(seen)`, and unlearning appends to `forgotten`, not to `seen`. The
number therefore does not change across the unlearn step, and `BoundaryLog`
returns immediately from a number already written. A branch job would record the
state before unlearning and the state after unlearning plus a whole new task,
with the interesting point in between skipped.

### Loss progress has no value during an unlearn step

`loss_progress` computes `(before - after) / before`, where `before` is the
step's `task_loss`. An unlearn step has none. The forget branch of
`uncle/study_training.py` never calls `on_step`, records no `task_loss`, and has
no batch of images at all: the objective is built from generated noise. Even if
a number were manufactured there, it would measure progress toward destroying
the task, which is the opposite sign of the learn-step number. The two must not
share a column. The existing code already writes a null there, which is correct.

### Nothing in R2 fills the gap

The `indicator_stream_v1` prefix holds only the five source jobs and their
indicator files. No branch or control job has ever run under it.
`paired_generalization_v2` ran the branches in October but kept only three
source checkpoints; every branch and control checkpoint was cleaned up. So there
is no post-unlearning model anywhere in the bucket to measure.

## The decision

A new run is needed, and it is small. The branch and control jobs already exist
in the `indicator_stream_v1` plan, resuming from finished source checkpoints, so
no new study and no plan change is required. A two-line change to the boundary
numbering in `uncle/paired_study.py` gives the unlearn step its own boundary
number, and leaves every existing source record numbered exactly as it is.

The run is being merged with a second planned experiment, which moves the
generator part of the way back toward its starting values and probes it. Both
ask the same thing: take the eight-task model, move its generator, read what
happened. The unlearned model becomes one more point in that probe grid. That
puts the direct probe on a damaged model for the first time, which the
indicator-only plan would not have done.

Both lines will run on order 01, seeds 1 and 2. Seed 0 was the first choice for
the unlearning arm but has only 1.22 points of probe headroom between its two
endpoints, against a noise floor near 3, so it cannot carry the probe
comparison. Seeds 1 and 2 have 4.27 and 4.73.
