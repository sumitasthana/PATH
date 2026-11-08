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

## The indicators read healthier after an unlearn

**Partly superseded on 8 November.** The weight-norm explanation flagged as
open at the end of this section was tested and carries about two thirds of
the rank and srank rise. The size claim below is withdrawn; the conclusion
about the indicators stands, for a reason given in the
[8 November record](Experiment-trajectory-2026-11-08).

This is the first reading of the plasticity indicators on a model that has just
been damaged. Every earlier reading was taken on a model that had only learned.

### The run

Job `branches/order_01/seed_1/forget_3/learn_15` under `indicator_stream_v1`.
It restored the finished eight-task source model for order 01 seed 1, unlearned
task 3 in 100 steps, measured the indicators, and stopped before task 15 was
trained. Twelve minutes and thirteen seconds on an A100-SXM4-40GB, most of it
setup rather than the unlearning.

The record is `uncle/indicators/indicator_stream_v1/order_01/seed_1_forget_3.jsonl`,
ten rows: a copy of the source run's nine rows, boundary 0 to boundary 8, plus
the new boundary 9. The copy means boundary 9 is measured against the same
boundary zero as every row before it, so the weight-change columns are
comparable along the whole stream. The post-unlearn model is kept in R2 at
`uncle/paired_generalization/indicator_stream_v1/branches/order_01/seed_1/forget_3/learn_15/boundary_9.pt`.

The job is left resumable rather than complete. Learning task 15 would have
measured nothing new, because the October records already settled how fast task
15 is learned after an unlearn.

### The unlearning worked

| Task | Before | After | Change |
| ---: | ---: | ---: | ---: |
| 3 | 34.8 | 10.0 | -24.8 |
| 0 | 42.6 | 43.2 | 0.6 |
| 9 | 48.8 | 50.6 | 1.8 |
| 5 | 32.8 | 35.4 | 2.6 |
| 17 | 50.2 | 48.0 | -2.2 |
| 1 | 54.4 | 52.2 | -2.2 |
| 7 | 44.0 | 41.0 | -3.0 |
| 14 | 45.8 | 45.8 | 0.0 |

Task 3 is at chance for a ten-class task. The other seven hold within three
points. This is the damaged model the question asks about, and by the project's
own earlier measurements the damage is real: a forgotten task is about six
points harder to relearn than a task never seen, and its task code costs 40 to
50 steps of dead time.

### All three weight-reading indicators moved the healthy way

| Indicator | t=7 | t=8 | t=9, after the unlearn | Change at 9 |
| --- | ---: | ---: | ---: | ---: |
| effective rank, in stream | 2.9748 | 2.9708 | 3.2664 | +0.2956 |
| effective rank, probe | 2.9745 | 2.9684 | 3.2759 | +0.3075 |
| effective rank, eurosat | 2.2170 | 2.2345 | 2.3239 | +0.0894 |
| srank, in stream | 175 | 175 | 188 | +13 |
| srank, probe | 178 | 178 | 190 | +12 |
| srank, eurosat | 116 | 117 | 125 | +8 |
| dormant 0.01, in stream | 0.2408 | 0.2421 | 0.2379 | -0.0042 |
| dormant 0.01, probe | 0.2408 | 0.2419 | 0.2383 | -0.0036 |
| dormant 0.01, eurosat | 0.2448 | 0.2461 | 0.2439 | -0.0022 |

Rank up, srank up, dormant units down. On all three image sets, including
EuroSAT, which is a different dataset entirely. Every one of these is the
direction these indicators call healthy.

As a share of what the whole eight-task stream did, the single unlearn step
gives back 18.6 percent of effective rank's fall on the in-stream images, 19.2
percent on the probe set and 22.6 percent on EuroSAT. For srank it is 36.1,
35.3 and 28.6 percent. Dormant units give back much less, 3 to 5 percent.

### The size of that move, against ordinary learning

Change in each number from one boundary to the next, in stream:

| Step | Generator distance | Generator angle | Effective rank |
| --- | ---: | ---: | ---: |
| 0 to 1 | 0.0320 | 1.833 | -1.183 |
| 1 to 2 | 0.0227 | 1.315 | +0.802 |
| 2 to 3 | 0.0153 | 0.875 | -0.555 |
| 3 to 4 | 0.0121 | 0.693 | -0.283 |
| 4 to 5 | 0.0127 | 0.715 | -0.076 |
| 5 to 6 | 0.0106 | 0.602 | -0.109 |
| 6 to 7 | 0.0077 | 0.437 | -0.183 |
| 7 to 8 | 0.0069 | 0.382 | -0.004 |
| 8 to 9, the unlearn | 0.0096 | 0.559 | +0.296 |

The generator itself moved an ordinary amount. Its distance and its angle at the
unlearn sit in the middle of the range the learning steps produced, larger than
the last two tasks and smaller than the first six.

Effective rank did not behave ordinarily. By the end of the stream the learning
steps had stopped moving it, the last three being -0.109, -0.183 and -0.004. The
unlearn moved it +0.296, larger than any of those and in the opposite direction.
The stream had not produced a rise that size since its second task. The same
holds for srank: the last three learning steps gave -3, 0 and 0, and the unlearn
gave +13.

### The generator stepped forward, but the weights it writes stepped back

| Measure | t=0 | t=8 | t=9 |
| --- | ---: | ---: | ---: |
| Generator distance from start | 0.0000 | 0.1200 | 0.1295 |
| Generator angle from start | 0.000 | 6.852 | 7.411 |
| Target distance from start | 0.0000 | 0.3845 | 0.3617 |
| Target angle from start | 0.000 | 21.559 | 21.187 |
| Target weight norm | 274.194 | 286.338 | 262.362 |

Target columns are the weights the generator writes for one fixed reference
code that is never trained, so they describe the generator rather than any
learned task.

The generator moved further from its starting point, as it had at every earlier
boundary. The weights it writes moved back toward theirs. Distance fell by
0.0228 and angle by 0.372. This is small, 5.9 percent of the distance travelled
and 1.7 percent of the rotation, but it is the first backward step in the whole
record.

The weight norm is the loudest number on the page. It rose from 274.19 to 286.34
across eight tasks, then fell to 262.36 in 100 unlearning steps. The generated
weights are now 4.3 percent smaller than they were in the untrained generator.

### What this means for the indicators

This is the third of the three outcomes set out before the run, and the
strongest result against the indicators. They were asked whether they can detect
damage when it is genuinely present. On this run they report the opposite: at
the exact moment a task was destroyed, all three read healthier, and by a margin
the late stream had not produced.

So the disagreement in this project is no longer "five measurements say no
damage, three say damage, and we cannot tell which is right." The three now
have a case against them that does not depend on believing the other five.
Whatever effective rank, srank and dormant units are tracking, it moves the
healthy way when the model is damaged, so it cannot be a damage measure.

### Limits, and the one that matters most

One seed. Seed 2 is the obvious repeat and has not run.

The weight norm is a live alternative explanation and it should be ruled out
before this is treated as settled. Both effective rank and srank divide by the
sum of the singular values, so neither can be changed by rescaling the feature
matrix. But the path from generator weights to those features is not a rescale:
it passes through ReLU and through batch normalisation with fixed statistics, so
shrinking the weights by 8.4 percent changes which units fire and can change the
shape of the spectrum without anything becoming healthier. A cheap check is to
rescale the post-unlearn generated weights back to the t=8 norm and measure
again. If the rise survives, the norm is not the cause.

This says nothing yet about whether the generator's general capacity was
damaged. That is what the direct probe measures, and the probe has not been run
on this model. The kept `boundary_9.pt` is exactly what it needs.
