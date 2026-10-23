# TODO

Working backlog, sequenced. Last updated 23 October 2026.

Experiment results live in [the experiment trajectory](wiki/Experiment-trajectory.md).
The implementation plan lives in [PLAN.md](PLAN.md). This file is only what is next.

## Goal

Show that this unlearning reaches chance accuracy without removing the task, and that the
method's own stopping point leaves the task recoverable.

Two measurements on 23 October are most of the argument already. Accuracy on the removed
task reaches chance at step 20 of 100, where the weights are still 99.19 percent aligned
with where they started. And chance accuracy turns out to be a model answering one class for
every image, which accuracy alone cannot distinguish from a model that genuinely forgot.

Batches 1 to 3 serve this goal. Batch 4 serves the group. Batch 5 is the plasticity line and
waits.

## Already done

- Direction measurement during unlearning, seed 0, task 3. Notebook 30. Weights turn 49 degrees and lose 43 percent of their length over 100 steps.
- Accuracy joined to direction on the step number. Notebook 31. Join validated to 0.13 percent.
- `measure.direction_change` in the library, with tests.

All four batches below are built and ready to run. They are independent of each other and
none needs a GPU. Run them in any order.

## Batch 1: one unlearned state, every measurement

**Notebook 32, as it stands. Roughly 25 minutes on CPU including the dataset download.**

Loads the source model, unlearns to step 20, stops, then scores the same model six ways with
no training anywhere: before unlearning, unlearned, rescaled as a whole, rescaled per tensor,
with neutral BatchNorm statistics, and rescaled with neutral statistics. Each arm reports
accuracy, classes used, input sensitivity and effective rank. Every arm also runs on the
kept task as a control.

Answers: whether the task survives the method's own stopping point, and if it does, whether
the route back is the weight scale, the running statistics, or neither.

Gate: if rescaling restores accuracy, the privacy result stands and batch 2 becomes
confirmation. If it does not, the BatchNorm arm in the same run says whether the statistics
are carrying the removal instead.

Dropped from this batch: per-channel activation variance. It needs the hook machinery that
notebook 27 already has, so it belongs there as a one-line addition rather than here.
Effective rank covers the feature-level version of the same question.

## Batch 2: repeat batch 1 where it is hardest

**Notebook 32 with two settings changed. No new code.**

- `FORGET_TASK = "14"`, learned last and still at high accuracy. Task 3 is first in the order and had already fallen to 35.6 percent.
- `STOP_AT = 100`, the end of unlearning. Expected to fail, since 49 degrees cannot be rescaled away, and that failure is the contrast that makes step 20 meaningful.

Run these only after batch 1, because batch 1 decides whether either is worth the time.

## Batch 3: more than one seed

**Notebook 30, as it stands. About one hour on CPU.**

The readings are weight-only and nearly free, so the finer interval costs almost nothing and
the extra seeds are the only real cost.

- Confirms the direction trajectory holds beyond seed 0.
- Locates where accuracy reaches chance inside steps 1 to 20, instead of bracketing it.

## Batch 4: tell the group

**Notebook 33, which runs nothing. Seconds, no GPU, no images.**

Notebook 29 already recorded effective rank beside the accuracy at every probe point, so
this downloads that file and lays the two side by side. It says whether the literature's
metric and direct measurement agree in our architecture, which is the control for Matas's
head learning rate explanation.

The one-page note for Andrii is written, at
`ops-docs/andrii-hypernetwork-plasticity-note.md`. Send it this week regardless of the other
batches. It carries two caveats that need settling before any shared table: our effective
rank is measured on the model after it has adapted to the probe task, not on the stream
model itself, and ours is an absolute count out of 768 while his 0.6 and 0.94 look
normalised.

## Batch 5: plasticity infrastructure, blocked

Every item here is blocked behind the first one, and every item is unblocked by it.

- Retrain the stream saving a checkpoint after every task. `checkpoint.save` already takes a path, so this needs a per-task path and nothing else.
- Then: mid-stream probe points, orders 02 and 03, trained reference points for the probes, per-task weight geometry across the stream, and the EuroSAT external probe.

Do not start until the unlearning line is written up.

## Parked

- Experiment B baseline and Experiment C code-only arm, from the 17 October record. Each is one run and each decides whether a larger experiment is worth starting.
- Anything needing a GPU, until RunPod is ready. Nothing in batches 1 to 4 does.
