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

## Batch 1: one unlearned state, every measurement

One notebook. Load the source, unlearn to step 20, stop, then measure. Every item below
needs the same unlearned model and one forward pass over the test set, so they cost one
model load and one dataset download between them.

- Measure accuracy at step 20 with no change, to confirm the joined table on a fresh run.
- Multiply the generated weights by 1.174 to restore their original length, then measure accuracy with no training.
- Replace the forgotten task's BatchNorm running statistics with neutral ones, then measure accuracy. Unlearning never updates those statistics.
- Measure effective rank of the features before unlearning, at step 20, and after rescaling.
- Record per-channel variance across images, not only peak response. A channel giving the same number for every image carries nothing.

Needs: Tiny ImageNet test split, CPU, roughly an hour including the download.

Answers: whether the task survives the method's own stopping point, and if it does, whether
the route back is the scale, the statistics, or neither.

Gate: if rescaling restores accuracy, the privacy result stands and batch 2 becomes
confirmation. If it does not, the BatchNorm arm in this same batch says whether the
statistics are carrying the removal instead.

## Batch 2: repeat batch 1 where it is hardest

Same notebook, different settings. Run only after batch 1.

- Repeat with task 14, learned last and still at high accuracy. Task 3 is first in the order and had already fallen to 35.6 percent.
- Repeat at step 100 instead of step 20. Expected to fail, since 49 degrees cannot be rescaled away, and that failure is the contrast that makes step 20 meaningful.

Answers: whether the batch 1 result depends on an easy task or an early stopping point.

## Batch 3: more than one seed

One run of notebook 30 with `SEEDS = [0, 1, 2]` and `INTERVAL = 5`. The readings are
weight-only and nearly free, so the finer interval costs almost nothing and the extra seeds
are the only real cost.

- Confirms the direction trajectory holds beyond seed 0.
- Locates where accuracy reaches chance inside steps 1 to 20, instead of bracketing it.

Needs: about one hour CPU. Independent of batch 1, so it can run while batch 1 is written.

## Batch 4: tell the group

- Send Andrii one page with the joined table. The meeting notes say the spoken version did not land.
- Measure effective rank on our t = 0 and t = 8 states, beside the existing direct plasticity numbers. Fills the architecture cell Matas cannot reach.

The second item needs a forward pass on two saved states and no retraining. Both are
cheap, and the page should go this week regardless of batch 1.

## Batch 5: plasticity infrastructure, blocked

Every item here is blocked behind the first one, and every item is unblocked by it.

- Retrain the stream saving a checkpoint after every task. `checkpoint.save` already takes a path, so this needs a per-task path and nothing else.
- Then: mid-stream probe points, orders 02 and 03, trained reference points for the probes, per-task weight geometry across the stream, and the EuroSAT external probe.

Do not start until the unlearning line is written up.

## Parked

- Experiment B baseline and Experiment C code-only arm, from the 17 October record. Each is one run and each decides whether a larger experiment is worth starting.
- Anything needing a GPU, until RunPod is ready. Nothing in batches 1 to 4 does.
