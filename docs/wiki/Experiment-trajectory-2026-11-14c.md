# Experiment trajectory: 14 November 2026, third record

Back to the [index](Experiment-trajectory).

This page covers the direct plasticity probe on the revised streams, order A.
The [first record for the day](Experiment-trajectory-2026-11-14) covers job 3
and the task-order result. The [second](Experiment-trajectory-2026-11-14b)
covers the rescaling control.

## The probe and the indicators disagree on the revised design too

The central claim of the paper was measured on the retired eight-task design.
It is now measured on the current one, and it holds.

## What was run

Notebook 47. Each probe takes a finished model, freezes it, teaches a fresh
task code on a task the stream never learned, and reads the accuracy. Nothing
in the stream model is trained. The probe task is group 9, the twenty classes
held out of every stream, so guessing scores 5 percent.

- 5 runs: order A, seeds 0 to 4
- 2 points per run: boundary 0, the untrained generator, and boundary 9, the
  end of the nine-task stream
- 3 learning rates, best on a validation split, 11 repetitions each
- 330 probe runs of 250 steps

The noise floor is built from the spread of those repetitions. A change has to
clear its own floor before it counts.

## The result

| Run | Probe at t = 0 | Probe at t = 9 | Change | Floor | Effective rank |
| --- | --- | --- | --- | --- | --- |
| order_A seed 0 | 26.13% | 43.72% | +17.59 | 1.70 | -29.8% |
| order_A seed 1 | 27.25% | 43.65% | +16.40 | 2.72 | -66.6% |
| order_A seed 2 | 26.06% | 43.75% | +17.68 | 2.10 | -31.7% |
| order_A seed 3 | 26.55% | 42.76% | +16.22 | 1.26 | -57.9% |
| order_A seed 4 | 26.78% | 43.90% | +17.12 | 2.07 | -65.3% |

Probe accuracy rose past its floor in 5 of 5 runs and fell past it in none.
Mean change +17.00 points. Over the same span effective rank fell by 50.2
percent on average, in 5 of 5 runs.

So the diagnostics say the representation collapsed. The direct measurement
says the model learns a new task far better than it did before any training.

## The sizes are new, and they are larger

On the old design the probe moved by about 1 to 3 points and only some seeds
cleared their floor. Here every seed clears its floor by six to thirteen times.

The reason is the probe task changed with the split. It is now twenty classes
instead of ten, so chance is 5 percent instead of 10, and the probe has much
further it can move. The old readings and these are not comparable in size.
What is comparable is the direction and whether the floor was cleared, and
both are stronger here.

## The seed spread falls on one side only

The first record for the day found that order A's seeds split into two groups
on effective rank, around 30 percent and around 60 percent, with no
explanation. That split is still here: seeds 0 and 2 read -29.8 and -31.7,
seeds 1, 3 and 4 read -66.6, -57.9 and -65.3.

The probe does not split at all. Its five readings span 1.46 points, from
+16.22 to +17.68, and the two rank clusters land on top of each other.

**So the quantity that doubles between seeds has no effect on the thing it is
supposed to predict.** A seed can lose a third of its effective rank or two
thirds of it and learn the new task equally well either way. That is a
stronger statement against the measure than the average disagreement is,
because it is a within-design comparison with the seed as the only difference.

## Limits

Order A only. Orders B and C are 660 more probe runs and resume from the same
file, so nothing already measured is repeated.

The probe task shares a dataset with the stream, so ordinary transfer is not
ruled out. The probe is a fresh task code on a frozen generator, so it measures
what the generator can be asked to do, not what the stream model does unaided.

## What this moves

Of the six direct measurements that say capacity held, this is the first one
re-established on the current design. Random-label fitting and curvature are
still on the retired eight-task design. Learning speed comes out of the probe
shapes this notebook already saved and needs no new run.
