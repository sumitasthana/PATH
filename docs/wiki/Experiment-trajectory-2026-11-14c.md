# Experiment trajectory: 14 November 2026, third record

Back to the [index](Experiment-trajectory).

This page covers the direct plasticity probe on the revised streams, all three
orders. The [first record for the day](Experiment-trajectory-2026-11-14)
covers job 3 and the task-order result. The
[second](Experiment-trajectory-2026-11-14b) covers the rescaling control.

## The probe and the indicators disagree on the revised design, on all fifteen streams

The central claim of the paper was measured on the retired eight-task design.
It is now measured on the current one, on every stream, and it holds.

## What was run

Notebook 47. Each probe takes a finished model, freezes it, teaches a fresh
task code on a task the stream never learned, and reads the accuracy. Nothing
in the stream model is trained. The probe task is group 9, the twenty classes
held out of every stream, so guessing scores 5 percent.

- 15 runs: three orders, five seeds each
- 2 points per run: boundary 0, the untrained generator, and boundary 9, the
  end of the nine-task stream
- 3 learning rates, best chosen on a validation split, 11 repetitions each
- 990 probe runs of 250 steps

The noise floor is built from the spread of those repetitions. A change has to
clear its own floor before it counts.

Order A ran first, on commit 18c8054. Orders B and C were added afterwards from
the same resume file, so no run was repeated.

## The result

| Run | Probe at t = 0 | Probe at t = 9 | Change | Floor | Effective rank |
| --- | --- | --- | --- | --- | --- |
| order_A seed 0 | 26.13% | 43.72% | +17.59 | 1.70 | -29.8% |
| order_A seed 1 | 27.25% | 43.65% | +16.40 | 2.72 | -66.6% |
| order_A seed 2 | 26.06% | 43.75% | +17.68 | 2.10 | -31.7% |
| order_A seed 3 | 26.55% | 42.76% | +16.22 | 1.26 | -57.9% |
| order_A seed 4 | 26.78% | 43.90% | +17.12 | 2.07 | -65.3% |
| order_B seed 0 | 25.50% | 42.39% | +16.89 | 1.62 | -77.2% |
| order_B seed 1 | 26.35% | 43.81% | +17.45 | 1.91 | -68.9% |
| order_B seed 2 | 26.15% | 44.24% | +18.09 | 1.81 | -30.7% |
| order_B seed 3 | 26.21% | 44.03% | +17.82 | 2.63 | -60.7% |
| order_B seed 4 | 26.05% | 45.28% | +19.24 | 1.60 | -80.6% |
| order_C seed 0 | 27.41% | 43.31% | +15.90 | 1.67 | -78.8% |
| order_C seed 1 | 25.27% | 44.45% | +19.18 | 1.82 | -68.2% |
| order_C seed 2 | 24.95% | 45.44% | +20.49 | 1.48 | -69.2% |
| order_C seed 3 | 26.39% | 45.36% | +18.97 | 2.31 | -62.2% |
| order_C seed 4 | 26.01% | 45.04% | +19.03 | 2.27 | -70.3% |

Probe accuracy rose past its floor in 15 of 15 runs and fell past it in none.
Each run clears its own floor by 6.0 to 13.8 times.

| Summary over the fifteen runs | Mean | Standard deviation | Lowest | Highest |
| --- | --- | --- | --- | --- |
| Probe at t = 0 | 26.20% | 0.66 | 24.95% | 27.41% |
| Probe at t = 9 | 44.08% | 0.92 | 42.39% | 45.44% |
| Change | +17.87 | 1.30 | +15.90 | +20.49 |
| Effective rank change | -61.2% | 17.0 | -30.7% | -80.6% |

So the diagnostics say the representation collapsed. The direct measurement
says the model learns a new task far better than it did before any training.

## The fall in effective rank does not predict anything

This is the sharper result, and order A alone could not have shown it.

Across the fifteen runs the fall in effective rank varies by a factor of 2.6,
from 30.7 to 80.6 percent. If that measure reported lost ability to learn, the
runs that lost the most of it should learn the new task worst.

| Relation across the fifteen runs | Value |
| --- | --- |
| Correlation between rank fall and probe change | -0.055 |
| Share of the probe spread it explains | 0.3% |
| Probe points per 1 percent of extra rank fall | -0.004 |

The correlation is almost exactly zero, and its sign is the opposite of the
one the measure's usual reading predicts. A run can lose a third of its
effective rank or four fifths of it and learn the held-out task equally well.

Per order the two quantities move the other way from each other, which is the
same point read coarsely:

| Order | Probe change, mean | Effective rank change, mean |
| --- | --- | --- |
| order_A | +17.00 | -50.3% |
| order_B | +17.90 | -63.6% |
| order_C | +18.71 | -69.7% |

The order with the largest rank fall has the largest probe gain. With five
seeds per order and a probe spread of 1.30 points those order means are not
far apart, so this is reported as the direction of the data, not as an order
effect.

## An unplanned repeat measurement, and what it bounds

Boundary 0 is the untrained generator, rebuilt from the seed rather than loaded
from a file, and the probe seed depends only on the task and the repetition.
So each seed's boundary-0 reading was taken three times, once per order, on a
configuration that should be identical.

| Seed | Three boundary-0 readings | Spread |
| --- | --- | --- |
| 0 | 26.13, 25.50, 27.41 | 1.91 |
| 1 | 27.25, 26.35, 25.27 | 1.98 |
| 2 | 26.06, 26.15, 24.95 | 1.20 |
| 3 | 26.55, 26.21, 26.39 | 0.34 |
| 4 | 26.78, 26.05, 26.01 | 0.77 |

The readings are not identical, so something is not reproducible: either GPU
arithmetic that does not repeat exactly, or the task order reaching the
starting generator through the configuration. Which of the two it is has not
been established and is worth one cheap check.

What the table does establish is a bound. Repeating a nominally identical
measurement moves it by at most 1.98 points. The effect being measured is
+17.87 points, so the effect is about nine times that bound.

## The sizes are new, and they are larger

On the old design the probe moved by about 1 to 3 points and only some seeds
cleared their floor. Here every run clears its floor by six to fourteen times.

The reason is that the probe task changed with the split. It is now twenty
classes instead of ten, so chance is 5 percent instead of 10, and the probe
has much further it can move. The old readings and these are not comparable in
size. What is comparable is the direction and whether the floor was cleared,
and both are stronger here.

## Limits

The probe task shares a dataset with the stream, so ordinary transfer is not
ruled out. The probe trains a fresh task code on a frozen generator, so it
measures what the generator can be asked to do, not what the stream model does
unaided.

Correction, 15 November 2026. The sentence above is wrong. Every record of
this run has `arm="full"`, `code_only=false` and `train_code=true`, and in
`uncle/probe.py` `code_only=false` puts the generator's parameters in the
trainable list. So the probe trained the generator together with the fresh
code. The `code_only` arm exists in the code but was not run here. The
measurement is therefore how well the model fits an unseen task when the whole
of it is free to move, with the model restored afterwards. The numbers in this
record are unaffected; only their description was wrong. Boundary 4 was not probed, so the two endpoints could in principle
hide a dip in the middle; the models for that are kept.

## What this moves

Of the six direct measurements that say capacity held, this is the first one
re-established on the current design. Random-label fitting and curvature are
still on the retired eight-task design. Learning speed comes out of the probe
shapes this notebook already saved and needs no new run.
