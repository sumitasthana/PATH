# Experiment trajectory: 14 November 2026, second record

Back to the [index](Experiment-trajectory).

This page covers the rescaling control. The [first record for the
day](Experiment-trajectory-2026-11-14) covers job 3 and the task-order result.

## Breaking a task on purpose raises effective rank in all fifteen runs, and holding the weight size fixed does not take it away

This is the strongest result the study has, and it is now measured on fifteen
runs instead of one.

## What was run

Four readings per pair, all forward passes, nothing trained:

- the model at the end of the nine-task stream
- the model after one task was unlearned
- the post-unlearn weights multiplied by one constant so their norm matches the
  earlier model
- the earlier weights shrunk to the post-unlearn norm

Fifteen pairs: three orders, five seeds each. About 16.2 GB of model files read
from R2.

## The result

| | Effective rank, in-stream images |
| --- | --- |
| What the unlearn did | **+1.580** on average, the same direction in **15 of 15** runs |
| What is left after the weight size is held fixed | **+0.754** on average, the same direction in **14 of 15** |
| So the weight size explains | **52%** of it |

The same pattern holds on the probe images and on EuroSAT, and on srank.

Effective rank rose in every single run after a task was destroyed. Rank going
up is the direction these measures call healthy.

## How big is that, really

The comparison the earlier record lacked. Taken from the same fifteen runs:

| Step | Change in effective rank |
| --- | --- |
| An ordinary learning step, tasks 2 through 9, 120 of them | 0.162 in typical size |
| The unlearn step, 15 of them | +1.581 |

So the unlearn moves effective rank about **9.7 times** as far as an ordinary
learning step, in the opposite direction. After the weight size is held fixed
it is still **4.7 times** an ordinary step.

A measure that barely notices nine tasks of learning, and then jumps ten times
further in the healthy direction when a task is destroyed, is not reporting
damage.

## This reverses two things from 8 November

**The two thirds figure does not replicate.** On 8 November this control was
run on a single pair from the old study and found the weight size carried
about 67 percent of the rank rise. Across fifteen pairs it is 52 percent. One
pair could not have told the difference, which is the same lesson the
task-order result taught.

**The size claim, withdrawn on 8 November, is supported again.** That
withdrawal rested on the leftover being +0.098 on one pair, which sat inside
the range of ordinary late-stream steps. Across fifteen pairs the leftover is
+0.754, which is 4.7 times an ordinary step. The claim should be reinstated
with the new number, and the record should say it was withdrawn on one pair
and restored on fifteen.

## A statistic that was reported wrongly, and is now fixed

The notebook first printed the share as 29 percent with a spread of 62 points.
That number came from averaging each run's own ratio of leftover to raw
change.

That statistic is unsafe here. It divides by each run's own change, and some of
those changes are near zero, which sends a single run's share to -78 percent or
+124 percent and drags the average with it.

The ratio of the averages, 52 percent, does not have that problem, and it is
what the notebook now reports. The per-run ratios still print underneath,
labelled as a diagnostic.

Nothing was measured wrongly. The readings were correct and only the summary
line was the wrong statistic.

## Why this matters for the paper

The headline finding no longer needs the old design to support it.

Before today, the claim that these diagnostics call a broken model healthy
rested on one run of the eight-task study, with an open question about whether
the weight scale explained it away. It now rests on fifteen runs of the current
design, with the weight scale measured and only accounting for about half.

## What is still on the old design

The six direct measurements that say the model keeps its ability to learn, the
probe, random labels, learning speed and curvature, are still measured on the
retired eight-task design. Re-running those against the fifteen current streams
is the remaining work.
