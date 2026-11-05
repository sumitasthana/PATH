# Experiment trajectory: 4 November 2026

[Back to the experiment index](Experiment-trajectory)

## Curvature, run 2

Run 1 is recorded on the [3 November page](Experiment-trajectory-2026-11-03). It used five
random directions per point, and five was too few: the spread between directions at the end
of the stream was 17 to 46 times the spread at the start, so its noise floor, built from the
start alone, was far too narrow at the far end.

Run 2 (`curvature2`) fixed two things and added an arm. Forty directions instead of five. A
noise floor built from the spread at both ends, `1.96 * sqrt(s_first**2 / n + s_last**2 / n)`,
printed beside the old narrow floor so the difference stays visible. And a second arm,
because run 1's measurement point had a loss of 12443 where chance for ten classes is 2.303.
2040 measurements, both arms reading the same nine model files.

### The reference arm: sound, and it answers the question

This arm is run 1 repeated with forty directions. It stands at the fixed task code that
belongs to no task and reads the loss on held-out task 2, which is where every other
indicator in this project reads.

| Seed | t | Loss | Mean curvature | Spread | vs t = 0 | Curvature / loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 68.55 | 0.03953 | 0.0206 | 1.00 | 5.77e-4 |
| 0 | 8 | 1786.36 | 0.5411 | 0.215 | 13.7 | 3.03e-4 |
| 1 | 0 | 1089.62 | 0.2264 | 0.106 | 1.00 | 2.08e-4 |
| 1 | 2 | 7840.80 | 1.281 | 0.636 | 5.66 | 1.63e-4 |
| 1 | 4 | 12443.04 | 4.069 | 1.49 | 18.0 | 3.27e-4 |
| 1 | 6 | 13487.95 | 1.463 | 1.26 | 6.46 | 1.09e-4 |
| 1 | 8 | 8363.18 | 1.470 | 1.04 | 6.50 | 1.76e-4 |
| 2 | 0 | 42.74 | 0.01532 | 0.00482 | 1.00 | 3.59e-4 |
| 2 | 2 | 249.11 | 0.06152 | 0.0292 | 4.02 | 2.47e-4 |
| 2 | 4 | 463.57 | 0.07307 | 0.0592 | 4.77 | 1.58e-4 |
| 2 | 6 | 478.65 | 0.1056 | 0.0543 | 6.90 | 2.21e-4 |
| 2 | 8 | 1140.99 | 0.08478 | 0.127 | 5.53 | 0.74e-4 |

#### The raw numbers rise, and now the rise is real

| Seed | Change | Floor from both ends | Floor from the start alone | Clears by |
| --- | ---: | ---: | ---: | ---: |
| 0 | +0.5016 | 0.0670 | 0.00903 | 7.5 |
| 1 | +1.244 | 0.3227 | 0.0465 | 3.9 |
| 2 | +0.06946 | 0.0392 | 0.00211 | 1.8 |

All three clear the honest floor. Run 1 could not say this: at five directions its seed 2
endpoint read 0.02237 against 0.08478 here, so the number itself was wrong by about four
times, not merely uncertain. More directions changed the answer, which is the point of
having run them.

The three step sizes now agree far better than at five directions. Seed 0 at t = 8 reads
0.541, 0.360 and 0.315, and seed 1 at t = 4 reads 4.07, 3.75 and 3.24. Both drift mildly
downward as the step grows, in the same direction, which is what a step slightly too large
looks like, so the smallest radius is the one read above. Seed 2 at t = 8 still spans
0.085 to 0.174, and its spread there, 0.127 against a mean of 0.0848, says that point is
the noisiest in the arm.

#### Divided by the loss, nothing rises

The last column of the table. Cross-entropy curvature carries the scale of the loss with
it, and the loss at these points rises by 26, 7.7 and 27 times along the stream. Dividing
one by the other:

| Seed | At t = 0 | At t = 8 | Ratio |
| --- | ---: | ---: | ---: |
| 0 | 5.77e-4 | 3.03e-4 | 0.53 |
| 1 | 2.08e-4 | 1.76e-4 | 0.85 |
| 2 | 3.59e-4 | 0.74e-4 | 0.21 |

Every seed falls. So the raw rise and the loss rise move together, and the surface measured
in units of its own loss did not get sharper on any seed.

#### What the reference arm says

Read as the literature reads it, the raw mean curvature rises along the stream on all three
seeds and clears a properly built floor. That puts curvature with effective rank, srank and
dormant units: a measurement reporting damage where the direct probe, which trains an unseen
task and counts the score, finds none.

Read with the loss scale divided out, it rises on none. Which of those two is the right
reading is not settled by this run. The honest statement is that the rise exists and is not
noise, and that it is not separable from the rise in the loss at the same point with the
evidence here.

Seed 1 still traces no shape across the stream: 0.226, 1.281, 4.069, 1.463, 1.470, peaking
at t = 4. Seed 2 is closer to steady: 0.0153, 0.0615, 0.0731, 0.1056, 0.0848. Neither is the
slow climb into a sharp place the literature describes.

### The learned arm: discarded, measured on the wrong statistics

The second arm was meant to answer the objection the reference arm cannot. It stands at the
code the model trained for task 3, the first task in order 01, on task 3's own pictures,
which is a place the stream actually settled rather than a code belonging to nothing.

It reported losses of 41,196,425 and 55,298,938 on seed 1, and 102,770,726 and 127,400,100
on seed 2. Chance for a ten-class task is 2.303. Seed 0 at the same boundary read 18,070,
three to four orders of magnitude away from the other two. Forgetting does not vary between
seeds like that.

The cause is in `uncle/trainer.py`, in the first lines of the file. The hypernetwork writes
the target's weights; it does not write the target's buffers, and BatchNorm's running mean
and variance are buffers. So `UnCLe` keeps a separate set of running statistics for every
task, updated only while that task is being learned, and `trainer.accuracy` runs a task's
network on that task's own set.

The notebook passed `trainer.buffer_template`, the clean set every new task starts from, to
both arms. For the reference arm that is right and deliberate: it stands at a code belonging
to no task, so no task's statistics are its own, and the clean template is both the only
defensible choice and the one every indicator in this project uses. For the learned arm it
is wrong. A trained network run on statistics it was never calibrated for produces garbage
at every layer, and the loss follows.

So the learned arm measured the bend of a surface that does not correspond to any model the
stream produced. Its rows are discarded. The reference arm is unaffected: its rows were
already measured the only way they can be, and they are kept.

#### The correction

Run 3 gives each arm the statistics that belong to it, the clean template for the reference
arm and `trainer.task_buffers[LEARNED_TASK]` for the learned arm, which the checkpoints
carry. Every row now records which set it used. On load, any row whose recorded set does not
match what its arm now asks for is dropped and measured again, so the correction applies
itself: run 2's reference rows match and stay, run 2's learned rows do not and go, and
running the notebook twice after the fix changes nothing.

The notebook also refuses to continue if task 3's loss at the boundary right after it was
learned is more than fifty times chance. That is the check that would have stopped run 2
within its first minute of the learned arm.

Run 3 costs the learned arm only, 600 measurements, on the same nine model files already
needed.

## What is still open

Whether the raw rise or the loss-scaled flatness is the right reading of the reference arm.
The learned arm is the evidence that would settle it, which is why it is worth measuring
properly rather than dropping.

The largest Hessian eigenvalue rather than the mean. Most of the sharpness literature
reports the largest. It needs Hessian-vector products and a power iteration, and it is a
separate notebook.

## Curvature, run 3: the learned arm, measured properly

Run 3 remeasured the learned arm on the statistics that belong to it. 600 measurements, no
new downloads.

### The fix is confirmed by the losses alone

| Seed | t | Loss, run 2 | Loss, run 3 |
| --- | ---: | ---: | ---: |
| 0 | 8 | 18,070 | 2.4403 |
| 1 | 2 | 41,196,425 | 2.4780 |
| 1 | 8 | 55,298,938 | 2.0607 |
| 2 | 2 | 102,770,726 | 1.5634 |
| 2 | 8 | 127,400,100 | 2.1156 |

Chance for a ten-class task is 2.303. The run 3 losses sit around it, which is where a
partly forgotten task belongs.

They can be checked against something the runs recorded for themselves. Each source run
saved `final_accuracies`, and task 3 ends the stream at 34.8 percent on seed 1 and 32.0
percent on seed 2, against 10 percent for chance. A cross-entropy near 2.06 and 2.12 is what
those accuracies look like. So the arm is now standing on a real model, and run 2's rows
were the artifact they appeared to be.

One thing to note rather than explain away: seed 1's loss on task 3 falls from 2.478 at
boundary 2 to 2.061 at boundary 8, so it gets better at an old task as the stream goes on,
while seed 2 gets worse, 1.563 to 2.116. Two seeds, opposite directions, so this is variance
and not a finding.

### The result

| Seed | t | Loss | Mean curvature | Spread | vs first | Curvature / loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 2.4403 | 4.043e-4 | 4.91e-4 | | 1.657e-4 |
| 1 | 2 | 2.4780 | 4.287e-4 | 4.56e-4 | 1.000 | 1.730e-4 |
| 1 | 8 | 2.0607 | 4.554e-4 | 6.27e-4 | 1.062 | 2.210e-4 |
| 2 | 2 | 1.5634 | 1.299e-4 | 5.22e-4 | 1.000 | 0.831e-4 |
| 2 | 8 | 2.1156 | 1.907e-4 | 5.14e-4 | 1.467 | 0.901e-4 |

Neither seed moves. Seed 1 changes by 2.67e-5 against a floor of 2.402e-4; seed 2 by 6.07e-5
against 2.269e-4. Both are well inside their floors, so both are no finding.

Two things stand out before the comparison. The curvature at a trained code is three to four
orders of magnitude smaller than at the reference code: 1.3e-4 to 4.6e-4 here against 0.015
to 4.07 there. And the spread between directions is larger than the mean at every point, so
individual directions scatter around zero.

### Why no finding is informative here

A floor wider than the change usually means the run was too small to say anything. Here it
does not, because the question is whether the rise the reference arm found also appears at a
trained code, and that rise is large.

| Seed | Smallest rise this arm could detect | What the reference arm reported | That rise would clear the floor by |
| --- | ---: | ---: | ---: |
| 1 | 1.56x | 6.50x | 9.8x |
| 2 | 2.75x | 5.53x | 2.6x |

So this arm has the power to see a rise the size the reference arm reported, on both seeds,
and it does not see one. What it cannot rule out is a smaller change: anything under 1.56x
on seed 1 and under 2.75x on seed 2 would be invisible here.

### What runs 2 and 3 say together

The reference arm, standing at a code belonging to no task where the loss runs into the
thousands, finds the loss surface getting sharper along the stream by 13.7, 6.5 and 5.5
times, clearing an honest floor on all three seeds.

The learned arm, standing at a code the model trained with the statistics that go with it,
finds nothing of the sort, and had enough power to find it.

Read together, the sharpening is a property of the reference point rather than of the place
the model settles. That also fits what dividing by the loss already suggested in the
reference arm, where every seed fell once the loss scale was taken out. Two separate routes
to the same conclusion: the rise in raw curvature tracks the rise in loss at a point the
model was never trained to occupy, and it does not appear at a point it was.

On the reading rule written down before run 1, curvature does not join effective rank, srank
and dormant units. It goes with the direct probe and random-label fitting: no plasticity
loss.

### What this still does not establish

Three seeds, and only two of them have two points on the learned arm. The learned arm
follows one task, the first in the order, so a task learned later in the stream might behave
differently. And this is the mean Hessian eigenvalue, not the largest, which is what most of
the sharpness literature reports; that needs Hessian-vector products and a power iteration
and is a separate notebook.

A smaller sharpening at the learned point is not ruled out. Settling seed 2 down to its own
measured change would need about 560 directions per point instead of 40, which is roughly
2800 measurements on one radius and no new downloads. Seed 1 would need about 3200, which is
not worth buying.
