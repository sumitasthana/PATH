# Experiment trajectory: 18 November 2026

Notebook 53, `53_curvature_on_the_revised_streams.ipynb`, measured curvature on
the current design. Notebook 42 had measured it on the retired eight-task
design and found that the answer depends on where you stand to read it. This
run repeats that on `plasticity_revised_v1`, the design the paper describes,
and the separation is far larger than it was on the old design.

## What curvature means here

Think of the training loss as a landscape. The model stands at one point on it.
Curvature says how sharply the ground bends there. A flat valley is easy to move
around in; a narrow steep one is not. The plasticity literature says a model
trained on many tasks in a row drifts into sharp places, and that this is one
reason it stops being able to learn.

The measurement steps the generator a small distance in a random direction,
reads the loss, comes back, steps the same distance the other way and reads the
loss again. Averaging the two and subtracting the loss in the middle cancels the
slope and leaves the bend. Averaged over random directions this is the mean
eigenvalue of the Hessian.

## Settings

Intended and actual were the same. Nothing was trained.

| Setting | Value |
| --- | --- |
| Study | `plasticity_revised_v1` |
| Ordering | A, seeds 0 to 4 |
| Reference arm | fixed code seed 20261030, clean BatchNorm template, group 9 pictures |
| Reference arm boundaries | 0, 4, 9 |
| Learned arm | group 0's trained code, group 0's own BatchNorm statistics, group 0 pictures |
| Learned arm boundaries | 4, 9 |
| Step sizes | 0.001, 0.003, 0.01 of the generator's own length |
| Directions per step size | 40 |
| Pictures per reading | 256, the same set the indicators use |
| Measurements | 3,000, all completed |

The learned arm starts at boundary 4 and not at boundary 1, where group 0 is
first learned, because only boundaries 4, 9 and 10 are kept in storage for this
study. The run time was not recorded.

## The reference arm: curvature rises by twelve orders of magnitude

| Seed | Boundary 0 | Boundary 9 | Ratio | Floor | Verdict |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.025 | 2.784e10 | 1.11e12 | 2.502e9 | moved |
| 1 | 0.06752 | 4.02e11 | 5.95e12 | 3.755e10 | moved |
| 2 | 0.6707 | 3.628e10 | 5.41e10 | 3.879e9 | moved |
| 3 | 248.7 | 1.126e12 | 4.53e9 | 1.412e11 | moved |
| 4 | 0.008776 | 5.73e5 | 6.53e7 | 7.361e4 | moved |

All three step sizes agree: 5 of 5 seeds rise at 0.001, at 0.003 and at 0.01.
Read raw, curvature joins effective rank, srank and dormant units in calling the
model damaged, and it does so far more loudly than any of them.

## The learned arm: no consistent movement

| Seed | Boundary 4 | Boundary 9 | Change | Floor | Verdict |
| --- | --- | --- | --- | --- | --- |
| 0 | 3.483e-6 | -2.202e-5 | -2.550e-5 | 6.299e-6 | moved |
| 1 | 2.331e-5 | 2.701e-5 | 3.704e-6 | 1.207e-5 | flat |
| 2 | -5.117e-7 | 2.421e-5 | 2.473e-5 | 7.546e-6 | moved |
| 3 | 1.518e-5 | 9.395e-6 | -5.783e-6 | 2.603e-6 | moved |
| 4 | 2.673e-5 | 5.516e-5 | 2.844e-5 | 1.212e-5 | moved |

Four of five clear their floor, but two rise and two fall, and the three step
sizes disagree with each other: 2, 3 and 3 seeds up out of 5. So this arm shows
movement with no direction, which is not the same as the clean flat reading
notebook 42 got on the old design. What it does not show is a rise.

The sizes are the point. Every reading on this arm is between about 1e-6 and
6e-5. Every reading on the reference arm is between 0.0088 and 1.1e12. The two
arms differ by ten to sixteen orders of magnitude on the same models at the same
boundaries.

## The reference point is not a slightly wrong place to stand

Chance for a group of twenty classes is a loss of 2.996.

| Seed | Reference, b0 | Reference, b9 | Learned, b4 | Learned, b9 |
| --- | --- | --- | --- | --- |
| 0 | 74.86 | 5.819e13 | 1.827 | 2.374 |
| 1 | 276.4 | 9.392e14 | 1.977 | 2.485 |
| 2 | 1878 | 8.14e13 | 2.600 | 2.325 |
| 3 | 6.632e5 | 3.52e15 | 1.969 | 2.200 |
| 4 | 26.72 | 1.39e9 | 2.329 | 2.351 |

The learned arm sits below chance at every point, which is what a model that
knows the group should do. The reference arm sits at ten to the thirteenth or
worse at boundary 9, and it is already far above chance at boundary 0.

This is the same 1e14 scale the four-arm BatchNorm test reported on 16 November
for the protocol route. The reference code and the clean template send the
network's activations off the scale, and the loss follows.

## The rise is the loss scale

Dividing each curvature reading by the loss at the same point removes the rise
on every seed.

| Seed | Curvature ratio | Loss ratio | Scaled, b0 | Scaled, b9 | Scaled ratio |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.11e12 | 7.77e11 | 3.34e-4 | 4.78e-4 | 1.43 |
| 1 | 5.95e12 | 3.40e12 | 2.44e-4 | 4.28e-4 | 1.75 |
| 2 | 5.41e10 | 4.33e10 | 3.57e-4 | 4.46e-4 | 1.25 |
| 3 | 4.53e9 | 5.31e9 | 3.75e-4 | 3.20e-4 | 0.85 |
| 4 | 6.53e7 | 5.20e7 | 3.28e-4 | 4.12e-4 | 1.26 |

Raw curvature spans fourteen orders of magnitude across these ten points.
Curvature divided by the loss at the same point stays between 2.44e-4 and
4.78e-4 at every one of them, a spread of less than a factor of two. The
sharpening tracks the loss and carries almost nothing of its own.

## What this means

Curvature is not a rank measure. It does not read features, it does not divide
by a norm, and it has no shared machinery with effective rank, srank or the
dormant count. It splits the same way they do. Read at a code that belongs to no
task, on statistics that belong to no weights, it says the model has been
damaged, by twelve orders of magnitude. Read at a code the model trained, on the
statistics that go with it, it says nothing has happened.

So the finding is not about the three indicators the paper measures. It is about
the measuring point. A measure that reads a model at a place the model never
visits reports the state of that place.

This is the old design's conclusion repeated on the current one, with a much
larger separation: notebook 42 saw a rise of 13.7, 6.5 and 5.5 times on the
reference arm, and this run sees 1e9 to 1e12.

## What this does not establish

This is the mean Hessian eigenvalue, taken over random directions by central
difference. Most of the sharpness literature reports the largest eigenvalue,
which needs Hessian-vector products and a power iteration. That has never been
measured in this project.

The learned arm follows one group, group 0, which is the first in ordering A.
A group learned later in the stream might behave differently. Only ordering A
was read.

The learned arm starts at boundary 4, so the first four groups of learning are
not covered on that arm.

Seed 2 reads a small negative mean curvature at boundary 4, which means that
point is not a local minimum in the averaged sense. This was not investigated.

Curvature divided by loss staying near constant is evidence that the rise is the
loss scale. It is not a proof that the shape of the surface is unchanged.

## Decision

Whether this becomes a section of version 3 is open. The result is a second,
independent demonstration of the paper's leading claim, on a measure that shares
nothing with the three it currently uses.
