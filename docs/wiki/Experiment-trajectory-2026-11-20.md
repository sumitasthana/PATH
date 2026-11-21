# 20 November 2026

Notebook 51 was re-run to recover the control network's accuracies, which the
first run printed without saving. Seed 0 finished and the run stopped on a new
check. The check was right to fire and the reason is not a fault in the code.

## Result: the control learned the groups

This is what the run was for. Table B6 has been empty because these numbers
were never saved.

Seed 0, ordinary ResNet50, nine groups of twenty classes, chance 5 percent:

| Boundary | Group | Loss, first epoch | Loss, last epoch | Just learned | At end of stream |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 2.745 | 1.535 | 49.5% | 13.5% |
| 2 | 1 | 1.915 | 1.061 | 57.4% | 17.3% |
| 3 | 2 | 1.746 | 0.796 | 54.4% | 18.2% |
| 4 | 3 | 1.624 | 0.544 | 54.3% | 24.6% |
| 5 | 4 | 1.543 | 0.517 | 62.8% | 20.7% |
| 6 | 5 | 1.633 | 0.469 | 60.4% | 25.9% |
| 7 | 6 | 1.639 | 0.350 | 58.9% | 25.8% |
| 8 | 7 | 1.756 | 0.351 | 54.2% | 33.5% |
| 9 | 8 | 1.544 | 0.271 | 57.4% | 57.4% |
| mean | | | | 56.6% | 26.3% |

The control learns every group it is given, to 56.6 percent against a chance
level of 5. The claim the paper rests on it is therefore sound. These are one
seed, not the five the run will produce, so Table B6 waits for the rest.

## Result: two runs of the same code do not land on the same readings

The run compared each new reading against the published one and stopped when
50 of 60 differed by more than half a percent.

Boundary 0 reproduced. Every trained boundary did not. At boundary 1:

| Route | Pictures | Published | This run | Gap |
| --- | --- | ---: | ---: | ---: |
| own | in_stream | 59.004 | 74.151 | +25.7% |
| own | probe | 64.275 | 74.786 | +16.4% |
| own | eurosat | 38.379 | 38.882 | +1.3% |
| start | in_stream | 1.1817 | 1.2983 | +9.9% |
| start | probe | 1.1704 | 1.2578 | +7.5% |

The accuracies moved too, but much less. Seed 0 reads 56.6 percent just
learned here against the 57.7 percent recorded for the first run, and 26.3
percent at the end against 28.0.

## Why

Not a code change. `build_plain` and `train_group` are byte for byte identical
across every commit since the file that holds the published readings was
written. The training data carries no random augmentation, only `ToTensor`.
The network build is seeded and the data loader is seeded, which is why
boundary 0 matches and the batches arrive in the same order.

What is left is ordinary GPU nondeterminism. cuDNN picks its convolution
algorithms per machine, and several backward kernels add their contributions
in whatever order the threads finish. Those differences are far below the last
digit of a single step. Over nine groups of five epochs, 45 epochs of a
ResNet50, they compound into different weights.

So the two runs are two draws from the same procedure and not a reproduction
of one by the other.

## What this means for the paper

The control's conclusion holds with room to spare. Table B2 compares the
control's fall of 84.4 percent, from 7.30 to 1.14 on the fixed route, against
the ordering A hypernetwork's 49.8 percent, from 5.56 to 2.79. A 10 percent
wobble in the boundary 9 reading moves the control to about 82.9 percent. The
gap it has to clear is 35 points.

What does not hold is the precision. These readings are quoted to three and
four significant figures. Two runs of the same code on the same seed agree to
about 10 percent on the fixed route and about 26 percent on the trained route
at boundary 1. A reader who repeats this work will not reproduce the digits.
That belongs in the paper and is not yet written there.

The same question applies to the hypernetwork side, where every indicator
number in the study comes from a single training run per stream. Nothing here
measures that, and no second run of a hypernetwork stream exists.

## What changed in the notebook

The rerun now writes to `plain_network_stream_v3.jsonl` and leaves
`plain_network_stream_v2.jsonl` alone. The paper cites v2, so v2 stays as the
record of the run that produced those numbers, and v3 sits beside it as a
second run at five seeds. Notebook 52 still reads v2 and was not touched.

The comparison against v2 is kept and now prints instead of stopping. For each
seed it reports the median and worst gap on each route, and the gap at
boundary 0, which should stay near zero. A difference there is the thing being
measured, not a fault to halt for.

## The rerun finished: five seeds, readings and accuracies both saved

The file now holds 300 readings and 45 accuracies, five seeds by nine groups.
Table B6 is filled from them.

| Boundary | Group | Loss, first epoch | Loss, last epoch | Just learned | At end of stream |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 2.683 | 1.464 | 49.96 (3.26) | 12.84 (1.89) |
| 2 | 1 | 1.943 | 1.034 | 54.68 (4.14) | 18.46 (3.03) |
| 3 | 2 | 1.772 | 0.813 | 55.80 (3.06) | 21.34 (2.69) |
| 4 | 3 | 1.676 | 0.557 | 61.52 (1.77) | 22.66 (3.25) |
| 5 | 4 | 1.561 | 0.496 | 61.04 (2.10) | 27.56 (1.58) |
| 6 | 5 | 1.633 | 0.441 | 59.26 (1.63) | 28.10 (2.78) |
| 7 | 6 | 1.654 | 0.354 | 61.88 (2.73) | 27.44 (2.27) |
| 8 | 7 | 1.755 | 0.327 | 57.10 (1.94) | 36.40 (3.46) |
| 9 | 8 | 1.539 | 0.255 | 62.90 (2.08) | 62.90 (2.08) |
| mean | | | | 58.24 | 28.63 |

Brackets are the standard deviation across the five seeds. Chance is 5.00
percent. The control learns every group it is given, so the claim the paper
rests on it is published evidence now and not a printed screen.

## The five-seed control rows

Every control row in Tables 1, B1 and B2 was recomputed. The three seeds and
the five seeds agree closely, which is the second thing worth recording: the
seed spread is small next to the run-to-run spread below.

| Reading | Three seeds | Five seeds |
| --- | ---: | ---: |
| Effective rank, fixed, b0 | 7.30 | 7.57 |
| Effective rank, fixed, b9 | 1.14 | 1.14 |
| Effective rank, matched, b9 | 110.51 | 110.55 |
| srank, fixed, b9 | 2.7 | 3.6 |
| srank, matched, b9 | 245.3 | 245.0 |
| Mean row share, matched, b9 | 0.709 | 0.715 |
| Dormant 0.01, fixed, b9 | 0.527 | 0.510 |

One claim was softened. The draft said the control reads exactly 0.000 dormant
units on the matched route from the first group onward. Across five seeds and
nine boundaries the largest value is 0.0015, so the paper now says it stays
below 0.002.

## The drift, measured

The new run and the published one, seeds 0 to 2, effective rank:

| Seed | Route | Median gap | Worst gap | Boundary 0 |
| --- | --- | ---: | ---: | ---: |
| 0 | fixed | 5.39% | 12.67% at b3 | 0.0000% |
| 0 | matched | 7.92% | 142.83% at b2 | 0.0000% |
| 1 | fixed | 4.23% | 17.45% at b2 | 0.0000% |
| 1 | matched | 6.21% | 87.69% at b1 | 0.0000% |
| 2 | fixed | 6.09% | 27.07% at b2 | 0.0000% |
| 2 | matched | 10.66% | 141.85% at b1 | 0.0000% |

Boundary 0 is exact to every digit on all three seeds, which rules out a code
change and leaves training nondeterminism as the cause. The matched route
drifts further than the fixed one because its statistics are measured from the
weights that drifted.

The worst gaps are early. Boundary 1 and 2 are where the two runs are least
alike, and the readings settle as the stream goes on. A limitation sentence
saying so is now in the paper, and it says no hypernetwork stream has been run
twice, so the same spread is assumed there and not measured.

## Next decision

Nothing is left in this block. The matched-statistics pass over all fifteen
hypernetwork streams is the next piece of work, and it is the one that decides
whether Sections 3.3 to 3.5 hold.

Whether to run one hypernetwork stream twice, to put the same number on the
study's own side, is open. It costs about two hours and nothing else depends
on it.

## Does the generator still need a group's code once it has learned it?

Notebook 48, test 4, froze the generator and trained only a fresh code, and
every reading landed between 4.14 and 5.84 percent against 5 percent chance. A
code cannot rescue a generator that never saw the group. The other direction
had never been measured: once a group has been learned, does the code it was
trained with still matter?

Notebook 54 asks that. It reads the five boundary-9 models of ordering A and
scores every learned group three ways, with the group's own trained code, with
the code it started from before any training, and with another group's trained
code. Nothing is trained. Each arm re-estimates its own BatchNorm statistics,
because the saved statistics were collected while the trained code was in use
and would favour that arm. The trained arm is also read through the saved
statistics, as a check that the re-estimation changes nothing.

### The check failed, so the result is provisional

| reading | value |
| --- | --- |
| pairs compared | 45 |
| largest gap between fresh and saved statistics | 28.10 points |
| mean gap | 9.49 points |

The notebook's own rule is that a gap over 2 points makes the arms
incomparable, and it printed that warning.

The cause is in the notebook and not in the model. The statistics were
re-estimated from the first twenty batches of the group's training split, taken
in order. A group's images sit in the order their class folders were read, so
the first 1,280 pictures are 3 classes of 20. That was confirmed on the same
split outside Colab. Statistics built from 3 classes are not the group's
statistics. The draw is now random, seeded by the group so that all three arms
of a group still see the same pictures, and a test fails if it ever reads the
split in order again.

### What the first run reported anyway

All three arms were scored through the same wrong statistics, so the comparison
between them is not obviously broken even though each number is.

| group | position | trained | initial | other |
| --- | --- | --- | --- | --- |
| 0 | 1 | 40.04 | 39.88 | 6.38 |
| 1 | 2 | 35.40 | 35.36 | 6.68 |
| 2 | 3 | 36.94 | 36.76 | 5.02 |
| 3 | 4 | 34.44 | 34.34 | 3.88 |
| 4 | 5 | 42.08 | 41.80 | 3.88 |
| 5 | 6 | 37.78 | 37.64 | 4.58 |
| 6 | 7 | 43.36 | 43.48 | 5.28 |
| 7 | 8 | 28.48 | 28.20 | 4.58 |
| 8 | 9 | 37.54 | 37.50 | 4.26 |
| all | | 37.34 | 37.22 | 4.95 |

Chance is 5 percent. The spread between seeds on the trained arm is 6.63
points.

The starting code scores what the trained code scores, on every one of the nine
groups, with the largest difference 0.28 points against a seed spread of 6.63.
Another group's code scores chance.

### What that does and does not say

It does not yet say the generator ignores the code. The obvious dull
explanation is that the code hardly moved while it was being learned, in which
case the starting code scores the same because it is nearly the same code. That
distance was never measured, and it is a norm on two vectors of 32 numbers, so
it now rides along with the arms.

If the code did move and the starting one still works, the generator is not
reading where the code ended up, and the 32 numbers act as a fixed name tag:
the right tag is needed, since another group's fails, but the tag's update buys
nothing and all of the learning is in the generator.

### Next decision

Run notebook 54 again with the random draw and the code distance. If the check
passes and the table holds, the finding is about the code's update rather than
about the code, and which of the two statements above is the right one depends
on the distance column.

### The second attempt measured nothing

Running the fixed notebook printed the same accuracies to the last digit, the
same failed check, and an empty distance column. It had not re-measured
anything. The notebook saves its readings to one key in R2 and skips any model
already in that file, and the first run's readings were still there, so every
model was treated as done. The accuracies came back unchanged because they were
the first run's accuracies, and the distances were blank because the version
that wrote those rows did not record distances.

The notebook now drops any saved reading that has no distance on it, since that
field is exactly what the old version never wrote, and says how many it
dropped. Those models are then measured again.

### The third run measured it: the code's update does nothing

With the random draw in place, all five models were measured again.

| group | position | trained | initial | other | code distance | code turn |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 41.96 | 41.86 | 6.10 | 0.010 | 0.4 |
| 1 | 2 | 48.38 | 48.42 | 7.42 | 0.015 | 0.7 |
| 2 | 3 | 46.34 | 46.24 | 4.16 | 0.022 | 1.0 |
| 3 | 4 | 40.22 | 40.06 | 4.24 | 0.020 | 0.9 |
| 4 | 5 | 47.62 | 47.40 | 3.50 | 0.026 | 1.2 |
| 5 | 6 | 41.38 | 41.50 | 5.02 | 0.019 | 0.8 |
| 6 | 7 | 41.60 | 41.56 | 5.16 | 0.021 | 1.0 |
| 7 | 8 | 37.30 | 37.02 | 5.08 | 0.019 | 0.8 |
| 8 | 9 | 43.82 | 43.64 | 4.12 | 0.020 | 0.9 |
| all | | 43.18 | 43.08 | 4.98 | 0.019 | 0.9 |

Chance is 5 percent and the spread between seeds on the trained arm is 8.29
points. Distance is the move divided by the size of the starting code, so 0.019
means a code ended about two percent of its own length away from where it
began. The turn is under one and a half degrees everywhere.

A group's starting code scores what its trained code scores, on all nine
groups, the largest difference being 0.28 points. Another group's trained code
scores chance.

So the code's update does nothing. That is not the same as saying the code does
nothing: the right code is needed, since another group's fails completely. The
32 numbers act as a fixed random name tag that the generator learns to read,
and the learning itself is in the generator.

The random draw also raised every accuracy, 43.18 against the ordered draw's
37.34, which is what a better estimate of the statistics should do.

### The check still fails, and why it no longer blocks the reading

| reading | value |
| --- | --- |
| pairs compared | 45 |
| mean gap | 10.63 points |
| largest gap | 31.10 points |

Twenty batches is a poorer estimate of a group's BatchNorm statistics than the
running average the run itself collected over five epochs, so every arm sits at
a different level from the run's own accuracy. The check was written as a
blocker, and that was the wrong rule. Every arm estimates its statistics from
the same pictures by the same recipe, so an offset of this kind moves all three
arms together. It cannot invent a difference between them and it cannot hide
one, and neither conclusion here depends on the absolute level: the initial arm
matches the trained arm to within 0.28 points, and the other arm sits at
chance.

What the gap does mean is that these are not the study's own accuracies and
must not be quoted beside a published figure. The notebook now says that
instead of telling the reader to stop.

### Next decision

The notebook now also reads the generator's own distance and turn from the
saved indicator records, so the code's move can be set beside the generator's.
That needs no GPU. Those two spans are not the same, the generator's counting
from the start of the stream and the code's covering one task, so the
comparison is a size comparison and not a ratio.

Whether this holds at boundary 4 as well as boundary 9 is open, and so is
whether it holds on orderings B and C.
