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

## The matched-statistics pass finished on all fifteen streams

Notebook 55 read every stream at boundaries 0, 9 and 10, twice: through the
fixed statistics the study uses everywhere, and through statistics measured
from each model's own weights. It also repeated the rescaling control through
both routes. No training. The sanity gate passed on ordering A before the other
ten streams started.

This is the run that was supposed to decide whether sections 3.3 to 3.5
survive the correction. It does decide it, and the answer has two halves that
point different ways.

### The direction of travel is a property of the route

The probe improves over these streams. An indicator that tracked the ability to
learn should improve with it: rank measures up, dormant units down.

| Measure | Route | Mean change, b0 to b9 | Streams moving as predicted |
| --- | --- | ---: | ---: |
| Effective rank | fixed | -3.469 | 0 of 15 |
| Effective rank | matched | +34.376 | 15 of 15 |
| srank | fixed | -134.467 | 0 of 15 |
| srank | matched | +0.000 | 6 of 15 |
| Dormant 0.01 | fixed | +0.182 | 0 of 15 |
| Dormant 0.01 | matched | +0.133 | 0 of 15 |
| Dormant 0.025 | fixed | +0.184 | 0 of 15 |
| Dormant 0.025 | matched | +0.159 | 0 of 15 |
| Dormant 0.1 | fixed | +0.179 | 0 of 15 |
| Dormant 0.1 | matched | +0.225 | 0 of 15 |

Effective rank turns completely over. Read through the fixed protocol it falls
on every one of the fifteen streams; read through measured statistics it rises
on every one of the fifteen. The claim that it points the wrong way is a claim
about the protocol and not about the measure.

srank is saturated on the matched route, 248.2 at both ends against a ceiling
of 256, so it reports nothing either way.

The dormant fraction fails under both routes. It rises while the ability to
learn improves, on all fifteen streams, whichever statistics it is read
through. That one is not the protocol.

### The absence of a relationship is not a property of the route

`scripts/upgrade_suite/matched_correlations.py` runs the paper's own
correlation test on the new readings. The probe side is untouched: the same
selection rule, the same eleven repetitions, imported from the script that
produced the published columns.

Change test, indicator movement over the stream against probe gain:

| Measure | fixed, N=15 | fixed, N=5 | matched, N=15 | matched, N=5 |
| --- | ---: | ---: | ---: | ---: |
| Effective rank | +0.097 | +0.340 | -0.154 | +0.148 |
| srank | -0.189 | +0.269 | -0.121 | +0.118 |
| Dormant 0.01 | +0.113 | +0.021 | +0.229 | +0.163 |
| Dormant 0.025 | +0.110 | +0.021 | +0.257 | +0.186 |
| Dormant 0.1 | +0.100 | +0.052 | +0.286 | +0.219 |

End-value test, indicator at boundary 9 against probe at boundary 9:

| Measure | fixed, N=15 | fixed, N=5 | matched, N=15 | matched, N=5 |
| --- | ---: | ---: | ---: | ---: |
| Effective rank | -0.165 | -0.352 | -0.618 | -0.362 |
| srank | -0.212 | -0.049 | -0.571 | +0.022 |
| Dormant 0.01 | +0.165 | +0.187 | +0.305 | +0.249 |
| Dormant 0.025 | +0.160 | +0.172 | +0.334 | +0.267 |
| Dormant 0.1 | +0.140 | +0.180 | +0.371 | +0.294 |

Nothing in either table is separable from zero. Fifteen runs need 0.514 and
five seeds need 0.878, and the largest value anywhere is 0.618 at a count that
overstates the evidence. Correcting the reading does not make any of these
measures predict how much a stream gains on the probe. Where the correction
moves a number at all it usually moves it further from the predicted sign, not
closer.

So the paper's central claim survives the correction, and the claim it used to
make about the sign does not. These indicators do not track the ability to
learn, under either reading. What the fixed protocol added was a false story
about direction.

The script checks itself against the published log before it reports: the fixed
route re-read from the checkpoints lands within 0.143 percent of the study's
own numbers on the worst of the fifteen streams, and its percent-change column
reproduces Table D2 to the third decimal.

### The rescaling confound is entirely the protocol

Notebook 46 found that about two thirds of the rank change at the unlearn came
from the weights simply getting smaller. Across all fifteen streams, read both
ways:

| Route | b9 | b10 | Raw change | Change at fixed weight size | Share from size |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 2.095 | 3.675 | +1.580 | +0.754 | 52.3% |
| matched | 140.648 | 149.411 | +8.763 | +8.763 | 0.0% |

The weights end the unlearn at 0.757 of their former size on average, between
0.490 and 0.980 across streams, so there is a real scale change to find.

On the fixed route, scaling the weights moves the reading by as much as 6.3
rank points. On the matched route the largest shift anywhere in the thirty
scaled readings is 0.0017, which is arithmetic noise. That is what BatchNorm
is supposed to do: it divides by the spread the weights actually produce, so a
constant factor on the weights divides straight back out. The two thirds figure
is a measurement of the fixed protocol, not of the unlearn.

### What this does to the paper

Three things change and one gets stronger.

Sections 3.3 to 3.5 no longer rest on a circular argument. The correlation
result is the same under both readings, so it is evidence about the measures
and not about the protocol. The limitation the paper currently declares can be
removed and replaced with this run.

The claim that the indicators point the wrong way has to be split. Effective
rank points the wrong way only under the fixed protocol. The dormant fraction
points the wrong way under both. srank reports nothing under either.

Section 4.3's rescaling result has to be rewritten as a property of the
protocol.

Table 2's curvature split still has three confounded variables and this run
does not touch it.

## Next decision

The paper edits above are the next piece of work. Nothing else is left in this
block. The matched-statistics pass over all fifteen
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

The run collected its BatchNorm statistics over five epochs. This notebook
rebuilds them from twenty batches. Twenty batches is a rougher estimate, so
every arm sits at a different level from the accuracy the run itself reported.

The check was written to stop the reader here, and that was the wrong rule.
All three arms build their statistics from the same pictures in the same way.
So this kind of error pushes all three arms up or down together. It cannot
create a difference between them, and it cannot hide one. Neither thing we
conclude depends on the level: the starting code is within 0.28 points of the
trained one, and another group's code sits at chance.

What the gap does mean is that these are not the study's own accuracies. They
must not be printed next to a published number. The notebook now says that,
instead of telling the reader to stop.

### Next decision

The notebook now also reads the generator's own distance and turn from the
saved indicator records, so the code's move can be set beside the generator's.
That needs no GPU. Those two spans are not the same, the generator's counting
from the start of the stream and the code's covering one task, so the
comparison is a size comparison and not a ratio.

Whether this holds at boundary 4 as well as boundary 9 is open, and so is
whether it holds on orderings B and C.

### What the code's move is small against

The generator's own distance and turn come from the indicator records the
streams already wrote, averaged over the five seeds of ordering A. They count
from the start of the stream, so each row covers everything up to that
boundary.

| boundary | distance | turn (deg) | distance since the last | turn since the last |
| --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.0 | | |
| 1 | 0.166 | 9.5 | 0.166 | 9.5 |
| 2 | 0.377 | 20.7 | 0.211 | 11.2 |
| 3 | 0.546 | 28.6 | 0.169 | 7.9 |
| 4 | 0.698 | 34.8 | 0.152 | 6.2 |
| 5 | 0.828 | 39.5 | 0.130 | 4.7 |
| 6 | 0.959 | 43.6 | 0.131 | 4.1 |
| 7 | 1.082 | 46.9 | 0.123 | 3.3 |
| 8 | 1.193 | 49.6 | 0.111 | 2.7 |
| 9 | 1.294 | 51.8 | 0.101 | 2.2 |
| 10 | 1.297 | 52.4 | 0.003 | 0.6 |

The last two columns are the difference between one row and the row above it.
They say how much further from the start the generator ended up after that
task. They do not say how long a path it walked to get there. A path is never
shorter than the straight line, so these are the smallest the move can have
been.

Now compare. A code moves 0.019 of its own length and turns 0.9 degrees while
its group is being learned. No group goes above 0.026 or 1.2 degrees.

The generator moves more than that on every task. Its smallest task is the
ninth, at the end of the stream, where it ends 0.101 further out and 2.2
degrees round. That is still about four times the largest code distance and
about twice the largest code turn. On the first task it is 0.166 and 9.5
degrees, about nine and eleven times what a code does.

So it is fair to say the code barely moved. The generator moves much more, on
every task.

One caution. Both numbers are divided by the size of the thing that moved, so
neither depends on how big that thing is, and that is what lets them be
compared. But a code is 32 numbers and the generator is about 25 million. These
numbers say how much each one changed. They do not say how much each one can
hold.

### An aside, not this experiment's question

Boundary 10 is ordering A's unlearn of group 0. The generator ends it 0.003
further from the start and 0.6 degrees round, the smallest move anywhere in the
table. The 14 November record found that the same unlearn moves effective rank
about ten times further than an ordinary learning step. Those two readings sit
oddly together and nobody has looked at them side by side. Noted here so it is
not lost.
