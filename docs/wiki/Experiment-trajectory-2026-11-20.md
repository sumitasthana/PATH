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
