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

## Next decision

Re-run the notebook. It will produce five seeds of readings, five seeds of
accuracies, and a reproducibility figure for each seed.

Then Table B6 gets filled, the control column of Table B2 gets its five-seed
means, the five places that say three seeds get corrected, and a sentence on
reproducibility goes into the limitations.

Whether to run one hypernetwork stream twice, to put the same number on the
study's own side, is open. It costs about two hours and nothing else depends
on it.
