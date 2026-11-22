# 22 November 2026

Notebook 56 ran. Random-label fitting is now measured on the revised study, and
it agrees with the direct probe: the model memorises shuffled labels far better
at the end of the stream than at the start.

## What ran

Ordering A of `plasticity_revised_v1`, five seeds, boundaries 0 and 9. 150 runs:
two points, three learning rates, five repetitions. The measurement is the final
training loss on 2,000 pictures of group 9 whose labels were shuffled once from
seed 1020, 600 steps, batch 64. Lower means the model memorised more, so lower
means more capacity to change.

Everything about the measurement is notebook 41's: the same label seed, the same
image count, the same step budget, the same three rates 1e-4, 3e-4 and 1e-3, and
the same rule that keeps the run with the lowest final training loss. What
changed is the study underneath it, and with it the task: group 9 is held out by
design, and it has twenty classes rather than ten, so chance is a loss of
ln(20) = 2.996.

The gate passed. It stops the session if the first stream's fit sits within 0.20
of chance, which would mean the run had measured the step budget rather than the
model. Twenty wrong labels are harder to memorise than ten on the same budget,
and that was the risk in this run. It did not happen.

## The result

| Stream | t = 0 | t = 9 | Change | Floor | Margin | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| seed 0 | 1.5870 | 0.0093 | -1.5778 | 0.3495 | 4.5x | no loss, more capacity |
| seed 1 | 1.6468 | 0.0003 | -1.6465 | 0.4694 | 3.5x | no loss, more capacity |
| seed 2 | 1.6398 | 0.0005 | -1.6393 | 0.2077 | 7.9x | no loss, more capacity |
| seed 3 | 1.5659 | 0.0041 | -1.5618 | 0.2108 | 7.4x | no loss, more capacity |
| seed 4 | 1.7199 | 0.0002 | -1.7197 | 0.3205 | 5.4x | no loss, more capacity |

Mean change -1.6290, spread across the five seeds 0.0628. Fell past its own
floor in 5 of 5, rose past it in 0.

## Against the retired design

Notebook 41 measured the same thing on `indicator_stream_v1`, eight tasks of ten
classes, three seeds. There the loss fell from about 1.40 to about 0.73, a change
of -0.6721. Here it falls from about 1.63 to about 0.003, a change of -1.6290.

Direction is the same on every stream of both studies. The size is about two and
a half times larger, and the spread across seeds is much smaller: 0.0628 against
0.1037 with five seeds against three.

## The end point is saturated, and that limits what the number can say

At boundary 9 four of the five seeds read a loss between 0.0002 and 0.0041. That
is not a low loss, it is memorisation of all 2,000 pictures with almost nothing
left. The measurement has hit the bottom of its own scale.

So this run establishes a direction and nothing finer. It cannot rank one late
boundary against another, because every late boundary would read zero. If a
reading at boundary 4 or 6 is ever wanted, the budget has to be cut until the
end of the stream stops saturating, and that is a different experiment from this
one.

The starting point carries a confound that has been on every direct measurement
in this study. At boundary 0 the generator is untrained, so it is a poor place to
start an optimisation from, and part of the gap is the optimiser having an easier
job at boundary 9 rather than the model having more capacity. The direct probe
has the same confound and reports the same direction. Nothing here separates the
two, and the conclusion the paper draws is the weaker one: whatever the rank
indicators are reporting, it is not an inability to fit new data.

## What this closes

Random-label fitting was the last of the six direct measurements still sitting on
the retired eight-task design, whose model files the current class split cannot
load. All six are now on `plasticity_revised_v1`. The "different training run"
caveat comes off that row in the paper.

The split stays where it was, now on one design throughout. The measurements that
train something and count the result say capacity held. The measurements that
read the weights or the activations and infer say it was lost.

## What is open

Orderings B and C were not run, to keep the session at three and a half hours
rather than ten. The per-run resume file means adding them repeats nothing
already done. On the direct probe the three orderings agreed in direction and
differed in size, so the expectation is that B and C agree here too, and that is
an expectation rather than a measurement.

The result has to go into the paper: the random-label row, its method paragraph
and the different-training-run caveat.

## Next decision

Block F, the induced plasticity-loss stress test, is what remains of the staged
plan, at fourteen hours or more of GPU time. Before that, the paper needs this
result written in, and the LaTeX draft regenerated from the Markdown.
