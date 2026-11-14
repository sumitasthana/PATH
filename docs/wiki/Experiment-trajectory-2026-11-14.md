# Experiment trajectory: 14 November 2026

Back to the [index](Experiment-trajectory).

## The task-order effect on effective rank does not survive five seeds

Job 3 finished. Ten runs: order B and order C, five seeds each. With job 2's
five runs of order A, there are now fifteen full streams.

This is the first time the order comparison has been made with more than one
run per order. The short answer is that the earlier result does not hold.

## The runs themselves

| Order | Seeds | Average accuracy | Spread | Standard deviation |
| --- | --- | --- | --- | --- |
| order_A | 5 | 40.18% | 7.03 | 2.89 |
| order_B | 5 | 41.59% | 8.09 | 3.07 |
| order_C | 5 | 39.87% | 3.87 | 1.42 |

The widest gap between order averages is 1.72 points. The seeds of a single
order disagree by about 2.46 points on average. So the gap between orders is
smaller than the noise inside one order. On accuracy, the three orders are the
same.

That is the result we wanted here. If one order had finished far ahead, the
indicator comparison below would be reading two different models rather than
two different orders.

Every run's unlearn step landed on exactly 5.00 percent, which is chance for
20 classes. Fifteen out of fifteen.

## The comparison this job exists for

For every measure, the change from boundary 0 to boundary 9 was taken in each
of the five seeds of each order. That gives an average and a spread per order.
Two orders differ only if the gap between their averages is large next to how
much their own seeds disagree.

The gap is reported in standard errors. A value of 2.3 or more is the usual
bar at five runs per group. Below 1 there is nothing to see.

| Measure | Widest gap between orders | Reading |
| --- | --- | --- |
| srank, EuroSAT | 2.50 | at the bar |
| srank, probe | 2.43 | at the bar |
| srank, in-stream | 2.36 | at the bar |
| Effective rank, all three sets | 1.36 to 1.37 | no difference |
| Distance moved, weight norm, angle | 0.78 to 1.19 | no difference |
| Dormant units, all cut-offs | 0.64 to 0.90 | no difference |

## What this means for the earlier claim

The 6 November record reported that the size of the effective-rank fall is set
by which task comes first, quoting 22.6, 44.6 and 71.8 percent for the three
orders from an identical starting generator. That was one run per order.

With five seeds per order the same measurement reads:

| Order | Fall in effective rank, in-stream | Spread across its own seeds |
| --- | --- | --- |
| order_A | -50.2% | 18.1 |
| order_B | -63.6% | 20.0 |
| order_C | -69.7% | 6.0 |

The five seeds of a single order vary by about as much as the three orders
differed from each other in the earlier result. The gap between order averages
is 1.36 standard errors, which is not a difference.

So the earlier number was the seed, not the order. One run per order could not
have told the two apart, and this is exactly the failure the restart was built
to catch.

## The one result that is near the bar, and why it is one result and not three

srank reaches 2.36 to 2.50 across the three image sets. That is at the usual
bar, so it deserves attention rather than dismissal.

Two cautions, both of which have to go in the paper.

The three image sets are not three independent tests. They are the same model
read on three sets of pictures, so they move together. This is one marginal
finding, not three agreeing ones.

And 21 measures were compared across 3 pairs of orders, which is 63
comparisons. At a bar of 2.3 you would expect one or two to clear it by
chance alone. A single marginal result out of 63 is what chance looks like.

Our reading: srank is suggestive, not established. It needs its own test
rather than a mention inside a table of 63.

## One thing worth following up

The direction is the same in all six rank measurements: order A falls least,
order C falls most, on both measures and all three image sets. Six out of six
pointing the same way is not nothing, even when no single one clears the bar.
But those six are not independent of each other either, so it is weak evidence
for a small real effect rather than proof of one.

Separately, order A's seeds split into two groups on effective rank. Seeds 0
and 2 fall by about 30 percent, while seeds 1, 3 and 4 fall by about 60. That
is not a spread around one value, it is two clusters. The same split shows on
all three image sets. Order C shows nothing like it, with a spread of 6.0
against order A's 18.1.

We do not have an explanation. It is recorded because a measure with two
regimes behaves differently from a noisy one, and averaging the two together
hides it.

## Where this leaves the study

Section 4.1 stands. Effective rank falls over nine tasks of learning, and five
seeds support it.

Section 4.2 does not have the result the earlier version reported. The honest
statement is that the task order has no detectable effect on effective rank,
dormant units, weight norm, angle or distance moved, with five seeds per
order, and that srank sits at the bar and needs a dedicated test.

That is a weaker claim than the draft was built around. It is also a useful
one: a one-seed comparison in this setting produces an effect that is not
there, and the paper can say so with fifteen runs behind it.

Still to do: the rescaling control, which is four forward passes per
before-and-after pair and trains nothing, now across fifteen pairs instead of
one.
