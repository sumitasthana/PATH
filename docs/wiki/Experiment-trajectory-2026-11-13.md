# Experiment trajectory: 13 November 2026

Back to the [index](Experiment-trajectory).

## The question of the day

The learning rate is the size of the step the model takes each time it
corrects itself. Too small and it barely learns. Too large and it overshoots
and damages what it already knew.

The earlier version of this work never chose one. It copied the value from the
original hypernetwork paper, 1e-3, and used it everywhere without checking. The
new methodology says to settle this first, before measuring anything else.

So today's question was simple. Out of five candidate step sizes, which one
should every later run use?

## What was run

Five short training runs, one per candidate.

Each run learned the same three tasks, in the same order, starting from the
same place. Only the step size differed between them. Each run was then scored
on its average accuracy across those three tasks.

Three tasks rather than the full nine, because a sweep only has to compare the
candidates against each other. Nine would have cost three times as much and
answered the same question.

Everything else was held still: one random starting point, five passes over
each task's images, batches of 64 images, the ResNet50 written by the
generator, and the same protection setting of 0.01 that holds earlier tasks in
place.

A task holds 20 classes with 500 training images each, so one task is 10,000
images. In batches of 64, five times over, that is 785 correction steps per
task. The generator holds 134,076,432 numbers and writes 23,541,332 of them
into the network it builds.

All five runs finished in one Colab session on an L4 GPU, about 14 minutes
each. Memory use peaked at 5.08 GiB of the card's 22, so the batch size was
never near the limit.

## The result

| Step size | Average of the 3 tasks | Task 0 | Task 1 | Task 2 |
| --- | --- | --- | --- | --- |
| 3e-4 | 48.80% | 43.5 | 51.5 | 51.4 |
| **5e-4** | **49.33%** | 39.8 | 52.3 | 55.9 |
| 1e-3 | 34.60% | 11.1 | 43.2 | 49.5 |
| 2e-3 | 41.23% | 23.6 | 48.9 | 51.2 |
| 4e-3 | 39.07% | 31.1 | 40.8 | 45.3 |

A task has 20 classes, so a model guessing at random would score 5 percent.
Every value here is far above that. The model does learn at all five step
sizes, and none of them is broken.

The losing values stay in this table on purpose. The methodology says the
tuning runs are reported in full. A chosen number with the rejected ones thrown
away cannot be checked by anyone.

## How the winner was picked

5e-4 scored highest, so 5e-4 is the choice.

That rule, take the highest average, was written down before the runs started.
It was not picked afterwards to suit the numbers.

That matters here, because 5e-4 beats 3e-4 by only 0.53 points. For comparison,
the next stage of the study treats a gap of up to 3 points between runs as
ordinary variation rather than a real difference. The gap between first and
second place is about a sixth of that.

So this sweep cannot really tell 5e-4 and 3e-4 apart. Had the scoring been
"which value keeps the first task best", 3e-4 would have won instead, 43.5
against 39.8.

That is exactly why the rule was fixed in advance. Once the results are in
front of you there is nearly always some way of reading them that favours
whichever answer you already prefer. Choosing the measure after seeing the
numbers lets you land anywhere and still feel justified.

The choice is not final either. Five more runs have to agree with each other
before 5e-4 is accepted. If they do not, 3e-4 is next, and no new sweep is
needed to get there.

## A sign that single runs are noisy

Read down the average column. It goes 48.80, 49.33, drops to 34.60, then climbs
back to 41.23 and 39.07.

If the step size were the only thing driving the score, that column should rise
to a peak and then fall away steadily. It should not dip in the middle and
climb back out.

1e-3 sitting below both of its neighbours suggests each of these numbers
carries a few points of randomness of its own. Every value was run once, from a
single random starting point, so there is nothing to average that randomness
away.

This does not change the decision. It does mean the gaps in the table should
not be read too precisely, and it is a second reason not to treat the top two
as genuinely separated.

## What the step size actually controls here

This is the part worth remembering.

Compare how far each task's score moves across the five step sizes:

| | Lowest | Highest | Range |
| --- | --- | --- | --- |
| Task 0 | 11.1 | 43.5 | 32.4 points |
| Task 1 | 40.8 | 52.3 | 11.5 points |
| Task 2 | 45.3 | 55.9 | 10.6 points |

Task 2 is the last one learned and is scored straight afterwards. It barely
moves, whatever the step size.

Task 0 is the first one learned, and by the time it is scored the model has
been through two more tasks. It moves three times as much.

So the step size is not mainly deciding how well the model learns something
new. It is deciding how much of the earlier work survives as new tasks arrive.
A large step learns the task in front of it and damages what came before.

## The old default was the worst value tested

1e-3 came last of the five. At that setting the first task ended at 11.1
percent, close to the 5 percent a random guess would score. Almost nothing of
it was left.

1e-3 is the value the earlier version of this work ran on, taken from the paper
and never tested.

That does not by itself invalidate the earlier results. Those runs used eight
tasks of ten classes, a different split, so their numbers do not carry across.
But it does support the reason the methodology gave for tuning first, which was
that untuned runs make every later comparison harder to read.

## The next stage will cost about twice the first estimate

The sweep's first run printed an estimate of 5.1 hours for the seven nine-task
runs still to come. That estimate was wrong, and the notebook has been fixed.

Here is why. A correction step took 0.244 seconds during the first task, 0.361
during the second, and 0.470 during the third.

Steps get slower as the stream goes on. The model carries a term that holds
earlier tasks in place, and at each new task there is one more old task to hold
on to. More to protect means more work per step.

The old estimate took the average time per task and multiplied it by the tasks
remaining. That quietly assumed the ninth task would cost what the first one
did. It will not.

Measuring the rise instead, and carrying it out to nine tasks, gives about 1.4
hours for one nine-task run and about **9.6 hours for the seven**. The rise was
measured from three points only, so treat 9.6 hours as a floor rather than a
firm figure.

## Decision: the damage step now runs on all seven runs

A separate decision was settled today, about the damage arm.

The study breaks a model on purpose at the end of a run. It picks one task and
unlearns it deliberately, so the indicators can be tested against damage that
is certainly there. An indicator that reads healthy on a model just broken
cannot tell learning from damage.

The methodology first asked for this on one run. The run notebooks were written
to do it on every run instead, each time on whichever task that run learned
first: task 0 in order A, task 4 in order B, task 8 in order C.

The seven version was chosen, and `docs/plasticity/methodology.md` has been
updated to say so rather than left contradicting the code. `run-plan.md` still
described it as a run of its own, "run 13", in two places. There is no run 13
now, because the unlearn is a step inside each of the seven runs. Both places
are corrected.

It is nearly free. Breaking a model takes 100 steps against the 7,065 a
nine-task run spends learning, so the whole arm adds under 2 percent.

What it buys is something to compare against. Five of the seven runs differ
only in their random starting point. So the damage test now has its own measure
of ordinary run-to-run variation, and anything smaller than that is not
reported as a finding. The rescaling check in section 3.8 gains the same thing.
It used to rest on a single before-and-after pair, which left it nothing to
judge its own result against.

One numbering detail was fixed in writing at the same time. Section 3.5 called
the reading after the damage step "the tenth boundary". That wording is left
over from when a run was eight tasks long. With nine tasks the stream ends at
boundary 9 and the damage step is boundary 10, counting the untrained model as
boundary 0. The notebooks already number it that way; now the specification
does too. A boundary off by one already cost this study a run on 7 November.

## The extra reading confuses the report, and has to be worked around

This follows directly from the decision above, and it would have been easy to
miss.

`indicator_report.endpoints` takes the smallest and the largest boundary number
in a record and reports the change between them. Until now the largest number
was always the end of learning, because records stopped at the last task.

These records do not stop there. Boundary 10 is the model after the damage
step. So the largest number is now a model broken on purpose, and the headline
change for every measure would run from the untrained generator all the way to
the damaged model.

Section 4.1 asks what nine tasks of learning do to effective rank. Read that
way, the answer would quietly have the damage mixed into it.

Nothing is wrong in either run notebook and nothing in the library needs
changing. The report simply has to be run twice: over boundaries 0 to 9 for the
learning claim, and over all eleven for the damage claim. It is written down
here because the wrong version would look perfectly reasonable in the output.

## Every class is now used, for the first time

Worth stating plainly, because it is easy to assume this was always true.

Tiny ImageNet has 200 classes. They are split into 10 groups of 20. Nine of
those groups, 180 classes, are learned one after another. The tenth group, the
remaining 20, is never trained on. It is held back as the probe: images the
model has never seen, from the same dataset, used to ask what the generator
does with unfamiliar data.

So every class has a job. 180 are learned, 20 are the held-out probe, none is
left over.

That was not true before. The old split cut the same 200 classes into 20 groups
of 10, but a run only learned 8 of those groups, which is 80 classes, plus one
group of 10 as the probe. So 90 classes were used and 110 were never touched.
More than half the dataset sat idle.

This is one of the reasons for the restart. A claim about what continual
learning does to a model is easier to defend when the stream covers the whole
dataset rather than a bit under half of it.

EuroSAT is separate from all of this. It is satellite imagery, a different
dataset, used as a second probe to ask what happens on images unlike anything
in Tiny ImageNet. It is never trained on either.

## Job 2 result: five seeds done, and the gate as written fails

| Seed | Mean over 9 tasks | Task 0 after the unlearn | Hours |
| --- | --- | --- | --- |
| 0 | 41.41% | 5.00% | 0.7 |
| 1 | 37.60% | 5.00% | 0.7 |
| 2 | 39.12% | 5.00% | 0.7 |
| 3 | 44.63% | 5.00% | 0.7 |
| 4 | 38.11% | 5.00% | 0.7 |

The damage arm is clean. Task 0 landed on exactly 5.00 percent in all five
runs, which is the guessing floor for 20 classes. The break worked, every time,
in the place it was aimed at.

The gate did not pass. Best minus worst is 7.03 points against a target of 3.

Two things need separating before anyone retunes.

**The runs are not behaving oddly.** The standard deviation across the five
seeds is 2.89 points. For data spread like that, the gap between the highest
and lowest of five draws should be about 6.7 points. The observed 7.03 is what
that spread predicts. Nothing anomalous happened.

**The target is the part that is broken.** It is written as a range, best minus
worst, and a range gets wider the more runs you do, because more draws reach
further into the tails. The same "3 points" therefore demands a standard
deviation under 1.77 with three seeds, under 1.29 with five, and under 0.97
with ten. A study could pass it by running fewer seeds, which is the opposite
of what a stability check should reward.

So the failure is partly an artifact of how the target was worded, not proof
that 5e-4 is unusable.

That is not a reason to wave it through. The floor exists because every claim
in the results compares numbers from different runs, and section 4.2 compares
single runs of different orderings. If seed noise alone moves mean accuracy by
a standard deviation of 2.9 points, an order difference smaller than about 6
points says nothing. That is exactly the failure the restart was meant to fix.

**Decision: measure the right quantity before changing anything.** The gate
scores mean accuracy, but almost every claim in the paper compares an
indicator, not an accuracy. Effective rank, dormant units and weight norm have
their own spread, and five seeds of indicator records are now in R2. Reading
that spread costs no GPU time and answers the real question, which is whether
these runs are stable enough for the comparisons the paper actually makes.

Job 3 is on hold until that reading exists. No retuning has been started.

## Where this leaves the study

Done: the learning rate is chosen, at 5e-4.

Next: five runs of the full nine-task stream, the same ordering, differing only
in their random starting point. They have to finish within 3 points of each
other on average accuracy. If they do, the setting is good enough to measure
against and the two remaining orderings can run. If they do not, the next thing
to try is 3e-4, before changing the number of passes or the protection setting.

## Gaps in this record

The per-run `environment` files record the code version as `None`. The runs
were started from a Colab copy of the repository, where the notebook's own
version lookup did not reach that file. The version is recorded correctly in
`sweep_summary.json`, which the notebook writes itself, so the runs can still
be identified. But the two files disagree, and `environment` is the wrong one.
