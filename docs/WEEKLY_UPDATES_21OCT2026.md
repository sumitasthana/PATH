# Weekly update: 21 October 2026

[Back to the weekly update index](WEEKLY_UPDATES.md)

## Words used on this page

Every term here is defined once, in one line.

| Term | What it means |
| --- | --- |
| Hypernetwork, or generator | The part that produces the weights of a classifier, instead of being the classifier. |
| Task code | 32 numbers belonging to one task. They tell the generator which classifier to produce. |
| Backbone | The large model that does the main work of reading an image. |
| Frozen | None of that part's weights are allowed to change during training. |
| Adapter | A small set of trainable weights added beside a frozen backbone. The backbone stays fixed and the adapter changes what it computes. |
| Hypernetwork-generated adapters | The generator produces the adapter weights from a task code, instead of the adapter being trained directly. |
| Matas's setup | Plain continual learning: train the model on one task after another, with no hypernetwork. |
| Plasticity | The ability to learn a **new** task. Not the same as forgetting, which is old tasks getting worse. |
| Fork-and-discard probe | Copy the model, teach the copy a task it has never seen, measure, then throw the copy away. |
| Input sensitivity | The widest spread any single output score shows across the test set. A dead model reads near zero. |
| Chance | The score a model gets by guessing. With 10 classes it is 10 percent. |


## Plasticity: measured on three seeds, no loss found

We asked whether the generator gets worse at learning new tasks after it has learned
eight. It does not. It is slightly better.

We measured this two ways, and both agree.

| Measurement | What it is | Mean change | Spread between seeds |
| --- | --- | ---: | ---: |
| Learning a new task | Test accuracy on three tasks the model never saw | +4.74 points | 1.68 |
| Raw ability to change | Training loss on images with random wrong labels | -0.70 | 0.06 |

Accuracy going up and loss going down both mean better learning.

Each seed, against its own measured noise floor. The noise floor is how much the
measurement wobbles when we repeat it on an unchanged model. A result only counts when
it is larger than that.

| Measurement | Seed | Before | After 8 tasks | Change | Floor | Counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Learning a new task | 0 | 51.87 | 58.23 | +6.37 | 5.53 | Yes |
| Learning a new task | 1 | 53.09 | 56.10 | +3.01 | 3.23 | No |
| Learning a new task | 2 | 52.51 | 57.37 | +4.86 | 2.28 | Yes |
| Raw ability to change | 0 | 1.37 | 0.67 | -0.70 | 0.29 | Yes |
| Raw ability to change | 1 | 1.54 | 0.78 | -0.76 | 0.42 | Yes |
| Raw ability to change | 2 | 1.31 | 0.67 | -0.64 | 0.18 | Yes |

Five of six readings count. Seed 1's accuracy reading is 3.01 against a floor of 3.23,
so it is just inside and we record it as no finding. It points the same way as the other
two. The second measurement is the stronger one: all three seeds count, and the three
values differ from each other by only 0.06.

**Two further results from the same runs.**

Training a task's 32 numbers buys nothing. We left them at their random starting value
and trained only the generator. The differences were -1.48, +1.01 and +1.27 points
against floors of 5.53, 3.23 and 2.28. No consistent direction, and far inside the floor.

The task code alone holds nothing. Freeze the generator, train only the 32 numbers, and
the result is 10.0 to 10.9 percent against a chance level of 10. We knew this from one
seed. It now holds on three.

**What it means.**

- The paper says unlearning restores plasticity. There is no plasticity loss for it to
  restore, so that claim is not supported here. Measuring unlearning directly agreed:
  +5.40 and -0.20 points across three seeds, both inside the noise floor.
- The control arm we planned, with the protection term switched off, is no longer
  needed. It only handicaps the eight-task model, and that model already scored higher
  while carrying the handicap.
- This agrees with the adapter study, which finds no plasticity loss on a different
  architecture over twenty tasks.
- We measured feature effective rank and found no change. Matt has since shown that
  metric is confounded by the learning rate, so we do not rely on it either way.

**Limits.** Three seeds, and the spread between them is about a third of the effect. One
task order, so "after eight tasks" and "after these particular eight tasks" cannot be
separated. Two measurement points only, with nothing in between. The new tasks come from
the same dataset as the stream. We do not know why it got better.

## Which neurons still fire after unlearning

Covered in the 18 October entry under Q2, because it answers that question. One run,
seed 0, task 3 removed.

## Plan for next: can a removed task be recovered?

This is the largest open question in the study and we have not run it. The neuron
measurements changed what we think the right experiment is.

**What we now believe happens.** Each generated weight is a raw number times a scale,
plus an offset. Unlearning pushes the raw number toward zero, so each weight moves
toward its offset rather than toward zero. The offsets differ by layer type.

| Parameter | Offset | Where unlearning sends it |
| --- | ---: | --- |
| Convolution and classifier weights | 0.0 | toward zero |
| BatchNorm scale | 1.0 | toward one |
| BatchNorm shift | 0.0 | toward zero |

If a convolution's weights approach zero, its output approaches zero for every picture.
The BatchNorm layer then subtracts a running average and divides by a running spread,
and neither of those was touched by unlearning. The result is close to a fixed number
per channel. Positive ones pass through, negative ones become zero. That would explain
both measurements: 59 percent silent, and 41 percent responding weakly.

**Three steps, cheapest first.**

1. **Check whether the surviving neurons are saying anything.** Record how much each
   channel varies from picture to picture, not just its average. If a channel gives
   nearly the same number for every picture, it carries no information about the input.
   This is one extra line in the measurement we already run.

2. **Look at the BatchNorm running statistics.** Our forgetting step changes generated
   parameters only. The running average and spread for the forgotten task were computed
   from that task's own pictures and are still stored, untouched. After unlearning they
   may be the only part of the model that still holds anything from that task. That is
   both a route to recovery and a gap in any privacy claim.

3. **Undo the weakening and see what returns.** We measured how much each part of the
   network was turned down: 0.29, 0.14, 0.13, 0.16 and 0.19. Multiply the generated
   weights back up by those amounts and measure accuracy, with no training at all. If
   accuracy rises above chance, the structure survived and only the scale was removed.
   If it stays at 10 percent, the weights are genuinely gone.

Step 1 decides whether steps 2 and 3 are worth doing, so it goes first. None of this is
a recovery experiment yet. A proper one needs a fixed training budget and a control that
separates transfer from other tasks from information specific to the removed task.
