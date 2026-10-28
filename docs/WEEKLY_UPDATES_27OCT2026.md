# Weekly update: 27 October 2026

[Back to the weekly update index](WEEKLY_UPDATES.md)

This update states where the UnCLe work stands. There are ten questions below. Each one
is followed by the answer and the measurements behind it.

An earlier version of this page claimed that UnCLe's forgetting step acts on the task
code and therefore aims at something that holds no skill. That was wrong. The forgetting
step trains the shared generator, not the task code. Question 2 explains the correction
and what it costs us. The earlier version is in the repository history.

## Words used on this page

Every term is defined once, in one line.

| Term | What it means |
| --- | --- |
| Hypernetwork, or generator | A network that produces the weights of a second network, instead of being used directly. |
| Task code | A small learned vector, 32 numbers, that identifies one task. It is the input to the generator. Think of it as an address. |
| Trunk | The shared part of the generator that every task passes through. |
| Forgetting, or unlearning | Extra training steps run so that one task's code no longer produces a working classifier. |
| Chance | The score a model gets by guessing. With 10 classes it is 10 percent. |
| Tie threshold | 3 points. A difference smaller than this is read as no difference. |

## 1. What are we claiming?

That UnCLe silences one route into a shared generator, and that we still cannot tell
whether the skill went with it.

The method is honest about what it does. The open question is what that leaves behind.
We tried to settle it by repeating the paper's own appendix measurement after forgetting,
and that attempt failed for a reason worth knowing. Question 9 gives it.

We are not saying the method fails. We are saying the evidence offered for removal does
not separate removal from silence, and our own attempt to separate them did not work
either.

## 2. What does the forgetting step actually change?

**What we assumed earlier.** That it changes the task code.

**What the code shows.** It changes the shared generator. The task code is frozen
throughout.

This matters enough to state carefully, because an earlier version of this page was built
on the wrong reading.

The paper's appendix has a baseline that replaces a task's code with a random one and
trains nothing. Accuracy on the target task stays above chance on all four of their
datasets. On Tiny ImageNet, which is the dataset family our own work uses, it is 15.54
percent against 10 percent chance. Their largest figure, 80.91 percent, is from a
different dataset and should not be used as the comparison for our results.

We have run the same measurement on our own model and it gives chance. See question 9.
So the baseline tells us about their setting, and the gap to ours is smaller than we
first wrote.

But the forgetting method does not replace the code. It keeps the code fixed and retrains
the generator so that this code now produces near-zero weights. Our implementation makes
this explicit: the optimiser is given the trunk and the heads, and everything else is
frozen.

So the appendix baseline and the method are consistent. The authors never claimed the
skill sits in the code. They claim it sits in the mapping from code to weights, and the
method edits that mapping.

This removes our strongest-sounding criticism. What follows is what survives it.

## 3. What does the forgetting step do to the generated weights?

**The paper's position.** It turns them into noise.

**What the derivation shows.** It drives them to zero.

This needs no experiment. The forgetting loss measures the squared distance between the
generated weights and fresh samples of random noise. Those samples average to zero. So
the average squared distance works out to the squared length of the weights, plus a
constant that does not depend on the weights.

That quantity is smallest when the weights are exactly zero. Minimising the loss drives
the weights to zero. It does not make them random.

Zero and noise are not the same thing. A zeroed classifier is silent. A randomised one is
scrambled. The difference matters for every question below, because a silent model and an
empty model look alike from the outside.

## 4. What does the forgotten model actually output?

**The paper's position.** A uniform spread across all classes, which it scores as maximum
confusion.

**What we measured.** One single class, for every image.

We recorded the model's predictions directly, immediately after forgetting. On a task that
stood at 42.60 percent beforehand, the model after 6 forgetting steps used 1 of 10 classes
and put 100 percent of its predictions in that one class. It scored exactly 10.00 percent.
After 100 steps, the same.

Exactly 10.00 percent on a balanced ten-class test set is what a constant answer gives.

This has been seen on two tasks of different strength, at two stopping points, in three
separate runs.

## 5. How does the paper decide that forgetting worked?

**The paper's position.** Three tests: forget accuracy at chance, uniformity at 100, and a
membership inference attack at 50.

**The problem.** A model that answers one fixed class for every image passes all three.

The previous section shows that this is not a hypothetical model. It is the model the
method produces. Forget accuracy at chance we measured directly. The other two follow from
near-zero outputs, and we have not measured them ourselves.

These tests show that the model has gone quiet. They do not show that the information was
removed. This point is not ours alone, and other recent work in unlearning evaluation makes
a similar argument.

## 6. How many forgetting steps are actually needed?

**The paper's position.** 100 steps, reduced over repeated operations down to 20.

**What we measured.** The collapse is complete after about 6.

Two measurements agree. First, accuracy. The task reaches chance by step 6 on one task and
by step 20 on another, and never moves again after that.

Second, the direction of the weights. At step 20 the weights still point almost exactly
where they started, having turned between 7.3 and 10.5 degrees, measured over three seeds.
By step 100 they have turned between 49.1 and 54.1 degrees.

So the steps taken after the task is gone remove nothing further. They move the shared
generator a long way for no measured gain, and they make the change harder to undo.

## 7. What happens to the forgotten task's own code?

**What we measured.** It stops working, as designed.

After forgetting, we tried to teach the same task again. Some arms kept the forgotten
task's own code. Others used a brand new one.

| Arm | Task code used | Steps before it responds at all |
| --- | --- | ---: |
| Attacker | The forgotten task's own | 38 |
| Code only | The forgotten task's own | 38 |
| Statistics only | Brand new | 2 |
| Fresh start | Brand new | 2 |

The split follows the code and nothing else. It held across three runs on two different
tasks, at every image budget and both stopping points.

We first read this as damage. It is better read as confirmation. The method trains the
generator so that this one code produces zeros, so this one code going quiet is the method
doing exactly what it says. The measurement is still useful, because it shows the effect is
sharply targeted: a new code is unaffected and starts responding immediately.

## 8. Is the skill gone?

**The paper's position.** Yes, removed.

**What we measured.** Relearning the forgotten task is no faster than learning it from
nothing.

We took the forgotten model and gave it a small number of images from the forgotten task,
between 1 and 20 per class. We then compared how fast it recovered against a model that had
never seen that task.

| Task probed | Gap at the 6-step model | Gap at the 100-step model |
| --- | ---: | ---: |
| Task 3, run 1 | -7.3 | -2.3 |
| Task 3, run 2 | -6.5 | -3.7 |
| Task 14 | +1.9 | +1.3 |

A positive number means the forgotten model did better than a fresh start. The task 14
numbers are inside the 3-point tie threshold, so they count as a tie, not a win.

Task 14 matters because it answers the obvious objection. Task 3 was weak before forgetting,
at 35.60 percent, so the probe had no room to show a gain. Task 14 stood at 42.60 percent
with about 12 points of room. The answer did not change.

There is a limit to what this probe can show. Every arm trains the generator. Training can
rebuild a skill, so a gain would have been hard to read, and the absence of a gain does not
prove the skill is absent. The next question describes a probe without that weakness.

## 9. Did a direct test settle it?

We ran one. It came back void, and the reason is itself a result.

The idea was to repeat the appendix baseline after forgetting instead of before. Give the
model an unused random task code, train nothing, and see whether the forgotten task is
still readable through that other code. If it were, forgetting would have blocked one
route and left the skill in place.

Five random codes, two stopping points, with each code given the target task's stored
BatchNorm statistics. A task the model kept was probed the same way as a control.

| Stopping point and stage | Task 14, own code | Task 14 via random codes | Task 7 via random codes |
| --- | ---: | ---: | ---: |
| 6, before forgetting | 42.6 | 9.5 | 10.0 |
| 6, after forgetting | 10.0 | 10.0 | 10.0 |
| 100, before forgetting | 42.6 | 9.5 | 10.0 |
| 100, after forgetting | 10.0 | 10.0 | 10.0 |

Read the before rows first. On an untouched model, random codes already read nothing. 9.5
and 10.0 percent against 10 percent chance, and almost every reading used a single class.
The instrument was dead before forgetting ever ran, so the after rows carry no information
about removal.

This is not a surprise in hindsight. Our 21 October runs measured a frozen generator with
a fresh code and found chance on three seeds. This is the same fact reached another way.

What it does tell us is that our model and the paper's behave differently on this point.
In our model an untrained code carries a task nowhere. On the comparable dataset theirs
carries one to 15.54 percent, above chance but not by much.

The authors have since been found to have released code, and it does not settle this. It
contains no discard-embedding baseline, so that table was produced by code that is not
public. It also evaluates differently from our probe: their test path leaves the target
network in training mode, so its normalisation layers use each batch's own statistics,
while our probe uses stored statistics. That difference alone could move these numbers and
has not been tested. The detailed audit is in the 27 October experiment record.

## 10. What is still missing?

Four things, in order of importance.

We still cannot say whether the skill was removed or only blocked. The probe in question 9
was meant to decide it and could not, so the question is open with no instrument that can
reach it in this model.

Everything above is one seed. Seeds 1 and 2 are needed before any of it is reported as a
result.

One observation is not yet a claim. The forgotten task's old code costs about 40 steps of
dead time, and then saves about 20 to 40 steps of learning once it responds again. On one
task the two roughly cancel. The pattern was mixed on the other task, so it needs the extra
seeds.

We have not checked the collapse using the paper's exact published settings. That needs the
journal version of the paper, which we do not have access to.
