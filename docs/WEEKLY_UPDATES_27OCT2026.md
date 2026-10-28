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

That UnCLe silences one route into a shared generator, and that nobody has yet checked
whether the skill is still reachable by a different route.

The method is honest about what it does. The open question is what that leaves behind.
The paper's own appendix contains the measurement that would settle it, but the appendix
runs it before forgetting rather than after. Question 9 sets this out.

We are not saying the method fails. We are saying one test is missing, and without it the
claim of removal is not supported.

## 2. What does the forgetting step actually change?

**What we assumed earlier.** That it changes the task code.

**What the code shows.** It changes the shared generator. The task code is frozen
throughout.

This matters enough to state carefully, because an earlier version of this page was built
on the wrong reading.

The paper's appendix has a baseline that replaces a task's code with a random one and
trains nothing. Accuracy on that task stays at 80.91 percent, against 10 percent chance.
That result is real, and it does show that a task code by itself holds no skill.

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

## 9. What test would settle it?

Run the paper's own appendix baseline again, but after forgetting instead of before.

The appendix takes a trained model, gives it an unused random task code, and measures
accuracy on the target task with no training at all. Before forgetting, that reaches 80.91
percent. So an arbitrary code can reach the task through the generator.

Nobody has run this after forgetting. It is the question the method raises and does not
answer. Forgetting changes the shared generator, so there are two possible outcomes, and
they mean opposite things.

| Result of the probe | What it means |
| --- | --- |
| An unused code still scores well above chance | Forgetting blocked one route. The skill is still in the generator and still reachable |
| An unused code scores at chance | The skill really was removed from the generator, and the method does what it claims |

This is cheap. There is no training. It is one forward pass per test image, using code we
already have. It also extends the authors' own experiment rather than arguing against it,
which makes it easy to defend.

## 10. What is still missing?

Four things, in order of importance.

The probe in question 9 has not been run. It is the one measurement that separates a
blocked route from a removed skill, and until it exists we cannot say which happened.

Everything above is one seed. Seeds 1 and 2 are needed before any of it is reported as a
result.

One observation is not yet a claim. The forgotten task's old code costs about 40 steps of
dead time, and then saves about 20 to 40 steps of learning once it responds again. On one
task the two roughly cancel. The pattern was mixed on the other task, so it needs the extra
seeds.

We have not checked the collapse using the paper's exact published settings. That needs the
journal version of the paper, which we do not have access to.
