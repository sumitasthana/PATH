# Weekly update: 27 October 2026

[Back to the weekly update index](WEEKLY_UPDATES.md)

This update states the claim we want to make about UnCLe, and gives the evidence for each
part of it. Each section is one question, followed by the answer and the measurements
behind it.

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

## What are we claiming?

UnCLe removes a task by changing that task's address. But the task's skill is not stored
at that address. So the method does not work the way the paper says it works.

We are not saying the method fails to forget. It does appear to forget. We are saying the
reason the paper gives for why it works is not the reason it works.

## Where is a task's skill stored?

**The paper's position.** In the task code. That is the only thing the forgetting step
acts on, so the paper's account only makes sense if the skill is there.

**What the measurement shows.** The skill is in the shared trunk, not in the code.

The evidence here is the paper's own, in its appendix. The authors take a fully trained
model, replace the forgotten task's code with a random one, and train nothing at all. If
the skill lived in the code, accuracy should fall to chance. It does not.

| Dataset | Accuracy after the code is replaced | Chance |
| --- | ---: | ---: |
| 1 | 80.91 | 10 |
| 2 | 20.70 | 10 |
| 3 | 61.79 | 10 |
| 4 | 15.54 | 10 |

A code the model has never trained with still reaches the task through the trunk. The
authors report this as a baseline that failed. We read it as a measurement of where the
task actually lives.

These are the paper's numbers. We have not reproduced them.

## What does the forgetting step do to the weights?

**The paper's position.** It turns the generated weights into noise.

**What the derivation shows.** It pushes them to zero.

This one needs no experiment. The forgetting loss measures the squared distance between
the generated weights and fresh samples of random noise. Those samples average to zero.
So the average squared distance works out to the squared length of the weights, plus a
constant that does not depend on the weights.

That quantity is smallest when the weights are exactly zero. Minimising the loss
therefore drives the weights to zero. It does not make them random.

Zero and noise are not the same thing. A zeroed classifier is silent. A randomised one
is scrambled. Our implementation follows the paper's equation, and what we observe
matches the derivation.

## What does the forgotten model actually output?

**The paper's position.** A uniform spread across all classes, which it scores as
maximum confusion.

**What we measured.** One single class, for every image.

We recorded the model's predictions directly, immediately after forgetting. On a task
that stood at 42.60 percent beforehand, the model after 6 forgetting steps used 1 of 10
classes and put 100 percent of its predictions in that one class. It scored exactly
10.00 percent. After 100 steps, the same.

Exactly 10.00 percent on a balanced ten-class test set is what a constant answer gives.

This has now been seen on two tasks of different strength, at two stopping points, in
three separate runs.

## How does the paper decide that forgetting worked?

**The paper's position.** Three tests: forget accuracy at chance, uniformity at 100, and
a membership inference attack at 50.

**The problem.** A model that answers one fixed class for every image passes all three.

The previous section shows that this is not a hypothetical model. It is the model the
method produces. Forget accuracy at chance is measured directly. The other two follow
from near-zero outputs, and we have not measured them ourselves.

These tests measure whether the model has gone silent. They do not measure whether the
information was removed.

## How many forgetting steps are actually needed?

**The paper's position.** 100 steps, reduced over repeated operations down to 20.

**What we measured.** The collapse is complete after about 6.

Two measurements agree. First, accuracy. The task reaches chance by step 6 on one task
and by step 20 on another, and never moves again after that.

Second, the direction of the weights. At step 20 the weights still point almost exactly
where they started, having turned between 7.3 and 10.5 degrees, measured over three
seeds. By step 100 they have turned between 49.1 and 54.1 degrees.

So the steps taken after the task is gone remove nothing further. They make the change
harder to undo.

## Can the forgotten task code be used again?

**The paper's position.** Not discussed.

**What we measured.** No. The old code is damaged, and a new code is not.

After forgetting, we tried to teach the same task again. Some arms kept the forgotten
task's own code. Others used a brand new one.

| Arm | Task code used | Steps before it responds at all |
| --- | --- | ---: |
| Attacker | The forgotten task's own | 38 |
| Code only | The forgotten task's own | 38 |
| Statistics only | Brand new | 2 |
| Fresh start | Brand new | 2 |

The split follows the code and nothing else. It held across three runs on two different
tasks, at every image budget and both stopping points. Eighteen runs on tasks the model
had never seen also started responding immediately.

Taken with the appendix result above, this gives the same conclusion from two
directions. The paper's own baseline shows that an untrained code still reaches the
task. Our runs show that the forgotten code is the one part that stopped working.

## Is the skill truly gone?

**The paper's position.** Yes, removed.

**What we measured.** We cannot tell. Relearning the forgotten task is no faster than
learning it from nothing.

This is the result that limits what we can claim, so it is stated carefully.

We took the forgotten model and gave it a small number of images from the forgotten
task, between 1 and 20 per class. We then compared how fast it recovered against a model
that had never seen that task. If something had been left behind, the forgotten model
should come back faster.

| Task probed | Gap at the 6-step model | Gap at the 100-step model |
| --- | ---: | ---: |
| Task 3, run 1 | -7.3 | -2.3 |
| Task 3, run 2 | -6.5 | -3.7 |
| Task 14 | +1.9 | +1.3 |

A positive number means the forgotten model did better than a fresh start. The task 14
numbers are inside the 3-point tie threshold, so they count as a tie, not a win.

Task 14 matters because it answers the obvious objection. Task 3 was weak before
forgetting, at 35.60 percent, so the probe had no room to show a gain. Task 14 stood at
42.60 percent with about 12 points of room. The answer did not change.

The forgotten tasks were also no easier than tasks the model had never seen. Task 14 was
in fact 5.0 and 5.6 points harder.

## What do we not claim?

We do not claim the skill survives forgetting. Our own probe says it does not, three
times over.

The likely explanation is that the gradients pass through the trunk on their way to the
task code, and damage the trunk along the way. If so, forgetting works by side effect.
The paper credits the task code, and the task code was never holding the skill.

A second explanation remains open: the trunk is intact and our relearning probe is
simply unable to detect what is left. We cannot rule this out.

Our paper separates the two questions. Whether the method forgets is one question, and
the answer appears to be yes. How it forgets is a different question, and the paper's
answer to that one does not match the measurements.

## What is still missing?

Three things, in order of importance.

The recovery probe and the collapse measurement have been run on one seed. Two more
seeds are needed before any of this is reported as a result.

One observation is not yet a claim. The old code costs about 40 steps of dead time, and
then saves about 20 to 40 steps of learning once it wakes. On one task the two roughly
cancel. The pattern was mixed on the other task, so it needs the extra seeds.

We have not checked the collapse using the paper's exact published settings. That needs
the journal version of the paper, which we do not have access to.
