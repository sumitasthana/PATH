# Experiment trajectory: 27 October 2026

[Back to the experiment index](Experiment-trajectory)

## The recovery probe on task 14

Notebook 24 ran again with `FORGET_TASK = "14"`. Everything else was the same as the 26 October runs: study `paired_generalization_v2`, order 01, seed 0, stopping points at 6 and 100 steps, four arms, image budgets of 1, 5 and 20 per class, 100 recovery steps measured every 10, generator trainable, unseen controls 2, 4 and 6.

This run existed to fix one problem. Task 3 is first in order 01 and had fallen to 35.60 percent before unlearning. The 26 October runs showed that 200 images bring any ten-class task in this model to roughly 37 to 40 percent, which is at or above where task 3 ever stood. The probe had no room to show a recovery gain. Task 14 is last in the order, so nothing was learned after it.

It worked as intended. Task 14 scored 42.60 percent before unlearning, and a fresh code reaches 29.6 and 30.8 percent at the largest budget. That leaves about 12 points of room.

## Result 1: with room, there is still no recovery advantage

The attacker arm beat a fresh start by 1.9 points at the 6-step stopping point and by 1.3 points at the 100-step one. Both are inside the 3-point tie threshold.

The sign flipped from the task 3 runs, where the attacker lost by 6.5 and 3.7 points. That fits the headroom explanation. On task 3 the floor sat above the ceiling and the old code's dead time dominated. On task 14 there is room, and the old code roughly breaks even.

| Peak accuracy | attacker | code only | stats only | fresh start |
| --- | --- | --- | --- | --- |
| 6 steps, 1 per class | 16.6 | 19.2 | 16.8 | 19.0 |
| 6 steps, 5 per class | 24.4 | 23.8 | 20.0 | 20.0 |
| 6 steps, 20 per class | 33.4 | 34.4 | 34.0 | 29.6 |
| 100 steps, 1 per class | 16.6 | 17.4 | 16.4 | 18.2 |
| 100 steps, 5 per class | 22.4 | 25.2 | 19.8 | 19.8 |
| 100 steps, 20 per class | 33.6 | 35.8 | 32.6 | 30.8 |

**This closes the main hole in the negative result.** The headroom objection was the strongest argument that the probe had been unfair. The probe was made fair and the answer did not change.

Task 14 was also harder than the unseen controls, not easier. With a fresh code it fell short of their mean by 5.0 and 5.6 points. So there is no sign that the model finds its own forgotten task easier than a task it has never seen.

## Result 2: a strong task collapses just as fast

Task 14 stood at 42.60 percent. After 6 unlearning steps it scored exactly 10.00 percent and used 1 of 10 classes, with 100 percent of predictions in that one class. After 100 steps, the same.

This is the clearest confirmation of the step-6 finding so far. The earlier measurement was made on task 3, which was already weak, so it was open to the reply that a strong task would need longer. It does not. A task at 42.60 percent collapses to a single answer within 6 steps, and the remaining 94 steps add nothing.

The collapse figure is now measured on two tasks of different strength, at two stopping points, in three separate runs.

## Result 3: the dead code replicates on a second task

| Arm | task code | wakes at, across both stopping points and all three budgets |
| --- | --- | --- |
| attacker | task 14's own | 40, 40, 40, 40, 40, 40 |
| code only | task 14's own | 30, 40, 40, 40, 30, 40 |
| stats only | brand new | 0, 0, 0, 20, 10, 0 |
| fresh start | brand new | 0, 0, 0, 0, 0, 0 |
| unseen 2, 4, 6 | brand new | step 0 in 16 of 18 runs, step 10 in the other two |

Mean dead time was 38 steps for the arms keeping task 14's own code and 2 steps for the arms using a new one. The task 3 runs gave 43 and 2, then 42 and 2.

Three runs, two tasks, same split. The result does not depend on which task is forgotten.

The paper's own discard-embedding baseline, recorded in the [26 October page](Experiment-trajectory-2026-10-26), points the same way from the other side. Replacing the forgotten task's embedding with a random one and training nothing leaves forget-task accuracy at 80.91, 20.70, 61.79 and 15.54 percent on their four datasets. A code the model never trained with still reaches the task through the shared generator. Our arms show the matching fact from inside: the task's own code is the part that stopped working.

## A new observation: the old code pays the dead time back

The run records how many steps each arm needs to reach halfway back after it starts responding. On task 14 at the 20-image budget the pattern is consistent in all four cells.

| Stopping point | attacker | code only | stats only | fresh start |
| --- | --- | --- | --- | --- |
| 6 steps | 20 | 20 | 40 | 40 |
| 100 steps | 20 | 20 | 50 | 60 |

The two arms holding task 14's own code take 20 steps once they wake. The two arms with a new code take 40 to 60.

Read together with the dead time, this says something more precise than the earlier wording. The old code is not simply a liability. It costs about 40 steps of dead time and then saves about 20 to 40 steps of learning. On this task the two roughly cancel, which is why the attacker and the fresh start finish in a tie.

This is recorded as an observation, not a claim. The same contrast on the task 3 runs was mixed: 10 against 30 at one stopping point and 50 against 20 at the other. It needs seeds 1 and 2 before it can be stated.

## A candidate signal that needs a repeat

At the 20-image budget every arm beat the fresh start, by 3.8, 4.8 and 4.4 points at the 6-step stopping point and by 2.8, 5.0 and 1.8 at the 100-step one. Several of those sit above the tie threshold.

It should not be reported yet, for two reasons. The arm doing worst is the fresh start rather than any arm doing well, which can simply mean the fresh start had a bad run. And the 26 October repeat showed that the 20-image cells vary by up to 8.8 points between identical runs, which is wider than these gaps.

Seeds 1 and 2 decide it.

## Unchanged findings

The BatchNorm statistics show no measurable recovery value: 0.7 and 0.0 points on the contrast that isolates them, both inside the tie threshold. That now holds on two tasks.

The stopping point makes no difference: 0.6 points between stopping at 6 and stopping at 100, in favour of the earlier stop. The task 3 runs gave 4.5, then 1.1 the other way, then this. Three runs, no stable difference.

The retained tasks held, at 36.8 to 40.8 percent throughout.

## One defect the run exposed

Three printed lines still said "Task 3" when the forgotten task was 14, so the unseen-task section of the output is mislabelled. The numbers in it are correct and belong to task 14. The earlier pass over the notebook replaced only the lower-case spelling. Fixed.

## Where this leaves the work

The negative privacy result is now solid. It survives the fix that was meant to rescue it, on a task with real room above the floor, and the forgotten task is harder than tasks the model never saw rather than easier.

The mechanism result is also solid: three runs, two tasks, the same split by task code every time.

Next:

1. Seeds 1 and 2 on task 14. This settles the 20-image signal and the pay-back observation at once, and it is the last experiment the privacy claim needs.
2. The accuracy trace, notebook 27, on seeds 1 and 2.
3. The collapse check on the paper's own settings, once the journal version and any released code are in hand.
