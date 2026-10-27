# Experiment trajectory: 26 October 2026

[Back to the experiment index](Experiment-trajectory)

## The recovery probe ran, at two stopping points

Notebook 24 had been built on 13 October and never run. It ran today on Colab, after four changes recorded below. This is the first recovery measurement in the project.

The headline is not the one the study plan expected. No recovery advantage was found. A different and more specific result appeared instead, and it is consistent across every comparison in the run.

### Settings

| Setting | Value |
| --- | --- |
| Study, order, seed | `paired_generalization_v2`, order 01, seed 0 |
| Forgotten task | 3, which scored 35.60 percent before unlearning |
| Unlearning budgets | 6 steps and 100 steps, each from its own fresh build of the saved source model |
| Arms | attacker, code only, stats only, fresh start |
| Image budgets | 1, 5 and 20 images per class, from task 3's own training set |
| Recovery steps | 100, measured every 10 |
| Generator during recovery | trainable |
| Unseen-task controls | 2, 4 and 6 |

The four arms are the four combinations of the two things unlearning leaves behind. The attacker keeps task 3's own 32-number code and its BatchNorm running statistics. Code only keeps the code and resets the statistics. Stats only keeps the statistics and uses a brand new code. Fresh start keeps neither.

### The retained tasks held

Mean accuracy on the seven retained tasks stayed at 40.3, 40.3 and 41.9 percent after recovery at the 6-step stopping point, and 39.9, 39.8 and 40.2 percent at the 100-step one.

This settles an open precondition. The study plan recorded that in our reproduction forgetting drove all tasks to chance, including retained ones, and said that had to be fixed before these experiments could be read. It is fixed. Nothing here is confounded by broken retained tasks.

## Result 1: no recovery advantage

The attacker arm never beat a fresh start. It lost by 7.3 points on average at the 6-step stopping point and by 2.3 points at the 100-step one.

| Peak accuracy | attacker | code only | stats only | fresh start |
| --- | --- | --- | --- | --- |
| 6 steps, 1 per class | 21.2 | 19.6 | 20.6 | 22.0 |
| 6 steps, 5 per class | 20.2 | 20.6 | 28.2 | 27.8 |
| 6 steps, 20 per class | 23.4 | 24.8 | 38.0 | 36.8 |
| 100 steps, 1 per class | 21.6 | 19.6 | 20.6 | 16.8 |
| 100 steps, 5 per class | 22.0 | 21.0 | 30.8 | 28.2 |
| 100 steps, 20 per class | 34.0 | 23.4 | 40.6 | 39.6 |

Task 3 was also no easier than a task the model had never seen. Against the mean of unseen tasks 2, 4 and 6, task 3 with a fresh code was ahead by 1.5 points at 6 steps and by 0.1 points at 100. Both are inside the 3-point tie threshold the notebook uses.

So on this seed, with this probe, there is no evidence that information about task 3 survived in a form an attacker could use. That is a negative result for the privacy hypothesis, and it is the second negative after the rescaling arms of 23 October.

## Result 2: the old task code is a liability, and this is the real finding

The run measures the step at which each arm starts responding to its input again, before which there is no learning signal to follow. The pattern is the same in all twelve comparisons.

| Arm | task code | wakes at, across both stopping points and all three budgets |
| --- | --- | --- |
| attacker | task 3's own | 40, 40, 50, 40, 40, 50 |
| code only | task 3's own | 40, 40, 50, 40, 40, 50 |
| stats only | brand new | 0, 0, 10, 0, 10, 10 |
| fresh start | brand new | 0, 0, 0, 0, 0, 0 |
| unseen 2, 4, 6 | brand new | 0 in all eighteen runs |

Every arm that keeps task 3's own code spends 40 to 50 steps unable to learn anything. Every arm with a new code starts immediately. The split follows the code and nothing else: it does not follow the BatchNorm statistics, the image budget, or the stopping point.

Read plainly: unlearning did not remove the model's ability to serve task 3. It made task 3's own address unusable. The generator still learns task 3 from a new code as readily as it learns a task it has never seen.

There is a second observation on top of this, and it is thin. At the 20-image budget the attacker arm, once awake, reaches halfway back in 10 steps, against 30 for a fresh start, at both stopping points. That would say the old code relearns faster once it escapes the dead region. It rests on two numbers, and at the smaller budgets the attacker arm never reaches halfway at all, so it is recorded as an observation and not a claim.

## Result 3: the collapse is confirmed, at both stopping points

Immediately after unlearning, at 6 steps and at 100 steps alike, the model used 1 of 10 classes and placed 100 percent of its predictions in that one class.

This is the direct confirmation of the claim the project has been making from accuracy traces. A model answering a single class scores exactly 10.00 percent on a balanced ten-class test set. The method's forget-task accuracy metric cannot distinguish that from a model that has genuinely forgotten.

It also shows the collapse is complete by step 6. The remaining 94 steps do not deepen it.

## Correction: two summary lines in the notebook read the wrong contrast

The run printed two conclusions that do not survive checking. Both are analysis errors in the notebook, not problems with the data, and both are recomputable from the saved CSV without rerunning anything.

### The BatchNorm verdict

The notebook reported that keeping the BatchNorm statistics is worth +4.5 points at the 100-step stopping point, and printed the paragraph about a data remnant the method does not attempt to remove.

It computed that as attacker minus code only. Both of those arms carry task 3's dead code, so the comparison is made between two crippled runs. Its +4.5 average comes almost entirely from one cell, the 20-image budget, where 34.0 against 23.4 contributes +10.6 while the other two budgets contribute +2.0 and +1.0.

The contrast that actually isolates the statistics is stats only against fresh start, since those two differ only in the statistics and both start from a new code. That gives +0.1 points on average at 6 steps and +2.5 at 100. Both are inside the 3-point tie threshold.

**So the BatchNorm statistics do not show a measurable recovery value on this seed.** The architectural point stands unchanged: unlearning never touches them, and they are computed from the forgotten task's images. But the claim that they carry usable recoverable information is not supported by this run, and the +4.5 figure should not be quoted.

### The collapse table

The table reporting the fewest classes a run ever used shows 1 for almost every arm, including fresh start and the unseen-task controls. That is because it takes the minimum over all measured steps, and the first measurement happens before any training.

As printed, the table says that learning a task the model has never seen also involves collapse, which is not a useful statement. The informative number is how many classes the model uses at the step where its accuracy peaks. Both fixes are in the notebook now.

## A limit that the run exposed: the probe has no headroom

Task 3 scored 35.60 percent before unlearning. At the 20-image budget, a fresh code reaches 36.8 and 39.6 percent on task 3, and 38.7 and 40.1 percent on tasks the model has never seen.

So 200 images bring any ten-class task to roughly 37 to 40 percent in this model, which is at or above what task 3 ever scored. At that budget the probe cannot show a recovery advantage, because the floor is already above the ceiling.

This follows from task 3 being the weakest possible case. It is first in the order and had lost most of its accuracy to ordinary forgetting before unlearning began. The 1-image and 5-image budgets keep some headroom but sit close to chance.

Any repeat should use a task that was still strong when it was unlearned.

## Changes made to notebook 24 before this run

Recorded because the notebook as committed on 13 October would have produced a different and partly corrupted result.

1. **Two stopping points instead of one.** Every arm now runs from a model unlearned for 6 steps and from one unlearned for 100, each built fresh from the saved source checkpoint.
2. **Collapse reporting added.** Every measurement records how many distinct classes the model predicts and what share went to the commonest one.
3. **`FREEZE_GENERATOR` now defaults to false.** With the generator frozen, a fresh code sits at chance by construction, as the 20 and 21 October record measured on three seeds. The fresh-start arm was therefore a pinned floor rather than a control.
4. **A bug fixed that corrupted the saved record.** The speed table reused the variable name `after`, overwriting the after-unlearning measurement with a formatting string before `summary.json` was written. Every earlier run of this notebook would have saved a wrong value for that field.

## Where this leaves the work

Two recovery attempts have now failed: rescaling on 23 October, and relearning today. Neither establishes that the information was removed, and today's run gives a concrete reason why a null result was likely: the old code's 40 to 50 steps of dead time, and the probe's missing headroom at the only budget with enough signal.

The positive results from today are the collapse confirmation and the dead-code finding.

Next decisions, in order:

1. Seed the recovery training end to end. The second run below shows the probe moving by up to 8.8 points between identical runs, so this comes before any further comparison.
2. Repeat on a task that was strong when unlearned, for example task 14, so the probe has room above the floor.
3. Repeat on seeds 1 and 2. Everything above is one seed.
4. Treat the dead-code result as a candidate contribution in its own right and decide whether it is the paper's mechanism claim.

## A second run of the same configuration, with the corrected reporting

The notebook was rerun after the two reporting fixes above were applied. Same study, same seed, same forgotten task, same two stopping points, same three image budgets. The few-shot images are drawn with a fixed seed, so both runs trained on the same pictures. Nothing else in the recovery training is seeded: the batch order and the new task codes differ between runs. So the two runs are independent repeats of one configuration, and the differences between them measure this probe's own noise.

### What replicated

**No recovery advantage.** The attacker arm lost to a fresh start by 6.5 points at the 6-step stopping point and by 3.7 points at the 100-step one. The first run gave 7.3 and 2.3. Same sign, same size, both runs.

**The dead code.** Average dead time was 42 steps for the arms keeping task 3's own code and 2 steps for the arms using a new code. The first run gave 43 and 2. Every individual wake step matched to within 10 steps, and all eighteen unseen-task runs woke at step 0 again.

**The collapse after unlearning.** 1 of 10 classes, 100 percent of predictions in that class, at both stopping points.

**Task 3 is not easier than an unseen task.** With a fresh code it was ahead by 1.3 points at 6 steps and behind by 2.0 points at 100. The first run gave +1.5 and +0.1. All four values are inside the tie threshold.

**The BatchNorm statistics show no recovery value.** The isolating contrast gave +0.7 and +2.1 points, against +0.1 and +2.5 in the first run. Four values, all inside the tie threshold.

### What did not replicate, and why that matters

The withdrawn +4.5 BatchNorm figure rested almost entirely on one cell: the 100-step attacker arm at 20 images per class, which scored 34.0 against code only's 23.4, a gap of +10.6. In the second run the same cell reads 25.2 against 29.6, a gap of -4.4. The attacker value moved by 8.8 points between identical runs.

That cell was the single largest source of run-to-run variation, and it was also the only evidence for the figure. The correction made by hand before this run is now confirmed by measurement.

### The noise floor is wider than 3 points, and not evenly spread

Across the 24 peak-accuracy cells the two runs differ by 2.0 points on average and by 8.8 at most. The 3-point tie threshold the notebook uses is too optimistic at the 20-image budget.

The variation is not spread evenly across the arms. The two arms carrying task 3's dead code differ between runs by 3.0 points on average, while the two arms using a new code differ by 0.9. This fits the dead-code finding: those arms are measured near the step where they start responding at all, so small changes in training move their peak a long way.

Two consequences. Any single-cell difference in a dead-code arm should not be reported. And the probe should be seeded end to end, so that a repeat tests the question rather than the randomness.

### One line retracted

The first run printed that stopping at 100 steps leaves more recoverable than stopping at 6. The second run reads the difference as -1.1 points, in the other direction, and the notebook now prints "no difference worth reporting". Both runs agree on the substance: the extra 94 unlearning steps change neither how much was removed nor how reachable it is.

## A reading of the paper's own discard baseline, and what it says about where the task lives

This section records a check against the paper, not a new run. It came out of asking whether a cheaper unlearning operation exists: delete the forgotten task's code and train nothing. The paper already tested that, and the result changes how today's dead-code finding should be read.

### The baseline exists and it fails

Appendix F, Table 8 of the paper lists `Discard e_f`: the forget task's embedding is thrown away and replaced with a random one. Nothing else changes and no unlearning steps are run. Forget-task accuracy after that operation:

| Dataset | Discard e_f | UnCLe | Chance |
| --- | --- | --- | --- |
| 5-Tasks | 80.91 | 10.04 | 10 |
| CIFAR-100 | 20.70 | 10.00 | 10 |
| Permuted-MNIST | 61.79 | 10.00 | 10 |
| Tiny-ImageNet | 15.54 | 10.00 | 10 |

Unlearning time is 0.00 in every row, so the operation is free. It is also not forgetting: the task still runs above chance on all four datasets, and at 80.91 percent on the 5-task set.

### What that means for today's result

The paper does not read this number as a finding. For this project it is one.

It says the forgotten task is not stored in its code. A code the model was never trained with still reaches the task through the shared generator. The generator is therefore only loosely conditioned on which code it is given.

That sits alongside today's dead-code result and points the same way. Our probe found that a new code starts learning task 3 immediately while task 3's own code costs 40 to 50 steps. The paper's number says a random code already serves a learned task above chance before any training at all. Both say the code is a weak address and the ability is held in the trunk.

The combined reading is that UnCLe's forget operation breaks the address rather than the ability. Today's measurements show the address broken. The paper's own baseline shows the ability surviving an address change.

Two limits on this reading. The numbers are the paper's, measured on its setup, not reproduced here. And the paper does not state whether the replacement random embedding is then frozen or adapted, which `docs/PLAN.md` line 140 already flags as an open question and which could change the figures.

### A contradiction with the paper's uniformity argument

The paper's appendix "Connecting MSE and L2" argues that minimising the MSE term drives the generated parameters toward zero, so the logits for the unlearned task become zero on every output node, so the output distribution is uniform and the model is maximally uncertain. The paper scores this with its UNI metric and reports UNI = 100.0 for UnCLe on all four datasets.

Today's run measures the same state as 1 class of 10 used, with 100 percent of predictions in that class.

A possible reconciliation, which has not been measured: near-zero logits are near-tied, so the softmax is close to uniform while the argmax falls on whichever output node keeps the largest leftover bias, and it falls there for every image. If that holds, both readings are correct and they describe the same collapsed model. It would also mean UNI, like forget-task accuracy at chance, is a criterion a zero head passes without forgetting anything.

This is an interpretation, not a result. Confirming it needs the per-class logit values at the collapsed state, which the recovery notebook does not currently save.

### Next decision from this section

Reproduce `Discard e_f` on our own seed-0 checkpoint: replace task 3's code with a random one, train nothing, and read forget-task accuracy, class coverage and input sensitivity straight away. It costs no training and gives a residue number that does not depend on the recovery probe, which has now been the weak link in two attempts.

If forget-task accuracy comes out above chance, the residue is in the trunk and measurable directly. If it comes out at chance, our implementation differs from the paper's on a point that changes the paper's own baseline, and locating that difference is itself worth recording.
