# Experiment trajectory: 27 October 2026

[Back to the experiment index](Experiment-trajectory)

## Validation of the task-conditioning claim

This audit supersedes the interpretations below that call the route probe a reproduction of Table 8, identify the code-only probe with a no-training probe, or say author code is unavailable. No new training result was produced. Local numerical results were checked against the working records and implementation, not recomputed from checkpoints or raw run logs.

### Author code is public

The [journal abstract](https://link.springer.com/article/10.1007/s10994-026-07125-8) links to [visprasadh/uncle](https://github.com/visprasadh/uncle). The inspected default-branch commit is `f9f1e218588f4382aac4429393510f782626438a`. The journal full text remains subscription content in the accessed page. The inspected source tree and README did not supply a discard-baseline driver, trained checkpoints, or Table 8 run logs. Code availability therefore removes one obstacle but does not establish the provenance of the reported baseline.

The released [test function](https://github.com/visprasadh/uncle/blob/f9f1e218588f4382aac4429393510f782626438a/main.py#L250) switches only the hypernetwork to evaluation mode. The separately constructed target ResNet remains in its default training mode. Its [BatchNorm implementation](https://github.com/visprasadh/uncle/blob/f9f1e218588f4382aac4429393510f782626438a/toolbox/batchnorm.py#L144) uses current-batch statistics and updates running buffers in that mode, including inside a no-gradient context. Our notebook 34 instead evaluates the target with stored task buffers. This is a concrete implementation difference and a candidate explanation to test, not an established explanation of Table 8. The released main driver is not evidence that the unreleased discard driver used the same policy.

The release also imports modules absent from its inspected tree in `toolbox/models.py`. It should not be described as verified runnable without repair. This audit did not execute the author training code.

### The comparisons do not establish input independence

The [preprint's Table 8](https://arxiv.org/html/2509.17530v1) reports replacement-embedding results for Request Sequence 1. Its definition of FA is an average over forgotten tasks at the end of the sequence. The same-dataset comparison for our Tiny ImageNet probe is 15.54 percent, not the 80.91 percent from 5-Tasks or 61.79 percent from Permuted-MNIST. Replacement is described without adaptation, and the reported unlearning time is zero. The text supports that intended operation, but does not verify implementation or establish immediate post-replacement accuracy.

Our notebook 34 reports one seed, task 14 and retained task 7, and five random codes per task. It measures before and after a single request. The repeated before readings at stopping points 6 and 100 use the same source and codes, not independent replications. Chance accuracy on these sampled routes does not exclude every possible alternative route.

High accuracy under sampled replacement codes would establish that useful prediction does not require the original code in that protocol. It would not establish that generated weights barely change with the code. Many different classifiers can have similar accuracy, and a task-independent useful component can coexist with a substantial task-dependent component. Nor would it establish independence across the entire embedding space. Testing that requires cross-code predictions and generated-weight comparisons on a fixed checkpoint, with all other task information controlled.

### Frozen during learning is different from replaced after learning

`uncle/probe.py:probe_once` confirms that the generator-only arm fixes a random code while fitting the generator to that code and task. The full arm fits both. The reported full-minus-generator-only gaps of -1.48, +1.01 and +1.27 percentage points are at the fresh-generator starting point, after probe training. They are not zero-shot scores or a whole-stream fixed-code ablation.

These results support the narrower statement that code optimization showed no clear advantage in these probes. They do not establish equivalence. Notebook 29 borrows a threshold calculated for the full arm's change between probe points, rather than calculating uncertainty in paired full-versus-generator-only differences. From the three rounded seed gaps, the mean is +0.27 points; an illustrative Student-t interval is approximately -3.50 to +4.04 points under independent, normally distributed seed differences. This small sample does not justify an exact-zero claim. Learning-rate selection also uses test accuracy in `best_of_grid`, so these are selected probe endpoints rather than untouched test estimates.

The probe training loop fits cross-entropy without the stream's preservation penalty. BatchNorm statistics also adapt. Its results cannot be extended directly to maintaining all old tasks through continual learning with every task code frozen.

The code-only arm trains the code with the generator frozen and initializes new BatchNorm buffers. It is not a frozen random-code forward pass. Its chance-level result does not prove that a 32-dimensional code cannot select task knowledge from a suitable trained generator.

A fixed random code can still identify a task after the generator learns its association with that code. An arbitrary identifier does not need a learned numerical meaning to carry identity. The claim that the code carries no information is therefore too broad; the claim that its numerical optimization may be unnecessary is testable and consistent with these observations.

The PLAN component table describes proposed interventions in our study. It is not evidence that the paper adapted replacement embeddings. The earlier attribution to that table was mistaken.

### Next decision

Request the exact Table 8 discard script and configuration, relevant trained checkpoints, per-task immediate and final accuracies, embedding sampling and optimizer policy, and BatchNorm evaluation policy. Ask whether discarded embeddings enter later preservation losses. The existing local author letter asks for code and the journal version but not these baseline details or weights; it remains unsent.

With a verified checkpoint, compare original, swapped, fresh-random and zero codes on the same tasks. Hold label mapping and other task state fixed. Separately compare stored-statistics evaluation with current-batch normalization. Measure prediction disagreement and layer-wise generated-weight changes as well as accuracy. For post-unlearning survival claims, add a matched model that never learned the forgotten task to separate retained task information from transfer. A general claim about hypernetwork continual learning requires replication beyond this implementation and these probe tasks.

The current conclusion is a protocol and implementation discrepancy worth resolving. Neither the paper's aggregate baseline nor our fixed-code probes establish that the generator ignores task identity or that UnCLe's learning and unlearning results are invalid.

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

## The alternative-route probe, and why it is void

Notebook 34 was written to answer one question. Unlearning trains the shared generator so that the forgotten task's own code produces near-zero weights. Is the task still reachable through a different code? The paper's Appendix F, Table 8 says an untrained random embedding reaches a task at up to 80.91 percent, so the probe takes that measurement and repeats it after unlearning.

Settings: study `paired_generalization_v2`, order 01, seed 0, `FORGET_TASK = "14"`, stopping points 6 and 100, five unused random task codes per reading, each given the target task's stored BatchNorm statistics. Nothing is trained anywhere in the notebook. Task 7 is the kept control, chosen as the last retained task in the order.

| Stopping point and stage | Task 14, own code | Task 14 through random codes | Task 7 through random codes |
| --- | ---: | ---: | ---: |
| 6, before unlearning | 42.6 | 9.5 | 10.0 |
| 6, after unlearning | 10.0 | 10.0 | 10.0 |
| 100, before unlearning | 42.6 | 9.5 | 10.0 |
| 100, after unlearning | 10.0 | 10.0 | 10.0 |

Per-code detail, which matters more than the means. Before unlearning, the five random codes read task 14 at 10.0, 8.6, 10.0, 10.0 and 8.8 percent, using 1, 5, 1, 1 and 3 of the ten classes. On task 7 all five read exactly 10.0 percent using a single class. After unlearning every reading is 10.0 percent on a single class.

So an untrained code in this model does not produce a weak classifier. It produces a collapsed one: a constant answer, which scores chance on a balanced set. The two readings that used more than one class scored 8.6 and 8.8 percent, below chance rather than above it.

### The probe is void, and the before column is why

The notebook printed "cannot decide" at both stopping points, and its stated reason is wrong. It reads the after column and says that reading through an unused code stopped working. The before column shows it never worked. Random codes read 9.5 and 10.0 percent on an untouched model, against 10 percent chance.

So there is no alternative route in this model, before unlearning or after. The question the probe was built to ask does not arise here, and the after column carries no information about removal.

### This repeats a result the study already has

The 20 and 21 October record measured a frozen generator with a fresh task code and found chance on all three seeds. This is the same measurement reached by a different path, now also with the target task's own BatchNorm statistics supplied, and at two unlearning stopping points. It agrees.

### What it says about Table 8

Our reproduction does not reproduce the behaviour Table 8 reports. In this model an untrained code carries a task nowhere. In the paper it carries one to 80.91 percent on the strongest of four datasets.

Two readings are open and this run does not separate them.

1. Their generator generalises across codes far more than ours does. Their own spread is wide, at 80.91, 20.70, 61.79 and 15.54 percent, and the lowest of those is close to chance, so the effect may be dataset-dependent.
2. Their discard baseline adapts the replacement embedding rather than freezing it. `docs/PLAN.md` already flags this as unresolved, in the component table under "Task embedding". If the replacement is adapted, Table 8 measures how fast an address can be retrained, not whether the task is reachable from an arbitrary address.

Settling this needs the journal version or released code, which we do not have.

### What this costs the current reading

The 80.91 figure has been used in our working documents as evidence that a task code is only an address. It cannot carry that weight here, because the same measurement in our model gives chance. Any statement built on it has to be marked as the paper's result in the paper's setting, not as a property of hypernetwork unlearning in general.

Together with the correction recorded in the 27 October weekly update, where the forgetting step was confirmed to train the generator and leave the task code frozen, this closes the address-versus-container reading from both directions.

### Next decision from this section

The route probe is not worth repeating on seeds 1 and 2 as it stands, because the instrument reads chance on an untouched model. It would become worth running again only if the Table 8 ambiguity is resolved in favour of an adapted embedding, in which case the probe should adapt the replacement code under a stated budget and the control stays as it is.

## Where this leaves the work

The negative privacy result is now solid. It survives the fix that was meant to rescue it, on a task with real room above the floor, and the forgotten task is harder than tasks the model never saw rather than easier.

The mechanism result is also solid: three runs, two tasks, the same split by task code every time.

Next:

1. Seeds 1 and 2 on task 14. This settles the 20-image signal and the pay-back observation at once, and it is the last experiment the privacy claim needs.
2. The accuracy trace, notebook 27, on seeds 1 and 2.
3. The collapse check on the paper's own settings, once the journal version and any released code are in hand.
