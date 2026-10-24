# Experiment trajectory: 23 October 2026

[Back to the experiment index](Experiment-trajectory)

## Review of the EuroSAT plan, and a correction to that review

No training was run today. This page first recorded a review of the [22 October plan](Experiment-trajectory-2026-10-22) against the adapter study in `reference/adapter-plasticity-study`. The main claim of that review was wrong. This page now records both the claim and its correction, because the wrong version was committed and the record should show what happened.

### What the review claimed

That the 22 October plan compares a generator at t = 0, which has random weights, against one that has learned eight tasks, and that this comparison cannot answer a plasticity question. The reasoning was that a random generator has never produced working convolution filters, so it would learn poorly from 800 images in 250 steps, and the t = 8 model would win for a reason unrelated to plasticity.

On that basis the review recommended deferring the EuroSAT plan and restating the 20 and 21 October finding.

### Why it was wrong

The 20 and 21 October record already contains the numbers that settle it. At t = 0 the full probe reaches 51.87, 53.09 and 52.51 percent on ten-class tasks. Chance is 10 percent.

The untrained generator learns the probe task well inside the budget. The gap to the eight-task model is about 5 points, not the collapse the review assumed. The claim was a prediction presented as a fact, and the measurement contradicting it was already in our own record.

The review also leaned on a second mistake. It cited the 21 October observation that a frozen generator with a fresh task code sits at chance, and treated that as evidence that the reference model is weak. That observation is about the code-only control arm, where the generator cannot change. It says nothing about the full probe, which trains the generator. The two arms have to be read separately.

The recommendation to defer the EuroSAT plan does not follow and has been withdrawn. The plan's status is planned again.

### A related error in uncle/probe.py

The warning on `relative_contribution` said that because the control at t = 0 sits near chance, G_0 is close to zero and the ratio divides by almost nothing.

That does not follow. G_0 is the gap between the two arms, not the control's own score:

```text
G_0 = full-probe performance - code-only performance
```

With the measured values, G_0 is about 51.9 minus 10.8, or 41 points. Nothing is unstable. The 21 October record confirms it: the ratio computed to 1.174, 1.076 and 1.124.

The real limitation is different, and that record states it correctly. Our control is pinned near chance at every point, so the gap tracks the full probe's accuracy almost exactly and the ratio repeats that accuracy in different units. In the adapter study the control is a linear probe on frozen pretrained features, which scores well and moves as the features change, so their ratio carries information their raw accuracy does not.

The docstring has been corrected.

### What the t = 0 comparison actually measures

The comparison measures learnability relative to initialization. It does not separate transfer from plasticity changes, and it does not rule out deterioration relative to an earlier trained checkpoint.

Two limits follow. Useful transfer can hide deterioration, because a model that has become harder to train may still score higher on a related probe. And a rise early in the sequence followed by a fall later cannot be seen when only the endpoints are measured.

The correct reading of the existing result: after eight tasks the generator performed better on the tested probes than a freshly initialized generator, under the chosen training budgets. This does not rule out deterioration relative to an earlier trained checkpoint, and it does not separate transfer from changes in plasticity.

### What survives from the review

Adding trained reference points. Not because t = 0 is invalid, but because one trained reference does not fix the problem either. A single task does not guarantee a capable reference, and transfer can affect a comparison between any two trained states. Keep t = 0 and add t = 1, 2, 4 and 6 where practical, so the shape across the sequence is visible instead of one difference between endpoints.

This needs the sequence trained again. Notebook 29 records that no mid-stream checkpoints were kept. The review wrongly blamed `uncle/checkpoint.save` for this: that function writes to whatever path it is given, so per-task files need only a per-task path. The states were not saved; the function did not prevent saving them.

Reading the two probe arms separately, as described above.

Logging diagnostics between probe points, which we do not do at all. The 22 October page now lists them with their real costs. Two corrections to the review's version: the fraction of silent channels needs a forward pass over a defined input set, not weights alone, and relative weight change across the sequence needs the earlier states, which an endpoint checkpoint cannot reconstruct.

These diagnostics are exploratory. The condition number of a generator head is not an established equivalent of the adapter study's LoRA factor conditioning. And the review's claim that a small weight movement would make any plasticity experiment on this stream pointless is wrong: weight distance does not determine learning ability, and geometry cannot replace the probe.

The revised compute estimate. With five repetitions at the reference, two at each later state, and both arms, two model states come to 3 x (5 + 2) x 3 x 2 = 126 probe runs, not the 90 of the original design. Each further state adds 36. The plan now also states which repetitions are paired when the counts differ.

### What does not port from the adapter study

Two of its mechanisms have no counterpart here. One concerns the Cayley map used by orthogonal finetuning, the other the two factor matrices in LoRA. We have neither.

Its argument that capacity cannot shrink also does not carry over. That holds there because the backbone is frozen and the adapter is a rotation or a low-rank term, so the set of reachable weights is fixed by construction. Our generator is the model and can lose expressivity, so we cannot assume that a slowdown means bad coordinates.

On re-anchoring, the defensible statement is narrower than the review's. We currently have no defined and validated equivalent of the adapter study's re-anchoring intervention. Their operation folds the adapter into the frozen backbone and restarts it at identity; we have no frozen backbone to fold into. That rules out the specific operation, not every function-preserving reparameterization of a hypernetwork.

What does port is the split between the part carrying the history and the part new to the probe: the generator against the fresh 32-number task code. `uncle/probe.py` implements both arms through `code_only` and `train_code`.

### The adapter study's own result, for context

At the learning rates that maximise accuracy, that study found no plasticity loss in either LoRA or orthogonal finetuning across 20 tasks. Its measurable effect appeared only when the adapter learning rate was pushed past the accuracy optimum, and then only for orthogonal finetuning.

After 20 tasks its mean rotation angle is 3.0 degrees, with a 99th percentile of 11.5 degrees. At that angle the slowdown factor it predicts is 0.99, so its own mechanism is inactive at realistic settings. Its informative arm was the one that pushed the learning rate, not the one at the best settings.

### Decisions

The EuroSAT plan stands. Its deferral is withdrawn.

Add intermediate measurement points when the sequence is next trained, saving the generator after each task.

Correct the `relative_contribution` docstring. Done.

## First direction measurement: unlearning is not a pure scaling

Run: notebook 30, seed 0, order 01, forgetting task 3, control task 0, 100 unlearning steps.
Colab CPU, 11.3 seconds per step, 18.9 minutes. No GPU and no images: the unlearning
objective is a noise term plus a weight-space preservation term, so it never reads a dataset.
One seed. This is an observation, not a result.

The post-unlearning models from the paired study were not kept, for any of the three seeds.
The notebook therefore loads the source model and applies one unlearning request in the
session, as notebook 27 does.

### The trajectory

| Step | Forgotten direction | Forgotten size change | Forgotten raw size | Control direction | Control size change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.0000 | 4699.98 | 1.0000 | 0.0000 |
| 20 | 0.9919 | 0.1977 | 4003.17 | 0.9998 | 0.0222 |
| 40 | 0.9673 | 0.3383 | 3537.74 | 0.9998 | 0.0212 |
| 60 | 0.9059 | 0.4741 | 3187.64 | 0.9998 | 0.0227 |
| 80 | 0.7915 | 0.6188 | 2917.46 | 0.9997 | 0.0237 |
| 100 | 0.6542 | 0.7601 | 2687.07 | 0.9997 | 0.0243 |

Direction is cosine similarity against the weights before unlearning, with each tensor's
offset removed. 1.0 means the weights point the same way and only their size changed.

### Three checks pass

The step 0 row reads 1.0000 and 0.0000, so the reference state was captured before anything
changed it. The notebook asserts this rather than printing it.

The raw size falls from 4699.98 to 2687.07, a loss of 42.8 percent. Notebook 27 recorded
about 43 percent over the same 100 steps. The two runs are doing the same thing.

The three numbers agree with each other. For two vectors, the relative change is fixed by
the size ratio and the angle between them. With a size ratio of 0.5717 and a cosine of
0.6542, the relative change should be 0.7608. The measurement says 0.7601. Nothing in the
calculation is wrong.

### What it says

**Unlearning is not a pure scaling.** By step 100 the weights have turned 49 degrees. A
rescaling cannot undo a rotation, so multiplying the weights back up will not restore the
task on its own. The rescaling experiment in step 3 of the 21 October plan should expect a
partial result at best.

**The turning is late, and it accelerates.** The size change grows almost linearly, by about
0.14 every 20 steps. The direction does not. It loses 7.3 degrees in the first 20 steps and
11.4 degrees in the last 20. At step 20 the weights have shrunk by a fifth and turned by 7
degrees. At step 40 they have shrunk by a third and turned by 15 degrees.

So early unlearning is close to a pure scaling and late unlearning is not. The two effects
are separable in time, which the endpoint alone would have hidden.

**That makes the unlearning length a variable, not a setting.** `burn_in` is 100 by choice.
If the removed task reaches chance accuracy well before step 100, the steps after that point
are not buying forgetting; they are buying irreversibility. Notebook 27 recorded the removed
task's accuracy at the same 20-step intervals, so this can be checked against saved numbers
without a new run.

**No channel died in weight space.** The died column is 0 for all 54 weight tensors. Every
channel still has weights. Median size ratios sit between 0.45 and 0.63, and median
per-channel cosines between 0.88 and 0.96.

This does not contradict the 21 October finding that 58.9 percent of channels went silent.
The two measure different things. Silent was defined on activations, with images passing
through the network. Died here is defined on weights. A channel can keep its weights and
still produce no response, which is what the 21 October record proposed: the weights shrink,
the pre-activation shrinks with them, and the BatchNorm layer then divides by a running
spread computed from the forgotten task's own data, which unlearning never touches.

The weight measurement supports that explanation. It does not prove it, and the BatchNorm
statistics have still not been examined.

**The damage stays where it was aimed.** The control task turned by 0.02 degrees and its
weights moved by 2.4 percent, against 49 degrees and 76 percent for the forgotten task. The
preservation term is doing its job. The control is not perfectly still, and its small
movement appears by step 20 and then holds, which is worth one line in any writeup rather
than being rounded away.

## The direction result holds on three seeds

Run: notebook 30, seeds 0, 1 and 2, order 01, forgetting task 3, 100 steps each. Colab CPU,
about 11.2 seconds per step, 18.7 minutes per seed. The reading interval stayed at 20, so
the finer trace over the first 20 steps was not taken and the crossing to chance is still
bracketed rather than located.

| Step | Seed 0 turned | Seed 0 length | Seed 1 turned | Seed 1 length | Seed 2 turned | Seed 2 length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 7.3 deg | 0.852 | 10.0 deg | 0.797 | 10.5 deg | 0.775 |
| 40 | 14.7 deg | 0.753 | 20.7 deg | 0.675 | 20.8 deg | 0.646 |
| 60 | 25.1 deg | 0.678 | 33.5 deg | 0.593 | 32.8 deg | 0.563 |
| 80 | 37.7 deg | 0.621 | 45.6 deg | 0.533 | 44.4 deg | 0.505 |
| 100 | 49.1 deg | 0.572 | 54.1 deg | 0.482 | 53.0 deg | 0.457 |

Every seed shows the same shape. The length falls steadily. The turning is small early and
accelerates. At step 20 the direction is 0.9833 to 0.9919, a mean of 0.9866, which is a turn
of 7.3 to 10.5 degrees. By step 100 it is 0.5864 to 0.6542, a turn of 49.1 to 54.1 degrees.

Seed 0 is the mildest of the three on every reading. Any statement made from seed 0 alone
therefore understated the effect rather than overstating it.

No channel lost its weights on any seed. The died column is 0 for all 54 tensors across all
three runs.

The control task held on every seed: direction 0.9997 to 0.9999 throughout, with its weights
moving 1.6 to 2.5 percent against 76 to 81 percent for the forgotten task.

### What this does and does not extend

It extends the weight measurement to three seeds. The direction trajectory is not a
single-seed accident.

It does not extend the accuracy side. Notebook 27 ran only seed 0, so the joined table that
shows accuracy reaching chance at step 20 still rests on one seed. The claim that the task is
gone at step 20 is one seed; the claim that the weights have barely turned at step 20 is
three.

### The rescaling factor differs by seed

The factor that restores the original length at step 20 is 1.174 for seed 0, 1.255 for seed 1
and 1.290 for seed 2. A recovery experiment has to compute this per seed rather than reuse
seed 0's number.

### Per-channel, seed 2 turned more in its early layers

Seed 2's first convolution has a median channel cosine of 0.7712 and a lowest of 0.4938,
against 0.9315 and 0.8526 for seed 0. Its early layer1 blocks are similar. Seeds 0 and 1 keep
their early layers closer to where they started. This is one observation on one seed and may
be noise, but if a recovery attempt works on seeds 0 and 1 and fails on seed 2, this is the
first place to look.

## Forgetting finishes at step 20; the other 80 steps only turn the weights

Run: notebook 31, which joins notebook 27's saved accuracy readings to notebook 30's
direction readings on the step number. No new training. Seed 0, order 01, task 3.

The join was checked before it was used. Both notebooks record the raw weight size for the
forgotten task. The two runs differ by 0.113 to 0.132 percent at every shared step, and the
gap does not grow, so this is one event measured twice and not two different runs.

| Step | Removed task accuracy | Classes used | Input sensitivity | Direction | Turned | Length left |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35.60% | 9 | 1.324e+01 | 1.0000 | 0.0 deg | 1.000 |
| 20 | 10.00% | 2 | 3.943e-01 | 0.9919 | 7.3 deg | 0.852 |
| 40 | 10.00% | 1 | 2.595e-02 | 0.9673 | 14.7 deg | 0.753 |
| 60 | 10.00% | 1 | 1.743e-03 | 0.9059 | 25.1 deg | 0.678 |
| 80 | 10.00% | 1 | 7.385e-05 | 0.7915 | 37.7 deg | 0.621 |
| 100 | 10.00% | 1 | 2.280e-06 | 0.6542 | 49.1 deg | 0.572 |

### The task is gone while the weights are still almost unchanged

Accuracy reaches chance at step 20 and never moves again. At that step the weights are 99.19
percent aligned with where they started, a turn of 7.3 degrees, and they have kept 85 percent
of their length.

The remaining 80 steps remove no further accuracy. They take the direction from 0.9919 to
0.6542, which is the part a rescaling cannot undo.

That is the hypothesis from the direction measurement, confirmed on this seed: the unlearning
length past step 20 buys irreversibility, not forgetting.

### Accuracy is the wrong measure after step 20, and it says so itself

The classes-used column is what shows this. From step 40 the model answers the same single
class for every test image. A balanced ten-class test set gives exactly 10.00 percent for
that behaviour, which is also the chance score. The two are indistinguishable by accuracy
alone.

So a model at chance accuracy is not evidence that a task was removed. A model that has
stopped responding at all scores the same. Any claim of unlearning that rests on a
chance-level accuracy needs a second measurement beside it. This matters beyond our own work,
because accuracy at chance is the usual evidence offered.

Input sensitivity is the measurement that keeps working. It is the widest range any single
output score covers across the test set, so it says whether the model still reacts to what it
is shown. It falls by a factor of about 5.8 million from start to end, and by a factor of 15
to 34 in every 20-step window, including the windows where accuracy is flat.

### A 15 percent shrink and a 7 degree turn are enough to collapse the function

Between step 0 and step 20 the weights lose 15 percent of their length and turn 7.3 degrees.
Over the same 20 steps, accuracy falls from 35.60 percent to chance and input sensitivity
falls by a factor of 34.

A small change in the weights produces a very large change in the behaviour. That is what the
BatchNorm explanation in the 21 October record predicts: the signal entering each BatchNorm
layer shrinks, the layer divides by a spread computed from the forgotten task's own data,
unlearning never updates that spread, and the mismatch compounds through the network.

The weight measurement is consistent with that explanation. It is not a test of it. The
BatchNorm statistics have still not been examined.

### What limits this reading

The removed task started at 35.60 percent, not at the 99 percent it would reach when freshly
learned. Task 3 is first in order 01, so it had already lost most of its accuracy to ordinary
forgetting across the following seven tasks. The unlearning only had to remove 25.6 points
above chance. A task unlearned from full accuracy may behave differently, and the first task
in the order is the easiest case.

Readings are 20 steps apart, so the crossing to chance happened somewhere in steps 1 to 20
and is not located. A run with a smaller interval would find it, and would cost about four
minutes on CPU.

One seed.

### The experiment this points to

Unlearn to step 20 only. Multiply the generated weights by 1.174, the factor that restores
their original length. Measure accuracy with no training.

At step 20 the direction is 0.9919, so after rescaling the weights sit within 7.3 degrees of
the originals. If accuracy returns, the model forgot the task in a way that a multiplication
undoes, and the privacy claim for this unlearning fails at its own stopping point rather
than at some artificial early step. If accuracy does not return, the 7.3 degree turn or the
BatchNorm statistics are carrying the removal, and the next question is which.

Rescaling the step 100 state is the weaker version of the same test, since 49 degrees cannot
be undone by any multiplier. Run the step 20 version first.

### Next, in order

Done: the accuracy join, in the section above.

Rescale the step 20 state and measure accuracy, as described above. This needs a forward
pass over the test set, so it needs images, but one pass over 1,000 test images runs on CPU.

Run seeds 1 and 2 of the direction measurement. About 38 minutes on CPU, so RunPod is not
needed for this.

Repeat the direction measurement with a smaller reading interval over the first 20 steps, to
locate the point where accuracy reaches chance instead of bracketing it.

Examine the BatchNorm running statistics. Still untouched, and still the largest open
thread.

### Open question, not yet decided

Whether plasticity remains the headline of the work, or becomes a supporting measurement while the unlearning results take the main line.

The argument for moving it is weaker than this page first claimed, since the plasticity measurements are sound and only their interpretation needed narrowing. It rests instead on what the two threads can deliver. The adapter study found no loss over a longer and better-powered stream, and our own eight-task result shows improvement rather than decline. The unlearning thread has a published claim that has never been tested, a mechanism the 21 October measurements contradict, and untouched per-task BatchNorm running statistics, which are both a route to recovery and a gap in any privacy claim.

The three steps at the end of the 21 October record belong to that thread, and the first is one extra line in a measurement already written.

Recorded as a recommendation. No decision has been made.
