# Experiment trajectory: 17 October 2026

## Plasticity probe added and its missing baseline measured

Plasticity means the ability to learn a new task. Loss of plasticity means that ability gets worse as more tasks are learned. It is a separate question from forgetting, which is about old tasks getting worse, and the study had not measured it.

A specification for three plasticity experiments was written to `ops-docs/guides/plasticity-probes.html`, which stays local because `ops-docs` is not committed. Notebook 26, `notebooks/26_plasticity_baseline.ipynb`, implements the first of them and was committed as `e7a3a30`.

Writing it showed that most of this experiment already existed under other names. The study's `controls/order_01/seed_N/learn_15` jobs learn task 15 after the eight tasks of order 01 with no unlearning, which is the "after a full sequence" arm. The `branches/.../forget_T/learn_15` jobs are the "after unlearning" arm. Both were finished for three seeds in the 10 October work. The only arm that had never been run was the baseline: a generator that has learned nothing, taught task 15 alone. Without that baseline, the existing task-15 scores had nothing to be compared against.

Notebook 26 therefore reads the two finished arms from R2 and trains only the baseline.

## Seed 0 baseline result

The user reported the following from one notebook-26 run on seed 0, probe task 15, with the saved study settings unchanged.

| Arm | What it is | Task 15 accuracy |
| --- | --- | ---: |
| A0, chunk embeddings trainable | Fresh generator, task 15 as its first task | 45.2 |
| A0, chunk embeddings frozen | Fresh generator, chunk embeddings held fixed | 37.6 |
| A1 | After the eight tasks of order 01, then task 15 | 50.8 |

Accuracies are percentages. The A1 value matches the seed-0 L15 control already recorded on 10 October, which is a consistency check on the reading path rather than a new measurement.

The baseline is reported twice because a first task is allowed to train the chunk embeddings and a ninth task is not. Appendix B freezes them after the first task. The plain baseline therefore enjoys an advantage that has nothing to do with plasticity, and the frozen baseline is the fair comparison. The gap between the two is 7.6 points, which is the size of that structural advantage.

Against the frozen baseline the difference is +13.2 points. Against the plain baseline it is +5.6 points. Both exceed the 3-point noise floor this study has measured between identical runs, and both point the same way.

**Reading: no plasticity loss at this probe on this seed.** The generator learns a new task better after eight tasks than it did fresh. The direction is positive transfer, not stiffness.

All baseline runs passed the input-sensitivity gate, so these accuracies describe models that still respond to their input rather than models that have stopped reading it. The individual sensitivity values were not included in the supplied output.

## A second seed-0 run, and what it shows about the noise

The user repeated the same run with no settings changed and supplied its output. It does not reproduce the first.

| Quantity | First run | Second run | Difference |
| --- | ---: | ---: | ---: |
| A0, chunk embeddings trainable | 45.2 | 45.2 | 0.0 |
| A0, chunk embeddings frozen | 37.6 | 40.4 | 2.8 |
| A1 minus A0 frozen | +13.2 | +10.4 | 2.8 |
| Chunk effect | +7.6 | +4.8 | 2.8 |

The second run also supplied the input-sensitivity values the first did not: 13.01 with the chunk embeddings trainable and 11.02 with them frozen. Both sit inside the 10.8 to 13.9 band recorded for working models before unlearning. All ten classes were used in both. These are healthy models, not collapsed ones. Each run took 0.9 minutes on the reported accelerator.

A 2.8-point gap between identical runs is consistent with the roughly 3-point run-to-run variation this study has already recorded, which comes from the graphics card not always adding numbers in the same order. It is not evidence of a mistake. It is the noise floor appearing in a place where only one measurement had been taken.

Two things follow.

The plasticity reading is unchanged. Both runs give a positive difference, +13.2 and +10.4, and both exceed the noise floor. The direction is the same and the conclusion does not depend on which run is used.

The chunk effect is not usable as a number. It is 7.6 in one run and 4.8 in the other, so the variation is more than half the quantity being measured. It is a difference between two noisy measurements, which is the worst case for precision. Treat it as a rough size, not a figure to quote. What it was for still works: the plasticity change keeps its sign after the structural advantage is removed.

Notebook 26 was corrected so a seed may be listed more than once. Previously a repeated seed silently replaced the earlier run in the comparison table, which would have hidden exactly the variation seen here. The table now shows how many runs each seed had and their spread, and averages the repeats. The default is now two runs per seed across all three seeds.

## Three seeds, two runs each: no plasticity change

The user ran notebook 26 at its new default, twelve runs in total. This supersedes both single-seed readings above.

| Seed | A0 fair, mean of 2 runs | Spread | A1 | A1 minus A0 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 44.5 | 3.0 | 50.8 | +6.3 |
| 1 | 43.2 | 3.2 | 44.0 | +0.8 |
| 2 | 39.0 | 3.6 | 34.4 | -4.6 |

The three-seed mean is +0.8 points with a sample SD of 5.5. The mean is inside the 3-point noise floor and far smaller than its own spread across seeds.

**Reading: no plasticity change at this probe.** The generator does not learn a new task worse after eight tasks, and it does not learn one better either. The seed-0 result recorded earlier was the most favourable of the three and did not survive the other two, which is what the caution on that section anticipated.

Every one of the twelve runs used all ten classes and reported input sensitivity between 10.67 and 14.33, inside the band recorded for working models. No run was voided by the gate. Each took 0.9 minutes.

## The noise floor is larger than 3 points

Seed 0 with the chunk embeddings frozen has now been run four times across three sessions, with no settings changed: 37.6, 40.4, 43.0 and 46.0 percent. That is a range of 8.4 points.

Within a single session the repeat spreads were smaller, at 3.0, 3.2 and 3.6 points. The wider range appears across sessions.

The 3-point floor used in the decision rule was carried over from the replay work and is too optimistic for this measurement. Four samples are too few to replace it with a number, and no cause has been established for the session-to-session component. The practical effect here is one-directional: a larger floor makes the "no change" reading safer, not weaker. Any future result from this notebook that lands between 3 and 8 points should be treated as unresolved rather than as a finding.

## The sequence makes new-task performance depend on the seed

One pattern is worth recording, as an observation rather than a claim.

The fresh baseline varies little across seeds: 44.5, 43.2 and 39.0 percent, a range of 5.5 points. The same task learned after eight tasks varies a great deal: 50.8, 44.0 and 34.4 percent, a range of 16.4 points.

So the eight-task model's ability to learn task 15 depends heavily on which seed it was, while a fresh generator's does not. Learning the sequence appears to introduce seed dependence rather than a uniform change in learning ability. This rests on three seeds and one probe task, and no mechanism has been tested.

## Shared measurement module, and a weight-change measurement beside the behaviour one

Input sensitivity had been copied into notebooks 21, 22, 23, 24 and 26 as five versions that had drifted apart. `uncle/measure.py` now holds one copy, with ten tests. It also adds `relative_change`, the quantity the teammate's adapter plasticity study calls `rel_dW`:

    rel_dW = || W_now - W_start || / || W_start ||

pooled over every generated tensor and computed in float64, the same definition that study uses, so the two projects report comparable numbers.

Notebook 27 runs both on one unlearning request. The user supplied its first run: order 01, seed 0, task 3 removed, 100 steps, 0.9 minutes.

| Step | Accuracy | Classes used | Weight size | rel_dW | Input sensitivity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35.6% | 9 | 4706.2 | 0.0000 | 1.324e+01 |
| 20 | 10.0% | 2 | 4007.7 | 0.1636 | 3.943e-01 |
| 40 | 10.0% | 1 | 3542.3 | 0.2799 | 2.595e-02 |
| 60 | 10.0% | 1 | 3191.9 | 0.3921 | 1.743e-03 |
| 80 | 10.0% | 1 | 2921.1 | 0.5119 | 7.385e-05 |
| 100 | 10.0% | 1 | 2690.3 | 0.6289 | 2.280e-06 |

**This reproduces the conclusion report exactly**, which is what the run was for. Table 8 of that report gives a starting weight size of 4,706 falling to 2,690, which is 57%, and input sensitivity of 13.2 falling to 0.00000228. Its Table 4 gives the per-class counts: 494 images to class 3 at step 20, all 500 to class 3 at step 40, and all 500 to class 0 by step 100. Every one of those appears again here. The shared module produces what the five separate copies produced.

The seven kept tasks moved from a mean of 40.4% to 40.0% across the same steps, with no task approaching chance. Unlearning stayed specific.

Two numbers are new, because the earlier work never computed them. The relative weight change reaches 0.629. The tensor that moved furthest is `layer1.1.bn1.bias`, an early BatchNorm offset, which is generated like every other weight and so is not covered by the report's finding that BatchNorm running statistics are untouched.

## A correction to how this contrast was being described

The notebook's first version printed that the weights "moved a modest amount". The measurement contradicts that: rel_dW reaches 0.629, which is a large move. That line was wrong and has been replaced.

The contrast still holds, but it is not the one that wording described. Both measurements register the unlearning. They differ in what they can express. A relative weight change is bounded in practice: weights driven toward zero give a value approaching 1, so 0.629 is a large reading on a scale with a ceiling. Input sensitivity fell by 6.8 factors of ten and has no such ceiling.

So the accurate statement is narrower than "a weight measurement would show nothing". It is that no bounded weight measurement can distinguish a model that changed a great deal from one that stopped responding to its input, because both sit near the top of its range. The behavioural measurement separates them. This does not change the study's finding; it corrects how the finding was being explained.

## Stream metrics, for comparison with the adapter plasticity study

The teammate's adapter plasticity study reports four summary numbers and a curve, all taken from one accuracy matrix per run. None had been computed for this framework. They turn out to need no new runs: every source job already stores the accuracy of every seen task after every request, so the matrix was recoverable from records saved on 10 October.

`uncle/stream_metrics.py` implements the five, with eighteen tests against a hand-worked three-task matrix. Notebook 28 computes them from R2 and needs no GPU.

Results for order 01, all three seeds, eight tasks, task-incremental.

| Metric | Mean | SD | 95% interval | Per seed |
| --- | ---: | ---: | --- | --- |
| Final accuracy, percent | 40.59 | 2.11 | [38.98, 42.98] | 39.83, 38.98, 42.98 |
| Learning accuracy, percent | 48.53 | 0.99 | [47.50, 49.48] | 49.48, 47.50, 48.62 |
| Forgetting, points | 9.47 | 2.49 | [6.60, 11.06] | 11.06, 10.74, 6.60 |
| Backward transfer, points | -9.08 | 2.36 | [-11.03, -6.46] | -11.03, -9.74, -6.46 |
| Curve slope, points per task | -0.79 | 0.71 | [-1.60, -0.26] | -0.26, -1.60, -0.51 |

The new-task accuracy curve, meaning how well each task scored at the moment it was learned, averaged over the three seeds:

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task | 3 | 0 | 9 | 5 | 17 | 1 | 7 | 14 |
| Mean accuracy | 46.9 | 50.9 | 49.4 | 52.7 | 49.5 | 54.3 | 42.9 | 41.6 |

The curve is flat or rising for the first six positions and falls at the last two.

## Why the falling curve is not a plasticity finding

The slope is negative in all three seeds and the bootstrap interval excludes zero. That is the first of the two conditions the adapter study fixes before declaring plasticity loss. It is not the finding, for two reasons.

**The second condition cannot be evaluated.** The rule also requires the mean change to exceed a noise floor estimated from repeated measurements at the first probe point. The stream runs were never repeated within a seed, so there is no repetition spread to estimate one from. One condition out of two is not the rule.

**Task identity and position are confounded.** All three seeds run the same learning order, so position 7 is always task 7 and position 8 is always task 14. If those two tasks are simply harder than the others, the curve falls for that reason and plasticity has nothing to do with it. The adapter study avoids this by probing the same held-out tasks at every point; an in-stream diagonal cannot, because every point is a different task.

The study plan already contains two further orders that would separate the two. Position 8 holds task 14 in order 01, task 5 in order 02 and task 3 in order 03. Running those would show whether the dip follows the position or the task. This is the same confound the conclusion report records as an open limitation, now with a specific measurement that depends on it.

## What of the adapter study's measurement set is still missing here

| Metric | Status in this framework |
| --- | --- |
| Final accuracy, learning accuracy, forgetting, backward transfer, curve slope | Done, above |
| E1, random-label training loss | Not run |
| E2, three unrelated probe tasks at several probe points | Approximated only. Notebook 26 uses one probe task from the same dataset family and two probe points |
| Head-only control and R(t) | Not run, and may not transfer. See below |
| Feature effective rank | Not run |
| Joint upper bound | Not run |
| Noise floor from repeated probes at t = 0 | Not run, so the decision rule cannot be applied to anything |
| rel_dW | Measured once during unlearning, not after each task across the stream |
| Adapter geometry: rotation angles, conditioning of B | Not applicable, there is no adapter |

Two differences make some of these not directly comparable, and both should be stated wherever these numbers appear beside the adapter study's.

**Only the task-incremental form exists here.** That study reports every metric twice because its model has one head over all classes. This framework has no such head: a task code generates a ten-output classifier and the code must be supplied to get any prediction. The adapter study calls the task-incremental form the cleaner read, so this is the comparable one, but it was not a choice.

**R(t) may not transfer.** The adapter study's head-only control is a linear probe on frozen pretrained features, which gives a meaningful floor. This framework has no pretrained backbone, so its equivalent is a frozen random generator plus a fresh 32-number code, which sits near chance. The denominator of R(t) would then be close to zero and the ratio would not mean what it means there.

## The adapter study's plasticity probe, ported and run

Notebook 29 runs that study's probe protocol properly, replacing the approximation in notebook 26. It takes a copy of the model, trains a fresh 32-number task code on a task the stream never saw, measures, then restores the model and checks it is unchanged. The generator plays the part that study gives the adapter, and the fresh code the part it gives the head, so its head-only control becomes a code-only arm.

Against notebook 26 it adds three probe tasks instead of one, a three-point learning-rate grid, five repetitions at the first probe point so the noise floor is measured rather than borrowed, feature effective rank, and the fork-and-discard protocol. It also removes notebook 26's chunk-embedding confound: the probe never trains the chunk embeddings, so both probe points are on equal footing.

An integration run on a small model caught a defect before any GPU time was spent. `add_task` draws a task code from a stream keyed by the run's seed and the task name, so every repetition began from the same code and differed only in batch order. The five repetitions came out identical, the noise floor computed as exactly zero, and every comparison would then have read as significant. The probe now redraws the code from the probe seed, which carries the repetition.

The user supplied the first full run: seed 0, order 01, 168 probe runs, 118 minutes.

| Endpoint | Arm | t = 0 | t = 8 | Change | Noise floor | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E2 | full | 52.45 | 59.07 | +6.62 | 4.85 | No loss, change is the other way |
| E1 | full | 2.26 | 2.24 | -0.03 | 0.04 | Void, see below |
| E2 | code-only | 10.35 | 10.00 | | | |
| E1 | code-only | 2.72 | 2.66 | | | |

E2 is test accuracy on tasks 2, 4 and 6, averaged, in percent. E1 is the final training loss on randomly labelled images.

**E2 shows no plasticity loss.** After eight tasks the generator learns an unseen task 6.62 points better than a generator that has learned nothing, and that exceeds the noise floor measured from the five repetitions. The direction and size agree with notebook 26's seed-0 reading of +6.3 points, which used a different probe task and a different protocol.

**E1 is void.** Chance loss for ten classes is ln(10) = 2.303. The full arm ends at 2.26 and 2.24, which is at chance, and the code-only arm ends at 2.72 and 2.66, which is worse than guessing. The probe never memorised anything, so its "no finding" says the budget was too small and says nothing about the model. The dose was 5,000 images for 400 steps, about 5 passes. The adapter study hit this same problem and fixed it with 2,000 images for 600 steps, about 19 passes, which is now the default here. The notebook also now refuses to issue an E1 verdict while the loss sits near chance.

**The code-only arm sits at chance at both points**, 10.35 and 10.00 percent against a chance level of 10. A frozen generator with a fresh 32-number code cannot learn a task at all. This confirms directly what section 5.2 of the conclusion report argued from the unlearned case: 32 numbers cannot hold a task.

That has a consequence for R(t), and it is the opposite of what was expected. The worry recorded earlier was that the denominator would be near zero. It is not: the generator's contribution is the full arm minus the code-only arm, which is 42.11 at t = 0 and 49.07 at t = 8, giving R(8) = 1.17. The ratio is computable. But because the control is pinned at chance at every point, R(t) carries no information beyond E2 itself, which it tracks. It should be reported as a number and described as not independent.

Feature effective rank read 467.9 at t = 0 and 396.6 at t = 8, but with a standard deviation near 275 in both, because the first version averaged over every learning rate rather than only the runs the grid selected. That is corrected; the figure should be read again from the next run.

## The protection term works against this result, not for it

When A1 learns task 15, a protection term holds the generator near its existing weights for the eight tasks already learned. Its strength is beta, set to 0.1. The baseline has no earlier task, so that term is absent there.

So A1 carried a handicap the baseline did not, and still scored higher. The planned control arm A1z, which repeats A1 with beta set to zero, was there to rule out the protection term as the cause of a plasticity loss. With the difference positive, that arm would only widen the gap and is no longer needed. It stays available if a later seed reverses the sign.

## Limits

Three seeds, two runs each, one probe task, one learning order.

Task 15 is one probe. A single task can be unusually easy or hard by accident. Tasks 2, 4 and 6 are also unseen in order 01 and would serve as further probes, but the comparison arms do not exist for them and would have to be run.

Three seeds give a sample SD but not a narrow one. The seed-to-seed SD of 5.5 points is larger than the mean effect, so this rules out a large plasticity change rather than establishing that the change is exactly zero. A small effect in either direction would not be visible at this sample size.

This result says nothing about erasure, recovery or privacy. It also says nothing about the paper's claim that unlearning restores plasticity: the relevant comparison there is the branch minus control difference, whose three-seed means are +5.40 points for U3/L15 and -0.20 for U14/L15, both at or inside the noise floor.

## Next

Experiment A is answered for this probe. Plasticity is not where the problem is, and the planned beta-zero control arm is not needed, because there is no plasticity loss for the protection term to explain.

That removes a candidate rather than producing one. The main finding of the study is unchanged and remains the one in the conclusion report: the method reaches chance accuracy by making the model stop responding to its input, and the reported measurements do not separate that state from an untrained one.

Two cheap checks remain from the local guide, each one run, each deciding whether a larger experiment is worth starting. Experiment C's code-only arm at one learned task asks whether a 32-number task code carries anything at all in a healthy model. Experiment B's baseline asks whether the 5-epoch budget allows a model to memorise random labels at all. Run either only if the question is still wanted; neither follows from this result.

The seed-dependence observation above is the one new thread. It would need more seeds and a second probe task before it is worth pursuing.

Do not start Experiment B or Experiment C until that is done. Each has a one-run check first, described in the local guide, that decides whether it is worth running at all.
