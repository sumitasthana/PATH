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
