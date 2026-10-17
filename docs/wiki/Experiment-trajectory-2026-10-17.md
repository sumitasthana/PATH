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

## The protection term works against this result, not for it

When A1 learns task 15, a protection term holds the generator near its existing weights for the eight tasks already learned. Its strength is beta, set to 0.1. The baseline has no earlier task, so that term is absent there.

So A1 carried a handicap the baseline did not, and still scored higher. The planned control arm A1z, which repeats A1 with beta set to zero, was there to rule out the protection term as the cause of a plasticity loss. With the difference positive, that arm would only widen the gap and is no longer needed. It stays available if a later seed reverses the sign.

## Limits

One seed, one probe task, one learning order. The result is provisional.

The seed spread is the main concern, and it is larger than the effect. The three recorded A1 values are 50.8, 44.0 and 34.4 percent, a spread of 16.4 points. Seed 0 is the highest of the three. If the baseline stays near 37.6 while A1 falls to 34.4 at seed 2, the difference changes sign. Seeds 1 and 2 are required before this is stated as a finding.

Task 15 is one probe. A single task can be unusually easy or hard by accident. Tasks 2, 4 and 6 are also unseen in order 01 and would serve as further probes, but the comparison arms do not exist for them and would have to be run.

This result says nothing about erasure, recovery or privacy. It also says nothing about the paper's claim that unlearning restores plasticity: the relevant comparison there is the branch minus control difference, whose three-seed means are +5.40 points for U3/L15 and -0.20 for U14/L15, both at or inside the noise floor.

## Next

Rerun notebook 26 with `SEEDS = [1, 2]` and compare the baseline against each seed's own control. Nothing else in the notebook changes. If all three seeds agree in direction, record it as a three-seed result with its spread. If they disagree, record the disagreement rather than the mean.

Do not start Experiment B or Experiment C until that is done. Each has a one-run check first, described in the local guide, that decides whether it is worth running at all.
