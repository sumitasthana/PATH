# Experiment trajectory: 29 October 2026

[Back to the experiment index](Experiment-trajectory)

## The EuroSAT probe, second pass

Notebook 35 was rebuilt and rerun after the [28 October record](Experiment-trajectory-2026-10-28) found that its first pass chose the probe learning rate by test accuracy, which the [22 October plan](Experiment-trajectory-2026-10-22) says to stop doing.

Commit `bbcfb28`. `uncle/probe.py` gained a `validation_set` argument and reports its accuracy beside the test accuracy.

### What changed between the two passes

Four things changed, not the three that were planned.

| Setting | First pass | Second pass |
| --- | --- | --- |
| Learning rate chosen on | Test accuracy, ties on test loss | Validation accuracy, ties on the lower rate |
| Repetitions at t = 0 and t = 8 | 5 and 2 | 11 and 11 |
| Arms | full and code-only | full only |
| Training images per probe task | 5,000 | 800 |
| Test images, EuroSAT | 500 | 2,000 |
| Validation images | none | 200 |

The split sizes were changed as well, to the sizes the 22 October plan asks for. That was not among the three changes requested, and it is the reason the two passes cannot be compared to find a cause. The section "What this run cannot say" below sets out the consequence.

Seed 0 only, order 01, probe points t = 0 and t = 8, three learning rates, 132 runs.

### Result

| Probe task | t = 0 | t = 8 | Change | Noise floor | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| EuroSAT | 58.18 | 56.09 | -2.09 | 5.67 | Inside the floor |
| Tiny ImageNet task 2 | 50.42 | 51.60 | +1.18 | 2.89 | Inside the floor |

Accuracies are percentages. Full arm, seed 0.

**Still no verdict on the main question.** Neither change clears its own floor.

### The floors came down as intended, and so did the spread

This is the part that worked.

| Probe task | Floor, first pass | Floor, second pass | Repetition spread, first pass | Repetition spread, second pass |
| --- | ---: | ---: | ---: | ---: |
| EuroSAT | 14.66 | 5.67 | 8.94 | 6.78 |
| Tiny ImageNet task 2 | 6.44 | 2.89 | 3.93 | 3.46 |

The spreads are worked back from the floors, using `d = 1.96 * s * sqrt(1/n_0 + 1/n_t)`.

Raising the repetitions from five and two to eleven and eleven accounts for most of the drop, because the floor falls as the square root of the count. The spread itself also fell, by 24 percent on EuroSAT and 12 percent on task 2.

An earlier version of this page credited that spread drop to the new selection rule, reasoning that choosing on validation removes a maximum taken over three noisy numbers. The re-scoring below shows that was wrong. Inside this pass the two rules give spreads of 3.46 and 3.48 on task 2, and 6.78 either way on EuroSAT. The selection rule does not move the spread here. The drop belongs to the split change along with everything else.

### The measured changes collapsed

| Probe task | Change, first pass | Change, second pass |
| --- | ---: | ---: |
| EuroSAT | +7.76 | -2.09 |
| Tiny ImageNet task 2 | +5.76 | +1.18 |

EuroSAT changed sign. The Tiny ImageNet gain fell to about a fifth of what it was.

The absolute accuracies moved too. EuroSAT at t = 0 rose slightly, from 55.84 to 58.18, despite training on 800 images instead of 5,000. Task 2 at t = 0 fell from 56.64 to 50.42.

### What this run does settle

**A tighter bound on the change.** Whatever the change is at eight tasks on these probes, it is smaller than 5.67 points on EuroSAT and smaller than 2.89 points on Tiny ImageNet task 2. The first pass could only bound it at 14.66 and 6.44. That is the first real narrowing this line has produced.

**No plasticity loss shows at eight tasks.** Nothing fell past a floor. That part of the project's reading is unaffected by this run, and it is the part that bears on the paper's saturation claim.

**The gain is the part now in doubt.** The 17 and 21 October runs reported a rise of +6.62 on one seed and +6.37, +3.01 and +4.86 on three. The same comparison here, on a single task, gives +1.18, inside its floor. This does not withdraw the earlier finding, because the two protocols differ in several ways. It does mean the earlier finding is not safe to state until one of them is shown to be the cause.

### What this run cannot say

It cannot say why the change collapsed. Two causes are available and this run changed both at once.

The rate selection may have been inflating the earlier numbers. Taking the best of three rates on the test score is a maximum over noise, and if it falls on the two probe points unevenly it can manufacture a difference.

Or the training split may be the cause. Dropping from 5,000 training images to 800 is a large change in regime. At 250 steps and batch 64 the probe now makes about 20 passes over 800 images, where before it made about 3 passes over 5,000. A model late in the stream may benefit from the larger set differently than a fresh one.

Changing both in one pass was an error of sequencing, not of method. Each change was individually justified by the 22 October plan. They should not have gone in together.

### The re-scoring check clears the selection rule

Every run in this pass recorded both its validation accuracy and its test accuracy. So the same 132 runs were scored again under the old rule, choosing the rate by test accuracy and breaking ties on test loss. Nothing was trained. The runs were read from `partial_v2.jsonl` in R2 and scored on a laptop.

| Rule | Probe task | t = 0 | t = 8 | Change | Floor | Spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation, this pass | Tiny ImageNet task 2 | 50.42 | 51.60 | +1.18 | 2.89 | 3.46 |
| Test, the first pass | Tiny ImageNet task 2 | 50.71 | 51.82 | +1.11 | 2.91 | 3.48 |
| Validation, this pass | EuroSAT | 58.18 | 56.09 | -2.09 | 5.67 | 6.78 |
| Test, the first pass | EuroSAT | 58.18 | 57.23 | -0.95 | 5.67 | 6.78 |

The two rules disagree on which rate to pick in only 3 of 22 cases per probe task, and the resulting numbers differ by 0.07 points on task 2 and 1.14 on EuroSAT, both far inside the floor.

**So the selection rule is not what inflated the earlier numbers.** Scored the old way, this pass still gives +1.11 and -0.95, not the +5.76 and +7.76 the first pass reported. One candidate cause is removed.

Two candidates remain for task 2, whose test split did not change between the passes. The training split fell from 5,000 images to 800. And the repetitions at t = 8 rose from two to eleven, so the first pass's t = 8 mean came from two numbers. A mean of two draws from a spread of about 3.9 points moves easily by several points, which is exactly why that reading was recorded as inside its floor at the time.

That second possibility deserves saying plainly. The first pass's +5.76 was never a finding. It was reported as unresolved, and the honest reading of the two passes together is that it was noise in a two-sample mean, now shrunk toward +1 by nine more repetitions. No artifact is needed to explain it.

The 17 and 21 October runs are a separate matter, because they also averaged three probe tasks and trained on 5,000 images. The selection rule is cleared for them too, but the other differences are not.

## The training split, isolated

Commit `c62424e`. Notebook 35 with one change from the second pass and nothing else: the training split back to 4,800 images, which is the 5,000 of the earlier runs with 200 held out for validation. EuroSAT switched off behind `INCLUDE_EUROSAT`. 66 runs, seed 0, order 01, full arm, Tiny ImageNet task 2 only.

Finished runs are saved under `RUN_LABEL`, here `pass3_train4800`, so the three protocols cannot pool.

### Result

| Probe task | t = 0 | t = 8 | Change | Noise floor | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Tiny ImageNet task 2 | 57.47 | 59.51 | +2.04 | 2.63 | Inside the floor |

### The three task 2 readings together

| Pass | Training images | Repetitions | t = 0 | t = 8 | Change | Floor | Spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First, 28 Oct | 5,000 | 5 and 2 | 56.64 | 62.40 | +5.76 | 6.44 | 3.93 |
| Second, 29 Oct | 800 | 11 and 11 | 50.42 | 51.60 | +1.18 | 2.89 | 3.46 |
| Third, 29 Oct | 4,800 | 11 and 11 | 57.47 | 59.51 | +2.04 | 2.63 | 3.15 |

Spreads are worked back from the floors.

**The training split controls the level, not the gain.** Restoring the images lifted t = 0 by 7.05 points and t = 8 by 7.91, so the earlier accuracy range is back and the protocol is comparable with the first pass again. The gap between the two points moved only from +1.18 to +2.04. That is nowhere near the +5.76 the first pass reported.

**The split is cleared, and so the repetition count is what is left.** The first pass differed from the third in three ways. The rate rule was ruled out by the re-scoring check above. The training split is ruled out here. The only remaining difference is that the first pass took two repetitions at t = 8 against eleven.

The arithmetic supports that. With a spread near 3.9 and two repetitions at t = 8 against five at t = 0, the uncertainty on one difference is about 3.3 points. Reading +5.76 when the underlying value is near +2.0 is a gap of a little over one standard error. That is an ordinary draw.

### What this withdraws, and what it does not

**The large gain is withdrawn as an artifact of too few repetitions.** The readings of 17 October, +6.62 against a floor of 4.85, and of 20 and 21 October, +6.37, +3.01 and +4.86, all used five repetitions at t = 0 and two at the later point. Three passes at eleven repetitions now put the change at +1.18, +2.04 and, on EuroSAT, -2.09. None of these clears its floor.

This is not a claim that those runs were wrong. They were measured correctly and reported as inside or near their floors at the time. It is a claim that the number they produced was imprecise, and that the imprecision was large enough to account for the whole apparent gain.

**No plasticity loss is withdrawn.** Nothing in any pass has fallen past a floor. The reading that bears on the paper's saturation claim is unaffected.

**The 21 October E1 result is untouched by this.** The random-label measurement fell by 0.70, 0.76 and 0.64 against floors of 0.29, 0.42 and 0.18, and all three cleared. That is a separate endpoint with a much smaller spread, and nothing here bears on it.

### How close this is to resolving

The change is +2.04 against a floor of 2.63. At the measured spread of 3.15, nineteen repetitions at each point would bring the floor to about 2.00.

Notebook 35 resumes by run identity, and the identity carries the repetition number. So raising the repetitions from eleven to nineteen skips the 66 runs already finished and adds 48 new ones, about 30 minutes, writing to the same saved file.

That would be optional stopping: the count would be raised after seeing the result, which biases toward a finding. If it is run, the record must say so and the reading is weaker than a pre-planned one.

The alternative avoids the bias. Run seeds 1 and 2 at the settings used here, unchanged. Three independent seeds agreeing on a small positive change would be stronger evidence than one seed pushed to more repetitions, and it matches what the 20 and 21 October run did. 132 runs, about 80 minutes.

## Next decision

The decisive test has now run and the collapse is explained. The remaining choice is how to finish the task 2 reading.

Preferred: seeds 1 and 2 at exactly the settings of the third pass, 4,800 training images, eleven repetitions, validation selection, full arm, task 2 only. 132 runs, about 80 minutes. Three seeds at a small positive change would be a result worth reporting, and the protocol is fixed in advance.

Cheaper but weaker: raise seed 0 to nineteen repetitions, 48 new runs and about 30 minutes, resuming into the same file. This is optional stopping and must be recorded as such.

After either, EuroSAT should be rerun at the settled protocol with enough repetitions to clear its wider floor. EuroSAT is the probe that separates transfer from plasticity, and it is the number the write-up needs. Its spread of 6.78 means it needs roughly four times the repetitions of task 2 for the same resolution.
