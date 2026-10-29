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

## Next decision

Do not start seeds 1 and 2 on this protocol. Seed 0 did not resolve, and two more seeds would measure the same unexplained mixture.

The decisive test is cheap and isolates one variable. Run Tiny ImageNet task 2 alone, validation selection kept, eleven repetitions at both points, and put the training split back to 5,000 images. If the change returns to about +5, the training split is the cause and the earlier gains are real at that split. If it stays near +1, the remaining difference is the probe-task set, and the 21 October gain rests on averaging three tasks. Roughly 66 runs, about 40 minutes.
