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

Two separate things pulled the floor down. Raising the repetitions from five and two to eleven and eleven accounts for most of it, because the floor falls as the square root of the count. The rest is the spread itself falling, by 24 percent on EuroSAT and 12 percent on task 2. Choosing the rate on validation instead of on test removes a maximum taken over three noisy numbers, and that is the expected effect.

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

**The gain is the part now in doubt.** The 17 and 21 October runs reported a rise of +6.62 on one seed and +6.37, +3.01 and +4.86 on three. Both used test-based rate selection. Under validation-based selection the same comparison on a single task gives +1.18, inside its floor. This does not withdraw the earlier finding, because the two protocols differ in more than the selection rule. It does mean the earlier finding is no longer safe to state without a check.

### What this run cannot say

It cannot say why the change collapsed. Two causes are available and this run changed both at once.

The rate selection may have been inflating the earlier numbers. Taking the best of three rates on the test score is a maximum over noise, and if it falls on the two probe points unevenly it can manufacture a difference.

Or the training split may be the cause. Dropping from 5,000 training images to 800 is a large change in regime. At 250 steps and batch 64 the probe now makes about 20 passes over 800 images, where before it made about 3 passes over 5,000. A model late in the stream may benefit from the larger set differently than a fresh one.

Changing both in one pass was an error of sequencing, not of method. Each change was individually justified by the 22 October plan. They should not have gone in together.

### The check that costs nothing

Both accuracies were recorded for every run in this pass, the validation one and the test one. So the same 132 runs can be scored again under the old rule, choosing the rate by test accuracy, with no new training.

If re-scoring returns a large positive change, the selection rule was the cause and the earlier gains were an artifact. If it returns something close to +1.18 and -2.09, the smaller training split was the cause, and the selection rule is cleared.

This is the next thing to run, and it needs no GPU.

## Next decision

Run the free re-scoring check before anything else. It decides whether the 21 October headline stands, and it costs one cell.

Do not start seeds 1 and 2 on this protocol yet. Seed 0 did not resolve, and until the cause of the collapse is known, two more seeds would measure an unknown mixture.

If the re-scoring clears the selection rule, the open question becomes the training-split size, and the cheap test is one pass on Tiny ImageNet task 2 alone at the old 5,000-image split, validation selection kept, eleven repetitions. Roughly 66 runs.
