# Experiment trajectory: 28 October 2026

[Back to the experiment index](Experiment-trajectory)

## A defect: the two stopping points were the same model

Notebook 24 has offered two unlearning stopping points, 6 steps and 100 steps, since 26 October. It did not have two. Every arm at both settings was unlearned for 100 steps.

`run_request` in `uncle/study_training.py` takes the step count from the configuration, at line 188.

```
steps = config.burn_in
```

`forget_interval`, which the notebook passed the stopping point to, controls only how often the boundary callback fires, at line 209.

```
if step % forget_interval == 0 or step == steps:
```

The notebook built its configuration with `"burn_in": max(UNLEARN_BUDGETS)`, which is 100 whichever stopping point is being run. So the stopping point changed the save frequency and nothing else.

### How the seed 1 run exposed it

In the seed 1 output, every arm that keeps task 14's own code produces an identical trace at both stopping points. Identical accuracies, identical class counts, and identical input sensitivities to four significant figures, at all eleven measured steps, in all three image budgets.

The arms that use a brand new code do differ between the two settings. That is not evidence against the defect. Those arms name their fresh code after the stopping point, so the code itself differs, which is enough to change the trajectory even from an identical model.

Two conditions that were supposed to start from models 49 degrees apart cannot agree to four significant figures. They agree because they were the same model.

### What this invalidates

**The stopping-point comparison, wherever it appears.** The 26 and 27 October records report that the stopping point makes no difference, quoting gaps of 4.5, 1.1 and 0.6 points. There was one stopping point, so those gaps measure run-to-run noise.

**The step-6 collapse figure from notebook 24.** The 27 October record states that task 14 was at exactly 10.00 percent on one of ten classes after 6 unlearning steps. That measurement was taken after 100 steps. The claim that a strong task collapses by step 6 is not supported by this notebook.

**Notebook 34.** It copies the same pattern, so its two stopping points were also one. The 27 October reading, that identical before and after numbers at both settings showed the extra 94 steps changing nothing, is withdrawn. They were the same run.

### What survives

The collapse itself. Task 14 reaches exactly 10.00 percent on one of ten classes with near-zero input sensitivity. Only the step count attached to it is wrong.

The dead-code split, the recovery result and the harder-than-unseen result. None of these compares stopping points. They compare arms within one unlearning setting, and that setting was the same for every arm.

The direction measurement from 23 October, which used notebooks 30 and 31 rather than this path.

### The fix

Both notebooks now take the stopping point as an argument to `build` and pass it as `burn_in`. The step count is therefore the stopping point.

This defect is a likely explanation for the step-count inconsistency recorded earlier between notebook 24 and notebooks 30 and 31, which had been left open.

## Seed 1 of the recovery probe, incomplete

Notebook 24 with `SEED = 1`, everything else as the seed 0 run: study `paired_generalization_v2`, order 01, `FORGET_TASK = "14"`, image budgets 1, 5 and 20 per class, 100 recovery steps measured every 10, generator trainable, unseen controls 2, 4 and 6.

The run reached 40 of 42 cells. The last two unseen-task cells at the second setting are missing. Read every stopping-point label below as 100 unlearning steps, for the reason above.

Task 14 stood at 40.4 percent before unlearning, with full input sensitivity and all ten classes in use. After unlearning it was at exactly 10.00 percent, on one of ten classes, with input sensitivity 2.4e-07.

### The dead code replicates, and wider

First step at which each arm responds to its input, taking input sensitivity above 0.1.

| Arm | Task code | 1 per class | 5 per class | 20 per class |
| --- | --- | ---: | ---: | ---: |
| attacker | task 14's own | 50 | 50 | 60 |
| code only | task 14's own | 50 | 50 | 50 |
| stats only | brand new | 0 | 0 | 0 |
| fresh start | brand new | 0 | 0 | 0 |

Seed 0 gave 38 steps against 2. Seed 1 gives 50 to 60 against 0. The split holds and is wider here. This is now four runs across two tasks and two seeds.

### Still no recovery advantage, and a larger loss at the useful budget

Peak accuracy reached by each arm.

| Images per class | attacker | code only | stats only | fresh start |
| --- | ---: | ---: | ---: | ---: |
| 1 | 14.8 | 14.8 | 17.4 | 17.6 |
| 5 | 19.0 | 19.0 | 20.8 | 20.0 |
| 20 | 16.0 | 15.8 | 29.2 | 31.4 |

The attacker is behind the fresh start by 2.8, 1.0 and 15.4 points. Seed 0 had the attacker marginally ahead, by 1.9 points, inside the tie threshold. Seed 1 puts it clearly behind.

The 20-image gap is the dead time showing up as lost budget. An arm that spends 60 of its 100 steps unable to respond has 40 steps to learn in, against 100 for an arm with a new code.

### The forgotten task is harder than tasks never seen

At the 20-image budget the unseen controls reached 43.2, 33.6 and 35.4 percent, a mean of 37.4. Task 14 from a fresh code reached 31.4.

So task 14 is about 6 points harder than a task the model never saw. Seed 0 gave 5.0 and 5.6 points in the same direction. This replicates.

### The 20-image candidate signal does not replicate

On seed 0 every arm beat the fresh start at the 20-image budget, by 1.8 to 5.0 points, which was recorded as a candidate signal needing seeds. On seed 1 the fresh start is the best arm at that budget by a wide margin. The signal does not survive and should not be reported.

### The pay-back observation is not readable here

Seed 0 showed the old-code arms reaching halfway back in 20 steps once awake, against 40 to 60 for a new code. On seed 1 the old-code arms never reach a comparable level at the 20-image budget, so there is no halfway point to compare. It remains unsettled.

### One measurement to stop using

Input sensitivity at step 0 for a brand new code is not usable. Across these runs it ranges from 6.0e-03 to 1.27e+09, which is eleven orders of magnitude. The wake step is meaningful for the arms carrying the forgotten task's own code, where it is consistent, and not for the arms with a fresh code.

### An instability worth noting

The attacker at the 20-image budget peaked at 16.0 percent at step 90 and fell to 11.4 at step 100, while its input sensitivity rose to 5.1e+01. It is diverging rather than converging. The 20-image attacker cells are the noisiest in this experiment.

### Runtime

About eleven hours for 40 cells, against an estimate of half an hour given in conversation. That estimate was wrong and the seed 2 run should be planned as an overnight job.

## Next decision

Seed 2 should not start until the stopping-point fix is in the notebook, so that its two settings are genuinely 6 and 100 steps.

The stopping-point question is now unanswered rather than answered, and it is the cheapest thing the fixed notebook buys back.

The step-6 collapse claim needs its own measurement from a path that does not use `run_request`, or a short run of the fixed notebook at a 6-step setting.
