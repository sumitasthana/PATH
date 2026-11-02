# Experiment trajectory: 2 November 2026

[Back to the experiment index](Experiment-trajectory)

## Reading the first indicator run

Notebook 37 on CPU, reading the record that notebook 36 wrote on 1 November. Study
`indicator_stream_v1`, order 01, seed 0. 233 rows, nine boundaries, one run.

The provenance check passes: `images` is 256 everywhere and `units` is 22,720 everywhere, so
every boundary measured the same number of images through the same number of ReLU positions.

### The headline, and why it is not what it looks like

**The representation narrows on data the model never trains on, by the same proportion as on
data it does.** That removes the explanation offered on 1 November, where the falling rank on
`in_stream` could have been the model tuning itself to the pictures it kept seeing. `probe` and
`eurosat` are fixed throughout and were never trained on, and they fall the same way.

| Image set | Effective rank at t = 0 | At t = 8 | Change | Fall |
| --- | ---: | ---: | ---: | ---: |
| `in_stream` | 7.4249 | 4.0169 | -3.4079 | 46 percent |
| `probe` | 7.3057 | 4.0473 | -3.2584 | 45 percent |
| `eurosat` | 4.0041 | 2.1648 | -1.8393 | 46 percent |

Three image sets, one of them from a different dataset entirely, all falling by 45 to 46
percent. The two Tiny ImageNet sets track each other closely at both ends, and EuroSAT sits
lower throughout, which is the domain gap rather than a different behaviour.

srank falls as well, and it separates the two datasets: `in_stream` 226 to 200 and `probe` 227
to 202, both about 11 percent, against `eurosat` 170 to 115, which is 32 percent. So the
unrelated dataset loses a larger share of the directions it was using.

Dormant units rise everywhere, by about the same amount on all three sets.

| Threshold | `in_stream` | `probe` | `eurosat` |
| --- | --- | --- | --- |
| 0.01 | 0.1639 to 0.2299 | 0.1639 to 0.2299 | 0.1790 to 0.2384 |
| 0.025 | 0.2002 to 0.2668 | 0.2002 to 0.2669 | 0.2135 to 0.2743 |
| 0.1 | 0.2901 to 0.3630 | 0.2900 to 0.3626 | 0.3008 to 0.3685 |

About a sixteenth of all units went quiet over the stream at every threshold. The two Tiny
ImageNet sets agree to three decimal places, which is close enough to be worth checking against
a second seed, though the sets are built from disjoint tasks and the statistic is coarse.

### What this contradicts

The direct plasticity probe, recorded on [29 October](Experiment-trajectory-2026-10-29), teaches
a fresh code on **Tiny ImageNet task 2** at t = 0 and t = 8 and measures how well it learns.
Task 2 is the same task as the `probe` image set here. Across three seeds it reads +3.10 points
with a 95 percent interval of +0.32 to +5.88, which excludes zero. Learning a new task is
slightly **easier** after eight tasks.

So on the same held-out task, in the same stream, two measurements disagree:

| Measurement | What it says about task 2 after eight tasks |
| --- | --- |
| Effective rank, this run | The representation uses 45 percent fewer directions |
| Dormant units, this run | 6.6 more units in every hundred have gone quiet |
| Direct probe, 29 October | A fresh code learns the task 3.10 points better |

**This is the most useful thing the indicators have produced, and it is not a plasticity
finding.** It is a finding about the indicators. The indicators read the representation the
generator offers through one fixed reference code that belongs to no task. The probe trains a
new code. A narrowed representation at a neutral code does not stop the generator from
re-expanding when a code is trained for the job.

That matters for the critique. The adapter plasticity study treats falling rank and rising
dormancy as evidence of lost plasticity, and the UnCLe paper claims unlearning restores
plasticity. Here the proxies move hard in the direction that would be called plasticity loss,
while the thing they are proxies for moves the other way. Taken at face value the proxies would
have produced a confident wrong answer.

### The weights turned, and the generated weights turned much further

The angle was not recorded by this run, which loaded its code before commit `5ac79bf`. For the
whole stream, start to end, it can be recovered from the recorded norms by the law of cosines.
The recovery was ruled out on 1 November for the task-by-task steps, where the angle is near
5e-5 of a cosine and the method's error is the same size. End to end the move is about two
thousand times larger, so the same arithmetic is sound here. Rounding the printed values to
four places leaves a band 0.01 degrees wide.

| | Norm at t = 0 | At t = 8 | Growth | `delta_w` | Angle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generator | 479.5684 | 483.4920 | +0.82 percent | 0.1196 | 6.81 degrees |
| Generated target | 273.3581 | 293.8820 | +7.51 percent | 0.4095 | 22.39 degrees |

Pure growth with no turn at all would have given `delta_w` of 0.0082 and 0.0751. The measured
values are 15 and 5 times larger, so almost all of both moves is a turn, not growth. This is
the distinction the angle was added for, and it holds without it.

The generated weights move much further than the generator that writes them: 0.4095 against
0.1196, and 22.4 degrees against 6.8. A small change in the generator is amplified into a large
change in what it produces.

### One measurement is not readable from this run

`loss_progress_mean` rose from 0.1397 at t = 1 to 0.2160 at t = 8. Higher means one update moves
the loss more, which is the opposite of learning getting harder. It is not interpretable here.
It is measured on the task being learned, and the eight tasks are different tasks with different
starting losses, so the change confounds the task with the stream position. Its endpoints are
also t = 1 and t = 8 rather than t = 0 and t = 8, because there is no update to sample before
training starts.

### What this run still cannot say

- **One seed.** Every spread column is empty. No change here clears a noise floor, because
  there is no floor. The three image sets agreeing is internal corroboration, not a substitute.
- **No control for movement alone.** Nothing here shows whether a generator moved the same
  distance by any means would narrow the same way. A random-direction control would settle it
  and has not been run.
- **The contradiction rests on one seed against three.** The probe result has three seeds; this
  has one. Seeds 1 and 2 are needed before the disagreement is stated as a result.

## Next decision

1. Run seeds 1 and 2 of `indicator_stream_v1`, about three hours of A100. They record the angle
   directly and they keep a model at boundaries 2, 4 and 6, added in commit `9b746a4`.
2. With three seeds, decide whether the rank and dormancy changes clear their own floor, and
   whether the disagreement with the direct probe survives.
3. Seed 0 kept no mid-stream models, because its job finished before `9b746a4`. Probing it at
   two, four and six tasks would need a fresh run under a new study name.
