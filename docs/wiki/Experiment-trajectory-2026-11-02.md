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
percent. **Seed 1, below, does not reproduce this agreement**; its three sets fall by 35, 35 and
15 percent. The two Tiny ImageNet sets track each other closely at both ends, and EuroSAT sits
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
ImageNet sets agree to three decimal places. Seed 1 agrees with seed 0 on dormancy, so this is
the one indicator here that reproduced.

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

## The second indicator run, seed 1

Notebook 36, same study, same order, `SEED` changed to 1. Started 2 November at 16:01 and
completed at 18:16, runtime clock. Two hours 15 minutes on an NVIDIA A100-SXM4-40GB, against one
hour 22 minutes for seed 0. The per-task times are 13:21, 15:57, 21:45, 16:02, 16:23, 15:30,
17:03 and 16:30, so the whole run was slower rather than one task stalling. Nothing about the run
differed except the seed, so this is the machine, not the method.

Status complete, nine boundaries recorded. This run carried the angle, and it kept a model at
boundaries 2, 4 and 6, the first use of the code added in `9b746a4`.

| Position in the order | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task | 3 | 0 | 9 | 5 | 17 | 1 | 7 | 14 |
| Accuracy, percent | 34.8 | 42.6 | 48.8 | 32.8 | 50.2 | 54.4 | 44.0 | 45.8 |

Mean 44.17 percent, against 39.20 for seed 0.

### The starting level is set by the initialisation, not by anything learned

Boundary zero is a fresh generator that has seen no data. The two seeds disagree there by a large
amount.

| Image set | Effective rank at t = 0, seed 0 | Seed 1 |
| --- | ---: | ---: |
| `in_stream` | 7.4249 | 4.5626 |
| `probe` | 7.3057 | 4.5691 |
| `eurosat` | 4.0041 | 2.6302 |

Seed 0 starts 63 percent higher on the two Tiny ImageNet sets. No training has happened. So the
absolute value of effective rank at the reference code is a property of the random
initialisation. A single number such as "effective rank 4.0" carries no meaning on its own, and
two studies that report different levels may differ only in their seeds.

### The size of the fall is not stable either

| Image set | Seed 0, t = 0 to t = 8 | Seed 1 |
| --- | ---: | ---: |
| `in_stream` | 7.4249 to 4.0169, down 45.9 percent | 4.5626 to 2.9708, down 34.9 percent |
| `probe` | 7.3057 to 4.0473, down 44.6 percent | 4.5691 to 2.9684, down 35.0 percent |
| `eurosat` | 4.0041 to 2.1648, down 45.9 percent | 2.6302 to 2.2345, down 15.0 percent |

Both seeds fall. The direction holds. The size does not: 46 against 35 percent on `in_stream`,
and 46 against 15 percent on EuroSAT.

**This removes a claim made earlier on this page.** The three image sets of seed 0 all fell by 45
to 46 percent, and that agreement was called internal corroboration. Seed 1 does not reproduce
it. Its three sets fall by 35, 35 and 15 percent. The two Tiny ImageNet sets still track each
other closely, which is expected because they are built the same way from the same dataset.
EuroSAT does not track them. With one seed the three-way agreement looked like a signal; with two
it looks like a coincidence of seed 0.

Seed 1 is also not monotone. Its `in_stream` rank goes 4.56, 3.38, 4.18, 3.63, 3.34, 3.27, 3.16,
2.97, 2.97. It falls on the first task, rises above its own second reading on the third, then
falls again. Seed 0 falls almost monotonically. So the shape of the curve is not stable across
seeds either.

### Almost all of the movement happens on the first task

| Measurement | Share of the whole stream's change that happens on task 1 |
| --- | ---: |
| Dormant units at 0.01, `in_stream`, seed 0 | 80 percent |
| Dormant units at 0.01, `in_stream`, seed 1 | 83 percent |
| Angle of the generated target weights, seed 1 | 90 percent |
| Effective rank, `in_stream`, seed 0 | 34 percent |
| Effective rank, `in_stream`, seed 1 | 74 percent |

Dormancy rises from about 0.16 to about 0.22 on the first task and then sits nearly flat for the
remaining seven. The generated weights turn 19.32 of their 21.56 degrees on the first task and
then barely turn again.

The generator itself behaves differently. Its angle grows steadily across the whole stream: 1.83,
3.15, 4.02, 4.72, 5.43, 6.03, 6.47, 6.85 degrees. So the generator keeps moving while what it
writes at the reference code stops moving. That is new and was not visible before the angle was
recorded.

A reasonable reading is that the first task moves the generator out of its random starting region
and into a region where it writes trained-looking weights, and that most of what these indicators
report is that one event. Later tasks move the generator further without changing much of what it
produces at a neutral code. This is an interpretation of two runs, not a result.

### Lower rank came with better accuracy

Seed 1 ends with `in_stream` effective rank 2.97 and mean accuracy 44.17 percent. Seed 0 ends with
rank 4.02 and mean accuracy 39.20 percent. The seed with less of the quantity being read as
capacity performed better.

Two points are not a trend, and the two runs differ in many ways at once. But this is the only
direct comparison available between the proxy and performance, and it points the opposite way to
the reading the proxy is usually given.

### The angle recovery from the norms is correct

Seed 0 has no angle column, so its start-to-end angle was recovered from its recorded norms by the
law of cosines. Seed 1 records both, which allows the method to be checked against the truth on
the same run.

| | Recovered from the norms | Measured directly | Error |
| --- | ---: | ---: | ---: |
| Generator | 6.835 degrees | 6.852 degrees | 0.25 percent |
| Generated target | 21.542 degrees | 21.559 degrees | 0.08 percent |

The recovery is sound end to end, as argued on this page, and the recovered figures for seed 0,
6.81 and 22.39 degrees, stand. The same recovery remains unusable for the task-by-task steps,
where the signal and the method's error are the same size.

### What the two runs together still cannot say

- **Two seeds, not three.** Seed 2 has not been run. A spread from two points is weak.
- **Still no control for movement alone.** Nothing yet shows whether a generator moved the same
  distance by any means would narrow the same way.
- **The disagreement with the direct probe is untested on this seed.** The probe has not been run
  on seed 1. The boundary models at 2, 4 and 6 now exist for it.
- **The slower runtime is unexplained.** It has no effect on the numbers, but it is unaccounted
  for.

## Next decision

1. Run seed 2 of `indicator_stream_v1`, about two hours of A100. Three seeds is the minimum for a
   spread, and seeds 0 and 1 already disagree enough that the third will decide whether anything
   except the direction of the fall survives.
2. Run the direct plasticity probe on seed 1 at boundaries 2, 4 and 6, using the models this run
   kept. That is the first test of whether the proxies and the probe disagree in the middle of a
   stream as well as at its ends, and it needs no new stream run.
3. Report the indicators as a change from boundary one, not boundary zero, or report both. Most of
   what boundary zero contributes is the random initialisation and the one-off move away from it.

## Seed 2 attempt: stopped by a stalled upload

Two faults showed up before any result did.

**The dataset path was wrong in notebook 36.** `DATA_ROOT` pointed at `/content/data`, but the
downloader only fetches Tiny ImageNet when the folder itself is named `tiny-imagenet-200`, because
it unzips into the parent. Every other notebook uses the full path; this one had drifted. Fixed in
the notebook.

**The run then died on a read timeout to R2.** Settings as intended: seed 2, order 01, eight
tasks, 240 minute budget with a 15 minute reserve, on an A100 40GB. Boundary 0 was measured at
20:41:42 on the fresh generator. Saves at 20:43:57, 20:46:26 and 20:48:41 all reported "still on
task 1 of 8". About eight and a half minutes in, during task 1, a save raised
`ReadTimeoutError: Read timeout on endpoint URL`.

The upload itself had gone through. The failure was in reading the file back to check it. The
retry settings on the R2 client cover sending a request, not the body that arrives afterwards, so
one stalled read ended the session. The record kept one boundary and no boundary models.

Fixed in `uncle/storage.py`: the read-back now starts again, up to three times, before giving up.
Reading changes nothing, so a repeat is safe. `tests/test_storage.py` covers it.

No seed 2 numbers exist. The run has to be started again from the beginning.
