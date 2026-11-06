# Experiment trajectory: 5 November 2026

[Back to the experiment index](Experiment-trajectory)

## New-task learning speed, read out of a run that already existed

This was the last indicator in the plan with no reading, and the [3 November
table](Experiment-trajectory-2026-11-03) listed it as "not measured". That was wrong. No new
run was needed. The measurement was already in a file on R2, unread.

### Why it looked unmeasured

Loss progress cannot be read in this design, for a reason that is in the 3 November record: it
is taken on whatever task is being learned at that moment, and the eight tasks differ, so a
change mixes the task with the position in the stream. The fix is to hold one task out of the
stream and train on that same task at every boundary, so the only thing that changes between
readings is the model.

That is exactly what notebook 40 does. At each of t = 0, 2, 4, 6 and 8 it loads the saved
model, forks a copy, trains the copy on held-out Tiny ImageNet task 2 for 250 steps, scores it,
and throws the copy away. Same task, same budget, same three learning rates, eleven repetitions,
seeds 1 and 2.

The part that was not known is that `probe_once` also saves the shape of each probe, not only
its end. It records four readings per run, at steps 62, 125, 188 and 250, each carrying the
training loss and the test accuracy at that point. All 330 runs carry them. They sit in
`uncle/matched_probe/indicator_stream_v1/order_01/curve1.jsonl`.

So the learning-speed reading is arithmetic on a file that already exists. No GPU time, no
model downloads, no new run.

### What it says

One run is selected per repetition by validation accuracy, ties going to the lower rate, which
is notebook 40's own rule. The curve reported is the curve of that selected run.

Test accuracy on the held-out task, part way through learning it, mean over eleven repetitions.

| Seed | t | step 62 | step 125 | step 188 | step 250 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 29.36 | 42.58 | 49.09 | 55.67 |
| 1 | 2 | 40.13 | 48.36 | 54.05 | 57.75 |
| 1 | 4 | 45.33 | 48.40 | 55.80 | 59.25 |
| 1 | 6 | 43.82 | 48.64 | 55.33 | 59.64 |
| 1 | 8 | 47.53 | 48.45 | 57.02 | 62.42 |
| 2 | 0 | 32.05 | 43.13 | 51.76 | 56.71 |
| 2 | 2 | 42.40 | 49.75 | 55.15 | 57.65 |
| 2 | 4 | 41.76 | 46.93 | 54.75 | 60.44 |
| 2 | 6 | 40.67 | 52.29 | 54.73 | 59.75 |
| 2 | 8 | 39.76 | 47.18 | 56.16 | 58.29 |

The clearest single number is how long the probe takes to reach half of what a fresh model
reaches in 250 steps. Fewer steps means faster learning.

| t | Seed 1 | Seed 2 |
| ---: | ---: | ---: |
| 0 | 61.2 | 55.6 |
| 2 | 43.7 | 41.9 |
| 4 | 38.3 | 43.1 |
| 6 | 39.6 | 44.3 |
| 8 | 36.5 | 45.1 |

Both seeds get faster, not slower. Seed 1 goes from 61.2 steps to 36.5, with repetition spreads
of 13.1 and 2.5, so the move is several times its own spread. Seed 2 goes from 55.6 to 45.1,
spreads 6.5 and 6.9, giving a both-ends floor of 5.6 against a change of 10.5, so it clears but
by less than twice.

Compared at the first checkpoint, where a speed difference shows most, the end of the stream is
ahead of the start by 18.16 points on seed 1 and 7.71 on seed 2, against floors of 3.81 and
3.90. Both clear.

### What this does not establish

The first checkpoint is at step 62, so the steps-to-half figures interpolate inside a 62-step
gap. The direction is safe; the exact step counts are not. Reading speed properly would want
checkpoints every ten steps or so, which is a `checkpoints` setting on `probe_once` and a rerun,
not a design change.

The selected learning rate does not drift along the stream. It is 1e-4 on 7 to 10 of the eleven
repetitions at every point on both seeds, and 3e-4 otherwise. So the speed-up is not a rate
artifact.

Seed 0 is absent, because its stream ran before the code that keeps a model at the middle
boundaries. This is two seeds, not three.

The probe task comes from the same dataset as the stream, which is a limit this project has
carried since the probe was built: it cannot separate a genuine capacity change from ordinary
transfer between similar classes. A faster start late in the stream is exactly what transfer
would also look like. What transfer does not explain is the absence of any slowing, which is
what the indicator was being asked about.

### Where it leaves the split

Four measurements now train something and count the result: the direct probe, random-label
fitting, curvature, and new-task learning speed. All four say capacity held. Three read the
weights or the activations and infer: effective rank, srank and dormant units. All three say it
was lost. No indicator in the plan is now unread except loss progress, which this reading
replaces rather than rescues.
