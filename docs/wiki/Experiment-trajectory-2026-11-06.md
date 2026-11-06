# Experiment trajectory: 6 November 2026

Back to the [index](Experiment-trajectory).

## A second task order, run to separate the task from its position

### What was run

Seed 0 of `order_02`, the same study (`indicator_stream_v1`), the same code, the
same probe task. The only difference from the finished `order_01` runs is the
sequence the eight tasks are learned in. The run completed on an A100 in
2 hours 16 minutes, nine boundaries, and wrote
`uncle/indicators/indicator_stream_v1/order_02/seed_0.jsonl`. Boundary models 2,
4 and 6 were kept. Final accuracies: task 14 36.0, task 0 40.2, task 9 42.6,
task 3 37.6, task 17 45.0, task 1 53.2, task 7 40.2, task 5 57.8.

The reason for the run: loss progress is measured on whatever task is being
learned at the time, so in a single order each task sits at exactly one
position, and a change along the stream could be the model or could be the task.
Between the two orders, tasks 3, 5 and 14 change position while five tasks do
not, so the five fixed tasks are a control and the three moving tasks are the
test.

### Loss progress follows the position, not the task

| Task | Position in `order_01` | `order_01`, three seeds | Position in `order_02` | `order_02`, seed 0 | Change |
| ---: | ---: | --- | ---: | ---: | ---: |
| 14 | 8 | 0.2160, 0.2093, 0.2293 (mean 0.2182) | 1 | 0.1169 | -0.1013 |
| 3 | 1 | 0.1397, 0.1485, 0.1643 (mean 0.1509) | 4 | 0.2777 | +0.1269 |
| 5 | 4 | 0.2755, 0.2521, 0.2205 (mean 0.2493) | 8 | 0.2905 | +0.0412 |
| 0 | 2 | 0.2062, 0.1820, 0.1720 (mean 0.1868) | 2 | 0.2002 | +0.0135 |
| 9 | 3 | 0.2856, 0.2292, 0.2540 (mean 0.2562) | 3 | 0.2668 | +0.0106 |
| 17 | 5 | 0.2774, 0.2942, 0.2557 (mean 0.2758) | 5 | 0.2752 | -0.0006 |
| 1 | 6 | 0.2495, 0.2557, missing (mean 0.2526) | 6 | 0.2396 | -0.0130 |
| 7 | 7 | 0.2932, 0.2420, 0.2020 (mean 0.2457) | 7 | 0.2895 | +0.0438 |

The last five rows are the control. They are the same task at the same position
in both orders, so their changes measure noise. The largest is task 7 at 0.0438.
Tasks 14 and 3 move by 0.1013 and 0.1269, which is 2.3 and 2.9 times that
control band. Task 5 moves by 0.0412, which is inside it and settles nothing.

Read by position instead of by task, the two orders line up:

| Position | `order_01` task and value | `order_02` task and value |
| ---: | --- | --- |
| 1 | task 3, 0.1509 | task 14, 0.1169 |
| 2 | task 0, 0.1868 | task 0, 0.2002 |
| 3 | task 9, 0.2562 | task 9, 0.2668 |
| 4 | task 5, 0.2493 | task 3, 0.2777 |
| 5 | task 17, 0.2758 | task 17, 0.2752 |
| 6 | task 1, 0.2526 | task 1, 0.2396 |
| 7 | task 7, 0.2457 | task 7, 0.2895 |
| 8 | task 14, 0.2182 | task 5, 0.2905 |

Position 1 reads low in both orders, with two different tasks sitting there. The
low first reading is therefore a property of being first, not of task 3. The
earlier reading that loss progress jumps over the first two tasks and then
flattens stands. The fall at position 8 seen in `order_01` does not repeat here:
position 8 reads 0.2182 there and 0.2905 in this run, so that end-of-stream dip
is not established. The third order, recorded below, settles it: there is no dip.

**The verdict on the row does not change.** Loss progress still does not fall
across the stream, which is what the plasticity question asks. It is now readable
rather than confounded, and what it reads is a rise over the first two tasks
followed by a flat stretch.

### The size of the effective-rank fall depends on the task order

This was not the question the run was built for. It is the larger result.

Seed 0 of `order_02` starts from the identical generator as seed 0 of
`order_01`: both read effective rank 7.3057 on the probe images at boundary 0,
before any training. Same seed, same code, same images. Only the order differs.

| Measurement at boundary 8 | `order_01` seed 0 | `order_02` seed 0 |
| --- | ---: | ---: |
| Effective rank, probe images | 4.0473 | 2.0635 |
| Fall from boundary 0 | 44.6 percent | 71.8 percent |
| srank, probe images | 202 | 139 |
| srank fall | 11.0 percent | 38.8 percent |
| Dormant units at 0.025, rise | 33.3 percent | 43.2 percent |
| Generator weight change | 0.1196 | 0.1168 |

The weight change is the same to within 2.3 percent. The generator travelled the
same distance in both runs. The effective-rank fall is 1.6 times larger in one
than the other, and the srank fall is 3.5 times larger.

Most of the gap appears at the first boundary. After one task, `order_01` seed 0
reads 6.2475 and this run reads 2.4727. Learning task 14 first drives the rank
reading down much further than learning task 3 first does, and the stream never
recovers from it.

**What that means for the paper.** The direction of the compression indicators is
order-independent: effective rank and srank fall and dormant units rise in all
four runs. The size is not. A fall quoted as a number, 31 percent in the adapter
arm or 35 to 61 percent across the `order_01` seeds, is partly a fact about which
task came first, not about how much capacity the model lost. Since the weight
movement is the same in both runs, the rank fall is not tracking how far the
model moved.

### What this does not establish

- One seed of `order_02`. The three-seed spread quoted for `order_01` is the
  noise floor used here; this run has no spread of its own.
- Task 5 moved from position 4 to position 8 and changed by less than the
  control band, so it contributes nothing either way.
- Position 8 is still measured by one task per order. Whether the dip seen in
  `order_01` is real needs the third order.
- Why learning task 14 first collapses the rank reading so much further is not
  explained here. Task 14 ends at 36.0 percent accuracy, the lowest of the eight,
  which is consistent with it being a hard task, but nothing here tests that.
- The loss-progress gap at `order_01` seed 2, boundary 6, is still missing and is
  excluded from that task's mean.

### Where it left the decision

`order_03` puts task 5 at position 1, task 14 at position 4 and task 3 at
position 8. It was worth more after this run than before it, for two reasons
rather than one. It gives position 1 a third task, which turns two agreeing
readings into three. And it gives a second measurement of how much the
effective-rank fall varies with the order, which at that point rested on a single
pair. It was run the same day, and the result is below.

## The third task order, which completes the rotation

### What was run

Seed 0 of `order_03`, the same study, the same code, the same settings. It
completed on an A100 in 1 hour 46 minutes and wrote
`uncle/indicators/indicator_stream_v1/order_03/seed_0.jsonl`. Boundary models 2,
4 and 6 were kept. Final accuracies: task 5 31.0, task 0 38.2, task 9 39.0,
task 14 37.8, task 17 39.8, task 1 56.8, task 7 34.2, task 3 36.8.

An earlier attempt the same day started on an L4 rather than an A100, saved its
pre-training checkpoint, and stopped. The second attempt refused to resume it,
because `run_job` compares the recorded GPU and software settings against the
current ones and stops rather than continuing one run across two kinds of
hardware. That checkpoint held 0.45 seconds of compute, no completed tasks and no
measured boundaries, so it was deleted and the run started fresh. Nothing was
lost. The guard is worth keeping in mind for any run that disconnects partway: a
checkpoint holding real training should not be cleared to get past it.

### Loss progress rises with the position, on all three tasks

Each of the three moving tasks has now been measured at positions 1, 4 and 8.

| Task | Position 1 | Position 4 | Position 8 | Spread |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.1509 (`order_01`, three seeds) | 0.2777 (`order_02`) | 0.2869 (`order_03`) | 0.1361 |
| 5 | 0.1426 (`order_03`) | 0.2493 (`order_01`, three seeds) | 0.2905 (`order_02`) | 0.1479 |
| 14 | 0.1169 (`order_02`) | 0.1862 (`order_03`) | 0.2182 (`order_01`, three seeds) | 0.1013 |

All three rise with the position, in the same order, with no exception. The five
tasks that never moved give the noise floor:

| Task | Position | `order_01`, three seeds | `order_02` | `order_03` | Largest gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 0.1868 | 0.2002 | 0.1992 | 0.0135 |
| 9 | 3 | 0.2562 | 0.2668 | 0.2890 | 0.0327 |
| 17 | 5 | 0.2758 | 0.2752 | 0.2719 | 0.0039 |
| 1 | 6 | 0.2526 | 0.2396 | 0.2457 | 0.0130 |
| 7 | 7 | 0.2457 | 0.2895 | 0.2989 | 0.0531 |

The largest control gap is 0.0531. The three moving tasks span 0.1013 to 0.1479,
which clears it by 1.9 to 2.8 times. **Loss progress is a property of the position
in the stream, measured three times over.**

There is a task offset as well as a position effect. Task 14 reads lower than
tasks 3 and 5 at every position it was measured at. The position ordering holds
inside each task regardless, which is what the crossed design was built to show.

Two earlier readings are corrected by this. The end-of-stream dip seen in
`order_01` is gone: position 8 is the highest reading for every one of the three
tasks. That dip was task 14 sitting at position 8, and task 14 is the
lowest-reading of the three. And the series is a rise over the first two positions
followed by a flat stretch, with no fall anywhere, so the row's verdict of no
plasticity loss is unchanged and now rests on a complete rotation.

### The effective-rank fall is set by which task comes first

This is the result that matters most, and it is not what the run was built for.
Three runs now share seed 0, so all three start from the identical generator,
reading effective rank 7.3057 on the probe images before any training. Same seed,
same code, same images. Only the sequence differs.

| Run | First task | Rank after task 1 | Rank at boundary 8 | Rank fall | srank fall | Generator moved | Mean final accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `order_03` | 5 | 6.3853 | 5.6552 | 22.6 percent | 4.8 percent | 0.1165 | 39.20 |
| `order_01` | 3 | 6.2475 | 4.0473 | 44.6 percent | 11.0 percent | 0.1196 | 39.20 |
| `order_02` | 14 | 2.4727 | 2.0635 | 71.8 percent | 38.8 percent | 0.1168 | 44.08 |

The generator travels the same distance in all three, 0.1165 to 0.1196, a spread
of 2.6 percent. The effective-rank fall spans 22.6 to 71.8 percent, a spread of
3.2 times. The srank fall spans 4.8 to 38.8 percent, a spread of 8.1 times.

Under `order_03` there is almost no compression to report at all. Effective rank
drops once on the first task, from 7.3057 to 6.3853, then sits between 5.84 and
6.07 for the rest of the stream and ends at 5.6552. srank goes from 227 to 216
across eight tasks. A run that learns the same eight tasks, moves the same
distance, and reaches the same mean accuracy as `order_01` seed 0 shows half its
rank fall and less than half its srank fall.

**Accuracy does not follow the fall.** `order_03` falls 22.6 percent and
`order_01` seed 0 falls 44.6 percent, and both end at a mean final accuracy of
39.20. `order_02` falls the furthest, 71.8 percent, and ends highest at 44.08. So
among these three the indicator read as lost capacity is, if anything, pointing
the wrong way.

**Dormant units are the stable one.** Across all five runs the rise sits between
33.3 and 43.2 percent, a spread of 1.3 times, against effective rank's 3.2 and
srank's 8.1. Whatever dormant units measure, it survives a change of task order.
Effective rank largely does not.

### What this does not establish

- One seed per new order. The three `order_01` seeds supply the noise floor for
  the loss-progress tables. The rank comparison rests on three runs that share
  seed 0 and differ only in order, which is the comparison it needs, but each of
  the two new orders has been run once.
- Why learning task 14 first collapses the rank reading, while tasks 3 and 5 first
  do not, is not explained here. Task 14 ends lowest in `order_02` at 36.0 percent
  and task 5 ends lowest in `order_03` at 31.0 percent, so reading task 14 as
  simply the hard one does not survive. Nothing here tests the mechanism.
- The share of the whole fall that lands on the first task is 32.5, 92.2 and 55.8
  percent for `order_01`, `order_02` and `order_03`. It is large in all three and
  does not order with anything else measured here.
- Mean final accuracy over eight tasks is a coarse summary. It is used only to
  show the fall does not track performance, not to rank the runs.
- The loss-progress gap at `order_01` seed 2, boundary 6, is still missing and is
  excluded from that task's mean.
