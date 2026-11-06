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
is not established.

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

### Where it leaves the decision

`order_03` puts task 5 at position 1, task 14 at position 4 and task 3 at
position 8. It is now worth more than it was before this run, for two reasons
rather than one. It gives position 1 a third task, which turns two agreeing
readings into three. And it gives a second measurement of how much the
effective-rank fall varies with the order, which is the finding above and
currently rests on a single pair.
