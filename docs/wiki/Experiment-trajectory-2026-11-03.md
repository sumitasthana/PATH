# Experiment trajectory: 3 November 2026

[Back to the experiment index](Experiment-trajectory)

## The matched probe: the indicators and the probe finally read the same models

### Why this run exists

Every earlier statement of the form "the indicator says one thing and the direct probe says
another" compared two different training runs.

The probe runs of 29 October read study `paired_generalization_v2`. The indicator runs of 1 and
2 November read study `indicator_stream_v1`. The two studies share all 23 configuration keys,
the same class order and the same seeds, and differ only in the code version they were trained
under. But they are separate training runs, and their final models are different files:

| Study | Final model, seed 0 |
| --- | ---: |
| `paired_generalization_v2` | 539,277,190 bytes, sha256 `d721ba3a...` |
| `indicator_stream_v1` | 539,277,254 bytes, sha256 `b3422029...` |

So the disagreement recorded so far was not measured. It was inferred across studies. Notebook
38 is notebook 35 with `STUDY_ID` changed, and nothing else about the protocol touched, so it
points the same probe at the models the indicators were actually read from.

### The run

Notebook 38, study `indicator_stream_v1`, order 01, seeds 0, 1 and 2, label `pass3_train4800`.
198 probe runs, 128.8 GPU minutes in total on an NVIDIA A100-SXM4-40GB, across two sessions: 80
runs in the first and 118 in the second, which took 77.5 minutes. The resume mechanism worked;
seed 0 was skipped at both probe points in the second session.

Protocol, unchanged from 29 October: Tiny ImageNet task 2, a task no stream ever learns. 4,800
training images and 200 validation images cut from task 2's own training split, 500 test
images. Three learning rates, 250 steps, the full arm only, eleven repetitions at each probe
point. The rate is chosen on validation accuracy and the result read from the test split, so no
image picks the rate and scores the answer.

EuroSAT was off. Its spread is about twice task 2's, so at eleven repetitions its noise floor
swallows any change it could show. It needs about forty repetitions and its own run label.

### The result

Probe accuracy at boundary 0, and at boundary 8 after eight tasks of continual training, on the
same model each indicator was read from.

| Seed | Probe t = 0 | Probe t = 8 | Change | Noise floor | Ratio | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 58.53 | 59.75 | +1.22 | 2.33 | 1.021 | inside the floor |
| 1 | 54.53 | 58.80 | +4.27 | 4.19 | 1.078 | higher than at t = 0 |
| 2 | 55.27 | 60.00 | +4.73 | 2.08 | 1.086 | higher than at t = 0 |

**No seed loses plasticity. All three move upward, and two of the three clear their own noise
floor.** The mean ratio across the three runs is 1.062, with a spread of 0.065.

The noise floor is the study's own definition: 1.96 times the spread of single repetitions at
boundary 0, times the square root of (1/n at each point). Seed 1's floor is wide because its
eleven repetitions at boundary 0 were themselves spread out, not because its change was small.

### The indicators, on the same models

| Seed | Effective rank, `probe` images, t = 0 | At t = 8 | Fall | Dormant units, rise | Generator weight change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.3057 | 4.0473 | 44.6 percent | 0.0660 | 0.1196 |
| 1 | 4.5691 | 2.9684 | 35.0 percent | 0.0831 | 0.1200 |
| 2 | 9.1953 | 3.5702 | 61.2 percent | 0.0680 | 0.1142 |

The `probe` image set is Tiny ImageNet task 2, the same task the probe trains on. So the
indicator and the probe are reading the same model on the same pictures.

### What this says, under the rule written down before the run

The implementation plan, section 7.5, fixed the reading rule in advance. An indicator **agrees**
if it moves in the direction the probe moved and clears its own spread across seeds. It
**disagrees** if it moves the other way, or moves while the probe does not. It is **silent** if
neither moves.

- **Effective rank disagrees.** It falls between 35 and 61 percent while the probe rises on
  every seed.
- **Dormant units disagree.** They rise on every seed, which is the direction read as damage,
  while the probe rises.
- **The pre-registered prediction failed.** Section 7.5 predicted both indicators would stay
  flat here, because the mechanism the adapter study blames for moving them, a slowly trained
  classifier, does not exist in this architecture. There is no such classifier: `uncle/trainer.py`
  builds one optimiser over the generator's trunk and heads, and the target's classifier is
  written by a hypernetwork head rather than trained on its own. The indicators moved anyway.

That failure is the useful part. The adapter study's own explanation for why these indicators
mislead cannot be the whole explanation, because the indicators mislead in the same direction in
an architecture the explanation does not apply to.

### What this does not establish

**No correlation can be claimed from three runs.** The rank fall does not order with the probe
ratio: seed 1 falls least (35.0 percent) and ends at 1.078, seed 0 falls in the middle (44.6
percent) and ends lowest at 1.021, seed 2 falls most (61.2 percent) and ends highest at 1.086.
Weight change does not order with it either. Three points cannot distinguish a real relationship
from none, and the adapter study's own table uses 75 and 36 points while still warning that
checkpoints from one run are not independent.

**One class order, one probe task.** Everything here is order 01 and Tiny ImageNet task 2. The
indicators also moved on EuroSAT, by 15.0, 45.9 and 53.5 percent, and this run gives no probe
reading there. Closing that gap needs EuroSAT at about forty repetitions and its own run label.

**Two probe points.** Seeds 1 and 2 kept models at boundaries 2, 4 and 6 for exactly this, and
those have not been probed. Loading a boundary file is code notebook 38 does not have.

**The movement control was still missing when this section was written.** It was run the same
day, and the next section reports it.

## The random-direction control: the fall is not a property of movement

### Why this run exists

Effective rank falls across a stream and the direct probe does not. Two explanations fit that.
Either the fall records something about what learning did, or it records only that the generator
moved, in which case moving it anywhere would do the same and the indicator says nothing about
learning at all.

The control separates them. It rebuilds the generator each stream started from, from the seed
alone, with no download and no training. It then pushes that generator by hand the same
distances the stream travelled, read from the stream's own record, but in directions chosen by
nothing, and reads the same indicators on the same images.

### The run

Notebook 39, study `indicator_stream_v1`, order 01, seeds 0, 1 and 2, label `control1`.
147 measurements, 7.0 minutes in total. Nothing trains, so this is forward passes only.

Two kinds of random direction, because "a random direction" is ambiguous. The **global** arm
spreads the move evenly over the whole generator, which concentrates it in the largest tensors.
The **layerwise** arm moves every tensor by the same fraction of its own size, which is closer to
what training does. Both travel the same total distance. Three draws at each boundary, so the
spread of the draws is reported beside each number.

Two checks guard the result. The rebuilt generator was compared against the numbers the stream
recorded at boundary 0 before anything was measured, and on all three seeds and all three image
sets the gap was 0.00 percent. Every move was checked against the distance it was asked for, and
the worst was 0.39 percent out.

### The result

Effective rank on the `probe` images at boundary 8, with the stream's own distance applied in a
random direction instead of the direction learning chose.

| Seed | Fresh | Stream at t = 8 | Stream fall | Random, global | Random, layerwise | Share of the fall, global | Share, layerwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.3057 | 4.0473 | 44.6 percent | 7.4453 | 7.3495 | -4.3 percent | -1.3 percent |
| 1 | 4.5691 | 2.9684 | 35.0 percent | 4.3312 | 4.9343 | 14.9 percent | -22.8 percent |
| 2 | 9.1953 | 3.5702 | 61.2 percent | 7.3133 | 7.9582 | 33.5 percent | 22.0 percent |

Dormant units at threshold 0.01, on the same images.

| Seed | Fresh | Stream rise | Random rise, global | Random rise, layerwise |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.1639 | +0.0660 | +0.0001 | -0.0011 |
| 1 | 0.1588 | +0.0831 | +0.0031 | -0.0032 |
| 2 | 0.1531 | +0.0680 | +0.0085 | +0.0051 |

### What it says

**Dormancy is learning's doing, almost entirely.** The stream raises dormant units by 0.066 to
0.083 on every seed. The same distance in a random direction raises them by at most 0.0085, and
on two seeds one arm moves them down. Random movement reproduces between nothing and a ninth of
the rise.

**Most of the rank fall is learning's doing as well, but not all of it.** On seed 0 random
movement reproduces none of the fall: both arms end above where the generator started, and the
change is inside the spread of the three draws. On seed 1 neither arm reproduces the fall
either, and the two arms disagree in sign, which is what no effect looks like at three draws. On
seed 2 both arms do fall, by 20.5 and 13.5 percent, which is a fifth to a third of the stream's
61.2 percent, and the global arm's fall is well outside its draw spread.

**Seed 2 is the one that started highest.** Its fresh rank is 9.1953 against 7.3057 and 4.5691.
The 2 November record already found that the size of the fall is set by the random starting
point, and that rank converges to the same narrow band whatever it started from. The control
fits that: a generator that starts high has further to converge, so even a direction chosen by
nothing drags it part of the way down. A generator that starts near the band does not move.

**The timing is the clearest part.** Seed 2's stream loses 38 percent of its rank at boundary 1
alone, where the distance travelled is only 0.0303. Random movement over that identical distance
leaves it at 9.0623 and 9.2109, which is where it started. Whatever the stream does in its first
task, distance does not explain it.

### What this does not establish

Three draws per point is thin, and the spreads are wide, up to 1.9 rank units on seed 2. The
control rules out "the fall is only movement" on seeds 0 and 1 and leaves seed 2 partly open. It
does not identify what about learning's direction causes the fall.

Only boundary 8 is summarised above. The full curve was measured at every boundary and is in R2
under `uncle/random_direction/indicator_stream_v1/order_01/control1.jsonl`.

## What each measurement showed

Every indicator the plan asked for has been measured on `indicator_stream_v1`, seeds 0, 1 and 2,
order 01. The direct probe is the reference they are judged against, because it trains an unseen
task on the same model and counts the score, which is what the word plasticity means. The probe
ends at 1.021, 1.078 and 1.086 of where it started.

Read one at a time, the split is clean and it falls along one line. The measurements that train something and count the result, the direct probe and random-label fitting, both say capacity held. The three that read the weights or the activations and infer, effective rank, srank and dormant units, all say it was lost. Weight change gives no verdict on its own, which is what it was built to do, and loss progress cannot be read in this design.

Curvature looked like it sat on the line and then came down on the training side. Read raw at the reference code it rises and joins the three that infer. But the rise tracks the rise in loss at that point, and at a code the model actually trained it is absent, on a measurement with enough power to have seen it. So the rise belongs to where it was measured, not to the model.

| Measurement | What it did | Isolated inference | Compared with the probe |
| --- | --- | --- | --- |
| Direct probe | Ends at 1.021, 1.078 and 1.086 of its start, and does not dip at t = 2, 4 or 6 | **No plasticity loss.** It trains an unseen task on the model and counts the score, which is what the word names. The score held and rose a little | This is the reference the others are compared with |
| Random-label fitting | Final training loss falls 0.5953, 0.7902 and 0.6310, against floors of 0.2672, 0.2580 and 0.2017. All three clear, by 2.2 to 3.1 times | **No plasticity loss.** Lower loss means the model memorised more of the shuffled labels, so its capacity to change rose | Agrees, on all three seeds, on these same models. October's separate training run gave the same direction and sizes within 0.1 |
| Effective rank | Falls 35.0, 44.6 and 61.2 percent | **Plasticity loss, severe.** A fall of this size is well past the point at which the literature calls a model saturated | Points the other way. The size of each fall also tracks where that seed started: 4.57, 7.31 and 9.20, ending in one narrow band. As of 6 November the size is known to be set by which task comes first. Three orders run at seed 0 start from the identical generator and fall 22.6, 44.6 and 71.8 percent, a spread of 3.2 times, while the generator travels the same distance in all three, 0.1165 to 0.1196. Mean final accuracy does not follow the fall: the two runs ending at 39.20 fell 22.6 and 44.6 percent, and the furthest fall ended highest, [record](Experiment-trajectory-2026-11-06) |
| srank | Falls about 11 percent on the in-stream and probe images, 32 percent on EuroSAT | **Plasticity loss, mild.** Larger on an unrelated domain than on the stream's own images | Points the other way, by much less. Its spread widens rather than narrowing, so the convergence that shapes effective rank does not appear to shape it. Its fall is also set by the task order: 4.8, 11.0 and 38.8 percent across three orders run at seed 0 from the identical starting generator, a spread of 8.1 times, [record](Experiment-trajectory-2026-11-06) |
| Dormant units | Rises 0.0660, 0.0831 and 0.0680 at threshold 0.01 | **Plasticity loss.** More units stop responding to anything, which is read as capacity dying | Points the other way. The control puts movement alone at 0.0085 at most, so the rise comes from the direction training takes. Real effect; not what the probe measures |
| Loss progress | Not a rise. Averaged over three seeds it reads 0.151, 0.187, 0.256, 0.249, 0.276, 0.253, 0.246, 0.218 at t = 1 to 8. It jumps on the first two tasks, flattens, and comes back down at the end. Quoting the two endpoints as a rise from 0.1397 to 0.2160, as earlier versions of this row did, hides that shape | **No plasticity loss.** Each task's loss falls further late in the stream than early, which reads as learning getting easier | Readable as of 6 November, when two more task orders separated the two. All three moving tasks were measured at positions 1, 4 and 8 and all three rise with the position, clearing the five fixed tasks' largest gap of 0.0531 by 1.9 to 2.8 times, so the shape belongs to the position in the stream and not to the task, [record](Experiment-trajectory-2026-11-06). The end-of-stream dip was task 14, the lowest-reading task, sitting at position 8; position 8 is the highest reading for all three tasks. It also has a gap at seed 2, boundary 6. The held-out version of the same question is the learning-speed row below |
| Weight change | Reaches about 0.12 of the generator's own size by boundary 8 | **No verdict either way.** It has no direction that means loss. It is a predictor of the probe, not an indicator of damage | Not tested here. Three seeds cannot separate a real relationship from none |
| Curvature | Mean Hessian eigenvalue, 40 directions per point, [runs 2 and 3](Experiment-trajectory-2026-11-04). At the reference code it rises 13.7, 6.5 and 5.5 times, clearing a both-ends floor on all three seeds. At a code the model trained it does not move at all, on an arm with the power to see a rise that size | **No plasticity loss.** The rise is real but belongs to the reference point, not to the model. It tracks the rise in loss at that point, and it is absent where the model actually settles | Agrees, by two separate routes. Dividing out the loss scale removes the rise on every seed, and the trained-code arm never shows it. The raw rise would disagree; the raw rise is the artifact |
| New-task learning speed | Steps to reach half of a fresh model's score on held-out task 2, read out of notebook 40's saved probe curves on 5 November, [record](Experiment-trajectory-2026-11-05). Seed 1 falls 61.2 to 36.5 steps, seed 2 55.6 to 45.1. Both clear a both-ends floor | **No plasticity loss.** The model learns an unseen task faster after eight tasks, not slower, which is the opposite of what a saturating model does | Agrees. It is the same fork-and-discard probe read earlier in its training, so it is not independent of the direct probe, but it answers the speed question the direct probe's fixed budget could not |

### Random-label fitting, the one that agrees

It was measured on 20 and 21 October by notebook 29, not by the indicator stream, and it was
missing from earlier versions of this table. The numbers above were recomputed from the saved
runs in `uncle/plasticity_probe/paired_generalization_v2/order_01/partial.jsonl`, 189 random-label
runs out of 756, under notebook 29's own rule: keep the learning rate with the lowest final
training loss, then average over repetitions.

Two thousand images from Tiny ImageNet task 8 are given fixed wrong labels, drawn once from seed
1020 so every probe point sees the same wrong labels, and the model is trained on them for 600
steps, about nineteen passes. Random labels carry no pattern, so the only way to fit them is to
memorise each image. Lower final loss means more raw capacity to change.

| Seed | Loss at t = 0 | At t = 8 | Change | Noise floor |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.3667 | 0.6710 | -0.6957 | 0.2900 |
| 1 | 1.5435 | 0.7829 | -0.7606 | 0.4225 |
| 2 | 1.3097 | 0.6653 | -0.6443 | 0.1845 |

Chance loss for ten classes is ln(10) = 2.303, and every reading sits well below it, so the model
really did memorise rather than run out of budget. An earlier attempt on 17 October was void for
exactly that reason: the model memorised nothing, so the number measured the budget and not the
model. The fix was fewer images and more passes over them, and 21 October is the first valid run.

Three caveats. It was measured on study `paired_generalization_v2`, not on `indicator_stream_v1`,
so it carries the same across-study gap that notebook 38 closed for the probe. That gap matters
less here than it did there, because the direct probe ran in the same notebook on the same
models, so these two are matched to each other even though neither is matched to the indicator
stream. It uses five repetitions at t = 0 and only two at t = 8, which is why the floors are wide.
And it is the `full` arm only.

### What follows

The direct probe says there is no plasticity loss here. Effective rank, srank and dormant units
all say there is. Loss progress cannot be read, and the held-out measurement that replaces it,
new-task learning speed, says there is not. Random-label fitting says there is not, and it is
the only one of the five that matches the direct measurement.

That ordering is worth stating plainly, because it is the reverse of the adapter study's. There,
random-label loss was the weakest predictor of the probe ratio, at -0.25 and -0.67, while
effective rank and dormant units were middling and weight change was strongest. Here the
indicator the adapter study ranked near the bottom is the only one that gets the answer right.

## Next decision

1. Probe seeds 1 and 2 at boundaries 2, 4 and 6, which turns three points into nine and shows
   whether the two sides disagree in the middle of a stream as well as at its ends. Notebook 40
   does this. It measures five points, 0, 2, 4, 6 and 8, and re-measures the two endpoints in the
   same session as the middle ones, because the probe's spread across sessions is wider than its
   spread inside one and a borrowed endpoint would put a session difference in the middle of the
   curve. 330 probe runs, about 3.3 hours, half of that per seed.
2. Decide whether seed 2's partial movement effect needs more draws. Raising repetitions from
   three to ten costs about twenty minutes and would say whether a fifth to a third is real.
3. Leave correlation claims out of any write-up until there are more than three runs. The
   results that stand today are the direction and the control: the probe does not fall, both
   indicators say it did, and moving the generator the same distance in a direction unrelated to
   learning does not reproduce what they report.

## The probe curve: five points instead of two

Notebook 38 probed only the start and the end of the stream. That leaves one shape unmeasured:
the probe could fall during the stream and recover by the end, and two points would miss it.
Notebook 40 probes seeds 1 and 2 at t = 0, 2, 4, 6 and 8, same study, same protocol, eleven
repetitions at every point, label `curve1`. Seed 0 is absent because its stream kept no middle
models.

### The result

Seed 1, probe task 2, spread of the eleven repetitions at t = 0 is 3.27, floor 2.73:

| t | Accuracy | Change | Ratio | Reading |
| ---: | ---: | ---: | ---: | --- |
| 0 | 55.67 | +0.00 | 1.000 | baseline |
| 2 | 57.75 | +2.07 | 1.037 | inside the floor |
| 4 | 59.25 | +3.58 | 1.064 | higher than at t = 0 |
| 6 | 59.64 | +3.96 | 1.071 | higher than at t = 0 |
| 8 | 62.42 | +6.75 | 1.121 | higher than at t = 0 |

Seed 2, probe task 2, spread 3.67, floor 3.07:

| t | Accuracy | Change | Ratio | Reading |
| ---: | ---: | ---: | ---: | --- |
| 0 | 56.71 | +0.00 | 1.000 | baseline |
| 2 | 57.65 | +0.95 | 1.017 | inside the floor |
| 4 | 60.44 | +3.73 | 1.066 | higher than at t = 0 |
| 6 | 59.75 | +3.04 | 1.054 | inside the floor |
| 8 | 58.29 | +1.58 | 1.028 | inside the floor |

No middle point falls below its floor on either seed. The dip the two-point design could have
hidden is not there.

### What it says and what it does not

The probe does not sag during the stream. Every point on both seeds is at or above its starting
value, and the two points that are clear of the floor on seed 1 and the one on seed 2 are all
above. So the indicators fall while the probe is flat or rising at every point measured, not only
at the ends.

Seed 1 rises steadily. Seed 2 peaks at t = 4 and drifts back inside its floor by t = 8. With one
seed each way, the shape between the ends is not established; only the absence of a fall is.

### How it compares with notebook 38

Notebook 38 measured the same two seeds at the same two ends, in a different sitting.

| Seed | t = 0, nb 38 | t = 0, nb 40 | t = 8, nb 38 | t = 8, nb 40 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 54.53 | 55.67 | 58.80 | 62.42 |
| 2 | 55.27 | 56.71 | 60.00 | 58.29 |

The accuracies repeat to within 1.1 to 3.6 points, which is about one floor. The ratios differ
more: seed 1 gives 1.078 then 1.121, seed 2 gives 1.086 then 1.028. That is expected, because a
ratio divides two noisy numbers and carries both errors. Both sittings agree on the sign and on
the verdict. Neither shows a fall. A ratio from a single sitting should be read as roughly 1,
not as a precise number.

## Random-label fitting, now on the indicator models

The result above was measured in October on `paired_generalization_v2`, a different training
run from the one everything else on this page uses. Notebook 41 repeats it on
`indicator_stream_v1`, the models the indicators, the matched probe and the random-direction
control were all read from. Method unchanged: task 8, labels drawn once from seed 1020, 2,000
images, 600 steps, rates 1e-4, 3e-4 and 1e-3, the rate with the lowest final training loss kept.
Two changes: five repetitions at both probe points instead of five and two, and the `full` arm
only. 90 runs, 121.1 minutes, which matches the 2.0 hours estimated from October's per-run time.

### The result

Final training loss on 2,000 images with shuffled labels. Lower means the model memorised more
of them, so lower means more capacity to change.

| Seed | t = 0 | t = 8 | Change | Floor | Margin | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1.3339 | 0.7386 | -0.5953 | 0.2672 | 2.2x | no loss, more capacity to change |
| 1 | 1.4759 | 0.6857 | -0.7902 | 0.2580 | 3.1x | no loss, more capacity to change |
| 2 | 1.3921 | 0.7611 | -0.6310 | 0.2017 | 3.1x | no loss, more capacity to change |

Mean change -0.6721, spread across seeds 0.1037. Chance loss for ten classes is ln(10) = 2.303,
and every mean sits well below it, so the probe memorised rather than measuring its own budget.

### Against October

| Seed | October change | October floor | Here | Floor here |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.6957 | 0.2900 | -0.5953 | 0.2672 |
| 1 | -0.7606 | 0.4225 | -0.7902 | 0.2580 |
| 2 | -0.6443 | 0.1845 | -0.6310 | 0.2017 |

Same direction on every seed, every one clear of its floor, and the sizes land within 0.1 of
each other on two seeds and 0.03 on the third. Two separately trained sets of models from one
study design give the same answer.

Seed 1's floor fell from 0.4225 to 0.2580. That was the reason for running five repetitions at
t = 8 instead of two, and it worked: the October floor was wide because the later point had only
two repetitions to build it from.

### What this closes

Random-label fitting is now measured on the same models as everything else on this page. The
"different training run" caveat on its row is gone. On one set of models, with one set of
seeds: the direct probe says capacity held, random-label fitting agrees, and effective rank,
srank and dormant units all point the other way.

What it does not close: two probe points only, so nothing says what happens between them for
this measurement; the `full` arm only; and the rate is chosen on the same training loss that is
then reported, which is the standard definition of the measure but makes the number a
best-achievable fit rather than a held-out score. The same rule runs at both points, so the
comparison between the points is fair.

## Curvature: measured, and not readable as it stands

Notebook 42 measured the curvature of the loss surface at five points on the stream, on the same
models as everything else on this page. It was the last named gap on the measurement side.

### The run

At each point the generator is loaded, the indicators' fixed task code is attached, and the loss
is read on the same 256 held-out images from task 2 that the indicators use. Then a random
direction is drawn, the generator is moved a small distance along it and the loss is read again,
and it is moved the same distance the other way and read a third time. Curvature along that
direction is `2 * ((L_plus + L_minus) / 2 - L0) / r**2`. Stepping both ways cancels the slope, so
what is left is the bend. Averaged over random directions this is the mean eigenvalue of the
Hessian, which is the overall bend of the surface, not its sharpest direction.

Three step sizes, 0.001, 0.003 and 0.01 of the generator's own length. Five random directions at
each. Five points for seeds 1 and 2, two for seed 0, which kept no middle models. 180
measurements, nothing trained.

### The result, at the smallest step

| Seed | t | Loss at the point | Mean curvature | Spread | vs t = 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 68.55 | 0.03391 | 0.0107 | 1.00 |
| 0 | 8 | 1786.36 | 0.7159 | 0.194 | 21.1 |
| 1 | 0 | 1089.62 | 0.243 | 0.0757 | 1.00 |
| 1 | 2 | 7840.80 | 1.381 | 1.07 | 5.68 |
| 1 | 4 | 12443.04 | 3.529 | 1.64 | 14.5 |
| 1 | 6 | 13487.95 | 0.926 | 1.45 | 3.81 |
| 1 | 8 | 8363.18 | 1.704 | 1.32 | 7.01 |
| 2 | 0 | 42.74 | 0.01065 | 0.00263 | 1.00 |
| 2 | 2 | 249.11 | 0.0258 | 0.0186 | 2.42 |
| 2 | 4 | 463.57 | 0.08593 | 0.0519 | 8.07 |
| 2 | 6 | 478.65 | 0.126 | 0.0432 | 11.8 |
| 2 | 8 | 1140.99 | 0.02237 | 0.121 | 2.10 |

The notebook printed all three endpoints as sharper at the end, which it reads as plasticity
loss. That print uses a noise floor built from the spread at t = 0 only, which is how the other
measurements on this page build theirs. For this measurement that is the wrong floor, because the
spread does not stay the same along the stream. It grows by 18, 17 and 46 times.

### Rebuilding the floor from both ends

A floor that uses the spread at both points, `1.96 * sqrt((s0**2 + st**2) / 5)`:

| Seed | Change | Floor from t = 0 only | Floor from both ends | Clears by |
| --- | ---: | ---: | ---: | ---: |
| 0 | +0.682 | 0.0133 | 0.170 | 4.0 |
| 1 | +1.461 | 0.0939 | 1.159 | 1.3 |
| 2 | +0.0117 | 0.00326 | 0.106 | does not clear |

So one seed clears clearly, one clears narrowly, and one does not clear at all. That is a weaker
statement than three out of three.

### Three further problems

The three step sizes do not agree, and not in one direction. Seed 0 at t = 8 reads 0.716, 0.426
and 0.272 as the step grows. Seed 1 at t = 6 reads 0.926, 1.592 and 2.027, which goes the other
way. Seed 2 at t = 8 reads 0.0224, 0.194 and 0.118. A step that is simply too large makes the
number drift one way. A number that moves both ways is noise.

The middle points do not trace a line. Seed 1 runs 0.24, 1.38, 3.53, 0.93, 1.70. Seed 2 runs
0.011, 0.026, 0.086, 0.126, 0.022. Neither is a slow climb into a sharper place.

The loss at the measurement point rises by about as much as the curvature does. Seed 0's loss
goes up 26 times while its curvature goes up 21 times. Seed 1's goes up 7.7 times against 7.0.
Cross-entropy curvature carries the scale of the loss with it, so curvature divided by loss is
worth a look:

| Seed | Curvature / loss at t = 0 | At t = 8 | Ratio |
| --- | ---: | ---: | ---: |
| 0 | 4.95e-4 | 4.01e-4 | 0.81 |
| 1 | 2.23e-4 | 2.04e-4 | 0.91 |
| 2 | 2.49e-4 | 0.20e-4 | 0.08 |

Divided by the loss, no seed rises. This is a diagnostic, not a proof: it says the raw rise and
the loss rise move together, which is what a scale effect looks like.

One more thing the table makes plain. The loss at these points reaches 12443. Chance loss for ten
classes is 2.303. The measurement sits at a fixed reference task code on a held-out task, which is
the same point the other indicators read, so it is consistent with them. But it is a long way from
anywhere the model was trained to be, and curvature there is not obviously the curvature the
literature means when it says a continually trained model ends up in a sharp place.

### What this says

Curvature is measured and the answer is that it is not readable in this design. The raw numbers
rise. The rise survives a proper floor on one seed of three, does not hold a shape across the
stream, does not repeat across step sizes, and tracks the loss scale closely enough that the scale
is a live explanation for it.

That is a result about the measurement, not about the model. It does not join effective rank,
srank and dormant units, and it does not join the probe. It sits on its own, saying that with 5
directions per point and 3 seeds there is not enough signal here to say anything.

What would make it readable: many more directions per point, say 50 instead of 5, which costs
forward passes and no training; a measurement point nearer a trained solution rather than a fixed
reference code; and the top Hessian eigenvalue rather than the mean, which is what most of the
sharpness literature actually reports and which needs Hessian-vector products and a power
iteration.
