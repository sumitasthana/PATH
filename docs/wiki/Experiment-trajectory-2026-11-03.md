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

## What each indicator is now worth

Every indicator the plan asked for has been measured on `indicator_stream_v1`, seeds 0, 1 and 2,
order 01. The direct probe is the reference they are judged against, because it trains an unseen
task on the same model and counts the score, which is what the word plasticity means. The probe
ends at 1.021, 1.078 and 1.086 of where it started.

| Indicator | What it did | Read on its own it says | Against the probe |
| --- | --- | --- | --- |
| Effective rank | Falls 35.0, 44.6 and 61.2 percent | Severe loss of plasticity | **Wrong.** Also mostly an artifact: the three seeds start at 4.57, 7.31 and 9.20 and converge to one narrow band, so the size of each fall is set by its random starting point |
| srank | Falls about 11 percent on the in-stream and probe images, 32 percent on EuroSAT | Mild loss, larger on an unrelated domain | **Wrong in the same direction, but far less loudly.** Its spread across seeds widens rather than narrowing, so it is not driven by the convergence that drives effective rank |
| Dormant units | Rises 0.0660, 0.0831 and 0.0680 at threshold 0.01 | Units are dying, so capacity is being lost | **Wrong.** The random control shows the rise is genuinely caused by the direction training takes, not by movement, so it is a real effect that is not plasticity |
| Weight change | Reaches about 0.12 of the generator's own size by boundary 8 | Nothing on its own; it is a predictor, not a verdict | **Untested here.** It was the adapter study's strongest predictor of the probe ratio, but across three seeds it does not order with the probe ratio, and three points cannot tell a real relationship from none |
| Loss progress | Rises from 0.1397 at t = 1 to 0.2160 at t = 8 | Learning is getting easier, not harder | **Not readable in this design.** It is measured on the task being learned, and the eight tasks differ, so the change mixes the task with the position in the stream. It also has gaps at boundaries 1 and 6 |

Random-label fitting, the fifth indicator in the adapter study's table, has not been measured in
this architecture at all.

Two conclusions follow. The direct probe says there is no plasticity loss here. Three of the
four readable indicators say there is, and the fourth cannot be read, so on this architecture the
indicators do not track plasticity.

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
