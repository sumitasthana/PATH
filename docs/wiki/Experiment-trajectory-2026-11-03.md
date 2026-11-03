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

**The movement control is still missing.** The 2 November record showed that effective rank
converges to the same narrow band whatever it started at. Until a generator is moved the same
distance, 0.118 of its own size, in a direction unrelated to learning, the fall could be a
property of movement rather than of learning. That experiment needs no training run and is still
the one that decides what all of this means.

## Next decision

1. Run the random-direction control. It is cheap, it needs no stream, and it decides whether
   the rank fall measures learning at all.
2. Probe seeds 1 and 2 at boundaries 2, 4 and 6, which turns three points into nine and shows
   whether the two sides disagree in the middle of a stream as well as at its ends.
3. Leave correlation claims out of any write-up until there are more than three runs. The
   result that stands today is the direction: the probe does not fall, and both indicators say
   it did.
