# Experiment trajectory: 1 November 2026

[Back to the experiment index](Experiment-trajectory)

## Plasticity indicators, built and run once

The plasticity probe is expensive and runs twice, before the stream and after it. It cannot say
when a change happened or whether it happened gradually. This adds five cheap indicators that are
read during training, at the end of every task, so the whole trajectory is visible instead of its
two endpoints.

The indicators are the ones the adapter plasticity study in `reference/adapter-plasticity-study`
reports, applied to the hypernetwork set-up so the two can be compared without reshaping either.

### What is measured

| Indicator | What it reads | Where |
| --- | --- | --- |
| Relative weight change | Distance from the boundary-zero weights, as a fraction | Generated target weights and generator parameters |
| Weight norm | Size of the weight vector | Both |
| Angle | Degrees turned from the boundary-zero weights | Both |
| Effective rank and srank | Directions still used to describe an image, delta 0.01 | Classifier input features |
| Dormant units | Units whose response has gone to nothing, at 0.01, 0.025 and 0.1 | Keyed by ReLU call position |
| Loss progress | How much one update moves the loss, sampled every tenth update | Training loop |

The weight indicators are read through one fixed reference code, seed 20261030, that belongs to no
task. Without a fixed code the reading would mix the generator moving with the code changing.

### The fixed image sets

Three sets of 256 images each, the same images at every boundary.

| Set | Contents |
| --- | --- |
| `in_stream` | Held-out images of the learned tasks, concatenated |
| `probe` | Tiny ImageNet task 2, never learned |
| `eurosat` | Satellite images, already 64 by 64 |

Measuring on the current task's images would confuse "the representation of this data changed"
with "the representation changed". The adapter study had to correct this after the fact. The
correction is built in here rather than applied later.

### Why the indicators had to be measured during training, not recovered afterwards

`StudyStore` keeps two rolling resume slots, `resume_0` and `resume_1`, not one checkpoint per
task. A finished job has no per-task history left to read. Four of the five indicators are read at
a task boundary, so the only way to get them is to attach an observer to the training loop.

### Commits

| Commit | What it added |
| --- | --- |
| `a27ac3f` | Dormant unit counting |
| `f759a85` | Loss progress |
| `120216d` | The boundary record, one JSON line per boundary |
| `01a4f33` | The long table and the curves |
| `260816f` | The observer that attaches the indicators to a stream run |
| `4be0eaa` | Notebook 36, which runs the stream, and notebook 37, which reads the records |
| `82065e4` | Wall-clock lines, a progress bar, and handing the Colab runtime back when the run completes |
| `5ac79bf` | The angle |

Notebook 36 creates its own study, `indicator_stream_v1`, whose `code_version` is whatever `HEAD`
is at creation. The frozen `paired_generalization_v2` study and notebooks 19 and 20 are untouched.

### The angle, added after the run had started

The first seven commits recorded how far the weights moved and how large they are, never which way
they point. Weights that double in size without turning travel a long way; weights that turn
without growing can travel the same distance. `delta_w` alone cannot tell those apart.

Two implementation facts, both established by measurement rather than assumed:

- The angle cannot be computed in float32. A ResNet50 weight vector has about 25 million entries,
  and a float32 dot product over that many terms reads a cosine of about 1.003. The error is near
  3e-3 while a one percent move turns the vector by about 5e-5 of a cosine, so float32 buries
  exactly the signal being looked for. The sum is taken in double precision.
- The angle is taken from the distance between the two unit vectors, `2*asin(norm(a-b)/2)`, not
  from `acos` of the dot product. Near zero degrees, which is where this measurement lives, `acos`
  loses about half its digits and cannot read exactly zero for a vector against itself, which is
  what the t = 0 row has to say.

The angle cannot be recovered after the fact from the recorded norms. The law of cosines on those
numbers has an error near 6e-5, the same order as the signal.

75 tests cover the indicators. Four mutations of the angle code were checked and all four fail at
least one test. One earlier test was wrong and the code caught it: `nudge` steps on the sum of
squares, whose gradient is the parameter itself, so it multiplies every parameter by the same
factor. It is pure shrinkage and turns nothing. That is now the test.

## The first indicator run, seed 0

Notebook 36, study `indicator_stream_v1`, order 01, seed 0. Tiny ImageNet, 20 tasks of 10 classes
at 64 by 64, of which order 01 learns eight: 3, 0, 9, 5, 17, 1, 7, 14.

Started 31 October 2026 at 23:56 and completed 1 November at 01:18, runtime clock. One hour 22
minutes on an NVIDIA A100-SXM4-40GB. Status complete, nine boundaries recorded in
`uncle/indicators/indicator_stream_v1/order_01/seed_0.jsonl` with its baseline file beside it.

This run loaded its code before `5ac79bf` was pushed, so **its record carries no angle column.**
Seeds 1 and 2 will carry it. The reader takes whatever numeric keys each row has, so a mixed set
of records reads without error; the angle row simply has fewer runs in it.

### Two indicators were printed while the run went

| Tasks learned | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `delta_w_generator` | 0 | 0.0321 | 0.0575 | 0.0716 | 0.0816 | 0.0949 | 0.1035 | 0.1124 | 0.1196 |
| `effective_rank_in_stream` | not printed | 6.26 | 6.04 | 5.12 | 4.92 | 4.52 | 4.15 | 3.89 | 4.02 |

The weight change is monotone and decelerating. Each task moves the generator less than the one
before: 0.0321, then 0.0254, 0.0141, 0.0100, 0.0133, 0.0086, 0.0089, 0.0072.

Effective rank on the held-out images of the learned tasks falls from 6.26 after the first task to
3.89 after the seventh, then recovers slightly to 4.02. That is a fall of about 38 percent across
the stream.

### What this run cannot say

- **It is one seed.** There is no spread, so no change here clears a noise floor, because there is
  no floor yet. Nothing in this section is a finding.
- **Boundary zero's effective rank was not printed on screen.** It is in the record. The `t = 0`
  cell above is left empty rather than guessed.
- **The other indicators were not read.** Weight norm, srank, dormant units, loss progress, and the
  rank on the `probe` and `eurosat` image sets are all in the record and have not been read out of
  it. Notebook 37 does that on CPU.
- **The `in_stream` set grows with the stream.** It is the concatenation of the held-out images of
  the tasks learned so far, so its contents at boundary 1 are not its contents at boundary 8. A
  falling rank on it is consistent with the model narrowing onto what it has seen as well as with a
  loss of capacity. The `probe` and `eurosat` sets are fixed throughout and are the comparison that
  separates those two readings. They have not been read.
- **No angle.** See above.

### Final accuracies

| Position in the order | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task | 3 | 0 | 9 | 5 | 17 | 1 | 7 | 14 |
| Accuracy, percent | 33.6 | 33.8 | 35.0 | 34.0 | 41.4 | 54.2 | 39.2 | 42.4 |

Ten classes per task, so chance is 10 percent. Earlier tasks sit lower than later ones, which is
ordinary forgetting and matches earlier runs on this order. Nothing indicates the run itself was
unhealthy.

### Cost

Per task, in order: 8:28, 9:48, 9:35, 10:01, 10:17, 9:51, 11:13, 11:28. About ten minutes a task
and rising slowly. A three-seed set is therefore about four hours of A100 time.

## Next decision

1. Read seed 0's record with notebook 37. CPU only, no GPU cost. This fills boundary zero's rank,
   the other three indicators, and the `probe` and `eurosat` comparison that the `in_stream`
   reading needs.
2. Run seeds 1 and 2, about three hours of A100, to get a spread. Those records carry the angle.
3. Recover seed 0's start and end angle offline from its baseline file and the finished source
   model, if the angle curve turns out to matter. Deferred.
