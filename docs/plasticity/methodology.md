# Methodology

Companions: `docs/plasticity/run-plan.md` implements this and
`docs/plasticity/paper-brief.md` writes the paper around it.

Section 3 of the plasticity-indicators paper, which is a standalone paper. It
is not a section of anything larger. If that changes later, this section is
written to survive the move, but nothing here should be shaped around that
possibility.

Status note, to be deleted before submission: the design below is the revised
one. The results currently in the draft (effective-rank convergence, the
task-order comparison, the norm control) were measured under the earlier
design of eight tasks of ten classes. They need re-running against this
methodology before they can be reported alongside it.

## 3.1 Data

All experiments use Tiny ImageNet. The dataset has 200 classes, each with 500
training images and 50 validation images at 64 by 64 pixels.

Every one of the 200 classes is used. We do not discard any part of the
dataset.

The classes are split once into ten disjoint groups of twenty. The split is
made by sorting the WordNet class identifiers, permuting that sorted list with
a NumPy PCG64 generator seeded at 42, and cutting the permuted list into ten
consecutive blocks of twenty. The permutation is independent of any training
randomness, so the split is the same no matter which seed a run trains with.
The resulting assignment is written to a file and checked on every later run,
so a run that loads a different split fails rather than proceeding quietly.

Nine of the ten groups form the continual-learning stream. The model learns
them one after another, twenty classes at a time. The tenth group is never
trained on. It is held out as an in-distribution probe: images the model has
never seen, drawn from the same dataset and the same image distribution as the
nine it learns.

A second probe comes from outside Tiny ImageNet. We use EuroSAT, a set of
satellite images with no overlap in subject matter, resolution convention or
capture process. It is never trained on either.

The two probes answer different questions. The held-out Tiny ImageNet group
asks what the model does with familiar-looking images it has not learned. The
satellite images ask what it does with images unlike anything in the stream. A
measure that moves on both is describing the generator. A measure that moves
only on the stream's own images is describing what the model learned about
those particular classes.

Three image sets are therefore read at every measurement point:

| Name | Source | Role |
|---|---|---|
| `in_stream` | Held-out test images of the nine learned tasks | Inside the training distribution |
| `probe` | Test images of the tenth Tiny ImageNet group | Unseen, in distribution |
| `eurosat` | Satellite images | Unseen, out of distribution |

Each set contributes 256 images, drawn once and fixed across every boundary,
every seed and every ordering.

## 3.2 Model

A hypernetwork learns the nine tasks in sequence. A generator holds a
32-dimensional code for each task and 200 chunk codes, and from these it writes
the complete weights of a ResNet50 target network.

The target network stores nothing between tasks. It is regenerated from the
codes at every use. No classifier is trained separately: the final layer is
written by the generator like every other layer. A single Adam optimizer covers
all trainable parameters at one learning rate, with no parameter groups.

Because the network is written fresh at every use, the quantities usually
tracked in plasticity work change meaning. Weight growth and rank are
properties of the generator, not of a parameter vector that accumulates across
tasks. We note this because it affects which published results are comparable
to ours and which are not.

## 3.3 Hyperparameter tuning

The earlier version of this work took the learning rate from the original
hypernetwork paper and did not tune it. Accuracy was low and varied widely
between seeds, which makes every downstream comparison harder to read: a
difference between two runs could be a real effect or could be the noise floor.

We therefore tune before measuring anything.

**What is tuned.** The learning rate first, since it is the setting with the
largest effect on both final accuracy and seed-to-seed spread. Epochs per task
and the regularization strength that holds earlier tasks still are tuned only
if the learning-rate sweep alone does not reach the target below.

**How it is tuned.** A sweep over learning rates on a shortened stream of the
first three tasks, one seed per value, choosing the value with the highest mean
accuracy over those three tasks. The shortened stream keeps the sweep cheap.
The chosen value is then confirmed on the full nine-task stream.

**The target.** Five seeds of the full stream, differing only in random
initialization and data order, must finish with mean accuracy over the nine
tasks spread no wider than 3 percentage points between the highest and lowest
seed. If the spread is wider, the setting is not yet good enough to measure
against and we tune further rather than reporting on top of it.

**Why this target.** Every claim in the results rests on comparing two numbers
read from different runs. A seed-to-seed spread of 3 points puts a floor under
what a difference has to clear before it means anything. Without that floor we
cannot tell an effect from the noise, and the results of the earlier version
were measured without it.

The tuning runs are reported in full, including the values rejected. The
learning rate used for every result in this paper is the one chosen here, and
it is the same in every run.

## 3.4 Task orderings

The order in which tasks arrive is one of the things we test, so it is varied
deliberately rather than left to chance.

Three orderings of the same nine tasks are used. Six of the nine tasks occupy
the same position in all three. Three tasks rotate through positions 1, 5 and
9, so that each of the three is learned first in one ordering, in the middle in
another, and last in the third.

The rotation is what separates an effect of the order from an effect of one
task being harder than another. If a measure reads differently when task A
comes first than when task B comes first, a design that shuffled everything
could not say whether the order mattered or whether A and B simply differ. Here
the same three tasks appear in the same three positions in every ordering, only
permuted among themselves, and the other six never move.

Task IDs 0 to 8 are the stream and task 9 is the held-out group of 3.1.

| Ordering | Task sequence | Position 1 / 5 / 9 |
|---|---|---|
| order_A | 0, 1, 2, 3, 4, 5, 6, 7, 8 | 0 / 4 / 8 |
| order_B | 4, 1, 2, 3, 8, 5, 6, 7, 0 | 4 / 8 / 0 |
| order_C | 8, 1, 2, 3, 0, 5, 6, 7, 4 | 8 / 0 / 4 |

Tasks 1, 2, 3, 5, 6 and 7 never move. Tasks 0, 4 and 8 form a three-cycle, so
each is learned first in one ordering, in the middle in another, and last in
the third.

These orderings were chosen for this study. An earlier version of this work
used the request sequences of the original hypernetwork paper, which address
twenty task IDs and cannot be expressed over ten groups.

Orderings are compared with the random initialization held identical, and the
sharing is confirmed rather than assumed: the two runs being compared must read
the same value at boundary 0, before any training, or the comparison is not
made.

## 3.5 The unlearn arm

One further step is run after the nine tasks, on every finished stream run. A
single task is unlearned deliberately, using the UnCLe procedure, which reverts
the generated parameters of that task toward a standard normal initialization.
It runs for 100 update steps.

The task unlearned is the one learned first in that ordering: task 0 for
order_A, task 4 for order_B, task 8 for order_C. Running this on all seven
stream runs rather than one costs 700 update steps in total, less than one
task's training, and gives seven damaged models across three orderings and five
seeds rather than a single case.

The purpose is narrow. It produces a model whose damage is known, located in a
known place, and confirmed by accuracy: the unlearned task should sit at chance
for a twenty-class task, which is 5 percent, while the other eight tasks should
be close to unchanged. That check is made on each of the seven runs before any
reading from it is used. A reading is taken at the boundary after this step:
boundary 10, counting the fresh generator as boundary 0, which is the eleventh
reading of that run.

This arm is what lets us ask whether the indicators detect damage that is
certainly present. It is not a claim about unlearning, and nothing in our
conclusions depends on the UnCLe method working as its authors describe. We use
it only as a way to break a model on purpose.

The rescaling control in 3.8 is applied to each run's pair of checkpoints, the
model before the unlearn and the model after it, because the unlearn moves the
weight norm much further in one step than a learning task does. That is seven
pairs, of which five differ only in the seed.

## 3.6 Reading protocol

Readings are taken at ten boundaries: before the first task, and after each of
the nine. Every run then takes an eleventh after its unlearn step, as 3.5
describes. A reading is a single forward pass. Nothing is trained to take one.
There are no gradient steps and no classifier head fitted on top of the
features.

A measure should describe the generator, not whichever task happens to be
loaded. Four choices follow from that, and all four are held fixed everywhere.

**One code.** Every reading uses the same reference code, drawn once at random
with seed 20261030. It belongs to no task, is never trained, and is identical
across boundaries, seeds and orderings.

**One set of statistics.** A BatchNorm layer keeps the numbers passing through
the network at a sensible size by dividing them by a stored mean and spread.
Every reading uses the same stored pair, the one each task begins from, so it
is the same at boundary 0 as at boundary 9. Evaluation mode is forced. In
training mode the layer works the mean and spread out from the images in front
of it instead, so a collapsed signal would be divided by a correspondingly
small spread and come out looking healthy.

**One layer.** Features are the 2048 numbers each image produces at the input
to the target network's final classifier layer. They are copied out by a
forward hook, which reads them in passing and alters nothing, so the generated
weights are loaded exactly as the generator wrote them.

**The same images.** Each of the three image sets contributes the same 256
images at every boundary and in every run.

At every boundary we check that the reading used all 256 images and all 22,720
ReLU outputs, so no result rests on a short or partial reading. The feature
matrix is 256 rows by 2048 columns, so no rank count can exceed 256.

## 3.7 Measures

**Effective rank** (Roy and Vetterli, 2007). With `s_i` the singular values of
the feature matrix and `p_i = s_i / sum(s)`, effective rank is
`exp(-sum(p_i log p_i))`. It need not be a whole number. A matrix with equal
singular values has effective rank equal to its column count; one with a single
non-zero singular value has effective rank 1. A fall during continual training
is a published warning sign of lost plasticity.

**srank** (Kumar et al., 2021), with delta = 0.01. Sort the singular values in
descending order, accumulate them as a running share of the total, and count
how many are needed to reach 1 - delta. It is a whole number. Unlike effective
rank it discards the tail of near-zero directions rather than letting them
contribute.

**Dormant units** (Sokar et al., 2023). A site is one place in the network
where the ReLU function is applied. A unit's score is its mean absolute
activation divided by the mean over all units at that same site. A score of 1
is an ordinary unit and a score near 0 is one that has stopped contributing.
Scoring within a site matters because sites differ in scale by orders of
magnitude. A site whose units are all exactly zero counts as wholly dormant.
Cut-offs of 0.01, 0.025 and 0.1 are recorded.

Both rank measures normalize the singular values by their own sum, so rescaling
the feature matrix leaves them unchanged. Whether that invariance survives the
path from weights to features is one of the things the results test.

**Weight norm.** Alongside every reading we record the norm of the generated
weights, the single number saying how large the whole set of weights is. It
costs nothing to compute and it is the quantity we find moves the rank measures
without touching the representation.

**Accuracy.** Top-1 accuracy on the held-out test images of each learned task,
measured at every boundary after that task, using that task's own code rather
than the reference code.

## 3.8 Controls

**Seed control.** One ordering is run with five random initializations, with
everything else held fixed, to separate what training does from what the
initialization supplies. Across runs we report the spread, not the mean,
because the spread is what reveals convergence toward a common band.

**Order control.** All three orderings are run with the same five random
initializations. Runs sharing a verified identical initialization are compared
across orderings, as described in 3.4, and the five runs of an ordering are
also averaged, so a difference between orderings is read against how far that
ordering's own seeds disagree.

An earlier version ran the two further orderings with one seed each and called
the result a controlled comparison rather than a sampled one. Measuring the
seed spread of the indicators themselves showed that this cannot work for the
rank measures. Across five seeds of one ordering, effective rank at the end of
the stream moves about half as much between seeds as nine tasks of learning
move it, so a gap between two single runs would have to reach about 72 percent
of the measure's own value before it meant anything, and for srank 81 percent.
The weight measures would have been safe on one seed, needing 2 to 3 percent,
but the rank measures are the ones this control exists to compare. Five seeds
per ordering is what makes the comparison readable.

**Rescaling control.** To separate a change in a rank measure from a
simultaneous change in weight scale, we read four arms in four forward passes
with nothing trained: each of the two checkpoints unmodified, the later
checkpoint's weights multiplied by a single constant that restores the earlier
norm, and the earlier checkpoint's weights multiplied by the constant that
imposes the later norm. Whatever difference survives the rescaling is the part
attributable to the representation.

This runs on all seven before-and-after pairs from 3.5. The five order A pairs
share everything but the seed, so the spread across them is what a difference
has to clear before the rescaling is said to have changed anything. The earlier
version of this control rested on one pair and could not make that comparison.

## 3.9 Compute

Each full nine-task run trains nine tasks of twenty classes on a ResNet50
written by the generator. The hyperparameter sweep adds shortened three-task
runs, one per learning-rate value. The unlearn arm adds 100 update steps to each
of the fifteen stream runs, which is under 2 percent of a run: a nine-task run
is 7,065 optimizer steps. The three controls in 3.8 add no training: the seed
and order controls read checkpoints the stream runs already produce, and the
rescaling control costs four forward passes.

The run count, to be confirmed against the compute budget before launching:

| Arm | Runs | Training cost each |
|---|---|---|
| Learning-rate sweep | one per candidate value | 3 tasks |
| Seed control, one ordering | 5 | 9 tasks |
| Order control, two further orderings | 10, five seeds each | 9 tasks |
| Unlearn arm | 15, one continuing each finished stream run | 100 steps |

None of the runs behind the earlier version of this work can be reused. They
were trained on eight tasks of ten classes, so every number in every arm above
comes from a fresh run.
